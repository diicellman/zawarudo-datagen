"""A forged task: one request a colleague would make, in 1 to 4 parts, each with the answer its proposer wrote and the
messages it rests on. Code checks what code can (the asker, the evidence in their sight, the answer's shape, nothing
given away, no repeat); the verifier's tries decide whether the answer is right."""

import hashlib
import json
import re
import unicodedata
from typing import Literal

from pydantic import Field, model_validator
from worldgen_slack.release import NonEmptyText, PrivateAnswer, SafeId, StrictModel

from ..workspace import Workspace
from ..world.text import normalized

# What a task is, as a request: what people ask a colleague who reads Slack for them.
FAMILIES = {
    "catch-up": "what the asker missed on something, or what changed about it since a moment",
    "decisions": "what was decided about something, and why; whether it is still the plan",
    "commitments": "what someone promised, to whom, and whether it was done",
    "status": "who handles something now, where it stands, or who to ask about it",
}
SETS = ("initial", "harder", "easier", "repaired")
# What a request needs of whoever answers it, where a solver goes wrong: each initial id aims at one, as code deals
# them, and code checks the shape it can tell (`skill_problems`).
SKILLS = {
    "aggregation": "every item, across channels, threads and days (every open blocker, every change, everyone who "
    "still owes something): a part answered by the whole set, gathered from 2 or more places",
    "temporal": "the answer as of a moment, a value that changed more than once, how long something ran or how far "
    "it moved",
    "abstention": "what is asked is not there, not settled, or out of the asker's sight, and the right answer says so: "
    "a status or a refusal part",
    "synthesis": "what is stated apart, joined: a decision and its reason, an owner handed over, a promise and whether "
    "it was kept: a request of 2 or more parts",
}


class Part(StrictModel):
    """One part of a request: what it asks, in the request's own words (`label`); its answer as the asker can tell it
    from Slack (one value; a set's every row; none for a refusal, which is right when the request cannot be fulfilled
    from what the asker can see); the messages it rests on, which the asker can read; and, for a status or a refusal,
    the messages out of the asker's sight that settle it or hold the truth. Its `context` is what the answer rests on
    that the request does not ask for (when, who said it, what it replaced): an answer may give it or leave it out,
    and only `answer` is required (local-docs/calibration/POLICY.md)."""

    label: NonEmptyText
    answer_type: Literal["text", "number", "set", "status", "refusal"]
    answer: list[NonEmptyText] = Field(default_factory=list)
    context: list[NonEmptyText] = Field(default_factory=list)
    evidence: list[int] = Field(default_factory=list)
    unseen: list[int] = Field(default_factory=list)


class Task(StrictModel):
    """One request a colleague would make, of a family: a single question (one part) or a composite request of 2 to 4
    parts, answered right only when every part is. A version of another task names it (`variant_of`) and why: harder,
    easier, or repaired."""

    id: SafeId
    family: Literal["catch-up", "decisions", "commitments", "status"]
    actor_id: SafeId
    question: NonEmptyText
    parts: list[Part] = Field(min_length=1, max_length=4)
    variant_of: SafeId | None = None
    why: Literal["harder", "easier", "repaired"] | None = None

    @model_validator(mode="after")
    def versioned(self):
        if (self.variant_of is None) != (self.why is None):
            raise ValueError("a version of a task names both the task (variant_of) and why")
        return self


def key(task: Task) -> str:
    """What a task's results are kept under: its content. A changed task is a new task."""
    return hashlib.sha256(json.dumps(task.model_dump(mode="json"), sort_keys=True).encode()).hexdigest()[:16]


def tokens(value: str) -> set[str]:
    return {t for t in re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", value).casefold()) if len(t) >= 2}


COUNTS = {"text": (1, 1), "number": (1, 1), "status": (1, 1), "set": (2, 50), "refusal": (0, 0)}


def skill_problems(world, task: Task, skill: str | None) -> list[str]:
    """Why a task lacks the shape of the skill its id aims at, as far as code can tell: aggregation, a set part of 3 or
    more answers resting on messages in 2 or more places (channels, or threads of one); abstention, a status or a
    refusal part; synthesis, 2 or more parts. Temporal has no shape code can tell."""
    if skill == "aggregation":
        for part in task.parts:
            if part.answer_type != "set" or len(part.answer) < 3 or not part.evidence:
                continue
            marks = ", ".join("?" * len(part.evidence))
            places = world.db.execute(
                "SELECT COUNT(DISTINCT CASE WHEN parent_id IS NULL THEN channel_id ELSE channel_id || ':' || parent_id "
                f"END) FROM messages WHERE id IN ({marks})",  # a channel, or a thread in it
                part.evidence,
            ).fetchone()[0]
            if places >= 2:
                return []
        return [
            "it aims at aggregation: give it a set part of 3 or more answers, gathered from messages in 2 or more "
            "places (channels, or threads)"
        ]
    if skill == "abstention" and not any(p.answer_type in ("status", "refusal") for p in task.parts):
        return ["it aims at abstention: give it a status or a refusal part"]
    if skill == "synthesis" and len(task.parts) < 2:
        return ["it aims at synthesis: join what is stated apart, in 2 or more parts"]
    return []


def check(world, task: Task, written: list[Task], skill: str | None = None) -> None:
    """Everything wrong with a task that code can tell, refused at once with every reason: its asker is a person still
    in the workspace; each part's answer has the shape of its type; its evidence exists, is no deleted message, and its
    asker can read it (a status's and a refusal's `unseen` they cannot); its parts ask different things; its question
    gives no answer away; its asker asked nothing the same before; and it has the shape of the skill its id aims at."""
    problems = skill_problems(world, task, skill)
    who = world.db.execute("SELECT is_bot, is_deleted FROM users WHERE id = ?", (task.actor_id,)).fetchone()
    if who is None or who["is_bot"] or who["is_deleted"]:
        problems.append(f"its asker {task.actor_id} is no person in the workspace")
    reader = Workspace(world.path, actor=task.actor_id) if who is not None else None
    for n, part in enumerate(task.parts, 1):
        low, high = COUNTS[part.answer_type]
        if not low <= len(part.answer) <= high:
            problems.append(
                f"part {n}: a {part.answer_type} answer has {low}-{high} values, not {len(part.answer)}"
            )
        if part.answer_type != "refusal" and not part.evidence:
            problems.append(f"part {n}: name the messages its answer rests on (evidence)")
        if part.answer_type == "status" and not part.unseen:
            problems.append(
                f"part {n}: a status names the message out of the asker's sight that settles it (unseen)"
            )
        for message in part.evidence + part.unseen:
            row = world.db.execute("SELECT is_deleted FROM messages WHERE id = ?", (message,)).fetchone()
            if row is None or row["is_deleted"]:
                problems.append(f"part {n}: message {message} is no message of the workspace")
            elif reader is not None:
                seen = reader.db.execute("SELECT 1 FROM messages WHERE id = ?", (message,)).fetchone()
                if message in part.evidence and not seen:
                    problems.append(f"part {n}: its asker cannot read message {message}")
                if message in part.unseen and seen:
                    problems.append(
                        f"part {n}: its asker can read message {message}, which unseen is for what they cannot"
                    )
    labels = [normalized(p.label) for p in task.parts]
    answers = [tuple(sorted(normalized(a) for a in p.answer)) for p in task.parts if p.answer]
    if len(set(labels)) < len(labels) or len(set(answers)) < len(answers):
        problems.append("each part asks for something of its own, with an answer of its own")
    if given := [
        a
        for p in task.parts
        if p.answer_type != "number"
        for a in p.answer
        if not tokens(a) - tokens(task.question)
    ]:
        problems.append(f"the question gives away its answer {given}")
    if any(
        t.actor_id == task.actor_id and normalized(t.question) == normalized(task.question)
        for t in written
        if t.id != task.id
    ):
        problems.append("its asker already asks the same")
    if reader is not None:
        reader.close()
    if problems:
        raise ValueError(f"{task.id}: " + "; ".join(problems))


def private_answer(world, task: Task) -> PrivateAnswer:
    """A task's private answer: each part's answer as its rows, the messages it rests on (channel, ts), and those out
    of its asker's sight; a composite request's, each part's under its label, its evidence theirs together."""

    def place(ids: list[int]) -> list[list[str]]:
        marks = ", ".join("?" * len(ids))
        rows = world.db.execute(
            f"SELECT channel_id, ts FROM messages WHERE id IN ({marks}) ORDER BY ts_us", ids
        ).fetchall()
        return [[r["channel_id"], r["ts"]] for r in rows] if ids else []

    answers = [
        PrivateAnswer(
            answer_type=p.answer_type,
            rows=[{"answer": a} for a in p.answer],
            messages=place(p.evidence),
            unseen=place(p.unseen),
            users=[],
            label=p.label if len(task.parts) > 1 else None,
            context=p.context,
        )
        for p in task.parts
    ]
    if len(answers) == 1:
        return answers[0]
    return PrivateAnswer(
        answer_type="composite",
        rows=[],
        messages=place(sorted({m for p in task.parts for m in p.evidence})),
        unseen=place(sorted({m for p in task.parts for m in p.unseen})),
        users=[],
        parts=answers,
    )


def answer_text(task: Task) -> list[str]:
    """What a task answers, as a person would read it: each part's answer under its label."""
    shown = [
        ", ".join(p.answer) or "(nothing to find: it cannot be answered from what the asker can see)"
        for p in task.parts
    ]
    return shown if len(task.parts) == 1 else [f"{p.label}: {a}" for p, a in zip(task.parts, shown)]
