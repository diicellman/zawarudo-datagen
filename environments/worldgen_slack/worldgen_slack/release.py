"""The release: one solver copy of the world, public questions, and private gold answers."""

import json
import os
import tempfile
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from .world import World, canonical, file_hash


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


SafeId = Annotated[str, Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")]
NonEmptyText = Annotated[str, Field(min_length=1)]
Evidence = Annotated[list[str], Field(min_length=2, max_length=2)]  # [channel, ts]


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".pending-")
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(canonical(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path: Path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(path.read_bytes(), object_pairs_hook=unique)


class PublicTask(StrictModel):
    """A task as a solver gets it, with how hard it measured: the solver model's tries at the release (those that
    crashed, and those that never answered), its share of right answers (with its 90% interval), of right and
    grounded ones, and of those that declined to answer, the gold evidence the tries saw, and the witness's share of
    right answers when it tried. Training can filter a curriculum on them, as prime-envs filters on avg@k columns. A
    composite request (v11) is one question in 2-4 parts, right only when every part is. A forged task (v12) names the
    set it came from (the initial set, or a harder, easier or repaired version of one: `variant_of`), the verifier's
    share of right answers, and the solver's by part."""

    task_id: SafeId
    question: NonEmptyText
    actor_id: SafeId
    category: NonEmptyText
    level: int
    answer_type: Literal["text", "set", "number", "refusal", "status", "composite"]
    world_hash: str
    solver: str | None = None
    tries: int | None = None
    crashed: int | None = None
    unanswered: int | None = None
    right_rate: float | None = None
    right_interval: Annotated[list[float], Field(min_length=2, max_length=2)] | None = (
        None  # its 90% interval
    )
    strict_rate: float | None = None
    abstain_rate: float | None = None
    coverage: float | None = None
    witness: str | None = None
    witness_right: float | None = None
    # v10: the tool surface and turn limit its tries had (rates on one do not carry to another), and how its wrong
    # tries went wrong, by code (search, budget, abstained, overclaim, trap, saw_all).
    surface: str | None = None
    turns: int | None = None
    miss_classes: dict[str, int] | None = None
    intent: str | None = None  # v10: the request it is, a person's to an assistant
    family: str | None = None  # v11: the request it is, of four (catch-up, decisions, commitments, status)
    set: Literal["initial", "harder", "easier", "repaired"] | None = None  # v12
    variant_of: SafeId | None = None  # v12: the task it is a version of
    verifier: str | None = None  # v12: the model that verified its gold, and its share of right answers
    verifier_right: float | None = None
    part_rates: dict[str, float] | None = None  # v12: the solver's share of right answers, by part


class Solution(StrictModel):
    """A right and grounded try of a task (v11): an agent's answer from Slack alone and the calls it made, each tool
    with its arguments, in order. It shows its gold is reached by reading Slack, in an agent's words, and not only by
    the ledger's query."""

    seat: Literal["solver", "witness", "verifier"]
    model: str | None = None
    trace_id: NonEmptyText
    answer: NonEmptyText
    calls: list[dict[str, JsonValue]]


class PrivateAnswer(StrictModel):
    """The gold rows of a task, the messages (channel, ts) and people its answer rests on, and, for a refusal, the
    truth out of its actor's sight (`unseen`). Its rows come from a gold query, or (v12) are the answer an agent's
    tries verified, with no query. A composite request's answer (v11) is its parts', each such an answer
    under its label, in the request's order: it has no rows or query of its own, and its messages, people and unseen
    are its parts' together. `solution` is a right and grounded try of it. Its `context` (v13) is what its answer
    rests on that the request does not ask for: an answer may give it or leave it out, and is wrong to contradict it."""

    answer_type: Literal["text", "set", "number", "refusal", "status", "composite"]
    rows: list[dict[str, JsonValue]]
    gold_sql: str = ""
    messages: list[Evidence]
    unseen: list[Evidence] = []
    users: list[str]
    label: NonEmptyText | None = None  # v11: a part's, what of its request it answers
    parts: list["PrivateAnswer"] = []  # v11: a composite request's
    solution: Solution | None = None  # v11: a try of it that reached its gold, the request's and not a part's
    context: list[NonEmptyText] = []  # v13: supporting details, not required (a composite's are its parts')

    @model_validator(mode="after")
    def shaped(self):
        if any(p.solution for p in self.parts):
            raise ValueError("a composite answer's solution is the request's, not a part's")
        if self.answer_type != "composite":
            if self.parts or not (self.gold_sql or self.rows or self.answer_type == "refusal"):
                raise ValueError("an answer has its rows (or its gold query, or is a refusal) and no parts")
            return self
        if self.context:
            raise ValueError("a composite answer's context is its parts'")
        labels = [p.label for p in self.parts]
        if not 2 <= len(self.parts) <= 4 or self.rows or self.gold_sql:
            raise ValueError("a composite answer is its 2-4 parts', with no rows or query of its own")
        if (
            None in labels
            or len(set(labels)) < len(labels)
            or any(p.answer_type == "composite" for p in self.parts)
        ):
            raise ValueError("a composite answer's parts are answers, each under a label of its own")
        for field in ("messages", "unseen", "users"):
            own = {json.dumps(v) for v in getattr(self, field)}
            if own != {json.dumps(v) for p in self.parts for v in getattr(p, field)}:
                raise ValueError(f"a composite answer's {field} are its parts' together")
        return self

    def answers(self) -> list["PrivateAnswer"]:
        """What its grade decides on: its parts, or itself."""
        return self.parts or [self]

    def reference(self) -> dict:
        """What the answer judge grades a response against: the answer type and gold rows, or each part's under its
        label."""
        if self.parts:
            return {
                "answer_type": "composite",
                "parts": [p.reference() | {"label": p.label} for p in self.parts],
            }
        return {"answer_type": self.answer_type, "rows": self.rows} | (
            {"context": self.context} if self.context else {}
        )


class Manifest(StrictModel):
    format: Literal["worldgen-slack.v13", "worldgen-slack.v12", "worldgen-slack.v11"] = (
        "worldgen-slack.v13"  # v11, v12 still read
    )
    world_hash: str
    files: dict[str, str]


FILES = {"world.sqlite", "tasks.json", "answers.json"}


def load_release(root: Path) -> tuple[Path, list[PublicTask], dict[str, PrivateAnswer]]:
    """Verify a release before use: file hashes, one world, matching tasks and answers, readable evidence."""
    root = root.resolve()
    manifest = Manifest.model_validate_json((root / "manifest.json").read_bytes())
    if set(manifest.files) != FILES:
        raise ValueError("a release holds world.sqlite, tasks.json and answers.json")
    for name, expected in manifest.files.items():
        if (root / name).is_symlink() or file_hash(root / name) != expected:
            raise ValueError(f"release file hash mismatch: {name}")
    rows = [PublicTask.model_validate(r) for r in read_json(root / "tasks.json")]
    answers = {k: PrivateAnswer.model_validate(v) for k, v in read_json(root / "answers.json").items()}
    if not rows or len({r.task_id for r in rows}) != len(rows) or set(answers) != {r.task_id for r in rows}:
        raise ValueError("public tasks and private answers must match exactly")
    for row in rows:
        if row.world_hash != manifest.world_hash:
            raise ValueError("a task references another world")
        reader, answer = World(root / "world.sqlite", actor=row.actor_id), answers[row.task_id]
        if answer.answer_type != row.answer_type or answer.label is not None:
            raise ValueError(f"{row.task_id}: its answer is another kind than its question")
        for channel, ts in answer.messages + answer.unseen:
            readable = reader.db.execute(
                "SELECT 1 FROM messages WHERE channel_id = ? AND ts = ?", (channel, ts)
            ).fetchone()
            if not readable and [channel, ts] in answer.messages:
                raise ValueError(f"{row.task_id}: evidence its actor cannot read")
            if readable and [channel, ts] in answer.unseen:
                raise ValueError(f"{row.task_id}: evidence out of sight that its actor can read")
        for user in answers[row.task_id].users:
            if not reader.db.execute("SELECT 1 FROM users WHERE id = ?", (user,)).fetchone():
                raise ValueError(f"{row.task_id}: a gold person who is not in the workspace")
        reader.close()
    return root / "world.sqlite", rows, answers
