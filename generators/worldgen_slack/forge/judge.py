"""The task judge: one call a task, on what code and the tries already found (its answer and the messages it rests on,
the answers that reached it, the student's wrong ones, what tries named beyond a set's answer). It rules on what no try
can: whether a colleague would ask it, whether its words allow another reading, whether its answer asks more than its
label, whether its messages state it, and whether a set's answer holds every member. A task is approved when all five
hold."""

import json

import verifiers.v1 as vf
from worldgen_slack.judge import READINGS, Misread, asked, replied
from worldgen_slack.release import NonEmptyText, StrictModel

from ..prompt import prompt, render
from ..world.clock import clock
from .tasks import Task
from .tries import patient

PROMPT = prompt(__file__, "judge.md")


class Ruling(StrictModel):
    realistic: bool
    one_reading: bool
    exact: bool
    supported: bool
    complete: bool
    reason: NonEmptyText

    @property
    def approved(self) -> bool:
        return self.realistic and self.one_reading and self.exact and self.supported and self.complete


class TaskJudge(vf.Judge[Ruling]):
    """Asked for its ruling as plain JSON, as the answer judge is (not structured output, which Prime overbills on GLM
    5.3); a reply that does not read as a ruling is a Misread."""

    def build_messages(self, **fields):
        return [
            vf.UserMessage(
                content=render(PROMPT, task=json.dumps(fields["task"], ensure_ascii=False, indent=1))
                + asked(Ruling)
            )
        ]

    def parse(self, response: vf.JudgeResponse) -> Ruling:
        try:
            return replied(response.text, Ruling)
        except ValueError as error:
            raise Misread(f"a reply that does not read as a ruling: {error}"[:300], response.usage) from error


def said(world, ids: list[int]) -> list[dict]:
    """Messages as a person reads them: who wrote what, where and when."""
    out = []
    for message in ids:
        row = world.db.execute(
            """SELECT m.ts_us, m.text, m.parent_id, c.name, c.type, u.real_name FROM messages m
            JOIN channels c ON c.id = m.channel_id JOIN users u ON u.id = m.user_id WHERE m.id = ?""",
            (message,),
        ).fetchone()
        if row is not None:
            where = f"#{row['name']}" if row["name"] else ("a DM" if row["type"] == "im" else "a group DM")
            out.append(
                {
                    "id": message,
                    "from": row["real_name"],
                    "where": where + (" (thread reply)" if row["parent_id"] else ""),
                    "when": clock(world, row["ts_us"]),
                    "text": row["text"],
                }
            )
    return out


def named_beyond(task: Task, n: int, tries: list[tuple[str, dict]]) -> list[dict]:
    """What each try presented beyond a set part's answer, as its grade names it (`extra`: a composite's by part, a
    single part's whole): tries that search the world name the members its writer missed (smoke i011: the verifier's
    and the student's tries named three promises its answer lacked, and were graded wrong for them)."""
    out = []
    for seat, result in tries:
        parts = result.get("parts") or []
        own = next((p for p in parts if p.get("label") == task.parts[n].label), None)
        if own is None and len(parts) == len(task.parts):
            own = parts[n]
        items = (own or {}).get("extra") if parts else result.get("extra")
        if items:
            out.append({"by": seat, "items": list(items)})
    return out


def hearing(world, task: Task, verified: list[dict], measured: list[dict]) -> dict:
    """What the judge reads of a task: its asker, its request, each part with its answer and its messages (and what is
    out of the asker's sight), and for a set what tries named beyond its answer; the answers that reached it (the
    verifier's tries, or, where the student's verified it and no verifier ran, the student's right ones, grounded
    first) and the student's wrong ones with their grades."""
    asker = world.db.execute("SELECT real_name, title FROM users WHERE id = ?", (task.actor_id,)).fetchone()
    finished = [r for r in measured if r and not r.get("crashed")]
    wrong = [r for r in finished if not r.get("correct")]
    tried = [r for r in verified if r and not r.get("crashed")]
    right = sorted((r for r in finished if r.get("correct")), key=lambda r: not r.get("grounded"))
    graded = [("the verifier", r) for r in tried] + [("the student", r) for r in finished]
    return {
        "asker": f"{asker['real_name']} ({asker['title']})" if asker else task.actor_id,
        "family": task.family,
        "request": task.question,
        "parts": [
            {
                "label": p.label,
                "answer_type": p.answer_type,
                "answer": p.answer,
                "context": p.context,
                "messages": said(world, p.evidence),
                "out_of_sight": said(world, p.unseen),
            }
            | ({"named_beyond": named_beyond(task, n, graded)} if p.answer_type == "set" else {})
            for n, p in enumerate(task.parts)
        ],
        "reached_by": "the verifier" if tried else "the student",
        "reaching_answers": [(r.get("response") or "")[:1500] for r in tried or right[:3]],
        "student_tries": len(finished),
        "student_wrong_answers": [
            {"answer": (r.get("response") or "")[:1200], "grade": (r.get("reason") or "")[:600]}
            for r in wrong
        ],
    }


async def rule(
    config: vf.JudgeConfig, world, task: Task, verified: list[dict], measured: list[dict]
) -> tuple[Ruling, dict]:
    """The judge's ruling on one task and what its calls cost: asked again when a call fails (patient), and when its
    reply does not read as a ruling (a Misread, up to READINGS readings, each paid and counted)."""
    judge = TaskJudge(config)

    async def call():
        return await judge.evaluate(task=hearing(world, task, verified, measured))

    usages = []
    for n in range(1, READINGS + 1):
        try:
            result = await patient(call, timeout=600)
        except Misread as misread:
            usages.append(misread.usage)
            if n == READINGS:
                raise
        else:
            usages.append(result.usage)
            break
    usage = vf.Usage.aggregate([u for u in usages if u is not None])
    return result.parsed, (usage.model_dump(mode="json") if usage else {})
