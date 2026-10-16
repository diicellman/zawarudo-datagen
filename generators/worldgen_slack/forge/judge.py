"""The task judge: one structured call a task, on what code and the tries already found (its answer and the messages
it rests on, the verifier's answers, the student's wrong ones). It rules on what no try can: whether a colleague
would ask it, whether its words allow another reading, whether its answer asks more than its label, and whether its
messages state it. A task is approved when all four hold."""

import json

import verifiers.v1 as vf
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
    reason: NonEmptyText

    @property
    def approved(self) -> bool:
        return self.realistic and self.one_reading and self.exact and self.supported


class TaskJudge(vf.Judge[Ruling]):
    schema = Ruling

    def build_messages(self, **fields):
        return [
            vf.UserMessage(
                content=render(PROMPT, task=json.dumps(fields["task"], ensure_ascii=False, indent=1))
            )
        ]


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


def hearing(world, task: Task, verified: list[dict], measured: list[dict]) -> dict:
    """What the judge reads of a task: its asker, its request, each part with its answer and its messages (and what is
    out of the asker's sight), the verifier's answers and the student's wrong ones with their grades."""
    asker = world.db.execute("SELECT real_name, title FROM users WHERE id = ?", (task.actor_id,)).fetchone()
    wrong = [r for r in measured if r and not r.get("crashed") and not r.get("correct")]
    return {
        "asker": f"{asker['real_name']} ({asker['title']})" if asker else task.actor_id,
        "family": task.family,
        "request": task.question,
        "parts": [
            {
                "label": p.label,
                "answer_type": p.answer_type,
                "answer": p.answer,
                "messages": said(world, p.evidence),
                "out_of_sight": said(world, p.unseen),
            }
            for p in task.parts
        ],
        "verifier_answers": [
            (r.get("response") or "")[:1500] for r in verified if r and not r.get("crashed")
        ],
        "student_tries": len([r for r in measured if r and not r.get("crashed")]),
        "student_wrong_answers": [
            {"answer": (r.get("response") or "")[:1200], "grade": (r.get("reason") or "")[:600]}
            for r in wrong
        ],
    }


async def rule(
    config: vf.JudgeConfig, world, task: Task, verified: list[dict], measured: list[dict]
) -> tuple[Ruling, dict]:
    """The judge's ruling on one task and what its call cost, asked again when a call fails (patient)."""
    judge = TaskJudge(config)

    async def call():
        return await judge.evaluate(task=hearing(world, task, verified, measured))

    result = await patient(call, timeout=600)
    return result.parsed, (result.usage.model_dump(mode="json") if getattr(result, "usage", None) else {})
