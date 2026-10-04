"""The agentic judge: semantic review of the tasks and the workspace, in a fresh runtime."""

import json
from pathlib import Path
from typing import Self

import verifiers.v1 as vf
from pydantic import Field
from worldgen_slack.db import SCHEMA, World
from worldgen_slack.tools import WorldTaskData, file_hash

from ..contracts import PHASE_CRITERIA, Verdict, quality, validate_verdict, verdict_schema
from .inspection import ReviewState, ReviewTools, ReviewToolsConfig

REVIEW_PROMPT = """Review what /task/input.json names, as /task/review.md says. /task/world.sqlite is the workspace
with its facts and tasks; /task/schema.sql is its schema. inspect_check runs code's checks and re-runs the reviewed
tasks' gold queries; inspect_read reads Slack as a task's actor; inspect_sql runs SQL as a task's actor. Other JSON
files in /task are agents' traces: solver_<task_id>_<n>.json are independent solves of that task. Write the verdict to
/task/verdict.json following /task/schemas.json.
"""

COMMON_GUIDE = """Approve only after investigating. Call inspect_check before writing the verdict.
- tasks: one review per task in input.json; when a task is valid depends on the review, below.
- issues: each defect, with its artifact, the ids it concerns, the defect and the requested_change. artifact is ledger
  for facts, storylines, people's titles and channels; tasks for a task's question, actor, answer type or gold query;
  workspace for messages: message_ids are the messages to change, evidence_message_ids the messages that show the
  defect and stay as they are.
- blocking is true when a defect changes, contradicts, hides or hands over a task's answer, or breaks the ledger;
  otherwise it is false.
- approved is true only when every task is valid and no issue blocks.
- When input.json has previous_issues, check that each is resolved.
"""

PHASE_GUIDES = {
    "task": """Task review. A task is valid when its actor can answer its question as asked from what they can read,
and its gold rows are the complete, correct answer. Score each criterion of schemas.json from 0 to 1:
- question_fit: each gold query answers its question as asked; test variants with inspect_sql.
- discoverability: the actor can find the answer with Slack's read tools; try inspect_read, and read the solves.
  Solvers that answer differently or fail can reveal ambiguity or a missing time scope; a mistake the evidence rules
  out is the solver's.
- shortcut_free: nothing in the workspace hands over the answer outside its evidence.
- level_fit, on each task's review, from 0 to 4: how fully answering it needs the level and concept its `means`
  names. 4: it needs them, with no shortcut; 3: it needs them, but one route is easier than the level; 2: partly;
  1: in name only; 0: one obvious step answers it. inspect_check measures each task: how many read_channel pages deep
  its evidence sits for its actor, the tables its gold query reads, and its evidence's search rank for the question's
  own words; the solves show the calls it took. level_fit is reported beside the task; it decides neither valid nor
  blocking.
""",
    "world": """World review: the workspace as a record of real work. Score each criterion of schemas.json from 0 to 1:
- scenario_alignment: the messages tell the storylines.
- world_coherence: messages agree with each other and with the facts; nothing is reported done or known before it
  happens; promises are kept or explicitly changed.
- professional_realism: it reads as real work. inspect_check's style shows each author's messages beside their typing
  (people in input.json); different people writing in one register is a defect.
Message times and order are code's; report text that contradicts them, not the times themselves. When input.json has
ledger, it gives every planned event and fact with its moment on the company clock. When it has written_through, the
workspace is written up to that moment and goes on later: judge what exists, and leave open what is still open.
""",
}


class JudgeConfig(vf.TaskConfig):
    tools: ReviewToolsConfig
    payload: str = Field(default="", exclude=True, repr=False)
    files: dict[str, str] = Field(default_factory=dict, exclude=True, repr=False)


class JudgeTask(vf.Task[WorldTaskData, ReviewState, JudgeConfig]):
    NEEDS_CONTAINER = True

    @classmethod
    def toolsets(cls, config: JudgeConfig) -> list[vf.Toolset]:
        return [ReviewTools(config.tools)]

    @classmethod
    def create(
        cls, payload: dict, world: Path, attempt: str, max_rows: int, files: dict[str, str] | None = None
    ) -> Self:
        """`world` is a host-side snapshot that stays unchanged while the judge works."""
        tools = ReviewToolsConfig(
            db_path=str(world),
            db_hash=file_hash(world),
            task_ids=[t["id"] for t in payload["tasks"]],
            actors=sorted({t["actor_id"] for t in payload["tasks"]}),
            max_rows=max_rows,
        )
        if tools.colocated or tools.runtime.type != "subprocess" or tools.url is not None:
            raise ValueError("the world file requires a host-side tool server")
        return cls(
            WorldTaskData(
                prompt=REVIEW_PROMPT,
                attempt=attempt,
                world_hash=tools.db_hash,
                storyline=payload.get("storyline", ""),
                network_allow=[],
                network_block=["*"],
            ),
            JudgeConfig(tools=tools, payload=json.dumps(payload), files=files or {}),
        )

    async def setup(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        payload = json.loads(self.config.payload)
        for name, text in self.config.files.items():
            await runtime.write("/task/" + name, text.encode())
        await runtime.write("/task/input.json", self.config.payload.encode())
        await runtime.write("/task/schema.sql", SCHEMA.read_text().encode())
        await runtime.write("/task/world.sqlite", Path(self.config.tools.db_path).read_bytes())
        schema = verdict_schema(payload["phase"], [t["id"] for t in payload["tasks"]])
        await runtime.write("/task/schemas.json", json.dumps(schema).encode())
        guide = COMMON_GUIDE + PHASE_GUIDES[payload["phase"]]
        await runtime.write("/task/review.md", guide.encode())

    async def finalize(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        payload = json.loads(self.config.payload)
        raw = await runtime.read("/task/verdict.json", max_bytes=2_000_000)
        trace.info["verdict_candidate"] = raw.decode()
        trace.info["observations"] = [c.model_dump(mode="json") for c in trace.state.calls]
        verdict = Verdict.model_validate_json(raw)
        validate_verdict(verdict, payload)
        if not trace.state.checked:
            raise ValueError("the reviewer did not run inspect_check")
        trace.info["verdict"] = verdict.model_dump(mode="json")
        trace.record_metric("approved", float(verdict.approved))
        if payload["phase"] in PHASE_CRITERIA:
            trace.record_metric(payload["phase"] + "_quality", quality(verdict))


def review_payload(world: World, phase: str, task_ids: list[str], taxonomy=None, **extra) -> dict:
    """What a review covers: its phase, the world's version, and the requested tasks with their gold and, given
    the taxonomy, what their category and level mean."""
    tasks = []
    for task_id in task_ids:
        row = dict(world.db.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone())
        facts = [
            r[0] for r in world.db.execute("SELECT fact_id FROM task_facts WHERE task_id = ?", (task_id,))
        ]
        tasks.append(
            {
                k: row[k]
                for k in ("id", "category", "level", "actor_id", "question", "answer_type", "gold_sql")
            }
            | {"gold": json.loads(row["gold_json"]), "facts": facts}
            | (
                {
                    "means": f"{spec.definition} Level {row['level']}: {spec.levels[row['level'] - 1]}. Concept: {row['concept']}"
                }
                if taxonomy and (spec := taxonomy.get(row["category"]))
                else {}
            )
        )
    return {"phase": phase, "world_hash": file_hash(world.path), "tasks": tasks, **extra}
