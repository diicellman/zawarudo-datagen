"""The agentic judge: semantic review of the tasks and the workspace, in a fresh runtime."""

import json
from pathlib import Path
from typing import Self

import verifiers.v1 as vf
from pydantic import Field
from worldgen_slack.taskset import WorldTaskData
from worldgen_slack.world import SCHEMA, file_hash
from ..workspace import Workspace

from ..contracts import frozen_before
from ..session import require_trace
from .tools import ReviewState, ReviewTools, ReviewToolsConfig
from .verdict import PHASE_CRITERIA, Issue, Verdict, accepted, quality, validate_verdict, verdict_schema

REVIEW_PROMPT = """Review what /task/input.json names, as /task/review.md says. /task/world.sqlite is the workspace
with its facts and tasks; /task/schema.sql is its schema. inspect_check runs code's checks and re-runs the reviewed
tasks' gold queries; inspect_read reads Slack as a task's actor; inspect_sql runs SQL as a task's actor. Other JSON
files in /task are agents' traces: solver_<task_id>_<n>.json are independent solves of that task, and
witness_<task_id>_<n>.json a stronger solver's tries of a task the solves rarely answer. Write the verdict to
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
and its gold rows are the complete, correct answer; a status task's answer is its gold value given as not settled, as
the actor can see it, while the fact that settles it is out of their sight. Score each criterion of schemas.json from 0 to 1:
- question_fit: each gold query answers its question as asked; test variants with inspect_sql.
- discoverability: the actor can find the answer with Slack's read tools; try inspect_read, and read the solves.
  When input.json has witness, a stronger solver tried the tasks the solves rarely answer: one it answers can be
  solved, which shows nothing about whether its question has one reading; one it fails too may be unanswerable.
- One reading. Each wrong solve's grade gives the answer it committed to and the gold rows it missed or added. For
  each, decide whether the question's own words allow a reading under which that answer is right: does a thread
  reply count as posting, is the poster among those who replied, does a reaction break a quiet spell, what comes
  before what in "then", whose clock and which week, a team or a channel. A mistake the question's words rule out is
  the solver's. If a reading allows it, the task is not valid: set its review's misread to that answer and the
  reading, and raise a blocking issue asking for the wording that rules the reading out with the same gold.
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
ledger, it gives every planned event and fact with its moment on the company clock. A fact marked (decoy) is a planned
near-miss: a wrong claim someone makes, which tasks rely on a reader taking for the answer. Judge a decoy's messages
for realism only (would this person say it, on what they knew), and leave the claim standing: it is a coherence defect
only when the story later acts on it as true and nothing in the workspace ever settles it; never ask for a decoy to be
corrected or hedged. When it has written_through, the
workspace is written up to that moment and goes on later: judge what exists, and leave open what is still open.
When it has written_from, the workspace before that moment is an earlier chapter's, written and frozen: read it as the
record this chapter goes on from, and judge the chapter's messages, from that moment on, and how they go on from it. An
earlier chapter's message is not changed; name it only as evidence.
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


def review_payload(world: Workspace, phase: str, task_ids: list[str], taxonomy=None, **extra) -> dict:
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


def thaw(verdict: Verdict, world) -> tuple[Verdict, list[Issue]]:
    """A verdict on a chapter's world, its earlier chapters read as written and frozen: an issue's messages from before
    the chapter become its evidence, and an issue left with nothing of this chapter's to change (no message, and no
    fact or task of its own) leaves the verdict, kept apart: nothing can act on it. A world written in one run has
    nothing frozen. Returns the verdict and the issues it set apart."""
    if not (start := frozen_before(world)):
        return verdict, []
    old = {r[0] for r in world.db.execute("SELECT id FROM messages WHERE ts_us < ?", (start,))}
    parent = {
        r[0]
        for r in world.db.execute(
            "SELECT f.id FROM facts f JOIN calendar c ON c.day = f.day WHERE c.start_us < ?", (start,)
        )
    }
    kept, apart = [], []
    for issue in verdict.issues:
        live = [m for m in issue.message_ids if m not in old]
        moved = issue.model_copy(
            update={
                "message_ids": live,
                "evidence_message_ids": list(
                    dict.fromkeys([*issue.evidence_message_ids, *(m for m in issue.message_ids if m in old)])
                ),
            }
        )
        frozen = (
            (issue.message_ids or issue.fact_ids)
            and not live
            and set(issue.fact_ids) <= parent
            and not issue.task_ids
        )
        (apart if frozen else kept).append(moved)
    return verdict.model_copy(update={"issues": kept}), apart


async def review(store, agents, payload: dict, attempt: str, max_rows: int, files=None, label="") -> Verdict:
    """One review by the judge, kept with its attempt: its input, its verdict, and the issues a chapter's frozen past
    sets apart. `files` maps /task file names to trace records the judge may investigate; `label` names one of an
    attempt's several reviews. A malformed verdict is absorbed by one fresh rerun."""
    suffix, files = ("-" + label if label else ""), files or {}
    store.artifact(attempt, "review_input" + suffix, payload)
    store.event("review_started", attempt=attempt, label=label, task_ids=[t["id"] for t in payload["tasks"]])
    world = store.snapshot(attempt, f"review{suffix}.sqlite")
    # One fresh rerun absorbs a malformed verdict; the review round is spent once.
    for _ in range(2):
        task = JudgeTask.create(
            payload,
            world,
            attempt,
            max_rows,
            {n: json.dumps(r) for n, r in files.items()},
        )
        trace = await agents.judge.run(task)
        store.trace(trace)
        if trace.ok:
            break
    require_trace(trace)
    verdict = Verdict.model_validate(trace.info["verdict"])
    store.artifact(attempt, "verdict" + suffix, verdict.model_dump(mode="json"))
    verdict, frozen = thaw(verdict, store.world)
    if frozen:
        store.artifact(attempt, "frozen_issues" + suffix, [i.model_dump(mode="json") for i in frozen])
    verdict = verdict.model_copy(update={"approved": accepted(verdict)})
    if payload["phase"] in PHASE_CRITERIA:
        store.artifact(
            attempt, "acceptance" + suffix, {"approved": verdict.approved, "quality": quality(verdict)}
        )
    return verdict
