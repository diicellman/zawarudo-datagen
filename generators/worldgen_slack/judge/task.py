"""The agentic judge: semantic review of the tasks and the workspace, in a fresh runtime."""

import json
from pathlib import Path
from typing import Self

import verifiers.v1 as vf
from pydantic import Field
from worldgen_slack.taskset import WorldTaskData
from worldgen_slack.world import SCHEMA, file_hash

from ..prompt import prompt
from ..session import require_trace
from ..workspace import Workspace
from ..world.clock import frozen_before
from .tools import ReviewState, ReviewTools, ReviewToolsConfig
from .verdict import PHASE_CRITERIA, Issue, Verdict, accepted, quality, validate_verdict, verdict_schema

REVIEW_PROMPT = prompt(__file__, "review.md")
COMMON_GUIDE = prompt(__file__, "common.md")
PHASE_GUIDES = {phase: prompt(__file__, f"{phase}.md") for phase in ("task", "world")}


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


async def review(
    store, agents, payload: dict, attempt: str, max_rows: int = 5, files=None, label=""
) -> Verdict:
    """One review by the judge, kept with its attempt: its input, its verdict, and the issues a chapter's frozen past
    sets apart. `max_rows` bounds a reviewed task's gold rows (a world review has none); `files` maps /task file
    names to trace records the judge may investigate; `label` names one of an attempt's several reviews. A malformed
    verdict is absorbed by one fresh rerun."""
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
