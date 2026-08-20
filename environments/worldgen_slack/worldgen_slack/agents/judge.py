from __future__ import annotations

import json
from typing import Any

import verifiers.v1 as vf

from ..contracts import (
    INTERFACE_ID,
    JudgeVerdict,
    ScenarioSpec,
    ValidationReport,
    redact_secrets,
)
from ..slack.models import SlackWorld, TaskContract
from ..slack.toolset import SlackState, SlackToolset, SlackToolsetConfig
from .solver import SolverSummary

SOLVER_TRACE_FILE = "/tmp/worldgen_solver_trace.json"
JUDGE_CONTEXT_FILE = "/tmp/worldgen_judge_context.json"
VERDICT_FILE = "/tmp/worldgen_judge_verdict.json"

JUDGE_PROMPT = f"""Audit one generated Slack QA item empirically. The complete solver trace is at
`{SOLVER_TRACE_FILE}` and the private audit context is at `{JUDGE_CONTEXT_FILE}`. Inspect those
files selectively. Use the fixed Slack actions to verify the evidence and current world yourself.
Do not trust the solver, builder metadata, or recorded validation without checking.

Score each quality criterion with this common scale: 1 clearly fails; 2 materially weak;
3 acceptable but limited; 4 strong; 5 exceptional.

- scenario_alignment: 1 contradicts the scenario; 2 has major mismatches; 3 broadly fits;
  4 strongly realizes the organization and workflow; 5 is unusually specific and complete.
- world_coherence: 1 is contradictory/broken; 2 has material timeline or relationship problems;
  3 is internally usable; 4 is consistent and well connected; 5 is exceptionally coherent.
- professional_realism: 1 is implausible; 2 is artificial; 3 is credible enough; 4 resembles a
  real workplace; 5 has exceptional professional detail without noise.
- discoverability: 1 cannot be found; 2 needs guessing or hidden knowledge; 3 is reachable with
  some friction; 4 has a clear evidence path; 5 supports robust discovery through several cues.
- shortcut_free: 1 exposes an answer cache; 2 has an obvious shortcut; 3 has minor leakage;
  4 requires genuine Slack retrieval; 5 is robust against shortcuts and prompt artifacts.

Set `task_unambiguous` false when more than one materially different answer fits the question.
Set `world_supports_task` false when actor-visible Slack evidence cannot establish the answer.
Use failure ownership carefully: ambiguous contracts belong to synthesizer; missing or incoherent
world evidence belongs to builder; a capable world with an incorrect answer belongs to solver;
fixed action defects belong to interface. File/tool/runtime failures are infrastructure failures.

Write exactly one JSON object matching the required schema to `{VERDICT_FILE}`. The chat reply is
not used as the verdict.
"""


class JudgeData(vf.TaskData):
    instance_id: str
    question: str
    interface_id: str = INTERFACE_ID


class JudgeConfig(vf.TaskConfig):
    tools: SlackToolsetConfig = SlackToolsetConfig()
    solver_trace: dict[str, Any]
    judge_context: dict[str, Any]


class JudgeTask(vf.Task[JudgeData, SlackState, JudgeConfig]):
    NEEDS_CONTAINER = True

    @classmethod
    def toolsets(cls, config: JudgeConfig) -> list[vf.Toolset]:
        return [SlackToolset(config.tools)]

    async def setup(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        paths = [VERDICT_FILE, SOLVER_TRACE_FILE, JUDGE_CONTEXT_FILE]
        removed = await runtime.run(["rm", "-f", *paths], {})
        if removed.exit_code:
            raise RuntimeError(f"could not clear judge protocol files: {removed.stderr[-500:]}")
        await runtime.write(
            SOLVER_TRACE_FILE,
            json.dumps(
                self.config.solver_trace,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
            ).encode(),
        )
        await runtime.write(
            JUDGE_CONTEXT_FILE,
            json.dumps(
                self.config.judge_context,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
            ).encode(),
        )

    async def finalize(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        try:
            raw = await runtime.read(VERDICT_FILE, max_bytes=64_000)
        except Exception as exc:
            raise ValueError(f"judge wrote no bounded verdict file at {VERDICT_FILE}") from exc
        verdict = JudgeVerdict.model_validate_json(raw)
        if not trace.state.completed_calls:
            raise ValueError("judge completed no empirically recorded Slack action")
        payload = verdict.model_dump(mode="json")
        trace.info["judge_tool_calls"] = trace.state.completed_calls
        trace.info["judge_verdict"] = payload


def make_judge_task(
    *,
    instance_id: str,
    scenario: ScenarioSpec,
    contract: TaskContract,
    world: SlackWorld,
    validation: ValidationReport,
    solver_trace: vf.Trace,
    solver_summary: SolverSummary,
    builder_metadata: dict[str, Any],
) -> JudgeTask:
    context = {
        "scenario": scenario.model_dump(mode="json"),
        "question": contract.question,
        "canonical_answer": contract.answer.canonical_answer,
        "required_claims": contract.answer.required_claims,
        "forbidden_claims": contract.answer.forbidden_claims,
        "required_evidence": [item.model_dump(mode="json") for item in contract.required_evidence],
        "deterministic_validation": validation.model_dump(
            mode="json",
            exclude={"public_snapshot"},
        ),
        "gold_call_log": validation.gold_call_log,
        "builder_artifact": builder_metadata,
        "solver_summary": solver_summary.model_dump(mode="json"),
        "verdict_schema": JudgeVerdict.model_json_schema(),
    }
    data = JudgeData(
        idx=solver_trace.task.data.idx,
        name=instance_id,
        prompt=JUDGE_PROMPT,
        network_allow=[],
        network_block=["*"],
        instance_id=instance_id,
        question=contract.question,
    )
    config = JudgeConfig(
        tools=SlackToolsetConfig.from_world(world, contract.actor_id),
        solver_trace=redact_secrets(solver_trace.to_record()),
        judge_context=context,
    )
    return JudgeTask(data, config)


__all__ = [
    "JUDGE_CONTEXT_FILE",
    "SOLVER_TRACE_FILE",
    "VERDICT_FILE",
    "JudgeTask",
    "make_judge_task",
]
