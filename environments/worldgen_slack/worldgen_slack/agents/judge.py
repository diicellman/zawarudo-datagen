from __future__ import annotations

import json
import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Self

import verifiers.v1 as vf
from pydantic import Field, model_validator

from ..contracts import ValidationReport, WorldJudgeVerdict, redact_secrets
from ..slack.models import INTERFACE_ID, QUALITY_CRITERIA, ScenarioSpec, SlackWorld, StrictModel, TaskContract
from ..slack.tools import SlackState, SlackToolset, SlackToolsetConfig

BUILDER_TRACE_FILE = "/tmp/worldgen_builder_trace.json"
WORLD_SOURCE_FILE = "/tmp/worldgen_world.py"
JUDGE_CONTEXT_FILE = "/tmp/worldgen_world_judge_context.json"
VERDICT_FILE = "/tmp/worldgen_world_judge_verdict.json"
_RUBRIC_PATH = Path(__file__).resolve().parents[1] / "slack" / "world_rubric.toml"
_CHOICE_SCORE = {
    "fails": 0.0,
    "weak": 0.25,
    "adequate": 0.5,
    "strong": 0.75,
    "exceptional": 1.0,
}

WORLD_JUDGE_PROMPT = f"""Grade only the generated Slack world and builder work.
Read the builder trace at `{BUILDER_TRACE_FILE}`, generated source at `{WORLD_SOURCE_FILE}`, and
rubric/context at `{JUDGE_CONTEXT_FILE}`. Empirically inspect the validated Slack world with at least
one Slack action. Do not trust supplied validation summaries as a substitute for inspection.
Write exactly one verdict JSON matching `verdict_schema` to `{VERDICT_FILE}`. The chat reply is not
the verdict. Do not assess or infer anything about a solver; no solver artifact is available.
"""


class RubricCriterion(StrictModel):
    name: str
    description: str
    weight: float = Field(ge=0.0)
    choices: list[str]


class WorldRubric(StrictModel):
    version: Literal[1]
    criteria: list[RubricCriterion]

    @model_validator(mode="after")
    def exact_contract(self) -> Self:
        names = [item.name for item in self.criteria]
        expected = ["task_unambiguous", "world_supports_task", *QUALITY_CRITERIA]
        if names != expected:
            raise ValueError(f"world rubric criteria must be exactly {expected}")
        for criterion in self.criteria[:2]:
            if criterion.weight != 0.0 or criterion.choices != ["no", "yes"]:
                raise ValueError("world hard-gate criteria require zero weight and no/yes choices")
        for criterion in self.criteria[2:]:
            if criterion.weight <= 0 or criterion.choices != list(_CHOICE_SCORE):
                raise ValueError("quality criteria require positive weight and ordered quality choices")
        return self


@lru_cache(maxsize=1)
def load_world_rubric() -> WorldRubric:
    return WorldRubric.model_validate(tomllib.loads(_RUBRIC_PATH.read_text(encoding="utf-8")))


def world_reward_scores(
    validation_ok: bool,
    verdict: WorldJudgeVerdict,
    rubric: WorldRubric | None = None,
) -> dict[str, Any]:
    rubric = rubric or load_world_rubric()
    weights = {item.name: item.weight for item in rubric.criteria[2:]}
    criteria = {name: _CHOICE_SCORE[getattr(verdict, name)] for name in QUALITY_CRITERIA}
    raw = sum(criteria[name] * weights[name] for name in QUALITY_CRITERIA) / sum(weights.values())
    hard_gates = {
        "task_unambiguous": verdict.task_unambiguous,
        "world_supports_task": verdict.world_supports_task,
    }
    hard_gate = validation_ok and all(hard_gates.values())
    return {
        "deterministic_validation": float(validation_ok),
        "hard_gates": hard_gates,
        "criteria": criteria,
        "world_quality_raw": raw,
        "world_quality": raw if hard_gate else 0.0,
    }


class WorldJudgeData(vf.TaskData):
    instance_id: str
    question: str
    interface_id: str = INTERFACE_ID


class WorldJudgeConfig(vf.TaskConfig):
    tools: SlackToolsetConfig = SlackToolsetConfig()
    builder_trace: dict[str, Any]
    source: str
    judge_context: dict[str, Any]


class WorldJudgeTask(vf.Task[WorldJudgeData, SlackState, WorldJudgeConfig]):
    NEEDS_CONTAINER = True

    @classmethod
    def toolsets(cls, config: WorldJudgeConfig) -> list[vf.Toolset]:
        return [SlackToolset(config.tools)]

    async def setup(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        paths = [VERDICT_FILE, BUILDER_TRACE_FILE, WORLD_SOURCE_FILE, JUDGE_CONTEXT_FILE]
        removed = await runtime.run(["rm", "-f", *paths], {})
        if removed.exit_code:
            raise RuntimeError(f"could not clear world-judge protocol files: {removed.stderr[-500:]}")
        await runtime.write(
            BUILDER_TRACE_FILE,
            json.dumps(
                self.config.builder_trace,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
            ).encode(),
        )
        await runtime.write(WORLD_SOURCE_FILE, self.config.source.encode())
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
        except vf.SandboxError as exc:
            raise ValueError(f"world judge wrote no bounded verdict file at {VERDICT_FILE}") from exc
        verdict = WorldJudgeVerdict.model_validate_json(raw)
        if not isinstance(trace.state, SlackState):
            raise TypeError("world judge trace requires SlackState")
        if not trace.state.completed_actions:
            raise ValueError("world judge completed no Slack action")
        trace.info["completed_actions"] = [
            action.model_dump(mode="json") for action in trace.state.completed_actions
        ]
        trace.info["world_verdict"] = verdict.model_dump(mode="json")


def _builder_trace_context(trace: vf.Trace) -> dict[str, Any]:
    record = trace.to_record()
    info = record.get("info") if isinstance(record.get("info"), dict) else {}
    return redact_secrets(
        {
            "version": record.get("version"),
            "id": record.get("id"),
            "agent": record.get("agent"),
            "nodes": record.get("nodes", []),
            "calls": record.get("calls", []),
            "errors": record.get("errors", []),
            "timing": record.get("timing"),
            "stop_condition": record.get("stop_condition"),
            "info": {key: info[key] for key in ("public_checks", "builder_metadata") if key in info},
        }
    )


def make_world_judge_task(
    *,
    generation_seed: int,
    instance_id: str,
    scenario: ScenarioSpec,
    contract: TaskContract,
    world: SlackWorld,
    validation: ValidationReport,
    builder_trace: vf.Trace,
    source: bytes,
) -> WorldJudgeTask:
    rubric = load_world_rubric()
    context = {
        "scenario": scenario.model_dump(mode="json"),
        "task_contract": contract.model_dump(mode="json"),
        "deterministic_validation": validation.model_dump(
            mode="json",
            exclude={"public_snapshot", "gold_call_log"},
        ),
        "rubric": rubric.model_dump(mode="json"),
        "verdict_schema": WorldJudgeVerdict.model_json_schema(),
    }
    data = WorldJudgeData(
        idx=generation_seed,
        name=instance_id,
        prompt=WORLD_JUDGE_PROMPT,
        network_allow=[],
        network_block=["*"],
        instance_id=instance_id,
        question=contract.question,
    )
    return WorldJudgeTask(
        data,
        WorldJudgeConfig(
            tools=SlackToolsetConfig.from_world(world, contract.actor_id),
            builder_trace=_builder_trace_context(builder_trace),
            source=source.decode("utf-8"),
            judge_context=context,
        ),
    )


__all__ = [
    "BUILDER_TRACE_FILE",
    "JUDGE_CONTEXT_FILE",
    "VERDICT_FILE",
    "WORLD_SOURCE_FILE",
    "WorldJudgeConfig",
    "WorldJudgeTask",
    "load_world_rubric",
    "make_world_judge_task",
    "world_reward_scores",
]
