from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Literal, Self

import verifiers.v1 as vf
from pydantic import Field, model_validator
from verifiers.v1.configs.retries import RetryConfig
from verifiers.v1.harnesses.null.harness import NullHarnessConfig
from verifiers.v1.harnesses.rlm.harness import RLMHarnessConfig

from .agents.builder import build_world, make_builder_task
from .agents.judge import make_world_judge_task, world_reward_scores
from .agents.solver import SolverTask, summarize_solver
from .agents.synthesizer import synthesis_brief_at, synthesize
from .contracts import (
    CheckResult,
    FailureOwner,
    GenerationSeedData,
    ValidationReport,
    WorldJudgeVerdict,
    finalize_generation,
    item_identity,
)
from .progress import NullProgressJournal
from .slack.models import QUALITY_CRITERIA
from .slack.validate import evaluate_candidate_in_runtime

RLM_REVISION = "83ef01f7a6c97328919387343bd30cf4edaac20d"
DEFAULT_IMAGE = "python:3.12-slim@sha256:2c941e860699f878900b0edc2403613c234d4b32eda3cc9fa7036991a2a63c4a"
_IMMUTABLE_IMAGE = re.compile(r"^[^@\s]+@sha256:[0-9a-f]{64}$")


def _prime(*, cpu: float, memory: float, disk: float) -> vf.PrimeConfig:
    return vf.PrimeConfig(
        vm=True,
        image=DEFAULT_IMAGE,
        workdir="/task",
        allow=[],
        block=["*"],
        cpu=cpu,
        memory=memory,
        disk=disk,
        idle_timeout=900,
        labels=["worldgen-slack"],
    )


def _secure_prime(name: str, runtime: vf.RuntimeConfig) -> None:
    if not isinstance(runtime, vf.PrimeConfig):
        raise TypeError(f"{name} runtime must be PrimeConfig")
    if not runtime.vm:
        raise ValueError(f"{name} must use a Prime VM")
    if not _IMMUTABLE_IMAGE.fullmatch(runtime.image):
        raise ValueError(f"{name} image must use an immutable sha256 digest")
    if runtime.workdir != "/task":
        raise ValueError(f"{name} workdir must be /task")
    if runtime.allow or runtime.block != ["*"]:
        raise ValueError(f"{name} must use framework-only egress")


class SlackDataGenerationConfig(vf.EnvConfig):
    retries: RetryConfig = RetryConfig(max_retries=0)
    max_concurrent_agents: int | None = 2
    synthesizer: vf.AgentConfig = vf.AgentConfig(
        model="openai/gpt-5.6-luna",
        harness=NullHarnessConfig(id="null"),
        runtime=_prime(cpu=1, memory=2, disk=5),
        sampling=vf.Sampling(temperature=0.0, max_tokens=6_000),
        max_turns=3,
    )
    builder: vf.AgentConfig = vf.AgentConfig(
        model="openai/gpt-5.6-terra",
        harness=RLMHarnessConfig(
            id="rlm",
            version=RLM_REVISION,
            builtin_skills=["edit"],
        ),
        runtime=_prime(cpu=2, memory=4, disk=8),
        sampling=vf.Sampling(temperature=0.2, max_tokens=16_000),
        max_turns=18,
    )
    solver: vf.AgentConfig = vf.AgentConfig(
        model="z-ai/glm-5.2",
        harness=NullHarnessConfig(id="null"),
        runtime=_prime(cpu=1, memory=2, disk=5),
        sampling=vf.Sampling(temperature=0.1, max_tokens=6_000),
        max_turns=8,
    )
    solver_judge: vf.JudgeConfig = vf.JudgeConfig(
        model="openai/gpt-5.6-luna",
        sampling=vf.Sampling(temperature=0.0, max_tokens=2_000),
    )
    judge: vf.AgentConfig = vf.AgentConfig(
        model="openai/gpt-5.6-sol",
        harness=RLMHarnessConfig(id="rlm", version=RLM_REVISION, builtin_skills=[]),
        runtime=_prime(cpu=1, memory=3, disk=5),
        sampling=vf.Sampling(temperature=0.0, max_tokens=6_000),
        max_turns=10,
        retries=RetryConfig(
            max_retries=1,
            include=[
                "TaskError",
                "ProviderError",
                "SandboxError",
                "HarnessError",
                "InterceptionError",
                "TunnelError",
                "ToolsetError",
            ],
        ),
    )
    candidate_runtime: vf.PrimeConfig = _prime(cpu=2, memory=3, disk=8)
    hidden_seeds: list[int] = Field(default_factory=lambda: [101, 202], min_length=2)
    public_seed: Literal[0] = 0
    brief_schedule_seed: int = Field(default=0, ge=0)
    max_builder_checks: int = Field(default=3, ge=1, le=3)
    candidate_timeout_seconds: float = Field(default=90.0, gt=0)
    public_check_timeout_seconds: float = Field(default=45.0, gt=0)

    @model_validator(mode="after")
    def validate_security(self) -> Self:
        if self.taskset.id != "worldgen-slack-generation":
            raise ValueError("generation taskset id must be worldgen-slack-generation")
        if self.retries.max_retries != 0:
            raise ValueError("generation environment whole-episode retries must be zero")
        if len(set(self.hidden_seeds)) != len(self.hidden_seeds):
            raise ValueError("hidden_seeds must be distinct")
        if self.public_seed in self.hidden_seeds:
            raise ValueError("hidden_seeds must not include the public seed")
        for name in ("synthesizer", "builder", "solver", "judge"):
            agent = getattr(self, name)
            _secure_prime(name, agent.runtime)
            harness = agent.harness
            if harness and (harness.env or harness.forward_env or harness.skills):
                raise ValueError(f"{name} harness must not inject env or extra skills")
        _secure_prime("candidate_runtime", self.candidate_runtime)
        if not isinstance(self.synthesizer.harness, NullHarnessConfig):
            raise TypeError("synthesizer must use the null harness")
        if not isinstance(self.solver.harness, NullHarnessConfig):
            raise TypeError("solver must use the null harness")
        for name, skills in (("builder", ["edit"]), ("judge", [])):
            harness = getattr(self, name).harness
            if not isinstance(harness, RLMHarnessConfig):
                raise TypeError(f"{name} must use the RLM harness")
            if not re.fullmatch(r"[0-9a-f]{40}", harness.version):
                raise ValueError(f"{name} RLM revision must be a pinned 40-hex commit")
            if harness.max_depth != 0 or harness.builtin_skills != skills:
                raise ValueError(f"{name} RLM capabilities exceed the fixed protocol")
        if not self.solver_judge.model:
            raise ValueError("solver answer judge model must be explicit")
        return self


async def _validate_candidate(
    source: bytes,
    proposal,
    config: SlackDataGenerationConfig,
    *,
    generation_seed: int,
    progress,
) -> ValidationReport:
    if source.strip():
        return await evaluate_candidate_in_runtime(
            source,
            proposal.task,
            config.candidate_runtime,
            seeds=(config.public_seed, *config.hidden_seeds),
            timeout_seconds=config.candidate_timeout_seconds,
            on_seed_complete=lambda validation_seed, ok: progress.event(
                generation_seed,
                "validation",
                "seed_finished",
                ok=ok,
                validation_seed=validation_seed,
            ),
        )
    return ValidationReport(
        ok=False,
        failure_owner=FailureOwner.BUILDER,
        checks=[
            CheckResult(
                name="builder_artifact",
                ok=False,
                detail="builder produced an empty world.py",
                failure_owner=FailureOwner.BUILDER,
            )
        ],
    )


def _require_ok(trace: vf.Trace, role: str) -> vf.Trace:
    if not trace.ok:
        raise RuntimeError(f"{role} rollout failed")
    return trace


def _discarded_attempt_record(trace: vf.Trace) -> dict[str, object]:
    timing = trace.timing
    duration_ms = int(
        round(
            sum(
                float(getattr(getattr(timing, name, None), "duration", 0.0))
                for name in ("boot", "setup", "agent", "finalize", "scoring")
            )
            * 1_000
        )
    )
    return {
        "trace_id": trace.id,
        "model": trace.agent.config.model if trace.agent else None,
        "usage": trace.usage.model_dump(mode="json") if trace.usage else None,
        "extra_usage": [usage.model_dump(mode="json") for usage in trace.extra_usage],
        "duration_ms": duration_ms,
    }


async def _synthesis_stage(seed, config, agent, progress):
    number = seed.generation_seed
    brief = synthesis_brief_at(config.brief_schedule_seed, number)
    progress.stage_started(number, "synthesis", brief=brief.model_dump(mode="json"))
    proposal = await synthesize(seed, agent, brief, progress=progress)
    progress.stage_finished(number, "synthesis", ok=proposal is not None)
    return proposal


async def _builder_stage(seed, proposal, config, agent, progress):
    number = seed.generation_seed
    instance_id, _ = item_identity(proposal)
    task = make_builder_task(
        generation_seed=number,
        interface_id=seed.interface_id,
        scenario=proposal.scenario,
        contract=proposal.task,
        public_seed=config.public_seed,
    )
    progress.stage_started(number, "builder", instance_id=instance_id)
    builder, source = await build_world(
        task,
        agent,
        max_checks=config.max_builder_checks,
        check_timeout_seconds=config.public_check_timeout_seconds,
        progress=progress,
    )
    checks = builder.info.get("public_checks") or []
    progress.stage_finished(
        number,
        "builder",
        ok=bool(checks and checks[-1].get("ok")),
        checks=len(checks),
    )
    return instance_id, builder, source


async def _validation_stage(seed, proposal, source, builder, config, progress):
    number = seed.generation_seed
    progress.stage_started(number, "validation", seeds=1 + len(config.hidden_seeds))
    builder.info["current_pipeline_stage"] = "validation"
    validation = await _validate_candidate(
        source,
        proposal,
        config,
        generation_seed=number,
        progress=progress,
    )
    builder.info["worldgen_validation"] = validation.model_dump(mode="json")
    progress.stage_finished(
        number,
        "validation",
        ok=validation.ok,
        failed_check=next((check.name for check in validation.checks if not check.ok), None),
    )
    return validation


async def _solver_branch(
    *,
    number,
    instance_id,
    proposal,
    world,
    answer_judge,
    agent,
    progress,
):
    started = time.monotonic()
    progress.event(number, "solver", "started")
    trace = await agent.run(
        SolverTask.from_snapshot(
            instance_id=instance_id,
            contract=proposal.task,
            world=world,
            answer_judge=answer_judge,
            idx=number,
        )
    )
    progress.event(
        number,
        "solver",
        "finished",
        ok=trace.ok,
        duration_ms=int((time.monotonic() - started) * 1_000),
    )
    progress.event(
        number,
        "solver_judge",
        "finished",
        ok="solver_semantic_verdict" in trace.info,
        solver_score=(
            trace.rewards["semantic_correctness"].score
            if trace.rewards.get("semantic_correctness") is not None
            else None
        ),
    )
    return trace


async def _world_judge_branch(
    *,
    number,
    instance_id,
    proposal,
    world,
    validation,
    builder,
    source,
    agent,
    progress,
):
    started = time.monotonic()
    progress.event(number, "world_judge", "started", retry=0)
    attempts: list[vf.Trace] = []

    def observe_attempt(trace: vf.Trace) -> None:
        if attempts:
            progress.event(number, "world_judge", "retry", retry=len(attempts))
        attempts.append(trace)

    trace = await agent.run(
        make_world_judge_task(
            generation_seed=number,
            instance_id=instance_id,
            scenario=proposal.scenario,
            contract=proposal.task,
            world=world,
            validation=validation,
            builder_trace=builder,
            source=source,
        ),
        on_trace=observe_attempt,
    )
    retry_count = max(0, len(attempts) - 1)
    trace.info["retry_count"] = retry_count
    trace.info["discarded_attempt_usage"] = [_discarded_attempt_record(attempt) for attempt in attempts[:-1]]
    progress.event(
        number,
        "world_judge",
        "finished",
        ok=trace.ok,
        duration_ms=int((time.monotonic() - started) * 1_000),
        retry=retry_count,
    )
    return trace


async def _evaluation_stage(
    *,
    seed,
    instance_id,
    proposal,
    world,
    validation,
    builder,
    source,
    config,
    agents,
    progress,
):
    number = seed.generation_seed
    progress.stage_started(number, "evaluation")
    async with asyncio.TaskGroup() as group:
        solver_future = group.create_task(
            _solver_branch(
                number=number,
                instance_id=instance_id,
                proposal=proposal,
                world=world,
                answer_judge=config.solver_judge,
                agent=agents.solver,
                progress=progress,
            )
        )
        judge_future = group.create_task(
            _world_judge_branch(
                number=number,
                instance_id=instance_id,
                proposal=proposal,
                world=world,
                validation=validation,
                builder=builder,
                source=source,
                agent=agents.judge,
                progress=progress,
            )
        )
    solution = _require_ok(solver_future.result(), "solver")
    _require_ok(judge_future.result(), "world judge")
    solution.info["solver_summary"] = summarize_solver(solution).model_dump(mode="json")
    progress.stage_finished(number, "evaluation", ok=True)


class SlackDataGenerationEnv(vf.Env[SlackDataGenerationConfig]):
    async def setup(self, agents: vf.Agents) -> None:
        for agent in agents:
            agent.trainable = False

    def __init__(self, config: SlackDataGenerationConfig, *, progress=None) -> None:
        self.progress = progress or NullProgressJournal()
        super().__init__(config)

    async def run(self, task: vf.Task, agents: vf.Agents) -> None:
        if not isinstance(task.data, GenerationSeedData):
            raise TypeError("SlackDataGenerationEnv requires GenerationSeedData")
        seed = task.data
        proposal = await _synthesis_stage(seed, self.config, agents.synthesizer, self.progress)
        if proposal is None:
            return
        instance_id, builder, source = await _builder_stage(
            seed, proposal, self.config, agents.builder, self.progress
        )
        validation = await _validation_stage(seed, proposal, source, builder, self.config, self.progress)
        if not validation.ok:
            return
        world = validation.public_snapshot
        if world is None:
            raise RuntimeError("successful validation omitted the public snapshot")
        builder.info["current_pipeline_stage"] = "evaluation"
        await _evaluation_stage(
            seed=seed,
            instance_id=instance_id,
            proposal=proposal,
            world=world,
            validation=validation,
            builder=builder,
            source=source,
            config=self.config,
            agents=agents,
            progress=self.progress,
        )
        builder.info["current_pipeline_stage"] = "finalization"

    async def finalize(self, task: vf.Task, episode: vf.Episode) -> None:
        builder_traces = episode.by_agent.get("builder", [])
        if builder_traces:
            if len(builder_traces) != 1:
                raise ValueError("generation episode requires exactly one builder trace")
            builder = builder_traces[0]
            validation = ValidationReport.model_validate_json(
                json.dumps(builder.info.get("worldgen_validation"))
            )
            if validation.ok:
                judges = episode.by_agent.get("judge", [])
                if len(judges) != 1:
                    raise ValueError("validated world requires exactly one world-judge trace")
                verdict = WorldJudgeVerdict.model_validate_json(
                    json.dumps(judges[0].info.get("world_verdict"))
                )
                scores = world_reward_scores(True, verdict)
                judges[0].record_metrics(
                    {
                        "task_unambiguous": float(verdict.task_unambiguous),
                        "world_supports_task": float(verdict.world_supports_task),
                        **{f"{name}_normalized": scores["criteria"][name] for name in QUALITY_CRITERIA},
                    }
                )
            else:
                scores = {
                    "deterministic_validation": 0.0,
                    "hard_gates": {
                        "task_unambiguous": False,
                        "world_supports_task": False,
                    },
                    "criteria": dict.fromkeys(QUALITY_CRITERIA, 0.0),
                    "world_quality_raw": 0.0,
                    "world_quality": 0.0,
                }
            builder.info["world_reward"] = scores
            builder.record_reward("deterministic_validation", scores["deterministic_validation"], 0.0)
            builder.record_reward("task_unambiguous", float(scores["hard_gates"]["task_unambiguous"]), 0.0)
            builder.record_reward(
                "world_supports_task",
                float(scores["hard_gates"]["world_supports_task"]),
                0.0,
            )
            builder.record_reward("world_quality_raw", scores["world_quality_raw"], 0.0)
            for name in QUALITY_CRITERIA:
                builder.record_reward(name, scores["criteria"][name], 0.0)
            builder.record_reward("world_quality", scores["world_quality"], 1.0)

        result = finalize_generation(task, episode)
        result_trace = builder_traces[0] if builder_traces else episode.by_agent["synthesizer"][0]
        if any("generation_result" in trace.info for trace in episode.traces):
            raise ValueError("generation result was already attached")
        result_trace.info["generation_result"] = result.model_dump(mode="json")


__all__ = [
    "DEFAULT_IMAGE",
    "RLM_REVISION",
    "SlackDataGenerationConfig",
    "SlackDataGenerationEnv",
]
