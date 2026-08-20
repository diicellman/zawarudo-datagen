from __future__ import annotations

import asyncio
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Self

import verifiers.v1 as vf
from pydantic import Field, model_validator
from verifiers.v1.configs.retries import RetryConfig
from verifiers.v1.harnesses.null.harness import NullHarnessConfig
from verifiers.v1.harnesses.rlm.harness import RLMHarnessConfig

from .agents.builder import BuilderTask, make_builder_task
from .agents.judge import make_judge_task
from .agents.solver import SolverTask, summarize_solver
from .agents.synthesizer import make_synthesizer_task, repair_prompt
from .contracts import (
    FailureOwner,
    GenerationResult,
    GenerationSeedData,
    ItemStatus,
    JudgeVerdict,
    QualityFilterConfig,
    authentication_failure,
    classify_item,
    decide_persistence,
    parse_synthesized_item,
    preflight_synthesized_item,
    redact_secrets,
)
from .dataset import DatasetWriter
from .slack.validation import MAX_SOURCE_BYTES, evaluate_candidate_in_runtime

RLM_REVISION = "e0080b25afddbc71ecf5476e4f50c3c63edddedb"
_IMMUTABLE_IMAGE = re.compile(r"^[^@\s]+@sha256:[0-9a-f]{64}$")
DEFAULT_IMAGE = "python:3.12-slim@sha256:2c941e860699f878900b0edc2403613c234d4b32eda3cc9fa7036991a2a63c4a"
_JUDGE_RETRY_TYPES = [
    "TaskError",
    "ProviderError",
    "SandboxError",
    "HarnessError",
    "InterceptionError",
    "TunnelError",
    "ToolsetError",
]
_OPERATIONAL_ERRORS = (
    vf.ProviderError,
    vf.SandboxError,
    vf.HarnessError,
    vf.RolloutError,
    vf.InterceptionError,
    vf.ToolsetError,
    vf.TunnelError,
    TimeoutError,
)


def _safe_operational_text(value: object, *, limit: int = 1_000) -> str:
    return str(redact_secrets(str(value)))[:limit]


def _safe_errors(errors: list[Any], *, limit: int = 4_000) -> str:
    rendered = "; ".join(
        f"{_safe_operational_text(error.type, limit=128)}: {_safe_operational_text(error.message)}"
        for error in errors[:20]
    )
    return rendered[:limit]


def _prime(*, cpu: float, memory: float, disk: float, labels: list[str]) -> vf.PrimeConfig:
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
        labels=labels,
    )


def _default_quality_filter() -> QualityFilterConfig:
    return QualityFilterConfig(
        min_accept_score=0.80,
        accepted_statuses={"solved"},
        retain_rejected_artifacts=False,
        weights={
            "scenario_alignment": 1.0,
            "world_coherence": 1.0,
            "professional_realism": 0.75,
            "discoverability": 1.25,
            "shortcut_free": 1.25,
        },
        minimum_scores={
            "scenario_alignment": 4,
            "world_coherence": 4,
            "professional_realism": 3,
            "discoverability": 4,
            "shortcut_free": 4,
        },
    )


class SlackDataGenerationConfig(vf.EnvConfig):
    max_concurrent_agents: int | None = 1
    synthesizer: vf.AgentConfig = vf.AgentConfig(
        model="z-ai/glm-5.2",
        harness=NullHarnessConfig(id="null"),
        runtime=_prime(cpu=1, memory=2, disk=5, labels=["worldgen-slack"]),
        sampling=vf.Sampling(temperature=0.5, max_tokens=8192),
        max_turns=5,
    )
    builder: vf.AgentConfig = vf.AgentConfig(
        model="z-ai/glm-5.2",
        harness=RLMHarnessConfig(
            id="rlm",
            version=RLM_REVISION,
            builtin_skills=["edit"],
        ),
        runtime=_prime(cpu=2, memory=4, disk=8, labels=["worldgen-slack"]),
        sampling=vf.Sampling(temperature=0.2, max_tokens=16384),
        max_turns=18,
    )
    solver: vf.AgentConfig = vf.AgentConfig(
        model="z-ai/glm-5.2",
        harness=NullHarnessConfig(id="null"),
        runtime=_prime(cpu=1, memory=2, disk=5, labels=["worldgen-slack"]),
        sampling=vf.Sampling(temperature=0.2, max_tokens=8192),
        max_turns=10,
    )
    judge: vf.AgentConfig = vf.AgentConfig(
        model="z-ai/glm-5.2",
        harness=RLMHarnessConfig(id="rlm", version=RLM_REVISION, builtin_skills=[]),
        runtime=_prime(cpu=1, memory=3, disk=5, labels=["worldgen-slack"]),
        sampling=vf.Sampling(temperature=0.1, max_tokens=8192),
        max_turns=24,
        retries=RetryConfig(max_retries=1, include=_JUDGE_RETRY_TYPES),
    )
    candidate_runtime: vf.PrimeConfig = _prime(
        cpu=2,
        memory=3,
        disk=8,
        labels=["worldgen-slack"],
    )
    hidden_seeds: list[int] = Field(default_factory=lambda: [101, 202], min_length=2)
    public_seed: Literal[0] = 0
    max_builder_checks: int = Field(default=3, ge=1, le=3)
    synthesizer_repair_attempts: int = Field(default=1, ge=0, le=1)
    min_gold_calls: int = Field(default=1, ge=1, le=8)
    near_duplicate_threshold: float = Field(default=0.92, gt=0.0, le=1.0)
    recent_summary_limit: int = Field(default=10, ge=0, le=50)
    candidate_timeout_seconds: float = Field(default=90.0, gt=0)
    public_check_timeout_seconds: float = Field(default=45.0, gt=0)
    output_dir: Path = Path("data/slack-v0")
    run_id: str = "slack-generator-v0"
    quality_filter: QualityFilterConfig = _default_quality_filter()
    runner_provenance: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_generation_policy(self) -> Self:
        if self.taskset.id != "worldgen-slack-generation":
            raise ValueError("generation taskset id must be worldgen-slack-generation")
        if len(set(self.hidden_seeds)) != len(self.hidden_seeds):
            raise ValueError("hidden_seeds must be distinct")
        if self.public_seed in self.hidden_seeds:
            raise ValueError("hidden_seeds must not include the public seed")
        images = {self.candidate_runtime.image}
        for name in ("synthesizer", "builder", "solver", "judge"):
            agent = getattr(self, name)
            if agent.model is None:
                raise ValueError(f"{name} must pin its model")
            runtime = agent.runtime
            if not isinstance(runtime, vf.PrimeConfig):
                raise TypeError(f"{name} runtime must be PrimeConfig")
            self._validate_prime(name, runtime)
            images.add(runtime.image)
        self._validate_prime("candidate_runtime", self.candidate_runtime)
        if len(images) != 1:
            raise ValueError("all roles and candidate execution must share one Prime image")
        if not isinstance(self.synthesizer.harness, NullHarnessConfig):
            raise TypeError("synthesizer must use the null harness")
        if not isinstance(self.solver.harness, NullHarnessConfig):
            raise TypeError("solver must use the null harness")
        for name in ("synthesizer", "builder", "solver", "judge"):
            harness = getattr(self, name).harness
            if harness.env or harness.forward_env or harness.skills:
                raise ValueError(f"{name} harness must not inject env or extra skills")
        if not isinstance(self.builder.harness, RLMHarnessConfig):
            raise TypeError("builder must use the RLM harness")
        if not re.fullmatch(r"[0-9a-f]{40}", self.builder.harness.version):
            raise ValueError("builder RLM revision must be a pinned 40-hex commit")
        if self.builder.harness.max_depth != 0:
            raise ValueError("builder RLM delegation must remain disabled")
        if self.builder.harness.builtin_skills != ["edit"]:
            raise ValueError("builder RLM may enable only the edit skill")
        if not isinstance(self.judge.harness, RLMHarnessConfig):
            raise TypeError("judge must use the RLM harness")
        if not re.fullmatch(r"[0-9a-f]{40}", self.judge.harness.version):
            raise ValueError("judge RLM revision must be a pinned 40-hex commit")
        if self.judge.harness.max_depth != 0:
            raise ValueError("judge RLM delegation must remain disabled")
        if self.judge.harness.builtin_skills:
            raise ValueError("judge RLM must not enable built-in skills")
        if self.judge.retries.max_retries != 1 or set(self.judge.retries.include) != set(_JUDGE_RETRY_TYPES):
            raise ValueError("judge must have exactly one protocol/operational retry")
        labels = ["worldgen-slack", self.run_id]
        for runtime in [
            self.synthesizer.runtime,
            self.builder.runtime,
            self.solver.runtime,
            self.judge.runtime,
            self.candidate_runtime,
        ]:
            runtime.labels = labels.copy()
        return self

    @staticmethod
    def _validate_prime(name: str, runtime: vf.PrimeConfig) -> None:
        if not runtime.vm:
            raise ValueError(f"{name} must use a Prime VM")
        if not _IMMUTABLE_IMAGE.fullmatch(runtime.image):
            raise ValueError(f"{name} image must use an immutable sha256 digest")
        if runtime.workdir != "/task":
            raise ValueError(f"{name} workdir must be /task")
        if runtime.allow or runtime.block != ["*"]:
            raise ValueError(f"{name} must use framework-only egress")


@dataclass
class PendingAttempt:
    result: GenerationResult
    signature_text: str | None = None
    source: str | None = None
    public_checks: list[dict[str, Any]] = field(default_factory=list)
    builder_metadata: dict[str, Any] = field(default_factory=dict)


class SlackDataGenerationEnv(vf.Env[SlackDataGenerationConfig]):
    def __init__(self, config: SlackDataGenerationConfig) -> None:
        super().__init__(config)
        self.writer = DatasetWriter(
            config.output_dir,
            run_id=config.run_id,
            quality_filter=config.quality_filter,
            prime_image=config.candidate_runtime.image,
            rlm_revision=config.builder.harness.version,
            environment_config=config.model_dump(mode="python", exclude={"output_dir"}),
        )
        self._pending: dict[int, PendingAttempt] = {}
        self._attempt_traces: dict[int, list[vf.Trace]] = {}
        self._auth_failure_traces: dict[int, list[vf.Trace]] = {}

    async def setup(self, agents: vf.Agents) -> None:
        for agent in agents:
            agent.trainable = False

    async def stop(self) -> None:
        self.writer.write_summary()

    async def run(self, task: vf.Task, agents: vf.Agents) -> None:
        if not isinstance(task.data, GenerationSeedData):
            raise TypeError("SlackDataGenerationEnv requires GenerationSeedData")
        seed = task.data
        previous = self._pending.pop(seed.generation_seed, None)
        previous_traces = self._attempt_traces.pop(seed.generation_seed, [])
        for trace in previous_traces:
            if trace.info.get("worldgen_tolerated_for_episode_retry"):
                trace.ok = False
        prior_errors = [error for trace in previous_traces for error in trace.errors]
        auth_errors = list(prior_errors)
        if (
            previous is not None
            and previous.result.failure_owner == FailureOwner.INFRASTRUCTURE
            and previous.result.reason
        ):
            auth_errors.append(
                vf.Error(
                    type=previous.result.status.value,
                    message=previous.result.reason,
                )
            )
        if auth := authentication_failure(auth_errors):
            message, status_code = auth
            safe_message = str(redact_secrets(message))[:1_000]
            self._auth_failure_traces[seed.generation_seed] = previous_traces
            self.writer.release_uncommitted_signatures(seed.generation_seed)
            raise vf.TaskError(f"HTTP status {status_code}: {safe_message}")
        if previous is not None:
            retry_errors = _safe_errors(prior_errors)
            retry_result = previous.result.model_copy(
                update={
                    "status": ItemStatus.INFRASTRUCTURE_ERROR,
                    "failure_owner": FailureOwner.INFRASTRUCTURE,
                    "reason": retry_errors or previous.result.reason,
                    "decision": None,
                }
            )
            self.writer.write_attempt(
                retry_result,
                signature_text=previous.signature_text,
                source=previous.source,
                traces=previous_traces,
                public_checks=previous.public_checks,
                builder_metadata=previous.builder_metadata,
                retry_discarded=True,
            )
        self.writer.release_uncommitted_signatures(seed.generation_seed)
        synthesized, synth_trace, synth_error = await self._run_synthesizer(
            seed,
            agents.synthesizer,
        )
        self._attempt_traces[seed.generation_seed] = [synth_trace]
        if not synth_trace.ok:
            self._pending[seed.generation_seed] = PendingAttempt(
                GenerationResult(
                    generation_seed=seed.generation_seed,
                    status=ItemStatus.INFRASTRUCTURE_ERROR,
                    failure_owner=FailureOwner.INFRASTRUCTURE,
                    reason=_trace_error(synth_trace),
                )
            )
            return
        if synthesized is None:
            self._pending[seed.generation_seed] = PendingAttempt(
                GenerationResult(
                    generation_seed=seed.generation_seed,
                    status=ItemStatus.SYNTHESIZER_FAILURE,
                    failure_owner=FailureOwner.SYNTHESIZER,
                    reason=synth_error,
                )
            )
            return

        reserved, signature, signature_text, duplicate_reason = self.writer.reserve_signature(
            synthesized,
            generation_seed=seed.generation_seed,
            near_duplicate_threshold=self.config.near_duplicate_threshold,
        )
        instance_id, _ = self.writer.identity(synthesized)
        if not reserved:
            self._pending[seed.generation_seed] = PendingAttempt(
                GenerationResult(
                    generation_seed=seed.generation_seed,
                    status=ItemStatus.DUPLICATE,
                    failure_owner=FailureOwner.SYNTHESIZER,
                    reason=duplicate_reason,
                    instance_id=instance_id,
                    signature=signature,
                    synthesized=synthesized,
                ),
                signature_text=signature_text,
            )
            return

        builder_task = make_builder_task(
            generation_seed=seed.generation_seed,
            interface_id=seed.interface_id,
            scenario=synthesized.scenario,
            contract=synthesized.task,
            public_seed=self.config.public_seed,
        )
        try:
            builder_trace, source, public_checks = await self._run_builder(
                agents.builder,
                builder_task,
            )
            self._attempt_traces[seed.generation_seed].append(builder_trace)
        except _OPERATIONAL_ERRORS as exc:
            self._pending[seed.generation_seed] = PendingAttempt(
                GenerationResult(
                    generation_seed=seed.generation_seed,
                    status=ItemStatus.INFRASTRUCTURE_ERROR,
                    failure_owner=FailureOwner.INFRASTRUCTURE,
                    reason=(f"{type(exc).__name__}: {_safe_operational_text(exc)}"),
                    instance_id=instance_id,
                    signature=signature,
                    synthesized=synthesized,
                ),
                signature_text=signature_text,
            )
            raise
        builder_metadata = _builder_metadata(builder_trace, source, public_checks)
        if not source.strip():
            infrastructure = not builder_trace.ok
            self._pending[seed.generation_seed] = PendingAttempt(
                GenerationResult(
                    generation_seed=seed.generation_seed,
                    status=(
                        ItemStatus.INFRASTRUCTURE_ERROR if infrastructure else ItemStatus.BUILDER_FAILURE
                    ),
                    failure_owner=(FailureOwner.INFRASTRUCTURE if infrastructure else FailureOwner.BUILDER),
                    reason=(
                        _trace_error(builder_trace)
                        if infrastructure
                        else "builder produced no nonempty world.py artifact"
                    ),
                    instance_id=instance_id,
                    signature=signature,
                    synthesized=synthesized,
                ),
                signature_text=signature_text,
                public_checks=public_checks,
                builder_metadata=builder_metadata,
            )
            return

        try:
            validation = await evaluate_candidate_in_runtime(
                source.encode(),
                synthesized.task,
                self.config.candidate_runtime,
                seeds=(self.config.public_seed, *self.config.hidden_seeds),
                timeout_seconds=self.config.candidate_timeout_seconds,
            )
        except vf.SandboxError as exc:
            self._pending[seed.generation_seed] = PendingAttempt(
                GenerationResult(
                    generation_seed=seed.generation_seed,
                    status=ItemStatus.INFRASTRUCTURE_ERROR,
                    failure_owner=FailureOwner.INFRASTRUCTURE,
                    reason=(f"{type(exc).__name__}: {_safe_operational_text(exc)}"),
                    instance_id=instance_id,
                    signature=signature,
                    synthesized=synthesized,
                ),
                signature_text=signature_text,
                source=source,
                public_checks=public_checks,
                builder_metadata=builder_metadata,
            )
            raise
        if not validation.ok:
            status = (
                ItemStatus.INFRASTRUCTURE_ERROR
                if validation.failure_owner == FailureOwner.INFRASTRUCTURE
                else ItemStatus.HARD_GATE_REJECTED
            )
            self._pending[seed.generation_seed] = PendingAttempt(
                GenerationResult(
                    generation_seed=seed.generation_seed,
                    status=status,
                    failure_owner=validation.failure_owner,
                    reason="; ".join(check.detail for check in validation.checks if not check.ok)[:4_000],
                    instance_id=instance_id,
                    signature=signature,
                    synthesized=synthesized,
                    validation=validation,
                ),
                signature_text=signature_text,
                source=source,
                public_checks=public_checks,
                builder_metadata=builder_metadata,
            )
            if validation.failure_owner == FailureOwner.INFRASTRUCTURE:
                raise vf.SandboxError("candidate validation infrastructure failure")
            return

        if not builder_trace.ok:
            builder_trace.info["worldgen_tolerated_for_episode_retry"] = True
            builder_trace.ok = True
        world = validation.public_snapshot
        if world is None:
            raise RuntimeError("successful validation omitted the public snapshot")
        solver_task = SolverTask.from_snapshot(
            instance_id=instance_id,
            contract=synthesized.task,
            world=world,
            idx=seed.generation_seed,
        )
        solver_trace = await agents.solver.run(solver_task)
        self._attempt_traces[seed.generation_seed].append(solver_trace)
        if not solver_trace.ok:
            self._pending[seed.generation_seed] = PendingAttempt(
                GenerationResult(
                    generation_seed=seed.generation_seed,
                    status=ItemStatus.INFRASTRUCTURE_ERROR,
                    failure_owner=FailureOwner.INFRASTRUCTURE,
                    reason=_trace_error(solver_trace),
                    instance_id=instance_id,
                    signature=signature,
                    synthesized=synthesized,
                    validation=validation,
                ),
                signature_text=signature_text,
                source=source,
                public_checks=public_checks,
                builder_metadata=builder_metadata,
            )
            return
        solver_summary = summarize_solver(solver_trace)
        judge_task = make_judge_task(
            instance_id=instance_id,
            scenario=synthesized.scenario,
            contract=synthesized.task,
            world=world,
            validation=validation,
            solver_trace=solver_trace,
            solver_summary=solver_summary,
            builder_metadata=builder_metadata,
        )
        judge_trace = await agents.judge.run(judge_task)
        self._attempt_traces[seed.generation_seed].append(judge_trace)
        if not judge_trace.ok:
            reason = _trace_error(judge_trace)
            protocol = any(
                term in reason.casefold() for term in ("verdict", "empirically recorded", "judge protocol")
            )
            self._pending[seed.generation_seed] = PendingAttempt(
                GenerationResult(
                    generation_seed=seed.generation_seed,
                    status=(
                        ItemStatus.JUDGE_PROTOCOL_FAILURE if protocol else ItemStatus.INFRASTRUCTURE_ERROR
                    ),
                    failure_owner=FailureOwner.INFRASTRUCTURE,
                    reason=reason,
                    instance_id=instance_id,
                    signature=signature,
                    synthesized=synthesized,
                    validation=validation,
                    solver=solver_summary,
                ),
                signature_text=signature_text,
                source=source,
                public_checks=public_checks,
                builder_metadata=builder_metadata,
            )
            operational = any(
                error.type in set(_JUDGE_RETRY_TYPES) - {"TaskError"} for error in judge_trace.errors
            )
            if operational:
                judge_trace.info["worldgen_tolerated_for_episode_retry"] = True
                judge_trace.ok = True
                raise vf.TaskError("judge operational retry exhausted")
            return
        try:
            verdict = JudgeVerdict.model_validate_json(json.dumps(judge_trace.info["judge_verdict"]))
        except (KeyError, ValueError) as exc:
            self._pending[seed.generation_seed] = PendingAttempt(
                GenerationResult(
                    generation_seed=seed.generation_seed,
                    status=ItemStatus.JUDGE_PROTOCOL_FAILURE,
                    failure_owner=FailureOwner.INFRASTRUCTURE,
                    reason=(f"{type(exc).__name__}: {_safe_operational_text(exc)}"),
                    instance_id=instance_id,
                    signature=signature,
                    synthesized=synthesized,
                    validation=validation,
                    solver=solver_summary,
                ),
                signature_text=signature_text,
                source=source,
                public_checks=public_checks,
                builder_metadata=builder_metadata,
            )
            return
        semantic_status = classify_item(validation, verdict)
        decision = decide_persistence(
            semantic_status,
            verdict,
            self.config.quality_filter,
        )
        item_status = _item_status(decision, verdict)
        self._pending[seed.generation_seed] = PendingAttempt(
            GenerationResult(
                generation_seed=seed.generation_seed,
                status=item_status,
                failure_owner=verdict.failure_owner,
                reason=decision.rejection_reason or verdict.reason,
                instance_id=instance_id,
                signature=signature,
                synthesized=synthesized,
                validation=validation,
                solver=solver_summary,
                verdict=verdict,
                decision=decision,
            ),
            signature_text=signature_text,
            source=source,
            public_checks=public_checks,
            builder_metadata=builder_metadata,
        )

    async def _run_synthesizer(
        self,
        seed: GenerationSeedData,
        agent: vf.Agent,
    ) -> tuple[Any | None, vf.Trace, str | None]:
        task = make_synthesizer_task(
            seed,
            self.writer.recent_summaries(self.config.recent_summary_limit),
        )
        synthesized = None
        error: Exception | None = None
        async with agent.interaction(task) as interaction:
            segment = await interaction.turn()
            for attempt in range(self.config.synthesizer_repair_attempts + 1):
                try:
                    synthesized = parse_synthesized_item(segment.last_reply)
                    preflight_synthesized_item(synthesized)
                    if len(synthesized.task.gold_calls) < self.config.min_gold_calls:
                        raise ValueError(f"gold path needs at least {self.config.min_gold_calls} calls")
                    error = None
                    break
                except (ValueError, TypeError) as exc:
                    synthesized = None
                    error = exc
                    if attempt >= self.config.synthesizer_repair_attempts or segment.terminated:
                        break
                    segment = await interaction.turn(repair_prompt(exc))
            trace = interaction.trace
        if synthesized is not None:
            trace.info["synthesized_item"] = synthesized.model_dump(mode="json")
        if error is not None:
            trace.info["synthesizer_error"] = f"{type(error).__name__}: {error}"
        return synthesized, trace, None if error is None else trace.info["synthesizer_error"]

    async def _run_builder(
        self,
        agent: vf.Agent,
        task: BuilderTask,
    ) -> tuple[vf.Trace, str, list[dict[str, Any]]]:
        source = b""
        attempts: list[dict[str, Any]] = []
        trace: vf.Trace | None = None
        interaction_error: Exception | None = None
        async with agent.provision(task) as runtime:
            try:
                async with agent.interaction(task, runtime=runtime) as interaction:
                    trace = interaction.trace
                    segment = await interaction.turn()
                    for attempt_number in range(1, self.config.max_builder_checks + 1):
                        source = await runtime.read(
                            "/task/workspace/world.py",
                            max_bytes=MAX_SOURCE_BYTES,
                        )
                        checked = await self._public_check(runtime)
                        attempts.append({"attempt": attempt_number, **checked})
                        if checked["ok"] or attempt_number == self.config.max_builder_checks:
                            break
                        if segment.terminated:
                            break
                        segment = await interaction.turn(
                            "Public checker feedback:\n"
                            + checked["report"]
                            + "\nRepair only /task/workspace/world.py, then rerun /task/check-world."
                        )
            except _OPERATIONAL_ERRORS as exc:
                interaction_error = exc
            try:
                source = await runtime.read(
                    "/task/workspace/world.py",
                    max_bytes=MAX_SOURCE_BYTES,
                )
            except vf.SandboxError:
                source = b""
        if trace is None:
            if interaction_error is not None:
                raise interaction_error
            raise vf.HarnessError("builder failed before minting a trace")
        trace.state.candidate_source = source.decode("utf-8", errors="replace")
        trace.state.public_attempts = attempts
        trace.state.conversational_completion = str(trace.stop_condition or "")
        trace.state.classification = "artifact_collected" if source.strip() else "empty_artifact"
        trace.info["public_checks"] = attempts
        trace.info["conversational_completion"] = trace.state.conversational_completion
        if interaction_error is not None:
            trace.info["builder_interaction_error"] = (
                f"{type(interaction_error).__name__}: {interaction_error}"
            )
        return trace, trace.state.candidate_source, attempts

    async def _public_check(self, runtime: vf.Runtime) -> dict[str, Any]:
        try:
            async with asyncio.timeout(self.config.public_check_timeout_seconds):
                process = await runtime.run(["/task/check-world"], {})
            raw = await runtime.read("/task/public-result.json", max_bytes=1_000_000)
            result = json.loads(raw)
        except TimeoutError:
            result = {"ok": False, "errors": ["public checker timed out"]}
            process = None
        except (vf.SandboxError, json.JSONDecodeError) as exc:
            result = {
                "ok": False,
                "errors": [f"{type(exc).__name__}: {_safe_operational_text(exc)}"],
            }
            process = None
        errors = [str(value) for value in result.get("errors", [])]
        if process is not None and process.exit_code not in (0, 2):
            errors.append((process.stderr or process.stdout)[-1_000:])
        return {
            "ok": bool(result.get("ok")) and not errors,
            "report": "\n".join(["RESULT PASS"] if not errors else ["RESULT FAIL", *errors]),
            "result": result,
        }

    def pop_auth_failure_traces(self, generation_seed: int) -> list[vf.Trace]:
        return self._auth_failure_traces.pop(generation_seed, [])

    async def persist_episode(self, task: vf.Task, episode: vf.Episode) -> dict[str, Any]:
        if not isinstance(task.data, GenerationSeedData):
            raise TypeError("cannot persist a non-generation task")
        seed = task.data.generation_seed
        for trace in episode.traces:
            if trace.info.get("worldgen_tolerated_for_episode_retry"):
                trace.ok = False
        pending = self._pending.pop(seed, None)
        self._attempt_traces.pop(seed, None)
        if pending is None:
            reason = _safe_errors(list(episode.errors))
            pending = PendingAttempt(
                GenerationResult(
                    generation_seed=seed,
                    status=ItemStatus.INFRASTRUCTURE_ERROR,
                    failure_owner=FailureOwner.INFRASTRUCTURE,
                    reason=reason or "episode ended without a generation result",
                )
            )
        if episode.errors:
            pending.builder_metadata["episode_errors"] = [
                {
                    "type": _safe_operational_text(error.type, limit=128),
                    "message": _safe_operational_text(error.message),
                }
                for error in list(episode.errors)[:20]
            ]
        return self.writer.write_attempt(
            pending.result,
            signature_text=pending.signature_text,
            source=pending.source,
            traces=episode.traces,
            public_checks=pending.public_checks,
            builder_metadata=pending.builder_metadata,
        )


def _trace_error(trace: vf.Trace) -> str:
    if not trace.errors:
        return f"{trace.agent.name if trace.agent else 'agent'} trace did not complete"
    return _safe_errors(list(trace.errors))


def _builder_metadata(
    trace: vf.Trace,
    source: str,
    public_checks: list[dict[str, Any]],
) -> dict[str, Any]:
    runtime_id = trace.agent.runtime.id if trace.agent and trace.agent.runtime else None
    duration = trace.timing.agent.duration if trace.timing and trace.timing.agent else 0.0
    return {
        "trace_id": trace.id,
        "runtime_id": runtime_id,
        "turns": trace.num_turns,
        "tokens": trace.num_total_tokens,
        "wall_seconds": duration,
        "source_lines": len(source.splitlines()),
        "source_hash": "sha256:" + hashlib.sha256(source.encode()).hexdigest(),
        "public_checker_attempts": len(public_checks),
        "conversational_completion": str(trace.stop_condition or ""),
        "trace_ok": trace.ok,
    }


def _item_status(decision: Any, verdict: JudgeVerdict) -> ItemStatus:
    if verdict.failure_owner == FailureOwner.INFRASTRUCTURE:
        return ItemStatus.INFRASTRUCTURE_ERROR
    if decision.write_to_dataset:
        return ItemStatus.SOLVED if decision.status == "solved" else ItemStatus.CHALLENGING
    if decision.status == "rejected" or (decision.rejection_reason or "").startswith("status_not_accepted"):
        return ItemStatus.SEMANTIC_REJECTED
    if decision.criterion_failures:
        return ItemStatus.CRITERION_FLOOR_REJECTED
    return ItemStatus.QUALITY_THRESHOLD_REJECTED


__all__ = [
    "DEFAULT_IMAGE",
    "RLM_REVISION",
    "SlackDataGenerationConfig",
    "SlackDataGenerationEnv",
]
