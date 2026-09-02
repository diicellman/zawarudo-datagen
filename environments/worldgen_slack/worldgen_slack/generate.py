from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import subprocess
import sys
import tomllib
from collections import Counter
from pathlib import Path
from typing import Any

import verifiers.v1 as vf
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from verifiers.v1.clients import EvalClientConfig, ModelContext

from .contracts import (
    AnswerKind,
    EvidenceLayout,
    FailedStage,
    FailureKind,
    FailureOwner,
    GenerationResult,
    GenerationSeedData,
    ItemStatus,
    ReasoningShape,
    ReleaseAcceptanceConfig,
    Workflow,
    decide_persistence,
    generation_result_from_episode,
    persistence_owner,
    preflight_synthesized_item,
    redact_secrets,
    release_item_status,
)
from .agents.synthesizer import (
    build_synthesis_schedule,
    synthesis_brief_at,
    synthesize,
)
from .dataset import DatasetWriter, validate_release_integrity
from .env import SlackDataGenerationConfig, SlackDataGenerationEnv
from .progress import ProgressJournal
from .slack.models import SynthesizedItem


class GenerationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str = "slack-qualification"
    default_model: str = "z-ai/glm-5.2"
    target_accepted: int = Field(default=10, ge=1)
    max_attempts: int = Field(default=25, ge=1)
    concurrency: int = Field(default=5, ge=1, le=16)
    seed_start: int = Field(default=0, ge=0)
    progress_interval_seconds: float = Field(default=10.0, gt=0, le=10)
    output: Path = Path("data/slack-qualification")
    near_duplicate_threshold: float = Field(default=0.92, gt=0.0, le=1.0)
    client: EvalClientConfig = EvalClientConfig()
    sampling: vf.Sampling = vf.Sampling(temperature=0.2, max_tokens=16_384)
    env: SlackDataGenerationConfig
    acceptance: ReleaseAcceptanceConfig = ReleaseAcceptanceConfig()

    @model_validator(mode="after")
    def validate_runner(self) -> "GenerationConfig":
        if self.target_accepted > self.max_attempts:
            raise ValueError("target_accepted cannot exceed max_attempts")
        if self.env.retries.max_retries != 0:
            raise ValueError("whole-episode retries must remain disabled")
        return self


class SynthesisDiagnosticConfig(vf.EnvConfig):
    retries: vf.RetryConfig = vf.RetryConfig(max_retries=0)
    max_concurrent_agents: int | None = 1
    synthesizer: vf.AgentConfig = vf.AgentConfig()
    brief_schedule_seed: int = Field(default=0, ge=0)


class SynthesisDiagnosticEnv(vf.Env[SynthesisDiagnosticConfig]):
    async def setup(self, agents: vf.Agents) -> None:
        agents.synthesizer.trainable = False

    async def run(self, task: vf.Task, agents: vf.Agents) -> None:
        if not isinstance(task.data, GenerationSeedData):
            raise TypeError("synthesis diagnostic requires GenerationSeedData")
        brief = synthesis_brief_at(
            self.config.brief_schedule_seed,
            task.data.generation_seed,
        )
        await synthesize(task.data, agents.synthesizer, brief)


def load_generation_config(path: str | Path, overrides: dict[str, Any] | None = None) -> GenerationConfig:
    config_path = Path(path).expanduser().resolve()
    try:
        values = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"could not load generation config: {config_path}") from exc
    values.update({key: value for key, value in (overrides or {}).items() if value is not None})
    return GenerationConfig.model_validate(values)


def _runner_provenance(config: GenerationConfig) -> dict[str, Any]:
    client = config.client.model_dump(mode="json")
    headers = client.pop("headers", {})
    return {
        "default_model": config.default_model,
        "client": client,
        "header_names": sorted(headers),
        "sampling": config.sampling.model_dump(mode="json"),
        "target_accepted": config.target_accepted,
        "max_attempts": config.max_attempts,
        "seed_start": config.seed_start,
        "concurrency": config.concurrency,
        "progress_interval_seconds": config.progress_interval_seconds,
        "retain_rejected_artifacts": config.acceptance.retain_rejected_artifacts,
        "environment": config.env.model_dump(mode="json"),
    }


def configured_models(config: GenerationConfig) -> set[str]:
    models = {config.default_model, config.env.solver_judge.model}
    models.update(getattr(config.env, role).model for role in ("synthesizer", "builder", "solver", "judge"))
    return {model for model in models if model}


def validate_required_models(required: set[str]) -> set[str]:
    try:
        completed = subprocess.run(
            ["prime", "inference", "models", "--output", "json", "--plain"],
            check=True,
            capture_output=True,
            text=True,
            timeout=45,
        )
        payload = json.loads(completed.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        raise RuntimeError("could not query Prime Inference models; run `prime login` and retry") from exc
    rows = payload.get("data", payload.get("models", [])) if isinstance(payload, dict) else []
    available = {
        row.get("id") if isinstance(row, dict) else row for row in rows if isinstance(row, (dict, str))
    }
    missing = sorted(required - available)
    if missing:
        raise ValueError(f"configured Prime Inference models are unavailable: {missing}")
    return required


def validate_model_availability(config: GenerationConfig) -> set[str]:
    return validate_required_models(configured_models(config))


async def _with_console_heartbeat(awaitable, *, label: str, interval: float):
    task = asyncio.ensure_future(awaitable)
    while True:
        try:
            return await asyncio.wait_for(asyncio.shield(task), timeout=interval)
        except TimeoutError:
            print(f"{label}: still active", flush=True)


def _episode_errors(episode: vf.Episode) -> list[vf.Error]:
    return [*episode.errors, *(error for trace in episode.traces for error in trace.errors)]


def _failed_episode_reason(episode: vf.Episode) -> str:
    errors = _episode_errors(episode)
    return "; ".join(f"{error.type}: {error.message}" for error in errors[:20])[:4_000] or (
        "episode failed without a recorded error"
    )


def _failed_trace_roles(episode: vf.Episode) -> set[str]:
    return {trace.agent.name for trace in episode.traces if trace.agent is not None and not trace.ok}


def _failure_stage(episode: vf.Episode) -> FailedStage:
    failed = _failed_trace_roles(episode)
    if len(failed) != 1:
        stages = {
            trace.info.get("current_pipeline_stage")
            for trace in episode.traces
            if trace.agent and trace.agent.name == "builder"
        }
        if stages == {"validation"}:
            return FailedStage.VALIDATION
        return FailedStage.INFRASTRUCTURE
    role = next(iter(failed))
    if role == "judge":
        return FailedStage.WORLD_JUDGE
    if role == "solver":
        joined = " ".join(f"{error.type} {error.message}" for error in _episode_errors(episode)).lower()
        if any(token in joined for token in ("judge", "verdict", "semantic", "scoring")):
            return FailedStage.SOLVER_ANSWER_JUDGE
        return FailedStage.SOLVER
    if role == "builder":
        return FailedStage.BUILDER
    if role == "synthesizer":
        return FailedStage.SYNTHESIS
    return FailedStage.INFRASTRUCTURE


def _failure_kind(episode: vf.Episode) -> FailureKind:
    errors = _episode_errors(episode)
    joined = " ".join(f"{error.type} {error.message}" for error in errors).lower()
    if any(error.status_code in {401, 403, 429} for error in errors) or any(
        token in joined for token in ("provider", "authentication", "http 401", "http 403", "rate limit")
    ):
        return FailureKind.PROVIDER_ERROR
    if any(token in joined for token in ("sandbox", "prime", "provision", "tunnel")):
        return FailureKind.SANDBOX_ERROR
    if any(
        token in joined
        for token in (
            "taskerror",
            "valueerror",
            "verdict",
            "no slack action",
            "no parsed",
            "no visible reply",
            "max_turns",
        )
    ):
        return FailureKind.PROTOCOL_ERROR
    return FailureKind.RUNTIME_ERROR


def _failure_owner(episode: vf.Episode, kind: FailureKind) -> FailureOwner:
    if kind != FailureKind.PROTOCOL_ERROR:
        return FailureOwner.INFRASTRUCTURE
    stage = _failure_stage(episode)
    if stage in {FailedStage.SOLVER, FailedStage.SOLVER_ANSWER_JUDGE}:
        return FailureOwner.SOLVER
    return FailureOwner.INTERFACE


def _release_result(
    result: GenerationResult,
    acceptance: ReleaseAcceptanceConfig,
) -> GenerationResult:
    if (
        result.validation is None
        or result.world_verdict is None
        or result.solver_score is None
        or result.builder_score is None
        or result.world_criteria is None
    ):
        return result
    decision = decide_persistence(result, acceptance)
    owner = persistence_owner(result, decision)
    rejected = not decision.write_to_dataset
    updated = result.model_copy(
        update={
            "status": release_item_status(decision),
            "failure_owner": owner,
            "failed_stage": (
                FailedStage.SOLVER
                if rejected and owner == FailureOwner.SOLVER
                else FailedStage.WORLD_JUDGE
                if rejected
                else None
            ),
            "failure_kind": FailureKind.QUALITY_REJECTION if rejected else None,
            "reason": decision.rejection_reason or result.reason,
            "decision": decision,
        }
    )
    return updated.consistent()


def _deduplicate_result(
    result: GenerationResult,
    writer: DatasetWriter,
    threshold: float,
) -> tuple[GenerationResult, str | None]:
    if result.synthesized is None:
        return result, None
    reserved, signature, signature_text, reason = writer.reserve_signature(
        result.synthesized,
        generation_seed=result.generation_seed,
        near_duplicate_threshold=threshold,
    )
    if reserved:
        return result.model_copy(update={"signature": signature}), signature_text
    decision = result.decision
    if decision is not None:
        decision = decision.model_copy(update={"write_to_dataset": False, "rejection_reason": reason})
    return (
        result.model_copy(
            update={
                "status": ItemStatus.DUPLICATE,
                "failure_owner": FailureOwner.SYNTHESIZER,
                "failed_stage": FailedStage.PERSISTENCE,
                "failure_kind": FailureKind.DUPLICATE,
                "reason": reason,
                "signature": signature,
                "decision": decision,
            }
        ),
        signature_text,
    )


def _builder_artifacts(
    episode: vf.Episode,
) -> tuple[str | None, list[dict[str, Any]], dict[str, Any]]:
    traces = episode.by_agent.get("builder", [])
    if not traces:
        return None, [], {}
    if len(traces) != 1:
        raise ValueError("generation episode has multiple final builder traces")
    trace = traces[0]
    source = trace.info.get("candidate_source")
    checks = trace.info.get("public_checks", [])
    metadata = trace.info.get("builder_metadata", {})
    if not isinstance(source, str) or not isinstance(checks, list) or not isinstance(metadata, dict):
        raise ValueError("builder trace contains malformed persistence artifacts")
    return source, checks, metadata


def _authentication_status(episode: vf.Episode) -> int | None:
    return next(
        (error.status_code for error in _episode_errors(episode) if error.status_code in {401, 403}),
        None,
    )


async def generate(
    config: GenerationConfig,
    *,
    check_models: bool = True,
) -> dict[str, Any]:
    print("startup: validating configured Prime Inference models", flush=True)
    if check_models:
        models = await _with_console_heartbeat(
            asyncio.to_thread(validate_model_availability, config),
            label="startup model validation",
            interval=config.progress_interval_seconds,
        )
    else:
        models = configured_models(config)

    writer = DatasetWriter(
        config.output,
        run_id=config.run_id,
        acceptance=config.acceptance,
        target_accepted=config.target_accepted,
        max_attempts=config.max_attempts,
        concurrency=config.concurrency,
        prime_image=config.env.candidate_runtime.image,
        rlm_revision=config.env.builder.harness.version,
        environment_config=_runner_provenance(config),
    )
    progress = ProgressJournal(
        config.output,
        run_id=config.run_id,
        target_accepted=config.target_accepted,
        max_attempts=config.max_attempts,
        concurrency=config.concurrency,
        interval_seconds=config.progress_interval_seconds,
        initial_accepted=writer.accepted_count,
        initial_attempted=writer.attempted_count,
    )
    for interrupted_seed in progress.interrupted_seeds(writer.completed_seeds):
        interrupted_data = GenerationSeedData(
            idx=interrupted_seed,
            name=f"seed-{interrupted_seed}",
            prompt="generate",
            network_allow=[],
            network_block=["*"],
            generation_seed=interrupted_seed,
            interface_id="slack.readonly.v1",
        )
        interrupted_episode = vf.Episode(
            task=vf.TraceTask(type="GenerationSeedTask", data=interrupted_data),
            errors=[
                vf.Error(
                    type="InterruptedAttempt",
                    message="attempt was active when the prior generator process stopped",
                )
            ],
        )
        await asyncio.to_thread(
            writer.write_failed_episode,
            generation_seed=interrupted_seed,
            episode=interrupted_episode,
            reason="attempt interrupted before transactional persistence",
            failed_stage=FailedStage.INFRASTRUCTURE,
            failure_kind=FailureKind.RUNTIME_ERROR,
            failure_owner=FailureOwner.INFRASTRUCTURE,
        )
        progress.recovered_attempt(interrupted_seed)

    env = SlackDataGenerationEnv(config.env, progress=progress)
    context = ModelContext(
        model=config.default_model,
        client=config.client,
        sampling=config.sampling,
    )
    task_iterator = iter(env.taskset)
    completed_seeds = writer.completed_seeds
    state = {
        "accepted": writer.accepted_count,
        "attempted": writer.attempted_count,
        "issued": writer.attempted_count,
    }
    dispatch_lock = asyncio.Lock()
    fatal_errors: list[str] = []

    async def next_task() -> vf.Task | None:
        async with dispatch_lock:
            if (
                fatal_errors
                or state["accepted"] >= config.target_accepted
                or state["issued"] >= config.max_attempts
            ):
                return None
            while True:
                task = next(task_iterator)
                if not isinstance(task.data, GenerationSeedData):
                    raise TypeError("generation taskset yielded an unexpected task type")
                seed = task.data.generation_seed
                if seed >= config.seed_start and seed not in completed_seeds:
                    break
            state["issued"] += 1
            progress.attempt_started(seed)
            return task

    async def persist(task: vf.Task, episode: vf.Episode) -> None:
        seed = task.data.generation_seed
        failure_reason = _failed_episode_reason(episode) if not episode.ok else None
        if not episode.ok:
            progress.fail_active(
                seed,
                error_type=_failure_kind(episode).value,
                detail=failure_reason,
            )
        progress.stage_started(seed, "persistence")

        def persist_episode() -> dict[str, Any]:
            if not episode.ok:
                failure_kind = _failure_kind(episode)
                return writer.write_failed_episode(
                    generation_seed=seed,
                    episode=episode,
                    reason=failure_reason or "episode failed without a reason",
                    failed_stage=_failure_stage(episode),
                    failure_kind=failure_kind,
                    failure_owner=_failure_owner(episode, failure_kind),
                )
            result = _release_result(
                generation_result_from_episode(episode),
                config.acceptance,
            )
            source, public_checks, builder_metadata = _builder_artifacts(episode)
            try:
                result, signature_text = _deduplicate_result(
                    result,
                    writer,
                    config.near_duplicate_threshold,
                )
                return writer.write_attempt(
                    result,
                    signature_text=signature_text,
                    source=source,
                    traces=episode.traces,
                    public_checks=public_checks,
                    builder_metadata=builder_metadata,
                )
            except BaseException:
                writer.release_uncommitted_signatures(seed)
                raise

        try:
            row = await asyncio.to_thread(persist_episode)
        except BaseException as exc:
            progress.stage_finished(seed, "persistence", ok=False, error_type=type(exc).__name__)
            raise
        accepted = bool(row["written_to_dataset"])
        state["attempted"] += 1
        state["accepted"] += int(accepted)
        completed_seeds.add(seed)
        progress.stage_finished(
            seed,
            "persistence",
            ok=True,
            status="accepted" if accepted else "rejected",
        )
        progress.attempt_finished(
            seed,
            status="accepted" if accepted else str(row["item_status"]),
            accepted=accepted,
            solver_score=row.get("solver_score"),
            builder_score=row.get("builder_score"),
            failure_detail=row.get("reason"),
        )
        if status := _authentication_status(episode):
            fatal_errors.append(
                "Prime authentication failed. Run `prime login` and verify Prime "
                f"Inference access (HTTP {status})."
            )

    async def worker() -> None:
        while task := await next_task():
            slot = env.slots(task)[0]
            await env.run_slot(slot, context, on_complete=lambda episode: persist(task, episode))

    await progress.start()
    print(
        "models "
        f"synth={config.env.synthesizer.model} builder={config.env.builder.model} "
        f"solver={config.env.solver.model} answer_judge={config.env.solver_judge.model} "
        f"world_judge={config.env.judge.model} validated={len(models)}",
        flush=True,
    )
    exit_reason = "error"
    try:
        if state["accepted"] < config.target_accepted and state["attempted"] < config.max_attempts:
            async with env.serving():
                async with asyncio.TaskGroup() as group:
                    for _ in range(config.concurrency):
                        group.create_task(worker())
        if fatal_errors:
            exit_reason = "fatal_infrastructure_error"
        elif state["accepted"] >= config.target_accepted:
            exit_reason = "target_reached"
        elif state["attempted"] >= config.max_attempts:
            exit_reason = "attempt_cap_exhausted"
        else:
            exit_reason = "stopped"
    finally:
        progress.finish_run(exit_reason)
        await progress.stop()

    writer.write_summary(exit_reason=exit_reason, release_integrity_ok=False)
    await asyncio.to_thread(validate_release_integrity, config.output)
    summary = writer.write_summary(exit_reason=exit_reason, release_integrity_ok=True)
    if fatal_errors:
        print(fatal_errors[0], file=sys.stderr, flush=True)
    return summary


def _trace_tokens(trace: vf.Trace) -> int:
    return sum(int(call.usage.total_tokens or 0) for call in trace.calls if call.usage is not None)


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


async def run_synthesis_diagnostic(
    config: GenerationConfig,
    *,
    attempts: int = 10,
    output: str | Path | None = None,
    check_models: bool = True,
) -> dict[str, Any]:
    if attempts < 1 or attempts > 100:
        raise ValueError("synthesis diagnostic attempts must be between 1 and 100")
    synth_models = {config.env.synthesizer.model} if config.env.synthesizer.model else set()
    print("synthesis diagnostic: validating synthesizer model", flush=True)
    if check_models:
        await _with_console_heartbeat(
            asyncio.to_thread(validate_required_models, synth_models),
            label="synthesis diagnostic model validation",
            interval=config.progress_interval_seconds,
        )
    root = Path(output or config.output.parent / f"{config.run_id}-synthesis-diagnostic")
    root = root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    diagnostic_config = SynthesisDiagnosticConfig(
        taskset=config.env.taskset,
        synthesizer=config.env.synthesizer,
        brief_schedule_seed=config.env.brief_schedule_seed,
    )
    env = SynthesisDiagnosticEnv(diagnostic_config)
    context = ModelContext(
        model=config.default_model,
        client=config.client,
        sampling=config.sampling,
    )
    tasks = list(env.taskset.head(attempts))
    semaphore = asyncio.Semaphore(min(config.concurrency, attempts))

    async def run_task(task: vf.Task) -> vf.Episode:
        return await env.run_slot(env.slots(task)[0], context, semaphore=semaphore)

    async def run_all() -> list[vf.Episode]:
        async with env.serving():
            async with asyncio.TaskGroup() as group:
                futures = [group.create_task(run_task(task)) for task in tasks]
        return [future.result() for future in futures]

    episodes = await _with_console_heartbeat(
        run_all(),
        label="synthesis diagnostic rollouts",
        interval=config.progress_interval_seconds,
    )

    rows: list[dict[str, Any]] = []
    valid_items: list[SynthesizedItem] = []
    schedule_matches = True
    no_builder_invoked = True
    for episode in episodes:
        seed = episode.task.data.generation_seed
        planned = synthesis_brief_at(config.env.brief_schedule_seed, seed)
        traces = episode.by_agent.get("synthesizer", [])
        no_builder_invoked &= set(episode.by_agent) <= {"synthesizer"}
        trace = traces[-1] if traces else None
        info = trace.info if trace is not None else {}
        actual_brief = info.get("synthesis_brief")
        schedule_matches &= actual_brief == planned.model_dump(mode="json")
        item_payload = info.get("synthesized_item")
        valid = bool(episode.ok and item_payload is not None)
        if valid:
            try:
                item = SynthesizedItem.model_validate_json(json.dumps(item_payload))
                preflight_synthesized_item(item, brief=planned)
            except (ValueError, ValidationError) as exc:
                valid = False
                info = {
                    **info,
                    "synthesizer_error": f"diagnostic revalidation failed: {exc}",
                }
            else:
                valid_items.append(item)
        errors = [*episode.errors]
        if trace is not None:
            errors.extend(trace.errors)
        reason = info.get("synthesizer_error") or "; ".join(
            f"{error.type}: {error.message}" for error in errors[:20]
        )
        raw_response = info.get("synthesizer_raw_response")
        raw_available = isinstance(raw_response, dict)
        rows.append(
            {
                "schema_version": 1,
                "generation_seed": seed,
                "brief": planned.model_dump(mode="json"),
                "valid": valid,
                "repair_count": int(info.get("synthesis_repairs", 0)),
                "failure_reason": (
                    str(redact_secrets(reason))[:4_000]
                    if not valid and reason
                    else "synthesis failed without a recorded reason"
                    if not valid
                    else None
                ),
                "raw_response_available": raw_available if not valid else None,
                "raw_response": raw_response if not valid and raw_available else None,
                "trace_id": trace.id if trace is not None else None,
                "tokens": _trace_tokens(trace) if trace is not None else 0,
            }
        )
    rows.sort(key=lambda row: row["generation_seed"])
    schedule = [row["brief"] for row in rows]
    axes = {
        "workflow": [item.value for item in Workflow],
        "reasoning_shape": [item.value for item in ReasoningShape],
        "evidence_layout": [item.value for item in EvidenceLayout],
        "answer_kind": [item.value for item in AnswerKind],
    }
    distribution: dict[str, dict[str, int]] = {}
    for field, values in axes.items():
        counts = Counter(brief[field] for brief in schedule)
        distribution[field] = {value: counts[value] for value in values}
    expected_schedule = [
        brief.model_dump(mode="json")
        for brief in build_synthesis_schedule(config.env.brief_schedule_seed, attempts)
    ]
    balanced = schedule == expected_schedule and all(
        max(counts.values()) - min(counts.values()) <= 1 for counts in distribution.values()
    )
    no_self_roots = all(
        evidence.thread_root_id != evidence.message_id
        for item in valid_items
        for evidence in item.task.required_evidence
        if evidence.thread_root_id is not None
    )
    failures = [row for row in rows if not row["valid"]]

    def retained_failure_is_valid(row: dict[str, Any]) -> bool:
        raw = row.get("raw_response")
        if row.get("raw_response_available") is not True or not isinstance(raw, dict):
            return False
        if set(raw) != {"text", "utf8_bytes", "sha256", "truncated"}:
            return False
        text = raw.get("text")
        total = raw.get("utf8_bytes")
        digest = raw.get("sha256")
        truncated = raw.get("truncated")
        if (
            not isinstance(text, str)
            or isinstance(total, bool)
            or not isinstance(total, int)
            or not isinstance(digest, str)
            or len(digest) != 64
            or not isinstance(truncated, bool)
            or not row.get("failure_reason")
        ):
            return False
        retained = text.encode("utf-8")
        if len(retained) > 65_536 or total < len(retained) or truncated != (total > 65_536):
            return False
        if not truncated and (total != len(retained) or hashlib.sha256(retained).hexdigest() != digest):
            return False
        return all(char in "0123456789abcdef" for char in digest)

    bounded_failures = all(retained_failure_is_valid(row) for row in failures)
    valid_count = sum(row["valid"] for row in rows)
    checks = {
        "yield_at_least_80_percent": valid_count / attempts >= 0.8,
        "brief_distribution_balanced": balanced and schedule_matches,
        "no_self_thread_roots": no_self_roots,
        "failed_responses_bounded_and_explained": bounded_failures,
        "builder_not_invoked": no_builder_invoked,
    }
    report = {
        "schema_version": 1,
        "run_id": config.run_id,
        "attempts": attempts,
        "valid": valid_count,
        "yield": valid_count / attempts,
        "passed": all(checks.values()),
        "checks": checks,
        "brief_distribution": distribution,
        "rows_path": "synthesis_diagnostic_attempts.jsonl",
        "models": sorted(synth_models),
    }
    rows_path = root / "synthesis_diagnostic_attempts.jsonl"
    temporary = rows_path.with_name(f".{rows_path.name}.tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(rows_path)
    _atomic_json(root / "synthesis_diagnostic.json", report)
    return {**report, "output": str(root)}


def _print_summary(summary: dict[str, Any]) -> None:
    role_usage = summary["role_usage"]
    lines = [
        ("run_id", summary["run_id"]),
        ("target accepted", summary["target_accepted"]),
        ("actual accepted", summary["actual_accepted"]),
        ("attempted", summary["attempted"]),
        ("attempt cap", summary["max_attempts"]),
        ("qualification passed", summary["qualification_passed"]),
        ("exit reason", summary["exit_reason"]),
        ("attempts per accepted world", summary["attempts_per_accepted_world"]),
        ("solver score mean", summary["solver_score_distribution"]["mean"]),
        ("builder score mean", summary["builder_score_distribution"]["mean"]),
        ("reported cost", summary["total_reported_cost"]),
        ("role usage", role_usage),
        ("output directory", summary["output"]),
    ]
    for label, value in lines:
        print(f"{label}: {value}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="worldgen-slack")
    subparsers = parser.add_subparsers(dest="command", required=True)
    command = subparsers.add_parser("generate", help="generate a Slack QA/world dataset")
    command.add_argument("--config", type=Path, required=True)
    command.add_argument("--target-accepted", type=int)
    command.add_argument("--max-attempts", type=int)
    command.add_argument("--concurrency", type=int)
    command.add_argument("--run-id")
    command.add_argument("--output", type=Path)
    diagnostic = subparsers.add_parser(
        "synthesis-diagnostic",
        help="run synthesis/schema/preflight only; never invoke the builder",
    )
    diagnostic.add_argument("--config", type=Path, required=True)
    diagnostic.add_argument("--attempts", type=int, default=10)
    diagnostic.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "synthesis-diagnostic":
        config = load_generation_config(args.config)
        report = asyncio.run(
            run_synthesis_diagnostic(
                config,
                attempts=args.attempts,
                output=args.output,
            )
        )
        for name in ("attempts", "valid", "yield", "passed", "brief_distribution", "output"):
            print(f"{name}: {report[name]}")
        return 0 if report["passed"] else 2
    config = load_generation_config(
        args.config,
        {
            "target_accepted": args.target_accepted,
            "max_attempts": args.max_attempts,
            "concurrency": args.concurrency,
            "run_id": args.run_id,
            "output": args.output,
        },
    )
    summary = asyncio.run(generate(config))
    _print_summary(summary)
    return 0 if summary["qualification_passed"] else 2


__all__ = [
    "GenerationConfig",
    "SynthesisDiagnosticConfig",
    "SynthesisDiagnosticEnv",
    "configured_models",
    "generate",
    "load_generation_config",
    "main",
    "run_synthesis_diagnostic",
    "validate_model_availability",
]


if __name__ == "__main__":
    raise SystemExit(main())
