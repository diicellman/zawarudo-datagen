from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest
import verifiers.v1 as vf

import worldgen_slack.dataset as dataset_module
from worldgen_slack.agents.judge import world_reward_scores
from worldgen_slack.agents.solver import SolverConfig
from worldgen_slack.contracts import (
    CheckResult,
    FailedStage,
    FailureKind,
    FailureOwner,
    GenerationResult,
    GenerationSeedData,
    ItemStatus,
    ReleaseAcceptanceConfig,
    SolverSummary,
    ValidationReport,
    WorldJudgeVerdict,
    decide_persistence,
    item_identity,
)
from worldgen_slack.dataset import (
    DatasetWriter,
    synthesized_signature,
    synthesized_signature_text,
    validate_release_integrity,
)
from worldgen_slack.generate import _release_result
from worldgen_slack.progress import ProgressJournal
from worldgen_slack.tasksets.generated import GeneratedSlackTaskset, GeneratedSlackTasksetConfig


def acceptance(*, retain: bool = True) -> ReleaseAcceptanceConfig:
    return ReleaseAcceptanceConfig(retain_rejected_artifacts=retain)


def verdict() -> WorldJudgeVerdict:
    return WorldJudgeVerdict(
        task_unambiguous=True,
        world_supports_task=True,
        scenario_alignment="exceptional",
        world_coherence="exceptional",
        professional_realism="strong",
        discoverability="exceptional",
        shortcut_free="exceptional",
        evidence_composition="exceptional",
        reason="Empirical inspection confirmed a coherent and discoverable world.",
    )


def seed_data(seed: int = 0) -> GenerationSeedData:
    return GenerationSeedData(
        idx=seed,
        name=f"seed-{seed}",
        prompt="generate",
        network_allow=[],
        network_block=["*"],
        generation_seed=seed,
        interface_id="slack.readonly.v1",
    )


def make_trace(role: str, seed: int = 0) -> vf.Trace:
    return vf.Trace(
        task=vf.TraceTask(type="GenerationSeedTask", data=seed_data(seed)),
        agent=vf.AgentInfo(config=vf.AgentConfig(model=f"test/{role}"), name=role),
        is_completed=True,
        ok=True,
    )


def role_traces(seed: int = 0) -> list[vf.Trace]:
    return [make_trace(role, seed) for role in ("synthesizer", "builder", "solver", "judge")]


def accepted_result(synthesized, world, traces: list[vf.Trace], seed: int = 0) -> GenerationResult:
    validation = ValidationReport(
        ok=True,
        failure_owner=FailureOwner.NONE,
        checks=[CheckResult(name="world", ok=True, detail="PASS")],
        public_snapshot=world,
        hidden_snapshot_hashes=["sha256:" + "1" * 64, "sha256:" + "2" * 64],
        gold_call_log=[],
    )
    world_verdict = verdict()
    world_scores = world_reward_scores(True, world_verdict)
    semantic_verdict = {
        "required_claims": [
            {"claim_index": 0, "grade": "supported", "reason": "present"},
            {"claim_index": 1, "grade": "supported", "reason": "present"},
        ],
        "forbidden_claim_indexes": [],
        "material_contradiction": False,
        "reason": "complete",
    }
    metrics = {
        "required_claim_coverage": 1.0,
        "contradiction_free": 1.0,
        "forbidden_claim_count": 0.0,
        "answer_present": 1.0,
    }
    solver_trace = next(trace for trace in traces if trace.agent.name == "solver")
    summary = SolverSummary(
        final_answer_present=True,
        final_answer=synthesized.task.answer.canonical_answer,
        completed_actions=[
            {
                "action": "get_thread",
                "arguments": {"conversation_id": "incidents", "root_message_id": "root"},
            }
        ],
        semantic_score=1.0,
        metrics=metrics,
        semantic_verdict=semantic_verdict,
        visible_errors=[],
        trace_id=solver_trace.id,
    )
    instance_id, _ = item_identity(synthesized)
    result = GenerationResult(
        generation_seed=seed,
        status=ItemStatus.SOLVED,
        failure_owner=FailureOwner.NONE,
        reason=world_verdict.reason,
        instance_id=instance_id,
        signature=synthesized_signature(synthesized),
        synthesized=synthesized,
        validation=validation,
        solver=summary,
        world_verdict=world_verdict,
        solver_score=1.0,
        solver_metrics=metrics,
        builder_score=world_scores["world_quality"],
        builder_raw_score=world_scores["world_quality_raw"],
        world_criteria=world_scores["criteria"],
        world_hard_gates=world_scores["hard_gates"],
        trace_ids={trace.agent.name: [trace.id] for trace in traces},
    )
    decision = decide_persistence(result, acceptance())
    by_role = {trace.agent.name: trace for trace in traces}
    by_role["synthesizer"].info["synthesized_item"] = synthesized.model_dump(mode="json")
    by_role["builder"].info["worldgen_validation"] = validation.model_dump(mode="json")
    by_role["builder"].info["world_reward"] = world_scores
    by_role["builder"].record_reward("deterministic_validation", 1.0, 0.0)
    by_role["builder"].record_reward("task_unambiguous", 1.0, 0.0)
    by_role["builder"].record_reward("world_supports_task", 1.0, 0.0)
    by_role["builder"].record_reward("world_quality_raw", world_scores["world_quality_raw"], 0.0)
    for name, score in world_scores["criteria"].items():
        by_role["builder"].record_reward(name, score, 0.0)
    by_role["builder"].record_reward("world_quality", world_scores["world_quality"], 1.0)
    by_role["solver"].info["solver_semantic_verdict"] = semantic_verdict
    by_role["solver"].info["solver_semantic_scores"] = {
        "semantic_correctness": 1.0,
        "required_claim_coverage": 1.0,
        "contradiction_free": 1.0,
        "forbidden_claim_count": 0.0,
    }
    by_role["solver"].info["solver_answer_judge_model"] = "test/answer-judge"
    by_role["solver"].record_reward("required_claim_coverage", 1.0, 0.0)
    by_role["solver"].record_reward("contradiction_free", 1.0, 0.0)
    by_role["solver"].record_reward("forbidden_claim_count", 0.0, 0.0)
    by_role["solver"].record_reward("semantic_correctness", 1.0, 1.0)
    by_role["judge"].info["world_verdict"] = world_verdict.model_dump(mode="json")
    by_role["judge"].info["completed_actions"] = [
        {"action": "get_thread", "arguments": {"conversation_id": "incidents", "root_message_id": "root"}}
    ]
    return result.model_copy(update={"decision": decision})


def writer(path: Path, *, retain: bool = True) -> DatasetWriter:
    return DatasetWriter(
        path,
        run_id="fixture-run",
        acceptance=acceptance(retain=retain),
        target_accepted=1,
        max_attempts=3,
        concurrency=1,
        prime_image="python:test@sha256:" + "a" * 64,
        rlm_revision="b" * 40,
        environment_config={"protocol": "synthesize-build-validate-solve-worldjudge"},
    )


def write_accepted(path: Path, synthesized, world) -> tuple[DatasetWriter, GenerationResult]:
    output = writer(path)
    traces = role_traces()
    result = accepted_result(synthesized, world, traces)
    output.write_attempt(
        result,
        signature_text=synthesized_signature_text(synthesized),
        source='raise RuntimeError("world.py must never run during generated evaluation")\n',
        traces=traces,
        public_checks=[{"ok": True}],
        builder_metadata={"turns": 1},
    )
    return output, result


def test_atomic_commit_public_private_isolation_and_snapshot_only_loading(
    tmp_path, synthesized, world
) -> None:
    output, result = write_accepted(tmp_path, synthesized, world)
    validate_release_integrity(tmp_path, require_tasks=True)
    public = json.loads((tmp_path / "public_tasks.jsonl").read_text())
    private = json.loads((tmp_path / "private_oracles.jsonl").read_text())
    attempt = json.loads((tmp_path / "attempts.jsonl").read_text())
    catalog = json.loads((tmp_path / "dataset.jsonl").read_text())
    artifacts = [json.loads(line) for line in (tmp_path / "artifacts.jsonl").read_text().splitlines()]
    assert json.loads((tmp_path / "manifest.json").read_text())["tables"] == {
        "attempts": {"path": "attempts.jsonl", "format": "jsonl"},
        "artifacts": {"path": "artifacts.jsonl", "format": "jsonl"},
        "progress": {"path": "progress.jsonl", "format": "jsonl"},
    }
    paths = {row["path"] for row in artifacts}
    assert {
        f"worlds/{result.instance_id}/snapshot.json",
        f"worlds/{result.instance_id}/solver_verdict.json",
        f"worlds/{result.instance_id}/world_verdict.json",
    } <= paths
    assert all(len(row["sha256"]) == 64 for row in artifacts)
    assert result.instance_id == public["instance_id"] == private["instance_id"]
    assert "answer" not in public and "required_claims" not in public
    assert private["answer"] == synthesized.task.answer.canonical_answer
    assert attempt["solver_score"] == 1.0
    assert attempt["builder_score"] >= 0.8
    assert attempt["world_criteria"] == result.world_criteria
    assert attempt["progress_path"] == "progress.jsonl"
    assert catalog["solver_verdict_ref"].endswith("/solver_verdict.json")
    assert catalog["world_verdict_ref"].endswith("/world_verdict.json")
    assert output.completed_seeds == {0}

    taskset = GeneratedSlackTaskset(GeneratedSlackTasksetConfig(release_dir=tmp_path, task=SolverConfig()))
    loaded = taskset.load()
    assert len(loaded) == 1
    assert loaded[0].data.prompt_text == synthesized.task.question
    assert loaded[0].config.oracle is not None
    assert synthesized.task.answer.canonical_answer not in loaded[0].config.model_dump_json()


def test_rejected_artifacts_failure_taxonomy_summary_and_resume(tmp_path, synthesized) -> None:
    output = writer(tmp_path, retain=True)
    result = GenerationResult(
        generation_seed=7,
        status=ItemStatus.HARD_GATE_REJECTED,
        failure_owner=FailureOwner.BUILDER,
        failed_stage=FailedStage.VALIDATION,
        failure_kind=FailureKind.DETERMINISTIC_REJECTION,
        reason="missing evidence",
        instance_id=item_identity(synthesized)[0],
        signature=synthesized_signature(synthesized),
        synthesized=synthesized,
        validation=ValidationReport(
            ok=False,
            failure_owner=FailureOwner.BUILDER,
            checks=[
                CheckResult(
                    name="required_evidence",
                    ok=False,
                    detail="missing",
                    failure_owner=FailureOwner.BUILDER,
                )
            ],
        ),
    )
    row = output.write_attempt(
        result,
        signature_text=synthesized_signature_text(synthesized),
        source="def build(seed, contract): return None\n",
        traces=[make_trace("synthesizer", 7), make_trace("builder", 7)],
    )
    assert not row["written_to_dataset"]
    assert (tmp_path / row["artifact_ref"]).is_dir()
    assert (tmp_path / "dataset.jsonl").read_text() == ""

    failed_seed = seed_data(8)
    failed_episode = vf.Episode(
        task=vf.TraceTask(type="GenerationSeedTask", data=failed_seed),
        ok=False,
        errors=[vf.Error(type="ProviderError", message="provider failed")],
    )
    output.write_failed_episode(
        generation_seed=8,
        episode=failed_episode,
        reason="provider failed",
        failed_stage=FailedStage.SYNTHESIS,
        failure_kind=FailureKind.PROVIDER_ERROR,
        failure_owner=FailureOwner.INFRASTRUCTURE,
    )
    resumed = writer(tmp_path, retain=True)
    assert resumed.completed_seeds == {7, 8}
    summary = resumed.write_summary(exit_reason="attempt_cap_exhausted")
    assert summary["rejections_by_stage"] == {"synthesis": 1, "validation": 1}
    assert summary["failure_kind_counts"] == {
        "deterministic_rejection": 1,
        "provider_error": 1,
    }
    assert summary["actual_accepted"] == 0


def test_transaction_recovery_and_failed_commit_rollback(tmp_path, synthesized, world, monkeypatch) -> None:
    output = writer(tmp_path)
    attempt_id = "fixture-run--seed-00000000--attempt-0001"
    orphan = tmp_path / "worlds" / "orphan--000000000000"
    orphan.mkdir()
    sizes = {name: (tmp_path / name).stat().st_size for name in dataset_module.JSONL_FILES}
    for name in dataset_module.JSONL_FILES:
        with (tmp_path / name).open("a") as handle:
            handle.write("partial")
    (tmp_path / ".transactions" / f"{attempt_id}.json").write_text(
        json.dumps(
            {
                "attempt_id": attempt_id,
                "instance_id": orphan.name,
                "world_existed": False,
                "rejected_existed": False,
                "jsonl_sizes": sizes,
            }
        )
    )
    output = writer(tmp_path)
    assert not orphan.exists()
    assert all((tmp_path / name).stat().st_size == size for name, size in sizes.items())

    original = dataset_module._append_jsonl
    calls = 0

    def fail_second(path, row):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated append failure")
        return original(path, row)

    monkeypatch.setattr(dataset_module, "_append_jsonl", fail_second)
    traces = role_traces()
    result = accepted_result(synthesized, world, traces)
    with pytest.raises(OSError, match="simulated append failure"):
        output.write_attempt(
            result,
            signature_text=synthesized_signature_text(synthesized),
            source="def build(seed, contract): return None\n",
            traces=traces,
        )
    assert not (tmp_path / "worlds" / result.instance_id).exists()
    assert not list((tmp_path / ".transactions").glob("*.json"))
    assert all((tmp_path / name).read_text() == "" for name in dataset_module.JSONL_FILES)


def test_tamper_path_traversal_and_managed_symlink_fail_loudly(tmp_path, synthesized, world) -> None:
    _, result = write_accepted(tmp_path, synthesized, world)
    snapshot = tmp_path / "worlds" / result.instance_id / "snapshot.json"
    snapshot.write_text("{}\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_release_integrity(tmp_path, require_tasks=True)

    other = tmp_path.parent / "symlink-release"
    other.mkdir()
    external = tmp_path.parent / "external-worlds"
    external.mkdir(exist_ok=True)
    (other / "worlds").symlink_to(external, target_is_directory=True)
    try:
        with pytest.raises(ValueError, match="must not be a symlink"):
            writer(other)
        assert list(external.iterdir()) == []
    finally:
        (other / "worlds").unlink(missing_ok=True)
        other.rmdir()
        external.rmdir()

    legacy = tmp_path.parent / "legacy-release"
    legacy.mkdir()
    (legacy / "run_manifest.json").write_text(
        json.dumps({"generator_schema_version": "worldgen-slack.dataset.v2"})
    )
    with pytest.raises(ValueError, match="incompatible dataset schema"):
        writer(legacy)
    assert not (legacy / "manifest.json").exists()


def test_qualification_report_requires_all_independent_gates(tmp_path, synthesized, world) -> None:
    output, _ = write_accepted(tmp_path, synthesized, world)
    progress = ProgressJournal(
        tmp_path,
        run_id="fixture-run",
        target_accepted=1,
        max_attempts=3,
        concurrency=1,
        interval_seconds=10,
        initial_accepted=0,
        initial_attempted=0,
    )
    progress.attempt_started(0)
    progress.stage_started(0, "evaluation")
    progress.event(0, "solver", "started")
    progress.event(0, "world_judge", "started")
    progress.event(0, "solver", "finished", ok=True)
    progress.event(0, "world_judge", "finished", ok=True)
    progress.stage_finished(0, "evaluation", ok=True)
    progress.attempt_finished(0, status="accepted", accepted=True)
    validate_release_integrity(tmp_path, require_tasks=True)
    summary = output.write_summary(exit_reason="target_reached", release_integrity_ok=True)
    report = json.loads((tmp_path / "qualification_report.json").read_text())
    assert summary["qualification_passed"] is True
    assert report["passed"] is True
    assert all(report["end_conditions"].values())
    assert report["world_rubric_hash"].startswith("sha256:")

    events = [json.loads(line) for line in (tmp_path / "progress.jsonl").read_text().splitlines()]
    first = datetime.fromisoformat(events[0]["timestamp_utc"].replace("Z", "+00:00"))
    events[1]["timestamp_utc"] = (first + timedelta(seconds=10.001)).isoformat()
    (tmp_path / "progress.jsonl").write_text(
        "".join(json.dumps(event, sort_keys=True) + "\n" for event in events)
    )
    delayed = output.write_summary(exit_reason="target_reached", release_integrity_ok=True)
    delayed_report = json.loads((tmp_path / "qualification_report.json").read_text())
    assert delayed["qualification_passed"] is False
    assert delayed_report["end_conditions"]["progress_heartbeat_within_interval"] is False

    events[0]["timestamp_utc"] = "not-an-iso-timestamp"
    (tmp_path / "progress.jsonl").write_text(
        "".join(json.dumps(event, sort_keys=True) + "\n" for event in events)
    )
    with pytest.raises(ValueError, match="invalid timestamp_utc"):
        output.write_summary(exit_reason="target_reached", release_integrity_ok=True)


def test_release_quality_and_configured_solver_threshold_have_mechanical_taxonomy(synthesized, world) -> None:
    traces = role_traces()
    result = accepted_result(synthesized, world, traces)
    floors = acceptance().minimum_world_scores | {"professional_realism": 1.0}
    rejected = _release_result(
        result,
        acceptance().model_copy(update={"minimum_world_scores": floors}),
    )
    assert rejected.status == ItemStatus.CRITERION_FLOOR_REJECTED
    assert rejected.failure_owner == FailureOwner.BUILDER
    assert rejected.failed_stage == FailedStage.WORLD_JUDGE
    assert rejected.failure_kind == FailureKind.QUALITY_REJECTION

    solver = result.solver.model_copy(
        update={
            "semantic_score": 0.75,
            "metrics": result.solver.metrics | {"required_claim_coverage": 0.75},
        }
    )
    raw = result.model_copy(
        update={
            "status": ItemStatus.CHALLENGING,
            "failure_owner": FailureOwner.SOLVER,
            "failed_stage": FailedStage.SOLVER,
            "failure_kind": FailureKind.QUALITY_REJECTION,
            "solver": solver,
            "solver_score": 0.75,
            "solver_metrics": solver.metrics,
            "decision": None,
        }
    ).consistent()
    accepted_at_configured_threshold = _release_result(
        raw,
        acceptance().model_copy(update={"min_solver_score": 0.75}),
    )
    assert accepted_at_configured_threshold.status == ItemStatus.SOLVED
    assert accepted_at_configured_threshold.failure_owner == FailureOwner.NONE
    assert accepted_at_configured_threshold.failed_stage is None
    assert accepted_at_configured_threshold.decision.write_to_dataset is True


def _rehash_world_artifact(root: Path, instance_id: str, filename: str) -> None:
    world_dir = root / "worlds" / instance_id
    target = world_dir / filename
    manifest_path = world_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifact_hashes"][filename] = "sha256:" + hashlib.sha256(target.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    relative_target = target.relative_to(root).as_posix()
    relative_manifest = manifest_path.relative_to(root).as_posix()
    rows = [json.loads(line) for line in (root / "artifacts.jsonl").read_text().splitlines()]
    for row in rows:
        if row["path"] == relative_target:
            row["sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
            row["size_bytes"] = target.stat().st_size
        elif row["path"] == relative_manifest:
            row["sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            row["size_bytes"] = manifest_path.stat().st_size
    (root / "artifacts.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def test_integrity_recomputes_named_rewards_and_empirical_actions(tmp_path, synthesized, world) -> None:
    solver_root = tmp_path / "solver"
    _, solver_result = write_accepted(solver_root, synthesized, world)
    solver_path = solver_root / "worlds" / solver_result.instance_id / "solver_trace.json"
    solver_record = json.loads(solver_path.read_text())
    solver_record["rewards"]["semantic_correctness"]["score"] = 0.0
    solver_path.write_text(json.dumps(solver_record, indent=2, sort_keys=True) + "\n")
    _rehash_world_artifact(solver_root, solver_result.instance_id, "solver_trace.json")
    with pytest.raises(ValueError, match="solver named reward mismatch"):
        validate_release_integrity(solver_root, require_tasks=True)

    judge_root = tmp_path / "judge"
    _, judge_result = write_accepted(judge_root, synthesized, world)
    judge_path = judge_root / "worlds" / judge_result.instance_id / "judge_trace.json"
    judge_record = json.loads(judge_path.read_text())
    judge_record["info"]["completed_actions"] = []
    judge_path.write_text(json.dumps(judge_record, indent=2, sort_keys=True) + "\n")
    _rehash_world_artifact(judge_root, judge_result.instance_id, "judge_trace.json")
    with pytest.raises(ValueError, match="no empirical Slack actions"):
        validate_release_integrity(judge_root, require_tasks=True)


def test_role_stats_include_discarded_judge_retry_usage_and_exact_retry_count() -> None:
    trace = make_trace("judge")
    trace.info["retry_count"] = 1
    trace.info["discarded_attempt_usage"] = [
        {
            "trace_id": "discarded",
            "model": "test/judge",
            "usage": vf.Usage(
                prompt_tokens=100,
                completion_tokens=20,
                cost=0.12,
            ).model_dump(mode="json"),
            "extra_usage": [],
            "duration_ms": 2_500,
        }
    ]
    stats = DatasetWriter._role_stats([trace], "judge")
    assert stats["tokens"] == 120
    assert stats["cost"] == pytest.approx(0.12)
    assert stats["duration_ms"] == 2_500
    assert stats["retry_count"] == 1
    assert stats["model"] == "test/judge"
