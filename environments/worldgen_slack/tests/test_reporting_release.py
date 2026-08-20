from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import worldgen_slack.dataset as dataset_module
from conftest import WORLD_PROGRAM, make_contract, make_world
from worldgen_slack.agents.solver import SolverSummary
from worldgen_slack.contracts import (
    FailureOwner,
    GenerationResult,
    ItemStatus,
    JudgeVerdict,
    QualityFilterConfig,
    decide_persistence,
    synthesized_signature,
    synthesized_signature_text,
)
from worldgen_slack.dataset import DatasetWriter
from worldgen_slack.tasksets.generated import (
    GeneratedSlackTaskset,
    GeneratedSlackTasksetConfig,
)
from worldgen_slack.slack.validation import validate_compiled_snapshots


class FakeTrace:
    def __init__(self, role: str) -> None:
        self.id = f"trace-{role}"
        self.agent = SimpleNamespace(name=role, runtime=None)

    def to_record(self):
        return {"id": self.id, "agent": {"name": self.agent.name}, "nodes": []}


def quality(retain: bool = False) -> QualityFilterConfig:
    return QualityFilterConfig(
        min_accept_score=0.75,
        accepted_statuses={"solved"},
        retain_rejected_artifacts=retain,
        weights={
            "scenario_alignment": 1,
            "world_coherence": 1,
            "professional_realism": 1,
            "discoverability": 1,
            "shortcut_free": 1,
        },
        minimum_scores={
            "scenario_alignment": 3,
            "world_coherence": 3,
            "professional_realism": 3,
            "discoverability": 3,
            "shortcut_free": 3,
        },
    )


def accepted_result(synthesized) -> GenerationResult:
    snapshots = {
        "0": make_world(0).model_dump(mode="json"),
        "101": make_world(101).model_dump(mode="json"),
        "202": make_world(202).model_dump(mode="json"),
        "repeat_0": make_world(0).model_dump(mode="json"),
    }
    validation = validate_compiled_snapshots(snapshots, synthesized.task, [0, 101, 202])
    verdict = JudgeVerdict(
        solver_correct=True,
        task_unambiguous=True,
        world_supports_task=True,
        scenario_alignment=4,
        world_coherence=4,
        professional_realism=4,
        discoverability=4,
        shortcut_free=4,
        failure_owner=FailureOwner.NONE,
        reason="Verified against Slack evidence.",
    )
    decision = decide_persistence("solved", verdict, quality())
    instance_id, _ = DatasetWriter.identity(synthesized)
    return GenerationResult(
        generation_seed=0,
        status=ItemStatus.SOLVED,
        failure_owner=FailureOwner.NONE,
        instance_id=instance_id,
        signature=synthesized_signature(synthesized),
        synthesized=synthesized,
        validation=validation,
        solver=SolverSummary(
            final_answer_present=True,
            final_answer=synthesized.task.answer.canonical_answer,
            tool_call_count=2,
            tool_names=["slack_search_messages", "slack_get_thread"],
            visible_errors=[],
            trace_id="trace-solver",
        ),
        verdict=verdict,
        decision=decision,
    )


def writer(tmp_path, *, retain: bool = False) -> DatasetWriter:
    return DatasetWriter(
        tmp_path,
        run_id="fixture-run",
        quality_filter=quality(retain),
        prime_image="python:3.12-slim@sha256:2c941e860699f878900b0edc2403613c234d4b32eda3cc9fa7036991a2a63c4a",
        rlm_revision="e0080b25afddbc71ecf5476e4f50c3c63edddedb",
        environment_config={"fixture": True},
    )


def traces() -> list[FakeTrace]:
    return [FakeTrace(role) for role in ("synthesizer", "builder", "solver", "judge")]


def test_host_identity_does_not_trust_repeated_model_slug(synthesized) -> None:
    first, _ = DatasetWriter.identity(synthesized)
    changed_contract = make_contract(
        task_slug=synthesized.task.task_slug,
        question="Which action restored the payments service?",
    )
    changed = synthesized.model_copy(update={"task": changed_contract})
    second, _ = DatasetWriter.identity(changed)
    assert first != second
    assert first.startswith("payments-incident--")


def test_accepted_item_is_atomic_and_public_private_rows_are_one_to_one(tmp_path, synthesized) -> None:
    output = writer(tmp_path)
    result = accepted_result(synthesized)
    row = output.write_attempt(
        result,
        signature_text=synthesized_signature_text(synthesized),
        source=WORLD_PROGRAM,
        traces=traces(),
    )
    assert row["written_to_dataset"]
    world_dir = tmp_path / "worlds" / result.instance_id
    required = {
        "scenario.json",
        "qa.json",
        "task_contract.json",
        "world.py",
        "snapshot.json",
        "hidden_snapshot_hashes.json",
        "gold.json",
        "validation.json",
        "solver_trace.json",
        "judge_verdict.json",
        "manifest.json",
    }
    assert required <= {path.name for path in world_dir.iterdir()}
    assert not list((tmp_path / "worlds").glob(".*"))
    ids = []
    for name in ("dataset.jsonl", "public_tasks.jsonl", "private_oracles.jsonl"):
        rows = [json.loads(line) for line in (tmp_path / name).read_text().splitlines()]
        ids.append({item["instance_id"] for item in rows})
    assert ids[0] == ids[1] == ids[2] == {result.instance_id}


def test_generated_taskset_loads_snapshot_without_executing_world(tmp_path, synthesized) -> None:
    output = writer(tmp_path)
    result = accepted_result(synthesized)
    output.write_attempt(
        result,
        signature_text=synthesized_signature_text(synthesized),
        source="raise RuntimeError('must not execute')\n",
        traces=traces(),
    )
    taskset = GeneratedSlackTaskset(GeneratedSlackTasksetConfig(release_dir=tmp_path))
    loaded = list(taskset)
    assert len(loaded) == 1
    assert loaded[0].data.instance_id == result.instance_id


def test_rejected_attempt_never_partially_enters_public_dataset(tmp_path, synthesized) -> None:
    output = writer(tmp_path)
    result = GenerationResult(
        generation_seed=0,
        status=ItemStatus.HARD_GATE_REJECTED,
        failure_owner=FailureOwner.BUILDER,
        reason="invalid world",
        signature=synthesized_signature(synthesized),
        synthesized=synthesized,
    )
    row = output.write_attempt(
        result,
        signature_text=synthesized_signature_text(synthesized),
        source=WORLD_PROGRAM,
        traces=[FakeTrace("synthesizer"), FakeTrace("builder")],
    )
    assert not row["written_to_dataset"]
    assert (tmp_path / "public_tasks.jsonl").read_text() == ""
    assert not (tmp_path / "rejected").exists()
    assert not (tmp_path / "traces").exists()
    assert row["trace_refs"] == {}


def test_rejected_retention_is_separate_and_resume_is_insert_only(tmp_path, synthesized) -> None:
    output = writer(tmp_path, retain=True)
    rejected = GenerationResult(
        generation_seed=0,
        status=ItemStatus.BUILDER_FAILURE,
        failure_owner=FailureOwner.BUILDER,
        signature=synthesized_signature(synthesized),
        synthesized=synthesized,
    )
    row = output.write_attempt(
        rejected,
        signature_text=synthesized_signature_text(synthesized),
        source=WORLD_PROGRAM,
        traces=[FakeTrace("builder")],
    )
    assert row["artifact_ref"].startswith("rejected/")
    assert not list((tmp_path / "worlds").iterdir())
    resumed = writer(tmp_path, retain=True)
    assert resumed.completed_seeds == {0}


def test_manifest_tampering_and_cross_world_reference_fail_loudly(tmp_path, synthesized) -> None:
    output = writer(tmp_path)
    result = accepted_result(synthesized)
    output.write_attempt(
        result,
        signature_text=synthesized_signature_text(synthesized),
        source=WORLD_PROGRAM,
        traces=traces(),
    )
    snapshot = tmp_path / "worlds" / result.instance_id / "snapshot.json"
    snapshot.write_text(snapshot.read_text() + " ")
    with pytest.raises(ValueError, match="hash mismatch"):
        list(GeneratedSlackTaskset(GeneratedSlackTasksetConfig(release_dir=tmp_path)))


def test_run_manifest_is_secret_safe_and_binds_immutable_config(tmp_path) -> None:
    kwargs = {
        "run_id": "bound-run",
        "quality_filter": quality(),
        "prime_image": "python:3.12-slim@sha256:2c941e860699f878900b0edc2403613c234d4b32eda3cc9fa7036991a2a63c4a",
        "rlm_revision": "e0080b25afddbc71ecf5476e4f50c3c63edddedb",
    }
    environment = {
        "accepted": {"challenging", "solved"},
        "harness": {"env": {"UNUSUAL_NAME": "literal-secret"}},
    }
    DatasetWriter(tmp_path, environment_config=environment, **kwargs)
    manifest_text = (tmp_path / "run_manifest.json").read_text()
    assert "literal-secret" not in manifest_text
    DatasetWriter(tmp_path, environment_config=environment, **kwargs)
    changed = {**environment, "harness": {"env": {"UNUSUAL_NAME": "changed-secret"}}}
    with pytest.raises(ValueError, match="different immutable run config"):
        DatasetWriter(tmp_path, environment_config=changed, **kwargs)


def test_startup_recovers_an_uncommitted_attempt_journal(tmp_path) -> None:
    writer(tmp_path)
    attempt_id = "fixture-run--seed-00000000--attempt-0001"
    orphan = tmp_path / "worlds" / "orphan--000000000000"
    orphan.mkdir()
    sizes = {name: (tmp_path / name).stat().st_size for name in dataset_module.JSONL_FILES}
    for name in dataset_module.JSONL_FILES:
        with (tmp_path / name).open("a") as handle:
            handle.write("partial")
    journal = tmp_path / ".transactions" / f"{attempt_id}.json"
    journal.write_text(
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
    resumed = writer(tmp_path)
    assert resumed.completed_seeds == set()
    assert not orphan.exists()
    assert not journal.exists()
    assert all((tmp_path / name).stat().st_size == size for name, size in sizes.items())


def test_transaction_journal_rejects_path_traversal(tmp_path) -> None:
    writer(tmp_path)
    attempt_id = "fixture-run--seed-00000000--attempt-0001"
    sizes = {name: (tmp_path / name).stat().st_size for name in dataset_module.JSONL_FILES}
    journal = tmp_path / ".transactions" / f"{attempt_id}.json"
    journal.write_text(
        json.dumps(
            {
                "attempt_id": attempt_id,
                "instance_id": "../..",
                "world_existed": False,
                "rejected_existed": False,
                "jsonl_sizes": sizes,
            }
        )
    )
    sentinel = tmp_path.parent / "worldgen-sentinel"
    sentinel.write_text("safe")
    try:
        with pytest.raises(ValueError, match="unsafe instance_id"):
            writer(tmp_path)
        assert sentinel.read_text() == "safe"
    finally:
        sentinel.unlink(missing_ok=True)


def test_failed_multi_file_commit_rolls_back_rows_world_and_journal(
    tmp_path, synthesized, monkeypatch
) -> None:
    output = writer(tmp_path)
    result = accepted_result(synthesized)
    original = dataset_module._append_jsonl
    calls = 0

    def fail_second(path, row):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated append failure")
        return original(path, row)

    monkeypatch.setattr(dataset_module, "_append_jsonl", fail_second)
    with pytest.raises(OSError, match="simulated append failure"):
        output.write_attempt(
            result,
            signature_text=synthesized_signature_text(synthesized),
            source=WORLD_PROGRAM,
            traces=traces(),
        )
    assert not (tmp_path / "worlds" / result.instance_id).exists()
    assert not list((tmp_path / ".transactions").glob("*.json"))
    assert all((tmp_path / name).read_text() == "" for name in dataset_module.JSONL_FILES)


def test_whole_episode_retry_releases_uncommitted_signature(tmp_path, synthesized) -> None:
    output = writer(tmp_path)
    reserved, _, _, _ = output.reserve_signature(
        synthesized,
        generation_seed=0,
        near_duplicate_threshold=0.92,
    )
    assert reserved
    duplicate, _, _, _ = output.reserve_signature(
        synthesized,
        generation_seed=1,
        near_duplicate_threshold=0.92,
    )
    assert not duplicate
    output.release_uncommitted_signatures(0)
    reserved_after_retry, _, _, _ = output.reserve_signature(
        synthesized,
        generation_seed=1,
        near_duplicate_threshold=0.92,
    )
    assert reserved_after_retry


def test_retry_discarded_row_does_not_complete_seed(tmp_path) -> None:
    output = writer(tmp_path)
    failed = GenerationResult(
        generation_seed=0,
        status=ItemStatus.INFRASTRUCTURE_ERROR,
        failure_owner=FailureOwner.INFRASTRUCTURE,
        reason="transient provider failure",
    )
    output.write_attempt(
        failed,
        signature_text=None,
        source=None,
        traces=[],
        retry_discarded=True,
    )
    assert output.completed_seeds == set()
    resumed = writer(tmp_path)
    assert resumed.completed_seeds == set()
    resumed.write_attempt(
        failed,
        signature_text=None,
        source=None,
        traces=[],
    )
    assert resumed.completed_seeds == {0}


def test_managed_symlink_is_rejected_before_writes(tmp_path) -> None:
    external = tmp_path.parent / "external-worlds"
    external.mkdir(exist_ok=True)
    (tmp_path / "worlds").symlink_to(external, target_is_directory=True)
    try:
        with pytest.raises(ValueError, match="must not be a symlink"):
            writer(tmp_path)
        assert list(external.iterdir()) == []
    finally:
        (tmp_path / "worlds").unlink(missing_ok=True)
        external.rmdir()


def test_retained_rejected_artifacts_are_hash_bound(tmp_path, synthesized) -> None:
    output = writer(tmp_path, retain=True)
    rejected = GenerationResult(
        generation_seed=0,
        status=ItemStatus.BUILDER_FAILURE,
        failure_owner=FailureOwner.BUILDER,
        signature=synthesized_signature(synthesized),
        synthesized=synthesized,
    )
    row = output.write_attempt(
        rejected,
        signature_text=synthesized_signature_text(synthesized),
        source=WORLD_PROGRAM,
        traces=[FakeTrace("builder")],
    )
    result_path = tmp_path / row["artifact_ref"] / "result.json"
    result_path.write_text(result_path.read_text() + " ")
    with pytest.raises(ValueError, match="hash mismatch"):
        writer(tmp_path, retain=True)
