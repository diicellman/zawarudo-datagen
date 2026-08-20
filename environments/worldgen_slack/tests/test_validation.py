from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from conftest import deep_world_payload, make_world
from worldgen_slack.contracts import (
    FailureOwner,
    JudgeVerdict,
    QualityFilterConfig,
    decide_persistence,
    normalized_quality_score,
    parse_synthesized_item,
    preflight_synthesized_item,
)
from worldgen_slack.slack.models import SlackWorld
from worldgen_slack.slack.validation import (
    static_source_errors,
    validate_compiled_snapshots,
    validate_world,
)


def test_valid_world_recovers_required_evidence_through_gold_calls(world, contract) -> None:
    report = validate_world(world, contract)
    assert report["ok"]
    assert len(report["gold_call_log"]) == 2
    assert all(item["ok"] for item in report["evidence"])


def test_archived_public_evidence_is_not_actor_visible(world, contract) -> None:
    payload = deep_world_payload(world)
    payload["conversations"][0]["is_archived"] = True
    report = validate_world(SlackWorld.model_validate(payload), contract)
    evidence = next(item for item in report["checks"] if item["name"] == "required_evidence")
    assert not evidence["ok"]
    assert "invisible" in evidence["detail"]


def test_source_policy_rejects_host_io_network_and_bad_signature() -> None:
    source = "import os\n\ndef build(other, contract):\n    return open('/tmp/x')\n"
    errors = static_source_errors(source)
    assert any("IMPORT" in error for error in errors)
    assert any("NAME" in error for error in errors)
    assert any("build arguments" in error for error in errors)

    bypass = """from worldgen_slack.slack.models import SlackWorld, TaskContract
importer = __builtins__["__import__"]
os = importer("os")
def build(seed, contract):
    os.system("touch /tmp/outside-task")
"""
    bypass_errors = static_source_errors(bypass)
    assert any("__builtins__" in error for error in bypass_errors)
    assert any("system" in error for error in bypass_errors)

    import_alias = """from random import _os as harmless
from worldgen_slack.slack.models import SlackWorld, TaskContract
def build(seed, contract):
    harmless.execv("/bin/sh", ["sh", "-c", "touch /tmp/outside-task"])
"""
    alias_errors = static_source_errors(import_alias)
    assert any("_os" in error for error in alias_errors)
    assert any("execv" in error for error in alias_errors)


def test_compiled_snapshots_require_determinism_variation_and_evidence_stability(contract) -> None:
    snapshots = {
        "0": make_world(0).model_dump(mode="json"),
        "101": make_world(101).model_dump(mode="json"),
        "202": make_world(202).model_dump(mode="json"),
        "repeat_0": make_world(0).model_dump(mode="json"),
    }
    report = validate_compiled_snapshots(snapshots, contract, [0, 101, 202])
    assert report.ok
    assert report.failure_owner == FailureOwner.NONE
    assert len(report.hidden_snapshot_hashes) == 2
    broken = json.loads(json.dumps(snapshots))
    message = next(item for item in broken["101"]["messages"] if item["id"] == "msg_cause")
    message["text"] += " changed"
    report = validate_compiled_snapshots(broken, contract, [0, 101, 202])
    assert not report.ok
    assert any(check.name == "hidden_seed_answer_stability" and not check.ok for check in report.checks)


def test_synthesized_parser_requires_exactly_one_strict_object(synthesized) -> None:
    parsed = parse_synthesized_item(synthesized.model_dump_json())
    assert parsed == synthesized
    with pytest.raises(ValueError, match="exactly one JSON object"):
        parse_synthesized_item("```json\n{}\n```")
    with pytest.raises(ValueError, match="strict JSON"):
        parse_synthesized_item('{"scenario": NaN}')


def test_preflight_rejects_self_rooted_and_contradictory_evidence(synthesized) -> None:
    payload = synthesized.model_dump(mode="json")
    payload["task"]["required_evidence"][0]["thread_root_id"] = "msg_cause"
    with pytest.raises(ValueError, match="own thread root"):
        preflight_synthesized_item(type(synthesized).model_validate(payload))
    payload = synthesized.model_dump(mode="json")
    payload["task"]["required_evidence"][1]["message_id"] = "msg_cause"
    payload["task"]["required_evidence"][1]["conversation_id"] = "dm_agent_alice"
    payload["task"]["min_distinct_evidence_messages"] = 1
    with pytest.raises(ValueError, match="contradictory"):
        preflight_synthesized_item(type(synthesized).model_validate(payload))


def _quality() -> QualityFilterConfig:
    return QualityFilterConfig(
        min_accept_score=0.75,
        accepted_statuses={"solved"},
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


def _verdict(**updates) -> JudgeVerdict:
    values = {
        "solver_correct": True,
        "task_unambiguous": True,
        "world_supports_task": True,
        "scenario_alignment": 4,
        "world_coherence": 4,
        "professional_realism": 4,
        "discoverability": 4,
        "shortcut_free": 4,
        "failure_owner": FailureOwner.NONE,
        "reason": "Verified.",
    }
    values.update(updates)
    return JudgeVerdict.model_validate(values)


def test_weighted_quality_and_inclusive_threshold() -> None:
    verdict = _verdict()
    assert normalized_quality_score(verdict, _quality().weights) == 0.75
    decision = decide_persistence("solved", verdict, _quality())
    assert decision.write_to_dataset


def test_criterion_floor_beats_high_average_and_status_policy() -> None:
    config = _quality()
    verdict = _verdict(
        scenario_alignment=2, world_coherence=5, professional_realism=5, discoverability=5, shortcut_free=5
    )
    decision = decide_persistence("solved", verdict, config)
    assert not decision.write_to_dataset
    assert decision.criterion_failures == ["scenario_alignment"]
    challenging = _verdict(solver_correct=False, failure_owner=FailureOwner.SOLVER)
    assert not decide_persistence("challenging", challenging, config).write_to_dataset
    config.accepted_statuses.add("challenging")
    assert decide_persistence("challenging", challenging, config).write_to_dataset
    infrastructure = _verdict(failure_owner=FailureOwner.INFRASTRUCTURE)
    assert not decide_persistence("solved", infrastructure, config).write_to_dataset


def test_invalid_quality_configuration_fails_at_startup() -> None:
    with pytest.raises(ValidationError, match="every quality criterion"):
        QualityFilterConfig(
            weights={"scenario_alignment": 1},
            minimum_scores={"scenario_alignment": 3},
        )
    with pytest.raises(ValidationError, match="cannot be empty"):
        QualityFilterConfig(
            accepted_statuses=set(),
            weights=_quality().weights,
            minimum_scores=_quality().minimum_scores,
        )
