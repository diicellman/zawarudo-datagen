from __future__ import annotations

import json
import pytest
from pydantic import ValidationError

from conftest import deep_world_payload, make_contract
from worldgen_slack.contracts import FailureOwner, JudgeVerdict, redact_secrets
from worldgen_slack.slack.models import (
    AnswerSpec,
    Conversation,
    Message,
    SlackWorld,
    User,
)


def test_public_models_are_strict_and_forbid_extra_fields() -> None:
    with pytest.raises(ValidationError):
        User.model_validate({"id": "u1", "name": 123})
    with pytest.raises(ValidationError):
        User.model_validate({"id": "u1", "name": "One", "unknown": True})


def test_duplicate_and_unknown_references_are_rejected(world) -> None:
    payload = deep_world_payload(world)
    payload["users"].append(payload["users"][0])
    with pytest.raises(ValidationError, match="duplicate user id"):
        SlackWorld.model_validate(payload)
    payload = deep_world_payload(world)
    payload["messages"][0]["conversation_id"] = "missing"
    with pytest.raises(ValidationError, match="unknown conversation"):
        SlackWorld.model_validate(payload)


def test_message_authors_and_reactors_must_be_conversation_members(world) -> None:
    payload = deep_world_payload(world)
    private = next(item for item in payload["messages"] if item["id"] == "msg_budget")
    private["author_id"] = "user_dan"
    with pytest.raises(ValidationError, match="author is not a conversation member"):
        SlackWorld.model_validate(payload)


def test_reply_and_edit_chronology_is_enforced(world) -> None:
    payload = deep_world_payload(world)
    reply = next(item for item in payload["messages"] if item["id"] == "msg_cause")
    reply["timestamp"] = "2025-01-15T09:00:00Z"
    with pytest.raises(ValidationError, match="predates its thread root"):
        SlackWorld.model_validate(payload)
    with pytest.raises(ValidationError, match="precedes timestamp"):
        Message(
            id="m1",
            conversation_id="c1",
            author_id="u1",
            text="x",
            timestamp="2025-01-02T00:00:00Z",
            edited_timestamp="2025-01-01T00:00:00Z",
        )


def test_conversation_membership_shapes_are_strict() -> None:
    with pytest.raises(ValidationError, match="exactly two"):
        Conversation(id="d1", kind="dm", member_ids=["u1", "u2", "u3"])
    with pytest.raises(ValidationError, match="at least 3"):
        Conversation(id="g1", kind="group_dm", member_ids=["u1", "u2"])


def test_contract_rejects_duplicate_evidence_and_impossible_spread() -> None:
    contract = make_contract()
    payload = contract.model_dump(mode="json")
    payload["required_evidence"][1]["evidence_id"] = payload["required_evidence"][0]["evidence_id"]
    with pytest.raises(ValidationError, match="duplicate evidence IDs"):
        make_contract(required_evidence=payload["required_evidence"])
    with pytest.raises(ValidationError, match="exceeds specified evidence"):
        make_contract(min_distinct_evidence_messages=3)


def test_answer_claims_are_nonblank_and_noncontradictory() -> None:
    with pytest.raises(ValidationError, match="must not be blank"):
        AnswerSpec(
            kind="entity",
            canonical_answer="x",
            required_claims=["  "],
        )
    with pytest.raises(ValidationError, match="contradict"):
        AnswerSpec(
            kind="entity",
            canonical_answer="x",
            required_claims=["same"],
            forbidden_claims=["SAME"],
        )


def test_judge_verdict_consistency_is_strict() -> None:
    base = {
        "solver_correct": False,
        "task_unambiguous": True,
        "world_supports_task": True,
        "scenario_alignment": 4,
        "world_coherence": 4,
        "professional_realism": 4,
        "discoverability": 4,
        "shortcut_free": 4,
        "failure_owner": FailureOwner.SOLVER,
        "reason": "The solver missed supported evidence.",
    }
    assert JudgeVerdict.model_validate(base).failure_owner == FailureOwner.SOLVER
    with pytest.raises(ValidationError, match="requires solver_correct"):
        JudgeVerdict.model_validate({**base, "failure_owner": FailureOwner.NONE})
    with pytest.raises(ValidationError, match="ambiguous task"):
        JudgeVerdict.model_validate(
            {**base, "task_unambiguous": False, "failure_owner": FailureOwner.BUILDER}
        )


def test_trace_redaction_removes_header_and_environment_credentials() -> None:
    record = {
        "agent": {
            "config": {
                "client": {"headers": {"X-Custom": "raw-secret", "Authorization": "Bearer abc"}},
                "harness": {"env": {"ARBITRARY": "raw-secret", "SAFE": "also-secret"}},
            }
        },
        "message": "token=abcdefghijk",
    }
    redacted = redact_secrets(record)
    assert "raw-secret" not in json.dumps(redacted)
    assert "abcdefghijk" not in json.dumps(redacted)
