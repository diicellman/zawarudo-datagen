from __future__ import annotations

import pytest
from pydantic import ValidationError

from worldgen_slack.slack.api import SlackAPI, SlackNotFoundError
from worldgen_slack.slack.models import SlackWorld

from conftest import deep_world_payload


def test_visibility_deletion_archive_and_public_shapes(world) -> None:
    actor = SlackAPI(world, "agent_user")
    visible = {item["conversation_id"] for item in actor.list_conversations()}
    assert visible == {"incidents", "dm"}
    assert actor.get_conversation_history("dm")[0]["message_id"] == "dm_msg"
    for hidden in ("private", "archived"):
        with pytest.raises(SlackNotFoundError):
            actor.get_conversation_history(hidden)

    results = actor.search_messages("secret answer confidential roadmap")
    ids = {item["message_id"] for item in results}
    assert "deleted" not in ids
    assert "private_msg" not in ids
    private_actor = SlackAPI(world, "alice")
    assert private_actor.get_conversation_history("private")[0]["message_id"] == "private_msg"

    private_keys = {
        "canonical_answer",
        "required_claims",
        "forbidden_claims",
        "required_evidence",
        "gold_calls",
    }
    for row in [*actor.search_messages("payments timeout"), *actor.get_thread("incidents", "root")]:
        assert not private_keys & set(row)


def test_search_history_and_thread_order_are_deterministic(world) -> None:
    api = SlackAPI(world, "agent_user")
    assert [row["message_id"] for row in api.get_thread("incidents", "root")] == [
        "root",
        "msg_cause",
        "msg_fix",
    ]
    history = [row["message_id"] for row in api.get_conversation_history("incidents")]
    assert history[:2] == ["newer", "root"]
    search = [row["message_id"] for row in api.search_messages("payments timeout")]
    assert search[:3] == ["newer", "msg_cause", "root"]
    assert len(api.search_messages("payments", limit=1)) == 1


def test_world_referential_membership_and_chronology_are_enforced(world) -> None:
    payload = deep_world_payload(world)
    next(item for item in payload["messages"] if item["id"] == "msg_fix")["author_id"] = "missing"
    with pytest.raises(ValidationError, match="unknown author"):
        SlackWorld.model_validate(payload)

    payload = deep_world_payload(world)
    next(item for item in payload["messages"] if item["id"] == "msg_fix")["timestamp"] = (
        "2025-01-15T09:59:00Z"
    )
    with pytest.raises(ValidationError, match="predates its thread root"):
        SlackWorld.model_validate(payload)

    payload = deep_world_payload(world)
    next(item for item in payload["conversations"] if item["id"] == "dm")["member_ids"].append("bob")
    with pytest.raises(ValidationError, match="dm conversations"):
        SlackWorld.model_validate(payload)
