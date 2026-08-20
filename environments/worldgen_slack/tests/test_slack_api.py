from __future__ import annotations

import json

import pytest
from mcp.server.fastmcp import FastMCP

from conftest import make_world
from worldgen_slack.slack.models import Conversation, Message, SlackWorld, User
from worldgen_slack.slack import (
    MAX_HISTORY_RESULTS,
    MAX_THREAD_RESULTS,
    SlackAPI,
    SlackNotFoundError,
    SlackToolset,
    SlackToolsetConfig,
)


def test_private_channel_membership_is_enforced_for_every_read(world):
    api = SlackAPI(world, "agent_user")
    visible_ids = {item["conversation_id"] for item in api.list_conversations()}
    assert "channel_leadership" not in visible_ids
    assert not api.is_conversation_visible("channel_leadership")
    with pytest.raises(SlackNotFoundError, match="not found or not visible"):
        api.get_conversation_history("channel_leadership")


def test_search_never_returns_private_or_deleted_messages(world):
    payload = world.model_dump(mode="json")
    payload["messages"].append(
        {
            "id": "msg_deleted_public",
            "conversation_id": "channel_engineering",
            "author_id": "user_alice",
            "text": "confidential roadmap deleted copy",
            "timestamp": "2025-01-16T00:00:00Z",
            "deleted": True,
        }
    )
    api = SlackAPI(SlackWorld.model_validate(payload), "agent_user")
    results = api.search_messages("confidential roadmap")
    assert results == []


def test_search_ranking_is_deterministic_and_documented(world):
    api = SlackAPI(world, "agent_user")
    first = api.search_messages("payments timeout")
    assert first == api.search_messages("payments timeout")
    assert [row["message_id"] for row in first[:3]] == [
        "msg_cause",
        "msg_incident_root",
        "msg_fix",
    ]
    assert all(
        set(row) == {"message_id", "conversation_id", "author_id", "timestamp", "thread_root_id", "text"}
        for row in first
    )


def test_history_and_thread_are_ordered_as_documented(world):
    api = SlackAPI(world, "agent_user")
    history = api.get_conversation_history("channel_incidents")
    assert [row["message_id"] for row in history] == ["msg_monitoring", "msg_incident_root"]

    thread = api.get_thread("channel_incidents", "msg_incident_root")
    assert [row["message_id"] for row in thread] == [
        "msg_incident_root",
        "msg_cause",
        "msg_fix",
    ]


def test_history_and_thread_outputs_are_bounded():
    users = [User(id="actor", name="Actor")]
    channel = Conversation(id="channel", name="channel", kind="public_channel", member_ids=["actor"])
    roots = [
        Message(
            id=f"root_{number}",
            conversation_id="channel",
            author_id="actor",
            text=f"root number {number}",
            timestamp=f"2025-01-01T00:00:{number:02d}Z",
        )
        for number in range(55)
    ]
    world = SlackWorld(users=users, conversations=[channel], messages=roots)
    api = SlackAPI(world, "actor")
    result = api.get_conversation_history("channel")
    assert len(result) == MAX_HISTORY_RESULTS
    assert result[0]["message_id"] == "root_54"

    root = Message(
        id="thread_root",
        conversation_id="channel",
        author_id="actor",
        text="thread root",
        timestamp="2025-01-02T00:00:00Z",
    )
    replies = [
        Message(
            id=f"reply_{number}",
            conversation_id="channel",
            author_id="actor",
            text=f"reply number {number}",
            timestamp=f"2025-01-02T00:00:{number:02d}Z",
            thread_root_id="thread_root",
        )
        for number in range(55)
    ]
    threaded = SlackWorld(users=users, conversations=[channel], messages=[root, *replies])
    result = SlackAPI(threaded, "actor").get_thread("channel", "thread_root")
    assert len(result) == MAX_THREAD_RESULTS
    assert result[0]["message_id"] == "thread_root"
    assert result[-1]["message_id"] == "reply_48"


async def _schemas(world: SlackWorld) -> list[dict]:
    toolset = SlackToolset(SlackToolsetConfig.from_world(world, "agent_user"))
    mcp = FastMCP("schema-test")
    toolset.register(mcp)
    return [tool.model_dump(mode="json") for tool in await mcp.list_tools()]


@pytest.mark.asyncio
async def test_tool_schemas_are_identical_across_task_worlds(world):
    other_world = make_world(seed=6)
    assert await _schemas(world) == await _schemas(other_world)

    schemas = await _schemas(world)
    names = {schema["name"] for schema in schemas}
    assert names == {
        "list_conversations",
        "search_messages",
        "get_conversation_history",
        "get_thread",
        "get_user",
    }
    assert SlackToolset(SlackToolsetConfig.from_world(world, "agent_user")).server_name == "slack"


@pytest.mark.asyncio
async def test_tool_outputs_never_contain_private_contract_fields(world):
    toolset = SlackToolset(SlackToolsetConfig.from_world(world, "agent_user"))
    await toolset.setup()
    outputs = [
        await toolset.list_conversations(),
        await toolset.search_messages("payments timeout"),
        await toolset.get_conversation_history("channel_incidents"),
        await toolset.get_thread("channel_incidents", "msg_incident_root"),
        await toolset.get_user("user_alice"),
    ]
    rendered = json.dumps(outputs, sort_keys=True).casefold()
    for private_name in (
        "required_evidence",
        "gold_calls",
        "canonical_answer",
        "forbidden_claims",
        "task_contract",
    ):
        assert private_name not in rendered


def test_limit_and_identifier_inputs_are_strict(world):
    api = SlackAPI(world, "agent_user")
    with pytest.raises(TypeError, match="integer"):
        api.search_messages("payments", True)
    with pytest.raises(ValueError, match="between"):
        api.get_conversation_history("channel_incidents", 51)
    with pytest.raises(ValueError, match="must match"):
        api.get_user("../user")
