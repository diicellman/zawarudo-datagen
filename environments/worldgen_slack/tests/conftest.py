from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from worldgen_slack.slack.models import (
    Conversation,
    Message,
    ScenarioSpec,
    SlackWorld,
    SynthesizedItem,
    TaskContract,
    User,
)


def make_contract(**updates: Any) -> TaskContract:
    payload: dict[str, Any] = {
        "task_slug": "payments-incident",
        "question": "What caused the payments timeout, and what action resolved it?",
        "actor_id": "agent_user",
        "answer": {
            "kind": "fact_summary",
            "canonical_answer": ("A stale DNS cache caused the timeout; recycling edge workers resolved it."),
            "required_claims": ["stale DNS cache caused timeout", "edge workers recycled"],
            "forbidden_claims": ["database overload"],
        },
        "required_evidence": [
            {
                "evidence_id": "cause",
                "message_id": "msg_cause",
                "conversation_id": "incidents",
                "author_id": "alice",
                "thread_root_id": "root",
                "required_terms": ["stale DNS cache", "payments timeout"],
                "fact_description": "Confirmed cause.",
            },
            {
                "evidence_id": "fix",
                "message_id": "msg_fix",
                "conversation_id": "incidents",
                "author_id": "bob",
                "thread_root_id": "root",
                "required_terms": ["recycled", "edge workers"],
                "fact_description": "Confirmed resolution.",
            },
        ],
        "gold_calls": [
            {"tool": "slack_search_messages", "arguments": {"query": "payments timeout"}},
            {
                "tool": "slack_get_thread",
                "arguments": {"conversation_id": "incidents", "root_message_id": "root"},
            },
        ],
        "min_distinct_evidence_messages": 2,
    }
    payload.update(updates)
    return TaskContract.model_validate(payload)


def make_world(seed: int = 0) -> SlackWorld:
    users = [
        User(id="agent_user", name="Agent User"),
        User(id="alice", name="Alice Chen"),
        User(id="bob", name="Bob Diaz"),
        User(id="cara", name="Cara Evans"),
    ]
    conversations = [
        Conversation(
            id="incidents",
            name="incidents",
            kind="public_channel",
            member_ids=[user.id for user in users],
        ),
        Conversation(
            id="private",
            name="leadership",
            kind="private_channel",
            member_ids=["alice", "bob"],
        ),
        Conversation(id="dm", kind="dm", member_ids=["agent_user", "cara"]),
        Conversation(
            id="archived",
            name="old-incidents",
            kind="public_channel",
            member_ids=[user.id for user in users],
            is_archived=True,
        ),
    ]
    rows = [
        ("root", "incidents", "cara", "Payments timeout investigation started.", "10:00:00", None, False),
        (
            "msg_cause",
            "incidents",
            "alice",
            "A stale DNS cache caused the payments timeout.",
            "10:05:00",
            "root",
            False,
        ),
        (
            "msg_fix",
            "incidents",
            "bob",
            "We recycled the edge workers and service recovered.",
            "10:09:00",
            "root",
            False,
        ),
        ("older", "incidents", "alice", "Payments timeout alert opened.", "09:00:00", None, False),
        ("newer", "incidents", "bob", "Payments timeout monitoring is green.", "11:00:00", None, False),
        ("private_msg", "private", "alice", "Confidential roadmap in Cedar.", "09:10:00", None, False),
        ("dm_msg", "dm", "cara", "Direct handoff is complete.", "09:20:00", None, False),
        ("deleted", "incidents", "alice", "Deleted secret answer cache.", "09:30:00", None, True),
        ("archived_msg", "archived", "bob", "Old incident detail.", "09:40:00", None, False),
        (
            "variant",
            "incidents",
            "cara",
            f"Rehearsal group {seed % 7} meets tomorrow.",
            "09:50:00",
            None,
            False,
        ),
    ]
    messages = [
        Message(
            id=message_id,
            conversation_id=conversation_id,
            author_id=author_id,
            text=text,
            timestamp=f"2025-01-15T{time}Z",
            thread_root_id=root,
            deleted=deleted,
        )
        for message_id, conversation_id, author_id, text, time, root, deleted in rows
    ]
    return SlackWorld(users=users, conversations=conversations, messages=messages)


@pytest.fixture
def contract() -> TaskContract:
    return make_contract()


@pytest.fixture
def world() -> SlackWorld:
    return make_world()


@pytest.fixture
def synthesized(contract: TaskContract) -> SynthesizedItem:
    return SynthesizedItem(
        scenario=ScenarioSpec(
            organization="Northstar Payments",
            workflow="Payments incident response",
            description="An SRE team diagnoses a payment timeout.",
        ),
        task=contract,
    )


def deep_world_payload(world: SlackWorld) -> dict[str, Any]:
    return deepcopy(world.model_dump(mode="json"))
