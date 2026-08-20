from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from worldgen_slack.contracts import ScenarioSpec, SynthesizedItem
from worldgen_slack.slack.models import Conversation, Message, SlackWorld, TaskContract, User


def make_contract(**updates: Any) -> TaskContract:
    payload: dict[str, Any] = {
        "task_slug": "payments-incident",
        "question": "What caused the payments timeout, and what action resolved it?",
        "actor_id": "agent_user",
        "answer": {
            "kind": "fact_summary",
            "canonical_answer": (
                "A stale DNS cache caused the payments timeout; recycling the edge workers resolved it."
            ),
            "required_claims": [
                "stale DNS cache caused timeout",
                "edge workers recycled",
            ],
            "forbidden_claims": ["database overload"],
        },
        "required_evidence": [
            {
                "evidence_id": "evidence_cause",
                "message_id": "msg_cause",
                "conversation_id": "channel_incidents",
                "author_id": "user_alice",
                "thread_root_id": "msg_incident_root",
                "required_terms": ["stale DNS cache", "payments timeout"],
                "fact_description": "The confirmed incident cause.",
            },
            {
                "evidence_id": "evidence_fix",
                "message_id": "msg_fix",
                "conversation_id": "channel_incidents",
                "author_id": "user_bob",
                "thread_root_id": "msg_incident_root",
                "required_terms": ["recycled", "edge workers"],
                "fact_description": "The action that resolved the incident.",
            },
        ],
        "gold_calls": [
            {
                "tool": "slack_search_messages",
                "arguments": {"query": "payments timeout"},
            },
            {
                "tool": "slack_get_thread",
                "arguments": {
                    "conversation_id": "channel_incidents",
                    "root_message_id": "msg_incident_root",
                },
            },
        ],
        "min_distinct_evidence_messages": 2,
        "min_distinct_evidence_conversations": 1,
    }
    payload.update(updates)
    return TaskContract.model_validate(payload)


def make_world(seed: int = 0) -> SlackWorld:
    users = [
        User(id="agent_user", name="Agent User", display_name="Agent", team="SRE"),
        User(id="user_alice", name="Alice Chen", display_name="Alice", team="SRE"),
        User(id="user_bob", name="Bob Diaz", display_name="Bob", team="Platform"),
        User(id="user_cara", name="Cara Evans", display_name="Cara", team="Payments"),
        User(id="user_dan", name="Dan Frost", display_name="Dan", team="Security"),
    ]
    conversations = [
        Conversation(
            id="channel_incidents",
            name="incidents",
            kind="public_channel",
            topic="Operational incidents",
            member_ids=[user.id for user in users],
        ),
        Conversation(
            id="channel_engineering",
            name="engineering",
            kind="public_channel",
            topic="Engineering discussion",
            member_ids=[user.id for user in users],
        ),
        Conversation(
            id="channel_leadership",
            name="leadership",
            kind="private_channel",
            topic="Leadership planning",
            member_ids=["user_alice", "user_bob"],
        ),
    ]
    messages = [
        Message(
            id="msg_incident_root",
            conversation_id="channel_incidents",
            author_id="user_cara",
            text="Payments timeout investigation started after an elevated error alert.",
            timestamp="2025-01-15T10:00:00Z",
        ),
        Message(
            id="msg_cause",
            conversation_id="channel_incidents",
            author_id="user_alice",
            text="Confirmed: a stale DNS cache caused the payments timeout.",
            timestamp="2025-01-15T10:05:00Z",
            thread_root_id="msg_incident_root",
        ),
        Message(
            id="msg_fix",
            conversation_id="channel_incidents",
            author_id="user_bob",
            text="We recycled the edge workers, and the payments service recovered.",
            timestamp="2025-01-15T10:09:00Z",
            thread_root_id="msg_incident_root",
        ),
        Message(
            id="msg_monitoring",
            conversation_id="channel_incidents",
            author_id="user_cara",
            text="Monitoring remained green during the thirty-minute observation window.",
            timestamp="2025-01-15T10:30:00Z",
        ),
        Message(
            id="msg_lunch",
            conversation_id="channel_engineering",
            author_id="user_dan",
            text="Team lunch will be in the north cafeteria on Friday.",
            timestamp="2025-01-14T09:00:00Z",
        ),
        Message(
            id="msg_docs",
            conversation_id="channel_engineering",
            author_id="user_alice",
            text="The onboarding guide now includes the local development commands.",
            timestamp="2025-01-14T09:05:00Z",
        ),
        Message(
            id="msg_demo",
            conversation_id="channel_engineering",
            author_id="user_bob",
            text="The dashboard demo moved to Thursday afternoon.",
            timestamp="2025-01-14T09:10:00Z",
        ),
        Message(
            id="msg_hiring",
            conversation_id="channel_engineering",
            author_id="user_cara",
            text="Two backend candidates are scheduled for interviews next week.",
            timestamp="2025-01-14T09:15:00Z",
        ),
        Message(
            id="msg_release",
            conversation_id="channel_engineering",
            author_id="user_dan",
            text=f"Release rehearsal group {seed % 7} meets tomorrow morning.",
            timestamp="2025-01-14T09:20:00Z",
        ),
        Message(
            id="msg_budget",
            conversation_id="channel_leadership",
            author_id="user_alice",
            text="The quarterly travel budget review is on Monday.",
            timestamp="2025-01-14T09:25:00Z",
        ),
        Message(
            id="msg_private_roadmap",
            conversation_id="channel_leadership",
            author_id="user_bob",
            text="The confidential roadmap review moved to room Cedar.",
            timestamp="2025-01-14T09:30:00Z",
        ),
        Message(
            id="msg_metrics",
            conversation_id="channel_engineering",
            author_id="user_cara",
            text="January adoption metrics will be ready after the data refresh.",
            timestamp="2025-01-14T09:35:00Z",
        ),
        Message(
            id="msg_training",
            conversation_id="channel_engineering",
            author_id="user_dan",
            text="Security training office hours are posted on the team calendar.",
            timestamp="2025-01-14T09:40:00Z",
        ),
        Message(
            id="msg_design",
            conversation_id="channel_engineering",
            author_id="user_alice",
            text="Please leave comments on the navigation design by Tuesday.",
            timestamp="2025-01-14T09:45:00Z",
        ),
        Message(
            id="msg_rotation",
            conversation_id="channel_engineering",
            author_id="user_bob",
            text="The next on-call rotation handoff is documented in the runbook.",
            timestamp="2025-01-14T09:50:00Z",
        ),
    ]
    return SlackWorld(users=users, conversations=conversations, messages=messages)


WORLD_PROGRAM = r"""from worldgen_slack.slack.models import Conversation, Message, SlackWorld, User


def build(seed, contract):
    users = [
        User(id="agent_user", name="Agent User", display_name="Agent", team="SRE"),
        User(id="user_alice", name="Alice Chen", display_name="Alice", team="SRE"),
        User(id="user_bob", name="Bob Diaz", display_name="Bob", team="Platform"),
        User(id="user_cara", name="Cara Evans", display_name="Cara", team="Payments"),
        User(id="user_dan", name="Dan Frost", display_name="Dan", team="Security"),
    ]
    conversations = [
        Conversation(id="channel_incidents", name="incidents", kind="public_channel",
                     topic="Operational incidents", member_ids=[u.id for u in users]),
        Conversation(id="channel_engineering", name="engineering", kind="public_channel",
                     topic="Engineering discussion", member_ids=[u.id for u in users]),
        Conversation(id="channel_leadership", name="leadership", kind="private_channel",
                     topic="Leadership planning", member_ids=["user_alice", "user_bob"]),
    ]
    rows = [
        ("msg_incident_root", "channel_incidents", "user_cara",
         "Payments timeout investigation started after an elevated error alert.",
         "2025-01-15T10:00:00Z", None),
        ("msg_cause", "channel_incidents", "user_alice",
         "Confirmed: a stale DNS cache caused the payments timeout.",
         "2025-01-15T10:05:00Z", "msg_incident_root"),
        ("msg_fix", "channel_incidents", "user_bob",
         "We recycled the edge workers, and the payments service recovered.",
         "2025-01-15T10:09:00Z", "msg_incident_root"),
        ("msg_monitoring", "channel_incidents", "user_cara",
         "Monitoring remained green during the thirty-minute observation window.",
         "2025-01-15T10:30:00Z", None),
        ("msg_lunch", "channel_engineering", "user_dan",
         "Team lunch will be in the north cafeteria on Friday.", "2025-01-14T09:00:00Z", None),
        ("msg_docs", "channel_engineering", "user_alice",
         "The onboarding guide now includes local development commands.", "2025-01-14T09:05:00Z", None),
        ("msg_demo", "channel_engineering", "user_bob",
         "The dashboard demo moved to Thursday afternoon.", "2025-01-14T09:10:00Z", None),
        ("msg_hiring", "channel_engineering", "user_cara",
         "Two backend candidates are scheduled for interviews next week.", "2025-01-14T09:15:00Z", None),
        ("msg_release", "channel_engineering", "user_dan",
         f"Release rehearsal group {seed % 7} meets tomorrow morning.", "2025-01-14T09:20:00Z", None),
        ("msg_budget", "channel_leadership", "user_alice",
         "The quarterly travel budget review is on Monday.", "2025-01-14T09:25:00Z", None),
        ("msg_private_roadmap", "channel_leadership", "user_bob",
         "The confidential roadmap review moved to room Cedar.", "2025-01-14T09:30:00Z", None),
        ("msg_metrics", "channel_engineering", "user_cara",
         "January adoption metrics will be ready after the data refresh.", "2025-01-14T09:35:00Z", None),
        ("msg_training", "channel_engineering", "user_dan",
         "Security training office hours are posted on the team calendar.", "2025-01-14T09:40:00Z", None),
        ("msg_design", "channel_engineering", "user_alice",
         "Please leave comments on the navigation design by Tuesday.", "2025-01-14T09:45:00Z", None),
        ("msg_rotation", "channel_engineering", "user_bob",
         "The next on-call rotation handoff is documented in the runbook.", "2025-01-14T09:50:00Z", None),
    ]
    messages = [Message(id=i, conversation_id=c, author_id=a, text=t, timestamp=ts,
                        thread_root_id=root) for i, c, a, t, ts, root in rows]
    return SlackWorld(users=users, conversations=conversations, messages=messages)
"""


@pytest.fixture
def contract() -> TaskContract:
    return make_contract()


@pytest.fixture
def synthesized(contract: TaskContract) -> SynthesizedItem:
    return SynthesizedItem(
        scenario=ScenarioSpec(
            organization="Northstar Payments",
            workflow="Payments incident response",
            description="An SRE team diagnoses and resolves a payment timeout incident.",
        ),
        task=contract,
    )


@pytest.fixture
def world() -> SlackWorld:
    return make_world()


@pytest.fixture
def world_program(tmp_path: Path) -> Path:
    path = tmp_path / "world.py"
    path.write_text(WORLD_PROGRAM, encoding="utf-8")
    return path


@pytest.fixture
def contract_file(tmp_path: Path, contract: TaskContract) -> Path:
    path = tmp_path / "contract.json"
    path.write_text(contract.model_dump_json(indent=2), encoding="utf-8")
    return path


def deep_world_payload(world: SlackWorld) -> dict[str, Any]:
    return deepcopy(world.model_dump(mode="json"))


def json_line(value: Any) -> str:
    return json.dumps(value, sort_keys=True) + "\n"
