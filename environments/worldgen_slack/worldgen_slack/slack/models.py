from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

INTERFACE_ID = "slack.readonly.v1"
QUALITY_CRITERIA = (
    "scenario_alignment",
    "world_coherence",
    "professional_realism",
    "discoverability",
    "shortcut_free",
    "evidence_composition",
)
SAFE_IDENTIFIER_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$"
_SAFE_IDENTIFIER = re.compile(SAFE_IDENTIFIER_PATTERN)
TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


SafeId = Annotated[str, Field(min_length=1, max_length=128, pattern=SAFE_IDENTIFIER_PATTERN)]
NonEmptyText = Annotated[str, Field(min_length=1)]


def is_safe_identifier(value: object) -> bool:
    return isinstance(value, str) and _SAFE_IDENTIFIER.fullmatch(value) is not None


def validate_safe_identifier(value: str, field_name: str = "identifier") -> str:
    if not is_safe_identifier(value):
        raise ValueError(f"{field_name} must match {SAFE_IDENTIFIER_PATTERN!r}")
    return value


def _nonblank(value: str, field_name: str) -> str:
    if not value.strip():
        raise ValueError(f"{field_name} must not be blank")
    return value


def _parse_timestamp(value: str, field_name: str) -> datetime:
    try:
        parsed = datetime.strptime(value, TIMESTAMP_FORMAT).replace(tzinfo=UTC)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be UTC in YYYY-MM-DDTHH:MM:SSZ format") from exc
    if parsed.strftime(TIMESTAMP_FORMAT) != value:
        raise ValueError(f"{field_name} must be UTC in YYYY-MM-DDTHH:MM:SSZ format")
    return parsed


class User(StrictModel):
    id: SafeId
    name: Annotated[str, Field(min_length=1, max_length=256)]
    display_name: Annotated[str, Field(max_length=256)] | None = None
    email: Annotated[str, Field(max_length=320)] | None = None
    team: Annotated[str, Field(max_length=256)] | None = None
    is_bot: bool = False
    is_admin: bool = False

    @field_validator("name")
    @classmethod
    def valid_name(cls, value: str) -> str:
        return _nonblank(value, "user name")


class Conversation(StrictModel):
    id: SafeId
    name: Annotated[str, Field(max_length=256)] | None = None
    kind: Literal["public_channel", "private_channel", "dm", "group_dm"]
    topic: Annotated[str, Field(max_length=2_000)] = ""
    purpose: Annotated[str, Field(max_length=2_000)] = ""
    member_ids: list[SafeId] = Field(default_factory=list)
    is_archived: bool = False

    @model_validator(mode="after")
    def valid_members(self) -> Self:
        if len(self.member_ids) != len(set(self.member_ids)):
            raise ValueError("conversation member_ids must be unique")
        minimum = 2 if self.kind == "dm" else 3 if self.kind == "group_dm" else 1
        maximum = 2 if self.kind == "dm" else None
        if len(self.member_ids) < minimum:
            raise ValueError(f"{self.kind} requires at least {minimum} members")
        if maximum is not None and len(self.member_ids) != maximum:
            raise ValueError("dm conversations require exactly two members")
        if self.kind in {"public_channel", "private_channel"} and not (self.name or "").strip():
            raise ValueError("channels require a nonblank name")
        return self


class Message(StrictModel):
    id: SafeId
    conversation_id: SafeId
    author_id: SafeId
    text: Annotated[str, Field(max_length=10_000)]
    timestamp: str
    thread_root_id: SafeId | None = None
    edited_timestamp: str | None = None
    deleted: bool = False

    @field_validator("timestamp", "edited_timestamp")
    @classmethod
    def valid_timestamp(cls, value: str | None, info: Any) -> str | None:
        if value is not None:
            _parse_timestamp(value, info.field_name)
        return value

    @model_validator(mode="after")
    def valid_edit_time(self) -> Self:
        if self.edited_timestamp is not None and _parse_timestamp(
            self.edited_timestamp, "edited_timestamp"
        ) < _parse_timestamp(self.timestamp, "timestamp"):
            raise ValueError("edited_timestamp precedes timestamp")
        return self


class Reaction(StrictModel):
    message_id: SafeId
    user_id: SafeId
    emoji: Annotated[str, Field(min_length=1, max_length=128)]


class SlackWorld(StrictModel):
    users: list[User]
    conversations: list[Conversation]
    messages: list[Message]
    reactions: list[Reaction] = Field(default_factory=list)

    @model_validator(mode="after")
    def valid_references(self) -> Self:
        user_ids = [user.id for user in self.users]
        conversation_ids = [conversation.id for conversation in self.conversations]
        message_ids = [message.id for message in self.messages]
        for label, identifiers in (
            ("user", user_ids),
            ("conversation", conversation_ids),
            ("message", message_ids),
        ):
            if len(identifiers) != len(set(identifiers)):
                raise ValueError(f"duplicate {label} id")

        users = set(user_ids)
        conversations = {item.id: item for item in self.conversations}
        messages = {item.id: item for item in self.messages}
        for conversation in self.conversations:
            missing = set(conversation.member_ids) - users
            if missing:
                raise ValueError(f"conversation {conversation.id!r} has unknown members: {sorted(missing)}")
        for message in self.messages:
            conversation = conversations.get(message.conversation_id)
            if conversation is None:
                raise ValueError(
                    f"message {message.id!r} has unknown conversation {message.conversation_id!r}"
                )
            if message.author_id not in users:
                raise ValueError(f"message {message.id!r} has unknown author {message.author_id!r}")
            if message.author_id not in conversation.member_ids:
                raise ValueError(f"message {message.id!r} author is not a conversation member")
            if message.thread_root_id is not None:
                root = messages.get(message.thread_root_id)
                if root is None:
                    raise ValueError(
                        f"message {message.id!r} has unknown thread root {message.thread_root_id!r}"
                    )
                if root.thread_root_id is not None:
                    raise ValueError(f"message {message.id!r} thread root is not a root message")
                if root.conversation_id != message.conversation_id:
                    raise ValueError(f"message {message.id!r} thread root is in another conversation")
                if _parse_timestamp(message.timestamp, "timestamp") < _parse_timestamp(
                    root.timestamp, "root timestamp"
                ):
                    raise ValueError(f"message {message.id!r} predates its thread root")
        reaction_keys: set[tuple[str, str, str]] = set()
        for reaction in self.reactions:
            message = messages.get(reaction.message_id)
            if message is None:
                raise ValueError(f"reaction has unknown message {reaction.message_id!r}")
            if reaction.user_id not in users:
                raise ValueError(f"reaction has unknown user {reaction.user_id!r}")
            if reaction.user_id not in conversations[message.conversation_id].member_ids:
                raise ValueError("reaction user is not a conversation member")
            key = (reaction.message_id, reaction.user_id, reaction.emoji)
            if key in reaction_keys:
                raise ValueError("duplicate reaction")
            reaction_keys.add(key)
        return self

    def require_actor(self, actor_id: str) -> User:
        validate_safe_identifier(actor_id, "actor_id")
        for user in self.users:
            if user.id == actor_id:
                return user
        raise ValueError(f"actor {actor_id!r} does not exist in the Slack world")


class ScenarioSpec(StrictModel):
    organization: NonEmptyText
    workflow: NonEmptyText
    description: NonEmptyText

    @field_validator("organization", "workflow", "description")
    @classmethod
    def nonblank(cls, value: str, info: Any) -> str:
        return _nonblank(value, info.field_name)


class AnswerSpec(StrictModel):
    kind: Literal["exact_string", "date", "entity", "list", "fact_summary"]
    canonical_answer: NonEmptyText
    required_claims: Annotated[list[NonEmptyText], Field(min_length=1)]
    forbidden_claims: list[NonEmptyText] = Field(default_factory=list)
    list_order_matters: bool = False

    @field_validator("canonical_answer")
    @classmethod
    def valid_answer(cls, value: str) -> str:
        return _nonblank(value, "canonical_answer")

    @field_validator("required_claims", "forbidden_claims")
    @classmethod
    def valid_claims(cls, values: list[str], info: Any) -> list[str]:
        for value in values:
            _nonblank(value, info.field_name)
        normalized = [" ".join(value.casefold().split()) for value in values]
        if len(normalized) != len(set(normalized)):
            raise ValueError(f"{info.field_name} must not contain duplicates")
        return values

    @model_validator(mode="after")
    def noncontradictory(self) -> Self:
        required = {" ".join(value.casefold().split()) for value in self.required_claims}
        forbidden = {" ".join(value.casefold().split()) for value in self.forbidden_claims}
        overlap = required & forbidden
        if overlap:
            raise ValueError(f"required and forbidden claims contradict: {sorted(overlap)}")
        if self.kind != "list" and self.list_order_matters:
            raise ValueError("list_order_matters is only valid for list answers")
        return self


class EvidenceRequirement(StrictModel):
    evidence_id: SafeId
    message_id: SafeId
    conversation_id: SafeId
    author_id: SafeId | None = None
    thread_root_id: SafeId | None = None
    required_terms: Annotated[list[NonEmptyText], Field(min_length=1)]
    fact_description: NonEmptyText

    @field_validator("required_terms")
    @classmethod
    def valid_terms(cls, values: list[str]) -> list[str]:
        for value in values:
            _nonblank(value, "required_terms")
        return values

    @field_validator("fact_description")
    @classmethod
    def valid_description(cls, value: str) -> str:
        return _nonblank(value, "fact_description")


_ALLOWED_GOLD_TOOLS = {
    "slack_list_conversations": frozenset(),
    "slack_search_messages": frozenset({"query", "limit"}),
    "slack_get_conversation_history": frozenset({"conversation_id", "limit"}),
    "slack_get_thread": frozenset({"conversation_id", "root_message_id"}),
    "slack_get_user": frozenset({"user_id"}),
}


class GoldCall(StrictModel):
    tool: Literal[
        "slack_list_conversations",
        "slack_search_messages",
        "slack_get_conversation_history",
        "slack_get_thread",
        "slack_get_user",
    ]
    arguments: dict[str, Any]

    @model_validator(mode="after")
    def valid_arguments(self) -> Self:
        expected = _ALLOWED_GOLD_TOOLS[self.tool]
        actual = set(self.arguments)
        required = expected - {"limit"}
        if actual - expected:
            raise ValueError(f"unsupported arguments for {self.tool}: {sorted(actual - expected)}")
        if required - actual:
            raise ValueError(f"missing arguments for {self.tool}: {sorted(required - actual)}")
        for name, value in self.arguments.items():
            if name in {"conversation_id", "root_message_id", "user_id"}:
                if not isinstance(value, str):
                    raise ValueError(f"{name} must be a string")
                validate_safe_identifier(value, name)
            elif name == "query":
                if not isinstance(value, str) or not value.strip():
                    raise ValueError("query must be a nonblank string")
            elif name == "limit":
                maximum = 10 if self.tool == "slack_search_messages" else 50
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ValueError("limit must be an integer")
                if not 1 <= value <= maximum:
                    raise ValueError(f"limit must be between 1 and {maximum}")
        return self


class TaskContract(StrictModel):
    question: NonEmptyText
    actor_id: SafeId
    answer: AnswerSpec
    required_evidence: Annotated[list[EvidenceRequirement], Field(min_length=1)]
    gold_calls: Annotated[list[GoldCall], Field(min_length=1)]
    task_slug: SafeId | None = None
    min_distinct_evidence_messages: Annotated[int, Field(ge=1)] = 1
    min_distinct_evidence_conversations: Annotated[int, Field(ge=1)] = 1

    @field_validator("question")
    @classmethod
    def valid_question(cls, value: str) -> str:
        value = _nonblank(value, "question")
        normalized = re.sub(r"[\s_-]+", "", value.casefold())
        if "requiredevidence" in normalized or "goldcalls" in normalized:
            raise ValueError("question contains a private task-contract field name")
        return value

    @model_validator(mode="after")
    def valid_evidence_spread(self) -> Self:
        evidence_ids = [item.evidence_id for item in self.required_evidence]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("required_evidence contains duplicate evidence IDs")
        message_count = len({item.message_id for item in self.required_evidence})
        conversation_count = len({item.conversation_id for item in self.required_evidence})
        if self.min_distinct_evidence_messages > message_count:
            raise ValueError("min_distinct_evidence_messages exceeds specified evidence")
        if self.min_distinct_evidence_conversations > conversation_count:
            raise ValueError("min_distinct_evidence_conversations exceeds specified evidence")
        return self


class SynthesizedItem(StrictModel):
    scenario: ScenarioSpec
    task: TaskContract


__all__ = [
    "INTERFACE_ID",
    "QUALITY_CRITERIA",
    "SAFE_IDENTIFIER_PATTERN",
    "TIMESTAMP_FORMAT",
    "AnswerSpec",
    "Conversation",
    "EvidenceRequirement",
    "GoldCall",
    "Message",
    "Reaction",
    "ScenarioSpec",
    "SlackWorld",
    "StrictModel",
    "SynthesizedItem",
    "TaskContract",
    "User",
    "is_safe_identifier",
    "validate_safe_identifier",
]
