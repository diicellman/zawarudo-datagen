"""Deterministic actor-scoped Slack reads with complete paginated traversal."""

import base64
import hashlib
import json
import re
import unicodedata
from datetime import UTC, datetime
from typing import Any, Literal
from pydantic import Field, JsonValue
from .models import Message, SlackWorld, StrictModel, validate_safe_identifier

INTERFACE_ID = "slack.readonly.v2"
ActionName = Literal[
    "list_conversations", "search_messages", "get_conversation_history", "get_thread", "get_user"
]


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).encode()


class ReadCall(StrictModel):
    action: ActionName
    arguments: dict[str, JsonValue] = Field(default_factory=dict)


class SlackNotFoundError(LookupError):
    """A resource is absent or not visible to the configured actor."""


def _normalized_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(value.split())


def _tokens(value: str) -> frozenset[str]:
    return frozenset(re.findall(r"[^\W_]+", _normalized_text(value), flags=re.UNICODE))


def _message_result(message: Message) -> dict[str, str | None]:
    # Keep this public shape deliberately small and identical for all read methods.
    return {
        "message_id": message.id,
        "conversation_id": message.conversation_id,
        "author_id": message.author_id,
        "timestamp": message.timestamp,
        "thread_root_id": message.thread_root_id,
        "text": message.text,
    }


def _validate_limit(value: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("limit must be an integer")
    if not 1 <= value <= maximum:
        raise ValueError(f"limit must be between 1 and {maximum}")
    return value


class SlackAPI:
    """Deterministic, read-only Slack behavior over one validated snapshot.

    Non-archived public channels are visible to every workspace user. Private channels,
    DMs, and group DMs are visible only to members. Archived conversations are excluded
    from all reads. History contains root messages newest-first; threads contain the root
    and replies oldest-first. Deleted messages never appear in history, threads, or search.
    """

    def __init__(self, world: SlackWorld, actor_id: str) -> None:
        if not isinstance(world, SlackWorld):
            world = SlackWorld.model_validate(world)
        # API state must not share mutable nested lists with builder/host state.
        self.world = world.model_copy(deep=True)
        self.actor = self.world.require_actor(actor_id)
        self.actor_id = self.actor.id
        self._users = {user.id: user for user in self.world.users}
        self._conversations = {conversation.id: conversation for conversation in self.world.conversations}
        self._messages = {message.id: message for message in self.world.messages}
        self.snapshot_hash = digest(self.world.model_dump(mode="json"))

    def is_conversation_visible(self, conversation_id: str) -> bool:
        """Return visibility without revealing whether an invisible conversation exists."""
        if not isinstance(conversation_id, str):
            return False
        conversation = self._conversations.get(conversation_id)
        if conversation is None or conversation.is_archived:
            return False
        if conversation.kind == "public_channel":
            return True
        return self.actor_id in conversation.member_ids

    def _visible_conversation(self, conversation_id: str):
        validate_safe_identifier(conversation_id, "conversation_id")
        if not self.is_conversation_visible(conversation_id):
            raise SlackNotFoundError("conversation not found or not visible")
        return self._conversations[conversation_id]

    def _conversations_list(self) -> list[dict[str, Any]]:
        """List visible, non-archived conversations ordered by name then ID."""
        visible = [
            conversation
            for conversation in self.world.conversations
            if self.is_conversation_visible(conversation.id)
        ]
        visible.sort(key=lambda item: ((item.name or "").casefold(), item.id))
        return [
            {
                "conversation_id": conversation.id,
                "name": conversation.name,
                "kind": conversation.kind,
                "topic": conversation.topic,
                "purpose": conversation.purpose,
                "member_ids": list(conversation.member_ids),
                "is_archived": conversation.is_archived,
            }
            for conversation in visible
        ]

    def get_user(self, user_id: str) -> dict[str, Any]:
        """Return a workspace directory entry; directory users are actor-visible."""
        validate_safe_identifier(user_id, "user_id")
        user = self._users.get(user_id)
        if user is None:
            raise SlackNotFoundError("user not found")
        return {
            "user_id": user.id,
            "name": user.name,
            "display_name": user.display_name,
            "email": user.email,
            "team": user.team,
            "is_bot": user.is_bot,
            "is_admin": user.is_admin,
        }

    def _page(self, items: list, scope: Any, cursor: str | None, limit: int) -> dict:
        _validate_limit(limit, 100)
        key = digest([self.snapshot_hash, self.actor_id, scope])
        offset = 0
        if cursor is not None:
            if not isinstance(cursor, str) or len(cursor) > 256:
                raise ValueError("invalid cursor")
            try:
                saved_key, offset = json.loads(base64.b64decode(cursor, validate=True))
            except (ValueError, TypeError) as exc:
                raise ValueError("invalid cursor") from exc
            if saved_key != key or type(offset) is not int or not 0 <= offset <= len(items):
                raise ValueError("cursor does not belong to this actor, snapshot, or query")
        end = offset + limit
        next_cursor = base64.b64encode(canonical([key, end])).decode() if end < len(items) else None
        return {"items": items[offset:end], "next_cursor": next_cursor}

    def list_conversations(self, cursor: str | None = None, limit: int = 50) -> dict:
        return self._page(self._conversations_list(), ["list"], cursor, limit)

    def search_messages(
        self,
        query: str,
        conversation_id: str | None = None,
        author_id: str | None = None,
        after: str | None = None,
        before: str | None = None,
        cursor: str | None = None,
        limit: int = 50,
    ) -> dict:
        if not isinstance(query, str) or not query.strip() or len(query) > 512:
            raise ValueError("query must contain 1–512 characters")
        if conversation_id is not None:
            self._visible_conversation(conversation_id)
        if author_id is not None:
            validate_safe_identifier(author_id, "author_id")
        for bound in (after, before):
            if bound is not None:
                if not isinstance(bound, str):
                    raise ValueError("time filters must be UTC timestamps")
                parsed = datetime.strptime(bound, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
                if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != bound:
                    raise ValueError("time filters must be UTC timestamps")
        if after is not None and before is not None and after >= before:
            raise ValueError("after must precede before")
        normalized, tokens = _normalized_text(query), _tokens(query)
        ranked = []
        for message in self.world.messages:
            if message.deleted or not self.is_conversation_visible(message.conversation_id):
                continue
            if conversation_id is not None and message.conversation_id != conversation_id:
                continue
            if author_id is not None and message.author_id != author_id:
                continue
            if after is not None and message.timestamp <= after:
                continue
            if before is not None and message.timestamp >= before:
                continue
            phrase = normalized in _normalized_text(message.text)
            overlap = len(tokens & _tokens(message.text))
            if phrase or overlap:
                ranked.append(((int(phrase), overlap, message.timestamp, message.id), message))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return self._page(
            [_message_result(message) for _, message in ranked],
            ["search", query, conversation_id, author_id, after, before],
            cursor,
            limit,
        )

    def get_conversation_history(
        self, conversation_id: str, cursor: str | None = None, limit: int = 50
    ) -> dict:
        self._visible_conversation(conversation_id)
        messages = sorted(
            (
                m
                for m in self.world.messages
                if m.conversation_id == conversation_id and not m.deleted and m.thread_root_id is None
            ),
            key=lambda m: (m.timestamp, m.id),
            reverse=True,
        )
        return self._page([_message_result(m) for m in messages], ["history", conversation_id], cursor, limit)

    def get_thread(
        self,
        conversation_id: str,
        root_message_id: str,
        cursor: str | None = None,
        limit: int = 50,
    ) -> dict:
        self._visible_conversation(conversation_id)
        validate_safe_identifier(root_message_id, "root_message_id")
        root = self._messages.get(root_message_id)
        if root is None or root.deleted or root.conversation_id != conversation_id or root.thread_root_id:
            raise SlackNotFoundError("thread root not found or not visible")
        replies = sorted(
            (m for m in self.world.messages if m.thread_root_id == root.id and not m.deleted),
            key=lambda m: (m.timestamp, m.id),
        )
        return self._page(
            [_message_result(m) for m in [root, *replies]],
            ["thread", conversation_id, root_message_id],
            cursor,
            limit,
        )

    def execute(self, call: ReadCall) -> dict:
        return getattr(self, call.action)(**call.arguments)
