import re
import unicodedata
from typing import Any

from .models import Message, SlackWorld, validate_safe_identifier

MAX_SEARCH_RESULTS = 10
MAX_HISTORY_RESULTS = 50
MAX_THREAD_RESULTS = 50


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

    def list_conversations(self) -> list[dict[str, Any]]:
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

    def search_messages(self, query: str, limit: int = MAX_SEARCH_RESULTS) -> list[dict[str, str | None]]:
        """Search visible messages using phrase, token-overlap, timestamp, and ID rank."""
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        normalized_query = _normalized_text(query)
        if not normalized_query:
            raise ValueError("query must not be blank")
        _validate_limit(limit, MAX_SEARCH_RESULTS)
        query_tokens = _tokens(query)
        ranked: list[tuple[tuple[int, int, str, str], Message]] = []
        for message in self.world.messages:
            if message.deleted or not self.is_conversation_visible(message.conversation_id):
                continue
            text = _normalized_text(message.text)
            phrase_match = normalized_query in text
            overlap = len(query_tokens & _tokens(message.text))
            if not phrase_match and overlap == 0:
                continue
            rank = (int(phrase_match), overlap, message.timestamp, message.id)
            ranked.append((rank, message))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [_message_result(message) for _, message in ranked[:limit]]

    def get_conversation_history(
        self, conversation_id: str, limit: int = MAX_HISTORY_RESULTS
    ) -> list[dict[str, str | None]]:
        """Return visible root messages, newest first, bounded to 50 items."""
        self._visible_conversation(conversation_id)
        _validate_limit(limit, MAX_HISTORY_RESULTS)
        messages = [
            message
            for message in self.world.messages
            if message.conversation_id == conversation_id
            and message.thread_root_id is None
            and not message.deleted
        ]
        messages.sort(key=lambda item: (item.timestamp, item.id), reverse=True)
        return [_message_result(message) for message in messages[:limit]]

    def get_thread(self, conversation_id: str, root_message_id: str) -> list[dict[str, str | None]]:
        """Return a root and its replies oldest first, bounded to 50 items."""
        self._visible_conversation(conversation_id)
        validate_safe_identifier(root_message_id, "root_message_id")
        root = self._messages.get(root_message_id)
        if (
            root is None
            or root.deleted
            or root.conversation_id != conversation_id
            or root.thread_root_id is not None
        ):
            raise SlackNotFoundError("thread root not found or not visible")
        replies = [
            message
            for message in self.world.messages
            if message.thread_root_id == root_message_id
            and message.conversation_id == conversation_id
            and not message.deleted
        ]
        replies.sort(key=lambda item: (item.timestamp, item.id))
        messages = [root, *replies[: MAX_THREAD_RESULTS - 1]]
        return [_message_result(message) for message in messages]

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


__all__ = [
    "MAX_HISTORY_RESULTS",
    "MAX_SEARCH_RESULTS",
    "MAX_THREAD_RESULTS",
    "SlackAPI",
    "SlackNotFoundError",
]
