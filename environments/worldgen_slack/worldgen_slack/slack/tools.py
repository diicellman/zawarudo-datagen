import json
from typing import Annotated, Any, Literal, Self

import verifiers.v1 as vf
from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

from .api import MAX_HISTORY_RESULTS, MAX_SEARCH_RESULTS, SlackAPI
from .models import SAFE_IDENTIFIER_PATTERN, SlackWorld, validate_safe_identifier

TOOL_PREFIX = "slack"
SearchLimit = Annotated[int, Field(ge=1, le=MAX_SEARCH_RESULTS)]
HistoryLimit = Annotated[int, Field(ge=1, le=MAX_HISTORY_RESULTS)]
SearchQuery = Annotated[str, Field(min_length=1, max_length=512)]
SlackIdentifier = Annotated[str, Field(min_length=1, max_length=128, pattern=SAFE_IDENTIFIER_PATTERN)]
SlackAction = Literal[
    "list_conversations",
    "search_messages",
    "get_conversation_history",
    "get_thread",
    "get_user",
]


class SlackActionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    action: SlackAction
    arguments: dict[str, JsonValue]

    @model_validator(mode="after")
    def bounded_arguments(self) -> Self:
        if len(json.dumps(self.arguments, ensure_ascii=False, sort_keys=True).encode()) > 4_096:
            raise ValueError("Slack action arguments exceed 4096 bytes")
        return self


class SlackState(vf.State):
    completed_actions: list[SlackActionRecord] = Field(default_factory=list, max_length=128)


class SlackToolsetConfig(vf.ToolsetConfig):
    model_config = ConfigDict(extra="forbid", strict=True)

    snapshot_json: str = Field(default="", repr=False)
    actor_id: str = Field(default="", max_length=128)

    @field_validator("snapshot_json")
    @classmethod
    def valid_snapshot(cls, value: str) -> str:
        if not value:
            return value
        return SlackWorld.model_validate_json(value).model_dump_json()

    @field_validator("actor_id")
    @classmethod
    def valid_actor(cls, value: str) -> str:
        return validate_safe_identifier(value, "actor_id") if value else value

    @model_validator(mode="after")
    def valid_pair(self) -> Self:
        if bool(self.snapshot_json) != bool(self.actor_id):
            raise ValueError("snapshot_json and actor_id must be configured together")
        if self.snapshot_json:
            SlackWorld.model_validate_json(self.snapshot_json).require_actor(self.actor_id)
        return self

    @classmethod
    def from_world(cls, world: SlackWorld, actor_id: str, **kwargs: Any) -> Self:
        validated = SlackWorld.model_validate(world)
        return cls(snapshot_json=validated.model_dump_json(), actor_id=actor_id, **kwargs)


class SlackToolset(vf.Toolset[SlackToolsetConfig, SlackState]):
    TOOL_PREFIX = TOOL_PREFIX

    def __init__(self, config: SlackToolsetConfig) -> None:
        super().__init__(config)
        self._api: SlackAPI | None = None

    async def setup(self) -> None:
        if not self.config.snapshot_json or not self.config.actor_id:
            raise ValueError("SlackToolset requires a snapshot_json and actor_id")
        self._api = SlackAPI(
            SlackWorld.model_validate_json(self.config.snapshot_json),
            self.config.actor_id,
        )

    @property
    def api(self) -> SlackAPI:
        if self._api is None:
            raise RuntimeError("SlackToolset.setup() has not completed")
        return self._api

    def _record(self, action: SlackAction, arguments: dict[str, JsonValue]) -> None:
        if len(self.state.completed_actions) >= 128:
            raise ValueError("Slack action history exceeds 128 successful calls")
        self.state.completed_actions.append(SlackActionRecord(action=action, arguments=arguments))

    @vf.tool
    async def list_conversations(self) -> list[dict[str, Any]]:
        """List conversations visible to the current Slack actor."""
        result = self.api.list_conversations()
        self._record("list_conversations", {})
        return result

    @vf.tool
    async def search_messages(
        self, query: SearchQuery, limit: SearchLimit = MAX_SEARCH_RESULTS
    ) -> list[dict[str, str | None]]:
        """Search visible undeleted messages; exact phrases rank before token overlap."""
        result = self.api.search_messages(query, limit)
        self._record("search_messages", {"query": query, "limit": limit})
        return result

    @vf.tool
    async def get_conversation_history(
        self,
        conversation_id: SlackIdentifier,
        limit: HistoryLimit = MAX_HISTORY_RESULTS,
    ) -> list[dict[str, str | None]]:
        """Read visible root messages newest first; absent or invisible IDs fail."""
        result = self.api.get_conversation_history(conversation_id, limit)
        self._record(
            "get_conversation_history",
            {"conversation_id": conversation_id, "limit": limit},
        )
        return result

    @vf.tool
    async def get_thread(
        self,
        conversation_id: SlackIdentifier,
        root_message_id: SlackIdentifier,
    ) -> list[dict[str, str | None]]:
        """Read a visible root and its replies oldest first; absent or invisible IDs fail."""
        result = self.api.get_thread(conversation_id, root_message_id)
        self._record(
            "get_thread",
            {
                "conversation_id": conversation_id,
                "root_message_id": root_message_id,
            },
        )
        return result

    @vf.tool
    async def get_user(self, user_id: SlackIdentifier) -> dict[str, Any]:
        """Read one workspace directory entry; unknown user IDs fail."""
        result = self.api.get_user(user_id)
        self._record("get_user", {"user_id": user_id})
        return result


__all__ = [
    "TOOL_PREFIX",
    "SlackActionRecord",
    "SlackState",
    "SlackToolset",
    "SlackToolsetConfig",
]


if __name__ == "__main__":
    SlackToolset.run()
