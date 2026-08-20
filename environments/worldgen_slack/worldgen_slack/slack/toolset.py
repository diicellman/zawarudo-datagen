from typing import Annotated, Any, Literal

import verifiers.v1 as vf
from pydantic import ConfigDict, Field, field_validator, model_validator

from .api import MAX_HISTORY_RESULTS, MAX_SEARCH_RESULTS, SlackAPI
from .models import SlackWorld, validate_safe_identifier

TOOL_PREFIX = "slack"
SearchLimit = Annotated[int, Field(ge=1, le=MAX_SEARCH_RESULTS)]
HistoryLimit = Annotated[int, Field(ge=1, le=MAX_HISTORY_RESULTS)]


class SlackToolsetConfig(vf.ToolsetConfig):
    model_config = ConfigDict(extra="forbid", strict=True)

    colocated: Literal[False] = False
    snapshot_json: str = Field(default="", repr=False)
    actor_id: str = Field(default="", max_length=128)

    @field_validator("snapshot_json")
    @classmethod
    def valid_snapshot(cls, value: str) -> str:
        if not value:
            return value
        world = SlackWorld.model_validate_json(value)
        return world.model_dump_json()

    @field_validator("actor_id")
    @classmethod
    def valid_actor(cls, value: str) -> str:
        return validate_safe_identifier(value, "actor_id") if value else value

    @model_validator(mode="after")
    def valid_pair(self):
        if self.colocated or self.runtime.type != "subprocess":
            raise ValueError("Slack Toolset must use the trusted host-local server")
        if bool(self.snapshot_json) != bool(self.actor_id):
            raise ValueError("snapshot_json and actor_id must be configured together")
        if self.snapshot_json:
            SlackWorld.model_validate_json(self.snapshot_json).require_actor(self.actor_id)
        return self

    @classmethod
    def from_world(cls, world: SlackWorld, actor_id: str, **kwargs: Any):
        validated = SlackWorld.model_validate(world)
        return cls(snapshot_json=validated.model_dump_json(), actor_id=actor_id, **kwargs)


class SlackState(vf.State):
    completed_calls: list[dict[str, Any]] = Field(default_factory=list)


class SlackToolset(vf.Toolset[SlackToolsetConfig, SlackState]):
    TOOL_PREFIX = TOOL_PREFIX

    def __init__(self, config: SlackToolsetConfig) -> None:
        super().__init__(config)
        self._api: SlackAPI | None = None

    async def setup(self) -> None:
        if not self.config.snapshot_json or not self.config.actor_id:
            raise ValueError("SlackToolset requires a snapshot_json and actor_id")
        world = SlackWorld.model_validate_json(self.config.snapshot_json)
        self._api = SlackAPI(world, self.config.actor_id)

    @property
    def api(self) -> SlackAPI:
        if self._api is None:
            raise RuntimeError("SlackToolset.setup() has not completed")
        return self._api

    def _record(self, name: str, arguments: dict[str, Any]) -> None:
        self.state.completed_calls.append({"tool": f"slack_{name}", "arguments": arguments})

    @vf.tool
    async def list_conversations(self) -> list[dict[str, Any]]:
        """List conversations visible to the current Slack actor."""
        result = self.api.list_conversations()
        self._record("list_conversations", {})
        return result

    @vf.tool
    async def search_messages(
        self, query: str, limit: SearchLimit = MAX_SEARCH_RESULTS
    ) -> list[dict[str, str | None]]:
        """Search visible messages. Exact phrases rank above token overlap."""
        result = self.api.search_messages(query, limit)
        self._record("search_messages", {"query": query, "limit": limit})
        return result

    @vf.tool
    async def get_conversation_history(
        self, conversation_id: str, limit: HistoryLimit = MAX_HISTORY_RESULTS
    ) -> list[dict[str, str | None]]:
        """Get root messages in one visible conversation, newest first."""
        result = self.api.get_conversation_history(conversation_id, limit)
        self._record(
            "get_conversation_history",
            {"conversation_id": conversation_id, "limit": limit},
        )
        return result

    @vf.tool
    async def get_thread(self, conversation_id: str, root_message_id: str) -> list[dict[str, str | None]]:
        """Get a visible thread root and its replies in chronological order."""
        result = self.api.get_thread(conversation_id, root_message_id)
        self._record(
            "get_thread",
            {"conversation_id": conversation_id, "root_message_id": root_message_id},
        )
        return result

    @vf.tool
    async def get_user(self, user_id: str) -> dict[str, Any]:
        """Look up one user in the workspace directory."""
        result = self.api.get_user(user_id)
        self._record("get_user", {"user_id": user_id})
        return result


__all__ = ["TOOL_PREFIX", "SlackState", "SlackToolset", "SlackToolsetConfig"]


if __name__ == "__main__":
    SlackToolset.run()
