"""Native Slack reads and reviewer inspection, with host-only private data."""

import asyncio
import functools
import json
import tempfile
from pathlib import Path
from typing import Annotated, Literal, TypeVar

import verifiers.v1 as vf
from pydantic import Field, JsonValue

from .models import SlackWorld, StrictModel
from .api import ActionName, SlackAPI, ReadCall, digest

PageSize = Annotated[int, Field(ge=1, le=100)]


class ReadRecord(StrictModel):
    actor_id: str
    call: ReadCall
    output: dict[str, JsonValue]


class ReadState(vf.State):
    reads: list[ReadRecord] = Field(default_factory=list, max_length=4096)
    checked: bool = False


ToolConfig = TypeVar("ToolConfig", bound=vf.ToolsetConfig)


class RecordedTools(vf.Toolset[ToolConfig, ReadState]):
    async def setup(self) -> None:
        self._state_lock = asyncio.Lock()

    def _with_state(self, fn):
        synced = super()._with_state(fn)

        @functools.wraps(synced)
        async def serialized(*args, **kwargs):
            # This pin replaces whole state on PUT. Serialize per-server reads until native atomic updates exist.
            async with self._state_lock:
                return await synced(*args, **kwargs)

        return serialized


def stage_tool_data(task, config, data: dict) -> None:
    if config.colocated or config.runtime.type != "subprocess" or config.url is not None:
        raise ValueError("private workspace data requires a host-side tool server")
    # Verifiers serializes tool config in an environment variable; large worlds need a host file reference.
    task._tool_directory = tempfile.TemporaryDirectory(prefix="zawarudo-tools-")
    path = Path(task._tool_directory.name) / "data.json"
    path.write_text(json.dumps(data, ensure_ascii=False))
    config.data_path, config.data_hash = str(path), digest(data)


def read_tool_data(config) -> dict:
    path = Path(config.data_path)
    if not path.is_file() or path.stat().st_size > 24_000_000:
        raise ValueError("missing or oversized tool data")
    data = json.loads(path.read_text())
    if digest(data) != config.data_hash:
        raise ValueError("tool data hash mismatch")
    return data


class SlackToolsetConfig(vf.ToolsetConfig):
    kind: Literal["read"] = "read"
    snapshot_json: str = Field(default="", exclude=True, repr=False)
    data_path: str = ""
    data_hash: str = ""
    actor_id: str


class SlackToolset(RecordedTools[SlackToolsetConfig]):
    TOOL_PREFIX = "slack"

    async def setup(self) -> None:
        await super().setup()
        self.api = SlackAPI(
            SlackWorld.model_validate_json(json.dumps(read_tool_data(self.config))), self.config.actor_id
        )

    def _read(self, action: ActionName, **arguments) -> dict:
        if len(self.state.reads) >= 4096:
            raise ValueError("read history exhausted")
        call = ReadCall(action=action, arguments=arguments)
        output = self.api.execute(call)
        self.state.reads.append(ReadRecord(actor_id=self.config.actor_id, call=call, output=output))
        return output

    @vf.tool
    async def list_conversations(self, cursor: str | None = None, limit: PageSize = 50) -> dict:
        """List visible non-archived conversations, with a continuation cursor."""
        return self._read("list_conversations", cursor=cursor, limit=limit)

    @vf.tool
    async def search_messages(
        self,
        query: str,
        conversation_id: str | None = None,
        author_id: str | None = None,
        after: str | None = None,
        before: str | None = None,
        cursor: str | None = None,
        limit: PageSize = 50,
    ) -> dict:
        """Search visible messages; optional UTC after/before bounds are exclusive. Results are paginated."""
        return self._read(
            "search_messages",
            query=query,
            conversation_id=conversation_id,
            author_id=author_id,
            after=after,
            before=before,
            cursor=cursor,
            limit=limit,
        )

    @vf.tool
    async def get_conversation_history(
        self, conversation_id: str, cursor: str | None = None, limit: PageSize = 50
    ) -> dict:
        """Read visible root messages newest first, with a continuation cursor."""
        return self._read(
            "get_conversation_history", conversation_id=conversation_id, cursor=cursor, limit=limit
        )

    @vf.tool
    async def get_thread(
        self,
        conversation_id: str,
        root_message_id: str,
        cursor: str | None = None,
        limit: PageSize = 50,
    ) -> dict:
        """Read a thread oldest first, with a continuation cursor."""
        return self._read(
            "get_thread",
            conversation_id=conversation_id,
            root_message_id=root_message_id,
            cursor=cursor,
            limit=limit,
        )

    @vf.tool
    async def get_user(self, user_id: str) -> dict:
        """Read a workspace directory entry."""
        return self._read("get_user", user_id=user_id)


class SlackTaskData(vf.TaskData):
    workspace_id: str = ""
    group_id: str = ""
    candidate_id: str = ""
    task_id: str = ""
    snapshot_hash: str = ""


if __name__ == "__main__":
    SlackToolset.run()
