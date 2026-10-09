"""Slack read tools over one world file, as the task's actor; every call is recorded for grading."""

import asyncio
import functools
import hashlib
import os
import threading
import time
from pathlib import Path
from typing import Annotated, Literal

import verifiers.v1 as vf
from pydantic import Field, JsonValue

from .db import World
from .dataset import StrictModel

PageSize = Annotated[int, Field(ge=1, le=100)]


class Call(StrictModel):
    tool: str
    arguments: dict[str, JsonValue]
    output: dict[str, JsonValue]


class CallState(vf.State):
    calls: list[Call] = Field(default_factory=list, max_length=4096)


class WorldTaskData(vf.TaskData):
    """Which task, attempt and storyline a run belongs to, and which world file it was given."""

    task_id: str = ""
    attempt: str = ""
    storyline: str = ""
    world_hash: str = ""


class WorldToolsConfig(vf.ToolsetConfig):
    db_path: str = ""
    db_hash: str = ""
    actor_id: str


def file_hash(path: Path | str) -> str:
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def watch_parent() -> None:
    """A tool server lives as long as the process that started it: once that process is gone, the server exits.
    Verifiers' own guard is Linux-only, and a stopped generator left its server running on macOS."""
    parent = os.getppid()
    if parent == 1:  # the parent was gone before the server started watching it
        os._exit(0)

    def watch():
        while os.getppid() == parent:
            time.sleep(1)
        os._exit(0)

    threading.Thread(target=watch, daemon=True).start()


def stage_world(config: WorldToolsConfig, path: Path | str) -> None:
    """The world file never leaves the host: the tool server opens it by path and checks its hash."""
    if config.colocated or config.runtime.type != "subprocess" or config.url is not None:
        raise ValueError("the world file requires a host-side tool server")
    config.db_path, config.db_hash = str(Path(path).resolve()), file_hash(path)


class SlackTools(vf.Toolset[WorldToolsConfig, CallState]):
    TOOL_PREFIX = "slack"

    async def setup(self) -> None:
        self._lock = asyncio.Lock()
        if file_hash(self.config.db_path) != self.config.db_hash:
            raise ValueError("world file hash mismatch")
        self.world = World(self.config.db_path, actor=self.config.actor_id)

    def _with_state(self, fn):
        synced = super()._with_state(fn)

        @functools.wraps(synced)
        async def serialized(*args, **kwargs):
            # State is replaced whole on every push, so concurrent calls must not interleave.
            async with self._lock:
                return await synced(*args, **kwargs)

        return serialized

    def _call(self, tool: str, **arguments) -> dict:
        if len(self.state.calls) >= 4096:
            raise ValueError("call history exhausted")
        output = self.world.call(tool, arguments)
        self.state.calls.append(Call(tool=tool, arguments=arguments, output=output))
        return output

    @vf.tool
    async def search_messages(
        self,
        query: str,
        sort: Literal["score", "timestamp"] = "score",
        cursor: str | None = None,
        limit: PageSize = 20,
    ) -> dict:
        """Search messages in conversations the user can see. Every keyword must match; "quoted phrases" match
        exactly. Modifiers: in:#channel or in:<channel id>, from:@handle or from:<user id>, before:, after: and
        on:YYYY-MM-DD in the user's timezone."""
        return self._call("search_messages", query=query, sort=sort, cursor=cursor, limit=limit)

    @vf.tool
    async def search_users(self, query: str, cursor: str | None = None, limit: PageSize = 20) -> dict:
        """Find workspace users by part of their real name, display name, handle or email, or by user ID."""
        return self._call("search_users", query=query, cursor=cursor, limit=limit)

    @vf.tool
    async def search_channels(self, query: str, cursor: str | None = None, limit: PageSize = 20) -> dict:
        """Find channels the user can see by part of their name, topic or purpose."""
        return self._call("search_channels", query=query, cursor=cursor, limit=limit)

    @vf.tool
    async def list_user_channels(
        self, types: str = "public,private,mpim,im", cursor: str | None = None, limit: PageSize = 50
    ) -> dict:
        """List the conversations the user is a member of, archived ones included. types is a comma-separated
        subset of public, private, mpim and im."""
        return self._call("list_user_channels", types=types, cursor=cursor, limit=limit)

    @vf.tool
    async def read_channel(
        self,
        channel_id: str,
        oldest: str | None = None,
        latest: str | None = None,
        cursor: str | None = None,
        limit: PageSize = 50,
    ) -> dict:
        """Read a conversation's top-level messages, newest first, optionally between the oldest and latest
        message timestamps. Replies are not included; a message with replies has thread_ts and reply_count."""
        return self._call(
            "read_channel", channel_id=channel_id, oldest=oldest, latest=latest, cursor=cursor, limit=limit
        )

    @vf.tool
    async def read_thread(
        self, channel_id: str, thread_ts: str, cursor: str | None = None, limit: PageSize = 50
    ) -> dict:
        """Read a thread: its parent message, then the replies oldest first."""
        return self._call(
            "read_thread", channel_id=channel_id, thread_ts=thread_ts, cursor=cursor, limit=limit
        )

    @vf.tool
    async def get_user(self, user_id: str) -> dict:
        """Fetch a user's profile: names, title, timezone, status and custom fields."""
        return self._call("get_user", user_id=user_id)

    @vf.tool
    async def list_channel_members(
        self, channel_id: str, cursor: str | None = None, limit: PageSize = 50
    ) -> dict:
        """List the user IDs of a conversation's current members."""
        return self._call("list_channel_members", channel_id=channel_id, cursor=cursor, limit=limit)

    @vf.tool
    async def get_reactions(self, channel_id: str, ts: str) -> dict:
        """Get the emoji reactions on a message and who added them."""
        return self._call("get_reactions", channel_id=channel_id, ts=ts)

    @vf.tool
    async def whoami(self) -> dict:
        """Who the user is (ID, handle, name, title, timezone) and the current time on their clock."""
        return self._call("whoami")


if __name__ == "__main__":
    watch_parent()
    SlackTools.run()
