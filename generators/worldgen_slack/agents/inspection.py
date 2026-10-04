"""The judge's tools, on the world file: code's checks, Slack reads and SQL as a task's actor."""

import asyncio
import functools
import json

import verifiers.v1 as vf
from pydantic import Field, JsonValue
from worldgen_slack.db import World
from worldgen_slack.tools import Call, file_hash

from ..contracts import measures, style


class ReviewToolsConfig(vf.ToolsetConfig):
    phase: str = ""
    db_path: str = ""
    db_hash: str = ""
    task_ids: list[str] = Field(default_factory=list)
    actors: list[str] = Field(default_factory=list)
    max_rows: int = 5


class ReviewState(vf.State):
    calls: list[Call] = Field(default_factory=list, max_length=4096)
    checked: bool = False


class ReviewTools(vf.Toolset[ReviewToolsConfig, ReviewState]):
    TOOL_PREFIX = "inspect"

    async def setup(self) -> None:
        self._lock = asyncio.Lock()
        if file_hash(self.config.db_path) != self.config.db_hash:
            raise ValueError("world file hash mismatch")

    def _with_state(self, fn):
        synced = super()._with_state(fn)

        @functools.wraps(synced)
        async def serialized(*args, **kwargs):
            async with self._lock:
                return await synced(*args, **kwargs)

        return serialized

    def _actor(self, actor_id: str) -> str:
        if actor_id not in self.config.actors:
            raise ValueError(f"actor_id is one of this review's task actors: {self.config.actors}")
        return actor_id

    @vf.tool
    async def check(self) -> dict:
        """Run code's rules over the world, re-run each reviewed task's gold query as its actor, measure each task
        (how many read_channel pages deep its evidence sits for its actor, the tables its gold query reads, its
        evidence's search rank for the question's own words), and report each author's style. In the ledger review
        no message exists yet, so whether the reviewed tasks' facts are stated, and the measures, wait for the
        written storylines."""
        world = World(self.config.db_path)
        written = self.config.phase != "ledger"
        tasks = {}
        for task_id in self.config.task_ids:
            row = world.db.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
            try:
                gold = world.gold(row["actor_id"], row["gold_sql"], max_rows=self.config.max_rows)
                tasks[task_id] = {"rows": gold["rows"]}
                if written:
                    tasks[task_id] |= measures(world, task_id, gold["rows"], gold["tables"])
            except ValueError as error:
                tasks[task_id] = {"error": str(error)}
        due = self.config.task_ids if written else []
        result = {"violations": world.violations(complete=due), "tasks": tasks, "style": style(world)}
        self.state.checked = True
        return json.loads(json.dumps(result, default=str))

    @vf.tool
    async def read(self, actor_id: str, tool: str, arguments: dict[str, JsonValue]) -> dict:
        """Call a Slack read tool as a task's actor: search_messages, search_users, search_channels,
        list_user_channels, read_channel, read_thread, get_user, list_channel_members or get_reactions."""
        output = World(self.config.db_path, actor=self._actor(actor_id)).call(tool, arguments)
        self.state.calls.append(Call(tool=tool, arguments=arguments | {"actor_id": actor_id}, output=output))
        return output

    @vf.tool
    async def sql(self, actor_id: str, sql: str) -> dict:
        """Run one SELECT over the world as a task's actor sees it, the way gold queries run."""
        return World(self.config.db_path).gold(self._actor(actor_id), sql, max_rows=50)


if __name__ == "__main__":
    ReviewTools.run()
