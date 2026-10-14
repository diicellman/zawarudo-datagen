"""The judge's tools, on the world file: code's checks, Slack reads and SQL as a task's actor."""

import asyncio
import functools
import json

import verifiers.v1 as vf
from pydantic import Field, JsonValue
from worldgen_slack import slack
from worldgen_slack.toolset import Call, watch_parent
from worldgen_slack.world import World, file_hash

from ..contracts import frozen_before, measures, style


class ReviewToolsConfig(vf.ToolsetConfig):
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
        evidence's search rank for the question's own words), and report each author's style."""
        world = World(self.config.db_path)
        tasks = {}
        for task_id in self.config.task_ids:
            row = world.db.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
            try:
                gold = world.gold(row["actor_id"], row["gold_sql"], max_rows=self.config.max_rows)
                tasks[task_id] = {"rows": gold["rows"]} | measures(
                    world, task_id, gold["rows"], gold["tables"]
                )
            except ValueError as error:
                tasks[task_id] = {"error": str(error)}
        result = {
            "violations": world.violations(complete=self.config.task_ids),
            "tasks": tasks,
            "style": style(world, frozen_before(world)),
        }
        self.state.checked = True
        return json.loads(json.dumps(result, default=str))

    @vf.tool
    async def read(self, actor_id: str, tool: str, arguments: dict[str, JsonValue]) -> dict:
        """Call a Slack read tool as a task's actor, by the solver's tool names without slack_: search_public,
        search_public_and_private, search_channels, search_users, read_channel, read_thread, read_user_profile,
        get_reactions, list_channel_members or list_user_channels. Returns what it showed, structured."""
        output = slack.read(World(self.config.db_path, actor=self._actor(actor_id)), tool, arguments)
        self.state.calls.append(Call(tool=tool, arguments=arguments | {"actor_id": actor_id}, output=output))
        return output

    @vf.tool
    async def sql(self, actor_id: str, sql: str) -> dict:
        """Run one SELECT over the world as a task's actor sees it, the way gold queries run."""
        return World(self.config.db_path).gold(self._actor(actor_id), sql, max_rows=50)


if __name__ == "__main__":
    watch_parent()
    ReviewTools.run()
