"""The task forge: tasks proposed on a finished, frozen world, kept when their tries say they land.
uv run --frozen worldgen-slack forge --config configs/worldgen_slack/forge.toml [--dry-run]

A world run writes the workspace and its ledger (backward generation: the board plants each task slot's facts before
day 1); the forge then writes tasks on that world, in rounds: a proposer in the author's seat writes candidates, the
solver tries each, and code keeps what lands. The world is never written again, so no task goes stale under it."""

import asyncio
import hashlib
import json
import re
import shutil
import tomllib
from pathlib import Path

from pydantic import Field
from verifiers.v1.clients import EvalClientConfig, ModelContext
from verifiers.v1.runtimes.prime import set_base_sandbox_labels
from worldgen_slack.dataset import atomic_json, sha256
from worldgen_slack.db import SCHEMA, World

import verifiers.v1 as vf

from .agents.world import WorldAuthorTask, forge_prompt
from .config import ROOT, Config, Section, load_config
from .env import SETUP_FILES, GenerationEnv, keep_awake, require_session
from .generate import failure, run_label
from .store import Store


class ForgeConfig(Section):
    """One forge, configured in one file: the world run it writes on, and what it changes of that run's settings."""

    base: Path  # the configuration the world was generated with (its sector, people, taxonomy and roles)
    world: Path  # the finished run: its final world.sqlite, with the ledger and the board
    output: Path
    rounds: int = Field(default=6, ge=1, le=50)
    candidates: int = Field(default=12, ge=1, le=50)  # the candidate tasks one round writes
    settings: dict = Field(
        default_factory=dict
    )  # sections of the base configuration it replaces, merged by key


def merged(base: dict, changes: dict) -> dict:
    return {k: merged(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v for k, v in changes.items()} | {k: v for k, v in base.items() if k not in changes}  # fmt: skip


def load_forge(path: Path) -> tuple[ForgeConfig, Config]:
    """The forge's configuration, and the run settings it works under: the base's, with its changes and output."""
    forge = ForgeConfig.model_validate(tomllib.loads(path.read_text()))
    for field in ("base", "world", "output"):
        if not getattr(forge, field).is_absolute():
            setattr(forge, field, (ROOT / getattr(forge, field)).resolve())
    base = load_config(forge.base).model_dump(mode="json")
    settings = Config.model_validate(merged(base, forge.settings) | {"output": str(forge.output)})
    return forge, settings


def provenance(forge: ForgeConfig, settings: Config) -> dict:
    """Everything a resumed forge must share with its start: its configuration and settings, the world it writes on
    (by hash), the pinned framework and the lock file."""
    return {
        "forge": forge.model_dump(mode="json"),
        "config": settings.model_dump(mode="json"),
        "world_hash": sha256(forge.world / "world.sqlite"),
        "verifiers_revision": "ac2ec29",
        "lock_hash": hashlib.sha256((ROOT / "uv.lock").read_bytes()).hexdigest(),
    }


TASK_TABLES = ("task_facts", "tasks")


class ForgeEnv(GenerationEnv):
    def __init__(self, forge: ForgeConfig, settings: Config, store: Store):
        self.forge = forge
        super().__init__(settings, store)

    async def run(self, task, agents):
        if self.store.state.phase == "premise":
            self.adopt()
        return await self.author_world(agents)

    def adopt(self) -> None:
        """The finished world, frozen: its final file with no tasks (their tables as the schema now has them), its board
        and ledger, and the world author's last notes for the proposer to start from."""
        source = World(self.forge.world / "world.sqlite")
        source.db.backup(self.world.db)
        source.close()
        tables = {m.group(1): m.group(0) for m in re.finditer(r"CREATE TABLE (\w+) \(.*?\n\);", SCHEMA.read_text(), re.S)}  # fmt: skip
        self.world.db.executescript("".join(f"DROP TABLE {t};" for t in TASK_TABLES))
        self.world.db.executescript("\n".join(tables[t] for t in reversed(TASK_TABLES)))
        state = json.loads((self.forge.world / "state.json").read_text())
        notes = self.forge.world / "attempts" / state.get("restore_point", "") / "notes"
        if state.get("restore_point") and notes.is_dir():
            shutil.copytree(notes, self.store.path("source", "notes"), dirs_exist_ok=True)
        self.store.state.phase, self.store.state.batch, self.store.state.restore_point = "tasks", 0, "source"
        self.store.event("world_adopted", world=str(self.forge.world), messages=self.world.db.execute("SELECT COUNT(*) FROM messages").fetchone()[0])  # fmt: skip
        self.store.save()

    async def blocks(self, agents, runtime, setup) -> None:
        await runtime.run(["rm", "-f", *(f"/task/{name}" for name in SETUP_FILES)], {})
        while self.store.state.phase == "tasks":
            await self.forge_round(agents, runtime)

    def written(self, ids: list[str]) -> list[str]:
        return [t for t in ids if self.world.db.execute("SELECT 1 FROM tasks WHERE id = ?", (t,)).fetchone()]

    async def forge_round(self, agents, runtime) -> None:
        """One round: the proposer writes a candidate for each open id in a fresh interaction that starts from the last
        round's notes; then the solver tries them, and what the judge approves is kept."""
        state, cfg = self.store.state, self.forge
        number = state.batch + 1
        attempt = self.store.reserve("round", 2 * cfg.rounds)  # a round cut off is written again once
        ids = [f"r{number:02d}-{n:02d}" for n in range(1, cfg.candidates + 1)]
        self.allow(ids)
        await self.bring_notes(runtime, state.restore_point)
        task = WorldAuthorTask.create("forge", 0, self.world.path, self.author_context(), attempt)
        async with agents.author.interaction(task, runtime=runtime) as interaction:
            for n in range(2):
                if not (missing := [t for t in ids if t not in self.written(ids)]):
                    break
                prompt = forge_prompt(number, cfg.rounds, ids) if n == 0 else f"These open ids have no candidate yet: {missing}. Write them with world_add_task, then end your turn."  # fmt: skip
                await self.author_step(interaction, runtime, prompt, attempt, "forge", self.pages())
        self.store.trace(interaction.trace)
        require_session(interaction.trace)
        await self.keep_notes(runtime, attempt)
        async with keep_awake(runtime, self.settings.author.keepalive):
            await self.weigh(agents, attempt, self.written(ids))
        state.batch, state.restore_point = number, attempt
        if number == cfg.rounds:
            state.phase = "done"
        self.store.finish_attempt(True)
        self.store.save()

    async def weigh(self, agents, attempt: str, ids: list[str]) -> None:
        """The round's candidates tried and reviewed (the screen, the solver's tries, the witness, the judge); a
        candidate the judge does not approve is dropped."""
        if not ids:
            return
        self.refresh_gold()
        keys = {t: self.task_key(t) for t in ids}
        await self.judge_tasks(agents, ids, keys, attempt, self.settings.author.tries)
        reviews = self.store.state.task_reviews
        for t in [t for t in ids if reviews.get(t, {}).get("key") != keys[t]]:
            self.drop(t, attempt, "not approved")

    def drop(self, task_id: str, attempt: str, why: str) -> None:
        self.world.db.execute("DELETE FROM task_facts WHERE task_id = ?", (task_id,))
        self.world.db.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
        self.store.event("candidate_dropped", attempt=attempt, task_id=task_id, why=why)

    def pages(self) -> dict[str, str]:
        """The forge's own memory pages, beside the world's: the board the world was planned for."""
        lines = ["# Board: the tasks the world was planned for, and the facts each rests on", ""]
        for slot, facts in self.world.db.execute("SELECT slot, group_concat(fact_id, ', ') FROM board GROUP BY slot ORDER BY slot"):  # fmt: skip
            lines.append(f"- {slot}: {facts}")
        kept = [f"- {r['id']} ({r['category']} L{r['level']}, right {r['right_rate']}): {r['question']}" for r in self.world.db.execute("SELECT * FROM tasks ORDER BY id")]  # fmt: skip
        return {"memory/board.md": "\n".join(lines) + "\n", "memory/archive.md": "# Archive\n\n" + ("\n".join(kept) or "(nothing kept yet)") + "\n"}  # fmt: skip


async def run(forge: ForgeConfig, settings: Config) -> dict:
    set_base_sandbox_labels([run_label(forge.output)])
    store = Store(forge.output, provenance(forge, settings))
    try:
        if store.state.phase != "done":
            env = ForgeEnv(forge, settings, store)
            context = ModelContext(
                model=settings.env.solver.model, client=EvalClientConfig(), sampling=vf.Sampling()
            )
            seed = vf.Task(vf.TaskData(idx=settings.seed, prompt="Forge tasks on the finished workspace."))
            async with env.serving():
                episode = await env.run_episode(seed, context)
            for trace in episode.traces:
                store.trace(trace)
            atomic_json(store.root / f"episode-{episode.id}.json", episode.to_record())
            if (reason := failure(store.state.phase, episode)) is not None:
                return store.summary(
                    "incomplete" if "ReviewLimit" in reason else "infrastructure_error", reason
                )
        summary = store.summary("complete")
        store.publish()
        return summary
    except asyncio.CancelledError:
        store.summary(
            "interrupted", "forge interrupted; committed progress and available traces are preserved"
        )
        raise
    finally:
        store.close()


def dry_run(forge: ForgeConfig, settings: Config) -> dict:
    """What a forge would start from, spending nothing: the world, its board, and the rounds."""
    world = World(forge.world / "world.sqlite")
    count = lambda table: world.db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]  # noqa: E731
    return {
        "world": str(forge.world),
        "world_hash": sha256(forge.world / "world.sqlite"),
        "messages": count("messages"),
        "facts": count("facts"),
        "board": dict(world.db.execute("SELECT slot, group_concat(fact_id, ', ') FROM board GROUP BY slot")),
        "rounds": forge.rounds,
        "candidates": forge.candidates,
        "tries": settings.author.tries,
        "proposer": settings.env.author.model,
        "solver": settings.env.solver.model,
    }
