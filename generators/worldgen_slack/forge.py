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
from .contracts import features
from .env import SETUP_FILES, GenerationEnv, keep_awake, rates, require_session
from .generate import failure, run_label
from .store import Store


class ForgeConfig(Section):
    """One forge, configured in one file: the world run it writes on, and what it changes of that run's settings."""

    base: Path  # the configuration the world was generated with (its sector, people, taxonomy and roles)
    world: Path  # the finished run: its final world.sqlite, with the ledger and the board
    output: Path
    rounds: int = Field(default=6, ge=1, le=50)
    candidates: int = Field(default=12, ge=1, le=50)  # the candidate tasks one round writes
    target: int = Field(
        default=2, ge=1, le=10
    )  # the tasks kept per category and level before the forge stops
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


def bucket(shape: dict) -> str:
    """A task's structure, as one of the archive's coordinates: how many values its asker can see, where its answer
    sits, a set, another asker's other answer, an answer to work out, a status."""
    parts = [f"k{min(shape['changes'], 3)}"] if shape["changes"] else []
    if shape["last_link"] and not shape["last_link"].startswith("channel"):
        parts.append(shape["last_link"].split("·")[0])
    parts += ["set"] * (shape["rows"] > 1) + [k for k in ("perspective", "derived", "status") if shape[k]]
    return "+".join(parts) or "plain"


def measured_level(right: float, bands) -> int:
    """The level a task's tries put it at: the first band, from level 1 down, whose floor its right rate reaches."""
    return next(i for i, (low, _) in enumerate(bands, 1) if right >= low)


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
            added = await self.weigh(agents, attempt, self.written(ids))
        state.batch, state.restore_point = number, attempt
        counts = state.plans.setdefault("forge", {})
        counts["idle"] = idle = 0 if added else counts.get("idle", 0) + 1
        # It stops when its rounds run out, when the archive is full, or after two rounds that keep nothing.
        if number == cfg.rounds or self.done() or idle >= 2:
            state.phase = "done"
        self.store.finish_attempt(True)
        self.store.save()

    async def weigh(self, agents, attempt: str, ids: list[str]) -> int:
        """The round's candidates tried and reviewed (the screen, the solver's tries, the witness, the judge); those
        GLM misses half the time or more are tried twice more with the messages their answer rests on. Each approved
        one is placed in the archive. Returns how many it kept."""
        state = self.store.state
        if not ids:
            return 0
        self.refresh_gold()
        keys = {t: self.task_key(t) for t in ids}
        await self.judge_tasks(agents, ids, keys, attempt, self.settings.author.tries)
        approved = [t for t in ids if state.task_reviews.get(t, {}).get("key") == keys[t]]
        low = [t for t in approved if state.task_reviews[t]["right_rate"] <= 0.5 and self.answers_on(t)]
        hinted = await self.solves(agents, low, attempt, 2, hint=True) if low else {}
        for t in ids:
            state.forged[t] = self.place(
                t, state.task_reviews[t] if t in approved else None, hinted.get(t), attempt
            )
        self.store.save()
        return sum(state.forged[t]["verdict"].startswith("kept") for t in ids)

    def answers_on(self, task_id: str) -> bool:
        """Whether a task's answer rests on messages a hint can show."""
        return bool(self.store.release_rows("hint")[1][task_id].messages)

    def place(self, task_id: str, review: dict | None, hinted, attempt: str) -> dict:
        """One candidate's outcome. Its tries give its level; the lowest band also needs the witness to answer it
        (hard, not broken). Its cell is its category, that level and its structure (MAP-Elites over task features,
        OMNI-EPIC's archive); a cell keeps the candidate GLM's tries are most mixed on, 4p(1-p), and the shorter
        question when they tie."""
        state, bands = self.store.state, self.settings.tasks.bands
        row = self.world.db.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        gold = json.loads(row["gold_json"])
        shape = features(self.world, task_id, gold)
        out = {"round": state.batch + 1, "category": row["category"], "declared": row["level"], "question": row["question"], "features": shape}  # fmt: skip
        if review is None:
            issues = [i.defect for i in state.task_issues.get(task_id, [])] + state.screens.get(task_id, {}).get("failed", [])  # fmt: skip
            self.drop(task_id, attempt, "not approved")
            return out | {"verdict": "dropped: the judge did not approve it", "issues": issues[:3]}
        right, witness = review["right_rate"], review.get("witness_right")
        hint = rates([o for o, _ in hinted])["right_rate"] if hinted else None
        level = measured_level(right, bands)
        misses = [{"answer": r.get("response", "")[:240], "why": r.get("reason", "")[:240]} for r in state.solves[task_id]["results"] if not r.get("correct") and not r.get("crashed")][:2]  # fmt: skip
        out |= {"right": right, "interval": review.get("right_interval"), "tries": review.get("tries"), "witness": witness, "hint": hint, "level": level, "misses": misses}  # fmt: skip
        if bands[level - 1][0] == 0 and not witness:
            self.drop(task_id, attempt, "neither solver answers it")
            return out | {"verdict": "dropped: the witness does not answer it either: hard, or broken"}
        cell, score, words = f"{row['category']}·L{level}·{bucket(shape)}", 4 * right * (1 - right), len(row["question"].split())  # fmt: skip
        held = state.archive.get(cell)
        if held and (held["score"], -held["words"]) >= (score, -words):
            self.drop(task_id, attempt, f"{cell} holds {held['task']}")
            return out | {"verdict": f"dropped: its cell {cell} holds {held['task']}, as mixed or more"}
        if held:
            self.drop(held["task"], attempt, f"replaced by {task_id}")
            state.forged[held["task"]]["verdict"] = f"replaced by {task_id}"
        self.world.db.execute("UPDATE tasks SET level = ? WHERE id = ?", (level, task_id))
        state.archive[cell] = {"task": task_id, "score": score, "words": words, "right": right}
        return out | {"verdict": f"kept in {cell}"}

    def covered(self) -> tuple[dict, list[str]]:
        """What the archive covers: its tasks per category and level, and the board's entries no kept task rests on (a
        kept task of the entry's category on one of its facts)."""
        kept = {(c, lvl): n for c, lvl, n in self.world.db.execute("SELECT category, level, COUNT(*) FROM tasks GROUP BY category, level")}  # fmt: skip
        rests = {(c, f) for c, f in self.world.db.execute("SELECT t.category, tf.fact_id FROM tasks t JOIN task_facts tf ON tf.task_id = t.id")}  # fmt: skip
        open_ = [slot for slot, facts in self.world.db.execute("SELECT slot, group_concat(fact_id) FROM board GROUP BY slot") if not any((slot.split("-l")[0], f) in rests for f in facts.split(","))]  # fmt: skip
        return kept, sorted(open_)

    def done(self) -> bool:
        """The archive is full: every category and level holds `target` tasks, and every board entry is covered."""
        kept, open_ = self.covered()
        cells = [
            (c, lvl) for c, spec in self.settings.taxonomy.items() for lvl in range(1, len(spec.levels) + 1)
        ]
        return not open_ and all(kept.get(cell, 0) >= self.forge.target for cell in cells)

    def drop(self, task_id: str, attempt: str, why: str) -> None:
        self.world.db.execute("DELETE FROM task_facts WHERE task_id = ?", (task_id,))
        self.world.db.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
        self.store.event("candidate_dropped", attempt=attempt, task_id=task_id, why=why)

    def pages(self) -> dict[str, str]:
        """The forge's own memory pages, beside the world's: the taxonomy, the board the world was planned for, and the archive:
        what each category and level holds against its target, the board's open entries, each kept task, and every
        candidate of the last two rounds with its tries, its traps and how GLM went wrong (SENTINEL's failure-driven
        proposer, 2606.12908)."""
        kept, open_ = self.covered()
        board = ["# Board: the tasks the world was planned for, and the facts each rests on", ""]
        for slot, facts in self.world.db.execute("SELECT slot, group_concat(fact_id, ', ') FROM board GROUP BY slot ORDER BY slot"):  # fmt: skip
            board.append(f"- {slot}: {facts}" + ("  (open)" if slot in open_ else "  (covered)"))
        lines = ["# Archive", "", f"Target: {self.forge.target} kept tasks per category and level, and every board entry. GLM tries each candidate {self.settings.author.tries} times.", ""]  # fmt: skip
        for category, spec in self.settings.taxonomy.items():
            lines.append(f"- {category}: " + " · ".join(f"L{lvl} {kept.get((category, lvl), 0)}/{self.forge.target}" for lvl in range(1, len(spec.levels) + 1)))  # fmt: skip
        lines += ["", f"Board entries no kept task rests on: {', '.join(open_) or 'none'}", "", "## Kept"]
        for cell, held in sorted(self.store.state.archive.items()):
            question = self.world.db.execute(
                "SELECT question FROM tasks WHERE id = ?", (held["task"],)
            ).fetchone()
            lines.append(
                f"- {cell}: {held['task']}, right {held['right']:.2f}: {question[0] if question else ''}"
            )
        last = [t for t, o in self.store.state.forged.items() if o["round"] >= self.store.state.batch - 1]
        lines += ["", "## The last rounds' candidates"]
        for t in last:
            o = self.store.state.forged[t]
            tried = f"right {o['right']:.2f} of {o['tries']} {o.get('interval')}, witness {o.get('witness')}, hint {o.get('hint')}" if "right" in o else ""  # fmt: skip
            lines.append(f"- {t} ({o['category']}, written as L{o['declared']}): {o['question']}\n  {tried} → {o['verdict']}; structure {bucket(o['features'])}" + "".join(f"\n  GLM answered: {m['answer']!r}; the grade: {m['why']!r}" for m in o.get("misses", [])) + "".join(f"\n  issue: {i}" for i in o.get("issues", [])))  # fmt: skip
        taxonomy = ["# Taxonomy: the categories and levels a candidate takes", ""]
        for category, spec in self.settings.taxonomy.items():
            taxonomy += [f"## {category}", spec.definition, f"gold: {spec.gold}; answer types: {', '.join(spec.answer_types)}"]  # fmt: skip
            taxonomy += [f"- level {lvl}: {text} (for example: {'; '.join(spec.concepts[lvl - 1])})" for lvl, text in enumerate(spec.levels, 1)]  # fmt: skip
            taxonomy.append("")
        return {"memory/board.md": "\n".join(board) + "\n", "memory/archive.md": "\n".join(lines) + "\n", "memory/taxonomy.md": "\n".join(taxonomy)}  # fmt: skip


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
