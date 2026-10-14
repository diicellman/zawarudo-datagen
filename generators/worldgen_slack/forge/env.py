"""The task forge: tasks proposed on a finished, frozen world, kept when their tries say they land.
uv run --frozen worldgen-slack forge --config configs/worldgen_slack/forge.toml [--dry-run]

A world run writes the workspace and its ledger (backward generation: the board plants each task slot's facts before
day 1); the forge then writes tasks on that world, in rounds: a proposer in the author's seat writes candidates, the
solver tries each, and code keeps what lands. The world is never written again, so no task goes stale under it."""

import json
import re
import shutil
import tomllib
from collections import Counter
from pathlib import Path

from pydantic import Field, model_validator
from worldgen_slack.world import SCHEMA, file_hash
from ..workspace import Workspace


from ..agents.world import WorldAuthorTask, forge_prompt
from .tasks import answer_of, forge_task
from ..world.board import brief
from ..config import Config, Section, load_config, pinned, rooted
from .difficulty import features
from .tasks import INTENTS, Task
from .tries import miss_class
from ..judge.verdict import READING
from ..env import SETUP_FILES, GenerationEnv, rates
from ..session import keep_awake, require_session
from ..store import Store


class ForgeConfig(Section):
    """One forge, configured in one file: the world run it writes on, and what it changes of that run's settings.
    `tasks` is its goal, the tasks the archive holds when it is done; `rounds` the most rounds it runs toward it;
    `candidates` the candidates one round writes; `target` the tasks a place (a category and a level) holds;
    `measure` has round 0 measure the finished run's own tasks, as written, before any proposer round; `settings`
    the sections of the base configuration it replaces, merged by key."""

    base: Path  # the configuration the world was generated with (its sector, people, taxonomy and roles)
    world: Path  # the finished run: its final world.sqlite, with the ledger and the board
    output: Path
    tasks: int = Field(default=40, ge=1, le=10_000)
    rounds: int = Field(default=12, ge=0, le=100)
    candidates: int = Field(default=12, ge=1, le=50)
    target: int = Field(default=2, ge=1, le=100)
    measure: bool = False
    settings: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def works(self):
        if self.rounds == 0 and not self.measure:
            raise ValueError("a forge with no rounds only measures: set measure = true")
        return self


def merged(base: dict, changes: dict) -> dict:
    return {
        k: merged(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
        for k, v in changes.items()
    } | {k: v for k, v in base.items() if k not in changes}


def load_forge(path: Path) -> tuple[ForgeConfig, Config]:
    """The forge's configuration, and the run settings it works under: the base's, with its changes and output."""
    forge = ForgeConfig.model_validate(tomllib.loads(path.read_text()))
    forge.base, forge.world, forge.output = rooted(forge.base), rooted(forge.world), rooted(forge.output)
    base = load_config(forge.base).model_dump(mode="json")
    settings = Config.model_validate(merged(base, forge.settings) | {"output": str(forge.output)})
    return forge, settings


def provenance(forge: ForgeConfig, settings: Config) -> dict:
    """Everything a resumed forge must share with its start: its configuration and settings, the world it writes on
    (by hash), the pinned framework and the lock file."""
    return {
        "forge": forge.model_dump(mode="json"),
        "config": settings.model_dump(mode="json"),
        "world_hash": file_hash(forge.world / "world.sqlite"),
    } | pinned()


TASK_TABLES = ("task_facts", "tasks")


def reset_tasks(world) -> None:
    """A world's task tables emptied, as the schema now has them: a forge and a chapter start with no tasks."""
    tables = {
        m.group(1): m.group(0)
        for m in re.finditer(r"CREATE TABLE (\w+) \(.*?\n\);", SCHEMA.read_text(), re.S)
    }
    world.db.executescript("".join(f"DROP TABLE {t};" for t in TASK_TABLES))
    world.db.executescript("\n".join(tables[t] for t in reversed(TASK_TABLES)))


def measured(forged: dict) -> list[str]:
    """This forge's own numbers, by category and intent: candidates tried, kept, sent back for another reading, the
    mean share of right answers of those tried, and how the wrong tries went wrong. Steering comes from these, not from
    lore about any solver."""
    cells: dict[tuple, dict] = {}
    for o in forged.values():
        cell = cells.setdefault(
            (o["category"], o.get("intent") or "-"),
            {"tried": 0, "kept": 0, "back": 0, "right": [], "misses": Counter()},
        )
        cell["tried"] += 1
        cell["kept"] += o["verdict"].startswith("kept")
        cell["back"] += o["verdict"].startswith("sent back")
        if o.get("right") is not None:
            cell["right"].append(o["right"])
        cell["misses"].update(o.get("miss_classes") or {})
    return [
        f"- {c}·{i}: {x['tried']} tried, {x['kept']} kept, {x['back']} sent back; right {sum(x['right']) / len(x['right']):.2f} on average; wrong tries: {dict(x['misses']) or 'none'}"
        for (c, i), x in sorted(cells.items())
        if x["right"] or x["back"]
    ]


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
        places = sum(len(spec.levels) for spec in settings.taxonomy.values())
        if forge.tasks > forge.target * places:
            raise ValueError(
                f"a goal of {forge.tasks} tasks is more than the archive's {places} places hold at {forge.target} each"
            )
        super().__init__(settings, store)

    async def run(self, task, agents):
        state = self.store.state
        if state.phase == "premise":
            self.adopt()
        if self.forge.measure and not state.plans.get("forge", {}).get("measured"):
            await self.measure(agents)
        if state.phase == "tasks":  # a forge that only measures needs no proposer, and no VM
            await self.author_world(agents)

    async def measure(self, agents) -> None:
        """Round 0: the finished run's own tasks, under their own ids, checked for validity only (T1-T7, the columns
        the tools show) and not by the forge's own gates, then measured as any candidate is: the screen, 8 tries, the
        witness, the hint, the judge. S3's tasks and the forge's are then read with one measuring stick: S3's rates
        came from 4 tries at temperature 0."""
        state = self.store.state
        attempt = self.store.reserve("measure", 2)
        source = Workspace(self.forge.world / "world.sqlite")
        rows = source.db.execute("SELECT * FROM tasks ORDER BY id").fetchall()
        ids = [r["id"] for r in rows]
        for row in rows:
            facts = [
                f
                for (f,) in source.db.execute(
                    "SELECT fact_id FROM task_facts WHERE task_id = ?", (row["id"],)
                )
            ]
            task = Task(
                id=row["id"],
                category=row["category"],
                level=row["level"],
                actor_id=row["actor_id"],
                question=row["question"],
                answer_type=row["answer_type"],
                gold_sql=row["gold_sql"],
                facts=facts,
            )
            try:
                with self.world.trial() as copy:
                    forge_task(copy, task, self.settings, ids, gates=False)
            except ValueError as error:  # a task the world's rules now refuse is itself a finding
                state.forged[row["id"]] = {
                    "round": 0,
                    "category": row["category"],
                    "declared": row["level"],
                    "question": row["question"],
                    "features": {},
                    "verdict": f"not imported: {error}",
                }
        source.close()
        self.store.event("tasks_imported", attempt=attempt, tasks=len(self.written(ids)), of=len(ids))
        await self.weigh(agents, attempt, self.written(ids))
        state.plans.setdefault("forge", {})["measured"] = True
        if self.forge.rounds == 0:
            state.phase = "done"
        self.store.finish_attempt(True)
        self.store.save()

    def adopt(self) -> None:
        """The finished world, frozen: its final file with no tasks (their tables as the schema now has them), its board
        and ledger, and the world author's last notes for the proposer to start from."""
        source = Workspace(self.forge.world / "world.sqlite")
        source.db.backup(self.world.db)
        source.close()
        self.world.migrate()
        reset_tasks(self.world)
        state = json.loads((self.forge.world / "state.json").read_text())
        notes = self.forge.world / "attempts" / state.get("restore_point", "") / "notes"
        if state.get("restore_point") and notes.is_dir():
            shutil.copytree(notes, self.store.path("source", "notes"), dirs_exist_ok=True)
        self.store.state.phase, self.store.state.batch, self.store.state.restore_point = "tasks", 0, "source"
        self.store.event(
            "world_adopted",
            world=str(self.forge.world),
            messages=self.world.db.execute("SELECT COUNT(*) FROM messages").fetchone()[0],
        )
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
        self.allow(
            ids,
            rework={
                t: o["rework"]
                for t, o in state.forged.items()
                if o.get("rework") and o["round"] == number - 1
            },
        )
        await self.bring_notes(runtime, state.restore_point)
        task = WorldAuthorTask.create("forge", 0, self.world.path, self.author_context(), attempt)
        async with agents.author.interaction(task, runtime=runtime) as interaction:
            for n in range(2):
                if not (missing := [t for t in ids if t not in self.written(ids)]):
                    break
                held = self.world.db.execute(
                    "SELECT COUNT(*) FROM tasks WHERE id NOT IN (" + ", ".join("?" * len(ids)) + ")", ids
                ).fetchone()[0]
                prompt = (
                    forge_prompt(number, cfg.rounds, ids, held, cfg.tasks)
                    if n == 0
                    else f"These open ids have no candidate yet: {missing}. Write them with world_add_task, then end your turn."
                )
                await self.author_step(interaction, runtime, prompt, attempt, "forge", self.pages())
        self.store.trace(interaction.trace)
        require_session(interaction.trace)
        await self.keep_notes(runtime, attempt)
        async with keep_awake(runtime, self.settings.author.keepalive):
            added = await self.weigh(agents, attempt, self.written(ids))
        state.batch, state.restore_point = number, attempt
        counts = state.plans.setdefault("forge", {})
        counts["idle"] = idle = 0 if added else counts.get("idle", 0) + 1
        # It stops at its goal, when its rounds run out, or after two rounds that place nothing.
        if number == cfg.rounds or self.done() or idle >= 2:
            state.phase = "done"
        self.store.finish_attempt(True)
        self.store.save()

    async def weigh(self, agents, attempt: str, ids: list[str]) -> int:
        """The round's candidates tried and reviewed (the screen, the solver's tries, the witness, the judge); those
        the solver misses half the time or more are tried twice more with the messages their answer rests on. Each approved
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
        (hard, not broken). Its place is its category and that level, which holds up to `target` tasks (MAP-Elites
        over task features, as OMNI-EPIC keeps its archive). A newcomer to a full place replaces a holder when it
        brings a structure the place lacks while two holders share one, or when the solver's tries are more mixed on it than
        on the weakest holder, by 4p(1-p), the shorter question winning a tie. forge-t1's 10 tasks filled 10 cells of
        their own, so nothing competed and every easy task was kept."""
        state, bands = self.store.state, self.settings.tasks.bands
        row = self.world.db.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        gold = json.loads(row["gold_json"])
        shape = features(self.world, task_id, gold)
        out = {
            "round": state.batch + 1,
            "category": row["category"],
            "intent": row["intent"],
            "declared": row["level"],
            "question": row["question"],
            "features": shape,
        }
        if review is None:
            issues = [i.defect for i in state.task_issues.get(task_id, [])] + state.screens.get(
                task_id, {}
            ).get("failed", [])
            readings = [i.removeprefix(READING) for i in issues if i.startswith(READING)]
            # Another reading is the question's fault, not the solver's: the next round may reword it, for the same
            # asker and gold answer; it is never kept as hard.
            rework = (
                {"answer": answer_of(self.world, task_id), "actor": row["actor_id"], "reading": readings[0]}
                if readings
                else None
            )
            self.drop(task_id, attempt, "another reading" if readings else "not approved")
            if rework:
                return out | {
                    "verdict": "sent back: its question can be read another way",
                    "issues": issues[:3],
                    "rework": rework,
                }
            return out | {"verdict": "dropped: the judge did not approve it", "issues": issues[:3]}
        right, witness = review["right_rate"], review.get("witness_right")
        hint = rates([o for o, _ in hinted])["right_rate"] if hinted else None
        level = measured_level(right, bands)
        wrong = [r for r in state.solves[task_id]["results"] if not r.get("correct") and not r.get("crashed")]
        misses = [{"answer": r.get("response", "")[:240], "why": r.get("reason", "")[:240]} for r in wrong][
            :2
        ]
        classes = dict(Counter(miss_class(r, row["answer_type"]) for r in wrong))
        out |= {
            "right": right,
            "interval": review.get("right_interval"),
            "tries": review.get("tries"),
            "witness": witness,
            "hint": hint,
            "level": level,
            "misses": misses,
            "miss_classes": classes,
        }
        if bands[level - 1][0] == 0 and not witness:
            self.drop(task_id, attempt, "neither solver answers it")
            return out | {"verdict": "dropped: the witness does not answer it either: hard, or broken"}
        where, shaped = (
            f"{row['category']}·L{level}",
            bucket(shape) + (f"·{row['intent']}" if row["intent"] else ""),
        )
        mine = {
            "task": task_id,
            "score": 4 * right * (1 - right),
            "words": len(row["question"].split()),
            "right": right,
            "bucket": shaped,
        }
        holders = state.archive.setdefault(where, [])
        if len(holders) >= self.forge.target:
            weakest = lambda hs: min(hs, key=lambda h: (h["score"], -h["words"]))  # noqa: E731
            shared = [h for h in holders if [g["bucket"] for g in holders].count(h["bucket"]) > 1]
            if shaped not in {h["bucket"] for h in holders} and shared:
                out_ = weakest(shared)
            elif (mine["score"], -mine["words"]) > (weakest(holders)["score"], -weakest(holders)["words"]):
                out_ = weakest(holders)
            else:
                held = ", ".join(h["task"] for h in holders)
                self.drop(task_id, attempt, f"{where} is full")
                return out | {
                    "verdict": f"dropped: {where} is full ({held}), with tasks as mixed or more and its structure"
                }
            holders.remove(out_)
            self.drop(out_["task"], attempt, f"replaced by {task_id}")
            state.forged[out_["task"]]["verdict"] = f"replaced by {task_id}"
        self.world.db.execute("UPDATE tasks SET level = ? WHERE id = ?", (level, task_id))
        holders.append(mine)
        return out | {"verdict": f"kept in {where} as {shaped}"}

    def covered(self) -> tuple[dict, list[str]]:
        """What the archive covers: its tasks per category and level, and the board's entries no kept task rests on (a
        kept task of the entry's category on one of its facts)."""
        kept = {
            (c, lvl): n
            for c, lvl, n in self.world.db.execute(
                "SELECT category, level, COUNT(*) FROM tasks GROUP BY category, level"
            )
        }
        rests = {
            (c, f)
            for c, f in self.world.db.execute(
                "SELECT t.category, tf.fact_id FROM tasks t JOIN task_facts tf ON tf.task_id = t.id"
            )
        }
        # A chapter's world: its brief is the board, and an earlier version's entries were covered by that version's
        # own tasks, which stay with it.
        ours = brief(self.world) or None
        open_ = [
            slot
            for slot, facts in self.world.db.execute(
                "SELECT slot, group_concat(fact_id) FROM board GROUP BY slot"
            )
            if (ours is None or slot in ours)
            and not any((slot.split("-l")[0], f) in rests for f in facts.split(","))
        ]
        return kept, sorted(open_)

    def done(self) -> bool:
        """The forge is at its goal: the archive holds `tasks` tasks, and every board entry is covered."""
        kept, open_ = self.covered()
        return not open_ and sum(kept.values()) >= self.forge.tasks

    def drop(self, task_id: str, attempt: str, why: str) -> None:
        self.world.db.execute("DELETE FROM task_facts WHERE task_id = ?", (task_id,))
        self.world.db.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
        self.store.event("candidate_dropped", attempt=attempt, task_id=task_id, why=why)

    def pages(self) -> dict[str, str]:
        """The forge's own memory pages, beside the world's: the taxonomy, the board the world was planned for, and the archive:
        what each category and level holds against its target, the board's open entries, each kept task, and every
        candidate of the last two rounds with its tries, its traps and how the solver went wrong (SENTINEL's failure-driven
        proposer, 2606.12908)."""
        kept, open_ = self.covered()
        board = ["# Board: the tasks the world was planned for, and the facts each rests on", ""]
        for slot, facts in self.world.db.execute(
            "SELECT slot, group_concat(fact_id, ', ') FROM board GROUP BY slot ORDER BY slot"
        ):
            board.append(f"- {slot}: {facts}" + ("  (open)" if slot in open_ else "  (covered)"))
        lines = [
            "# Archive",
            "",
            f"Goal: {self.forge.tasks} kept tasks, at most {self.forge.target} per category and level, and every board entry covered. It holds {sum(kept.values())}. The solver tries each candidate {self.settings.author.tries} times.",
            "",
        ]
        for category, spec in self.settings.taxonomy.items():
            lines.append(
                f"- {category}: "
                + " · ".join(
                    f"L{lvl} {kept.get((category, lvl), 0)}/{self.forge.target}"
                    for lvl in range(1, len(spec.levels) + 1)
                )
            )
        lines += [
            "",
            f"Board entries no kept task rests on: {', '.join(open_) or 'none'}",
            "",
            "## Kept (a full place takes a newcomer only if the solver's tries are more mixed on it, or it brings a new structure)",
        ]
        for where, holders in sorted(self.store.state.archive.items()):
            full = " (full)" if len(holders) >= self.forge.target else ""
            for held in holders:
                question = self.world.db.execute(
                    "SELECT question FROM tasks WHERE id = ?", (held["task"],)
                ).fetchone()
                lines.append(
                    f"- {where}{full}: {held['task']}, right {held['right']:.2f}, structure {held['bucket']}: {question[0] if question else ''}"
                )
        last = [t for t, o in self.store.state.forged.items() if o["round"] >= self.store.state.batch - 1]
        lines += ["", "## The last rounds' candidates"]
        for t in last:
            o = self.store.state.forged[t]
            tried = (
                f"right {o['right']:.2f} of {o['tries']} {o.get('interval')}, witness {o.get('witness')}, hint {o.get('hint')}"
                if "right" in o
                else ""
            )
            lines.append(
                f"- {t} ({o['category']}, written as L{o['declared']}): {o['question']}\n  {tried} → {o['verdict']}; structure {bucket(o['features']) if o['features'] else '-'}"
                + "".join(
                    f"\n  the solver answered: {m['answer']!r}; the grade: {m['why']!r}"
                    for m in o.get("misses", [])
                )
                + "".join(f"\n  issue: {i}" for i in o.get("issues", []))
            )
            if o.get("rework") and o["round"] == self.store.state.batch:
                lines.append(
                    f'  reword it this round: world_add_task with rewords="{t}", asked by {o["rework"]["actor"]}, whose gold answer stays {o["rework"]["answer"]}'
                )
        lines += ["", "## What this forge has measured (its own tries, by category and intent)"]
        lines += measured(self.store.state.forged) or ["- nothing yet"]
        taxonomy = ["# Taxonomy: the categories and levels a candidate takes", ""]
        for category, spec in self.settings.taxonomy.items():
            taxonomy += [
                f"## {category}",
                spec.definition,
                f"gold: {spec.gold}; answer types: {', '.join(spec.answer_types)}",
            ]
            taxonomy += [
                f"- level {lvl}: {text} (for example: {'; '.join(spec.concepts[lvl - 1])})"
                for lvl, text in enumerate(spec.levels, 1)
            ]
            taxonomy.append("")
        taxonomy += [
            "# Intents: the request a task is, a person's to an assistant (name one on each candidate)",
            "",
        ]
        taxonomy += [f"- {k}: {v}" for k, v in INTENTS.items()]
        return {
            "memory/board.md": "\n".join(board) + "\n",
            "memory/archive.md": "\n".join(lines) + "\n",
            "memory/taxonomy.md": "\n".join(taxonomy) + "\n",
        }


def dry_run(forge: ForgeConfig, settings: Config) -> dict:
    """What a forge would start from, spending nothing: the world, its board, its goal, and the rounds."""
    world = Workspace(forge.world / "world.sqlite")
    count = lambda table: world.db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]  # noqa: E731
    return {
        "world": str(forge.world),
        "world_hash": file_hash(forge.world / "world.sqlite"),
        "messages": count("messages"),
        "facts": count("facts"),
        "board": dict(world.db.execute("SELECT slot, group_concat(fact_id, ', ') FROM board GROUP BY slot")),
        "tasks": forge.tasks,
        "target": forge.target,
        "measure": forge.measure,
        "rounds": forge.rounds,
        "candidates": forge.candidates,
        "tries": settings.author.tries,
        "proposer": settings.env.author.model,
        "solver": settings.env.solver.model,
    }
