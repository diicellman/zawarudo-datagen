"""A chapter: a finished world continued forward in time, into a new version of it.
uv run --frozen worldgen-slack chapter --config configs/worldgen_slack/chapter.toml [--dry-run]

A world run (or an earlier chapter) is written and frozen. A chapter appends days after its last, plans what they hold
(the earlier storylines going on, its own after them, and the brief: the material a later forge's tasks will rest on,
checked by code on the board), and the same author writes them in time order, as the run wrote its own. Nothing
before the chapter's first day is written again. The chapter writes no tasks: the forge writes them on the version it
leaves, and tasks forged on an earlier version stay with that version."""

import asyncio
import hashlib
import json
import random
import re
import shutil
import tempfile
import tomllib
from datetime import date, timedelta
from pathlib import Path

from pydantic import Field, model_validator
from verifiers.v1.clients import EvalClientConfig, ModelContext
from verifiers.v1.runtimes.prime import set_base_sandbox_labels
from worldgen_slack.dataset import atomic_json, sha256
from worldgen_slack.db import World, digest

import verifiers.v1 as vf

from .chronicle import spread, start_clock
from .config import ROOT, ActivityConfig, AuthorSettings, Config, Needs, Section, Text, load_config
from .contracts import Slot, calendar_days, extend_calendar, frozen_digest, meta
from .env import GenerationEnv
from .forge import merged, reset_tasks
from .generate import failure, run_label
from .store import RunState, Store


class Material(Section):
    """One kind of material the chapter plants: `count` board entries of one taxonomy cell, each needing its level's
    needs with `needs` laid over them (and always fresh: first stated in the chapter)."""

    category: Text
    level: int = Field(ge=1)
    count: int = Field(default=1, ge=1, le=20)
    concept: Text | None = (
        None  # what the entries are about; the seed draws one of the level's concepts when unset
    )
    needs: dict = Field(default_factory=dict)


class ChapterConfig(Section):
    """One chapter, configured in one file: the finished world it continues, its days and messages, the storylines it
    adds, when the judge reviews it (days counted from its first), its brief, and what it changes of the base
    configuration's settings, merged by key. `seed` is the chapter's own, so that two chapters of one world differ."""

    base: Path  # the configuration the world was generated with (its sector, people, taxonomy and roles)
    world: Path  # the finished run or chapter it continues
    output: Path
    seed: int = Field(ge=0)
    days: int = Field(ge=1, le=30)
    messages: int = Field(ge=1, le=5000)
    storylines: int = Field(default=0, ge=0, le=10)  # new storylines, after the earlier ones
    review_days: list[int] = Field(default_factory=list)  # 1 is the chapter's first day
    material: list[Material] = Field(min_length=1)
    settings: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def within(self):
        if bad := [d for d in self.review_days if not 1 <= d < self.days]:
            raise ValueError(f"review_days are days of the chapter before its last: {bad}")
        return self


def parent_world(path: Path) -> World:
    return World(path / "world.sqlite")


def load_chapter(path: Path) -> tuple[ChapterConfig, Config]:
    """The chapter's configuration, and the settings it runs under (`chapter_settings`)."""
    chapter = ChapterConfig.model_validate(tomllib.loads(path.read_text()))
    for field in ("base", "world", "output"):
        if not getattr(chapter, field).is_absolute():
            setattr(chapter, field, (ROOT / getattr(chapter, field)).resolve())
    return chapter, chapter_settings(chapter, load_config(chapter.base))


def chapter_settings(chapter: ChapterConfig, base: Config) -> Config:
    """The settings a chapter runs under: the base's, with its changes, its seed and output, its messages, the
    storylines in all (the parent's and its own), and its review days on the calendar. The parent is a finished
    world that breaks none of the world's rules, and each material is a cell of the taxonomy."""
    state = json.loads((chapter.world / "state.json").read_text())
    if state.get("phase") != "done":
        raise ValueError(
            f"{chapter.world} is not finished (phase {state.get('phase')}): a chapter continues a finished world"
        )
    world = parent_world(chapter.world)
    try:
        last = world.db.execute("SELECT MAX(day) FROM calendar").fetchone()[0]
        storylines = world.db.execute("SELECT COUNT(*) FROM storylines").fetchone()[0]
        if broken := world.violations(complete=True):
            raise ValueError(f"{chapter.world} breaks the world's rules: " + "; ".join(broken))
    finally:
        world.close()
    changes = merged(
        chapter.settings,
        {
            "author": {"review_days": [last + d for d in chapter.review_days]},
            "activity": {"messages": chapter.messages},
        },
    )
    fixed = {
        "output": str(chapter.output),
        "seed": chapter.seed,
        "storylines": storylines + chapter.storylines,
    }
    settings = Config.model_validate(merged(base.model_dump(mode="json"), changes) | fixed)
    for m in chapter.material:
        spec = settings.taxonomy.get(m.category)
        if spec is None or not 1 <= m.level <= len(spec.levels):
            raise ValueError(f"material {m.category} level {m.level} is no cell of the taxonomy")
        Needs.model_validate((spec.needs[m.level - 1].model_dump() if spec.needs else {}) | m.needs)
    return settings


def version_of(world) -> int:
    return int(meta(world, "version", 1))


def brief_slots(chapter: ChapterConfig, settings: Config, version: int) -> list[Slot]:
    """The brief as slots on the board: each material's entries, ids `{category}-l{level}-v{version}-{n}` (a forge
    reads an entry's category from its id), each with its level's needs under the material's, always fresh, and a
    concept (a cell's entries walk its level's concepts in a seeded order) and a question style the seed draws."""
    rng, slots, seen, drawn = random.Random(digest([chapter.seed, "brief"])), [], {}, {}
    for m in chapter.material:
        spec = settings.taxonomy[m.category]
        needs = Needs.model_validate(
            (spec.needs[m.level - 1].model_dump() if spec.needs else {}) | m.needs | {"fresh": True}
        )
        pool = drawn.setdefault(
            (m.category, m.level), rng.sample(spec.concepts[m.level - 1], len(spec.concepts[m.level - 1]))
        )
        for _ in range(m.count):
            seen[(m.category, m.level)] = n = seen.get((m.category, m.level), 0) + 1
            concept = (
                m.concept or pool[(n - 1) % len(pool)]
            )  # a cell's entries walk its concepts, as a world's slots do
            slots.append(
                Slot(
                    id=f"{m.category}-l{m.level}-v{version}-{n}",
                    category=m.category,
                    level=m.level,
                    concept=concept,
                    style=rng.choice(settings.tasks.styles),
                    needs=needs,
                )
            )
    return slots


def parent_segments(path: Path, world) -> list[dict]:
    """The parent's calendar segments: its own when it is a chapter, else its run's, rebuilt from the configuration
    it ran with (read as written: a field a later version of the settings dropped or added changes nothing) and the
    days its first plan put an event on."""
    if (stored := meta(world, "segments")) is not None:
        return json.loads(stored)
    config = json.loads((path / "run.json").read_text())["config"]
    activity, author = config.get("activity", {}), config.get("author", {})
    first, last = world.db.execute("SELECT MIN(day), MAX(day) FROM calendar").fetchone()
    busy = meta(world, "busy_days")
    return [
        dict(
            first=first,
            last=last,
            messages=activity.get("messages", ActivityConfig().messages),
            busy=json.loads(busy) if busy else None,
            weekend=author.get("weekend", AuthorSettings().weekend),
            parts=activity.get("parts", ActivityConfig().parts),
        )
    ]


def same_schema(world) -> list[str]:
    """The tables, views and triggers the world's schema differs in from the schema now (the task tables aside, which
    a chapter empties): a table by its columns (a column a migration added stands after the table's constraints, so
    its text differs), anything else by its text, comments and spacing aside."""

    def defined(db) -> dict:
        rows = db.execute(
            "SELECT type, name, sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'"
        ).fetchall()
        return {
            name: sorted(tuple(c)[1:] for c in db.execute(f"PRAGMA table_info({name})"))
            if kind == "table" and not (sql or "").startswith("CREATE VIRTUAL")
            else " ".join(re.sub(r"--[^\n]*", "", sql or "").split())
            for kind, name, sql in rows
        }

    with tempfile.TemporaryDirectory() as directory:
        fresh = World.create(Path(directory) / "schema.sqlite")
        want = defined(fresh.db)
        fresh.close()
    have = defined(world.db)
    return sorted(
        n
        for n in want.keys() | have.keys()
        if want.get(n) != have.get(n) and n not in ("tasks", "task_facts")
    )


def last_day_attempt(path: Path) -> str:
    """The attempt the parent's last closed day was kept in: its notes are where the world left off."""
    closed = [
        json.loads(line)
        for line in (path / "progress.jsonl").read_text().splitlines()
        if '"day_closed"' in line
    ]
    if not closed:
        raise ValueError(f"{path} closed no day")
    return closed[-1]["attempt"]


def adopt(store: Store, chapter: ChapterConfig, settings: Config, slots: list[Slot]) -> None:
    """The parent, frozen, as the chapter's start: its world with no tasks, its calendar running on by the chapter's
    days, its segments and the chapter's, the lineage (version, the parent's hash, the chapter's first day, the digest
    of everything before it), the brief's needs, the present at the chapter's first morning; its premise, people,
    organization and drawn conversations; and the notes its last day left."""
    world, parent = store.world, chapter.world
    source = parent_world(parent)
    source.db.backup(world.db)
    source.close()
    world.migrate()  # a world of an older schema comes forward before anything is measured on it
    if drift := same_schema(world):
        raise ValueError(
            f"{parent}'s world has another schema in {drift}: a chapter continues a world of this one"
        )
    reset_tasks(world)
    spans = parent_segments(parent, world)
    version = version_of(world) + 1
    with world.batch():
        first = extend_calendar(world, chapter.days)
        spans.append(
            dict(
                first=first,
                last=first + chapter.days - 1,
                messages=chapter.messages,
                busy=None,
                weekend=settings.author.weekend,
                parts=settings.activity.parts,
            )
        )
        lineage = {
            "segments": json.dumps(spans),
            "version": str(version),
            "parent_hash": sha256(parent / "world.sqlite"),
            "chapter_start_day": str(first),
            "brief": json.dumps({s.id: s.needs.model_dump() for s in slots}),
        }
        for key, value in lineage.items():
            world.db.execute("INSERT OR REPLACE INTO world_meta (key, value) VALUES (?, ?)", (key, value))
        start_clock(world, first)
        world.db.execute(
            "INSERT OR REPLACE INTO world_meta (key, value) VALUES ('frozen_digest', ?)",
            (frozen_digest(world),),
        )
    was = RunState.model_validate(
        {
            k: v
            for k, v in json.loads((parent / "state.json").read_text()).items()
            if k in ("premise", "cast", "organization", "drafts", "plans")
        }
    )
    state = store.state
    state.premise, state.cast, state.organization = was.premise, was.cast, was.organization
    state.drafts = {"organization": was.drafts["organization"]}
    state.plans = {"agenda": dict(was.plans.get("agenda", {}))}
    notes = parent / "attempts" / last_day_attempt(parent) / "notes"
    shutil.copytree(notes, store.path("source", "notes"), dirs_exist_ok=True)
    store.snapshot("source")
    state.quota, state.phase, state.day, state.restore_point = slots, "plan", first, "source"
    store.event(
        "chapter_adopted",
        world=str(parent),
        version=version,
        first=first,
        days=chapter.days,
        brief=[s.id for s in slots],
    )
    store.save()


class ChapterEnv(GenerationEnv):
    """The world author in a chapter: the plan of its days, its days, and a final review of the world, with no
    tasks; its reviews read the earlier chapters as written and frozen."""

    def __init__(self, chapter: ChapterConfig, settings: Config, store: Store, slots: list[Slot]):
        self.chapter, self.slots = chapter, slots
        super().__init__(settings, store)

    def author_context(self) -> dict:
        first = int(meta(self.world, "chapter_start_day", 1))
        return super().author_context() | {"chapter": {"first": first, "version": version_of(self.world)}}

    async def run(self, task, agents):
        if self.store.state.phase == "premise":
            adopt(self.store, self.chapter, self.settings, self.slots)
        return await self.author_world(agents)


def provenance(chapter: ChapterConfig, settings: Config) -> dict:
    """Everything a resumed chapter must share with its start: its configuration and settings, the world it continues
    (by hash), the pinned framework and the lock file."""
    return {
        "chapter": chapter.model_dump(mode="json"),
        "config": settings.model_dump(mode="json"),
        "parent": {"path": str(chapter.world), "world_hash": sha256(chapter.world / "world.sqlite")},
        "verifiers_revision": "ac2ec29",
        "lock_hash": hashlib.sha256((ROOT / "uv.lock").read_bytes()).hexdigest(),
    }


async def run(chapter: ChapterConfig, settings: Config) -> dict:
    set_base_sandbox_labels([run_label(chapter.output)])
    store = Store(chapter.output, provenance(chapter, settings))
    try:
        if store.state.phase != "done":
            world = parent_world(chapter.world)
            slots = brief_slots(chapter, settings, version_of(world) + 1)
            world.close()
            env = ChapterEnv(chapter, settings, store, slots)
            context = ModelContext(
                model=settings.env.solver.model, client=EvalClientConfig(), sampling=vf.Sampling()
            )
            seed = vf.Task(vf.TaskData(idx=settings.seed, prompt="Continue the shared Slack workspace."))
            async with env.serving():
                episode = await env.run_episode(seed, context)
            for trace in episode.traces:
                store.trace(trace)
            atomic_json(store.root / f"episode-{episode.id}.json", episode.to_record())
            if (reason := failure(store.state.phase, episode)) is not None:
                return store.summary(
                    "incomplete" if "ReviewLimit" in reason else "infrastructure_error", reason
                )
        return store.summary("complete")
    except asyncio.CancelledError:
        store.summary(
            "interrupted", "chapter interrupted; committed progress and available traces are preserved"
        )
        raise
    finally:
        store.close()


def dry_run(chapter: ChapterConfig, settings: Config) -> dict:
    """What a chapter would start from, spending nothing: the parent, the version it makes, the new days with their
    quotas before the plan marks event days, the brief with each entry's needs, and the storylines."""
    world = parent_world(chapter.world)
    try:
        last, on = world.db.execute("SELECT day, date FROM calendar ORDER BY day DESC LIMIT 1").fetchone()
        zone = meta(world, "zone")
        days = calendar_days(date.fromisoformat(on) + timedelta(days=1), chapter.days, zone, last + 1)
        dates = [d["date"] for d in days]
        shares = spread(dates, last + 1, chapter.messages, settings.author.weekend, settings.activity.parts)
        version = version_of(world) + 1
        out = {
            "parent": str(chapter.world),
            "parent_hash": sha256(chapter.world / "world.sqlite"),
            "parent_days": last,
            "messages_before": world.db.execute("SELECT COUNT(*) FROM messages").fetchone()[0],
            "version": version,
            "days": {f"{d['day']} {d['date']}": q["messages"] for d, q in zip(days, shares)},
            "review_days": settings.author.review_days,
            "storylines": {
                "earlier": [r[0] for r in world.db.execute("SELECT id FROM storylines ORDER BY position")],
                "in_all": settings.storylines,
            },
            "brief": [
                s.model_dump(mode="json", exclude_none=True) for s in brief_slots(chapter, settings, version)
            ],
            "author": settings.env.author.model,
            "judge": settings.env.judge.model,
            "seed": settings.seed,
        }
    finally:
        world.close()
    return out
