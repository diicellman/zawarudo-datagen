"""Building blocks of the checks: refusals, the fixture worlds and settings, and serving a toolset in process."""

import asyncio
import json
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from verifiers.v1.mcp.launch import serve

from ..config import Needs, load_taxonomy
from ..forge.config import ForgeConfig
from ..session import context_of
from ..workspace import Workspace
from ..world.board import Slot
from ..world.cast import Premise, SeedPersona, Typing, user_id
from ..world.clock import PARTS, part_of, present
from ..world.config import WorldConfig
from ..world.ledger import Event, Plan, PlanFact, Storyline
from ..world.organization import Channel, Organization, Person, Routine, organize


def fails(function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except (ValueError, TypeError, LookupError):
        return
    raise AssertionError("expected rejection")


def refused(expected, function, *args, **kwargs):
    """The call is refused, for the reason given."""
    try:
        function(*args, **kwargs)
    except ValueError as error:
        assert expected in str(error), (expected, str(error))
        return
    raise AssertionError(f"not refused: {expected}")


def onward(world) -> str:
    """Where advance moves next: the next part of today, or from the night, tomorrow."""
    part, order = part_of(world, present(world)), list(PARTS)
    return "tomorrow" if part == "night" else order[order.index(part) + 1]


def world_fixture(path):
    """Three people on Chicago time: #releases (public: U1, U2), #exec (private: U1), a DM (U1, U2)."""
    world = Workspace.create(path)
    zone = ZoneInfo("America/Chicago")

    def at(day, hour, minute=0):
        return int(datetime(2026, 6, 14 + day, hour, minute, tzinfo=zone).timestamp() * 1e6)

    def message(id, channel, user, day, hour, text, parent=None, minute=0):
        return dict(
            id=id, channel_id=channel, user_id=user, ts_us=at(day, hour, minute), text=text, parent_id=parent
        )

    with world.batch():
        world.insert("world_meta", [{"key": "now_us", "value": str(at(4, 0))}])
        world.insert(
            "calendar",
            [
                dict(day=d, date=f"2026-06-{14 + d}", start_us=at(d, 0), end_us=at(d + 1, 0))
                for d in (1, 2, 3)
            ],
        )
        world.insert(
            "users",
            [
                dict(
                    id=u,
                    handle=u.lower(),
                    real_name=n,
                    display_name=n.split()[0],
                    email=f"{u}@example.com",
                    tz="America/Chicago",
                    created_us=at(1, 0) - 1,
                )
                for u, n in (("U1", "Alicia Rao"), ("U2", "Owen Sato"), ("U3", "Mia Chen"))
            ],
        )
        world.insert(
            "channels",
            [
                dict(id="C1", name="releases", type="public", created_us=at(1, 0) - 1),
                dict(id="G1", name="exec", type="private", created_us=at(1, 0) - 1),
                dict(id="D1", name=None, type="im", created_us=at(1, 0) - 1),
            ],
        )
        world.insert(
            "members",
            [
                dict(channel_id=c, user_id=u, joined_us=at(1, 0) - 1)
                for c, u in (("C1", "U1"), ("C1", "U2"), ("G1", "U1"), ("D1", "U1"), ("D1", "U2"))
            ]
            + [dict(channel_id="G1", user_id="U3", joined_us=at(1, 0) - 1, left_us=at(1, 1))],
        )
        world.insert(
            "messages",
            [
                message(1, "C1", "U1", 1, 10, "Release 4.2 rollback: release blocked, rollback decided"),
                message(3, "C1", "U2", 2, 9, "notes posted for release 4.2"),
                message(4, "G1", "U1", 1, 11, "private: rollback budget approved"),
                message(5, "D1", "U2", 1, 12, "can you check the rollback"),
                message(6, "C1", "U1", 1, 11, "deleted rollback note", parent=1) | {"is_deleted": 1},
            ],
        )
        world.insert("messages", [message(2, "C1", "U2", 1, 10, "ack, rolling back now", parent=1, minute=5)])
        world.insert("storylines", [dict(id="r", summary="the 4.2 release", position=1)])
        world.insert(
            "scenes",
            [
                dict(
                    id="s1",
                    channel_id="C1",
                    storyline="r",
                    day=1,
                    part="morning",
                    slot_start_us=at(1, 9),
                    slot_end_us=at(1, 12),
                    situation="decide",
                )
            ],
        )
        world.insert("scene_messages", [dict(message_id=m, scene_id="s1") for m in (1, 2, 6)])
        world.insert(
            "facts",
            [
                dict(
                    id="f1",
                    storyline="r",
                    subject="Release 4.2",
                    attribute="decision",
                    value="rollback",
                    channel_id="C1",
                    author_id="U1",
                    day=1,
                    summary="rollback decided",
                )
            ],
        )
        world.insert("evidence", [dict(fact_id="f1", message_id=1, role="anchor", anchor_token="rollback")])
        world.insert(
            "tasks",
            [
                dict(
                    id="t1",
                    category="lookup",
                    level=1,
                    actor_id="U2",
                    question="What was decided for 4.2?",
                    answer_type="text",
                    gold_source="ledger",
                    gold_sql="SELECT value AS answer FROM facts WHERE id = 'f1'",
                )
            ],
        )
        world.insert("task_facts", [dict(task_id="t1", fact_id="f1")])
    return world, at, message


def seed_person(i, zone="America/Chicago"):
    names = ["Alicia Rao", "Owen Sato", "Mia Chen", "Raj Patel", "Lena Ortiz", "Tom Berg"]
    return SeedPersona(
        uuid=f"p{i}",
        name=names[i],
        sex="F",
        age=30 + i,
        marital_status="single",
        education_level="bachelors",
        bachelors_field="cs",
        occupation="software_developer",
        city="Austin",
        state="TX",
        country="USA",
        timezone=zone,
        persona="calm",
        professional_persona="engineer",
        cultural_background="midwest",
        skills=["go"],
        hobbies=["chess"],
        typing=Typing(
            id=f"t{i}",
            messages=40,
            median_words=8,
            short_share=0.2,
            long_share=0.1,
            question_share=0.2,
            lowercase_share=0.3,
            emoji_share=0.1,
        ),
    )


# The checks' own taxonomy, a copy of configs/worldgen_slack/taxonomy.toml, and the bands its forge configs state.
TAXONOMY = Path(__file__).parent / "fixtures/taxonomy.toml"


BANDS = [(0.75, 1.0), (0.5, 0.75), (0.25, 0.5), (0.0, 0.25)]


def contracts_settings(root) -> WorldConfig:
    """A world's settings, with the checks' taxonomy as a chapter's."""
    gaps = root / "gaps.json"
    gaps.write_text(json.dumps({"quantiles": [i * 7.2 for i in range(1001)]}) + "\n")
    taxonomy = load_taxonomy(TAXONOMY)
    return WorldConfig.model_validate(
        {
            "sector": "software",
            "seed": 3,
            "output": str(root / "run"),
            "personas": {"path": str(root / "p.jsonl"), "typing": str(root / "t.jsonl"), "gaps": str(gaps)},
            "storylines": 2,
            "taxonomy": taxonomy.model_dump(mode="json")["categories"],
            "styles": taxonomy.styles,
        }
    )


def forge_of(settings: WorldConfig, **changes) -> ForgeConfig:
    """A forge on a world with these settings, its proposer seated as the world's author is (as each forge config
    states it)."""
    seats = settings.env.model_dump(mode="json")
    return ForgeConfig.model_validate(
        {
            "world": str(settings.output),
            "output": str(settings.output),
            "env": {"proposer": seats["author"], "judge": seats["judge"]},
        }
        | changes
    )


def needing(settings, category, level, **needs):
    """The settings with one level of a category needing only these."""
    spec = settings.taxonomy[category]
    tables = [Needs()] * len(spec.levels)
    tables[level - 1] = Needs(**needs)
    return settings.model_copy(
        update={"taxonomy": settings.taxonomy | {category: spec.model_copy(update={"needs": tables})}}
    )


def needless(taxonomy):
    """The taxonomy without its levels' needs, for checks whose tasks are not built to them."""
    return {name: c.model_copy(update={"needs": None}) for name, c in taxonomy.items()}


def world_settings(root, **sections: dict) -> WorldConfig:
    """The checks' settings for writing a world: their taxonomy without needs, and each named section's fields
    changed."""
    base = contracts_settings(root)
    changed = {name: getattr(base, name).model_copy(update=fields) for name, fields in sections.items()}
    return base.model_copy(update=changed | {"taxonomy": needless(base.taxonomy)})


def ledger_plan(ops, leads, a, b, **changes):
    """Two storylines on an organized world: a rollback decided on day 1 (f1), service back at e2 on day 2 (f2), and
    an audit's dry run and owner at e1 on day 3 (f3, f4)."""
    events = [
        Event(id="e1", storyline="s2", title="release window", day=3, time="10:00"),
        Event(id="e2", storyline="s1", title="service restored", day=2, time="14:00"),
    ]
    facts = [
        PlanFact(
            id="f1",
            storyline="s1",
            subject="Release 4.2",
            attribute="decision",
            value="rollback",
            anchor="rollback",
            channel_id=ops,
            author_id=a,
            day=1,
            summary="s",
        ),
        PlanFact(
            id="f2",
            storyline="s1",
            subject="Release 4.2",
            attribute="state",
            value="service back",
            channel_id=ops,
            author_id=b,
            day=2,
            after=["f1"],
            event="e2",
            kind="happened",
            summary="s",
        ),
        PlanFact(
            id="f3",
            storyline="s2",
            subject="Audit",
            attribute="window",
            value="dry run",
            anchor="dry run",
            channel_id=leads,
            author_id=a,
            day=3,
            event="e1",
            kind="scheduled",
            summary="s",
        ),
        PlanFact(
            id="f4",
            storyline="s2",
            subject="Audit",
            attribute="owner",
            value="Owen",
            channel_id=ops,
            author_id=a,
            day=3,
            after=["f3"],
            supersedes="f3",
            event="e1",
            kind="scheduled",
            summary="s",
        ),
    ]
    document = dict(
        storylines=[Storyline(id="s1", summary="the 4.2 release"), Storyline(id="s2", summary="the audit")],
        events=events,
        facts=facts,
    )
    return Plan(**document | changes)


def organized(root, settings, apps=(), groups=()):
    """An organized world on Chicago time: people a, b, c (and d, in New York), #ops (public: everyone), #leads
    (private: a, b), DMs a-c and code's own, a 10-day calendar from a Monday; `apps`, its integrations."""
    cast = [seed_person(i, "America/New_York" if i == 3 else "America/Chicago") for i in range(4)]
    a, b, c, d = (user_id(p.uuid) for p in cast)
    routines = [
        Routine(kind=k, probability=p) for k, p in (("standup", 0.5), ("review", 0.3), ("lunch", 0.2))
    ]
    org = Organization(
        people=[Person(user_id=u, title="Engineer", team="Platform") for u in (a, b, c, d)],
        channels=[
            Channel(name="ops", type="public", members=[a, b, c, d], routines=routines),
            Channel(name="leads", type="private", members=[a, b], routines=routines),
            Channel(type="im", members=[a, c]),
        ],
        dm_routines=routines,
        integrations=list(apps),
        usergroups=list(groups),
    )
    premise = Premise(
        company="Lattice Creek", niche="n", region="r", size="s", culture="c", cast="x", staffing={}
    )
    root.mkdir(parents=True, exist_ok=True)
    world = Workspace.create(root / "world.sqlite")
    with world.trial() as copy:
        organize(copy, org, cast, premise, settings)
    return world, cast, org, (a, b, c, d)


def author_context(settings, cast, org, root, agenda=None):
    """What the author's world tools are configured with, as the env builds it; `agenda` maps a day to its drawn
    scenes."""
    state = SimpleNamespace(
        quota=[
            Slot(id=f"t{i}", category=c, level=n, concept="a concept", style="a style")
            for i, (c, n) in enumerate((("semantic", 3), ("lookup", 1)), 1)
        ],
        cast=cast,
        plans={
            "agenda": {
                str(d): [s.model_dump(mode="json") for s in scenes] for d, scenes in (agenda or {}).items()
            }
        },
    )
    return context_of(settings, state, root, org.model_dump(mode="json"))


@asynccontextmanager
async def mcp_client(toolset):
    """A toolset served as verifiers launches one (python -m, in a process of its own), and an MCP session on it."""
    async with (
        asyncio.timeout(30),
        serve(toolset) as url,
        streamable_http_client(url) as (reader, writer),
        ClientSession(reader, writer) as client,
    ):
        await client.initialize()
        yield client


async def served_tools(toolset) -> dict[str, str]:
    async with mcp_client(toolset) as client:
        return {t.name: t.description for t in (await client.list_tools()).tools}


def with_state(toolset):
    """Serve calls against an in-memory state, as the framework's state channel would."""
    stored = {"state": toolset._state_cls()}

    async def pull():
        return stored["state"].model_copy(deep=True)

    async def push(before):
        stored["state"] = toolset.state.model_copy(deep=True)

    toolset._pull_state, toolset._push_state = pull, push
    return stored
