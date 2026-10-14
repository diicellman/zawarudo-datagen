"""The organization: people, channels, routines, DMs and usergroups as rows, and the background plan of everyday
talk."""

import json
import random
import re
from collections import Counter
from itertools import combinations
from typing import Literal

from pydantic import Field
from worldgen_slack.release import NonEmptyText, SafeId, StrictModel
from worldgen_slack.world import digest, words

from ..config import Ids
from ..workspace import SCHEMA_VERSION
from .cast import Premise, SeedPersona, company_zone, user_id
from .clock import DAY, Part, calendar
from .text import TIME_LITERAL


def channel_id(kind: str, name: str | None, members: list[str]) -> str:
    prefix = {"public": "C", "private": "G", "mpim": "G", "im": "D"}[kind]
    return prefix + digest(["channel", name or sorted(members)])[:10].upper()


class Person(StrictModel):
    user_id: SafeId
    title: NonEmptyText
    team: NonEmptyText
    status_emoji: str | None = Field(
        default=None, pattern=r"^:[a-z0-9_+-]+:$"
    )  # their Slack status, if they set one
    status_text: str | None = Field(default=None, max_length=100)


class Routine(StrictModel):
    kind: NonEmptyText
    probability: float = Field(gt=0, le=1)


class Channel(StrictModel):
    name: str | None = None
    type: Literal["public", "private", "mpim", "im"]
    topic: str = ""
    purpose: str = ""
    members: Ids = Field(min_length=2)
    routines: list[Routine] = Field(default_factory=list)


APP_KINDS = ("deploys", "alerts", "tickets", "ci", "reminders")


class Integration(StrictModel):
    """An app the workspace runs: its bot's name, what it posts, the channel it posts in, the matters it posts about
    (services, checks, ticket subjects, reminders), and for tickets the prefix of their numbers."""

    name: NonEmptyText
    kind: Literal["deploys", "alerts", "tickets", "ci", "reminders"]
    channel: NonEmptyText
    subjects: list[NonEmptyText] = Field(min_length=3, max_length=12)
    prefix: str | None = Field(default=None, pattern=r"^[A-Z]{2,6}$")


class UserGroup(StrictModel):
    """A group people @-mention at once: @oncall, @eng-leads."""

    handle: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    name: NonEmptyText
    description: str = ""
    members: Ids = Field(min_length=2)


class Organization(StrictModel):
    people: list[Person] = Field(min_length=3)
    channels: list[Channel] = Field(min_length=1)
    dm_routines: list[Routine] = Field(default_factory=list)
    integrations: list[Integration] = Field(default_factory=list)
    usergroups: list[UserGroup] = Field(default_factory=list, max_length=6)


class Scene(StrictModel):
    """An everyday conversation code draws for the author's agenda: where, who, which part of which day, what kind,
    about how many lines."""

    id: SafeId
    channel_id: SafeId
    participants: Ids = Field(min_length=1)
    day: int = Field(ge=1)
    part: Part
    situation: NonEmptyText
    length: int = Field(ge=1, le=60)


def channel_rows(world, channels: list[Channel], created_us: int) -> tuple[list[dict], list[dict]]:
    people = {r[0] for r in world.db.execute("SELECT id FROM users")}
    rows, members = [], []
    for c in channels:
        if c.name is not None and not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", c.name):
            raise ValueError(f"channel {c.name!r}: a name is lowercase letters, digits, - and _")
        if unknown := set(c.members) - people:
            raise ValueError(f"channel {c.name or c.members}: unknown members {sorted(unknown)}")
        if (c.type in ("im", "mpim")) != (c.name is None):
            raise ValueError(
                f"channel {c.name or c.members}: public and private channels have a name; DMs do not"
            )
        if c.type == "im" and len(set(c.members)) != 2 or c.type == "mpim" and len(set(c.members)) < 3:
            raise ValueError(f"{c.members}: a DM has exactly 2 members, a group DM at least 3")
        identifier = channel_id(c.type, c.name, c.members)
        rows.append(
            dict(id=identifier, name=c.name, type=c.type, topic=c.topic, purpose=c.purpose)
            | dict(creator_id=c.members[0], created_us=created_us)
        )
        members += [
            dict(channel_id=identifier, user_id=m, joined_us=created_us) for m in dict.fromkeys(c.members)
        ]
    return rows, members


def organize(world, org: Organization, cast: list[SeedPersona], premise: Premise, settings) -> None:
    """S1: people chosen from the candidates and the company's channels. Code adds ids, handles, emails, display
    names, timezones and the calendar."""
    pool = {user_id(p.uuid): p for p in cast}
    chosen = [p.user_id for p in org.people]
    if unknown := set(chosen) - pool.keys():
        raise ValueError(f"people are candidates: {sorted(unknown)} is no candidate's user_id")
    if missing := pool.keys() - set(chosen):
        raise ValueError(f"people are every candidate: {sorted(missing)} has no title and team")
    for c in org.channels:
        if c.name is not None and not 3 <= len(c.routines) <= 5:
            raise ValueError(f"channel {c.name}: routines lists 3 to 5 recurring kinds of conversation")
    if not 3 <= len(org.dm_routines) <= 5:
        raise ValueError("dm_routines lists 3 to 5 recurring kinds of direct conversation")
    place = company_zone(cast, chosen)
    days = calendar(settings.seed, settings.calendar, place)
    start = days[0]["start_us"]
    domain = "".join(words(premise.company)[:2]) + ".example"
    handles = Counter()
    users = []
    for person in org.people:
        seed = pool[person.user_id]
        handle = ".".join(words(seed.name))
        handles[handle] += 1
        handle += str(handles[handle]) if handles[handle] > 1 else ""
        users.append(
            dict(id=person.user_id, handle=handle, real_name=seed.name, display_name=seed.name.split()[0])
            | dict(
                email=f"{handle}@{domain}", title=person.title, tz=seed.timezone, created_us=start - 365 * DAY
            )
            | dict(profile_json=json.dumps({"Team": person.team}))
            | dict(status_emoji=person.status_emoji, status_text=person.status_text)
        )
        if person.status_text and TIME_LITERAL.search(person.status_text):
            raise ValueError(f"{person.user_id}: a status names no time: {person.status_text!r}")
    named = {ch.name for ch in org.channels if ch.name}
    if settings.activity.bot_posts > 0 and not 2 <= len(org.integrations) <= 5:
        raise ValueError(
            "integrations lists 2 to 5 apps the company runs, each posting in one of its channels"
        )
    for app in org.integrations:
        if app.channel not in named:
            raise ValueError(
                f"integration {app.name}: channel is one of the organization's channels, not {app.channel!r}"
            )
        if spoken := [x for x in app.subjects if TIME_LITERAL.search(x)]:
            raise ValueError(f"integration {app.name}: a subject names no time: {spoken}")
    if len({a.name for a in org.integrations}) != len(org.integrations):
        raise ValueError("each integration has a name of its own")
    for app in org.integrations:
        handle = re.sub(r"[^a-z0-9]+", "-", app.name.lower()).strip("-")
        users.append(
            dict(id=bot_id(app.name), handle=handle, real_name=app.name, display_name=app.name, is_bot=1)
            | dict(email=f"{handle}@bots.{domain}", title="App", tz=place, created_us=start - 30 * DAY)
            | dict(
                profile_json=json.dumps(
                    {"subjects": app.subjects} | ({"prefix": app.prefix} if app.prefix else {})
                )
            )
        )
    meta = {
        "schema_version": str(SCHEMA_VERSION),
        "company": premise.company,
        "zone": place,
        "now_us": str(days[-1]["end_us"]),
        "time_style": "habit",  # each person writes times their own way
    }
    with world.batch():
        world.insert("world_meta", [{"key": k, "value": v} for k, v in meta.items()])
        world.insert("calendar", days)
        world.insert("users", users)
        channels = [*org.channels, *direct_messages(org, settings)]
        rows, members = channel_rows(world, channels, start - 30 * DAY)
        world.insert("channels", rows)
        world.insert("members", members)
        by_name = {r["name"]: r["id"] for r in rows if r["name"]}
        world.insert(
            "members",
            [
                dict(channel_id=by_name[a.channel], user_id=bot_id(a.name), joined_us=start - 30 * DAY)
                for a in org.integrations
            ],
        )
        world.insert(
            "integrations",
            [
                dict(user_id=bot_id(a.name), kind=a.kind, channel_id=by_name[a.channel])
                for a in org.integrations
            ],
        )
        if len({g.handle for g in org.usergroups}) != len(org.usergroups):
            raise ValueError("each usergroup has a handle of its own")
        for g in org.usergroups:
            if strangers := set(g.members) - set(chosen):
                raise ValueError(
                    f"usergroup @{g.handle}: members are people of the company, not {sorted(strangers)}"
                )
            group = "S" + digest(["group", g.handle])[:10].upper()
            world.insert(
                "usergroups",
                [
                    dict(
                        id=group,
                        handle=g.handle,
                        name=g.name,
                        description=g.description,
                        created_us=start - 30 * DAY,
                    )
                ],
            )
            world.insert(
                "usergroup_members", [dict(usergroup_id=group, user_id=m) for m in dict.fromkeys(g.members)]
            )


def bot_id(name: str) -> str:
    return "U" + digest(["bot", name])[:10].upper()


def direct_messages(org: Organization, settings) -> list[Channel]:
    """Code's DMs: the seed draws pairs of people, weighted by the channels they share and by a shared team, until
    the company has `dms_per_person` DMs per person."""
    rng = random.Random(digest([settings.seed, "dms"]))
    team = {p.user_id: p.team for p in org.people}
    shared = Counter(
        pair for c in org.channels if c.name is not None for pair in combinations(sorted(set(c.members)), 2)
    )
    have = {tuple(sorted(c.members)) for c in org.channels if c.type == "im"}
    pairs = [p for p in combinations(sorted(team), 2) if p not in have]
    weights = [1 + shared[p] + 2 * (team[p[0]] == team[p[1]]) for p in pairs]
    picked = []
    while pairs and len(picked) < round(settings.activity.dms_per_person * len(team)) - len(have):
        i = rng.choices(range(len(pairs)), weights)[0]
        picked.append(pairs.pop(i))
        weights.pop(i)
    return [Channel(type="im", members=list(p)) for p in picked]


def background_plan(world, org: Organization, activity, seed: int, day: int, budget: int) -> list[Scene]:
    """A day's everyday conversations, drawn by code from the activity targets: `budget` messages in conversations of
    about `conversation_lines` lines, in channels by Zipf weight on their members and in DMs by `dm_share`, in parts
    by the day's rhythm. Each takes its kind from its conversation's routines, drawn by their stated probabilities."""
    rng = random.Random(digest([seed, "agenda", day]))
    span = world.db.execute("SELECT start_us, end_us FROM calendar WHERE day = ?", (day,)).fetchone()[:]
    # Conversations happen in channels open all day (not archived by its end), among people there all day.
    archived = {
        r[0]
        for r in world.db.execute(
            "SELECT id FROM channels WHERE archived_us IS NOT NULL AND archived_us < ?", (span[1],)
        )
    }
    routines = {
        i: c.routines
        for c in org.channels
        if c.routines and (i := channel_id(c.type, c.name, c.members)) not in archived
    }
    named = sorted(
        routines,
        key=lambda c: (
            -world.db.execute("SELECT COUNT(*) FROM members WHERE channel_id = ?", (c,)).fetchone()[0],
            c,
        ),
    )
    dms = [r[0] for r in world.db.execute("SELECT id FROM channels WHERE type = 'im' ORDER BY id")]
    channel_share = 1 - (activity.dm_share if dms and org.dm_routines else 0)
    zipf = [1 / (rank + 1) ** activity.channel_skew for rank in range(len(named))]
    weights = [channel_share * z / sum(zipf) for z in zipf] + [(1 - channel_share) / len(dms)] * len(
        dms
    ) * bool(org.dm_routines)
    places = named + dms * bool(org.dm_routines)
    scenes = []
    while places and budget >= 2:
        length = min(20, budget, 2 + int(rng.expovariate(1 / max(activity.conversation_lines - 2, 0.5))))
        place = rng.choices(places, weights)[0]
        kinds = routines.get(place, org.dm_routines)
        members = [
            r[0]
            for r in world.db.execute(
                "SELECT m.user_id FROM members m JOIN users u ON u.id = m.user_id WHERE m.channel_id = ? AND m.joined_us <= ? AND (m.left_us IS NULL OR m.left_us >= ?) AND u.is_bot = 0 ORDER BY m.user_id",
                (place, *span),
            )
        ]
        talkers = min(len(members), 2 + min(int(rng.expovariate(1.0)), 2))
        scenes.append(
            Scene(
                id=f"bg-{day:02d}-{len(scenes) + 1:03d}",
                channel_id=place,
                participants=rng.sample(members, talkers),
                day=day,
                part=rng.choices(list(activity.parts), list(activity.parts.values()))[0],
                situation=rng.choices([k.kind for k in kinds], [k.probability for k in kinds])[0],
                length=length,
            )
        )
        budget -= length
    return scenes
