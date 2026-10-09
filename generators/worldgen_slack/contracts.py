"""What agents write, how code turns it into the world, and the deterministic rules in between."""

import bisect
import json
import random
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from itertools import combinations
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Annotated, Literal, Self
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import AfterValidator, BeforeValidator, Field, model_validator

from worldgen_slack.db import SHOWN, World, canonical, digest
from worldgen_slack.db import words
from worldgen_slack.dataset import NonEmptyText, SafeId, StrictModel


def _meaningful_tokens(value: str) -> set[str]:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return {token for token in re.findall(r"[^\W_]+", normalized) if len(token) >= 2}


def normalized(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.casefold()))


def zone(value: str) -> str:
    try:
        ZoneInfo(value)
    except (ZoneInfoNotFoundError, ValueError) as error:
        raise ValueError(f"unknown IANA timezone {value!r}") from error
    return value


class Premise(StrictModel):
    company: NonEmptyText
    niche: NonEmptyText
    region: NonEmptyText
    size: NonEmptyText
    culture: NonEmptyText
    cast: NonEmptyText
    staffing: dict[NonEmptyText, Annotated[int, Field(ge=1)]]


class Premises(StrictModel):
    premises: list[Premise] = Field(min_length=2)


def pick_premise(premises: Premises, count: int, used: list[str], seed: int) -> Premise:
    """Entropy comes from code: the model proposes distinct options, the run seed chooses one."""
    names = [normalized(p.company).split()[0] for p in premises.premises]
    taken = {normalized(name).split()[0] for name in used if normalized(name)}
    if len(names) != count or len(set(names)) != count:
        raise ValueError(f"propose exactly {count} premises whose company names start with distinct words")
    if reused := sorted(set(names) & taken):
        raise ValueError(f"company names reuse names from used_names: {reused}")
    return random.Random(seed).choice(premises.premises)


class Typing(StrictModel):
    """How one real Slack user types: medians and shares over their messages."""

    id: SafeId
    messages: int
    median_words: float
    short_share: float
    long_share: float
    question_share: float
    lowercase_share: float
    emoji_share: float


class SeedPersona(StrictModel):
    """A person from the seed persona file (scripts/worldgen_slack/personas.sql); `typing` is paired by the seed."""

    uuid: SafeId
    name: NonEmptyText
    sex: NonEmptyText
    age: int
    marital_status: NonEmptyText
    education_level: NonEmptyText
    bachelors_field: str | None
    occupation: NonEmptyText
    city: NonEmptyText
    state: NonEmptyText
    country: NonEmptyText
    timezone: Annotated[str, AfterValidator(zone)]
    persona: NonEmptyText
    professional_persona: NonEmptyText
    cultural_background: NonEmptyText
    skills: list[str]
    hobbies: list[str]
    typing: Typing | None = None


def census(personas, used=()) -> tuple[list[str], Counter]:
    """The persona file's countries, and how many people unused elsewhere in the corpus hold each occupation."""
    people, taken = [json.loads(line) for line in personas.path.read_text().splitlines()], set(used)
    supply = Counter(p["occupation"] for p in people if p["name"] not in taken)
    return sorted({p["country"] for p in people}), supply


def pick_cast(personas, seed: int, used: list[str], staffing: dict[str, int]) -> list[SeedPersona]:
    """Entropy comes from code: the run seed draws the premise's staffing, occupation by occupation, from people
    unused elsewhere in the corpus, and pairs each candidate with a real user's typing."""
    rng, taken, cast, roles = random.Random(seed), set(used), [], Counter(staffing)
    lines = personas.path.read_text().splitlines()
    for line in rng.sample(lines, len(lines)):
        if not +roles:
            break
        person = json.loads(line)
        if roles[person["occupation"]] > 0 and person["name"] not in taken:
            roles[person["occupation"]] -= 1
            cast.append(SeedPersona.model_validate(person))
    if short := +roles:
        raise ValueError(f"the persona file has too few people unused in the corpus for {dict(short)}")
    typing = rng.sample(personas.typing.read_text().splitlines(), len(cast))
    return [p.model_copy(update={"typing": Typing.model_validate_json(t)}) for p, t in zip(cast, typing)]


class Slot(StrictModel):
    """One task a world holds, drawn by the run seed: its id (the task's id), its taxonomy cell, the concept it
    requires and how it is asked."""

    id: SafeId
    category: NonEmptyText
    level: int = Field(ge=1)
    concept: NonEmptyText
    style: NonEmptyText


def quota(taxonomy: dict, styles: list[str], seed: int, count: int) -> list[Slot]:
    """Entropy comes from code: a world's tasks walk a seeded shuffle of every taxonomy cell, again and again when it
    holds more tasks than cells. A cell's repeats take its level's concepts in a seeded order, so a repeat requires
    another concept while one is left, and every task draws a question style, as DataDesigner's category and
    subcategory samplers do."""
    rng = random.Random(seed)
    cells = [(name, level) for name, c in taxonomy.items() for level in range(1, len(c.levels) + 1)]
    order = rng.sample(cells, len(cells))
    concepts = {(c, level): rng.sample(taxonomy[c].concepts[level - 1], len(taxonomy[c].concepts[level - 1])) for c, level in order}  # fmt: skip
    slots = []
    for i in range(count):
        (category, level), n = order[i % len(order)], i // len(order)
        pool = concepts[(category, level)]
        slots.append(Slot(id=f"{category}-l{level}-{n + 1}", category=category, level=level, concept=pool[n % len(pool)], style=rng.choice(styles)))  # fmt: skip
    return slots


def cards(cast: list) -> dict[str, dict]:
    """Each person's typing card, by user id: how they write, as the author and the judge read it."""
    return {user_id(p.uuid): p.typing.model_dump(exclude={"id", "messages"}) for p in cast if p.typing}


def selves(cast: list) -> dict[str, dict]:
    """Each person as their seed draws them, by user id: who they are beyond their title, as the author reads it. The
    seed's own job was drawn for another occupation, so its professional persona is left out."""
    return {
        user_id(p.uuid): {
            "age": p.age, "home": f"{p.city}, {p.state}", "education": p.education_level.replace("_", " "),
            "persona": p.persona, "background": p.cultural_background, "hobbies": p.hobbies[:6],
        }
        for p in cast
    }  # fmt: skip


def user_id(seed_uuid: str) -> str:
    return "U" + digest(["user", seed_uuid])[:10].upper()


def channel_id(kind: str, name: str | None, members: list[str]) -> str:
    prefix = {"public": "C", "private": "G", "mpim": "G", "im": "D"}[kind]
    return prefix + digest(["channel", name or sorted(members)])[:10].upper()


# ---------------------------------------------------------------------- the documents agents write

Clock = Annotated[str, Field(pattern=r"^([01]\d|2[0-3]):[0-5]\d$")]
Zone = Annotated[str, AfterValidator(zone)]
Part = Literal["early", "morning", "afternoon", "evening", "night"]
# A part of the day on the company clock: [start hour, end hour).
PARTS = {"early": (6, 9), "morning": (9, 12), "afternoon": (12, 17), "evening": (17, 21), "night": (21, 24)}


def listed(value):
    """Models often write null for an empty list, a bare string for a one-item list, or a list as a string."""
    if value is None:
        return []
    if isinstance(value, str):
        return json.loads(value) if value.startswith("[") else [value]
    return value


Ids = Annotated[list[SafeId], BeforeValidator(listed)]


class Moment(StrictModel):
    """A moment the story is about: a calendar day, a time on a clock, and that clock's IANA zone."""

    day: int = Field(ge=1)
    time: Clock
    zone: Zone


class Person(StrictModel):
    user_id: SafeId
    title: NonEmptyText
    team: NonEmptyText


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


class Organization(StrictModel):
    people: list[Person] = Field(min_length=3)
    channels: list[Channel] = Field(min_length=1)
    dm_routines: list[Routine] = Field(default_factory=list)


class Storyline(StrictModel):
    id: SafeId
    summary: NonEmptyText


class Task(StrictModel):
    id: SafeId
    category: NonEmptyText
    level: int = Field(ge=1)
    actor_id: SafeId
    question: NonEmptyText
    answer_type: Literal["text", "set", "number", "refusal", "status"]
    gold_sql: NonEmptyText
    facts: Ids = Field(default_factory=list)
    # The gold query with one condition relaxed (a day, a channel, a reaction left out): what a hasty reader answers.
    near_sql: list[NonEmptyText] = Field(default_factory=list, max_length=3)


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


class Reaction(StrictModel):
    user_id: SafeId
    emoji: Annotated[str, Field(pattern=r"^[a-z0-9_+'-]{1,40}$")]


# Words: a short line has at most SHORT, a long one more than LONG, as the typing profiles measure.
SHORT, LONG = 4, 20


# ---------------------------------------------------------------------- time: code owns every timestamp

PLACEHOLDER = re.compile(r"\{at:([A-Za-z0-9][A-Za-z0-9_.-]*)\}")
MENTION = re.compile(r"<@([A-Za-z0-9]+)>")
# ponytail: clock times and calendar dates only; relative words ("tomorrow", "Friday") are left to the judge,
# and "may 2" (the verb) is read as a date.
TIME_LITERAL = re.compile(
    r"\b(?:\d{1,2}:\d{2}|\d{1,2}\s?[ap]\.?m\b\.?|\d{4}-\d{2}-\d{2}|"
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sept?(?:ember)?|"
    r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?\s\d{1,2}(?:st|nd|rd|th)?\b)",
    re.IGNORECASE,
)


def us(moment: datetime) -> int:
    return int(moment.timestamp() * 1_000_000)


def company_zone(cast: list[SeedPersona], chosen: list[str]) -> str:
    """The company clock: the timezone most of its people live in (alphabetical on a tie)."""
    counts = Counter(p.timezone for p in cast if user_id(p.uuid) in chosen)
    return min(counts, key=lambda z: (-counts[z], z))


def calendar(seed: int, settings, place: str) -> list[dict]:
    """Day 1 is a Monday of the configured year picked by the run seed; days are company-clock midnights."""
    first = date(settings.year, 1, 1)
    mondays = [first + timedelta(days=d) for d in range(366) if (first + timedelta(days=d)).weekday() == 0]
    start, tz = random.Random(seed).choice(mondays), ZoneInfo(place)
    days = [start + timedelta(days=n) for n in range(settings.days + 1)]
    return [
        {
            "day": n + 1,
            "date": d.isoformat(),
            "start_us": us(datetime(d.year, d.month, d.day, tzinfo=tz)),
            "end_us": us(datetime(nd.year, nd.month, nd.day, tzinfo=tz)),
        }
        for n, (d, nd) in enumerate(zip(days, days[1:]))
    ]


def at(world, moment: Moment) -> int:
    row = world.db.execute("SELECT date FROM calendar WHERE day = ?", (moment.day,)).fetchone()
    if row is None:
        raise ValueError(f"day {moment.day} is outside the calendar")
    d, (hour, minute) = date.fromisoformat(row[0]), map(int, moment.time.split(":"))
    return us(datetime(d.year, d.month, d.day, hour, minute, tzinfo=ZoneInfo(moment.zone)))


def window(world, day: int, part: str) -> tuple[int, int]:
    """A part of a calendar day on the company clock, in microseconds."""
    row = world.db.execute("SELECT date FROM calendar WHERE day = ?", (day,)).fetchone()
    if row is None:
        raise ValueError(f"day {day} is outside the calendar")
    d, tz = date.fromisoformat(row[0]), ZoneInfo(world_meta(world, "zone"))
    start, end = PARTS[part]
    return us(datetime(d.year, d.month, d.day, start, tzinfo=tz)), us(
        datetime(d.year, d.month, d.day, tzinfo=tz) + timedelta(hours=end)
    )


def clock(world, moment_us: int) -> str:
    """A moment on the company clock, as an author reads it."""
    return (
        f"{datetime.fromtimestamp(moment_us / 1e6, ZoneInfo(world_meta(world, 'zone'))):%a %Y-%m-%d %H:%M %Z}"
    )


def world_meta(world, key: str) -> str:
    return world.db.execute("SELECT value FROM world_meta WHERE key = ?", (key,)).fetchone()[0]


def render(moment_us: int, place: str, said_us: int) -> str:
    """A moment as its writer reads it on their own clock; the day is named when it is not the day of writing."""
    tz = ZoneInfo(place)
    when, said = datetime.fromtimestamp(moment_us / 1e6, tz), datetime.fromtimestamp(said_us / 1e6, tz)
    return f"{when:%H:%M %Z}" if when.date() == said.date() else f"{when:%a %b} {when.day} {when:%H:%M %Z}"


class Gaps:
    """How long real Slack users take to answer: seconds between consecutive messages of a channel, as quantiles
    (scripts/worldgen_slack/personas.sql). A line with no pause takes a gap under an hour; "hours" takes one above."""

    def __init__(self, path: Path):
        self.quantiles = json.loads(path.read_text().splitlines()[0])["quantiles"]
        self.hour = bisect.bisect_left(self.quantiles, 3600) / (len(self.quantiles) - 1)

    def sample(self, rng: random.Random, pause: str) -> int:
        u = rng.random() * self.hour if pause == "" else self.hour + rng.random() * (1 - self.hour)
        position = u * (len(self.quantiles) - 1)
        low = int(position)
        high = min(low + 1, len(self.quantiles) - 1)
        seconds = self.quantiles[low] + (self.quantiles[high] - self.quantiles[low]) * (position - low)
        return int(seconds * 1_000_000) + rng.randrange(1, 1_000_000)


# ---------------------------------------------------------------------- documents → rows, checked

LEDGER_TABLES = {"facts", "fact_relations", "evidence"}
STRUCTURE = {"channels", "members", "messages", "message_mentions", "reactions", "thread_stats"}


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
        )
    meta = {
        "schema_version": "1",
        "company": premise.company,
        "zone": place,
        "now_us": str(days[-1]["end_us"]),
    }
    with world.batch():
        world.insert("world_meta", [{"key": k, "value": v} for k, v in meta.items()])
        world.insert("calendar", days)
        world.insert("users", users)
        channels = [*org.channels, *direct_messages(org, settings)]
        rows, members = channel_rows(world, channels, start - 30 * DAY)
        world.insert("channels", rows)
        world.insert("members", members)


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
    routines = {channel_id(c.type, c.name, c.members): c.routines for c in org.channels if c.routines}
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
                "SELECT user_id FROM members WHERE channel_id = ? AND left_us IS NULL ORDER BY user_id",
                (place,),
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


DAY = 86_400_000_000


def same_rows(rows: list[dict], again: list[dict], back) -> bool:
    """Whether a renumbered copy's rows, their message ids mapped back, are the world's rows."""
    mapped = [
        r | {"message_id": back(r["message_id"])} if r.get("message_id") is not None else r for r in again
    ]
    return sorted(map(canonical, rows)) == sorted(map(canonical, mapped))


def check_task(world, task: Task, settings, renumbered=None) -> list[dict]:
    """T1-T5, and T7 given the world `renumbered`: the gold query runs as the task's actor, reads what the task's
    category promises, and finds its messages by what they say; the rows are the gold answer. Returns them."""
    category = settings.taxonomy.get(task.category)
    if category is None:
        raise ValueError(f"{task.id}: category must be one of {sorted(settings.taxonomy)}")
    if not 1 <= task.level <= len(category.levels) or task.answer_type not in category.answer_types:
        raise ValueError(
            f"{task.id}: a {task.category} task has a level from 1 to {len(category.levels)} and an answer type in "
            f"{category.answer_types}"
        )
    try:
        out = world.gold(task.actor_id, task.gold_sql, max_rows=settings.tasks.max_answer_rows)
    except LookupError:
        raise ValueError(f"{task.id}: actor_id {task.actor_id} is not a person in the world") from None
    except ValueError as error:
        raise ValueError(f"{task.id}: {error}") from None
    read, rows, source = set(out["tables"]), out["rows"], gold_source(task, settings)
    if renumbered is not None:
        copy, back = renumbered
        try:
            again = copy.gold(task.actor_id, task.gold_sql, max_rows=settings.tasks.max_answer_rows)["rows"]
        except ValueError:
            again = None
        if again is None or not same_rows(rows, again, back):
            raise ValueError(
                f"{task.id}: the gold query finds messages by what the question names (words, people, channels, "
                "threads, reactions, times), never by a message id or by id order"
            )
    if source == "sql" and read & LEDGER_TABLES:
        raise ValueError(
            f"{task.id}: a {task.category} gold query reads only the workspace, not {sorted(read & LEDGER_TABLES)}"
        )
    if source in ("ledger", "hybrid") and not read & LEDGER_TABLES:
        raise ValueError(
            f"{task.id}: a {task.category} {task.answer_type} gold query reads the facts it answers from"
        )
    if source == "hybrid" and not read & STRUCTURE:
        raise ValueError(
            f"{task.id}: a hybrid gold query also reads the workspace (messages, members, reactions)"
        )
    if source in ("ledger", "hybrid") and not task.facts:
        raise ValueError(f"{task.id}: name the facts its answer rests on in facts")
    unshown = [r for r in out["reads"] if (t := r.split(".")[0]) in SHOWN and r.split(".")[1] not in SHOWN[t]]
    if unshown:
        raise ValueError(
            f"{task.id}: the gold query reads {unshown}, which no Slack tool shows its solver; ask only about what "
            "the tools show"
        )
    if task.answer_type == "refusal":
        if rows:
            raise ValueError(
                f"{task.id}: a refusal's gold query returns no rows as its actor; it returns {len(rows)}"
            )
        return rows
    if "answer" not in out["columns"]:
        raise ValueError(f"{task.id}: the gold query returns an `answer` column")
    if not 1 <= len(rows) <= settings.tasks.max_answer_rows:
        raise ValueError(
            f"{task.id}: the gold query returns 1 to {settings.tasks.max_answer_rows} rows as its actor, not "
            f"{len(rows) if len(rows) <= settings.tasks.max_answer_rows else 'more'}"
        )
    if task.answer_type != "set" and len(rows) != 1:
        raise ValueError(
            f"{task.id}: a {task.answer_type} answer is one row; use answer_type set for several"
        )
    if task.answer_type == "status":
        unsettled(world, task, rows)
    for near in task.near_sql:  # a hasty reader's answer: another one, as the actor sees it
        try:
            relaxed = world.gold(task.actor_id, near, max_rows=50)["rows"]
        except ValueError as error:
            raise ValueError(f"{task.id}: its near-miss query fails: {error}") from None
        if sorted(str(r.get("answer")) for r in relaxed) == sorted(str(r.get("answer")) for r in rows):
            raise ValueError(
                f"{task.id}: its near-miss query returns its gold answer; relax a condition that matters"
            )
    if given := [
        str(r["answer"])
        for r in rows
        if task.answer_type != "number"
        and not _meaningful_tokens(str(r["answer"])) - _meaningful_tokens(task.question)
    ]:
        raise ValueError(f"{task.id}: the question gives away its answer {given}")
    return rows


def record_task(world, task: Task, settings, slot: Slot, needs: bool = True) -> dict:
    """T6: a task fills its slot. Code checks it (its slot's cell, a question no other task asks, facts its actor can
    read, T1-T5, T7, and, with `needs`, its level's needs) and inserts it with its slot's concept and its gold rows.
    Returns its gold rows and code's measures. Run inside `World.trial()`."""
    if (task.id, task.category, task.level) != (slot.id, slot.category, slot.level):
        raise ValueError(f"{slot.id} is a {slot.category} level {slot.level} task, with the slot's id")
    asked = {
        normalized(r[0]) for r in world.db.execute("SELECT question FROM tasks WHERE id != ?", (task.id,))
    }
    if normalized(task.question) in asked:
        raise ValueError(f"{task.id}: a task asks a question no other task asks")
    for fact in (
        task.facts if task.answer_type not in ("refusal", "status") else ()
    ):  # a refusal or a status may rest on the truth out of sight
        row = world.db.execute("SELECT channel_id FROM facts WHERE id = ?", (fact,)).fetchone()
        if row and not readable(world, task.actor_id, row[0]):
            raise ValueError(f"{task.id}: its actor cannot read {row[0]}, where {fact} is stated")
    with world.renumbered() as renumbered:
        gold = check_task(world, task, settings, renumbered)
    with world.batch():
        world.insert("tasks", [task_row(task, settings) | {"concept": slot.concept}])
        world.insert("task_facts", [dict(task_id=task.id, fact_id=f) for f in dict.fromkeys(task.facts)])
    world.db.execute("UPDATE tasks SET gold_json = ? WHERE id = ?", (json.dumps(gold), task.id))
    tables = world.gold(task.actor_id, task.gold_sql, max_rows=settings.tasks.max_answer_rows)["tables"]
    measured = measures(world, task.id, gold, tables)
    spec = settings.taxonomy.get(task.category)
    if needs and spec and spec.needs and (short := unmet(spec.needs[task.level - 1], measured)):
        raise ValueError(f"{task.id}: a level-{task.level} {task.category} task needs " + "; ".join(short))
    return {"gold": gold} | measured


def unmet(needs, measured: dict) -> list[str]:
    """Each of a level's needs that a task's measures fall short of, with what was measured; a need whose measure is
    not given (a board entry has only its facts') is not checked."""
    out = []
    if measured["channels"] < needs.channels:
        out.append(f"its facts first stated in at least {needs.channels} channels (they are in {measured['channels']})")  # fmt: skip
    if measured["relations"] < needs.relations:
        out.append(f"at least {needs.relations} supersedes or after relations among its facts (it has {measured['relations']})")  # fmt: skip
    if measured["decoys"] < needs.decoys:
        out.append(f"at least {needs.decoys} near-misses: decoys, or values whose change its actor cannot see, on its answer's subject and attribute, that its actor can read and no change it can see retracts (it has {measured['decoys']})")  # fmt: skip
    if needs.hidden and measured.get("hidden") is False:
        out.append("its answer stated where it is harder to see than its near-misses: in a thread reply, in a DM, or on a later day than every near-miss")  # fmt: skip
    if needs.named is not None and measured.get("named", 0) > needs.named:
        out.append(f"a question naming at most {needs.named} of its evidence's channels and identifiers (it names {measured['named']})")  # fmt: skip
    return out


def needs_text(needs) -> str:
    """A level's needs, as the author reads them in now.md."""
    many = lambda n, word: f"{n} {word}" + ("s" if n > 1 else "")  # noqa: E731
    out = []
    if needs.channels:
        out.append(f"facts first stated in {many(needs.channels, 'channel')}")
    if needs.relations:
        out.append(f"{many(needs.relations, 'relation')} among its facts")
    if needs.decoys:
        out.append(f"{many(needs.decoys, 'near-miss')} on its answer's subject and attribute that its actor can read, unretracted in its sight")  # fmt: skip
    if needs.hidden:
        out.append("its answer in a thread reply, a DM, or later than its near-misses")
    if needs.named is not None:
        out.append("a question that names none of its evidence's channels and identifiers" if needs.named == 0 else f"a question naming at most {needs.named} of its evidence's channels and identifiers")  # fmt: skip
    return "; ".join(out)


IDENTIFIER = re.compile(
    r"(?<![\w-])(?=[\w-]*\d)(?=[\w-]*[^\W\d_])\w[\w-]*\w(?![\w-])"
)  # PWSQL-03, CHG-2291, CU12


def named(world, question: str, evidence: list[int]) -> int:
    """How many of the evidence's channel names and identifiers the question names."""
    marks = ", ".join("?" * len(evidence))
    rows = world.db.execute(f"SELECT m.text, c.name FROM messages m JOIN channels c ON c.id = m.channel_id WHERE m.id IN ({marks})", evidence).fetchall()  # fmt: skip
    terms = {n.casefold() for _, n in rows if n} | {
        t.casefold() for text, _ in rows for t in IDENTIFIER.findall(text)
    }
    return sum(bool(re.search(rf"(?<![\w-]){re.escape(t)}(?![\w-])", question, re.IGNORECASE)) for t in terms)


def fact_measures(world, ids: list[str], actor: str | None = None, gold=None, refusal: bool = False) -> dict:
    """What a task's facts give it: the channels they are first stated in, the relations among them, and its
    near-misses (`decoys`): facts on an answer fact's subject and attribute, other than its answer facts, that `actor`
    can read and that no fact `actor` can see supersedes: an unretracted decoy, or a value whose change is out of the
    actor's sight. A near-miss the world corrects in plain sight is no near-miss: S2's solver followed every such
    chain. Planned before any actor is chosen, a near-miss counts wherever it is stated, and a change counts as seen
    when it is public."""
    marks = ", ".join("?" * len(ids))
    channels = world.db.execute(f"SELECT COUNT(DISTINCT channel_id) FROM facts WHERE id IN ({marks})", ids).fetchone()[0]  # fmt: skip
    relations = world.db.execute(f"SELECT COUNT(*) FROM fact_relations WHERE src_fact IN ({marks}) AND dst_fact IN ({marks})", [*ids, *ids]).fetchone()[0]  # fmt: skip
    answers = answer_facts(world, ids, actor, gold, refusal)
    return {"channels": channels, "relations": relations, "decoys": len(near_misses(world, answers, actor))}


def places(world, fact: str) -> set[str]:
    """Where a fact is stated, or, before it is, the channel it is planned to be first stated in."""
    stated = {r[0] for r in world.db.execute("SELECT DISTINCT m.channel_id FROM evidence e JOIN messages m ON m.id = e.message_id WHERE e.fact_id = ?", (fact,))}  # fmt: skip
    return stated or {world.db.execute("SELECT channel_id FROM facts WHERE id = ?", (fact,)).fetchone()[0]}


def answer_facts(
    world, ids: list[str], actor: str | None = None, gold=None, refusal: bool = False
) -> list[str]:
    """Which of a task's facts it answers with: those its gold answers are the values of; for a refusal, those out of
    its actor's sight (the truth it cannot see); else, or when none is, those that are no decoy and that no fact
    supersedes; and for a task about an earlier state, its facts that are no decoy (a later value is its near-miss)."""
    marks = ", ".join("?" * len(ids))
    values = {str(v).strip().casefold() for v in gold or ()}
    if refusal and actor:
        found = [f for f in ids if not any(readable(world, actor, c) for c in places(world, f))]
    else:
        found = [f for f, v in world.db.execute(f"SELECT id, value FROM facts WHERE id IN ({marks})", ids) if v.strip().casefold() in values]  # fmt: skip
    current = f"SELECT id FROM facts WHERE id IN ({marks}) AND is_decoy = 0 AND id NOT IN (SELECT dst_fact FROM fact_relations WHERE kind = 'supersedes')"  # fmt: skip
    earlier = f"SELECT id FROM facts WHERE id IN ({marks}) AND is_decoy = 0"
    return (
        found
        or [f for (f,) in world.db.execute(current, ids)]
        or [f for (f,) in world.db.execute(earlier, ids)]
    )


def near_misses(world, answers: list[str], actor: str | None = None) -> list[str]:
    """The near-misses of a task answering with facts `answers` (`fact_measures`), as fact ids."""
    reads = (lambda c: readable(world, actor, c)) if actor else (lambda c: True)

    def sees(channel: str) -> bool:
        if actor:
            return readable(world, actor, channel)
        return (
            world.db.execute("SELECT type FROM channels WHERE id = ?", (channel,)).fetchone()[0] == "public"
        )

    out = []
    for answer in answers:
        subject, attribute = world.db.execute("SELECT lower(trim(subject)), lower(trim(attribute)) FROM facts WHERE id = ?", (answer,)).fetchone()  # fmt: skip
        for (fact,) in world.db.execute(
            "SELECT id FROM facts WHERE lower(trim(subject)) = ? AND lower(trim(attribute)) = ? ORDER BY id",
            (subject, attribute),
        ):
            if fact in answers or fact in out or not any(map(reads, places(world, fact))):
                continue
            changes = [r[0] for r in world.db.execute("SELECT src_fact FROM fact_relations WHERE kind = 'supersedes' AND dst_fact = ?", (fact,))]  # fmt: skip
            if not any(sees(c) for change in changes for c in places(world, change)):
                out.append(fact)
    return out


def hidden(world, ids: list[str], actor: str, gold=None) -> bool | None:
    """Whether a task's answer is stated where it is harder to see than its near-misses: an anchor of an answer fact
    in a thread reply or a direct conversation, or on a later day than every near-miss is first stated. None for a
    task with no answer fact stated."""
    answers = answer_facts(world, ids, actor, gold)
    misses = near_misses(world, answers, actor)
    marks = ", ".join("?" * len(answers))
    anchors = world.db.execute(
        f"""SELECT m.parent_id, c.type, f.day FROM evidence e JOIN messages m ON m.id = e.message_id
        JOIN channels c ON c.id = m.channel_id JOIN facts f ON f.id = e.fact_id
        WHERE e.role = 'anchor' AND e.fact_id IN ({marks})""",
        answers,
    ).fetchall()
    if not anchors:
        return None
    last = max((world.db.execute("SELECT day FROM facts WHERE id = ?", (f,)).fetchone()[0] for f in misses), default=None)  # fmt: skip
    return any(a["parent_id"] is not None or a["type"] in ("im", "mpim") or (last is not None and a["day"] > last) for a in anchors)  # fmt: skip


def measures(world, task_id: str, rows: list[dict], tables) -> dict:
    """What makes a task hard, measured for its actor: how many read_channel pages deep its evidence sits and how many
    messages it sits under, the tables its gold query reads, its evidence's best rank when the actor searches the
    question's own words, how many of its evidence's channels and identifiers the question names, how much of its
    evidence the actor cannot read (`unseen`: a refusal's truth out of sight), and what its facts give it
    (`fact_measures`)."""
    task = world.db.execute(
        "SELECT actor_id, question, answer_type FROM tasks WHERE id = ?", (task_id,)
    ).fetchone()
    answers, refusal = [r.get("answer") for r in rows], task["answer_type"] == "refusal"
    evidence = [r["message_id"] for r in rows if r.get("message_id") is not None]
    # Its own evidence: a decoy it lists among its facts is a near-miss to search past, not evidence to find.
    evidence += [
        r[0]
        for r in world.db.execute(
            """SELECT e.message_id FROM task_facts tf JOIN evidence e ON e.fact_id = tf.fact_id
            JOIN facts f ON f.id = tf.fact_id WHERE tf.task_id = ? AND e.role = 'anchor' AND f.is_decoy = 0""",
            (task_id,),
        )
    ]
    reader = World(world.path, actor=task["actor_id"])
    # A refusal may rest on the truth out of its actor's sight: what the actor cannot read is counted, not measured.
    evidence = list(dict.fromkeys(evidence))
    seen = [m for m in evidence if reader.db.execute("SELECT 1 FROM messages WHERE id = ?", (m,)).fetchone()]
    pages = [
        reader.db.execute(
            """SELECT COUNT(*) / 50 + 1 FROM messages m, messages e
            WHERE e.id = ? AND m.channel_id = e.channel_id AND m.parent_id IS NULL
            AND m.ts_us > (SELECT ts_us FROM messages WHERE id = COALESCE(e.parent_id, e.id))""",
            (message_id,),
        ).fetchone()[0]
        for message_id in seen
    ]
    under = [
        reader.db.execute(
            """SELECT (SELECT COUNT(*) FROM messages m WHERE m.channel_id = e.channel_id AND m.parent_id IS NULL
            AND m.ts_us > (SELECT ts_us FROM messages WHERE id = COALESCE(e.parent_id, e.id)))
            + (SELECT COUNT(*) FROM messages r WHERE r.parent_id = e.parent_id AND r.ts_us < e.ts_us)
            FROM messages e WHERE e.id = ?""",
            (message_id,),
        ).fetchone()[0]
        for message_id in seen
    ]
    facts = [r[0] for r in world.db.execute("SELECT fact_id FROM task_facts WHERE task_id = ?", (task_id,))]
    (busiest,) = reader.db.execute("SELECT MAX(n) FROM (SELECT COUNT(*) AS n FROM messages WHERE parent_id IS NULL GROUP BY channel_id)").fetchone()  # fmt: skip
    return (
        {
            "evidence_pages": max(pages, default=None),
            "depth": max(under, default=0),
            "depth_share": round(max(under, default=0) / busiest, 3) if busiest else 0.0,
            "tables": sorted(tables),
            "bm25_rank": reader.rank(task["question"], seen),
            "named": named(world, task["question"], evidence),
            "unseen": len(evidence) - len(seen),
        }
        | fact_measures(world, facts, task["actor_id"], answers, refusal)
        | ({"hidden": hidden(world, facts, task["actor_id"], answers)} if facts and not refusal else {})
    )


def stated_values(world, facts: list[str], actor: str) -> dict[str, list[int]]:
    """The values its actor can read stated on the subjects and attributes of a task's facts, each with the messages
    that state it, oldest first."""
    reader, out = World(world.path, actor=actor), {}
    seen = {r[0] for r in reader.db.execute("SELECT id FROM messages")}
    reader.close()
    marks = ", ".join("?" * len(facts))
    for value, message, _ in world.db.execute(
        f"""SELECT lower(trim(f.value)), m.id, m.ts_us FROM facts f JOIN evidence e ON e.fact_id = f.id
        JOIN messages m ON m.id = e.message_id WHERE (lower(trim(f.subject)), lower(trim(f.attribute))) IN
        (SELECT lower(trim(subject)), lower(trim(attribute)) FROM facts WHERE id IN ({marks})) ORDER BY m.ts_us""",
        facts,
    ):
        if message in seen:
            out.setdefault(value, []).append(message)
    return out


def panel(world, task_id: str, rows: list[dict]) -> dict:
    """Lazy solvers, played by code over the ledger as a task's actor can read it (StateMemBench's heuristics,
    2608.19652): the first value stated on its answer's subject and attribute, the latest, the most stated, and the
    first a search for the question's own words reaches. `traps` are the ones that answer wrong. A task the search
    for its own words answers right is a shortcut: in S2 and S3, GLM never missed one of its 20 such tasks, and
    missed 8 of the 17 where that search found a wrong value first."""
    task = world.db.execute(
        "SELECT actor_id, question, answer_type FROM tasks WHERE id = ?", (task_id,)
    ).fetchone()
    facts = [r[0] for r in world.db.execute("SELECT fact_id FROM task_facts WHERE task_id = ?", (task_id,))]
    stated = stated_values(world, facts, task["actor_id"]) if facts else {}
    if not stated:
        return {"guesses": {}, "traps": [], "shortcut": False}
    when = {v: world.db.execute(f"SELECT MIN(ts_us), MAX(ts_us) FROM messages WHERE id IN ({', '.join('?' * len(ms))})", ms).fetchone() for v, ms in stated.items()}  # fmt: skip
    reader = World(world.path, actor=task["actor_id"])
    ranks = {v: reader.rank(task["question"], ms) for v, ms in stated.items()}
    reader.close()
    found = [v for v in stated if ranks[v] is not None]
    guesses = {
        "first": min(stated, key=lambda v: when[v][0]),
        "latest": max(stated, key=lambda v: when[v][1]),
        "most": max(stated, key=lambda v: len(stated[v])),
        "top": min(found, key=lambda v: ranks[v]) if found else None,
    }
    # The value the task rests on: its answer facts' (a hybrid task answers about the message that states it).
    marks = ", ".join("?" * len(facts))
    answered = answer_facts(world, facts, task["actor_id"], [r.get("answer") for r in rows])
    gold = {v.strip().casefold() for f, v in world.db.execute(f"SELECT id, value FROM facts WHERE id IN ({marks})", facts) if f in answered}  # fmt: skip
    if task["answer_type"] == "refusal":  # nothing is the answer: any value given is wrong
        return {
            "guesses": guesses,
            "traps": [k for k, v in guesses.items() if v is not None],
            "shortcut": False,
        }
    traps = [k for k, v in guesses.items() if v is not None and v not in gold]
    return {"guesses": guesses, "traps": traps, "shortcut": task["answer_type"] != "status" and guesses["top"] in gold}  # fmt: skip


def features(world, task_id: str, rows: list[dict]) -> dict:
    """What a task's structure is, measured by code: how many values its asker can see on its answer's subject and
    attribute (`changes`), where its answer's statement sits (`last_link`: a thread reply, a direct conversation, a
    private channel, or a channel, and later than the first value), how many rows it answers, whether another person
    asking gets another answer (`perspective`), and whether its answer is in no message as written (`derived`)."""
    task = world.db.execute(
        "SELECT actor_id, gold_sql, answer_type FROM tasks WHERE id = ?", (task_id,)
    ).fetchone()
    facts = [r[0] for r in world.db.execute("SELECT fact_id FROM task_facts WHERE task_id = ?", (task_id,))]
    answers = [str(r.get("answer")) for r in rows]
    stated = stated_values(world, facts, task["actor_id"]) if facts else {}
    link = None
    golden = [stated[a.strip().casefold()] for a in answers if a.strip().casefold() in stated]
    if golden:
        message = world.db.execute("SELECT m.parent_id, c.type, m.ts_us FROM messages m JOIN channels c ON c.id = m.channel_id WHERE m.id = ?", (golden[0][0],)).fetchone()  # fmt: skip
        link = "thread" if message["parent_id"] else {"im": "dm", "mpim": "dm", "private": "private"}.get(message["type"], "channel")  # fmt: skip
        first = min(world.db.execute(f"SELECT MIN(ts_us) FROM messages WHERE id IN ({', '.join('?' * len(ms))})", ms).fetchone()[0] for ms in stated.values())  # fmt: skip
        link += "·later" if message["ts_us"] > first and len(stated) > 1 else ""
    others = []
    for (user,) in world.db.execute("SELECT id FROM users WHERE id != ? ORDER BY id", (task["actor_id"],)):
        try:
            others.append(sorted(str(r.get("answer")) for r in world.gold(user, task["gold_sql"], max_rows=50)["rows"]))  # fmt: skip
        except ValueError:
            continue  # the query returns what that person cannot read
    readable = World(world.path, actor=task["actor_id"])
    verbatim = all(readable.db.execute("SELECT 1 FROM messages WHERE instr(lower(text), lower(?)) LIMIT 1", (a,)).fetchone() for a in answers)  # fmt: skip
    named = all(world.db.execute("SELECT 1 FROM users WHERE real_name = ?", (a,)).fetchone() for a in answers)
    readable.close()
    return {
        "changes": len(stated),
        "last_link": link,
        "rows": len(rows),
        "perspective": any(o != sorted(answers) for o in others),
        "derived": bool(answers) and not verbatim and not named,
        "status": task["answer_type"] == "status",
    }


def failure_cases(world, task_id: str, rows: list[dict]) -> tuple[str, list[tuple[str, str]]]:
    """A task's gold as a plain answer, and the wrong answers its grade must reject, built by code: each near-miss
    value its actor can read on its answer's subject and attribute, a hedge between the gold and the first of them, a
    set with a row left out, declining to answer, a bare opener (judges accept one 60-90% of the time, 2507.08794),
    and a status given as final. A grade that accepts one, or rejects the gold, is ambiguous as asked
    (Self-Challenging's failure cases, 2506.01716)."""
    task = world.db.execute(
        "SELECT actor_id, answer_type, near_sql FROM tasks WHERE id = ?", (task_id,)
    ).fetchone()
    kind, answers = task["answer_type"], [str(r.get("answer")) for r in rows]
    facts = [r[0] for r in world.db.execute("SELECT fact_id FROM task_facts WHERE task_id = ?", (task_id,))]
    misses = []
    if facts and kind != "refusal":
        golden = {a.strip().casefold() for a in answers}
        for fact in answer_facts(world, facts, task["actor_id"], answers):
            subject, attribute = world.db.execute("SELECT lower(trim(subject)), lower(trim(attribute)) FROM facts WHERE id = ?", (fact,)).fetchone()  # fmt: skip
            for other, value in world.db.execute("SELECT id, value FROM facts WHERE lower(trim(subject)) = ? AND lower(trim(attribute)) = ? ORDER BY day, id", (subject, attribute)):  # fmt: skip
                sight = any(readable(world, task["actor_id"], c) for c in places(world, other))
                if sight and value.strip().casefold() not in golden and value not in misses:
                    misses.append(value)
    elif facts:  # a refusal: any value its actor can see, given as the answer
        marks = ", ".join("?" * len(facts))
        misses = [v for f, v in world.db.execute(f"SELECT id, value FROM facts WHERE id IN ({marks}) ORDER BY day, id", facts) if any(readable(world, task["actor_id"], c) for c in places(world, f))]  # fmt: skip
    gold = ", ".join(answers)
    cases = [(f"the near-miss {value!r}", f"{value}.") for value in misses[:3]]
    for near in json.loads(
        task["near_sql"]
    ):  # a hasty reader's answer, from the gold with a condition relaxed
        relaxed = ", ".join(
            str(r.get("answer")) for r in world.gold(task["actor_id"], near, max_rows=50)["rows"]
        )
        cases.append((f"the near-miss query's answer {relaxed!r}", f"{relaxed or 'Nobody'}."))
    cases.append(("a bare opener", "Thought process:"))
    if kind == "refusal":
        return "I can't answer that from anything you can see: it isn't there.", cases
    if misses:
        cases.append(("a hedge", f"Either {gold} or {misses[0]}; I can't tell which."))
    if kind == "set" and len(answers) > 1:
        cases.append(("a set with a row left out", ", ".join(answers[:-1]) + "."))
    if kind == "status":
        cases.append(("the status given as final", f"{gold}. That is final and confirmed."))
        return f"{gold}, though it is not settled: nothing you can see confirms it.", cases
    cases.append(("declining to answer", "I can't find that in anything you can see."))
    return f"{gold}.", cases


def band_move(probe: dict) -> str | None:
    """What a probed task needs: a fix the judge asks for, or to be harder or easier to land in its level's band of
    right-answer rates. A band that starts at 0 also needs the witness to answer it: one even it fails is too hard."""
    (lo, hi), right, witness = probe["band"], probe["right_rate"], probe.get("witness_right")
    if not probe["approved"]:
        return "fix"
    if right > hi:
        return "harder"
    if right < lo or (lo == 0 and witness == 0):
        return "easier"
    return None


def gold_source(task: Task, settings) -> str:
    """Where a task's answer comes from: its category's gold, but a status answer is a value of the ledger as its
    actor can see it stated."""
    if task.answer_type == "status":
        return "ledger"
    return settings.taxonomy[task.category].gold if task.category in settings.taxonomy else "sql"


def unsettled(world, task: Task, rows: list[dict]) -> None:
    """A status answer's ground: the value its gold gives is a fact its actor can read, that no fact in their sight
    supersedes, and that a fact it rests on, out of their sight, does. S3's care team could read only the provisional
    "15%... Grant can correct me", while a DM had settled 10%: the useful answer gives 15% as not settled."""
    values = {str(r.get("answer")).strip().casefold() for r in rows}
    seen = lambda f: any(readable(world, task.actor_id, c) for c in places(world, f))  # noqa: E731
    marks = ", ".join("?" * len(task.facts))
    given = [f for f, v in world.db.execute(f"SELECT id, value FROM facts WHERE id IN ({marks})", task.facts) if v.strip().casefold() in values and seen(f)]  # fmt: skip
    if not given:
        raise ValueError(
            f"{task.id}: a status answer's gold gives the value of one of its facts its actor can read"
        )
    # The values that replace it: later facts on its subject and attribute down its supersedes chain.
    later = [r[0] for r in world.db.execute(
        f"""WITH RECURSIVE chain(id) AS (SELECT src_fact FROM fact_relations WHERE kind = 'supersedes'
        AND dst_fact IN ({", ".join("?" * len(given))}) UNION SELECT r.src_fact FROM chain c JOIN fact_relations r
        ON r.dst_fact = c.id AND r.kind = 'supersedes')
        SELECT f.id FROM chain JOIN facts f ON f.id = chain.id JOIN facts g ON g.id = ?
        WHERE lower(trim(f.subject)) = lower(trim(g.subject)) AND lower(trim(f.attribute)) = lower(trim(g.attribute))""",
        [*given, given[0]],
    )]  # fmt: skip
    if in_sight := [f for f in later if seen(f)]:
        raise ValueError(f"{task.id}: its actor can read {in_sight}, which supersede the value it gives: that is its answer, not a status")  # fmt: skip
    if not [f for f in task.facts if f in later]:
        raise ValueError(f"{task.id}: a status answer rests on a fact, out of its actor's sight, that supersedes the value it gives")  # fmt: skip


def task_row(task: Task, settings) -> dict:
    return task.model_dump(
        include={"id", "category", "level", "actor_id", "question", "answer_type", "gold_sql"}
    ) | {"gold_source": gold_source(task, settings), "near_sql": json.dumps(task.near_sql)}


def readable(world, actor: str, channel: str) -> bool:
    return bool(
        world.db.execute(
            """SELECT 1 FROM channels c WHERE c.id = ? AND (c.type = 'public' OR EXISTS (SELECT 1 FROM members m
            WHERE m.channel_id = c.id AND m.user_id = ? AND m.left_us IS NULL))""",
            (channel, actor),
        ).fetchone()
    )


def text_errors(world, lines, conveyable: set[str], moments: set[str]) -> list[str]:
    """Each line's text rules: a reply points at an earlier line, conveys names only facts the conversation may
    state, a time or a date appears only as {at:id} of a moment, no line names a fact or event id, and mentions name
    people."""
    ids = [r[0] for r in world.db.execute("SELECT id FROM facts UNION SELECT id FROM events")]
    people = {r[0] for r in world.db.execute("SELECT id FROM users")}
    errors = []
    for i, line in enumerate(lines):
        if line.reply_to is not None and line.reply_to >= i:
            errors.append(f"line {i}: reply_to names an earlier line of this conversation")
        if extra := set(line.conveys) - conveyable:
            errors.append(
                f"line {i}: conveys lists only facts this conversation may state; not {sorted(extra)}"
            )
        if unknown := set(PLACEHOLDER.findall(line.text)) - moments:
            errors.append(f"line {i}: {{at:...}} names a moment; not {sorted(unknown)}")
        if literal := TIME_LITERAL.findall(PLACEHOLDER.sub(" ", line.text)):
            errors.append(f"line {i}: write times and dates only as {{at:id}}; found {literal}")
        if named := [
            f
            for f in ids
            if re.search(rf"(?<![\w{{:]){re.escape(f)}(?![\w}}])", PLACEHOLDER.sub(" ", line.text))
        ]:
            errors.append(f"line {i}: no line names a fact or event id; found {named}")
        if strangers := set(MENTION.findall(line.text)) - people:
            errors.append(f"line {i}: <@...> names a person's user_id; not {sorted(strangers)}")
    return errors


def insert_lines(
    world, scene: dict, lines, stamps, moments, first, rng, gaps, thread=None, latest=None
) -> list[int]:
    """Insert one conversation at code's timestamps: its scenes row, each line's text with {at:id} rendered on its
    author's clock, replies in their thread (or every line in `thread`), mentions, reactions after their message
    (and before `latest`), and evidence: a fact in `first` is anchored by its author's first line conveying it."""
    facts = {r["id"]: dict(r) for r in world.db.execute("SELECT * FROM facts")}
    zones = {r[0]: r[1] for r in world.db.execute("SELECT id, tz FROM users")}
    world.insert("scenes", [scene])
    ids = [None] * len(lines)
    for i, (line, ts) in enumerate(zip(lines, stamps)):
        text = PLACEHOLDER.sub(lambda m: render(moments[m[1]], zones[line.author_id], ts), line.text)
        root = i
        while lines[root].reply_to is not None:
            root = lines[root].reply_to
        parent = thread if thread is not None else ids[root] if root != i else None
        (ids[i],) = world.insert(
            "messages",
            [
                dict(
                    channel_id=scene["channel_id"],
                    ts_us=ts,
                    user_id=line.author_id,
                    text=text,
                    parent_id=parent,
                )
            ],
        )
        world.insert("scene_messages", [dict(message_id=ids[i], scene_id=scene["id"])])
        world.insert(
            "message_mentions",
            [dict(message_id=ids[i], user_id=u) for u in dict.fromkeys(MENTION.findall(text))],
        )
        reacted = [(r, ts + 1 + gaps.sample(rng, "")) for r in line.reactions]
        world.insert(
            "reactions",
            [
                dict(message_id=ids[i], user_id=r.user_id, emoji=r.emoji)
                | dict(created_us=min(at, max(ts + 1, latest - 1)) if latest else at)
                for r, at in reacted
            ],
        )
        for fact_id in dict.fromkeys(line.conveys):
            fact = facts[fact_id]
            token = fact["anchor"] or (
                render(fact["moment_us"], zones[line.author_id], ts)
                if fact["moment_us"] is not None
                else None
            )
            role = "anchor" if fact_id in first and line.author_id == fact["author_id"] else "supporting"
            if role == "anchor":
                first.discard(fact_id)
            world.insert(
                "evidence", [dict(fact_id=fact_id, message_id=ids[i], role=role, anchor_token=token)]
            )
    return ids


EMOJI = re.compile(r":[a-z0-9_+\-]+:|[☀-➿\U0001F300-\U0001FAFF]")


def activity(world, start_us: int | None = None, end_us: int | None = None) -> dict:
    """The workspace's shape as `[activity]` measures it, over all messages or those in [start_us, end_us): how many,
    and the shares that are thread replies, carry a reaction, sit in DMs, mention someone or carry an emoji."""
    span = (start_us if start_us is not None else -1 << 62, end_us if end_us is not None else 1 << 62)
    rows = world.db.execute(
        """SELECT m.id, m.parent_id IS NOT NULL AS reply, c.type IN ('im', 'mpim') AS dm, m.text,
        EXISTS (SELECT 1 FROM reactions r WHERE r.message_id = m.id) AS reacted,
        EXISTS (SELECT 1 FROM message_mentions x WHERE x.message_id = m.id) AS mentions
        FROM messages m JOIN channels c ON c.id = m.channel_id WHERE m.ts_us >= ? AND m.ts_us < ?""",
        span,
    ).fetchall()
    share = lambda values: round(sum(values) / len(rows), 3) if rows else 0.0  # noqa: E731
    return {
        "messages": len(rows),
        "reply_share": share([r["reply"] for r in rows]),
        "reaction_rate": share([r["reacted"] for r in rows]),
        "dm_share": share([r["dm"] for r in rows]),
        "mention_rate": share([r["mentions"] for r in rows]),
        "emoji_rate": share([bool(EMOJI.search(r["text"])) for r in rows]),
    }


def style(world) -> dict:
    """Surface statistics of voice and timing, overall and per author; the world judge compares them with people's
    typing."""
    rows = world.db.execute(
        "SELECT m.user_id, m.text, m.ts_us, u.tz FROM messages m JOIN users u ON u.id = m.user_id WHERE m.is_deleted = 0"
    ).fetchall()

    def measure(items):
        texts = [r["text"].strip() for r in items]
        local = [datetime.fromtimestamp(r["ts_us"] / 1e6, ZoneInfo(r["tz"])) for r in items]
        share = lambda values: round(sum(values) / len(values), 2)  # noqa: E731
        return {
            "messages": len(texts),
            "median_words": statistics.median(len(t.split()) for t in texts),
            "short": share([len(t.split()) <= SHORT for t in texts]),
            "long": share([len(t.split()) > LONG for t in texts]),
            "question": share(["?" in t for t in texts]),
            "lowercase_start": share([t[:1].islower() for t in texts]),
            "off_hours": share([t.hour < 7 or t.hour >= 20 or t.weekday() >= 5 for t in local]),
        }

    if not rows:
        return {}
    by_author = defaultdict(list)
    for r in rows:
        by_author[r["user_id"]].append(r)
    return measure(rows) | {"authors": {a: measure(items) for a, items in sorted(by_author.items())}}


# ---------------------------------------------------------------------- reviews


class Issue(StrictModel):
    """A defect a judge reports, on the ledger, a task or the workspace (messages); the author repairs it."""

    artifact: Literal["ledger", "tasks", "workspace"]
    task_ids: Ids = Field(default_factory=list)
    fact_ids: Ids = Field(default_factory=list)
    message_ids: list[int] = Field(default_factory=list)  # the messages to change
    evidence_message_ids: list[int] = Field(
        default_factory=list
    )  # the messages that show the defect; they stay
    defect: NonEmptyText
    requested_change: NonEmptyText
    blocking: bool = True


class TaskReview(StrictModel):
    task_id: SafeId
    valid: bool
    reason: NonEmptyText
    level_fit: int | None = Field(default=None, ge=0, le=4)


# A task review scores one task; a world review scores the workspace text.
PHASE_CRITERIA = {
    "task": ("question_fit", "discoverability", "shortcut_free"),
    "world": ("scenario_alignment", "world_coherence", "professional_realism"),
}
QUALITY_CRITERIA = tuple(name for names in PHASE_CRITERIA.values() for name in names)


class Verdict(StrictModel):
    approved: bool
    tasks: list[TaskReview]
    issues: list[Issue]
    criteria: dict[str, float] = Field(default_factory=dict)
    summary: NonEmptyText

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if set(self.criteria) - set(QUALITY_CRITERIA):
            raise ValueError("unknown quality criterion")
        if len({t.task_id for t in self.tasks}) != len(self.tasks):
            raise ValueError("duplicate task review")
        if any(not 0 <= value <= 1 for value in self.criteria.values()):
            raise ValueError("criteria must be finite scores from zero to one")
        if self.approved and (any(i.blocking for i in self.issues) or any(not t.valid for t in self.tasks)):
            raise ValueError("approval contradicts blocking defects")
        if not self.approved and not self.issues:
            raise ValueError("rejection must include actionable issues")
        return self


def verdict_schema(phase: str, task_ids: list[str]) -> dict:
    """The contract `validate_verdict` enforces: exactly one review per requested task, and the phase's criteria."""
    schema = Verdict.model_json_schema()
    schema["properties"]["tasks"] |= {"minItems": len(task_ids), "maxItems": len(task_ids)}
    if task_ids:
        schema["$defs"]["TaskReview"]["properties"]["task_id"]["enum"] = task_ids
    names = PHASE_CRITERIA.get(phase, ())
    schema["properties"]["criteria"] = {
        "type": "object",
        "properties": {name: {"type": "number", "minimum": 0, "maximum": 1} for name in names},
        "additionalProperties": False,
    }
    if phase in PHASE_CRITERIA:
        schema["properties"]["criteria"]["required"] = list(names)
        schema["required"].append("criteria")
    if phase == "task":
        schema["$defs"]["TaskReview"]["required"].append("level_fit")
    return schema


def quality(verdict: Verdict) -> float:
    if not verdict.criteria:
        raise ValueError("a task or world review must score its criteria")
    return sum(verdict.criteria.values()) / len(verdict.criteria)


def validate_verdict(verdict: Verdict, payload: dict) -> None:
    if payload["phase"] in PHASE_CRITERIA and set(verdict.criteria) != set(PHASE_CRITERIA[payload["phase"]]):
        raise ValueError(
            f"a {payload['phase']} review scores exactly {list(PHASE_CRITERIA[payload['phase']])}"
        )
    if {r.task_id for r in verdict.tasks} != {t["id"] for t in payload["tasks"]}:
        raise ValueError("judge must review every requested task exactly once")
    if payload["phase"] == "task" and any(r.level_fit is None for r in verdict.tasks):
        raise ValueError("a task review scores each task's level_fit from 0 to 4")
    if payload["phase"] == "task" and any(i.blocking and not i.task_ids for i in verdict.issues):
        raise ValueError(
            "a blocking issue in a task review names the tasks it blocks (task_ids); a defect that blocks no task "
            "is not blocking"
        )


def deciding(verdict: Verdict, acceptance) -> list[Issue]:
    """The issues that decide acceptance and a rejection's route: blocking ones, and minor ones when they block."""
    return [i for i in verdict.issues if i.blocking or acceptance.minor_issues_block]


def accepted(verdict: Verdict, acceptance) -> bool:
    """Blocking issues decide (`config.Acceptance`); the judge's scores are reported, not thresholds."""
    return all(t.valid for t in verdict.tasks) and not deciding(verdict, acceptance)


def accepted_task(verdict: Verdict, acceptance, task_id: str) -> bool:
    """One task of a batched review passes on its own review and the issues that name it: an issue naming no task is
    the world's, and the final review decides it."""
    return accepted(
        verdict.model_copy(
            update={
                "tasks": [r for r in verdict.tasks if r.task_id == task_id],
                "issues": [i for i in verdict.issues if task_id in i.task_ids],
            }
        ),
        acceptance,
    )
