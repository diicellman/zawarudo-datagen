"""The company clock: the calendar, the present and the parts of the day, frozen chapters, and natural times as
each person writes them."""

import json
import random
import re
from datetime import date, datetime, timedelta
from typing import Annotated, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import AfterValidator, Field
from worldgen_slack.release import StrictModel
from worldgen_slack.world import digest


def zone(value: str) -> str:
    try:
        ZoneInfo(value)
    except (ZoneInfoNotFoundError, ValueError) as error:
        raise ValueError(f"unknown IANA timezone {value!r}") from error
    return value


Clock = Annotated[str, Field(pattern=r"^([01]\d|2[0-3]):[0-5]\d$")]


Zone = Annotated[str, AfterValidator(zone)]


Part = Literal["early", "morning", "afternoon", "evening", "night"]


# A part of the day on the company clock: [start hour, end hour).
PARTS = {"early": (6, 9), "morning": (9, 12), "afternoon": (12, 17), "evening": (17, 21), "night": (21, 24)}


class Moment(StrictModel):
    """A moment the story is about: a calendar day, a time on a clock, and that clock's IANA zone."""

    day: int = Field(ge=1)
    time: Clock
    zone: Zone


def us(moment: datetime) -> int:
    return int(moment.timestamp() * 1_000_000)


def calendar_days(start: date, count: int, place: str, first: int = 1) -> list[dict]:
    """`count` calendar days from `start`, numbered from `first`; days are company-clock midnights."""
    tz = ZoneInfo(place)
    days = [start + timedelta(days=n) for n in range(count + 1)]
    return [
        {
            "day": first + n,
            "date": d.isoformat(),
            "start_us": us(datetime(d.year, d.month, d.day, tzinfo=tz)),
            "end_us": us(datetime(nd.year, nd.month, nd.day, tzinfo=tz)),
        }
        for n, (d, nd) in enumerate(zip(days, days[1:]))
    ]


def calendar(seed: int, settings, place: str) -> list[dict]:
    """Day 1 is a Monday of the configured year picked by the run seed; days are company-clock midnights."""
    first = date(settings.year, 1, 1)
    mondays = [first + timedelta(days=d) for d in range(366) if (first + timedelta(days=d)).weekday() == 0]
    return calendar_days(random.Random(seed).choice(mondays), settings.days, place)


def extend_calendar(world, count: int) -> int:
    """`count` more days after the calendar's last, the dates running on (a chapter's days); returns the first."""
    last, on = world.db.execute("SELECT day, date FROM calendar ORDER BY day DESC LIMIT 1").fetchone()
    world.insert(
        "calendar",
        calendar_days(date.fromisoformat(on) + timedelta(days=1), count, world_meta(world, "zone"), last + 1),
    )
    return last + 1


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


def meta(world, key: str, default=None):
    """A world_meta value that may be absent: a world with no chapter has no lineage."""
    row = world.db.execute("SELECT value FROM world_meta WHERE key = ?", (key,)).fetchone()
    return row[0] if row else default


def chapter_start(world) -> int:
    """The first day of the chapter being written: 1 for a world written in one run."""
    return int(meta(world, "chapter_start_day", 1))


def frozen_before(world) -> int:
    """The moment before which the world is an earlier chapter's, written and frozen: 0 when there is none."""
    first = chapter_start(world)
    if first == 1:
        return 0
    return world.db.execute("SELECT start_us FROM calendar WHERE day = ?", (first,)).fetchone()[0]


# What an earlier chapter wrote, by table: its days, people and channels, its messages with what hangs on them, its
# conversations, its ledger (facts, their relations, events) and its promises. :t is the chapter's first moment.
FROZEN = {
    "calendar": "SELECT * FROM calendar WHERE start_us < :t",
    "users": "SELECT * FROM users",
    "channels": "SELECT * FROM channels",
    "members": "SELECT * FROM members",
    "messages": "SELECT * FROM messages WHERE ts_us < :t",
    "message_mentions": "SELECT x.* FROM message_mentions x JOIN messages m ON m.id = x.message_id WHERE m.ts_us < :t",
    "reactions": "SELECT x.* FROM reactions x JOIN messages m ON m.id = x.message_id WHERE m.ts_us < :t",
    "evidence": "SELECT x.* FROM evidence x JOIN messages m ON m.id = x.message_id WHERE m.ts_us < :t",
    "scenes": "SELECT * FROM scenes WHERE slot_start_us < :t",
    "scene_messages": "SELECT x.* FROM scene_messages x JOIN messages m ON m.id = x.message_id WHERE m.ts_us < :t",
    "facts": "SELECT f.* FROM facts f JOIN calendar c ON c.day = f.day WHERE c.start_us < :t",
    "fact_relations": "SELECT r.* FROM fact_relations r JOIN facts f ON f.id = r.src_fact JOIN calendar c ON c.day = f.day WHERE c.start_us < :t",
    "events": "SELECT * FROM events WHERE moment_us < :t",
    # a promise as it was made: one that slipped may be kept late or dropped in a later chapter
    "commitments": "SELECT x.id, x.owner_id, x.text, x.message_id, x.due_us, x.slips FROM commitments x "
    "JOIN messages m ON m.id = x.message_id WHERE m.ts_us < :t",
}


def frozen_digest(world) -> str:
    """A digest of everything before the chapter's first day: a chapter changes none of it."""
    t = frozen_before(world)
    return digest(
        {
            name: sorted((list(r) for r in world.db.execute(sql, {"t": t})), key=repr)
            for name, sql in FROZEN.items()
        }
    )


# How a person writes a time, drawn once per person: "5pm", "5 PM" or "17:00". A world written before 2026-10-14 has
# no `time_style` and keeps its clock style ("17:00 EDT", "Tue Aug 25 17:00 EDT"), chapters included.
HABITS = ("12h", "12h spaced", "24h")


def habit(person: str) -> str:
    return HABITS[int(digest(["habit", person])[:8], 16) % len(HABITS)]


def writing(world) -> tuple[dict[str, str], dict[str, str], str | None]:
    """How each person writes a time: their clock, their habit (none in a world kept in the clock style), and the
    company's clock, whose people leave the zone out."""
    zones = {r[0]: r[1] for r in world.db.execute("SELECT id, tz FROM users")}
    styled = world.db.execute("SELECT value FROM world_meta WHERE key = 'time_style'").fetchone()
    home = world.db.execute("SELECT value FROM world_meta WHERE key = 'zone'").fetchone()
    habits = {u: habit(u) for u in zones} if styled and styled[0] == "habit" else {}
    return zones, habits, home[0] if home else None


def render(moment_us: int, place: str, said_us: int, way: str | None = None, home: str | None = None) -> str:
    """A moment as its writer writes it on their own clock. In a person's habit (`way`): the time ("5pm", "5:30 PM",
    "17:00", "12pm"), the day only when it is not the day of writing ("tomorrow", "yesterday", a weekday within the
    next six days, else the date), and the zone only when theirs is not the company's (`home`). Every form reads back
    to the exact moment given the writer's clock and the day they wrote it (`reread`). With no habit, the clock style:
    the day is named when it is not the day of writing."""
    tz = ZoneInfo(place)
    when, said = datetime.fromtimestamp(moment_us / 1e6, tz), datetime.fromtimestamp(said_us / 1e6, tz)
    if way is None:
        return (
            f"{when:%H:%M %Z}" if when.date() == said.date() else f"{when:%a %b} {when.day} {when:%H:%M %Z}"
        )
    if way == "24h":
        hour = f"{when:%H:%M}"
    else:  # "12pm", not "noon": a word people also write freely, which code could not tell from its own
        h, m = when.hour % 12 or 12, f":{when.minute:02d}" if when.minute else ""
        hour = f"{h}{m}{' ' if way == '12h spaced' else ''}{('pm' if when.hour >= 12 else 'am') if way == '12h' else ('PM' if when.hour >= 12 else 'AM')}"
    gap = (when.date() - said.date()).days
    day = (
        ""
        if gap == 0
        else "tomorrow "
        if gap == 1
        else "yesterday "
        if gap == -1
        else f"{when:%a} "
        if 1 < gap < 7
        else f"{when:%b} {when.day} "
    )
    return day + hour + ("" if home is None or place == home else f" {when:%Z}")


READ_BACK = re.compile(
    r"^(?:(?P<rel>tomorrow|yesterday) |(?P<wd>Mon|Tue|Wed|Thu|Fri|Sat|Sun) |(?P<mon>[A-Z][a-z]{2}) (?P<md>\d{1,2}) )?"
    r"(?P<h>\d{1,2})(?::(?P<m>\d{2}))?(?: ?(?P<ap>am|pm|AM|PM))?(?: (?P<zone>[A-Z]{2,5}))?$"
)


def reread(text: str, place: str, said_us: int) -> int:
    """A rendered time read back: the moment it names on its writer's clock, from the day they wrote it."""
    tz, said = ZoneInfo(place), datetime.fromtimestamp(said_us / 1e6, ZoneInfo(place))
    found = READ_BACK.match(text)
    if not found:
        raise ValueError(f"not a rendered time: {text!r}")
    g = found.groupdict()
    hour, minute = int(g["h"]), int(g["m"] or 0)
    if g["ap"]:
        hour = hour % 12 + (12 if g["ap"].lower() == "pm" else 0)
    day = said.date()
    if g["rel"]:
        day += timedelta(days=1 if g["rel"] == "tomorrow" else -1)
    elif g["wd"]:
        day += timedelta(
            days=next(k for k in range(2, 7) if (said.date() + timedelta(days=k)).strftime("%a") == g["wd"])
        )
    elif g["mon"]:
        day = datetime.strptime(f"{g['mon']} {g['md']} {said.year}", "%b %d %Y").date()
    return us(datetime(day.year, day.month, day.day, hour, minute, tzinfo=tz))


DAY = 86_400_000_000


def start_clock(world, day: int = 1) -> None:
    """From now on the world only grows forward: its present moves to `day` (day 1, or a chapter's first) at the start
    of the early part."""
    world.db.execute(
        "UPDATE world_meta SET value = ? WHERE key = 'now_us'", (str(window(world, day, "early")[0]),)
    )
    world.db.execute("INSERT OR REPLACE INTO world_meta (key, value) VALUES ('chronological', '1')")


def present(world) -> int:
    return int(world_meta(world, "now_us"))


def today(world) -> int | None:
    """The calendar day of the present; None once the last day is closed."""
    row = world.db.execute(
        "SELECT day FROM calendar WHERE start_us <= ? AND ? < end_us", (present(world), present(world))
    ).fetchone()
    return row[0] if row else None


def part_of(world, moment_us: int) -> str:
    hour = datetime.fromtimestamp(moment_us / 1e6, ZoneInfo(world_meta(world, "zone"))).hour
    return next((p for p, (start, end) in PARTS.items() if start <= hour < end), "early")


def moments(world) -> dict[str, int]:
    """What {at:id} may name: every event, and every fact that carries a moment."""
    rows = world.db.execute(
        "SELECT id, moment_us FROM events UNION ALL SELECT id, moment_us FROM facts WHERE moment_us IS NOT NULL"
    )
    return {r[0]: r[1] for r in rows}


def bounds(world, day: int) -> tuple[int, int]:
    return tuple(world.db.execute("SELECT start_us, end_us FROM calendar WHERE day = ?", (day,)).fetchone())


def apportion(total: int, weights: list[float]) -> list[int]:
    """Whole numbers in proportion to `weights` that add up to `total` (largest remainders)."""
    exact = [total * w / sum(weights) if sum(weights) else 0.0 for w in weights]
    counts = [int(x) for x in exact]
    for i in sorted(range(len(exact)), key=lambda i: (counts[i] - exact[i], i))[: total - sum(counts)]:
        counts[i] += 1
    return counts


def spread(dates: list[str], first: int, messages: int, weekend: float, parts: dict, busy=()) -> list[dict]:
    """`messages` over the dates (day numbers from `first`), a weekend day weighing `weekend` of a workday unless it is
    `busy` (a day number with a planned event), and each day split by the parts' rhythm."""
    weights = [
        1.0 if day in busy or date.fromisoformat(d).weekday() < 5 else weekend
        for day, d in enumerate(dates, first)
    ]
    rhythm = [parts.get(p, 0.0) for p in PARTS]
    return [
        {"messages": n, "parts": dict(zip(PARTS, apportion(n, rhythm)))} for n in apportion(messages, weights)
    ]


def daily(dates: list[str], settings, busy=()) -> list[dict]:
    """Code's share of the world per calendar date and part: `[activity].messages` over the dates, a weekend day
    weighing `[author].weekend` of a workday unless it is `busy` (a day number with a planned event), and each day
    split by the parts' rhythm."""
    return spread(
        dates, 1, settings.activity.messages, settings.author.weekend, settings.activity.parts, busy
    )


def segments(world, settings) -> list[dict]:
    """The calendar's segments, each spreading its own messages over its own days: a world written in one run is one,
    by its settings, with the days its first plan put an event on; a chapter appends one of its own after its parent's
    (`world_meta.segments`, written when it starts), so a closed day's quota never changes."""
    if (stored := meta(world, "segments")) is not None:
        return json.loads(stored)
    first, last = world.db.execute("SELECT MIN(day), MAX(day) FROM calendar").fetchone()
    busy = meta(world, "busy_days")
    return [
        dict(
            first=first,
            last=last,
            messages=settings.activity.messages,
            busy=json.loads(busy) if busy else None,
            weekend=settings.author.weekend,
            parts=settings.activity.parts,
        )
    ]


def segment_of(world, settings, day: int) -> dict:
    return next(s for s in segments(world, settings) if s["first"] <= day <= s["last"])
