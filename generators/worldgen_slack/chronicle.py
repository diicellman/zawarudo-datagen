"""The world written in time order: one author plans the ledger and its events, posts each conversation at the
world's present, and moves the present forward. Code checks every step, owns every time, and never writes into the
past. Pure functions over `World`; the author's tools (`agents/world.py`) call them inside `World.trial()`."""

import json
import math
import re
import statistics
from datetime import date, datetime
from typing import Literal, Self
from zoneinfo import ZoneInfo

from pydantic import Field, model_validator
from worldgen_slack.release import NonEmptyText, SafeId, StrictModel
from worldgen_slack.world import digest

from .integrations import app_id, event_errors, queue_day, queue_events
from .contracts import (
    LONG,
    SHORT,
    MENTION,
    PARTS,
    PLACEHOLDER,
    TIME_LITERAL,
    Clock,
    Ids,
    Moment,
    Reaction,
    Storyline,
    Task,
    Zone,
    at,
    clock,
    board_measures,
    chapter_start,
    fact_measures,
    frozen_before,
    near_misses,
    answer_facts,
    on_board,
    slot_needs,
    insert_lines,
    meta,
    normalized,
    Slot,
    features,
    panel,
    record_task,
    INTENTS,
    PEOPLE,
    drain,
    linked,
    queue,
    render,
    writing,
    text_errors,
    unmet,
    window,
    world_meta,
)

# ---------------------------------------------------------------------- the documents the author writes


class Event(StrictModel):
    """A moment of the story that facts and messages refer to: a calendar day, a time, and the clock it is told on
    (the company's when unset)."""

    id: SafeId
    storyline: SafeId
    title: NonEmptyText
    day: int = Field(ge=1)
    time: Clock
    zone: Zone | None = None
    # The app that posts when it happens (a deploy finished, an alert fired), by its name: code posts its title then.
    post_by: str | None = None


class PlanFact(StrictModel):
    """What someone states in Slack, and where it is first stated: by its author, in its channel, on its day."""

    id: SafeId
    storyline: SafeId
    subject: NonEmptyText
    attribute: NonEmptyText
    value: NonEmptyText
    anchor: str | None = None
    channel_id: SafeId
    author_id: SafeId
    day: int = Field(ge=1)
    after: Ids = Field(default_factory=list)
    supersedes: SafeId | None = None
    event: SafeId | None = None
    kind: Literal["happened", "scheduled"] | None = None
    decoy: bool = False
    summary: NonEmptyText

    @model_validator(mode="after")
    def about_an_event(self) -> Self:
        if (self.event is None) != (self.kind is None):
            raise ValueError(
                f"{self.id}: a fact about an event names it and whether it happened or is scheduled"
            )
        return self


class BoardEntry(StrictModel):
    """The planned facts one ledger or hybrid task slot will rest on."""

    slot: SafeId
    facts: Ids = Field(min_length=1)


class Move(StrictModel):
    """A change in who is where, on a day of the calendar: someone joins or leaves a channel (Slack posts that they
    did), someone leaves the company (they leave every conversation), or a channel is archived (no one posts in it
    again). Code times it in the day's morning."""

    kind: Literal["join", "leave", "depart", "archive"]
    day: int = Field(ge=1)
    user_id: SafeId | None = None
    channel_id: SafeId | None = None

    @model_validator(mode="after")
    def named(self) -> Self:
        if (self.kind != "archive") != (self.user_id is not None) or (self.kind != "depart") != (
            self.channel_id is not None
        ):
            raise ValueError(
                "a join or a leave names a person and a channel; a departure a person; an archive a channel"
            )
        return self


class Artifact(StrictModel):
    """A ticket, pull request, doc or incident people link to ({link:id}): what they call it, its title, who owns it, and
    the day it is made. Code gives it its URL; a ticket app announces a new ticket."""

    id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9#._/-]{1,40}$")
    kind: Literal["ticket", "pull_request", "doc", "incident"]
    title: NonEmptyText
    owner_id: SafeId
    day: int = Field(ge=1)


class Plan(StrictModel):
    storylines: list[Storyline] = Field(min_length=1)
    events: list[Event] = Field(default_factory=list)
    facts: list[PlanFact] = Field(min_length=1)
    board: list[BoardEntry] = Field(default_factory=list)
    moves: list[Move] = Field(default_factory=list)
    artifacts: list[Artifact] = Field(default_factory=list)


HOSTS = {"ticket": "tickets", "pull_request": "git", "doc": "docs", "incident": "status"}


def artifact_url(world, artifact: Artifact) -> str:
    """Where it lives, on the company's own hosts."""
    company = world_meta(world, "company")
    domain = re.sub(r"[^a-z0-9]+", "", company.lower()) + ".example"
    path = {
        "ticket": f"browse/{artifact.id}",
        "pull_request": f"app/pull/{re.sub(r'[^0-9]', '', artifact.id) or artifact.id}",
        "doc": "d/" + re.sub(r"[^a-z0-9]+", "-", artifact.title.lower()).strip("-"),
        "incident": f"incidents/{artifact.id}",
    }[artifact.kind]
    return f"https://{HOSTS[artifact.kind]}.{domain}/{path}"


def apply_artifacts(world, artifacts: list[Artifact], anchors: list[str], first: int, day: int) -> None:
    """Record the plan's artifacts, once each: earlier ones stay as they were; a new one is made in its day's morning,
    by a person, under a title that states no fact (a ticket app posts it), and a ticket app announces a ticket."""
    done = {r["id"]: r for r in world.db.execute("SELECT * FROM artifacts")}
    planned = {a.id: a for a in artifacts}
    if len(planned) < len(artifacts):
        raise ValueError("each artifact has an id of its own")
    if gone := sorted(done.keys() - planned.keys()):
        raise ValueError(f"recorded artifacts stay in the plan: {gone}")
    people = {r[0] for r in world.db.execute("SELECT id FROM users WHERE is_bot = 0")}
    for a in (a for i, a in planned.items() if i not in done):
        at = window(world, a.day, "morning")[0] + int(digest(a.model_dump())[:6], 16) % 90 * 60_000_000 + 11
        if a.day < max(day, first) or at <= present(world):
            raise ValueError(f"artifact {a.id}: made on a later moment than the present")
        if a.owner_id not in people:
            raise ValueError(f"artifact {a.id}: its owner is a person of the company, not {a.owner_id}")
        found = [x for x in anchors if re.search(rf"(?<!\w){re.escape(x)}(?!\w)", a.title, re.IGNORECASE)]
        if TIME_LITERAL.search(a.title) or found:
            raise ValueError(f"artifact {a.id}: its title states no fact or time: {found or 'a time'}")
        world.insert(
            "artifacts",
            [
                dict(
                    id=a.id,
                    kind=a.kind,
                    title=a.title,
                    url=artifact_url(world, a),
                    owner_id=a.owner_id,
                    created_us=at,
                )
            ],
        )
        if a.kind == "ticket" and (
            app := world.db.execute(
                "SELECT user_id, channel_id FROM integrations WHERE kind = 'tickets' ORDER BY user_id"
            ).fetchone()
        ):
            queue(
                world,
                at,
                app[1],
                app[0],
                f"<{artifact_url(world, a)}|{a.id}> created: {a.title} · assigned to <@{a.owner_id}>",
                f"artifact:{a.id}",
                "bot_message",
            )


def move_moment(world, move: Move) -> int:
    """When a move happens: in its day's morning, at a minute the move itself draws."""
    return (
        window(world, move.day, "morning")[0] + int(digest(move.model_dump())[:6], 16) % 120 * 60_000_000 + 7
    )


def member_through(world, channel_id: str, user_id: str, start_us: int, end_us: int) -> bool:
    """Whether someone is a member of a channel from `start_us` to `end_us`."""
    return bool(
        world.db.execute(
            "SELECT 1 FROM members WHERE channel_id = ? AND user_id = ? AND joined_us <= ? AND (left_us IS NULL OR left_us >= ?)",
            (channel_id, user_id, start_us, end_us),
        ).fetchone()
    )


def apply_moves(world, moves: list[Move], first: int, day: int) -> None:
    """Record the plan's moves, once each: earlier ones stay as they were; new ones are checked at their moment and
    written as members' joins and leaves, departures and archives, with Slack's own posts queued for joins and leaves."""
    done = (
        json.loads(world_meta(world, "moves") or "[]")
        if world.db.execute("SELECT 1 FROM world_meta WHERE key = 'moves'").fetchone()
        else []
    )
    planned = [m.model_dump() for m in moves]
    if gone := [m for m in done if m not in planned]:
        raise ValueError(f"recorded moves stay in the plan: {gone}")
    people = {r[0] for r in world.db.execute("SELECT id FROM users WHERE is_bot = 0")}
    for move in (Move(**m) for m in planned if m not in done):
        at, who, where = move_moment(world, move), move.user_id, move.channel_id
        if move.day < max(day, first) or at <= present(world):
            raise ValueError(
                f"a move is planned for a later moment than the present: {move.model_dump(exclude_none=True)}"
            )
        if who is not None and who not in people:
            raise ValueError(f"{move.kind}: {who} is no person of the company")
        channel = (
            world.db.execute("SELECT * FROM channels WHERE id = ?", (where,)).fetchone() if where else None
        )
        if where and (channel is None or channel["name"] is None):
            raise ValueError(f"{move.kind}: {where} is a named channel of the company")
        if channel is not None and channel["archived_us"] is not None and channel["archived_us"] <= at:
            raise ValueError(f"{move.kind}: {where} is archived by then")
        inside = who and where and member_through(world, where, who, at, at)
        if move.kind == "join":
            if inside:
                raise ValueError(f"join: {who} is a member of {where} already")
            world.insert("members", [dict(channel_id=where, user_id=who, joined_us=at)])
            queue(
                world,
                at,
                where,
                who,
                f"<@{who}> has joined the channel",
                f"move:join:{where}:{who}:{move.day}",
                "channel_join",
            )
        elif move.kind == "leave":
            if not inside:
                raise ValueError(f"leave: {who} is no member of {where} then")
            queue(
                world,
                at - 1,
                where,
                who,
                f"<@{who}> has left the channel",
                f"move:leave:{where}:{who}:{move.day}",
                "channel_leave",
            )
            world.db.execute(
                "UPDATE members SET left_us = ? WHERE channel_id = ? AND user_id = ? AND joined_us <= ? AND (left_us IS NULL OR left_us > ?)",
                (at, where, who, at, at),
            )
        elif move.kind == "depart":
            world.db.execute(
                "UPDATE members SET left_us = ? WHERE user_id = ? AND joined_us < ? AND (left_us IS NULL OR left_us > ?)",
                (at, who, at, at),
            )
            world.db.execute("UPDATE users SET deactivated_us = ?, is_deleted = 1 WHERE id = ?", (at, who))
        else:
            world.db.execute("UPDATE channels SET archived_us = ?, is_archived = 1 WHERE id = ?", (at, where))
    world.db.execute(
        "INSERT OR REPLACE INTO world_meta (key, value) VALUES ('moves', ?)", (json.dumps(planned),)
    )


class Commit(StrictModel):
    """A promise the line makes, due by the end of a calendar day."""

    id: SafeId
    text: NonEmptyText
    due_day: int = Field(ge=1)


class Close(StrictModel):
    """How an open promise ends: kept, changed or dropped."""

    id: SafeId
    status: Literal["kept", "changed", "dropped"]


class PostLine(StrictModel):
    author_id: SafeId
    text: NonEmptyText
    reply_to: int | None = Field(default=None, ge=0)
    conveys: Ids = Field(default_factory=list)
    pause: Literal["", "hours"] = ""
    reactions: list[Reaction] = Field(default_factory=list)
    commits: list[Commit] = Field(default_factory=list)
    closes: list[Close] = Field(default_factory=list)


class Conversation(StrictModel):
    """One stretch of talk in one channel, posted at the present. `thread` continues an earlier thread of the channel:
    every line then replies in it."""

    channel_id: SafeId
    about: NonEmptyText
    thread: int | None = None
    lines: list[PostLine] = Field(min_length=1, max_length=40)


# ---------------------------------------------------------------------- the clock


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


def quotas(world, settings) -> dict[int, dict]:
    """Each day's quota, by its segment. A day the segment's first plan put an event on weighs as a workday, so a
    storyline's busiest night is not a quiet day's handful of messages; the days are fixed by that plan, so a closed
    day's quota never changes."""
    dates = dict(world.db.execute("SELECT day, date FROM calendar ORDER BY day").fetchall())
    out = {}
    for s in segments(world, settings):
        days = range(s["first"], s["last"] + 1)
        out |= dict(
            zip(
                days,
                spread(
                    [dates[d] for d in days],
                    s["first"],
                    s["messages"],
                    s["weekend"],
                    s["parts"],
                    s["busy"] or (),
                ),
            )
        )
    return out


def keep_busy(world, events: list) -> None:
    """The days the first plan of a segment puts an event on, kept from then on: a world in one run's first plan, or
    a chapter's, for its own days only."""
    if (stored := meta(world, "segments")) is None:
        world.db.execute(
            "INSERT OR IGNORE INTO world_meta (key, value) VALUES ('busy_days', ?)",
            (json.dumps(sorted({e.day for e in events})),),
        )
        return
    spans = json.loads(stored)
    current = next(s for s in spans if s["first"] <= today(world) <= s["last"])
    if current["busy"] is None:
        current["busy"] = sorted({e.day for e in events if current["first"] <= e.day <= current["last"]})
        world.db.execute("UPDATE world_meta SET value = ? WHERE key = 'segments'", (json.dumps(spans),))


# A day's shares, each checked on that day's own messages at its close, and named as the author reads them.
SHARES = {
    "reply_share": "thread replies",
    "reaction_rate": "reacted to",
    "dm_share": "in DMs",
    "mention_rate": "mention someone",
}


def share_band(settings, share: str, n: int) -> tuple[int, int]:
    """A share's band for n messages: mentions keep a tighter one, as their measured rate is small."""
    slack = settings.author.mention_tolerance if share == "mention_rate" else settings.author.share_tolerance
    return band(getattr(settings.activity, share), slack, n)


def band(target: float, slack: float, n: int) -> tuple[int, int]:
    """How many of n messages may have a property whose share is target ± slack, rounded outward so that every count
    of messages has a range."""
    return max(0, math.floor(round((target - slack) * n, 9))), min(
        n, math.ceil(round((target + slack) * n, 9))
    )


def tally(world, day: int) -> dict[str, int]:
    """Today's messages, and how many are thread replies, carry a reaction, sit in DMs, mention someone."""
    row = world.db.execute(
        """SELECT COUNT(*), COALESCE(SUM(m.parent_id IS NOT NULL), 0),
        COALESCE(SUM(EXISTS (SELECT 1 FROM reactions r WHERE r.message_id = m.id)), 0),
        COALESCE(SUM(c.type IN ('im', 'mpim')), 0),
        COALESCE(SUM(EXISTS (SELECT 1 FROM message_mentions x WHERE x.message_id = m.id)), 0)
        FROM messages m JOIN channels c ON c.id = m.channel_id JOIN users u ON u.id = m.user_id
        WHERE {PEOPLE} AND m.ts_us >= ? AND m.ts_us < ?""".format(PEOPLE=PEOPLE),
        bounds(world, day),
    ).fetchone()
    return dict(zip(("messages", *SHARES), row))


def promises_made(world, day: int) -> int:
    """The promises today's messages made."""
    return world.db.execute(
        "SELECT COUNT(*) FROM commitments c JOIN messages m ON m.id = c.message_id WHERE m.ts_us >= ? AND m.ts_us < ?",
        bounds(world, day),
    ).fetchone()[0]


def owed(world, day: int) -> tuple[int, int]:
    """The lines today still needs, at least, in DMs and elsewhere: one for each channel and author with a fact planned
    for today still unstated, or one for the promises due today if there is no such line."""
    groups = world.db.execute(
        """SELECT DISTINCT c.type IN ('im', 'mpim') AS dm, f.channel_id, f.author_id FROM facts f
        JOIN channels c ON c.id = f.channel_id WHERE f.day = ?
        AND NOT EXISTS (SELECT 1 FROM evidence e WHERE e.fact_id = f.id AND e.role = 'anchor')""",
        (day,),
    ).fetchall()
    due = world.db.execute(
        "SELECT 1 FROM commitments WHERE status = 'open' AND due_us <= ?", (bounds(world, day)[1],)
    ).fetchone()
    in_dms = sum(g[0] for g in groups)
    return in_dms, len(groups) - in_dms + (1 if due and not groups else 0)


def room(world, settings, day: int) -> dict | None:
    """Whether today can still close from where it stands, and how: None when no count of messages within its quota,
    with the lines it still owes, puts all three shares in their bands at once (past messages never change; each
    later line may be a reply, reacted to, in a DM or not). Otherwise the counts it can close at, and, at the count
    nearest its quota, how many more lines of each kind it needs."""
    quota, slack = quotas(world, settings)[day]["messages"], settings.author.tolerance
    lo, hi = math.ceil(round(quota * (1 - slack), 9)), math.floor(round(quota * (1 + slack), 9))
    have, (in_dms, elsewhere) = tally(world, day), owed(world, day)
    fits = {}
    for n in range(max(lo, have["messages"] + in_dms + elsewhere), hi + 1):
        more, extra = {}, n - have["messages"]
        for share in SHARES:
            low, high = share_band(settings, share, n)
            least = have[share] + (in_dms if share == "dm_share" else 0)
            most = have[share] + extra - (elsewhere if share == "dm_share" else 0)
            if max(low, least) > min(high, most):
                break
            more[share] = (max(low, least) - have[share], min(high, most) - have[share])
        else:
            fits[n] = more
    if not fits:
        return None
    best = min(fits, key=lambda n: (abs(n - quota), n))
    return {"closes": (min(fits), max(fits)), "owed": in_dms + elsewhere, "at": best, "more": fits[best]}


def today_line(world, settings, day: int) -> str:
    """Today's count and shares against their bands, what closing it still takes, and the time left for it."""
    have, ways = tally(world, day), room(world, settings, day)
    shares = ", ".join(f"{SHARES[s]} {have[s]}" for s in SHARES)
    start, end = bounds(world, day)
    minutes = max(end - max(present(world), start), 0) // 60_000_000
    left = f"{minutes // 60} h {minutes % 60} min left today"
    if ways is None:
        return f"{have['messages']} messages today ({shares}); today can no longer close within its quota and shares; {left}"
    lo, hi = ways["closes"]
    needs = ", ".join(f"{a}-{b} {SHARES[s]}" for s, (a, b) in ways["more"].items())
    promised = (
        f"; promises made {promises_made(world, day)}, at least {math.floor(settings.activity.promise_rate * ways['at'])}"
        if settings.activity.promise_rate
        else ""
    )
    return (
        f"{have['messages']} messages today ({shares}); it closes with {lo}-{hi} messages. At {ways['at']}: "
        f"{ways['at'] - have['messages']} more lines, of them {needs}{promised}; {left}"
    )


def agenda_budget(world, settings, day: int) -> int:
    """The messages of a day's everyday conversations: its quota, less a conversation's worth of lines for each
    channel a fact planned for the day is first stated in."""
    channels = world.db.execute(
        "SELECT COUNT(DISTINCT channel_id) FROM facts WHERE day = ?", (day,)
    ).fetchone()[0]
    return max(
        0, quotas(world, settings)[day]["messages"] - round(settings.activity.conversation_lines * channels)
    )


def posted(world, start_us: int, end_us: int) -> int:
    """People's messages in the span: a bot's post or a join is code's, and counts in no quota or share."""
    return world.db.execute(
        f"SELECT COUNT(*) FROM messages m JOIN users u ON u.id = m.user_id WHERE {PEOPLE} AND m.ts_us >= ? AND m.ts_us < ?",
        (start_us, end_us),
    ).fetchone()[0]


# ---------------------------------------------------------------------- the ledger, planned and re-planned


def check_facts(facts: list, storylines: set[str]) -> None:
    """A ledger's facts: a value with no time in it, a known storyline, an anchor made of words of the value,
    relations to facts of the same ledger and no fact that comes after itself, and anchors a line can tell apart:
    a line containing a fact's anchor states that fact, so two facts with one anchor could never be first stated
    apart, and a fact whose anchor contains another's comes after it."""
    known = {f.id for f in facts}
    for fact in facts:
        if TIME_LITERAL.search(fact.value):
            raise ValueError(
                f"{fact.id}: a value has no time or date in it; its moment is planned apart from it"
            )
        if fact.storyline not in storylines:
            raise ValueError(f"{fact.id}: storyline {fact.storyline} is not in storylines")
        if fact.anchor and f" {normalized(fact.anchor)} " not in f" {normalized(fact.value)} ":
            raise ValueError(
                f"{fact.id}: an anchor is words of its value; {fact.anchor!r} is not in {fact.value!r}"
            )
        if unknown := set(fact.after) - known | ({fact.supersedes} - known - {None}):
            raise ValueError(
                f"{fact.id}: after and supersedes name facts of this ledger; unknown {sorted(unknown)}"
            )
    earlier = {f.id: {*f.after, *([f.supersedes] if f.supersedes else [])} for f in facts}
    before = {}  # each fact's earlier facts, through after and supersedes

    def reach(fact: str, path: tuple = ()) -> set[str]:
        if fact in path:
            raise ValueError(
                f"{fact} comes after itself, through {' > '.join((*path[path.index(fact) :], fact))}"
            )
        if fact not in before:
            before[fact] = set().union(*({d} | reach(d, (*path, fact)) for d in earlier[fact]))
        return before[fact]

    for fact in facts:
        reach(fact.id)
    anchors = {f.id: (f.anchor, f" {normalized(f.anchor)} ") for f in facts if f.anchor}
    for f, (word, padded) in anchors.items():
        for g, (other, inside) in anchors.items():
            if f == g or inside not in padded:
                continue
            if inside == padded and f < g:
                raise ValueError(
                    f"{f} and {g} have one anchor, {word!r}: a line that states one states both. A repeat of a value is "
                    "the same fact conveyed again; plan it once, or give each fact words of its own"
                )
            if inside != padded and g not in before[f]:
                raise ValueError(
                    f"{f}'s anchor {word!r} contains {g}'s, {other!r}: a line that states {f} states {g} too, so {f} "
                    f"comes after {g}"
                )


def fact_row(fact: PlanFact, events: dict[str, Event], moment: dict[str, int], zone: str) -> dict:
    row = fact.model_dump(
        include={
            "id",
            "storyline",
            "subject",
            "attribute",
            "value",
            "anchor",
            "channel_id",
            "author_id",
            "day",
            "summary",
        }
    ) | {"event_id": fact.event, "is_decoy": int(fact.decoy)}
    if fact.event is None:
        return row | {"moment_us": None, "moment_zone": None, "moment_kind": None}
    return row | {
        "moment_us": moment[fact.event],
        "moment_zone": events[fact.event].zone or zone,
        "moment_kind": fact.kind,
    }


def front_loaded(world, settings, facts: list, day: int) -> str | None:
    """Why the plan's facts are front-loaded, if they are: each third of the segment being written (the world's
    calendar, or a chapter's days) not yet over first states at least half its share of the facts first stated in that
    segment, its share of the segment's messages (an event day weighs as a workday). S1's storylines were told by day
    8, and its last days had nothing left to state. A third still open can always take more facts, today or later."""
    span = segment_of(world, settings, day)
    shares = quotas(world, settings)
    days = list(range(span["first"], span["last"] + 1))
    facts = [f for f in facts if f.day >= span["first"]]  # an earlier chapter's facts are its own
    total = sum(shares[d]["messages"] for d in days) or 1
    whose = "the world's" if span["first"] == 1 else "this chapter's"
    start = 0
    for size in apportion(len(days), [1, 1, 1]):
        third, start = days[start : start + size], start + size
        if not third or third[-1] < day:
            continue
        share = len(facts) * sum(shares[d]["messages"] for d in third) / total
        if (planned := sum(f.day in third for f in facts)) < (least := int(share / 2)):
            return (
                f"days {third[0]}-{third[-1]} first state {planned} of the plan's {len(facts)} facts; they hold "
                f"{share / len(facts):.0%} of {whose} messages, so they first state at least {least}: plan facts "
                "for them, so the storylines run to the end"
            )
    return None


def record_plan(world, plan: Plan, settings, slots=()) -> None:
    """The ledger in time order. Storylines are fixed once planned (summaries may change; a chapter adds its own after
    them); an event, once planned, never moves or goes; a stated fact keeps everything but its summary, while an
    unstated one may change or go and is planned for today or later. A fact about an event carries the event's moment.
    An earlier chapter's facts and events stay as they are, and a new event is in this chapter. The board gives each
    ledger and hybrid slot of `slots` the planned facts it will rest on (and each slot whose level needs something of
    its facts, and each entry of a chapter's brief), as many channels, relations, near-misses and values as it needs:
    the material of a hard backward task exists before its first day, when it can still be posted. Board entries of
    other slots (an earlier chapter's) stay."""
    count = settings.storylines
    if len(plan.storylines) != count:
        raise ValueError(f"plan exactly {count} storylines")
    ids = [x.id for x in (*plan.storylines, *plan.events, *plan.facts)]
    if repeated := sorted({i for i in ids if ids.count(i) > 1}):
        raise ValueError(f"storyline, event and fact ids are distinct; repeated {repeated}")
    kept = [r[0] for r in world.db.execute("SELECT id FROM storylines ORDER BY position")]
    if kept and kept != [s.id for s in plan.storylines[: len(kept)]]:
        raise ValueError(
            f"the storylines are planned: keep {kept} in this order, first (summaries may change)"
        )
    check_facts(plan.facts, {s.id for s in plan.storylines})
    anchors = [f.anchor for f in plan.facts if f.anchor] + [
        r[0] for r in world.db.execute("SELECT anchor FROM facts WHERE anchor IS NOT NULL")
    ]
    if errors := event_errors(world, plan.events, anchors):
        raise ValueError("; ".join(errors))
    day, zone = today(world), world_meta(world, "zone")
    if day is None:
        raise ValueError("the calendar is closed: the ledger is final")
    events, moment = {e.id: e for e in plan.events}, {}
    old = {r["id"]: dict(r) for r in world.db.execute("SELECT * FROM events")}
    old_ids, frozen, first = set(old), frozen_before(world), chapter_start(world)
    for e in plan.events:
        if e.storyline not in {s.id for s in plan.storylines}:
            raise ValueError(f"{e.id}: storyline {e.storyline} is not in storylines")
        if TIME_LITERAL.search(e.title):
            raise ValueError(f"{e.id}: a title has no time or date in it; day and time say when")
        moment[e.id] = at(world, Moment(day=e.day, time=e.time, zone=e.zone or zone))
        if e.id not in old_ids and moment[e.id] < frozen:
            raise ValueError(
                f"{e.id}: an earlier chapter's days are written and frozen; a new event is in this one (day {first} or later)"
            )
        if e.id in old and (old[e.id]["moment_us"], old[e.id]["zone"], old[e.id]["title"]) != (
            moment[e.id],
            e.zone or zone,
            e.title,
        ):
            raise ValueError(
                f"{e.id} is planned: an event never moves; a change of plan is a new event, with a fact that "
                "supersedes the old one"
            )
    if gone := sorted(old.keys() - events.keys()):
        raise ValueError(f"planned events stay in the ledger: {gone}")
    apply_moves(
        world, plan.moves, first, day
    )  # a plan is recorded on a trial copy: a refusal later drops these
    apply_artifacts(world, plan.artifacts, anchors, first, day)
    stated = {r[0] for r in world.db.execute("SELECT DISTINCT fact_id FROM evidence")}
    current = {r["id"]: dict(r) for r in world.db.execute("SELECT * FROM facts")}
    relations = {
        (r[0], r[1], r[2]) for r in world.db.execute("SELECT src_fact, dst_fact, kind FROM fact_relations")
    }
    planned = {f.id: f for f in plan.facts}
    if gone := sorted(stated - planned.keys()):
        raise ValueError(f"stated facts stay in the ledger: {gone}")
    backward = {s.id: s for s in slots if on_board(settings, s)}
    board = {e.slot: list(dict.fromkeys(e.facts)) for e in plan.board}
    if len(board) < len(plan.board):
        raise ValueError("the board has one entry for each slot")
    if strays := sorted(board.keys() - backward.keys()):
        raise ValueError(f"the board is for the slots of now.md that rest on planned facts; {strays} do not")
    if missing := sorted(backward.keys() - board.keys()):
        raise ValueError(
            f"the board names the facts each slot that rests on planned facts will rest on; {missing} have none"
        )
    if unknown := sorted({f for facts in board.values() for f in facts} - planned.keys()):
        raise ValueError(f"the board names facts of the plan; {unknown} are not")
    rows = {}
    for f in plan.facts:
        if f.event is not None and f.event not in events:
            raise ValueError(f"{f.id}: event {f.event} is not in events")
        rows[f.id] = fact_row(f, events, moment, zone)
        mine = {(f.id, d, "after") for d in f.after} | (
            {(f.id, f.supersedes, "supersedes")} if f.supersedes else set()
        )
        if f.id in stated:
            if {k: v for k, v in rows[f.id].items() if k != "summary"} != {
                k: current[f.id][k] for k in rows[f.id] if k != "summary"
            } or mine != {r for r in relations if r[0] == f.id}:
                raise ValueError(f"{f.id} is stated: only its summary may change")
            if current[f.id]["day"] < first and f.summary != current[f.id]["summary"]:
                raise ValueError(
                    f"{f.id} is an earlier chapter's, written and frozen: it stays as it is, its summary too"
                )
            continue
        if not member_through(world, f.channel_id, f.author_id, *bounds(world, f.day)):
            raise ValueError(
                f"{f.id}: its author {f.author_id} is a member of {f.channel_id} all of day {f.day}"
            )
        if (
            (
                gone := world.db.execute(
                    "SELECT archived_us FROM channels WHERE id = ?", (f.channel_id,)
                ).fetchone()
            )
            and gone[0] is not None
            and gone[0] < bounds(world, f.day)[1]
        ):
            raise ValueError(f"{f.id}: {f.channel_id} is archived by day {f.day}")
        if f.day < day:
            raise ValueError(f"{f.id}: an unstated fact is planned for today (day {day}) or later")
        if f.kind == "happened" and f.day < events[f.event].day:
            raise ValueError(f"{f.id}: what happened is first stated on or after the day of {f.event}")
        if f.kind == "scheduled" and f.day > events[f.event].day:
            raise ValueError(f"{f.id}: what is scheduled is first stated on or before the day of {f.event}")
        if f.kind == "scheduled" and f.day == day and moment[f.event] <= present(world):
            raise ValueError(f"{f.id}: {f.event} is past; what is scheduled for it is told before it happens")
        earlier = [*f.after, *([f.supersedes] if f.supersedes else [])]
        if later := [d for d in earlier if planned[d].day > f.day]:
            raise ValueError(f"{f.id} comes after {later}, which are first stated on a later day")
    with world.batch():
        # The plan's board is whole, as its facts are; in a chapter, an earlier chapter's entries (on stated facts) stay.
        scope = (
            list(backward)
            if first > 1
            else [r[0] for r in world.db.execute("SELECT DISTINCT slot FROM board")]
        )
        world.db.execute(f"DELETE FROM board WHERE slot IN ({', '.join('?' * len(scope))})", scope)
        for s in plan.storylines[: len(kept)]:
            world.db.execute("UPDATE storylines SET summary = ? WHERE id = ?", (s.summary, s.id))
        world.insert(
            "storylines",
            [s.model_dump() | {"position": i} for i, s in enumerate(plan.storylines, 1) if i > len(kept)],
        )
        world.insert(
            "events",
            [
                dict(
                    id=e.id, storyline=e.storyline, title=e.title, moment_us=moment[e.id], zone=e.zone or zone
                )
                | dict(post_by=app_id(world, e.post_by) if e.post_by else None)
                for e in plan.events
                if e.id not in old
            ],
        )
        keep_busy(world, plan.events)
        world.db.execute("DELETE FROM fact_relations")
        unstated = [f for f in current if f not in stated]
        world.db.execute(f"DELETE FROM facts WHERE id IN ({', '.join('?' * len(unstated))})", unstated)
        world.insert("facts", [rows[f] for f in planned if f not in stated])
        for f in stated:
            world.db.execute("UPDATE facts SET summary = ? WHERE id = ?", (planned[f].summary, f))
        world.insert(
            "fact_relations",
            [
                dict(src_fact=f.id, dst_fact=d, kind="after")
                for f in plan.facts
                for d in dict.fromkeys(f.after)
            ]
            + [
                dict(src_fact=f.id, dst_fact=f.supersedes, kind="supersedes")
                for f in plan.facts
                if f.supersedes
            ],
        )
        if short := front_loaded(world, settings, plan.facts, day):
            raise ValueError(short)
        world.insert("board", [dict(slot=s, fact_id=f) for s, facts in board.items() for f in facts])
        for s, facts in board.items():
            needs, level = slot_needs(settings, backward[s]), backward[s].level
            if needs and (short := unmet(needs, fact_measures(world, facts) | board_measures(world, facts))):
                raise ValueError(
                    f"the board's {s} (level-{level} {backward[s].category}) needs " + "; ".join(short)
                )
        if room(world, settings, day) is None:  # inside the batch: the refusal takes the plan back
            raise ValueError(f"this plan leaves today no way to close: {today_line(world, settings, day)}")
        # The apps' posts: each event one reports, at its moment, and today's routine ones.
        queue_events(world, settings, present(world))
        queue_day(world, settings, day)


def unplanned(world, settings, slots=()) -> list[str]:
    """What a plan turn still owes: its storylines, a chapter's own facts (a chapter that plants nothing new is no
    chapter), and a board entry for each slot that rests on planned facts."""
    errors, count = [], settings.storylines
    if world.db.execute("SELECT COUNT(*) FROM storylines").fetchone()[0] != count:
        errors.append(f"plan the ledger with world_plan ({count} storylines)")
    first = chapter_start(world)
    if first > 1 and not world.db.execute("SELECT 1 FROM facts WHERE day >= ?", (first,)).fetchone():
        errors.append(f"plan this chapter's facts with world_plan: none is planned for day {first} or later")
    boarded = {r[0] for r in world.db.execute("SELECT DISTINCT slot FROM board")}
    if missing := [s.id for s in slots if on_board(settings, s) and s.id not in boarded]:
        errors.append(f"give the board an entry for {missing} with world_plan")
    return errors


# ---------------------------------------------------------------------- conversations, at the present


def opening(world, settings, rng, now: int, day: int) -> int:
    """How long the present waits before a conversation starts: the part's remaining time, spread over the
    conversations it can still expect (what its quota still holds, or what its pace so far brings in the time left,
    whichever is more); never longer than the day can spare, the rest of the day spread over the conversations it
    still needs to close. An author who writes past a part's quota keeps its pace: each conversation does not halve
    the time left, which bunched S1's conversations at the end of the morning and of the evening."""
    part = part_of(world, now)
    start, end = window(world, day, part)
    left = max(quotas(world, settings)[day]["parts"][part] - posted(world, start, now), 0)
    started = world.db.execute(
        "SELECT COUNT(*) FROM scenes WHERE slot_start_us >= ? AND slot_start_us < ?", (start, now)
    ).fetchone()[0]
    pace = started * (end - now) / (now - start) if now > start else 0.0
    conversations = max(1.0, left / settings.activity.conversation_lines, pace)
    wait = int(rng.random() * 2 * max(end - now, 0) / (conversations + 1))
    if (ways := room(world, settings, day)) is not None and (
        need := ways["closes"][0] - tally(world, day)["messages"]
    ) > 0:
        still = math.ceil(need / settings.activity.conversation_lines)
        wait = min(wait, max(bounds(world, day)[1] - now, 0) // (still + 1))
    return wait + rng.randrange(1_000_000, 60_000_000)


def stated_anchors(facts: dict[str, dict], text: str) -> list[str]:
    """The facts whose anchor the text contains."""
    said = f" {normalized(PLACEHOLDER.sub(' ', text))} "
    return [f for f, x in facts.items() if x["anchor"] and f" {normalized(x['anchor'])} " in said]


def brief(world) -> dict[str, dict]:
    """A chapter's brief, as it was written when the chapter started: each entry's needs, by slot."""
    return json.loads(meta(world, "brief", "{}"))


def hidden_answers(world) -> dict[str, list[tuple[str, int | None]]]:
    """The answer facts of the brief's entries that need their answer out of easy sight, by fact: each entry, with
    the last day a near-miss of it is first stated (None when it has none)."""
    needs, out = brief(world), {}
    for slot, ids in world.db.execute(
        "SELECT slot, group_concat(fact_id) FROM board GROUP BY slot"
    ).fetchall():
        if not needs.get(slot, {}).get("hidden"):
            continue
        answers = answer_facts(world, ids.split(","))
        misses = near_misses(world, answers)
        last = max(
            (world.db.execute("SELECT day FROM facts WHERE id = ?", (m,)).fetchone()[0] for m in misses),
            default=None,
        )
        for answer in answers:
            out.setdefault(answer, []).append((slot, last))
    return out


def post(world, conversation: Conversation, settings, rng, gaps) -> dict:
    """One conversation at the present, in one channel among its members. A line that contains a planned fact's
    anchor states that fact and lists it in conveys; a fact's first statement is its planned author's, in its planned
    channel, on its planned day. Code times each line after the last, before midnight, and moves the present to the
    last line; promises open and close with the lines that make and settle them."""
    day = today(world)
    if day is None:
        raise ValueError("the calendar is closed; nothing more is posted")
    start, end = bounds(world, day)
    lines, channel = conversation.lines, conversation.channel_id
    if (
        row := world.db.execute("SELECT archived_us FROM channels WHERE id = ?", (channel,)).fetchone()
    ) is None:
        raise ValueError(f"no channel {channel}")
    if row[0] is not None and row[0] <= end:
        raise ValueError(f"{channel} is archived today: no one posts in it from then on")
    facts = {r["id"]: dict(r) for r in world.db.execute("SELECT * FROM facts")}
    moment = moments(world)
    errors = []  # members post and react, and replies stay in their channel's threads: the world file's triggers
    for i, line in enumerate(lines):
        if line.author_id in {r.user_id for r in line.reactions}:
            errors.append(f"line {i}: nobody reacts to their own line")
        if untagged := [f for f in stated_anchors(facts, line.text) if f not in line.conveys]:
            errors.append(
                f"line {i}: it contains the anchors of {untagged}, so it states them: list them in conveys"
            )
    errors += text_errors(world, lines, set(facts), set(moment))
    anchored = {r[0] for r in world.db.execute("SELECT fact_id FROM evidence WHERE role = 'anchor'")}
    follows = {}
    for src, dst in world.db.execute(
        "SELECT src_fact, dst_fact FROM fact_relations WHERE kind IN ('after', 'supersedes')"
    ):
        follows.setdefault(src, {})[dst] = None  # a fact may both follow and supersede another
    first, hide = set(), hidden_answers(world)
    direct = world.db.execute(
        "SELECT type IN ('im', 'mpim') FROM channels WHERE id = ?", (channel,)
    ).fetchone()
    for i, line in enumerate(lines):
        for f in line.conveys:
            if f in facts and f not in anchored and f not in first:
                x = facts[f]
                if (channel, line.author_id, day) != (x["channel_id"], x["author_id"], x["day"]):
                    errors.append(
                        f"line {i}: {f} is first stated by {x['author_id']} in {x['channel_id']} on day {x['day']}"
                    )
                reply = line.reply_to is not None or conversation.thread is not None
                if (
                    f in hide
                    and not (reply or (direct and direct[0]))
                    and [s for s, last in hide[f] if last is None or day <= last]
                ):
                    errors.append(
                        f"line {i}: {f} answers the brief's {[s for s, _ in hide[f]]}, which needs its answer stated where "
                        "it is harder to see than its near-misses: in a thread reply, in a DM, or on a later day than "
                        "every near-miss"
                    )
                if pending := [d for d in follows.get(f, []) if d not in anchored and d not in first]:
                    errors.append(f"line {i}: {f} comes after {pending}, which are not stated yet")
                first.add(f)
    if conversation.thread is not None and any(line.reply_to is not None for line in lines):
        errors.append("in a continued thread every line replies to it: leave reply_to empty")
    promises = {r["id"]: r["status"] for r in world.db.execute("SELECT id, status FROM commitments")}
    days = {r[0] for r in world.db.execute("SELECT day FROM calendar")}
    for i, line in enumerate(lines):
        for c in line.commits:
            if c.due_day < day or c.due_day not in days:
                errors.append(f"line {i}: {c.id} is due today or on a later day of the calendar")
        for c in line.closes:
            if promises.get(c.id) != "open":
                errors.append(f"line {i}: {c.id} is no open commitment")
    if errors:
        raise ValueError("; ".join(errors))
    now, before = present(world), today_line(world, settings, day)
    stamps = [
        now + (gaps.sample(rng, "hours") if lines[0].pause else opening(world, settings, rng, now, day))
    ]
    for line in lines[1:]:
        stamps.append(stamps[-1] + gaps.sample(rng, line.pause))
    if stamps[-1] >= end:
        raise ValueError(
            f"this conversation would run past midnight ({clock(world, stamps[-1])}); post fewer lines or pauses "
            "today, or carry it on tomorrow"
        )
    told = {}
    for stamp, line in zip(stamps, lines):
        for f in line.conveys:
            told.setdefault(f, stamp)
    if late := [
        f
        for f, moment_us in owed_moments(world, day)
        if moment_us <= stamps[-1] and not told.get(f, moment_us) < moment_us
    ]:
        raise ValueError(
            f"this conversation would run to {clock(world, stamps[-1])}, past the moment of {late}, which are "
            "scheduled and still to be told before it: tell them first, or post less now"
        )
    number = world.db.execute("SELECT COUNT(*) FROM scenes WHERE day = ?", (day,)).fetchone()[0] + 1
    conveyed = [f for line in lines for f in line.conveys]
    scene = dict(
        id=f"d{day:02d}-{number:03d}", channel_id=channel, day=day, part=part_of(world, stamps[0])
    ) | dict(
        storyline=facts[conveyed[0]]["storyline"] if conveyed else None,
        slot_start_us=stamps[0],
        slot_end_us=stamps[-1] + 1,
        situation=conversation.about,
        plan_json=json.dumps(conversation.model_dump(mode="json")),
    )
    with world.batch():
        world.db.execute("UPDATE world_meta SET value = ? WHERE key = 'now_us'", (str(stamps[-1]),))
        ids = insert_lines(world, scene, lines, stamps, moment, first, rng, gaps, conversation.thread, end)
        for message, line in zip(ids, lines):
            world.insert(
                "commitments",
                [
                    dict(
                        id=c.id,
                        owner_id=line.author_id,
                        text=c.text,
                        message_id=message,
                        due_us=bounds(world, c.due_day)[1],
                    )
                    for c in line.commits
                ],
            )
            for c in line.closes:
                world.db.execute(
                    "UPDATE commitments SET status = ?, closed_by = ? WHERE id = ?", (c.status, message, c.id)
                )
        if room(world, settings, day) is None:  # inside the batch: the refusal takes the conversation back
            raise ValueError(
                f"after this conversation today could not close: {today_line(world, settings, day)}. Before it, "
                f"{before}"
            )
    texts = {
        r[0]: r[1]
        for r in world.db.execute(
            f"SELECT id, text FROM messages WHERE id IN ({', '.join('?' * len(ids))})", ids
        )
    }
    return {
        "conversation": scene["id"],
        "messages": [
            {"id": m, "at": clock(world, ts), "author": line.author_id, "text": texts[m]}
            for m, ts, line in zip(ids, stamps, lines)
        ],
        "now": clock(world, stamps[-1]),
        "today": today_line(world, settings, day),
    }


# ---------------------------------------------------------------------- moving the present, closing a day


def owed_moments(world, day: int) -> list[tuple[str, int]]:
    """The scheduled facts planned for `day` and not stated yet, with their moments: each must be told before it."""
    return world.db.execute(
        """SELECT id, moment_us FROM facts f WHERE day = ? AND moment_kind = 'scheduled'
        AND NOT EXISTS (SELECT 1 FROM evidence e WHERE e.fact_id = f.id AND e.role = 'anchor') ORDER BY id""",
        (day,),
    ).fetchall()


def close_day(world, settings, day: int) -> list[str]:
    """Why the day cannot close yet: its message count outside its quota's tolerance, its shares outside their bands
    on its own messages, a fact planned for it still unstated, a promise due by its end still open."""
    start, end = bounds(world, day)
    quota, slack, errors = quotas(world, settings)[day]["messages"], settings.author.tolerance, []
    count = posted(world, start, end)
    if not quota * (1 - slack) <= count <= quota * (1 + slack):
        errors.append(f"{count} messages today; the day holds about {quota} (within {slack:.0%})")
    for (fact,) in world.db.execute(
        "SELECT id FROM facts f WHERE day = ? AND NOT EXISTS (SELECT 1 FROM evidence e WHERE e.fact_id = f.id AND e.role = 'anchor')",
        (day,),
    ):
        errors.append(f"{fact} is planned for today: its author states it in its channel")
    for (promise,) in world.db.execute(
        "SELECT id FROM commitments WHERE status = 'open' AND due_us <= ?", (end,)
    ):
        errors.append(f"{promise} is due by today's end: keep it, change it or drop it in a message")
    made, least = promises_made(world, day), math.floor(settings.activity.promise_rate * count)
    if made < least:
        errors.append(f"{made} promises made today; the day holds at least {least} (a line's commits)")
    have = tally(world, day)
    for share, name in SHARES.items():
        low, high = share_band(settings, share, count)
        if not low <= have[share] <= high:
            errors.append(f"{have[share]} of today's {count} messages are {name}; the day holds {low}-{high}")
    return errors


def advance(world, settings, to: str) -> dict:
    """Move the present to the start of the part `to` of today; "tomorrow", from the night, closes the day, if it can
    close, and moves to the next day's early part, or past the calendar's end. The part the present is in already
    leaves it where it is, and "tomorrow" again is refused, so a call repeated after a lost response never moves
    twice."""
    day = today(world)
    if day is None:
        raise ValueError("the calendar is closed")
    now, order = present(world), list(PARTS)
    part = part_of(world, now)
    closed, later = False, order[order.index(part) + 1 :]
    if to == part:
        target = now
    elif to in later:
        target = window(world, day, to)[0]
    elif to != "tomorrow":
        raise ValueError(
            f"advance moves to a later part of today, one of {later}, or from the night to tomorrow"
        )
    elif part != "night":
        raise ValueError(f"the day closes from the night: it is the {part}; advance to night first")
    else:
        if errors := close_day(world, settings, day):
            raise ValueError(f"day {day} stays open: " + "; ".join(errors))
        last = world.db.execute("SELECT MAX(day) FROM calendar").fetchone()[0]
        target, closed = (window(world, day + 1, "early")[0] if day < last else bounds(world, day)[1]), True
        if day < last:
            queue_day(world, settings, day + 1)
    if late := [f for f, moment_us in owed_moments(world, day) if moment_us <= target]:
        raise ValueError(
            f"{late} are scheduled before {clock(world, target)} and still to be told: post them before moving on"
        )
    drain(world, max(target, now))  # code's posts due on the way go in first
    world.db.execute("UPDATE world_meta SET value = ? WHERE key = 'now_us'", (str(max(target, now)),))
    return {
        "now": clock(world, max(target, now)),
        "closed_day": day if closed else None,
        "today": today(world),
    }


def drift(world, cards: dict[str, dict], since_us: int = 0) -> dict[str, str]:
    """Each person's messages since `since_us` beside the typing card they were given: what is far from it, by person."""
    rows = world.db.execute(
        "SELECT user_id, text FROM messages WHERE ts_us >= ? AND is_deleted = 0", (since_us,)
    ).fetchall()
    by = {}
    for user, text in rows:
        by.setdefault(user, []).append(text.strip())
    notes = {}
    for user, texts in sorted(by.items()):
        card = cards.get(user)
        if card is None or len(texts) < 4:
            continue
        share = lambda flags: sum(flags) / len(flags)  # noqa: E731
        measured = {
            "short_share": share([len(t.split()) <= SHORT for t in texts]),
            "long_share": share([len(t.split()) > LONG for t in texts]),
            "question_share": share(["?" in t for t in texts]),
            "lowercase_share": share([t[:1].islower() for t in texts]),
        }
        far = [
            f"{k.removesuffix('_share')} {v:.2f} vs {card[k]:.2f}"
            for k, v in measured.items()
            if abs(v - card[k]) > 0.2
        ]
        median = statistics.median(len(t.split()) for t in texts)
        if not card["median_words"] / 1.6 <= median <= card["median_words"] * 1.6:
            far.append(f"median words {median:g} vs {card['median_words']:g}")
        if far:
            notes[user] = "; ".join(far)
    return notes


# ---------------------------------------------------------------------- fixes and tasks


def revise(world, message_id: int, text: str) -> dict:
    """A fix to one message's text where it stands: same author, time, thread and statements. Every rule runs again.
    Run inside `World.trial()`."""
    row = world.db.execute("SELECT * FROM messages WHERE id = ?", (message_id,)).fetchone()
    if row is None:
        raise ValueError(f"no message {message_id}")
    if row["ts_us"] < frozen_before(world):
        raise ValueError(f"m{message_id} is in an earlier chapter, written and frozen: it is not revised")
    facts = {r["id"]: dict(r) for r in world.db.execute("SELECT * FROM facts")}
    conveyed = [
        r[0] for r in world.db.execute("SELECT fact_id FROM evidence WHERE message_id = ?", (message_id,))
    ]
    line = PostLine(author_id=row["user_id"], text=text, conveys=conveyed)
    errors = text_errors(world, [line], set(conveyed), set(moments(world)))
    if new := [f for f in stated_anchors(facts, text) if f not in conveyed]:
        errors.append(f"a revision states nothing new: it contains the anchors of {new}")
    if errors:
        raise ValueError("; ".join(errors))
    zones, habits, home = writing(world)
    moment, who = moments(world), row["user_id"]
    rendered = linked(
        world,
        PLACEHOLDER.sub(
            lambda m: render(moment[m[1]], zones[who], row["ts_us"], habits.get(who), home), text
        ),
        row["ts_us"],
    )
    with world.batch():
        world.db.execute("UPDATE messages SET text = ? WHERE id = ?", (rendered, message_id))
        world.db.execute("DELETE FROM message_mentions WHERE message_id = ?", (message_id,))
        world.insert(
            "message_mentions",
            [dict(message_id=message_id, user_id=u) for u in dict.fromkeys(MENTION.findall(rendered))],
        )
    return {"message": message_id, "text": rendered}


def edit(world, message_id: int, text: str) -> dict:
    """A person's edit of their own message of today, at the present: as a revision (same statements, no task's answer
    changed), and Slack marks it edited. What it said before is gone, as in Slack."""
    row = world.db.execute(
        "SELECT m.ts_us, m.subtype, u.is_bot FROM messages m JOIN users u ON u.id = m.user_id WHERE m.id = ?",
        (message_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f"no message {message_id}")
    day, now = today(world), present(world)
    if row["is_bot"] or row["subtype"] or day is None or not bounds(world, day)[0] <= row["ts_us"] <= now:
        raise ValueError("an edit is of a person's own message, posted earlier today")
    out = revise(world, message_id, text)
    world.db.execute("UPDATE messages SET edited_us = ? WHERE id = ?", (now, message_id))
    return out | {"edited": clock(world, now)}


def answer_of(world, task_id: str) -> list:
    """What a task answers: its gold rows' answers, or, for a refusal, the facts it rests on."""
    row = world.db.execute("SELECT answer_type, gold_json FROM tasks WHERE id = ?", (task_id,)).fetchone()
    if row["answer_type"] == "refusal":
        return sorted(
            f for (f,) in world.db.execute("SELECT fact_id FROM task_facts WHERE task_id = ?", (task_id,))
        )
    return sorted(str(r.get("answer")) for r in json.loads(row["gold_json"]))


# ---------------------------------------------------------------------- the forge: tasks on a frozen world


def forge_task(
    world, task: Task, settings, open_ids: list[str], gates: bool = True, rework: dict | None = None
) -> dict:
    """One candidate task on a finished, frozen world, for one of the round's open ids. Its category and level are the
    proposer's own, and its tries re-level it; it is checked as every task is (T1-T7, readable facts, the columns the
    tools show), with no planned level's needs: the forge measures difficulty instead of planning it. Without `gates`
    (a finished run's own task, measured as it was written), the forge's own gates (the shortcut panel, the near-miss
    queries) are left out. Returns its gold rows and code's measures."""
    if task.id not in open_ids:
        raise ValueError(f"a candidate's id is one of this round's open ids: {open_ids}")
    if gates and task.intent is None:
        raise ValueError(f"{task.id}: name its intent, the request it is: {sorted(INTENTS)}")
    was = None
    if task.rewords is not None:
        if (was := (rework or {}).get(task.rewords)) is None:
            raise ValueError(
                f"rewords names a task the last round sent back for another reading: {sorted(rework or {})}"
            )
        if task.actor_id != was["actor"]:
            raise ValueError(f"{task.id} rewords {task.rewords}: the same asker, {was['actor']}")
    spec = settings.taxonomy.get(task.category)
    concept = (
        spec.levels[task.level - 1] if spec and 1 <= task.level <= len(spec.levels) else "the proposer's own"
    )
    slot = Slot(
        id=task.id, category=task.category, level=task.level, concept=concept, style="the proposer's own"
    )
    world.db.execute("DELETE FROM task_facts WHERE task_id = ?", (task.id,))
    world.db.execute("DELETE FROM tasks WHERE id = ?", (task.id,))
    result = {"task": task.id} | record_task(world, task, settings, slot)
    if was is not None and (now := answer_of(world, task.id)) != was["answer"]:
        raise ValueError(
            f"{task.id} rewords {task.rewords} and keeps its gold answer: that one answers {was['answer']}, this one "
            f"{now}. Change the words so they rule the other reading out ({was['reading']}), not the answer"
        )
    lazy = panel(world, task.id, result["gold"])
    if gates and task.level > 1 and not task.facts and task.answer_type != "refusal" and not task.near_sql:
        raise ValueError(
            f"{task.id}: a task above level 1 with no facts names in near_sql its gold query with one condition "
            "relaxed (a day, a channel, a reaction or a thread left out), which answers otherwise: the answer a hasty "
            "reader gives"
        )
    if gates and task.level > 1 and lazy["shortcut"]:
        raise ValueError(
            f"{task.id}: a search for the question's own words reaches its answer first ({lazy['guesses']['top']!r}), "
            "so it is a level-1 task: the first search a solver makes answers it. Put a near-miss where that search "
            "reaches first (ask with the words its stale or wrong value uses, or where the correction is out of that "
            "search's way), or write it as level 1"
        )
    return result | {"panel": lazy, "features": features(world, task.id, result["gold"])}


class Annotation(StrictModel):
    """A fact the finished world already states, registered on the messages that state it: the ledger grows from the
    chatter while the workspace stays as it is. Its channel, author and day are its first message's."""

    id: SafeId
    storyline: SafeId
    subject: NonEmptyText
    attribute: NonEmptyText
    value: NonEmptyText
    anchor: NonEmptyText  # words every one of its messages contains
    message_ids: list[int] = Field(min_length=1)
    supersedes: SafeId | None = None
    decoy: bool = False
    summary: NonEmptyText


def annotate(world, fact: Annotation) -> dict:
    """Register a fact the world states (the forge's world_annotate): checked by the world's rules on insert (each
    message contains the anchor, a supersedes link points forward in time, two values of one subject and attribute
    are a change or a decoy)."""
    if world.db.execute("SELECT 1 FROM facts WHERE id = ?", (fact.id,)).fetchone():
        raise ValueError(f"fact {fact.id} exists: an annotation registers a new fact")
    if (
        fact.supersedes
        and not world.db.execute("SELECT 1 FROM facts WHERE id = ?", (fact.supersedes,)).fetchone()
    ):
        raise ValueError(f"fact {fact.supersedes}, which it supersedes, does not exist")
    marks = ", ".join("?" * len(fact.message_ids))
    stated = world.db.execute(
        f"SELECT id, channel_id, user_id, ts_us FROM messages WHERE id IN ({marks}) AND is_deleted = 0 ORDER BY ts_us",
        fact.message_ids,
    ).fetchall()
    if len(stated) != len(set(fact.message_ids)):
        raise ValueError(f"messages {sorted(set(fact.message_ids) - {r['id'] for r in stated})} do not exist")
    first = stated[0]
    (day,) = world.db.execute(
        "SELECT day FROM calendar WHERE ? >= start_us AND ? < end_us", (first["ts_us"], first["ts_us"])
    ).fetchone()
    with world.batch():
        world.insert(
            "facts",
            [
                dict(
                    id=fact.id,
                    storyline=fact.storyline,
                    subject=fact.subject,
                    attribute=fact.attribute,
                    value=fact.value,
                    anchor=fact.anchor,
                    channel_id=first["channel_id"],
                    author_id=first["user_id"],
                    day=day,
                    is_decoy=int(fact.decoy),
                    summary=fact.summary,
                )
            ],
        )
        world.insert(
            "evidence",
            [
                dict(fact_id=fact.id, message_id=r["id"], role="anchor", anchor_token=fact.anchor)
                for r in stated
            ],
        )
        if fact.supersedes:
            world.insert(
                "fact_relations", [dict(src_fact=fact.id, dst_fact=fact.supersedes, kind="supersedes")]
            )
    return {
        "fact": fact.id,
        "channel_id": first["channel_id"],
        "author_id": first["user_id"],
        "day": day,
        "messages": len(stated),
    }
