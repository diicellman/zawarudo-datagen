"""The world written in time order: one author plans the ledger and its events, posts each conversation at the
world's present, and moves the present forward. Code checks every step, owns every time, and never writes into the
past. Pure functions over `World`; the author's tools (`agents/world.py`) call them inside `World.trial()`."""

import json
import math
import statistics
from datetime import date, datetime
from typing import Literal, Self
from zoneinfo import ZoneInfo

from pydantic import Field, model_validator
from worldgen_slack.dataset import NonEmptyText, SafeId, StrictModel

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
    fact_measures,
    insert_lines,
    normalized,
    record_task,
    render,
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


class Plan(StrictModel):
    storylines: list[Storyline] = Field(min_length=1)
    events: list[Event] = Field(default_factory=list)
    facts: list[PlanFact] = Field(min_length=1)
    board: list[BoardEntry] = Field(default_factory=list)


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


def start_clock(world) -> None:
    """From now on the world only grows forward: its present moves to day 1 at the start of the early part."""
    world.db.execute(
        "UPDATE world_meta SET value = ? WHERE key = 'now_us'", (str(window(world, 1, "early")[0]),)
    )
    world.insert("world_meta", [{"key": "chronological", "value": "1"}])


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


def daily(dates: list[str], settings, busy=()) -> list[dict]:
    """Code's share of the world per calendar date and part: `[activity].messages` over the dates, a weekend day
    weighing `[author].weekend` of a workday unless it is `busy` (a day number with a planned event), and each day
    split by the parts' rhythm."""
    weights = [
        1.0 if day in busy or date.fromisoformat(d).weekday() < 5 else settings.author.weekend
        for day, d in enumerate(dates, 1)
    ]
    rhythm = [settings.activity.parts.get(p, 0.0) for p in PARTS]
    return [
        {"messages": n, "parts": dict(zip(PARTS, apportion(n, rhythm)))}
        for n in apportion(settings.activity.messages, weights)
    ]


def quotas(world, settings) -> dict[int, dict]:
    """Each day's quota. A day the first plan put an event on weighs as a workday, so a storyline's busiest night is
    not a quiet day's handful of messages; the days are fixed by that plan, so a closed day's quota never changes."""
    days = world.db.execute("SELECT day, date FROM calendar ORDER BY day").fetchall()
    busy = world.db.execute("SELECT value FROM world_meta WHERE key = 'busy_days'").fetchone()
    shares = daily([d for _, d in days], settings, json.loads(busy[0]) if busy else ())
    return {day: quota for (day, _), quota in zip(days, shares)}


# A day's shares, each checked on that day's own messages at its close, and named as the author reads them.
SHARES = {"reply_share": "thread replies", "reaction_rate": "reacted to", "dm_share": "in DMs"}


def band(target: float, slack: float, n: int) -> tuple[int, int]:
    """How many of n messages may have a property whose share is target ± slack, rounded outward so that every count
    of messages has a range."""
    return max(0, math.floor(round((target - slack) * n, 9))), min(
        n, math.ceil(round((target + slack) * n, 9))
    )


def tally(world, day: int) -> dict[str, int]:
    """Today's messages, and how many are thread replies, carry a reaction, sit in DMs."""
    row = world.db.execute(
        """SELECT COUNT(*), COALESCE(SUM(m.parent_id IS NOT NULL), 0),
        COALESCE(SUM(EXISTS (SELECT 1 FROM reactions r WHERE r.message_id = m.id)), 0),
        COALESCE(SUM(c.type IN ('im', 'mpim')), 0)
        FROM messages m JOIN channels c ON c.id = m.channel_id WHERE m.ts_us >= ? AND m.ts_us < ?""",
        bounds(world, day),
    ).fetchone()
    return dict(zip(("messages", *SHARES), row))


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
            low, high = band(getattr(settings.activity, share), settings.author.share_tolerance, n)
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
        return f"{have['messages']} messages today ({shares}); today can no longer close within its quota and shares; {left}"  # fmt: skip
    lo, hi = ways["closes"]
    needs = ", ".join(f"{a}-{b} {SHARES[s]}" for s, (a, b) in ways["more"].items())
    return (
        f"{have['messages']} messages today ({shares}); it closes with {lo}-{hi} messages. At {ways['at']}: "
        f"{ways['at'] - have['messages']} more lines, of them {needs}; {left}"
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
    return world.db.execute(
        "SELECT COUNT(*) FROM messages WHERE ts_us >= ? AND ts_us < ?", (start_us, end_us)
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


def record_plan(world, plan: Plan, settings, slots=()) -> None:
    """The ledger in time order. Storylines are fixed once planned (summaries may change); an event, once planned,
    never moves or goes; a stated fact keeps everything but its summary, while an unstated one may change or go and is
    planned for today or later. A fact about an event carries the event's moment. The board gives each ledger and
    hybrid slot of `slots` the planned facts it will rest on, as many channels, relations and decoys as its level
    needs: the material of a hard backward task exists before day 1, when it can still be posted."""
    count = settings.storylines
    if len(plan.storylines) != count:
        raise ValueError(f"plan exactly {count} storylines")
    ids = [x.id for x in (*plan.storylines, *plan.events, *plan.facts)]
    if repeated := sorted({i for i in ids if ids.count(i) > 1}):
        raise ValueError(f"storyline, event and fact ids are distinct; repeated {repeated}")
    kept = [r[0] for r in world.db.execute("SELECT id FROM storylines ORDER BY position")]
    if kept and kept != [s.id for s in plan.storylines]:
        raise ValueError(f"the storylines are planned: keep {kept} in this order (summaries may change)")
    check_facts(plan.facts, {s.id for s in plan.storylines})
    day, zone = today(world), world_meta(world, "zone")
    if day is None:
        raise ValueError("the calendar is closed: the ledger is final")
    events, moment = {e.id: e for e in plan.events}, {}
    old = {r["id"]: dict(r) for r in world.db.execute("SELECT * FROM events")}
    for e in plan.events:
        if e.storyline not in {s.id for s in plan.storylines}:
            raise ValueError(f"{e.id}: storyline {e.storyline} is not in storylines")
        if TIME_LITERAL.search(e.title):
            raise ValueError(f"{e.id}: a title has no time or date in it; day and time say when")
        moment[e.id] = at(world, Moment(day=e.day, time=e.time, zone=e.zone or zone))
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
    members = {
        (r[0], r[1])
        for r in world.db.execute("SELECT channel_id, user_id FROM members WHERE left_us IS NULL")
    }
    stated = {r[0] for r in world.db.execute("SELECT DISTINCT fact_id FROM evidence")}
    current = {r["id"]: dict(r) for r in world.db.execute("SELECT * FROM facts")}
    relations = {
        (r[0], r[1], r[2]) for r in world.db.execute("SELECT src_fact, dst_fact, kind FROM fact_relations")
    }
    planned = {f.id: f for f in plan.facts}
    if gone := sorted(stated - planned.keys()):
        raise ValueError(f"stated facts stay in the ledger: {gone}")
    backward = {s.id: s for s in slots if settings.taxonomy[s.category].gold in ("ledger", "hybrid")}
    board = {e.slot: list(dict.fromkeys(e.facts)) for e in plan.board}
    if len(board) < len(plan.board):
        raise ValueError("the board has one entry for each slot")
    if strays := sorted(board.keys() - backward.keys()):
        raise ValueError(f"the board is for the ledger and hybrid slots of now.md; {strays} are not")
    if missing := sorted(backward.keys() - board.keys()):
        raise ValueError(
            f"the board names the facts each ledger and hybrid slot will rest on; {missing} have none"
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
            continue
        if (f.channel_id, f.author_id) not in members:
            raise ValueError(f"{f.id}: its author {f.author_id} is a member of {f.channel_id}")
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
        world.db.execute("DELETE FROM board")  # the plan's board is whole, as its facts are
        if kept:
            for s in plan.storylines:
                world.db.execute("UPDATE storylines SET summary = ? WHERE id = ?", (s.summary, s.id))
        else:
            world.insert(
                "storylines", [s.model_dump() | {"position": i} for i, s in enumerate(plan.storylines, 1)]
            )
        world.insert(
            "events",
            [
                dict(
                    id=e.id, storyline=e.storyline, title=e.title, moment_us=moment[e.id], zone=e.zone or zone
                )
                for e in plan.events
                if e.id not in old
            ],
        )
        world.db.execute(  # the first plan's event days, kept from then on
            "INSERT OR IGNORE INTO world_meta (key, value) VALUES ('busy_days', ?)",
            (json.dumps(sorted({e.day for e in plan.events})),),
        )
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
        world.insert("board", [dict(slot=s, fact_id=f) for s, facts in board.items() for f in facts])
        for s, facts in board.items():
            spec, level = settings.taxonomy[backward[s].category], backward[s].level
            if spec.needs and (short := unmet(spec.needs[level - 1], fact_measures(world, facts))):
                raise ValueError(
                    f"the board's {s} (level-{level} {backward[s].category}) needs " + "; ".join(short)
                )
        if room(world, settings, day) is None:  # inside the batch: the refusal takes the plan back
            raise ValueError(f"this plan leaves today no way to close: {today_line(world, settings, day)}")


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
    if (ways := room(world, settings, day)) is not None and (need := ways["closes"][0] - tally(world, day)["messages"]) > 0:  # fmt: skip
        still = math.ceil(need / settings.activity.conversation_lines)
        wait = min(wait, max(bounds(world, day)[1] - now, 0) // (still + 1))
    return wait + rng.randrange(1_000_000, 60_000_000)


def stated_anchors(facts: dict[str, dict], text: str) -> list[str]:
    """The facts whose anchor the text contains."""
    said = f" {normalized(PLACEHOLDER.sub(' ', text))} "
    return [f for f, x in facts.items() if x["anchor"] and f" {normalized(x['anchor'])} " in said]


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
    if world.db.execute("SELECT 1 FROM channels WHERE id = ?", (channel,)).fetchone() is None:
        raise ValueError(f"no channel {channel}")
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
    for src, dst in world.db.execute("SELECT src_fact, dst_fact FROM fact_relations WHERE kind IN ('after', 'supersedes')"):  # fmt: skip
        follows.setdefault(src, {})[dst] = None  # a fact may both follow and supersede another
    first = set()
    for i, line in enumerate(lines):
        for f in line.conveys:
            if f in facts and f not in anchored and f not in first:
                x = facts[f]
                if (channel, line.author_id, day) != (x["channel_id"], x["author_id"], x["day"]):
                    errors.append(
                        f"line {i}: {f} is first stated by {x['author_id']} in {x['channel_id']} on day {x['day']}"
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
    if late := [f for f, moment_us in owed_moments(world, day) if moment_us <= stamps[-1] and not told.get(f, moment_us) < moment_us]:  # fmt: skip
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
    texts = {r[0]: r[1] for r in world.db.execute(f"SELECT id, text FROM messages WHERE id IN ({', '.join('?' * len(ids))})", ids)}  # fmt: skip
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
    have = tally(world, day)
    for share, name in SHARES.items():
        low, high = band(getattr(settings.activity, share), settings.author.share_tolerance, count)
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
    if late := [f for f, moment_us in owed_moments(world, day) if moment_us <= target]:
        raise ValueError(
            f"{late} are scheduled before {clock(world, target)} and still to be told: post them before moving on"
        )
    world.db.execute("UPDATE world_meta SET value = ? WHERE key = 'now_us'", (str(max(target, now)),))
    return {
        "now": clock(world, max(target, now)),
        "closed_day": day if closed else None,
        "today": today(world),
    }


def drift(world, cards: dict[str, dict], since_us: int = 0) -> dict[str, str]:
    """Each person's messages since `since_us` beside the typing card they were given: what is far from it, by person."""
    rows = world.db.execute("SELECT user_id, text FROM messages WHERE ts_us >= ? AND is_deleted = 0", (since_us,)).fetchall()  # fmt: skip
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
    """A fix to one message's text where it stands: same author, time, thread and statements. Every rule runs again,
    and no task's gold answer may change. Run inside `World.trial()`."""
    row = world.db.execute("SELECT * FROM messages WHERE id = ?", (message_id,)).fetchone()
    if row is None:
        raise ValueError(f"no message {message_id}")
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
    zone = world.db.execute("SELECT tz FROM users WHERE id = ?", (row["user_id"],)).fetchone()[0]
    moment = moments(world)
    rendered = PLACEHOLDER.sub(lambda m: render(moment[m[1]], zone, row["ts_us"]), text)
    gold = {t["id"]: json.loads(t["gold_json"]) for t in world.db.execute("SELECT id, gold_json FROM tasks")}
    with world.batch():
        world.db.execute("UPDATE messages SET text = ? WHERE id = ?", (rendered, message_id))
        world.db.execute("DELETE FROM message_mentions WHERE message_id = ?", (message_id,))
        world.insert(
            "message_mentions",
            [dict(message_id=message_id, user_id=u) for u in dict.fromkeys(MENTION.findall(rendered))],
        )
    for task in world.db.execute("SELECT id, actor_id, gold_sql FROM tasks").fetchall():
        if world.gold(task["actor_id"], task["gold_sql"], max_rows=50)["rows"] != gold[task["id"]]:
            raise ValueError(f"the revision changes task {task['id']}'s gold answer")
    return {"message": message_id, "text": rendered}


def add_task(world, task: Task, settings, slots: list) -> dict:
    """One task for one of the slots being written, on the finished world; it replaces the slot's task, if any. Its
    gold query is checked as every task is (T1-T7, readable facts, the level's needs). Returns its gold rows and
    code's measures of how hard it is."""
    if today(world) is not None:
        raise ValueError("tasks are written once the calendar is closed")
    slot = next((s for s in slots if s.id == task.id), None)
    if slot is None:
        raise ValueError(f"a task's id is one of the slots you write now: {[s.id for s in slots]}")
    world.db.execute("DELETE FROM task_facts WHERE task_id = ?", (task.id,))
    world.db.execute("DELETE FROM tasks WHERE id = ?", (task.id,))
    return {"task": task.id} | record_task(world, task, settings, slot)  # every planned fact is stated by now
