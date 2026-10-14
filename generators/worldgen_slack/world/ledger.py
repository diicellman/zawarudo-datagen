"""The ledger an author plans: storylines, events, facts, moves and artifacts, recorded with record_plan."""

import json
import re
from typing import Literal, Self

from pydantic import Field, model_validator
from worldgen_slack.release import NonEmptyText, SafeId, StrictModel
from worldgen_slack.world import digest

from ..config import Ids
from .board import board_measures, fact_measures, on_board, slot_needs, unmet
from .clock import (
    Clock,
    Moment,
    Zone,
    apportion,
    at,
    bounds,
    chapter_start,
    frozen_before,
    present,
    segment_of,
    today,
    window,
    world_meta,
)
from .integrations import app_id, event_errors, queue, queue_day, queue_events
from .text import TIME_LITERAL, normalized
from .writing import keep_busy, quotas, room, today_line


class Storyline(StrictModel):
    id: SafeId
    summary: NonEmptyText


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
    segment, its share of the segment's messages (an event day weighs as a workday). Unchecked, an author tells its
    storylines early and leaves its last days nothing to state. A third still open can always take more facts, today or later."""
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
