"""Writing a day: conversations posted at the present, their lines checked, the day's quotas and shares, the clock
moved on, and fixes (revise, edit)."""

import json
import math
import re
import statistics
from typing import Annotated, Literal

from pydantic import Field
from worldgen_slack.release import NonEmptyText, SafeId, StrictModel
from .board import hidden_answers
from .clock import (
    PARTS,
    bounds,
    clock,
    frozen_before,
    meta,
    moments,
    part_of,
    present,
    render,
    segments,
    spread,
    today,
    window,
    writing,
)
from .integrations import drain, queue_day
from .style import LONG, SHORT
from .text import DAY_WORD, GROUP, LINK, MENTION, PEOPLE, PLACEHOLDER, RAW_URL, TIME_LITERAL, normalized
from ..config import Ids


class Reaction(StrictModel):
    user_id: SafeId
    emoji: Annotated[str, Field(pattern=r"^[a-z0-9_+'-]{1,40}$")]


def linked(world, text: str, at_us: int) -> str:
    """A line's links and group mentions as Slack writes them; an artifact is linked once it exists."""

    def link(m):
        row = world.db.execute("SELECT url, created_us FROM artifacts WHERE id = ?", (m[1],)).fetchone()
        if row["created_us"] > at_us:
            raise ValueError(f"{m[1]} does not exist yet: it is made on a later day")
        return f"<{row['url']}|{m[1]}>"

    def group(m):
        row = world.db.execute("SELECT id FROM usergroups WHERE handle = ?", (m[1],)).fetchone()
        return f"<!subteam^{row[0]}|@{m[1]}>"

    return GROUP.sub(group, LINK.sub(link, text))


def text_errors(world, lines, conveyable: set[str], moments: set[str]) -> list[str]:
    """Each line's text rules: a reply points at an earlier line, conveys names only facts the conversation may
    state, a time or a date appears only as {at:id} of a moment, no line names a fact or event id, and mentions name
    people."""
    ids = [r[0] for r in world.db.execute("SELECT id FROM facts UNION SELECT id FROM events")]
    people = {r[0] for r in world.db.execute("SELECT id FROM users")}
    artifacts = {r[0] for r in world.db.execute("SELECT id FROM artifacts")}
    groups = {r[0] for r in world.db.execute("SELECT handle FROM usergroups")}
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
        if named_twice := [
            m.group(1) for m in PLACEHOLDER.finditer(line.text) if DAY_WORD.search(line.text[: m.start()])
        ]:
            errors.append(f"line {i}: {{at:...}} names its own day; drop the day word before {named_twice}")
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
        if lost := set(LINK.findall(line.text)) - artifacts:
            errors.append(f"line {i}: {{link:...}} names an artifact of the plan; not {sorted(lost)}")
        if lost := set(GROUP.findall(line.text)) - groups:
            errors.append(f"line {i}: {{group:...}} names a usergroup's handle; not {sorted(lost)}")
        if RAW_URL.search(LINK.sub(" ", line.text)):
            errors.append(f"line {i}: a link is {{link:id}} of an artifact; code writes its URL")
    return errors


def insert_lines(
    world, scene: dict, lines, stamps, moments, first, rng, gaps, thread=None, latest=None
) -> list[int]:
    """Insert one conversation at code's timestamps: its scenes row, each line's text with {at:id} rendered on its
    author's clock, replies in their thread (or every line in `thread`), mentions, reactions after their message
    (and before `latest`), and evidence: a fact in `first` is anchored by its author's first line conveying it."""
    facts = {r["id"]: dict(r) for r in world.db.execute("SELECT * FROM facts")}
    zones, habits, home = writing(world)
    world.insert("scenes", [scene])
    ids = [None] * len(lines)
    for i, (line, ts) in enumerate(zip(lines, stamps)):
        drain(world, ts)  # code's posts due before this line come first
        text = linked(
            world,
            PLACEHOLDER.sub(
                lambda m: render(moments[m[1]], zones[line.author_id], ts, habits.get(line.author_id), home),
                line.text,
            ),
            ts,
        )
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
                render(fact["moment_us"], zones[line.author_id], ts, habits.get(line.author_id), home)
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
