"""The world author: the memory code renders from the world before every turn, the author's tools over the world
file, its task and its turns. The author writes the world in time order through these tools; code is the only
writer, owns every time, and checks every write."""

import asyncio
import functools
import inspect
import json
import random
import re
import time
import typing
from datetime import datetime
from pathlib import Path
from typing import Literal, Self
from zoneinfo import ZoneInfo

import verifiers.v1 as vf
from pydantic import JsonValue, TypeAdapter, ValidationError
from verifiers.v1.utils.decorators import discover_decorated
from worldgen_slack.db import World, digest
from worldgen_slack.tools import WorldTaskData, file_hash, watch_parent

from ..chronicle import (
    Conversation,
    Plan,
    add_task,
    advance,
    bounds,
    drift,
    part_of,
    post,
    posted,
    present,
    quotas,
    record_plan,
    revise,
    today,
    today_line,
)
from ..config import Config
from ..contracts import (
    LONG,
    SHORT,
    Gaps,
    Part,
    Slot,
    Task,
    cards,
    channel_id,
    clock,
    selves,
    window,
    world_meta,
)

GOLD_SQL = """- gold_sql is one SELECT over world.sqlite as the task's actor sees it: channels, members, messages,
  message_mentions, reactions and thread_stats hold only what the actor can read; users, calendar, storylines, facts,
  fact_relations and evidence are whole. local(us) writes a time on the actor's clock, local(us, zone) on another.
  It returns the answer in a column named answer, and may name the evidence in message_id and user_id columns.
- What the gold query reads follows the category's gold in `taxonomy`: a sql query reads only the workspace, a
  ledger query reads the facts it answers from, a hybrid query reads both.
- The gold query finds messages by what the question names (words, people, channels, threads, reactions, times),
  never by a message id or by id order: code checks that it answers the same with the messages renumbered.
- answer_type: text or number is one row; set is 1 to max_answer_rows rows; refusal is no rows as the actor.
- question is what actor_id asks, in `language`; it does not contain its answer.
"""

AUTHOR_GUIDE = (
    """You write one company's Slack workspace, as its people live it, in time order: first the company and its
people, then the ledger of what will be stated, then each day's conversations, then tasks on the finished world.
The workspace is a SQLite file that only code writes. You change it with the world_* tools: async functions in your
IPython (`r = await world_post(conversation={...})`) that return JSON text and raise with the reason when a write
breaks a rule; fix and call again. They are live: world_now(), world_view(ref), world_sql(sql, actor_id=None) and
world_read(actor_id, tool, arguments) read the world as it is now.

Memory and notes
- /task/memory/ is code's, rendered from the world before every turn: now.md (the present, today's quota, the
  conversations code drew for today, each storyline's facts, events, open promises, task slots, style drift),
  ledger.json (the ledger as world_plan takes it), and a page per person, channel, storyline and event. Pages name
  each other as [[id]]; grep them. Every time in them is on the company clock, as a weekday and a date.
- /task/notes/ is yours: plan.md (the story's arcs day by day, and a board of what each task slot will rest on) and
  recap.md (what happened today, what is open). Besides the world, they are all you keep from one turn to the next.

Time
- The world has a present. world_post writes a conversation at the present and moves it to its last line;
  world_advance(to=part) moves it to a later part of the day: early 06-09, morning 09-12, afternoon 12-17, evening
  17-21, night 21-24. From the night, world_advance(to="tomorrow") closes the day, once the day's checks pass, and
  moves to the next morning. Nothing is ever written before the present, so write the day in the order it happens.
- A message never contains a clock time or a date. It writes {at:id} for an event, or for a fact about one; code
  renders it on the author's clock. Relative words (tomorrow, Friday, this morning) are fine when true at the present.

The ledger (world_plan)
- storylines: exactly the number now.md asks for, the company's workstreams. Fixed once planned; summaries may change.
- events: the story's moments: a day of the calendar, a time and, if not the company's, a clock (IANA zone). An event
  never moves once planned: a change of plan is a new event, and a fact that supersedes the old one.
- facts: what someone states in Slack: subject, attribute and value (no time in a value); its author, its channel and
  the day it is first stated; event and kind (happened or scheduled) when it is about an event, and it then carries
  the event's time; after and supersedes, facts it follows or replaces; decoy, a value that is wrong; summary, what it
  means and on whose authority. anchor: distinctive words of the value. Every line that contains a fact's anchor
  states that fact, so each fact has words of its own: no two facts share an anchor, and a fact whose anchor contains
  another's comes after it. A value said again is the same fact conveyed again, not a new fact. A fact about an event
  with no anchor is stated with its time, {at:...}.
- A stated fact never changes; an unstated one may be re-planned for today or later.

Conversations (world_post)
- One channel; lines by its members, each in its author's way (their typing card is on their page). reply_to is the
  index of an earlier line whose thread the line joins; thread continues an earlier thread of the channel.
  reactions come from other members. pause is "hours" when an hour or more passes before the line.
- conveys lists the facts a line states. A fact's first statement is by its author, in its channel, on its day; a
  happened fact is stated only after it happens, a scheduled one before it happens; after and supersedes keep their
  order. Anyone may repeat a stated
  fact later, listing it in conveys.
- commits opens a promise with the day it is due; closes ends one: kept, changed or dropped. Every promise ends by
  its due day.
- Each day holds about the number of messages now.md gives, spread by the parts' rhythm, and its own shares of thread
  replies, reacted messages and DMs. A weekend day is quiet, unless your first plan puts an event on it: a day with
  an event gets a workday's messages; now.md's today line shows them, and what closing the day still takes. A
  conversation or a plan after which the day could no longer close is refused. now.md lists the everyday
  conversations code drew for the day from the channels' routines; the storylines happen among them. Each person
  writes like their typing card; now.md reports who drifts from it.

Voice
- This is Slack, not email and not a status report. People type quickly: fragments, contractions (I'll, can't,
  it's), dropped subjects ("on it", "looks good"), lowercase starts, the odd typo, a question when they don't know
  something, a one-word reply when that is all it takes.
- Each person sounds like themselves. Their page says who they are and how they type: how long they write, how often
  they ask, start lowercase or use emoji. A director and a new engineer, a careful analyst and a chatty support lead,
  do not sound alike.
- Address people as <@USERID> when a line is aimed at them; a reaction often replaces a "thanks" line; emoji in text
  like :tada: or 🙏 where the person would use them.
- Work talk is concrete and uneven: names of systems, versions, ticket numbers, links, half-formed ideas, mild
  complaints, jokes, side topics, disagreement. Not every line moves the story; not every line is an update.
- A line that states a fact contains the fact's anchor words; the rest of the line is still in its author's voice.

Tasks (after the last day, world_add_task)
- One task for each slot of now.md, with the slot's id: a question an actor asks, as hard as its level and concept
  say, in its style. Nothing can be posted after the last day, so plan from day 1 what each slot will rest on: facts in several
  channels, decoys, buried or split evidence, private conversations.
"""
    + GOLD_SQL
    + """- world_add_task checks the task and returns its gold rows and code's measures (read_channel pages deep, tables
  read, search rank). A solver then tries each task; reword a task, or rest it on other evidence, to fit its level.

Reviews
- An independent judge reviews the world after some days and at the end. Its issues come back to you: message_ids
  are the messages to change (world_revise rewrites one where it stands), evidence_message_ids show the defect.
"""
)


# ---------------------------------------------------------------------- memory: rendered from the world


def people(world) -> dict[str, dict]:
    return {r["id"]: dict(r) for r in world.db.execute("SELECT * FROM users ORDER BY real_name")}


def label(world, channel_id: str) -> str:
    row = world.db.execute("SELECT name FROM channels WHERE id = ?", (channel_id,)).fetchone()
    if row["name"]:
        return f"#{row['name']}"
    members = world.db.execute(
        "SELECT u.real_name FROM members m JOIN users u ON u.id = m.user_id WHERE m.channel_id = ? ORDER BY u.real_name",
        (channel_id,),
    )
    return "DM " + ", ".join(r[0] for r in members)


def slug(world, channel_id: str) -> str:
    row = world.db.execute("SELECT name FROM channels WHERE id = ?", (channel_id,)).fetchone()
    if row["name"]:
        return row["name"]
    handles = world.db.execute(
        "SELECT u.handle FROM members m JOIN users u ON u.id = m.user_id WHERE m.channel_id = ? ORDER BY u.handle",
        (channel_id,),
    )
    return "dm-" + "-".join(r[0] for r in handles)


def card(typing: dict) -> str:
    """A typing card in words: the shares as the typing profiles measure them, short and long by SHORT and LONG."""
    return (
        f"usually about {typing['median_words']:g} words; {typing['short_share']:.0%} of their messages are short "
        f"({SHORT} words or fewer) and {typing['long_share']:.0%} long (more than {LONG}); "
        f"{typing['question_share']:.0%} ask something; {typing['lowercase_share']:.0%} start lowercase; "
        f"{typing['emoji_share']:.0%} carry an emoji"
    )


def line_of(world, row, who: dict) -> str:
    thread = f" (in the thread of [[m{row['parent_id']}]])" if row["parent_id"] else ""
    return f"- {clock(world, row['ts_us'])} [[m{row['id']}]] {who[row['user_id']]['real_name']}{thread}: {row['text']}"


def fact_line(world, fact: dict, who: dict) -> str:
    first = world.db.execute(
        """SELECT m.id, m.ts_us FROM evidence e JOIN messages m ON m.id = e.message_id
        WHERE e.fact_id = ? AND e.role = 'anchor'""",
        (fact["id"],),
    ).fetchone()
    where = f"in {label(world, fact['channel_id'])} by {who[fact['author_id']]['real_name']}"
    if first:
        status = f"stated {clock(world, first['ts_us'])} {where} ([[m{first['id']}]])"
    else:
        status = f"planned for day {fact['day']} ({clock(world, bounds(world, fact['day'])[0])[:14]}) {where}"
    about = (
        f"; {fact['moment_kind']} [[{fact['event_id']}]] {clock(world, fact['moment_us'])}"
        if fact["event_id"]
        else ""
    )
    extra = (" (decoy)" if fact["is_decoy"] else "") + (
        f", anchor {fact['anchor']!r}" if fact["anchor"] else ""
    )
    return (
        f"- [[{fact['id']}]] {fact['subject']}: {fact['attribute']} = {fact['value']}{extra}{about}; {status}"
    )


def now_page(world, settings, context: dict) -> str:
    """The one-pager: everything the author needs at the present, on one page."""
    who, total = people(world), world.db.execute("SELECT COUNT(*) FROM calendar").fetchone()[0]
    day, now = today(world), present(world)
    out = []
    if day is None:
        out.append(
            f"# Now: the calendar is closed; its {total} days ended {clock(world, now)}. Tasks are written now."
        )
    else:
        out.append(f"# Now: {clock(world, now)}, day {day} of {total}, {part_of(world, now)}")
        start, end = bounds(world, day)
        quota = quotas(world, settings)[day]
        parts = ", ".join(
            f"{p} {posted(world, *window(world, day, p))} of {n}" for p, n in quota["parts"].items()
        )
        out += [
            "",
            "## Today",
            f"- messages: {posted(world, start, end)} posted of about {quota['messages']} ({parts})",
            f"- today: {today_line(world, settings, day)}",
        ]
        for f in world.db.execute(
            """SELECT * FROM facts f WHERE day = ? AND NOT EXISTS (SELECT 1 FROM evidence e WHERE e.fact_id = f.id
            AND e.role = 'anchor') ORDER BY id""",
            (day,),
        ):
            out.append(f"- to state today: {fact_line(world, dict(f), who)[2:]}")
        for c in world.db.execute("SELECT * FROM commitments WHERE status = 'open' AND due_us <= ?", (end,)):
            out.append(f"- due by today's end: [[{c['id']}]] {who[c['owner_id']]['real_name']}: {c['text']}")
        drawn = context.get("agenda", {}).get(str(day), [])
        if drawn:
            out.append("- everyday conversations code drew for today:")
            out += [
                f"  - {label(world, a['channel_id'])}, {a['part']}: {a['situation']} "
                f"({', '.join(who[p]['real_name'] for p in a['participants'])}; about {a['length']} lines)"
                for a in drawn
            ]
    for s in world.db.execute("SELECT * FROM storylines ORDER BY position"):
        out += ["", f"## Storyline [[{s['id']}]]: {s['summary']}"]
        out += [
            fact_line(world, dict(f), who)
            for f in world.db.execute("SELECT * FROM facts WHERE storyline = ? ORDER BY day, id", (s["id"],))
        ]
    events = world.db.execute("SELECT * FROM events ORDER BY moment_us").fetchall()
    if events:
        out += ["", "## Events"]
        out += [
            f"- [[{e['id']}]] {e['title']}: {clock(world, e['moment_us'])}"
            + (" (past)" if e["moment_us"] <= now else "")
            for e in events
        ]
    promises = world.db.execute("SELECT * FROM commitments WHERE status = 'open' ORDER BY due_us").fetchall()
    if promises:
        out += ["", "## Open promises"]
        out += [
            f"- [[{c['id']}]] {who[c['owner_id']]['real_name']}: {c['text']}; due by {clock(world, c['due_us'] - 1)[:14]} "
            f"(made in [[m{c['message_id']}]])"
            for c in promises
        ]
    out += ["", "## Task slots (written after the last day; a task's id is its slot's)"]
    for slot in context.get("slots", []):
        spec, level = settings.taxonomy[slot["category"]], slot["level"]
        spread = spec.spread[level - 1] if spec.spread else 1
        out.append(
            f"- {slot['id']}: {slot['category']} level {level}: {spec.levels[level - 1]}; concept: {slot['concept']}; "
            f"style: {slot['style']}"
            + (f"; its facts first stated in at least {spread} channels" if spread > 1 else "")
            + ("; written in this turn" if slot["id"] in context.get("writable", []) else "")
        )
    if notes := drift(world, context.get("cards", {})):
        out += ["", "## Style drift (people far from their typing card)"]
        out += [f"- {who[user]['real_name']} [[{user}]]: {note}" for user, note in notes.items()]
    return "\n".join(out) + "\n"


def frontmatter(id: str, kind: str, aliases: list[str]) -> str:
    return f"---\nid: {id}\ntype: {kind}\naliases: [{', '.join(a for a in aliases if a)}]\n---\n"


def person_page(world, user_id: str, context: dict) -> str:
    who = people(world)
    me = who[user_id]
    team = json.loads(me["profile_json"]).get("Team", "")
    out = [frontmatter(user_id, "person", [me["handle"], me["display_name"], me["real_name"]])]
    out.append(f"# {me['real_name']}: {me['title']}, {team} ({me['tz']})")
    if typing := context.get("cards", {}).get(user_id):
        out.append(f"- how they type: {card(typing)}")
    channels = [r[0] for r in world.db.execute("SELECT channel_id FROM members WHERE user_id = ? AND left_us IS NULL", (user_id,))]  # fmt: skip
    out.append("- conversations: " + ", ".join(f"{label(world, c)} [[{c}]]" for c in channels))
    lines = world.db.execute(
        "SELECT * FROM messages WHERE user_id = ? ORDER BY ts_us DESC LIMIT 8", (user_id,)
    ).fetchall()[::-1]
    if lines:
        out += ["", "## Recent lines"] + [line_of(world, r, who) for r in lines]
    promises = world.db.execute(
        "SELECT * FROM commitments WHERE owner_id = ? AND status = 'open'", (user_id,)
    )
    out += [
        f"- open promise [[{c['id']}]]: {c['text']}; due by {clock(world, c['due_us'] - 1)[:14]}"
        for c in promises
    ]
    if self := context.get("selves", {}).get(user_id):
        out += [
            "",
            "## Who they are",
            f"- {self['age']}, from {self['home']}; {self['education']}",
            f"- as a person (any job named here was before this one): {self['persona']}",
            f"- background: {self['background']}",
            f"- outside work: {', '.join(self['hobbies'])}",
        ]
    return "\n".join(out) + "\n"


def channel_page(world, channel_id: str, context: dict) -> str:
    who = people(world)
    channel = world.db.execute("SELECT * FROM channels WHERE id = ?", (channel_id,)).fetchone()
    out = [frontmatter(channel_id, channel["type"], [label(world, channel_id)])]
    out.append(f"# {label(world, channel_id)} ({channel['type']})")
    if channel["topic"] or channel["purpose"]:
        out.append(f"- topic: {channel['topic']}; purpose: {channel['purpose']}")
    members = [r[0] for r in world.db.execute("SELECT user_id FROM members WHERE channel_id = ? AND left_us IS NULL", (channel_id,))]  # fmt: skip
    out.append("- members: " + ", ".join(f"{who[m]['real_name']} [[{m}]]" for m in members))
    if routines := context.get("routines", {}).get(channel_id):
        out.append(
            "- everyday conversations: " + ", ".join(f"{r['kind']} ({r['probability']})" for r in routines)
        )
    lines = world.db.execute(
        "SELECT * FROM messages WHERE channel_id = ? ORDER BY ts_us DESC LIMIT 30", (channel_id,)
    ).fetchall()[::-1]
    if lines:
        out += ["", "## Latest messages"] + [line_of(world, r, who) for r in lines]
    return "\n".join(out) + "\n"


def storyline_page(world, storyline: str) -> str:
    who = people(world)
    row = world.db.execute("SELECT * FROM storylines WHERE id = ?", (storyline,)).fetchone()
    out = [frontmatter(storyline, "storyline", []), f"# {row['summary']}"]
    out += [fact_line(world, dict(f), who) for f in world.db.execute("SELECT * FROM facts WHERE storyline = ? ORDER BY day, id", (storyline,))]  # fmt: skip
    out += [
        f"- event [[{e['id']}]] {e['title']}: {clock(world, e['moment_us'])}"
        for e in world.db.execute("SELECT * FROM events WHERE storyline = ? ORDER BY moment_us", (storyline,))
    ]
    return "\n".join(out) + "\n"


def event_page(world, event: str) -> str:
    who = people(world)
    row = world.db.execute("SELECT * FROM events WHERE id = ?", (event,)).fetchone()
    out = [frontmatter(event, "event", [row["title"]]), f"# {row['title']}: {clock(world, row['moment_us'])}"]
    out.append(f"- storyline [[{row['storyline']}]]")
    out += [fact_line(world, dict(f), who) for f in world.db.execute("SELECT * FROM facts WHERE event_id = ?", (event,))]  # fmt: skip
    return "\n".join(out) + "\n"


def message_page(world, message_id: int) -> str:
    who = people(world)
    row = world.db.execute("SELECT * FROM messages WHERE id = ?", (message_id,)).fetchone()
    if row is None:
        raise ValueError(f"no message m{message_id}")
    root = row["parent_id"] or row["id"]
    thread = world.db.execute(
        "SELECT * FROM messages WHERE id = ? OR parent_id = ? ORDER BY ts_us", (root, root)
    )
    out = [f"# [[m{message_id}]] in {label(world, row['channel_id'])}"]
    out += [line_of(world, r, who) for r in thread]
    out += [
        f"- states [[{f}]] ({role})"
        for f, role in world.db.execute(
            "SELECT fact_id, role FROM evidence WHERE message_id = ?", (message_id,)
        )
    ]
    return "\n".join(out) + "\n"


def task_page(world, task: dict) -> str:
    out = [
        f"- [[{task['id']}]] {task['category']} level {task['level']} ({task['concept']}), asked by [[{task['actor_id']}]]: "
        f"{task['question']}",
        f"  gold {task['gold_json']}; right answers {task['right_rate']}, right and grounded {task['strict_rate']}, "
        f"fewest calls {task['min_calls']}",
    ]
    return "\n".join(out)


def view(world, ref: str, context: dict) -> str:
    """One page by what it is called: a person (id, handle or name), a channel (id or #name), a storyline, an event, a
    fact, a task, a conversation or a message (m123)."""
    ref = ref.strip()
    if re.fullmatch(r"m\d+", ref):
        return message_page(world, int(ref[1:]))
    key = ref.lstrip("#@").casefold()
    for row in world.db.execute("SELECT id, handle, display_name, real_name FROM users"):
        if key in {row["id"].casefold(), row["handle"], row["real_name"].casefold()}:
            return person_page(world, row["id"], context)
    for row in world.db.execute("SELECT id, name FROM channels"):
        if key in {row["id"].casefold(), (row["name"] or "").casefold()}:
            return channel_page(world, row["id"], context)
    if world.db.execute("SELECT 1 FROM storylines WHERE id = ?", (ref,)).fetchone():
        return storyline_page(world, ref)
    if world.db.execute("SELECT 1 FROM events WHERE id = ?", (ref,)).fetchone():
        return event_page(world, ref)
    if fact := world.db.execute("SELECT * FROM facts WHERE id = ?", (ref,)).fetchone():
        return fact_line(world, dict(fact), people(world)) + "\n"
    if task := world.db.execute("SELECT * FROM tasks WHERE id = ?", (ref,)).fetchone():
        return task_page(world, dict(task)) + "\n"
    if world.db.execute("SELECT 1 FROM scenes WHERE id = ?", (ref,)).fetchone():
        who = people(world)
        rows = world.db.execute(
            "SELECT m.* FROM scene_messages sm JOIN messages m ON m.id = sm.message_id WHERE sm.scene_id = ? ORDER BY m.ts_us",
            (ref,),
        )
        return f"# conversation {ref}\n" + "\n".join(line_of(world, r, who) for r in rows) + "\n"
    raise ValueError(
        f"nothing is called {ref!r}: name a person (id, handle or name), a channel (id or #name), a storyline, an "
        "event, a fact, a task, a conversation (dNN-NNN) or a message (m123)"
    )


def ledger_document(world) -> dict:
    """The ledger as world_plan takes it, so the author can edit and plan it again."""
    company = world_meta(world, "zone")
    days = {d: n for n, d in world.db.execute("SELECT day, date FROM calendar")}

    def moment(us: int, zone: str) -> dict:
        local = datetime.fromtimestamp(us / 1e6, ZoneInfo(zone))
        return {"day": days[local.date().isoformat()], "time": f"{local:%H:%M}"} | (
            {"zone": zone} if zone != company else {}
        )

    relations = world.db.execute("SELECT src_fact, dst_fact, kind FROM fact_relations").fetchall()
    return {
        "storylines": [
            {"id": s, "summary": t}
            for s, t in world.db.execute("SELECT id, summary FROM storylines ORDER BY position")
        ],
        "events": [
            {"id": e["id"], "storyline": e["storyline"], "title": e["title"]}
            | moment(e["moment_us"], e["zone"])
            for e in world.db.execute("SELECT * FROM events ORDER BY moment_us")
        ],
        "facts": [
            {
                k: f[k]
                for k in (
                    "id",
                    "storyline",
                    "subject",
                    "attribute",
                    "value",
                    "anchor",
                    "channel_id",
                    "author_id",
                    "day",
                )
            }
            | {
                "after": [d for s, d, k in relations if s == f["id"] and k == "after"],
                "supersedes": next((d for s, d, k in relations if s == f["id"] and k == "supersedes"), None),
                "event": f["event_id"],
                "kind": f["moment_kind"] if f["event_id"] else None,
                "decoy": bool(f["is_decoy"]),
                "summary": f["summary"],
            }
            for f in world.db.execute("SELECT * FROM facts ORDER BY day, id")
        ],
    }


def ledger_digest(world) -> dict:
    """Events and facts with every moment rendered on the company clock, for a reviewer."""
    who = people(world)
    return {
        "events": [
            f"{e['id']}: {e['title']}, {clock(world, e['moment_us'])}"
            for e in world.db.execute("SELECT * FROM events ORDER BY moment_us")
        ],
        "facts": [
            fact_line(world, dict(f), who)[2:]
            for f in world.db.execute("SELECT * FROM facts ORDER BY day, id")
        ],
    }


def files(world, settings, context: dict, mode: str) -> dict[str, str]:
    """The memory code writes to /task/memory before a turn."""
    out = {
        "memory/now.md": now_page(world, settings, context),
        "memory/ledger.json": json.dumps(ledger_document(world), ensure_ascii=False, indent=1),
    }
    for user in people(world).values():
        out[f"memory/people/{user['handle']}.md"] = person_page(world, user["id"], context)
    for (channel,) in world.db.execute("SELECT id FROM channels ORDER BY id"):
        out[f"memory/channels/{slug(world, channel)}.md"] = channel_page(world, channel, context)
    for (storyline,) in world.db.execute("SELECT id FROM storylines"):
        out[f"memory/storylines/{storyline}.md"] = storyline_page(world, storyline)
    for (event,) in world.db.execute("SELECT id FROM events"):
        out[f"memory/events/{event}.md"] = event_page(world, event)
    if mode == "tasks":
        tasks = [task_page(world, dict(t)) for t in world.db.execute("SELECT * FROM tasks ORDER BY id")]
        out["memory/tasks.md"] = "# Tasks\n" + "\n".join(tasks) + "\n"
    return out


# ---------------------------------------------------------------------- the author's tools


class AuthoringToolsConfig(vf.ToolsetConfig):
    mode: Literal["plan", "day", "tasks"] = "plan"
    day: int = 0
    db_path: str = ""
    db_hash: str = ""
    context: str = (
        "{}"  # settings, task slots, typing cards, the day's drawn conversations, routines, the call log
    )


TYPES = {"string": "str", "integer": "int", "number": "float", "boolean": "bool", "object": "object"}


def shape(schema: dict, defs: dict) -> str:
    """One JSON schema, compact: a document's fields with ? when optional, a list's bounds, a choice's values."""
    if "$ref" in schema:
        return shape(defs[schema["$ref"].rsplit("/", 1)[1]], defs)
    if "anyOf" in schema:
        return " | ".join(shape(s, defs) for s in schema["anyOf"] if s.get("type") != "null")
    if "enum" in schema:
        return "|".join(json.dumps(v) for v in schema["enum"])
    if schema.get("type") == "array":
        lo, hi = schema.get("minItems"), schema.get("maxItems")
        bound = f"{lo or 0}-{hi} × " if hi else (f"{lo}+ × " if lo else "")
        return f"[{bound}{shape(schema.get('items', {}), defs)}]"
    if "properties" in schema:
        required = set(schema.get("required", []))
        return "{" + ", ".join(f"{k}{'' if k in required else '?'}: {shape(v, defs)}" for k, v in schema["properties"].items()) + "}"  # fmt: skip
    return TYPES.get(schema.get("type"), "any")


def arguments(fn) -> str:
    """A tool's arguments as the author passes them, rendered from the types that check the call: the author's
    help() shows only a tool's description."""
    hints, out = typing.get_type_hints(fn), []
    for name, param in inspect.signature(fn).parameters.items():
        schema = TypeAdapter(hints[name]).json_schema()
        optional = "?" if param.default is not inspect.Parameter.empty else ""
        out.append(f"{name}{optional}: {shape(schema, schema.get('$defs', {}))}")
    return ", ".join(out) or "none"


class WorldTools(vf.Toolset[AuthoringToolsConfig, vf.State]):
    """The author's world tools. They keep no state of their own: with the base state, a call makes no state round
    trip through the tunnel. world-calls.jsonl logs every call, its arguments and its outcome."""

    TOOL_PREFIX = "world"

    async def setup(self) -> None:
        self._lock = asyncio.Lock()
        if file_hash(self.config.db_path) != self.config.db_hash:
            raise ValueError("world file hash mismatch")
        self.context = json.loads(self.config.context)
        self.settings = Config.model_validate(self.context["settings"])
        self.slots = [Slot.model_validate(s) for s in self.context.get("slots", []) if s["id"] in self.context.get("writable", [])]  # fmt: skip
        self.gaps = Gaps(self.settings.personas.gaps)
        self.posted = None  # the last post: its conversation, the present it left, and its result

    def _with_state(self, fn):
        synced = super()._with_state(fn)

        @functools.wraps(synced)
        async def serialized(*args, **kwargs):
            # One call at a time: each write is applied to a trial of the world file and replaces it whole.
            async with self._lock:
                return await synced(*args, **kwargs)

        return serialized

    def register(self, mcp) -> None:
        """Each tool's description ends with its arguments, so the author reads what a call takes before making one.
        Arguments the tools' schemas refuse never reach a tool: log them too, with a short reason."""
        for fn in discover_decorated(self, "tool"):
            doc = (fn.__doc__ or "").strip()
            name = getattr(fn, "tool_name", None) or fn.__name__
            mcp.add_tool(self._with_state(fn), name=name, description=f"{doc}\n\nArguments: {arguments(fn)}")
        call = mcp.call_tool

        async def call_tool(name, arguments, context=None):
            try:
                return await call(name, arguments, context)
            except Exception as error:
                if not isinstance(error.__cause__, ValidationError):
                    raise
                reason = "; ".join(
                    f"{'.'.join(map(str, e['loc']))}: {e['msg']}"
                    for e in error.__cause__.errors(include_url=False)
                )
                self._log(name, arguments, f"invalid arguments: {reason}")
                raise ValueError(f"{name}: invalid arguments: {reason}") from None

        mcp.call_tool = call_tool

    def _log(self, tool: str, arguments: dict, error: str | None) -> None:
        if log := self.context.get("log"):
            entry = {"time": time.time(), "mode": self.config.mode, "day": self.config.day, "tool": tool}
            with open(log, "a") as stream:
                stream.write(
                    json.dumps(entry | {"ok": error is None, "error": error, "args": arguments}) + "\n"
                )

    def _logged(self, tool: str, arguments: dict, act, *modes: str):
        """Run one call, refused when it is not this turn's, and log it with its arguments and outcome."""
        error = None
        try:
            if modes:
                self._turn(tool, *modes)
            return act()
        except (ValueError, LookupError) as caught:
            error = str(caught)
            raise ValueError(error) from None
        except Exception as caught:  # a fault, not a refusal: logged as failed, and raised as it is
            error = f"{type(caught).__name__}: {caught}"
            raise
        finally:
            self._log(tool, arguments, error)

    def _world(self, writable: bool = False) -> World:
        return World(self.config.db_path, writable=writable)

    def _turn(self, tool: str, *modes: str) -> None:
        if self.config.mode not in modes:
            raise ValueError(
                f"{tool} is for the {' or '.join(modes)} turn; this is the {self.config.mode} turn"
            )
        if (
            self.config.mode == "day"
            and tool in ("post", "advance")
            and today(self._world()) != self.config.day
        ):
            raise ValueError(
                f"day {self.config.day} is closed: write recap.md, update plan.md and end your turn"
            )

    def _write(self, act):
        world = self._world(writable=True)
        try:
            with world.trial() as copy:
                return act(copy)
        finally:
            world.close()

    @vf.tool
    async def now(self) -> str:
        """The one-pager: the present, today's quota and drawn conversations, each storyline's facts, events, open
        promises, task slots and style drift."""
        return self._logged("now", {}, lambda: now_page(self._world(), self.settings, self.context))

    @vf.tool
    async def view(self, ref: str) -> str:
        """One page: a person (id, handle or name), a channel (id or #name), a storyline, an event, a fact, a task, a
        conversation (dNN-NNN) or a message (m123), with its latest messages and links."""
        return self._logged("view", {"ref": ref}, lambda: view(self._world(), ref, self.context))

    @vf.tool
    async def sql(self, sql: str, actor_id: str | None = None) -> dict:
        """One read-only SELECT over the whole world file, or, with actor_id, over the world as that person sees it
        (the way gold queries run). At most 200 rows."""

        def act():
            world = self._world()
            return world.gold(actor_id, sql, max_rows=200) if actor_id else world.select(sql, max_rows=200)

        return self._logged("sql", {"sql": sql, "actor_id": actor_id}, act)

    @vf.tool
    async def read(self, actor_id: str, tool: str, arguments: dict[str, JsonValue]) -> dict:
        """A Slack read tool as any person: search_messages, search_users, search_channels, list_user_channels,
        read_channel, read_thread, get_user, list_channel_members or get_reactions."""
        return self._logged(
            "read",
            {"actor_id": actor_id, "tool": tool, "arguments": arguments},
            lambda: World(self.config.db_path, actor=actor_id).call(tool, arguments),
        )

    @vf.tool
    async def plan(self, ledger: Plan) -> str:
        """Plan the ledger, whole: storylines, events and facts. Planned events and stated facts stay as they are; the
        rest may change. Returns what the ledger now holds."""

        def act():
            self._write(lambda copy: record_plan(copy, ledger, self.settings))
            world = self._world()
            counts = [
                world.db.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                for t in ("storylines", "events", "facts")
            ]
            return f"the ledger holds {counts[0]} storylines, {counts[1]} events and {counts[2]} facts"

        return self._logged("plan", {"ledger": ledger.model_dump(mode="json")}, act, "plan", "day")

    @vf.tool
    async def post(self, conversation: Conversation) -> dict:
        """Post one conversation at the present: its lines, timed by code after the present, the facts they state,
        their reactions and promises. Returns the messages as written and the new present."""

        document = conversation.model_dump(mode="json")

        def act():
            now = present(self._world())
            if self.posted and self.posted[:2] == (document, now):
                # The same conversation again, and nothing written since: a retry after a lost response, not a
                # second conversation.
                return self.posted[2]
            seed = digest([self.settings.seed, "post", now, document])
            result = self._write(
                lambda copy: post(copy, conversation, self.settings, random.Random(seed), self.gaps)
            )
            self.posted = (document, present(self._world()), result)
            return result

        return self._logged("post", {"conversation": document}, act, "day")

    @vf.tool
    async def advance(self, to: Part | Literal["tomorrow"]) -> dict:
        """Move the present to the part `to` of today; from the night, "tomorrow" closes the day and moves to the
        next morning. The present's own part leaves it where it is. Reports what keeps a day open, and the day's style
        drift once it closes."""

        def act():
            day = today(self._world())
            moved = self._write(lambda copy: advance(copy, self.settings, to))
            if moved["closed_day"]:
                world = self._world()
                who, notes = people(world), drift(world, self.context.get("cards", {}), bounds(world, day)[0])
                moved["drift"] = {who[user]["real_name"]: note for user, note in notes.items()}
            return moved

        return self._logged("advance", {"to": to}, act, "day")

    @vf.tool
    async def revise(self, message_id: int, text: str) -> dict:
        """Rewrite one message's text where it stands: same author, time, thread and statements; no task's gold
        answer may change."""
        return self._logged(
            "revise",
            {"message_id": message_id, "text": text},
            lambda: self._write(lambda copy: revise(copy, message_id, text)),
            "day",
            "tasks",
        )

    @vf.tool
    async def add_task(self, task: Task) -> dict:
        """Write the task of one slot, with the slot's id (it replaces the slot's task, if any): checked, then its gold
        rows and code's measures returned."""
        return self._logged(
            "add_task",
            {"task": task.model_dump(mode="json")},
            lambda: self._write(lambda copy: add_task(copy, task, self.settings, self.slots)),
            "tasks",
        )


# ---------------------------------------------------------------------- the author's task and turns


class WorldAuthorConfig(vf.TaskConfig):
    tools: AuthoringToolsConfig


class WorldAuthorTask(vf.Task[WorldTaskData, vf.State, WorldAuthorConfig]):
    """One block of the author's work in its VM (the plan, a day, the tasks): the env opens it with the turn's prompt,
    and the author works through the world tools."""

    NEEDS_CONTAINER = True

    @classmethod
    def toolsets(cls, config: WorldAuthorConfig) -> list[vf.Toolset]:
        return [WorldTools(config.tools)]

    @classmethod
    def create(cls, mode: str, day: int, world: Path, context: dict, attempt: str) -> Self:
        tools = AuthoringToolsConfig(
            mode=mode, day=day, db_path=str(world), db_hash=file_hash(world), context=json.dumps(context)
        )
        if tools.colocated or tools.runtime.type != "subprocess" or tools.url is not None:
            raise ValueError("the world file requires a host-side tool server")
        data = WorldTaskData(
            prompt=None,
            system_prompt=AUTHOR_GUIDE,
            attempt=attempt,
            world_hash=tools.db_hash,
            network_allow=[],
            network_block=["*"],
        )
        return cls(data, WorldAuthorConfig(tools=tools))


def rejected(feedback: str, restored: str = "") -> str:
    """Why the last attempt at this block was rejected, and what was put back."""
    return f"\nYour last attempt at this was rejected{restored}: {feedback}" if feedback else ""


def plan_prompt(world, settings, feedback: str = "") -> str:
    count = settings.storylines
    return (
        f"Before day 1 ({clock(world, bounds(world, 1)[0])[:14]}): plan the ledger with world_plan: exactly {count} "
        "storylines, their events and their facts, over the calendar in now.md. Then write /task/notes/plan.md: each "
        "storyline's arc day by day, and a board of what each task slot in now.md will rest on. End your turn when the "
        f"ledger is planned.{rejected(feedback)}"
    )


def day_prompt(world, day: int, issues: list | None = None, feedback: str = "") -> str:
    total = world.db.execute("SELECT COUNT(*) FROM calendar").fetchone()[0]
    review = (
        "\nThe judge reviewed the world so far. Fix what you can in what you write next (world_revise rewrites a "
        f"message where it stands): {json.dumps(issues, ensure_ascii=False)}"
        if issues
        else ""
    )
    return (
        f"It is {clock(world, present(world))}, day {day} of {total}.{review}"
        f"{rejected(feedback, ', and the world and your notes are back at the start of the day')}\nRead world_now() and your notes, and "
        "write today: post its conversations in the order they happen with world_post, and move on with "
        'world_advance(to=part); from the night, to="tomorrow" closes the day once today\'s checks pass. Then write '
        "/task/notes/recap.md "
        "(what happened today, what is open), update plan.md, and end your turn."
    )


def tasks_prompt(settings, ids: list[str]) -> str:
    return (
        f"The last day is closed: nothing more is posted. Write the tasks of these slots of now.md: {', '.join(ids)}; "
        "one each, with world_add_task and the slot's id. It returns the task's gold rows and code's measures. The "
        "other slots are written in turns of their own. After this turn a solver tries each "
        f"task {settings.author.tries} times and the judge reviews it; a task whose share of right answers misses its "
        "level's band, or that the judge does not approve, comes back to you."
    )


def harden_prompt(back: dict) -> str:
    return (
        "These tasks come back to you, each with its slot's cell, the solver's tries (right_rate, the share of right "
        "answers, is how hard the task is; strict_rate also needs every claim grounded; solves, each try's calls and the "
        "step at which the gold evidence first appeared: the route to beat; the whole tries are in "
        "/task/memory/solves/<task>/), the judge's review (level_fit "
        "from 0 to 4: how fully answering it needs its level and concept), code's measures (evidence pages, tables "
        "read, search rank), its level's band of right-answer rates, and what it needs (move): harder, easier, or the "
        f"fix the judge asks for: {json.dumps(back, ensure_ascii=False)}\nRewrite each in its slot so that answering it "
        "needs its level and concept with no easier route, and lands in its band: reword it, or rest it on other "
        "evidence; world_revise can remove a giveaway from a message. A task you change is tried and reviewed again; "
        "one you keep stands as it is, and its round still counts (rounds_left). End your turn when the tasks stand."
    )


def fix_prompt(issues: list) -> str:
    return (
        f"The review rejected the world: {json.dumps(issues, ensure_ascii=False)}\nFix each issue with world_revise or "
        "world_add_task: message_ids are the messages to change, evidence_message_ids show the defect. If you judge an "
        "issue mistaken, write why in /task/notes/plan.md. End your turn when done."
    )


def context_of(settings, state, store_root: Path, organization: dict | None, writable=()) -> dict:
    """What the author's tools are configured with: the settings, the world's task slots and those this turn writes,
    each person's typing card
    and seeded self, the conversations code drew per day, the channels' routines, and the call log."""
    routines = {}
    if organization:
        for c in organization["channels"]:
            if c.get("routines"):
                routines[channel_id(c["type"], c.get("name"), c["members"])] = c["routines"]
    agenda = {
        day: [
            {k: s[k] for k in ("channel_id", "part", "situation", "participants", "length")} for s in scenes
        ]
        for day, scenes in (state.plans.get("agenda") or {}).items()
    }
    return {
        "settings": json.loads(settings.model_dump_json()),
        "slots": [s.model_dump(mode="json") for s in state.quota],
        "writable": list(writable),
        "cards": cards(state.cast),
        "selves": selves(state.cast),
        "agenda": agenda,
        "routines": routines,
        "log": str(store_root / "world-calls.jsonl"),
    }


if __name__ == "__main__":
    watch_parent()  # the server process ends with the generator that started it
    WorldTools.run()
