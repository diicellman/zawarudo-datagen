"""The memory code renders from the world before every turn of an agent in a VM: /task/memory's now.md (the
present, today's quota, the storylines, events and promises, a chapter's brief), ledger.json (the ledger as world_plan
takes it) and a page per person, channel, storyline and event; the pages world_now and world_view show; and the
ledger a reviewer reads. Every time in them is on the company clock."""

import json
import re
from datetime import datetime
from zoneinfo import ZoneInfo

from ...config import Needs
from ..board import Slot, needs_text, on_board, slot_needs
from ..clock import bounds, chapter_start, clock, frozen_before, part_of, present, today, window, world_meta
from ..style import LONG, SHORT
from ..writing import drift, posted, quotas, today_line


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
    day, now, first = today(world), present(world), chapter_start(world)
    out = []
    span = (
        f" (this chapter writes days {first}-{total}; days 1-{first - 1} are written and frozen)"
        if first > 1
        else ""
    )
    if day is None:
        out.append(f"# Now: the calendar is closed; its {total} days ended {clock(world, now)}.")
    else:
        out.append(f"# Now: {clock(world, now)}, day {day} of {total}{span}, {part_of(world, now)}")
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
    if context.get("slots"):
        out += [
            "",
            "## Brief: what this chapter plants, each entry on the board (a later forge writes its tasks)",
        ]
    board = {}
    for slot, fact in world.db.execute("SELECT slot, fact_id FROM board ORDER BY fact_id"):
        board.setdefault(slot, []).append(f"[[{fact}]]")
    for slot in map(Slot.model_validate, context.get("slots", [])):
        spec, level = settings.taxonomy[slot.category], slot.level
        needs = needs_text(slot_needs(settings, slot) or Needs())
        out.append(
            f"- {slot.id}: {slot.category} level {level}: {spec.levels[level - 1]}; concept: {slot.concept}; "
            f"style: {slot.style}"
            + (f"; needs: {needs}" if needs else "")
            + (
                f"; the board rests it on {', '.join(board[slot.id])}"
                if slot.id in board
                else "; for the board"
                if on_board(settings, slot)
                else ""
            )
        )
    if notes := drift(world, context.get("cards", {}), frozen_before(world)):
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
    now = int(world.db.execute("SELECT value FROM world_meta WHERE key = 'now_us'").fetchone()[0])
    channels = [
        r[0]
        for r in world.db.execute(
            "SELECT channel_id FROM members WHERE user_id = ? AND joined_us <= ? AND (left_us IS NULL OR left_us > ?)",
            (user_id, now, now),
        )
    ]
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
    now = int(world.db.execute("SELECT value FROM world_meta WHERE key = 'now_us'").fetchone()[0])
    members = [
        r[0]
        for r in world.db.execute(
            "SELECT user_id FROM members WHERE channel_id = ? AND joined_us <= ? AND (left_us IS NULL OR left_us > ?)",
            (channel_id, now, now),
        )
    ]
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
    out += [
        fact_line(world, dict(f), who)
        for f in world.db.execute("SELECT * FROM facts WHERE storyline = ? ORDER BY day, id", (storyline,))
    ]
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
    out += [
        fact_line(world, dict(f), who)
        for f in world.db.execute("SELECT * FROM facts WHERE event_id = ?", (event,))
    ]
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


def task_page(task: dict) -> str:
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
        return task_page(dict(task)) + "\n"
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


def ledger_document(world, slots=None) -> dict:
    """The ledger as world_plan takes it, so the author can edit and plan it again: its board holds the entries of
    `slots` (an earlier chapter's entries stay as they are, apart from the plan)."""
    company = world_meta(world, "zone")
    days = {d: n for n, d in world.db.execute("SELECT day, date FROM calendar")}

    def moment(us: int, zone: str) -> dict:
        local = datetime.fromtimestamp(us / 1e6, ZoneInfo(zone))
        return {"day": days[local.date().isoformat()], "time": f"{local:%H:%M}"} | (
            {"zone": zone} if zone != company else {}
        )

    relations = world.db.execute("SELECT src_fact, dst_fact, kind FROM fact_relations").fetchall()
    board = world.db.execute("SELECT slot, fact_id FROM board ORDER BY slot, fact_id").fetchall()
    if slots is not None:
        board = [(t, f) for t, f in board if t in slots]
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
        "board": [
            {"slot": s, "facts": [f for t, f in board if t == s]} for s in dict.fromkeys(t for t, _ in board)
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


def files(world, settings, context: dict) -> dict[str, str]:
    """The memory code writes to /task/memory before a turn."""
    out = {
        "memory/now.md": now_page(world, settings, context),
        "memory/ledger.json": json.dumps(
            ledger_document(
                world, {x["id"] for x in context.get("slots", [])} if context.get("chapter") else None
            ),
            ensure_ascii=False,
            indent=1,
        ),
    }
    for user in people(world).values():
        out[f"memory/people/{user['handle']}.md"] = person_page(world, user["id"], context)
    for (channel,) in world.db.execute("SELECT id FROM channels ORDER BY id"):
        out[f"memory/channels/{slug(world, channel)}.md"] = channel_page(world, channel, context)
    for (storyline,) in world.db.execute("SELECT id FROM storylines"):
        out[f"memory/storylines/{storyline}.md"] = storyline_page(world, storyline)
    for (event,) in world.db.execute("SELECT id FROM events"):
        out[f"memory/events/{event}.md"] = event_page(world, event)
    return out
