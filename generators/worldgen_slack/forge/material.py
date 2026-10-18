"""The proposer's map of a finished world's task material, by code: each value group (a subject and attribute) with
every value it took, in order, the messages that state it, where they sit, and which value is current; who can read
each private channel; and the people who could ask. What world_add_task needs (an answer's messages, an asker who can
read them) is on it, so the proposer reads the world to walk a task's route, not to survey it.

The world's profile (world/profile.py, `worldgen-slack profile`) is the same world measured for a person: counts, for
the one who writes a chapter."""

import math
from collections import defaultdict

from worldgen_slack.slack import PAGE

from ..world.board import answer_facts
from ..world.profile import counts, promises


def places(world) -> dict[str, str]:
    """Each channel as a reader names it: #name, #name (private), or a DM by the people in it."""
    names = dict(world.db.execute("SELECT id, real_name FROM users").fetchall())
    members = defaultdict(list)
    for channel, user in world.db.execute(
        "SELECT channel_id, user_id FROM members WHERE left_us IS NULL ORDER BY joined_us, user_id"
    ):
        members[channel].append(names[user])
    out = {}
    for channel, name, kind in world.db.execute("SELECT id, name, type FROM channels"):
        out[channel] = {
            "public": f"#{name}",
            "private": f"#{name} (private)",
            "im": "DM " + ", ".join(members[channel]),
            "mpim": "group DM " + ", ".join(members[channel]),
        }[kind]
    return out


def pages(world) -> dict[int, int]:
    """How deep each message sits in its channel as read_channel pages it, newest first: a top-level message's page;
    a thread reply, its parent's."""
    page, rank = {}, defaultdict(int)
    for message, channel in world.db.execute(
        "SELECT id, channel_id FROM messages WHERE parent_id IS NULL ORDER BY ts_us DESC, id DESC"
    ):
        rank[channel] += 1
        page[message] = math.ceil(rank[channel] / PAGE)
    for message, parent in world.db.execute("SELECT id, parent_id FROM messages WHERE parent_id IS NOT NULL"):
        page[message] = page.get(parent, 1)
    return page


def promise_list(world, where: dict[str, str]) -> list[dict]:
    """Every promise: its owner, what it promised, the message that made it and where, its due day, how it went (kept,
    kept late, changed, dropped, slipped: past due with nothing said; open), and the message that settled it."""
    names = dict(world.db.execute("SELECT id, real_name FROM users").fetchall())
    days = [(d, end) for d, end in world.db.execute("SELECT day, end_us FROM calendar ORDER BY day")]
    out = []
    for c in world.db.execute(
        """SELECT c.*, m.channel_id, m.ts_us AS made_us, k.ts_us AS closed_us FROM commitments c
        JOIN messages m ON m.id = c.message_id LEFT JOIN messages k ON k.id = c.closed_by ORDER BY c.due_us, c.id"""
    ):
        outcome = {"kept": "kept late" if c["closed_us"] and c["closed_us"] > c["due_us"] else "kept"}.get(
            c["status"], c["status"]
        )
        if outcome == "open":
            outcome = "slipped" if c["due_us"] <= max(e for _, e in days) else "open"
        out.append(
            {
                "promise": c["id"],
                "owner": f"{names[c['owner_id']]} ({c['owner_id']})",
                "text": c["text"],
                "made": f"m{c['message_id']}",
                "where": where[c["channel_id"]],
                "due_day": next(d for d, end in days if c["due_us"] <= end),
                "outcome": outcome,
            }
            | ({"settled_by": f"m{c['closed_by']}"} if c["closed_by"] else {})
        )
    return out


def material(world) -> dict:
    """What the world holds for tasks: a summary (its material counted, by storyline, and how its promises went); the
    value groups, most values first, each value with its fact id, day, place, how many read_channel pages deep its
    first message sits, and its messages (m123; a thread reply names its parent), and, where so, that it is a decoy,
    what it supersedes, that it is current; the collections, sets of items each stated where it came up; every
    promise and how it went; who can read each private channel (a public one, everyone; a DM, the people it names);
    and the people who could ask: name (id), title, clock."""
    where, deep = places(world), pages(world)
    messages, first = defaultdict(list), {}
    for fact, message, parent in world.db.execute(
        """SELECT e.fact_id, m.id, m.parent_id FROM evidence e JOIN messages m ON m.id = e.message_id
        ORDER BY e.fact_id, e.role = 'supporting', m.id"""
    ):
        messages[fact].append(f"m{message}" + (f" (reply to m{parent})" if parent else ""))
        first.setdefault(fact, message)
    supersedes = dict(
        world.db.execute("SELECT src_fact, dst_fact FROM fact_relations WHERE kind = 'supersedes'").fetchall()
    )
    by = defaultdict(list)
    for row in world.db.execute("SELECT * FROM facts ORDER BY day, id"):
        by[(row["subject"].strip().lower(), row["attribute"].strip().lower())].append(dict(row))
    groups = []
    for facts in by.values():
        current = set(answer_facts(world, [f["id"] for f in facts]))
        collection = next((f["collection"] for f in facts if f.get("collection")), None)
        groups.append(
            {
                "subject": facts[0]["subject"],
                "attribute": facts[0]["attribute"],
                "storyline": facts[0]["storyline"],
            }
            | ({"collection": collection} if collection else {})
            | {
                "values": [
                    {
                        "fact": f["id"],
                        "value": f["value"],
                        "day": f["day"],
                        "where": where[f["channel_id"]],
                        "page": deep.get(first.get(f["id"]), 1),
                        "messages": messages[f["id"]],
                    }
                    # only what is so: a decoy, what it supersedes, the current value
                    | ({"decoy": True} if f["is_decoy"] else {})
                    | ({"supersedes": supersedes[f["id"]]} if f["id"] in supersedes else {})
                    | ({"current": True} if f["id"] in current else {})
                    for f in facts
                ],
            }
        )
    readers = defaultdict(list)
    for channel, name in world.db.execute(
        """SELECT c.id, u.real_name FROM channels c JOIN members m ON m.channel_id = c.id AND m.left_us IS NULL
        JOIN users u ON u.id = m.user_id WHERE c.type = 'private' ORDER BY c.id, u.real_name"""
    ):
        readers[where[channel]].append(name)
    people = [
        f"{name} ({u}), {title or 'no title'}, {tz}"
        for u, name, title, tz in world.db.execute(
            "SELECT id, real_name, title, tz FROM users WHERE is_bot = 0 AND is_deleted = 0 ORDER BY real_name"
        )
    ]
    sets = defaultdict(list)  # a collection's items: each a group, its current value and every message on it
    for g in groups:
        if "collection" in g:
            sets[g["collection"]].append(g)
    found = counts(world)
    return {
        "summary": {k: v for k, v in found.items() if k != "by_storyline"}
        | {"by_storyline": found["by_storyline"], "promises": promises(world)},
        "groups": sorted(groups, key=lambda g: (-len(g["values"]), g["subject"], g["attribute"])),
        "collections": [
            {
                "collection": name,
                "storyline": items[0]["storyline"],
                "items": [
                    {
                        "item": g["subject"],
                        "attribute": g["attribute"],
                        "current": [v["value"] for v in g["values"] if v.get("current")],
                        "messages": [m for v in g["values"] for m in v["messages"]],
                    }
                    for g in items
                ],
            }
            for name, items in sorted(sets.items())
        ],
        "promises": promise_list(world, where),
        "private_readers": dict(readers),
        "people": people,
    }
