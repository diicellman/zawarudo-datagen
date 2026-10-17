"""The proposer's map of a finished world's task material, by code: each value group (a subject and attribute) with
every value it took, in order, the messages that state it, where they sit, and which value is current; who can read
each private channel; and the people who could ask. What world_add_task needs (an answer's messages, an asker who can
read them) is on it, so the proposer reads the world to walk a task's route, not to survey it.

The world's profile (world/profile.py, `worldgen-slack profile`) is the same world measured for a person: counts, for
the one who writes a chapter."""

from collections import defaultdict

from ..world.board import answer_facts


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


def material(world) -> dict:
    """The value groups, most values first, each value with its fact id, day, place and messages (m123; a thread reply
    names its parent), and, where so, that it is a decoy, what it supersedes, that it is current; who can read each
    private channel (a public one, everyone; a DM, the people it names); and the people who could ask: name (id),
    title, clock."""
    where = places(world)
    messages = defaultdict(list)
    for fact, message, parent in world.db.execute(
        """SELECT e.fact_id, m.id, m.parent_id FROM evidence e JOIN messages m ON m.id = e.message_id
        ORDER BY e.fact_id, e.role = 'supporting', m.id"""
    ):
        messages[fact].append(f"m{message}" + (f" (reply to m{parent})" if parent else ""))
    supersedes = dict(
        world.db.execute("SELECT src_fact, dst_fact FROM fact_relations WHERE kind = 'supersedes'").fetchall()
    )
    by = defaultdict(list)
    for row in world.db.execute("SELECT * FROM facts ORDER BY day, id"):
        by[(row["subject"].strip().lower(), row["attribute"].strip().lower())].append(dict(row))
    groups = []
    for facts in by.values():
        current = set(answer_facts(world, [f["id"] for f in facts]))
        groups.append(
            {
                "subject": facts[0]["subject"],
                "attribute": facts[0]["attribute"],
                "storyline": facts[0]["storyline"],
                "values": [
                    {
                        "fact": f["id"],
                        "value": f["value"],
                        "day": f["day"],
                        "where": where[f["channel_id"]],
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
    return {
        "groups": sorted(groups, key=lambda g: (-len(g["values"]), g["subject"], g["attribute"])),
        "private_readers": dict(readers),
        "people": people,
    }
