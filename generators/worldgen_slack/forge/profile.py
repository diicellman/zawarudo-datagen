"""A world's profile: what material it holds for tasks, read off the world file with no model call.
uv run --frozen worldgen-slack profile --world data/v7-05/software

The value groups (a subject and attribute, the values it took, how they supersede each other, its decoys, where its
current value is stated and who can read it), threads, recurring conversations, people, and what the world's tasks
and board already rest on. A chapter's brief is written from it, and what a chapter planted is measured by it."""

import json
from collections import Counter, defaultdict
from pathlib import Path

from ..workspace import Workspace, asked
from ..world.board import answer_facts, readable
from ..world.clock import meta


def chains(world) -> dict[str, int]:
    """Each fact's depth in its supersedes chain: 1 for a fact that supersedes nothing."""
    supersedes = dict(
        world.db.execute("SELECT src_fact, dst_fact FROM fact_relations WHERE kind = 'supersedes'").fetchall()
    )
    depth = {}

    def of(fact: str) -> int:
        if fact not in depth:
            depth[fact] = 1 + (of(supersedes[fact]) if fact in supersedes else 0)
        return depth[fact]

    for (fact,) in world.db.execute("SELECT id FROM facts"):
        of(fact)
    return depth


def version_of_day(world) -> dict[int, int]:
    """The world version each calendar day was written in: 1 for a world in one run; a chapter's days, its own."""
    segments = json.loads(meta(world, "segments", "null") or "null")
    days = [d for (d,) in world.db.execute("SELECT day FROM calendar")]
    if not segments:
        return dict.fromkeys(days, 1)
    return {d: next(i for i, s in enumerate(segments, 1) if s["first"] <= d <= s["last"]) for d in days}


def groups(world) -> list[dict]:
    """The value groups: facts on one subject and attribute (case and spacing aside), with their values, chain depth,
    decoys, channels and days, where the current value is stated (its channel's kind, a thread reply), how many
    people can read it, the version that last added to it, and the tasks and board entries resting on it."""
    depth, versions = chains(world), version_of_day(world)
    people = [u for (u,) in world.db.execute("SELECT id FROM users WHERE is_bot = 0")]
    by = defaultdict(list)
    for row in world.db.execute(
        "SELECT f.*, c.type, c.name FROM facts f JOIN channels c ON c.id = f.channel_id ORDER BY f.day, f.id"
    ):
        by[(row["subject"].strip().lower(), row["attribute"].strip().lower())].append(dict(row))
    tasks = Counter(f for (f,) in world.db.execute("SELECT fact_id FROM task_facts"))
    board = defaultdict(set)
    for slot, fact in world.db.execute("SELECT slot, fact_id FROM board"):
        board[fact].add(slot)
    out = []
    for (subject, attribute), facts in by.items():
        ids = [f["id"] for f in facts]
        current = answer_facts(world, ids)
        anchors = world.db.execute(
            f"""SELECT m.channel_id, m.parent_id, c.type FROM evidence e JOIN messages m ON m.id = e.message_id
            JOIN channels c ON c.id = m.channel_id WHERE e.role = 'anchor' AND e.fact_id IN ({", ".join("?" * len(current))})""",
            current,
        ).fetchall()
        readers = (
            sum(any(readable(world, u, a["channel_id"]) for a in anchors) for u in people) if anchors else 0
        )
        out.append(
            {
                "subject": facts[0]["subject"],
                "attribute": facts[0]["attribute"],
                "values": len({f["value"].strip().lower() for f in facts}),
                "facts": ids,
                "current": current,
                "decoys": [f["id"] for f in facts if f["is_decoy"]],
                "depth": max(depth[f] for f in ids),
                "channels": len({f["channel_id"] for f in facts}),
                "days": [facts[0]["day"], max(f["day"] for f in facts)],
                "current_where": sorted(
                    {("thread reply in " if a["parent_id"] else "") + a["type"] for a in anchors}
                ),
                "current_private": bool(anchors)
                and all(a["type"] in ("im", "mpim", "private") for a in anchors),
                "readers": [readers, len(people)],
                "version": max(versions[f["day"]] for f in facts),
                "tasks": sum(tasks[f] for f in ids),
                "board": sorted(set().union(*(board[f] for f in ids))),
            }
        )
    return sorted(out, key=lambda g: (-g["values"], -g["depth"], g["subject"], g["attribute"]))


def profile(world) -> dict:
    """The world's profile: its size, its value groups and what they add up to, its threads, recurring conversations
    and people, and what tasks and the board rest on."""

    def count(sql):
        return world.db.execute(sql).fetchone()[0]

    found = groups(world)
    replies = [
        n
        for (n,) in world.db.execute(
            "SELECT COUNT(*) FROM messages WHERE parent_id IS NOT NULL GROUP BY parent_id"
        )
    ]
    kinds = Counter(s for (s,) in world.db.execute("SELECT situation FROM scenes"))
    posts = dict(world.db.execute("SELECT user_id, COUNT(*) FROM messages GROUP BY user_id").fetchall())
    names = dict(world.db.execute("SELECT id, real_name FROM users").fetchall())
    used = {f for (f,) in world.db.execute("SELECT fact_id FROM task_facts")}
    decoys = [f for (f,) in world.db.execute("SELECT id FROM facts WHERE is_decoy = 1")]
    return {
        "world": {
            "company": meta(world, "company"),
            "version": int(meta(world, "version", 1)),
            "days": count("SELECT COUNT(*) FROM calendar"),
            "messages": count("SELECT COUNT(*) FROM messages"),
            "channels": dict(
                world.db.execute("SELECT type, COUNT(*) FROM channels GROUP BY type").fetchall()
            ),
            "people": count("SELECT COUNT(*) FROM users"),
            "facts": count("SELECT COUNT(*) FROM facts"),
            "tasks": count(f"SELECT COUNT(*) FROM tasks WHERE {asked(world.db)}"),
            "board_entries": count("SELECT COUNT(DISTINCT slot) FROM board"),
        },
        "material": {
            "groups": len(found),
            "groups_4_values": sum(g["values"] >= 4 for g in found),
            "groups_3_values": sum(g["values"] >= 3 for g in found),
            "longest_chain": max((g["depth"] for g in found), default=0),
            "private_truths": sum(g["current_private"] for g in found),
            "decoys": len(decoys),
            "decoys_unused": len([d for d in decoys if d not in used]),
            "groups_without_tasks": sum(not g["tasks"] for g in found),
        },
        "groups": found,
        "threads": {
            "threads": len(replies),
            "with_5_replies": sum(n >= 5 for n in replies),
            "most_replies": max(replies, default=0),
        },
        "recurring": [{"situation": s, "conversations": n} for s, n in kinds.most_common(15)],
        "people": {
            "busiest": [[names[u], n] for u, n in sorted(posts.items(), key=lambda x: -x[1])[:5]],
            "quiet": sorted(names[u] for u in names if posts.get(u, 0) < 5),
        },
    }


def page(found: dict) -> str:
    """The profile as a page to read."""
    w, m = found["world"], found["material"]
    out = [
        f"# Profile: {w['company']}, version {w['version']}",
        "",
        f"{w['days']} days, {w['messages']} messages, {w['people']} people, channels {w['channels']}; {w['facts']} facts, "
        f"{w['tasks']} tasks, {w['board_entries']} board entries.",
        "",
        "## Material",
        *(f"- {k.replace('_', ' ')}: {v}" for k, v in m.items()),
        "",
        "## Value groups (by values, then chain depth)",
    ]
    for g in found["groups"]:
        if g["values"] < 2:
            continue
        out.append(
            f"- {g['subject']} / {g['attribute']}: {g['values']} values, depth {g['depth']}, decoys {len(g['decoys'])}, "
            f"{g['channels']} channels, days {g['days'][0]}-{g['days'][1]}, current {g['current']} in "
            f"{', '.join(g['current_where']) or 'nothing stated'} (readable by {g['readers'][0]}/{g['readers'][1]}), "
            f"version {g['version']}, tasks {g['tasks']}"
            + (f", board {', '.join(g['board'])}" if g["board"] else "")
        )
    t = found["threads"]
    out += [
        "",
        f"## Threads: {t['threads']}, {t['with_5_replies']} with 5 or more replies, at most {t['most_replies']}",
        "",
        "## Recurring conversations",
        *(f"- {r['situation']}: {r['conversations']}" for r in found["recurring"]),
        "",
        f"## People: busiest {found['people']['busiest']}; quiet {found['people']['quiet']}",
    ]
    return "\n".join(out) + "\n"


def write(run: Path) -> dict:
    """A run's profile, written beside its world as profile.json and profile.md."""
    world = Workspace(run / "world.sqlite")
    try:
        found = profile(world)
    finally:
        world.close()
    (run / "profile.json").write_text(json.dumps(found, indent=1, ensure_ascii=False))
    (run / "profile.md").write_text(page(found))
    return found["material"]
