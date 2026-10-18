"""A world's profile: what material it holds for tasks, read off the world file with no model call.
uv run --frozen worldgen-slack profile --world data/v7-05/software

The value groups (a subject and attribute, the values it took, how they supersede each other, its decoys, where its
current value is stated and how many can read it), threads, recurring conversations, people, and what the world's
tasks and board already rest on: counts for a person, who writes a chapter's brief from it and measures by it what a
chapter planted. A forge's proposer reads the same world as its task material (forge/material.py): the messages each
value rests on and who can read them."""

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from worldgen_slack.slack import PAGE

from ..workspace import Workspace, asked
from .board import answer_facts, readable
from .clock import bounds, meta, present
from .text import PEOPLE

PRIVATE = ("private", "im", "mpim")
# What a world's ledger is held to (`[ledger]`) and measured by: the counts of `counts`, each as the author reads it.
MATERIAL = {
    "facts": "facts",
    "chains": "chains (a subject and attribute whose values supersede one another 3 deep, across 2 or more channels)",
    "deep_chains": "deep chains (4 deep)",
    "decoys": "decoys (each beside a real value of its subject and attribute)",
    "private_truths": "private truths (a value that changed, its current one planned in a private channel or a DM)",
    "collections": "collections (4 or more items of one set, planned across 3 or more channels, an item's value "
    "changed)",
    "archives": "channels archived (a move)",
    "departures": "people who leave the company (a move)",
}
PLANNED = list(MATERIAL)[:6]  # counted by storyline


def shortfalls(world, targets) -> list[str]:
    """What the ledger holds less of than `targets` (a world's `[ledger]`) asks."""
    have = counts(world)
    return [
        f"{have[k]} {text} of the {want} it holds at least"
        for k, text in MATERIAL.items()
        if have[k] < (want := getattr(targets, k))
    ]


def target_line(world, targets) -> str:
    """The ledger against its targets, on one line; empty when nothing is asked."""
    if targets is None or not any(getattr(targets, k) for k in MATERIAL):
        return ""
    have = counts(world)
    return ", ".join(
        f"{k.replace('_', ' ')} {have[k]}/{getattr(targets, k)}" for k in MATERIAL if getattr(targets, k)
    )


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


def counts(world, through_day: int | None = None) -> dict:
    """The world's task material, counted off the ledger as planned, so a plan is held to it before a line is written:
    its facts; chains, value groups (a subject and attribute) whose values supersede one another 3 deep (deep chains:
    4) across 2 or more channels; decoys beside a real value of their group; private truths, groups of 2 or more
    values whose current value is planned in a private channel or a DM; collections, sets of 4 or more items planned
    across 3 or more channels, an item's value changed at least once; the channels archived and the people who leave.
    Each but the last two also by storyline. `through_day` counts only what is first stated by that day."""
    last = through_day if through_day is not None else math.inf
    # a world of an older schema, read as it is (not brought forward), has no collections
    collection = (
        "lower(trim(f.collection))"
        if "collection" in {r[1] for r in world.db.execute("PRAGMA table_info(facts)")}
        else "NULL"
    )
    facts = [
        dict(r)
        for r in world.db.execute(
            f"""SELECT f.id, f.storyline, f.day, lower(trim(f.subject)) AS subject, lower(trim(f.attribute)) AS attribute,
            lower(trim(f.value)) AS value, f.channel_id, f.is_decoy, {collection} AS collection, c.type
            FROM facts f JOIN channels c ON c.id = f.channel_id ORDER BY f.day, f.id"""
        )
        if r["day"] <= last
    ]
    known = {f["id"] for f in facts}
    supersedes = {
        src: dst
        for src, dst in world.db.execute(
            "SELECT src_fact, dst_fact FROM fact_relations WHERE kind = 'supersedes'"
        )
        if src in known and dst in known
    }
    replaced, depth = set(supersedes.values()), {}

    def of(
        fact: str,
    ) -> int:  # its depth in its chain: 1 for a fact that supersedes nothing (a ledger has no cycle)
        if fact not in depth:
            depth[fact] = 1 + (of(supersedes[fact]) if fact in supersedes else 0)
        return depth[fact]

    groups, sets = defaultdict(list), defaultdict(list)
    for f in facts:
        groups[(f["subject"], f["attribute"])].append(f)
        if f["collection"]:
            sets[f["collection"]].append(f)
    tally = defaultdict(Counter)  # storyline -> material -> count
    for f in facts:
        tally[f["storyline"]]["facts"] += 1
    for group in groups.values():
        storyline, real = group[0]["storyline"], [f for f in group if not f["is_decoy"]]
        deepest, channels = max(of(f["id"]) for f in group), len({f["channel_id"] for f in group})
        tally[storyline]["chains"] += deepest >= 3 and channels >= 2
        tally[storyline]["deep_chains"] += deepest >= 4 and channels >= 2
        tally[storyline]["decoys"] += sum(f["is_decoy"] for f in group) if real else 0
        current = [f for f in real if f["id"] not in replaced]
        tally[storyline]["private_truths"] += (
            len({f["value"] for f in group}) >= 2
            and bool(current)
            and all(f["type"] in PRIVATE for f in current)
        )
    for items in sets.values():
        tally[items[0]["storyline"]]["collections"] += (
            len({f["subject"] for f in items}) >= 4
            and len({f["channel_id"] for f in items}) >= 3
            and any(f["id"] in replaced for f in items)
        )
    end = bounds(world, through_day)[1] if through_day is not None else math.inf

    def moved(sql: str) -> int:
        return sum(us < end for (us,) in world.db.execute(sql))

    by_storyline = {s: {k: c[k] for k in PLANNED} for s, c in sorted(tally.items())}
    return {k: sum(c[k] for c in by_storyline.values()) for k in PLANNED} | {
        "archives": moved("SELECT archived_us FROM channels WHERE archived_us IS NOT NULL"),
        "departures": moved(
            "SELECT deactivated_us FROM users WHERE is_bot = 0 AND deactivated_us IS NOT NULL"
        ),
        "by_storyline": by_storyline,
    }


def promises(world) -> dict[str, int]:
    """How the world's promises went, as of its present: kept by their due day or later, changed, dropped, slipped
    (past due with nothing said), and still open."""
    now, out = present(world), Counter()
    for status, due, closed in world.db.execute(
        """SELECT c.status, c.due_us, m.ts_us FROM commitments c LEFT JOIN messages m ON m.id = c.closed_by"""
    ):
        if status == "kept":
            out["kept late" if closed > due else "kept"] += 1
        else:
            out[status if status != "open" else "slipped" if due < now else "open"] += 1
    return {k: out[k] for k in ("kept", "kept late", "changed", "dropped", "slipped", "open")}


def depth_of_channels(world, top: int = 5) -> list[dict]:
    """The busiest channels, how many top-level messages each holds, and how many of read_channel's pages they fill."""
    rows = world.db.execute(
        """SELECT c.name, c.type, COUNT(*) FROM messages m JOIN channels c ON c.id = m.channel_id
        WHERE m.parent_id IS NULL AND c.name IS NOT NULL GROUP BY c.id ORDER BY COUNT(*) DESC, c.name LIMIT ?""",
        (top,),
    ).fetchall()
    return [{"channel": f"#{n}", "type": t, "top_level": k, "pages": math.ceil(k / PAGE)} for n, t, k in rows]


def fact_share(world) -> float:
    """The share of people's messages that state or repeat a fact."""
    total, stating = world.db.execute(
        f"""SELECT COUNT(*), COUNT(DISTINCT e.message_id) FROM messages m JOIN users u ON u.id = m.user_id
        LEFT JOIN evidence e ON e.message_id = m.id WHERE {PEOPLE}"""
    ).fetchone()
    return round(stating / total, 3) if total else 0.0


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
        "counts": counts(world),
        "promises": promises(world),
        "deepest_channels": depth_of_channels(world),
        "fact_share": fact_share(world),
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
    w, m, c = found["world"], found["material"], found["counts"]
    out = [
        f"# Profile: {w['company']}, version {w['version']}",
        "",
        f"{w['days']} days, {w['messages']} messages, {w['people']} people, channels {w['channels']}; {w['facts']} facts, "
        f"{w['tasks']} tasks, {w['board_entries']} board entries; {found['fact_share']:.1%} of people's messages state "
        "a fact.",
        "",
        "## Material",
        *(f"- {k.replace('_', ' ')}: {v}" for k, v in m.items()),
        "",
        "## The ledger's counts (what `[ledger]` holds a world to)",
        "- " + ", ".join(f"{k.replace('_', ' ')} {c[k]}" for k in MATERIAL),
        *(
            f"- {s}: " + ", ".join(f"{k.replace('_', ' ')} {n}" for k, n in t.items())
            for s, t in c["by_storyline"].items()
        ),
        "",
        "## Promises: " + ", ".join(f"{k} {n}" for k, n in found["promises"].items()),
        "",
        "## Deepest channels: "
        + "; ".join(
            f"{d['channel']} {d['top_level']} top-level ({d['pages']} pages)"
            for d in found["deepest_channels"]
        ),
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
    return {
        "material": found["material"],
        "counts": {k: v for k, v in found["counts"].items() if k != "by_storyline"},
    }
