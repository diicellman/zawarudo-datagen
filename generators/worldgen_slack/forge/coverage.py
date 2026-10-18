"""What a forge's tasks cover of their world, by code: the families and skills they ask, the storylines and askers
they rest on, the material no task rests on yet (value groups, collections, promises), and the openings that repeat.
The proposer reads it on its tasks page, to write where the world still has material; the report carries it."""

import re
from collections import Counter, defaultdict

from .tasks import FAMILIES, Task


def opening(question: str) -> str:
    """A request's first two words, lowercased: how it opens."""
    return " ".join(re.findall(r"[\w']+", question.lower())[:2])


def coverage(world, records: list[dict]) -> dict:
    """What `records` (forged records: a task and the skill it aimed at) cover of the world."""
    tasks = [Task.model_validate(r["task"]) for r in records]
    used = {m for t in tasks for p in t.parts for m in (*p.evidence, *p.unseen)}
    facts_of = defaultdict(set)
    for fact, message in world.db.execute("SELECT fact_id, message_id FROM evidence"):
        facts_of[message].add(fact)
    touched = {f for m in used for f in facts_of[m]}
    groups, storyline_of, sets = defaultdict(set), {}, defaultdict(set)
    for row in world.db.execute("SELECT * FROM facts"):
        key = (row["subject"].strip(), row["attribute"].strip())
        groups[key].add(row["id"])
        storyline_of[key] = row["storyline"]
        if "collection" in row.keys() and row["collection"]:
            sets[row["collection"]].add(row["id"])
    unused = [key for key, facts in groups.items() if not facts & touched]
    by_storyline = Counter(
        storyline_of[key] for key, facts in groups.items() if facts & touched
    )  # tasks' material, by storyline
    names = dict(world.db.execute("SELECT id, real_name FROM users").fetchall())
    promises = [
        f"{c['id']} ({names[c['owner_id']]}: {c['text']})"
        for c in world.db.execute("SELECT * FROM commitments ORDER BY due_us, id")
        if c["message_id"] not in used and (c["closed_by"] is None or c["closed_by"] not in used)
    ]
    openings = Counter(opening(t.question) for t in tasks)
    return {
        "tasks": len(tasks),
        "families": Counter(t.family for t in tasks),
        "skills": Counter(r.get("skill") or "none" for r in records),
        "askers": len({t.actor_id for t in tasks}),
        "storylines": dict(sorted(by_storyline.items())),
        "unused_groups": len(unused),
        "unused_by_storyline": dict(sorted(Counter(storyline_of[k] for k in unused).items())),
        "unused_examples": [f"{s} / {a}" for s, a in sorted(unused)[:12]],
        "unused_collections": sorted(name for name, facts in sets.items() if not facts & touched),
        "unused_promises": promises,
        "openings": [[o, n] for o, n in openings.most_common() if n > 1][:8],
    }


def page(found: dict) -> list[str]:
    """Coverage as the head of the proposer's tasks page."""
    families = ", ".join(f"{f} {found['families'].get(f, 0)}" for f in FAMILIES)
    skills = ", ".join(f"{s} {n}" for s, n in sorted(found["skills"].items()))
    lines = [
        f"## Coverage: {found['tasks']} tasks by {found['askers']} askers",
        f"- families: {families}; skills: {skills}",
        "- material tasks rest on, by storyline: "
        + (", ".join(f"{s} {n}" for s, n in found["storylines"].items()) or "none yet"),
    ]
    if found["unused_groups"]:
        lines.append(
            f"- value groups no task rests on yet: {found['unused_groups']} ("
            + ", ".join(f"{s} {n}" for s, n in found["unused_by_storyline"].items())
            + "), among them: "
            + "; ".join(found["unused_examples"])
        )
    if found["unused_collections"]:
        lines.append("- collections no task rests on yet: " + "; ".join(found["unused_collections"]))
    if found["unused_promises"]:
        lines.append(
            f"- promises no task rests on yet ({len(found['unused_promises'])}): "
            + "; ".join(found["unused_promises"][:15])
        )
    if found["openings"]:
        lines.append(
            "- openings already used more than once (open new requests otherwise): "
            + "; ".join(f"{o!r} {n}" for o, n in found["openings"])
        )
    return lines + [""]
