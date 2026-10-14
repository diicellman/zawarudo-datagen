"""The board: the slots of a chapter's brief, the needs of their levels, and the measures of the facts an entry
rests on (where its answer is stated, its near-misses, the values its subject and attribute take)."""

import json

from pydantic import Field
from worldgen_slack.release import NonEmptyText, SafeId, StrictModel

from ..config import Needs
from .clock import chapter_start, meta


class Slot(StrictModel):
    """One task a world holds, drawn by the run seed: its id (the task's id), its taxonomy cell, the concept it
    requires and how it is asked."""

    id: SafeId
    category: NonEmptyText
    level: int = Field(ge=1)
    concept: NonEmptyText
    style: NonEmptyText
    needs: Needs | None = None  # a chapter's brief: what its board entry needs, in place of its level's


def slot_needs(settings, slot: Slot) -> Needs | None:
    """What a slot's board entry needs: a chapter's brief sets its own, else its level's."""
    if slot.needs is not None:
        return slot.needs
    spec = settings.taxonomy[slot.category]
    return spec.needs[slot.level - 1] if spec.needs else None


def on_board(settings, slot: Slot) -> bool:
    """Whether a slot rests on planned facts, so the board plans them before its first day: a ledger or hybrid slot,
    one whose level needs something of its facts (robustness level 4: a near-miss in sight, the truth out of it), and
    every entry of a chapter's brief."""
    needs = slot_needs(settings, slot)
    return (
        slot.needs is not None
        or settings.taxonomy[slot.category].gold in ("ledger", "hybrid")
        or bool(needs and needs.facts())
    )


def unmet(needs, measured: dict) -> list[str]:
    """Each of a level's needs that a board entry's facts fall short of, with what was measured. Whether its answer is
    hidden is checked when it is posted."""
    out = []
    if measured["channels"] < needs.channels:
        out.append(
            f"its facts first stated in at least {needs.channels} channels (they are in {measured['channels']})"
        )
    if measured["relations"] < needs.relations:
        out.append(
            f"at least {needs.relations} supersedes or after relations among its facts (it has {measured['relations']})"
        )
    if measured["decoys"] < needs.decoys:
        out.append(
            f"at least {needs.decoys} near-misses: decoys, or values whose change its actor cannot see, on its answer's subject and attribute, that its actor can read and no change it can see retracts (it has {measured['decoys']})"
        )
    if measured.get("values", needs.values) < needs.values:
        out.append(
            f"its answer's subject and attribute taking at least {needs.values} values, its answer among them (it takes {measured['values']})"
        )
    if needs.fresh and measured.get("fresh") is False:
        out.append(
            "its answer first stated in this chapter, on its first day or later: what it plants is new"
        )
    return out


def needs_text(needs) -> str:
    """A level's needs, as the author reads them in now.md."""

    def many(n, word):
        return f"{n} {word}" + ("s" if n > 1 else "")

    out = []
    if needs.channels:
        out.append(f"facts first stated in {many(needs.channels, 'channel')}")
    if needs.relations:
        out.append(f"{many(needs.relations, 'relation')} among its facts")
    if needs.decoys:
        out.append(
            f"{many(needs.decoys, 'near-miss')} on its answer's subject and attribute that its actor can read, unretracted in its sight"
        )
    if needs.hidden:
        out.append("its answer in a thread reply, a DM, or later than its near-misses")
    if needs.named is not None:
        out.append(
            "a question that names none of its evidence's channels and identifiers"
            if needs.named == 0
            else f"a question naming at most {needs.named} of its evidence's channels and identifiers"
        )
    if needs.values:
        out.append(
            f"its answer's subject and attribute taking {needs.values} or more values, its answer among them"
        )
    if needs.fresh:
        out.append("its answer first stated in this chapter")
    return "; ".join(out)


def fact_measures(world, ids: list[str], actor: str | None = None, gold=None, refusal: bool = False) -> dict:
    """What a task's facts give it: the channels they are first stated in, the relations among them, and its
    near-misses (`decoys`): facts on an answer fact's subject and attribute, other than its answer facts, that `actor`
    can read and that no fact `actor` can see supersedes: an unretracted decoy, or a value whose change is out of the
    actor's sight. A near-miss the world corrects in plain sight is no near-miss: a solver follows every such
    chain. Planned before any actor is chosen, a near-miss counts wherever it is stated, and a change counts as seen
    when it is public."""
    marks = ", ".join("?" * len(ids))
    channels = world.db.execute(
        f"SELECT COUNT(DISTINCT channel_id) FROM facts WHERE id IN ({marks})", ids
    ).fetchone()[0]
    relations = world.db.execute(
        f"SELECT COUNT(*) FROM fact_relations WHERE src_fact IN ({marks}) AND dst_fact IN ({marks})",
        [*ids, *ids],
    ).fetchone()[0]
    answers = answer_facts(world, ids, actor, gold, refusal)
    return {
        "channels": channels,
        "relations": relations,
        "decoys": len(near_misses(world, answers, actor)),
        "values": values(world, answers, actor),
    }


def values(world, answers: list[str], actor: str | None = None) -> int:
    """How many values an answer's subject and attribute takes, its own among them, as `actor` can read them (every
    one, before any actor is chosen): the most of any answer fact's."""
    most = 0
    for answer in answers:
        subject, attribute = world.db.execute(
            "SELECT lower(trim(subject)), lower(trim(attribute)) FROM facts WHERE id = ?", (answer,)
        ).fetchone()
        rows = world.db.execute(
            "SELECT id, lower(trim(value)) FROM facts WHERE lower(trim(subject)) = ? AND lower(trim(attribute)) = ?",
            (subject, attribute),
        ).fetchall()
        seen = {
            v
            for f, v in rows
            if f == answer or not actor or any(readable(world, actor, c) for c in places(world, f))
        }
        most = max(most, len(seen))
    return most


def board_measures(world, ids: list[str]) -> dict:
    """What only the plan can say of a board entry's facts: whether its answer is first stated where it is harder to
    see than its near-misses (`hidden`: in a DM, or on a later day than every near-miss; None when it may yet be, in a
    thread reply, which code checks when it is posted), and whether it is `fresh`, first stated in the chapter being
    written."""
    answers = answer_facts(world, ids)
    if not answers:
        return {"hidden": None, "fresh": None}
    marks = ", ".join("?" * len(answers))
    rows = world.db.execute(
        f"SELECT f.day, c.type FROM facts f JOIN channels c ON c.id = f.channel_id WHERE f.id IN ({marks})",
        answers,
    ).fetchall()
    misses = near_misses(world, answers)
    last = max(
        (world.db.execute("SELECT day FROM facts WHERE id = ?", (f,)).fetchone()[0] for f in misses),
        default=None,
    )
    hidden = any(t in ("im", "mpim") or (last is not None and d > last) for d, t in rows)
    return {"hidden": True if hidden else None, "fresh": any(d >= chapter_start(world) for d, _ in rows)}


def places(world, fact: str) -> set[str]:
    """Where a fact is stated, or, before it is, the channel it is planned to be first stated in."""
    stated = {
        r[0]
        for r in world.db.execute(
            "SELECT DISTINCT m.channel_id FROM evidence e JOIN messages m ON m.id = e.message_id WHERE e.fact_id = ?",
            (fact,),
        )
    }
    return stated or {world.db.execute("SELECT channel_id FROM facts WHERE id = ?", (fact,)).fetchone()[0]}


def answer_facts(
    world, ids: list[str], actor: str | None = None, gold=None, refusal: bool = False
) -> list[str]:
    """Which of a task's facts it answers with: those its gold answers are the values of; for a refusal, those out of
    its actor's sight (the truth it cannot see); else, or when none is, those that are no decoy and that no fact
    supersedes; and for a task about an earlier state, its facts that are no decoy (a later value is its near-miss)."""
    marks = ", ".join("?" * len(ids))
    values = {str(v).strip().casefold() for v in gold or ()}
    if refusal and actor:
        found = [f for f in ids if not any(readable(world, actor, c) for c in places(world, f))]
    else:
        found = [
            f
            for f, v in world.db.execute(f"SELECT id, value FROM facts WHERE id IN ({marks})", ids)
            if v.strip().casefold() in values
        ]
    current = f"SELECT id FROM facts WHERE id IN ({marks}) AND is_decoy = 0 AND id NOT IN (SELECT dst_fact FROM fact_relations WHERE kind = 'supersedes')"
    earlier = f"SELECT id FROM facts WHERE id IN ({marks}) AND is_decoy = 0"
    return (
        found
        or [f for (f,) in world.db.execute(current, ids)]
        or [f for (f,) in world.db.execute(earlier, ids)]
    )


def near_misses(world, answers: list[str], actor: str | None = None) -> list[str]:
    """The near-misses of a task answering with facts `answers` (`fact_measures`), as fact ids."""
    reads = (lambda c: readable(world, actor, c)) if actor else (lambda c: True)

    def sees(channel: str) -> bool:
        if actor:
            return readable(world, actor, channel)
        return (
            world.db.execute("SELECT type FROM channels WHERE id = ?", (channel,)).fetchone()[0] == "public"
        )

    out = []
    for answer in answers:
        subject, attribute = world.db.execute(
            "SELECT lower(trim(subject)), lower(trim(attribute)) FROM facts WHERE id = ?", (answer,)
        ).fetchone()
        for (fact,) in world.db.execute(
            "SELECT id FROM facts WHERE lower(trim(subject)) = ? AND lower(trim(attribute)) = ? ORDER BY id",
            (subject, attribute),
        ):
            if fact in answers or fact in out or not any(map(reads, places(world, fact))):
                continue
            changes = [
                r[0]
                for r in world.db.execute(
                    "SELECT src_fact FROM fact_relations WHERE kind = 'supersedes' AND dst_fact = ?", (fact,)
                )
            ]
            if not any(sees(c) for change in changes for c in places(world, change)):
                out.append(fact)
    return out


def hidden(world, ids: list[str], actor: str, gold=None) -> bool | None:
    """Whether a task's answer is stated where it is harder to see than its near-misses: an anchor of an answer fact
    in a thread reply or a direct conversation, or on a later day than every near-miss is first stated. None for a
    task with no answer fact stated."""
    answers = answer_facts(world, ids, actor, gold)
    misses = near_misses(world, answers, actor)
    marks = ", ".join("?" * len(answers))
    anchors = world.db.execute(
        f"""SELECT m.parent_id, c.type, f.day FROM evidence e JOIN messages m ON m.id = e.message_id
        JOIN channels c ON c.id = m.channel_id JOIN facts f ON f.id = e.fact_id
        WHERE e.role = 'anchor' AND e.fact_id IN ({marks})""",
        answers,
    ).fetchall()
    if not anchors:
        return None
    last = max(
        (world.db.execute("SELECT day FROM facts WHERE id = ?", (f,)).fetchone()[0] for f in misses),
        default=None,
    )
    return any(
        a["parent_id"] is not None or a["type"] in ("im", "mpim") or (last is not None and a["day"] > last)
        for a in anchors
    )


def readable(world, actor: str, channel: str) -> bool:
    return bool(
        world.db.execute(
            """SELECT 1 FROM channels c WHERE c.id = ? AND (c.type = 'public' OR EXISTS (SELECT 1 FROM members m
            WHERE m.channel_id = c.id AND m.user_id = ? AND m.left_us IS NULL))""",
            (channel, actor),
        ).fetchone()
    )


def brief(world) -> dict[str, dict]:
    """A chapter's brief, as it was written when the chapter started: each entry's needs, by slot."""
    return json.loads(meta(world, "brief", "{}"))


def hidden_answers(world) -> dict[str, list[tuple[str, int | None]]]:
    """The answer facts of the brief's entries that need their answer out of easy sight, by fact: each entry, with
    the last day a near-miss of it is first stated (None when it has none)."""
    needs, out = brief(world), {}
    for slot, ids in world.db.execute(
        "SELECT slot, group_concat(fact_id) FROM board GROUP BY slot"
    ).fetchall():
        if not needs.get(slot, {}).get("hidden"):
            continue
        answers = answer_facts(world, ids.split(","))
        misses = near_misses(world, answers)
        last = max(
            (world.db.execute("SELECT day FROM facts WHERE id = ?", (m,)).fetchone()[0] for m in misses),
            default=None,
        )
        for answer in answers:
            out.setdefault(answer, []).append((slot, last))
    return out
