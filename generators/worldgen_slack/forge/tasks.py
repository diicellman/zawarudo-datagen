"""The task contract: what a task is, its gold query checked as its actor, and a candidate recorded on a world."""

import json
import re
import unicodedata
from typing import Literal

from pydantic import Field, model_validator
from worldgen_slack.release import NonEmptyText, SafeId, StrictModel
from worldgen_slack.world import SHOWN, canonical

from ..config import Ids
from ..evidence import measures
from ..world.board import Slot, places, readable
from ..world.text import normalized
from .difficulty import features, panel


def _meaningful_tokens(value: str) -> set[str]:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return {token for token in re.findall(r"[^\W_]+", normalized) if len(token) >= 2}


# What a person asks an assistant for, each checkable against the workspace and its ledger.
INTENTS = {
    "find": "find a fact: what was decided, said or chosen",
    "status": "the status of something, or what changed about it",
    "owner": "who owns or handles something, now or at a moment",
    "promised": "what someone promised, and whether it is still open",
    "when": "when something happens or happened, on whose clock",
    "stance": "who said, agreed or objected",
    "structure": "who posted, reacted or replied where: the workspace's own record",
}
# What a task is, as a request: what people ask a colleague who reads Slack for them. A forge balances its tasks by
# family (they replace the intents), and a request of several parts asks for what a real catch-up or report needs.
FAMILIES = {
    "catch-up": "what the asker missed on something, or what changed about it since a moment",
    "decisions": "what was decided about something, and why; whether it is still the plan",
    "commitments": "what someone promised, to whom, and whether it was done",
    "status": "who handles something now, where it stands, or who to ask about it",
}


class Part(StrictModel):
    """One part of a composite request: what it asks, in the request's own words (`label`), and its gold as a task's:
    a category of the taxonomy, an answer type, the gold query, the facts it rests on and its near-miss queries."""

    label: NonEmptyText
    category: NonEmptyText
    answer_type: Literal["text", "set", "number", "refusal", "status"]
    gold_sql: NonEmptyText
    facts: Ids = Field(default_factory=list)
    near_sql: list[NonEmptyText] = Field(default_factory=list, max_length=3)


class Task(StrictModel):
    """One request a colleague would make, of a family: a single question (a category, a level, its gold query) or a
    composite request in 2 to 4 parts, each with its own gold, answered right only when every part is."""

    id: SafeId
    category: NonEmptyText
    level: int = Field(ge=1)
    actor_id: SafeId
    question: NonEmptyText
    answer_type: Literal["text", "set", "number", "refusal", "status", "composite"]
    gold_sql: NonEmptyText | None = None
    facts: Ids = Field(default_factory=list)
    # The gold query with one condition relaxed (a day, a channel, a reaction left out): what a hasty reader answers.
    near_sql: list[NonEmptyText] = Field(default_factory=list, max_length=3)
    # In the forge, the id of a task sent back for another reading of its question: this one rewords it, for the same
    # asker and the same gold answer.
    rewords: SafeId | None = None
    # In the forge, the request it is (INTENTS): the archive balances them, so the tasks read like real requests.
    intent: Literal["find", "status", "owner", "promised", "when", "stance", "structure"] | None = None
    # The request family it is (FAMILIES): a forge's places are its families by measured level.
    family: Literal["catch-up", "decisions", "commitments", "status"] | None = None
    # A composite request (category and answer_type composite): its 2 to 4 parts, each with its own gold; it is
    # answered right only when every part is.
    parts: list[Part] = Field(default_factory=list, max_length=4)

    @model_validator(mode="after")
    def shaped(self):
        if self.parts or "composite" in (self.answer_type, self.category):
            if not 2 <= len(self.parts) <= 4 or (self.answer_type, self.category) != (
                "composite",
                "composite",
            ):
                raise ValueError(
                    "a composite request has category and answer_type composite, and 2 to 4 parts"
                )
            if self.gold_sql or self.facts or self.near_sql:
                raise ValueError("a composite request's gold, facts and near-miss queries are its parts'")
        elif not self.gold_sql:
            raise ValueError("a task names its gold_sql, or is a composite request of parts")
        return self


def part_of(task: Task, n: int, part: Part) -> Task:
    """A composite's `n`th part as a task of its own, never asked alone: its request's asker and level, its own gold."""
    return Task(
        id=f"{task.id}-p{n}",
        category=part.category,
        level=task.level,
        actor_id=task.actor_id,
        question=part.label,
        answer_type=part.answer_type,
        gold_sql=part.gold_sql,
        facts=part.facts,
        near_sql=part.near_sql,
    )


LEDGER_TABLES = {
    "facts",
    "fact_relations",
    "evidence",
    "commitments",
    "artifacts",
}  # promises and the registry too


STRUCTURE = {"channels", "members", "messages", "message_mentions", "reactions", "thread_stats"}


def same_rows(rows: list[dict], again: list[dict], back) -> bool:
    """Whether a renumbered copy's rows, their message ids mapped back, are the world's rows."""
    mapped = [
        r | {"message_id": back(r["message_id"])} if r.get("message_id") is not None else r for r in again
    ]
    return sorted(map(canonical, rows)) == sorted(map(canonical, mapped))


def check_task(world, task: Task, settings, renumbered=None) -> list:
    """T1-T5, and T7 given the world `renumbered`: the gold query runs as the task's actor, reads what the task's
    category promises, and finds its messages by what they say; the rows are the gold answer. Returns them; for a
    composite request, each part's, each part checked as a task, its answers its own and none given in the request."""
    if task.parts:
        golds = [
            check_task(world, part_of(task, n, p), settings, renumbered) for n, p in enumerate(task.parts, 1)
        ]
        answers = [tuple(sorted(str(r.get("answer")) for r in rows)) for rows in golds]
        if len(set(answers)) < len(answers):
            raise ValueError(
                f"{task.id}: each part asks for something of its own; two parts give the same answer"
            )
        if given := [
            str(r["answer"])
            for p, rows in zip(task.parts, golds)
            if p.answer_type != "number"
            for r in rows
            if not _meaningful_tokens(str(r["answer"])) - _meaningful_tokens(task.question)
        ]:
            raise ValueError(f"{task.id}: the request gives away its answer {given}")
        return golds
    category = settings.taxonomy.get(task.category)
    if category is None:
        raise ValueError(f"{task.id}: category must be one of {sorted(settings.taxonomy)}")
    if not 1 <= task.level <= len(category.levels) or task.answer_type not in category.answer_types:
        raise ValueError(
            f"{task.id}: a {task.category} task has a level from 1 to {len(category.levels)} and an answer type in "
            f"{category.answer_types}"
        )
    asker = world.db.execute("SELECT is_bot, is_deleted FROM users WHERE id = ?", (task.actor_id,)).fetchone()
    if asker is not None and (asker["is_bot"] or asker["is_deleted"]):
        raise ValueError(
            f"{task.id}: actor_id {task.actor_id} is a bot or no longer at the company; a person asks"
        )
    try:
        out = world.gold(task.actor_id, task.gold_sql, max_rows=settings.max_answer_rows)
    except LookupError:
        raise ValueError(f"{task.id}: actor_id {task.actor_id} is not a person in the world") from None
    except ValueError as error:
        raise ValueError(f"{task.id}: {error}") from None
    read, rows, source = set(out["tables"]), out["rows"], gold_source(task, settings)
    if renumbered is not None:
        copy, back = renumbered
        try:
            again = copy.gold(task.actor_id, task.gold_sql, max_rows=settings.max_answer_rows)["rows"]
        except ValueError:
            again = None
        if again is None or not same_rows(rows, again, back):
            raise ValueError(
                f"{task.id}: the gold query finds messages by what the question names (words, people, channels, "
                "threads, reactions, times), never by a message id or by id order"
            )
    if source == "sql" and read & LEDGER_TABLES:
        raise ValueError(
            f"{task.id}: a {task.category} gold query reads only the workspace, not {sorted(read & LEDGER_TABLES)}"
        )
    if source in ("ledger", "hybrid") and not read & LEDGER_TABLES:
        raise ValueError(
            f"{task.id}: a {task.category} {task.answer_type} gold query reads the facts it answers from"
        )
    if source == "hybrid" and not read & STRUCTURE:
        raise ValueError(
            f"{task.id}: a hybrid gold query also reads the workspace (messages, members, reactions)"
        )
    if source in ("ledger", "hybrid") and not task.facts:
        raise ValueError(f"{task.id}: name the facts its answer rests on in facts")
    unshown = [r for r in out["reads"] if (t := r.split(".")[0]) in SHOWN and r.split(".")[1] not in SHOWN[t]]
    if unshown:
        raise ValueError(
            f"{task.id}: the gold query reads {unshown}, which no Slack tool shows its solver; ask only about what "
            "the tools show"
        )
    if task.answer_type == "refusal":
        if rows:
            raise ValueError(
                f"{task.id}: a refusal's gold query returns no rows as its actor; it returns {len(rows)}"
            )
        return rows
    if "answer" not in out["columns"]:
        raise ValueError(f"{task.id}: the gold query returns an `answer` column")
    if not 1 <= len(rows) <= settings.max_answer_rows:
        raise ValueError(
            f"{task.id}: the gold query returns 1 to {settings.max_answer_rows} rows as its actor, not "
            f"{len(rows) if len(rows) <= settings.max_answer_rows else 'more'}"
        )
    if task.answer_type != "set" and len(rows) != 1:
        raise ValueError(
            f"{task.id}: a {task.answer_type} answer is one row; use answer_type set for several"
        )
    if task.answer_type == "status":
        unsettled(world, task, rows)
    for near in task.near_sql:  # a hasty reader's answer: another one, as the actor sees it
        try:
            relaxed = world.gold(task.actor_id, near, max_rows=50)["rows"]
        except ValueError as error:
            raise ValueError(f"{task.id}: its near-miss query fails: {error}") from None
        if sorted(str(r.get("answer")) for r in relaxed) == sorted(str(r.get("answer")) for r in rows):
            raise ValueError(
                f"{task.id}: its near-miss query returns its gold answer; relax a condition that matters"
            )
    if given := [
        str(r["answer"])
        for r in rows
        if task.answer_type != "number"
        and not _meaningful_tokens(str(r["answer"])) - _meaningful_tokens(task.question)
    ]:
        raise ValueError(f"{task.id}: the question gives away its answer {given}")
    return rows


def record_task(world, task: Task, settings, slot: Slot) -> dict:
    """T6: a task fills its slot. Code checks it (its slot's cell, a question no other task asks, facts its actor can
    read, T1-T5, T7) and inserts it with its slot's concept and its gold rows. Returns its gold rows and code's
    measures. Run inside `World.trial()`."""
    if (task.id, task.category, task.level) != (slot.id, slot.category, slot.level):
        raise ValueError(f"{slot.id} is a {slot.category} level {slot.level} task, with the slot's id")
    if task.parts:
        return record_composite(world, task, settings, slot)
    for fact in (
        task.facts if task.answer_type not in ("refusal", "status") else ()
    ):  # a refusal or a status may rest on the truth out of sight
        row = world.db.execute("SELECT channel_id FROM facts WHERE id = ?", (fact,)).fetchone()
        if row and not readable(world, task.actor_id, row[0]):
            raise ValueError(f"{task.id}: its actor cannot read {row[0]}, where {fact} is stated")
    with world.renumbered() as renumbered:
        gold = check_task(world, task, settings, renumbered)
    # A question no other task asks, but for a perspective twin: another asker, for whom its answer differs.
    for question, actor, rows in world.db.execute(
        "SELECT question, actor_id, gold_json FROM tasks WHERE id != ? AND parent IS NULL", (task.id,)
    ):
        if normalized(question) == normalized(task.question) and (
            actor == task.actor_id or json.loads(rows) == gold
        ):
            raise ValueError(
                f"{task.id}: a task asks a question no other task asks, unless another person asks it and its answer differs for them"
            )
    with world.batch():
        world.insert("tasks", [task_row(task, settings) | {"concept": slot.concept}])
        world.insert("task_facts", [dict(task_id=task.id, fact_id=f) for f in dict.fromkeys(task.facts)])
    world.db.execute("UPDATE tasks SET gold_json = ? WHERE id = ?", (json.dumps(gold), task.id))
    tables = world.gold(task.actor_id, task.gold_sql, max_rows=settings.max_answer_rows)["tables"]
    return {"gold": gold} | measures(world, task.id, gold, tables)


def record_composite(world, task: Task, settings, slot: Slot) -> dict:
    """A composite request recorded: its parts checked (T1-T7, readable facts, each as a task), then the request and
    each part inserted, a part a task of its own under its request (parent, part, label) with its gold rows and facts.
    Returns each part's gold rows and code's measures."""
    parts = [part_of(task, n, p) for n, p in enumerate(task.parts, 1)]
    for part in parts:
        for fact in part.facts if part.answer_type not in ("refusal", "status") else ():
            row = world.db.execute("SELECT channel_id FROM facts WHERE id = ?", (fact,)).fetchone()
            if row and not readable(world, part.actor_id, row[0]):
                raise ValueError(f"{part.id}: its actor cannot read {row[0]}, where {fact} is stated")
    with world.renumbered() as renumbered:
        golds = check_task(world, task, settings, renumbered)
    for (question,) in world.db.execute(
        "SELECT question FROM tasks WHERE id != ? AND parent IS NULL AND actor_id = ?",
        (task.id, task.actor_id),
    ):
        if normalized(question) == normalized(task.question):
            raise ValueError(f"{task.id}: a request no other task of its asker makes")
    with world.batch():
        world.insert("tasks", [task_row(task, settings) | {"concept": slot.concept}])
        for n, (part, rows) in enumerate(zip(parts, golds), 1):
            world.insert(
                "tasks",
                [
                    task_row(part, settings)
                    | {"concept": slot.concept, "parent": task.id, "part": n, "label": part.question}
                    | {"gold_json": json.dumps(rows)}
                ],
            )
            world.insert("task_facts", [dict(task_id=part.id, fact_id=f) for f in dict.fromkeys(part.facts)])
    out = []
    for part, rows in zip(parts, golds):
        tables = world.gold(part.actor_id, part.gold_sql, max_rows=settings.max_answer_rows)["tables"]
        out.append(
            {"part": part.id, "label": part.question, "gold": rows} | measures(world, part.id, rows, tables)
        )
    return {"gold": golds, "parts": out}


def gold_source(task: Task, settings) -> str:
    """Where a task's answer comes from: its category's gold, but a status answer is a value of the ledger as its
    actor can see it stated, and a composite request's answer is its parts'."""
    if task.parts:
        return "composite"
    if task.answer_type == "status":
        return "ledger"
    return settings.taxonomy[task.category].gold if task.category in settings.taxonomy else "sql"


def unsettled(world, task: Task, rows: list[dict]) -> None:
    """A status answer's ground: the value its gold gives is a fact its actor can read, that no fact in their sight
    supersedes, and that a fact it rests on, out of their sight, does: a team that can read only a provisional
    "15%... Grant can correct me", while a DM has settled 10%, is best answered 15%, as not settled."""
    values = {str(r.get("answer")).strip().casefold() for r in rows}

    def seen(f):
        return any(readable(world, task.actor_id, c) for c in places(world, f))

    marks = ", ".join("?" * len(task.facts))
    given = [
        f
        for f, v in world.db.execute(f"SELECT id, value FROM facts WHERE id IN ({marks})", task.facts)
        if v.strip().casefold() in values and seen(f)
    ]
    if not given:
        raise ValueError(
            f"{task.id}: a status answer's gold gives the value of one of its facts its actor can read"
        )
    # The values that replace it: later facts on its subject and attribute down its supersedes chain.
    later = [
        r[0]
        for r in world.db.execute(
            f"""WITH RECURSIVE chain(id) AS (SELECT src_fact FROM fact_relations WHERE kind = 'supersedes'
        AND dst_fact IN ({", ".join("?" * len(given))}) UNION SELECT r.src_fact FROM chain c JOIN fact_relations r
        ON r.dst_fact = c.id AND r.kind = 'supersedes')
        SELECT f.id FROM chain JOIN facts f ON f.id = chain.id JOIN facts g ON g.id = ?
        WHERE lower(trim(f.subject)) = lower(trim(g.subject)) AND lower(trim(f.attribute)) = lower(trim(g.attribute))""",
            [*given, given[0]],
        )
    ]
    if in_sight := [f for f in later if seen(f)]:
        raise ValueError(
            f"{task.id}: its actor can read {in_sight}, which supersede the value it gives: that is its answer, not a status"
        )
    if not [f for f in task.facts if f in later]:
        raise ValueError(
            f"{task.id}: a status answer rests on a fact, out of its actor's sight, that supersedes the value it gives"
        )


def task_row(task: Task, settings) -> dict:
    return task.model_dump(
        include={"id", "category", "level", "actor_id", "question", "answer_type", "intent", "family"}
    ) | {
        "gold_sql": task.gold_sql or "",
        "gold_source": gold_source(task, settings),
        "near_sql": json.dumps(task.near_sql),
    }


def answer_of(world, task_id: str) -> list:
    """What a task answers: its gold rows' answers, or, for a refusal, the facts it rests on; for a composite request,
    each part's, by its label."""
    row = world.db.execute("SELECT answer_type, gold_json FROM tasks WHERE id = ?", (task_id,)).fetchone()
    if row["answer_type"] == "composite":
        return [
            f"{label}: {', '.join(answer_of(world, part))}"
            for part, label in world.db.execute(
                "SELECT id, label FROM tasks WHERE parent = ? ORDER BY part", (task_id,)
            ).fetchall()
        ]
    if row["answer_type"] == "refusal":
        return sorted(
            f for (f,) in world.db.execute("SELECT fact_id FROM task_facts WHERE task_id = ?", (task_id,))
        )
    return sorted(str(r.get("answer")) for r in json.loads(row["gold_json"]))


def reference_of(db, task_id: str) -> dict:
    """What the answer judge grades a task against, from its world's rows: its answer type and gold rows; a
    composite request's, each part's under its label (as its release's `PrivateAnswer.reference`)."""
    row = db.execute("SELECT answer_type, gold_json FROM tasks WHERE id = ?", (task_id,)).fetchone()
    if row[0] != "composite":
        return {"answer_type": row[0], "rows": json.loads(row[1])}
    parts = db.execute(
        "SELECT answer_type, gold_json, label FROM tasks WHERE parent = ? ORDER BY part", (task_id,)
    ).fetchall()
    return {
        "answer_type": "composite",
        "parts": [{"answer_type": k, "rows": json.loads(g), "label": label} for k, g, label in parts],
    }


def forget(world, task_id: str) -> None:
    """A task taken off the world, its parts with it."""
    for t in [r[0] for r in world.db.execute("SELECT id FROM tasks WHERE parent = ?", (task_id,))] + [
        task_id
    ]:
        world.db.execute("DELETE FROM task_facts WHERE task_id = ?", (t,))
    world.db.execute("DELETE FROM tasks WHERE parent = ?", (task_id,))
    world.db.execute("DELETE FROM tasks WHERE id = ?", (task_id,))


def forge_task(
    world, task: Task, settings, open_ids: list[str], gates: bool = True, rework: dict | None = None
) -> dict:
    """One candidate task on a finished, frozen world, for one of the round's open ids. Its category and level are the
    proposer's own, and its tries re-level it; it is checked as every task is (T1-T7, readable facts, the columns the
    tools show), with no planned level's needs: the forge measures difficulty instead of planning it. Without `gates`
    (a finished run's own task, measured as it was written), the forge's own gates (the shortcut panel, the near-miss
    queries) are left out. Returns its gold rows and code's measures."""
    if task.id not in open_ids:
        raise ValueError(f"a candidate's id is one of this round's open ids: {open_ids}")
    if gates and task.family is None:
        raise ValueError(f"{task.id}: name its family, the request it is: {sorted(FAMILIES)}")
    was = None
    if task.rewords is not None:
        if (was := (rework or {}).get(task.rewords)) is None:
            raise ValueError(
                f"rewords names a task the last round sent back for another reading: {sorted(rework or {})}"
            )
        if task.actor_id != was["actor"]:
            raise ValueError(f"{task.id} rewords {task.rewords}: the same asker, {was['actor']}")
    spec = settings.taxonomy.get(task.category)
    concept = (
        spec.levels[task.level - 1] if spec and 1 <= task.level <= len(spec.levels) else "the proposer's own"
    )
    slot = Slot(
        id=task.id, category=task.category, level=task.level, concept=concept, style="the proposer's own"
    )
    forget(world, task.id)
    result = {"task": task.id} | record_task(world, task, settings, slot)
    if was is not None and (now := answer_of(world, task.id)) != was["answer"]:
        raise ValueError(
            f"{task.id} rewords {task.rewords} and keeps its gold answer: that one answers {was['answer']}, this one "
            f"{now}. Change the words so they rule the other reading out ({was['reading']}), not the answer"
        )
    if (
        task.parts
    ):  # its difficulty is its parts', each measured as a task's; no one search answers a whole request
        return result | {"features": features(world, task.id, [])}
    lazy = panel(world, task.id, result["gold"])
    if gates and task.level > 1 and not task.facts and task.answer_type != "refusal" and not task.near_sql:
        raise ValueError(
            f"{task.id}: a task above level 1 with no facts names in near_sql its gold query with one condition "
            "relaxed (a day, a channel, a reaction or a thread left out), which answers otherwise: the answer a hasty "
            "reader gives"
        )
    if gates and task.level > 1 and lazy["shortcut"]:
        raise ValueError(
            f"{task.id}: a search for the question's own words reaches its answer first ({lazy['guesses']['top']!r}), "
            "so it is a level-1 task: the first search a solver makes answers it. Put a near-miss where that search "
            "reaches first (ask with the words its stale or wrong value uses, or where the correction is out of that "
            "search's way), or write it as level 1"
        )
    return result | {"panel": lazy, "features": features(world, task.id, result["gold"])}


class Annotation(StrictModel):
    """A fact the finished world already states, registered on the messages that state it: the ledger grows from the
    chatter while the workspace stays as it is. Its channel, author and day are its first message's."""

    id: SafeId
    storyline: SafeId
    subject: NonEmptyText
    attribute: NonEmptyText
    value: NonEmptyText
    anchor: NonEmptyText  # words every one of its messages contains
    message_ids: list[int] = Field(min_length=1)
    supersedes: SafeId | None = None
    decoy: bool = False
    summary: NonEmptyText


def annotate(world, fact: Annotation) -> dict:
    """Register a fact the world states (the forge's world_annotate): checked by the world's rules on insert (each
    message contains the anchor, a supersedes link points forward in time, two values of one subject and attribute
    are a change or a decoy)."""
    if world.db.execute("SELECT 1 FROM facts WHERE id = ?", (fact.id,)).fetchone():
        raise ValueError(f"fact {fact.id} exists: an annotation registers a new fact")
    if (
        fact.supersedes
        and not world.db.execute("SELECT 1 FROM facts WHERE id = ?", (fact.supersedes,)).fetchone()
    ):
        raise ValueError(f"fact {fact.supersedes}, which it supersedes, does not exist")
    marks = ", ".join("?" * len(fact.message_ids))
    stated = world.db.execute(
        f"SELECT id, channel_id, user_id, ts_us FROM messages WHERE id IN ({marks}) AND is_deleted = 0 ORDER BY ts_us",
        fact.message_ids,
    ).fetchall()
    if len(stated) != len(set(fact.message_ids)):
        raise ValueError(f"messages {sorted(set(fact.message_ids) - {r['id'] for r in stated})} do not exist")
    first = stated[0]
    (day,) = world.db.execute(
        "SELECT day FROM calendar WHERE ? >= start_us AND ? < end_us", (first["ts_us"], first["ts_us"])
    ).fetchone()
    with world.batch():
        world.insert(
            "facts",
            [
                dict(
                    id=fact.id,
                    storyline=fact.storyline,
                    subject=fact.subject,
                    attribute=fact.attribute,
                    value=fact.value,
                    anchor=fact.anchor,
                    channel_id=first["channel_id"],
                    author_id=first["user_id"],
                    day=day,
                    is_decoy=int(fact.decoy),
                    summary=fact.summary,
                )
            ],
        )
        world.insert(
            "evidence",
            [
                dict(fact_id=fact.id, message_id=r["id"], role="anchor", anchor_token=fact.anchor)
                for r in stated
            ],
        )
        if fact.supersedes:
            world.insert(
                "fact_relations", [dict(src_fact=fact.id, dst_fact=fact.supersedes, kind="supersedes")]
            )
    return {
        "fact": fact.id,
        "channel_id": first["channel_id"],
        "author_id": first["user_id"],
        "day": day,
        "messages": len(stated),
    }
