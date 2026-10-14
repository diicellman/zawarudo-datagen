"""Where a task's evidence sits for its asker, measured by code: how many read_channel pages deep and under how many
messages, its search rank for the question's own words, how much of it the question names and how much the asker
cannot read, with what its facts give it. The judge's check reports it, the forge's difficulty builds on it, and a
run's summary lists it for each task."""

import re

from worldgen_slack.slack import PAGE

from .workspace import Workspace
from .world.board import fact_measures, hidden

IDENTIFIER = re.compile(
    r"(?<![\w-])(?!\d{1,2}(?:am|pm)\b)(?=[\w-]*\d)(?=[\w-]*[^\W\d_])\w[\w-]*\w(?![\w-])", re.IGNORECASE
)  # PWSQL-03, CHG-2291, CU12; a time ("5pm") is not one


def named(world, question: str, evidence: list[int]) -> int:
    """How many of the evidence's channel names and identifiers the question names."""
    marks = ", ".join("?" * len(evidence))
    rows = world.db.execute(
        f"SELECT m.text, c.name FROM messages m JOIN channels c ON c.id = m.channel_id WHERE m.id IN ({marks})",
        evidence,
    ).fetchall()
    terms = {n.casefold() for _, n in rows if n} | {
        t.casefold() for text, _ in rows for t in IDENTIFIER.findall(text)
    }
    return sum(bool(re.search(rf"(?<![\w-]){re.escape(t)}(?![\w-])", question, re.IGNORECASE)) for t in terms)


def measures(world, task_id: str, rows: list[dict], tables) -> dict:
    """What makes a task hard, measured for its actor: how many read_channel pages deep its evidence sits and how many
    messages it sits under, the tables its gold query reads, its evidence's best rank when the actor searches the
    question's own words, how many of its evidence's channels and identifiers the question names, how much of its
    evidence the actor cannot read (`unseen`: a refusal's truth out of sight), and what its facts give it
    (`fact_measures`)."""
    task = world.db.execute(
        "SELECT actor_id, question, answer_type FROM tasks WHERE id = ?", (task_id,)
    ).fetchone()
    answers, refusal = [r.get("answer") for r in rows], task["answer_type"] == "refusal"
    evidence = [r["message_id"] for r in rows if r.get("message_id") is not None]
    # Its own evidence: a decoy it lists among its facts is a near-miss to search past, not evidence to find.
    evidence += [
        r[0]
        for r in world.db.execute(
            """SELECT e.message_id FROM task_facts tf JOIN evidence e ON e.fact_id = tf.fact_id
            JOIN facts f ON f.id = tf.fact_id WHERE tf.task_id = ? AND e.role = 'anchor' AND f.is_decoy = 0""",
            (task_id,),
        )
    ]
    reader = Workspace(world.path, actor=task["actor_id"])
    # A refusal may rest on the truth out of its actor's sight: what the actor cannot read is counted, not measured.
    evidence = list(dict.fromkeys(evidence))
    seen = [m for m in evidence if reader.db.execute("SELECT 1 FROM messages WHERE id = ?", (m,)).fetchone()]
    pages = [
        reader.db.execute(
            f"""SELECT COUNT(*) / {PAGE} + 1 FROM messages m, messages e
            WHERE e.id = ? AND m.channel_id = e.channel_id AND m.parent_id IS NULL
            AND m.ts_us > (SELECT ts_us FROM messages WHERE id = COALESCE(e.parent_id, e.id))""",
            (message_id,),
        ).fetchone()[0]
        for message_id in seen
    ]
    under = [
        reader.db.execute(
            """SELECT (SELECT COUNT(*) FROM messages m WHERE m.channel_id = e.channel_id AND m.parent_id IS NULL
            AND m.ts_us > (SELECT ts_us FROM messages WHERE id = COALESCE(e.parent_id, e.id)))
            + (SELECT COUNT(*) FROM messages r WHERE r.parent_id = e.parent_id AND r.ts_us < e.ts_us)
            FROM messages e WHERE e.id = ?""",
            (message_id,),
        ).fetchone()[0]
        for message_id in seen
    ]
    facts = [r[0] for r in world.db.execute("SELECT fact_id FROM task_facts WHERE task_id = ?", (task_id,))]
    (busiest,) = reader.db.execute(
        "SELECT MAX(n) FROM (SELECT COUNT(*) AS n FROM messages WHERE parent_id IS NULL GROUP BY channel_id)"
    ).fetchone()
    return (
        {
            "evidence_pages": max(pages, default=None),
            "depth": max(under, default=0),
            "depth_share": round(max(under, default=0) / busiest, 3) if busiest else 0.0,
            "tables": sorted(tables),
            "bm25_rank": reader.rank(task["question"], seen),
            "named": named(world, task["question"], evidence),
            "unseen": len(evidence) - len(seen),
        }
        | fact_measures(world, facts, task["actor_id"], answers, refusal)
        | ({"hidden": hidden(world, facts, task["actor_id"], answers)} if facts and not refusal else {})
    )
