"""How hard a task is, measured by code beyond where its evidence sits (evidence.py): the lazy solver's panel, its
features, and the wrong answers its grade must reject."""

import json

from ..workspace import Workspace
from ..world.board import answer_facts, places, readable


def stated_values(world, facts: list[str], actor: str) -> dict[str, list[int]]:
    """The values its actor can read stated on the subjects and attributes of a task's facts, each with the messages
    that state it, oldest first."""
    reader, out = Workspace(world.path, actor=actor), {}
    seen = {r[0] for r in reader.db.execute("SELECT id FROM messages")}
    reader.close()
    marks = ", ".join("?" * len(facts))
    for value, message, _ in world.db.execute(
        f"""SELECT lower(trim(f.value)), m.id, m.ts_us FROM facts f JOIN evidence e ON e.fact_id = f.id
        JOIN messages m ON m.id = e.message_id WHERE (lower(trim(f.subject)), lower(trim(f.attribute))) IN
        (SELECT lower(trim(subject)), lower(trim(attribute)) FROM facts WHERE id IN ({marks})) ORDER BY m.ts_us""",
        facts,
    ):
        if message in seen:
            out.setdefault(value, []).append(message)
    return out


def panel(world, task_id: str, rows: list[dict]) -> dict:
    """Lazy solvers, played by code over the ledger as a task's actor can read it (StateMemBench's heuristics,
    2608.19652): the first value stated on its answer's subject and attribute, the latest, the most stated, and the
    first a search for the question's own words reaches. `traps` are the ones that answer wrong. A task the search
    for its own words answers right is a shortcut: in S2 and S3, GLM never missed one of its 20 such tasks, and
    missed 8 of the 17 where that search found a wrong value first."""
    task = world.db.execute(
        "SELECT actor_id, question, answer_type FROM tasks WHERE id = ?", (task_id,)
    ).fetchone()
    facts = [r[0] for r in world.db.execute("SELECT fact_id FROM task_facts WHERE task_id = ?", (task_id,))]
    stated = stated_values(world, facts, task["actor_id"]) if facts else {}
    if not stated:
        return {"guesses": {}, "traps": [], "shortcut": False}
    when = {
        v: world.db.execute(
            f"SELECT MIN(ts_us), MAX(ts_us) FROM messages WHERE id IN ({', '.join('?' * len(ms))})", ms
        ).fetchone()
        for v, ms in stated.items()
    }
    reader = Workspace(world.path, actor=task["actor_id"])
    ranks = {v: reader.rank(task["question"], ms) for v, ms in stated.items()}
    reader.close()
    found = [v for v in stated if ranks[v] is not None]
    guesses = {
        "first": min(stated, key=lambda v: when[v][0]),
        "latest": max(stated, key=lambda v: when[v][1]),
        "most": max(stated, key=lambda v: len(stated[v])),
        "top": min(found, key=lambda v: ranks[v]) if found else None,
    }
    # The value the task rests on: its answer facts' (a hybrid task answers about the message that states it).
    marks = ", ".join("?" * len(facts))
    answered = answer_facts(world, facts, task["actor_id"], [r.get("answer") for r in rows])
    gold = {
        v.strip().casefold()
        for f, v in world.db.execute(f"SELECT id, value FROM facts WHERE id IN ({marks})", facts)
        if f in answered
    }
    if task["answer_type"] == "refusal":  # nothing is the answer: any value given is wrong
        return {
            "guesses": guesses,
            "traps": [k for k, v in guesses.items() if v is not None],
            "shortcut": False,
        }
    traps = [k for k, v in guesses.items() if v is not None and v not in gold]
    return {
        "guesses": guesses,
        "traps": traps,
        "shortcut": task["answer_type"] != "status" and guesses["top"] in gold,
    }


def features(world, task_id: str, rows: list[dict]) -> dict:
    """What a task's structure is, measured by code: how many values its asker can see on its answer's subject and
    attribute (`changes`), where its answer's statement sits (`last_link`: a thread reply, a direct conversation, a
    private channel, or a channel, and later than the first value), how many rows it answers, whether another person
    asking gets another answer (`perspective`), and whether its answer is in no message as written (`derived`)."""
    task = world.db.execute(
        "SELECT actor_id, gold_sql, answer_type FROM tasks WHERE id = ?", (task_id,)
    ).fetchone()
    facts = [r[0] for r in world.db.execute("SELECT fact_id FROM task_facts WHERE task_id = ?", (task_id,))]
    answers = [str(r.get("answer")) for r in rows]
    stated = stated_values(world, facts, task["actor_id"]) if facts else {}
    link = None
    # Where its answer is stated: its answer facts' values, as the panel reads them (a hybrid or a person answer is
    # about the message that states a fact).
    answered = answer_facts(world, facts, task["actor_id"], answers) if facts else []
    values = (
        [
            v.strip().casefold()
            for f, v in world.db.execute(
                f"SELECT id, value FROM facts WHERE id IN ({', '.join('?' * len(answered))})", answered
            )
        ]
        if answered
        else []
    )
    golden = [stated[v] for v in values if v in stated]
    if golden:
        message = world.db.execute(
            "SELECT m.parent_id, c.type, m.ts_us FROM messages m JOIN channels c ON c.id = m.channel_id WHERE m.id = ?",
            (golden[0][0],),
        ).fetchone()
        link = (
            "thread"
            if message["parent_id"]
            else {"im": "dm", "mpim": "dm", "private": "private"}.get(message["type"], "channel")
        )
        first = min(
            world.db.execute(
                f"SELECT MIN(ts_us) FROM messages WHERE id IN ({', '.join('?' * len(ms))})", ms
            ).fetchone()[0]
            for ms in stated.values()
        )
        link += "·later" if message["ts_us"] > first and len(stated) > 1 else ""
    others = []
    for (user,) in world.db.execute(
        "SELECT id FROM users WHERE id != ? AND is_bot = 0 ORDER BY id", (task["actor_id"],)
    ):
        try:
            others.append(
                sorted(str(r.get("answer")) for r in world.gold(user, task["gold_sql"], max_rows=50)["rows"])
            )
        except ValueError:
            continue  # the query returns what that person cannot read
    readable = Workspace(world.path, actor=task["actor_id"])
    verbatim = all(
        readable.db.execute(
            "SELECT 1 FROM messages WHERE instr(lower(text), lower(?)) LIMIT 1", (a,)
        ).fetchone()
        for a in answers
    )
    named = all(world.db.execute("SELECT 1 FROM users WHERE real_name = ?", (a,)).fetchone() for a in answers)
    readable.close()
    return {
        "changes": len(stated),
        "last_link": link,
        "rows": len(rows),
        "perspective": any(o != sorted(answers) for o in others),
        "derived": bool(answers) and not verbatim and not named,
        "status": task["answer_type"] == "status",
    }


def failure_cases(world, task_id: str, rows: list[dict]) -> tuple[str, list[tuple[str, str]]]:
    """A task's gold as a plain answer, and the wrong answers its grade must reject, built by code: each near-miss
    value its actor can read on its answer's subject and attribute, a hedge between the gold and the first of them, a
    set with a row left out, declining to answer, a bare opener (judges accept one 60-90% of the time, 2507.08794),
    and a status given as final. A grade that accepts one, or rejects the gold, is ambiguous as asked
    (Self-Challenging's failure cases, 2506.01716)."""
    task = world.db.execute(
        "SELECT actor_id, answer_type, near_sql FROM tasks WHERE id = ?", (task_id,)
    ).fetchone()
    kind, answers = task["answer_type"], [str(r.get("answer")) for r in rows]
    facts = [r[0] for r in world.db.execute("SELECT fact_id FROM task_facts WHERE task_id = ?", (task_id,))]
    misses = []
    if facts and kind != "refusal":
        golden = {a.strip().casefold() for a in answers}
        for fact in answer_facts(world, facts, task["actor_id"], answers):
            subject, attribute = world.db.execute(
                "SELECT lower(trim(subject)), lower(trim(attribute)) FROM facts WHERE id = ?", (fact,)
            ).fetchone()
            for other, value in world.db.execute(
                "SELECT id, value FROM facts WHERE lower(trim(subject)) = ? AND lower(trim(attribute)) = ? ORDER BY day, id",
                (subject, attribute),
            ):
                sight = any(readable(world, task["actor_id"], c) for c in places(world, other))
                if sight and value.strip().casefold() not in golden and value not in misses:
                    misses.append(value)
    elif facts:  # a refusal: any value its actor can see, given as the answer
        marks = ", ".join("?" * len(facts))
        misses = [
            v
            for f, v in world.db.execute(
                f"SELECT id, value FROM facts WHERE id IN ({marks}) ORDER BY day, id", facts
            )
            if any(readable(world, task["actor_id"], c) for c in places(world, f))
        ]
    gold = ", ".join(answers)
    cases = [(f"the near-miss {value!r}", f"{value}.") for value in misses[:3]]
    for near in json.loads(
        task["near_sql"]
    ):  # a hasty reader's answer, from the gold with a condition relaxed
        relaxed = ", ".join(
            str(r.get("answer")) for r in world.gold(task["actor_id"], near, max_rows=50)["rows"]
        )
        cases.append((f"the near-miss query's answer {relaxed!r}", f"{relaxed or 'Nobody'}."))
    cases.append(("a bare opener", "Thought process:"))
    if kind == "refusal":
        return "I can't answer that from anything you can see: it isn't there.", cases
    if misses:
        cases.append(("a hedge", f"Either {gold} or {misses[0]}; I can't tell which."))
    if kind == "set" and len(answers) > 1:
        cases.append(("a set with a row left out", ", ".join(answers[:-1]) + "."))
    if kind == "status":
        cases.append(("the status given as final", f"{gold}. That is final and confirmed."))
        return f"{gold}, though it is not settled: nothing you can see confirms it.", cases
    cases.append(("declining to answer", "I can't find that in anything you can see."))
    return f"{gold}.", cases
