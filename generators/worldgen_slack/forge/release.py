"""A forge's release: the solver's copy of its finished world, each kept task's public row with how hard it measured,
and its private answer, written atomically."""

import json
import os
import shutil
import tempfile
from collections import Counter
from decimal import Decimal
from pathlib import Path

from worldgen_slack import slack
from worldgen_slack.config import role
from worldgen_slack.judge import EMPTY
from worldgen_slack.release import (
    FILES,
    Manifest,
    PrivateAnswer,
    PublicTask,
    Solution,
    atomic_json,
    load_release,
)
from worldgen_slack.world import file_hash

from ..workspace import Workspace


def miss_class(result: dict, answer_type: str) -> str:
    """How a wrong try went wrong, by code: `budget` (no answer: empty, or out of turns), `abstained` (declined),
    `overclaim` (a status answer's value given as final), `trap` (a refusal answered with a value), `search` (some of
    the evidence its answer rests on never shown) or `saw_all` (shown all of it, and answered otherwise). Whether a
    wrong answer is a reasoning slip or a reading the question allows is the judge's ruling, on every wrong answer."""
    if result.get("unanswered") or result.get("reason") == EMPTY:
        return "budget"
    if result.get("abstained"):
        return "abstained"
    aligned = "missing" in result and not result["missing"] and not result.get("extra")
    if answer_type == "status" and aligned:
        return "overclaim"
    if answer_type == "refusal" and result.get("committed"):
        return "trap"
    coverage = result.get("evidence_coverage")
    return "search" if coverage is not None and coverage < 1 else "saw_all"


def wrong_answers(world, task_id: str, results: list[dict]) -> list[tuple[dict, str]]:
    """A task's wrong tries, crashed ones left out, each with its answer type; a composite request's, each wrong
    part of each, as its own answer type, on its own evidence."""
    kind = world.db.execute("SELECT answer_type FROM tasks WHERE id = ?", (task_id,)).fetchone()[0]
    kinds = [
        k
        for (k,) in world.db.execute(
            "SELECT answer_type FROM tasks WHERE parent = ? ORDER BY part", (task_id,)
        )
    ]
    out = []
    for r in results:
        if r.get("correct") or r.get("crashed"):
            continue
        if kind != "composite" or len(r.get("parts") or []) != len(kinds):  # a grade that failed has no parts
            out.append((r, kind))
            continue
        cover = r.get("part_coverage") or [None] * len(kinds)
        for part, k, c in zip(r["parts"], kinds, cover):
            if not part["correct"]:
                own = part | {key: r.get(key) for key in ("unanswered", "reason")} | {"evidence_coverage": c}
                out.append((own, k))
    return out


def miss_classes(world, task_id: str, results: list[dict]) -> Counter:
    """How a task's wrong tries went wrong (`miss_class`), by each wrong answer (`wrong_answers`)."""
    return Counter(miss_class(r, k) for r, k in wrong_answers(world, task_id, results))


def miss_shape(result: dict) -> str | None:
    """What a wrong answer's alignment with its reference shows: `short` (it lacks a row and adds none: an
    incomplete aggregation), `extra` (it adds a value and lacks none), `wrong` (another value in place of one), or
    None (aligned yet wrong: a status given as final, a refusal answered)."""
    missing, extra = bool(result.get("missing")), bool(result.get("extra"))
    return {(True, False): "short", (False, True): "extra", (True, True): "wrong"}.get((missing, extra))


def miss_shapes(world, task_id: str, results: list[dict]) -> Counter:
    return Counter(
        shape for r, _ in wrong_answers(world, task_id, results) if (shape := miss_shape(r)) is not None
    )


def right_by_part(world, task_id: str, results: list[dict]) -> dict[str, float] | None:
    """A composite request's share of right answers by part, over its graded tries: which of its parts the solver
    misses. None for a task with no parts, or no graded try."""
    labels = [
        label
        for (label,) in world.db.execute("SELECT label FROM tasks WHERE parent = ? ORDER BY part", (task_id,))
    ]
    graded = [
        r["parts"]
        for r in results
        if labels and not r.get("crashed") and len(r.get("parts") or []) == len(labels)
    ]
    if not graded:
        return None
    return {label: sum(g[n]["correct"] for g in graded) / len(graded) for n, label in enumerate(labels)}


def solution(store, task_id: str) -> Solution | None:
    """A try that reached a task's gold from Slack alone, right and grounded: the solver's first, else the witness's
    of the same version of the task, with its answer and its calls from its saved trace. None when no try did: such a
    task's gold is the ledger's word alone, and it is not kept."""
    state, env = store.state, store.settings.get("env", {})
    solves = state.solves.get(task_id, {})
    witness = state.witness.get(task_id, {})
    books = [("solver", solves)] + ([("witness", witness)] if witness.get("key") == solves.get("key") else [])
    for seat, book in books:
        for result, trace_id in zip(book.get("results", []), book.get("traces", [])):
            path = store.root / "traces" / f"{trace_id}.json"
            if result and result.get("semantic_correctness") == 1.0 and path.exists():
                calls = json.loads(path.read_text())["info"].get("observations", [])
                return Solution(
                    seat=seat,
                    model=env.get(seat, {}).get("model"),
                    trace_id=trace_id,
                    answer=result["response"],
                    calls=[{"tool": c["tool"], "arguments": c["arguments"]} for c in calls],
                )
    return None


def private_answer(store, task) -> PrivateAnswer:
    """A task's private answer: its gold rows, and the messages and people they rest on; a refusal's truth out of
    its actor's sight goes apart, as `unseen`. The messages an answer rests on are its gold rows' and those stating
    its answer facts: the task's facts its actor can read that are no decoy and that no other of them supersedes. A
    stale value or a decoy beside them is not its evidence. A composite request's answer is its parts', each under
    its label."""
    db = store.world.db
    if task["answer_type"] == "composite":
        parts = [
            private_answer(store, p).model_copy(update={"label": p["label"]})
            for p in db.execute(
                "SELECT * FROM tasks WHERE parent = ? ORDER BY part", (task["id"],)
            ).fetchall()
        ]

        def together(field: str) -> list:
            return sorted({tuple(m) for p in parts for m in getattr(p, field)}, key=lambda m: Decimal(m[1]))

        return PrivateAnswer(
            answer_type="composite",
            rows=[],
            gold_sql="",
            messages=[list(m) for m in together("messages")],
            unseen=[list(m) for m in together("unseen")],
            users=sorted({u for p in parts for u in p.users}),
            parts=parts,
        )
    gold = json.loads(task["gold_json"])
    anchors = {
        (f, m)
        for f, m in db.execute(
            """SELECT e.fact_id, e.message_id FROM task_facts tf JOIN evidence e ON e.fact_id = tf.fact_id
            WHERE tf.task_id = ?""",
            (task["id"],),
        )
    }
    stating = {m for _, m in anchors} | {r["message_id"] for r in gold if r.get("message_id") is not None}
    marks = ", ".join("?" * len(stating))
    reader = Workspace(store.world.path, actor=task["actor_id"])
    readable = {
        r[0] for r in reader.db.execute(f"SELECT id FROM messages WHERE id IN ({marks})", sorted(stating))
    }
    reader.close()
    decoys = {
        f
        for (f,) in db.execute(
            "SELECT f.id FROM task_facts tf JOIN facts f ON f.id = tf.fact_id WHERE tf.task_id = ? AND f.is_decoy = 1",
            (task["id"],),
        )
    }
    seen = {f for f, m in anchors if m in readable} - decoys
    stale = {
        d
        for s, d in db.execute("SELECT src_fact, dst_fact FROM fact_relations WHERE kind = 'supersedes'")
        if s in seen
    }
    ids = {m for f, m in anchors if f in seen - stale} | {
        r["message_id"] for r in gold if r.get("message_id") is not None
    }
    stated = db.execute(
        f"SELECT id, channel_id, ts FROM messages WHERE id IN ({marks}) ORDER BY ts_us",
        sorted(stating),
    ).fetchall()
    messages = [[r["channel_id"], r["ts"]] for r in stated if r["id"] in readable and r["id"] in ids]
    unseen = [[r["channel_id"], r["ts"]] for r in stated if r["id"] not in readable]
    return PrivateAnswer(
        answer_type=task["answer_type"],
        rows=gold,
        gold_sql=task["gold_sql"],
        messages=messages,
        unseen=unseen,
        users=sorted({r["user_id"] for r in gold if r.get("user_id") is not None}),
    )


def release_rows(store, world_hash: str) -> tuple[list[PublicTask], dict[str, PrivateAnswer]]:
    """Each task's public row, with how hard it measured, and its private answer (`private_answer`)."""
    rows, answers = [], {}
    for task in store.world.db.execute("SELECT * FROM tasks WHERE parent IS NULL ORDER BY id").fetchall():
        review, env = store.state.task_reviews.get(task["id"]), store.settings.get("env", {})
        measured = (
            {}
            if review is None
            else {
                k: review.get(k)
                for k in (
                    "tries",
                    "crashed",
                    "unanswered",
                    "right_rate",
                    "right_interval",
                    "strict_rate",
                    "abstain_rate",
                    "coverage",
                    "witness_right",
                )
            }
            | {
                "solver": env.get("solver", {}).get("model"),
                "witness": env.get("witness", {}).get("model")
                if review.get("witness_right") is not None
                else None,
            }
        )
        if review is not None:
            measured |= {
                "surface": slack.VERSION,
                "turns": env.get("solver", {}).get("max_turns") or role("", solver=True).max_turns,
                "miss_classes": dict(miss_classes(store.world, task["id"], review.get("results", []))),
            }
        rows.append(
            PublicTask(
                task_id=task["id"],
                question=task["question"],
                actor_id=task["actor_id"],
                category=task["category"],
                level=task["level"],
                answer_type=task["answer_type"],
                world_hash=world_hash,
                intent=task["intent"],
                family=task["family"],
                **measured,
            )
        )
        answers[task["id"]] = private_answer(store, task).model_copy(
            update={"solution": solution(store, task["id"])}
        )
    return rows, answers


def publish(store) -> None:
    """A finished forge's release: the solver's copy of its world, its public tasks and their private answers."""
    if store.state.phase != "done":
        raise ValueError("release requires a finished run")
    if missing := [
        t
        for (t,) in store.world.db.execute("SELECT id FROM tasks WHERE parent IS NULL")
        if t not in store.state.task_reviews
    ]:
        raise ValueError(f"release requires an approved review of every task; missing {missing}")
    if unsolved := [
        t
        for (t,) in store.world.db.execute("SELECT id FROM tasks WHERE parent IS NULL")
        if solution(store, t) is None
    ]:
        raise ValueError(f"release requires a right and grounded try of every task; none of {unsolved}")
    if store.world.violations(complete=True):
        raise ValueError("release requires a world that breaks no rule")
    copy = store.root / "solver.sqlite"
    copy.unlink(missing_ok=True)
    store.world.solver_copy(copy)
    rows, answers = release_rows(store, file_hash(copy))
    write_release(store.root / "release", copy, rows, answers)


def write_release(root: Path, world: Path, rows: list[PublicTask], answers: dict[str, PrivateAnswer]) -> None:
    """Publish atomically; an existing release must be the same one."""
    if root.exists():
        _, old_rows, old_answers = load_release(root)
        if (old_rows, old_answers) != (rows, answers) or file_hash(root / "world.sqlite") != file_hash(world):
            raise ValueError("refusing to overwrite a different release")
        return
    root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root.parent, prefix=".release-") as temporary:
        stage = Path(temporary) / "release"
        stage.mkdir()
        shutil.copyfile(world, stage / "world.sqlite")
        atomic_json(stage / "tasks.json", [r.model_dump(mode="json") for r in rows])
        atomic_json(stage / "answers.json", {k: v.model_dump(mode="json") for k, v in answers.items()})
        files = {name: file_hash(stage / name) for name in sorted(FILES)}
        atomic_json(
            stage / "manifest.json", Manifest(world_hash=files["world.sqlite"], files=files).model_dump()
        )
        if load_release(stage)[1:] != (rows, answers):
            raise ValueError("published data differs from approved inputs")
        os.rename(stage, root)
