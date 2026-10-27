"""A forge's release: the solver's copy of its frozen world; each approved task's public row (its family, the set it
came from, how hard it measured, by part) and its private answer (the verified gold, the messages it rests on, a try
that reached it); and the material report, which tells a person writing the next chapter where the
world ran short."""

import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from worldgen_slack import slack
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

from .coverage import coverage
from .tasks import FAMILIES, SETS, Task, private_answer
from .tries import agreeing, by_part, rates

KEPT = ("solved", "window", "unsolved")  # the labels of approved tasks
MEASURED = (
    "tries",
    "crashed",
    "unanswered",
    "right_rate",
    "right_interval",
    "strict_rate",
    "abstain_rate",
    "coverage",
)


def trace_record(store, trace_id: str | None) -> dict | None:
    path = store.root / "traces" / f"{trace_id}.json"
    return json.loads(path.read_text()) if trace_id and path.exists() else None


def level(right: float) -> int:
    """A task's level by its pass rate: 1 easy (most tries right), 4 hard (few)."""
    return 1 if right >= 0.8 else 2 if right >= 0.5 else 3 if right >= 0.2 else 4


def solution(store, task_id: str) -> Solution | None:
    """The first try that reached the task's gold (a grounded one before any other), with its answer and calls, from its
    saved trace: the verifier's, or the student's where the student's tries verified it and no verifier ran."""
    for seat, book in (("verifier", store.state.verifier), ("solver", store.state.solves)):
        entry = book.get(task_id, {})
        reached = [
            (r, t) for r, t in zip(entry.get("results", []), entry.get("traces", [])) if r in agreeing([r])
        ]
        if reached:
            break
    else:
        return None
    result, trace_id = min(
        reached, key=lambda pair: not pair[0].get("grounded")
    )  # the first grounded, else the first
    record = trace_record(store, trace_id) or {}
    calls = (record.get("info") or {}).get("observations", [])
    return Solution(
        seat=seat,
        model=store.settings.get("env", {}).get(seat, {}).get("model"),
        trace_id=trace_id,
        answer=result["response"],
        calls=[{"tool": c["tool"], "arguments": c["arguments"]} for c in calls],
    )


def release_rows(store, world_hash: str) -> tuple[list[PublicTask], dict[str, PrivateAnswer]]:
    """Each approved task's public row and private answer, in the order they were written."""
    rows, answers, env = [], {}, store.settings.get("env", {})
    for task_id, record in store.state.forged.items():
        if record.get("label") not in KEPT:
            continue
        task = Task.model_validate(record["task"])
        tried = store.state.solves[task_id]["results"]
        measured, verified = rates(tried), store.state.verifier.get(task_id, {}).get("results", [])
        done = [r for r in verified if r and not r.get("crashed")]
        rows.append(
            PublicTask(
                task_id=task_id,
                question=task.question,
                actor_id=task.actor_id,
                category=task.family,
                level=level(measured["right_rate"]),
                answer_type="composite" if len(task.parts) > 1 else task.parts[0].answer_type,
                world_hash=world_hash,
                solver=env.get("solver", {}).get("model"),
                **{k: measured[k] for k in MEASURED},
                surface=slack.VERSION,
                turns=env.get("solver", {}).get("max_turns"),
                family=task.family,
                set=record["set"],
                variant_of=task.variant_of,
                verifier=env.get("verifier", {}).get("model"),
                verifier_right=len(agreeing(done)) / len(done) if done else None,
                part_rates=by_part(task, tried),
            )
        )
        answers[task_id] = private_answer(store.world, task).model_copy(
            update={"solution": solution(store, task_id)}
        )
    return rows, answers


def report(store) -> dict:
    """Where the world ran short, for whoever writes the next chapter: per family and set, the tasks written, verified,
    approved and in the window; and the proposer's last notes, on what it looked for."""
    counts: dict[str, Counter] = {f: Counter() for f in FAMILIES}
    for record in store.state.forged.values():
        family, label = record["task"]["family"], record.get("label")
        counts[family]["written"] += 1
        counts[family]["verified"] += bool(record.get("verified"))
        counts[family]["approved"] += label in KEPT
        counts[family]["window"] += label == "window"
        counts[family][record["set"]] += 1
    skills: dict[str, Counter] = defaultdict(Counter)
    for record in store.state.forged.values():
        skill, label = record.get("skill") or "none", record.get("label")
        skills[skill]["written"] += 1
        skills[skill]["verified"] += bool(record.get("verified"))
        skills[skill]["approved"] += label in KEPT
        skills[skill]["window"] += label == "window"
    covered = coverage(store.world, list(store.state.forged.values()))
    # The proposer's notes as its last clean turn kept them.
    notes = store.root / "attempts" / store.state.restore_point / "notes" / "forge.md"
    return {
        "families": {
            f: {k: c[k] for k in ("written", "verified", "approved", "window", *SETS)}
            for f, c in counts.items()
        },
        "skills": {s: dict(c) for s, c in sorted(skills.items())},
        "short": sorted(f for f, c in counts.items() if c["approved"] < 2),
        "dropped": store.state.plans.get("forge", {}).get("dropped", []),
        "askers": covered["askers"],
        "openings": covered["openings"],
        "material": {
            k: covered[k]
            for k in ("storylines", "unused_groups", "unused_by_storyline", "unused_collections")
        }
        | {"unused_promises": len(covered["unused_promises"])},
        "proposer_notes": notes.read_text()[-4000:] if notes.is_file() else "",
    }


def publish(store) -> None:
    """A finished forge's release, and its material report beside it."""
    if store.state.phase != "done":
        raise ValueError("release requires a finished run")
    copy = store.root / "solver.sqlite"
    if not copy.exists():
        store.world.solver_copy(copy)
    atomic_json(store.root / "report.json", report(store))
    rows, answers = release_rows(store, file_hash(copy))
    if rows:  # a forge that approved nothing has no release
        relabelled = store.state.plans.get("forge", {}).get("relabelled", False)
        write_release(store.root / "release", copy, rows, answers, supersede=relabelled)


def write_release(
    root: Path,
    world: Path,
    rows: list[PublicTask],
    answers: dict[str, PrivateAnswer],
    supersede: bool = False,
) -> None:
    """Publish atomically; an existing release must be the same one, unless the forge labelled tasks again after it
    (`supersede`): the earlier release is then kept beside it, as release.superseded."""
    if root.exists():
        _, old_rows, old_answers = load_release(root)
        if (old_rows, old_answers) == (rows, answers) and file_hash(root / "world.sqlite") == file_hash(
            world
        ):
            return
        if not supersede:
            raise ValueError("refusing to overwrite a different release")
        kept = root.with_name(root.name + ".superseded")
        if kept.exists():
            shutil.rmtree(kept)
        os.rename(root, kept)
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
