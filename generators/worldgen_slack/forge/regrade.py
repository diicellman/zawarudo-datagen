"""A forge's stored tries graded again, by another grader or on golds split into answer and context: each try's grade
replaced, each task's reach, rates and label recomputed from them, and its release published anew (the earlier kept as
release.superseded) when the forge is finished; a forge mid-label keeps the new grades and publishes when it finishes.
No try runs again: a regrade costs one grading call a try.

uv run --frozen worldgen-slack regrade --config configs/worldgen_slack/forge-w2.toml [--splits <folder>] [--dry-run]

The run keeps its own manifest (run.json): the grader comes from the config given (its `grading` and `answer_judge`),
and the regrade is recorded in the run's state and events. `--splits` names a folder of reference batches whose
`split` entries move a part's un-asked details into its context (local-docs/calibration/POLICY.md); a task's gold
before its split is kept beside it."""

import asyncio
import glob
import json
from pathlib import Path
from types import SimpleNamespace

from worldgen_slack.taskset import SolverTask

from ..store import Store
from .config import ForgeConfig
from .env import label_of
from .release import publish
from .tasks import Task
from .tries import Tries, agreeing, by_part, patient, rates, reached

GRADE = (
    "correct", "abstained", "grounded", "unsupported_asides", "committed", "missing", "extra", "reason", "parts",
    "outcome", "semantic_correctness", "evidence_coverage", "part_coverage",
)  # fmt: skip


def splits_of(folder: Path | None) -> dict[str, dict[str, dict]]:
    """Each task's split parts, by label: the reference batches' `split` entries that changed a gold."""
    found = {}
    for path in sorted(glob.glob(str(folder / "batch-*.json"))) if folder else []:
        for row in json.loads(Path(path).read_text()):
            found[row["task_id"]] = {s["part"]: s for s in row.get("split") or [] if s.get("changed")}
    return found


def split(store: Store, splits: dict[str, dict[str, dict]]) -> list[str]:
    """Each split task's parts given their answer and context, the gold before it kept as `gold_before_split`."""
    changed = []
    for task_id, parts in splits.items():
        record = store.state.forged.get(task_id)
        if not record or not parts:
            continue
        task = record["task"]
        record.setdefault("gold_before_split", [dict(p) for p in task["parts"]])
        for part in task["parts"]:
            if (own := parts.get(part["label"])) is not None:
                part["answer"], part["context"] = list(own["answer"]), list(own["context"])
        Task.model_validate(task)  # still a task
        changed.append(task_id)
    return changed


def stored_tries(store: Store) -> list[tuple[str, str, int]]:
    """Every graded try of the forge: (task, book, place)."""
    return [
        (task_id, book, n)
        for book in ("solves", "verifier")
        for task_id, entry in getattr(store.state, book).items()
        if task_id in store.state.forged
        for n, result in enumerate(entry.get("results", []))
        if result and not result.get("crashed") and entry["traces"][n]
    ]


async def regrade(config: ForgeConfig, splits: Path | None = None) -> dict:
    manifest = json.loads((config.output / "run.json").read_text())
    store = Store(config.output, manifest)
    try:
        changed = split(store, splits_of(splits))
        store.save()
        tries = Tries(config, store)
        world, gate = tries.copy(), asyncio.Semaphore(config.solvers)
        todo = stored_tries(store)
        failed = []

        async def one(task_id: str, book: str, n: int) -> None:
            entry = getattr(store.state, book)[task_id]
            result, trace_id = entry["results"][n], entry["traces"][n]
            trace = json.loads((store.root / "traces" / f"{trace_id}.json").read_text())
            task = Task.model_validate(store.state.forged[task_id]["task"])
            solve = tries.solver_task(task, world)
            graded = SimpleNamespace(
                info={"observations": (trace.get("info") or {}).get("observations", [])},
                last_reply=result.get("response") or "",
                id=trace_id,
                record_metrics=lambda metrics: None,
            )
            try:
                async with gate:
                    await patient(lambda: SolverTask.semantic_correctness(solve, graded), timeout=300)
            except Exception as error:
                failed.append(f"{task_id}/{book}/{n}: {type(error).__name__}: {error}"[:300])
                return
            evaluation = graded.info["evaluation"]
            entry["results"][n] = (
                result | {k: evaluation[k] for k in GRADE if k in evaluation} | {"regraded": True}
            )

        await asyncio.gather(*(one(*t) for t in todo))
        labels, unjudged = {}, []
        for task_id, record in store.state.forged.items():
            if record.get("label") is None:
                continue
            task = Task.model_validate(record["task"])
            verified = store.state.verifier.get(task_id, {}).get("results", [])
            record["agree"] = len(agreeing(verified))
            record["reached"] = reached(task, verified)
            record["verified"] = min(record["reached"]) >= 2
            measured = store.state.solves.get(task_id, {}).get("results", [])
            if measured:
                record["rates"], record["parts"] = rates(measured), by_part(task, measured)
            before = record["label"]
            record["label"] = label_of(record, config.solved)
            labels[task_id] = (before, record["label"])
            if record["verified"] and "ruling" not in record:
                unjudged.append(
                    task_id
                )  # verified only now: the task judge never ruled on it, so it stays rejected
        store.state.plans.setdefault("forge", {})["relabelled"] = True
        store.state.plans["forge"]["regraded"] = {
            "grading": config.grading.model_dump(),
            "judge": config.answer_judge.model,
            "splits": str(splits) if splits else None,
            "split_tasks": len(changed),
        }
        moved = {t: f"{a}->{b}" for t, (a, b) in labels.items() if a != b}
        store.event(
            "forge_regraded", tries=len(todo), failed=len(failed), split_tasks=len(changed), moved=moved
        )
        store.save()
        if failed:
            raise RuntimeError(f"{len(failed)} of {len(todo)} tries failed to regrade: {failed[0]}")
        finished = (
            store.state.phase == "done"
        )  # a forge mid-label publishes when it finishes, from these grades
        if finished:
            publish(store)
        return {
            "status": "complete",
            "published": finished,
            "tries": len(todo),
            "split_tasks": len(changed),
            "moved": moved,
            "verified_unjudged": unjudged,
            "labels": {
                k: sum(1 for _, b in labels.values() if b == k)
                for k in ("solved", "window", "unsolved", "rejected")
            },
        }
    finally:
        store.close()


def dry_regrade(config: ForgeConfig, splits: Path | None = None) -> dict:
    """What a regrade would grade, spending nothing."""
    manifest = json.loads((config.output / "run.json").read_text())
    state = json.loads((config.output / "state.json").read_text())
    store = SimpleNamespace(
        state=SimpleNamespace(**{k: state.get(k, {}) for k in ("solves", "verifier", "forged")})
    )
    found = splits_of(splits)
    return {
        "run": manifest.get("config", {}).get("output"),
        "tries": len(stored_tries(store)),
        "split_tasks": sum(1 for t, parts in found.items() if parts and t in state.get("forged", {})),
        "grading": config.grading.model_dump(),
        "judge": config.answer_judge.model,
    }
