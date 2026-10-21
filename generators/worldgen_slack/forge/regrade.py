"""A forge's stored tries graded again, by another grader or on golds split into answer and context: each try's grade
replaced, each task's reach, rates and label recomputed from them, and its release published anew (the earlier kept as
release.superseded) when the forge is finished; a forge mid-label keeps the new grades and publishes when it finishes.
No try runs again: a regrade costs one grading call a try.

uv run --frozen worldgen-slack regrade --config configs/worldgen_slack/forge-w2.toml [--splits <folder>] [--dry-run]

The run keeps its own manifest (run.json): the grader comes from the config given (its `answer_judge`, whose `prompt`
may name another prompt file), and the regrade is recorded in the run's state and events. `--splits` names a folder of reference batches whose
`split` entries move a part's un-asked details into its context (local-docs/calibration/POLICY.md); a task's gold
before its split is kept beside it."""

import asyncio
import glob
import json
import random
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from worldgen_slack.taskset import SolverTask

from ..prices import priced
from ..store import Store
from .config import ForgeConfig
from .env import label_of
from .release import publish
from .tasks import Task
from .tries import Tries, agreeing, by_part, patient, rates, reached

GRADE = (
    "correct", "abstained", "grounded", "unsupported_asides", "committed", "missing", "extra", "reason", "parts",
    "outcome", "readings", "semantic_correctness", "evidence_coverage", "part_coverage",
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


def grader(config: ForgeConfig) -> str:
    """Which grader grades: its model and its prompt file (the default's, or the one `answer_judge.prompt` names), the
    mark a try regraded by it carries, so a regrade by another prompt on the same model grades it again."""
    prompt = config.answer_judge.prompt
    return f"{config.answer_judge.model}:{prompt.name if prompt else 'judge_prompt.md'}"


FAIL_FAST = 20  # grades that fail before any lands, and the regrade stops: a bug, not a provider's burst
SAVE_EVERY = 50  # grades between a regrade's saves and progress events
PER_GRADE = (
    0.02  # $ a regrade may spend on a try unless told otherwise: about twice grader v2's measured $0.0105
)


class Regrading:
    """A stored try as the grader gets it again: what its calls showed and its answer. It keeps each grading call the
    grader records on it once the call is paid, as a trace does: a stand-in without that record made every grade of
    forge-w2's first regrade fail after its call was billed, four times over ($166, 2026-10-21)."""

    def __init__(self, observations: list, reply: str, trace_id: str, model: str):
        self.info = {"observations": observations}
        self.last_reply, self.id, self.model = reply, trace_id, model
        self.usage = []

    def record_metrics(self, metrics: dict) -> None:
        pass

    def record_judge_call(self, *, name: str, request: dict, response) -> None:
        if response.usage is not None:
            self.usage.append(response.usage.model_dump(mode="json"))

    @property
    def spend(self) -> float:
        return sum(priced(u, self.model) for u in self.usage)


async def regrade(
    config: ForgeConfig,
    splits: Path | None = None,
    sample: int | None = None,
    budget: float | None = None,
    seed: int = 20261021,
) -> dict:
    """See the module. `sample` grades that many tries at random and changes nothing: a live check of the grader, its
    cost and its outcomes before a whole regrade. A regrade stops when its first FAIL_FAST grades all fail, or its
    measured spend passes `budget` (PER_GRADE a try by default); it keeps what it graded, saved every SAVE_EVERY grades,
    and a run again skips the tries this grader has already regraded."""
    manifest = json.loads((config.output / "run.json").read_text())
    store = Store(config.output, manifest)
    try:
        changed = split(store, splits_of(splits))
        mark = grader(config)
        todo = [
            t
            for t in stored_tries(store)
            if getattr(store.state, t[1])[t[0]]["results"][t[2]].get("regraded") != mark
        ]
        if sample:
            todo = random.Random(seed).sample(todo, min(sample, len(todo)))
        else:
            store.save()  # the gold splits, before any grade
        budget = budget if budget is not None else PER_GRADE * len(todo)
        tries = Tries(config, store)
        world, gate = tries.copy(), asyncio.Semaphore(config.solvers)
        done, failed, outcomes, spent, reread, stop = [], [], Counter(), [0.0], [0], []

        async def one(task_id: str, book: str, n: int) -> None:
            if stop:
                return
            entry = getattr(store.state, book)[task_id]
            result, trace_id = entry["results"][n], entry["traces"][n]
            trace = json.loads((store.root / "traces" / f"{trace_id}.json").read_text())
            solve = tries.solver_task(Task.model_validate(store.state.forged[task_id]["task"]), world)
            graded = Regrading(
                (trace.get("info") or {}).get("observations", []),
                result.get("response") or "",
                trace_id,
                config.answer_judge.model,
            )
            try:
                async with gate:
                    if stop:
                        return
                    await patient(lambda: SolverTask.semantic_correctness(solve, graded), timeout=300)
            except Exception as error:
                failed.append(f"{task_id}/{book}/{n}: {type(error).__name__}: {error}"[:300])
            else:
                evaluation = graded.info["evaluation"]
                outcomes[evaluation.get("outcome")] += 1
                reread[0] += evaluation.get("readings", 1) > 1  # the reader misread it first: asked again
                done.append((task_id, book, n))
                if not sample:
                    entry["results"][n] = (
                        result | {k: evaluation[k] for k in GRADE if k in evaluation} | {"regraded": mark}
                    )
            finally:
                spent[0] += graded.spend
            if not stop and not done and len(failed) >= FAIL_FAST:
                stop.append(f"its first {len(failed)} grades all failed: {failed[0]}")
            if not stop and spent[0] > budget:
                stop.append(f"it spent ${spent[0]:.2f}, past its ${budget:.2f} budget")
            if not sample and (len(done) + len(failed)) % SAVE_EVERY == 0:
                store.event(
                    "regrade_progress",
                    graded=len(done),
                    failed=len(failed),
                    spent=round(spent[0], 2),
                    of=len(todo),
                )
                store.save()

        await asyncio.gather(*(one(*t) for t in todo))
        measured = {
            "graded": len(done),
            "failed": len(failed),
            "spent": round(spent[0], 4),
            "per_grade": round(spent[0] / max(len(done) + len(failed), 1), 4),
            "outcomes": dict(outcomes),
            "reread": reread[0],
        }
        if sample:
            return {"status": "sampled", **measured, "failures": failed[:5]}
        store.save()
        if stop:
            store.event("regrade_stopped", why=stop[0], **measured)
            raise RuntimeError(f"the regrade stopped: {stop[0]} ({measured})")
        if failed:
            raise RuntimeError(
                f"{len(failed)} of {len(todo)} tries failed to regrade, the rest kept (run again to finish): {failed[0]}"
            )
        labels, unjudged = {}, []
        for task_id, record in store.state.forged.items():
            if record.get("label") is None:
                continue
            task = Task.model_validate(record["task"])
            verified = store.state.verifier.get(task_id, {}).get("results", [])
            record["agree"] = len(agreeing(verified))
            record["reached"] = reached(task, verified)
            record["verified"] = min(record["reached"]) >= 2
            measured_rates = store.state.solves.get(task_id, {}).get("results", [])
            if measured_rates:
                record["rates"], record["parts"] = rates(measured_rates), by_part(task, measured_rates)
            before = record["label"]
            record["label"] = label_of(record, config.solved)
            labels[task_id] = (before, record["label"])
            if record["verified"] and "ruling" not in record:
                unjudged.append(
                    task_id
                )  # verified only now: the task judge never ruled on it, so it stays rejected
        store.state.plans.setdefault("forge", {})["relabelled"] = True
        store.state.plans["forge"]["regraded"] = {
            "grader": mark,
            "splits": str(splits) if splits else None,
            "split_tasks": len(changed),
            **measured,
        }
        moved = {t: f"{a}->{b}" for t, (a, b) in labels.items() if a != b}
        store.event("forge_regraded", tries=len(todo), split_tasks=len(changed), moved=moved, **measured)
        store.save()
        finished = (
            store.state.phase == "done"
        )  # a forge mid-label publishes when it finishes, from these grades
        if finished:
            publish(store)
        return {
            "status": "complete",
            "published": finished,
            "tries": len(todo),
            **measured,
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
        "grader": grader(config),
        "budget": round(PER_GRADE * len(stored_tries(store)), 2),
    }
