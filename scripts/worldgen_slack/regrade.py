"""Regrade a forge's stored tries with the grader and compare with the grades the run recorded: the calibration of a
grader before it takes over.

uv run --frozen python scripts/worldgen_slack/regrade.py --run data/v8-05/forge-w2 --model z-ai/glm-5.3 \
    --set local-docs/calibration/forge-w2/set.json --splits local-docs/calibration/forge-w2/reference \
    --truth local-docs/calibration/forge-w2/reference.json --out local-docs/calibration/forge-w2/runs/<name>.jsonl \
    [--effort high] [--temperature 0] [--prompt FILE] [--tries N] [--jobs 6] [--dry-run]

Each try is graded as the forge and the eval grade it: `SolverTask.semantic_correctness` on a stand-in of its stored
trace (the regrade's `Regrading`, which keeps each paid call). The reader is asked the request, the gold (split into
answer and context by the readers' `--splits`) with what is out of the asker's sight, and the fenced answer; code
decides, a misread is asked again, and grounding is code's. `--prompt` names a prompt file in place of the judge's
own (`JudgeConfig.prompt`): a variant is a file, not a code change. A provider's error is asked again (the forge's
`patient`); any other fails that try. Each grade is written to --out as it lands, with its usage; then the agreement
with the recorded grades, the spend, and with `--truth` the agreement with the calibration's truth, overall, by
stratum and on its random sample. `--set` regrades only a calibration set's tries (the verifier's among them),
`--tries N` a seeded sample that keeps every try the run graded wrong. An --out that exists is gone on with: its grades
stand, and only the tries it lacks (or failed) are graded. A dry run spends nothing."""

import argparse
import asyncio
import glob
import json
import random
from collections import Counter
from pathlib import Path

import verifiers.v1 as vf
from worldgen_slack.taskset import SolverTask

from generators.worldgen_slack.forge.regrade import Regrading
from generators.worldgen_slack.forge.tasks import Task, private_answer
from generators.worldgen_slack.forge.tries import patient, public_task
from generators.worldgen_slack.prices import priced
from generators.worldgen_slack.workspace import Workspace


def outcome(grade: dict) -> str:
    return "right" if grade["correct"] else "abstained" if grade["abstained"] else "wrong"


def kappa(pairs: list[tuple[str, str]]) -> float:
    """Cohen's kappa of two graders' outcomes."""
    n = len(pairs)
    agree = sum(a == b for a, b in pairs) / n
    first, second = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    chance = sum(first[k] * second[k] for k in first) / (n * n)
    return 1.0 if chance == 1 else (agree - chance) / (1 - chance)


def tries(run: Path, seats=("solver",)) -> list[dict]:
    """Every finished try of the run by `seats`, with what its calls showed and how it was graded."""
    found = []
    for path in sorted((run / "traces").glob("*.json")):
        trace = json.loads(path.read_text())
        info, data = trace.get("info") or {}, trace["task"]["data"]
        grade = info.get("evaluation")
        if trace["agent"]["name"] not in seats or not grade or not grade.get("execution_ok"):
            continue
        found.append(
            dict(
                trace=path.stem,
                task=data["task_id"],
                question=data["prompt"],
                response=grade.get("response") or "",
                observations=info.get("observations", []),
                recorded=grade,
            )
        )
    return found


def usage_of(graded: Regrading) -> dict:
    """A grade's calls in one usage: their tokens summed, their cost summed when every call reported one."""
    calls = graded.usage
    total = {
        k: sum(c.get(k) or 0 for c in calls)
        for k in ("prompt_tokens", "completion_tokens", "cached_input_tokens", "reasoning_tokens")
    }
    costs = [c.get("cost") for c in calls]
    return total | {"cost": sum(costs) if costs and None not in costs else None, "calls": len(calls)}


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--effort", help="the grader's reasoning effort (low, medium, high, max)")
    parser.add_argument("--temperature", type=float, help="the grader's sampling temperature")
    parser.add_argument("--prompt", type=Path, help="a prompt file in place of the judge's own")
    parser.add_argument("--set", type=Path, help="a calibration set: regrade only its tries")
    parser.add_argument(
        "--splits", type=Path, help="the readers' gold splits (a folder of reference batches)"
    )
    parser.add_argument("--truth", type=Path, help="the calibration's reference.json, to score against")
    parser.add_argument("--tries", type=int, default=None)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=300, help="seconds one grading call may take")
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Each task's gold, as the forge's tries are graded against it: its answer, a composite request's by part, each
    # part split into answer and context where the readers split it.
    state, world = json.loads((args.run / "state.json").read_text()), Workspace(args.run / "world.sqlite")
    splits = {}
    for path in sorted(glob.glob(str(args.splits / "batch-*.json"))) if args.splits else []:
        for row in json.loads(Path(path).read_text()):
            splits[row["task_id"]] = {s["part"]: s for s in row.get("split") or [] if s.get("changed")}
    tasks, answers = {}, {}
    for task_id, record in state.get("forged", {}).items():
        if "task" in record:
            task = Task.model_validate(record["task"])
            answer = private_answer(world, task)
            own = splits.get(task_id, {})
            parts = [
                p.model_copy(
                    update={
                        "rows": [{"answer": a} for a in own[label]["answer"]],
                        "context": own[label]["context"],
                    }
                )
                if (label := p.label or task.parts[0].label) in own
                else p
                for p in answer.answers()
            ]
            answers[task_id] = answer.model_copy(update={"parts": parts}) if answer.parts else parts[0]
            tasks[task_id] = task
    world.close()
    seat = {"solves": "S", "verifier": "V"}
    named = {}  # trace id -> (task, try)
    for book, letter in seat.items():
        for task_id, entry in state.get(book, {}).items():
            for n, trace_id in enumerate(entry.get("traces", []), 1):
                if trace_id:
                    named[trace_id] = (task_id, f"{letter}{n}")
    wanted = None
    if args.set:
        wanted = {(t["task_id"], x["try"]) for t in json.loads(args.set.read_text()) for x in t["tries"]}
    work = [
        t | {"try": named.get(t["trace"], (None, None))[1]}
        for t in tries(args.run, ("solver", "verifier") if wanted else ("solver",))
        if t["task"] in tasks and (wanted is None or named.get(t["trace"]) in wanted)
    ]
    if args.tries is not None and args.tries < len(work):
        wrong = [t for t in work if not t["recorded"]["correct"]]
        rest = [t for t in work if t["recorded"]["correct"]]
        work = wrong + random.Random(args.seed).sample(rest, max(0, args.tries - len(wrong)))
    # A run stopped midway goes on: the grades --out already holds stand (each was paid for), its failed ones are
    # graded again, and the rest are graded and added. A fresh run takes a fresh --out.
    before = []
    if args.out.exists():
        before = [
            r for r in map(json.loads, args.out.read_text().splitlines()) if not (r["new"] or {}).get("error")
        ]
        held = {(r["task"], r["try"]) for r in before}
        work = [t for t in work if (t["task"], t["try"]) not in held]
    print(
        f"{len(work)} tries ({sum(not t['recorded']['correct'] for t in work)} graded wrong) for {args.model}"
        f" at effort {args.effort or 'default'}, prompt {args.prompt or 'the judge’s own'}"
        + (f"; {len(before)} already graded in {args.out}" if before else "")
    )
    if args.dry_run:
        return

    sampling = vf.SamplingConfig(
        **({"reasoning_effort": args.effort} if args.effort else {}),
        **({"temperature": args.temperature} if args.temperature is not None else {}),
    )
    judge = vf.JudgeConfig(model=args.model, sampling=sampling, prompt=args.prompt)
    solver_world = args.run / "solver.sqlite"  # the solver copy the tries read
    gate = asyncio.Semaphore(args.jobs)

    async def regrade(t):
        if not t["response"].strip():
            return t | {"new": None, "usage": {}}
        task = tasks[t["task"]]
        solve = SolverTask.create(
            public_task(task, solver_world), solver_world, answers[t["task"]], judge, network_policy=False
        )
        graded = Regrading(t["observations"], t["response"], t["trace"], args.model)
        try:
            async with gate:
                await patient(lambda: SolverTask.semantic_correctness(solve, graded), timeout=args.timeout)
            new = graded.info["evaluation"]
        except Exception as error:
            new = {"error": f"{type(error).__name__}: {error}"[:300]}
        return t | {"new": new, "usage": usage_of(graded)}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as stream:
        for row in before:
            stream.write(json.dumps(row) + "\n")

        async def kept(graded):
            """A grade, written as it lands: a run stopped midway keeps every grade it paid for."""
            row = await graded
            stream.write(json.dumps({k: v for k, v in row.items() if k != "observations"}) + "\n")
            stream.flush()
            return row

        graded = before + await asyncio.gather(*(kept(regrade(t)) for t in work))

    done = [g for g in graded if g["new"] and "error" not in g["new"]]
    failed = [g for g in graded if g["new"] and "error" in g["new"]]
    pairs = [(outcome(g["recorded"]), outcome(g["new"])) for g in done]
    wrong = [(a, b) for a, b in pairs if a == "wrong"]
    print(
        f"\ngraded {len(done)} of {len(graded)} tries ({len(graded) - len(done) - len(failed)} empty,"
        f" {len(failed)} failed, {sum(g['new'].get('readings', 1) > 1 for g in done)} read again)"
    )
    for g in failed[:5]:
        print(f"  failed {g['task']} {g['try']}: {g['new']['error']}")
    if pairs:  # nothing to compare when every request failed
        print(
            f"agreement {sum(a == b for a, b in pairs) / len(pairs):.1%}, on the recorded wrong"
            f" {sum(a == b for a, b in wrong) / max(len(wrong), 1):.1%} ({len(wrong)}), kappa {kappa(pairs):.3f}"
        )
    print("recorded → new:", dict(Counter(f"{a}→{b}" for a, b in pairs if a != b)))
    spend = sum(priced(g["usage"], args.model) for g in graded if g["usage"])
    print(
        f"\nspend ${spend:.2f} ({args.model}, prices.py where the gateway gave no cost),"
        f" ${spend / max(len(done), 1):.4f} a grade"
    )
    if args.truth:
        truth = {(r["task_id"], r["try"]): r for r in json.loads(args.truth.read_text())}
        scored = [
            (truth[g["task"], g["try"]], g["new"]["outcome"]) for g in done if (g["task"], g["try"]) in truth
        ]
        for name, sel in (
            [("all", scored)]
            + [
                (k, [x for x in scored if x[0]["stratum"] == k])
                for k in sorted({x[0]["stratum"] for x in scored})
            ]
            + [("the random sample", [x for x in scored if x[0]["picked"] == "sample"])]
        ):
            if sel:
                pairs = [(r["truth"], o) for r, o in sel]
                print(
                    f"  against the truth, {name:18} n={len(sel):3}: outcome {sum(a == b for a, b in pairs) / len(sel):.0%}"
                    f" (κ {kappa(pairs):.2f}), success {sum((a == 'success') == (b == 'success') for a, b in pairs) / len(sel):.0%}"
                )


if __name__ == "__main__":
    asyncio.run(main())
