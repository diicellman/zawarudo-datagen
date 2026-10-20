"""Regrade a forge's stored tries with another answer grader and compare with the grades the run recorded: the
calibration before a grader takes over (P1).

uv run --frozen python scripts/worldgen_slack/regrade.py --run data/v7-06/forge-c1 --model z-ai/glm-5.3 \
    --out runs/regrade-glm.jsonl [--view compact|raw] [--tries N] [--dry-run]

Reads each solver trace (`traces/*.json`: the question, the response, the observations and the recorded grade) and
each task's gold from the forge's state (its answer, each part's; a forge before the simple one is read at its tag). Writes every new grade beside the recorded one, then prints the
agreement on right, abstained and wrong: overall, on the tries the run graded wrong, Cohen's kappa, and every
disagreement. Each grade is written to --out as it lands, and a request that takes longer than --timeout seconds
counts as failed and is tried once more. `--view raw` sends the raw outputs, as the recorded grades had them, to tell the grader from the view.
`--tries N` takes a seeded sample that keeps every try the run graded wrong. A dry run spends nothing.

A calibration (2026-10-20): `--set local-docs/calibration/<run>/set.json` regrades only its tries (the verifier's
among them), `--view current` shows the grader what the run's own grading showed (the environment's `compact`),
`--judge v2` reads the answer and grades it by code (local-docs/calibration/POLICY.md) with the readers' gold splits
(`--splits`, the set's reference/) applied, `--effort` sets the grader's reasoning, and `--truth` (the set's
reference.json) scores every grade against the reference outcome, overall and by stratum."""

import argparse
import asyncio
import glob
import json
import random
from collections import Counter
from pathlib import Path

import verifiers.v1 as vf
from worldgen_slack.judge import AnswerJudge, AnswerReader, reader_reference
from worldgen_slack.judge import graded as grade_reading
from worldgen_slack.slack import compact

from generators.worldgen_slack.forge.tasks import Task, private_answer
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


# FC1's tries were recorded on the tools before Slack's own: a person read by get_user or whoami, and messages with
# time_utc. Grading's view of them, as it was when they were graded, so a regrade sees what the recorded grade saw.
PROFILES = {"read_user_profile": "id", "get_user": "id", "whoami": "user_id"}


def recorded_view(observations: list[dict]) -> dict:
    """What a try's calls showed, each thing once, as grading read the tools of the time (`compact` then)."""

    def items(call: dict) -> list:
        listed = call["output"].get("items")
        listed = listed if isinstance(listed, list) else []
        return listed + [
            c
            for i in listed
            if isinstance(i, dict)
            for side in ("before", "after")
            for c in (i.get("context") or {}).get(side, [])
        ]

    messages, people, channels, members, reactions, calls = {}, {}, {}, {}, {}, []
    for call in observations:
        out, tool = call["output"], call["tool"]
        listed = out.get("items") if isinstance(out.get("items"), list) else []
        calls.append(
            {"tool": tool, "arguments": call["arguments"], "results": len(listed) if "items" in out else 1}
        )
        for item in items(call):
            if isinstance(item, dict) and "ts" in item:
                key = (item["channel"], item["ts"])
                messages[key] = messages.get(key, {}) | {
                    k: v for k, v in item.items() if k not in ("time_utc", "context")
                }
            elif isinstance(item, dict) and "real_name" in item and tool != "list_channel_members":
                people.setdefault(item["id"], item)
            elif isinstance(item, dict) and "id" in item and "real_name" not in item:
                channels.setdefault(item["id"], item)
        if tool == "list_channel_members":
            members.setdefault(call["arguments"]["channel_id"], set()).update(
                i["id"] if isinstance(i, dict) else i for i in listed
            )
        elif tool == "get_reactions":
            reactions[(out["channel"], out["ts"])] = out["reactions"]
        elif tool in PROFILES:
            people[out[PROFILES[tool]]] = out
    for key, read in reactions.items():
        messages.setdefault(key, {"channel": key[0], "ts": key[1]})["reactions"] = read
    return {
        "messages": sorted(messages.values(), key=lambda m: (float(m["ts"]), m["channel"])),
        "people": list(people.values()),
        "channels": list(channels.values()),
        "members": {channel: sorted(ids) for channel, ids in members.items()},
        "calls": calls,
    }


def tries(run: Path, seats=("solver",)) -> list[dict]:
    """Every finished try of the run by `seats`, with what its grader saw and how it graded."""
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


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--view", choices=["compact", "raw", "current"], default="compact")
    parser.add_argument("--judge", choices=["v1", "v2"], default="v1")
    parser.add_argument("--ground", action="store_true", help="v2: the reader also judges grounding")
    parser.add_argument("--effort", help="the grader's reasoning effort (low, high, max)")
    parser.add_argument("--temperature", type=float, help="the grader's sampling temperature")
    parser.add_argument("--set", type=Path, help="a calibration set: regrade only its tries")
    parser.add_argument(
        "--splits", type=Path, help="v2: the readers' gold splits (a folder of reference batches)"
    )
    parser.add_argument("--truth", type=Path, help="the calibration's reference.json, to score against")
    parser.add_argument("--tries", type=int, default=None)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=300, help="seconds one grading request may take")
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Each task's reference, as the forge's tries were graded against it: its answer, a composite request's by part.
    state, world = json.loads((args.run / "state.json").read_text()), Workspace(args.run / "world.sqlite")
    splits = {}
    for path in sorted(glob.glob(str(args.splits / "batch-*.json"))) if args.splits else []:
        for row in json.loads(Path(path).read_text()):
            splits[row["task_id"]] = {s["part"]: s for s in row.get("split") or [] if s.get("changed")}
    tasks, reference, answers = {}, {}, {}
    for task_id, record in state.get("forged", {}).items():
        if "task" in record:
            task = Task.model_validate(record["task"])
            answer = private_answer(world, task)
            own = splits.get(task_id, {})
            parts = [
                p.model_copy(
                    update={
                        "rows": [{"answer": a} for a in own[p.label or task.parts[0].label]["answer"]],
                        "context": own[p.label or task.parts[0].label]["context"],
                    }
                )
                if (p.label or task.parts[0].label) in own
                else p
                for p in answer.answers()
            ]
            answer = answer.model_copy(update={"parts": parts}) if answer.parts else parts[0]
            tasks[task_id], reference[task_id], answers[task_id] = (
                {"question": task.question},
                answer.reference(),
                answer,
            )
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

    def view_of(observations):
        if args.view == "current":
            return compact(observations)
        return recorded_view(observations) if args.view == "compact" else observations

    views = (
        [len(json.dumps(view_of(t["observations"]))) for t in work]
        if args.judge == "v1" or args.ground
        else [0]
    )
    print(
        f"{len(work)} tries ({sum(not t['recorded']['correct'] for t in work)} graded wrong) for {args.model}, {args.view} view: {sum(views) / max(len(views), 1):,.0f} characters of observations a try"
    )
    if args.dry_run:
        return

    sampling = vf.SamplingConfig(
        **({"reasoning_effort": args.effort} if args.effort else {}),
        **({"temperature": args.temperature} if args.temperature is not None else {}),
    )
    config = vf.JudgeConfig(model=args.model, sampling=sampling)
    judge = AnswerReader(config) if args.judge == "v2" else AnswerJudge(config)
    gate = asyncio.Semaphore(args.jobs)

    async def grade(question, ref, response, observations, answer=None):
        """A rate limit waits and tries again (GLM 5.3 takes 8 requests at a time); any other error, a timeout
        included, once more."""
        async with gate:
            failures = 0
            for wait in (2, 4, 8, 16, 32, 60, 60, 60):
                try:
                    async with asyncio.timeout(args.timeout):
                        if args.judge == "v2":
                            fields = {"observations": observations} if args.ground else {}
                            result = await judge.evaluate(
                                question=question,
                                reference=reader_reference(answer),
                                response=response,
                                **fields,
                            )
                            reading = result.parsed
                            grounded = bool(reading.grounded) if args.ground else bool(observations)
                            new, verdict = grade_reading(reading, answer, grounded)
                            parsed = new.model_dump(mode="json") | {
                                "outcome": verdict,
                                "reading": reading.model_dump(mode="json"),
                            }
                        else:
                            result = await judge.evaluate(
                                question=question, reference=ref, response=response, observations=observations
                            )
                            parsed = result.parsed.model_dump(mode="json")
                    usage = result.usage.model_dump(mode="json") if getattr(result, "usage", None) else {}
                    return parsed, usage
                except Exception as error:
                    limited = "429" in repr(error) or "rate" in repr(error).lower()
                    failures += not limited
                    if failures > 1:
                        return {"error": repr(error)}, {}
                    await asyncio.sleep(wait + random.random())
            return {"error": "rate limited past every retry"}, {}

    async def regrade(t):
        view = view_of(t["observations"])
        if not t["response"].strip():
            return t | {"new": None, "usage": {}}
        new, usage = await grade(t["question"], reference[t["task"]], t["response"], view, answers[t["task"]])
        return t | {"new": new, "usage": usage}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as stream:

        async def kept(graded):
            """A grade, written as it lands: a run stopped midway keeps every grade it paid for."""
            row = await graded
            stream.write(json.dumps({k: v for k, v in row.items() if k != "observations"}) + "\n")
            stream.flush()
            return row

        graded = await asyncio.gather(*(kept(regrade(t)) for t in work))

    done = [g for g in graded if g["new"] and "error" not in g["new"]]
    failed = [g for g in graded if g["new"] and "error" in g["new"]]
    pairs = [(outcome(g["recorded"]), outcome(g["new"])) for g in done]
    wrong = [(a, b) for a, b in pairs if a == "wrong"]
    print(
        f"\ngraded {len(done)} of {len(graded)} tries ({len(graded) - len(done) - len(failed)} empty, {len(failed)} failed)"
    )
    if pairs:  # nothing to compare when every request failed
        print(
            f"agreement {sum(a == b for a, b in pairs) / len(pairs):.1%}, on the recorded wrong {sum(a == b for a, b in wrong) / max(len(wrong), 1):.1%} ({len(wrong)}), kappa {kappa(pairs):.3f}"
        )
    print("recorded → new:", dict(Counter(f"{a}→{b}" for a, b in pairs if a != b)))
    for g in done:
        if outcome(g["recorded"]) != outcome(g["new"]):
            print(
                f"  {g['task']} {g['trace'][:8]} {outcome(g['recorded'])}→{outcome(g['new'])}: {g['new']['reason'][:220]}"
            )
    spend = sum(priced(g["usage"], args.model) for g in graded)
    print(
        f"\nspend ${spend:.2f} ({args.model}, prices.py where the gateway gave no cost), ${spend / max(len(done), 1):.4f} a grade"
    )
    if args.truth:
        policy = {"right": "success", "abstained": "fallback", "wrong": "error"}
        truth = {(r["task_id"], r["try"]): r for r in json.loads(args.truth.read_text())}
        scored = [
            (truth[g["task"], g["try"]], g["new"].get("outcome") or policy[outcome(g["new"])])
            for g in done
            if (g["task"], g["try"]) in truth
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
                    f"  against the reference, {name:18} n={len(sel):3}: outcome {sum(a == b for a, b in pairs) / len(sel):.0%}"
                    f" (κ {kappa(pairs):.2f}), success {sum((a == 'success') == (b == 'success') for a, b in pairs) / len(sel):.0%}"
                )


if __name__ == "__main__":
    asyncio.run(main())
