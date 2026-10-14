"""Regrade a forge's stored tries with another answer grader and compare with the grades the run recorded: the
calibration before a grader takes over (P1).

uv run --frozen python scripts/worldgen_slack/regrade.py --run data/v7-06/forge-c1 --model z-ai/glm-5.3 \
    --out runs/regrade-glm.jsonl [--view compact|raw] [--tries N] [--screen] [--dry-run]

Reads each solver trace (`traces/*.json`: the question, the response, the observations and the recorded grade) and
each task's gold from the run's `world.sqlite`. Writes every new grade beside the recorded one, then prints the
agreement on right, abstained and wrong: overall, on the tries the run graded wrong, Cohen's kappa, and every
disagreement. Each grade is written to --out as it lands, and a request that takes longer than --timeout seconds
counts as failed and is tried once more. `--view raw` sends the raw outputs, as the recorded grades had them, to tell the grader from the view.
`--screen` also regrades each task's failure cases, which a grader must all judge wrong, and its gold, which it must
judge right. `--tries N` takes a seeded sample that keeps every try the run graded wrong. A dry run spends nothing."""

import argparse
import asyncio
import json
import random
import sqlite3
from collections import Counter
from pathlib import Path

import verifiers.v1 as vf
from worldgen_slack.judge import AnswerJudge

from generators.worldgen_slack.forge.difficulty import failure_cases
from generators.worldgen_slack.workspace import Workspace

# $ per million input, output and cached input tokens (list prices)
PRICES = {"z-ai/glm-5.3": (1.4, 4.4, 0.26)}


def outcome(grade: dict) -> str:
    return "right" if grade["correct"] else "abstained" if grade["abstained"] else "wrong"


def kappa(pairs: list[tuple[str, str]]) -> float:
    """Cohen's kappa of two graders' outcomes."""
    n = len(pairs)
    agree = sum(a == b for a, b in pairs) / n
    first, second = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    chance = sum(first[k] * second[k] for k in first) / (n * n)
    return 1.0 if chance == 1 else (agree - chance) / (1 - chance)


def cost(model: str, usage: dict) -> float:
    if usage.get("cost") is not None:
        return usage["cost"]
    fresh, out, cached = PRICES.get(model, (0, 0, 0))
    hit = usage.get("cached_input_tokens") or 0
    return (
        (usage.get("prompt_tokens", 0) - hit) * fresh + hit * cached + usage.get("completion_tokens", 0) * out
    ) / 1e6


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


def tries(run: Path) -> list[dict]:
    """Every finished solver try of the run, with what its grader saw and how it graded."""
    found = []
    for path in sorted((run / "traces").glob("*.json")):
        trace = json.loads(path.read_text())
        info, data = trace.get("info") or {}, trace["task"]["data"]
        grade = info.get("evaluation")
        if trace["agent"]["name"] != "solver" or not grade or not grade.get("execution_ok"):
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
    parser.add_argument("--view", choices=["compact", "raw"], default="compact")
    parser.add_argument("--tries", type=int, default=None)
    parser.add_argument("--screen", action="store_true")
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=300, help="seconds one grading request may take")
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # A task the run dropped is gone from its world; its gold is in the round's review snapshots.
    tasks, kept = {}, set()
    for path in [*sorted(args.run.glob("attempts/round-*/review-tasks-*.sqlite")), args.run / "world.sqlite"]:
        db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        db.row_factory = sqlite3.Row
        rows = {
            r["id"]: dict(r) for r in db.execute("SELECT id, question, answer_type, gold_json FROM tasks")
        }
        tasks |= rows
        kept = set(rows)
        db.close()
    reference = {
        t: {"answer_type": r["answer_type"], "rows": json.loads(r["gold_json"])} for t, r in tasks.items()
    }
    work = [t for t in tries(args.run) if t["task"] in tasks]
    if args.tries is not None and args.tries < len(work):
        wrong = [t for t in work if not t["recorded"]["correct"]]
        rest = [t for t in work if t["recorded"]["correct"]]
        work = wrong + random.Random(args.seed).sample(rest, max(0, args.tries - len(wrong)))
    cases = []
    if args.screen:
        world = Workspace(args.run / "world.sqlite")
        for t in sorted(kept):
            gold, failures = failure_cases(world, t, reference[t]["rows"])
            cases += [dict(task=t, case="gold", response=gold, expect=True)]
            cases += [dict(task=t, case=name, response=response, expect=False) for name, response in failures]
        world.close()
    views = [
        len(json.dumps(recorded_view(t["observations"]) if args.view == "compact" else t["observations"]))
        for t in work
    ]
    print(
        f"{len(work)} tries ({sum(not t['recorded']['correct'] for t in work)} graded wrong) and {len(cases)} screen cases for {args.model}, {args.view} view: {sum(views) / max(len(views), 1):,.0f} characters of observations a try"
    )
    if args.dry_run:
        return

    judge, gate = AnswerJudge(vf.JudgeConfig(model=args.model)), asyncio.Semaphore(args.jobs)

    async def grade(question, ref, response, observations):
        """A rate limit waits and tries again (GLM 5.3 takes 8 requests at a time); any other error, a timeout
        included, once more."""
        async with gate:
            failures = 0
            for wait in (2, 4, 8, 16, 32, 60, 60, 60):
                try:
                    async with asyncio.timeout(args.timeout):
                        result = await judge.evaluate(
                            question=question, reference=ref, response=response, observations=observations
                        )
                    usage = result.usage.model_dump(mode="json") if getattr(result, "usage", None) else {}
                    return result.parsed.model_dump(mode="json"), usage
                except Exception as error:
                    limited = "429" in repr(error) or "rate" in repr(error).lower()
                    failures += not limited
                    if failures > 1:
                        return {"error": repr(error)}, {}
                    await asyncio.sleep(wait + random.random())
            return {"error": "rate limited past every retry"}, {}

    async def regrade(t):
        view = recorded_view(t["observations"]) if args.view == "compact" else t["observations"]
        if not t["response"].strip():
            return t | {"new": None, "usage": {}}
        new, usage = await grade(t["question"], reference[t["task"]], t["response"], view)
        return t | {"new": new, "usage": usage}

    async def rescreen(c):
        new, usage = await grade(tasks[c["task"]]["question"], reference[c["task"]], c["response"], [])
        return c | {"new": new, "usage": usage}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as stream:

        async def kept(graded):
            """A grade, written as it lands: a run stopped midway keeps every grade it paid for."""
            row = await graded
            stream.write(json.dumps({k: v for k, v in row.items() if k != "observations"}) + "\n")
            stream.flush()
            return row

        graded = await asyncio.gather(*(kept(regrade(t)) for t in work))
        screened = await asyncio.gather(*(kept(rescreen(c)) for c in cases))

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
    if screened:
        lost = [s for s in screened if "error" in s["new"]]
        flips = [s for s in screened if "error" not in s["new"] and s["new"]["correct"] != s["expect"]]
        print(f"\nscreen: {len(screened)} cases, {len(lost)} failed, {len(flips)} judged against expectation")
        for s in flips:
            print(f"  {s['task']} {s['case']}: {s['new']['reason'][:220]}")
    spend = sum(cost(args.model, g["usage"]) for g in [*graded, *screened])
    print(f"\nspend ${spend:.2f} ({args.model}, list prices where the gateway gave no cost)")


if __name__ == "__main__":
    asyncio.run(main())
