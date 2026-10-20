"""Prime's bill against our own books: spend by UTC day and model as Prime's resource-spending CSV has it, beside the
same days rebuilt from runs' traces (each call's reported cost, else prices.py), and, for each model with calls that
report no cost, the cached-input rate the bill implies.

uv run --frozen python scripts/worldgen_slack/bill.py references/resource-spending-7_days-2026-10-20.csv \
    data/v8-05/* [--days 2026-10-18 2026-10-19]

A run's traces carry each agent's calls (`usage`) and the answer judge's grades of them (`extra_usage`); a forge's
state carries its task judge's rulings, dated by their `task_labelled` events. A rebuild that runs short of the bill
means calls nobody recorded (a run on another machine, a probe) or a price that is wrong; a model whose calls report
their cost and still run short means the former."""

import argparse
import csv
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from generators.worldgen_slack.prices import DEAREST, PRICES, priced


def read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def day(seconds: float) -> str:
    return datetime.fromtimestamp(seconds, UTC).strftime("%Y-%m-%d")


def calls(run: Path):
    """Every priced call of a run, as (UTC day, model, usage)."""
    config = (read_json(run / "run.json") or {}).get("config") or {}
    grader = (config.get("answer_judge") or {}).get("model")
    for path in (run / "traces").glob("*.json"):
        trace = read_json(path)
        if not trace:
            continue
        start = (trace.get("timing") or {}).get("start")
        if not start:
            continue
        if trace.get("usage"):
            yield day(start), trace["agent"]["config"].get("model"), trace["usage"]
        for usage in trace.get("extra_usage") or []:
            yield day(start), grader, usage
    state = read_json(run / "state.json") or {}
    judge = (config.get("judge") or {}).get("model")
    if state.get("forged") and judge and (run / "progress.jsonl").exists():
        labelled = {}
        for line in (run / "progress.jsonl").read_text().splitlines():
            if '"task_labelled"' in line:
                event = json.loads(line)
                labelled[event["task_id"]] = event["time"]
        for task_id, record in state["forged"].items():
            if record.get("usage") and task_id in labelled:
                yield day(labelled[task_id]), judge, record["usage"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("bill", type=Path, help="Prime's resource-spending CSV")
    parser.add_argument("runs", nargs="+", type=Path, help="run directories whose traces the days hold")
    parser.add_argument("--days", nargs="*", help="the days to show (default: every day of the bill)")
    args = parser.parse_args()
    billed = defaultdict(dict)  # day -> model -> $
    with args.bill.open() as stream:
        for row in csv.DictReader(stream):
            for column, value in row.items():
                if column.startswith("Inference: "):
                    billed[row["Date"]][column.removeprefix("Inference: ")] = float(value or 0)
    # day -> model -> [$ reported by the provider, $ of unreported calls' fresh and output tokens, their cached M]
    books = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0, 0.0]))
    for run in args.runs:
        if not (run / "traces").is_dir():
            continue
        for when, model, usage in calls(run):
            own = books[when][model]
            if usage.get("cost") is not None:
                own[0] += usage["cost"]
            else:
                own[1] += priced({**usage, "cached_input_tokens": 0}, model)
                own[2] += (usage.get("cached_input_tokens") or 0) / 1e6
    print(f"{'day':10}  {'model':32} {'billed':>9} {'rebuilt':>9} {'ratio':>6}  implied cached $/M")
    for when in args.days or sorted(billed):
        for model in sorted(set(billed.get(when, {})) | set(books.get(when, {}))):
            bill = billed.get(when, {}).get(model, 0.0)
            reported, unreported, cached = books.get(when, {}).get(model, [0.0, 0.0, 0.0])
            ours = reported + unreported + cached * PRICES.get(model or "", DEAREST)[1]
            if bill < 0.01 and ours < 0.01:
                continue
            implied = (
                f"{(bill - reported - unreported) / cached:.2f} (on {cached:.0f}M cached)"
                if cached > 1
                else ""
            )
            ratio = f"{ours / bill:.2f}" if bill else "–"
            print(f"{when:10}  {str(model):32} {bill:9.2f} {ours:9.2f} {ratio:>6}  {implied}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
