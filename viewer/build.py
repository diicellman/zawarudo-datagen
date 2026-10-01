"""Export a worldgen run as a standalone, offline HTML page: lineage, world, tasks, cast and agent traces."""

import argparse
import json
from pathlib import Path

from worldgen_slack.slack.api import digest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
QUIET = {"heartbeat", "agent_progress", "author_started"}
DISCOVERED = ("conversation_id", "root_message_id", "user_id", "author_id", "cursor")


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else None


def artifact(run, reference):
    return read_json(run / "artifacts" / f"{reference}.json") if isinstance(reference, str) else reference


def summarize(record, full):
    """One agent run: role, what it worked on, when, and what it cost; `full` keeps its messages."""
    data, info, usage = record["task"]["data"], record.get("info") or {}, record.get("usage") or {}
    timing = record.get("timing") or {}
    row = {
        "id": record["id"],
        "role": record["agent"]["name"],
        "model": record["agent"]["config"].get("model"),
        "attempt": data.get("candidate_id") or None,
        "task_id": data.get("task_id") or None,
        "scene_id": info.get("scene_id"),
        "start": timing.get("start"),
        "end": (timing.get("finalize") or {}).get("end") or (timing.get("agent") or {}).get("end"),
        "ok": record.get("ok"),
        "calls": len(record.get("calls") or []),
        "input_tokens": usage.get("prompt_tokens"),
        "cached_tokens": usage.get("cached_input_tokens"),
        "output_tokens": usage.get("completion_tokens"),
        "cost": usage.get("cost"),
    }
    if "evaluation" in info:
        row["evaluation"] = info["evaluation"]
        row["route"] = [
            {
                "call": o["call"],
                "messages": [
                    i["message_id"] for i in (o.get("output") or {}).get("items", []) if "message_id" in i
                ],
            }
            for o in info.get("observations", [])
        ]
    if "verdict" in info:
        row["approved"] = info["verdict"]["approved"]
    if full:
        row["messages"] = [
            {k: n["message"].get(k) for k in ("role", "content", "tool_calls")} for n in record["nodes"]
        ]
    return row


def chain(binding, outputs):
    """The gold route step by step: what each call returns, which results are bound evidence, and which
    returned IDs a later call uses."""
    evidence = {m for claim in binding["claims"] for m in claim["message_ids"]}
    steps = []
    for call, output in zip(binding["gold_calls"], outputs):
        items = output.get("items", [output] if "user_id" in output else [])
        text = json.dumps(output)
        steps.append(
            {
                "call": call,
                "returned": len(items),
                "evidence": [i["message_id"] for i in items if i.get("message_id") in evidence],
                "text": text,
            }
        )
    for index, step in enumerate(steps):
        later = [(name, s["call"]["arguments"].get(name)) for s in steps[index + 1 :] for name in DISCOVERED]
        step["feeds"] = sorted(
            {f"{name}={value}" for name, value in later if value and str(value) in step["text"]}
        )
    for step in steps:
        del step["text"]
    return steps


def load_run(path, full=False):
    """Everything the page shows, read from the run's saved files."""
    run = path.resolve()
    state = read_json(run / "state.json")
    if state is None:
        raise ValueError(f"No run checkpoint in {run}")
    catalog, candidate = artifact(run, state["catalog"]), artifact(run, state["candidate"])
    events = [json.loads(line) for line in (run / "progress.jsonl").read_text().splitlines()]
    order = {e["attempt"]: i for i, e in reversed(list(enumerate(events))) if e.get("attempt")}
    events = [e for e in events if e["event"] not in QUIET]
    traces = [summarize(read_json(p), full) for p in (run / "traces").glob("*.json")]
    snapshots, attempts, gold = {}, [], {}
    folders = sorted(
        (p for p in (run / "attempts").iterdir() if p.is_dir()),
        key=lambda p: (order.get(p.name, 1e9), p.name),
    )
    for folder in folders:
        files = {p.stem: read_json(p) for p in folder.glob("*.json")}
        world = (files.get("review_input") or files.get("review_input-world") or {}).get("candidate")
        key = digest(world["snapshot"])[:12] if world else None
        if world:
            snapshots[key] = world["snapshot"]
        validation = files.get("validation") or {}
        for task_id, outputs in validation.get("gold_outputs", {}).items():
            gold[task_id] = outputs
        labels = sorted(name.removeprefix("verdict-") for name in files if name.startswith("verdict-"))
        attempts.append(
            {
                "id": folder.name,
                "snapshot": key,
                "verdict": files.get("verdict"),
                "acceptance": files.get("acceptance"),
                "validation": {k: validation.get(k) for k in ("ok", "errors", "counts")}
                if validation
                else None,
                "corrections": files.get("corrections"),
                "reviews": {
                    label: {
                        "verdict": files[f"verdict-{label}"],
                        "solves": (files.get(f"review_input-{label}") or {}).get("solves"),
                        "changed_messages": (files.get(f"review_input-{label}") or {}).get(
                            "changed_messages"
                        ),
                    }
                    for label in labels
                },
            }
        )
    scenes, owner = [], {}
    for scene in (state.get("plan") or {}).get("scenes", []):
        written = state["scenes"].get(scene["id"], {}).get("scene", {})
        ids = ["m" + digest([scene["id"], i])[:10] for i in range(len(written.get("lines", [])))]
        owner.update(dict.fromkeys(ids, scene["id"]))
        scenes.append(scene | {"messages": ids, "promises": written.get("promises", [])})
    bindings = {b["task_id"]: b for b in (candidate or {}).get("bindings", [])}
    return {
        "name": f"{run.parent.name}/{run.name}",
        "summary": read_json(run / "summary.json"),
        "phase": state["phase"],
        "premise": state.get("premise"),
        "catalog": catalog,
        "world": (candidate or {}).get("snapshot"),
        "bindings": bindings,
        # Saved route outputs belong to the current bindings only when every gold call has its output.
        "chains": {
            t: chain(b, gold[t]) for t, b in bindings.items() if len(gold.get(t, [])) == len(b["gold_calls"])
        },
        "task_reviews": state.get("task_reviews") or {},
        "scenes": scenes,
        "scene_of": owner,
        "attempts": attempts,
        "snapshots": snapshots,
        "events": events,
        "traces": sorted(traces, key=lambda t: t["start"] or 0),
    }


def render(data):
    # Escape HTML delimiters even inside JSON strings: artifacts are untrusted text.
    encoded = (
        json.dumps(data, ensure_ascii=True)
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )
    template = (HERE / "template.html").read_text()
    assert template.count("__VIEWER_DATA__") == 1
    return template.replace("__VIEWER_DATA__", encoded)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path, help="Run directories (data/<run>/<world>)")
    parser.add_argument("--output", type=Path, default=HERE / "artifacts.html")
    parser.add_argument("--traces", action="store_true", help="Embed every agent transcript (large)")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(HERE):
        parser.error("Output must be inside viewer/ so run artifacts cannot be overwritten")
    if output.suffix != ".html" or output == HERE / "template.html":
        parser.error("Choose a generated .html output other than template.html")
    output.write_text(render({"runs": [load_run(path, args.traces) for path in args.runs]}))
    print(f"Wrote {output} ({output.stat().st_size:,} bytes)")
    print("Open this file in a browser. It contains private answers and full world snapshots.")


if __name__ == "__main__":
    main()
