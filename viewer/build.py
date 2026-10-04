"""Export a worldgen run as a standalone, offline HTML page: lineage, world, tasks, cast and agent traces."""

import argparse
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from worldgen_slack.db import digest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
QUIET = {"agent_progress", "author_started"}
KIND = {"public": "public_channel", "private": "private_channel", "im": "dm", "mpim": "group_dm"}


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else None


def connect(path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    return db


def iso(us: int) -> str:
    return datetime.fromtimestamp(us / 1e6, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def snapshot(path: Path) -> dict:
    """A world file as the page draws it: people, conversations and messages."""
    db = connect(path)
    members = {}
    for channel, user in db.execute(
        "SELECT channel_id, user_id FROM members WHERE left_us IS NULL ORDER BY user_id"
    ):
        members.setdefault(channel, []).append(user)
    world = {
        "users": [
            {"id": r["id"], "name": r["real_name"], "display_name": r["display_name"], "team": json.loads(r["profile_json"]).get("Team")}
            for r in db.execute("SELECT * FROM users ORDER BY id")
        ],
        "conversations": [
            {"id": r["id"], "name": r["name"], "kind": KIND[r["type"]], "topic": r["topic"], "purpose": r["purpose"], "member_ids": members.get(r["id"], [])}
            for r in db.execute("SELECT * FROM channels ORDER BY id")
        ],
        "messages": [
            {"id": str(r["id"]), "conversation_id": r["channel_id"], "author_id": r["user_id"], "text": r["text"], "timestamp": iso(r["ts_us"]), "thread_root_id": str(r["parent_id"]) if r["parent_id"] else None, "deleted": bool(r["is_deleted"])}
            for r in db.execute("SELECT * FROM messages ORDER BY ts_us")
        ],
    }  # fmt: skip
    db.close()
    return world


def summarize(record, full):
    """One agent run: role, what it worked on, when, and what it cost; `full` keeps its messages."""
    data, info, usage = record["task"]["data"], record.get("info") or {}, record.get("usage") or {}
    timing = record.get("timing") or {}
    row = {
        "id": record["id"],
        "role": record["agent"]["name"],
        "model": record["agent"]["config"].get("model"),
        "attempt": data.get("attempt") or None,
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
            {"call": {"action": o["tool"], "arguments": o["arguments"]}, "messages": []}
            for o in info.get("observations", [])
        ]
    if "verdict" in info:
        row["approved"] = info["verdict"]["approved"]
    if full:
        row["messages"] = [
            {k: n["message"].get(k) for k in ("role", "content", "tool_calls")} for n in record["nodes"]
        ]
    return row


def ledger(path: Path, state: dict) -> tuple[dict, dict, list, dict]:
    """The world file's answer key as the page draws it: catalog (people, storylines, facts, tasks), evidence per
    task, scenes, and each message's scene."""
    db = connect(path)
    cast = {c["uuid"]: c for c in state.get("cast", [])}
    from_seed = {"U" + digest(["user", uuid])[:10].upper(): c for uuid, c in cast.items()}
    users = db.execute("SELECT * FROM users ORDER BY id").fetchall()
    facts = [dict(r) for r in db.execute("SELECT * FROM facts ORDER BY day, id")]
    storyline_of = {f["id"]: f["storyline"] for f in facts}
    tasks, bindings = [], {}
    for t in db.execute("SELECT * FROM tasks ORDER BY id").fetchall():
        gold = json.loads(t["gold_json"])
        fact_ids = [r[0] for r in db.execute("SELECT fact_id FROM task_facts WHERE task_id = ?", (t["id"],))]
        evidence = {str(r["message_id"]) for r in gold if r.get("message_id") is not None}
        evidence |= {str(r[0]) for r in db.execute("SELECT e.message_id FROM task_facts tf JOIN evidence e ON e.fact_id = tf.fact_id WHERE tf.task_id = ? AND e.role = 'anchor'", (t["id"],))}  # fmt: skip
        claims = [str(r.get("answer")) for r in gold] or ["(refusal: nothing to find)"]
        tasks.append(
            {"id": t["id"], "group_id": next((storyline_of[f] for f in fact_ids), "tasks"), "question": t["question"], "actor_id": t["actor_id"], "fact_ids": fact_ids, "reasoning": f"{t['category']} · level {t['level']} · {t['answer_type']}", "gold_sql": t["gold_sql"], "category": t["category"], "level": t["level"], "answer": {"canonical_answer": "; ".join(claims), "required_claims": claims}}
        )  # fmt: skip
        bindings[t["id"]] = {"claims": [{"claim_index": k, "message_ids": sorted(evidence), "user_ids": []} for k in range(len(claims))]}  # fmt: skip
    catalog = {
        "company": (db.execute("SELECT value FROM world_meta WHERE key = 'company'").fetchone() or [None])[0],
        "people": [{"id": u["id"], "name": u["real_name"], "display_name": u["display_name"], "team": json.loads(u["profile_json"]).get("Team")} for u in users],
        "personas": [{"id": u["id"], "role": u["title"], "seniority": "", "timezone": u["tz"], "voice": "", "profile": from_seed.get(u["id"])} for u in users],
        "groups": [{"id": r["id"], "description": r["summary"]} for r in db.execute("SELECT * FROM storylines ORDER BY position")],
        "facts": [
            {"id": f["id"], "group_id": f["storyline"], "subject": f["subject"], "predicate": f["attribute"], "value": f["value"], "valid_from": f"day {f['day']}" + (f" · {f['moment_kind']} {iso(f['moment_us'])}" if f["moment_us"] else ""), "valid_until": None, "description": f["summary"]}
            for f in facts
        ],
        "tasks": tasks,
    }  # fmt: skip
    scenes, owner = [], {}
    for s in db.execute("SELECT * FROM scenes ORDER BY slot_start_us").fetchall():
        plan = json.loads(s["plan_json"])
        ids = [str(r[0]) for r in db.execute("SELECT sm.message_id FROM scene_messages sm JOIN messages m ON m.id = sm.message_id WHERE sm.scene_id = ? ORDER BY m.ts_us", (s["id"],))]  # fmt: skip
        owner.update(dict.fromkeys(ids, s["id"]))
        # A conversation keeps its lines, whose authors, statements and promises it shows; older runs' scenes keep a plan.
        lines = plan.get("lines", [])
        participants = plan.get("participants") or list(dict.fromkeys(line["author_id"] for line in lines))
        beats = [{"fact_id": b["fact"], "author_id": b["author_id"]} for b in plan.get("beats", [])] + [{"fact_id": f, "author_id": line["author_id"]} for line in lines for f in line.get("conveys", [])]  # fmt: skip
        promises = plan.get("promises") or [c["text"] for line in lines for c in line.get("commits", [])]
        scenes.append(
            {"id": s["id"], "conversation_id": s["channel_id"], "participant_ids": participants, "start": iso(s["slot_start_us"]), "end": iso(s["slot_end_us"]), "situation": s["situation"], "beats": beats, "messages": ids, "promises": promises, "revision_note": plan.get("revision_note", "")}
        )  # fmt: skip
    db.close()
    return catalog, bindings, scenes, owner


def load_run(path, full=False):
    """Everything the page shows, read from the run's saved files and world snapshots."""
    run = path.resolve()
    state = read_json(run / "state.json")
    if state is None or not (run / "world.sqlite").exists():
        raise ValueError(f"No run (state.json and world.sqlite) in {run}")
    catalog, bindings, scenes, owner = ledger(run / "world.sqlite", state)
    events = [json.loads(line) for line in (run / "progress.jsonl").read_text().splitlines()]
    order = {e["attempt"]: i for i, e in reversed(list(enumerate(events))) if e.get("attempt")}
    events = [e for e in events if e["event"] not in QUIET]
    traces = [summarize(read_json(p), full) for p in (run / "traces").glob("*.json")]
    snapshots, attempts = {}, []
    folders = sorted((p for p in (run / "attempts").iterdir() if p.is_dir()), key=lambda p: (order.get(p.name, -1 if p.name in ("premise", "organization") else 1e9), p.name))  # fmt: skip
    for folder in folders:
        files = {p.stem: read_json(p) for p in folder.glob("*.json")}
        key = None
        if (folder / "world.sqlite").exists():
            key = folder.name
            snapshots[key] = snapshot(folder / "world.sqlite")
        validation = files.get("validation") or {}
        labels = sorted(name.removeprefix("verdict-") for name in files if name.startswith("verdict-"))
        attempts.append(
            {
                "id": folder.name,
                "snapshot": key,
                "verdict": files.get("verdict") or (files.get(f"verdict-{labels[0]}") if len(labels) == 1 else None),
                "acceptance": files.get("acceptance"),
                "validation": {k: validation.get(k) for k in ("ok", "errors")} if validation else None,
                "corrections": files.get("corrections"),
                "reviews": {
                    label: {
                        "verdict": files[f"verdict-{label}"],
                        "solves": (files.get(f"review_input-{label}") or {}).get("solves"),
                    }
                    for label in labels
                },
            }
        )  # fmt: skip
    return {
        "name": f"{run.parent.name}/{run.name}",
        "summary": read_json(run / "summary.json"),
        "phase": state["phase"],
        "premise": state.get("premise"),
        "quota": state.get("quota"),
        "catalog": catalog,
        "world": snapshot(run / "world.sqlite"),
        "bindings": bindings,
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
