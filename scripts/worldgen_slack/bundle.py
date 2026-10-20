"""A forge's release as a dataset to hand on: the world, its tasks with their answers and metadata, every try behind
their labels, a QC note and its readings, and the viewer, in one folder (and a zip).

uv run --frozen python scripts/worldgen_slack/bundle.py data/v8-05/forge-w2 --name W2 \
    --qc-note local-docs/bundle-notes/forge-w2-QC.md --out data/v8-05/bundle/forge-w2 [--zip]

The folder holds:
- `release/`: the environment's release as the forge published it (tasks.json, answers.json, manifest.json and the
  world the solver reads, world.sqlite), which the eval loads as is;
- `world-full.sqlite`: the same world with the generator's answer key (storylines, facts, evidence, scenes);
- `tasks.jsonl`: one row per released task;
- `tries.jsonl`: every student and verifier try behind the labels;
- `qc.jsonl` and `QC.md`: the QC readings and note, when given (the run's qc.json, and --qc-note);
- `viewer.html`: the offline viewer of the world and the tasks;
- `README.md`: what all of it is.

It reads the run's saved files and spends nothing. The bundle carries gold answers: share it only on purpose."""

import argparse
import json
import shutil
import sqlite3
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "viewer"))
from build import load_run, render  # noqa: E402  (the viewer's page builder)


def here(path: Path) -> Path:
    """A run directory as this checkout has it: a forge run elsewhere (the research VM) names its world by that
    machine's absolute path, under the same data/ folder."""
    if path.exists() or "data" not in path.parts:
        return path
    return ROOT.joinpath(*path.parts[path.parts.index("data") :])


def read_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def slack_ts(us: int) -> str:
    return f"{us // 1_000_000}.{us % 1_000_000:06d}"


def stamp(us: float) -> str:
    return datetime.fromtimestamp(us, UTC).strftime("%Y-%m-%d %H:%M UTC")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path, help="a forge's run directory (its state.json and release/)")
    parser.add_argument("--name", required=True, help="the world's name, e.g. W2")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--qc-note", type=Path, help="a QC note (markdown) to include as QC.md")
    parser.add_argument("--zip", action="store_true", help="also write <out>.zip")
    parser.add_argument(
        "--world", type=Path, help="the world's run directory, for its premise (default: the forge's)"
    )
    args = parser.parse_args()
    run = args.run
    state, config = read_json(run / "state.json"), read_json(run / "run.json")["config"]
    rows, answers = read_json(run / "release" / "tasks.json"), read_json(run / "release" / "answers.json")
    manifest = read_json(run / "release" / "manifest.json")
    if not rows:
        sys.exit(f"{run}/release holds no tasks")
    qc = {v["task_id"]: v for v in read_json(run / "qc.json") or []}
    premise = (read_json(args.world or here(Path(config["world"])) / "state.json") or {}).get("premise") or {}
    models = {
        "proposer": config["env"]["proposer"]["model"],
        "student": config["env"]["solver"]["model"],
        "verifier": config["env"]["verifier"]["model"],
        "task_judge": config["judge"]["model"],
        "grader": config["answer_judge"]["model"],
    }
    out = args.out
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    shutil.copytree(run / "release", out / "release")
    shutil.copy(run / "world.sqlite", out / "world-full.sqlite")

    db = sqlite3.connect(f"file:{run / 'release' / 'world.sqlite'}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row

    def message(mid: int) -> dict:
        r = db.execute(
            """SELECT m.id, m.ts_us, m.parent_id, m.text, c.id AS channel_id, c.name, c.type, u.real_name
            FROM messages m JOIN channels c ON c.id = m.channel_id JOIN users u ON u.id = m.user_id WHERE m.id = ?""",
            (mid,),
        ).fetchone()
        return {
            "message_id": r["id"],
            "channel_id": r["channel_id"],
            "channel": r["name"] or {"im": "DM", "mpim": "group DM"}.get(r["type"], r["type"]),
            "ts": slack_ts(r["ts_us"]),
            "thread_reply": r["parent_id"] is not None,
            "author": r["real_name"],
            "text": r["text"],
        }

    world = {
        "name": args.name,
        "company": premise.get("company")
        or db.execute("SELECT value FROM world_meta WHERE key = 'company'").fetchone()[0],
        "hash": manifest["world_hash"],
        "zone": db.execute("SELECT value FROM world_meta WHERE key = 'zone'").fetchone()[0],
    }
    tasks, tries = [], []
    for row in rows:
        task_id = row["task_id"]
        record = state["forged"][task_id]
        task = record["task"]
        asker = db.execute(
            "SELECT id, real_name, title FROM users WHERE id = ?", (task["actor_id"],)
        ).fetchone()
        solution = answers[task_id].get("solution") or {}
        review = qc.get(task_id)
        parts = [
            {
                "label": p["label"],
                "answer_type": p["answer_type"],
                "answer": p.get("answer") or [],
                "evidence": [message(m) for m in p["evidence"]],
                "out_of_sight": [message(m) for m in p.get("unseen") or []],
            }
            for p in task["parts"]
        ]
        tasks.append(
            {
                "task_id": task_id,
                "world": world,
                "forge": f"{run.parent.name}/{run.name}",
                "question": task["question"],
                "asker": {"id": asker["id"], "name": asker["real_name"], "title": asker["title"]},
                "answer": parts,
                "answer_text": " | ".join(
                    (f"{p['label']}: " if len(parts) > 1 else "")
                    + ("; ".join(p["answer"]) or "(not there to find)")
                    for p in parts
                ),
                "answer_type": row["answer_type"],
                "family": task["family"],
                "skill": record.get("skill"),
                "set": record["set"],
                "variant_of": task.get("variant_of"),
                "label": record["label"],
                "right_rate": row["right_rate"],
                "right_interval": row["right_interval"],
                "strict_rate": row["strict_rate"],
                "abstain_rate": row["abstain_rate"],
                "part_rates": row["part_rates"],
                "graded_tries": row["tries"],
                "crashed_tries": row["crashed"],
                "verifier_right": row["verifier_right"],
                "judge": record.get("ruling"),
                "solution": {
                    "seat": solution.get("seat"),
                    "answer": solution.get("answer"),
                    "calls": solution.get("calls"),
                }
                if solution
                else None,
                "models": models,
                "qc": {
                    k: review[k] for k in ("task_verdict", "reason", "skill_match", "gold_ok", "gold_note")
                }
                if review
                else None,
            }
        )
        for seat, book in (("student", "solves"), ("verifier", "verifier")):
            entry = state[book].get(task_id, {})
            for n, (r, trace_id) in enumerate(zip(entry.get("results", []), entry.get("traces", [])), 1):
                if not r:
                    continue
                trace = read_json(run / "traces" / f"{trace_id}.json") or {}
                calls = [
                    {"tool": o["tool"], "arguments": o["arguments"]}
                    for o in (trace.get("info") or {}).get("observations") or []
                ]
                tries.append(
                    {
                        "task_id": task_id,
                        "seat": seat,
                        "try": f"{seat[0].upper()}{n}",
                        "model": models[seat],
                        "crashed": bool(r.get("crashed")),
                        "correct": r.get("correct"),
                        "grounded": r.get("grounded"),
                        "abstained": r.get("abstained"),
                        "evidence_coverage": r.get("evidence_coverage"),
                        "parts": [
                            {k: p.get(k) for k in ("label", "correct", "committed", "missing", "extra")}
                            for p in r.get("parts") or []
                        ],
                        "answer": r.get("response"),
                        "grade": r.get("reason"),
                        "calls": calls,
                    }
                )
    db.close()
    jsonl(out / "tasks.jsonl", tasks)
    jsonl(out / "tries.jsonl", tries)
    if qc:
        jsonl(out / "qc.jsonl", [qc[t["task_id"]] for t in tasks if t["task_id"] in qc])
    if args.qc_note:
        shutil.copy(args.qc_note, out / "QC.md")
    (out / "viewer.html").write_text(render({"runs": [load_run(run)]}))
    (out / "README.md").write_text(
        readme(run, state, world, premise, tasks, tries, models, qc, bool(args.qc_note))
    )
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"{out}: {len(tasks)} tasks, {len(tries)} tries, {size / 1e6:.1f} MB")
    if args.zip:
        archive = shutil.make_archive(str(out), "zip", out.parent, out.name)
        print(f"{archive}: {Path(archive).stat().st_size / 1e6:.1f} MB")
    return 0


def table(counter: Counter, head: str) -> str:
    return f"| {head} | Tasks |\n|---|---|\n" + "".join(f"| {k} | {v} |\n" for k, v in counter.most_common())


def readme(run, state, world, premise, tasks, tries, models, qc: dict, has_note: bool) -> str:
    db = sqlite3.connect(f"file:{run / 'world.sqlite'}?mode=ro", uri=True)
    one = lambda sql: db.execute(sql).fetchone()[0]  # noqa: E731
    messages, people = (
        one("SELECT COUNT(*) FROM messages WHERE is_deleted = 0"),
        one("SELECT COUNT(*) FROM users WHERE is_bot = 0"),
    )
    days = one("SELECT MAX(day) FROM calendar")
    kinds = dict(db.execute("SELECT type, COUNT(*) FROM channels GROUP BY type").fetchall())
    storylines = [r for r in db.execute("SELECT id, summary FROM storylines ORDER BY position")]
    db.close()
    labelled = [r for r in state["forged"].values() if r.get("label")]
    unlabelled = sum(1 for r in state["forged"].values() if not r.get("label"))
    events = [json.loads(line) for line in (run / "progress.jsonl").read_text().splitlines()]
    last = max(e["time"] for e in events if e["event"] == "task_labelled")
    short = sum(1 for t in tasks if t["crashed_tries"])
    verdicts = Counter(t["qc"]["task_verdict"] for t in tasks if t["qc"])
    bad = [t["task_id"] for t in tasks if t["qc"] and t["qc"]["task_verdict"] == "bad"]
    lines = [
        f"# {world['name']} tasks: Slack-assistant questions over one synthetic workspace",
        "",
        f"{len(tasks)} tasks a colleague asks a Slack assistant, over {world['company']}'s workspace ({world['name']}).",
        "Each comes with its answer, the messages the answer rests on, the student model's measured pass rate and",
        "every try behind it. The tasks were labelled by "
        + stamp(last)
        + f", and the bundle was made {datetime.now(UTC):%Y-%m-%d}.",
        "",
        "## The world",
        "",
        f"- **{world['company']}**: {premise.get('niche', '')} {premise.get('region', '')}; {premise.get('size', '')}.",
        f"- **{messages:,} messages over {days} working days** from {people} people.",
        f"- **Conversations:** {kinds.get('public', 0)} public channels, {kinds.get('private', 0)} private ones,"
        f" {kinds.get('im', 0)} DMs and {kinds.get('mpim', 0)} group DMs. Times are on {world['zone']}.",
        "- **Writing:** one author model wrote it in time order, around storylines that run through the weeks:",
        "",
        *[f"  - **{sid}**: {summary}" for sid, summary in storylines],
        "",
        "  Values change, people correct each other, promises slip, and some facts sit only in private channels or DMs.",
        "- **Files:** `release/world.sqlite` is the world as an assistant reads it (Slack-shaped: channels, members,",
        "  messages, threads, reactions). `world-full.sqlite` adds the generator's answer key: storylines, the facts each",
        "  message states (decoys marked), and the scenes that wrote them.",
        "",
        "## How the tasks were made and labelled",
        "",
        f"- **Proposer** ({models['proposer']}): read the world and wrote each request as a named colleague (the",
        "  asker), with its answer and the messages it rests on. A request may have up to 4 parts. Code checked that",
        "  the asker can read every cited message.",
        f"- **Verifier** ({models['verifier']}, at max reasoning): answered each request as the asker, from Slack",
        "  alone. Each part had to be reached twice, or the task was dropped.",
        f"- **Student** ({models['student']}): tried each task 8 times. Each try was graded against the answer by",
        f"  {models['grader']}. An answer that hedges between values counts as wrong.",
        f"- **Task judge** ({models['task_judge']}): ruled each task realistic, unambiguous, exact (asking no more than",
        "  its parts) and supported by its messages. Only tasks with all four are in this release.",
        "- **Labels:** by the student's share of right tries.",
        "  - `solved`: at least 7 of 8 right;",
        "  - `window`: some right;",
        "  - `unsolved`: none right, though the verifier reached it.",
        "- **Sets:** `initial` tasks were written first. Then each solved task got a `harder` version, each unsolved one",
        "  an `easier` one, and each verified task the judge refused a `repaired` one.",
        "- **Skills:** each initial task was written for one skill:",
        "  - aggregation: gather every item across places;",
        "  - temporal: the value as of a moment, or the latest update;",
        "  - abstention: the answer is that it isn't there, isn't settled, or can't be seen;",
        "  - synthesis: join what is stated apart.",
        "",
        table(Counter(t["label"] for t in tasks), "Label"),
        table(Counter(t["skill"] for t in tasks), "Skill"),
        table(Counter(t["set"] for t in tasks), "Set"),
        table(Counter(t["family"] for t in tasks), "Request family"),
        "## Files",
        "",
        "| File | What |",
        "|---|---|",
        "| `tasks.jsonl` | One row per task (schema below) |",
        f"| `tries.jsonl` | Every try behind the labels ({len(tries)}): `task_id`, `seat` (student or verifier),"
        " `try`, `model`, `correct`, `grounded`, `abstained`, `crashed`, `evidence_coverage`, `parts` (by part:"
        " committed, missing, extra), `answer`, `grade` (the grader's reason), and `calls` (each tool call with its"
        " arguments) |",
        "| `release/` | The environment's own release format (tasks.json, answers.json, manifest.json, world.sqlite) |",
        "| `world-full.sqlite` | The world with the generator's answer key |",
        "| `viewer.html` | An offline viewer of the world and the tasks (below) |",
        *(["| `QC.md`, `qc.jsonl` | The QC note, and its reading of each task |"] if has_note or qc else []),
        "",
        "**A row of `tasks.jsonl`:**",
        "- `task_id`, `question`, `asker` (id, name, title), `world` (name, company, hash of `release/world.sqlite`,",
        "  zone), `forge`.",
        "- `answer`: a list of parts, each with `label`, `answer_type` (text, set, number, status or refusal) and",
        "  `answer` (the items).",
        "  - `evidence`: the messages it rests on, each with `message_id`, `channel_id`, `channel`, `ts` (Slack's),",
        "    `thread_reply`, `author` and `text`.",
        "  - `out_of_sight`: messages that hold the truth but that the asker can't read.",
        "- `answer_text`: the answer as one string.",
        "- **What the task is:** `answer_type`, `family`, `skill`, `set`, `variant_of`.",
        "- **How the student did:** `label`, `right_rate`, `right_interval` (90%), `strict_rate` (right and grounded),",
        "  `abstain_rate`, `part_rates`, `graded_tries`, `crashed_tries`.",
        "- **The checks:** `verifier_right`, `judge` (the task judge's ruling), `solution` (a verifier try that",
        "  reached the answer, with its calls).",
        "- `models`, and `qc` (the QC verdict and its reason).",
        "",
        "## Caveats",
        "",
        f"- {short} labels rest on fewer than 8 graded tries (`crashed_tries` > 0): some grades failed on a provider's",
        "  rate limits and were left out of the rate.",
        f"- The forge wrote {len(state['forged'])} tasks. {len(labelled)} were labelled; of those, the ones the judge",
        f"  refused or the verifier could not reach are not here. {unlabelled} were left unlabelled by the provider's",
        "  rate limits and are not here either.",
        *(
            [
                f"- **QC:** {'QC.md reads' if has_note else 'The QC readings cover'} every task and the tries behind it.",
                f"  {verdicts['bad']} tasks are rated bad ({', '.join(bad)}) and {verdicts['mixed']} mixed; filter on",
                "  `qc.task_verdict`.",
            ]
            if qc
            else []
        ),
        "",
        "## The viewer",
        "",
        "Open `viewer.html` in a browser; it works offline. It has these tabs:",
        "- **World:** the workspace by channel and day. A reader filter shows what one person can see.",
        "- **Tasks:** each task with its answer, its messages and the near-misses beside it (earlier values, later",
        "  values and decoys), its ruling, its QC reading, and every student and verifier try. A panel beside it",
        "  takes your labels, which stay in the browser; Export saves them as JSON.",
        "- **Cast:** the people, and how each of them writes.",
        "- **Lineage and Traces:** how the forge ran.",
        "",
        "'Answers shown' blurs answers on screen. It does not remove them.",
        "",
        "## Evaluating a model on it",
        "",
        "`release/` is the worldgen-slack environment's release format. With that environment installed, run:",
        "`eval --env.taskset.task.release_dir <bundle>/release`. The solver answers through Slack tools served over",
        "`release/world.sqlite`, and the grader compares its answer with each part's.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
