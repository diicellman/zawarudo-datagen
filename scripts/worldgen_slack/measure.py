"""How generated worlds type and what they cost, against real Slack users: style overall and per author (each
author beside the real typing profile it was given), the task mix, solve rates and spend.

uv run --frozen python scripts/worldgen_slack/measure.py data/v6-01/software [more runs]
"""

import argparse
import json
import statistics
from pathlib import Path

from generators.worldgen_slack.chronicle import bounds, posted, quotas
from generators.worldgen_slack.config import ROOT, ActivityConfig, Config
from generators.worldgen_slack.contracts import SeedPersona, activity, normalized, style, user_id
from worldgen_slack.db import World

TYPING = {
    "median_words": "median_words",
    "short": "short_share",
    "long": "long_share",
    "question": "question_share",
    "lowercase_start": "lowercase_share",
}


def real_users(path: Path) -> dict:
    """The median real user, from the typing profiles the casts are paired with."""
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    return {ours: round(statistics.median(r[theirs] for r in rows), 2) for ours, theirs in TYPING.items()}


def scorecard(world: World, targets: ActivityConfig) -> dict:
    """Each [activity] target beside the world's measured value."""
    measured = activity(world)
    one = lambda sql: world.db.execute(sql).fetchone()[0]  # noqa: E731
    shares = ("reply_share", "reaction_rate", "dm_share", "mention_rate", "emoji_rate")
    return (
        {"messages": {"target": targets.messages, "measured": measured["messages"]}}
        | {k: {"target": getattr(targets, k), "measured": measured[k]} for k in shares}
        | {
            "conversations_used": f"{one('SELECT COUNT(DISTINCT channel_id) FROM messages')} of {one('SELECT COUNT(*) FROM channels')}"
        }
    )


def author(run: Path, world: World, settings: Config) -> dict | None:
    """A v7 world, as its author wrote it: each day against its quota, the ledger and its events, promises, the
    author's turns and tool calls, the solver's probes, the reviews, and how often a ledger task's answer sits in one
    message."""
    if not world.db.execute("SELECT 1 FROM world_meta WHERE key = 'chronological'").fetchone():
        return None
    one = lambda sql, *args: world.db.execute(sql, args).fetchone()[0]  # noqa: E731
    events = [json.loads(line) for line in (run / "progress.jsonl").read_text().splitlines()]
    calls = [json.loads(line) for line in (run / "world-calls.jsonl").read_text().splitlines()] if (run / "world-calls.jsonl").exists() else []  # fmt: skip
    quota = quotas(world, settings)
    days = {
        day: {"messages": posted(world, *bounds(world, day)), "quota": q["messages"]}
        | {k: v for k, v in activity(world, *bounds(world, day)).items() if k != "messages"}
        for day, q in quota.items()
    }
    by_tool = {}
    for call in calls:
        row = by_tool.setdefault(call["tool"], {"calls": 0, "errors": 0})
        row["calls"] += 1
        row["errors"] += not call["ok"]
    # A shortcut: one message that states every fact a multi-fact ledger task rests on (v6-03's level-3 answer sat
    # whole in one announcement).
    texts = [normalized(t) for (t,) in world.db.execute("SELECT text FROM messages")]
    shortcuts = {}
    for (task,) in world.db.execute("SELECT id FROM tasks WHERE gold_source = 'ledger'"):
        values = [normalized(v) for (v,) in world.db.execute("SELECT f.value FROM task_facts tf JOIN facts f ON f.id = tf.fact_id WHERE tf.task_id = ?", (task,))]  # fmt: skip
        if len(values) > 1:
            shortcuts[task] = sum(all(f" {v} " in f" {text} " for v in values) for text in texts)
    terms = [t for (t,) in world.db.execute("SELECT anchor FROM facts WHERE anchor IS NOT NULL UNION SELECT subject FROM facts")]  # fmt: skip
    everyday = world.db.execute("SELECT s.id, group_concat(m.text, ' ') FROM scenes s JOIN scene_messages sm ON sm.scene_id = s.id JOIN messages m ON m.id = sm.message_id WHERE s.storyline IS NULL GROUP BY s.id").fetchall()  # fmt: skip
    return {
        "days": days,
        "ledger": {
            "storylines": one("SELECT COUNT(*) FROM storylines"),
            "events": one("SELECT COUNT(*) FROM events"),
            "events_shared_by_facts": one(
                "SELECT COUNT(*) FROM (SELECT event_id FROM facts WHERE event_id IS NOT NULL GROUP BY event_id HAVING COUNT(*) > 1)"
            ),  # fmt: skip
            "facts": one("SELECT COUNT(*) FROM facts"),
            "decoys": one("SELECT COUNT(*) FROM facts WHERE is_decoy = 1"),
            "supersedes": one("SELECT COUNT(*) FROM fact_relations WHERE kind = 'supersedes'"),
            "restated": one("SELECT COUNT(*) FROM evidence WHERE role = 'supporting'"),
        },
        "promises": {
            r[0]: r[1] for r in world.db.execute("SELECT status, COUNT(*) FROM commitments GROUP BY status")
        },
        "author_turns": sum(e["event"] == "author_turn" for e in events),
        "tools": by_tool,
        "probes": [e["rates"] for e in events if e["event"] == "probe"],
        "reviews": {
            e["attempt"]: e["approved"]
            for e in events
            if e["event"] == "candidate_finished" and str(e["attempt"]).startswith(("review", "final"))
        },
        "ledger_answer_shortcuts": shortcuts,
        "everyday_on_storylines": round(
            sum(any(f" {normalized(t)} " in f" {normalized(text)} " for t in terms) for _, text in everyday)
            / max(len(everyday), 1),
            2,
        ),
    }


def measure(run: Path) -> dict:
    state = json.loads((run / "state.json").read_text())
    summary = json.loads((run / "summary.json").read_text()) if (run / "summary.json").exists() else {}
    stats = style(World(run / "world.sqlite"))
    # Profiles exported before long_share was measured read it as unknown (0).
    cast = {
        user_id(p["uuid"]): SeedPersona.model_validate_json(
            json.dumps(p | {"typing": {"long_share": 0.0} | p["typing"]})
        )
        for p in state["cast"]
    }
    authors = {
        author: {k: values[k] for k in TYPING} | {"messages": values["messages"]}
        | {"target": {ours: getattr(cast[author].typing, theirs) for ours, theirs in TYPING.items()}}
        for author, values in stats.get("authors", {}).items()
        if author in cast
    }  # fmt: skip
    config = json.loads((run / "run.json").read_text())["config"]
    targets = ActivityConfig.model_validate(config.get("activity", {}))
    written = (
        author(run, World(run / "world.sqlite"), Config.model_validate(config))
        if "author" in config
        else None
    )
    return {
        "run": str(run),
        "author": written,
        "scorecard": scorecard(World(run / "world.sqlite"), targets),
        "overall": {k: stats.get(k) for k in ("messages", *TYPING, "off_hours")},
        "authors": authors,
        "task_mix": summary.get("task_mix"),
        "solve_rates": summary.get("solve_rates"),
        "difficulty": summary.get("difficulty"),
        "cost": summary.get("reported_model_cost"),
        "minutes": round(summary.get("elapsed_seconds", 0) / 60, 1),
        "rounds": summary.get("rounds"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--typing", type=Path, default=ROOT / "data/seeds/typing-profiles.jsonl")
    args = parser.parse_args()
    report = {"real_users": real_users(args.typing), "runs": [measure(run) for run in args.runs]}
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
