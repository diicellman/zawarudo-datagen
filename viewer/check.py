"""Offline check on a scripted run: uv run --frozen python viewer/check.py"""

import asyncio
import sqlite3
import tempfile
from pathlib import Path

from build import load_run, render
from generators.worldgen_slack.check import check_author


def main():
    with tempfile.TemporaryDirectory() as tmp:
        asyncio.run(check_author(Path(tmp) / "author"))
        run = load_run(Path(tmp) / "author" / "run")
        db = sqlite3.connect(Path(tmp) / "author" / "run" / "world.sqlite")
        stated = {
            (str(m), f, role) for m, f, role in db.execute("SELECT message_id, fact_id, role FROM evidence")
        }
        db.close()
    # A world written in time order by one author: its blocks in order, and each conversation by its lines.
    steps = [a["id"] for a in run["attempts"]]
    assert (
        steps.index("plan-01")
        < steps.index("day-01-01")
        < steps.index("review-02-02")
        < steps.index("day-03-02")
    ), steps
    assert steps.index("day-04-01") < steps.index("tasks-01") < steps.index("final-02"), steps
    assert all(
        run["snapshots"][a["snapshot"]]["messages"] for a in run["attempts"] if a["id"].startswith("final")
    ), "each attempt shows the world it ended with"
    assert set(run["scene_of"]) == {m["id"] for m in run["world"]["messages"]}, (
        "every message maps to its conversation"
    )
    assert all(s["participant_ids"] for s in run["scenes"]) and any(s["beats"] for s in run["scenes"])
    assert any(s["promises"] for s in run["scenes"]), "a conversation shows the promises its lines make"
    tasks = {t["id"]: t for t in run["catalog"]["tasks"]}
    assert tasks and all(t["gold_sql"] for t in tasks.values()), "every task shows its gold query"
    anchored = [t for t, b in run["bindings"].items() if tasks[t]["fact_ids"]]
    assert anchored and all(run["bindings"][t]["claims"][0]["message_ids"] for t in anchored), (
        "ledger tasks show the messages their facts are stated in"
    )
    assert all(p["profile"] for p in run["catalog"]["personas"]), "the cast shows seeded profiles"
    # A task shows its band and rounds, each of its probes, the witness's tries and the board's facts.
    assert run["summary"]["rates"] and all(
        "band" in r and "rounds" in r for r in run["summary"]["rates"].values()
    )
    assert {e["task_id"] for e in run["events"] if e["event"] == "task_reviewed"} == set(tasks), (
        "every task's probes"
    )
    assert any(t["role"] == "witness" and t["evaluation"] and t["task_id"] in tasks for t in run["traces"]), (
        "the witness"
    )
    assert run["board"] and set(run["board"]) <= set(tasks) and all(run["board"].values()), (
        "the board per ledger slot"
    )
    assert run["world"]["zone"], "times are drawn on the world's clock"
    # A message shows its reactions and whom it mentions, as Slack does, and the facts it states.
    people = {u["id"] for u in run["world"]["users"]}
    reacted = [m for m in run["world"]["messages"] if m["reactions"]]
    assert reacted and all(
        r["user_ids"] and set(r["user_ids"]) <= people for m in reacted for r in m["reactions"]
    ), "each message's reactions, by emoji, with who reacted"
    assert any("<@" in m["text"] for m in run["world"]["messages"]), (
        "a mention, which the page draws as a name"
    )
    marks = {(m, f["fact"], f["role"]) for m, fs in run["fact_marks"].items() for f in fs}
    assert marks and marks == stated, "every message marks the facts it states"
    assert all("decoy" in f for fs in run["fact_marks"].values() for f in fs)
    attack = "</script><script>alert('artifact')</script>&"
    html = render({"runs": [], "probe": attack})
    assert attack not in html and "__VIEWER_DATA__" not in html and "\\u003c/script\\u003e" in html
    print("Viewer checks passed: attempt order, snapshots, conversations, gold queries, cast, bands, probes, witness, board, reactions, mentions, facts, escaping.")  # fmt: skip


if __name__ == "__main__":
    main()
