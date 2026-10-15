"""Offline check on a scripted run: uv run --frozen python viewer/check.py"""

import asyncio
import sqlite3
import tempfile
from pathlib import Path

from build import load_run, render

from generators.worldgen_slack.tests.flows import run_flows, scripted_judges


def main():
    with tempfile.TemporaryDirectory() as tmp:
        with scripted_judges():
            asyncio.run(run_flows(Path(tmp) / "author"))
        world, forge, chapter = (
            load_run(Path(tmp) / "author" / name) for name in ("run", "forge", "chapter/run")
        )
        db = sqlite3.connect(Path(tmp) / "author" / "run" / "world.sqlite")
        stated = {
            (str(m), f, role) for m, f, role in db.execute("SELECT message_id, fact_id, role FROM evidence")
        }
        db.close()
    # A world written in time order by one author: its blocks in order, and each conversation by its lines.
    steps = [a["id"] for a in world["attempts"]]
    assert (
        steps.index("plan-01")
        < steps.index("day-01-01")
        < steps.index("review-02-02")
        < steps.index("day-03-02")
    ), steps
    assert steps.index("day-04-01") < steps.index("final-01") < steps.index("final-02"), steps
    assert all(
        world["snapshots"][a["snapshot"]]["messages"]
        for a in world["attempts"]
        if a["id"].startswith("final")
    ), "each attempt shows the world it ended with"
    assert set(world["scene_of"]) == {m["id"] for m in world["world"]["messages"]}, (
        "every message maps to its conversation"
    )
    assert all(s["participant_ids"] for s in world["scenes"]) and any(s["beats"] for s in world["scenes"])
    assert any(s["promises"] for s in world["scenes"]), "a conversation shows the promises its lines make"
    assert all(p["profile"] for p in world["catalog"]["personas"]), "the cast shows seeded profiles"
    assert world["world"]["zone"], "times are drawn on the world's clock"
    assert world["chapter"] is None and not world["catalog"]["tasks"], (
        "a world written in one run, with no tasks"
    )
    # A message shows its reactions and whom it mentions, as Slack does, and the facts it states.
    people = {u["id"] for u in world["world"]["users"]}
    reacted = [m for m in world["world"]["messages"] if m["reactions"]]
    assert reacted and all(
        r["user_ids"] and set(r["user_ids"]) <= people for m in reacted for r in m["reactions"]
    ), "each message's reactions, by emoji, with who reacted"
    assert any("<@" in m["text"] for m in world["world"]["messages"]), (
        "a mention, which the page draws as a name"
    )
    marks = {(m, f["fact"], f["role"]) for m, fs in world["fact_marks"].items() for f in fs}
    assert marks and marks == stated, "every message marks the facts it states"
    assert all("decoy" in f for fs in world["fact_marks"].values() for f in fs)
    # The forge's tasks on it: each shows its gold query, a ledger task the messages its facts are stated in, its
    # level and rates, each of its probes and the witness's tries.
    tasks = {t["id"]: t for t in forge["catalog"]["tasks"]}
    assert tasks and all(t["gold_sql"] for t in tasks.values()), "every task shows its gold query"
    joined = tasks["r09-07"]
    assert (
        not any(t.startswith("r09-07-p") for t in tasks)
        and [c.split(":")[0] for c in joined["answer"]["required_claims"]] == ["the newest", "the two newest"]
        and joined["gold_sql"].startswith("-- the newest\n")
        and joined["reasoning"].startswith("catch-up · composite")
    ), "a composite request is one task, its parts its labelled claims"
    anchored = [t for t in forge["bindings"] if tasks[t]["fact_ids"]]
    assert anchored and all(forge["bindings"][t]["claims"][0]["message_ids"] for t in anchored), (
        "ledger tasks show the messages their facts are stated in"
    )
    rates = forge["summary"]["rates"]
    assert rates and all("level" in r and "right_rate" in r for r in rates.values()), rates
    assert set(rates) <= {e["task_id"] for e in forge["events"] if e["event"] == "task_reviewed"}, (
        "every task's probes"
    )
    assert any(
        t["role"] == "witness" and t["evaluation"] and t["task_id"] in rates for t in forge["traces"]
    ), "the witness"
    # A chapter: the version it writes, from its first day, and its brief, each entry with the facts the board plants.
    brief = chapter["chapter"]["brief"]
    assert chapter["chapter"]["first_day"] == 5 and brief and all(b["facts"] for b in brief), chapter[
        "chapter"
    ]
    assert {b["slot"] for b in brief} <= set(chapter["board"]) and all(chapter["board"].values()), (
        "the board per slot"
    )
    attack = "</script><script>alert('artifact')</script>&"
    html = render({"runs": [], "probe": attack})
    assert attack not in html and "__VIEWER_DATA__" not in html and "\\u003c/script\\u003e" in html
    print(
        "Viewer checks passed: attempt order, snapshots, conversations, cast, reactions, mentions, facts, gold queries, composites, rates, probes, witness, chapter, board, escaping."
    )


if __name__ == "__main__":
    main()
