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
    # The forge's tasks on it, from its state: each with its set, its label, its answer and the messages it rests on.
    tasks = {t["id"]: t for t in forge["catalog"]["tasks"]}
    assert {"i001", "h101", "e102", "r103"} <= set(tasks), sorted(tasks)
    assert (
        tasks["h101"]["reasoning"].startswith("status · harder") and "2 parts" in tasks["h101"]["reasoning"]
    )
    assert [c.split(":")[0] for c in tasks["h101"]["answer"]["required_claims"]] == [
        "who posted it",
        "who answered it",
    ]
    assert all(forge["bindings"][t]["claims"][0]["message_ids"] for t in tasks), (
        "each task shows its messages"
    )
    assert (
        tasks["i001"]["gold_sql"].startswith("rests on m") and "verifier agreed" in tasks["i001"]["gold_sql"]
    )
    assert any(t["role"] == "verifier" and t["evaluation"] for t in forge["traces"]), "the verifier's tries"
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
        "Viewer checks passed: attempt order, snapshots, conversations, cast, reactions, mentions, facts, forge tasks, verifier, chapter, board, escaping."
    )


if __name__ == "__main__":
    main()
