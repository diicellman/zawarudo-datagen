"""Offline check on a scripted run: uv run --frozen python viewer/check.py"""

import asyncio
import tempfile
from pathlib import Path

from build import load_run, render
from generators.worldgen_slack.check import check_author, check_flow


def main():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "flow"
        asyncio.run(check_flow(root))
        run = load_run(root / "run")
        asyncio.run(check_author(Path(tmp) / "author"))
        written = load_run(Path(tmp) / "author" / "run")
    # A world written in time order by one author (v7): its blocks in order, and each conversation by its lines.
    steps = [a["id"] for a in written["attempts"]]
    assert (
        steps.index("plan-01")
        < steps.index("day-01-01")
        < steps.index("review-01")
        < steps.index("day-03-02")
    ), steps
    assert steps.index("day-04-01") < steps.index("tasks-01") < steps.index("final-02"), steps
    assert set(written["scene_of"]) == {m["id"] for m in written["world"]["messages"]}
    assert all(s["participant_ids"] for s in written["scenes"]) and any(s["beats"] for s in written["scenes"])
    assert any(s["promises"] for s in written["scenes"]), "a conversation shows the promises its lines make"
    ids = [a["id"] for a in run["attempts"]]
    assert ids.index("organization") < ids.index("ledger-01") < ids.index("build-s1-01"), ids
    assert ids.index("build-s2-02") < ids.index("tasks-01") < ids.index("final-01"), "attempts in run order"
    assert any(e["event"] == "attempt_routed" for e in run["events"]), "routes reach the page"
    assert all(
        run["snapshots"][a["snapshot"]]["messages"] for a in run["attempts"] if a["id"].startswith("final")
    ), "each attempt shows the world it ended with"
    assert set(run["scene_of"]) == {m["id"] for m in run["world"]["messages"]}, (
        "every message maps to its scene"
    )
    tasks = {t["id"]: t for t in run["catalog"]["tasks"]}
    assert tasks and all(t["gold_sql"] for t in tasks.values()), "every task shows its gold query"
    anchored = [t for t, b in run["bindings"].items() if tasks[t]["fact_ids"]]
    assert anchored and all(run["bindings"][t]["claims"][0]["message_ids"] for t in anchored), (
        "ledger tasks show the messages their facts are stated in"
    )
    assert all(p["profile"] for p in run["catalog"]["personas"]), "the cast shows seeded profiles"
    attack = "</script><script>alert('artifact')</script>&"
    html = render({"runs": [], "probe": attack})
    assert attack not in html and "__VIEWER_DATA__" not in html and "\\u003c/script\\u003e" in html
    print(
        "Viewer checks passed: attempt order, routes, snapshots, scene mapping, gold queries, cast, escaping."
    )


if __name__ == "__main__":
    main()
