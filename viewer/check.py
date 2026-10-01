"""Offline check on a scripted run: uv run --frozen python viewer/check.py"""

import asyncio
import tempfile
from pathlib import Path

from build import load_run, render
from generators.worldgen_slack.check import check_flow


def main():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "run"
        asyncio.run(check_flow(root))
        run = load_run(root)
    ids = [a["id"] for a in run["attempts"]]
    assert ids[:5] == ["build-g1-01", "build-g1-02", "build-g2-01", "build-g2-02", "final-01"], (
        "attempts in run order"
    )
    assert any(e["event"] == "attempt_routed" for e in run["events"]), "routes reach the page"
    assert run["chains"].keys() == run["bindings"].keys(), "every task shows its gold route"
    for task_id, steps in run["chains"].items():
        bound = {m for claim in run["bindings"][task_id]["claims"] for m in claim["message_ids"]}
        assert {m for step in steps for m in step["evidence"]} == bound, (
            "the route reaches all bound evidence"
        )
    assert set(run["scene_of"]) == {m["id"] for m in run["world"]["messages"]}, (
        "every message maps to its scene"
    )
    attack = "</script><script>alert('artifact')</script>&"
    html = render({"runs": [], "probe": attack})
    assert attack not in html and "__VIEWER_DATA__" not in html and "\\u003c/script\\u003e" in html
    print("Viewer checks passed: attempt order, routes, gold chains, scene mapping, HTML escaping.")


if __name__ == "__main__":
    main()
