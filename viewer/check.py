"""Small offline check: run with uv run --frozen python viewer/check.py."""

import json
import tempfile
from pathlib import Path

from build import ROOT, load_run, render


def main():
    run = load_run(ROOT / "data/qualification-01/software")
    assert run["attempts"][0]["id"] == "catalog-01"
    assert run["attempts"][0]["validation"]["ok"] is False
    world = next(a for a in run["attempts"] if a["id"] == "build-grp_helix48-01")
    assert len(world["candidate"]["snapshot"]["messages"]) == 150
    assert world["tasks"][0]["answer"]["canonical_answer"]
    assert world["verdict"]["approved"] is False
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp)
        attempt = path / "attempts/broken"
        attempt.mkdir(parents=True)
        (attempt / "author_output.json").write_text(json.dumps({"text": "not json"}))
        broken = load_run(path)["attempts"][0]
        assert broken["candidate"] is None
        assert "viewer_parse_error" in broken["validation"]
    attack = "</script><script>alert('artifact')</script>&"
    html = render({"runs": [], "probe": attack})
    assert attack not in html
    assert "__VIEWER_DATA__" not in html
    assert "\\u003c/script\\u003e" in html
    print("Viewer checks passed: ordering, rejected artifacts, evidence, malformed output, HTML escaping.")


if __name__ == "__main__":
    main()
