"""Export saved worldgen attempts as a standalone, offline HTML review page."""

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else None


def load_run(path):
    """Read saved artifacts only; do not import or execute generator code."""
    path = path.resolve()
    order = {}
    progress = path / "progress.jsonl"
    if progress.exists():
        for line in progress.read_text().splitlines():
            event = json.loads(line)
            if event.get("attempt"):
                order.setdefault(event["attempt"], len(order))
    attempts = []
    folders = sorted(
        (path / "attempts").iterdir(),
        key=lambda p: (order.get(p.name, len(order)), p.name),
    )
    for folder in folders:
        if not folder.is_dir():
            continue
        payload = read_json(folder / "review_input.json")
        validation = read_json(folder / "validation.json")
        raw = read_json(folder / "author_output.json") or read_json(folder / "author_catalog.json")
        # Invalid author JSON is research evidence, not a reason to hide an attempt.
        parsed = None
        if payload is None and raw is not None:
            try:
                parsed = json.loads(raw["text"])
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                validation = {"saved_validation": validation, "viewer_parse_error": str(exc)}
        payload = payload or {}
        catalog = payload.get("catalog")
        candidate = payload.get("candidate")
        if isinstance(parsed, dict):
            if "snapshot" in parsed:
                candidate = parsed
            elif "tasks" in parsed:
                catalog = parsed
        verdict = read_json(folder / "verdict.json")
        attempts.append(
            {
                "id": folder.name,
                "phase": payload.get("phase", "catalog" if folder.name.startswith("catalog") else "world"),
                "catalog": catalog,
                "candidate": candidate,
                "tasks": payload.get("tasks", (catalog or {}).get("tasks", [])),
                "verdict": verdict,
                "validation": validation,
                "source": str(folder),
            }
        )
    if not attempts:
        raise ValueError(f"No saved attempts found in {path}")
    return {
        "name": f"{path.parent.name} / {path.name}",
        "path": str(path),
        "attempts": attempts,
        "summary": read_json(path / "summary.json"),
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
    parser.add_argument("runs", nargs="*", type=Path, help="Run directories containing attempts/")
    parser.add_argument("--output", type=Path, default=HERE / "artifacts.html")
    args = parser.parse_args()
    paths = args.runs or [ROOT / "data/v2-09" / name for name in ("software", "employee", "services")]
    output = args.output.resolve()
    if not output.is_relative_to(HERE):
        parser.error("Output must be inside viewer/ so run artifacts cannot be overwritten")
    if output.suffix != ".html" or output == HERE / "template.html":
        parser.error("Choose a generated .html output other than template.html")
    data = {"runs": [load_run(path) for path in paths]}
    output.write_text(render(data))
    print(f"Wrote {output} ({output.stat().st_size:,} bytes)")
    print("Open this file in a browser. It contains private answers and full world snapshots.")


if __name__ == "__main__":
    main()
