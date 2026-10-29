"""Serve a built viewer page with labelling: the page at /, and the person's task labels at /labels, saved to a JSON
file the moment each is made (`<label set>/<task id>` → the fields, with when). Local only: the page carries answers.

uv run --frozen python viewer/serve.py viewer/artifacts-forge-w2.html [--port 8766] [--labels viewer/labels.json]

Opened as a file instead, the page keeps labels in the browser and exports them as JSON."""

import argparse
import json
import os
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIELDS = {
    "fair",
    "reading",
    "failure",
    "realistic",
    "skill",
    "note",
    "outcome",
    "parts",
    "codes",
}  # the label panel; a try's outcome, and a blind ruling's outcome by part with its reason codes
lock = threading.Lock()


def load(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, suffix=".tmp") as stream:
        json.dump(value, stream, indent=1, ensure_ascii=False)
    os.replace(stream.name, path)


def handler(page: Path, labels: Path):
    class Handler(BaseHTTPRequestHandler):
        def send(self, body: bytes, kind: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                return self.send(page.read_bytes(), "text/html; charset=utf-8")
            if self.path == "/labels":
                return self.send(json.dumps(load(labels)).encode(), "application/json")
            self.send(b"not found", "text/plain", 404)

        def do_POST(self):
            if self.path != "/labels":
                return self.send(b"not found", "text/plain", 404)
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            task, fields = body.get("id"), {k: v for k, v in (body.get("label") or {}).items() if k in FIELDS}
            if not isinstance(task, str) or not fields:
                return self.send(b'{"ok": false}', "application/json", 400)
            with lock:
                saved = load(labels)
                saved[task] = (
                    saved.get(task, {}) | fields | {"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
                )
                save(labels, saved)
            self.send(json.dumps({"ok": True, "label": saved[task]}).encode(), "application/json")

        def log_message(self, *args):
            pass

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("page", type=Path, help="a page viewer/build.py wrote")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--labels", type=Path, default=HERE / "labels.json")
    args = parser.parse_args()
    if not args.page.exists():
        parser.error(f"{args.page} does not exist: build it with viewer/build.py first")
    server = ThreadingHTTPServer(
        ("127.0.0.1", args.port), handler(args.page.resolve(), args.labels.resolve())
    )
    print(f"viewer on http://127.0.0.1:{args.port} (labels in {args.labels})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
