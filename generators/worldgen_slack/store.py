"""The run directory: the world file, one atomic checkpoint, attempt snapshots, events, traces and the release."""

import fcntl
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Literal

from pydantic import Field
from worldgen_slack.dataset import (
    PrivateAnswer,
    PublicTask,
    StrictModel,
    atomic_json,
    read_json,
    sha256,
    write_release,
)
from worldgen_slack.db import World, canonical

from .contracts import Issue, Premise, SeedPersona, Slot, Verdict, measures


def used_names(corpus: Path, exclude: Path) -> dict[str, list[str]]:
    """Companies and people of every other world in the corpus."""
    companies, people = set(), set()
    for path in corpus.glob("**/world.sqlite"):
        if (
            path.parent.resolve() == exclude.resolve()
            or path.parent.name in ("release", "attempts")
            or "attempts" in path.parts
        ):
            continue
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
            if company := db.execute("SELECT value FROM world_meta WHERE key = 'company'").fetchone():
                companies.add(company[0])
            people.update(r[0] for r in db.execute("SELECT real_name FROM users"))
    return {"companies": sorted(companies), "people": sorted(people)}


class RunState(StrictModel):
    phase: Literal["premise", "organization", "plan", "day", "review", "tasks", "final", "done"] = "premise"
    premise: Premise | None = None
    cast: list[SeedPersona] = Field(default_factory=list)
    quota: list[Slot] = Field(default_factory=list)  # the world's tasks, one per slot
    organization: list[str] = Field(default_factory=list)  # the channels the organization made
    drafts: dict[str, dict] = Field(default_factory=dict)  # each setup phase's last document
    plans: dict[str, dict] = Field(
        default_factory=dict
    )  # "agenda": day → the everyday conversations code drew
    task_reviews: dict[str, dict] = Field(default_factory=dict)
    rounds: dict[str, int] = Field(default_factory=dict)
    reviews: dict[str, dict] = Field(default_factory=dict)
    feedback: str = ""
    active_attempt: str | None = None
    last_verdict: Verdict | None = None
    # The world author: the day being written (or reviewed), the attempt whose world and notes it starts from, a
    # review's issues waiting for the author's next turn, the solver's runs of each task as it was when solved
    # (probes and the final review share them), and the solver runs spent on probing tasks.
    day: int = 0
    restore_point: str = ""
    issues: list[Issue] = Field(default_factory=list)
    solves: dict[str, dict] = Field(default_factory=dict)  # task → {key, results, traces}
    probe_solves: int = 0


class ReviewLimit(RuntimeError):
    pass


class Store:
    def __init__(self, root: Path, config: dict):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = (self.root / ".lock").open("a")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = self.root / "run.json"
        if manifest.exists():
            if read_json(manifest) != config:
                raise ValueError("resume configuration differs; use a new output directory")
            self.state = RunState.model_validate_json((self.root / "state.json").read_bytes())
            self.world = World(self.root / "world.sqlite", writable=True)
            if self.state.active_attempt:
                self.event("interrupted", attempt=self.state.active_attempt)
                self.state.active_attempt = None
                self.save()
        else:
            if any(p.name != ".lock" for p in self.root.iterdir()):
                raise ValueError("output is nonempty without a run manifest")
            self.state = RunState()
            self.world = World.create(self.root / "world.sqlite")
            atomic_json(manifest, config)
            self.save()

    def close(self):
        self.world.close()
        self.lock.close()

    def event(self, event, **fields):
        row = {"time": time.time(), "event": event, "phase": self.state.phase, **fields}
        with (self.root / "progress.jsonl").open("ab") as stream:
            stream.write(canonical(row) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        print(json.dumps(row), flush=True)

    def save(self):
        atomic_json(self.root / "state.json", self.state.model_dump(mode="json"))

    def reserve(self, key, maximum):
        used = self.state.rounds.get(key, 0)
        if used >= maximum:
            raise ReviewLimit(f"review limit exhausted for {key}: {used}/{maximum}")
        self.state.rounds[key] = used + 1
        attempt = f"{key.replace(':', '-')}-{used + 1:02d}"
        self.state.active_attempt = attempt
        self.save()
        self.event("candidate_started", attempt=attempt, round=used + 1)
        return attempt

    def path(self, attempt: str, name: str) -> Path:
        path = self.root / "attempts" / attempt / name
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def artifact(self, attempt, name, value):
        atomic_json(self.path(attempt, name + ".json"), value)

    def snapshot(self, attempt: str, name: str = "world.sqlite") -> Path:
        """The world as it is now, kept with the attempt; reviews and the viewer read these files."""
        path = self.path(attempt, name)
        path.unlink(missing_ok=True)
        self.world.snapshot(path)
        return path

    def restore(self, attempt: str) -> None:
        """The world as an attempt left it: a day is always written from where the last one closed."""
        source = sqlite3.connect(self.path(attempt, "world.sqlite"))
        try:
            source.backup(self.world.db)
        finally:
            source.close()

    def trace(self, trace):
        record = trace.to_record()
        record["usage"] = trace.usage.model_dump(mode="json") if trace.usage else None
        atomic_json(self.root / "traces" / (trace.id + ".json"), record)
        self.event(
            "agent_finished" if trace.is_completed else "agent_progress",
            role=trace.agent.name,
            trace_id=trace.id,
            ok=trace.ok,
            usage=record["usage"],
        )

    def finish_attempt(self, approved):
        attempt = self.state.active_attempt
        if attempt:
            self.snapshot(attempt)
        self.event("candidate_finished", attempt=attempt, approved=approved)
        self.state.active_attempt = None
        self.save()

    def summary(self, status, reason=""):
        usage = {}
        for path in (self.root / "traces").glob("*.json"):
            trace = read_json(path)
            accounts = [(trace["agent"]["name"], trace.get("usage") or {}, False)]
            accounts += [("answer_judge", value, True) for value in trace.get("extra_usage", [])]
            for name, account, is_judge in accounts:
                row = usage.setdefault(
                    name,
                    {
                        "traces": 0,
                        "judge_calls": 0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "reported_cost": 0.0,
                    }
                    | {"unpriced_calls": 0, "failed_traces": 0},
                )
                row["judge_calls" if is_judge else "traces"] += 1
                row["failed_traces"] += not is_judge and trace.get("ok") is False
                row["input_tokens"] += (account.get("prompt_tokens") or 0) + (
                    account.get("cached_input_tokens") or 0
                )
                row["output_tokens"] += account.get("completion_tokens") or 0
                if account.get("cost") is None:
                    row["unpriced_calls"] += 1
                else:
                    row["reported_cost"] += account["cost"]
        events = []
        if (self.root / "progress.jsonl").exists():
            events = [json.loads(line) for line in (self.root / "progress.jsonl").read_text().splitlines()]
        rates = {task: {k: review[k] for k in ("right_rate", "strict_rate", "coverage", "tries", "crashed")} for task, review in self.state.task_reviews.items()}  # fmt: skip
        right = [r["right_rate"] for r in rates.values()]
        count = lambda sql: self.world.db.execute(sql).fetchone()[0]  # noqa: E731
        summary = {
            "status": status,
            "reason": reason,
            "phase": self.state.phase,
            "day": self.state.day,
            "rounds": self.state.rounds,
            "final_approved": "final" in self.state.reviews,
            "tasks": count("SELECT COUNT(*) FROM tasks"),
            "messages": count("SELECT COUNT(*) FROM messages"),
            "task_mix": [
                list(r)
                for r in self.world.db.execute("SELECT category, level, COUNT(*) FROM tasks GROUP BY 1, 2")
            ],
            "rates": rates,
            "mean_learnability": sum(4 * p * (1 - p) for p in right) / len(right) if right else None,
            "usage_by_role": usage,
            "reported_model_cost": sum(r["reported_cost"] for r in usage.values()),
            "elapsed_seconds": events[-1]["time"] - events[0]["time"] if events else 0,
            "rejected_candidates": sum(
                e["event"] == "candidate_finished" and not e["approved"] for e in events
            ),
        }
        if status != "running":
            summary["difficulty"] = self.difficulty()
        atomic_json(self.root / "summary.json", summary)
        return summary

    def difficulty(self) -> dict:
        """Each task's intended level beside how hard it measures on the final world: the judge's level_fit, the
        solver's right-answer and strict rates, and code's measures (evidence pages, tables read, search rank)."""
        out = {}
        for task in self.world.db.execute(
            "SELECT id, category, level, concept, actor_id, gold_sql FROM tasks"
        ):
            review = self.state.task_reviews.get(task["id"], {})
            try:
                gold = self.world.gold(task["actor_id"], task["gold_sql"], max_rows=50)
                measured = measures(self.world, task["id"], gold["rows"], gold["tables"])
            except ValueError as error:
                measured = {"error": str(error)}
            out[task["id"]] = (
                {k: task[k] for k in ("category", "level", "concept")}
                | {
                    "level_fit": review.get("level_fit"),
                    "right_rate": review.get("right_rate"),
                    "strict_rate": review.get("strict_rate"),
                }
                | measured
            )
        return out

    def release_rows(self, world_hash: str) -> tuple[list[PublicTask], dict[str, PrivateAnswer]]:
        """Each task's public row and private answer: its gold rows, and the messages and people they rest on."""
        rows, answers = [], {}
        db = self.world.db
        for task in db.execute("SELECT * FROM tasks ORDER BY id").fetchall():
            gold = json.loads(task["gold_json"])
            ids = {r["message_id"] for r in gold if r.get("message_id") is not None}
            ids |= {
                r[0]
                for r in db.execute(
                    """SELECT e.message_id FROM task_facts tf JOIN evidence e ON e.fact_id = tf.fact_id
                    WHERE tf.task_id = ? AND e.role = 'anchor'""",
                    (task["id"],),
                )
            }
            marks = ", ".join("?" * len(ids))
            messages = [
                list(r)
                for r in db.execute(
                    f"SELECT channel_id, ts FROM messages WHERE id IN ({marks}) ORDER BY ts_us", sorted(ids)
                )
            ]
            rows.append(
                PublicTask(task_id=task["id"], question=task["question"], actor_id=task["actor_id"], category=task["category"], level=task["level"], answer_type=task["answer_type"], world_hash=world_hash)
            )  # fmt: skip
            answers[task["id"]] = PrivateAnswer(
                answer_type=task["answer_type"],
                rows=gold,
                gold_sql=task["gold_sql"],
                messages=messages,
                users=sorted({r["user_id"] for r in gold if r.get("user_id") is not None}),
            )
        return rows, answers

    def publish(self):
        if self.state.phase != "done":
            raise ValueError("release requires a finished run")
        if missing := [
            t for (t,) in self.world.db.execute("SELECT id FROM tasks") if t not in self.state.task_reviews
        ]:
            raise ValueError(f"release requires an approved review of every task; missing {missing}")
        if self.world.violations(complete=True):
            raise ValueError("release requires a world that breaks no rule")
        copy = self.root / "solver.sqlite"
        copy.unlink(missing_ok=True)
        self.world.solver_copy(copy)
        rows, answers = self.release_rows(sha256(copy))
        write_release(self.root / "release", copy, rows, answers)
