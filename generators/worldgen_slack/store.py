"""The run directory: the world file, one atomic checkpoint, attempt snapshots, events, traces and the release."""

import fcntl
from collections import Counter
import json
import os
import shutil
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import Literal

from pydantic import Field
from worldgen_slack import slack
from worldgen_slack.config import role
from worldgen_slack.release import (
    FILES,
    Manifest,
    PrivateAnswer,
    PublicTask,
    StrictModel,
    atomic_json,
    load_release,
    read_json,
)
from worldgen_slack.world import canonical, file_hash
from .workspace import Workspace

from .forge.difficulty import measures
from .forge.tries import miss_class
from .world.board import Slot
from .world.cast import Premise, SeedPersona
from .judge.verdict import Issue, Verdict


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
    quota: list[Slot] = Field(default_factory=list)  # the board's entries: a chapter's brief
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
    # review's issues waiting for the author's next turn. The forge: the solver's runs of each task as it was when
    # solved, and the round it writes next.
    day: int = 0
    restore_point: str = ""
    issues: list[Issue] = Field(default_factory=list)
    solves: dict[str, dict] = Field(default_factory=dict)  # task → {key, results, traces}
    batch: int = 0  # the round the forge writes next
    witness: dict[str, dict] = Field(default_factory=dict)  # task → the witness's runs, with its key
    task_issues: dict[str, list[Issue]] = Field(default_factory=dict)  # each task's latest review's issues
    lost: dict[str, int] = Field(default_factory=dict)  # attempts a lost session cut off: they spend no round
    # Each task's screen: the wrong answers its grade accepted, with its key. The forge: each candidate's outcome,
    # and the archive's places (category and measured level), each holding its tasks.
    screens: dict[str, dict] = Field(default_factory=dict)
    forged: dict[str, dict] = Field(default_factory=dict)
    archive: dict[str, list[dict]] = Field(default_factory=dict)


class ReviewLimit(RuntimeError):
    pass


class SessionLost(RuntimeError):
    """An author's session ended on an error that is none of its own limits: its sandbox, tunnel or provider."""


class Store:
    def __init__(self, root: Path, config: dict):
        self.root = root.resolve()
        self.settings = config.get("config", {})  # the run's configuration, as its manifest records it
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = (self.root / ".lock").open("a")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = self.root / "run.json"
        if manifest.exists():
            if read_json(manifest) != config:
                raise ValueError("resume configuration differs; use a new output directory")
            self.state = RunState.model_validate_json((self.root / "state.json").read_bytes())
            self.world = Workspace(self.root / "world.sqlite", writable=True)
            if self.state.active_attempt:
                self.event("interrupted", attempt=self.state.active_attempt)
                self.state.active_attempt = None
                self.save()
        else:
            if any(p.name != ".lock" for p in self.root.iterdir()):
                raise ValueError("output is nonempty without a run manifest")
            self.state = RunState()
            self.world = Workspace.create(self.root / "world.sqlite")
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
        if used - self.state.lost.get(key, 0) >= maximum:
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

    def lose(self) -> str | None:
        """The active attempt was cut off by a lost session, which is no attempt of the author's: it is closed, its
        files kept for reading, and it spends none of its key's rounds."""
        attempt = self.state.active_attempt
        if attempt:
            key = attempt.rsplit("-", 1)[0]
            self.state.lost[key] = self.state.lost.get(key, 0) + 1
            self.event("candidate_finished", attempt=attempt, approved=False, lost=True)
            self.state.active_attempt = None
            self.save()
        return attempt

    def summary(self, status, reason=""):
        usage = {}
        # Each trace's own calls and its answer judge's; then the answer judge's grades of the failure cases.
        books = [
            [(t["agent"]["name"], t.get("usage") or {}, False, t.get("ok"))]
            + [("answer_judge", u, True, None) for u in t.get("extra_usage", [])]
            for t in map(read_json, (self.root / "traces").glob("*.json"))
        ]
        books.append(
            [("answer_judge", u, True, None) for s in self.state.screens.values() for u in s.get("usage", [])]
        )
        for accounts in books:
            for name, account, is_judge, ok in accounts:
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
                row["failed_traces"] += not is_judge and ok is False
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
        levels = dict(self.world.db.execute("SELECT id, level FROM tasks").fetchall())
        rates = {}
        for task, review in self.state.task_reviews.items():
            rates[task] = {
                k: review.get(k)
                for k in (
                    "right_rate",
                    "strict_rate",
                    "coverage",
                    "tries",
                    "crashed",
                    "unanswered",
                    "witness_right",
                    "level_fit",
                )
            }
            if task in levels:
                rates[task]["level"] = levels[task]
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
            "crashed_solves": sum(r["crashed"] or 0 for r in rates.values()),
            "unanswered_tries": sum(r["unanswered"] or 0 for r in rates.values()),
            "witness_tries": sum(len(w["results"]) for w in self.state.witness.values()),
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
        """Each task's public row and private answer: its gold rows, and the messages and people they rest on; a
        refusal's truth out of its actor's sight goes apart, as `unseen`. The messages an answer rests on are its gold
        rows' and those stating its answer facts: the task's facts its actor can read that are no decoy and that no
        other of them supersedes. A stale value or a decoy beside them is not its evidence."""
        rows, answers = [], {}
        db = self.world.db
        for task in db.execute("SELECT * FROM tasks ORDER BY id").fetchall():
            gold = json.loads(task["gold_json"])
            anchors = {
                (f, m)
                for f, m in db.execute(
                    """SELECT e.fact_id, e.message_id FROM task_facts tf JOIN evidence e ON e.fact_id = tf.fact_id
                    WHERE tf.task_id = ?""",
                    (task["id"],),
                )
            }
            stating = {m for _, m in anchors} | {
                r["message_id"] for r in gold if r.get("message_id") is not None
            }
            marks = ", ".join("?" * len(stating))
            reader = Workspace(self.world.path, actor=task["actor_id"])
            readable = {
                r[0]
                for r in reader.db.execute(f"SELECT id FROM messages WHERE id IN ({marks})", sorted(stating))
            }
            reader.close()
            decoys = {
                f
                for (f,) in db.execute(
                    "SELECT f.id FROM task_facts tf JOIN facts f ON f.id = tf.fact_id WHERE tf.task_id = ? AND f.is_decoy = 1",
                    (task["id"],),
                )
            }
            seen = {f for f, m in anchors if m in readable} - decoys
            stale = {
                d
                for s, d in db.execute(
                    "SELECT src_fact, dst_fact FROM fact_relations WHERE kind = 'supersedes'"
                )
                if s in seen
            }
            ids = {m for f, m in anchors if f in seen - stale} | {
                r["message_id"] for r in gold if r.get("message_id") is not None
            }
            stated = db.execute(
                f"SELECT id, channel_id, ts FROM messages WHERE id IN ({marks}) ORDER BY ts_us",
                sorted(stating),
            ).fetchall()
            messages = [[r["channel_id"], r["ts"]] for r in stated if r["id"] in readable and r["id"] in ids]
            unseen = [[r["channel_id"], r["ts"]] for r in stated if r["id"] not in readable]
            review, env = self.state.task_reviews.get(task["id"]), self.settings.get("env", {})
            measured = (
                {}
                if review is None
                else {
                    k: review.get(k)
                    for k in (
                        "tries",
                        "crashed",
                        "unanswered",
                        "right_rate",
                        "right_interval",
                        "strict_rate",
                        "abstain_rate",
                        "coverage",
                        "witness_right",
                    )
                }
                | {
                    "solver": env.get("solver", {}).get("model"),
                    "witness": env.get("witness", {}).get("model")
                    if review.get("witness_right") is not None
                    else None,
                }
            )
            if review is not None:
                wrong = [
                    r for r in review.get("results", []) if not r.get("correct") and not r.get("crashed")
                ]
                measured |= {
                    "surface": slack.VERSION,
                    "turns": env.get("solver", {}).get("max_turns") or role("", solver=True).max_turns,
                    "miss_classes": dict(Counter(miss_class(r, task["answer_type"]) for r in wrong)),
                }
            rows.append(
                PublicTask(
                    task_id=task["id"],
                    question=task["question"],
                    actor_id=task["actor_id"],
                    category=task["category"],
                    level=task["level"],
                    answer_type=task["answer_type"],
                    world_hash=world_hash,
                    intent=task["intent"],
                    **measured,
                )
            )
            answers[task["id"]] = PrivateAnswer(
                answer_type=task["answer_type"],
                rows=gold,
                gold_sql=task["gold_sql"],
                messages=messages,
                unseen=unseen,
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
        rows, answers = self.release_rows(file_hash(copy))
        write_release(self.root / "release", copy, rows, answers)


def write_release(root: Path, world: Path, rows: list[PublicTask], answers: dict[str, PrivateAnswer]) -> None:
    """Publish atomically; an existing release must be the same one."""
    if root.exists():
        _, old_rows, old_answers = load_release(root)
        if (old_rows, old_answers) != (rows, answers) or file_hash(root / "world.sqlite") != file_hash(world):
            raise ValueError("refusing to overwrite a different release")
        return
    root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root.parent, prefix=".release-") as temporary:
        stage = Path(temporary) / "release"
        stage.mkdir()
        shutil.copyfile(world, stage / "world.sqlite")
        atomic_json(stage / "tasks.json", [r.model_dump(mode="json") for r in rows])
        atomic_json(stage / "answers.json", {k: v.model_dump(mode="json") for k, v in answers.items()})
        files = {name: file_hash(stage / name) for name in sorted(FILES)}
        atomic_json(
            stage / "manifest.json", Manifest(world_hash=files["world.sqlite"], files=files).model_dump()
        )
        if load_release(stage)[1:] != (rows, answers):
            raise ValueError("published data differs from approved inputs")
        os.rename(stage, root)
