"""Local artifacts and one atomic restart checkpoint; execution belongs to the Env."""

import fcntl
import json
import os
import time
from pathlib import Path
from typing import Literal
from pydantic import Field
from worldgen_slack.dataset import PublicTask, atomic_json, read_json, write_release
from worldgen_slack.slack.api import canonical, digest
from worldgen_slack.slack.models import StrictModel
from .contracts import Catalog, Candidate, Verdict, validate_candidate, validate_verdict


class RunState(StrictModel):
    phase: Literal["catalog", "build", "final", "evaluate", "done"] = "catalog"
    catalog: Catalog | None = None
    candidate: Candidate | None = None
    built_groups: list[str] = Field(default_factory=list)
    final_groups: list[str] = Field(default_factory=list)
    rounds: dict[str, int] = Field(default_factory=dict)
    reviews: dict[str, dict] = Field(default_factory=dict)
    evaluation: dict[str, dict] = Field(default_factory=dict)
    feedback: str = ""
    catalog_budget: str = "catalog"
    active_attempt: str | None = None
    last_approved_candidate: str | None = None


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
            checkpoint = read_json(self.root / "state.json")
            for key in ("catalog", "candidate"):
                if checkpoint[key]:
                    identifier = checkpoint[key]
                    if len(identifier) != 64 or any(c not in "0123456789abcdef" for c in identifier):
                        raise ValueError("invalid artifact reference")
                    content = read_json(self.root / "artifacts" / (identifier + ".json"))
                    if digest(content) != identifier:
                        raise ValueError("checkpoint artifact hash mismatch")
                    checkpoint[key] = content
            self.state = RunState.model_validate(checkpoint)
            if self.state.active_attempt:
                self.event("interrupted", attempt=self.state.active_attempt)
                self.state.active_attempt = None
                self.save()
        else:
            if any(p.name != ".lock" for p in self.root.iterdir()):
                raise ValueError("output is nonempty without a run manifest")
            self.state = RunState()
            atomic_json(manifest, config)
            self.save()

    def close(self):
        self.lock.close()

    def event(self, event, **fields):
        row = {"time": time.time(), "event": event, "phase": self.state.phase, **fields}
        with (self.root / "progress.jsonl").open("ab") as stream:
            stream.write(canonical(row) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        print(json.dumps(row), flush=True)

    def save(self, *, approved=False):
        checkpoint = self.state.model_dump(mode="json")
        for key in ("catalog", "candidate"):
            if checkpoint[key] is not None:
                value = checkpoint[key]
                identifier = digest(value)
                path = self.root / "artifacts" / (identifier + ".json")
                if not path.exists():
                    atomic_json(path, value)
                checkpoint[key] = identifier
        if (
            approved
            and checkpoint["candidate"]
            and any(key.startswith(("build:", "final:")) for key in self.state.reviews)
        ):
            self.state.last_approved_candidate = checkpoint["candidate"]
            checkpoint["last_approved_candidate"] = checkpoint["candidate"]
        atomic_json(self.root / "state.json", checkpoint)

    def reserve(self, key, maximum):
        used = self.state.rounds.get(key, 0)
        phase, _, group = key.partition(":")
        consumed = (
            sum(self.state.rounds.get(f"{stage}:{group}", 0) for stage in ("build", "final"))
            if phase in {"build", "final"}
            else used
        )
        if consumed >= maximum:
            raise ReviewLimit(f"review limit exhausted for {key}: {consumed}/{maximum}")
        self.state.rounds[key] = used + 1
        attempt = f"{key.replace(':', '-')}-{used + 1:02d}"
        self.state.active_attempt = attempt
        self.save()
        self.event("candidate_started", attempt=attempt, round=used + 1)
        return attempt

    def artifact(self, attempt, name, value):
        atomic_json(self.root / "attempts" / attempt / (name + ".json"), value)

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
        self.event("candidate_finished", attempt=self.state.active_attempt, approved=approved)
        self.state.active_attempt = None
        self.save(approved=approved)

    def invalidate(self, reason, *, catalog=True):
        self.state.final_groups.clear()
        self.state.reviews = {
            key: review for key, review in self.state.reviews.items() if key == "catalog" and not catalog
        }
        self.state.evaluation.clear()
        self.state.last_approved_candidate = None
        self.event("approvals_invalidated", reason=reason)

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
                        "unpriced_calls": 0,
                    },
                )
                row["judge_calls" if is_judge else "traces"] += 1
                row["input_tokens"] += (account.get("prompt_tokens") or 0) + (
                    account.get("cached_input_tokens") or 0
                )
                row["output_tokens"] += account.get("completion_tokens") or 0
                if account.get("cost") is None:
                    row["unpriced_calls"] += 1
                else:
                    row["reported_cost"] += account["cost"]
        events = (
            [json.loads(line) for line in (self.root / "progress.jsonl").read_text().splitlines()]
            if (self.root / "progress.jsonl").exists()
            else []
        )
        evidence = {}
        if self.state.candidate:
            for binding in self.state.candidate.bindings:
                for identifier in {m for c in binding.claims for m in c.message_ids}:
                    evidence.setdefault(identifier, set()).add(binding.task_id)
        results = list(self.state.evaluation.values())
        summary = {
            "status": status,
            "reason": reason,
            "phase": self.state.phase,
            "rounds": self.state.rounds,
            "built_groups": self.state.built_groups,
            "final_groups": self.state.final_groups,
            "task_count": len(self.state.catalog.tasks) if self.state.catalog else 0,
            "message_count": len(self.state.candidate.snapshot.messages) if self.state.candidate else 0,
            "workspace_hash": digest(self.state.candidate.snapshot.model_dump(mode="json"))
            if self.state.candidate
            else None,
            "evaluated_tasks": len(results),
            "solved_tasks": sum(r["semantic_correctness"] == 1 for r in results),
            "grounded_tasks": sum(r["grounded"] for r in results),
            "solver_calls": sum(r["read_count"] for r in results),
            "solver_execution_failures": sum(not r["execution_ok"] for r in results),
            "usage_by_role": usage,
            "reported_model_cost": sum(r["reported_cost"] for r in usage.values()),
            "elapsed_seconds": events[-1]["time"] - events[0]["time"] if events else 0,
            "rejected_candidates": sum(
                e["event"] == "candidate_finished" and not e["approved"] for e in events
            ),
            "evidence_messages": len(evidence),
            "reused_evidence_messages": sum(len(tasks) > 1 for tasks in evidence.values()),
            "last_approved_candidate": self.state.last_approved_candidate,
        }
        atomic_json(self.root / "summary.json", summary)
        return summary

    def publish(self):
        from .agents.judge import review_payload
        from .env import reference_for

        state = self.state
        if state.phase != "done" or set(state.evaluation) != {t.id for t in state.catalog.tasks}:
            raise ValueError("release requires every task to finish evaluation")
        catalog, candidate = state.catalog, state.candidate
        if not validate_candidate(catalog, candidate, [t.id for t in catalog.tasks])["ok"]:
            raise ValueError("release fails deterministic validation")
        scopes = [("catalog", None, [t.id for t in catalog.tasks], "catalog")]
        scopes += [
            ("final:" + g.id, candidate, [t.id for t in catalog.tasks if t.group_id == g.id], "world")
            for g in catalog.groups
        ]
        for key, snapshot, ids, phase in scopes:
            verdict = Verdict.model_validate(state.reviews[key])
            validate_verdict(verdict, review_payload(catalog, snapshot, ids, phase))
            if not verdict.approved:
                raise ValueError("release includes rejected work")
        world_hash = digest(candidate.snapshot.model_dump(mode="json"))
        rows = [
            PublicTask(
                task_id=t.id,
                workspace_id=catalog.workspace_id,
                question=t.question,
                actor_id=t.actor_id,
                snapshot_ref="snapshot.json",
                snapshot_hash=world_hash,
            )
            for t in catalog.tasks
        ]
        write_release(
            self.root / "release",
            candidate.snapshot,
            rows,
            {t.id: reference_for(t, candidate) for t in catalog.tasks},
        )
