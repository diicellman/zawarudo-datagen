from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .contracts import redact_secrets


@dataclass
class ActiveStage:
    stage: str
    started: float
    detail: dict[str, Any] = field(default_factory=dict)
    branches: dict[str, float] = field(default_factory=dict)


class NullProgressJournal:
    enabled = False

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    def attempt_started(self, seed: int) -> None:
        return None

    def stage_started(self, seed: int, stage: str, **detail: Any) -> None:
        return None

    def stage_updated(self, seed: int, event: str = "updated", **detail: Any) -> None:
        return None

    def stage_finished(self, seed: int, stage: str, *, ok: bool = True, **detail: Any) -> None:
        return None

    def event(self, seed: int, stage: str, event: str, **detail: Any) -> None:
        return None

    def attempt_finished(self, seed: int, *, status: str, accepted: bool, **detail: Any) -> None:
        return None

    def heartbeat(self) -> None:
        return None

    def finish_run(self, exit_reason: str) -> None:
        return None

    def fail_active(self, seed: int, *, error_type: str, detail: str | None = None) -> None:
        return None

    def interrupted_seeds(self, completed_seeds: set[int]) -> list[int]:
        return []

    def recovered_attempt(self, seed: int) -> None:
        return None


class ProgressJournal:
    enabled = True

    def __init__(
        self,
        output: str | Path,
        *,
        run_id: str,
        target_accepted: int,
        max_attempts: int,
        concurrency: int,
        interval_seconds: float,
        initial_accepted: int = 0,
        initial_attempted: int = 0,
    ) -> None:
        if interval_seconds <= 0 or interval_seconds > 10:
            raise ValueError("progress interval must be greater than zero and at most ten seconds")
        self.output = Path(output).expanduser().resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.path = self.output / "progress.jsonl"
        if self.path.is_symlink():
            raise ValueError("progress.jsonl must not be a symlink")
        self.run_id = run_id
        self.target_accepted = target_accepted
        self.max_attempts = max_attempts
        self.concurrency = concurrency
        self.interval_seconds = interval_seconds
        self.accepted = initial_accepted
        self.attempted = initial_attempted
        self.issued = initial_attempted
        self.active: dict[int, ActiveStage] = {}
        self._started = time.monotonic()
        self._lock = threading.RLock()
        self._stop = asyncio.Event()
        self._heartbeat_task: asyncio.Task[None] | None = None
        self.path.touch(exist_ok=True)
        self._validate_existing()

    def _validate_existing(self) -> None:
        data = self.path.read_bytes()
        lines = data.splitlines(keepends=True)
        offset = 0
        for number, raw in enumerate(lines, 1):
            next_offset = offset + len(raw)
            if not raw.strip():
                offset = next_offset
                continue
            try:
                row = json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                is_partial_tail = number == len(lines) and not raw.endswith((b"\n", b"\r"))
                if is_partial_tail:
                    with self.path.open("r+b") as handle:
                        handle.truncate(offset)
                        handle.flush()
                        os.fsync(handle.fileno())
                    return
                raise ValueError(f"invalid progress.jsonl line {number}") from exc
            if not isinstance(row, dict) or row.get("run_id") != self.run_id:
                raise ValueError("progress.jsonl belongs to another run")
            offset = next_offset

    async def start(self) -> None:
        if self._heartbeat_task is not None:
            raise RuntimeError("progress heartbeat is already running")
        print(
            f"run={self.run_id} target={self.target_accepted} max_attempts={self.max_attempts} "
            f"concurrency={self.concurrency}",
            flush=True,
        )
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def stop(self) -> None:
        self._stop.set()
        task = self._heartbeat_task
        if task is not None:
            await task
        self._heartbeat_task = None

    async def _heartbeat_loop(self) -> None:
        while not self._stop.is_set():
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.interval_seconds)
            except TimeoutError:
                self.heartbeat()

    @staticmethod
    def _bounded_detail(value: Any) -> Any:
        safe = redact_secrets(value)
        if isinstance(safe, str):
            raw = safe.encode("utf-8", errors="replace")[:500]
            return raw.decode("utf-8", errors="ignore")
        if safe is None or isinstance(safe, (bool, int, float)):
            return safe
        rendered = json.dumps(safe, ensure_ascii=False, sort_keys=True)
        raw = rendered.encode("utf-8", errors="replace")[:500]
        return raw.decode("utf-8", errors="ignore")

    def _emit(
        self,
        *,
        seed: int | None,
        stage: str,
        event: str,
        ok: bool | None = None,
        duration_ms: int | None = None,
        detail: Any = None,
        **fields: Any,
    ) -> None:
        now = datetime.now(UTC)
        row: dict[str, Any] = {
            "schema_version": 1,
            "timestamp_utc": now.isoformat().replace("+00:00", "Z"),
            "run_id": self.run_id,
            "seed": seed,
            "stage": stage,
            "event": event,
            "attempt": 1 if seed is not None else None,
            "ok": ok,
            "elapsed_ms": int((time.monotonic() - self._started) * 1_000),
            "duration_ms": duration_ms,
            "detail": self._bounded_detail(detail),
        }
        for name, value in fields.items():
            if isinstance(value, (str, bool, int, float)) or value is None:
                row[name] = self._bounded_detail(value)
            else:
                row[name] = self._bounded_detail(value)
        line = json.dumps(row, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n"
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
            elapsed = row["elapsed_ms"] // 1_000
            prefix = f"[{elapsed // 60:02d}:{elapsed % 60:02d}]"
            target = f"seed={seed}" if seed is not None else f"run={self.run_id}"
            extras = " ".join(f"{name}={row[name]}" for name in fields if row.get(name) is not None)
            print(
                f"{prefix} {target} {stage} {event}"
                + (f" {extras}" if extras else "")
                + (f' detail="{row["detail"]}"' if row["detail"] else ""),
                flush=True,
            )

    def attempt_started(self, seed: int) -> None:
        with self._lock:
            self.issued += 1
            self.active[seed] = ActiveStage("queued", time.monotonic())
        self._emit(seed=seed, stage="attempt", event="started", issued=self.issued)

    def stage_started(self, seed: int, stage: str, **detail: Any) -> None:
        with self._lock:
            self.active[seed] = ActiveStage(stage, time.monotonic(), dict(detail))
        self._emit(seed=seed, stage=stage, event="started", **detail)

    def stage_updated(self, seed: int, event: str = "updated", **detail: Any) -> None:
        with self._lock:
            active = self.active.get(seed)
            if active is None:
                raise ValueError(f"seed {seed} has no active progress stage")
            active.detail.update(detail)
            stage = active.stage
        self._emit(seed=seed, stage=stage, event=event, **detail)

    def stage_finished(self, seed: int, stage: str, *, ok: bool = True, **detail: Any) -> None:
        with self._lock:
            active = self.active.get(seed)
            if active is None or active.stage != stage:
                raise ValueError(f"seed {seed} is not active in stage {stage!r}")
            duration_ms = int((time.monotonic() - active.started) * 1_000)
            self.active[seed] = ActiveStage("between", time.monotonic())
        self._emit(
            seed=seed,
            stage=stage,
            event="finished",
            ok=ok,
            duration_ms=duration_ms,
            **detail,
        )

    def event(self, seed: int, stage: str, event: str, **detail: Any) -> None:
        ok = detail.pop("ok", None)
        duration_ms = detail.pop("duration_ms", None)
        with self._lock:
            active = self.active.get(seed)
            if active is not None:
                if event == "started":
                    active.branches[stage] = time.monotonic()
                    active.detail[f"{stage}_status"] = "active"
                elif event == "finished":
                    active.branches.pop(stage, None)
                    active.detail[f"{stage}_status"] = "finished" if ok is not False else "failed"
                elif event == "retry":
                    active.detail[f"{stage}_retry"] = detail.get("retry")
        self._emit(
            seed=seed,
            stage=stage,
            event=event,
            ok=ok,
            duration_ms=duration_ms,
            **detail,
        )

    def attempt_finished(self, seed: int, *, status: str, accepted: bool, **detail: Any) -> None:
        with self._lock:
            if seed not in self.active:
                raise ValueError(f"seed {seed} is not active")
            self.active.pop(seed)
            self.attempted += 1
            self.accepted += int(accepted)
        self._emit(
            seed=seed,
            stage="attempt",
            event="finished",
            ok=accepted,
            status=status,
            accepted=self.accepted,
            attempted=self.attempted,
            **detail,
        )

    def interrupted_seeds(self, completed_seeds: set[int]) -> list[int]:
        started: set[int] = set()
        finished: set[int] = set()
        for raw in self.path.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            row = json.loads(raw)
            seed = row.get("seed")
            if not isinstance(seed, int) or row.get("stage") != "attempt":
                continue
            if row.get("event") == "started":
                started.add(seed)
            elif row.get("event") == "finished":
                finished.add(seed)
        return sorted(started - finished - completed_seeds)

    def recovered_attempt(self, seed: int) -> None:
        with self._lock:
            self.attempted += 1
            self.issued = max(self.issued, self.attempted)
        self._emit(
            seed=seed,
            stage="attempt",
            event="finished",
            ok=False,
            status="interrupted_recovered",
            accepted=self.accepted,
            attempted=self.attempted,
        )

    def fail_active(self, seed: int, *, error_type: str, detail: str | None = None) -> None:
        with self._lock:
            active = self.active.get(seed)
            stage = active.stage if active is not None else None
        if stage not in {None, "between", "queued"}:
            self.stage_finished(
                seed,
                stage,
                ok=False,
                error_type=error_type,
                detail=detail,
            )

    def finish_run(self, exit_reason: str) -> None:
        self._emit(
            seed=None,
            stage="run",
            event="finished",
            ok=self.accepted >= self.target_accepted,
            accepted=self.accepted,
            attempted=self.attempted,
            exit_reason=exit_reason,
        )

    def heartbeat(self) -> None:
        now = time.monotonic()
        with self._lock:
            active = [
                (
                    seed,
                    state.stage,
                    int(now - state.started),
                    dict(state.detail),
                    {name: int(now - started) for name, started in state.branches.items()},
                )
                for seed, state in sorted(self.active.items())
            ]
            accepted = self.accepted
            completed = self.attempted
            issued = self.issued
        self._emit(
            seed=None,
            stage="run",
            event="heartbeat",
            accepted=accepted,
            target=self.target_accepted,
            attempted=issued,
            completed_attempts=completed,
            active=len(active),
        )
        for seed, stage, elapsed, detail, branches in active:
            scalar_detail = {
                name: value
                for name, value in detail.items()
                if isinstance(value, (str, bool, int, float)) or value is None
            }
            self._emit(
                seed=seed,
                stage=stage,
                event="heartbeat",
                stage_elapsed_seconds=elapsed,
                **scalar_detail,
            )
            for branch, branch_elapsed in sorted(branches.items()):
                self._emit(
                    seed=seed,
                    stage=branch,
                    event="heartbeat",
                    branch_elapsed_seconds=branch_elapsed,
                )


__all__ = ["NullProgressJournal", "ProgressJournal"]
