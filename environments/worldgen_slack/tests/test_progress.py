from __future__ import annotations

import asyncio
import json

import pytest

from worldgen_slack.progress import ProgressJournal


def journal(tmp_path, **updates) -> ProgressJournal:
    values = {
        "run_id": "progress-run",
        "target_accepted": 10,
        "max_attempts": 25,
        "concurrency": 5,
        "interval_seconds": 0.02,
    }
    values.update(updates)
    return ProgressJournal(tmp_path, **values)


@pytest.mark.asyncio
async def test_heartbeat_reports_all_active_seeds_and_issued_attempts(tmp_path) -> None:
    progress = journal(tmp_path)
    await progress.start()
    for seed in range(5):
        progress.attempt_started(seed)
        progress.stage_started(seed, "builder", check=0, max_checks=3)
    await asyncio.sleep(0.05)
    await progress.stop()
    rows = [json.loads(line) for line in (tmp_path / "progress.jsonl").read_text().splitlines()]
    run_heartbeats = [row for row in rows if row["stage"] == "run" and row["event"] == "heartbeat"]
    assert run_heartbeats
    assert run_heartbeats[-1]["attempted"] == 5
    assert run_heartbeats[-1]["completed_attempts"] == 0
    assert run_heartbeats[-1]["active"] == 5
    seed_heartbeats = {
        row["seed"] for row in rows if row["event"] == "heartbeat" and row["stage"] == "builder"
    }
    assert seed_heartbeats == set(range(5))
    assert all(
        {
            "schema_version",
            "timestamp_utc",
            "run_id",
            "seed",
            "stage",
            "event",
            "attempt",
            "ok",
            "elapsed_ms",
            "duration_ms",
            "detail",
        }
        <= set(row)
        for row in rows
    )


def test_branch_heartbeats_track_solver_and_world_judge_separately(tmp_path) -> None:
    progress = journal(tmp_path)
    progress.attempt_started(0)
    progress.stage_started(0, "evaluation")
    progress.event(0, "solver", "started")
    progress.event(0, "world_judge", "started")
    progress.event(0, "solver", "finished", ok=True)
    progress.heartbeat()
    rows = [json.loads(line) for line in (tmp_path / "progress.jsonl").read_text().splitlines()]
    assert any(row["stage"] == "world_judge" and row["event"] == "heartbeat" for row in rows)
    evaluation = [row for row in rows if row["stage"] == "evaluation" and row["event"] == "heartbeat"][-1]
    assert evaluation["solver_status"] == "finished"
    assert evaluation["world_judge_status"] == "active"


def test_progress_recovers_only_an_unterminated_tail(tmp_path) -> None:
    progress = journal(tmp_path)
    progress.attempt_started(0)
    valid = (tmp_path / "progress.jsonl").read_bytes()
    with (tmp_path / "progress.jsonl").open("ab") as handle:
        handle.write(b'{"partial"')
    resumed = journal(tmp_path)
    assert (tmp_path / "progress.jsonl").read_bytes() == valid
    assert resumed.interrupted_seeds(set()) == [0]

    with (tmp_path / "progress.jsonl").open("ab") as handle:
        handle.write(b"not-json\n")
    with pytest.raises(ValueError, match="invalid progress.jsonl"):
        journal(tmp_path)


def test_progress_bounds_and_redacts_file_and_console_detail(tmp_path, capsys) -> None:
    progress = journal(tmp_path)
    progress.attempt_started(0)
    progress.stage_started(0, "synthesis")
    progress.stage_updated(
        0,
        event="repair",
        detail="token=abcdefghijk " + "x" * 1_000,
    )
    output = capsys.readouterr().out
    row = json.loads((tmp_path / "progress.jsonl").read_text().splitlines()[-1])
    assert "abcdefghijk" not in output
    assert "abcdefghijk" not in row["detail"]
    assert "[REDACTED]" in row["detail"]
    assert len(row["detail"].encode()) <= 500


def test_completed_attempt_leaves_active_state(tmp_path) -> None:
    progress = journal(tmp_path)
    progress.attempt_started(4)
    progress.stage_started(4, "persistence")
    progress.stage_finished(4, "persistence", ok=True)
    progress.attempt_finished(4, status="rejected", accepted=False)
    assert 4 not in progress.active
    assert progress.attempted == 1
