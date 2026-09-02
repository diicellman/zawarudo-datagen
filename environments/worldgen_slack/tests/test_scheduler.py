from __future__ import annotations

import asyncio
import threading
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
import verifiers.v1 as vf

import worldgen_slack.generate as generation
from worldgen_slack.contracts import (
    FailedStage,
    FailureKind,
    FailureOwner,
    GenerationSeedData,
)
from worldgen_slack.env import SlackDataGenerationConfig
from worldgen_slack.generate import GenerationConfig, generate


def seed_data(seed: int) -> GenerationSeedData:
    return GenerationSeedData(
        idx=seed,
        name=f"seed-{seed}",
        prompt="generate",
        network_allow=[],
        network_block=["*"],
        generation_seed=seed,
        interface_id="slack.readonly.v1",
    )


class InfiniteTaskset:
    def __iter__(self):
        seed = 0
        while True:
            yield vf.Task(seed_data(seed))
            seed += 1


class FakeWriter:
    initial_attempted = 0
    initial_accepted = 0
    initial_completed: set[int] = set()
    accept_from_seed: int | None = None
    last: "FakeWriter | None" = None

    def __init__(self, _output, *, target_accepted, max_attempts, **_kwargs):
        self.target = target_accepted
        self.max_attempts = max_attempts
        self.attempted = self.initial_attempted
        self.accepted = self.initial_accepted
        self._completed = set(self.initial_completed)
        self.seeds: list[int] = []
        self._lock = threading.Lock()
        FakeWriter.last = self

    @property
    def attempted_count(self):
        return self.attempted

    @property
    def accepted_count(self):
        return self.accepted

    @property
    def completed_seeds(self):
        return set(self._completed)

    def _row(self, seed: int, accepted: bool):
        with self._lock:
            self.attempted += 1
            self.accepted += int(accepted)
            self._completed.add(seed)
            self.seeds.append(seed)
        return {
            "written_to_dataset": accepted,
            "item_status": "solved" if accepted else "infrastructure_error",
            "solver_score": 1.0 if accepted else None,
            "builder_score": 1.0 if accepted else None,
            "reason": None if accepted else "failed",
        }

    def write_failed_episode(self, *, generation_seed, **_kwargs):
        return self._row(generation_seed, False)

    def write_attempt(self, result, **_kwargs):
        accepted = self.accept_from_seed is not None and result.generation_seed >= self.accept_from_seed
        return self._row(result.generation_seed, accepted)

    def release_uncommitted_signatures(self, _seed):
        return None

    def write_summary(self, *, exit_reason=None, release_integrity_ok=None):
        return {
            "run_id": "scheduler-test",
            "output": "unused",
            "target_accepted": self.target,
            "actual_accepted": self.accepted,
            "attempted": self.attempted,
            "max_attempts": self.max_attempts,
            "qualification_passed": self.accepted >= self.target,
            "exit_reason": exit_reason,
            "attempts_per_accepted_world": self.attempted / self.accepted if self.accepted else None,
            "solver_score_distribution": {"mean": None},
            "builder_score_distribution": {"mean": None},
            "total_reported_cost": None,
            "role_usage": {},
        }


class FakeEnv:
    max_active = 0
    active = 0

    def __init__(self, _config, *, progress):
        self.taskset = InfiniteTaskset()
        self.progress = progress

    def slots(self, task, _n=1):
        return [SimpleNamespace(task=task)]

    @asynccontextmanager
    async def serving(self):
        yield

    async def run_slot(self, slot, _context, *, on_complete):
        FakeEnv.active += 1
        FakeEnv.max_active = max(FakeEnv.max_active, FakeEnv.active)
        await asyncio.sleep(0.001)
        seed = slot.task.data.generation_seed
        accepted_path = FakeWriter.accept_from_seed is not None and seed >= FakeWriter.accept_from_seed
        episode = vf.Episode(
            task=vf.TraceTask(type="GenerationSeedTask", data=slot.task.data),
            ok=accepted_path,
            errors=([] if accepted_path else [vf.Error(type="ProviderError", message="failed")]),
        )
        await on_complete(episode)
        FakeEnv.active -= 1
        return episode


def config(tmp_path, *, target=1, cap=5, concurrency=2) -> GenerationConfig:
    return GenerationConfig(
        run_id="scheduler-test",
        target_accepted=target,
        max_attempts=cap,
        concurrency=concurrency,
        progress_interval_seconds=0.1,
        output=tmp_path,
        env=SlackDataGenerationConfig(taskset={"id": "worldgen-slack-generation"}),
    )


@pytest.fixture(autouse=True)
def reset_fakes(monkeypatch):
    FakeWriter.initial_attempted = 0
    FakeWriter.initial_accepted = 0
    FakeWriter.initial_completed = set()
    FakeWriter.accept_from_seed = None
    FakeEnv.active = 0
    FakeEnv.max_active = 0
    monkeypatch.setattr(generation, "DatasetWriter", FakeWriter)
    monkeypatch.setattr(generation, "SlackDataGenerationEnv", FakeEnv)
    monkeypatch.setattr(generation, "validate_release_integrity", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        generation,
        "generation_result_from_episode",
        lambda episode: SimpleNamespace(generation_seed=episode.task.data.generation_seed),
    )
    monkeypatch.setattr(generation, "_release_result", lambda result, _acceptance: result)
    monkeypatch.setattr(generation, "_builder_artifacts", lambda _episode: (None, [], {}))
    monkeypatch.setattr(generation, "_deduplicate_result", lambda result, _writer, _threshold: (result, None))


@pytest.mark.asyncio
async def test_fixed_pool_schedules_replacements_and_drains_inflight_successes(tmp_path) -> None:
    FakeWriter.accept_from_seed = 2
    summary = await generate(config(tmp_path, target=1, cap=6, concurrency=2), check_models=False)
    writer = FakeWriter.last
    assert writer is not None
    assert writer.seeds == [0, 1, 2, 3]
    assert summary["actual_accepted"] == 2
    assert summary["qualification_passed"] is True
    assert FakeEnv.max_active == 2


@pytest.mark.asyncio
async def test_attempt_cap_is_exact_under_concurrency(tmp_path) -> None:
    summary = await generate(config(tmp_path, target=2, cap=5, concurrency=3), check_models=False)
    writer = FakeWriter.last
    assert writer is not None
    assert sorted(writer.seeds) == [0, 1, 2, 3, 4]
    assert len(writer.seeds) == 5
    assert summary["actual_accepted"] == 0
    assert summary["exit_reason"] == "attempt_cap_exhausted"
    assert summary["qualification_passed"] is False
    assert FakeEnv.max_active == 3


@pytest.mark.asyncio
async def test_resume_skips_noncontiguous_completed_seeds_and_stops_if_target_exists(tmp_path) -> None:
    FakeWriter.initial_attempted = 3
    FakeWriter.initial_accepted = 1
    FakeWriter.initial_completed = {0, 2, 7}
    summary = await generate(config(tmp_path, target=1, cap=5, concurrency=2), check_models=False)
    assert summary["attempted"] == 3
    assert FakeWriter.last is not None and FakeWriter.last.seeds == []
    assert FakeEnv.max_active == 0


def trace(role: str, *, ok: bool, error: vf.Error | None = None) -> vf.Trace:
    value = vf.Trace(
        task=vf.TraceTask(type="GenerationSeedTask", data=seed_data(0)),
        agent=vf.AgentInfo(config=vf.AgentConfig(model="test"), name=role),
        is_completed=True,
        ok=ok,
    )
    if error is not None:
        value.errors.append(error)
    return value


def test_operational_failure_taxonomy_uses_failed_role_not_present_role() -> None:
    episode = vf.Episode(
        task=vf.TraceTask(type="GenerationSeedTask", data=seed_data(0)),
        traces=[
            trace(
                "solver", ok=False, error=vf.Error(type="TaskError", message="semantic judge verdict missing")
            ),
            trace("judge", ok=True),
        ],
    )
    kind = generation._failure_kind(episode)
    assert generation._failure_stage(episode) == FailedStage.SOLVER_ANSWER_JUDGE
    assert kind == FailureKind.PROTOCOL_ERROR
    assert generation._failure_owner(episode, kind) == FailureOwner.SOLVER

    judged = vf.Episode(
        task=vf.TraceTask(type="GenerationSeedTask", data=seed_data(0)),
        traces=[
            trace("solver", ok=True),
            trace(
                "judge",
                ok=False,
                error=vf.Error(type="TaskError", message="world judge completed no Slack action"),
            ),
        ],
    )
    assert generation._failure_stage(judged) == FailedStage.WORLD_JUDGE
    assert generation._failure_kind(judged) == FailureKind.PROTOCOL_ERROR
    assert generation._failure_owner(judged, FailureKind.PROTOCOL_ERROR) == FailureOwner.INTERFACE


def test_episode_level_validation_failure_uses_builder_stage_marker() -> None:
    builder = trace("builder", ok=True)
    builder.info["current_pipeline_stage"] = "validation"
    episode = vf.Episode(
        task=vf.TraceTask(type="GenerationSeedTask", data=seed_data(0)),
        traces=[trace("synthesizer", ok=True), builder],
        errors=[
            vf.Error(
                type="EnvError",
                message="SlackDataGenerationEnv.run(): candidate validation provision failed",
            )
        ],
    )
    assert generation._failure_stage(episode) == FailedStage.VALIDATION
