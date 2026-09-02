from __future__ import annotations

from contextlib import asynccontextmanager

import pytest
import verifiers.v1 as vf

import worldgen_slack.generate as generation
from worldgen_slack.agents.synthesizer import bounded_response, synthesis_brief_at
from worldgen_slack.contracts import GenerationSeedData
from worldgen_slack.env import SlackDataGenerationConfig
from worldgen_slack.generate import GenerationConfig, run_synthesis_diagnostic


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


class DiagnosticTaskset:
    def head(self, count):
        return [vf.Task(seed_data(seed)) for seed in range(count)]


class FakeDiagnosticEnv:
    valid_count = 8
    retain_raw = True
    synthesized_payload = None

    def __init__(self, config):
        self.config = config
        self.taskset = DiagnosticTaskset()

    def slots(self, task, _n=1):
        return [task]

    @asynccontextmanager
    async def serving(self):
        yield

    async def run_slot(self, task, _context, *, semaphore):
        async with semaphore:
            seed = task.data.generation_seed
            data = task.data
            trace = vf.Trace(
                task=vf.TraceTask(type="SynthesizerTask", data=data),
                agent=vf.AgentInfo(config=vf.AgentConfig(model="test/synth"), name="synthesizer"),
                is_completed=True,
                ok=True,
            )
            trace.info["synthesis_brief"] = synthesis_brief_at(
                self.config.brief_schedule_seed, seed
            ).model_dump(mode="json")
            trace.info["synthesis_repairs"] = int(seed >= self.valid_count)
            if seed < self.valid_count:
                trace.info["synthesized_item"] = self.synthesized_payload
            else:
                trace.info["synthesizer_error"] = "ValueError: duplicate JSON key 'task'"
                if self.retain_raw:
                    trace.info["synthesizer_raw_response"] = bounded_response('{"task":1,"task":2}')
            return vf.Episode(
                task=vf.TraceTask(type="GenerationSeedTask", data=data),
                traces=[trace],
                ok=True,
            )


def config(tmp_path) -> GenerationConfig:
    return GenerationConfig(
        run_id="diagnostic-test",
        target_accepted=1,
        max_attempts=3,
        concurrency=3,
        progress_interval_seconds=0.1,
        output=tmp_path / "unused-release",
        env=SlackDataGenerationConfig(taskset={"id": "worldgen-slack-generation"}),
    )


@pytest.fixture(autouse=True)
def fake_diagnostic(monkeypatch, synthesized):
    FakeDiagnosticEnv.valid_count = 8
    FakeDiagnosticEnv.retain_raw = True
    FakeDiagnosticEnv.synthesized_payload = synthesized.model_dump(mode="json")
    monkeypatch.setattr(generation, "SynthesisDiagnosticEnv", FakeDiagnosticEnv)
    monkeypatch.setattr(generation, "preflight_synthesized_item", lambda *_args, **_kwargs: None)


@pytest.mark.asyncio
async def test_synthesis_diagnostic_passes_at_eight_of_ten_and_never_builds(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "worldgen_slack.agents.builder.build_world",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("builder invoked")),
    )
    output = tmp_path / "diagnostic"
    report = await run_synthesis_diagnostic(config(tmp_path), attempts=10, output=output, check_models=False)
    assert report["valid"] == 8
    assert report["yield"] == 0.8
    assert report["passed"] is True
    assert report["checks"]["builder_not_invoked"] is True
    assert report["checks"]["brief_distribution_balanced"] is True
    assert (output / "synthesis_diagnostic.json").is_file()
    rows = (output / "synthesis_diagnostic_attempts.jsonl").read_text().splitlines()
    assert len(rows) == 10


@pytest.mark.asyncio
async def test_synthesis_diagnostic_fails_below_yield_or_without_real_raw_response(
    tmp_path,
) -> None:
    FakeDiagnosticEnv.valid_count = 7
    report = await run_synthesis_diagnostic(
        config(tmp_path), attempts=10, output=tmp_path / "seven", check_models=False
    )
    assert report["passed"] is False
    assert report["checks"]["yield_at_least_80_percent"] is False

    FakeDiagnosticEnv.valid_count = 8
    FakeDiagnosticEnv.retain_raw = False
    missing = await run_synthesis_diagnostic(
        config(tmp_path), attempts=10, output=tmp_path / "missing-raw", check_models=False
    )
    assert missing["passed"] is False
    assert missing["checks"]["failed_responses_bounded_and_explained"] is False


@pytest.mark.asyncio
async def test_synthesis_diagnostic_validates_only_effective_synthesizer_model(tmp_path, monkeypatch) -> None:
    seen: list[set[str]] = []

    def validate(required):
        seen.append(required)
        return required

    monkeypatch.setattr(generation, "validate_required_models", validate)
    cfg = config(tmp_path)
    report = await run_synthesis_diagnostic(
        cfg, attempts=10, output=tmp_path / "model-scope", check_models=True
    )
    assert seen == [{cfg.env.synthesizer.model}]
    assert report["models"] == [cfg.env.synthesizer.model]
