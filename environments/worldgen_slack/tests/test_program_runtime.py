from __future__ import annotations

from pathlib import Path

import pytest
import verifiers.v1 as vf

from conftest import WORLD_PROGRAM
from worldgen_slack.env import SlackDataGenerationConfig
from worldgen_slack.slack.validation import evaluate_candidate_in_runtime, static_source_errors


def test_fixture_program_matches_the_fixed_build_contract() -> None:
    assert static_source_errors(WORLD_PROGRAM) == []
    assert "worldgen_slack.slack.models" in WORLD_PROGRAM


@pytest.mark.asyncio
async def test_candidate_execution_refuses_every_non_prime_or_permissive_runtime(contract) -> None:
    with pytest.raises(TypeError, match="PrimeConfig"):
        await evaluate_candidate_in_runtime(WORLD_PROGRAM.encode(), contract, vf.DockerConfig())
    with pytest.raises(ValueError, match="vm=true"):
        await evaluate_candidate_in_runtime(
            WORLD_PROGRAM.encode(),
            contract,
            vf.PrimeConfig(vm=False, allow=[], block=["*"]),
        )
    with pytest.raises(ValueError, match="framework-only"):
        await evaluate_candidate_in_runtime(
            WORLD_PROGRAM.encode(),
            contract,
            vf.PrimeConfig(vm=True, allow=["example.com"]),
        )


def test_production_source_has_no_docker_or_subprocess_fallback() -> None:
    package = Path(__file__).parents[1] / "worldgen_slack"
    production = "\n".join(path.read_text() for path in package.rglob("*.py"))
    assert "DockerConfig" not in production
    assert "SubprocessConfig" not in production
    assert "subprocess.run" not in production


def test_all_runtime_images_are_one_prime_family() -> None:
    config = SlackDataGenerationConfig(taskset=vf.TasksetConfig(id="worldgen-slack-generation"))
    runtimes = [
        config.synthesizer.runtime,
        config.builder.runtime,
        config.solver.runtime,
        config.judge.runtime,
        config.candidate_runtime,
    ]
    assert {runtime.image for runtime in runtimes} == {
        "python:3.12-slim@sha256:2c941e860699f878900b0edc2403613c234d4b32eda3cc9fa7036991a2a63c4a"
    }
    assert all(runtime.labels == ["worldgen-slack", config.run_id] for runtime in runtimes)
