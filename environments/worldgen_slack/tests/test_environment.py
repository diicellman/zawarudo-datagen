from __future__ import annotations

import inspect
import json
from types import SimpleNamespace

import pytest
import verifiers.v1 as vf
from verifiers.v1.configs.retries import RetryConfig
from verifiers.v1.utils.retries import episode_should_retry

from worldgen_slack.agents.builder import BUILDER_PROMPT, WORLD_STUB
from worldgen_slack.contracts import (
    CheckResult,
    FailureOwner,
    GenerationResult,
    ItemStatus,
    JudgeVerdict,
    QualityDecision,
    ValidationReport,
    authentication_failure,
    classify_item,
)
from worldgen_slack.env import (
    PendingAttempt,
    SlackDataGenerationConfig,
    SlackDataGenerationEnv,
    _item_status,
)
from worldgen_slack.generate import _EPISODE_RETRY_TYPES, _select_tasks, generate
from worldgen_slack.tasksets.generation import GenerationSeedTaskset


@pytest.mark.asyncio
async def test_environment_freezes_all_four_roles(tmp_path) -> None:
    config = SlackDataGenerationConfig(
        taskset=vf.TasksetConfig(id="worldgen-slack-generation"),
        output_dir=tmp_path,
    )
    env = SlackDataGenerationEnv(config)
    agents = [SimpleNamespace(trainable=True) for _ in range(4)]
    await env.setup(agents)
    assert all(not agent.trainable for agent in agents)


def test_builder_stub_is_empty_and_no_completion_token_is_required() -> None:
    assert "raise NotImplementedError" in WORLD_STUB
    assert "WORLD_READY" not in BUILDER_PROMPT
    assert "worldgen_slack.slack.models" in WORLD_STUB


def test_config_rejects_any_non_prime_role() -> None:
    with pytest.raises((TypeError, ValueError), match="PrimeConfig"):
        SlackDataGenerationConfig(
            taskset=vf.TasksetConfig(id="worldgen-slack-generation"),
            synthesizer=vf.AgentConfig(
                model="z-ai/glm-5.2",
                runtime=vf.DockerConfig(),
            ),
        )


def _verdict(**updates) -> JudgeVerdict:
    values = {
        "solver_correct": True,
        "task_unambiguous": True,
        "world_supports_task": True,
        "scenario_alignment": 4,
        "world_coherence": 4,
        "professional_realism": 4,
        "discoverability": 4,
        "shortcut_free": 4,
        "failure_owner": FailureOwner.NONE,
        "reason": "Verified.",
    }
    values.update(updates)
    return JudgeVerdict.model_validate(values)


def test_semantic_classification_separates_solved_challenging_and_rejected(world) -> None:
    validation = ValidationReport(
        ok=True,
        failure_owner=FailureOwner.NONE,
        checks=[CheckResult(name="all", ok=True, detail="PASS")],
        public_snapshot=world,
    )
    assert classify_item(validation, _verdict()) == "solved"
    assert (
        classify_item(
            validation,
            _verdict(solver_correct=False, failure_owner=FailureOwner.SOLVER),
        )
        == "challenging"
    )
    assert (
        classify_item(
            validation,
            _verdict(
                solver_correct=False,
                task_unambiguous=False,
                failure_owner=FailureOwner.SYNTHESIZER,
            ),
        )
        == "rejected"
    )
    infrastructure = _verdict(failure_owner=FailureOwner.INFRASTRUCTURE)
    assert classify_item(validation, infrastructure) == "rejected"


def test_quality_status_mapping_is_explicit() -> None:
    verdict = _verdict()
    solved = QualityDecision(
        status="solved",
        write_to_dataset=True,
        quality_score=0.8,
        criterion_failures=[],
    )
    assert _item_status(solved, verdict).value == "solved"
    floor = solved.model_copy(
        update={
            "write_to_dataset": False,
            "criterion_failures": ["discoverability"],
            "rejection_reason": "criterion_floor:discoverability",
        }
    )
    assert _item_status(floor, verdict).value == "criterion_floor_rejected"


def test_standalone_runner_uses_public_programmatic_env_lifecycle() -> None:
    source = inspect.getsource(generate)
    assert "env.serving()" in source
    assert "env.run_slot(" in source
    assert "asyncio.TaskGroup" in source
    assert "uv run eval" not in source


def test_tolerated_judge_trace_does_not_trigger_a_whole_episode_retry() -> None:
    error = SimpleNamespace(type="ProviderError")
    trace = SimpleNamespace(ok=False, errors=[error])
    episode = SimpleNamespace(errors=[], traces=[trace])
    retry = RetryConfig(max_retries=1, include=["ProviderError"])
    assert episode_should_retry(episode, retry)
    trace.ok = True
    assert not episode_should_retry(episode, retry)


def test_resume_count_targets_the_first_n_seeds() -> None:
    taskset = GenerationSeedTaskset(vf.TasksetConfig())
    env = SimpleNamespace(
        writer=SimpleNamespace(completed_seeds={0, 2}),
        taskset=taskset,
    )
    selected = _select_tasks(env, 3)
    assert [task.data.generation_seed for task in selected] == [1]


def test_config_rejects_mutable_prime_image_tags() -> None:
    with pytest.raises(ValueError, match="immutable sha256 digest"):
        SlackDataGenerationConfig(
            taskset=vf.TasksetConfig(id="worldgen-slack-generation"),
            candidate_runtime=vf.PrimeConfig(
                vm=True,
                image="python:3.12-slim",
                workdir="/task",
                allow=[],
                block=["*"],
            ),
        )


def test_config_rejects_mutable_or_secret_forwarding_harnesses() -> None:
    config = SlackDataGenerationConfig(taskset=vf.TasksetConfig(id="worldgen-slack-generation"))
    payload = config.model_dump(mode="python")
    payload["judge"]["harness"]["forward_env"] = ["PRIME_API_KEY"]
    with pytest.raises(ValueError, match="must not inject env"):
        SlackDataGenerationConfig.model_validate(payload)
    payload = config.model_dump(mode="python")
    payload["builder"]["harness"]["version"] = "latest"
    with pytest.raises(ValueError, match="pinned 40-hex commit"):
        SlackDataGenerationConfig.model_validate(payload)


def test_config_rejects_unrelated_generation_taskset() -> None:
    with pytest.raises(ValueError, match="generation taskset id"):
        SlackDataGenerationConfig(taskset=vf.TasksetConfig(id="harbor"))


@pytest.mark.asyncio
async def test_non_auth_operational_errors_are_redacted_from_attempts(tmp_path) -> None:
    config = SlackDataGenerationConfig(
        taskset=vf.TasksetConfig(id="worldgen-slack-generation"),
        output_dir=tmp_path,
    )
    env = SlackDataGenerationEnv(config)
    task = next(iter(GenerationSeedTaskset(config.taskset)))
    episode = vf.Episode(
        errors=[
            vf.Error(
                type="ProviderError",
                message="upstream transport failed with Bearer TOPSECRET123456789",
            )
        ]
    )

    await env.persist_episode(task, episode)

    raw = (tmp_path / "attempts.jsonl").read_text()
    assert "TOPSECRET123456789" not in raw
    row = json.loads(raw)
    assert "[REDACTED]" in row["reason"]
    assert "[REDACTED]" in row["builder"]["episode_errors"][0]["message"]


def test_authentication_detection_ignores_traceback_line_numbers() -> None:
    max_turns = vf.Error(
        type="HarnessError",
        message=(
            'Error code: 400 - rollout stopped: max_turns\nFile "/tmp/runner.py", line 401, in serve_stream'
        ),
    )
    assert authentication_failure([max_turns]) is None


@pytest.mark.parametrize(
    ("error", "expected_status"),
    [
        (vf.Error(type="ProviderError", message="denied", status_code=403), 403),
        (vf.Error(type="ProviderError", message="Error code: 401 - invalid request"), 401),
        (vf.Error(type="ProviderError", message="authentication failed"), 401),
        (vf.Error(type="ProviderError", message="HTTP status code 401"), 401),
        (vf.Error(type="ProviderError", message="status code: 403"), 403),
    ],
)
def test_authentication_detection_requires_context(error, expected_status) -> None:
    assert authentication_failure([error])[1] == expected_status


def test_structured_non_auth_status_is_authoritative() -> None:
    error = vf.Error(
        type="ProviderError",
        message="authentication failed; nested HTTP status 401",
        status_code=500,
    )
    assert authentication_failure([error]) is None


@pytest.mark.asyncio
async def test_authentication_retry_boundary_uses_non_retryable_sticky_sentinel(tmp_path) -> None:
    config = SlackDataGenerationConfig(
        taskset=vf.TasksetConfig(id="worldgen-slack-generation"),
        output_dir=tmp_path,
    )
    env = SlackDataGenerationEnv(config)
    task = next(iter(GenerationSeedTaskset(config.taskset)))
    env._attempt_traces[0] = [
        SimpleNamespace(
            info={},
            errors=[vf.Error(type="ProviderError", message="denied", status_code=403)],
        )
    ]

    with pytest.raises(vf.TaskError, match="HTTP status 403") as raised:
        await env.run(task, SimpleNamespace())

    converted = vf.Error(type="TaskError", message=str(raised.value))
    assert converted.status_code is None
    assert authentication_failure([converted])[1] == 403
    episode = SimpleNamespace(errors=[converted], traces=[])
    retry = RetryConfig(max_retries=3, include=_EPISODE_RETRY_TYPES)
    assert not episode_should_retry(episode, retry)


@pytest.mark.asyncio
async def test_pending_environment_auth_failure_never_writes_discarded_attempt(tmp_path) -> None:
    config = SlackDataGenerationConfig(
        taskset=vf.TasksetConfig(id="worldgen-slack-generation"),
        output_dir=tmp_path,
    )
    env = SlackDataGenerationEnv(config)
    task = next(iter(GenerationSeedTaskset(config.taskset)))
    env._pending[0] = PendingAttempt(
        GenerationResult(
            generation_seed=0,
            status=ItemStatus.INFRASTRUCTURE_ERROR,
            failure_owner=FailureOwner.INFRASTRUCTURE,
            reason="SandboxError: HTTP status code 401",
        )
    )
    env._attempt_traces[0] = [SimpleNamespace(info={}, errors=[])]

    with pytest.raises(vf.TaskError, match="HTTP status 401"):
        await env.run(task, SimpleNamespace())

    assert (tmp_path / "attempts.jsonl").read_text() == ""
