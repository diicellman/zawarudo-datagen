from __future__ import annotations

from types import SimpleNamespace

import pytest
import verifiers.v1 as vf
from pydantic import ValidationError
from verifiers.v1.utils.loaders import load_environment, resolve_env_config

from worldgen_slack.agents.judge import (
    JUDGE_CONTEXT_FILE,
    SOLVER_TRACE_FILE,
    VERDICT_FILE,
    JudgeConfig,
    JudgeData,
    JudgeTask,
)
from worldgen_slack.agents.solver import (
    SolverConfig,
    SolverTask,
    extract_final_answer,
    score_exact_answer,
    score_supported_answer,
    visible_errors,
)
from worldgen_slack.env import SlackDataGenerationConfig
from worldgen_slack.slack.models import AnswerSpec
from worldgen_slack.slack.toolset import SlackState, SlackToolsetConfig
from worldgen_slack.tasksets.generation import GenerationSeedTaskset


def test_solver_visible_data_has_no_private_oracle(world, contract) -> None:
    task = SolverTask.from_snapshot(
        instance_id="fixture--000000000000",
        contract=contract,
        world=world,
    )
    fields = set(task.data.model_dump())
    assert not fields & {
        "answer",
        "required_claims",
        "forbidden_claims",
        "gold_calls",
        "contract",
        "snapshot_json",
    }
    assert task.config.oracle is None
    assert contract.answer.canonical_answer not in task.data.prompt_text


def test_generated_consumer_keeps_oracle_host_side(world, contract) -> None:
    task = SolverTask.from_snapshot(
        instance_id="fixture--000000000000",
        contract=contract,
        world=world,
        include_oracle=True,
    )
    assert task.config.oracle == contract
    assert "oracle" not in task.data.model_dump()


@pytest.mark.asyncio
async def test_generated_solver_reward_requires_retrieval_and_accepts_supported_answer(
    world, contract
) -> None:
    task = SolverTask.from_snapshot(
        instance_id="fixture--000000000000",
        contract=contract,
        world=world,
        include_oracle=True,
    )
    trace = SimpleNamespace(
        last_reply=f"<final_answer>{contract.answer.canonical_answer}</final_answer>",
        state=SimpleNamespace(completed_calls=[{"tool": "slack_get_thread"}]),
    )
    assert await task.answer_correct(trace) == 1.0
    trace.state.completed_calls.clear()
    assert await task.answer_correct(trace) == 0.0


def test_answer_extraction_and_deterministic_types(contract) -> None:
    answer = contract.answer.canonical_answer
    assert extract_final_answer(f"<final_answer>{answer}</final_answer>") == answer
    assert extract_final_answer("missing") is None
    assert extract_final_answer("<final_answer>a</final_answer><final_answer>b</final_answer>") is None
    assert not score_exact_answer(answer, contract.answer)


def test_successful_tool_content_mentioning_errors_is_not_an_error() -> None:
    message = SimpleNamespace(
        model_dump=lambda **_: {
            "role": "tool",
            "content": '{"text":"Elevated 5xx errors are now resolved."}',
        }
    )
    trace = SimpleNamespace(errors=[], nodes=[SimpleNamespace(message=message)])
    assert visible_errors(trace) == []


def test_supported_answer_accepts_natural_exact_value_and_rejects_forbidden_claim() -> None:
    spec = AnswerSpec(
        kind="exact_string",
        canonical_answer="4.5%",
        required_claims=["The reported click-through rate is 4.5%."],
        forbidden_claims=["The click-through rate was 3.2%."],
    )
    assert score_supported_answer(
        "The Spring Sale click-through rate was 4.5%, according to #marketing-data.",
        spec,
    )
    assert not score_supported_answer(
        "The click-through rate was 3.2%, although another report says 4.5%.",
        spec,
    )


def test_supported_answer_requires_every_decisive_canonical_fact() -> None:
    spec = AnswerSpec(
        kind="exact_string",
        canonical_answer=(
            "An expired SSL certificate on the authentication gateway caused the patient "
            "portal to reject all session tokens."
        ),
        required_claims=["The expired certificate caused token rejection."],
    )
    assert score_supported_answer(
        "The root cause was an expired SSL certificate on the authentication gateway. "
        "Because the gateway failed validation, the patient portal rejected all session tokens.",
        spec,
    )
    assert not score_supported_answer(
        "A valid SSL certificate on the authentication gateway caused the patient portal "
        "to reject all session tokens.",
        spec,
    )
    assert not score_supported_answer(
        "The SSL certificate on the authentication gateway did not expire, but the patient "
        "portal rejected all session tokens.",
        spec,
    )
    assert not score_supported_answer(
        "An expired SSL certificate on the authentication gateway caused the patient portal "
        "to accept all session tokens.",
        spec,
    )


def test_supported_answer_rejects_clause_level_denial_and_reversed_roles() -> None:
    spec = AnswerSpec(
        kind="exact_string",
        canonical_answer=(
            "An expired SSL certificate on the authentication gateway caused the patient "
            "portal to reject all session tokens."
        ),
        required_claims=["The expired certificate caused token rejection."],
    )
    assert not score_supported_answer(
        "It is false that an expired SSL certificate on the authentication gateway caused "
        "the patient portal to reject all session tokens.",
        spec,
    )
    assert not score_supported_answer(
        "There was no evidence that an expired SSL certificate on the authentication gateway "
        "caused the patient portal to reject all session tokens.",
        spec,
    )
    assert score_supported_answer(
        "The expired SSL certificate on the authentication gateway not only caused the patient "
        "portal to reject all session tokens, it also blocked admins.",
        spec,
    )
    assert not score_supported_answer(
        "All session tokens rejected the patient portal because an expired authentication "
        "gateway caused the SSL certificate.",
        spec,
    )


def test_structured_quantities_are_atomic_and_ordered() -> None:
    spec = AnswerSpec(
        kind="exact_string",
        canonical_answer="4.5%",
        required_claims=["The reported rate is 4.5%."],
        forbidden_claims=["The reported rate is 3.2%."],
    )
    assert score_supported_answer("The answer is 4.5%.", spec)
    assert not score_supported_answer("The report had 4 sends and 5 clicks.", spec)
    assert not score_supported_answer("The answer is 5.4%.", spec)
    assert score_supported_answer("The report also mentions 2.3%, but the answer is 4.5%.", spec)
    assert not score_supported_answer("The report says 3.2%, then claims 4.5%.", spec)


def test_supported_date_can_appear_in_a_natural_answer() -> None:
    spec = AnswerSpec(
        kind="date",
        canonical_answer="2025-04-08",
        required_claims=["The launch date is April 8, 2025."],
    )
    assert score_supported_answer("The launch is scheduled for April 8, 2025.", spec)


def test_supported_list_rejects_substrings_and_extra_members() -> None:
    spec = AnswerSpec(
        kind="list",
        canonical_answer="Ann, Bob",
        required_claims=["Ann and Bob are owners."],
    )
    assert score_supported_answer("Ann and Bob", spec)
    assert not score_supported_answer("The announcement was written by Bobby.", spec)
    assert not score_supported_answer("Ann, Mallory, Bob", spec)


def test_supported_list_preserves_numeric_members() -> None:
    spec = AnswerSpec(
        kind="list",
        canonical_answer="120, 365",
        required_claims=["The totals are 120 and 365."],
    )
    assert score_supported_answer("120 and 365", spec)
    assert not score_supported_answer("120, 240, 365", spec)


def test_fact_summary_requires_every_discriminative_claim_token() -> None:
    spec = AnswerSpec(
        kind="fact_summary",
        canonical_answer="Alice owns the launch due April 8, 2025.",
        required_claims=[
            "The launch owner is Alice",
            "The launch date is April 8 2025",
        ],
    )
    assert score_supported_answer("Alice owns the launch due April 8, 2025.", spec)
    assert not score_supported_answer("The launch owner date is the 8.", spec)


def test_supported_date_rejects_multiple_distinct_dates() -> None:
    spec = AnswerSpec(
        kind="date",
        canonical_answer="2025-04-08",
        required_claims=["The launch date is April 8, 2025."],
    )
    assert not score_supported_answer("April 8, 2025 or April 9, 2025", spec)
    assert not score_supported_answer("April 9, 2025; correction: April 8, 2025", spec)


def test_generation_seed_stream_is_infinite_and_deterministic() -> None:
    taskset = GenerationSeedTaskset(vf.TasksetConfig())
    assert taskset.INFINITE
    first = list(taskset.head(4))
    second = list(taskset.head(4))
    assert [task.data.generation_seed for task in first] == [0, 1, 2, 3]
    assert [task.data.model_dump() for task in first] == [task.data.model_dump() for task in second]
    assert all(task.data.prompt is None for task in first)


def test_production_config_is_prime_vm_default_deny() -> None:
    config = SlackDataGenerationConfig(taskset=vf.TasksetConfig(id="worldgen-slack-generation"))
    for name in ("synthesizer", "builder", "solver", "judge"):
        runtime = getattr(config, name).runtime
        assert isinstance(runtime, vf.PrimeConfig)
        assert runtime.vm
        assert runtime.allow == []
        assert runtime.block == ["*"]
    assert isinstance(config.candidate_runtime, vf.PrimeConfig)
    assert config.candidate_runtime.vm
    assert config.builder.harness.builtin_skills == ["edit"]
    assert config.judge.harness.builtin_skills == []
    assert config.judge.retries.max_retries == 1
    assert set(config.judge.retries.include) == {
        "TaskError",
        "ProviderError",
        "SandboxError",
        "HarnessError",
        "InterceptionError",
        "TunnelError",
        "ToolsetError",
    }
    toolset = SlackToolsetConfig()
    assert toolset.colocated is False
    assert toolset.runtime.type == "subprocess"


class FakeRuntime:
    def __init__(self, verdict: bytes | None = None) -> None:
        self.files: dict[str, bytes] = {}
        self.verdict = verdict
        self.removed: list[str] = []

    async def run(self, argv, env):
        self.removed.extend(argv[2:])
        for path in argv[2:]:
            self.files.pop(path, None)
        return SimpleNamespace(exit_code=0, stderr="")

    async def write(self, path: str, content: bytes) -> None:
        self.files[path] = content

    async def read(self, path: str, max_bytes: int | None = None) -> bytes:
        if path == VERDICT_FILE and self.verdict is not None:
            return self.verdict
        raise FileNotFoundError(path)


def _judge_task(world, contract) -> JudgeTask:
    return JudgeTask(
        JudgeData(
            idx=0,
            name="judge-fixture",
            prompt="audit",
            instance_id="fixture--000000000000",
            question=contract.question,
        ),
        JudgeConfig(
            tools=SlackToolsetConfig.from_world(world, contract.actor_id),
            solver_trace={"id": "solver"},
            judge_context={"question": contract.question},
        ),
    )


@pytest.mark.asyncio
async def test_judge_protocol_writes_context_and_removes_stale_verdict(world, contract) -> None:
    runtime = FakeRuntime()
    task = _judge_task(world, contract)
    trace = SimpleNamespace(info={}, state=SlackState())
    await task.setup(trace, runtime)
    assert VERDICT_FILE in runtime.removed
    assert SOLVER_TRACE_FILE in runtime.files
    assert JUDGE_CONTEXT_FILE in runtime.files


@pytest.mark.asyncio
async def test_judge_protocol_requires_fresh_valid_file_and_empirical_call(world, contract) -> None:
    task = _judge_task(world, contract)
    trace = SimpleNamespace(info={}, state=SlackState())
    with pytest.raises(ValueError, match="no bounded verdict"):
        await task.finalize(trace, FakeRuntime())
    with pytest.raises(ValidationError):
        await task.finalize(trace, FakeRuntime(b"{}"))
    verdict = b'{"solver_correct":true,"task_unambiguous":true,"world_supports_task":true,"scenario_alignment":4,"world_coherence":4,"professional_realism":4,"discoverability":4,"shortcut_free":4,"failure_owner":"none","reason":"Verified."}'
    with pytest.raises(ValueError, match="empirically recorded"):
        await task.finalize(trace, FakeRuntime(verdict))
    trace.state.completed_calls.append({"tool": "slack_search_messages", "arguments": {}})
    await task.finalize(trace, FakeRuntime(verdict))
    assert trace.info["judge_verdict"]["solver_correct"] is True


def test_generated_environment_preflight_uses_typed_solver_tool_config() -> None:
    config = resolve_env_config(
        {
            "taskset": {
                "id": "worldgen-slack-generated",
                "release_dir": "/tmp/not-loaded-by-preflight",
            },
            "agent": {
                "model": "fixture-model",
                "harness": {"id": "null"},
                "runtime": {"type": "prime", "vm": True},
            },
        }
    )
    assert isinstance(config.taskset.task, SolverConfig)
    env = load_environment(config)
    assert env._requires_tunnel({}) is True
