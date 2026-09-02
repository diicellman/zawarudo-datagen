from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
import verifiers.v1 as vf

from worldgen_slack.agents.judge import (
    WorldJudgeData,
    WorldJudgeTask,
    make_world_judge_task,
    world_reward_scores,
)
from worldgen_slack.agents.solver import SolverTask, summarize_solver
from worldgen_slack.contracts import (
    CheckResult,
    FailureOwner,
    GenerationSeedData,
    ValidationReport,
    WorldJudgeVerdict,
    generation_result_from_episode,
)
from worldgen_slack.env import SlackDataGenerationConfig, SlackDataGenerationEnv
from worldgen_slack.progress import NullProgressJournal
from worldgen_slack.slack.tools import SlackActionRecord, SlackState


def seed_data(seed: int = 0) -> GenerationSeedData:
    return GenerationSeedData(
        idx=seed,
        name=f"seed-{seed}",
        prompt="generate",
        network_allow=[],
        network_block=["*"],
        generation_seed=seed,
        interface_id="slack.readonly.v1",
    )


def make_trace(
    name: str,
    data: vf.TaskData,
    *,
    ok: bool = True,
    reply: str = "",
    state: SlackState | None = None,
) -> vf.Trace:
    nodes = []
    if reply:
        nodes.append(vf.MessageNode(message=vf.AssistantMessage(content=reply), sampled=True))
    return vf.Trace(
        task=vf.TraceTask(type="TestTask", data=data),
        agent=vf.AgentInfo(config=vf.AgentConfig(model="test"), name=name),
        nodes=nodes,
        state=state or vf.State(),
        is_completed=True,
        ok=ok,
    )


def solver_trace(data: vf.TaskData, score: float = 1.0) -> vf.Trace:
    trace = make_trace(
        "solver",
        data,
        reply="A stale DNS cache caused the timeout; recycling edge workers fixed it.",
        state=SlackState(
            completed_actions=[SlackActionRecord(action="search_messages", arguments={"query": "timeout"})]
        ),
    )
    trace.record_reward("semantic_correctness", score)
    trace.info["solver_semantic_scores"] = {
        "semantic_correctness": score,
        "required_claim_coverage": score,
        "contradiction_free": 1.0,
        "forbidden_claim_count": 0.0,
    }
    trace.info["solver_semantic_verdict"] = {
        "required_claims": [
            {"claim_index": 0, "grade": "supported", "reason": "present"},
            {"claim_index": 1, "grade": "supported", "reason": "present"},
        ],
        "forbidden_claim_indexes": [],
        "material_contradiction": False,
        "reason": "complete",
    }
    return trace


def world_verdict(**updates) -> WorldJudgeVerdict:
    values = {
        "task_unambiguous": True,
        "world_supports_task": True,
        "scenario_alignment": "exceptional",
        "world_coherence": "strong",
        "professional_realism": "strong",
        "discoverability": "exceptional",
        "shortcut_free": "exceptional",
        "evidence_composition": "strong",
        "reason": "The inspected world supports the task.",
    }
    values.update(updates)
    return WorldJudgeVerdict.model_validate(values)


def good_validation(world) -> ValidationReport:
    return ValidationReport(
        ok=True,
        failure_owner=FailureOwner.NONE,
        checks=[CheckResult(name="world", ok=True, detail="PASS")],
        public_snapshot=world,
    )


class ConcurrentRunner:
    def __init__(
        self, name: str, trace: vf.Trace, events: list[str], barrier: asyncio.Event, starts: list[str]
    ):
        self.name = name
        self.trace = trace
        self.events = events
        self.barrier = barrier
        self.starts = starts

    async def run(self, _task, *, on_trace=None) -> vf.Trace:
        if on_trace is not None:
            on_trace(self.trace)
        self.events.append(f"{self.name}_started")
        self.starts.append(self.name)
        if len(self.starts) == 2:
            self.barrier.set()
        await asyncio.wait_for(self.barrier.wait(), 1)
        self.events.append(f"{self.name}_finished")
        return self.trace


def bare_env() -> SlackDataGenerationEnv:
    env = object.__new__(SlackDataGenerationEnv)
    env.config = SimpleNamespace(
        public_seed=0,
        hidden_seeds=[101, 202],
        brief_schedule_seed=0,
        max_builder_checks=3,
        public_check_timeout_seconds=10.0,
        candidate_timeout_seconds=20.0,
        candidate_runtime=object(),
        solver_judge=vf.JudgeConfig(model="test"),
    )
    env.progress = NullProgressJournal()
    return env


def test_config_has_narrow_retries_and_two_post_validation_agents() -> None:
    config = SlackDataGenerationConfig(taskset={"id": "worldgen-slack-generation"})
    assert config.retries.max_retries == 0
    assert config.max_concurrent_agents == 2
    assert config.synthesizer.retries.max_retries == 0
    assert config.builder.retries.max_retries == 0
    assert config.solver.retries.max_retries == 0
    assert config.judge.retries.max_retries == 1
    assert config.builder.max_turns == 18
    assert config.judge.max_turns == 10


@pytest.mark.asyncio
async def test_run_preserves_stages_and_starts_solver_and_world_judge_concurrently(
    monkeypatch, synthesized, world
) -> None:
    events: list[str] = []
    starts: list[str] = []
    barrier = asyncio.Event()
    seed = seed_data()
    builder = make_trace("builder", seed)
    builder.info["public_checks"] = [{"ok": True}]
    builder.info["builder_metadata"] = {}
    solver = solver_trace(seed)
    judge = make_trace("judge", seed, state=SlackState())
    judge.info["world_verdict"] = world_verdict().model_dump(mode="json")

    async def fake_synthesize(_seed, _agent, _brief, *, progress):
        events.append("synthesis")
        return synthesized

    async def fake_build(*_args, **_kwargs):
        events.append("builder")
        return builder, b"candidate"

    async def fake_validate(*_args, **_kwargs):
        events.append("validation")
        return good_validation(world)

    monkeypatch.setattr("worldgen_slack.env.synthesize", fake_synthesize)
    monkeypatch.setattr("worldgen_slack.env.build_world", fake_build)
    monkeypatch.setattr("worldgen_slack.env._validate_candidate", fake_validate)
    agents = SimpleNamespace(
        synthesizer=object(),
        builder=object(),
        solver=ConcurrentRunner("solver", solver, events, barrier, starts),
        judge=ConcurrentRunner("judge", judge, events, barrier, starts),
    )

    await bare_env().run(vf.Task(seed), agents)
    assert events[:3] == ["synthesis", "builder", "validation"]
    assert set(events[3:5]) == {"solver_started", "judge_started"}
    assert set(events[5:]) == {"solver_finished", "judge_finished"}
    assert solver.info["solver_summary"]["semantic_score"] == 1.0


@pytest.mark.asyncio
async def test_synthesis_rejection_ends_cleanly_without_builder(monkeypatch) -> None:
    seed = seed_data()
    calls: list[str] = []

    async def reject(*_args, **_kwargs):
        calls.append("synthesis")
        return None

    monkeypatch.setattr("worldgen_slack.env.synthesize", reject)
    await bare_env().run(vf.Task(seed), SimpleNamespace(synthesizer=object()))
    assert calls == ["synthesis"]


@pytest.mark.asyncio
async def test_deterministic_validation_rejection_stops_before_expensive_branches(
    monkeypatch, synthesized
) -> None:
    events: list[str] = []
    seed = seed_data()
    builder = make_trace("builder", seed)
    builder.info["public_checks"] = [{"ok": False}]

    async def fake_synthesize(*_args, **_kwargs):
        events.append("synthesis")
        return synthesized

    async def fake_build(*_args, **_kwargs):
        events.append("builder")
        return builder, b"candidate"

    async def reject(*_args, **_kwargs):
        events.append("validation")
        return ValidationReport(
            ok=False,
            failure_owner=FailureOwner.BUILDER,
            checks=[
                CheckResult(
                    name="required_evidence",
                    ok=False,
                    detail="evidence missing",
                    failure_owner=FailureOwner.BUILDER,
                )
            ],
        )

    monkeypatch.setattr("worldgen_slack.env.synthesize", fake_synthesize)
    monkeypatch.setattr("worldgen_slack.env.build_world", fake_build)
    monkeypatch.setattr("worldgen_slack.env._validate_candidate", reject)
    agents = SimpleNamespace(synthesizer=object(), builder=object())
    await bare_env().run(vf.Task(seed), agents)
    assert events == ["synthesis", "builder", "validation"]
    assert not builder.info["worldgen_validation"]["ok"]


@pytest.mark.asyncio
@pytest.mark.parametrize("final_roles", [("solver", "judge"), ("judge", "solver")])
async def test_finalize_uses_role_names_not_completion_order(synthesized, world, final_roles) -> None:
    seed = seed_data()
    synth = make_trace("synthesizer", seed)
    synth.info["synthesized_item"] = synthesized.model_dump(mode="json")
    builder = make_trace("builder", seed)
    builder.info["worldgen_validation"] = good_validation(world).model_dump(mode="json")
    solver = solver_trace(seed)
    solver.info["solver_summary"] = summarize_solver(solver).model_dump(mode="json")
    judge = make_trace("judge", seed, state=SlackState())
    judge.info["world_verdict"] = world_verdict().model_dump(mode="json")
    by_role = {"solver": solver, "judge": judge}
    episode = vf.Episode(
        task=vf.TraceTask(type="GenerationSeedTask", data=seed),
        traces=[synth, builder, *(by_role[name] for name in final_roles)],
    )

    await bare_env().finalize(vf.Task(seed), episode)
    episode.ok = True
    result = generation_result_from_episode(episode)
    assert result.status.value == "solved"
    assert result.solver_score == 1.0
    assert result.builder_score == pytest.approx(world_reward_scores(True, world_verdict())["world_quality"])
    assert builder.rewards["world_quality"].weight == 1.0
    assert solver.rewards["semantic_correctness"].weight == 1.0
    assert "generation_result" in builder.info
    assert "generation_result" not in solver.info
    assert "generation_result" not in judge.info


@pytest.mark.asyncio
async def test_finalize_hard_gate_zeroes_builder_reward_but_preserves_raw_diagnostics(
    synthesized, world
) -> None:
    seed = seed_data()
    synth = make_trace("synthesizer", seed)
    synth.info["synthesized_item"] = synthesized.model_dump(mode="json")
    builder = make_trace("builder", seed)
    builder.info["worldgen_validation"] = good_validation(world).model_dump(mode="json")
    solver = solver_trace(seed)
    solver.info["solver_summary"] = summarize_solver(solver).model_dump(mode="json")
    judge = make_trace("judge", seed, state=SlackState())
    judge.info["world_verdict"] = world_verdict(task_unambiguous=False).model_dump(mode="json")
    episode = vf.Episode(
        task=vf.TraceTask(type="GenerationSeedTask", data=seed),
        traces=[synth, builder, solver, judge],
    )
    await bare_env().finalize(vf.Task(seed), episode)
    result = generation_result_from_episode(episode)
    assert result.builder_raw_score > 0
    assert result.builder_score == 0
    assert result.status.value == "semantic_rejected"
    assert builder.rewards["world_quality"].score == 0


def test_solver_task_keeps_oracle_out_of_public_serialization(world, contract) -> None:
    task = SolverTask.from_snapshot(
        instance_id="fixture--000000000000",
        contract=contract,
        world=world,
    )
    assert task.data.prompt_text == contract.question
    assert task.config.oracle is not None
    assert contract.answer.canonical_answer not in task.data.model_dump_json()
    assert contract.answer.canonical_answer not in task.config.model_dump_json()
    assert "canonical_answer" not in task.config.tools.snapshot_json


def test_world_judge_receives_no_solver_artifact(synthesized, world) -> None:
    seed = seed_data()
    builder = make_trace("builder", seed)
    builder.info["generation_result"] = {"solver_response": "UNIQUE_PRIVATE_SOLVER_ANSWER"}
    builder.info["solver_reward"] = 1.0
    builder.record_reward("world_quality", 0.9)
    task = make_world_judge_task(
        generation_seed=0,
        instance_id="fixture--000000000000",
        scenario=synthesized.scenario,
        contract=synthesized.task,
        world=world,
        validation=good_validation(world),
        builder_trace=builder,
        source=b"def build(): pass",
    )
    payload = task.config.model_dump(mode="json")
    rendered = str(payload).lower()
    assert "solver_trace" not in rendered
    assert "solver_response" not in rendered
    assert "solver_reward" not in rendered
    assert "unique_private_solver_answer" not in rendered
    assert "generation_result" not in rendered


@pytest.mark.asyncio
async def test_world_judge_verdict_file_requires_typed_empirical_action() -> None:
    verdict = world_verdict()

    class Runtime:
        async def read(self, *_args, **_kwargs):
            return verdict.model_dump_json().encode()

    data = WorldJudgeData(
        idx=0,
        name="fixture",
        prompt="judge",
        network_allow=[],
        network_block=["*"],
        instance_id="fixture",
        question="question",
    )
    task = WorldJudgeTask(data, {"builder_trace": {}, "source": "x", "judge_context": {}})
    trace = make_trace(
        "judge",
        data,
        state=SlackState(completed_actions=[SlackActionRecord(action="list_conversations", arguments={})]),
    )
    await task.finalize(trace, Runtime())
    assert trace.info["world_verdict"]["task_unambiguous"] is True

    no_action = make_trace("judge", data, state=SlackState())
    with pytest.raises(ValueError, match="no Slack action"):
        await task.finalize(no_action, Runtime())

    outer_ipython = make_trace("judge", data, state=SlackState())
    outer_ipython.info["outer_trace_text"] = "slack_search_messages(query='payments')"
    with pytest.raises(ValueError, match="no Slack action"):
        await task.finalize(outer_ipython, Runtime())

    class MissingRuntime:
        async def read(self, *_args, **_kwargs):
            raise vf.SandboxError("missing")

    with pytest.raises(ValueError, match="no bounded verdict file"):
        await task.finalize(trace, MissingRuntime())

    class MalformedRuntime:
        async def read(self, *_args, **_kwargs):
            return b'{"task_unambiguous": "guess"}'

    with pytest.raises(ValueError):
        await task.finalize(trace, MalformedRuntime())
