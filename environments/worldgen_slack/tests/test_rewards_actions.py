from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import verifiers.v1 as vf

from worldgen_slack.agents.judge import world_reward_scores
from worldgen_slack.agents.solver import (
    RequiredClaimVerdict,
    SlackAnswerJudge,
    SolverAnswerVerdict,
    SolverTask,
    solver_verdict_scores,
)
from worldgen_slack.contracts import WorldJudgeVerdict
from worldgen_slack.slack.api import SlackNotFoundError
from worldgen_slack.slack.tools import SlackActionRecord, SlackState, SlackToolset, SlackToolsetConfig


def verdict(grades, *, forbidden=(), contradiction=False) -> SolverAnswerVerdict:
    return SolverAnswerVerdict(
        required_claims=[
            RequiredClaimVerdict(claim_index=index, grade=grade, reason="checked")
            for index, grade in enumerate(grades)
        ],
        forbidden_claim_indexes=list(forbidden),
        material_contradiction=contradiction,
        reason="checked against the private oracle",
    )


def test_solver_semantic_scoring_is_continuous_and_contradictions_zero(contract) -> None:
    full = solver_verdict_scores(verdict(["supported", "supported"]), contract.answer)
    partial = solver_verdict_scores(verdict(["supported", "partial"]), contract.answer)
    missing = solver_verdict_scores(verdict(["missing", "missing"]), contract.answer)
    contradicted = solver_verdict_scores(
        verdict(["supported", "contradicted"], contradiction=False), contract.answer
    )
    forbidden = solver_verdict_scores(verdict(["supported", "supported"], forbidden=[0]), contract.answer)
    assert full["semantic_correctness"] == 1.0
    assert partial["semantic_correctness"] == 0.75
    assert missing["semantic_correctness"] == 0.0
    assert contradicted["required_claim_coverage"] == 0.5
    assert contradicted["semantic_correctness"] == 0.0
    assert contradicted["contradiction_free"] == 0.0
    assert forbidden["semantic_correctness"] == 0.0


@pytest.mark.asyncio
async def test_solver_task_records_named_primary_and_diagnostic_rewards(monkeypatch, contract, world) -> None:
    task = SolverTask.from_snapshot(
        instance_id="fixture--000000000000",
        contract=contract,
        world=world,
        answer_judge=vf.JudgeConfig(model="test/answer-judge"),
    )
    data = task.data
    trace = vf.Trace(
        task=vf.TraceTask(type="SolverTask", data=data),
        agent=vf.AgentInfo(config=vf.AgentConfig(model="test/solver"), name="solver"),
        nodes=[vf.MessageNode(message=vf.AssistantMessage(content="complete answer"), sampled=True)],
        state=SlackState(),
        is_completed=True,
        ok=True,
    )
    judged = verdict(["supported", "partial"])

    async def evaluate(_self, **_kwargs):
        trace.extra_usage.append(vf.Usage(prompt_tokens=10, completion_tokens=3, cost=0.01))
        return SimpleNamespace(parsed=judged)

    monkeypatch.setattr(SlackAnswerJudge, "evaluate", evaluate)
    score = await task.semantic_correctness(trace)
    trace.record_reward("semantic_correctness", score, 1.0)
    assert score == 0.75
    assert trace.rewards["semantic_correctness"].weight == 1.0
    assert trace.rewards["required_claim_coverage"].weight == 0.0
    assert trace.rewards["contradiction_free"].weight == 0.0
    assert trace.rewards["forbidden_claim_count"].weight == 0.0
    assert trace.info["solver_answer_judge_model"] == "test/answer-judge"
    assert trace.extra_usage[0].total_tokens == 13


@pytest.mark.asyncio
async def test_slack_actions_append_only_after_success_and_are_bounded(world) -> None:
    toolset = SlackToolset(SlackToolsetConfig.from_world(world, "agent_user"))
    await toolset.setup()
    assert toolset.state.completed_actions == []
    conversations = await toolset.list_conversations()
    assert conversations
    assert toolset.state.completed_actions == [SlackActionRecord(action="list_conversations", arguments={})]
    before = list(toolset.state.completed_actions)
    with pytest.raises(SlackNotFoundError):
        await toolset.get_thread("incidents", "does-not-exist")
    assert toolset.state.completed_actions == before
    toolset.state.completed_actions = [
        SlackActionRecord(action="list_conversations", arguments={}) for _ in range(128)
    ]
    with pytest.raises(ValueError, match="exceeds 128"):
        await toolset.list_conversations()
    assert len(toolset.state.completed_actions) == 128


def world_verdict(**updates) -> WorldJudgeVerdict:
    payload = {
        "task_unambiguous": True,
        "world_supports_task": True,
        "scenario_alignment": "exceptional",
        "world_coherence": "exceptional",
        "professional_realism": "exceptional",
        "discoverability": "exceptional",
        "shortcut_free": "exceptional",
        "evidence_composition": "exceptional",
        "reason": "checked",
    }
    payload.update(updates)
    return WorldJudgeVerdict.model_validate(payload)


def test_world_reward_is_continuous_and_hard_gates_only_zero_final() -> None:
    exceptional = world_reward_scores(True, world_verdict())
    weak = world_reward_scores(
        True,
        world_verdict(
            scenario_alignment="weak",
            world_coherence="weak",
            professional_realism="weak",
            discoverability="weak",
            shortcut_free="weak",
            evidence_composition="weak",
        ),
    )
    ambiguous = world_reward_scores(True, world_verdict(task_unambiguous=False))
    invalid = world_reward_scores(False, world_verdict())
    assert exceptional["world_quality"] == 1.0
    assert weak["world_quality"] == 0.25
    assert ambiguous["world_quality"] == 0.0
    assert ambiguous["world_quality_raw"] == 1.0
    assert invalid["world_quality"] == 0.0
    assert invalid["world_quality_raw"] == 1.0


@pytest.mark.asyncio
async def test_empty_solver_response_scores_zero_without_calling_answer_judge(
    monkeypatch, contract, world
) -> None:
    task = SolverTask.from_snapshot(
        instance_id="fixture--empty",
        contract=contract,
        world=world,
        answer_judge=vf.JudgeConfig(model="test/answer-judge"),
    )
    trace = vf.Trace(
        task=vf.TraceTask(type="SolverTask", data=task.data),
        agent=vf.AgentInfo(config=vf.AgentConfig(model="test/solver"), name="solver"),
        state=SlackState(),
        is_completed=True,
        ok=True,
    )

    async def forbidden_call(*_args, **_kwargs):
        raise AssertionError("answer judge must not run for an empty response")

    monkeypatch.setattr(SlackAnswerJudge, "evaluate", forbidden_call)
    assert await task.semantic_correctness(trace) == 0.0
    assert trace.info["solver_semantic_verdict"] is None
    assert trace.info["solver_semantic_scores"]["required_claim_coverage"] == 0.0


def test_claim_verdict_cardinality_and_uniqueness_are_strict(contract) -> None:
    incomplete = verdict(["supported"])
    with pytest.raises(ValueError, match="appear exactly once"):
        incomplete.validate_against(contract.answer)
    with pytest.raises(ValueError, match="must be unique"):
        SolverAnswerVerdict(
            required_claims=[
                RequiredClaimVerdict(claim_index=0, grade="supported", reason="one"),
                RequiredClaimVerdict(claim_index=0, grade="missing", reason="duplicate"),
            ],
            material_contradiction=False,
            reason="invalid",
        )


@pytest.mark.asyncio
async def test_exact_match_is_only_diagnostic_and_cannot_override_semantic_judge(
    monkeypatch, contract, world
) -> None:
    answer = contract.answer.model_copy(
        update={
            "kind": "entity",
            "canonical_answer": "Alice Chen",
            "required_claims": ["Alice Chen"],
            "forbidden_claims": [],
        }
    )
    entity_contract = contract.model_copy(update={"answer": answer})
    task = SolverTask.from_snapshot(
        instance_id="fixture--entity",
        contract=entity_contract,
        world=world,
        answer_judge=vf.JudgeConfig(model="test/answer-judge"),
    )
    trace = vf.Trace(
        task=vf.TraceTask(type="SolverTask", data=task.data),
        agent=vf.AgentInfo(config=vf.AgentConfig(model="test/solver"), name="solver"),
        nodes=[vf.MessageNode(message=vf.AssistantMessage(content="Alice Chen"), sampled=True)],
        state=SlackState(),
        is_completed=True,
        ok=True,
    )

    async def evaluate(_self, **_kwargs):
        return SimpleNamespace(parsed=verdict(["missing"]))

    monkeypatch.setattr(SlackAnswerJudge, "evaluate", evaluate)
    assert await task.exact_answer_diagnostic(trace) == 1.0
    assert await task.semantic_correctness(trace) == 0.0


def test_world_reward_uses_configured_weights_and_unsupported_gate() -> None:
    mixed = world_reward_scores(
        True,
        world_verdict(
            scenario_alignment="exceptional",
            world_coherence="fails",
            professional_realism="fails",
            discoverability="fails",
            shortcut_free="fails",
            evidence_composition="fails",
        ),
    )
    assert mixed["world_quality_raw"] == pytest.approx(0.75 / 6.0)
    unsupported = world_reward_scores(True, world_verdict(world_supports_task=False))
    assert unsupported["world_quality"] == 0.0
    assert unsupported["world_quality_raw"] == 1.0


def test_action_arguments_are_strictly_typed_and_bounded() -> None:
    with pytest.raises(ValueError, match="4096"):
        SlackActionRecord(
            action="search_messages",
            arguments={"query": "x" * 5_000},
        )
    with pytest.raises(ValueError):
        SlackActionRecord(action="unknown", arguments={})


def test_native_tool_annotations_are_concrete_for_mcp_schema_generation() -> None:
    for name in (
        "list_conversations",
        "search_messages",
        "get_conversation_history",
        "get_thread",
        "get_user",
    ):
        signature = inspect.signature(getattr(SlackToolset, name))
        assert signature.return_annotation is not inspect.Signature.empty
        assert all(
            not isinstance(parameter.annotation, str)
            for parameter in signature.parameters.values()
            if parameter.name != "self"
        )
