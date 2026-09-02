from __future__ import annotations

from collections import Counter

import pytest
from pydantic import BaseModel, ValidationError, field_validator

from worldgen_slack.agents.synthesizer import (
    bounded_response,
    build_synthesis_schedule,
    make_synthesizer_task,
    repair_prompt,
    synthesis_brief_at,
)
from worldgen_slack.contracts import (
    AnswerKind,
    EvidenceLayout,
    GenerationSeedData,
    ReasoningShape,
    SynthesisBrief,
    Workflow,
    parse_synthesized_item,
    preflight_synthesized_item,
)


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


def test_actual_brief_schedule_is_prefix_stable_balanced_compatible_and_unique() -> None:
    for schedule_seed in range(100):
        schedule = build_synthesis_schedule(schedule_seed, 25)
        assert schedule[:10] == build_synthesis_schedule(schedule_seed, 10)
        assert schedule == tuple(synthesis_brief_at(schedule_seed, i) for i in range(25))
        assert len({brief.model_dump_json() for brief in schedule}) == 25
        first_ten = schedule[:10]
        for field, enum_type in (
            ("workflow", Workflow),
            ("reasoning_shape", ReasoningShape),
            ("evidence_layout", EvidenceLayout),
            ("answer_kind", AnswerKind),
        ):
            counts = Counter(getattr(brief, field) for brief in first_ten)
            values = [counts[value] for value in enum_type]
            assert max(values) - min(values) <= 1
        assert all(
            brief.evidence_layout == EvidenceLayout.MULTIPLE_CHANNELS
            for brief in schedule
            if brief.reasoning_shape == ReasoningShape.CROSS_CHANNEL
        )
        assert all(
            brief.evidence_layout != EvidenceLayout.ONE_THREAD
            for brief in schedule
            if brief.reasoning_shape == ReasoningShape.CROSS_THREAD
        )


def test_first_balanced_json_keeps_exact_safe_failure_causes(synthesized) -> None:
    fenced = parse_synthesized_item(f"prefix {synthesized.model_dump_json()} suffix")
    assert fenced == synthesized
    with pytest.raises(ValueError, match="duplicate JSON key 'scenario'"):
        parse_synthesized_item('{"scenario": null, "scenario": null}')
    with pytest.raises(ValueError, match="non-finite JSON number 'NaN'"):
        parse_synthesized_item('{"value": NaN}')
    with pytest.raises(ValueError, match="line 1, column"):
        parse_synthesized_item('{"scenario": }')
    deep = '{"value":' + "[" * 100 + "0" + "]" * 100 + "}"
    with pytest.raises(ValueError, match="exceeds depth 64"):
        parse_synthesized_item(deep)
    with pytest.raises(ValueError, match="exceeds 65536 bytes"):
        parse_synthesized_item("x" * 65_537 + synthesized.model_dump_json())
    with pytest.raises((ValueError, ValidationError)):
        parse_synthesized_item("{} " + synthesized.model_dump_json())


def test_brief_aware_preflight_enforces_answer_layout_and_reasoning(synthesized) -> None:
    one_thread = SynthesisBrief(
        workflow=Workflow.INCIDENT,
        reasoning_shape=ReasoningShape.LOOKUP,
        evidence_layout=EvidenceLayout.ONE_THREAD,
        answer_kind=AnswerKind.FACT_SUMMARY,
    )
    preflight_synthesized_item(synthesized, brief=one_thread)

    wrong_answer = one_thread.model_copy(update={"answer_kind": AnswerKind.DATE})
    with pytest.raises(ValueError, match="answer kind"):
        preflight_synthesized_item(synthesized, brief=wrong_answer)

    multiple_channels = one_thread.model_copy(
        update={
            "reasoning_shape": ReasoningShape.CROSS_CHANNEL,
            "evidence_layout": EvidenceLayout.MULTIPLE_CHANNELS,
        }
    )
    with pytest.raises(ValueError, match="multiple_channels"):
        preflight_synthesized_item(synthesized, brief=multiple_channels)

    identity = one_thread.model_copy(update={"reasoning_shape": ReasoningShape.IDENTITY_JOIN})
    with pytest.raises(ValueError, match="slack_get_user"):
        preflight_synthesized_item(synthesized, brief=identity)

    with pytest.raises(ValueError, match="requires multiple_channels"):
        SynthesisBrief(
            workflow=Workflow.INCIDENT,
            reasoning_shape=ReasoningShape.CROSS_CHANNEL,
            evidence_layout=EvidenceLayout.ONE_THREAD,
            answer_kind=AnswerKind.FACT_SUMMARY,
        )


def test_brief_preflight_requires_genuine_scalar_evidence_composition(synthesized) -> None:
    brief = SynthesisBrief(
        workflow=Workflow.HANDOFF,
        reasoning_shape=ReasoningShape.LOOKUP,
        evidence_layout=EvidenceLayout.ONE_THREAD,
        answer_kind=AnswerKind.ENTITY,
    )
    payload = synthesized.model_dump(mode="json")
    payload["task"].update(
        {
            "question": "Who owns the migration handoff tracked by ticket T-42?",
            "answer": {
                "kind": "entity",
                "canonical_answer": "Nora Chen",
                "required_claims": ["Nora Chen owns the migration handoff."],
                "forbidden_claims": ["The handoff is complete."],
            },
            "required_evidence": [
                {
                    "evidence_id": "mapping",
                    "message_id": "mapping-root",
                    "conversation_id": "incidents",
                    "author_id": "alice",
                    "thread_root_id": None,
                    "required_terms": ["ticket T-42", "migration handoff"],
                    "fact_description": "The migration handoff is tracked by ticket T-42.",
                },
                {
                    "evidence_id": "owner",
                    "message_id": "owner-reply",
                    "conversation_id": "incidents",
                    "author_id": "bob",
                    "thread_root_id": "mapping-root",
                    "required_terms": ["ticket T-42", "Nora Chen"],
                    "fact_description": "Nora Chen owns ticket T-42.",
                },
            ],
            "min_distinct_evidence_messages": 2,
        }
    )
    compositional = type(synthesized).model_validate(payload)
    preflight_synthesized_item(compositional, brief=brief)

    single = compositional.model_dump(mode="json")
    single["task"]["required_evidence"] = [single["task"]["required_evidence"][1]]
    single["task"]["required_evidence"][0]["thread_root_id"] = None
    single["task"]["min_distinct_evidence_messages"] = 1
    with pytest.raises(ValueError, match="at least two distinct evidence messages"):
        preflight_synthesized_item(type(synthesized).model_validate(single), brief=brief)

    repeated = compositional.model_dump(mode="json")
    repeated["task"]["required_evidence"][0]["required_terms"].append("Nora Chen")
    with pytest.raises(ValueError, match="scalar canonical answer must occur in at most one"):
        preflight_synthesized_item(type(synthesized).model_validate(repeated), brief=brief)


def test_synthesizer_prompt_contains_only_brief_actions_and_proposal_contract() -> None:
    brief = build_synthesis_schedule(0, 1)[0]
    task = make_synthesizer_task(seed_data(), brief)
    prompt = task.data.prompt_text
    assert "SYNTHESIS BRIEF" in prompt
    assert brief.workflow.value in prompt
    assert "SLACK ACTIONS" in prompt
    assert "PROPOSAL SCHEMA" in prompt
    assert "thread_root_id references a different message_id" in prompt
    assert "genuine evidence composition" in prompt
    assert "at least two distinct evidence messages" in prompt
    assert "hidden_seeds" not in prompt
    assert "world_rubric" not in prompt
    assert "builder_guide" not in prompt
    assert "slack_world.schema" not in prompt


def test_bounded_response_retains_hash_byte_count_and_prefix() -> None:
    response = "é" * 40_000
    bounded = bounded_response(response)
    assert bounded["utf8_bytes"] == 80_000
    assert bounded["truncated"] is True
    assert len(bounded["text"].encode()) <= 65_536
    assert len(bounded["sha256"]) == 64


def test_repair_prompt_serializes_validation_error_context_safely() -> None:
    class Broken(BaseModel):
        value: int

        @field_validator("value")
        @classmethod
        def reject(cls, _value: int) -> int:
            raise ValueError("semantic failure")

    with pytest.raises(ValidationError) as caught:
        Broken(value=1)
    prompt = repair_prompt(caught.value)
    assert "semantic failure" in prompt
    assert len(prompt.encode("utf-8")) < 4_000
