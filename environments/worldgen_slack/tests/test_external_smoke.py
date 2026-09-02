from __future__ import annotations

import os
from pathlib import Path

import pytest
import verifiers.v1 as vf

from worldgen_slack.agents.solver import SlackAnswerJudge, solver_verdict_scores
from worldgen_slack.dataset import validate_release_integrity
from worldgen_slack.generate import generate, load_generation_config
from worldgen_slack.slack.models import AnswerSpec


@pytest.mark.skipif(
    os.getenv("WORLDGEN_SLACK_EXTERNAL_SMOKE") != "1",
    reason="requires Prime Inference access",
)
@pytest.mark.asyncio
async def test_live_host_solver_judge_accepts_a_known_paraphrase() -> None:
    answer = AnswerSpec(
        kind="fact_summary",
        canonical_answer="A stale DNS cache caused the timeout; recycling edge workers resolved it.",
        required_claims=["stale DNS cache caused timeout", "edge workers recycled"],
        forbidden_claims=["database overload"],
    )
    trace = vf.Trace(
        task=vf.TraceTask(type="CredentialJudgeTest", data=vf.TaskData(prompt="judge")),
        agent=vf.AgentInfo(config=vf.AgentConfig(model="test"), name="solver"),
        is_completed=True,
        ok=True,
    )
    judged = await SlackAnswerJudge(
        vf.JudgeConfig(
            model="openai/gpt-5.6-luna",
            sampling=vf.Sampling(temperature=0.0, max_tokens=2_000),
        )
    ).evaluate(
        trace=trace,
        question="What caused the timeout and how was it fixed?",
        canonical_answer=answer.canonical_answer,
        required_claims=answer.required_claims,
        forbidden_claims=answer.forbidden_claims,
        response="The outage came from stale DNS entries, and the team restored service by restarting the edge worker fleet.",
    )
    assert judged.parsed is not None
    assert solver_verdict_scores(judged.parsed, answer)["semantic_correctness"] == 1.0
    assert trace.extra_usage


@pytest.mark.skipif(
    os.getenv("WORLDGEN_SLACK_EXTERNAL_SMOKE") != "1",
    reason="requires Prime Inference, Sandbox, and RLM access",
)
@pytest.mark.asyncio
async def test_one_live_prime_generation_episode(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[3]
    config = load_generation_config(
        root / "configs/generation.toml",
        {
            "run_id": "external-smoke",
            "target_accepted": 1,
            "max_attempts": 3,
            "concurrency": 1,
            "output": tmp_path,
        },
    )
    summary = await generate(config)
    assert summary["actual_accepted"] >= 1
    assert summary["qualification_passed"] is True
    validate_release_integrity(tmp_path, require_tasks=True)
