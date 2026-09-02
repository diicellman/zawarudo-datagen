from __future__ import annotations

import json
import re
import unicodedata
from datetime import datetime
from statistics import fmean
from typing import Literal, Self

import verifiers.v1 as vf
from pydantic import Field, field_validator, model_validator

from ..contracts import SolverSummary, redact_secrets
from ..slack.models import INTERFACE_ID, AnswerSpec, NonEmptyText, SlackWorld, StrictModel, TaskContract
from ..slack.tools import SlackState, SlackToolset, SlackToolsetConfig

CLAIM_SCORE = {"missing": 0.0, "contradicted": 0.0, "partial": 0.5, "supported": 1.0}


class RequiredClaimVerdict(StrictModel):
    claim_index: int = Field(ge=0)
    grade: Literal["missing", "contradicted", "partial", "supported"]
    reason: NonEmptyText

    @field_validator("reason")
    @classmethod
    def nonblank_reason(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("claim verdict reason must not be blank")
        return value


class SolverAnswerVerdict(StrictModel):
    required_claims: list[RequiredClaimVerdict]
    forbidden_claim_indexes: list[int] = Field(default_factory=list)
    material_contradiction: bool
    reason: NonEmptyText

    @field_validator("reason")
    @classmethod
    def nonblank_reason(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("solver verdict reason must not be blank")
        return value

    @model_validator(mode="after")
    def unique_indexes(self) -> Self:
        required = [item.claim_index for item in self.required_claims]
        if len(required) != len(set(required)):
            raise ValueError("required claim verdict indexes must be unique")
        if len(self.forbidden_claim_indexes) != len(set(self.forbidden_claim_indexes)):
            raise ValueError("forbidden claim indexes must be unique")
        if any(index < 0 for index in self.forbidden_claim_indexes):
            raise ValueError("forbidden claim indexes must be non-negative")
        return self

    def validate_against(self, answer: AnswerSpec) -> Self:
        required = {item.claim_index for item in self.required_claims}
        expected_required = set(range(len(answer.required_claims)))
        if required != expected_required:
            raise ValueError(
                "required claim verdict indexes must appear exactly once: "
                f"expected {sorted(expected_required)}, got {sorted(required)}"
            )
        unknown_forbidden = set(self.forbidden_claim_indexes) - set(range(len(answer.forbidden_claims)))
        if unknown_forbidden:
            raise ValueError(f"unknown forbidden claim indexes: {sorted(unknown_forbidden)}")
        return self


class SlackAnswerJudge(vf.Judge[SolverAnswerVerdict]):
    schema = SolverAnswerVerdict

    def build_messages(
        self,
        *,
        question: str,
        canonical_answer: str,
        required_claims: list[str],
        forbidden_claims: list[str],
        response: str,
    ) -> vf.Messages:
        payload = {
            "question": question,
            "canonical_answer": canonical_answer,
            "required_claims": [
                {"claim_index": index, "claim": claim} for index, claim in enumerate(required_claims)
            ],
            "forbidden_claims": [
                {"claim_index": index, "claim": claim} for index, claim in enumerate(forbidden_claims)
            ],
            "solver_response": response,
        }
        return [
            vf.SystemMessage(
                content=(
                    "Grade only the solver response against the supplied private answer oracle. "
                    "Treat semantically equivalent paraphrases as supported. Grade every required "
                    "claim exactly once using its zero-based claim_index. Use contradicted when the "
                    "response materially conflicts with a required claim. List every forbidden claim "
                    "asserted by the response. Set material_contradiction only for a material conflict "
                    "with the oracle. Do not infer missing entries."
                )
            ),
            vf.UserMessage(content=json.dumps(payload, ensure_ascii=False, sort_keys=True)),
        ]


def solver_verdict_scores(
    verdict: SolverAnswerVerdict,
    answer: AnswerSpec,
) -> dict[str, float]:
    verdict.validate_against(answer)
    coverage = fmean(CLAIM_SCORE[item.grade] for item in verdict.required_claims)
    contradiction_free = (
        not verdict.material_contradiction
        and not verdict.forbidden_claim_indexes
        and all(item.grade != "contradicted" for item in verdict.required_claims)
    )
    return {
        "semantic_correctness": coverage if contradiction_free else 0.0,
        "required_claim_coverage": coverage,
        "contradiction_free": float(contradiction_free),
        "forbidden_claim_count": float(len(verdict.forbidden_claim_indexes)),
    }


class SolverData(vf.TaskData):
    instance_id: str
    question: str
    interface_id: str = INTERFACE_ID


class SolverConfig(vf.TaskConfig):
    tools: SlackToolsetConfig = SlackToolsetConfig()
    oracle: AnswerSpec | None = Field(default=None, exclude=True, repr=False)
    answer_judge: vf.JudgeConfig = vf.JudgeConfig(
        model="openai/gpt-5.6-luna",
        sampling=vf.Sampling(temperature=0.0, max_tokens=2_000),
    )


class SolverTask(vf.Task[SolverData, SlackState, SolverConfig]):
    @classmethod
    def toolsets(cls, config: SolverConfig) -> list[vf.Toolset]:
        return [SlackToolset(config.tools)]

    @classmethod
    def from_snapshot(
        cls,
        *,
        instance_id: str,
        contract: TaskContract,
        world: SlackWorld,
        answer_judge: vf.JudgeConfig | None = None,
        idx: int | None = None,
    ) -> "SolverTask":
        data = SolverData(
            idx=idx,
            name=instance_id,
            prompt=contract.question,
            network_allow=[],
            network_block=["*"],
            instance_id=instance_id,
            question=contract.question,
        )
        config = SolverConfig(
            tools=SlackToolsetConfig.from_world(world, contract.actor_id),
            oracle=contract.answer.model_copy(deep=True),
            answer_judge=answer_judge or SolverConfig().answer_judge,
        )
        return cls(data, config)

    @vf.metric
    async def answer_present(self, trace: vf.Trace) -> float:
        return float(bool(trace.last_reply.strip()))

    @vf.metric
    async def completed_action_count(self, trace: vf.Trace) -> float:
        return float(len(require_slack_state(trace).completed_actions))

    @vf.metric
    async def exact_answer_diagnostic(self, trace: vf.Trace) -> float:
        oracle = self.config.oracle
        if oracle is None or oracle.kind == "fact_summary":
            return 0.0
        return float(score_exact_answer(trace.last_reply.strip(), oracle))

    @vf.reward(weight=1.0)
    async def semantic_correctness(self, trace: vf.Trace) -> float:
        oracle = self.config.oracle
        if oracle is None:
            raise ValueError("solver semantic reward requires a private answer oracle")
        response = trace.last_reply.strip()
        trace.info["solver_answer_judge_model"] = self.config.answer_judge.model
        if not response:
            scores = {
                "semantic_correctness": 0.0,
                "required_claim_coverage": 0.0,
                "contradiction_free": 1.0,
                "forbidden_claim_count": 0.0,
            }
            trace.info["solver_semantic_verdict"] = None
        else:
            judged = await SlackAnswerJudge(self.config.answer_judge).evaluate(
                trace=trace,
                question=self.data.question,
                canonical_answer=oracle.canonical_answer,
                required_claims=oracle.required_claims,
                forbidden_claims=oracle.forbidden_claims,
                response=response,
            )
            verdict = judged.parsed
            if verdict is None:
                raise ValueError("solver answer judge returned no parsed verdict")
            scores = solver_verdict_scores(verdict, oracle)
            trace.info["solver_semantic_verdict"] = verdict.model_dump(mode="json")
        trace.info["solver_semantic_scores"] = scores
        for name in (
            "required_claim_coverage",
            "contradiction_free",
            "forbidden_claim_count",
        ):
            trace.record_metric(name, scores[name])
            trace.record_reward(name, scores[name], 0.0)
        return scores["semantic_correctness"]


def require_slack_state(trace: vf.Trace) -> SlackState:
    if not isinstance(trace.state, SlackState):
        raise TypeError("Slack solver trace requires SlackState")
    return trace.state


def normalize_answer(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    value = re.sub(r"[^\w\s/-]", " ", value)
    return " ".join(value.split())


def _dates(value: str) -> set[str]:
    month = (
        r"(?:January|February|March|April|May|June|July|August|September|October|"
        r"November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"
    )
    candidates: list[str] = []
    for pattern in (
        r"\b\d{4}[-/]\d{2}[-/]\d{2}\b",
        rf"\b{month}\s+\d{{1,2}},?\s+\d{{4}}\b",
        rf"\b\d{{1,2}}\s+{month}\s+\d{{4}}\b",
    ):
        candidates.extend(match.group(0) for match in re.finditer(pattern, value, re.IGNORECASE))
    parsed: set[str] = set()
    for candidate in candidates:
        rendered = candidate.replace(",", "")
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%B %d %Y", "%b %d %Y", "%d %B %Y", "%d %b %Y"):
            try:
                parsed.add(datetime.strptime(rendered, fmt).date().isoformat())
                break
            except ValueError:
                continue
    return parsed


def _list_items(value: str) -> list[str]:
    return [
        item
        for piece in re.split(r"[,;\n]|\s+and\s+", value)
        if (item := normalize_answer(re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", piece)))
    ]


def score_exact_answer(value: str, spec: AnswerSpec) -> bool:
    if not value or spec.kind == "fact_summary":
        return False
    if spec.kind == "date":
        return len(_dates(value)) == 1 and _dates(value) == _dates(spec.canonical_answer)
    if spec.kind == "list":
        actual = _list_items(value)
        expected = _list_items(spec.canonical_answer)
        return actual == expected if spec.list_order_matters else sorted(actual) == sorted(expected)
    return normalize_answer(value) == normalize_answer(spec.canonical_answer)


def visible_errors(trace: vf.Trace) -> list[str]:
    return [str(redact_secrets(f"{error.type}: {error.message}"))[:1_000] for error in trace.errors[:20]]


def summarize_solver(trace: vf.Trace) -> SolverSummary:
    answer = trace.last_reply.strip()
    state = require_slack_state(trace)
    reward = trace.rewards.get("semantic_correctness")
    if reward is None:
        raise ValueError("solver trace is missing semantic_correctness reward")
    scores = trace.info.get("solver_semantic_scores")
    if not isinstance(scores, dict):
        raise ValueError("solver trace is missing semantic score details")
    runtime_id = trace.agent.runtime.id if trace.agent.runtime else None
    return SolverSummary(
        final_answer_present=bool(answer),
        final_answer=answer,
        completed_actions=[item.model_dump(mode="json") for item in state.completed_actions],
        semantic_score=reward.score,
        metrics={
            "required_claim_coverage": float(scores["required_claim_coverage"]),
            "contradiction_free": float(scores["contradiction_free"]),
            "forbidden_claim_count": float(scores["forbidden_claim_count"]),
            "answer_present": float(bool(answer)),
        },
        semantic_verdict=trace.info.get("solver_semantic_verdict"),
        visible_errors=visible_errors(trace),
        trace_id=trace.id,
        runtime_id=runtime_id,
    )


__all__ = [
    "CLAIM_SCORE",
    "RequiredClaimVerdict",
    "SlackAnswerJudge",
    "SolverAnswerVerdict",
    "SolverConfig",
    "SolverData",
    "SolverTask",
    "score_exact_answer",
    "solver_verdict_scores",
    "summarize_solver",
]
