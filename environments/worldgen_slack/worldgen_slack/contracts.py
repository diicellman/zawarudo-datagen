from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from enum import StrEnum
from typing import Annotated, Any, Literal, Self

import verifiers.v1 as vf
from pydantic import Field, field_validator, model_validator

from .slack.models import NonEmptyText, SlackWorld, StrictModel, TaskContract


_SECRET_KEY = re.compile(r"(?i)(?:api[_-]?key|authorization|cookie|password|secret|token)$")
_SECRET_VALUE = re.compile(
    r"(?i)(bearer\s+|(?:api[_-]?key|authorization|password|secret|token)\s*[=:]\s*)"
    r"[A-Za-z0-9_./+\-=]{8,}"
)
_SECRET_CONTAINERS = {"env", "headers"}


def redact_secrets(value: Any, *, parent_key: str | None = None) -> Any:
    if isinstance(value, str):
        if parent_key and _SECRET_KEY.search(parent_key):
            return "[REDACTED]"
        return _SECRET_VALUE.sub(lambda match: match.group(1) + "[REDACTED]", value)
    if isinstance(value, dict):
        output: dict[Any, Any] = {}
        redact_all = bool(parent_key and parent_key.casefold() in _SECRET_CONTAINERS)
        for key, item in value.items():
            key_text = str(key)
            output[key] = (
                "[REDACTED]"
                if redact_all or _SECRET_KEY.search(key_text)
                else redact_secrets(item, parent_key=key_text)
            )
        return output
    if isinstance(value, list):
        return [redact_secrets(item, parent_key=parent_key) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_secrets(item, parent_key=parent_key) for item in value)
    return value


INTERFACE_ID = "slack.readonly.v1"
QUALITY_CRITERIA = (
    "scenario_alignment",
    "world_coherence",
    "professional_realism",
    "discoverability",
    "shortcut_free",
)


class FailureOwner(StrEnum):
    NONE = "none"
    SYNTHESIZER = "synthesizer"
    BUILDER = "builder"
    SOLVER = "solver"
    INTERFACE = "interface"
    INFRASTRUCTURE = "infrastructure"


class ItemStatus(StrEnum):
    SOLVED = "solved"
    CHALLENGING = "challenging"
    HARD_GATE_REJECTED = "hard_gate_rejected"
    SEMANTIC_REJECTED = "semantic_rejected"
    QUALITY_THRESHOLD_REJECTED = "quality_threshold_rejected"
    CRITERION_FLOOR_REJECTED = "criterion_floor_rejected"
    SYNTHESIZER_FAILURE = "synthesizer_failure"
    DUPLICATE = "duplicate"
    BUILDER_FAILURE = "builder_failure"
    JUDGE_PROTOCOL_FAILURE = "judge_protocol_failure"
    INFRASTRUCTURE_ERROR = "infrastructure_error"


class GenerationSeedData(vf.TaskData):
    generation_seed: Annotated[int, Field(ge=0)]
    domain: Literal["slack"] = "slack"
    interface_id: Literal["slack.readonly.v1"] = INTERFACE_ID


class ScenarioSpec(StrictModel):
    organization: NonEmptyText
    workflow: NonEmptyText
    description: NonEmptyText

    @field_validator("organization", "workflow", "description")
    @classmethod
    def nonblank(cls, value: str, info: Any) -> str:
        if not value.strip():
            raise ValueError(f"{info.field_name} must not be blank")
        return value


class SynthesizedItem(StrictModel):
    scenario: ScenarioSpec
    task: TaskContract


class CheckResult(StrictModel):
    name: str
    ok: bool
    detail: str
    failure_owner: FailureOwner = FailureOwner.NONE


class ValidationReport(StrictModel):
    ok: bool
    failure_owner: FailureOwner
    checks: list[CheckResult]
    public_snapshot: SlackWorld | None = None
    hidden_snapshot_hashes: list[str] = Field(default_factory=list)
    gold_call_log: list[dict[str, Any]] = Field(default_factory=list)
    runtime: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if self.ok and self.failure_owner != FailureOwner.NONE:
            raise ValueError("a successful validation report must have failure_owner='none'")
        if self.ok and self.public_snapshot is None:
            raise ValueError("a successful validation report requires a public snapshot")
        if not self.ok and self.failure_owner == FailureOwner.NONE:
            raise ValueError("a failed validation report requires a failure owner")
        if self.ok and any(not check.ok for check in self.checks):
            raise ValueError("a successful validation report cannot contain failed checks")
        return self


class JudgeVerdict(StrictModel):
    solver_correct: bool
    task_unambiguous: bool
    world_supports_task: bool
    scenario_alignment: Annotated[int, Field(ge=1, le=5)]
    world_coherence: Annotated[int, Field(ge=1, le=5)]
    professional_realism: Annotated[int, Field(ge=1, le=5)]
    discoverability: Annotated[int, Field(ge=1, le=5)]
    shortcut_free: Annotated[int, Field(ge=1, le=5)]
    failure_owner: FailureOwner
    reason: NonEmptyText

    @field_validator("reason")
    @classmethod
    def reason_nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("reason must not be blank")
        return value

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if self.failure_owner == FailureOwner.NONE and not self.solver_correct:
            raise ValueError("failure_owner='none' requires solver_correct=true")
        if self.failure_owner == FailureOwner.SOLVER and (
            self.solver_correct or not self.world_supports_task
        ):
            raise ValueError("failure_owner='solver' requires an incorrect solver and a supporting world")
        if not self.task_unambiguous and self.failure_owner != FailureOwner.SYNTHESIZER:
            raise ValueError("an ambiguous task must be owned by the synthesizer")
        if not self.world_supports_task and self.failure_owner == FailureOwner.SOLVER:
            raise ValueError("an unsupported world cannot be owned by the solver")
        return self

    @property
    def scores(self) -> dict[str, int]:
        return {criterion: getattr(self, criterion) for criterion in QUALITY_CRITERIA}


SemanticStatus = Literal["solved", "challenging", "rejected"]


class QualityFilterConfig(StrictModel):
    min_accept_score: Annotated[float, Field(ge=0.0, le=1.0)] = 0.80
    accepted_statuses: set[Literal["solved", "challenging"]] = Field(default_factory=lambda: {"solved"})
    retain_rejected_artifacts: bool = False
    weights: dict[str, float]
    minimum_scores: dict[str, int]

    @field_validator("accepted_statuses", mode="before")
    @classmethod
    def parse_accepted_statuses(cls, value: Any) -> Any:
        return set(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_policy(self) -> Self:
        expected = set(QUALITY_CRITERIA)
        if set(self.weights) != expected:
            raise ValueError("weights must define every quality criterion exactly once")
        if set(self.minimum_scores) != expected:
            raise ValueError("minimum_scores must define every quality criterion exactly once")
        if not self.accepted_statuses:
            raise ValueError("accepted_statuses cannot be empty")
        if any(weight < 0 for weight in self.weights.values()):
            raise ValueError("quality weights must be non-negative")
        if sum(self.weights.values()) <= 0:
            raise ValueError("quality weights must have a positive total")
        if any(
            isinstance(score, bool) or not isinstance(score, int) for score in self.minimum_scores.values()
        ):
            raise ValueError("criterion minimums must be integers")
        if any(score < 1 or score > 5 for score in self.minimum_scores.values()):
            raise ValueError("criterion minimums must be integers from 1 to 5")
        return self


class QualityDecision(StrictModel):
    status: SemanticStatus
    write_to_dataset: bool
    quality_score: Annotated[float, Field(ge=0.0, le=1.0)]
    criterion_failures: list[str]
    rejection_reason: str | None = None


class SolverSummary(StrictModel):
    final_answer_present: bool
    final_answer: str
    tool_call_count: Annotated[int, Field(ge=0)]
    tool_names: list[str]
    visible_errors: list[str]
    trace_id: str
    runtime_id: str | None = None


class GenerationResult(StrictModel):
    generation_seed: int
    status: ItemStatus
    failure_owner: FailureOwner
    reason: str | None = None
    instance_id: str | None = None
    signature: str | None = None
    synthesized: SynthesizedItem | None = None
    validation: ValidationReport | None = None
    solver: SolverSummary | None = None
    verdict: JudgeVerdict | None = None
    decision: QualityDecision | None = None
    trace_ids: dict[str, list[str]] = Field(default_factory=dict)


def authentication_failure(errors: list[Any]) -> tuple[str, int] | None:
    auth_phrases = (
        "unauthorized",
        "not authenticated",
        "authentication failed",
        "authentication required",
        "invalid api key",
        "incorrect api key",
        "missing api key",
        "prime login",
    )
    status_pattern = re.compile(
        r"(?i)\b(?:error\s+code|http(?:\s+status)?(?:\s+code)?|"
        r"status(?:[_\s-]*code)?)\s*[\"']?\s*[:=]?\s*(401|403)\b"
    )
    for error in errors:
        rendered = f"{error.type}: {error.message}"
        status_code = getattr(error, "status_code", None)
        if status_code in {401, 403}:
            return rendered, status_code
        if status_code is not None:
            continue
        match = status_pattern.search(rendered)
        if match:
            return rendered, int(match.group(1))
        if any(phrase in rendered.casefold() for phrase in auth_phrases):
            return rendered, 401
    return None


def parse_synthesized_item(text: str) -> SynthesizedItem:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("synthesizer response is empty")
    stripped = text.strip()
    if not (stripped.startswith("{") and stripped.endswith("}")):
        raise ValueError("synthesizer response must be exactly one JSON object")
    try:
        value = json.loads(stripped, parse_constant=_reject_constant)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError("synthesizer response is not strict JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("synthesizer response must be a JSON object")
    return SynthesizedItem.model_validate(value)


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value!r} is not supported")


def normalize_signature_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


def synthesized_signature_text(item: SynthesizedItem) -> str:
    fields = [
        item.scenario.organization,
        item.scenario.workflow,
        item.task.question,
        *item.task.answer.required_claims,
    ]
    return " | ".join(normalize_signature_text(value) for value in fields)


def synthesized_signature(item: SynthesizedItem) -> str:
    return hashlib.sha256(synthesized_signature_text(item).encode()).hexdigest()


def token_jaccard(left: str, right: str) -> float:
    left_tokens = set(normalize_signature_text(left).split())
    right_tokens = set(normalize_signature_text(right).split())
    if not left_tokens and not right_tokens:
        return 1.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def preflight_synthesized_item(item: SynthesizedItem) -> None:
    task = item.task
    if task.question.count("?") > 1:
        raise ValueError("task must ask one answerable question")
    answer = normalize_signature_text(task.answer.canonical_answer)
    question = normalize_signature_text(task.question)
    if len(answer) >= 4 and answer in question:
        raise ValueError("question directly states the canonical answer")
    write_request = re.search(
        r"\b(send|post|write|update|edit|delete|archive|invite|react|create)\b",
        task.question.casefold(),
    )
    if write_request:
        raise ValueError("task requests a Slack write action")
    if not any(
        call.tool
        in {
            "slack_search_messages",
            "slack_get_conversation_history",
            "slack_get_thread",
        }
        for call in task.gold_calls
    ):
        raise ValueError("gold calls do not include a message-reading action")

    bindings: dict[str, tuple[str, str | None, str | None]] = {}
    for evidence in task.required_evidence:
        if evidence.thread_root_id == evidence.message_id:
            raise ValueError(f"evidence {evidence.evidence_id!r} is its own thread root")
        binding = (evidence.conversation_id, evidence.author_id, evidence.thread_root_id)
        previous = bindings.setdefault(evidence.message_id, binding)
        if previous != binding:
            raise ValueError(f"message {evidence.message_id!r} has contradictory evidence bindings")


def normalized_quality_score(verdict: JudgeVerdict, weights: dict[str, float]) -> float:
    weighted = sum(
        weights[criterion] * ((getattr(verdict, criterion) - 1) / 4) for criterion in QUALITY_CRITERIA
    )
    return weighted / sum(weights.values())


def classify_item(validation: ValidationReport, verdict: JudgeVerdict) -> SemanticStatus:
    if not validation.ok:
        return "rejected"
    if not verdict.task_unambiguous or not verdict.world_supports_task:
        return "rejected"
    if verdict.solver_correct and verdict.failure_owner == FailureOwner.NONE:
        return "solved"
    if not verdict.solver_correct and verdict.failure_owner == FailureOwner.SOLVER:
        return "challenging"
    return "rejected"


def decide_persistence(
    status: SemanticStatus,
    verdict: JudgeVerdict,
    config: QualityFilterConfig,
) -> QualityDecision:
    quality_score = normalized_quality_score(verdict, config.weights)
    criterion_failures = [
        criterion
        for criterion, minimum in config.minimum_scores.items()
        if getattr(verdict, criterion) < minimum
    ]
    ownership_allowed = (status == "solved" and verdict.failure_owner == FailureOwner.NONE) or (
        status == "challenging" and verdict.failure_owner == FailureOwner.SOLVER
    )
    write_to_dataset = (
        status in config.accepted_statuses
        and ownership_allowed
        and not criterion_failures
        and quality_score >= config.min_accept_score
    )
    reason = None
    if status not in config.accepted_statuses:
        reason = f"status_not_accepted:{status}"
    elif not ownership_allowed:
        reason = f"failure_owner_not_accepted:{verdict.failure_owner.value}"
    elif criterion_failures:
        reason = "criterion_floor:" + ",".join(criterion_failures)
    elif quality_score < config.min_accept_score:
        reason = "quality_below_threshold"
    return QualityDecision(
        status=status,
        write_to_dataset=write_to_dataset,
        quality_score=quality_score,
        criterion_failures=criterion_failures,
        rejection_reason=reason,
    )


__all__ = [
    "INTERFACE_ID",
    "QUALITY_CRITERIA",
    "CheckResult",
    "FailureOwner",
    "GenerationResult",
    "GenerationSeedData",
    "ItemStatus",
    "JudgeVerdict",
    "QualityDecision",
    "QualityFilterConfig",
    "ScenarioSpec",
    "SolverSummary",
    "SynthesizedItem",
    "TaskContract",
    "ValidationReport",
    "authentication_failure",
    "classify_item",
    "decide_persistence",
    "normalized_quality_score",
    "parse_synthesized_item",
    "preflight_synthesized_item",
    "redact_secrets",
    "synthesized_signature",
    "synthesized_signature_text",
    "token_jaccard",
]
