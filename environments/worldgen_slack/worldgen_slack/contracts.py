from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from enum import StrEnum
from typing import Annotated, Any, Literal, Self

import verifiers.v1 as vf
from pydantic import ConfigDict, Field, field_validator, model_validator

from .slack.models import (
    INTERFACE_ID,
    QUALITY_CRITERIA,
    NonEmptyText,
    SlackWorld,
    StrictModel,
    SynthesizedItem,
)

_SECRET_KEY = re.compile(r"(?i)(?:api[_-]?key|authorization|cookie|password|secret|token)$")
_SECRET_VALUE = re.compile(
    r"(?i)(bearer\s+|(?:api[_-]?key|authorization|password|secret|token)\s*[=:]\s*)"
    r"[A-Za-z0-9_./+\-=]{8,}"
)
_SECRET_CONTAINERS = {"env", "headers"}
_SAFE_SLUG = re.compile(r"[^a-z0-9]+")


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


class FailureOwner(StrEnum):
    NONE = "none"
    SYNTHESIZER = "synthesizer"
    BUILDER = "builder"
    SOLVER = "solver"
    INTERFACE = "interface"
    INFRASTRUCTURE = "infrastructure"


class FailedStage(StrEnum):
    SYNTHESIS = "synthesis"
    BUILDER = "builder"
    VALIDATION = "validation"
    SOLVER = "solver"
    SOLVER_ANSWER_JUDGE = "solver_answer_judge"
    WORLD_JUDGE = "world_judge"
    PERSISTENCE = "persistence"
    INFRASTRUCTURE = "infrastructure"


class FailureKind(StrEnum):
    CONTRACT_REJECTION = "contract_rejection"
    DETERMINISTIC_REJECTION = "deterministic_rejection"
    QUALITY_REJECTION = "quality_rejection"
    PROTOCOL_ERROR = "protocol_error"
    PROVIDER_ERROR = "provider_error"
    SANDBOX_ERROR = "sandbox_error"
    RUNTIME_ERROR = "runtime_error"
    DUPLICATE = "duplicate"


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
    PROTOCOL_ERROR = "protocol_error"
    INFRASTRUCTURE_ERROR = "infrastructure_error"


class Workflow(StrEnum):
    INCIDENT = "incident"
    PROJECT_DECISION = "project_decision"
    SECURITY_REVIEW = "security_review"
    CUSTOMER_SUPPORT = "customer_support"
    PLANNING = "planning"
    COMPLIANCE = "compliance"
    HIRING = "hiring"
    HANDOFF = "handoff"


class ReasoningShape(StrEnum):
    LOOKUP = "lookup"
    TEMPORAL = "temporal"
    COMPARE = "compare"
    AGGREGATE = "aggregate"
    CROSS_THREAD = "cross_thread"
    CROSS_CHANNEL = "cross_channel"
    EXCEPTION = "exception"
    IDENTITY_JOIN = "identity_join"


class EvidenceLayout(StrEnum):
    ONE_THREAD = "one_thread"
    MULTIPLE_THREADS = "multiple_threads"
    MULTIPLE_CHANNELS = "multiple_channels"


class AnswerKind(StrEnum):
    ENTITY = "entity"
    DATE = "date"
    LIST = "list"
    FACT_SUMMARY = "fact_summary"


class SynthesisBrief(StrictModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    workflow: Workflow
    reasoning_shape: ReasoningShape
    evidence_layout: EvidenceLayout
    answer_kind: AnswerKind

    @model_validator(mode="after")
    def compatible(self) -> Self:
        if (
            self.reasoning_shape == ReasoningShape.CROSS_CHANNEL
            and self.evidence_layout != EvidenceLayout.MULTIPLE_CHANNELS
        ):
            raise ValueError("cross_channel reasoning requires multiple_channels evidence")
        if (
            self.reasoning_shape == ReasoningShape.CROSS_THREAD
            and self.evidence_layout == EvidenceLayout.ONE_THREAD
        ):
            raise ValueError("cross_thread reasoning requires evidence in multiple threads")
        return self


class GenerationSeedData(vf.TaskData):
    generation_seed: Annotated[int, Field(ge=0)]
    domain: Literal["slack"] = "slack"
    interface_id: Literal["slack.readonly.v1"] = INTERFACE_ID


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


QualityChoice = Literal["fails", "weak", "adequate", "strong", "exceptional"]


class WorldJudgeVerdict(StrictModel):
    task_unambiguous: bool
    world_supports_task: bool
    scenario_alignment: QualityChoice
    world_coherence: QualityChoice
    professional_realism: QualityChoice
    discoverability: QualityChoice
    shortcut_free: QualityChoice
    evidence_composition: QualityChoice
    reason: NonEmptyText

    @field_validator("reason")
    @classmethod
    def reason_nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("reason must not be blank")
        return value

    @property
    def choices(self) -> dict[str, QualityChoice]:
        return {criterion: getattr(self, criterion) for criterion in QUALITY_CRITERIA}


SemanticStatus = Literal["solved", "challenging", "rejected"]


class ReleaseAcceptanceConfig(StrictModel):
    min_solver_score: Annotated[float, Field(ge=0.0, le=1.0)] = 1.0
    min_world_score: Annotated[float, Field(ge=0.0, le=1.0)] = 0.80
    retain_rejected_artifacts: bool = True
    minimum_world_scores: dict[str, float] = Field(
        default_factory=lambda: {
            "scenario_alignment": 0.75,
            "world_coherence": 0.75,
            "professional_realism": 0.50,
            "discoverability": 0.75,
            "shortcut_free": 0.75,
            "evidence_composition": 0.75,
        }
    )

    @model_validator(mode="after")
    def validate_policy(self) -> Self:
        if set(self.minimum_world_scores) != set(QUALITY_CRITERIA):
            raise ValueError("minimum_world_scores must define every world criterion exactly once")
        if any(not 0.0 <= score <= 1.0 for score in self.minimum_world_scores.values()):
            raise ValueError("minimum world scores must be between zero and one")
        return self


class QualityDecision(StrictModel):
    status: SemanticStatus
    write_to_dataset: bool
    solver_score: Annotated[float, Field(ge=0.0, le=1.0)]
    builder_score: Annotated[float, Field(ge=0.0, le=1.0)]
    criterion_failures: list[str]
    rejection_reason: str | None = None

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if len(set(self.criterion_failures)) != len(self.criterion_failures) or not set(
            self.criterion_failures
        ) <= set(QUALITY_CRITERIA):
            raise ValueError("criterion_failures contains invalid or duplicate names")
        if self.write_to_dataset and (self.criterion_failures or self.rejection_reason):
            raise ValueError("an accepted decision cannot contain rejection data")
        if not self.write_to_dataset and not self.rejection_reason:
            raise ValueError("a rejected decision requires a reason")
        return self


class SolverSummary(StrictModel):
    final_answer_present: bool
    final_answer: str
    completed_actions: list[dict[str, Any]]
    semantic_score: Annotated[float, Field(ge=0.0, le=1.0)]
    metrics: dict[str, float]
    semantic_verdict: dict[str, Any] | None = None
    visible_errors: list[str]
    trace_id: str
    runtime_id: str | None = None

    @model_validator(mode="after")
    def complete_metrics(self) -> Self:
        expected = {
            "required_claim_coverage",
            "contradiction_free",
            "forbidden_claim_count",
            "answer_present",
        }
        if set(self.metrics) != expected:
            raise ValueError("solver metrics are incomplete")
        if not 0.0 <= self.metrics["required_claim_coverage"] <= 1.0:
            raise ValueError("required claim coverage must be between zero and one")
        if self.metrics["contradiction_free"] not in {0.0, 1.0}:
            raise ValueError("contradiction_free must be binary")
        if self.metrics["answer_present"] not in {0.0, 1.0}:
            raise ValueError("answer_present must be binary")
        if self.metrics["forbidden_claim_count"] < 0:
            raise ValueError("forbidden claim count must be non-negative")
        for action in self.completed_actions:
            if set(action) != {"action", "arguments"}:
                raise ValueError("completed Slack action has an invalid public shape")
        return self


class GenerationResult(StrictModel):
    generation_seed: int
    status: ItemStatus
    failure_owner: FailureOwner
    failed_stage: FailedStage | None = None
    failure_kind: FailureKind | None = None
    reason: str | None = None
    instance_id: str | None = None
    signature: str | None = None
    synthesized: SynthesizedItem | None = None
    validation: ValidationReport | None = None
    solver: SolverSummary | None = None
    world_verdict: WorldJudgeVerdict | None = None
    solver_score: Annotated[float, Field(ge=0.0, le=1.0)] | None = None
    solver_metrics: dict[str, float] | None = None
    builder_score: Annotated[float, Field(ge=0.0, le=1.0)] | None = None
    builder_raw_score: Annotated[float, Field(ge=0.0, le=1.0)] | None = None
    world_criteria: dict[str, float] | None = None
    world_hard_gates: dict[str, bool] | None = None
    decision: QualityDecision | None = None
    trace_ids: dict[str, list[str]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def consistent(self) -> Self:
        semantic_fields = (
            self.solver,
            self.world_verdict,
            self.solver_score,
            self.solver_metrics,
            self.builder_score,
            self.builder_raw_score,
            self.world_criteria,
            self.world_hard_gates,
            self.decision,
        )
        if self.synthesized is None:
            if any(value is not None for value in (self.instance_id, self.signature, self.validation)):
                raise ValueError("an itemless result cannot have item artifacts")
            if any(value is not None for value in semantic_fields):
                raise ValueError("an itemless result cannot have reward artifacts")
            if self.status not in {
                ItemStatus.SYNTHESIZER_FAILURE,
                ItemStatus.PROTOCOL_ERROR,
                ItemStatus.INFRASTRUCTURE_ERROR,
            }:
                raise ValueError("an itemless result has an invalid status")
            if self.failed_stage is None or self.failure_kind is None:
                raise ValueError("a failed itemless result requires failure stage and kind")
            return self

        expected_id, _ = item_identity(self.synthesized)
        if self.instance_id != expected_id:
            raise ValueError("result instance_id does not match its synthesized item")
        if self.signature is not None and re.fullmatch(r"[0-9a-f]{64}", self.signature) is None:
            raise ValueError("result signature must be a lowercase SHA-256 digest")
        if self.validation is None:
            raise ValueError("a synthesized result requires deterministic validation")

        if not self.validation.ok:
            if any(value is not None for value in semantic_fields):
                raise ValueError("a deterministic rejection cannot contain semantic results")
            if self.status not in {
                ItemStatus.HARD_GATE_REJECTED,
                ItemStatus.BUILDER_FAILURE,
                ItemStatus.DUPLICATE,
            }:
                raise ValueError("failed validation has an inconsistent result status")
            if self.status != ItemStatus.DUPLICATE and self.failure_owner != self.validation.failure_owner:
                raise ValueError("failed validation has inconsistent failure ownership")
            return self

        if any(
            value is None
            for value in (
                self.solver,
                self.world_verdict,
                self.solver_score,
                self.solver_metrics,
                self.builder_score,
                self.builder_raw_score,
                self.world_criteria,
                self.world_hard_gates,
            )
        ):
            raise ValueError("validated data requires complete solver and world reward results")
        assert self.solver is not None
        assert self.world_verdict is not None
        assert self.solver_score is not None
        assert self.solver_metrics is not None
        assert self.builder_score is not None
        assert self.builder_raw_score is not None
        assert self.world_criteria is not None
        assert self.world_hard_gates is not None
        if abs(self.solver_score - self.solver.semantic_score) > 1e-9:
            raise ValueError("solver score disagrees with solver summary")
        if self.solver_metrics != self.solver.metrics:
            raise ValueError("solver metrics disagree with solver summary")
        if set(self.world_criteria) != set(QUALITY_CRITERIA):
            raise ValueError("world criteria are incomplete")
        if set(self.world_hard_gates) != {"task_unambiguous", "world_supports_task"}:
            raise ValueError("world hard gates are incomplete")
        expected_status = (
            self.decision.status
            if self.decision is not None
            else classify_item(self.validation, self.world_verdict, self.solver_score)
        )
        if self.decision is not None and self.decision.status != expected_status:
            raise ValueError("quality decision disagrees with semantic status")
        if self.status != ItemStatus.DUPLICATE:
            expected_owner = (
                persistence_owner(self, self.decision)
                if self.decision is not None
                else classify_owner(self.world_verdict, self.solver_score)
            )
            if self.failure_owner != expected_owner:
                raise ValueError("result failure ownership is not mechanically derived")
        semantic_item_status = {
            "solved": ItemStatus.SOLVED,
            "challenging": ItemStatus.CHALLENGING,
            "rejected": ItemStatus.SEMANTIC_REJECTED,
        }[expected_status]
        expected_item_status = (
            semantic_item_status if self.decision is None else release_item_status(self.decision)
        )
        if self.status not in {expected_item_status, ItemStatus.DUPLICATE}:
            raise ValueError("result status disagrees with rewards and acceptance")
        return self


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value!r} is not supported")


def first_balanced_json_object(
    text: str,
    *,
    max_bytes: int = 65_536,
    max_depth: int = 64,
) -> str:
    try:
        encoded = text.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError("synthesizer response is not valid UTF-8 text") from exc
    if len(encoded) > max_bytes:
        raise ValueError(f"synthesizer response exceeds {max_bytes} bytes")
    start: int | None = None
    stack: list[str] = []
    in_string = False
    escaped = False
    pairs = {"}": "{", "]": "["}
    for index, char in enumerate(text):
        if start is None:
            if char == "{":
                start = index
                stack.append(char)
            continue
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append(char)
            if len(stack) > max_depth:
                raise ValueError(f"synthesizer JSON object exceeds depth {max_depth}")
        elif char in "}]":
            if not stack or stack[-1] != pairs[char]:
                raise ValueError(
                    f"synthesizer JSON object has mismatched delimiter {char!r} at character {index}"
                )
            stack.pop()
            if not stack:
                return text[start : index + 1]
    raise ValueError("synthesizer response contains no balanced JSON object")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key {key!r}")
        output[key] = value
    return output


def parse_synthesized_item(text: str) -> SynthesizedItem:
    candidate = first_balanced_json_object(text)
    try:
        value = json.loads(
            candidate,
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"synthesizer response strict JSON error at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc
    except (ValueError, RecursionError) as exc:
        raise ValueError(f"synthesizer response strict JSON error: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("synthesizer response must contain a JSON object")
    return SynthesizedItem.model_validate(value)


def normalize_signature_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


def item_identity(item: SynthesizedItem) -> tuple[str, str]:
    contract_bytes = json.dumps(
        item.task.model_dump(mode="json"),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    contract_hash = hashlib.sha256(contract_bytes).hexdigest()
    source = item.task.task_slug or f"{item.scenario.organization}-{item.scenario.workflow}"
    slug = _SAFE_SLUG.sub("-", source.casefold()).strip("-")[:80].rstrip("-") or "slack-item"
    return f"{slug}--{contract_hash[:12]}", contract_hash


def preflight_synthesized_item(
    item: SynthesizedItem,
    *,
    brief: SynthesisBrief | None = None,
) -> None:
    task = item.task
    if task.question.count("?") > 1:
        raise ValueError("task must ask one answerable question")
    answer = normalize_signature_text(task.answer.canonical_answer)
    question = normalize_signature_text(task.question)
    if len(answer) >= 4 and answer in question:
        raise ValueError("question directly states the canonical answer")
    if re.search(
        r"\b(send|post|write|update|edit|delete|archive|invite|react|create)\b",
        task.question.casefold(),
    ):
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

    if brief is not None:
        if task.answer.kind != brief.answer_kind.value:
            raise ValueError(
                f"answer kind {task.answer.kind!r} does not match brief {brief.answer_kind.value!r}"
            )
        evidence_message_ids = {evidence.message_id for evidence in task.required_evidence}
        if len(evidence_message_ids) < 2:
            raise ValueError("qualification tasks require at least two distinct evidence messages")
        if task.answer.kind in {AnswerKind.ENTITY.value, AnswerKind.DATE.value}:
            answer_mentions = 0
            for evidence in task.required_evidence:
                texts = [evidence.fact_description, *evidence.required_terms]
                if any(answer and answer in normalize_signature_text(text) for text in texts):
                    answer_mentions += 1
            if answer_mentions > 1:
                raise ValueError(
                    "scalar canonical answer must occur in at most one evidence item; "
                    "other evidence must supply a distinct join or constraint"
                )
        thread_keys = {
            (evidence.conversation_id, evidence.thread_root_id or evidence.message_id)
            for evidence in task.required_evidence
        }
        conversations = {evidence.conversation_id for evidence in task.required_evidence}
        if brief.evidence_layout == EvidenceLayout.ONE_THREAD and len(thread_keys) != 1:
            raise ValueError("one_thread brief requires exactly one evidence thread")
        if brief.evidence_layout == EvidenceLayout.MULTIPLE_THREADS and len(thread_keys) < 2:
            raise ValueError("multiple_threads brief requires at least two evidence threads")
        if brief.evidence_layout == EvidenceLayout.MULTIPLE_CHANNELS and len(conversations) < 2:
            raise ValueError("multiple_channels brief requires at least two evidence conversations")
        if brief.reasoning_shape == ReasoningShape.CROSS_THREAD and len(thread_keys) < 2:
            raise ValueError("cross_thread reasoning requires at least two evidence threads")
        if brief.reasoning_shape == ReasoningShape.CROSS_CHANNEL and len(conversations) < 2:
            raise ValueError("cross_channel reasoning requires at least two evidence conversations")
        if brief.reasoning_shape == ReasoningShape.IDENTITY_JOIN and not any(
            call.tool == "slack_get_user" for call in task.gold_calls
        ):
            raise ValueError("identity_join reasoning requires a slack_get_user gold call")


def classify_item(
    validation: ValidationReport,
    verdict: WorldJudgeVerdict,
    solver_score: float,
    min_solver_score: float = 1.0,
) -> SemanticStatus:
    if not validation.ok or not verdict.task_unambiguous or not verdict.world_supports_task:
        return "rejected"
    return "solved" if solver_score >= min_solver_score else "challenging"


def classify_owner(verdict: WorldJudgeVerdict, solver_score: float) -> FailureOwner:
    if not verdict.task_unambiguous:
        return FailureOwner.SYNTHESIZER
    if not verdict.world_supports_task:
        return FailureOwner.BUILDER
    if solver_score < 1.0:
        return FailureOwner.SOLVER
    return FailureOwner.NONE


def decide_persistence(result: GenerationResult, config: ReleaseAcceptanceConfig) -> QualityDecision:
    if (
        result.validation is None
        or result.world_verdict is None
        or result.solver_score is None
        or result.builder_score is None
        or result.world_criteria is None
    ):
        raise ValueError("acceptance requires complete validated reward results")
    status = classify_item(
        result.validation,
        result.world_verdict,
        result.solver_score,
        config.min_solver_score,
    )
    criterion_failures = [
        criterion
        for criterion, minimum in config.minimum_world_scores.items()
        if result.world_criteria[criterion] < minimum
    ]
    write_to_dataset = (
        status == "solved"
        and result.validation.ok
        and result.world_verdict.task_unambiguous
        and result.world_verdict.world_supports_task
        and result.solver_score >= config.min_solver_score
        and result.builder_score >= config.min_world_score
        and not criterion_failures
    )
    reason = None
    if status != "solved":
        reason = f"status_not_accepted:{status}"
    elif result.solver_score < config.min_solver_score:
        reason = "solver_below_threshold"
    elif not result.world_verdict.task_unambiguous:
        reason = "task_ambiguous"
    elif not result.world_verdict.world_supports_task:
        reason = "world_does_not_support_task"
    elif criterion_failures:
        reason = "criterion_floor:" + ",".join(criterion_failures)
    elif result.builder_score < config.min_world_score:
        reason = "world_quality_below_threshold"
    return QualityDecision(
        status=status,
        write_to_dataset=write_to_dataset,
        solver_score=result.solver_score,
        builder_score=result.builder_score,
        criterion_failures=criterion_failures,
        rejection_reason=reason,
    )


def persistence_owner(result: GenerationResult, decision: QualityDecision) -> FailureOwner:
    verdict = result.world_verdict
    if verdict is None:
        raise ValueError("persistence ownership requires a world verdict")
    if decision.write_to_dataset:
        return FailureOwner.NONE
    if not verdict.task_unambiguous:
        return FailureOwner.SYNTHESIZER
    if not verdict.world_supports_task:
        return FailureOwner.BUILDER
    if decision.status == "challenging":
        return FailureOwner.SOLVER
    return FailureOwner.BUILDER


def release_item_status(decision: QualityDecision) -> ItemStatus:
    if decision.write_to_dataset:
        return ItemStatus.SOLVED if decision.status == "solved" else ItemStatus.CHALLENGING
    if decision.status == "challenging" or decision.rejection_reason == "solver_below_threshold":
        return ItemStatus.CHALLENGING
    if decision.status == "rejected" or (decision.rejection_reason or "").startswith("status_not_accepted"):
        return ItemStatus.SEMANTIC_REJECTED
    if decision.criterion_failures:
        return ItemStatus.CRITERION_FLOOR_REJECTED
    return ItemStatus.QUALITY_THRESHOLD_REJECTED


def _trace_map(episode: vf.Episode) -> dict[str, vf.Trace]:
    grouped = episode.by_agent
    if any(len(traces) != 1 for traces in grouped.values()):
        raise ValueError("a finalized generation episode requires one trace per agent")
    names = set(grouped)
    allowed = [
        {"synthesizer"},
        {"synthesizer", "builder"},
        {"synthesizer", "builder", "solver", "judge"},
    ]
    if names not in allowed:
        raise ValueError(f"invalid world-generation trace roles: {sorted(names)}")
    if any(not trace.ok for trace in episode.traces):
        raise ValueError("cannot finalize a generation result from a failed trace")
    return {name: traces[0] for name, traces in grouped.items()}


def finalize_generation(task: vf.Task, episode: vf.Episode) -> GenerationResult:
    if not isinstance(task.data, GenerationSeedData):
        raise TypeError("Slack world generation requires GenerationSeedData")
    traces = _trace_map(episode)
    seed = task.data.generation_seed
    trace_ids = {name: [trace.id] for name, trace in traces.items()}
    synthesized_payload = traces["synthesizer"].info.get("synthesized_item")
    if synthesized_payload is None:
        reason = traces["synthesizer"].info.get("synthesizer_error")
        if not isinstance(reason, str) or not reason:
            raise ValueError("synthesizer trace has neither an item nor a rejection")
        return GenerationResult(
            generation_seed=seed,
            status=ItemStatus.SYNTHESIZER_FAILURE,
            failure_owner=FailureOwner.SYNTHESIZER,
            failed_stage=FailedStage.SYNTHESIS,
            failure_kind=FailureKind.CONTRACT_REJECTION,
            reason=reason,
            trace_ids=trace_ids,
        )

    item = SynthesizedItem.model_validate(synthesized_payload)
    instance_id, _ = item_identity(item)
    if "builder" not in traces:
        raise ValueError("a valid synthesized item must be followed by a builder trace")

    validation_payload = traces["builder"].info.get("worldgen_validation")
    if validation_payload is None:
        raise ValueError("builder trace is missing deterministic validation")
    validation = ValidationReport.model_validate_json(json.dumps(validation_payload))
    if validation.failure_owner == FailureOwner.INFRASTRUCTURE:
        raise ValueError("infrastructure failure cannot be finalized as generated data")
    if not validation.ok:
        reason = "; ".join(check.detail for check in validation.checks if not check.ok)[:4_000]
        status = (
            ItemStatus.BUILDER_FAILURE
            if any(check.name == "builder_artifact" for check in validation.checks)
            else ItemStatus.HARD_GATE_REJECTED
        )
        return GenerationResult(
            generation_seed=seed,
            status=status,
            failure_owner=validation.failure_owner,
            failed_stage=FailedStage.VALIDATION,
            failure_kind=FailureKind.DETERMINISTIC_REJECTION,
            reason=reason,
            instance_id=instance_id,
            synthesized=item,
            validation=validation,
            trace_ids=trace_ids,
        )

    if set(traces) != {"synthesizer", "builder", "solver", "judge"}:
        raise ValueError("validated candidates require solver and world-judge traces")
    solver = SolverSummary.model_validate(traces["solver"].info.get("solver_summary"))
    verdict = WorldJudgeVerdict.model_validate_json(json.dumps(traces["judge"].info.get("world_verdict")))
    world_scores = traces["builder"].info.get("world_reward")
    if not isinstance(world_scores, dict):
        raise ValueError("builder trace is missing world reward details")
    solver_score = solver.semantic_score
    semantic_status = classify_item(validation, verdict, solver_score)
    owner = classify_owner(verdict, solver_score)
    status = {
        "solved": ItemStatus.SOLVED,
        "challenging": ItemStatus.CHALLENGING,
        "rejected": ItemStatus.SEMANTIC_REJECTED,
    }[semantic_status]
    failed_stage = None
    failure_kind = None
    if semantic_status == "challenging":
        failed_stage = FailedStage.SOLVER
        failure_kind = FailureKind.QUALITY_REJECTION
    elif semantic_status == "rejected":
        failed_stage = FailedStage.WORLD_JUDGE
        failure_kind = FailureKind.QUALITY_REJECTION
    return GenerationResult(
        generation_seed=seed,
        status=status,
        failure_owner=owner,
        failed_stage=failed_stage,
        failure_kind=failure_kind,
        reason=verdict.reason,
        instance_id=instance_id,
        synthesized=item,
        validation=validation,
        solver=solver,
        world_verdict=verdict,
        solver_score=solver_score,
        solver_metrics=solver.metrics,
        builder_score=float(world_scores["world_quality"]),
        builder_raw_score=float(world_scores["world_quality_raw"]),
        world_criteria={name: float(world_scores["criteria"][name]) for name in QUALITY_CRITERIA},
        world_hard_gates={
            "task_unambiguous": bool(world_scores["hard_gates"]["task_unambiguous"]),
            "world_supports_task": bool(world_scores["hard_gates"]["world_supports_task"]),
        },
        trace_ids=trace_ids,
    )


def generation_result_from_episode(episode: vf.Episode) -> GenerationResult:
    matches = [trace for trace in episode.traces if "generation_result" in trace.info]
    if len(matches) != 1:
        raise ValueError("episode must contain exactly one generation result")
    expected_role = "builder" if episode.by_agent.get("builder") else "synthesizer"
    if matches[0].agent is None or matches[0].agent.name != expected_role:
        raise ValueError(f"generation result must be attached to the {expected_role} trace")
    result = GenerationResult.model_validate_json(json.dumps(matches[0].info["generation_result"]))
    actual_ids: dict[str, list[str]] = {}
    for trace in episode.traces:
        actual_ids.setdefault(trace.agent.name, []).append(trace.id)
    if result.trace_ids != actual_ids:
        raise ValueError("generation result trace IDs do not match its episode")
    return result


__all__ = [
    "INTERFACE_ID",
    "QUALITY_CRITERIA",
    "AnswerKind",
    "CheckResult",
    "EvidenceLayout",
    "FailedStage",
    "FailureKind",
    "FailureOwner",
    "GenerationResult",
    "GenerationSeedData",
    "ItemStatus",
    "QualityDecision",
    "ReleaseAcceptanceConfig",
    "ReasoningShape",
    "SolverSummary",
    "SynthesisBrief",
    "ValidationReport",
    "Workflow",
    "WorldJudgeVerdict",
    "classify_item",
    "classify_owner",
    "decide_persistence",
    "finalize_generation",
    "first_balanced_json_object",
    "generation_result_from_episode",
    "item_identity",
    "parse_synthesized_item",
    "persistence_owner",
    "preflight_synthesized_item",
    "redact_secrets",
    "release_item_status",
]
