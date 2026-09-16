"""Private generation contracts and checks for a shared Slack workspace."""

import re
import unicodedata
from collections import Counter
from datetime import UTC, datetime
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from worldgen_slack.slack.models import (
    QUALITY_CRITERIA,
    AnswerSpec,
    NonEmptyText,
    SafeId,
    SlackWorld,
    StrictModel,
    User,
)

from worldgen_slack.slack.api import SlackAPI, ReadCall, canonical, digest


def _compact_identifier(value: str) -> str:
    return "".join(char for char in unicodedata.normalize("NFKC", value).casefold() if char.isalnum())


def _meaningful_tokens(value: str) -> set[str]:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return {token for token in re.findall(r"[^\W_]+", normalized) if len(token) >= 2}


def _identifier_reveals(identifier: str, value: str, *, match_tokens: bool = True) -> bool:
    segments = re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", identifier).casefold())
    compact_value = _compact_identifier(value)
    if len(compact_value) < 2:
        return False
    if any(
        "".join(segments[start:end]) == compact_value
        for start in range(len(segments))
        for end in range(start + 1, len(segments) + 1)
    ):
        return True
    compact_identifier = _compact_identifier(identifier)
    candidates = [compact_identifier]
    for prefix in ("conversation", "channel", "message", "user", "conv", "chan", "usr", "msg"):
        if compact_identifier.startswith(prefix):
            candidates.append(compact_identifier.removeprefix(prefix))
    if len(compact_value) >= 4 and compact_identifier.startswith("u"):
        candidates.append(compact_identifier.removeprefix("u"))
    for candidate in candidates:
        if candidate == compact_value:
            return True
        suffix = candidate.removeprefix(compact_value)
        if candidate.startswith(compact_value) and suffix.isdigit() and len(suffix) <= 2:
            return True
    if not match_tokens:
        return False
    for token in _meaningful_tokens(value):
        for segment in segments:
            if segment == token:
                return True
            if (
                len(token) >= 3
                and segment.startswith(token)
                and len(segment) - len(token) <= 2
                and segment[len(token) :].isdigit()
            ):
                return True
    return False


def normalized(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.casefold()))


class Fact(StrictModel):
    id: SafeId
    subject: NonEmptyText
    predicate: NonEmptyText
    value: NonEmptyText
    valid_from: str
    valid_until: str | None = None
    description: NonEmptyText

    @field_validator("valid_from", "valid_until")
    @classmethod
    def timestamp(cls, value):
        if value is not None:
            parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
            if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value:
                raise ValueError("fact dates must be canonical UTC timestamps")
        return value

    @model_validator(mode="after")
    def interval(self) -> Self:
        if self.valid_until is not None and self.valid_until <= self.valid_from:
            raise ValueError("fact intervals are half-open and must have positive duration")
        return self


class PlannedTask(StrictModel):
    id: SafeId
    group_id: SafeId
    question: NonEmptyText
    actor_id: SafeId
    answer: AnswerSpec
    fact_ids: list[SafeId] = Field(min_length=1)
    reasoning: NonEmptyText


class WorkGroup(StrictModel):
    id: SafeId
    description: NonEmptyText


class Catalog(StrictModel):
    workspace_id: SafeId
    sector: NonEmptyText
    company: NonEmptyText
    overview: NonEmptyText
    people: list[User] = Field(min_length=1)
    groups: list[WorkGroup] = Field(min_length=1)
    facts: list[Fact] = Field(min_length=1)
    tasks: list[PlannedTask] = Field(min_length=1)

    @model_validator(mode="after")
    def references(self) -> Self:
        for label, values in (("task", self.tasks), ("fact", self.facts), ("group", self.groups)):
            if len({v.id for v in values}) != len(values):
                raise ValueError(f"duplicate {label} ID")
        facts = {v.id for v in self.facts}
        groups = {v.id for v in self.groups}
        if len({p.id for p in self.people}) != len(self.people):
            raise ValueError("duplicate person ID")
        questions: set[str] = set()
        directory_leaks = []
        for task in self.tasks:
            if task.group_id not in groups or set(task.fact_ids) - facts:
                raise ValueError(f"unknown group/fact in {task.id}")
            if task.actor_id not in {p.id for p in self.people}:
                raise ValueError(f"unknown task reader: {task.actor_id}")
            key = normalized(task.question)
            if key in questions:
                raise ValueError(f"duplicate or contradictory QA question: {task.id}")
            questions.add(key)
            if task.answer.kind in {"entity", "date"}:
                leaking = [
                    p.id
                    for p in self.people
                    if reveals_answer(p.id, task.answer.canonical_answer, task.question)
                ]
                if leaking:
                    directory_leaks.append(f"{task.id}: {leaking}")
        if directory_leaks:
            raise ValueError(
                "catalog person IDs encode answer content; rename these IDs and their actor references: "
                + "; ".join(directory_leaks)
            )
        if groups != {t.group_id for t in self.tasks}:
            raise ValueError("every group must have tasks")
        # Catalogs contain hundreds of facts; replace this quadratic scan with an interval index at larger scales.
        for index, left in enumerate(self.facts):
            for right in self.facts[index + 1 :]:
                if (normalized(left.subject), normalized(left.predicate)) != (
                    normalized(right.subject),
                    normalized(right.predicate),
                ):
                    continue
                overlap = max(left.valid_from, right.valid_from) < min(
                    left.valid_until or "9999", right.valid_until or "9999"
                )
                if overlap and normalized(left.value) != normalized(right.value):
                    raise ValueError(f"contradictory overlapping facts: {left.id}, {right.id}")
        return self


class ClaimEvidence(StrictModel):
    claim_index: int = Field(ge=0)
    message_ids: list[SafeId] = Field(default_factory=list)
    user_ids: list[SafeId] = Field(default_factory=list)

    @model_validator(mode="after")
    def nonempty(self) -> Self:
        if not self.message_ids and not self.user_ids:
            raise ValueError("a claim needs message or directory evidence")
        return self


class Binding(StrictModel):
    task_id: SafeId
    claims: list[ClaimEvidence] = Field(min_length=1)
    gold_calls: list[ReadCall] = Field(min_length=1)


class Candidate(StrictModel):
    snapshot: SlackWorld
    bindings: list[Binding]

    @model_validator(mode="after")
    def unique(self) -> Self:
        if len({b.task_id for b in self.bindings}) != len(self.bindings):
            raise ValueError("duplicate task binding")
        return self


class Issue(StrictModel):
    owner: Literal["synthesizer", "builder"]
    artifact: Literal["catalog", "workspace", "bindings"]
    task_ids: list[SafeId] = Field(default_factory=list)
    fact_ids: list[SafeId] = Field(default_factory=list)
    message_ids: list[SafeId] = Field(default_factory=list)
    defect: NonEmptyText
    requested_change: NonEmptyText

    @model_validator(mode="after")
    def ownership(self) -> Self:
        expected = "synthesizer" if self.artifact == "catalog" else "builder"
        if self.owner != expected:
            raise ValueError(f"{self.artifact} defects belong to {expected}")
        return self


class TaskReview(StrictModel):
    task_id: SafeId
    valid: bool
    answer_complete: bool
    supported_claims: list[int]
    reason: NonEmptyText


class Verdict(StrictModel):
    reviewed_hash: str
    approved: bool
    tasks: list[TaskReview]
    issues: list[Issue]
    criteria: dict[str, float] = Field(default_factory=dict)
    summary: NonEmptyText

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if len({t.task_id for t in self.tasks}) != len(self.tasks):
            raise ValueError("duplicate task review")
        if any(not 0 <= value <= 1 for value in self.criteria.values()):
            raise ValueError("criteria must be finite scores from zero to one")
        if self.approved and (self.issues or any(not t.valid or not t.answer_complete for t in self.tasks)):
            raise ValueError("approval contradicts defects")
        if not self.approved and not self.issues:
            raise ValueError("rejection must include actionable issues")
        return self


def quality(verdict: Verdict) -> float:
    weights = dict(zip(QUALITY_CRITERIA, (1.0, 1.25, 1.25, 1.0, 1.0, 1.5), strict=True))
    if set(verdict.criteria) != set(weights):
        raise ValueError("world review must score all six quality criteria")
    return sum(weights[k] * verdict.criteria[k] for k in weights) / sum(weights.values())


def reveals_answer(identifier: str, answer: str, question: str) -> bool:
    private_tokens = {
        token for token in _meaningful_tokens(answer) - _meaningful_tokens(question) if not token.isdigit()
    }
    return _identifier_reveals(identifier, answer, match_tokens=False) or any(
        _identifier_reveals(identifier, token, match_tokens=False) for token in private_tokens
    )


def validate_verdict(verdict: Verdict, payload: dict) -> None:
    if verdict.reviewed_hash != digest(payload):
        raise ValueError("stale review: candidate hash differs")
    tasks = {t["id"]: t for t in payload["tasks"]}
    if {r.task_id for r in verdict.tasks} != set(tasks):
        raise ValueError("judge must review every requested task exactly once")
    for review in verdict.tasks:
        expected = set(range(len(tasks[review.task_id]["answer"]["required_claims"])))
        if (
            len(review.supported_claims) != len(set(review.supported_claims))
            or set(review.supported_claims) - expected
        ):
            raise ValueError("invalid supported claim indices")
        if verdict.approved and set(review.supported_claims) != expected:
            raise ValueError("approval requires support for every canonical answer claim")
    if payload["phase"] == "world":
        score = quality(verdict)
        if verdict.approved and (score < 0.8 or min(verdict.criteria.values()) < 0.75):
            raise ValueError("approval contradicts world quality floors")


def validate_candidate(catalog: Catalog, candidate: Candidate, task_ids: list[str]) -> dict:
    """Structural and replay checks. Entailment and naturalness remain agentic judgments."""
    errors = []
    world = candidate.snapshot
    counts = {key: len(getattr(world, key)) for key in ("users", "conversations", "messages", "reactions")}
    limits = {"users": 200, "conversations": 100, "messages": 10_000, "reactions": 20_000}
    if (
        any(counts[key] > limits[key] for key in limits)
        or len(canonical(world.model_dump(mode="json"))) > 20_000_000
    ):
        errors.append("workspace exceeds bounded v2 size limits")
    tasks = {t.id: t for t in catalog.tasks}
    bindings = {b.task_id: b for b in candidate.bindings}
    if set(bindings) - set(tasks):
        errors.append("binding references unknown task")
    messages = {m.id: m for m in world.messages}
    users = {u.id: u for u in world.users}
    for person in catalog.people:
        if users.get(person.id) != person:
            errors.append(f"shared directory differs from approved person {person.id}")
    routes = {}
    public_text = "\n".join(
        [
            *(m.text for m in world.messages),
            *(u.name for u in world.users),
            *(c.topic + " " + c.purpose + " " + (c.name or "") for c in world.conversations),
        ]
    )
    for task in catalog.tasks:
        if len(task.question) >= 12 and normalized(task.question) in normalized(public_text):
            errors.append(f"{task.id}: literal benchmark question in workspace")
        if task.answer.kind in {"entity", "date"}:
            identifiers = [
                *(u.id for u in world.users),
                *(c.id for c in world.conversations),
                *(m.id for m in world.messages),
            ]
            leaking = [
                identifier
                for identifier in identifiers
                if reveals_answer(identifier, task.answer.canonical_answer, task.question)
            ]
            if leaking:
                errors.append(f"{task.id}: identifiers encode private answer content: {leaking}")
    if any(
        '"' + field + '"' in public_text
        for field in ("canonical_answer", "required_claims", "gold_calls", "fact_ids")
    ):
        errors.append("serialized private contract fields in workspace")
    for task_id in task_ids:
        task, binding = tasks.get(task_id), bindings.get(task_id)
        if task is None or binding is None:
            errors.append(f"{task_id}: missing task/binding")
            continue
        try:
            api = SlackAPI(world, task.actor_id)
            indices = [c.claim_index for c in binding.claims]
            if sorted(indices) != list(range(len(task.answer.required_claims))):
                raise ValueError("bind every claim exactly once")
            required_messages = {m for claim in binding.claims for m in claim.message_ids}
            required_users = {u for claim in binding.claims for u in claim.user_ids}
            for message_id in required_messages:
                message = messages.get(message_id)
                if (
                    message is None
                    or message.deleted
                    or not api.is_conversation_visible(message.conversation_id)
                ):
                    raise ValueError(f"missing or invisible evidence {message_id}")
            for user_id in required_users:
                api.get_user(user_id)
            outputs = []
            observed_text = task.question
            for call in binding.gold_calls:
                for name in ("conversation_id", "root_message_id", "user_id", "author_id", "cursor"):
                    value = call.arguments.get(name)
                    if value is not None and not re.search(
                        r"(?<![\w.-])" + re.escape(str(value)) + r"(?![\w.-])", observed_text
                    ):
                        raise ValueError(
                            f"gold route uses undiscovered {name}={value!r}; begin with global search "
                            "or list_conversations, then use IDs/cursors from prior observations"
                        )
                output = api.execute(call)
                outputs.append(output)
                observed_text += "\n" + canonical(output).decode()
            observed_messages = {
                item["message_id"]
                for output in outputs
                for item in output.get("items", [])
                if "message_id" in item
            }
            observed_users = {output["user_id"] for output in outputs if "user_id" in output}
            if required_messages - observed_messages or required_users - observed_users:
                raise ValueError(
                    "gold route misses bound evidence: "
                    f"message_ids={sorted(required_messages - observed_messages)}, "
                    f"user_ids={sorted(required_users - observed_users)}"
                )
            routes[task_id] = outputs
        except (ValueError, TypeError, LookupError) as exc:
            errors.append(f"{task_id}: {exc}")
    replies = [m for m in world.messages if m.thread_root_id and not m.deleted]
    delays = Counter(
        int(
            (
                datetime.fromisoformat(m.timestamp)
                - datetime.fromisoformat(messages[m.thread_root_id].timestamp)
            ).total_seconds()
        )
        for m in replies
    )
    thread_sizes = Counter(Counter(m.thread_root_id for m in replies).values())
    texts = [normalized(m.text) for m in world.messages if not m.deleted]
    activity = {
        "distinct_message_texts": len(set(texts)),
        "duplicate_message_excess": len(texts) - len(set(texts)),
        "replies": len(replies),
        "threads_by_reply_count": dict(sorted(thread_sizes.items())),
        "common_reply_delays_seconds": [
            {"seconds": delay, "count": count} for delay, count in delays.most_common(5)
        ],
    }
    return {
        "ok": not errors,
        "errors": errors,
        "counts": counts,
        "activity": activity,
        "gold_outputs": routes,
    }
