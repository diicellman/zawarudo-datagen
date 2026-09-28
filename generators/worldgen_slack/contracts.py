"""Private generation contracts and checks for a shared Slack workspace."""

import json
import random
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal, Self
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import AfterValidator, BeforeValidator, Field, model_validator

from worldgen_slack.slack.models import (
    QUALITY_CRITERIA,
    AnswerSpec,
    Conversation,
    Message,
    NonEmptyText,
    SafeId,
    SlackWorld,
    StrictModel,
    User,
)

from worldgen_slack.slack.api import SlackAPI, ReadCall, canonical, digest

SEED_MAX_BYTES = 262_144
SEED_MAX_MESSAGES = 512
SEED_MAX_TEXT_CHARS = 128_000


class SeedMessage(StrictModel):
    text: NonEmptyText
    speaker: NonEmptyText | None = None
    timestamp: NonEmptyText | None = None


class SeedExample(StrictModel):
    id: SafeId
    dataset: NonEmptyText
    revision: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
    rows: list[Annotated[int, Field(ge=0)]] = Field(min_length=1, max_length=100)
    source_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    notes: NonEmptyText
    messages: list[SeedMessage] = Field(min_length=1, max_length=101)

    @model_validator(mode="after")
    def ordered_rows(self) -> Self:
        if self.rows != sorted(set(self.rows)):
            raise ValueError("seed row indices must be unique and ascending")
        return self


class SeedPacket(StrictModel):
    examples: list[SeedExample] = Field(min_length=1, max_length=48)

    @model_validator(mode="after")
    def bounded(self) -> Self:
        if len({example.id for example in self.examples}) != len(self.examples):
            raise ValueError("seed example IDs must be unique")
        messages = [message for example in self.examples for message in example.messages]
        if (
            len(messages) > SEED_MAX_MESSAGES
            or sum(len(message.text) for message in messages) > SEED_MAX_TEXT_CHARS
        ):
            raise ValueError("seed packet exceeds 512 messages or 128000 text characters")
        if any(not message.text.strip() for message in messages):
            raise ValueError("seed messages must not be blank")
        if len(self.model_dump_json().encode()) + 1 > SEED_MAX_BYTES:
            raise ValueError("seed packet exceeds 256 KiB")
        return self


def load_seed_packet(path: Path) -> SeedPacket:
    with path.open("rb") as stream:
        raw = stream.read(SEED_MAX_BYTES + 1)
    if len(raw) > SEED_MAX_BYTES:
        raise ValueError("seed packet exceeds 256 KiB")
    return SeedPacket.model_validate_json(raw)


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


def utc(value: str) -> str:
    parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value:
        raise ValueError("timestamps must be canonical UTC: YYYY-MM-DDTHH:MM:SSZ")
    return value


Timestamp = Annotated[str, AfterValidator(utc)]


def wall_clock(value: str) -> str:
    datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    return value


def to_utc(local: str, timezone: str) -> str:
    moment = datetime.strptime(local, "%Y-%m-%d %H:%M:%S").replace(tzinfo=ZoneInfo(timezone))
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def to_local(utc: str, timezone: str, form: str = "%Y-%m-%d %H:%M:%S") -> str:
    return datetime.fromisoformat(utc).astimezone(ZoneInfo(timezone)).strftime(form)


def clock(scene: "ScenePlan", zones: dict[str, str]) -> str:
    """A scene is written on one clock: its first participant's timezone."""
    return zones[scene.participant_ids[0]]


def zone(value: str) -> str:
    try:
        ZoneInfo(value)
    except (ZoneInfoNotFoundError, ValueError) as error:
        raise ValueError(f"unknown IANA timezone {value!r}") from error
    return value


class Premise(StrictModel):
    company: NonEmptyText
    niche: NonEmptyText
    region: NonEmptyText
    size: NonEmptyText
    culture: NonEmptyText
    cast: NonEmptyText


class Premises(StrictModel):
    premises: list[Premise] = Field(min_length=2)


def pick_premise(premises: Premises, count: int, used: list[str], seed: int) -> Premise:
    """Entropy comes from code: the model proposes distinct options, the run seed chooses one."""
    names = [normalized(p.company).split()[0] for p in premises.premises]
    taken = {normalized(name).split()[0] for name in used if normalized(name)}
    if len(names) != count or len(set(names)) != count:
        raise ValueError(f"propose exactly {count} premises whose company names start with distinct words")
    if reused := sorted(set(names) & taken):
        raise ValueError(f"company names reuse names from used_names: {reused}")
    return random.Random(seed).choice(premises.premises)


class Persona(StrictModel):
    id: SafeId
    role: NonEmptyText
    seniority: NonEmptyText
    timezone: Annotated[str, AfterValidator(zone)]
    voice: NonEmptyText


class Fact(StrictModel):
    id: SafeId
    subject: NonEmptyText
    predicate: NonEmptyText
    value: NonEmptyText
    valid_from: Timestamp
    valid_until: Timestamp | None = None
    description: NonEmptyText

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
    personas: list[Persona] = Field(default_factory=list)
    groups: list[WorkGroup] = Field(min_length=1)
    facts: list[Fact] = Field(min_length=1)
    tasks: list[PlannedTask] = Field(min_length=1)

    @model_validator(mode="after")
    def references(self) -> Self:
        for label, values in (("task", self.tasks), ("fact", self.facts), ("group", self.groups)):
            if len({v.id for v in values}) != len(values):
                raise ValueError(f"duplicate {label} ID")
        if self.personas and sorted(p.id for p in self.personas) != sorted(p.id for p in self.people):
            raise ValueError("personas must describe every person exactly once")
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


class Beat(StrictModel):
    fact_id: SafeId
    author_id: SafeId


class ScenePlan(StrictModel):
    id: SafeId
    conversation_id: SafeId
    participant_ids: list[SafeId] = Field(min_length=1)
    start: Timestamp
    end: Timestamp
    situation: NonEmptyText
    beats: list[Beat] = Field(default_factory=list)
    length: int = Field(ge=1, le=60)
    revision_note: str = ""

    @model_validator(mode="after")
    def ordered(self) -> Self:
        if self.end <= self.start:
            raise ValueError(f"{self.id}: end must come after start")
        return self


class Detail(StrictModel):
    value: NonEmptyText
    since: Timestamp | None = None
    at: Timestamp | None = None


class Plan(StrictModel):
    conversations: list[Conversation] = Field(min_length=1)
    scenes: list[ScenePlan] = Field(min_length=1)
    details: dict[NonEmptyText, Detail] = Field(default_factory=dict)


def listed(value):
    """Models often write null for an empty list, a bare string for a one-item list, or a list as a string."""
    if value is None:
        return []
    if isinstance(value, str):
        return json.loads(value) if value.startswith("[") else [value]
    return value


def noted(value):
    """Notes are strings; models often write a note as an object such as {who, what, by}."""
    return [", ".join(str(v) for v in n.values()) if isinstance(n, dict) else n for n in listed(value)]


Conveys = Annotated[list[SafeId], BeforeValidator(listed)]
Notes = Annotated[list[NonEmptyText], BeforeValidator(noted)]


class Line(StrictModel):
    author_id: SafeId
    text: NonEmptyText
    local_time: Annotated[str, AfterValidator(wall_clock)]
    reply_to: int | None = Field(default=None, ge=0)
    conveys: Conveys = Field(default_factory=list)


class WrittenScene(StrictModel):
    lines: list[Line] = Field(min_length=1)
    introduces: Notes = Field(default_factory=list)
    promises: Notes = Field(default_factory=list)


class SceneRecord(StrictModel):
    key: str
    scene: WrittenScene


class Rewrite(StrictModel):
    scene_id: SafeId
    note: NonEmptyText


class BindOutput(StrictModel):
    bindings: list[Binding]
    rewrites: list[Rewrite] = Field(default_factory=list)


def first_mentions(plan: Plan) -> dict[str, str]:
    """fact_id → the earliest scene that has a beat for it."""
    first = {}
    for scene in sorted(plan.scenes, key=lambda s: (s.start, s.id)):
        for beat in scene.beats:
            first.setdefault(beat.fact_id, scene.id)
    return first


def timed(catalog: Catalog) -> dict[str, str]:
    """Facts with a time of day: the world first states each in the minute of its valid_from. Date-only facts
    (00:00:00Z) are not timed."""
    return {f.id: f.valid_from for f in catalog.facts if not f.valid_from.endswith("T00:00:00Z")}


def check_plan(catalog: Catalog, plan: Plan) -> Plan:
    conversations = {c.id: c for c in plan.conversations}
    facts = {f.id: f for f in catalog.facts}
    if len(conversations) != len(plan.conversations):
        raise ValueError("duplicate conversation ID")
    if len({s.id for s in plan.scenes}) != len(plan.scenes):
        raise ValueError("duplicate scene ID")
    if unknown := {m for c in plan.conversations for m in c.member_ids} - {p.id for p in catalog.people}:
        raise ValueError(f"conversation members outside the catalog: {sorted(unknown)}")
    for scene in plan.scenes:
        conversation = conversations.get(scene.conversation_id)
        if conversation is None:
            raise ValueError(f"{scene.id}: unknown conversation {scene.conversation_id}")
        if outside := set(scene.participant_ids) - set(conversation.member_ids):
            raise ValueError(f"{scene.id}: participants are not members: {sorted(outside)}")
        for beat in scene.beats:
            if beat.fact_id not in facts or beat.author_id not in scene.participant_ids:
                raise ValueError(f"{scene.id}: beat needs a catalog fact and a participant author")
    starts, minutes = {s.id: s.start for s in plan.scenes}, timed(catalog)
    for fact_id, scene_id in first_mentions(plan).items():
        if (minute := minutes.get(fact_id)) and starts[scene_id][:16] > minute[:16]:
            raise ValueError(
                f"{scene_id}: fact {fact_id} is stated at {facts[fact_id].valid_from}; the earliest scene "
                "stating it must start by then"
            )
    return plan


def check_scene(
    plan: ScenePlan, scene: WrittenScene, zones: dict[str, str], timely: dict[str, str] | None = None
) -> list[str]:
    """`timely` maps facts first stated in this scene to the UTC minute they must be stated in."""
    errors = []
    if outside := {line.author_id for line in scene.lines} - set(plan.participant_ids):
        errors.append(f"authors must be participants; not participants: {sorted(outside)}")
    else:
        zone = clock(plan, zones)
        times = [to_utc(line.local_time, zone) for line in scene.lines]
        if times[0] < plan.start:
            errors.append(
                f"line 0 is at {scene.lines[0].local_time}, before the scene start {to_local(plan.start, zone)}"
            )
        if times[-1] > plan.end:
            errors.append(
                f"the last line is at {scene.lines[-1].local_time}, after the scene end {to_local(plan.end, zone)}"
            )
        errors += [
            f"line {i} is at {scene.lines[i].local_time}, before line {i - 1} at {scene.lines[i - 1].local_time}; "
            "messages must follow each other in time"
            for i in range(1, len(times))
            if times[i] < times[i - 1]
        ]
        for fact_id, minute in (timely or {}).items():
            stated = [(t, line) for t, line in zip(times, scene.lines) if fact_id in line.conveys]
            if stated and stated[0][0][:16] != minute[:16]:
                first = stated[0][1]
                errors.append(
                    f"fact {fact_id}: the first line stating it is at {first.local_time} but must be in "
                    f"the minute {to_local(minute, zone, '%Y-%m-%d %H:%M')}"
                )
    for index, line in enumerate(scene.lines):
        if line.reply_to is not None and line.reply_to >= index:
            errors.append(f"line {index}: reply_to {line.reply_to} must name an earlier line of this scene")
    beats = {b.fact_id for b in plan.beats}
    if extra := {f for line in scene.lines for f in line.conveys} - beats:
        errors.append(f"conveys may only list beat fact IDs; unknown: {sorted(extra)}")
    for beat in plan.beats:
        if not any(beat.fact_id in line.conveys and line.author_id == beat.author_id for line in scene.lines):
            errors.append(f"{beat.author_id} must state fact {beat.fact_id} in a line that conveys it")
    return errors


def assemble(catalog: Catalog, plan: Plan, scenes: dict[str, WrittenScene]) -> tuple[SlackWorld, dict]:
    """Deterministic scene → message mapping; returns the world and fact_id → conveying message IDs."""
    zones = {p.id: p.timezone for p in catalog.personas}
    messages, conveyed, placed = [], defaultdict(list), {}
    for scene_plan in plan.scenes:
        lines = scenes[scene_plan.id].lines

        def root(index, lines=lines):
            """Slack threads are one level deep: replying to a reply joins its thread."""
            return index if lines[index].reply_to is None else root(lines[index].reply_to)

        ids = placed[scene_plan.id] = [
            "m" + digest([scene_plan.id, index])[:10] for index in range(len(scenes[scene_plan.id].lines))
        ]
        for message_id, line in zip(ids, scenes[scene_plan.id].lines, strict=True):
            messages.append(
                Message(
                    id=message_id,
                    conversation_id=scene_plan.conversation_id,
                    author_id=line.author_id,
                    text=line.text,
                    timestamp=to_utc(line.local_time, clock(scene_plan, zones)),
                    thread_root_id=None if line.reply_to is None else ids[root(line.reply_to)],
                )
            )
            for fact_id in line.conveys:
                conveyed[fact_id].append(message_id)
    messages.sort(key=lambda m: (m.timestamp, m.id))
    world = SlackWorld(users=catalog.people, conversations=plan.conversations, messages=messages)
    promises = [{"scene_id": s.id, "promise": p} for s in plan.scenes for p in scenes[s.id].promises]
    return world, {"conveyed": dict(conveyed), "scene_messages": placed, "open_promises": promises}


def style(world: SlackWorld, personas: list[Persona]) -> dict:
    """Surface statistics of voice and timing; people are compared with their personas by the judge."""
    zones = {p.id: ZoneInfo(p.timezone) for p in personas}

    def share(values):
        values = list(values)
        return round(sum(values) / len(values), 2) if values else None

    def measure(messages):
        texts = [m.text.strip() for m in messages]
        local = [
            datetime.fromisoformat(m.timestamp).astimezone(zones[m.author_id])
            for m in messages
            if m.author_id in zones
        ]
        return {
            "messages": len(texts),
            "median_words": statistics.median(len(t.split()) for t in texts),
            "lowercase_start": share(t[:1].islower() for t in texts),
            "period_end": share(t.endswith(".") for t in texts),
            "question": share("?" in t for t in texts),
            "short": share(len(t.split()) <= 4 for t in texts),
            "seconds_zero": share(m.timestamp.endswith(":00Z") for m in messages),
            "off_hours": share(t.hour < 7 or t.hour >= 20 or t.weekday() >= 5 for t in local),
        }

    live = [m for m in world.messages if not m.deleted]
    if not live:
        return {}
    by_author = defaultdict(list)
    for message in live:
        by_author[message.author_id].append(message)
    return measure(live) | {
        "authors": {author: measure(items) for author, items in sorted(by_author.items())}
    }


class Issue(StrictModel):
    owner: Literal["synthesizer", "builder"]
    artifact: Literal["catalog", "workspace", "bindings"]
    task_ids: list[SafeId] = Field(default_factory=list)
    fact_ids: list[SafeId] = Field(default_factory=list)
    message_ids: list[SafeId] = Field(default_factory=list)
    defect: NonEmptyText
    requested_change: NonEmptyText
    blocking: bool = True

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
        if set(self.criteria) - set(QUALITY_CRITERIA):
            raise ValueError("unknown quality criterion")
        if len({t.task_id for t in self.tasks}) != len(self.tasks):
            raise ValueError("duplicate task review")
        if any(not 0 <= value <= 1 for value in self.criteria.values()):
            raise ValueError("criteria must be finite scores from zero to one")
        blocking = any(i.blocking for i in self.issues)
        if self.approved and (blocking or any(not t.valid or not t.answer_complete for t in self.tasks)):
            raise ValueError("approval contradicts blocking defects")
        if not self.approved and not self.issues:
            raise ValueError("rejection must include actionable issues")
        return self


def verdict_schema(phase: str) -> dict:
    schema = Verdict.model_json_schema()
    schema["properties"]["criteria"] = {
        "type": "object",
        "properties": {name: {"type": "number", "minimum": 0, "maximum": 1} for name in QUALITY_CRITERIA},
        "additionalProperties": False,
    }
    if phase == "world":
        schema["properties"]["criteria"]["required"] = list(QUALITY_CRITERIA)
        schema["required"].append("criteria")
    return schema


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


def accepted(verdict: Verdict, payload: dict, acceptance) -> bool:
    """World acceptance from the judge's findings and the configured floors (`config.Acceptance`)."""
    claims = {t["id"]: len(t["answer"]["required_claims"]) for t in payload["tasks"]}
    return (
        all(
            t.valid and t.answer_complete and len(set(t.supported_claims)) == claims[t.task_id]
            for t in verdict.tasks
        )
        and not any(i.blocking or acceptance.minor_issues_block for i in verdict.issues)
        and min(verdict.criteria.values()) >= acceptance.criterion_floor
        and quality(verdict) >= acceptance.quality_floor
    )


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
        "style": style(world, catalog.personas),
    }
    return {
        "ok": not errors,
        "errors": errors,
        "counts": counts,
        "activity": activity,
        "gold_outputs": routes,
    }
