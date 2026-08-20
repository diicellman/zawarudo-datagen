from __future__ import annotations

import re
import unicodedata
from collections import Counter
from datetime import datetime

import verifiers.v1 as vf
from pydantic import Field

from ..contracts import INTERFACE_ID, SolverSummary, redact_secrets
from ..slack.models import AnswerSpec, SlackWorld, TaskContract
from ..slack.toolset import SlackState, SlackToolset, SlackToolsetConfig

FINAL_TAG = "final_answer"


class SolverData(vf.TaskData):
    instance_id: str
    question: str
    interface_id: str = INTERFACE_ID


class SolverConfig(vf.TaskConfig):
    tools: SlackToolsetConfig = SlackToolsetConfig()
    oracle: TaskContract | None = Field(default=None, exclude=True, repr=False)


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
        include_oracle: bool = False,
        idx: int | None = None,
    ) -> "SolverTask":
        prompt = (
            f"{contract.question}\n\nUse only the fixed read-only Slack actions. "
            f"Reply naturally inside <{FINAL_TAG}>...</{FINAL_TAG}> when done."
        )
        data = SolverData(
            idx=idx,
            name=instance_id,
            prompt=prompt,
            network_allow=[],
            network_block=["*"],
            instance_id=instance_id,
            question=contract.question,
        )
        config = SolverConfig(
            tools=SlackToolsetConfig.from_world(world, contract.actor_id),
            oracle=contract.model_copy(deep=True) if include_oracle else None,
        )
        return cls(data, config)

    @vf.metric
    async def final_answer_present(self, trace: vf.Trace) -> float:
        answer = extract_final_answer(trace.last_reply)
        return float(bool(answer))

    @vf.metric
    async def tool_call_count(self, trace: vf.Trace) -> float:
        return float(len(trace.state.completed_calls))

    @vf.reward
    async def answer_correct(self, trace: vf.Trace) -> float:
        answer = extract_final_answer(trace.last_reply)
        oracle = self.config.oracle
        if answer is None or oracle is None or not trace.state.completed_calls:
            return 0.0
        return float(score_supported_answer(answer, oracle.answer))

    @vf.metric
    async def exact_answer(self, trace: vf.Trace) -> float:
        answer = extract_final_answer(trace.last_reply)
        oracle = self.config.oracle
        if answer is None or oracle is None or oracle.answer.kind == "fact_summary":
            return 0.0
        return float(score_exact_answer(answer, oracle.answer))

    @vf.metric
    async def required_claim_coverage(self, trace: vf.Trace) -> float:
        answer = extract_final_answer(trace.last_reply) or ""
        oracle = self.config.oracle
        if oracle is None:
            return 0.0
        claims = oracle.answer.required_claims
        return sum(_claim_coverage(claim, answer) for claim in claims) / len(claims)


def extract_final_answer(text: str) -> str | None:
    matches = re.findall(
        rf"<{FINAL_TAG}>\s*(.*?)\s*</{FINAL_TAG}>",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if len(matches) != 1 or not matches[0].strip():
        return None
    return matches[0].strip()


def normalize_answer(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    value = re.sub(r"[^\w\s/-]", " ", value)
    return " ".join(value.split())


def _dates(value: str) -> set[str]:
    month = (
        r"(?:January|February|March|April|May|June|July|August|September|October|"
        r"November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"
    )
    candidates = [value.strip()]
    for pattern in (
        r"\b\d{4}[-/]\d{2}[-/]\d{2}\b",
        rf"\b{month}\s+\d{{1,2}},?\s+\d{{4}}\b",
        rf"\b\d{{1,2}}\s+{month}\s+\d{{4}}\b",
    ):
        candidates.extend(match.group(0) for match in re.finditer(pattern, value, re.IGNORECASE))
    parsed: set[str] = set()
    for candidate in candidates:
        rendered = re.sub(r"[,]", "", candidate)
        for fmt in (
            "%Y-%m-%d",
            "%Y/%m/%d",
            "%B %d %Y",
            "%b %d %Y",
            "%d %B %Y",
            "%d %b %Y",
        ):
            try:
                parsed.add(datetime.strptime(rendered, fmt).date().isoformat())
                break
            except ValueError:
                continue
    return parsed


def _date(value: str) -> str | None:
    parsed = _dates(value)
    return next(iter(parsed)) if len(parsed) == 1 else None


def _list_items(value: str) -> list[str]:
    pieces = re.split(r"[,;\n]|\s+and\s+", value)
    output: list[str] = []
    for piece in pieces:
        unbulleted = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", piece)
        normalized = normalize_answer(unbulleted)
        if normalized:
            output.append(normalized)
    return output


def score_exact_answer(value: str, spec: AnswerSpec) -> bool:
    if spec.kind == "fact_summary":
        return False
    if spec.kind == "date":
        return _date(value) is not None and _date(value) == _date(spec.canonical_answer)
    if spec.kind == "list":
        actual, expected = _list_items(value), _list_items(spec.canonical_answer)
        return actual == expected if spec.list_order_matters else sorted(actual) == sorted(expected)
    return normalize_answer(value) == normalize_answer(spec.canonical_answer)


_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "being",
        "by",
        "for",
        "from",
        "i",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "with",
    }
)
_NEGATIONS = frozenset({"neither", "never", "no", "nor", "not", "without"})
_CONTRASTS = frozenset({"although", "but", "however", "yet"})
_TOKEN_PATTERN = re.compile(r"\d+(?:[.,]\d+)?%?|[a-z0-9_]+|[.!?;]")
_TOKEN_ALIASES = {
    "because": "caus",
    "expiration": "expir",
}


def _stem_token(token: str) -> str:
    if token in _TOKEN_ALIASES:
        return _TOKEN_ALIASES[token]
    if re.fullmatch(r"\d+(?:[.,]\d+)?%?", token):
        return token
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 4 and token.endswith("s"):
        return token[:-1]
    if len(token) > 4 and token.endswith("e"):
        return token[:-1]
    return token


def _is_effective_negation(tokens: list[str], index: int) -> bool:
    token = tokens[index]
    return token in _NEGATIONS and not (
        token == "not" and index + 1 < len(tokens) and tokens[index + 1] == "only"
    )


def _strong_denial_indexes(tokens: list[str]) -> set[int]:
    denied: set[int] = set()
    segment: list[int] = []

    def finish() -> None:
        if not segment:
            return
        words = [tokens[index] for index in segment]
        strong_denial = False
        for offset, word in enumerate(words):
            before = words[max(0, offset - 2) : offset]
            after = words[offset + 1 : offset + 5]
            if word in {"false", "incorrect", "untrue"} and (
                "that" in after or any(item in {"is", "was"} for item in before)
            ):
                strong_denial = True
            if word == "no" and "evidence" in after:
                evidence = offset + 1 + after.index("evidence")
                if "that" in words[evidence + 1 : evidence + 5]:
                    strong_denial = True
            if word in {"deny", "denied", "denies"} and "that" in after:
                strong_denial = True
            if word == "not" and any(item in {"case", "true"} for item in after) and "that" in after:
                strong_denial = True
        if strong_denial:
            denied.update(segment)

    for index, token in enumerate(tokens):
        if token in {".", "!", "?", ";"} or token in _CONTRASTS:
            finish()
            segment = []
        else:
            segment.append(index)
    finish()
    return denied


def _content_sequence(value: str) -> list[tuple[str, bool]]:
    normalized = unicodedata.normalize("NFKC", value).casefold().replace("’", "'")
    normalized = re.sub(r"n['’]t\b", " not", normalized)
    normalized = re.sub(r"\bcannot\b", "can not", normalized)
    normalized = normalized.replace("-", " ").replace("/", " ")
    tokens = _TOKEN_PATTERN.findall(normalized)
    strong_denials = _strong_denial_indexes(tokens)
    output: list[tuple[str, bool]] = []
    boundary = -1
    contrast = -1
    for index, token in enumerate(tokens):
        if token in {".", "!", "?", ";"}:
            boundary = index
            continue
        if token in _CONTRASTS:
            contrast = index
            continue
        if token in _STOPWORDS or token in _NEGATIONS:
            continue
        start = max(boundary, contrast, index - 4) + 1
        negated = index in strong_denials or any(
            _is_effective_negation(tokens, position) for position in range(start, index)
        )
        if len(token) > 1 or token.isdigit():
            output.append((_stem_token(token), negated))
    return output


def _supports_ordered_relations(
    available: list[tuple[str, bool]],
    expected: list[tuple[str, bool]],
    *,
    max_extra: int = 4,
) -> bool:
    pairs = list(zip(expected, expected[1:], strict=False))
    if not pairs:
        return True
    matches = 0
    for left, right in pairs:
        if any(
            item == left and right in available[index + 1 : index + max_extra + 2]
            for index, item in enumerate(available)
        ):
            matches += 1
    return matches * 5 >= len(pairs) * 4


def _supports_claim(value: str, claim: str) -> bool:
    expected = _content_sequence(claim)
    available = _content_sequence(value)
    required = Counter(expected)
    return (
        bool(required) and required <= Counter(available) and _supports_ordered_relations(available, expected)
    )


def _contains_near_sequence(
    value: list[tuple[str, bool]],
    expected: list[tuple[str, bool]],
    *,
    max_extra: int = 2,
) -> bool:
    if not expected:
        return False
    for start, item in enumerate(value):
        if item != expected[0]:
            continue
        cursor = start
        for target in expected[1:]:
            cursor += 1
            while cursor < len(value) and value[cursor] != target:
                cursor += 1
            if cursor >= len(value) or cursor - start + 1 > len(expected) + max_extra:
                break
        else:
            return True
    return False


def _matches_forbidden(value: str, claim: str) -> bool:
    available = _content_sequence(value)
    expected = _content_sequence(claim)
    quantities = Counter(item for item in expected if re.fullmatch(r"\d+(?:[.,]\d+)?%?", item[0]))
    if quantities and quantities <= Counter(available):
        return True
    return _contains_near_sequence(available, expected)


def score_supported_answer(value: str, spec: AnswerSpec) -> bool:
    if any(_matches_forbidden(value, forbidden) for forbidden in spec.forbidden_claims):
        return False
    if spec.kind == "date":
        return _date(value) is not None and _date(value) == _date(spec.canonical_answer)
    if spec.kind == "list":
        return score_exact_answer(value, spec)
    return _supports_claim(value, spec.canonical_answer)


def _claim_coverage(claim: str, answer: str) -> float:
    tokens = set(normalize_answer(claim).split())
    if not tokens:
        return 0.0
    return len(tokens & set(normalize_answer(answer).split())) / len(tokens)


def tool_names(trace: vf.Trace) -> list[str]:
    calls: dict[str, str] = {}
    results: set[str] = set()
    for node in trace.nodes:
        message = node.message.model_dump(mode="json", exclude_none=True)
        for call in message.get("tool_calls", []):
            name = call.get("name") or call.get("function", {}).get("name")
            call_id = call.get("id")
            if isinstance(call_id, str) and isinstance(name, str) and name.startswith("slack_"):
                calls[call_id] = name
        if message.get("role") == "tool" and isinstance(message.get("tool_call_id"), str):
            results.add(message["tool_call_id"])
    return [calls[call_id] for call_id in calls if call_id in results]


def visible_errors(trace: vf.Trace) -> list[str]:
    output = [str(redact_secrets(f"{error.type}: {error.message}"))[:1_000] for error in trace.errors[:20]]
    for node in trace.nodes:
        message = node.message.model_dump(mode="json", exclude_none=True)
        if message.get("role") != "tool":
            continue
        content = str(message.get("content", ""))
        normalized = content.lstrip().casefold()
        if normalized.startswith(("error", '{"error"', "{'error'")):
            output.append(str(redact_secrets(content))[:500])
    return output


def summarize_solver(trace: vf.Trace) -> SolverSummary:
    answer = extract_final_answer(trace.last_reply) or ""
    names = [call["tool"] for call in trace.state.completed_calls]
    runtime_id = trace.agent.runtime.id if trace.agent and trace.agent.runtime else None
    return SolverSummary(
        final_answer_present=bool(answer),
        final_answer=answer,
        tool_call_count=len(names),
        tool_names=names,
        visible_errors=visible_errors(trace),
        trace_id=trace.id,
        runtime_id=runtime_id,
    )


__all__ = [
    "SolverConfig",
    "SolverData",
    "SolverTask",
    "extract_final_answer",
    "score_exact_answer",
    "score_supported_answer",
    "summarize_solver",
    "tool_names",
]
