"""The answer judge: a solver's answer graded against the release's gold, part by part. The model reports what the answer
commits to and a few narrow criteria, each with its reason (judge_prompt.md); code decides each part's outcome by the
grading policy (local-docs/calibration/POLICY.md, v4, after "Training Language Models to Fail Gracefully"): a success
gives every item the request asks for and commits nothing the truth rules out, a fallback holds back honestly, and
anything else is an error. A grade reads as an AnswerGrade: a success correct, a fallback abstained."""

import json
import re
from pathlib import Path
from typing import Literal

import verifiers.v1 as vf
from pydantic import BaseModel, Field

from .release import NonEmptyText, PrivateAnswer, StrictModel

Outcome = Literal["success", "fallback", "error"]


class PartGrade(StrictModel):
    """One part of a composite request, graded on its own: whether its answer is right, whether it declined that part,
    and its alignment with the part's items."""

    label: NonEmptyText
    correct: bool
    abstained: bool
    committed: list[str]
    missing: list[str]
    extra: list[str]


class AnswerGrade(StrictModel):
    """The answer against the gold (correct), whether it held back (abstained), whether the messages its answer rests
    on were shown to the try (grounded), and how many context details it contradicts (counted, not scored). The
    alignment says what it committed to, which of the gold's items it lacks, and what it presents beyond them. A
    composite request's grade is its parts'."""

    correct: bool
    abstained: bool
    grounded: bool
    unsupported_asides: int = Field(ge=0)
    committed: list[str]
    missing: list[str]
    extra: list[str]
    reason: NonEmptyText
    parts: list[PartGrade] = []


def reward(grade: AnswerGrade) -> float:
    """+1 for a right answer the observations ground, 0 for a right one they don't or for declining to answer, -1 for
    a wrong one: under a 1-or-0 grade a guess always beats saying it cannot be told (TruthRL, Kalai et al.), and a
    solver gave a provisional figure as final in 38 of 65 misses. A composite request is all or nothing: +1 only when
    every part is right and grounded, -1 when any part is wrong, 0 when it declined what it could not tell."""
    if grade.correct:
        return 1.0 if grade.grounded else 0.0
    return 0.0 if grade.abstained else -1.0


class Check(StrictModel):
    reason: NonEmptyText
    verdict: bool


class Gives(StrictModel):
    item: int
    reason: NonEmptyText
    verdict: bool


class PartReading(StrictModel):
    """What an answer says to one part of a request, criterion by criterion: read, not graded."""

    part: NonEmptyText
    commits: str
    gives: list[Gives]
    conflicts: Check
    extra: Check | None
    hedged: Check
    holds_back: Check
    declines: Check | None
    false_asides: list[str]


class Reading(StrictModel):
    parts: list[PartReading]


READINGS = 3  # readings a grade asks for before a misread fails it: each a fresh call, paid once


class Misread(ValueError):
    """A reading that does not read the gold's parts and items, or a reply that holds none: the reader's slip, not the
    answer's, so it is asked again (READINGS) where an error of the grader's own code is not. `usage` is what its call
    cost, for a judge whose calls no trace records (the forge's task judge)."""

    def __init__(self, message: str, usage: vf.Usage | None = None):
        super().__init__(message)
        self.usage = usage


def asked(model: type[BaseModel]) -> str:
    """The reply a judge asks for, after its prompt: one JSON object of `model`'s schema, in plain text. Not structured
    output: Prime bills a GLM 5.3 call made with response_format about 20 times the cost it reports, and a plain call
    at that cost (2026-10-24, 40 calls each way on one grading prompt; 39 of the 40 plain replies read)."""
    return "\n\nReply with only a JSON object and nothing else, matching this JSON schema:\n" + json.dumps(
        model.model_json_schema()
    )


def replied[M: BaseModel](text: str, model: type[M]) -> M:
    """A reply as `model`: the first JSON object it holds (prose around it aside, a raw control character inside a
    string read as written), else a ValueError naming what failed."""
    decoder = json.JSONDecoder(strict=False)
    for start in (m.start() for m in re.finditer(r"\{", text)):
        try:
            value, _ = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            continue
        return model.model_validate(value)
    raise ValueError(f"no JSON object in the reply: {text[:200]!r}")


REPLY = asked(Reading) + (
    "\nextra, for a part that is not a set, and declines, for a part that can be answered, are null, not objects."
)


def label_of(answer: PrivateAnswer) -> str:
    return answer.label or "the answer"


def part_outcome(reading: PartReading, answer: PrivateAnswer) -> Outcome:
    """A part's outcome by the policy (v4): anything the truth rules out, a hedge or, in a set, a non-member is an
    error; a part that can't be answered succeeds when it declines and falls back when it holds back; any other
    succeeds when it gives every item and falls back when it holds back. Whatever else it commits is an error. A
    criterion a part's type does not ask (extra beside a set, gives beside its items) decides nothing."""
    member = answer.answer_type == "set" and reading.extra.verdict
    if reading.conflicts.verdict or reading.hedged.verdict or member:
        return "error"
    if answer.answer_type == "refusal":
        return (
            "success" if reading.declines.verdict else "fallback" if reading.holds_back.verdict else "error"
        )
    if all(g.verdict for g in reading.gives):
        return "success"
    return "fallback" if reading.holds_back.verdict else "error"


def request_outcome(outcomes: list[Outcome]) -> Outcome:
    """A request's outcome from its parts': a success when every part is, an error when any part is, else a fallback."""
    if all(o == "success" for o in outcomes):
        return "success"
    return "error" if "error" in outcomes else "fallback"


def matched(reading: Reading, reference: PrivateAnswer) -> list[PartReading]:
    """Each part's reading: by its label when the reading names every part once, in any order, else by its place when
    it reads as many parts as there are (a reader that renames them). Each reads every item of its part once (a part
    with no items reads none it is held to), a set whether it presents a non-member, and a part that can't be answered
    whether it declines; else a Misread."""
    answers = reference.answers()
    labels = [label_of(a) for a in answers]
    named = {r.part: r for r in reading.parts}
    if len(named) == len(reading.parts) == len(answers) and set(named) == set(labels):
        readings = [named[label] for label in labels]
    elif len(reading.parts) == len(answers):
        readings = list(reading.parts)
    else:
        raise Misread(f"a reading reads each of its {len(answers)} parts once, not {len(reading.parts)}")
    for r, a in zip(readings, answers):
        if a.rows and sorted(g.item for g in r.gives) != list(range(1, len(a.rows) + 1)):
            raise Misread(f"{label_of(a)}: gives reads each of its {len(a.rows)} items once")
        if (a.answer_type == "set" and r.extra is None) or (
            a.answer_type == "refusal" and r.declines is None
        ):
            raise Misread(f"{label_of(a)}: its {a.answer_type} criterion is unread")
    return readings


def why(reading: PartReading, outcome: Outcome) -> str:
    """The reason that decided a part: the criterion that failed it, or else what held it back or gave it."""
    for check in (reading.conflicts, reading.hedged, reading.extra):
        if check and check.verdict:
            return check.reason
    if outcome == "success":
        return (reading.declines.reason if reading.declines else "") or "; ".join(
            g.reason for g in reading.gives
        )
    if outcome == "fallback":
        return reading.holds_back.reason
    return next((g.reason for g in reading.gives if not g.verdict), reading.holds_back.reason)


def graded(reading: Reading, reference: PrivateAnswer, grounded: bool) -> tuple[AnswerGrade, Outcome]:
    """A reading graded by code, as an AnswerGrade (a success correct, a fallback abstained) and its outcome."""
    answers = reference.answers()
    readings = matched(reading, reference)
    outcomes = [part_outcome(r, a) for r, a in zip(readings, answers)]
    outcome = request_outcome(outcomes)
    composite = reference.answer_type == "composite"

    def aligned(r: PartReading, a: PrivateAnswer) -> dict:
        given = {g.item for g in r.gives if g.verdict}
        return {
            "committed": [r.commits] if r.commits.strip() else [],
            "missing": [str(row.get("answer")) for n, row in enumerate(a.rows, 1) if n not in given],
            "extra": [r.extra.reason] if r.extra and r.extra.verdict else [],
        }

    own = [aligned(r, a) for r, a in zip(readings, answers)]

    def labelled(key: str) -> list[str]:
        return [(f"{label_of(a)}: " if composite else "") + v for o, a in zip(own, answers) for v in o[key]]

    grade = AnswerGrade(
        correct=outcome == "success",
        abstained=outcome == "fallback",
        grounded=grounded,
        unsupported_asides=sum(len(r.false_asides) for r in readings),
        committed=labelled("committed"),
        missing=labelled("missing"),
        extra=labelled("extra"),
        reason=" | ".join(
            (f"{label_of(a)}: " if composite else "") + f"{o}: {why(r, o)}"
            for r, a, o in zip(readings, answers, outcomes)
        ),
        parts=[
            PartGrade(label=label_of(a), correct=o == "success", abstained=o == "fallback", **x)
            for a, o, x in zip(answers, outcomes, own)
        ]
        if composite
        else [],
    )
    return grade, outcome


EMPTY = "The response is empty: it gives no answer and does not say the request cannot be answered."


def empty_grade(reference: PrivateAnswer) -> AnswerGrade:
    """An empty response is wrong by rule, with no grader call: it lacks every item, of every part."""
    composite = reference.answer_type == "composite"
    missing = {label_of(a): [str(r.get("answer")) for r in a.rows] for a in reference.answers()}
    return AnswerGrade(
        correct=False,
        abstained=False,
        grounded=False,
        unsupported_asides=0,
        committed=[],
        missing=[(f"{label}: " if composite else "") + m for label, rows in missing.items() for m in rows],
        extra=[],
        reason=EMPTY,
        parts=[
            PartGrade(label=label, correct=False, abstained=False, committed=[], missing=rows, extra=[])
            for label, rows in missing.items()
        ]
        if composite
        else [],
    )


def gold(reference: PrivateAnswer, out_of_sight: list[list[dict]]) -> str:
    """The gold as the reader sees it: each part's label, answer type, numbered items and context, and the messages out
    of its asker's sight where it has any."""
    return json.dumps(
        [
            {
                "part": label_of(a),
                "answer_type": a.answer_type,
                "items": [{"item": n, "answer": str(r.get("answer"))} for n, r in enumerate(a.rows, 1)],
                "context": a.context,
            }
            | ({"out_of_sight": hidden} if hidden else {})
            for a, hidden in zip(reference.answers(), out_of_sight)
        ],
        ensure_ascii=False,
        indent=1,
    )


def fenced(text: str) -> str:
    """The answer as data: between fences longer than any run of backticks in it, so it cannot close them."""
    fence = "`" * max(3, 1 + max((len(run) for run in re.findall(r"`+", text)), default=0))
    return f"{fence}\n{text}\n{fence}"


class AnswerJudge(vf.Judge[Reading]):
    """Its prompt (judge_prompt.md, or the file JudgeConfig.prompt names) asks for a reading as plain JSON (REPLY), and
    code reads the reply: one that does not read as a Reading is a Misread, asked again. Each call is recorded on the
    trace as billed, whether its reply reads or not."""

    prompt = (Path(__file__).with_name("judge_prompt.md")).read_text(encoding="utf-8")

    def build_messages(self, **fields):
        return super().build_messages(**fields) + REPLY

    def parse(self, response: vf.JudgeResponse) -> Reading:
        try:
            return replied(response.text, Reading)
        except ValueError as error:
            raise Misread(
                f"a reply that does not read as a reading: {error}"[:300], response.usage
            ) from error
