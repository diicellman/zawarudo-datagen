"""The answer judge: a solver's answer graded against the release's reference, on what its calls showed.

Two graders. v1 (AnswerJudge) is one model call that decides right, declined or wrong. v2 (AnswerReader) has the model
read what an answer commits to, part by part, and code decide each part's outcome by the grading policy
(local-docs/calibration/POLICY.md, after "Training Language Models to Fail Gracefully"): a success gives every item the
request asks for and commits nothing false; a fallback holds back honestly and commits nothing false; anything else
that commits is an error. A v2 grade is mapped onto v1's fields (a success is correct, a fallback abstained), so what
reads grades reads either."""

import json
from typing import Literal

import verifiers.v1 as vf
from pydantic import BaseModel, Field

from .release import NonEmptyText, PrivateAnswer, StrictModel


class PartGrade(StrictModel):
    """One part of a composite request, graded on its own as an answer is: whether the response's answer to it is
    right, whether it declined that part, and its alignment with the part's rows."""

    label: NonEmptyText
    correct: bool
    abstained: bool
    committed: list[str]
    missing: list[str]
    extra: list[str]


class AnswerGrade(StrictModel):
    """The committed answer against the reference (correct), whether the response declined to give one (abstained),
    whether the observations support the claims that make its answer (grounded), and how many of the response's other
    material claims they don't (counted, not scored). The alignment says what it committed to, which of the
    reference's rows it lacks, and which values it has that the reference does not: code classes a miss from it. A
    composite request's grade is its parts' (`settled`)."""

    correct: bool
    abstained: bool
    grounded: bool
    unsupported_asides: int = Field(ge=0)
    committed: list[str]
    missing: list[str]
    extra: list[str]
    reason: NonEmptyText
    parts: list[PartGrade] = []


EMPTY = "The response is empty: it gives no answer and does not say the request cannot be answered."


def rows_of(reference: dict) -> list[str]:
    return [str(r["answer"]) for r in reference.get("rows", [])]


def settled(grade: AnswerGrade, reference: dict) -> AnswerGrade:
    """A composite request's grade, settled by code from its parts' and not by the grader's word on the whole: right
    only when every part is, declined when it is not and no part is wrong (it gave what it could tell and declined
    the rest), its alignment its parts' under their labels. Any other grade has no parts."""
    if reference.get("answer_type") != "composite":
        return grade.model_copy(update={"parts": []})
    labels = [p["label"] for p in reference["parts"]]
    if len(grade.parts) != len(labels):
        raise ValueError(f"a composite grade grades each of its {len(labels)} parts, in order")
    parts = [p.model_copy(update={"label": label}) for p, label in zip(grade.parts, labels)]
    correct = all(p.correct for p in parts)
    return grade.model_copy(
        update={
            "parts": parts,
            "correct": correct,
            "abstained": not correct and all(p.correct or p.abstained for p in parts),
            **{
                k: [f"{p.label}: {v}" for p in parts for v in getattr(p, k)]
                for k in ("committed", "missing", "extra")
            },
        }
    )


def empty_grade(reference: dict) -> AnswerGrade:
    """An empty response is wrong by rule, with no grader call: it lacks every reference row, of every part."""
    parts = [
        PartGrade(
            label=p["label"], correct=False, abstained=False, committed=[], missing=rows_of(p), extra=[]
        )
        for p in reference.get("parts", [])
    ]
    grade = AnswerGrade(
        correct=False,
        abstained=False,
        grounded=False,
        unsupported_asides=0,
        committed=[],
        missing=rows_of(reference),
        extra=[],
        reason=EMPTY,
        parts=parts,
    )
    return settled(grade, reference)


class AnswerJudge(vf.Judge[AnswerGrade]):
    schema = AnswerGrade

    async def evaluate(self, *, trace=None, **fields):
        """The grader's grade, settled by code for a composite request."""
        response = await super().evaluate(trace=trace, **fields)
        response.parsed = settled(response.parsed, fields.get("reference") or {})
        return response

    def build_messages(self, **fields):
        return [
            vf.UserMessage(
                content=(
                    "Grade a Slack answer against the question and its reference. First find the answer the response "
                    "commits to: one that lists alternatives without settling on one is wrong, while side options it "
                    "labels as such do not count against an answer it commits to, nor toward it: a value named only in "
                    "passing, as an aside or a side option, is not committed, so a set whose committed answer lacks a "
                    "reference row is wrong however the response mentions that row elsewhere. A response that gives no answer and "
                    "says the request cannot be answered or determined has abstained, whether or not that is right. "
                    "The reference's rows are the answer: for a set every row is required and order does not matter; "
                    "text and number are one value; a time asked for on the asker's clock is given on that clock; a "
                    "refusal is right when the response says the request cannot be fulfilled as asked from what the "
                    "asker can see: what it asks about does not exist, is out of their sight, or rests on a premise "
                    "the workspace contradicts. A status reference is the latest value the asker can see, which is "
                    "not settled: the response is right when it gives that value and says it is not final or "
                    "confirmed; giving it as final, or another value, is wrong; saying only that it cannot be "
                    "determined abstains. Accept equivalent wording. Then decide grounded on the answer's own claims: "
                    "the solver's RECORDED observations support the claims that make its answer (the value and what "
                    "it rests on; for a status, also that it is unsettled; for a refusal, that the thing is not there "
                    "to see or its premise is contradicted); the reference is not an observation, and no observations "
                    "means ungrounded. The observations are what the solver's calls showed, each thing once: messages "
                    "in time order, the people, channels and members it looked up, and a log of its calls with how "
                    "many results each returned. Count the response's other material claims the observations do not "
                    "support (unsupported_asides); they do not change grounded. Then align: committed lists the answer "
                    "the response commits to, one entry per value or row (empty when it abstains); missing lists the "
                    "reference rows it lacks; extra lists committed values the reference does not have. Return "
                    "correct, abstained, grounded, unsupported_asides, committed, missing, extra and a concise "
                    "reason. A composite reference is one request in parts, each with its label, answer type and "
                    "rows: grade each part on its own by the rules above, against its own rows and as the response "
                    "answers it, in parts (one entry per reference part, in its order: label, correct, abstained, "
                    "committed, missing, extra); a part the response leaves out lacks its rows. Grounded then covers "
                    "the claims of every part's answer. For any other reference, parts is empty.\n"
                    + json.dumps(fields, ensure_ascii=False)
                )
            )
        ]


def reward(grade: AnswerGrade) -> float:
    """+1 for a right answer the observations ground, 0 for a right one they don't or for declining to answer, -1 for
    a wrong one: under a 1-or-0 grade a guess always beats saying it cannot be told (TruthRL, Kalai et al.), and a
    solver gave a provisional figure as final in 38 of 65 misses. A composite request is all or nothing: +1 only when
    every part is right and grounded, -1 when any part is wrong, 0 when it declined what it could not tell."""
    if grade.correct:
        return 1.0 if grade.grounded else 0.0
    return 0.0 if grade.abstained else -1.0


# ---------------------------------------------------------------------- v2: the model reads, code grades

Outcome = Literal["success", "fallback", "error"]


class Grading(BaseModel):
    """Which grader grades a try, and whether it reads the try's tool outputs to judge grounding (else grounding is
    code's: each part's answer rests on a message the try was shown)."""

    version: Literal[1, 2] = 1
    ground: bool = True


class Commit(StrictModel):
    """A value the response commits to as its answer to a part: which reference answer item it states, if any (by
    index), and whether it states, for something the reference covers, another value than the reference's."""

    value: NonEmptyText
    item: int | None = None
    contradicts: bool = False


class PartReading(StrictModel):
    """What a response says to one part of a request: read, not graded."""

    label: NonEmptyText
    committed: list[Commit]
    hedged: bool
    holds_back: bool
    claims: Literal["settled", "unsettled", "not_visible", "never_happened", "none"]
    contradicts_context: list[str] = []
    slips: list[str] = []


class Reading(StrictModel):
    parts: list[PartReading]
    grounded: bool | None = None  # read only when the reader is given the try's observations
    reason: NonEmptyText


def golds(reference: PrivateAnswer) -> list[dict]:
    """What code grades each part against: its answer type, its answer items, its context, and whether the truth it
    rests on is out of its asker's sight."""
    return [
        {
            "label": a.label,
            "answer_type": a.answer_type,
            "items": [str(r.get("answer")) for r in a.rows],
            "context": a.context,
            "unseen": bool(a.unseen),
        }
        for a in reference.answers()
    ]


def part_outcome(reading: PartReading, gold: dict) -> Outcome:
    """A part's outcome by the grading policy. Any committed falsehood is an error: a hedge, a value that contradicts
    the reference's answer, a status given as settled, and a flat "it never happened" about a truth out of the asker's
    sight. A detail beside what was asked (its context, or anything the request does not ask for) decides nothing: a
    false one is counted as an aside. Then: a refusal succeeds when it holds back; a set when it states every item and nothing beyond
    them, and falls back when it flags a partial answer that adds nothing false; a status when it states the latest
    visible value and says it is unsettled or out of sight; a text or number when it states every answer item (a split
    gold may ask for more than one). Holding back with nothing committed, or with only some items and none beyond them,
    is a fallback; committing a value that states no item, or part of the answer as if whole, is an error."""
    kind, count = gold["answer_type"], len(gold["items"])
    stated = {c.item for c in reading.committed if c.item is not None and 0 <= c.item < count}
    stray = [c for c in reading.committed if c.item is None or not 0 <= c.item < count]
    if (
        reading.hedged
        or any(c.contradicts for c in reading.committed)
        or (kind == "status" and reading.claims == "settled")
        or (reading.claims == "never_happened" and gold["unseen"])
    ):
        return "error"
    if kind == "refusal":
        return "success" if reading.holds_back and not reading.committed else "error"
    complete = count > 0 and stated == set(range(count))
    if kind == "set":
        if complete and not stray:
            return "success"
        return "fallback" if reading.holds_back and not stray else "error"
    if kind == "status" and complete:
        return "success" if reading.claims in ("unsettled", "not_visible") else "error"
    if kind != "status" and complete:
        return "success"
    if reading.holds_back and not stray:
        return "fallback"  # nothing committed, or only some of what is asked, flagged as partial
    return "error"


def request_outcome(outcomes: list[Outcome]) -> Outcome:
    """A request's outcome from its parts': a success when every part is, an error when any part is, else a fallback."""
    if all(o == "success" for o in outcomes):
        return "success"
    return "error" if "error" in outcomes else "fallback"


READINGS = 3  # readings a grade asks for before a misread fails it: each a fresh call, paid once


class Misread(ValueError):
    """A reading that does not read the reference's parts: the reader's slip, not the answer's, so it is asked again
    (READINGS) where an error of the grader's own code is not."""


def matched(reading: Reading, parts: list[dict]) -> list[PartReading]:
    """Each part's reading: by its label when the reading names every part once, in any order; else by its place when
    it reads as many parts as there are (a reader that renames them); else a Misread."""
    labels = [g["label"] or "the answer" for g in parts]
    named = {r.label: r for r in reading.parts}
    if len(named) == len(reading.parts) == len(parts) and set(named) == set(labels):
        return [named[label] for label in labels]
    if len(reading.parts) == len(parts):
        return list(reading.parts)
    raise Misread(f"a reading reads each of its {len(parts)} parts once, not {len(reading.parts)}")


def graded(reading: Reading, reference: PrivateAnswer, grounded: bool) -> tuple[AnswerGrade, Outcome]:
    """A reading graded by code, as an AnswerGrade (a success correct, a fallback abstained) and its outcome."""
    parts = golds(reference)
    readings = matched(reading, parts)
    outcomes = [part_outcome(r, g) for r, g in zip(readings, parts)]
    outcome = request_outcome(outcomes)

    def aligned(r: PartReading, g: dict) -> dict:
        stated = {c.item for c in r.committed if c.item is not None}
        return {
            "committed": [c.value for c in r.committed],
            "missing": [item for i, item in enumerate(g["items"]) if i not in stated],
            "extra": [c.value for c in r.committed if c.item is None],
        }

    own = [aligned(r, g) for r, g in zip(readings, parts)]
    composite = reference.answer_type == "composite"
    grade = AnswerGrade(
        correct=outcome == "success",
        abstained=outcome == "fallback",
        grounded=grounded,
        unsupported_asides=sum(len(r.contradicts_context) for r in readings),
        committed=[
            (f"{g['label']}: " if composite else "") + v for a, g in zip(own, parts) for v in a["committed"]
        ],
        missing=[
            (f"{g['label']}: " if composite else "") + v for a, g in zip(own, parts) for v in a["missing"]
        ],
        extra=[(f"{g['label']}: " if composite else "") + v for a, g in zip(own, parts) for v in a["extra"]],
        reason=reading.reason,
        parts=[
            PartGrade(label=g["label"], correct=o == "success", abstained=o == "fallback", **a)
            for a, g, o in zip(own, parts, outcomes)
        ]
        if composite
        else [],
    )
    return grade, outcome


READ = (
    "Read a Slack answer against the question and its reference, and report what it commits to, part by part. Do not "
    "grade it: report what it says. The reference has one part per thing the question asks (a single question has "
    "one), each with its answer type, its answer items (numbered from 0) and its context (details the answer rests on "
    "that the question does not ask for). For each part, in order: committed lists each value the response commits to "
    "as its answer to that part (for a set, each member it presents as the set), with item the number of the answer "
    "item it states in any equivalent wording (a time on the asker's clock counts; a wrong weekday beside a right date "
    "still states the date, unless the question asks what day, which day or on what day, which asks for the weekday "
    "too: list such slips in slips), or null when it states none, and contradicts true when it gives, for something "
    "an answer item covers, another value. A flat headline or opening claim commits, even when the body later softens "
    "it with a caveat; an explicit correction of a value replaces it; a headline scoped to what is visible ('not as far "
    "as Slack shows') holds back. A value it settles on, though framed as an inference or with a caveat, is committed. "
    "A value named only in passing, as an aside or a side option, is not committed. hedged: it offers alternatives for "
    "the part without settling on one. holds_back: it says it cannot find, see or confirm the part, or flags its "
    "answer as partial. claims: what it says of the part's state: settled (final, done, decided), unsettled (not final "
    "or not confirmed), not_visible (out of the asker's sight), never_happened (it did not or will not happen, or does "
    "not exist), or none. contradicts_context: the details beside the answer that the context contradicts (false "
    "asides). Then a concise reason."
)
GROUND = (
    " Also decide grounded: whether the observations (what the solver's calls showed: messages in time order, the "
    "people, channels and members it looked up, and a log of its calls) support the claims that make its answer; the "
    "reference is not an observation, and no observations means not grounded."
)


def reader_reference(reference: PrivateAnswer) -> list[dict]:
    """The reference as the reader sees it: each part's label, answer type, numbered answer items and context."""
    return [
        {
            "label": g["label"] or "the answer",
            "answer_type": g["answer_type"],
            "answer": [{"item": i, "value": v} for i, v in enumerate(g["items"])],
            "context": g["context"],
        }
        for g in golds(reference)
    ]


class AnswerReader(vf.Judge[Reading]):
    schema = Reading

    def build_messages(self, **fields):
        grounding = "observations" in fields
        return [
            vf.UserMessage(
                content=READ + (GROUND if grounding else "") + "\n" + json.dumps(fields, ensure_ascii=False)
            )
        ]
