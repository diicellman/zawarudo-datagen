"""The answer judge: a solver's answer graded against the release's reference, on what its calls showed."""

import json

import verifiers.v1 as vf
from pydantic import Field

from .release import NonEmptyText, StrictModel


class AnswerGrade(StrictModel):
    """The committed answer against the reference (correct), whether the response declined to give one (abstained),
    whether the observations support the claims that make its answer (grounded), and how many of the response's other
    material claims they don't (counted, not scored). The alignment says what it committed to, which of the
    reference's rows it lacks, and which values it has that the reference does not: code classes a miss from it."""

    correct: bool
    abstained: bool
    grounded: bool
    unsupported_asides: int = Field(ge=0)
    committed: list[str]
    missing: list[str]
    extra: list[str]
    reason: NonEmptyText


EMPTY = "The response is empty: it gives no answer and does not say the request cannot be answered."


def empty_grade(reference: dict) -> AnswerGrade:
    """An empty response is wrong by rule, with no grader call: it lacks every reference row."""
    rows = [str(r["answer"]) for r in reference.get("rows", [])]
    return AnswerGrade(
        correct=False,
        abstained=False,
        grounded=False,
        unsupported_asides=0,
        committed=[],
        missing=rows,
        extra=[],
        reason=EMPTY,
    )


class AnswerJudge(vf.Judge[AnswerGrade]):
    schema = AnswerGrade

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
                    "reason.\n" + json.dumps(fields, ensure_ascii=False)
                )
            )
        ]


def reward(grade: AnswerGrade) -> float:
    """+1 for a right answer the observations ground, 0 for a right one they don't or for declining to answer, -1 for
    a wrong one: under a 1-or-0 grade a guess always beats saying it cannot be told (TruthRL, Kalai et al.), and a
    solver gave a provisional figure as final in 38 of 65 misses."""
    if grade.correct:
        return 1.0 if grade.grounded else 0.0
    return 0.0 if grade.abstained else -1.0
