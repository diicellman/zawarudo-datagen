"""Grader v2: the model reads what an answer commits to, part by part, and code grades it by the policy (a success,
a fallback or an error); the release's answer carries its context (v13); a solve graded by v2 reads as v1's."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from worldgen_slack.judge import (
    AnswerReader,
    Commit,
    Grading,
    PartReading,
    Reading,
    golds,
    graded,
    part_outcome,
    reader_reference,
    request_outcome,
)
from worldgen_slack.release import PrivateAnswer, PublicTask
from worldgen_slack.taskset import SolverTask

from .support import world_fixture


def read(*commits, hedged=False, back=False, claims="none", context=(), label="p") -> PartReading:
    return PartReading(
        label=label,
        committed=[Commit(value=v, item=i, contradicts=x) for v, i, x in commits],
        hedged=hedged,
        holds_back=back,
        claims=claims,
        contradicts_context=list(context),
    )


def gold(kind: str, items: int = 1, unseen: bool = False) -> dict:
    return {
        "label": "p",
        "answer_type": kind,
        "items": [f"v{i}" for i in range(items)],
        "context": [],
        "unseen": unseen,
    }


@pytest.mark.parametrize(
    "reading, part, outcome, why",
    [
        (read(("v0", 0, False)), gold("text"), "success", "the value stated"),
        (
            read(("v0", 0, False), ("an aside", None, False)),
            gold("text"),
            "success",
            "an aside costs nothing",
        ),
        (read(back=True), gold("text"), "fallback", "holds back, commits nothing"),
        (read(("w", None, False)), gold("text"), "error", "another value"),
        (read(("w", None, False), back=True), gold("text"), "error", "a wrong value, though it holds back"),
        (read(("v0", 0, False)), gold("text", 2), "error", "a text asked in two items, one given as whole"),
        (read(("v0", 0, False), back=True), gold("text", 2), "fallback", "one of two, flagged as partial"),
        (
            read(("v0", 0, False), ("w", None, True), hedged=True),
            gold("text"),
            "error",
            "a hedge fails (decided)",
        ),
        (
            read(("v0", 0, False), context=["the 1.4 dB margin"]),
            gold("text"),
            "success",
            "a false context detail is an aside, counted not failed (decided)",
        ),
        (read(("v0", 0, False), ("v1", 1, False)), gold("set", 2), "success", "every member, nothing beyond"),
        (read(("v0", 0, False), back=True), gold("set", 2), "fallback", "a partial set, flagged as partial"),
        (read(("v0", 0, False)), gold("set", 2), "error", "a partial set presented as whole"),
        (
            read(("v0", 0, False), ("v1", 1, False), ("v9", None, False)),
            gold("set", 2),
            "error",
            "a non-member",
        ),
        (
            read(("v0", 0, False), claims="unsettled"),
            gold("status"),
            "success",
            "the latest value, not final",
        ),
        (read(("v0", 0, False), claims="not_visible"), gold("status"), "success", "the latest visible value"),
        (
            read(("v0", 0, False), claims="settled"),
            gold("status"),
            "error",
            "an unsettled status given as final",
        ),
        (read(("v0", 0, False)), gold("status"), "error", "a status that never says it is unsettled"),
        (read(back=True, claims="not_visible"), gold("status"), "fallback", "cannot tell, says so"),
        (
            read(claims="never_happened"),
            gold("status", unseen=True),
            "error",
            "a flat no about the unseen (decided)",
        ),
        (
            read(back=True, claims="not_visible"),
            gold("refusal", 0, True),
            "success",
            "cannot be told, says so",
        ),
        (read(("w", None, False)), gold("refusal", 0), "error", "answers what cannot be answered"),
        (
            read(back=True, claims="never_happened"),
            gold("refusal", 0),
            "success",
            "it does not exist: a refusal",
        ),
        (
            read(back=True, claims="never_happened"),
            gold("refusal", 0, True),
            "error",
            "unseen, yet called false",
        ),
    ],
)
def test_policy(reading, part, outcome, why):
    assert part_outcome(reading, part) == outcome, why


def test_request_and_grade():
    """A request is a success when every part is, an error when any part is, else a fallback; its grade reads as v1's:
    a success correct, a fallback abstained, each part aligned under its label."""
    assert [
        request_outcome(o) for o in (["success"] * 2, ["success", "fallback"], ["fallback", "error"])
    ] == [
        "success",
        "fallback",
        "error",
    ]
    answer = PrivateAnswer(
        answer_type="composite",
        rows=[],
        messages=[["C1", "1.000001"]],
        users=[],
        parts=[
            PrivateAnswer(
                answer_type="text",
                rows=[{"answer": "Nov 6"}],
                messages=[["C1", "1.000001"]],
                users=[],
                label="when",
                context=["the date before the slip"],
            ),
            PrivateAnswer(
                answer_type="set", rows=[{"answer": "a"}, {"answer": "b"}], messages=[], users=[], label="who"
            ),
        ],
    )
    assert [g["context"] for g in golds(answer)] == [["the date before the slip"], []]
    assert reader_reference(answer)[1]["answer"] == [
        {"item": 0, "value": "a"},
        {"item": 1, "value": "b"},
    ] and ("unseen" not in json.dumps(reader_reference(answer))), (
        "the reader sees the numbered items and the context, not what code alone decides on"
    )
    reading = Reading(
        parts=[read(("Fri Nov 6", 0, False), label="when"), read(("a", 0, False), back=True, label="who")],
        reason="r",
    )
    grade, outcome = graded(reading, answer, grounded=True)
    assert (outcome, grade.correct, grade.abstained) == ("fallback", False, True)
    assert [(p.label, p.correct, p.abstained) for p in grade.parts] == [
        ("when", True, False),
        ("who", False, True),
    ]
    assert grade.missing == ["who: b"] and grade.committed == ["when: Fri Nov 6", "who: a"]
    aside = Reading(
        parts=[read(("Nov 6", 0, False), context=["Oct 22"], label="when"), reading.parts[1]], reason="r"
    )
    assert graded(aside, answer, grounded=True)[0].unsupported_asides == 1, "a false aside is counted"
    with pytest.raises(ValueError, match="reads each of its 2 parts"):
        graded(Reading(parts=[reading.parts[0]], reason="r"), answer, grounded=True)


def test_release_context():
    """A part's context is its own (v13): a composite's is its parts', and the reference shows it."""
    part = PrivateAnswer(answer_type="text", rows=[{"answer": "x"}], messages=[], users=[], context=["why"])
    assert part.reference() == {"answer_type": "text", "rows": [{"answer": "x"}], "context": ["why"]}
    assert PrivateAnswer.model_validate_json(part.model_dump_json()) == part
    with pytest.raises(ValueError, match="context is its parts'"):
        PrivateAnswer(
            answer_type="composite",
            rows=[],
            messages=[],
            users=[],
            context=["no"],
            parts=[part.model_copy(update={"label": "a"}), part.model_copy(update={"label": "b"})],
        )


async def test_solve_graded_by_v2(tmp_path):
    """A solve graded by v2: the reader is given no observations when grounding is code's, its reading graded by code,
    and the evaluation carries the outcome beside v1's fields."""
    world, _, _ = world_fixture(tmp_path / "world.sqlite")
    world.solver_copy(tmp_path / "solver.sqlite")
    ts = world.db.execute("SELECT ts FROM messages WHERE id = 1").fetchone()[0]
    answer = PrivateAnswer(answer_type="text", rows=[{"answer": "rollback"}], messages=[["C1", ts]], users=[])
    task = PublicTask(
        task_id="t1",
        question="What did we decide?",
        actor_id="U2",
        category="lookup",
        level=1,
        answer_type="text",
        world_hash="h",
    )
    solve = SolverTask.create(
        task,
        tmp_path / "solver.sqlite",
        answer,
        network_policy=False,
        grading=Grading(version=2, ground=False),
    )
    asked = {}

    async def reading(self, **fields):
        asked.update(fields)
        return SimpleNamespace(parsed=Reading(parts=[read(("the rollback", 0, False))], reason="r"))

    shown = [{"tool": "read_channel", "arguments": {}, "output": {"items": [{"channel": "C1", "ts": ts}]}}]
    trace = SimpleNamespace(
        info={"observations": shown}, last_reply="We rolled back.", id="try", record_metrics=lambda m: None
    )
    evaluate, AnswerReader.evaluate = AnswerReader.evaluate, reading
    try:
        assert await SolverTask.semantic_correctness(solve, trace) == 1.0
    finally:
        AnswerReader.evaluate = evaluate
    result = trace.info["evaluation"]
    assert "observations" not in asked and asked["reference"][0]["answer"] == [
        {"item": 0, "value": "rollback"}
    ]
    assert (result["outcome"], result["correct"], result["grounded"]) == ("success", True, True), (
        "grounded by code: its answer rests on a message the try was shown"
    )
    asks = AnswerReader(solve.config.judge).build_messages(question="q", reference=[], response="r")
    assert "Do not grade it" in asks[0].content and "decide grounded" not in asks[0].content
    grounding = AnswerReader(solve.config.judge).build_messages(
        question="q", reference=[], response="r", observations=[]
    )
    assert "decide grounded" in grounding[0].content
    await asyncio.sleep(0)
