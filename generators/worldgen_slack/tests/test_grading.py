"""The answer judge: the model reports what an answer commits to and the policy's narrow criteria, and code decides each
part's outcome (a success, a fallback or an error); the reader is shown the gold, and what is out of the asker's sight;
a try is graded on its own trace, through the judge's own call, only the network faked."""

import json

import pytest
import verifiers.v1 as vf
from worldgen_slack.judge import (
    READINGS,
    REPLY,
    AnswerJudge,
    Check,
    Gives,
    Misread,
    PartReading,
    Reading,
    fenced,
    gold,
    graded,
    part_outcome,
    replied,
    request_outcome,
)
from worldgen_slack.release import PrivateAnswer, PublicTask
from worldgen_slack.taskset import SolverTask, out_of_sight

from ..prices import priced
from .flows import FakeInference, answer_fields, live_judging
from .support import world_fixture


def check(verdict: bool) -> Check:
    return Check(reason="r", verdict=verdict)


def read(
    *gives,
    conflicts=False,
    hedged=False,
    held=False,
    extra=None,
    declines=None,
    asides=(),
    part="p",
    commits="c",
) -> PartReading:
    return PartReading(
        part=part,
        commits=commits,
        gives=[Gives(item=n, reason="r", verdict=v) for n, v in enumerate(gives, 1)],
        conflicts=check(conflicts),
        extra=None if extra is None else check(extra),
        hedged=check(hedged),
        holds_back=check(held),
        declines=None if declines is None else check(declines),
        false_asides=list(asides),
    )


def part(kind: str, items: int = 1, label: str = "p") -> PrivateAnswer:
    rows = [{"answer": f"v{n}"} for n in range(items)] if kind != "refusal" else []
    return PrivateAnswer(answer_type=kind, rows=rows, messages=[], users=[], label=label)


INVENTED = pytest.mark.xfail(
    strict=True,
    reason="18d: an invented quoted message fails the request; its check lands after the core's calibration",
)


@pytest.mark.parametrize(
    "reading, kind, items, outcome, why",
    [
        (read(True), "text", 1, "success", "gives the item (anchors 9, 10, 18, 18b, 20; i099 S4 derived)"),
        (read(True, asides=["Oct 9"]), "text", 1, "success", "a false aside is counted, not failed (16)"),
        (read(False, held=True), "text", 1, "fallback", "holds back without the item (3)"),
        (read(True, conflicts=True), "text", 1, "error", "another value, weekday or source (17, 18c)"),
        (
            read(False, conflicts=True, held=True),
            "text",
            1,
            "error",
            "a denial of the item, though held (i010)",
        ),
        (read(True, False), "text", 2, "error", "one of two items, given as whole"),
        (read(True, False, held=True), "text", 2, "fallback", "one of two, flagged as partial"),
        (read(True, hedged=True), "text", 1, "error", "a hedge (19)"),
        (read(False), "text", 1, "error", "cut off: no item and no holding back (i076 S7)"),
        (read(True), "text", 1, "success", "a flat 'never happened' gives 'nothing says it did' (i075 S5)"),
        (read(True, True, extra=False), "set", 2, "success", "every member, nothing beyond"),
        (read(True, False, extra=False, held=True), "set", 2, "fallback", "a partial set, flagged (12)"),
        (read(True, False, extra=False), "set", 2, "error", "a partial set presented as whole (13)"),
        (read(True, True, extra=True), "set", 2, "error", "a non-member (14)"),
        (read(True, extra=True), "text", 1, "success", "extra beside a set decides nothing (i092 S8)"),
        (
            read(True),
            "status",
            1,
            "success",
            "scoped, or flat and borne out by what is out of sight (1, 2; h139 S2)",
        ),
        (
            read(True, conflicts=True),
            "status",
            1,
            "error",
            "a claim what is out of sight contradicts (h145 S7)",
        ),
        (read(False, held=True), "status", 1, "fallback", "can't tell, without the latest word (3)"),
        (read(declines=True), "refusal", 0, "success", "declines: it isn't there, or not found (5, 5b, 7)"),
        (
            read(declines=False, held=True),
            "refusal",
            0,
            "fallback",
            "nearby facts flagged as inferred (h147 S3)",
        ),
        (
            read(declines=False, conflicts=True),
            "refusal",
            0,
            "error",
            "settles on an answer anyway (6; h147 S2)",
        ),
        (
            read(declines=True, conflicts=True),
            "refusal",
            0,
            "error",
            "'never happened', contradicted out of sight (8)",
        ),
        (read(declines=False), "refusal", 0, "error", "neither declines nor holds back"),
        pytest.param(
            read(True), "text", 1, "error", "a right answer on an invented quote (18d)", marks=INVENTED
        ),
    ],
)
def test_policy(reading, kind, items, outcome, why):
    assert part_outcome(reading, part(kind, items)) == outcome, why


def test_request_and_grade():
    """A request is a success when every part is, an error when any part is, else a fallback; its grade reads as an
    AnswerGrade (a success correct, a fallback abstained), each part aligned under its label. A reading must read
    every part and item, and a set's or a refusal's own criterion, or it misreads and is asked again."""
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
    reading = Reading(
        parts=[
            read(True, part="when", commits="Fri Nov 6"),
            read(True, False, extra=False, held=True, part="who", commits="a"),
        ]
    )
    grade, outcome = graded(reading, answer, grounded=True)
    assert (outcome, grade.correct, grade.abstained) == ("fallback", False, True)
    assert [(p.label, p.correct, p.abstained) for p in grade.parts] == [
        ("when", True, False),
        ("who", False, True),
    ]
    assert grade.missing == ["who: b"] and grade.committed == ["when: Fri Nov 6", "who: a"]
    assert grade.reason.startswith("when: success") and "who: fallback" in grade.reason
    aside = Reading(parts=[read(True, asides=["Oct 22"], part="when"), reading.parts[1]])
    assert graded(aside, answer, grounded=True)[0].unsupported_asides == 1, "a false aside is counted"
    assert graded(Reading(parts=reading.parts[::-1]), answer, grounded=True) == (grade, outcome), (
        "in any order"
    )
    renamed = [p.model_copy(update={"part": f"Part {n}"}) for n, p in enumerate(reading.parts, 1)]
    assert graded(Reading(parts=renamed), answer, grounded=True)[1] == outcome, "renamed: by place"
    for misread, why in (
        ([reading.parts[0]], "reads each of its 2 parts once, not 1"),
        ([*reading.parts, reading.parts[0]], "not 3"),
        (
            [reading.parts[0], read(True, extra=False, part="who")],
            "who: gives reads each of its 2 items once",
        ),
        ([reading.parts[0], read(True, True, part="who")], "who: its set criterion is unread"),
    ):
        with pytest.raises(Misread, match=why):
            graded(Reading(parts=misread), answer, grounded=True)
    with pytest.raises(Misread, match="its refusal criterion is unread"):
        graded(
            Reading(parts=[read(held=True, part="the answer")]),
            part("refusal").model_copy(update={"label": None}),
            True,
        )
    # A part with no items reads none it is held to: a gives the reader adds to a refusal decides nothing (h147 S4).
    refusal = part("refusal").model_copy(update={"label": None})
    assert (
        graded(Reading(parts=[read(True, declines=True, part="the answer")]), refusal, True)[1] == "success"
    )


def test_release_context():
    """A part's context is its own (v13): a composite's is its parts', and the reference shows it."""
    single = PrivateAnswer(answer_type="text", rows=[{"answer": "x"}], messages=[], users=[], context=["why"])
    assert single.reference() == {"answer_type": "text", "rows": [{"answer": "x"}], "context": ["why"]}
    assert PrivateAnswer.model_validate_json(single.model_dump_json()) == single
    with pytest.raises(ValueError, match="context is its parts'"):
        PrivateAnswer(
            answer_type="composite",
            rows=[],
            messages=[],
            users=[],
            context=["no"],
            parts=[single.model_copy(update={"label": "a"}), single.model_copy(update={"label": "b"})],
        )


def test_gold_as_read(tmp_path):
    """The reader is shown each part's numbered items and context, and, where its truth is out of the asker's sight,
    what those messages say, on the asker's clock: never their ids. The answer is fenced as data."""
    world, _, _ = world_fixture(tmp_path / "world.sqlite")
    world.solver_copy(tmp_path / "solver.sqlite")
    hidden = world.db.execute("SELECT channel_id, ts FROM messages WHERE id = 4").fetchone()
    shown = world.db.execute("SELECT channel_id, ts FROM messages WHERE id = 1").fetchone()
    answer = PrivateAnswer(
        answer_type="composite",
        rows=[],
        messages=[list(shown)],
        unseen=[list(hidden)],
        users=[],
        parts=[
            PrivateAnswer(
                answer_type="text",
                rows=[{"answer": "rollback"}],
                messages=[list(shown)],
                users=[],
                label="what",
            ),
            PrivateAnswer(
                answer_type="status",
                rows=[{"answer": "nothing visible says it is approved"}],
                messages=[],
                unseen=[list(hidden)],
                users=[],
                label="budget",
                context=["asked on Monday"],
            ),
        ],
    )
    seen = out_of_sight(str(tmp_path / "solver.sqlite"), "U2", answer)
    assert seen == [
        [],
        [
            {
                "from": "Alicia Rao",
                "where": "#exec",
                "when": "Mon Jun 15, 11:00",
                "text": "private: rollback budget approved",
            }
        ],
    ]
    shown_gold = json.loads(gold(answer, seen))
    assert shown_gold == [
        {"part": "what", "answer_type": "text", "items": [{"item": 1, "answer": "rollback"}], "context": []},
        {
            "part": "budget",
            "answer_type": "status",
            "items": [{"item": 1, "answer": "nothing visible says it is approved"}],
            "context": ["asked on Monday"],
            "out_of_sight": seen[1],
        },
    ]
    assert hidden[1] not in gold(answer, seen), "the reader is never given the message's id"
    assert fenced("plain") == "```\nplain\n```" and fenced("a ```b``` c").startswith("````\n"), (
        "it can't close the fence"
    )


def rollback(tmp_path):
    """A one-part task ("What did we decide?": rollback), what its try was shown, and the reader's right reading."""
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
    solve = SolverTask.create(task, tmp_path / "solver.sqlite", answer, network_policy=False)
    shown = [{"tool": "read_channel", "arguments": {}, "output": {"items": [{"channel": "C1", "ts": ts}]}}]
    return solve, shown, Reading(parts=[read(True, part="the answer", commits="We rolled back.")])


async def test_try_graded_on_a_real_trace(tmp_path):
    """A try's grade as the forge and the eval take it: on the harness's own trace, through the judge's own call, only
    the network faked. The reader is asked the request, the gold and the fenced answer; the grade lands with its
    reading, grounded by code; each paid call (a misread's second reading too, and a reply that does not read) is
    recorded on the trace, where the forge prices its spend. A reader that misreads every time fails the grade after READINGS readings; an empty answer is an
    error with no call."""
    solve, shown, right = rollback(tmp_path)
    asked = []

    def reply(fields):
        asked.append(fields)
        return right

    trace = vf.Trace.model_construct(root_reply="We rolled back.", info={"observations": shown})
    with live_judging(FakeInference(reply)) as network:
        assert await SolverTask.semantic_correctness(solve, trace) == 1.0
    assert asked == [
        {
            "request": "What did we decide?",
            "reference": [
                {
                    "part": "the answer",
                    "answer_type": "text",
                    "items": [{"item": 1, "answer": "rollback"}],
                    "context": [],
                }
            ],
            "response": "We rolled back.",
        }
    ]
    evaluation = trace.info["evaluation"]
    assert (evaluation["outcome"], evaluation["correct"], evaluation["grounded"], evaluation["readings"]) == (
        "success",
        True,
        True,
        1,
    ) and evaluation["reading"] == right.model_dump(mode="json"), (
        "graded by code, grounded by code, its reading kept"
    )
    assert (network.calls, len(trace.info["judge_calls"]), len(trace.extra_usage)) == (1, 1, 1)
    assert priced(trace.extra_usage[0].model_dump(mode="json"), "z-ai/glm-5.3") == pytest.approx(0.01)
    assert json.loads(json.dumps(trace.info)), "the trace saves"
    # A misread (one part too many) is asked again: two paid calls, both recorded; one that misreads every time fails.
    readings = iter([right.model_copy(update={"parts": right.parts * 2}), right])
    trace = vf.Trace.model_construct(root_reply="We rolled back.", info={"observations": shown})
    with live_judging(FakeInference(lambda fields: next(readings))) as network:
        assert await SolverTask.semantic_correctness(solve, trace) == 1.0
    assert (network.calls, len(trace.info["judge_calls"]), trace.info["evaluation"]["readings"]) == (2, 2, 2)
    trace = vf.Trace.model_construct(root_reply="We rolled back.", info={"observations": shown})
    with live_judging(
        FakeInference(lambda fields: right.model_copy(update={"parts": right.parts * 2}))
    ) as network:
        with pytest.raises(Misread):
            await SolverTask.semantic_correctness(solve, trace)
    assert network.calls == len(trace.info["judge_calls"]) == READINGS, (
        "asked READINGS times, each paid and kept"
    )
    # A reply that does not read as a reading (cut short) is the reader's slip, asked again, and each call is recorded
    # as billed.
    unparsed = []

    def unparseable_first(fields):
        if not unparsed:
            unparsed.append(1)
            return '{"parts": ['
        return right

    trace = vf.Trace.model_construct(root_reply="We rolled back.", info={"observations": shown})
    with live_judging(FakeInference(unparseable_first)) as network:
        assert await SolverTask.semantic_correctness(solve, trace) == 1.0
    assert (network.calls, len(trace.info["judge_calls"]), len(trace.extra_usage)) == (2, 2, 2)
    assert trace.info["evaluation"]["readings"] == 2
    # Grounded by message: a right answer whose try never saw the gold's message is right but ungrounded (0).
    elsewhere = [
        {"tool": "read_channel", "arguments": {}, "output": {"items": [{"channel": "C9", "ts": "1.000001"}]}}
    ]
    trace = vf.Trace.model_construct(root_reply="We rolled back.", info={"observations": elsewhere})
    with live_judging(FakeInference(lambda fields: right)):
        assert await SolverTask.semantic_correctness(solve, trace) == 0.0
    assert trace.info["evaluation"]["correct"] and not trace.info["evaluation"]["grounded"]
    trace = vf.Trace.model_construct(root_reply="  ", info={"observations": shown})
    with live_judging(FakeInference(lambda fields: right)) as network:
        assert await SolverTask.semantic_correctness(solve, trace) == -1.0
    assert network.calls == 0 and trace.info["evaluation"]["outcome"] == "error", "an empty answer: no call"


def test_prompt_names_every_criterion():
    """The prompt, the schema and code say the same thing: every criterion the reading reports is defined in the
    prompt under its own name, and the prompt fills the three fields the grade passes."""
    for name in PartReading.model_fields:
        if name != "part":
            assert f"**{name}" in AnswerJudge.prompt, name
    assert all(f"{{{field}}}" in AnswerJudge.prompt for field in ("request", "reference", "response"))


def test_prompt_fields():
    """The flows' provider reads the prompt as the reader does: the fenced answer comes back whole."""
    solve_prompt = (
        "x\n\nThe request:\nq\n\nThe gold, by part:\n[]\n\nThe answer, between the fences:\n"
        + fenced("a\n```b```")
    )
    assert answer_fields(solve_prompt) == {"request": "q", "reference": [], "response": "a\n```b```"}
    assert answer_fields(solve_prompt + REPLY)["response"] == "a\n```b```", (
        "the reply it asks for comes after"
    )


def test_reply_read():
    """A reading comes as plain JSON, which code reads (structured output is overbilled on GLM 5.3): the reply's first
    JSON object, prose around it aside and a raw control character in a string read as written (GLM wrote one in 3 of
    700 structured replies). A criterion a part doesn't ask is null, not an object with no verdict (the one of probe
    C's 40 plain replies that failed), and a reply with no object fails. The prompt asks for it last, in the reading's
    own schema."""
    reading = Reading(parts=[read(True, part="the answer", commits="We rolled back.")])
    text = reading.model_dump_json()
    assert replied(f"The reading:\n{text}\nThat is all.", Reading) == reading
    assert (
        replied(text.replace("rolled back", "rolled\nback"), Reading).parts[0].commits == "We rolled\nback."
    )
    unasked = json.loads(text)
    unasked["parts"][0]["declines"] = {"reason": "it can be answered", "verdict": None}
    with pytest.raises(ValueError, match="verdict"):
        replied(json.dumps(unasked), Reading)
    with pytest.raises(ValueError, match="no JSON object"):
        replied("I can't grade this.", Reading)
    prompt = AnswerJudge().build_messages(request="q", reference="[]", response="a")
    assert prompt.endswith(REPLY) and json.dumps(Reading.model_json_schema()) in REPLY
