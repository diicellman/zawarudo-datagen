"""The judge: its verdict, its tools as the reviewed tasks' actors, and its review."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from worldgen_slack.judge import EMPTY, AnswerGrade, AnswerJudge, PartGrade
from worldgen_slack.release import PrivateAnswer, PublicTask, Solution, load_release
from worldgen_slack.slack import compact, people_shown, shown
from worldgen_slack.taskset import SolverTask
from worldgen_slack.world import file_hash

from ..forge.release import write_release
from ..judge.task import PHASE_GUIDES, JudgeTask, review_payload
from ..judge.tools import ReviewState
from ..judge.verdict import (
    PHASE_CRITERIA,
    UNREALISTIC,
    Issue,
    TaskReview,
    Verdict,
    accepted_task,
    read_one_way,
    validate_verdict,
    verdict_schema,
)
from ..workspace import Workspace
from .support import contracts_settings, fails, served_tools, with_state, world_fixture


async def test_reviews(tmp_path):
    """Reviews bind to what they reviewed, decide on blocking issues per task, and inspect the world as its tasks'
    actors; solvers get the solver copy and no answer; a release holds exactly what was approved."""
    world, at, message = world_fixture(tmp_path / "world.sqlite")
    rows = world.gold("U2", "SELECT value AS answer FROM facts WHERE id = 'f1'")["rows"]
    world.db.execute("UPDATE tasks SET gold_json = ? WHERE id = 't1'", (json.dumps(rows),))
    payload = review_payload(world, "task", ["t1"])
    taxonomy = contracts_settings(tmp_path).taxonomy
    assert review_payload(world, "task", ["t1"], taxonomy)["tasks"][0]["means"].startswith(
        taxonomy["lookup"].definition
    )
    assert payload["tasks"][0]["gold"] == [{"answer": "rollback"}] and payload["world_hash"] == file_hash(
        world.path
    )
    schema = verdict_schema("task", ["t1"])
    assert schema["properties"]["tasks"]["maxItems"] == 1 and schema["$defs"]["TaskReview"]["properties"][
        "task_id"
    ]["enum"] == ["t1"]
    assert verdict_schema("world", [])["properties"]["tasks"]["maxItems"] == 0

    def verdict(approved=True, issues=(), tasks=(("t1", True),), criteria=None):
        return Verdict(
            approved=approved,
            tasks=[
                TaskReview(task_id=t, valid=v, reason="r", level_fit=3, unrealistic=None) for t, v in tasks
            ],
            issues=list(issues),
            criteria=criteria if criteria is not None else dict.fromkeys(PHASE_CRITERIA["task"], 1.0),
            summary="s",
        )

    validate_verdict(verdict(), payload)
    fails(validate_verdict, verdict(criteria={"question_fit": 1.0}), payload)
    fails(validate_verdict, verdict(tasks=()), payload)
    unscored = verdict().model_copy(update={"tasks": [TaskReview(task_id="t1", valid=True, reason="r")]})
    fails(validate_verdict, unscored, payload)  # a task review scores each task's level_fit
    assert {"level_fit", "unrealistic"} <= set(
        verdict_schema("task", ["t1"])["$defs"]["TaskReview"]["required"]
    )
    unruled = verdict().model_copy(
        update={"tasks": [TaskReview(task_id="t1", valid=True, reason="r", level_fit=3)]}
    )
    fails(
        validate_verdict, unruled, payload
    )  # a task review rules on each task's realism, null when it is real
    # A request no colleague would make is not valid, however hard; code blocks it with the judge's reason.
    fails(TaskReview, task_id="t1", valid=True, reason="r", unrealistic="audits who stayed quiet")
    audit = TaskReview(
        task_id="t1", valid=False, reason="r", level_fit=3, unrealistic="audits who stayed quiet"
    )
    blocked = read_one_way(
        Verdict.model_construct(approved=False, tasks=[audit], issues=[], criteria={}, summary="s")
    )
    assert [i.defect for i in blocked.issues] == [
        UNREALISTIC + "audits who stayed quiet"
    ] and not accepted_task(blocked, "t1")
    assert (
        "Realism, on each task's review" in PHASE_GUIDES["task"]
        and "Ask what its asker would do with the answer" in PHASE_GUIDES["task"]
        and "treats colleagues' activity as data" in PHASE_GUIDES["task"]
    ), "the reviewer is told what a colleague asks, in the user's terms"
    workspace = Issue(artifact="workspace", defect="d", requested_change="c")
    fails(
        validate_verdict, verdict(approved=False, issues=[workspace]), payload
    )  # a blocking one names its tasks
    validate_verdict(verdict(issues=[workspace.model_copy(update={"blocking": False})]), payload)
    assert "workspace" in str(schema["$defs"]["Issue"]["properties"]["artifact"])
    issue = Issue(artifact="tasks", task_ids=["t2"], defect="d", requested_change="c")
    fails(verdict, issues=[issue])
    fails(verdict, approved=False)
    batch = verdict(approved=False, issues=[issue], tasks=(("t1", True), ("t2", True)))
    assert accepted_task(batch, "t1") and not accepted_task(batch, "t2")
    general = batch.model_copy(update={"issues": [issue.model_copy(update={"task_ids": []})]})
    assert accepted_task(general, "t1"), "an issue naming no task blocks none: it is the world's"
    minor = batch.model_copy(update={"issues": [issue.model_copy(update={"blocking": False})]})
    assert accepted_task(minor, "t2"), "a minor issue blocks nothing"

    world.snapshot(tmp_path / "review.sqlite")
    judge = JudgeTask.create(payload, tmp_path / "review.sqlite", "final-01", 5)
    assert judge.config.tools.actors == ["U2"] and "rollback" not in judge.config.tools.model_dump_json()
    (tools,) = judge.toolsets(judge.config)
    await tools.setup()
    stored = with_state(tools)
    result = await tools._with_state(tools.check)()
    assert (
        stored["state"].checked
        and result["tasks"]["t1"]["rows"] == [{"answer": "rollback"}]
        and result["violations"] == []
    )
    rank = Workspace(tmp_path / "review.sqlite", actor="U2").rank(payload["tasks"][0]["question"], [1])
    assert result["tasks"]["t1"]["bm25_rank"] == rank == 2 and "U1" in result["style"]["authors"]
    assert (
        await tools._with_state(tools.read)(
            actor_id="U2", tool="read_channel", arguments={"channel_id": "C1"}
        )
    )["items"]
    # else a reviewer read as someone outside the review
    with pytest.raises(ValueError):
        await tools._with_state(tools.read)(
            actor_id="U1", tool="read_channel", arguments={"channel_id": "G1"}
        )
    assert (
        await tools._with_state(tools.sql)(
            actor_id="U2", sql="SELECT COUNT(*) AS n FROM messages WHERE channel_id = 'G1'"
        )
    )["rows"] == [{"n": 0}]
    assert set(await served_tools(tools)) == {"check", "read", "sql"} and tools.server_name == "inspect"
    files = {"/task/verdict.json": verdict().model_dump_json().encode()}
    runtime = SimpleNamespace(read=lambda path, max_bytes: asyncio.sleep(0, files[path]))
    unchecked = SimpleNamespace(state=ReviewState(), info={}, record_metric=lambda *a: None)
    # else a verdict without inspect_check was accepted
    with pytest.raises(ValueError):
        await judge.finalize(unchecked, runtime)
    checked = SimpleNamespace(state=ReviewState(checked=True), info={}, record_metric=lambda *a: None)
    await judge.finalize(checked, runtime)
    assert checked.info["verdict"]["approved"]

    world.snapshot(tmp_path / "planted.sqlite")
    planted = Workspace(tmp_path / "planted.sqlite", writable=True)
    with planted.batch():  # a fact t1 needs that no message states yet
        fact = dict(id="f9", storyline="r", subject="c", attribute="d", value="v", summary="s", day=2)
        planted.insert("facts", [fact | dict(channel_id="C1", author_id="U1")])
        planted.insert("task_facts", [dict(task_id="t1", fact_id="f9")])

    async def inspected(phase, task_ids):
        judge = JudgeTask.create(review_payload(planted, phase, task_ids), planted.path, phase, 5)
        (tools,) = judge.toolsets(judge.config)
        await tools.setup()
        with_state(tools)
        return judge, await tools._with_state(tools.check)()

    _, due = await inspected("task", ["t1"])
    _, untasked = await inspected("world", [])
    assert [v.split(":")[0] for v in due["violations"]] == ["unstated_fact"] and "bm25_rank" in due["tasks"][
        "t1"
    ]
    assert untasked["violations"] == [], "completion is due for reviewed tasks"

    world.solver_copy(tmp_path / "solver.sqlite")
    task = PublicTask(
        task_id="t1",
        question="q?",
        actor_id="U2",
        category="search",
        level=1,
        answer_type="text",
        world_hash=file_hash(tmp_path / "solver.sqlite"),
    )
    answer = PrivateAnswer(
        answer_type="text",
        rows=rows,
        gold_sql="SELECT 1",
        messages=[["C1", world.db.execute("SELECT ts FROM messages WHERE id = 1").fetchone()[0]]],
        users=["U1"],
    )
    solve = SolverTask.create(task, tmp_path / "solver.sqlite", answer, network_policy=False)
    assert solve.data.system_prompt.startswith("You are using Slack as the user with user_id U2. It is "), (
        "the client says who and when"
    )
    assert "rollback" not in solve.data.model_dump_json() + solve.config.tools.model_dump_json()
    assert solve.config.tools.db_hash == file_hash(tmp_path / "solver.sqlite")
    # The answer judge grades the answer the response commits to and grounds the claims that make it; an unsupported
    # aside is counted, and costs nothing.
    asks = AnswerJudge(solve.config.judge).build_messages(
        question="q", reference={}, response="r", observations=[]
    )
    assert (
        "lists alternatives without settling on one is wrong" in asks[0].content
        and "do not change grounded" in asks[0].content
        and "a value named only in passing, as an aside or a side option, is not committed" in asks[0].content
    ), "a row named in passing does not complete a set"
    assert "never ask for a decoy to be\ncorrected or hedged" in PHASE_GUIDES["world"], (
        "the world judge knows a decoy is planned"
    )
    assert "unsupported_asides" in AnswerGrade.model_json_schema()["required"]
    observed = [
        {
            "tool": "read_channel",
            "arguments": {},
            "output": {"items": [{"channel": "C1", "ts": answer.messages[0][1]}]},
        }
    ]
    assert (
        "A status reference is the latest value the asker can see" in asks[0].content
        and "has abstained" in asks[0].content
    )
    # The reward: +1 right and grounded, 0 right but ungrounded or declined, -1 wrong (an aside costs nothing).
    for correct, abstained, grounded, asides, score in (
        (True, False, True, 2, 1.0),
        (True, False, False, 0, 0.0),
        (False, True, False, 0, 0.0),
        (False, False, False, 0, -1.0),
        (True, True, True, 0, 1.0),
        (False, False, True, 0, -1.0),
    ):
        grade = AnswerGrade(
            correct=correct,
            abstained=abstained,
            grounded=grounded,
            unsupported_asides=asides,
            committed=["x"],
            missing=[],
            extra=[],
            reason="r",
        )
        trace = SimpleNamespace(
            info={"observations": observed}, last_reply="rollback", id="try", record_metrics=lambda m: None
        )
        judged = AnswerJudge.evaluate
        AnswerJudge.evaluate = lambda self, **fields: asyncio.sleep(0, SimpleNamespace(parsed=grade))
        try:
            assert await SolverTask.semantic_correctness(solve, trace) == score, (
                correct,
                abstained,
                grounded,
            )
        finally:
            AnswerJudge.evaluate = judged
        assert (
            trace.info["evaluation"]["unsupported_asides"] == asides
            and trace.info["evaluation"]["evidence_coverage"] == 0.5
        )
    # Grounded by message: tries never shown the gold's messages are ungrounded, whatever the judge says.
    elsewhere = [
        {"tool": "read_channel", "arguments": {}, "output": {"items": [{"channel": "C9", "ts": "1.000001"}]}}
    ]
    grade = AnswerGrade(
        correct=True,
        abstained=False,
        grounded=True,
        unsupported_asides=0,
        committed=["x"],
        missing=[],
        extra=[],
        reason="r",
    )
    trace = SimpleNamespace(
        info={"observations": elsewhere}, last_reply="rollback", id="try", record_metrics=lambda m: None
    )
    judged, AnswerJudge.evaluate = (
        AnswerJudge.evaluate,
        lambda self, **fields: asyncio.sleep(0, SimpleNamespace(parsed=grade)),
    )
    try:
        assert (
            await SolverTask.semantic_correctness(solve, trace) == 0.0
            and not trace.info["evaluation"]["grounded"]
        )
    finally:
        AnswerJudge.evaluate = judged
    # The grader reads what the calls showed, each thing once, and aligns the answer it commits to with the reference;
    # an empty response is wrong by rule, with no grader call.
    assert "Then align: committed lists the answer the response commits to" in asks[0].content
    gold_ts = answer.messages[0][1]
    calls = [
        {
            "tool": "search_public",
            "arguments": {"keywords": ["nothing"]},
            "output": {"items": [], "next_cursor": None},
        },
        {
            "tool": "read_channel",
            "arguments": {},
            "output": {"items": [{"channel": "C1", "ts": gold_ts, "user": "U1", "text": "a"}]},
        },
        {
            "tool": "search_public",
            "arguments": {"keywords": ["a"]},
            "output": {
                "items": [{"channel": "C1", "ts": gold_ts, "user": "U1", "text": "a", "channel_name": "c"}]
            },
        },
        {
            "tool": "get_reactions",
            "arguments": {},
            "output": {
                "channel": "C1",
                "ts": gold_ts,
                "reactions": [{"name": "eyes", "count": 1, "users": ["U3"]}],
            },
        },
        {
            "tool": "list_channel_members",
            "arguments": {"channel_id": "C1"},
            "output": {"items": ["U4", "U2"], "next_cursor": None},
        },
    ]
    view = compact(calls)
    assert [(m["ts"], m.get("channel_name")) for m in view["messages"]] == [(gold_ts, "c")], (
        "a message once, merged"
    )
    assert view["messages"][0]["reactions"][0]["users"] == ["U3"] and view["members"] == {"C1": ["U2", "U4"]}
    assert [c["results"] for c in view["calls"]] == [0, 1, 1, 1, 2], "an empty search stays in the log"
    assert set().union(*map(people_shown, calls)) == {"U1", "U2", "U3", "U4"}, (
        "authors, reactors, members are seen"
    )
    around = {
        "tool": "search_public_and_private",
        "arguments": {},
        "output": {
            "items": [
                {
                    "channel": "C1",
                    "ts": "5.000001",
                    "user": "U1",
                    "text": "x",
                    "context": {
                        "before": [{"channel": "C1", "ts": "4.000001", "user": "U5", "text": "y"}],
                        "after": [],
                    },
                }
            ]
        },
    }
    assert shown(around) == {("C1", "5.000001"), ("C1", "4.000001")} and people_shown(around) == {
        "U1",
        "U5",
    }, "a hit's context is shown too"
    passed = {}

    async def graded(self, **fields):
        passed.update(fields)
        return SimpleNamespace(
            parsed=AnswerGrade(
                correct=True,
                abstained=False,
                grounded=True,
                unsupported_asides=0,
                committed=["x"],
                missing=[],
                extra=[],
                reason="r",
            )
        )

    trace = SimpleNamespace(
        info={"observations": calls}, last_reply="rollback", id="try", record_metrics=lambda m: None
    )
    judged, AnswerJudge.evaluate = AnswerJudge.evaluate, graded
    try:
        assert await SolverTask.semantic_correctness(solve, trace) == 1.0
        assert passed["observations"] == view and trace.info["evaluation"]["evidence_coverage"] == 1.0, (
            "U1 is seen as the gold message's author"
        )
        passed.clear()
        trace = SimpleNamespace(
            info={"observations": calls}, last_reply="  ", id="try", record_metrics=lambda m: None
        )
        assert await SolverTask.semantic_correctness(solve, trace) == -1.0 and not passed, "no grader call"
        empty = trace.info["evaluation"]
        assert empty["reason"] == EMPTY and empty["missing"] == [str(r["answer"]) for r in answer.rows]
    finally:
        AnswerJudge.evaluate = judged
    write_release(tmp_path / "release", tmp_path / "solver.sqlite", [task], {"t1": answer})
    assert load_release(tmp_path / "release")[1:] == ([task], {"t1": answer})
    manifest = json.loads((tmp_path / "release" / "manifest.json").read_text())
    assert manifest["format"] == "worldgen-slack.v11"
    (tmp_path / "release" / "manifest.json").write_text(
        json.dumps(manifest | {"format": "worldgen-slack.v10"})
    )
    fails(load_release, tmp_path / "release")  # a release of another format
    (tmp_path / "release" / "manifest.json").write_text(json.dumps(manifest))
    write_release(tmp_path / "release", tmp_path / "solver.sqlite", [task], {"t1": answer})  # idempotent
    fails(
        write_release,
        tmp_path / "release",
        tmp_path / "solver.sqlite",
        [task],
        {"t1": answer.model_copy(update={"rows": []})},
    )
    tampered = json.loads((tmp_path / "release" / "answers.json").read_text())
    tampered["t1"]["rows"] = [{"answer": "something else"}]
    (tmp_path / "release" / "answers.json").write_text(json.dumps(tampered))
    fails(load_release, tmp_path / "release")
    hidden = answer.model_copy(
        update={"messages": [["G1", world.db.execute("SELECT ts FROM messages WHERE id = 4").fetchone()[0]]]}
    )
    fails(write_release, tmp_path / "release2", tmp_path / "solver.sqlite", [task], {"t1": hidden})
    # A refusal's truth out of its actor's sight ships apart, as unseen; evidence its actor can read may not.
    out_of_sight = answer.model_copy(update={"messages": [], "unseen": hidden.messages})
    write_release(tmp_path / "release3", tmp_path / "solver.sqlite", [task], {"t1": out_of_sight})
    assert load_release(tmp_path / "release3")[2]["t1"].unseen == hidden.messages
    fails(
        write_release,
        tmp_path / "release4",
        tmp_path / "solver.sqlite",
        [task],
        {"t1": answer.model_copy(update={"unseen": answer.messages})},
    )


async def test_composite_grade(tmp_path, monkeypatch):
    """A composite request's answer is its parts', each under its label; the answer judge grades each part, and code
    settles the whole, all or nothing: right only when every part is, declined when no part is wrong and one was
    declined, grounded only when each part was shown what it rests on. A release carries it whole (v11)."""
    world, at, message = world_fixture(tmp_path / "world.sqlite")
    world.solver_copy(tmp_path / "solver.sqlite")
    ts = dict(world.db.execute("SELECT id, ts FROM messages").fetchall())

    def part(label, answer, mid, kind="text"):
        return PrivateAnswer(
            answer_type=kind,
            rows=[{"answer": answer}],
            gold_sql="SELECT 1",
            messages=[["C1", ts[mid]]],
            users=[],
            label=label,
        )

    parts = [part("who called it", "Alicia Rao", 1), part("who posted the notes", "Owen Sato", 3)]
    answer = PrivateAnswer(
        answer_type="composite",
        rows=[],
        gold_sql="",
        messages=[["C1", ts[1]], ["C1", ts[3]]],
        users=[],
        parts=parts,
    )
    assert (
        answer.reference()
        == {
            "answer_type": "composite",
            "parts": [
                {"answer_type": "text", "rows": [{"answer": "Alicia Rao"}], "label": "who called it"},
                {"answer_type": "text", "rows": [{"answer": "Owen Sato"}], "label": "who posted the notes"},
            ],
        }
        and answer.answers() == parts
    )
    whole = answer.model_dump()
    solved = {
        "seat": "witness",
        "model": "z-ai/glm-5.3",
        "trace_id": "w-1",
        "answer": "Alicia Rao called it; Owen Sato posted the notes.",
        "calls": [{"tool": "read_channel", "arguments": {"channel_id": "C1"}}],
    }
    for broken in (
        whole
        | {
            "parts": whole["parts"][:1],
            "messages": whole["parts"][0]["messages"],
        },  # one part is no composite
        whole | {"rows": [{"answer": "x"}]},  # its rows are its parts'
        whole | {"gold_sql": "SELECT 1"},
        whole | {"messages": whole["messages"][:1]},  # its evidence is its parts' together
        whole | {"parts": [whole["parts"][0], whole["parts"][0]]},  # each part under a label of its own
        whole | {"parts": [whole["parts"][0], whole["parts"][1] | {"label": None}]},
        whole["parts"][0] | {"parts": whole["parts"]},  # a single answer has no parts
        whole["parts"][0] | {"gold_sql": ""},
        whole
        | {
            "parts": [whole["parts"][0] | {"solution": solved}, whole["parts"][1]]
        },  # the request's, not a part's
    ):
        fails(PrivateAnswer.model_validate, broken)
    answer = answer.model_copy(update={"solution": Solution.model_validate(solved)})
    task = PublicTask(
        task_id="t1",
        question="Catch me up on 4.2: who called the rollback, and who posted the notes?",
        actor_id="U2",
        category="composite",
        level=2,
        answer_type="composite",
        world_hash=file_hash(tmp_path / "solver.sqlite"),
        family="catch-up",
    )
    solve = SolverTask.create(task, tmp_path / "solver.sqlite", answer, network_policy=False)
    asks = AnswerJudge(solve.config.judge).build_messages(
        question="q", reference={}, response="r", observations=[]
    )
    assert "grade each part on its own by the rules above" in asks[0].content

    def read(*ids):
        return [
            {"tool": "read_channel", "arguments": {}, "output": {"items": [{"channel": "C1", "ts": ts[i]}]}}
            for i in ids
        ]

    def grade(*parts, correct=True, grounded=True):
        return AnswerGrade(
            correct=correct,  # the grader's word on the whole, which code does not read
            abstained=False,
            grounded=grounded,
            unsupported_asides=0,
            committed=[],
            missing=[],
            extra=[],
            reason="r",
            parts=[
                PartGrade(
                    label=f"part {n}",
                    correct=right,
                    abstained=declined,
                    committed=[] if declined else ["x"],
                    missing=[] if right else ["y"],
                    extra=[],
                )
                for n, (right, declined) in enumerate(parts, 1)
            ],
        )

    async def scored(graded, observations, reply="Alicia Rao called it; Owen Sato posted the notes."):
        async def complete(self, messages, **kwargs):
            return SimpleNamespace(parsed=graded)

        monkeypatch.setattr(AnswerJudge, "complete", complete)
        trace = SimpleNamespace(
            info={"observations": observations}, last_reply=reply, id="try", record_metrics=lambda m: None
        )
        return await SolverTask.semantic_correctness(solve, trace), trace.info["evaluation"]

    score, evaluation = await scored(grade((True, False), (True, False)), read(1, 3))
    assert score == 1.0 and [p["label"] for p in evaluation["parts"]] == [
        "who called it",
        "who posted the notes",
    ], "every part right and grounded; the labels are the reference's"
    score, evaluation = await scored(grade((True, False), (False, False), correct=True), read(1, 3))
    assert (
        score == -1.0 and not evaluation["correct"] and evaluation["missing"] == ["who posted the notes: y"]
    ), "one part wrong is the whole wrong, whatever the grader says of the whole"
    score, evaluation = await scored(grade((True, False), (False, True), correct=False), read(1, 3))
    assert score == 0.0 and evaluation["abstained"], "what it could not tell, declined, costs nothing"
    score, evaluation = await scored(grade((False, False), (False, True), correct=False), read(1, 3))
    assert score == -1.0 and not evaluation["abstained"], "a wrong part is not undone by a declined one"
    score, evaluation = await scored(grade((True, False), (True, False)), read(1))
    assert score == 0.0 and not evaluation["grounded"], "each part grounded by its own messages"
    assert evaluation["part_coverage"] == [1.0, 0.0] and evaluation["evidence_coverage"] == 0.5
    with pytest.raises(ValueError, match="each of its 2 parts"):
        await scored(grade((True, False)), read(1, 3))
    score, evaluation = await scored(grade(), read(1, 3), reply=" ")
    assert score == -1.0 and [p["missing"] for p in evaluation["parts"]] == [["Alicia Rao"], ["Owen Sato"]], (
        "an empty response lacks every part's rows"
    )
    write_release(tmp_path / "release", tmp_path / "solver.sqlite", [task], {"t1": answer})
    assert load_release(tmp_path / "release")[1:] == ([task], {"t1": answer})
    assert load_release(tmp_path / "release")[2]["t1"].solution.calls == solved["calls"], (
        "it ships how it was solved"
    )
    assert json.loads((tmp_path / "release" / "manifest.json").read_text())["format"] == "worldgen-slack.v11"
    fails(
        write_release,
        tmp_path / "release2",
        tmp_path / "solver.sqlite",
        [task.model_copy(update={"answer_type": "text"})],
        {"t1": answer},
    )  # its question and its answer say what it is alike
