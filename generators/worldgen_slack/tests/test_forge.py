"""The task contract: gold as its actor, the reads it may make, its rank and its trial."""

import asyncio
import json
from types import SimpleNamespace

import pytest
import verifiers.v1 as vf
from worldgen_slack.judge import AnswerJudge
from worldgen_slack.slack import PAGE
from worldgen_slack.world import SHADOWED, SHOWN

from ..evidence import measures
from ..forge import tries as tries_module
from ..forge.config import ForgeSeats
from ..forge.tasks import Part, Task, answer_of, check_task, forge_task, forget
from ..forge.tries import Tries
from ..workspace import Workspace, asked
from .support import contracts_settings, fails, forge_of, refused, world_fixture


def test_tasks(tmp_path):
    """Gold queries run as their actor: what the actor cannot see does not exist for them, nothing but a single
    read is allowed, returned evidence must be readable, and a document is applied on a trial copy."""
    world, at, message = world_fixture(tmp_path / "world.sqlite")

    def answers(actor, sql, **kwargs):
        return [r["answer"] for r in world.gold(actor, sql, **kwargs)["rows"]]

    every = "SELECT text AS answer FROM messages ORDER BY ts_us"
    assert "private: rollback budget approved" not in answers("U2", every)
    assert "private: rollback budget approved" in answers("U1", every)
    assert "deleted rollback note" not in answers("U1", every)
    assert answers("U1", "SELECT text AS answer FROM commitments WHERE status = 'open'") == [], (
        "what was promised is gold's to read"
    )
    assert answers("U2", "SELECT COUNT(*) AS answer FROM members WHERE channel_id = 'G1'") == [0]
    assert answers("U3", "SELECT reply_count AS answer FROM thread_stats") == [1]
    assert answers("U2", "SELECT local(ts_us) AS answer FROM messages WHERE id = 1") == [
        "2026-06-15 10:00 CDT"
    ]
    assert answers("U2", "SELECT local(ts_us, 'UTC') AS answer FROM messages WHERE id = 1") == [
        "2026-06-15 15:00 UTC"
    ]
    assert world.gold("U2", "SELECT value AS answer FROM facts WHERE id = 'f1'")["tables"] == ["facts"]
    assert "messages" in world.gold("U2", every)["tables"]
    assert set(SHOWN) == {*SHADOWED, "users"} and all(
        shown <= {r[1] for r in world.db.execute(f"PRAGMA table_xinfo({t})")} for t, shown in SHOWN.items()
    ), "SHOWN names the workspace's own columns"
    assert world.gold("U2", "SELECT emoji AS answer FROM reactions WHERE created_us > 0")["reads"] == [
        "reactions.created_us",
        "reactions.emoji",
    ], "a gold query's own columns are read off"
    assert len(world.gold("U1", every, max_rows=2)["rows"]) == 3, "one row past the cap shows the cap was hit"
    settings = forge_of(contracts_settings(tmp_path))
    who = "SELECT u.real_name AS answer, m.id AS message_id FROM messages m JOIN users u ON u.id = m.user_id"

    def lookup(sql):
        return Task(
            id="tw",
            category="lookup",
            level=1,
            actor_id="U2",
            question="Who answered the decision?",
            answer_type="text",
            gold_sql=sql,
        )

    with world.renumbered() as again:
        assert check_task(world, lookup(who + " WHERE m.text LIKE 'ack%'"), settings, again) == [
            {"answer": "Owen Sato", "message_id": 2}
        ]
        for pinned in (who + " WHERE m.id = 2", who + " WHERE m.channel_id = 'C1' ORDER BY m.id LIMIT 1"):
            fails(check_task, world, lookup(pinned), settings, again)  # T7: a message found by its id
    assert measures(world, "t1", [], ["facts"])["evidence_pages"] == 1
    try:
        with (
            world.trial() as copy
        ):  # a page of newer top-level messages pushes the evidence to read_channel's second page
            with copy.batch():
                copy.insert(
                    "messages",
                    [
                        dict(
                            channel_id="C1",
                            ts_us=at(1, 11, 7) + 1009 * (i + 1),
                            user_id="U1",
                            text=f"note {i}",
                        )
                        for i in range(PAGE)
                    ],
                )
            assert measures(copy, "t1", [], ["facts"])["evidence_pages"] == 2
            raise LookupError("drop the trial")
    except LookupError:
        pass
    for bad in (
        "SELECT text AS answer FROM main.messages",
        "SELECT * FROM main.members",
        "DELETE FROM facts",
        "SELECT 1; SELECT 2",
        "SELECT * FROM tasks",
        "SELECT * FROM scenes",
        "SELECT docid FROM messages_fts",
        "SELECT 'U9' AS user_id",
        "PRAGMA table_info(tasks)",
        "ATTACH DATABASE ':memory:' AS other",
    ):
        fails(world.gold, "U2", bad)
    # Unscoped, as the world's author reads it: every table, still one read-only statement, on the company clock.
    assert world.select("SELECT COUNT(*) AS n FROM tasks JOIN scenes")["rows"] == [{"n": 1}]
    assert world.select("SELECT local(ts_us) AS t FROM messages WHERE id = 1")["rows"] == [
        {"t": "2026-06-15 15:00 UTC"}
    ]
    for bad in ("DELETE FROM tasks", "SELECT 1; SELECT 2", "PRAGMA table_info(tasks)"):
        fails(world.select, bad)
    world.insert("evidence", [dict(fact_id="f1", message_id=4, role="supporting")])
    world.insert("reactions", [dict(message_id=4, user_id="U1", emoji="eyes", created_us=at(1, 12))])
    world.insert("message_mentions", [dict(message_id=4, user_id="U2")])
    for table in ("reactions", "message_mentions"):
        assert (
            answers("U2", f"SELECT COUNT(*) AS answer FROM {table}")
            == [0]
            != answers("U1", f"SELECT COUNT(*) AS answer FROM {table}")
        )
    fails(world.gold, "U2", "SELECT message_id FROM evidence WHERE fact_id = 'f1'")
    assert len(world.gold("U1", "SELECT message_id FROM evidence WHERE fact_id = 'f1'")["rows"]) == 2
    fails(
        world.gold,
        "U2",
        "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT COUNT(*) FROM c",
        seconds=0.2,
    )
    fails(Workspace(world.path, actor="U1").checks)

    scoped = Workspace(world.path, actor="U2")
    assert scoped.rank("rollback release", [1]) == 1 and scoped.rank("zebra", [1]) is None


async def test_tries_cut_and_kept(monkeypatch):
    """A try cut off (a hung request, at its seat's rollout time) runs once more and says why; a round cut off tops
    its tasks up when it resumes, only the empty places tried and each saved as it lands; a changed task is tried
    afresh; a hung screen grade is cut and asked once more. The forge's tries are held to 15 minutes, its grades to 5."""
    saved = []
    state = SimpleNamespace(solves={}, witness={})
    tries = Tries(SimpleNamespace(solvers=2), SimpleNamespace(state=state, trace=lambda trace: None))
    tries.store.save = lambda: saved.append(json.dumps(state.solves))
    monkeypatch.setattr(tries, "copy", lambda attempt, seat: (None, {}))
    monkeypatch.setattr(
        tries, "solver_task", lambda t, copy, release, hint: SimpleNamespace(data=SimpleNamespace(task_id=t))
    )

    class Solver:
        calls = 0

        async def run(self, task):
            Solver.calls += 1
            if Solver.calls == 1:
                return SimpleNamespace(
                    id="hung",
                    ok=False,
                    info={},
                    task=task,
                    last_reply="",
                    stop_condition=None,
                    errors=[SimpleNamespace(message="agent timeout: rollout exceeded its 900s budget")],
                )
            graded = {"task_id": task.data.task_id, "correct": True, "reason": "r", "response": "an answer"}
            return SimpleNamespace(
                id=f"t{Solver.calls}",
                ok=True,
                info={"evaluation": graded},
                task=task,
                errors=[],
                last_reply="an answer",
                stop_condition=None,
            )

    kept = {"correct": True, "crashed": False}
    state.solves["t1"] = {"key": "k", "results": [kept, None, None], "traces": ["kept", None, None]}
    await tries.tried(SimpleNamespace(solver=Solver()), state.solves, ["t1"], {"t1": "k"}, "a", 3, "solver")
    results = state.solves["t1"]["results"]
    assert Solver.calls == 3 and results[0] is kept, "only the empty places are tried"
    assert [r.get("rerun", "")[:13] for r in results[1:]].count("agent timeout") == 1 and not any(
        r["crashed"] for r in results[1:]
    ), "the hung try ran once more, and says why"
    assert len(saved) == 2 and "null" not in saved[-1], "each try is saved as it lands"
    await tries.tried(SimpleNamespace(solver=Solver()), state.solves, ["t1"], {"t1": "k2"}, "a", 3, "solver")
    assert Solver.calls == 6 and state.solves["t1"]["key"] == "k2", "a changed task is tried afresh"

    hangs = [True, False]

    async def evaluate(self, **kwargs):
        if hangs.pop(0):
            await asyncio.sleep(60)
        return "graded"

    monkeypatch.setattr(tries_module, "GRADE_TIMEOUT", 0.05)
    monkeypatch.setattr(AnswerJudge, "evaluate", evaluate)
    tries.settings = SimpleNamespace(solvers=2, answer_judge=vf.JudgeConfig(model="z-ai/glm-5.3"))
    assert await asyncio.wait_for(tries.grade("q", {}, "r"), 5) == "graded", "a hung grade is asked once more"
    hangs[:] = [True, True]
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(tries.grade("q", {}, "r"), 5)

    forge = ForgeSeats()
    assert (forge.solver.timeout.rollout, forge.solver.timeout.scoring, forge.witness.timeout.rollout) == (
        900,
        300,
        900,
    )


def test_composite(tmp_path):
    """A composite request (2 to 4 parts, each with its own gold) is recorded as its request and its parts, each part a
    task of its own under it, checked as a task is; its parts answer different things, its request gives none of them
    away, a part's fault names the part, and taking it off takes its parts with it."""
    world, at, message = world_fixture(tmp_path / "world.sqlite")
    settings = forge_of(contracts_settings(tmp_path))
    who = "SELECT u.real_name AS answer, m.id AS message_id FROM messages m JOIN users u ON u.id = m.user_id"
    called, answered = who + " WHERE m.text LIKE '%rollback decided%'", who + " WHERE m.text LIKE 'ack%'"

    def request(
        *sqls, question="Catch me up on the 4.2 rollback: who called it, and who answered?", **changes
    ):
        labels = ("who called it", "who answered it", "who else")
        return Task(
            id="tc",
            category="composite",
            level=1,
            actor_id="U2",
            question=question,
            answer_type="composite",
            family="catch-up",
            parts=[
                Part(label=label, category="lookup", answer_type="text", gold_sql=sql)
                for label, sql in zip(labels, sqls)
            ],
            **changes,
        )

    for shape in (
        lambda: request(called),  # one part is no composite
        lambda: request(called, answered, gold_sql="SELECT 1 AS answer"),  # its gold is its parts'
        lambda: request(called, answered)
        .model_copy(update={"category": "lookup"})
        .model_validate(request(called, answered).model_dump() | {"category": "lookup"}),
        lambda: Task(id="t", category="lookup", level=1, actor_id="U2", question="q?", answer_type="text"),
    ):
        fails(shape)
    caller = world.db.execute("SELECT real_name FROM users WHERE id = 'U1'").fetchone()[0]
    with world.trial() as copy:
        forged = forge_task(copy, request(called, answered), settings, ["tc"])
        assert [p["gold"] for p in forged["parts"]] == [
            [{"answer": caller, "message_id": 1}],
            [{"answer": "Owen Sato", "message_id": 2}],
        ] and len(forged["features"]) == 2
    rows = [dict(r) for r in world.db.execute("SELECT * FROM tasks WHERE id LIKE 'tc%' ORDER BY id")]
    assert [(r["id"], r["answer_type"], r["parent"], r["part"], r["label"], r["family"]) for r in rows] == [
        ("tc", "composite", None, None, None, "catch-up"),
        ("tc-p1", "text", "tc", 1, "who called it", None),
        ("tc-p2", "text", "tc", 2, "who answered it", None),
    ] and (rows[0]["gold_sql"], rows[0]["gold_source"]) == ("", "composite")
    assert answer_of(world, "tc") == [f"who called it: {caller}", "who answered it: Owen Sato"]
    assert world.db.execute(f"SELECT COUNT(*) FROM tasks WHERE {asked(world.db)}").fetchone()[0] == 2, (
        "a request is asked once, its parts never"
    )
    with world.trial() as copy:  # else two parts that ask for one thing were kept
        refused("two parts give the same answer", forge_task, copy, request(called, called), settings, ["tc"])
    with world.trial() as copy:  # else a request that tells its own answer was kept
        refused(
            "gives away its answer",
            forge_task,
            copy,
            request(
                called,
                answered,
                question="Did Owen Sato or someone else answer the 4.2 rollback, and who called it?",
            ),
            settings,
            ["tc"],
        )
    with world.trial() as copy:  # a part's fault is the part's: its id names it
        refused("tc-p2:", forge_task, copy, request(called, who + " WHERE m.id = 2"), settings, ["tc"])
    forget(world, "tc")
    assert not world.db.execute("SELECT 1 FROM tasks WHERE id LIKE 'tc%'").fetchone(), (
        "a request goes with its parts"
    )
