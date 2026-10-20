"""The forge's parts, each on its own: a task's contract and code's checks, its private answer, the tries (kept as they
land, a split decided by a third), the judge's hearing, the spend, the labels and the release rows."""

import json
from types import SimpleNamespace

import pytest
from worldgen_slack.judge import Grading

from ..forge import tries as tries_module
from ..forge.env import ForgeEnv, label_of, on_crashes, provenance, reopen, spent
from ..forge.judge import Ruling, hearing, rule
from ..forge.material import material
from ..forge.release import level, release_rows, report
from ..forge.tasks import Part, Task, answer_text, check, key, private_answer
from ..forge.tries import Tries, agreeing, by_part, patient, rates, reached
from ..prices import priced
from .support import contracts_settings, forge_of, refused, world_fixture


def part(label="who called it", answer=("Alicia Rao",), evidence=(1,), kind="text", unseen=()):
    return Part(
        label=label, answer_type=kind, answer=list(answer), evidence=list(evidence), unseen=list(unseen)
    )


def request(*parts, id="t9", actor="U2", question="Catch me up on the 4.2 release: who called the rollback?"):
    return Task(id=id, family="catch-up", actor_id=actor, question=question, parts=list(parts) or [part()])


def test_material(tmp_path):
    """The proposer's map: a collection's items with their current values and messages; each value's page depth in
    its channel; every promise with how it went (one past due with nothing said has slipped); and the world's material
    counted, by storyline."""
    world, at, _ = world_fixture(tmp_path / "w.sqlite")
    items = (
        (1, "C1", 1, "U1", "blocked"),
        (2, "G1", 4, "U1", "budget"),
        (3, "D1", 5, "U2", "check"),
        (4, "C1", 2, "U2", "ack"),
    )
    with world.batch():
        world.insert(
            "facts",
            [
                dict(
                    id=f"b{n}",
                    storyline="r",
                    subject=f"blocker {n}",
                    attribute="state",
                    value="open",
                    anchor=token,
                    channel_id=channel,
                    author_id=author,
                    day=1,
                    summary="s",
                    collection="4.2 blockers",
                )
                for n, channel, _, author, token in items
            ],
        )
        world.insert(
            "evidence",
            [
                dict(fact_id=f"b{n}", message_id=m, role="anchor", anchor_token=token)
                for n, _, m, _, token in items
            ],
        )
        world.insert(
            "commitments",
            [dict(id="c1", owner_id="U2", text="send the notes", message_id=3, due_us=at(3, 0), slips=1)],
        )
    found = material(world)
    (blockers,) = found["collections"]
    assert (blockers["collection"], blockers["storyline"], len(blockers["items"])) == ("4.2 blockers", "r", 4)
    assert {i["item"]: (i["current"], i["messages"]) for i in blockers["items"]}["blocker 4"] == (
        ["open"],
        ["m2 (reply to m1)"],
    ), "a thread reply names its parent"
    assert all(v["page"] == 1 for g in found["groups"] for v in g["values"]), "one page deep, every channel"
    assert found["promises"] == [
        {
            "promise": "c1",
            "owner": "Owen Sato (U2)",
            "text": "send the notes",
            "made": "m3",
            "where": "#releases",
            "due_day": 2,
            "outcome": "slipped",
        }
    ]
    assert found["summary"]["facts"] == 5 and found["summary"]["promises"]["slipped"] == 1, found["summary"]


def test_offers():
    """The ids still open, those offered least first; one offered `offers` times unwritten is dropped, short of
    material, and stays dropped."""
    events = []
    env = ForgeEnv.__new__(ForgeEnv)
    env.settings = SimpleNamespace(offers=2)
    env.store = SimpleNamespace(
        state=SimpleNamespace(
            plans={
                "forge": {
                    "planned": dict.fromkeys(("i001", "i002", "i003"), {}),
                    "offered": {"i001": 2, "i002": 1},
                }
            },
            forged={"i003": {}},
        ),
        event=lambda name, **fields: events.append((name, fields)),
        save=lambda: None,
    )
    assert env.open_ids() == ["i002", "i001"], "the least offered first; a written id is not open"
    env.drop_stuck()
    assert (
        env.open_ids() == ["i002"]
        and env.plan["dropped"] == ["i001"]
        and events == [("ids_dropped", {"ids": ["i001"], "offered": 2})]
    )


def test_contract(tmp_path):
    """A task is a request of 1 to 4 parts; code refuses, with every reason, what it can tell is wrong (its asker, its
    messages out of their sight, an answer of the wrong shape, two parts alike, a question that gives an answer away, a
    repeat); its private answer is each part's rows and the messages they rest on."""
    world, at, message = world_fixture(tmp_path / "world.sqlite")
    single = request()
    check(world, single, [])
    assert answer_text(single) == ["Alicia Rao"] and key(single) == key(single.model_copy())
    with pytest.raises(ValueError):
        Task.model_validate(single.model_dump() | {"parts": []})
    with pytest.raises(ValueError):
        Task.model_validate(single.model_dump() | {"variant_of": "t1"})  # a version names why too
    world.db.execute("UPDATE users SET is_bot = 1 WHERE id = 'U3'")
    for wrong, why in (
        (request(actor="U3"), "no person in the workspace"),
        (request(part(evidence=[4])), "its asker cannot read message 4"),  # #exec is out of U2's sight
        (request(part(evidence=[6])), "message 6 is no message"),  # deleted
        (request(part(evidence=[])), "name the messages its answer rests on"),
        (request(part(answer=["Alicia Rao", "Owen Sato"])), "a text answer has 1-1 values"),
        (request(part(kind="set")), "a set answer has 2-50 values"),
        (request(part(kind="status")), "a status names the message out of the asker's sight"),
        (request(part(kind="status", unseen=[3])), "its asker can read message 3"),
        (request(part(), part(label="who called it ")), "each part asks for something of its own"),
        (request(question="Did Alicia Rao call the rollback?"), "gives away its answer"),
    ):
        refused(why, check, world, wrong, [])
    refused("its asker already asks the same", check, world, single, [single.model_copy(update={"id": "t8"})])
    check(world, request(part(kind="status", unseen=[4])), [])  # the settling message, out of sight
    check(world, request(part(kind="refusal", answer=[], evidence=[])), [])
    composite = request(
        part(),
        part(label="who answered", answer=["Owen Sato"], evidence=[2]),
        question="Catch me up on the 4.2 rollback: who called it, and who answered?",
    )
    check(world, composite, [])
    # The skill an id aims at, by its shape as far as code can tell: aggregation, a set of 3 or more from 2 or more
    # places (a channel, or a thread in one); abstention, a status or a refusal; synthesis, 2 or more parts.
    many = part(label="who took part", answer=["Alicia Rao", "Owen Sato", "Mia Chen"], kind="set")
    for task, skill in (
        (request(many.model_copy(update={"evidence": [1, 3]})), "aggregation"),  # two posts, one channel
        (request(part(kind="set", answer=["Alicia Rao", "Owen Sato"], evidence=[1, 5])), "aggregation"),
        (single, "abstention"),
        (single, "synthesis"),
    ):
        refused(f"it aims at {skill}", check, world, task, [], skill)
    for task, skill in (
        (request(many.model_copy(update={"evidence": [1, 2]})), "aggregation"),  # a post and its thread
        (request(many.model_copy(update={"evidence": [1, 5]})), "aggregation"),  # a channel and a DM
        (request(part(kind="refusal", answer=[], evidence=[])), "abstention"),
        (composite, "synthesis"),
        (single, "temporal"),  # no shape code can tell
    ):
        check(world, task, [], skill)
    answer = private_answer(world, composite)
    ts = dict(world.db.execute("SELECT id, ts FROM messages").fetchall())
    assert answer.answer_type == "composite" and [p.label for p in answer.parts] == [
        "who called it",
        "who answered",
    ]
    assert answer.messages == [["C1", ts[1]], ["C1", ts[2]]] and answer.parts[1].rows == [
        {"answer": "Owen Sato"}
    ]
    assert answer.reference()["parts"][0] == {
        "answer_type": "text",
        "rows": [{"answer": "Alicia Rao"}],
        "label": "who called it",
    }
    assert answer_text(composite) == ["who called it: Alicia Rao", "who answered: Owen Sato"]
    assert private_answer(world, single).rows == [{"answer": "Alicia Rao"}]
    # What a part rests on but does not ask for is its context: carried to the release, not required of an answer.
    dated = single.model_copy(
        update={"parts": [single.parts[0].model_copy(update={"context": ["on Oct 14, in #release"]})]}
    )
    assert private_answer(world, dated).context == ["on Oct 14, in #release"]
    assert private_answer(world, dated).reference()["context"] == ["on Oct 14, in #release"]


async def test_tries(tmp_path):
    """Each try is kept the moment it lands: a run cut off tops its task up, and a changed task is tried afresh; a
    verifier split is decided by a third try; a call that fails is asked again, and its last failure raised."""
    saved = []
    state = SimpleNamespace(solves={}, verifier={})
    store = SimpleNamespace(
        state=state, trace=lambda trace: None, save=lambda: saved.append(1), root=tmp_path
    )
    tries = Tries(SimpleNamespace(solvers=2, verify=2, tries=3), store)
    tries.copy = lambda: None
    tries.solver_task = lambda task, world: SimpleNamespace(data=SimpleNamespace(task_id=task.id))
    # None crashes: the student's second try crashes all 3 of its attempts
    plan = {"verifier": [True, False, True], "solver": [True, None, None, None, False]}
    made = {"verifier": 0, "solver": 0}

    class Seat:
        def __init__(self, seat):
            self.seat = seat

        async def run(self, task):
            made[self.seat] += 1
            right = plan[self.seat][(made[self.seat] - 1) % len(plan[self.seat])]
            evaluation = {"correct": right, "semantic_correctness": 1.0 if right else -1.0, "response": "r"}
            return SimpleNamespace(
                id=f"{self.seat}-{made[self.seat]}",
                ok=right is not None,
                info={"evaluation": evaluation} if right is not None else {},
                errors=[SimpleNamespace(message="stream ended")] if right is None else [],
                last_reply="r",
                stop_condition=None,
                task=task,
            )

    agents = SimpleNamespace(verifier=Seat("verifier"), solver=Seat("solver"))
    task = request()
    verified = await tries.verify(agents, task, "k")
    assert len(verified) == 3 and len(agreeing(verified)) == 2, "a split is decided by a third try"
    stray = {"correct": True, "grounded": False, "semantic_correctness": 0.0}
    assert agreeing([stray, {"correct": True, "crashed": True}]) == [stray], (
        "a try right on every part reaches the gold, though a claim of its own is ungrounded"
    )
    measured = await tries.measure(agents, task, "k")
    assert [r["crashed"] for r in measured] == [False, True, False], (
        "a try that crashes runs again, up to 3 times"
    )
    assert rates(measured)["right_rate"] == 0.5 and rates(measured)["crashed"] == 1
    assert len(saved) == 6, "each try is saved as it lands"
    state.solves["t9"]["results"][2] = None  # a run cut off before its last try
    kept, crashed_trace = state.solves["t9"]["traces"][0], state.solves["t9"]["traces"][1]
    await tries.measure(agents, task, "k")
    after = state.solves["t9"]
    assert (
        after["traces"][0] == kept and after["traces"][1] != crashed_trace and after["results"][2] is not None
    ), "a resume tries the empty places and those that crashed, and keeps the rest"
    before = made["solver"]
    await tries.measure(agents, task, "k2")
    assert made["solver"] > before + 3 and state.solves["t9"]["key"] == "k2", "a changed task is tried afresh"
    composite = request(part(), part(label="who answered", answer=["Owen Sato"], evidence=[2]))
    graded = [{"parts": [{"correct": True}, {"correct": c}]} for c in (True, False)]
    assert (
        by_part(composite, graded) == {"who called it": 1.0, "who answered": 0.5}
        and by_part(task, graded) is None
    )
    calls = []

    async def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("unavailable")
        return "ok"

    # The verifier's tries that crashed (a provider's refusals) reach nothing and refute nothing: short of `verify`
    # answered, the task fails rather than its answer going unreached, and a later try of it tops them up.
    plan["verifier"] = [None]
    state.verifier.clear()
    with pytest.raises(RuntimeError, match="the verifier's tries crashed"):
        await tries.verify(agents, task, "k")
    plan["verifier"] = [True]
    assert [r["correct"] for r in await tries.verify(agents, task, "k")] == [True, True]
    assert await patient(flaky, timeout=5) == "ok" and len(calls) == 3
    with pytest.raises(RuntimeError):
        await patient(lambda: (_ for _ in ()).throw(RuntimeError("down")), timeout=5)
    assert tries_module.WAITS == (0, 0, 0), "the checks wait for no provider"


async def test_verifier_reach(tmp_path):
    """A composite's answer holds when each part is reached by two of the verifier's tries, though no one try reached
    them all: a split on a part is decided by a third. A verifier's try that ends with no answer (its turn cap, on calls
    that lost their arguments) runs again, as a crash does; a student's is its own failure, and counted."""
    state = SimpleNamespace(solves={}, verifier={})
    store = SimpleNamespace(state=state, trace=lambda trace: None, save=lambda: None, root=tmp_path)
    tries = Tries(SimpleNamespace(solvers=2, verify=2, tries=2), store)
    tries.copy = lambda: None
    tries.solver_task = lambda task, world: SimpleNamespace(data=SimpleNamespace(task_id=task.id))

    def graded(*parts):
        return {"correct": all(parts), "response": "r", "parts": [{"correct": c} for c in parts]}

    silent = {"correct": False, "response": "", "parts": [{"correct": False}, {"correct": False}]}

    class Seat:
        def __init__(self, plan):
            self.plan, self.made = list(plan), 0

        async def run(self, task):
            evaluation = self.plan[self.made]
            self.made += 1
            return SimpleNamespace(
                id=f"t{self.made}",
                ok=True,
                info={"evaluation": evaluation},
                errors=[],
                last_reply=evaluation["response"],
                stop_condition="max_turns" if evaluation is silent else None,
                task=task,
            )

    composite = request(part(), part(label="who answered", answer=["Owen Sato"], evidence=[2]))
    verifier = Seat([graded(True, False), silent, graded(False, True), graded(True, True)])
    results = await tries.verify(SimpleNamespace(verifier=verifier), composite, "k")
    assert verifier.made == 4 and len(results) == 3, (
        "a silent try ran again, and a split on a part took a third"
    )
    assert reached(composite, results) == [2, 2] and not agreeing(results[:2]), (
        "each part reached twice, apart"
    )
    assert results[1]["rerun"] == "no answer"
    student = Seat([silent, graded(True, True)])
    measured = await tries.measure(SimpleNamespace(solver=student), composite, "k")
    assert student.made == 2 and measured[0]["unanswered"] and rates(measured)["right_rate"] == 0.5, (
        "a student's silent try is counted wrong, not run again"
    )
    assert reached(request(), [{"correct": True}, None, {"correct": True, "crashed": True}]) == [1]


async def test_judge(tmp_path):
    """The judge reads a task as its asker would and the tries' answers, and rules on four things; all four make it
    approved, and its call's cost is kept."""
    world, at, message = world_fixture(tmp_path / "world.sqlite")
    world.db.execute("INSERT OR REPLACE INTO world_meta (key, value) VALUES ('zone', 'America/Chicago')")
    task = request()
    heard = hearing(
        world,
        task,
        [{"response": "Alicia Rao called it"}],
        [{"correct": False, "response": "Owen", "reason": "x"}],
    )
    assert heard["asker"].startswith("Owen Sato") and heard["parts"][0]["messages"][0]["from"] == "Alicia Rao"
    assert heard["student_wrong_answers"] == [{"answer": "Owen", "grade": "x"}] and heard["verifier_answers"]
    assert heard["parts"][0]["context"] == [], "the judge reads each part's context beside its answer"
    ruling, usage = await rule(forge_of(contracts_settings(tmp_path)).judge, world, task, [], [])
    assert ruling.approved and usage == {}
    precise = request(question="Who precisely called the 4.2 rollback?")
    ruling, _ = await rule(forge_of(contracts_settings(tmp_path)).judge, world, precise, [], [])
    assert not ruling.exact and not ruling.approved, "an answer that asks more than its part"
    assert not Ruling(realistic=False, one_reading=True, exact=True, supported=True, reason="r").approved


def test_provenance_of_later_settings(tmp_path):
    """A setting added after forges started on it is left out of a forge's provenance at its default, so those forges
    resume; set otherwise, it is part of it."""
    config = forge_of(contracts_settings(tmp_path))
    (config.world).mkdir(parents=True, exist_ok=True)
    (config.world / "world.sqlite").write_bytes(b"w")
    assert "grading" not in provenance(config)["config"]
    v2 = config.model_copy(update={"grading": Grading(version=2, ground=False)})
    assert provenance(v2)["config"]["grading"] == {"version": 2, "ground": False}


def test_spend_and_labels(tmp_path):
    """What a forge spent (reported, or else list prices), how a labelled task is labelled, and its release row."""
    (tmp_path / "traces").mkdir()
    (tmp_path / "traces" / "a.json").write_text(
        json.dumps(
            {"agent": {"name": "solver"}, "usage": {"cost": 0.5}, "extra_usage": [{"prompt_tokens": 1e6}]}
        )
    )
    (tmp_path / "traces" / "b.json").write_text(
        json.dumps({"agent": {"name": "verifier"}, "usage": {"prompt_tokens": 1e6, "completion_tokens": 1e6}})
    )
    settings = {
        "env": {"solver": {"model": "deepseek/deepseek-v4.1-flash"}, "verifier": {"model": "z-ai/glm-5.3"}},
        "answer_judge": {"model": "z-ai/glm-5.3"},
        "judge": {"model": "z-ai/glm-5.3"},
    }
    state = SimpleNamespace(forged={"t": {"usage": {"completion_tokens": 1e6}}})
    assert spent(SimpleNamespace(root=tmp_path, settings=settings, state=state)) == pytest.approx(
        0.5 + 1.4 + 5.8 + 4.4
    )
    assert priced({"prompt_tokens": 1e6}, "a model of no list") == 4.0, (
        "an unlisted model at the dearest price"
    )
    assert priced({"prompt_tokens": 0, "cached_input_tokens": 1e6}, "z-ai/glm-5.3") == 1.2, (
        "GLM's cached input as the bill charged it (about $1.20), not as listed ($0.26)"
    )
    assert priced({"cached_input_tokens": 1e6, "cost": 0.5}, "z-ai/glm-5.3") == 0.5, "a reported cost stands"
    measured = {"right_rate": 0.5}
    assert [
        label_of(r, 0.875)
        for r in (
            {"verified": False},
            {"verified": True, "approved": False, "rates": measured},
            {"verified": True, "approved": True, "rates": {"right_rate": 1.0}},
            {"verified": True, "approved": True, "rates": {"right_rate": 0.0}},
            {"verified": True, "approved": True, "rates": measured},
        )
    ] == ["rejected", "rejected", "solved", "unsolved", "window"]
    assert [level(r) for r in (1.0, 0.75, 0.25, 0.0)] == [1, 2, 3, 4]
    # A label that rests on tries that crashed is the provider's, not the task's: it is labelled again.
    lost = {"results": [{"crashed": True}, {"correct": False}]}
    assert [
        on_crashes(r, v)
        for r, v in (
            ({"label": "rejected", "verified": False}, lost),
            ({"label": "rejected", "verified": False}, {"results": [{"correct": False}] * 2}),
            ({"label": "window", "verified": True, "rates": {"crashed": 2}}, {}),
            ({"label": "solved", "verified": True, "rates": {"crashed": 0}}, lost),
            ({"verified": False}, lost),
        )
    ] == [True, False, True, False, False]
    # A finished forge is taken up again by its own command when a task's labelling failed, or its label rests on tries
    # that crashed (an earlier pass's too); else it stays done.
    for forged, again in (
        ({"i001": {"label": "solved", "rates": {"crashed": 0}}}, False),
        ({"i001": {"failed": "RuntimeError: 3 of the student's 8 tries crashed"}}, True),
        ({"i001": {"label": "solved", "verified": True, "rates": {"crashed": 4}}}, True),
    ):
        events = []
        store = SimpleNamespace(
            state=SimpleNamespace(phase="done", forged=forged, verifier={}, plans={}),
            event=lambda kind, **fields: events.append(kind),
            save=lambda: None,
        )
        assert (reopen(store), store.state.phase, events) == (
            (True, "label", ["forge_reopened"]) if again else (False, "done", [])
        ), forged


def test_release(tmp_path):
    """An approved task's row names its set, its rates (by part) and the verifier's; its answer carries the verifier's
    try that reached it; the report counts each family's tasks, and names those short of material."""
    world, at, message = world_fixture(tmp_path / "world.sqlite")
    (tmp_path / "traces").mkdir()
    (tmp_path / "traces" / "v1.json").write_text(
        json.dumps({"info": {"observations": [{"tool": "read_channel", "arguments": {"channel_id": "C1"}}]}})
    )
    task = request()
    right = {
        "correct": True,
        "grounded": True,
        "semantic_correctness": 1.0,
        "response": "Alicia Rao called it",
    }
    state = SimpleNamespace(
        forged={
            "t9": {
                "task": task.model_dump(mode="json"),
                "set": "harder",
                "label": "window",
                "verified": True,
                "skill": "aggregation",
            },
            "t8": {
                "task": task.model_dump(mode="json") | {"id": "t8"},
                "set": "initial",
                "label": "rejected",
            },
        },
        solves={"t9": {"results": [right, {"correct": False}], "traces": [None, None]}},
        verifier={"t9": {"results": [right | {"grounded": False}, right], "traces": ["v0", "v1"]}},
        restore_point="adapt1-02",
        plans={"forge": {"dropped": ["t7"]}},
    )
    for attempt, text in (("adapt1-02", "no owner named for the audit"), ("source", "the author's plan")):
        (tmp_path / "attempts" / attempt / "notes").mkdir(parents=True)
        (tmp_path / "attempts" / attempt / "notes" / "forge.md").write_text(text)
    store = SimpleNamespace(
        world=world, root=tmp_path, state=state, settings={"env": {"verifier": {"model": "z-ai/glm-5.3"}}}
    )
    rows, answers = release_rows(store, "h")
    (row,) = rows
    assert (row.task_id, row.set, row.right_rate, row.level, row.verifier_right) == (
        "t9",
        "harder",
        0.5,
        2,
        1.0,
    )
    assert answers["t9"].solution.trace_id == "v1" and answers["t9"].solution.calls == [
        {"tool": "read_channel", "arguments": {"channel_id": "C1"}}
    ], "the solution is a grounded try, though an ungrounded one came first"
    found = report(store)
    assert found["families"]["catch-up"]["written"] == 2 and found["short"] == sorted(
        ["catch-up", "decisions", "commitments", "status"]
    ), "a family with fewer than 2 approved tasks is short"
    assert found["proposer_notes"] == "no owner named for the audit", (
        "the notes of the proposer's last clean turn"
    )
    assert (found["skills"]["aggregation"], found["skills"]["none"]["written"], found["dropped"]) == (
        {"written": 1, "verified": 1, "approved": 1, "window": 1},
        1,
        ["t7"],
    ), "by skill, and the ids dropped short of material"
    assert (found["askers"], found["openings"], found["material"]["unused_groups"]) == (
        1,
        [["catch me", 2]],
        0,
    ), "its askers, the openings it repeats, and the world's one value group, which its tasks rest on"
    assert ForgeEnv.WHY["repaired"].startswith("The judge refused it")
