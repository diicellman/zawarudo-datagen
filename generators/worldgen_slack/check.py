"""Run: uv run python -m generators.worldgen_slack.check"""

import asyncio
import json
import tempfile
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from verifiers.v1.mcp.launch import serve
from worldgen_slack.slack.models import QUALITY_CRITERIA, AnswerSpec, Conversation, Message, SlackWorld, User
from worldgen_slack.slack.api import SlackAPI, ReadCall, digest
from worldgen_slack.slack.tools import ReadRecord, ReadState, SlackTaskData, stage_tool_data
from worldgen_slack.taskset import (
    SolverTask,
    EvaluationTask,
    EvaluationConfig,
    SlackTaskset,
    SlackTasksetConfig,
)
from worldgen_slack.dataset import atomic_json, load_release
from .agents.judge import JudgeTask, review_payload
from .agents.inspection import missing_evidence
from .config import Config
from .env import GenerationEnv, reference_for
from .store import Store, ReviewLimit
from .contracts import (
    Binding,
    Candidate,
    Catalog,
    ClaimEvidence,
    Fact,
    Issue,
    PlannedTask,
    TaskReview,
    Verdict,
    WorkGroup,
    reveals_answer,
    validate_candidate,
    validate_verdict,
)


def fails(function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except (ValueError, TypeError, LookupError, ReviewLimit):
        return
    raise AssertionError("expected rejection")


def fixture():
    users = [User(id="u1", name="Alicia Rao"), User(id="u2", name="Owen Sato")]
    facts = [
        Fact(
            id=f"f{i}",
            subject=f"work-{i}",
            predicate="approved destination",
            value=f"region-{i}",
            valid_from="2026-01-01T00:00:00Z",
            description=f"Approved region is region-{i}.",
        )
        for i in (1, 2)
    ]
    tasks = [
        PlannedTask(
            id=f"t{i}",
            group_id=f"g{i}",
            actor_id=f"u{i}",
            question=f"Which region was approved for work-{i}?",
            answer=AnswerSpec(
                kind="fact_summary",
                canonical_answer=f"region-{i}",
                required_claims=[f"region-{i}"],
            ),
            fact_ids=[f"f{i}"],
            reasoning="lookup",
        )
        for i in (1, 2)
    ]
    catalog = Catalog(
        workspace_id="ws1",
        sector="IT",
        company="Harbor",
        overview="Two related infrastructure workstreams.",
        people=users,
        groups=[WorkGroup(id=f"g{i}", description=f"work-{i}") for i in (1, 2)],
        facts=facts,
        tasks=tasks,
    )
    start = datetime(2026, 1, 1, tzinfo=UTC)
    messages = [
        Message(
            id=f"m{i}",
            conversation_id="c1",
            author_id="u1",
            text=f"Deployment observation {i}",
            timestamp=(start + timedelta(minutes=i)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
        for i in range(160)
    ]
    messages[0].text = "Approved destination for work-1 is region-1."
    messages += [
        Message(
            id=f"r{i}",
            conversation_id="c1",
            author_id="u1",
            thread_root_id="m0",
            text=f"Rollout check {i}",
            timestamp=(start + timedelta(hours=4, minutes=i)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
        for i in range(70)
    ]
    messages.append(
        Message(
            id="secret",
            conversation_id="c2",
            author_id="u2",
            text="work-2 is approved for region-2.",
            timestamp="2026-01-01T00:00:00Z",
        )
    )
    world = SlackWorld(
        users=users,
        conversations=[
            Conversation(id="c1", name="platform", kind="public_channel", member_ids=["u1", "u2"]),
            Conversation(id="c2", name="restricted", kind="private_channel", member_ids=["u2"]),
        ],
        messages=messages,
    )
    bindings = [
        Binding(
            task_id=f"t{i}",
            claims=[ClaimEvidence(claim_index=0, message_ids=[message])],
            gold_calls=[ReadCall(action="search_messages", arguments={"query": f"work-{i}"})],
        )
        for i, message in ((1, "m0"), (2, "secret"))
    ]
    return catalog, Candidate(snapshot=world, bindings=bindings)


def approval(payload):
    return Verdict(
        reviewed_hash=digest(payload),
        approved=True,
        tasks=[
            TaskReview(
                task_id=t["id"],
                valid=True,
                answer_complete=True,
                supported_claims=list(range(len(t["answer"]["required_claims"]))),
                reason="checked",
            )
            for t in payload["tasks"]
        ],
        issues=[],
        criteria=dict.fromkeys(QUALITY_CRITERIA, 1.0) if payload["phase"] == "world" else {},
        summary="checked",
    )


def collect(api, action, **arguments):
    seen = []
    cursor = None
    while True:
        result = getattr(api, action)(**arguments, cursor=cursor, limit=7)
        seen.extend(result["items"])
        cursor = result["next_cursor"]
        if cursor is None:
            return seen


def check_contracts_and_reads():
    catalog, candidate = fixture()
    assert validate_candidate(catalog, candidate, ["t1", "t2"])["ok"]
    answer = "Jon Bell approves Cedar privileged-access exceptions as of 2026-10-21."
    question = "As of 2026-10-21, who approves Cedar privileged-access exceptions?"
    assert not reveals_answer("cnv_cedar_rollout", answer, question)
    assert not reveals_answer("cnv_access_ops", answer, question)
    assert reveals_answer("usr_jon_bell", answer, question)
    assert reveals_answer("msg_20260915", "2026-09-15", "When does enforcement begin?")
    assert reveals_answer("m_2026_10_14", "2026-10-14", "When is launch?")
    assert not reveals_answer("m_rel_ops_10", "2026-10-14", "When is launch?")
    assert not reveals_answer("m_rel_ops_14", "2026-10-14", "When is launch?")
    activity = validate_candidate(catalog, candidate, ["t1"])["activity"]
    assert activity["replies"] == 70 and activity["threads_by_reply_count"] == {70: 1}
    assert activity["duplicate_message_excess"] == 0
    route = candidate.model_copy(deep=True)
    route.bindings[0].gold_calls[0].arguments["conversation_id"] = "c1"
    report = validate_candidate(catalog, route, ["t1"])
    assert not report["ok"] and "undiscovered conversation_id" in report["errors"][0]
    route.bindings[0].gold_calls.insert(0, ReadCall(action="list_conversations"))
    assert validate_candidate(catalog, route, ["t1"])["ok"]
    route.bindings[0].claims[0].user_ids = ["u1"]
    report = validate_candidate(catalog, route, ["t1"])
    assert not report["ok"] and "user_ids=['u1']" in report["errors"][0]
    route.bindings[0].gold_calls.append(ReadCall(action="get_user", arguments={"user_id": "u1"}))
    assert validate_candidate(catalog, route, ["t1"])["ok"]
    fails(
        Issue,
        owner="synthesizer",
        artifact="bindings",
        defect="Undiscovered ID",
        requested_change="Discover it",
    )
    directory_catalog = catalog.model_dump(mode="json")
    directory_catalog["people"][0]["id"] = "usr_alicia"
    directory_catalog["tasks"][0].update(
        actor_id="usr_alicia",
        question="Who approved work-1?",
        answer={
            "kind": "entity",
            "canonical_answer": "Alicia Rao",
            "required_claims": ["Alicia Rao approved work-1."],
        },
    )
    fails(Catalog.model_validate_json, json.dumps(directory_catalog))
    api = SlackAPI(candidate.snapshot, "u1")
    history = collect(api, "get_conversation_history", conversation_id="c1")
    assert len(history) == 160 and len({m["message_id"] for m in history}) == 160
    assert len(collect(api, "get_thread", conversation_id="c1", root_message_id="m0")) == 71
    assert len(collect(api, "search_messages", query="Deployment")) == 159
    assert {r["conversation_id"] for r in collect(api, "list_conversations")} == {"c1"}
    assert "secret" not in {m["message_id"] for m in api.search_messages("region-2")["items"]}
    fails(api.get_conversation_history, "c2")
    fails(api.get_thread, "c2", "secret")
    fails(api.search_messages, "work", conversation_id="c2")
    result = api.get_conversation_history("c1", limit=1)
    cursor = result["next_cursor"]
    fails(api.search_messages, "work", cursor=cursor)
    fails(
        SlackAPI(candidate.snapshot, "u2").get_conversation_history,
        "c1",
        cursor=cursor,
    )
    changed = candidate.model_copy(deep=True)
    changed.snapshot.messages[-2].text = "A different rollout check"
    fails(
        SlackAPI(changed.snapshot, "u1").get_conversation_history,
        "c1",
        cursor=cursor,
    )
    fails(api.get_conversation_history, "c1", limit=True)
    fails(api.search_messages, "work", before="nonsense")
    assert (
        len(
            api.search_messages(
                "Deployment",
                after="2026-01-01T02:00:00Z",
                before="2026-01-01T02:10:00Z",
            )["items"]
        )
        == 9
    )
    assert not api.search_messages("Deployment", author_id="u2")["items"]
    corrupted = catalog.model_dump(mode="json")
    corrupted["tasks"][1]["question"] = corrupted["tasks"][0]["question"]
    fails(Catalog.model_validate_json, json.dumps(corrupted))
    corrupted = catalog.model_dump(mode="json")
    other = dict(corrupted["facts"][0], id="conflict", value="different")
    corrupted["facts"].append(other)
    fails(Catalog.model_validate_json, json.dumps(corrupted))
    changed = candidate.model_copy(deep=True)
    changed.snapshot.messages = [m for m in changed.snapshot.messages if m.id != "secret"]
    assert not validate_candidate(catalog, changed, ["t1", "t2"])["ok"]
    changed = candidate.model_copy(deep=True)
    changed.bindings[0].gold_calls[0].arguments["limit"] = 1
    assert validate_candidate(catalog, changed, ["t1"])["ok"]
    changed.snapshot.messages.append(
        Message(
            id="later_update",
            conversation_id="c1",
            author_id="u2",
            timestamp="2026-01-02T00:00:00Z",
            text="work-1 monitoring continues; the approved destination is unchanged.",
        )
    )
    regression = validate_candidate(catalog, changed, ["t1"])
    assert not regression["ok"] and any("gold route" in error for error in regression["errors"])
    changed = candidate.model_copy(deep=True)
    changed.bindings[0].gold_calls = [
        ReadCall(
            action="get_conversation_history",
            arguments={"conversation_id": "c1", "limit": 1},
        )
    ]
    assert not validate_candidate(catalog, changed, ["t1"])["ok"]
    payload = review_payload(catalog, candidate, ["t1", "t2"], "world")
    assert set(missing_evidence(payload, [])) == {"t1", "t2"}
    inspected = []
    for task, binding in zip(catalog.tasks, candidate.bindings):
        api = SlackAPI(candidate.snapshot, task.actor_id)
        inspected.extend(
            ReadRecord(actor_id=task.actor_id, call=call, output=api.execute(call))
            for call in binding.gold_calls
        )
    assert not missing_evidence(payload, inspected)
    directory_payload = json.loads(json.dumps(payload))
    directory_payload["candidate"]["bindings"][0]["claims"][0]["user_ids"] = ["u2"]
    assert missing_evidence(directory_payload, inspected)["t1"]["user_ids"] == ["u2"]
    call = ReadCall(action="get_user", arguments={"user_id": "u2"})
    inspected.append(
        ReadRecord(actor_id="u1", call=call, output=SlackAPI(candidate.snapshot, "u1").execute(call))
    )
    assert not missing_evidence(directory_payload, inspected)
    inspected[1].actor_id = "u1"
    assert "t2" in missing_evidence(payload, inspected)
    verdict = approval(payload)
    validate_verdict(verdict, payload)
    incomplete = verdict.model_copy(deep=True)
    incomplete.tasks[0].supported_claims = []
    fails(validate_verdict, incomplete, payload)
    altered = dict(payload, candidate=changed.model_dump(mode="json"))
    fails(validate_verdict, verdict, altered)
    first = SolverTask.create(
        catalog.tasks[0], candidate.snapshot, "ws1", reference_for(catalog.tasks[0], candidate)
    )
    second = SolverTask.create(
        catalog.tasks[1], candidate.snapshot, "ws1", reference_for(catalog.tasks[1], candidate)
    )
    assert first is not second and first.config.tools.actor_id != second.config.tools.actor_id
    assert "answer" not in first.data.model_dump() and "fact_ids" not in first.data.model_dump()
    assert first.config.tools.colocated is False
    assert "snapshot_json" not in first.data.model_dump()
    print("PASS contracts, regression checks, pagination, cursor binding, privacy, stale/incomplete verdicts")


class TestStore(Store):
    def trace(self, trace):
        pass


class ScriptedAgent:
    @asynccontextmanager
    async def provision(self, task):
        assert task.config.files["schemas.json"]
        yield None

    @asynccontextmanager
    async def interaction(self, task, runtime):
        yield SimpleNamespace(trace=SimpleNamespace(ok=True, info={}, record_reward=lambda *args: None))


async def check_flow(root):
    catalog, candidate = fixture()

    class ScriptedEnv(GenerationEnv):
        async def author_turn(self, interaction, runtime, context, attempt, first):
            assert context.get("group_id") != "g2", "already supported groups need review, not regeneration"
            if not first:
                assert context["feedback"]
            return (catalog if context["phase"] == "catalog" else candidate).model_dump_json()

        async def review(self, agents, payload, attempt, author_trace=None):
            verdict = approval(payload)
            if attempt in {"catalog-01", "build-g1-01"}:
                upstream = attempt.startswith("catalog")
                return verdict.model_copy(
                    update={
                        "approved": False,
                        "issues": [
                            Issue(
                                owner="synthesizer" if upstream else "builder",
                                artifact="catalog" if upstream else "workspace",
                                task_ids=["t1"],
                                defect="Duplicated question" if upstream else "Repetitive activity",
                                requested_change="Repair the identified defect",
                            )
                        ],
                    }
                )
            return verdict

        async def evaluate(self, agents):
            assert self.store.state.final_groups == ["g1", "g2"]
            self.store.state.evaluation = {
                t.id: {
                    "task_id": t.id,
                    "semantic_correctness": 0.0,
                    "execution_ok": True,
                    "grounded": False,
                    "read_count": 1,
                }
                for t in catalog.tasks
            }
            self.store.state.phase = "done"
            self.store.save(approved=True)

    config = Config(sector="IT", task_count=2, group_size=1, output=root, seed=1)
    store = TestStore(root, {"test": 1})
    store.state.catalog = catalog
    agents = SimpleNamespace(synthesizer=ScriptedAgent(), builder=ScriptedAgent())
    env = ScriptedEnv(config, store)
    await env.run(None, agents)
    assert store.state.phase == "done"
    atomic_json(
        store.root / "traces/billing-check.json",
        {"agent": {"name": "solver"}, "usage": {"cost": 1.0}, "extra_usage": [{"cost": 0.25}]},
    )
    billing = store.summary("check")
    assert billing["reported_model_cost"] == 1.25
    assert billing["usage_by_role"]["answer_judge"]["judge_calls"] == 1
    assert billing["usage_by_role"]["answer_judge"]["traces"] == 0
    assert store.state.rounds == {"catalog": 2, "build:g1": 2, "build:g2": 1, "final:g1": 1, "final:g2": 1}
    store.publish()
    world, rows, answers = load_release(root / "release")
    assert world == candidate.snapshot and len(rows) == 2
    assert rows[0].snapshot_ref == rows[1].snapshot_ref
    assert "answer" not in rows[0].model_dump()
    taskset = SlackTaskset(SlackTasksetConfig(task=EvaluationConfig(release_dir=root / "release")))
    task = next(iter(taskset))
    rebuilt = EvaluationTask(SlackTaskData.model_validate_json(task.data.model_dump_json()), task.config)
    await rebuilt.validate(None)
    assert rebuilt.solver_task().config.reference == answers[task.data.task_id]
    altered = EvaluationTask(rebuilt.data.model_copy(update={"snapshot_hash": "wrong"}), task.config)
    fails(altered.solver_task)
    store.publish()
    atomic_json(root / "release/tasks.json", [{**rows[0].model_dump(mode="json"), "answer": "leak"}])
    fails(load_release, root / "release")
    saved = store.state.last_approved_candidate
    store.reserve("interrupted", 2)
    store.close()
    resumed = TestStore(root, {"test": 1})
    assert resumed.state.active_attempt is None and resumed.state.rounds["interrupted"] == 1
    assert resumed.state.last_approved_candidate == saved
    resumed.reserve("interrupted", 2)
    fails(resumed.reserve, "interrupted", 2)
    catalog_review = resumed.state.reviews["catalog"]
    resumed.invalidate("world-only regression", catalog=False)
    assert resumed.state.reviews == {"catalog": catalog_review}
    assert resumed.state.built_groups == ["g1", "g2"] and not resumed.state.final_groups
    assert resumed.state.last_approved_candidate is None
    before = dict(resumed.state.rounds)
    verdict = approval(review_payload(catalog, candidate, ["t1", "t2"], "world")).model_copy(
        update={
            "approved": False,
            "issues": [
                Issue(
                    owner="synthesizer",
                    artifact="catalog",
                    task_ids=["t2"],
                    defect="Ambiguous scope",
                    requested_change="Clarify scope",
                )
            ],
        }
    )
    repairing = ScriptedEnv(config, resumed)
    repairing.route_rejection(verdict, "build:g2")
    assert resumed.state.phase == "catalog" and not resumed.state.reviews
    assert resumed.state.built_groups == ["g1"]
    assert resumed.state.rounds == before
    assert resumed.state.last_approved_candidate is None
    catalog.tasks[1].question += " According to the final approval?"
    await repairing.synthesize(agents)
    assert resumed.state.rounds["build:g2"] == before["build:g2"] + 1
    assert resumed.state.phase == "build" and not resumed.state.final_groups
    assert resumed.state.last_approved_candidate is None
    await repairing.run(None, agents)
    assert resumed.state.phase == "done" and resumed.state.final_groups == ["g1", "g2"]
    assert resumed.state.rounds["build:g1"] == before["build:g1"]
    assert resumed.state.rounds["build:g2"] == before["build:g2"] + 2
    assert sum(resumed.state.rounds[f"{stage}:g2"] for stage in ("build", "final")) == 5
    fails(resumed.reserve, "build:g2", 5)
    fails(resumed.reserve, "final:g2", 5)
    resumed.close()
    print(
        "PASS complete feedback flow, upstream invalidation, bounded retries, crash/resume, release and task wire"
    )


async def check_native_servers():
    catalog, candidate = fixture()
    payload = review_payload(catalog, candidate, ["t1", "t2"], "world")
    solver = SolverTask.create(
        catalog.tasks[0], candidate.snapshot, catalog.workspace_id, reference_for(catalog.tasks[0], candidate)
    )
    review = JudgeTask.create(payload)
    for task, data, expected in (
        (solver, candidate.snapshot.model_dump(mode="json"), 5),
        (review, payload, 2),
    ):
        stage_tool_data(task, task.config.tools, data)
        try:
            assert len(task.config.tools.model_dump_json()) < 2000
            toolsets = task.toolsets(task.config)
            assert len(toolsets) == 1
            toolset = toolsets[0]
            await toolset.setup()
            stored = {"state": ReadState()}

            async def pull(stored=stored):
                state = stored["state"].model_copy(deep=True)
                await asyncio.sleep(0)
                return state

            async def push(before, stored=stored, toolset=toolset):
                await asyncio.sleep(0)
                stored["state"] = toolset.state.model_copy(deep=True)

            toolset._pull_state, toolset._push_state = pull, push
            if task is solver:
                call = toolset._with_state(toolset.get_user)
                await asyncio.gather(*(call(user_id="u1") for _ in range(20)))
            else:
                await toolset._with_state(toolset.check)()
                call = toolset._with_state(toolset.read)
                await asyncio.gather(
                    *(
                        call(
                            actor_id="u1",
                            action="get_user",
                            arguments={"user_id": "u1"},
                        )
                        for _ in range(20)
                    )
                )
                assert stored["state"].checked
            assert len(stored["state"].reads) == 20
            async with (
                asyncio.timeout(30),
                serve(toolsets[0]) as url,
                streamable_http_client(url) as (reader, writer),
                ClientSession(reader, writer) as client,
            ):
                await client.initialize()
                assert len((await client.list_tools()).tools) == expected
        finally:
            task._tool_directory.cleanup()
    print("PASS native subprocess tool entrypoints, tool discovery, bounded private-data configuration")


def make_world(seed: int = 0) -> SlackWorld:
    users = [
        User(id="agent_user", name="Agent User"),
        User(id="alice", name="Alice Chen"),
        User(id="bob", name="Bob Diaz"),
        User(id="cara", name="Cara Evans"),
    ]
    conversations = [
        Conversation(
            id="incidents",
            name="incidents",
            kind="public_channel",
            member_ids=[user.id for user in users],
        ),
        Conversation(
            id="private",
            name="leadership",
            kind="private_channel",
            member_ids=["alice", "bob"],
        ),
        Conversation(id="dm", kind="dm", member_ids=["agent_user", "cara"]),
        Conversation(
            id="archived",
            name="old-incidents",
            kind="public_channel",
            member_ids=[user.id for user in users],
            is_archived=True,
        ),
    ]
    rows = [
        ("root", "incidents", "cara", "Payments timeout investigation started.", "10:00:00", None, False),
        (
            "msg_cause",
            "incidents",
            "alice",
            "A stale DNS cache caused the payments timeout.",
            "10:05:00",
            "root",
            False,
        ),
        (
            "msg_fix",
            "incidents",
            "bob",
            "We recycled the edge workers and service recovered.",
            "10:09:00",
            "root",
            False,
        ),
        ("older", "incidents", "alice", "Payments timeout alert opened.", "09:00:00", None, False),
        ("newer", "incidents", "bob", "Payments timeout monitoring is green.", "11:00:00", None, False),
        ("private_msg", "private", "alice", "Confidential roadmap in Cedar.", "09:10:00", None, False),
        ("dm_msg", "dm", "cara", "Direct handoff is complete.", "09:20:00", None, False),
        ("deleted", "incidents", "alice", "Deleted secret answer cache.", "09:30:00", None, True),
        ("archived_msg", "archived", "bob", "Old incident detail.", "09:40:00", None, False),
        (
            "variant",
            "incidents",
            "cara",
            f"Rehearsal group {seed % 7} meets tomorrow.",
            "09:50:00",
            None,
            False,
        ),
    ]
    messages = [
        Message(
            id=message_id,
            conversation_id=conversation_id,
            author_id=author_id,
            text=text,
            timestamp=f"2025-01-15T{time}Z",
            thread_root_id=root,
            deleted=deleted,
        )
        for message_id, conversation_id, author_id, text, time, root, deleted in rows
    ]
    return SlackWorld(users=users, conversations=conversations, messages=messages)


def check_slack_rules():
    world = make_world()
    api = SlackAPI(world, "agent_user")
    assert {r["conversation_id"] for r in api.list_conversations()["items"]} == {"incidents", "dm"}
    assert api.get_conversation_history("dm")["items"][0]["message_id"] == "dm_msg"
    for hidden in ("private", "archived"):
        fails(api.get_conversation_history, hidden)
    assert {
        r["message_id"] for r in api.search_messages("secret answer confidential roadmap")["items"]
    }.isdisjoint({"deleted", "private_msg"})
    assert [r["message_id"] for r in api.get_thread("incidents", "root")["items"]] == [
        "root",
        "msg_cause",
        "msg_fix",
    ]
    assert [r["message_id"] for r in api.get_conversation_history("incidents")["items"]][:2] == [
        "newer",
        "root",
    ]
    for field, value in (("author_id", "missing"), ("timestamp", "2025-01-15T09:59:00Z")):
        changed = world.model_dump(mode="json")
        next(m for m in changed["messages"] if m["id"] == "msg_fix")[field] = value
        fails(SlackWorld.model_validate, changed)
    print("PASS retained Slack visibility, deletion, archive, ordering, membership, and chronology checks")


async def check_live_feedback():
    """Controlled bad candidates exercise the real native author/reviewer exchanges."""
    import verifiers.v1 as vf
    from verifiers.v1.clients import EvalClientConfig, ModelContext
    from .config import load_config
    from .generate import provenance

    config = load_config(Path("configs/generation.toml"))
    config.output = config.output.parent / "feedback-check"
    store = Store(config.output, provenance(config))

    class FeedbackEnv(GenerationEnv):
        async def author_turn(self, interaction, runtime, context, attempt, first):
            raw = await super().author_turn(interaction, runtime, context, attempt, first)
            injected = self.store.root / ("injected-" + context["phase"] + ".json")
            if injected.exists():
                return raw
            try:
                if context["phase"] == "catalog":
                    Catalog.model_validate_json(raw)
                else:
                    candidate = Candidate.model_validate_json(raw)
                    if not validate_candidate(
                        self.store.state.catalog, candidate, context["required_task_ids"]
                    )["ok"]:
                        return raw
            except ValueError:
                return raw
            content = json.loads(raw)
            self.store.artifact(attempt, "before_test_injection", content)
            if context["phase"] == "catalog":
                original, replacement = content["tasks"][:2]
                content["tasks"][1] = dict(
                    original,
                    id=replacement["id"],
                    group_id=replacement["group_id"],
                    question="In this workspace, "
                    + original["question"][:1].lower()
                    + original["question"][1:],
                )
            else:
                world = content["snapshot"]
                channel = next(
                    c for c in world["conversations"] if c["kind"] == "public_channel" and c["member_ids"]
                )
                latest = max(datetime.fromisoformat(m["timestamp"]) for m in world["messages"])
                for i in range(40):
                    world["messages"].append(
                        {
                            "id": f"xcheck{i:04d}",
                            "conversation_id": channel["id"],
                            "author_id": channel["member_ids"][0],
                            "text": "Status update: everything remains on track. No new information.",
                            "timestamp": (latest + timedelta(minutes=5 * (i + 1))).strftime(
                                "%Y-%m-%dT%H:%M:%SZ"
                            ),
                        }
                    )
            raw = json.dumps(content)
            await runtime.write("/task/output.json", raw.encode())
            self.store.artifact(attempt, "author_output", {"text": raw, "controlled_test_injection": True})
            atomic_json(injected, {"attempt": attempt})
            self.store.event("controlled_defect_injected", attempt=attempt)
            return raw

        async def run(self, task, agents):
            if self.store.state.phase == "catalog":
                await self.synthesize(agents)
            group = self.store.state.catalog.groups[0]
            if group.id not in self.store.state.built_groups:
                await self.build_group(agents, group)
            assert self.store.state.rounds["catalog"] >= 2
            assert self.store.state.rounds["build:" + group.id] >= 2
            assert group.id in self.store.state.built_groups
            for phase in ("catalog", "world"):
                name = json.loads((self.store.root / ("injected-" + phase + ".json")).read_text())["attempt"]
                verdict = json.loads((self.store.root / "attempts" / name / "verdict.json").read_text())
                assert not verdict["approved"]
                assert any(
                    issue["owner"] == ("synthesizer" if phase == "catalog" else "builder")
                    for issue in verdict["issues"]
                )
            self.store.event("live_feedback_verified")

    atomic_json(
        store.root / "fixture.json",
        {
            "kind": "controlled feedback test; not a released dataset",
            "defects": ["paraphrased duplicate QA", "40 repeated five-minute status posts"],
        },
    )
    try:
        env = FeedbackEnv(config, store)
        context = ModelContext(
            model=config.env.solver.model, client=EvalClientConfig(), sampling=vf.Sampling()
        )
        async with env.serving():
            episode = await env.run_episode(
                vf.Task(vf.TaskData(prompt="Check both generation feedback loops.")), context
            )
        for trace in episode.traces:
            store.trace(trace)
        atomic_json(store.root / f"episode-{episode.id}.json", episode.to_record())
        store.summary("feedback_check_complete" if episode.ok else "feedback_check_failed")
        assert episode.ok, [e.message for e in episode.errors]
        print("PASS real catalog and builder rejection → feedback → revision → independent approval")
    finally:
        store.close()


async def check_ablation(root):
    from .ablate_judge import AblationEnv, Case, Study, author_context, has_feedback
    from .agents.judge import REVIEW_GUIDE
    from .config import PipelineConfig

    catalog, candidate = fixture()
    payload = review_payload(catalog, candidate, [t.id for t in catalog.tasks], "world")
    files = {}

    async def write(path, value):
        files[path] = value

    for override in (None, "Experimental guide"):
        task = JudgeTask.create(payload, review_guide=override)
        await task.setup(None, SimpleNamespace(write=write))
        assert (
            files["/task/review.md"]
            == (
                (REVIEW_GUIDE if override is None else override) + "\nreviewed_hash: " + digest(payload)
            ).encode()
        )
        task._tool_directory.cleanup()

    cases = [Case(id=f"case{i}", source=Path("unused"), phase="world", repair=i < 3) for i in range(6)]
    study = Study(output=root, guide_addendum=Path("unused"), seed=1, cases=cases)
    revised = approval(payload).model_copy(
        update={"summary": "Nonblocking improvements: explain the follow-up."}
    )
    assert not has_feedback(approval(payload)) and has_feedback(revised)
    assert not has_feedback(revised.model_copy(update={"summary": "Nonblocking improvements: none."}))
    one, two = author_context(payload, revised), author_context(payload, revised)
    assert json.loads(one["feedback"])["summary"] == revised.summary
    one["previous_output"]["snapshot"]["messages"].clear()
    assert two["previous_output"]["snapshot"]["messages"] and payload["candidate"]["snapshot"]["messages"]
    turns = []

    class Author:
        @asynccontextmanager
        async def provision(self, task):
            context = json.loads(task.config.files["input.json"])
            assert json.loads(context["feedback"])["summary"] == revised.summary

            async def read(path, max_bytes):
                return candidate.model_dump_json().encode()

            yield SimpleNamespace(read=read)

        @asynccontextmanager
        async def interaction(self, task, runtime):
            async def turn():
                turns.append(task.data.candidate_id)
                return SimpleNamespace(terminated=False)

            yield SimpleNamespace(turn=turn, trace=SimpleNamespace(id="check", ok=True))

    class Experiment(AblationEnv):
        def trace(self, trace):
            pass

        async def review(self, agents, payload, guide, name):
            return {"verdict": (revised if guide == "revised" else approval(payload)).model_dump(mode="json")}

    env = Experiment(
        PipelineConfig(),
        study,
        {c.id: payload for c in cases},
        {"baseline": REVIEW_GUIDE, "revised": "revised"},
    )
    agents = SimpleNamespace(builder=Author())
    await env.arm(cases[0], "baseline", agents)
    await env.arm(cases[0], "revised", agents)
    await env.arm(cases[0], "revised", agents)
    assert len(turns) == 1, "resume must never dispatch a second revision"
    atomic_json(root / "jobs/interrupted/result.json", {"status": "running"})
    result = await env.job("interrupted", lambda: (_ for _ in ()).throw(AssertionError("retried")))
    assert result["status"] == "interrupted"
    atomic_json(root / "traces/budget.json", {"usage": {"cost": 25}})
    result = await env.job("over-budget", lambda: (_ for _ in ()).throw(AssertionError("dispatched")))
    assert result["status"] == "budget_stopped"
    print("PASS ablation default guide, feedback delivery, arm isolation, one revision, resume and budget")


async def check_seeds(root):
    from unittest.mock import patch
    from scripts.worldgen_slack import run_seed_study
    from .agents.author import FILE_GUIDE
    from .agents.builder import BuilderTask
    from .agents.synthesizer import SynthesizerTask
    from .config import ROOT, SeedDataConfig
    from .contracts import (
        SEED_MAX_BYTES,
        SEED_MAX_MESSAGES,
        SEED_MAX_TEXT_CHARS,
        SeedPacket,
        load_seed_packet,
        verdict_schema,
    )
    from .generate import provenance
    from scripts.worldgen_slack.prepare_seeds import Selection, normalize, normalize_flyte, normalize_software

    selection = Selection(
        id="example",
        dataset="unionai/flyte-slack-data",
        revision="a" * 40,
        start=0,
        end=2,
        notes="Manually checked sequence; metadata unavailable.",
        join_pairs=True,
    )
    response = {
        "rows": [
            {"row_idx": 0, "row": {"input": "Can you check?", "output": "Checking."}, "truncated_cells": []},
            {"row_idx": 1, "row": {"input": "Checking.", "output": "Fixed."}, "truncated_cells": []},
        ]
    }
    packet = SeedPacket(examples=[normalize(selection, response)])
    assert [message.text for message in packet.examples[0].messages] == [
        "Can you check?",
        "Checking.",
        "Fixed.",
    ]
    assert all(
        message.speaker is None and message.timestamp is None for message in packet.examples[0].messages
    )
    rows = [row["row"] for row in response["rows"]]
    fails(normalize_flyte, rows, join_pairs=False)
    fails(normalize_flyte, [rows[0], {"input": "Another conversation", "output": "OK"}], join_pairs=True)
    fails(normalize, selection, {"rows": response["rows"][:1]})
    truncated = json.loads(json.dumps(response))
    truncated["rows"][0]["truncated_cells"] = ["input"]
    fails(normalize, selection, truncated)
    software = [
        dict(workspace="workspace", channel="general", text="Done", user="A", ts="2018-01-01T12:00:00")
    ]
    messages = normalize_software(software)
    assert messages[0].speaker == "A" and messages[0].timestamp == "2018-01-01T12:00:00"
    fails(normalize_software, [software[0], dict(software[0], channel="elsewhere")])
    fails(Selection.model_validate, dict(selection.model_dump(), end=101))
    for roles in ([], ["judge"], ["solver"], ["builder", "builder"]):
        fails(SeedDataConfig, path=root / "seeds.json", roles=roles)
    assert SeedDataConfig(path=Path("data/seeds/example.json")).path == ROOT / "data/seeds/example.json"
    for update in (
        {"messages": [{"text": "x"}] * 102},
        {"messages": [{"text": "x" * (SEED_MAX_TEXT_CHARS + 1)}]},
        {"messages": [{"text": "   "}]},
        {"notes": "x" * SEED_MAX_BYTES},
        {"rows": [1, 0]},
    ):
        fails(SeedPacket.model_validate, {"examples": [dict(packet.examples[0].model_dump(), **update)]})
    fails(SeedPacket, examples=packet.examples * 2)
    expanded = [
        dict(packet.examples[0].model_dump(), id=f"example-{index}", messages=[{"text": "x"}] * 64)
        for index in range(SEED_MAX_MESSAGES // 64)
    ]
    assert (
        sum(len(example.messages) for example in SeedPacket.model_validate({"examples": expanded}).examples)
        == SEED_MAX_MESSAGES
    )
    expanded[0]["messages"].append({"text": "one too many"})
    fails(SeedPacket.model_validate, {"examples": expanded})
    root.mkdir(parents=True)
    path = root / "seeds.json"
    path.write_text(packet.model_dump_json())
    assert load_seed_packet(path) == packet
    oversized = root / "oversized.json"
    oversized.write_bytes(b" " * (SEED_MAX_BYTES + 1))
    fails(load_seed_packet, oversized)
    catalog, candidate = fixture()
    base = Config(sector="IT", task_count=2, group_size=1, output=root / "baseline")
    assert "seed_data" not in provenance(base)["config"]
    for roles in (None, ["builder"], ["synthesizer"], ["synthesizer", "builder"]):
        config = base.model_copy(
            update={"seed_data": SeedDataConfig(path=path, roles=roles) if roles else None}
        )
        seeds = load_seed_packet(path) if roles else None
        run_root = root / ("-".join(roles) if roles else "baseline")
        store = TestStore(run_root, provenance(config, seeds))
        store.state.catalog = catalog
        env = GenerationEnv(config, store, seeds)
        for role, task_type in (("synthesizer", SynthesizerTask), ("builder", BuilderTask)):
            for group in catalog.groups:
                context = env.catalog_context() if role == "synthesizer" else env.build_context(group)
                task = task_type.create(context, group.id, env.author_seeds(role))
                selected = roles is not None and role in roles
                assert ("seed_data" in context) == selected
                assert ("seeds.json" in task.config.files) == selected
                if not selected:
                    assert task.config.files["guide.md"] == FILE_GUIDE + task_type.instructions
                files = {}

                async def write(name, value):
                    files[name] = value

                async def read(name, max_bytes):
                    return candidate.model_dump_json().encode()

                async def turn(message):
                    assert ("/task/seeds.json" in files) == selected
                    assert ("seed_data" in json.loads(files["/task/input.json"])) == selected
                    if selected:
                        assert SeedPacket.model_validate_json(files["/task/seeds.json"]) == packet
                    return SimpleNamespace(terminated=False)

                runtime = SimpleNamespace(write=write, read=read)
                await task.setup(None, runtime)
                interaction = SimpleNamespace(turn=turn, trace=SimpleNamespace(info={}))
                await env.author_turn(interaction, runtime, context, "initial", True)
                context["feedback"] = "Revise the world"
                await env.author_turn(interaction, runtime, context, "repair", False)
        solver = SolverTask.create(
            catalog.tasks[0], candidate.snapshot, "ws1", reference_for(catalog.tasks[0], candidate)
        )
        assert "seeds.json" not in solver.data.model_dump_json() + solver.config.model_dump_json()
        store.save()
        store.close()
        resumed = TestStore(run_root, provenance(config, seeds))
        resumed.close()
        if seeds:
            changed = packet.model_copy(deep=True)
            changed.examples[0].messages[0].text = "Changed source"
            fails(Store, run_root, provenance(config, changed))
    schema = verdict_schema("world")
    assert set(schema["properties"]["criteria"]["required"]) == set(QUALITY_CRITERIA)
    assert schema["properties"]["criteria"]["additionalProperties"] is False
    assert "criteria" in schema["required"]
    assert "required" not in verdict_schema("catalog")["properties"]["criteria"]
    bad = approval(review_payload(catalog, candidate, ["t1", "t2"], "world")).model_dump()
    bad["criteria"]["weighted_quality"] = 0.9
    fails(Verdict.model_validate, bad)
    saved_failure = ROOT / "data/qualification-01/services/failed-verdict.json"
    if saved_failure.exists():
        fails(Verdict.model_validate_json, saved_failure.read_bytes())
    fixed_store = TestStore(root / "fixed", {"test": True})
    fixed_store.state.catalog = catalog
    fixed_store.state.phase = "build"
    fixed = GenerationEnv(base, fixed_store, fixed_catalog=True)
    rejection = approval(review_payload(catalog, candidate, ["t1", "t2"], "world")).model_copy(
        update={
            "approved": False,
            "issues": [
                Issue(owner="synthesizer", artifact="catalog", defect="Ambiguous", requested_change="Clarify")
            ],
        }
    )
    before = fixed_store.state.model_dump()
    fails(fixed.route_rejection, rejection, "build:g1")
    assert fixed_store.state.model_dump() == before
    fixed_store.close()
    dispatched = []

    async def fake_run(config, **kwargs):
        dispatched.append(config)
        summary = {"status": "complete", "reported_model_cost": 13}
        atomic_json(config.output / "run.json", {})
        atomic_json(config.output / "summary.json", summary)
        return summary

    study_root = root / "study"
    study = run_seed_study.Study(output="data/check", seed_path="unused", configs=["a", "b", "c"])
    cases = [(base, catalog, approval(review_payload(catalog, None, ["t1", "t2"], "catalog")))] * 3
    with (
        patch.object(run_seed_study, "inputs", return_value=(study_root, path, cases, {"test": 1})),
        patch.object(run_seed_study, "run", fake_run),
    ):
        await run_seed_study.execute(study)
        await run_seed_study.execute(study)
    assert len(dispatched) == 2 and [config.research_budget_usd for config in dispatched] == [25, 12]
    assert dispatched[0].seed_data is None and dispatched[1].seed_data.roles == ["builder"]
    result = json.loads((study_root / "results.json").read_text())
    assert len(result["arms"]) == 6 and result["reported_model_cost"] == 26
    assert all(arm["status"] == "budget_stopped" for arm in result["arms"][2:])
    dispatched.clear()
    study = study.model_copy(update={"prior_reported_cost_usd": 3.0})
    with (
        patch.object(run_seed_study, "inputs", return_value=(root / "parallel", path, cases, {"test": 1})),
        patch.object(run_seed_study, "run", fake_run),
    ):
        await asyncio.gather(
            run_seed_study.execute(study, "seeded"), run_seed_study.execute(study, "unseeded")
        )
        await run_seed_study.execute(study, "seeded")
        await run_seed_study.execute(study, "unseeded")
    assert len(dispatched) == 2
    assert all(config.research_budget_usd == 11 for config in dispatched)
    assert {config.output.parent.name for config in dispatched} == {"seeded", "unseeded"}
    assert all(
        (config.seed_data is not None) == (config.output.parent.name == "seeded") for config in dispatched
    )
    for arm in ("seeded", "unseeded"):
        result = json.loads((root / "parallel" / arm / "results.json").read_text())
        assert len(result["arms"]) == 3 and result["reported_model_cost"] == 13
        assert all(item["seeded"] == (arm == "seeded") for item in result["arms"])
        assert all(item["status"] == "budget_stopped" for item in result["arms"][1:])
    print(
        "PASS seed normalization, bounds, role targeting, repairs, resume, solver isolation, verdict schema"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-feedback", action="store_true")
    args = parser.parse_args()
    if args.live_feedback:
        asyncio.run(check_live_feedback())
    else:
        check_slack_rules()
        check_contracts_and_reads()
        asyncio.run(check_native_servers())
        with tempfile.TemporaryDirectory() as directory:
            asyncio.run(check_seeds(Path(directory) / "seeds"))
            asyncio.run(check_flow(Path(directory) / "run"))
            asyncio.run(check_ablation(Path(directory) / "ablation"))
        print("All generation checks passed.")
