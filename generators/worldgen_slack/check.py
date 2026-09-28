"""Run: uv run python -m generators.worldgen_slack.check"""

import asyncio
import json
import tempfile
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo
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
from .config import Acceptance, Config
from .env import GenerationEnv, reference_for
from .store import Store, ReviewLimit
from .agents.writer import brief, excerpts, observed, visible
from .store import used_names
from .contracts import (
    Beat,
    BindOutput,
    Binding,
    Candidate,
    Catalog,
    ClaimEvidence,
    Detail,
    Fact,
    Issue,
    Line,
    Persona,
    Plan,
    PlannedTask,
    Premise,
    Premises,
    Rewrite,
    ScenePlan,
    SeedExample,
    SeedMessage,
    SeedPacket,
    TaskReview,
    Verdict,
    WorkGroup,
    WrittenScene,
    accepted,
    assemble,
    check_plan,
    check_scene,
    first_mentions,
    listed,
    pick_premise,
    reveals_answer,
    timed,
    style,
    to_utc,
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
    clean = approval(payload)
    minor = Issue(owner="builder", artifact="workspace", defect="x", requested_change="y", blocking=False)
    assert accepted(clean.model_copy(update={"issues": [minor]}), payload, Acceptance()), (
        "minor issues do not block"
    )
    assert not accepted(
        clean.model_copy(update={"issues": [minor.model_copy(update={"blocking": True})]}),
        payload,
        Acceptance(),
    )
    low = clean.model_copy(update={"criteria": {**clean.criteria, "world_coherence": 0.6}})
    assert not accepted(low, payload, Acceptance()), "a criterion under its floor rejects"
    assert not accepted(
        clean.model_copy(update={"issues": [minor]}), payload, Acceptance(minor_issues_block=True)
    )
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
        assert set(json.loads(task.config.files["schemas.json"])) in (
            {"premise", "catalog"},
            {"plan", "bind"},
        )
        yield None

    @asynccontextmanager
    async def interaction(self, task, runtime):
        yield SimpleNamespace(trace=SimpleNamespace(ok=True, info={}, record_reward=lambda *args: None))


def premises(count=12):
    return Premises(
        premises=[
            Premise(
                company=f"{name} Freight",
                niche="cold-chain logistics IT",
                region="Poland",
                size="400 people",
                culture="direct, hybrid",
                cast="Polish and Ukrainian engineers",
            )
            for name in (
                "Harbor",
                "Kestrel",
                "Wawel",
                "Odra",
                "Lynx",
                "Bora",
                "Tatra",
                "Vistula",
                "Amber",
                "Mazur",
                "Sokol",
                "Brda",
            )[:count]
        ]
    )


def personas(catalog):
    return [
        Persona(
            id=p.id, role="engineer", seniority="senior", timezone="Europe/Warsaw", voice="terse, lowercase"
        )
        for p in catalog.people
    ]


def plan_fixture(catalog):
    conversations = fixture()[1].snapshot.conversations
    return Plan(
        conversations=conversations,
        scenes=[
            ScenePlan(
                id="s_route",
                conversation_id="c1",
                participant_ids=["u1", "u2"],
                start="2026-01-02T09:00:00Z",
                end="2026-01-02T09:45:00Z",
                situation="Choosing a destination",
                beats=[Beat(fact_id="f1", author_id="u1")],
                length=2,
            ),
            ScenePlan(
                id="s_private",
                conversation_id="c2",
                participant_ids=["u2"],
                start="2026-01-02T10:00:00Z",
                end="2026-01-02T10:45:00Z",
                situation="Private approval",
                beats=[Beat(fact_id="f2", author_id="u2")],
                length=1,
            ),
            ScenePlan(
                id="s_chat",
                conversation_id="c1",
                participant_ids=["u1", "u2"],
                start="2026-01-02T11:00:00Z",
                end="2026-01-02T11:45:00Z",
                situation="Lunch plans",
                length=2,
            ),
        ],
    )


ZONES = {"u1": "Europe/Warsaw", "u2": "Europe/Warsaw"}


def scripted_scene(scene, recent, previous):
    base = datetime.fromisoformat(scene.start).astimezone(ZoneInfo("Europe/Warsaw"))

    def at(minutes, seconds):
        return (base + timedelta(minutes=minutes, seconds=seconds)).strftime("%Y-%m-%d %H:%M:%S")

    texts = {"f1": "ok so work-1 goes to region-1", "f2": "work-2 is approved for region-2"}
    lines = [
        Line(author_id=b.author_id, text=texts[b.fact_id], local_time=at(0, 17), conveys=[b.fact_id])
        for b in scene.beats
    ] or [Line(author_id=scene.participant_ids[0], text="lunch?", local_time=at(0, 5))]
    lines.append(
        Line(
            author_id=scene.participant_ids[-1],
            text="rewritten" if scene.revision_note else f"seen {len(recent)}",
            local_time=at(30, 41),
            reply_to=0 if previous is None else None,
        )
    )
    promises = {"s_route": ["u1 posts the region map"], "s_chat": ["u2 books the lunch table by noon"]}.get(
        scene.id
    )
    written = WrittenScene(lines=lines, introduces=[f"{scene.id} detail"], promises=promises)
    assert not check_scene(scene, written, ZONES)
    return written


def check_scenes(root):
    catalog, candidate = fixture()
    chosen = pick_premise(premises(), 12, ["Northstar Systems"], seed=7)
    assert chosen == pick_premise(premises(), 12, [], seed=7)
    fails(pick_premise, premises(11), 12, [], 7)
    fails(pick_premise, premises(), 12, ["Harbor Cloud Systems"], 7)
    duplicated = premises()
    duplicated.premises[1].company = "Harbor Logistics"
    fails(pick_premise, duplicated, 12, [], 7)
    catalog.personas = personas(catalog)
    fails(Persona, id="u1", role="r", seniority="s", timezone="Mars/Olympus", voice="v")
    fails(Catalog.model_validate, dict(catalog.model_dump(), personas=catalog.model_dump()["personas"][:1]))
    plan = plan_fixture(catalog)
    check_plan(catalog, plan)
    outsider = plan.model_copy(deep=True)
    outsider.scenes[1].participant_ids = ["u1"]
    fails(check_plan, catalog, outsider)
    scenes = {s.id: scripted_scene(s, [], None) for s in plan.scenes}
    silent = WrittenScene(lines=[Line(author_id="u1", text="hi", local_time="2026-01-02 10:00:00")])
    assert any("f1" in error for error in check_scene(plan.scenes[0], silent, ZONES))
    local_start = WrittenScene(lines=[Line(author_id="u1", text="hi", local_time="2026-01-02 12:00:07")])
    assert not check_scene(plan.scenes[2], local_start, ZONES)
    overrun = WrittenScene(lines=[Line(author_id="u1", text="hi", local_time="2026-01-02 12:50:00")])
    assert any("after the scene end" in e for e in check_scene(plan.scenes[2], overrun, ZONES)), "scenes end"
    as_if_utc = WrittenScene(lines=[Line(author_id="u1", text="hi", local_time="2026-01-02 11:00:00")])
    assert any("before the scene start" in error for error in check_scene(plan.scenes[2], as_if_utc, ZONES))
    swapped = WrittenScene(
        lines=[
            Line(author_id="u1", text="a", local_time="2026-01-02 12:00:30"),
            Line(author_id="u2", text="b", local_time="2026-01-02 12:00:12"),
        ]
    )
    assert check_scene(plan.scenes[2], swapped, ZONES) == [
        "line 1 is at 2026-01-02 12:00:12, before line 0 at 2026-01-02 12:00:30; messages must follow each other in time"
    ]
    in_order = WrittenScene(lines=list(reversed(swapped.lines)))
    assert not check_scene(plan.scenes[2], in_order, {"u1": "Europe/Warsaw", "u2": "Asia/Singapore"}), (
        "every line is on the scene clock, not its author's"
    )
    assert to_utc("2026-01-02 00:30:00", "Europe/Warsaw") == "2026-01-01T23:30:00Z"
    assert to_utc("2026-01-01 22:00:00", "America/Santiago") == "2026-01-02T01:00:00Z"
    fails(Line, author_id="u1", text="hi", local_time="2026-01-02T10:00:00Z")
    lines = [
        Line(author_id="u1", text="a", local_time="2026-01-02 12:00:01"),
        Line(author_id="u2", text="b", local_time="2026-01-02 12:00:09"),
        Line(author_id="u1", text="c", local_time="2026-01-02 12:01:00", reply_to=1),
        Line(author_id="u2", text="d", local_time="2026-01-02 12:02:00", reply_to=2),
    ]
    assert not check_scene(plan.scenes[2], WrittenScene(lines=lines), ZONES)
    nested, placed = assemble(
        catalog, plan.model_copy(update={"scenes": [plan.scenes[2]]}), {"s_chat": WrittenScene(lines=lines)}
    )
    ids = placed["scene_messages"]["s_chat"]
    assert {m.id: m.thread_root_id for m in nested.messages}[ids[3]] == ids[1], (
        "a reply to a reply joins its thread"
    )
    lines[3].reply_to = 3
    assert check_scene(plan.scenes[2], WrittenScene(lines=lines), ZONES) == [
        "line 3: reply_to 3 must name an earlier line of this scene"
    ]
    lines[3].reply_to = 0
    two_threads, placed = assemble(
        catalog, plan.model_copy(update={"scenes": [plan.scenes[2]]}), {"s_chat": WrittenScene(lines=lines)}
    )
    ids = placed["scene_messages"]["s_chat"]
    roots = {m.id: m.thread_root_id for m in two_threads.messages}
    assert roots[ids[2]] == ids[1] and roots[ids[3]] == ids[0] and roots[ids[1]] is None
    world, index = assemble(catalog, plan, scenes)
    again, _ = assemble(catalog, plan, scenes)
    assert world == again and len(world.messages) == 6
    root_id, reply_id = index["scene_messages"]["s_route"]
    assert next(m for m in world.messages if m.id == reply_id).thread_root_id == root_id
    assert index["conveyed"] == {"f1": [root_id], "f2": [index["scene_messages"]["s_private"][0]]}
    assert not any(reveals_answer(m.id, "region-1", "Which region?") for m in world.messages)
    report = style(world, catalog.personas)
    assert report["messages"] == 6 and report["seconds_zero"] == 0 and report["lowercase_start"] == 1
    assert set(report["authors"]) == {"u1", "u2"} and report["off_hours"] == 0
    seeds = SeedPacket(examples=[example(f"e{i}") for i in range(5)])
    assert excerpts(seeds, "1:s") == excerpts(seeds, "1:s") and len(excerpts(seeds, "1:s")) == 2
    assert excerpts(None, "1:s") == []
    prompt = brief(
        catalog,
        chosen,
        plan.model_copy(
            update={
                "details": {
                    "QA pool": Detail(value="blr-fleet-2"),
                    "incident": Detail(value="INC-9", since="2026-01-03T00:00:00Z"),
                    "cutover": Detail(value="cutover window", at="2026-01-02T15:00:00Z"),
                }
            }
        ),
        plan.scenes[0],
        {
            "recent": [],
            "elsewhere": [("2026-01-02T08:00:00Z", "restricted", lines[0])],
            "established": ["release train is 8.4"],
            "commitments": ["u2 books the lunch table by noon"],
        },
        excerpts(seeds, "1:s"),
        None,
        "English",
        {"f1": "2026-01-02T09:15:00Z"},
    )
    assert "region-1" in prompt and "Which region" not in prompt and "canonical_answer" not in prompt
    assert '"start": "2026-01-02 10:00:00"' in prompt and '"language": "English"' in prompt
    assert '"utc"' not in prompt and "start_local" not in prompt, "the brief shows one clock"
    assert json.loads(prompt)["world_details"] == {
        "QA pool": "blr-fleet-2",
        "cutover": {"value": "cutover window", "at": "Friday 2026-01-02 16:00"},
    }, "details appear only once known; scheduled moments are on the scene clock"
    assert json.loads(prompt)["commitments"] == ["u2 books the lunch table by noon"]
    assert json.loads(prompt)["established_details"] == ["release train is 8.4"]
    conversations = {c.id: c for c in plan.conversations}
    route, private, chat = plan.scenes
    assert visible(route, private, conversations) and visible(route, chat, conversations)
    assert not visible(chat, route, conversations) and not visible(route, route, conversations)
    alone = chat.model_copy(update={"participant_ids": ["u1"]})
    assert not visible(private, alone, conversations), (
        "u1 is not in the private channel, so it waits for nothing there"
    )
    loose = WrittenScene.model_validate_json(
        '{"lines": [{"author_id": "u1", "text": "x", "local_time": "2026-01-02 12:00:01"}], "introduces": null, "promises": "lunch at noon"}'
    )
    assert loose.introduces == [] and loose.promises == ["lunch at noon"]
    objects = WrittenScene.model_validate(
        {
            "lines": loose.model_dump()["lines"],
            "promises": [{"who": "u2", "what": "book lunch", "by": "noon"}],
        }
    )
    assert objects.promises == ["u2, book lunch, noon"], "a promise written as an object becomes a note"
    assert '"state_at": "2026-01-02 10:15"' in prompt and '"established_facts"' in prompt
    assert json.loads(prompt)["seen_elsewhere"] == [
        {
            "conversation": "restricted",
            "author": "Alicia Rao",
            "time": "Friday 2026-01-02 09:00:00",
            "text": "a",
        }
    ]
    assert listed('["f1"]') == ["f1"]
    clocked = catalog.model_copy(deep=True)
    clocked.facts[0].valid_from = "2026-01-02T08:30:00Z"
    assert timed(clocked) == {"f1": "2026-01-02T08:30:00Z"}, (
        "a fact with a time of day is stated in its minute"
    )
    local_times = review_payload(clocked, None, [], "catalog")["fact_local_times"]
    assert local_times == {"f1": {"Europe/Warsaw": "Friday 2026-01-02 09:30"}}, (
        "judges see timed facts locally"
    )
    assert first_mentions(plan) == {"f1": "s_route", "f2": "s_private"}
    fails(check_plan, clocked, plan)
    clocked.facts[0].valid_from = "2026-01-02T09:00:00Z"
    check_plan(clocked, plan)
    written = scripted_scene(plan.scenes[0], [], None)
    assert not check_scene(plan.scenes[0], written, ZONES, {"f1": "2026-01-02T09:00:00Z"})
    assert check_scene(plan.scenes[0], written, ZONES, {"f1": "2026-01-02T09:05:00Z"}) == [
        "fact f1: the first line stating it is at 2026-01-02 10:00:17 but must be in the minute 2026-01-02 10:05"
    ], "a timed fact is stated in its own minute, not merely before a deadline"
    corpus = root / "corpus"
    atomic_json(
        corpus / "old/artifacts" / f"{digest(catalog.model_dump(mode='json'))}.json",
        catalog.model_dump(mode="json"),
    )
    atomic_json(corpus / "old/state.json", {"catalog": digest(catalog.model_dump(mode="json"))})
    atomic_json(corpus / "odd/state.json", [])
    assert used_names(corpus, corpus / "new") == {
        "companies": ["Harbor"],
        "people": ["Alicia Rao", "Owen Sato"],
    }
    assert used_names(corpus, corpus / "old") == {"companies": [], "people": []}
    print(
        "PASS premise pick, personas, plan/scene contracts, deterministic assembly, style, excerpts, corpus names"
    )


async def check_flow(root):
    catalog, _ = fixture()
    catalog.personas = personas(catalog)
    written, turns = [], []

    class ScriptedEnv(GenerationEnv):
        async def author_turn(self, interaction, runtime, context, attempt, first):
            assert context.get("group_id") != "g2", "already supported groups need review, not regeneration"
            turns.append((attempt, context["phase"], bool(context.get("feedback"))))
            if context["phase"] != "bind":
                assert context["language"] == "English"
            if context["phase"] == "premise":
                assert first and context["used_names"] == {"companies": [], "people": []}
                return premises().model_dump_json()
            if context["phase"] == "catalog":
                assert context["premise"]
                return catalog.model_copy(update={"company": context["premise"]["company"]}).model_dump_json()
            if context["phase"] == "plan":
                assert context["new_messages_hint"] == "9-19", "15 messages per task in a one-task group"
                plan = self.store.state.plan or plan_fixture(catalog)
                if [phase for _, phase, _ in turns].count("plan") == 1:
                    broken = plan.model_copy(deep=True)
                    broken.scenes[0].conversation_id = "c_missing"
                    return broken.model_dump_json()
                return plan.model_dump_json()
            conveyed = context["conveyed"]
            assert "open_promises" in context
            first_bind = not any(
                line.text == "rewritten" for r in self.store.state.scenes.values() for line in r.scene.lines
            )
            misses = not first_bind and not any(phase == "bind" and fed for _, phase, fed in turns)
            bindings = [
                Binding(
                    task_id=f"t{i}",
                    claims=[ClaimEvidence(claim_index=0, message_ids=conveyed[f"f{i}"])],
                    gold_calls=[
                        ReadCall(
                            action="search_messages", arguments={"query": "lunch" if misses else f"work-{i}"}
                        )
                    ],
                )
                for i in (1, 2)
            ]
            rewrites = [Rewrite(scene_id="s_chat", note="Make lunch plans specific")]
            # A broken bind that also asks for a rewrite must still be validated once rewrites run out.
            return BindOutput(
                bindings=bindings, rewrites=rewrites if first_bind or misses else []
            ).model_dump_json()

        async def write_scene(self, agents, plan, scene, observed, previous, timely):
            written.append(scene.id)
            return scripted_scene(scene, observed["recent"], previous)

        async def review(self, agents, payload, attempt, author_trace=None):
            verdict = approval(payload)
            if attempt == "build-g1-02":
                assert payload["changed_messages"] and payload["previous_issues"], (
                    "a repair is re-reviewed with focus"
                )
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
                                message_ids=[] if upstream else ["m" + digest(["s_chat", 0])[:10]],
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

    config = Config(sector="IT", task_count=2, group_size=1, output=root, seed=1, corpus=root / "corpus")
    store = TestStore(root, {"test": 1})
    store.state.catalog = catalog
    agents = SimpleNamespace(synthesizer=ScriptedAgent(), builder=ScriptedAgent())
    env = ScriptedEnv(config, store)
    await env.run(None, agents)
    assert store.state.phase == "done"
    assert store.state.premise == pick_premise(premises(), 12, [], 1)
    assert written == ["s_route", "s_private", "s_chat", "s_chat", "s_chat"], (
        "scenes are written in world time; only new, revised or repaired ones"
    )
    assert not any(a == "build-g1-02" and phase == "plan" for a, phase, _ in turns), "a repair keeps the plan"
    assert "Repetitive activity" in store.state.plan.scenes[2].revision_note, (
        "the writer gets the judge's words"
    )
    chat = store.state.plan.scenes[2]
    written_now = {k: v.scene for k, v in store.state.scenes.items()}
    seen = observed(store.state.catalog, store.state.plan, written_now, chat)
    assert [line.text for _, _, line in seen["recent"]] == ["ok so work-1 goes to region-1", "seen 0"]
    assert [(name, line.text) for _, name, line in seen["elsewhere"]] == [
        ("restricted", "work-2 is approved for region-2"),
        ("restricted", "seen 0"),
    ]
    assert not observed(store.state.catalog, store.state.plan, written_now, store.state.plan.scenes[0])[
        "elsewhere"
    ]
    assert seen["established"] == ["s_route detail", "s_private detail"]
    assert seen["commitments"] == ["u1 posts the region map"], "earlier promises travel to later writers"
    assert store.state.plan.scenes[2].revision_note and len(store.state.candidate.snapshot.messages) == 6
    corrected = [(phase, fed) for attempt, phase, fed in turns if attempt == "build-g1-01"]
    assert corrected[:2] == [("plan", False), ("plan", True)], "plan errors are corrected inside the attempt"
    assert ("bind", True) in corrected, "bind validation errors are corrected inside the attempt"
    assert json.loads((root / "attempts/build-g1-01/corrections.json").read_text())
    judged = []

    async def flaky(task):
        judged.append(task)
        verdict = approval(json.loads(task.config.tools.payload_json)).model_dump(mode="json")
        return SimpleNamespace(ok=len(judged) > 1, errors=[], info={"verdict": verdict})

    payload = review_payload(catalog, store.state.candidate, ["t1"], "world")
    verdict = await GenerationEnv.review(
        env, SimpleNamespace(judge=SimpleNamespace(run=flaky)), payload, "judge"
    )
    assert verdict.approved and len(judged) == 2 and judged[0] is not judged[1]
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
    candidate = store.state.candidate
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
    assert resumed.state.rounds["catalog"] == before["catalog"] + 1, (
        "catalog repairs use the catalog allowance"
    )
    assert resumed.state.rounds["build:g2"] == before["build:g2"]
    assert resumed.state.phase == "build" and not resumed.state.final_groups
    assert resumed.state.last_approved_candidate is None
    await repairing.run(None, agents)
    assert resumed.state.phase == "done" and resumed.state.final_groups == ["g1", "g2"]
    assert repairing.current()
    resumed.state.catalog.facts[0].value += " (revised)"
    assert not repairing.current(), "a world written against old facts is rebuilt, not re-reviewed"
    assert resumed.state.rounds["build:g1"] == before["build:g1"]
    assert resumed.state.rounds["build:g2"] == before["build:g2"] + 1
    assert resumed.state.rounds["final:g2"] == before["final:g2"] + 1
    fails(resumed.reserve, "final:g2", resumed.state.rounds["final:g2"])
    resumed.reserve("build:g2", resumed.state.rounds["build:g2"] + 1)
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
            injected = self.store.root / "injected-catalog.json"
            if context["phase"] != "catalog" or injected.exists():
                return raw
            try:
                Catalog.model_validate_json(raw)
            except ValueError:
                return raw
            content = json.loads(raw)
            self.store.artifact(attempt, "before_test_injection", content)
            original, replacement = content["tasks"][:2]
            content["tasks"][1] = dict(
                original,
                id=replacement["id"],
                group_id=replacement["group_id"],
                question="In this workspace, " + original["question"][:1].lower() + original["question"][1:],
            )
            raw = json.dumps(content)
            await runtime.write("/task/catalog.json", raw.encode())
            self.store.artifact(attempt, "author_catalog", {"text": raw, "controlled_test_injection": True})
            atomic_json(injected, {"attempt": attempt})
            self.store.event("controlled_defect_injected", attempt=attempt)
            return raw

        async def write_scene(self, agents, plan, scene, seen, previous, timely):
            written = await super().write_scene(agents, plan, scene, seen, previous, timely)
            injected = self.store.root / "injected-world.json"
            if scene.beats or injected.exists():
                return written
            author = next(line.author_id for line in written.lines)
            latest = datetime.strptime(max(line.local_time for line in written.lines), "%Y-%m-%d %H:%M:%S")
            written.lines += [
                Line(
                    author_id=author,
                    text="Status update: everything remains on track. No new information.",
                    local_time=(latest + timedelta(days=1, minutes=5 * i)).strftime("%Y-%m-%d %H:%M:%S"),
                )
                for i in range(40)
            ]
            atomic_json(injected, {"attempt": self.store.state.active_attempt, "scene_id": scene.id})
            self.store.event("controlled_defect_injected", scene_id=scene.id)
            return written

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


def example(identifier):
    return SeedExample(
        id=identifier,
        dataset="unionai/flyte-slack-data",
        revision="a" * 40,
        rows=[0],
        source_sha256="b" * 64,
        notes="Fixture excerpt.",
        messages=[SeedMessage(text=f"{identifier} question?", speaker="speaker_1")],
    )


def check_seeds(root):
    from .config import ROOT, SeedDataConfig
    from .contracts import (
        SEED_MAX_BYTES,
        SEED_MAX_MESSAGES,
        SEED_MAX_TEXT_CHARS,
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
    assert all(m.speaker is None and m.timestamp is None for m in packet.examples[0].messages)
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
        sum(len(e.messages) for e in SeedPacket.model_validate({"examples": expanded}).examples)
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
    base = Config(sector="IT", task_count=2, group_size=1, output=root / "baseline")
    assert "seed_data" not in provenance(base)["config"]
    seeded = base.model_copy(update={"seed_data": SeedDataConfig(path=path)})
    fails(provenance, seeded)
    TestStore(root / "seeded", provenance(seeded, packet)).close()
    changed = packet.model_copy(deep=True)
    changed.examples[0].messages[0].text = "Changed source"
    fails(Store, root / "seeded", provenance(seeded, changed))
    catalog, candidate = fixture()
    schema = verdict_schema("world")
    assert set(schema["properties"]["criteria"]["required"]) == set(QUALITY_CRITERIA)
    assert schema["properties"]["criteria"]["additionalProperties"] is False
    assert "criteria" in schema["required"]
    assert "required" not in verdict_schema("catalog")["properties"]["criteria"]
    bad = approval(review_payload(catalog, candidate, ["t1", "t2"], "world")).model_dump()
    bad["criteria"]["weighted_quality"] = 0.9
    fails(Verdict.model_validate, bad)
    print("PASS seed normalization, bounds, packet provenance and resume, verdict schema")


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
            check_seeds(Path(directory) / "seeds")
            check_scenes(Path(directory) / "scenes")
            asyncio.run(check_flow(Path(directory) / "run"))
        print("All generation checks passed.")
