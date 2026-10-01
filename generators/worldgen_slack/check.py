"""Offline invariant checks, one per rule the pipeline relies on: uv run --frozen python -m generators.worldgen_slack.check"""

import asyncio
import json
import tempfile
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from verifiers.v1.mcp.launch import serve
from verifiers.v1.utils.compile import resolve_runtime_config
from worldgen_slack.slack.models import AnswerSpec, Conversation, Message, SlackWorld, User
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
from .agents import synthesizer, writer
from .agents.judge import JudgeTask, review_payload
from .agents.inspection import missing_evidence
from .agents.writer import brief, excerpts, observed, visible
from .config import ROOT, Acceptance, Config, PersonasConfig, PipelineConfig, SeedDataConfig
from .env import GenerationEnv, reference_for
from .generate import provenance
from .store import ReviewLimit, Store, used_names
from .contracts import (
    PHASE_CRITERIA,
    SEED_MAX_BYTES,
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
    ScenePlan,
    SeedExample,
    SeedMessage,
    SeedPacket,
    SeedPersona,
    TaskReview,
    Typing,
    Verdict,
    WorkGroup,
    WrittenScene,
    accepted,
    accepted_task,
    assemble,
    check_plan,
    check_scene,
    listed,
    load_seed_packet,
    pick_cast,
    pick_premise,
    reveals_answer,
    timed,
    to_utc,
    validate_candidate,
    validate_verdict,
    verdict_schema,
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
            group_id=f"g{i}",
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
                kind="fact_summary", canonical_answer=f"region-{i}", required_claims=[f"region-{i}"]
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
    messages = [
        Message(
            id=f"m{i}",
            conversation_id="c1",
            author_id="u1",
            text="Approved destination for work-1 is region-1." if i == 0 else f"Deployment observation {i}",
            timestamp=f"2026-01-01T00:0{i}:00Z",
        )
        for i in range(4)
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
        criteria=dict.fromkeys(PHASE_CRITERIA.get(payload["phase"], ()), 1.0),
        summary="checked",
    )


def check_slack_reads():
    """Visibility, ordering and cursor binding of the actor-scoped Slack API."""
    users = [User(id=u, name=u.title()) for u in ("agent", "alice", "bob", "cara")]
    everyone = [u.id for u in users]
    world = SlackWorld(
        users=users,
        conversations=[
            Conversation(id="incidents", name="incidents", kind="public_channel", member_ids=everyone),
            Conversation(
                id="private", name="leadership", kind="private_channel", member_ids=["alice", "bob"]
            ),
            Conversation(id="dm", kind="dm", member_ids=["agent", "cara"]),
            Conversation(id="old", name="old", kind="public_channel", member_ids=everyone, is_archived=True),
        ],
        messages=[
            Message(
                id=i,
                conversation_id=c,
                author_id=a,
                text=t,
                timestamp=f"2025-01-15T{h}Z",
                thread_root_id=r,
                deleted=d,
            )
            for i, c, a, t, h, r, d in (
                ("root", "incidents", "cara", "Payments timeout investigation.", "10:00:00", None, False),
                ("cause", "incidents", "alice", "A stale DNS cache caused it.", "10:05:00", "root", False),
                ("newer", "incidents", "bob", "Payments monitoring is green.", "11:00:00", None, False),
                ("roadmap", "private", "alice", "Confidential roadmap in Cedar.", "09:10:00", None, False),
                ("handoff", "dm", "cara", "Direct handoff is complete.", "09:20:00", None, False),
                ("deleted", "incidents", "alice", "Deleted secret answer cache.", "09:30:00", None, True),
                ("archived", "old", "bob", "Old incident detail.", "09:40:00", None, False),
            )
        ],
    )
    api = SlackAPI(world, "agent")
    assert {r["conversation_id"] for r in api.list_conversations()["items"]} == {"incidents", "dm"}
    fails(api.get_conversation_history, "private")
    fails(api.get_conversation_history, "old")
    assert {r["message_id"] for r in api.search_messages("secret confidential roadmap")["items"]}.isdisjoint(
        {"deleted", "roadmap"}
    )
    assert [r["message_id"] for r in api.get_thread("incidents", "root")["items"]] == ["root", "cause"]
    assert [r["message_id"] for r in api.get_conversation_history("incidents")["items"]] == ["newer", "root"]
    early = world.model_dump(mode="json")
    early["messages"][1]["timestamp"] = "2025-01-15T09:59:00Z"
    fails(SlackWorld.model_validate, early)
    _, candidate = fixture()
    api = SlackAPI(candidate.snapshot, "u1")
    cursor = api.get_conversation_history("c1", limit=1)["next_cursor"]
    assert api.get_conversation_history("c1", cursor=cursor, limit=1)["items"]
    fails(api.search_messages, "work", cursor=cursor)
    fails(SlackAPI(candidate.snapshot, "u2").get_conversation_history, "c1", cursor=cursor)
    edited = candidate.snapshot.model_copy(deep=True)
    edited.messages[1].text = "edited"
    fails(SlackAPI(edited, "u1").get_conversation_history, "c1", cursor=cursor)
    fails(api.get_conversation_history, "c1", limit=True)
    fails(api.search_messages, "work", before="nonsense")
    print("PASS Slack reads: visibility, archive, deletion, ordering, chronology, cursor binding")


def check_replay():
    """validate_candidate: gold routes replay as the reader, discover every ID, and never start from the answer."""
    catalog, candidate = fixture()
    assert validate_candidate(catalog, candidate, ["t1", "t2"])["ok"]
    route = candidate.model_copy(deep=True)
    route.bindings[0].gold_calls[0].arguments["conversation_id"] = "c1"
    assert "undiscovered conversation_id" in validate_candidate(catalog, route, ["t1"])["errors"][0]
    route.bindings[0].gold_calls.insert(0, ReadCall(action="list_conversations"))
    assert validate_candidate(catalog, route, ["t1"])["ok"], "IDs from earlier outputs are discovered"
    route.bindings[0].claims[0].user_ids = ["u1"]
    assert "user_ids=['u1']" in validate_candidate(catalog, route, ["t1"])["errors"][0]
    leaky = catalog.model_copy(deep=True)
    leaky.tasks[0].answer.canonical_answer = "region-1 destination"
    route = candidate.model_copy(deep=True)
    route.bindings[0].gold_calls = [
        ReadCall(action="search_messages", arguments={"query": "work-1 destination"})
    ]
    assert "answer terms ['destination']" in validate_candidate(leaky, route, ["t1"])["errors"][0]
    route.bindings[0].gold_calls.insert(0, ReadCall(action="search_messages", arguments={"query": "work-1"}))
    assert validate_candidate(leaky, route, ["t1"])["ok"], "answer terms are usable once an output shows them"
    gone = candidate.model_copy(deep=True)
    gone.snapshot.messages = [m for m in gone.snapshot.messages if m.id != "secret"]
    assert not validate_candidate(catalog, gone, ["t2"])["ok"], "evidence must exist"
    private = candidate.model_copy(deep=True)
    private.bindings[0].claims[0].message_ids = ["secret"]
    assert "invisible evidence secret" in validate_candidate(catalog, private, ["t1"])["errors"][0], (
        "evidence must be visible to the task's reader"
    )
    buried = candidate.model_copy(deep=True)
    buried.bindings[0].gold_calls[0].arguments["limit"] = 1
    assert validate_candidate(catalog, buried, ["t1"])["ok"]
    buried.snapshot.messages.append(
        Message(
            id="later",
            conversation_id="c1",
            author_id="u2",
            timestamp="2026-01-02T00:00:00Z",
            text="work-1 monitoring continues.",
        )
    )
    assert "misses bound evidence" in validate_candidate(catalog, buried, ["t1"])["errors"][0], (
        "later content that hides earlier evidence is a regression"
    )
    literal = candidate.model_copy(deep=True)
    literal.snapshot.messages[1].text = catalog.tasks[0].question
    assert "literal benchmark question" in validate_candidate(catalog, literal, [])["errors"][0]
    answer, question = (
        "Jon Bell approves Cedar exceptions as of 2026-10-21.",
        "Who approves Cedar exceptions?",
    )
    assert reveals_answer("usr_jon_bell", answer, question) and not reveals_answer(
        "cnv_cedar", answer, question
    )
    assert reveals_answer("m_2026_10_14", "2026-10-14", "When?") and not reveals_answer(
        "m_ops_14", "2026-10-14", "When?"
    )
    directory = catalog.model_dump(mode="json")
    directory["people"][0]["id"] = "usr_alicia"
    directory["tasks"][0].update(
        actor_id="usr_alicia",
        question="Who approved work-1?",
        answer={
            "kind": "entity",
            "canonical_answer": "Alicia Rao",
            "required_claims": ["Alicia Rao approved it."],
        },
    )
    fails(Catalog.model_validate_json, json.dumps(directory))
    duplicate = catalog.model_dump(mode="json")
    duplicate["tasks"][1]["question"] = duplicate["tasks"][0]["question"]
    fails(Catalog.model_validate_json, json.dumps(duplicate))
    conflict = catalog.model_dump(mode="json")
    conflict["facts"].append(dict(conflict["facts"][0], id="conflict", value="different"))
    fails(Catalog.model_validate_json, json.dumps(conflict))
    print("PASS replay: discovery, answer-term guard, evidence, regressions, identifier and catalog leaks")


def check_verdicts():
    """Verdicts are fresh and complete; blocking issues decide; approval requires inspected evidence."""
    catalog, candidate = fixture()
    payload = review_payload(catalog, candidate, ["t1", "t2"], "world")
    verdict = approval(payload)
    validate_verdict(verdict, payload)
    incomplete = verdict.model_copy(deep=True)
    incomplete.tasks[0].supported_claims = []
    fails(validate_verdict, incomplete, payload)
    task_payload = review_payload(catalog, candidate, ["t1"], "task")
    validate_verdict(approval(task_payload), task_payload)
    world_scored = approval(task_payload).model_copy(update={"criteria": verdict.criteria})
    fails(validate_verdict, world_scored, task_payload)
    stale = candidate.model_copy(deep=True)
    stale.snapshot.messages[1].text = "changed"
    fails(validate_verdict, verdict, dict(payload, candidate=stale.model_dump(mode="json")))
    minor = Issue(owner="builder", artifact="workspace", defect="x", requested_change="y", blocking=False)
    assert accepted(verdict.model_copy(update={"issues": [minor]}), payload, Acceptance()), (
        "minor issues pass"
    )
    blocking = minor.model_copy(update={"blocking": True})
    assert not accepted(verdict.model_copy(update={"issues": [blocking]}), payload, Acceptance())
    assert not accepted(
        verdict.model_copy(update={"issues": [minor]}), payload, Acceptance(minor_issues_block=True)
    )
    named = blocking.model_copy(update={"task_ids": ["t2"]})
    assert [
        accepted_task(verdict.model_copy(update={"issues": [named]}), payload, Acceptance(), t)
        for t in ("t1", "t2")
    ] == [
        True,
        False,
    ], "one task of a batch passes on its own review and issues"
    assert not accepted_task(
        verdict.model_copy(update={"issues": [blocking]}), payload, Acceptance(), "t1"
    ), "an issue naming no task holds back every task"
    low = verdict.model_copy(update={"criteria": dict.fromkeys(verdict.criteria, 0.1)})
    assert accepted(low, payload, Acceptance()), "scores are reported, not thresholds"
    fails(Issue, owner="synthesizer", artifact="bindings", defect="x", requested_change="y")
    for phase, ids in (("world", []), ("task", ["t1"])):
        schema = verdict_schema(phase, ids)
        criteria = schema["properties"]["criteria"]
        assert (
            set(criteria["required"]) == set(PHASE_CRITERIA[phase])
            and criteria["additionalProperties"] is False
        )
        assert (
            schema["properties"]["tasks"]["maxItems"] == len(ids) == schema["properties"]["tasks"]["minItems"]
        ), "the schema asks for exactly the requested task reviews"
    assert verdict_schema("task", ["t1"])["$defs"]["TaskReview"]["properties"]["task_id"]["enum"] == ["t1"]
    unknown = verdict.model_dump()
    unknown["criteria"]["weighted_quality"] = 0.9
    fails(Verdict.model_validate, unknown)
    assert set(missing_evidence(payload, [])) == {"t1", "t2"}
    reads = [
        ReadRecord(
            actor_id=t.actor_id, call=call, output=SlackAPI(candidate.snapshot, t.actor_id).execute(call)
        )
        for t, b in zip(catalog.tasks, candidate.bindings)
        for call in b.gold_calls
    ]
    assert not missing_evidence(payload, reads)
    reads[1].actor_id = "u1"
    assert "t2" in missing_evidence(payload, reads), "evidence counts only when read as the task's reader"
    print("PASS verdicts: freshness, claims, phase criteria, blocking issues decide, inspected evidence")


def premises(count=12):
    names = (
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
    )
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
            for name in names[:count]
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
    return Plan(
        conversations=fixture()[1].snapshot.conversations,
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


def check_time_and_scenes(root):
    """The time model, the plan and scene contracts, assembly, and what a writer is shown."""
    catalog, _ = fixture()
    chosen = pick_premise(premises(), 12, ["Northstar Systems"], seed=7)
    assert chosen == pick_premise(premises(), 12, [], seed=7), "the seed picks the premise"
    fails(pick_premise, premises(11), 12, [], 7)
    fails(pick_premise, premises(), 12, ["Harbor Cloud Systems"], 7)
    catalog.personas = personas(catalog)
    fails(Persona, id="u1", role="r", seniority="s", timezone="Mars/Olympus", voice="v")
    fails(Catalog.model_validate, dict(catalog.model_dump(), personas=catalog.model_dump()["personas"][:1]))
    stray = dict(catalog.model_dump()["facts"][0], group_id="g9")
    fails(Catalog.model_validate, dict(catalog.model_dump(), facts=[stray, catalog.model_dump()["facts"][1]]))
    plan = plan_fixture(catalog)
    check_plan(catalog, plan)
    outsider = plan.model_copy(deep=True)
    outsider.scenes[1].participant_ids = ["u1"]
    fails(check_plan, catalog, outsider)
    clocked = catalog.model_copy(deep=True)
    clocked.facts[0].valid_from = "2026-01-02T08:30:00Z"
    assert timed(clocked) == {"f1": "2026-01-02T08:30:00Z"}
    fails(check_plan, clocked, plan)
    clocked.facts[0].valid_from = "2026-01-02T09:00:00Z"
    check_plan(clocked, plan)
    moved = plan.model_copy(deep=True)
    moved.scenes[0].situation = "Rewritten history"
    fails(check_plan, catalog, moved, [plan.scenes[0]])
    noted = plan.model_copy(deep=True)
    noted.scenes[0].revision_note = "Shorter"
    check_plan(catalog, noted, [plan.scenes[0]])
    assert review_payload(clocked, None, [], "catalog")["fact_local_times"] == {
        "f1": {"Europe/Warsaw": "Friday 2026-01-02 10:00"}
    }, "judges see timed facts on local clocks"
    chat = plan.scenes[2]

    def scene(*times, **line):
        return WrittenScene(lines=[Line(author_id="u1", text="hi", local_time=t, **line) for t in times])

    assert any("f1" in e for e in check_scene(plan.scenes[0], scene("2026-01-02 10:00:00"), ZONES))
    assert not check_scene(chat, scene("2026-01-02 12:00:07"), ZONES)
    assert any("after the scene end" in e for e in check_scene(chat, scene("2026-01-02 12:50:00"), ZONES))
    assert any("before the scene start" in e for e in check_scene(chat, scene("2026-01-02 11:00:00"), ZONES))
    assert any(
        "follow each other" in e
        for e in check_scene(chat, scene("2026-01-02 12:00:30", "2026-01-02 12:00:12"), ZONES)
    )
    two = WrittenScene(
        lines=[
            Line(author_id="u1", text="a", local_time="2026-01-02 12:00:12"),
            Line(author_id="u2", text="b", local_time="2026-01-02 12:00:30"),
        ]
    )
    assert not check_scene(chat, two, {"u1": "Europe/Warsaw", "u2": "Asia/Singapore"}), "one clock per scene"
    written = scripted_scene(plan.scenes[0], [], None)
    assert not check_scene(plan.scenes[0], written, ZONES, {"f1": "2026-01-02T09:00:00Z"})
    assert (
        "must be in the minute"
        in check_scene(plan.scenes[0], written, ZONES, {"f1": "2026-01-02T09:05:00Z"})[0]
    )
    fails(Line, author_id="u1", text="hi", local_time="2026-01-02T10:00:00Z")
    assert to_utc("2026-01-01 22:00:00", "America/Santiago") == "2026-01-02T01:00:00Z"
    lines = [
        Line(author_id="u1", text="a", local_time="2026-01-02 12:00:01"),
        Line(author_id="u2", text="b", local_time="2026-01-02 12:00:09", reply_to=0),
        Line(author_id="u1", text="c", local_time="2026-01-02 12:01:00", reply_to=1),
        Line(author_id="u2", text="d", local_time="2026-01-02 12:02:00", reply_to=2),
    ]
    only_chat = plan.model_copy(update={"scenes": [chat]})
    nested, placed = assemble(catalog, only_chat, {"s_chat": WrittenScene(lines=lines)})
    ids = placed["scene_messages"]["s_chat"]
    assert {m.id: m.thread_root_id for m in nested.messages}[ids[3]] == ids[0], (
        "a reply to a reply joins its thread"
    )
    lines[3].reply_to = 3
    assert "must name an earlier line" in check_scene(chat, WrittenScene(lines=lines), ZONES)[0]
    scenes = {s.id: scripted_scene(s, [], None) for s in plan.scenes}
    world, index = assemble(catalog, plan, scenes)
    assert world == assemble(catalog, plan, scenes)[0] and len(world.messages) == 6, (
        "assembly is deterministic"
    )
    assert index["conveyed"]["f2"] == index["scene_messages"]["s_private"][:1]
    prompt = json.loads(
        brief(
            catalog,
            chosen,
            plan.model_copy(
                update={
                    "details": {
                        "pool": Detail(value="blr-2"),
                        "incident": Detail(value="INC-9", since="2026-01-03T00:00:00Z"),
                        "cutover": Detail(value="cutover", at="2026-01-02T15:00:00Z"),
                    }
                }
            ),
            plan.scenes[0],
            {"recent": [], "elsewhere": [], "established": [], "commitments": ["u2 books lunch"]},
            [],
            None,
            "English",
            {"f1": "2026-01-02T09:15:00Z"},
        )
    )
    assert (
        prompt["scene"]["start"] == "2026-01-02 10:00:00"
        and prompt["scene"]["beats"][0]["state_at"] == "2026-01-02 10:15"
    )
    assert prompt["world_details"] == {
        "pool": "blr-2",
        "cutover": {"value": "cutover", "at": "Friday 2026-01-02 16:00"},
    }, "details appear once known, on the scene clock"
    assert "Which region" not in json.dumps(prompt) and prompt["commitments"] == ["u2 books lunch"]
    conversations = {c.id: c for c in plan.conversations}
    route, private, chat = plan.scenes
    assert visible(route, private, conversations) and not visible(chat, route, conversations)
    assert not visible(private, chat.model_copy(update={"participant_ids": ["u1"]}), conversations)
    seen = observed(catalog, plan, scenes, chat)
    assert [line.text for _, _, line in seen["recent"]] == ["ok so work-1 goes to region-1", "seen 0"]
    assert seen["commitments"] == ["u1 posts the region map"], "earlier promises travel to later writers"
    loose = WrittenScene.model_validate_json(
        '{"lines": [{"author_id": "u1", "text": "x", "local_time": "2026-01-02 12:00:01"}],'
        ' "introduces": null, "promises": [{"who": "u2", "what": "lunch"}]}'
    )
    assert loose.introduces == [] and loose.promises == ["u2, lunch"] and listed('["f1"]') == ["f1"]
    seeds = SeedPacket(examples=[example(f"e{i}") for i in range(5)])
    assert excerpts(seeds, "1:s") == excerpts(seeds, "1:s") and len(excerpts(seeds, "1:s")) == 2
    corpus = root / "corpus"
    atomic_json(
        corpus / f"old/artifacts/{digest(catalog.model_dump(mode='json'))}.json",
        catalog.model_dump(mode="json"),
    )
    atomic_json(corpus / "old/state.json", {"catalog": digest(catalog.model_dump(mode="json"))})
    assert used_names(corpus, corpus / "new") == {
        "companies": ["Harbor"],
        "people": ["Alicia Rao", "Owen Sato"],
    }
    assert used_names(corpus, corpus / "old") == {"companies": [], "people": []}
    print("PASS time model, plan/scene contracts, frozen scenes, assembly, writer brief, corpus names")


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


def check_config(root):
    """Seed packet bounds, resume refuses a changed configuration, and only code-free agents skip the VM."""
    packet = SeedPacket(examples=[example("e0")])
    for update in ({"messages": [{"text": "x"}] * 102}, {"messages": [{"text": "   "}]}, {"rows": [1, 0]}):
        fails(SeedPacket.model_validate, {"examples": [dict(packet.examples[0].model_dump(), **update)]})
    fails(SeedPacket, examples=packet.examples * 2)
    root.mkdir(parents=True)
    path = root / "seeds.json"
    path.write_text(packet.model_dump_json())
    assert load_seed_packet(path) == packet
    (root / "oversized.json").write_bytes(b" " * (SEED_MAX_BYTES + 1))
    fails(load_seed_packet, root / "oversized.json")
    assert SeedDataConfig(path=Path("data/seeds/x.json")).path == ROOT / "data/seeds/x.json"
    base = Config(sector="IT", task_count=2, group_size=1, output=root / "baseline")
    assert base.env.solver.runtime.type == "subprocess", "an agent that executes no code needs no VM"
    local = PipelineConfig().judge.model_copy(update={"runtime": base.env.writer.runtime})
    fails(Config, sector="IT", task_count=2, output=root / "x", env=PipelineConfig(judge=local))
    seeded = base.model_copy(update={"seed_data": SeedDataConfig(path=path)})
    fails(provenance, seeded)
    TestStore(root / "seeded", provenance(seeded, packet)).close()
    changed = packet.model_copy(deep=True)
    changed.examples[0].messages[0].text = "Changed source"
    fails(Store, root / "seeded", provenance(seeded, changed))
    print("PASS config: seed bounds, provenance-checked resume, VM rule")


def seed_person(i):
    return SeedPersona(
        uuid=f"p{i}",
        name=f"Person {i}",
        sex="Female",
        age=30 + i,
        marital_status="never_married",
        education_level="bachelors",
        bachelors_field="stem",
        occupation="accountant_or_auditor" if i % 2 else "software_developer",
        city="Austin",
        state="TX",
        country="USA",
        timezone="America/Chicago",
        persona="Curious and organized.",
        professional_persona="Careful reviewer.",
        cultural_background="Grew up in Texas.",
        skills=["Python"],
        hobbies=["Hiking"],
    )


def check_cast(root):
    """A seeded cast: the premise names occupations, the seed draws unused candidates from them with real typing,
    and people keep a candidate's identity."""
    root.mkdir(parents=True)
    (root / "personas.jsonl").write_text("\n".join(seed_person(i).model_dump_json() for i in range(10)))
    typing = Typing(
        id="t",
        messages=30,
        median_words=9,
        short_share=0.2,
        question_share=0.3,
        lowercase_share=0.4,
        emoji_share=0,
    )
    (root / "typing.jsonl").write_text("\n".join(typing.model_dump_json() for _ in range(10)))
    personas = PersonasConfig(path=root / "personas.jsonl", typing=root / "typing.jsonl", pool=4)
    settings = Config(sector="IT", task_count=2, group_size=1, output=root / "run", personas=personas)
    used = {"companies": [], "people": []}
    supply = synthesizer.premise_context(settings, used)
    assert supply["country"] == "USA" and supply["occupations"] == {
        "software_developer": 5,
        "accountant_or_auditor": 5,
    }
    staffed = premises().model_copy(deep=True)
    for p in staffed.premises:
        p.occupations = ["software_developer"]
    premise = synthesizer.parse_premise(staffed.model_dump_json(), settings, used)
    for occupations, pool in ([], 4), (["pilot"], 4), (["software_developer"], 6):
        wrong = staffed.model_copy(deep=True)
        wrong.premises[0].occupations = occupations
        short = settings.model_copy(update={"personas": personas.model_copy(update={"pool": pool})})
        fails(synthesizer.parse_premise, wrong.model_dump_json(), short, used)
    cast = pick_cast(personas, 3, ["Person 0"], premise.occupations)
    assert cast == pick_cast(personas, 3, ["Person 0"], premise.occupations) and all(p.typing for p in cast)
    assert {p.occupation for p in cast} == {"software_developer"}, (
        "people come from the company's occupations"
    )
    assert len(cast) == 4 and "Person 0" not in {p.name for p in cast}, "people used elsewhere are not drawn"
    fails(pick_cast, personas.model_copy(update={"pool": 5}), 3, ["Person 0"], premise.occupations)
    catalog, _ = fixture()
    catalog.workspace_id = "workspace-0"
    catalog.people = [u.model_copy(update={"name": p.name}) for u, p in zip(catalog.people, cast)]
    catalog.personas = [
        Persona(
            id=u.id, role="engineer", seniority="senior", timezone=p.timezone, voice="terse", seed_id=p.uuid
        )
        for u, p in zip(catalog.people, cast)
    ]
    state = SimpleNamespace(
        cast=cast, premise=premise.model_copy(update={"company": "Harbor"}), catalog=None, feedback=""
    )
    seeded = synthesizer.check_catalog(catalog.model_dump_json(), settings, state, used)
    assert [p.profile for p in seeded.personas] == cast[:2], "code attaches each chosen candidate's profile"
    # Unknown candidate, one candidate twice, a changed timezone, a changed name.
    for update, name in (
        ({"seed_id": "p99"}, cast[0].name),
        ({"seed_id": cast[1].uuid}, cast[1].name),
        ({"timezone": "Europe/Warsaw"}, cast[0].name),
        ({}, "Someone Else"),
    ):
        wrong = catalog.model_copy(deep=True)
        wrong.personas[0] = wrong.personas[0].model_copy(update=update)
        wrong.people[0].name = name
        fails(synthesizer.check_catalog, wrong.model_dump_json(), settings, state, used)
    assert "profile" not in Persona.model_json_schema()["properties"], "profiles are code-written"
    cards = synthesizer.catalog_context(settings, state, used)["candidates"]
    assert [c["seed_id"] for c in cards] == [p.uuid for p in cast] and cards[0]["typing"]["median_words"] == 9
    plan = plan_fixture(seeded)
    shown = json.loads(
        brief(
            seeded,
            premises().premises[0],
            plan,
            plan.scenes[0],
            observed(seeded, plan, {}, plan.scenes[0]),
            [],
            None,
            "English",
            {},
        )
    )["participants"][0]
    assert shown["typing"]["short_share"] == 0.2 and shown["profile"]["home"] == "Austin, TX"
    assert "seed_id" not in shown, "writers see the person, not the seed record"
    print("PASS cast: seeded draw, corpus names, candidate identity, code-written profiles, writer portrait")


async def check_tool_servers():
    """Private data stays host-side; tools serve the reader's view; the solver sees no answer."""
    catalog, candidate = fixture()
    # A group's judges see its scoped catalog: g2 has no built tasks yet.
    scoped = catalog.model_copy(update={"tasks": catalog.tasks[:1]})
    payload = review_payload(scoped, candidate, ["t1"], "world")
    solvers = [
        SolverTask.create(t, candidate.snapshot, "ws1", reference_for(t, candidate)) for t in catalog.tasks
    ]
    assert solvers[0].config.tools.actor_id != solvers[1].config.tools.actor_id
    assert not {"answer", "fact_ids", "snapshot_json"} & set(solvers[0].data.model_dump())
    review = JudgeTask.create(payload)
    for task, data, tools in (
        (solvers[0], candidate.snapshot.model_dump(mode="json"), 5),
        (review, payload, 2),
    ):
        stage_tool_data(task, task.config.tools, data)
        try:
            assert len(task.config.tools.model_dump_json()) < 2000 and not task.config.tools.colocated
            (toolset,) = task.toolsets(task.config)
            await toolset.setup()
            stored = {"state": ReadState()}

            async def pull(stored=stored):
                await asyncio.sleep(0)
                return stored["state"].model_copy(deep=True)

            async def push(before, stored=stored, toolset=toolset):
                await asyncio.sleep(0)
                stored["state"] = toolset.state.model_copy(deep=True)

            toolset._pull_state, toolset._push_state = pull, push
            if task is review:
                await toolset._with_state(toolset.check)()
                call = toolset._with_state(toolset.read)
                await asyncio.gather(
                    *(call(actor_id="u1", action="get_user", arguments={"user_id": "u1"}) for _ in range(20))
                )
                assert stored["state"].checked
            else:
                call = toolset._with_state(toolset.get_user)
                await asyncio.gather(*(call(user_id="u1") for _ in range(20)))
            assert len(stored["state"].reads) == 20, "concurrent reads are all recorded"
            async with (
                asyncio.timeout(30),
                serve(toolset) as url,
                streamable_http_client(url) as (reader, writer_),
                ClientSession(reader, writer_) as client,
            ):
                await client.initialize()
                assert len((await client.list_tools()).tools) == tools
        finally:
            task._tool_directory.cleanup()
    print("PASS tool servers: host-side private data, reader scope, recorded reads, discovery")


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
        trace = SimpleNamespace(ok=True, info={}, to_record=lambda: {"nodes": []})
        yield SimpleNamespace(trace=trace)


async def check_flow(root):
    """One scripted run through every feedback loop, then publish, the release wire, and interrupt/resume."""
    catalog, _ = fixture()
    catalog.personas = personas(catalog)
    # Background of g2's workstream that no task uses.
    catalog.facts.append(catalog.facts[1].model_copy(update={"id": "f3", "subject": "work-3"}))
    written, turns, reviewed, feedback = [], [], [], {}

    class ScriptedEnv(GenerationEnv):
        async def author_turn(self, interaction, runtime, context, attempt, first):
            turns.append((attempt, context["phase"], bool(context.get("feedback"))))
            feedback.setdefault(attempt, context.get("feedback"))
            if context["phase"] == "premise":
                return premises().model_dump_json()
            if context["phase"] == "catalog":
                return catalog.model_copy(update={"company": context["premise"]["company"]}).model_dump_json()
            if context["phase"] == "plan":
                facts = {f["id"] for f in context["catalog"]["facts"]}
                assert facts & {"f2", "f3"} == ({"f2", "f3"} if context["group_id"] == "g2" else set()), (
                    "a group sees only its own workstreams' facts"
                )
                earlier = {s.id: s for s in (self.store.state.plan.scenes if self.store.state.plan else [])}
                full = plan_fixture(catalog)
                scenes = [earlier.get(s.id, s) for s in full.scenes if {b.fact_id for b in s.beats} <= facts]
                plan = full.model_copy(update={"scenes": scenes})
                plans = [a for a, phase, _ in turns if phase == "plan"]
                if len(plans) == 1 or plans.count("build-g2-01") == 1:
                    plan = plan.model_copy(deep=True)
                    if context["group_id"] == "g1":
                        plan.scenes[0].conversation_id = "c_missing"
                    else:
                        plan.scenes[0].situation = "Rewritten history"
                return plan.model_dump_json()
            # The run's first bind misses its evidence; the validation error comes back inside the attempt.
            misses = not any(phase == "bind" and fed for _, phase, fed in turns)
            bindings = [
                Binding(
                    task_id=f"t{i}",
                    claims=[ClaimEvidence(claim_index=0, message_ids=context["conveyed"][f"f{i}"])],
                    gold_calls=[
                        ReadCall(
                            action="search_messages",
                            # Group g2 re-binds t1 with a page limit: a changed binding.
                            arguments={"query": "lunch" if misses else f"work-{i}"}
                            | ({"limit": 20} if context["group_id"] == "g2" and i == 1 else {}),
                        )
                    ],
                )
                for i in (int(task_id[1:]) for task_id in context["required_task_ids"])
            ]
            return BindOutput(bindings=bindings).model_dump_json()

        async def write_scene(self, agents, plan, scene, observed, previous, timely):
            written.append(scene.id)
            return scripted_scene(scene, observed["recent"], previous)

        async def attempt(self, agents, task, candidate):
            result = {
                "task_id": task.id,
                "semantic_correctness": 1.0,
                "execution_ok": True,
                "grounded": True,
                "read_count": 1,
            }
            return result, SimpleNamespace(to_record=lambda: {"nodes": []})

        async def review(self, agents, payload, attempt, files=None, label=""):
            phase, ids = payload["phase"], [t["id"] for t in payload["tasks"]]
            assert all(isinstance(record, dict) for record in (files or {}).values()), (
                "trace records by file name"
            )
            reviewed.append((attempt, phase, ids))
            if attempt.startswith("build-g1"):
                assert {t["group_id"] for t in payload["catalog"]["tasks"]} == {"g1"}, (
                    "judges see built tasks only"
                )
                assert [g["id"] for g in payload["catalog"]["groups"]] == ["g1"], "and only built groups"
                assert [g["id"] for g in payload["later_groups"]] == ["g2"], "later groups are named as later"
            verdict = approval(payload)
            if phase == "task":
                assert sorted(files) == sorted(f"solver_{t}_{n}.json" for t in ids for n in (1, 2, 3, 4)), (
                    "one judge sees four solves of each due task"
                )
                assert {r["task_id"] for r in payload["solves"]} == set(ids), "only the due tasks' solves"
            if (attempt, phase) == ("build-g1-02", "world"):
                assert payload["changed_messages"] and payload["previous_issues"], (
                    "a repair is reviewed with focus"
                )
            if (attempt, phase) == ("build-g2-02", "task"):
                assert payload["previous_issues"], "a rebind is judged against the previous issue"
            if (attempt, phase) == ("build-g2-01", "world"):
                # Minor issues are reported, not routed: the task review's bindings issue alone decides a rebind.
                minor = [
                    Issue(
                        owner="builder",
                        artifact="workspace",
                        message_ids=[chat],
                        defect="terse",
                        requested_change="x",
                    )
                    for chat in ["m" + digest(["s_chat", 0])[:10]]
                ] + [Issue(owner="synthesizer", artifact="catalog", defect="wording", requested_change="x")]
                return verdict.model_copy(
                    update={"issues": [i.model_copy(update={"blocking": False}) for i in minor]}
                )
            if (attempt, phase) == ("build-g2-01", "task"):
                leak = Issue(
                    owner="builder",
                    artifact="bindings",
                    task_ids=["t2"],
                    defect="leak",
                    requested_change="fix",
                )
                return verdict.model_copy(update={"approved": False, "issues": [leak]})
            if attempt == "catalog-01" or (attempt, phase) == ("build-g1-01", "world"):
                upstream = attempt.startswith("catalog")
                issue = Issue(
                    owner="synthesizer" if upstream else "builder",
                    artifact="catalog" if upstream else "workspace",
                    task_ids=["t1"],
                    message_ids=[] if upstream else ["m" + digest(["s_chat", 0])[:10]],
                    defect="Duplicated question" if upstream else "Repetitive activity",
                    requested_change="Repair the identified defect",
                )
                return verdict.model_copy(
                    update={"approved": False, "issues": [issue], "summary": issue.defect}
                )
            return verdict

    config = Config(sector="IT", task_count=2, group_size=1, output=root, seed=1, corpus=root / "corpus")
    store = TestStore(root, {"test": 1})
    store.state.catalog = catalog
    agents = SimpleNamespace(synthesizer=ScriptedAgent(), builder=ScriptedAgent())
    env = ScriptedEnv(config, store)
    await env.run(None, agents)
    assert store.state.phase == "done" and "final" in store.state.reviews
    solve = env.solver_task(catalog.tasks[0], store.state.candidate)
    resolve_runtime_config(config.env.solver.runtime, solve)  # the configured solver runtime accepts its task
    assert store.state.rounds == {"catalog": 2, "build:g1": 2, "build:g2": 2, "final": 1}, (
        "catalog repair once"
    )
    first = [(phase, fed) for attempt, phase, fed in turns if attempt == "build-g1-01"]
    assert first[:2] == [("plan", False), ("plan", True)] and ("bind", True) in first, (
        "plan and bind errors are corrected inside the attempt"
    )
    assert written == ["s_route", "s_chat", "s_chat", "s_private"], (
        "each group writes only its new or repaired scenes"
    )
    assert "Repetitive activity" in store.state.plan.scenes[2].revision_note, (
        "a repair carries the judge's words"
    )
    assert json.loads(feedback["build-g1-02"])["summary"] == "Repetitive activity", (
        "feedback carries only the rejecting reviews' words"
    )
    events = [json.loads(line) for line in (root / "progress.jsonl").read_text().splitlines()]
    assert [(e["attempt"], e["route"]) for e in events if e["event"] == "attempt_routed"] == [
        ("build-g1-01", "plan"),
        ("build-g1-02", "repair"),
        ("build-g2-01", "plan"),
        ("build-g2-02", "rebind"),
    ], "each build attempt logs its route"
    assert not any(a == "build-g1-02" and phase == "plan" for a, phase, _ in turns), "a repair keeps the plan"
    assert [(p, f) for a, p, f in turns if a == "build-g2-01"][:2] == [("plan", False), ("plan", True)], (
        "a later group cannot change the approved world's scenes"
    )
    assert [(p, f) for a, p, f in turns if a == "build-g2-02"] == [("bind", True)], (
        "bindings-only issues rebind"
    )
    assert ("build-g1-02", "task", ["t1"]) not in reviewed, "an unchanged task is not judged again"
    assert ("build-g2-01", "task", ["t1", "t2"]) in reviewed, (
        "a changed binding re-judges its task with the new one"
    )
    assert ("build-g2-02", "task", ["t2"]) in reviewed, "a rejected task review is never reused"
    assert [(p, i) for a, p, i in reviewed if a == "final-01"] == [("world", [])], "final reuses task reviews"
    summary = store.summary("check")
    assert (
        summary["solves"] == 8
        and summary["solved_tasks"] == 2
        and summary["solve_rates"] == {"t1": 1.0, "t2": 1.0}
    )
    kept = store.state.task_reviews.pop("t2")
    fails(store.publish)
    store.state.task_reviews["t2"] = kept
    store.publish()
    world, rows, answers = load_release(root / "release")
    assert world == store.state.candidate.snapshot and len(rows) == 2 and "answer" not in rows[0].model_dump()
    taskset = SlackTaskset(SlackTasksetConfig(task=EvaluationConfig(release_dir=root / "release")))
    task = next(iter(taskset))
    rebuilt = EvaluationTask(SlackTaskData.model_validate_json(task.data.model_dump_json()), task.config)
    await rebuilt.validate(None)
    assert rebuilt.solver_task().config.reference == answers[task.data.task_id]
    fails(EvaluationTask(rebuilt.data.model_copy(update={"snapshot_hash": "wrong"}), task.config).solver_task)
    judged = []

    async def flaky(task):
        judged.append(task)
        verdict = approval(json.loads(task.config.tools.payload_json)).model_dump(mode="json")
        return SimpleNamespace(ok=len(judged) > 1, errors=[], info={"verdict": verdict})

    payload = review_payload(catalog, store.state.candidate, ["t1"], "world")
    verdict = await GenerationEnv.review(
        env, SimpleNamespace(judge=SimpleNamespace(run=flaky)), payload, "judge"
    )
    assert verdict.approved and len(judged) == 2, "a malformed review is rerun once"
    store.reserve("interrupted", 2)
    store.close()
    resumed = TestStore(root, {"test": 1})
    assert resumed.state.active_attempt is None and resumed.state.rounds["interrupted"] == 1
    fails(resumed.reserve, "interrupted", 1)
    before = dict(resumed.state.rounds)
    upstream = Issue(
        owner="synthesizer", artifact="catalog", task_ids=["t2"], defect="scope", requested_change="fix"
    )
    repairing = ScriptedEnv(config, resumed)
    repairing.route_rejection(
        verdict.model_copy(update={"approved": False, "issues": [upstream]}), "build:g2"
    )
    assert (
        resumed.state.phase == "catalog"
        and not resumed.state.reviews
        and resumed.state.built_groups == ["g1"]
    )
    catalog.tasks[1].question += " According to the final approval?"
    await repairing.run(None, agents)
    assert resumed.state.rounds["catalog"] == before["catalog"] + 1, (
        "catalog repairs use the catalog allowance"
    )
    assert [(p, i) for a, p, i in reviewed if a == "build-g2-03"] == [("task", ["t2"])], (
        "a question-only catalog repair re-judges that task and no unchanged scene"
    )
    assert writer.current(resumed.state)
    resumed.state.catalog.facts[0].value += " (revised)"
    assert not writer.current(resumed.state), "a world written against old facts is rebuilt, not re-reviewed"
    resumed.close()
    print(
        "PASS flow: catalog repair, targeted repair, rebind, group scope, frozen scenes, review cache, final, publish, resume"
    )


if __name__ == "__main__":
    check_slack_reads()
    check_replay()
    check_verdicts()
    asyncio.run(check_tool_servers())
    with tempfile.TemporaryDirectory() as directory:
        check_config(Path(directory) / "config")
        check_cast(Path(directory) / "cast")
        check_time_and_scenes(Path(directory) / "scenes")
        asyncio.run(check_flow(Path(directory) / "run"))
    print("All generation checks passed.")
