"""Offline invariant checks, one per rule group the pipeline relies on: uv run --frozen python -m generators.worldgen_slack.check"""

import asyncio
import json
from collections import Counter
from contextlib import asynccontextmanager
from types import SimpleNamespace
import random
import re
import tempfile
import tomllib
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from verifiers.v1.errors import SandboxError
from verifiers.v1.mcp.launch import serve
from worldgen_slack.dataset import PrivateAnswer, PublicTask, load_release, sha256, write_release
from worldgen_slack.db import ANSWER_KEY, World
from worldgen_slack.taskset import SolverTask
from worldgen_slack.tools import SlackTools, WorldToolsConfig, file_hash, stage_world
from .agents.inspection import ReviewState
from .agents.judge import JudgeTask, review_payload
from .agents.synthesizer import parse_premise
from .agents.world import WorldAuthorTask, context_of, files, ledger_digest
from .chronicle import (
    Close,
    Commit,
    Conversation,
    Event,
    Plan,
    PlanFact,
    PostLine,
    add_task,
    advance,
    bounds,
    close_day,
    part_of,
    post,
    posted,
    present,
    quotas,
    record_plan,
    revise,
    start_clock,
    today,
)
from .env import GenerationEnv
from .generate import failure, provenance
from .store import Store
from .config import ROOT, Acceptance, Category, Config
from .contracts import (
    DAY,
    PHASE_CRITERIA,
    TIME_LITERAL,
    Beat,
    Channel,
    Fact,
    Gaps,
    Issue,
    Ledger,
    Line,
    Moment,
    Organization,
    Person,
    Layout,
    PlanError,
    Place,
    placed_lines,
    Reaction,
    Routine,
    Premise,
    Premises,
    Scene,
    ScenePlan,
    SeedPersona,
    Storyline,
    Task,
    TaskReview,
    TaskSet,
    Typing,
    Verdict,
    WrittenScene,
    accepted_task,
    at,
    background_plan,
    channel_id,
    check_lines,
    check_plan,
    check_task,
    deciding,
    direct_messages,
    layout,
    measures,
    organize,
    pick_cast,
    quota,
    record_ledger,
    record_tasks,
    render,
    unplanned,
    user_id,
    validate_verdict,
    verdict_schema,
    window,
    world_meta,
    write_scene,
)


def fails(function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except (ValueError, TypeError, LookupError):
        return
    raise AssertionError("expected rejection")


def world_fixture(path):
    """Three people on Chicago time: #releases (public: U1, U2), #exec (private: U1), a DM (U1, U2)."""
    world = World.create(path)
    zone = ZoneInfo("America/Chicago")

    def at(day, hour, minute=0):
        return int(datetime(2026, 6, 14 + day, hour, minute, tzinfo=zone).timestamp() * 1e6)

    def message(id, channel, user, day, hour, text, parent=None, minute=0):
        return dict(
            id=id, channel_id=channel, user_id=user, ts_us=at(day, hour, minute), text=text, parent_id=parent
        )

    with world.batch():
        world.insert("world_meta", [{"key": "now_us", "value": str(at(4, 0))}])
        world.insert(
            "calendar",
            [
                dict(day=d, date=f"2026-06-{14 + d}", start_us=at(d, 0), end_us=at(d + 1, 0))
                for d in (1, 2, 3)
            ],
        )
        world.insert(
            "users",
            [
                dict(
                    id=u,
                    handle=u.lower(),
                    real_name=n,
                    display_name=n.split()[0],
                    email=f"{u}@example.com",
                    tz="America/Chicago",
                    created_us=at(1, 0) - 1,
                )
                for u, n in (("U1", "Alicia Rao"), ("U2", "Owen Sato"), ("U3", "Mia Chen"))
            ],
        )
        world.insert(
            "channels",
            [
                dict(id="C1", name="releases", type="public", created_us=at(1, 0) - 1),
                dict(id="G1", name="exec", type="private", created_us=at(1, 0) - 1),
                dict(id="D1", name=None, type="im", created_us=at(1, 0) - 1),
            ],
        )
        world.insert(
            "members",
            [
                dict(channel_id=c, user_id=u, joined_us=at(1, 0) - 1)
                for c, u in (("C1", "U1"), ("C1", "U2"), ("G1", "U1"), ("D1", "U1"), ("D1", "U2"))
            ]
            + [dict(channel_id="G1", user_id="U3", joined_us=at(1, 0) - 1, left_us=at(1, 1))],
        )
        world.insert(
            "messages",
            [
                message(1, "C1", "U1", 1, 10, "Release 4.2 rollback: release blocked, rollback decided"),
                message(3, "C1", "U2", 2, 9, "notes posted for release 4.2"),
                message(4, "G1", "U1", 1, 11, "private: rollback budget approved"),
                message(5, "D1", "U2", 1, 12, "can you check the rollback"),
                message(6, "C1", "U1", 1, 11, "deleted rollback note", parent=1) | {"is_deleted": 1},
            ],
        )
        world.insert("messages", [message(2, "C1", "U2", 1, 10, "ack, rolling back now", parent=1, minute=5)])
        world.insert("storylines", [dict(id="r", summary="the 4.2 release", position=1)])
        world.insert(
            "scenes",
            [
                dict(
                    id="s1",
                    channel_id="C1",
                    storyline="r",
                    day=1,
                    part="morning",
                    slot_start_us=at(1, 9),
                    slot_end_us=at(1, 12),
                    situation="decide",
                )
            ],
        )
        world.insert("scene_messages", [dict(message_id=m, scene_id="s1") for m in (1, 2, 6)])
        world.insert(
            "facts",
            [
                dict(
                    id="f1",
                    storyline="r",
                    subject="Release 4.2",
                    attribute="decision",
                    value="rollback",
                    channel_id="C1",
                    author_id="U1",
                    day=1,
                    summary="rollback decided",
                )
            ],
        )
        world.insert("evidence", [dict(fact_id="f1", message_id=1, role="anchor", anchor_token="rollback")])
        world.insert(
            "tasks",
            [
                dict(
                    id="t1",
                    category="lookup",
                    level=1,
                    actor_id="U2",
                    question="What was decided for 4.2?",
                    answer_type="text",
                    gold_source="ledger",
                    gold_sql="SELECT value AS answer FROM facts WHERE id = 'f1'",
                )
            ],
        )
        world.insert("task_facts", [dict(task_id="t1", fact_id="f1")])
    return world, at, message


async def check_world(root):
    """The world file: triggers and rules reject defects at insert and roll back; reads are Slack as one actor."""
    root.mkdir(parents=True)
    world, at, message = world_fixture(root / "world.sqlite")
    assert world.violations(complete=True) == []
    tables = [
        "messages",
        "facts",
        "evidence",
        "reactions",
        "fact_relations",
        "tasks",
        "events",
        "commitments",
    ]

    def rejects(expected, **rows):
        before = [world.db.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in tables]
        try:
            with world.batch():
                for table, items in rows.items():
                    world.insert(table, items)
        except ValueError as error:
            assert expected in str(error), (expected, str(error))
        else:
            raise AssertionError(f"not rejected: {expected}")
        assert before == [world.db.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in tables], (
            "rolled back"
        )

    fact = dict(storyline="r", value="v", summary="s", channel_id="C1", author_id="U1")
    rejects("not a member", messages=[message(9, "C1", "U3", 1, 13, "hi")])
    rejects("earlier top-level", messages=[message(9, "C1", "U1", 1, 13, "re", parent=2)])
    rejects("earlier top-level", messages=[message(9, "C1", "U1", 1, 9, "re", parent=1)])
    rejects("earlier top-level", messages=[message(9, "G1", "U1", 1, 13, "re", parent=1)])
    rejects(
        "member of its channel",
        reactions=[dict(message_id=1, user_id="U3", emoji="eyes", created_us=at(1, 13))],
    )
    rejects(
        "member of its channel",
        reactions=[dict(message_id=1, user_id="U2", emoji="eyes", created_us=at(1, 9))],
    )
    rejects(
        "stated_off_day",
        facts=[fact | dict(id="f2", subject="a", attribute="b", day=1)],
        evidence=[dict(fact_id="f2", message_id=3, role="anchor")],
    )
    rejects(
        "before_it_happened",
        facts=[
            fact
            | dict(
                id="f2",
                subject="a",
                attribute="b",
                day=1,
                moment_us=at(1, 15),
                moment_zone="UTC",
                moment_kind="happened",
            )
        ],
        evidence=[dict(fact_id="f2", message_id=4, role="anchor")],
    )
    rejects(
        "out_of_order",
        facts=[fact | dict(id="f2", subject="a", attribute="b", day=1)],
        fact_relations=[dict(src_fact="f1", dst_fact="f2", kind="after")],
        evidence=[dict(fact_id="f2", message_id=4, role="anchor")],
    )
    rejects(
        "anchor_token_missing",
        evidence=[dict(fact_id="f1", message_id=3, role="supporting", anchor_token="budget")],
    )
    rejects(
        "ambiguous_value",
        facts=[fact | dict(id="f2", subject="release 4.2 ", attribute="Decision", value="hold", day=1)],
    )
    rejects(
        "unreadable_evidence",
        facts=[fact | dict(id="f2", subject="a", attribute="b", day=1)],
        evidence=[dict(fact_id="f2", message_id=4, role="anchor")],
        task_facts=[dict(task_id="t1", fact_id="f2")],
    )
    rejects("future_message", messages=[message(9, "C1", "U1", 4, 1, "late")])
    rejects(
        "outside_slot",
        messages=[message(9, "C1", "U1", 1, 13, "late")],
        scene_messages=[dict(message_id=9, scene_id="s1")],
    )
    window_event = dict(id="e1", title="release window", moment_us=at(2, 10), zone="America/Chicago")
    on_event = fact | dict(subject="a", attribute="b", day=1, event_id="e1", moment_zone="America/Chicago", moment_kind="scheduled")  # fmt: skip
    rejects("event_moment", events=[window_event], facts=[on_event | dict(id="f2", moment_us=at(2, 11))])
    # The fixture was written storyline by storyline: message 4 (day 1) has a later id than message 3 (day 2).
    rejects("written_out_of_order", world_meta=[{"key": "chronological", "value": "1"}])
    rejects(
        "commitment_order",
        commitments=[
            dict(
                id="c1",
                owner_id="U2",
                text="post notes",
                message_id=3,
                due_us=at(3, 0),
                status="kept",
                closed_by=1,
            )
        ],  # fmt: skip
    )
    with world.batch():  # facts about one event share its moment; a promise is closed after it is made
        world.insert("events", [window_event])
        world.insert("facts", [on_event | dict(id="f8", moment_us=at(2, 10))])
        world.insert("commitments", [dict(id="c1", owner_id="U1", text="post notes", message_id=1, due_us=at(3, 0), status="kept", closed_by=3)])  # fmt: skip
    world.db.execute("DELETE FROM commitments")
    world.db.execute("DELETE FROM facts WHERE id = 'f8'")
    world.db.execute("DELETE FROM events")
    with world.batch():  # a supersedes chain disambiguates transitively; a reaction after its message is fine
        world.insert(
            "facts",
            [
                fact | dict(id=f"f{i}", subject="Release 4.2", attribute="decision", value=f"v{i}", day=1)
                for i in (2, 3)
            ],
        )
        world.insert(
            "fact_relations",
            [
                dict(src_fact="f2", dst_fact="f1", kind="supersedes"),
                dict(src_fact="f3", dst_fact="f2", kind="supersedes"),
            ],
        )
        world.insert("reactions", [dict(message_id=1, user_id="U2", emoji="eyes", created_us=at(1, 13))])
    with world.batch():  # a planned task fact is a completion defect, not an insert defect
        world.insert("facts", [fact | dict(id="f4", subject="c", attribute="d", day=2)])
        world.insert("task_facts", [dict(task_id="t1", fact_id="f4")])
    assert [r for r, _ in world.checks(complete=True)] == ["unstated_fact"] and world.violations() == []
    assert (
        world.checks(complete=["t1"]) == world.checks(complete=True) and world.checks(complete=["t2"]) == []
    ), "completion is due only for the named tasks"

    world.solver_copy(root / "solver.sqlite")
    solver = World(root / "solver.sqlite", actor="U2")
    names = {r[0] for r in solver.db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert not names & set(ANSWER_KEY) and {"messages", "messages_fts"} <= names

    def texts(reader, query, **kwargs):
        return [i["text"] for i in reader.search_messages(query, **kwargs)["items"]]

    u1, u3 = World(root / "solver.sqlite", actor="U1"), World(root / "solver.sqlite", actor="U3")
    fails(solver.read_channel, "G1")
    fails(u3.read_channel, "G1")
    assert u3.list_user_channels()["items"] == [] and texts(solver, "deleted") == []
    assert texts(solver, "rollback budget") == [] and texts(u1, "rollback budget") == [
        "private: rollback budget approved"
    ]
    assert "can you check the rollback" in texts(
        solver, "rollback"
    ) and "can you check the rollback" not in texts(u3, "rollback")
    assert [c["id"] for c in solver.list_user_channels()["items"]] == ["D1", "C1"]
    assert solver.list_user_channels(types="im")["items"][0]["users"] == ["U1"]
    assert [m["text"] for m in u3.read_channel("C1")["items"]] == [
        "notes posted for release 4.2",
        texts(u3, "blocked")[0],
    ]
    assert u3.read_channel("C1")["items"][1]["reply_count"] == 1
    root_ts = u3.read_channel("C1")["items"][1]["ts"]
    assert [m["text"] for m in u3.read_thread("C1", root_ts)["items"]][1] == "ack, rolling back now"
    assert texts(solver, "release")[0].startswith("Release 4.2 rollback"), (
        "BM25 ranks the denser, older match first"
    )
    assert texts(solver, "release", sort="timestamp")[0] == "notes posted for release 4.2"
    assert texts(solver, '"rolling back"') == ["ack, rolling back now"]
    assert (
        texts(solver, "release from:@u2")
        == ["notes posted for release 4.2"]
        == texts(solver, "4.2 in:#releases on:2026-06-16")
    )
    assert texts(solver, "4.2 before:2026-06-16") == texts(solver, "blocked") and texts(
        solver, "4.2 after:2026-06-15"
    ) == texts(solver, "notes")
    assert texts(solver, "rollback in:#exec") == []
    page = solver.search_messages("release", limit=1)
    assert (
        solver.search_messages("release", cursor=page["next_cursor"], limit=1)["items"][0]["text"]
        == "notes posted for release 4.2"
    )
    fails(solver.search_messages, "rollback", cursor=page["next_cursor"])
    fails(solver.search_messages, "release", limit=101)
    assert solver.get_reactions("C1", root_ts)["reactions"] == [{"name": "eyes", "count": 1, "users": ["U2"]}]
    assert solver.get_user("U1")["name"] == "u1" and solver.search_users("ali")["items"][0]["id"] == "U1"
    assert solver.search_channels("e")["items"][0]["name"] == "releases", "private channels stay unlisted"

    fails(stage_world, WorldToolsConfig(actor_id="U2", colocated=True), root / "solver.sqlite")
    config = WorldToolsConfig(actor_id="U2")
    stage_world(config, root / "solver.sqlite")
    toolset = SlackTools(config)
    await toolset.setup()
    stored = {"state": toolset._state_cls()}

    async def pull():
        return stored["state"].model_copy(deep=True)

    async def push(before):
        stored["state"] = toolset.state.model_copy(deep=True)

    toolset._pull_state, toolset._push_state = pull, push
    search = toolset._with_state(toolset.search_messages)
    await asyncio.gather(*(search(query="rollback") for _ in range(10)))
    assert len(stored["state"].calls) == 10 and stored["state"].calls[0].tool == "search_messages"
    async with (
        asyncio.timeout(30),
        serve(toolset) as url,
        streamable_http_client(url) as (reader, writer_),
        ClientSession(reader, writer_) as client,
    ):
        await client.initialize()
        assert len((await client.list_tools()).tools) == 9
    bad = WorldToolsConfig(actor_id="U2", db_path=config.db_path, db_hash="0" * 64)
    try:
        await SlackTools(bad).setup()
        raise AssertionError("a changed world file was served")
    except ValueError:
        pass
    print(
        "PASS world: triggers, rules with rollback, completion, solver copy, visibility, search, paging, tools"
    )


def check_tasks(root):
    """Gold queries run as their actor: what the actor cannot see does not exist for them, nothing but a single
    read is allowed, returned evidence must be readable, and a document is applied on a trial copy."""
    root.mkdir(parents=True)
    world, at, message = world_fixture(root / "world.sqlite")

    def answers(actor, sql, **kwargs):
        return [r["answer"] for r in world.gold(actor, sql, **kwargs)["rows"]]

    every = "SELECT text AS answer FROM messages ORDER BY ts_us"
    assert "private: rollback budget approved" not in answers("U2", every)
    assert "private: rollback budget approved" in answers("U1", every)
    assert "deleted rollback note" not in answers("U1", every)
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
    assert len(world.gold("U1", every, max_rows=2)["rows"]) == 3, "one row past the cap shows the cap was hit"
    settings = contracts_settings(root)
    who = "SELECT u.real_name AS answer, m.id AS message_id FROM messages m JOIN users u ON u.id = m.user_id"

    def lookup(sql):
        return Task(id="tw", category="lookup", level=1, actor_id="U2", question="Who answered the decision?", answer_type="text", gold_sql=sql)  # fmt: skip

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
        ):  # 50 newer top-level messages push the evidence to read_channel's second page
            with copy.batch():
                copy.insert("messages", [dict(channel_id="C1", ts_us=at(1, 11, 7) + 1009 * (i + 1), user_id="U1", text=f"note {i}") for i in range(50)])  # fmt: skip
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
    assert world.select("SELECT local(ts_us) AS t FROM messages WHERE id = 1")["rows"] == [{"t": "2026-06-15 15:00 UTC"}]  # fmt: skip
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
    fails(World(world.path, actor="U1").checks)

    scoped = World(world.path, actor="U2")
    assert scoped.rank("rollback release", [1]) == 1 and scoped.rank("zebra", [1]) is None
    try:
        with world.trial() as copy:
            copy.clear_scene("s1")
            assert copy.db.execute("SELECT COUNT(*) FROM messages WHERE id IN (1, 2, 6)").fetchone()[0] == 0
            raise ValueError("drop the trial")
    except ValueError:
        pass
    assert world.db.execute("SELECT COUNT(*) FROM messages WHERE id IN (1, 2)").fetchone()[0] == 2
    with world.trial() as copy:
        copy.clear_scene("s1")
    assert world.db.execute("SELECT COUNT(*) FROM messages WHERE id IN (1, 2)").fetchone()[0] == 0
    assert world.db.execute("SELECT COUNT(*) FROM scenes").fetchone()[0] == 0
    assert [name for name, _ in world.checks(complete=True)] == ["unstated_fact"], (
        "the cleared anchor is missed"
    )
    print(
        "PASS tasks: gold as its actor, reads only, readable evidence, time limit, rank, trial, scene replace"
    )


def seed_person(i, zone="America/Chicago"):
    names = ["Alicia Rao", "Owen Sato", "Mia Chen", "Raj Patel", "Lena Ortiz", "Tom Berg"]
    return SeedPersona(
        uuid=f"p{i}", name=names[i], sex="F", age=30 + i, marital_status="single", education_level="bachelors",
        bachelors_field="cs", occupation="software_developer", city="Austin", state="TX", country="USA",
        timezone=zone, persona="calm", professional_persona="engineer", cultural_background="midwest",
        skills=["go"], hobbies=["chess"],
        typing=Typing(id=f"t{i}", messages=40, median_words=8, short_share=0.2, long_share=0.1, question_share=0.2, lowercase_share=0.3, emoji_share=0.1),
    )  # fmt: skip


def contracts_settings(root):
    gaps = root / "gaps.json"
    gaps.write_text(json.dumps({"quantiles": [i * 7.2 for i in range(1001)]}) + "\n")
    spec = tomllib.loads((ROOT / "configs/worldgen_slack/worldgen.toml").read_text())
    return Config.model_validate(
        {
            "sector": "software",
            "seed": 3,
            "output": str(root / "run"),
            "personas": {"path": str(root / "p.jsonl"), "typing": str(root / "t.jsonl"), "gaps": str(gaps)},
            "tasks": {
                "count": 4,
                "per_storyline": 2,
                "max_answer_rows": 2,
                "styles": spec["tasks"]["styles"],
            },
            "taxonomy": spec["taxonomy"],
        }
    )


def check_contracts(root):
    """Documents become rows only through deterministic rules: the organization, the ledger and its tasks (T2-T6),
    the scene plan, the written scene; time is code's (calendar, parts, gaps, {at:} rendering, no literal times)."""
    root.mkdir(parents=True)
    settings = contracts_settings(root)
    styles = settings.tasks.styles
    cells = quota(settings.taxonomy, styles, 3, 17)
    order = [c[:2] for c in cells]
    assert cells == quota(settings.taxonomy, styles, 3, 17) and order != [c[:2] for c in quota(settings.taxonomy, styles, 4, 17)]  # fmt: skip
    assert len({(c, level) for c, level, *_ in cells}) == 17, "every (category, level) once"
    assert all(concept in settings.taxonomy[c].concepts[level - 1] and style in styles for c, level, concept, style in cells), "concepts come from their own level"  # fmt: skip
    for wrong in ({"concepts": [["x"]]}, {"spread": [1]}, {"concepts": [["x"], [], ["y"]]}):
        fails(Category.model_validate, settings.taxonomy["semantic"].model_dump() | wrong)
    jobs = ["software_developer"] * 3 + ["accountant_or_auditor"] * 2 + ["manager"]
    people = [seed_person(i).model_copy(update={"occupation": job}) for i, job in enumerate(jobs)]
    (root / "p.jsonl").write_text("".join(p.model_dump_json() + "\n" for p in people))
    (root / "t.jsonl").write_text("".join(p.typing.model_dump_json() + "\n" for p in people))
    personas = settings.personas.model_copy(update={"pool": 3})
    staffing = {"software_developer": 2, "accountant_or_auditor": 1}
    drawn = pick_cast(personas, 3, ["Owen Sato"], staffing)
    assert Counter(p.occupation for p in drawn) == staffing and "Owen Sato" not in {p.name for p in drawn}
    assert drawn == pick_cast(personas, 3, ["Owen Sato"], staffing), "the seed decides the cast"
    fails(pick_cast, personas, 3, ["Owen Sato", "Alicia Rao"], staffing)  # one unused developer is left

    def premises(staffing):  # raw, as the synthesizer writes it
        return json.dumps({"premises": [dict(company=f"{w} Co", niche="n", region="r", size="s", culture="c", cast="x", staffing=s) for w, s in (("Birch", {"software_developer": 2, "manager": 1}), ("Cedar", staffing))]})  # fmt: skip

    hiring = settings.model_copy(update={"personas": personas, "premise_count": 2})
    used = {"companies": [], "people": ["Owen Sato"]}
    assert parse_premise(premises(staffing), hiring, used).staffing in (
        staffing,
        {"software_developer": 2, "manager": 1},
    )
    for bad in (
        {"software_developer": 2},  # adds up to 2, not 3
        {"software_developer": 2, "lawyer": 1},  # not an occupation of the persona file
        {"software_developer": 3},  # only 2 unused developers
        {"software_developer": 2, "manager": 1, "accountant_or_auditor": 0},
    ):
        fails(parse_premise, premises(bad), hiring, used)

    cast = [seed_person(i, "America/New_York" if i == 3 else "America/Chicago") for i in range(4)]
    a, b, c, d = (user_id(p.uuid) for p in cast)
    premise = Premise(
        company="Lattice Creek", niche="n", region="r", size="s", culture="c", cast="x", staffing={}
    )
    world = World.create(root / "world.sqlite")

    routines = [
        Routine(kind=k, probability=p) for k, p in (("standup", 0.5), ("review", 0.3), ("lunch", 0.2))
    ]

    def org(people=(a, b, c, d), channels=None, dm_routines=routines):
        return Organization(
            people=[Person(user_id=u, title="Engineer", team="Platform") for u in people],
            channels=channels
            or [
                Channel(name="ops", type="public", members=[a, b, c, d], routines=routines),
                Channel(name="leads", type="private", members=[a, b], routines=routines),
                Channel(type="im", members=[a, c]),
            ],
            dm_routines=dm_routines,
        )

    for bad in (
        org(people=(a, a, b, c, d)),
        org(people=(a, b, c, d, "Uxyz")),
        org(
            people=(a, b, c),
            channels=[Channel(name="ops", type="public", members=[a, b, c], routines=routines)],
        ),  # d left out  # fmt: skip
        org(channels=[Channel(name="ops", type="public", members=[a, b, c, d], routines=routines[:2])]),
        org(dm_routines=routines[:2]),
        org(channels=[Channel(type="im", members=[a, b, c])]),
    ):
        try:
            with world.trial() as copy:
                organize(copy, bad, cast, premise, settings)
            raise AssertionError("organization accepted")
        except ValueError:
            pass
    assert world.db.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0, (
        "a failed trial leaves the world untouched"
    )
    with world.trial() as copy:
        organize(copy, org(), cast, premise, settings)
    dms = [sorted(m.members) for m in direct_messages(org(), settings)]
    assert dms == [sorted(m.members) for m in direct_messages(org(), settings)] and len(dms) == 1, (
        "code tops the author's DM up to dms_per_person, drawn by the seed"
    )
    assert sorted([a, c]) not in dms and world.db.execute("SELECT COUNT(*) FROM channels WHERE type = 'im'").fetchone()[0] == 2  # fmt: skip
    users = {r["id"]: r for r in world.db.execute("SELECT * FROM users")}
    assert users[a]["tz"] == "America/Chicago" and users[a]["display_name"] == "Alicia"
    assert (
        users[a]["email"] == "alicia.rao@latticecreek.example"
        and world_meta(world, "zone") == "America/Chicago"
    )
    day1 = world.db.execute("SELECT date FROM calendar WHERE day = 1").fetchone()[0]
    assert (
        date.fromisoformat(day1).weekday() == 0
        and world.db.execute("SELECT COUNT(*) FROM calendar").fetchone()[0] == 10
    )
    ops, leads = channel_id("public", "ops", []), channel_id("private", "leads", [])

    def fact(id, storyline, author, day, channel=ops, value=None, places=None, **extra):
        places = places or [Place(channel_id=channel, author_id=author, probability=1.0)]
        return Fact(id=id, storyline=storyline, subject=id, attribute="state", value=value or f"v-{id}", places=places, day=day, summary="s", **extra)  # fmt: skip

    def cell(category, level):
        return (category, level, "a concept", "a style")

    def ledger(facts=None, tasks=None, storylines=("s1", "s2")):
        return Ledger(
            storylines=[Storyline(id=s, summary=s) for s in storylines],
            facts=facts
            or [
                fact("f1", "s1", a, 1, anchor="rollback", value="rollback"),
                fact(
                    "f2",
                    "s1",
                    b,
                    2,
                    after=["f1"],
                    happened_at=Moment(day=2, time="14:00", zone="America/Chicago"),
                ),
                fact("f3", "s2", a, 3, channel=leads),
                fact("f4", "s1", a, 1, after=["f1"]),
            ],
            tasks=tasks if tasks is not None else [search],
        )

    def task(
        id="t1",
        category="search",
        sql="SELECT value AS answer FROM facts WHERE id = 'f1'",
        facts=("f1",),
        **extra,
    ):
        return Task(id=id, category=category, level=1, actor_id=c, question="Which way did the release go?", answer_type="text", gold_sql=sql, facts=list(facts)) .model_copy(update=extra)  # fmt: skip

    search = task()
    found = [cell("search", 1)]
    rejected = {
        "storylines": (ledger(storylines=("s1", "s2", "s3")), found),
        "author not a member": (ledger(facts=[fact("f1", "s1", c, 1, channel=leads)], tasks=[]), []),
        "after a later storyline": (
            ledger(facts=[fact("f1", "s1", a, 1, after=["f3"]), fact("f3", "s2", a, 3)]),
            found,
        ),
        "T6 cells": (ledger(), [cell("search", 2)]),
        "T2 sql reads facts": (ledger(tasks=[task(category="lookup")]), [cell("lookup", 1)]),
        "T2 ledger reads no facts": (
            ledger(tasks=[task(sql="SELECT real_name AS answer FROM users LIMIT 1")]),
            found,
        ),
        "T2 hybrid reads no workspace": (ledger(tasks=[task(category="hybrid")]), [cell("hybrid", 1)]),
        "T4 no facts": (ledger(tasks=[task(facts=())]), found),
        "T3 refusal with rows": (
            ledger(
                tasks=[task(category="robustness", answer_type="refusal", sql="SELECT 1 AS answer", facts=())]
            ),
            [cell("robustness", 1)],
        ),
        "T3 too many rows": (
            ledger(tasks=[task(answer_type="set", sql="SELECT value AS answer FROM facts")]),
            found,
        ),
        "T3 text answer of 2 rows": (
            ledger(tasks=[task(sql="SELECT value AS answer FROM facts LIMIT 2")]),
            found,
        ),
        "T3 no answer column": (ledger(tasks=[task(sql="SELECT value FROM facts WHERE id = 'f1'")]), found),
        "T5 giveaway": (ledger(tasks=[task(question="Was it a rollback for the release?")]), found),
        "anchor not in the value": (
            ledger(facts=[fact("f1", "s1", a, 1, anchor="Release 4.2")], tasks=[]),
            [],
        ),
        "T6 four candidates": (
            ledger(tasks=[task(id=f"t{i}", question=f"Which way did release {i} go?") for i in range(4)]),
            found,
        ),  # fmt: skip
        "T6 candidate ids": (ledger(tasks=[task(), task(question="Where did the release go?")]), found),
        "unreadable fact": (
            ledger(tasks=[task(sql="SELECT value AS answer FROM facts WHERE id = 'f3'", facts=("f3",))]),
            found,
        ),
    }
    for name, (bad, cells_) in rejected.items():
        try:
            with world.trial() as copy:
                record_ledger(copy, bad, settings, cells_)
            raise AssertionError(f"ledger accepted: {name}")
        except ValueError:
            pass
    assert world.db.execute("SELECT COUNT(*) FROM facts").fetchone()[0] == 0
    invalid = task(id="bad", sql="SELECT value FROM facts WHERE id = 'f1'")  # no answer column
    rare = task(id="rare", question="Where did the release go?", probability=0.05)
    for candidates, picked, least in (
        ([invalid, task(id="good", question="Where did the release go?")], "good", 20),
        ([task(id="usual", probability=1.0), rare], "rare", 16),
    ):
        chosen = []
        for seed in range(20):  # the run seed picks: only valid candidates, mostly the less likely
            try:
                with world.trial() as copy:
                    record_ledger(
                        copy, ledger(tasks=candidates), settings.model_copy(update={"seed": seed}), found
                    )
                    chosen += [r[0] for r in copy.db.execute("SELECT id FROM tasks")]
                    raise LookupError("drop the trial")
            except LookupError:
                pass
        assert chosen.count(picked) >= least, (picked, chosen)
    with world.trial() as copy:
        record_ledger(copy, ledger(), settings, [cell("search", 1)])
    assert json.loads(world.db.execute("SELECT gold_json FROM tasks").fetchone()[0]) == [
        {"answer": "rollback"}
    ]
    assert world.db.execute("SELECT concept FROM tasks").fetchone()[0] == "a concept", (
        "the cell's concept is stored"
    )

    def settled(facts):  # a level-3 semantic task needs its facts first stated in 2 channels
        sql = f"SELECT value AS answer FROM facts WHERE id = '{facts[-1]}'"
        return Task(id="t9", category="semantic", level=3, actor_id=a, question="What did the review settle?", answer_type="text", gold_sql=sql, facts=facts)  # fmt: skip

    fails(record_tasks, world, [settled(["f1", "f4"])], settings, [cell("semantic", 3)])  # both in #ops
    try:
        with world.trial() as copy:
            record_tasks(copy, [settled(["f1", "f3"])], settings, [cell("semantic", 3)])  # #ops and #leads
            raise LookupError("drop the trial")
    except LookupError:
        pass
    here, there = Place(channel_id=ops, author_id=a, probability=0.5), Place(channel_id=leads, author_id=a, probability=0.5)  # fmt: skip
    outsider = Place(channel_id=leads, author_id=c, probability=1.0)

    def placed(facts, tasks=(), cells=(), seed=3):
        """Where code places each fact, on a dropped trial."""
        try:
            with world.trial() as copy:
                copy.db.execute("DELETE FROM task_facts")
                copy.db.execute("DELETE FROM tasks")
                for table in ("fact_relations", "facts", "storylines"):
                    copy.db.execute(f"DELETE FROM {table}")
                record_ledger(copy, ledger(facts=facts, tasks=list(tasks)), settings.model_copy(update={"seed": seed}), list(cells))  # fmt: skip
                where = {f: ch for f, ch in copy.db.execute("SELECT id, channel_id FROM facts")}
                raise LookupError(where)
        except LookupError as out:
            return out.args[0]

    assert placed([fact("f1", "s1", a, 1, places=[outsider, here])]) == {"f1": ops}, (
        "a place's author is a member"
    )
    spread_out = {placed([fact("f1", "s1", a, 1, places=[here, there])], seed=s)["f1"] for s in range(20)}
    assert spread_out == {ops, leads}, "the seed draws among valid places"
    reader = task(facts=("f1",))  # its actor c reads ops, not leads
    likely = Place(channel_id=leads, author_id=a, probability=1.0)
    unlikely = Place(channel_id=ops, author_id=a, probability=0.01)
    assert placed([fact("f1", "s1", a, 1, places=[likely, unlikely])], [reader], found) == {"f1": ops}, "the actor reads it"  # fmt: skip
    hard = Task(id="t3", category="semantic", level=3, actor_id=a, question="What was finally settled?", answer_type="text", gold_sql="SELECT value AS answer FROM facts WHERE id = 'f2'", facts=["f1", "f2"])  # fmt: skip
    pair = [fact("f1", "s1", a, 1, places=[here, there]), fact("f2", "s1", a, 2, places=[here, there], supersedes="f1")]  # fmt: skip
    assert all(
        len(set(placed(pair, [hard], [cell("semantic", 3)], seed=s).values())) == 2 for s in range(10)
    ), "spread"
    stuck = [fact("f1", "s1", a, 1, places=[here]), fact("f2", "s1", a, 2, places=[here], supersedes="f1")]
    try:
        placed(stuck, [hard], [cell("semantic", 3)])
        raise AssertionError("a level-3 task's facts were left in one channel")
    except ValueError as error:
        assert "at least 2 channels" in str(error), error

    def scene(id, day, part, beats, participants=(a, b, c), channel=ops, **extra):
        return Scene(id=id, channel_id=channel, participants=list(participants), day=day, part=part, situation="work", length=3, beats=[Beat(fact=f, author_id=u) for f, u in beats], **extra)  # fmt: skip

    first = scene("sc1", 1, "morning", [("f1", a), ("f4", a)])
    second = scene("sc2", 2, "afternoon", [("f2", b)], during="f2")
    for bad in (
        [first, scene("sc2", 2, "afternoon", [("f2", b)])],  # happened at 14:00, stated from 12:00
        [scene("sc0", 1, "early", [("f4", a)]), first, second],  # f4 comes after f1, stated a part earlier
        [scene("sc1", 1, "morning", [("f1", c), ("f4", a)]), second],  # f1 is the ledger's: a's
        [
            scene("sc1", 1, "morning", [("f1", a), ("f4", a)], participants=(b, c)),
            second,
        ],  # a is no participant
        [
            first,
            second,
            scene("sc4", 1, "evening", [], participants=(a, c), channel=leads),
        ],  # c is not in #leads
        [first, second, scene("sc5", 3, "morning", [], during="f2")],  # f2 happens on day 2
        [first],  # f2 has no beat
        [first, second, scene("sc3", 3, "morning", [("f3", a)], channel=leads, participants=(a, b))],
    ):
        fails(check_plan, world, ScenePlan(scenes=bad), "s1", {})
    plan = check_plan(world, ScenePlan(scenes=[first, second]), "s1", {})
    fails(
        check_plan,
        world,
        ScenePlan(scenes=[first.model_copy(update={"part": "evening"}), second]),
        "s1",
        {"sc1": first},
    )
    check_plan(
        world,
        ScenePlan(scenes=[first.model_copy(update={"revision_note": "shorter"}), second]),
        "s1",
        {"sc1": first},
    )

    gaps, rng = Gaps(settings.personas.gaps), random.Random(1)
    assert all(gaps.sample(rng, "") < 3_600_000_000 <= gaps.sample(rng, "hours") for _ in range(200))
    one = WrittenScene(
        lines=[
            Line(author_id=a, text="rollback it is, <@" + c + "> fyi", conveys=["f1"]),
            Line(author_id=b, text="ok, rollback", reply_to=0, conveys=["f1"]),
            Line(author_id=a, text="and the other part is settled", conveys=["f4"]),
        ]
    )
    for bad in (
        one.model_copy(
            update={"lines": [one.lines[0].model_copy(update={"text": "rollback at 3pm"}), *one.lines[1:]]}
        ),
        one.model_copy(
            update={"lines": [one.lines[0].model_copy(update={"text": "rollback {at:f1}"}), *one.lines[1:]]}
        ),
        one.model_copy(update={"lines": [one.lines[0].model_copy(update={"conveys": []}), *one.lines[1:]]}),
        one.model_copy(
            update={"lines": [one.lines[0], one.lines[1].model_copy(update={"reply_to": 1}), one.lines[2]]}
        ),
        one.model_copy(
            update={"lines": [one.lines[0], one.lines[1].model_copy(update={"author_id": d}), one.lines[2]]}
        ),
        one.model_copy(
            update={"lines": [one.lines[0].model_copy(update={"text": "we go back"}), *one.lines[1:]]}
        ),
        one.model_copy(
            update={"lines": [one.lines[0].model_copy(update={"conveys": ["f1", "f2"]}), *one.lines[1:]]}
        ),
    ):
        try:
            with world.trial() as copy:
                write_scene(copy, "s1", plan.scenes[0], bad, "k1", random.Random(1), gaps)
            raise AssertionError("scene accepted")
        except PlanError:
            raise
        except ValueError:
            pass
    with world.trial() as copy:
        write_scene(copy, "s1", plan.scenes[0], one, "k1", random.Random(1), gaps)
    rows = world.db.execute("SELECT * FROM messages ORDER BY ts_us").fetchall()
    early, late = window(world, 1, "morning")
    assert [r["text"] for r in rows] == [line.text for line in one.lines] and early <= rows[0]["ts_us"] < late
    assert rows[1]["parent_id"] == rows[0]["id"] and rows[1]["ts_us"] > rows[0]["ts_us"]
    roles = [tuple(r) for r in world.db.execute("SELECT fact_id, role FROM evidence ORDER BY message_id")]
    assert roles == [("f1", "anchor"), ("f1", "supporting"), ("f4", "anchor")], (
        "the ledger's placement is the anchor"
    )
    assert world.db.execute("SELECT user_id FROM message_mentions").fetchone()[0] == c
    stamps = placed_lines(world, plan.scenes[0], one, random.Random(1), gaps)  # where sc1 went: now taken
    assert stamps[0] > rows[-1]["ts_us"], "a scene never interleaves with another of its channel"
    two = WrittenScene(lines=[Line(author_id=b, text="restored at {at:f2}", conveys=["f2"])])
    with world.trial() as copy:
        write_scene(copy, "s1", plan.scenes[1], two, "k2", random.Random(2), gaps)
    text, ts = world.db.execute(
        "SELECT text, ts_us FROM messages WHERE user_id = ? AND text LIKE 'restored%'", (b,)
    ).fetchone()
    moment = world.db.execute("SELECT moment_us FROM facts WHERE id = 'f2'").fetchone()[0]
    assert text == "restored at 14:00 CDT" and ts >= moment
    assert render(moment, "UTC", moment) == "19:00 UTC" and render(moment, "UTC", moment + DAY).startswith(
        f"{datetime.fromtimestamp(moment / 1e6, ZoneInfo('UTC')):%a}"
    )
    late_night = scene("sc3", 3, "night", [("f3", a)], channel=leads, participants=(a, b))
    three = WrittenScene(
        lines=[Line(author_id=b, text="still here", pause="hours") for _ in range(4)]
        + [Line(author_id=a, text="decided", conveys=["f3"], pause="hours")]
    )
    try:
        with world.trial() as copy:
            write_scene(copy, "s2", late_night, three, "k3", random.Random(3), gaps)
        raise AssertionError("a statement past midnight was accepted")
    except PlanError as error:
        assert "stated_off_day" in str(error)
    talk = scene("sc9", 2, "afternoon", [], participants=(a, b))
    chat = WrittenScene(
        lines=[
            Line(author_id=a, text="lunch?"),
            Line(author_id=b, text="yes", reply_to=0, reactions=[Reaction(user_id=a, emoji="thumbsup")]),
        ]
    )
    thread = Layout(shape="thread", reactions=1)
    check_lines(world, talk, chat, thread)
    fails(check_lines, world, talk, chat.model_copy(update={"lines": [chat.lines[0], chat.lines[1].model_copy(update={"reply_to": None})]}), thread)  # fmt: skip
    for wrong in (
        Layout(shape="flat", reactions=1),
        Layout(reactions=0),
        Layout(shape="thread", reactions=2),
    ):
        fails(check_lines, world, talk, chat, wrong)
    first_reacted = chat.lines[0].model_copy(update={"reactions": [Reaction(user_id=b, emoji="eyes")]})
    fails(check_lines, world, talk, chat.model_copy(update={"lines": [first_reacted, chat.lines[1].model_copy(update={"reply_to": None})]}), Layout(shape="flat", reactions=1))  # fmt: skip
    check_lines(world, talk, chat.model_copy(update={"lines": [first_reacted, chat.lines[1].model_copy(update={"reply_to": None, "reactions": []})]}), Layout(shape="flat", reactions=1))  # fmt: skip
    own = chat.lines[1].model_copy(update={"reactions": [Reaction(user_id=b, emoji="eyes")]})
    fails(check_lines, world, talk, chat.model_copy(update={"lines": [chat.lines[0], own]}), thread)
    outsider = chat.lines[1].model_copy(update={"reactions": [Reaction(user_id=c, emoji="eyes")]})
    fails(check_lines, world, talk, chat.model_copy(update={"lines": [chat.lines[0], outsider]}), thread)
    check_lines(world, talk, chat.model_copy(update={"lines": [chat.lines[0], outsider]}), thread.model_copy(update={"audience": [c]}))  # fmt: skip
    with world.trial() as copy:
        write_scene(copy, None, talk, chat, "k9", random.Random(9), gaps, thread)
    reacted = world.db.execute(
        "SELECT r.user_id, r.emoji, r.created_us > m.ts_us FROM reactions r JOIN messages m ON m.id = r.message_id"
    ).fetchall()
    assert [tuple(r) for r in reacted] == [(a, "thumbsup", 1)], "a reaction is stored after its message"
    assert world.db.execute("SELECT storyline FROM scenes WHERE id = 'sc9'").fetchone()[0] is None
    quiet = settings.activity.model_copy(update={"reply_share": 0.0, "reaction_rate": 0.0})
    busy = settings.activity.model_copy(update={"reply_share": 0.95, "reaction_rate": 1.0})
    calm, lively = layout(world, talk, quiet, random.Random(1)), layout(world, talk, busy, random.Random(1))
    assert (calm.shape, calm.reactions, lively.shape, lively.reactions) == ("flat", 0, "thread", talk.length)
    assert set(calm.audience) == {c, d}, "the audience is the conversation's other members"
    voices = {a: seed_person(0).typing.model_copy(update={"short_share": 1.0, "long_share": 0.0}), b: seed_person(1).typing.model_copy(update={"short_share": 0.0, "long_share": 1.0})}  # fmt: skip
    talky = talk.model_copy(update={"length": 40})
    terse_voices, wordy_voices = {p: voices[a] for p in (a, b)}, {p: voices[b] for p in (a, b)}
    assert layout(world, talky, quiet, random.Random(1), terse_voices).short == 40, (
        "the budget follows typing"
    )
    assert layout(world, talky, quiet, random.Random(1), wordy_voices).long == 40, "the budget follows typing"
    banded = layout(world, talky, quiet, random.Random(1), voices)
    assert banded.short + banded.long == 40 and banded.short and banded.long, "each line's author is drawn"
    assert layout(world, talk, quiet, random.Random(1), {a: voices[a]}).long is None, (
        "no budget without every typing"
    )
    terse = Layout(short=1, long=0)
    lunch = Line(author_id=a, text="lunch?")
    check_lines(world, talk, WrittenScene(lines=[lunch, Line(author_id=b, text="yes please, the usual place")]), terse)  # fmt: skip
    for wrong in (
        [
            Line(author_id=a, text="lunch somewhere close by today?"),
            Line(author_id=b, text="yes please, the usual place"),
        ],  # fmt: skip
        [lunch, Line(author_id=b, text=" ".join(["word"] * 21))],
    ):
        fails(check_lines, world, talk, WrittenScene(lines=wrong), terse)
    check_lines(world, talk, WrittenScene(lines=[lunch, Line(author_id=b, text=" ".join(["word"] * 21))]), Layout(short=1, long=1))  # fmt: skip
    direct = scene("sc10", 2, "afternoon", [], participants=(a, c), channel=channel_id("im", None, [a, c]))
    assert layout(world, direct, busy, random.Random(1)).shape == "flat", "a DM is never threaded"
    count = world.db.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    small = settings.activity.model_copy(update={"messages": count + 12, "conversation_lines": 4})
    plan = background_plan(world, org(), small, 7)
    workdays = {
        d
        for d, x in world.db.execute("SELECT day, date FROM calendar")
        if date.fromisoformat(x).weekday() < 5
    }
    members = {(r[0], r[1]) for r in world.db.execute("SELECT channel_id, user_id FROM members")}
    assert plan == background_plan(world, org(), small, 7) and sum(s.length for s in plan) in (11, 12)
    assert all(s.situation in {r.kind for r in routines} and s.day in workdays and not s.beats for s in plan)
    assert all((s.channel_id, p) in members for s in plan for p in s.participants)
    assert background_plan(world, org(), small.model_copy(update={"messages": count}), 7) == []
    month = background_plan(world, org(), small.model_copy(update={"messages": count + 300}), 7)
    assert {s.day for s in month} == workdays, "a large plan covers every workday and no weekend"
    direct = background_plan(world, org(), small.model_copy(update={"dm_share": 1.0}), 7)
    kinds = {r[0]: r[1] for r in world.db.execute("SELECT id, type FROM channels")}
    assert {kinds[s.channel_id] for s in direct} == {"im"}, "dm_share places conversations in DMs"
    skewed = background_plan(world, org(), small.model_copy(update={"dm_share": 0.0, "channel_skew": 4.0, "messages": count + 60}), 7)  # fmt: skip
    places = Counter(kinds[s.channel_id] for s in skewed)
    assert places["public"] > 3 * places["private"], "busier channels draw more conversations"
    try:
        with world.trial() as copy:  # a planned value, said in a background conversation
            copy.db.execute("INSERT INTO facts (id, storyline, subject, attribute, value, channel_id, author_id, day, summary) VALUES ('f9', 's1', 'Lunch', 'place', 'taco stand', ?, ?, 1, 's')", (ops, a))  # fmt: skip
            for text in ("lunch at the Taco Stand?", "see you at {at:f2}"):
                fails(unplanned, copy, WrittenScene(lines=[Line(author_id=a, text=text)]))
            said = WrittenScene(lines=[Line(author_id=a, text="taco stand it is")])
            idle, story = scene("sc11", 3, "morning", [], participants=(a, b)), scene("sc12", 4, "morning", [], participants=(a, b))  # fmt: skip
            fails(write_scene, copy, None, idle, said, "k11", random.Random(1), gaps)
            write_scene(copy, "s1", story, said, "k12", random.Random(1), gaps)  # a storyline may discuss it
            unplanned(copy, WrittenScene(lines=[Line(author_id=a, text="lunch somewhere tacos?")]))
            raise LookupError("drop the trial")
    except LookupError:
        pass
    hybrid = dict(category="hybrid", level=1, actor_id=c, answer_type="number", gold_sql="SELECT COUNT(*) AS answer FROM messages m, facts f WHERE f.id = 'f1'")  # fmt: skip
    unread = Task(
        id="h1", question="How many messages surround the decision?", facts=["f3"], **hybrid
    )  # in leads
    read = Task(id="h2", question="How many messages surround the rollback?", facts=["f1"], **hybrid)
    picks = []
    for seed in range(
        20
    ):  # on the written world: a candidate whose actor cannot read its facts is never picked
        try:
            with world.trial() as copy:
                record_tasks(
                    copy, [unread, read], settings.model_copy(update={"seed": seed}), [cell("hybrid", 1)]
                )
                picks += [r[0] for r in copy.db.execute("SELECT id FROM tasks WHERE category = 'hybrid'")]
                raise LookupError("drop the trial")
        except LookupError:
            pass
    assert picks == ["h2"] * 20, picks
    assert TIME_LITERAL.findall("at 3pm, 15:00, 2026-06-15 or Jun 15") == [
        "3pm",
        "15:00",
        "2026-06-15",
        "Jun 15",
    ]
    assert TIME_LITERAL.findall("release 8.14 on v2.10, maybe 2 people") == []
    print("PASS contracts: organization, ledger and task gold (T2-T6), plan, written scenes, code-owned time")


def organized(root, settings):
    """An organized world on Chicago time: people a, b, c (and d, in New York), #ops (public: everyone), #leads
    (private: a, b), DMs a-c and code's own, a 10-day calendar from a Monday."""
    cast = [seed_person(i, "America/New_York" if i == 3 else "America/Chicago") for i in range(4)]
    a, b, c, d = (user_id(p.uuid) for p in cast)
    routines = [
        Routine(kind=k, probability=p) for k, p in (("standup", 0.5), ("review", 0.3), ("lunch", 0.2))
    ]
    org = Organization(
        people=[Person(user_id=u, title="Engineer", team="Platform") for u in (a, b, c, d)],
        channels=[
            Channel(name="ops", type="public", members=[a, b, c, d], routines=routines),
            Channel(name="leads", type="private", members=[a, b], routines=routines),
            Channel(type="im", members=[a, c]),
        ],
        dm_routines=routines,
    )
    premise = Premise(
        company="Lattice Creek", niche="n", region="r", size="s", culture="c", cast="x", staffing={}
    )
    world = World.create(root / "world.sqlite")
    with world.trial() as copy:
        organize(copy, org, cast, premise, settings)
    return world, cast, org, (a, b, c, d)


def refused(expected, function, *args, **kwargs):
    """The call is refused, for the reason given."""
    try:
        function(*args, **kwargs)
    except ValueError as error:
        assert expected in str(error), (expected, str(error))
        return
    raise AssertionError(f"not refused: {expected}")


def check_clock(root):
    """The world in time order (v7): a plan whose events never move and whose facts share their event's moment;
    conversations posted only at the present, their statements tagged and first made where the plan says; promises
    opened and closed; days closed only when their quota, facts and promises are settled; fixes that change no task;
    tasks written once the calendar is closed."""
    root.mkdir(parents=True)
    base = contracts_settings(root)
    settings = base.model_copy(
        update={
            "activity": base.activity.model_copy(update={"messages": 12}),
            "author": base.author.model_copy(update={"tolerance": 1.0, "share_tolerance": 1.0}),
        }
    )
    gaps = Gaps(settings.personas.gaps)
    world, cast, org, (a, b, c, d) = organized(root, settings)
    ops, leads = channel_id("public", "ops", []), channel_id("private", "leads", [])
    with world.trial() as copy:
        start_clock(copy)
    assert today(world) == 1 and present(world) == window(world, 1, "early")[0]
    shares = quotas(world, settings)
    assert sum(q["messages"] for q in shares.values()) == 12 and all(sum(q["parts"].values()) == q["messages"] for q in shares.values())  # fmt: skip
    assert shares[6]["messages"] == shares[7]["messages"] == 0 < shares[1]["messages"], "weekends are quiet"
    zone = "America/Chicago"

    def plan(**changes):
        events = [
            Event(id="e1", storyline="s2", title="release window", day=3, time="10:00"),
            Event(id="e2", storyline="s1", title="service restored", day=2, time="14:00"),
        ]
        facts = [
            PlanFact(id="f1", storyline="s1", subject="Release 4.2", attribute="decision", value="rollback", anchor="rollback", channel_id=ops, author_id=a, day=1, summary="s"),
            PlanFact(id="f2", storyline="s1", subject="Release 4.2", attribute="state", value="service back", channel_id=ops, author_id=b, day=2, after=["f1"], event="e2", kind="happened", summary="s"),
            PlanFact(id="f3", storyline="s2", subject="Audit", attribute="window", value="dry run", anchor="dry run", channel_id=leads, author_id=a, day=3, event="e1", kind="scheduled", summary="s"),
            PlanFact(id="f4", storyline="s2", subject="Audit", attribute="owner", value="Owen", channel_id=ops, author_id=a, day=3, after=["f3"], event="e1", kind="scheduled", summary="s"),
        ]  # fmt: skip
        document = dict(storylines=[Storyline(id="s1", summary="the 4.2 release"), Storyline(id="s2", summary="the audit")], events=events, facts=facts)  # fmt: skip
        return Plan(**document | changes)

    def recorded(document):
        with world.trial() as copy:
            record_plan(copy, document, settings)

    first_plan = plan()
    events, facts = first_plan.events, first_plan.facts
    for expected, bad in (
        ("plan exactly 2 storylines", plan(storylines=first_plan.storylines[:1])),
        ("repeated ['f1']", plan(events=[*events, Event(id="f1", storyline="s1", title="clash", day=1, time="09:00")])),
        ("a title has no time", plan(events=[events[0].model_copy(update={"title": "window at 10:00"}), events[1]])),
        ("event e9 is not in events", plan(facts=[*facts[:3], facts[3].model_copy(update={"event": "e9"})])),
        (f"is a member of {leads}", plan(facts=[*facts[:2], facts[2].model_copy(update={"author_id": c}), facts[3]])),
        ("on or after the day of e2", plan(facts=[facts[0], facts[1].model_copy(update={"day": 1}), *facts[2:]])),
        ("on a later day", plan(facts=[facts[0].model_copy(update={"day": 3}), *facts[1:]])),  # f2 follows f1
    ):  # fmt: skip
        refused(expected, recorded, bad)
    recorded(first_plan)
    shared = world.db.execute("SELECT DISTINCT f.moment_us, e.moment_us FROM facts f JOIN events e ON e.id = f.event_id WHERE e.id = 'e1'").fetchall()  # fmt: skip
    assert len(shared) == 1 and shared[0][0] == shared[0][1] == at(world, Moment(day=3, time="10:00", zone=zone)), "facts share their event's moment"  # fmt: skip
    refused(
        "an event never moves",
        recorded,
        plan(events=[events[0].model_copy(update={"time": "11:00"}), events[1]]),
    )
    refused("planned events stay", recorded, plan(events=events[:1]))
    refused("keep ['s1', 's2']", recorded, plan(storylines=[Storyline(id="s3", summary="x"), Storyline(id="s2", summary="y")]))  # fmt: skip
    rng = random.Random(5)

    cells = [("semantic", 3, "a concept", "a style"), ("lookup", 1, "a concept", "a style")]
    owner = Task(id="t1", category="semantic", level=3, actor_id=a, question="Who owns the audit after the dry run?", answer_type="text", gold_sql="SELECT value AS answer FROM facts WHERE id = 'f4'", facts=["f3", "f4"])  # fmt: skip

    def tasked(task):
        with world.trial() as copy:
            return add_task(copy, task, settings, cells)

    refused("once the calendar is closed", tasked, owner)

    def posted_(conversation):
        with world.trial() as copy:
            return post(copy, conversation, settings, rng, gaps)

    def talk(*lines, channel=ops, **extra):
        return Conversation(
            channel_id=channel, about="work", lines=[PostLine(**line) for line in lines], **extra
        )

    decided = talk(
        dict(author_id=a, text="we go with the rollback", conveys=["f1"]),
        dict(author_id=b, text="ok, I'll post the notes", reply_to=0, commits=[Commit(id="c1", text="post the notes", due_day=2)], reactions=[Reaction(user_id=a, emoji="eyes")]),
    )  # fmt: skip
    for expected, bad in (
        ("write times and dates only", talk(dict(author_id=a, text="rollback at 3pm", conveys=["f1"]))),
        ("list them in conveys", talk(dict(author_id=a, text="rollback is decided"))),
        ("f1 is first stated by", talk(dict(author_id=b, text="rollback is decided", conveys=["f1"]))),
        ("f3 is first stated by", talk(dict(author_id=a, text="dry run it is", conveys=["f3"]), channel=leads)),
        ("nobody reacts to their own line", talk(dict(author_id=a, text="ok", reactions=[Reaction(user_id=a, emoji="eyes")]))),
        ("from a member of its channel", talk(dict(author_id=a, text="ok", reactions=[Reaction(user_id=c, emoji="eyes")]), channel=leads)),
        ("not a member of the channel", talk(dict(author_id=c, text="hi"), channel=leads)),
        ("c9 is no open commitment", talk(dict(author_id=a, text="ok", closes=[Close(id="c9", status="kept")]))),
        ("due today or on a later day", talk(dict(author_id=a, text="ok", commits=[Commit(id="c2", text="t", due_day=11)]))),
    ):  # fmt: skip
        refused(expected, posted_, bad)
    before = present(world)
    out = posted_(decided)
    rows = world.db.execute("SELECT id, ts_us, parent_id FROM messages ORDER BY id").fetchall()
    assert rows[0]["ts_us"] > before and present(world) == rows[-1]["ts_us"] and rows[1]["parent_id"] == rows[0]["id"]  # fmt: skip
    assert out["conversation"] == "d01-001" and len(out["messages"]) == 2
    assert world.db.execute("SELECT role FROM evidence WHERE fact_id = 'f1'").fetchone()[0] == "anchor"
    assert world.db.execute("SELECT status, owner_id FROM commitments WHERE id = 'c1'").fetchone()[:] == (
        "open",
        b,
    )
    refused(
        "UNIQUE",
        posted_,
        talk(dict(author_id=a, text="ok", commits=[Commit(id="c1", text="again", due_day=2)])),
    )
    reaction = world.db.execute("SELECT created_us FROM reactions").fetchone()[0]
    assert rows[0]["ts_us"] < reaction < bounds(world, 1)[1], "a reaction stays within its message's day"
    again = posted_(talk(dict(author_id=c, text="noted"), thread=rows[0]["id"]))
    noted = world.db.execute("SELECT ts_us, parent_id FROM messages WHERE id = ?", (again["messages"][0]["id"],)).fetchone()  # fmt: skip
    assert noted["ts_us"] > rows[-1]["ts_us"] and noted["parent_id"] == rows[0]["id"], (
        "a later post comes later"
    )
    refused("earlier top-level message", posted_, talk(dict(author_id=c, text="noted"), thread=rows[1]["id"]))
    refused("leave reply_to empty", posted_, talk(dict(author_id=c, text="a"), dict(author_id=a, text="b", reply_to=0), thread=rows[0]["id"]))  # fmt: skip
    refused("past midnight", posted_, talk(*[dict(author_id=a, text="still here", pause="hours")] * 20))
    try:  # code's clock is the only way forward: a message dated before the present is refused
        with world.batch():
            world.insert(
                "messages", [dict(channel_id=ops, ts_us=rows[0]["ts_us"] - 1, user_id=a, text="late")]
            )
        raise AssertionError("a message was written into the past")
    except ValueError as error:
        assert "written_out_of_order" in str(error), error

    def advanced(**kwargs):
        with world.trial() as copy:
            return advance(copy, settings, **kwargs)

    assert advanced()["closed_day"] is None and part_of(world, present(world)) != "early"
    refused("a later part of today", advanced, to="early")
    advanced(to="night")
    strict = settings.model_copy(update={"author": settings.author.model_copy(update={"tolerance": 0.0})})
    assert shares[1]["messages"] == 2 and any("messages today" in e for e in close_day(world, strict, 1)), (
        "3 posted"
    )
    assert advanced()["closed_day"] == 1 and today(world) == 2
    advanced(to="morning")
    refused(
        "before_it_happened", posted_, talk(dict(author_id=b, text="service back at {at:e2}", conveys=["f2"]))
    )
    advanced(to="evening")
    assert any(e.startswith("c1 is due") for e in close_day(world, settings, 2)), (
        "the promise due today is open"
    )
    restored = posted_(talk(dict(author_id=b, text="service back at {at:e2}, notes are up", conveys=["f2"], closes=[Close(id="c1", status="kept")])))  # fmt: skip
    assert restored["messages"][0]["text"] == "service back at 14:00 CDT, notes are up", (
        "{at:} on the author's clock"
    )
    assert world.db.execute("SELECT status, closed_by FROM commitments").fetchone()[:] == ("kept", restored["messages"][0]["id"])  # fmt: skip
    refused("only its summary may change", recorded, plan(facts=[facts[0].model_copy(update={"value": "roll forward", "anchor": None}), *facts[1:]]))  # fmt: skip
    refused("only its summary may change", recorded, plan(facts=[facts[0], facts[1].model_copy(update={"after": []}), *facts[2:]]))  # fmt: skip
    recorded(plan(facts=[facts[0].model_copy(update={"summary": "decided early"}), *facts[1:]]))
    assert world.db.execute("SELECT summary FROM facts WHERE id = 'f1'").fetchone()[0] == "decided early"
    refused(
        "today (day 2) or later", recorded, plan(facts=[*facts[:3], facts[3].model_copy(update={"day": 1})])
    )
    advanced(to="night")
    advanced()
    advanced(to="night")
    refused("day 3 stays open", advanced)
    assert {e.split()[0] for e in close_day(world, settings, 3)} == {"f3", "f4"}, (
        "planned for day 3, unstated"
    )
    refused("comes after ['f3']", posted_, talk(dict(author_id=a, text="Owen owns the audit at {at:e1}", conveys=["f4"])))  # fmt: skip
    # Late, but on their day: a scheduled fact may be stated after its event.
    posted_(
        talk(
            dict(author_id=a, text="dry run for the release window at {at:e1}", conveys=["f3"]), channel=leads
        )
    )
    # A fact about an event and with no anchor of its own is stated with the event's time.
    posted_(talk(dict(author_id=a, text="Owen owns the audit at {at:e1}", conveys=["f4"]), dict(author_id=d, text="thanks")))  # fmt: skip
    while today(world) is not None:
        advanced(to="night") if part_of(world, present(world)) != "night" else advanced()
    refused("the calendar is closed", posted_, talk(dict(author_id=a, text="late")))
    refused("the calendar is closed", recorded, first_plan)
    measured = close_day(world, settings.model_copy(update={"author": settings.author.model_copy(update={"share_tolerance": 0.0})}), 10)  # fmt: skip
    assert any("reply_share" in e for e in measured), measured
    refused("at least 2 channels", tasked, owner.model_copy(update={"facts": ["f1", "f4"]}))  # both in #ops
    refused("fills one of the cells", tasked, owner.model_copy(update={"level": 2}))
    result = tasked(owner)
    assert result["gold"] == [{"answer": "Owen"}] and "bm25_rank" in result
    replier = Task(id="t2", category="lookup", level=1, actor_id=c, question="Who said thanks in #ops?", answer_type="text", gold_sql="SELECT u.real_name AS answer FROM messages m JOIN users u ON u.id = m.user_id WHERE m.text = 'thanks'")  # fmt: skip
    tasked(replier)
    tasked(replier.model_copy(update={"id": "t3"}))  # the cell's task is replaced
    assert [r[0] for r in world.db.execute("SELECT id FROM tasks ORDER BY id")] == ["t1", "t3"]
    thanks = world.db.execute("SELECT id FROM messages WHERE text = 'thanks'").fetchone()[0]
    stated = world.db.execute("SELECT message_id FROM evidence WHERE fact_id = 'f4'").fetchone()[0]

    def revised(message, text):
        with world.trial() as copy:
            return revise(copy, message, text)

    refused("changes task t3", revised, thanks, "thank you")  # t3's gold answer reads that text
    refused(
        "anchor_token_missing", revised, stated, "the audit has an owner"
    )  # f4's evidence keeps its token
    refused("states nothing new", revised, thanks, "thanks, rollback noted")
    assert revised(stated, "Owen owns the audit from {at:e1}")["text"] == "Owen owns the audit from 10:00 CDT"
    ids = [r[0] for r in world.db.execute("SELECT id FROM messages ORDER BY ts_us")]
    assert ids == sorted(ids) and world.violations(complete=True) == []
    print(
        "PASS clock: plan and events, posts at the present, tagging, first statements, promises, day close, revise, tasks"
    )


def author_context(settings, cast, org, root, agenda=None):
    """What WorldTools are configured with, as the driver builds it."""
    state = SimpleNamespace(
        quota=[("semantic", 3, "a concept", "a style"), ("lookup", 1, "a concept", "a style")],
        cast=cast,
        plans={"agenda": {"scenes": [s.model_dump(mode="json") for s in agenda or []]}},
    )
    return context_of(settings, state, root, org.model_dump(mode="json"))


async def check_tools(root):
    """The author's tools (v7): nine, served host-side over the world file; each turn's tools act only in their turn
    and on their day; writes are checked and logged, reads are live; the memory renders every time on the company
    clock, never as raw microseconds, and the one-pager stays small on a full world."""
    root.mkdir(parents=True)
    base = contracts_settings(root)
    quick = root / "quick-gaps.json"  # most replies within ten minutes, as in real channels
    quick.write_text(
        json.dumps({"quantiles": [i * 0.6 for i in range(900)] + [600 + i * 138 for i in range(101)]}) + "\n"
    )
    settings = base.model_copy(
        update={
            "activity": base.activity.model_copy(update={"messages": 400}),
            "author": base.author.model_copy(update={"tolerance": 1.0, "share_tolerance": 1.0}),
            "personas": base.personas.model_copy(update={"gaps": quick}),
        }
    )
    world, cast, org, (a, b, c, d) = organized(root, settings)
    ops, leads = channel_id("public", "ops", []), channel_id("private", "leads", [])
    with world.trial() as copy:
        start_clock(copy)
    agenda = background_plan(world, org, settings.activity, settings.seed)
    context = author_context(settings, cast, org, root, agenda)
    ledger = Plan(
        storylines=[Storyline(id="s1", summary="the 4.2 release"), Storyline(id="s2", summary="the audit")],
        events=[Event(id="e1", storyline="s2", title="audit window", day=3, time="10:00")],
        facts=[
            PlanFact(id="f1", storyline="s1", subject="Release 4.2", attribute="decision", value="rollback", anchor="rollback", channel_id=ops, author_id=a, day=1, summary="s"),
            PlanFact(id="f2", storyline="s2", subject="Audit", attribute="owner", value="Owen", channel_id=leads, author_id=a, day=3, after=["f1"], event="e1", kind="scheduled", summary="s"),
        ],
    )  # fmt: skip

    def tools_for(mode, day):
        task = WorldAuthorTask.create(mode, day, world.path, context, f"{mode}-{day:02d}")
        (tools,) = task.toolsets(task.config)
        return tools

    planner = tools_for("plan", 0)
    await planner.setup()
    stored = with_state(planner)
    assert "ledger holds 2 storylines, 1 events and 2 facts" in await planner._with_state(planner.plan)(
        ledger=ledger
    )
    for name, kwargs in (("post", {"conversation": Conversation(channel_id=ops, about="x", lines=[PostLine(author_id=a, text="hi")])}), ("add_task", {"task": Task(id="t", category="lookup", level=1, actor_id=a, question="q?", answer_type="text", gold_sql="SELECT 1 AS answer")})):  # fmt: skip
        try:
            await planner._with_state(getattr(planner, name))(**kwargs)
            raise AssertionError(f"{name} ran in the plan turn")
        except ValueError as error:
            assert "turn" in str(error), error
    day1 = tools_for("day", 1)
    await day1.setup()
    with_state(day1)
    call = lambda tools, name, **kwargs: tools._with_state(getattr(tools, name))(**kwargs)  # noqa: E731
    page = await call(day1, "now")
    assert (
        page.startswith("# Now: ") and "day 1 of 10" in page and "[[f1]]" in page and "## Task cells" in page
    )
    assert "everyday conversations code drew for today" in page or not [s for s in agenda if s.day == 1]
    out = await call(day1, "post", conversation=Conversation(channel_id=ops, about="decision", lines=[PostLine(author_id=a, text="rollback it is", conveys=["f1"]), PostLine(author_id=b, text="ok", reply_to=0)]))  # fmt: skip
    assert len(out["messages"]) == 2 and out["now"].endswith("CDT"), out
    try:
        await call(day1, "post", conversation=Conversation(channel_id=ops, about="x", lines=[PostLine(author_id=c, text="rollback again")]))  # fmt: skip
        raise AssertionError("an untagged anchor was posted")
    except ValueError as error:
        assert "conveys" in str(error), error
    for ref in (
        "#ops",
        a,
        "alicia.rao",
        "f1",
        "e1",
        "s1",
        f"m{out['messages'][0]['id']}",
        out["conversation"],
    ):
        assert (await call(day1, "view", ref=ref)).strip(), ref
    try:
        await call(day1, "view", ref="nobody")
        raise AssertionError("an unknown page was shown")
    except ValueError:
        pass
    assert (await call(day1, "sql", sql="SELECT COUNT(*) AS n FROM facts"))["rows"] == [{"n": 2}]
    assert (await call(day1, "sql", sql="SELECT COUNT(*) AS n FROM messages", actor_id=c))["rows"] == [
        {"n": 2}
    ]
    for bad in ({"sql": "DELETE FROM facts"}, {"sql": "SELECT * FROM tasks", "actor_id": c}):
        try:
            await call(day1, "sql", **bad)
            raise AssertionError(f"sql ran {bad}")
        except ValueError:
            pass
    assert (await call(day1, "read", actor_id=c, tool="read_channel", arguments={"channel_id": ops}))["items"]
    while today(world) == 1:
        await call(day1, "advance")
    try:
        await call(day1, "post", conversation=Conversation(channel_id=ops, about="x", lines=[PostLine(author_id=a, text="hi")]))  # fmt: skip
        raise AssertionError("day 1's tools posted on day 2")
    except ValueError as error:
        assert "day 1 is closed" in str(error), error
    log = [json.loads(line) for line in (root / "world-calls.jsonl").read_text().splitlines()]
    assert {e["tool"] for e in log} >= {"plan", "post", "view", "sql", "read", "advance", "now"}
    assert any(not e["ok"] and "conveys" in e["error"] for e in log) and stored["state"].calls == 1, (
        "one plan went through"
    )
    served = tools_for("day", 2)  # a tool server checks the world file's hash when it starts
    with_state(served)
    names = await served_tools(served)
    assert names == {"now", "view", "sql", "read", "plan", "post", "advance", "revise", "add_task"} and served.server_name == "world"  # fmt: skip
    async with (
        asyncio.timeout(30),
        serve(served) as url,
        streamable_http_client(url) as (reader, writer_),
        ClientSession(reader, writer_) as client,
    ):  # nested documents cross the wire as strict models, and a refused write carries its reason
        await client.initialize()
        refused = await client.call_tool("post", {"conversation": {"channel_id": ops, "about": "x", "lines": [{"author_id": a, "text": "hi", "commits": [{"id": "c1", "text": "t", "due_day": 2.5}]}]}})  # fmt: skip
        assert refused.is_error and "due_day" in str(refused.content), refused
    bad = WorldAuthorTask.create("day", 2, world.path, context, "day-02-01")
    world.db.execute("UPDATE storylines SET summary = 'changed' WHERE id = 's1'")
    try:
        await bad.toolsets(bad.config)[0].setup()
        raise AssertionError("a changed world file was served")
    except ValueError:
        pass
    world.db.execute("UPDATE storylines SET summary = 'the 4.2 release' WHERE id = 's1'")
    rng, gaps, lines = random.Random(3), Gaps(settings.personas.gaps), 0
    while today(world) is not None:  # a full world: about 400 messages over ten days, written at the present
        day = today(world)
        quota = quotas(world, settings)[day]["messages"]
        if day == 3 and not world.db.execute("SELECT 1 FROM evidence WHERE fact_id = 'f2'").fetchone():
            post(world, Conversation(channel_id=leads, about="audit", lines=[PostLine(author_id=a, text="Owen owns it from {at:e1}", conveys=["f2"])]), settings, rng, gaps)  # fmt: skip
        if posted(world, *bounds(world, day)) < quota and part_of(world, present(world)) != "night":
            try:
                post(world, Conversation(channel_id=rng.choice([ops, leads]), about="work", lines=[PostLine(author_id=rng.choice([a, b]), text=f"note {lines + i} on the work", reply_to=0 if i else None) for i in range(4)]), settings, rng, gaps)  # fmt: skip
                lines += 4
                continue
            except ValueError:
                pass
        advance(world, settings)
    assert world.db.execute("SELECT COUNT(*) FROM messages").fetchone()[0] > 300
    memory = files(world, settings, context, "tasks")
    raw = re.compile(r"\b1\d{15}\b")
    assert not [p for p, text in memory.items() if raw.search(text)], "no raw microseconds reach the author"
    assert len(memory["memory/now.md"].encode()) < 20_000, len(memory["memory/now.md"].encode())
    assert Plan.model_validate(json.loads(memory["memory/ledger.json"])) == ledger, (
        "ledger.json is world_plan's input"
    )
    assert not raw.search(json.dumps(ledger_digest(world))), "the judge's ledger is rendered too"
    print(
        "PASS tools: nine served, scoped by turn and day, checked and logged; memory rendered, small, round-trips"
    )


async def served_tools(toolset):
    async with (
        asyncio.timeout(30),
        serve(toolset) as url,
        streamable_http_client(url) as (reader, writer_),
        ClientSession(reader, writer_) as client,
    ):
        await client.initialize()
        return {t.name for t in (await client.list_tools()).tools}


def with_state(toolset):
    """Serve calls against an in-memory state, as the framework's state channel would."""
    stored = {"state": toolset._state_cls()}

    async def pull():
        return stored["state"].model_copy(deep=True)

    async def push(before):
        stored["state"] = toolset.state.model_copy(deep=True)

    toolset._pull_state, toolset._push_state = pull, push
    return stored


async def check_reviews(root):
    """Reviews bind to what they reviewed, decide on blocking issues per task, and inspect the world as its tasks'
    actors; solvers get the solver copy and no answer; a release holds exactly what was approved."""
    root.mkdir(parents=True)
    world, at, message = world_fixture(root / "world.sqlite")
    rows = world.gold("U2", "SELECT value AS answer FROM facts WHERE id = 'f1'")["rows"]
    world.db.execute("UPDATE tasks SET gold_json = ? WHERE id = 't1'", (json.dumps(rows),))
    payload = review_payload(world, "task", ["t1"])
    taxonomy = contracts_settings(root).taxonomy
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
            tasks=[TaskReview(task_id=t, valid=v, reason="r", level_fit=3) for t, v in tasks],
            issues=list(issues),
            criteria=criteria if criteria is not None else dict.fromkeys(PHASE_CRITERIA["task"], 1.0),
            summary="s",
        )

    validate_verdict(verdict(), payload)
    fails(validate_verdict, verdict(criteria={"question_fit": 1.0}), payload)
    fails(validate_verdict, verdict(tasks=()), payload)
    unscored = verdict().model_copy(update={"tasks": [TaskReview(task_id="t1", valid=True, reason="r")]})
    fails(validate_verdict, unscored, payload)  # a task review scores each task's level_fit
    assert "level_fit" in verdict_schema("task", ["t1"])["$defs"]["TaskReview"]["required"]
    workspace = Issue(artifact="workspace", defect="d", requested_change="c")
    validate_verdict(verdict(approved=False, issues=[workspace]), payload)
    ledger = review_payload(world, "ledger", ["t1"])

    def ledger_verdict(issue):
        tasks = [TaskReview(task_id="t1", valid=False, reason="r")]
        return Verdict(approved=False, tasks=tasks, issues=[issue], summary="s")

    fails(validate_verdict, ledger_verdict(workspace), ledger)  # no message exists to repair yet
    validate_verdict(ledger_verdict(workspace.model_copy(update={"artifact": "ledger"})), ledger)
    assert verdict_schema("ledger", ["t1"])["$defs"]["Issue"]["properties"]["artifact"]["enum"] == [
        "ledger",
        "tasks",
    ] and "workspace" in str(schema["$defs"]["Issue"]["properties"]["artifact"])
    issue = Issue(artifact="tasks", task_ids=["t2"], defect="d", requested_change="c")
    fails(verdict, issues=[issue])
    fails(verdict, approved=False)
    batch = verdict(approved=False, issues=[issue], tasks=(("t1", True), ("t2", True)))
    strict, lenient = Acceptance(minor_issues_block=True), Acceptance()
    assert accepted_task(batch, lenient, "t1") and not accepted_task(batch, lenient, "t2")
    general = batch.model_copy(update={"issues": [issue.model_copy(update={"task_ids": []})]})
    assert not accepted_task(general, lenient, "t1"), "an issue naming no task blocks every task"
    minor = batch.model_copy(update={"issues": [issue.model_copy(update={"blocking": False})]})
    assert (
        accepted_task(minor, lenient, "t2")
        and not accepted_task(minor, strict, "t2")
        and not deciding(minor, lenient)
    )

    world.snapshot(root / "review.sqlite")
    judge = JudgeTask.create(payload, root / "review.sqlite", "build-s1-01", 5)
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
    rank = World(root / "review.sqlite", actor="U2").rank(payload["tasks"][0]["question"], [1])
    assert result["tasks"]["t1"]["bm25_rank"] == rank == 2 and "U1" in result["style"]["authors"]
    assert (
        await tools._with_state(tools.read)(
            actor_id="U2", tool="read_channel", arguments={"channel_id": "C1"}
        )
    )["items"]
    try:
        await tools._with_state(tools.read)(
            actor_id="U1", tool="read_channel", arguments={"channel_id": "G1"}
        )
        raise AssertionError("a reviewer read as someone outside the review")
    except ValueError:
        pass
    assert (
        await tools._with_state(tools.sql)(
            actor_id="U2", sql="SELECT COUNT(*) AS n FROM messages WHERE channel_id = 'G1'"
        )
    )["rows"] == [{"n": 0}]
    assert await served_tools(tools) == {"check", "read", "sql"} and tools.server_name == "inspect"
    files = {"/task/verdict.json": verdict().model_dump_json().encode()}
    runtime = SimpleNamespace(read=lambda path, max_bytes: asyncio.sleep(0, files[path]))
    unchecked = SimpleNamespace(state=ReviewState(), info={}, record_metric=lambda *a: None)
    try:
        await judge.finalize(unchecked, runtime)
        raise AssertionError("a verdict without inspect_check was accepted")
    except ValueError:
        pass
    checked = SimpleNamespace(state=ReviewState(checked=True), info={}, record_metric=lambda *a: None)
    await judge.finalize(checked, runtime)
    assert checked.info["verdict"]["approved"]

    world.snapshot(root / "planted.sqlite")
    planted = World(root / "planted.sqlite", writable=True)
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
    ledger_judge, unwritten = await inspected("ledger", ["t1"])
    _, untasked = await inspected("world", [])
    assert [v.split(":")[0] for v in due["violations"]] == ["unstated_fact"] and "bm25_rank" in due["tasks"][
        "t1"
    ]
    assert unwritten["violations"] == untasked["violations"] == [], (
        "completion is due for reviewed, written tasks"
    )
    assert unwritten["tasks"]["t1"] == {"rows": due["tasks"]["t1"]["rows"]}, (
        "a ledger review ranks no evidence"
    )
    files["/task/verdict.json"] = ledger_verdict(workspace).model_dump_json().encode()
    try:
        await ledger_judge.finalize(checked, runtime)
        raise AssertionError("a ledger review routed a repair to messages that do not exist")
    except ValueError as error:
        assert "reports issues on ledger or tasks only" in str(error), error

    world.solver_copy(root / "solver.sqlite")
    task = PublicTask(
        task_id="t1",
        question="q?",
        actor_id="U2",
        category="search",
        level=1,
        answer_type="text",
        world_hash=sha256(root / "solver.sqlite"),
    )
    answer = PrivateAnswer(
        answer_type="text",
        rows=rows,
        gold_sql="SELECT 1",
        messages=[["C1", world.db.execute("SELECT ts FROM messages WHERE id = 1").fetchone()[0]]],
        users=["U1"],
    )
    solve = SolverTask.create(task, root / "solver.sqlite", answer, network_policy=False)
    assert "rollback" not in solve.data.model_dump_json() + solve.config.tools.model_dump_json()
    assert solve.config.tools.db_hash == sha256(root / "solver.sqlite")
    write_release(root / "release", root / "solver.sqlite", [task], {"t1": answer})
    assert load_release(root / "release")[1:] == ([task], {"t1": answer})
    write_release(root / "release", root / "solver.sqlite", [task], {"t1": answer})  # idempotent
    fails(
        write_release,
        root / "release",
        root / "solver.sqlite",
        [task],
        {"t1": answer.model_copy(update={"rows": []})},
    )
    tampered = json.loads((root / "release" / "answers.json").read_text())
    tampered["t1"]["rows"] = [{"answer": "something else"}]
    (root / "release" / "answers.json").write_text(json.dumps(tampered))
    fails(load_release, root / "release")
    hidden = answer.model_copy(
        update={"messages": [["G1", world.db.execute("SELECT ts FROM messages WHERE id = 4").fetchone()[0]]]}
    )
    fails(write_release, root / "release2", root / "solver.sqlite", [task], {"t1": hidden})
    print("PASS reviews: verdict contracts, per-task acceptance, judge tools as actors, solver copy, release")


class TestStore(Store):
    def trace(self, trace):
        pass


class ScriptedAgent:
    trainable = False

    @asynccontextmanager
    async def provision(self, task):
        yield None

    @asynccontextmanager
    async def interaction(self, task, runtime):
        yield SimpleNamespace(trace=SimpleNamespace(ok=True, info={}, to_record=lambda: {"nodes": []}))


def flow_settings(root, seed):
    settings = contracts_settings(root)
    (root / "p.jsonl").write_text("".join(seed_person(i).model_dump_json() + "\n" for i in range(6)))
    (root / "t.jsonl").write_text("".join(seed_person(i).typing.model_dump_json() + "\n" for i in range(6)))
    personas = settings.personas.model_copy(update={"pool": 4})
    activity = settings.activity.model_copy(
        update={"messages": 45, "conversation_lines": 3, "dm_share": 0.4, "reaction_rate": 0.3}
    )
    return settings.model_copy(
        update={
            "seed": seed,
            "premise_count": 2,
            "personas": personas,
            "corpus": root / "corpus",
            "activity": activity,
        }
    )


async def check_flow(root):
    """One scripted episode through the real control flow: every author's correction turn, ledger review and a
    tasks-only ledger repair, a scene repair from a storyline review, a crash and resume, the tasks phase, the final
    review, publish and the release."""
    root.mkdir(parents=True)
    # A seed whose 4 quota cells mix ledger and workspace categories.
    probe = contracts_settings(root)
    seed = next(s for s in range(100) if len({probe.taxonomy[c].gold == "ledger" for c, *_ in quota(probe.taxonomy, probe.tasks.styles, s, 4)}) == 2)  # fmt: skip
    settings = flow_settings(root, seed)
    log, turns = [], []
    crash = {"armed": True}

    def task_for(i, cell, facts=("f1",), sql=None):
        category, level = cell["category"], cell["level"]
        gold, kind = settings.taxonomy[category].gold, None
        if gold == "ledger":
            sql, kind = sql or "SELECT value AS answer FROM facts WHERE id = 'f1'", "text"
        elif category == "robustness":
            sql, kind, facts = "SELECT id AS answer FROM users WHERE real_name = 'Nobody Here'", "refusal", ()
        elif gold == "hybrid":
            sql, kind = (
                "SELECT COUNT(*) AS answer FROM messages m JOIN evidence e ON e.message_id = m.id WHERE e.fact_id = 'f1'",
                "number",
            )
        else:
            sql, kind, facts = "SELECT COUNT(*) AS answer FROM messages", "number", ()
        return Task(id=f"t{i}", category=category, level=level, actor_id=crash["reader"], question=f"Question {i} about the work?", answer_type=kind, gold_sql=sql, facts=list(facts))  # fmt: skip

    class ScriptedEnv(GenerationEnv):
        async def author_turn(self, interaction, runtime, task_cls, context, attempt, first):
            phase, fed = context["phase"], bool(context.get("feedback"))
            turns.append((attempt, phase, fed))
            if phase == "premise":
                return Premises(premises=[Premise(company=f"{w} Software", niche="n", region="r", size="s", culture="c", cast="x", staffing={"software_developer": 4}) for w in ("Lattice", "Birch")]).model_dump_json()  # fmt: skip
            if phase == "organization":
                ids = [c["user_id"] for c in context["candidates"]]
                crash["reader"] = ids[2]
                a, b, c, d = ids
                kinds = [Routine(kind=k, probability=0.5) for k in ("standup", "handoff", "lunch")]
                channels = [
                    Channel(name="ops", type="public", members=ids, routines=kinds),
                    Channel(name="leads", type="private", members=[a, b], routines=kinds),
                ]
                channels.append(
                    Channel(type="im", members=[a, c] if fed else [a, c, d])
                )  # first: a 3-person DM
                return Organization(people=[Person(user_id=u, title="Engineer", team="Platform") for u in ids], channels=channels, dm_routines=kinds).model_dump_json()  # fmt: skip
            if phase == "ledger":
                ops = channel_id("public", "ops", [])
                a, b = [r[0] for r in self.world.db.execute("SELECT id FROM users ORDER BY id")][:2]
                zone = world_meta(self.world, "zone")
                facts = [
                    Fact(id="f1", storyline="s1", subject="Release 4.2", attribute="decision", value="rollback", anchor="rollback", places=[Place(channel_id=ops, author_id=a, probability=1.0)], day=1, summary="s"),
                    Fact(id="f2", storyline="s1", subject="Release 4.2", attribute="restored", value="service back", places=[Place(channel_id=ops, author_id=b, probability=1.0)], day=2, after=["f1"], happened_at=Moment(day=2, time="14:00", zone=zone), summary="s"),
                    Fact(id="f3", storyline="s2", subject="Audit", attribute="owner", value="Owen", places=[Place(channel_id=ops, author_id=a, probability=1.0)], day=3, summary="s"),
                ]  # fmt: skip
                cells = context["cells"]
                tasks = [task_for(i, cell) for i, cell in enumerate(cells)]
                if not fed and attempt.endswith("01"):
                    tasks = tasks[:-1]  # first: one task short of the quota
                elif fed and attempt.endswith("02"):
                    tasks = [
                        t.model_copy(update={"question": t.question.replace("Question", "Clearer question")})
                        for t in tasks
                    ]
                storylines = [
                    Storyline(id="s1", summary="the 4.2 release"),
                    Storyline(id="s2", summary="the audit"),
                ]
                return Ledger(storylines=storylines, facts=facts, tasks=tasks).model_dump_json()
            if phase == "scenes":
                if context["storyline"] == "s2" and crash["armed"]:
                    crash["armed"] = False
                    raise KeyboardInterrupt("simulated crash")
                ops = channel_id("public", "ops", [])
                a, b, c = [r[0] for r in self.world.db.execute("SELECT id FROM users ORDER BY id")][:3]
                if context["storyline"] == "s2":
                    scenes = [Scene(id="sc4", channel_id=ops, participants=[a, b], day=3, part="morning", situation="audit", length=2, beats=[Beat(fact="f3", author_id=a)])]  # fmt: skip
                else:
                    restored = Scene(id="sc2", channel_id=ops, participants=[b, c], day=2, part="afternoon", situation="restore", length=2, beats=[Beat(fact="f2", author_id=b)], during="f2")  # fmt: skip
                    scenes = [
                        Scene(id="sc1", channel_id=ops, participants=[a, b, c], day=1, part="morning", situation="decide", length=3, beats=[Beat(fact="f1", author_id=a)]),
                        restored if fed else restored.model_copy(update={"during": None}),  # first: stated before it happened
                        Scene(id="sc3", channel_id=ops, participants=[b, c], day=1, part="evening", situation="wrap up", length=2),
                    ]  # fmt: skip
                return ScenePlan(scenes=scenes).model_dump_json()
            cells = context["cells"]
            tasks = [task_for(10 + i, cell) for i, cell in enumerate(cells)]
            if not fed:  # first: a question a ledger task already asks
                tasks[0] = tasks[0].model_copy(update={"question": context["existing_questions"][0]})
            return TaskSet(tasks=tasks).model_dump_json()

        async def compose(self, agents, scene_id, prompt, accept):
            brief_ = json.loads(prompt)
            beats = brief_["scene"]["beats"]
            speakers = [p["user_id"] for p in brief_["participants"]]
            if scene_id.startswith("bg-") and not crash.get(
                "dropped"
            ):  # first: a background line that names a moment
                crash["dropped"] = scene_id
                try:
                    accept(WrittenScene(lines=[Line(author_id=speakers[0], text="see you at {at:f2}")]))
                except ValueError as error:
                    return None, [str(error)]
            lines = [Line(author_id=speakers[0], text="morning all")]
            for beat in beats:
                text = f"{beat['anchor'] or 'it is settled'}, {beat['value']}"
                if beat["fact_id"] == "f2":
                    text = "service back at {at:f2}"
                lines.append(Line(author_id=beat["author_id"], text=text, conveys=[beat["fact_id"]]))
            lines.append(Line(author_id=speakers[-1], text="noted" if "revision_note" not in brief_ else "noted, thanks", reply_to=0))  # fmt: skip
            shape, reacted = brief_["layout"]["shape"], brief_["layout"]["reactions"]
            reactors = speakers + [a["user_id"] for a in brief_["layout"]["audience"]]
            if shape != "free":
                lines = [x.model_copy(update={"reply_to": None if shape == "flat" or i == 0 else 0}) for i, x in enumerate(lines)]  # fmt: skip
            for i in range(min(reacted, len(lines))):
                who = next(u for u in reactors if u != lines[i].author_id)
                lines[i] = lines[i].model_copy(update={"reactions": [Reaction(user_id=who, emoji="eyes")]})
            written = WrittenScene(lines=lines)
            log.append(("write", brief_["scene"]["situation"]))
            if brief_["scene"]["situation"] == "decide" and ("write", "decide") not in log[:-1]:
                try:  # first: a literal time, sent back as a correction turn
                    accept(
                        written.model_copy(
                            update={
                                "lines": [
                                    lines[0].model_copy(update={"text": "morning all, 9:30 sync"}),
                                    *lines[1:],
                                ]
                            }
                        )
                    )
                    raise AssertionError("a literal time was accepted")
                except PlanError:
                    raise
                except ValueError:
                    log.append(("corrected", "decide"))
            accept(written)
            return written, []

        async def solve(self, agents, task):
            path = Path(task.config.tools.db_path)
            assert path.name == "solver.sqlite" and not World(path).db.execute("SELECT name FROM sqlite_master WHERE name = 'tasks'").fetchone()  # fmt: skip
            result = {"task_id": task.data.task_id, "semantic_correctness": 1.0, "correct": True, "grounded": True, "execution_ok": True, "calls": 2}  # fmt: skip
            return result, SimpleNamespace(to_record=lambda: {"nodes": []})

        async def review(self, agents, payload, attempt, files=None, label=""):
            log.append(("review", attempt, label, [t["id"] for t in payload["tasks"]]))
            issues = []
            if payload["phase"] == "ledger" and attempt == "ledger-01":
                issues = [
                    Issue(
                        artifact="tasks",
                        task_ids=[payload["tasks"][0]["id"]],
                        defect="vague",
                        requested_change="sharpen",
                    )
                ]
            if payload["phase"] == "world" and payload.get("storyline") == "s1" and attempt == "build-s1-01":
                sc3 = [
                    m
                    for (m,) in self.world.db.execute(
                        "SELECT message_id FROM scene_messages WHERE scene_id = 'sc3'"
                    )
                ]
                issues = [
                    Issue(artifact="workspace", message_ids=sc3[:1], defect="flat", requested_change="warmer")
                ]
            if payload["phase"] == "world" and "storyline" not in payload and attempt == "final-01":
                (first,) = self.world.db.execute("SELECT MIN(sm.message_id) FROM scene_messages sm JOIN scenes s ON s.id = sm.scene_id WHERE s.storyline IS NULL").fetchone()  # fmt: skip
                issues = [
                    Issue(
                        artifact="workspace", message_ids=[first], defect="stiff", requested_change="looser"
                    )
                ]
            criteria = dict.fromkeys(PHASE_CRITERIA.get(payload["phase"], ()), 1.0)
            verdict = Verdict(approved=not issues, tasks=[TaskReview(task_id=t["id"], valid=True, reason="r", level_fit=4 if payload["phase"] == "task" else None) for t in payload["tasks"]], issues=issues, criteria=criteria, summary="s")  # fmt: skip
            validate_verdict(verdict, payload)
            self.store.artifact(
                attempt, "verdict-" + (label or payload["phase"]), verdict.model_dump(mode="json")
            )
            return verdict

    agents = SimpleNamespace(
        **{
            name: ScriptedAgent()
            for name in ("author", "synthesizer", "builder", "writer", "judge", "solver")
        }
    )
    manifest = provenance(settings)
    store = TestStore(settings.output, manifest)
    env = ScriptedEnv(settings, store)
    await env.setup(agents)
    try:
        await env.run(None, agents)
        raise AssertionError("the simulated crash did not happen")
    except KeyboardInterrupt:
        pass
    store.close()
    fails(Store, settings.output, manifest | {"lock_hash": "changed"})
    store = TestStore(settings.output, manifest)
    assert store.state.built == ["s1"] and store.state.phase == "build", (
        "the checkpoint keeps the approved storyline"
    )
    env = ScriptedEnv(settings, store)
    await env.run(None, agents)
    state = store.state
    assert state.phase == "done" and state.built == ["s1", "s2"]
    background = store.world.db.execute(
        "SELECT c.type, COUNT(DISTINCT s.id) FROM scenes s JOIN channels c ON c.id = s.channel_id WHERE s.storyline IS NULL GROUP BY 1"
    ).fetchall()
    assert {kind for kind, _ in background} >= {"public", "im"}, ("everyday conversations fill channels and DMs", background)  # fmt: skip
    assert store.world.db.execute("SELECT COUNT(*) FROM reactions").fetchone()[0], (
        "the layouts' reactions are written"
    )
    assert (
        any(
            e["event"] == "background_dropped" and e["scene_ids"] == [crash["dropped"]]
            for e in map(json.loads, (store.root / "progress.jsonl").read_text().splitlines())
        )
        and store.world.db.execute("SELECT 1 FROM scenes WHERE id = ?", (crash["dropped"],)).fetchone()
    ), (  # fmt: skip
        "a background conversation its writer could not write is dropped, then written on the next pass"
    )
    notes = [json.loads(p)["revision_note"] for (p,) in store.world.db.execute("SELECT plan_json FROM scenes WHERE storyline IS NULL")]  # fmt: skip
    assert notes.count("stiff Requested change: looser") == 1 and state.rounds["final"] == 2, (
        "a final-review issue on a background message rewrites only that conversation"
    )
    retried = SimpleNamespace(
        ok=False, errors=[SimpleNamespace(type="TaskError", message="malformed verdict")]
    )
    limit = SimpleNamespace(type="ReviewLimit", message="no valid ledger")
    assert failure("done", SimpleNamespace(errors=[], traces=[retried])) is None, (
        "a judge failure the review loop retried does not fail a finished run"
    )
    assert failure("build", SimpleNamespace(errors=[], traces=[retried])) == "TaskError: malformed verdict"
    assert failure("done", SimpleNamespace(errors=[limit], traces=[])) == "ReviewLimit: no valid ledger"
    assert [(a, p, f) for a, p, f in turns if p == "organization"] == [
        ("organization", "organization", False),
        ("organization", "organization", True),
    ]
    assert [(a, f) for a, p, f in turns if p == "ledger"] == [
        ("ledger-01", False),
        ("ledger-01", True),
        ("ledger-02", True),
    ]
    assert [(a, f) for a, p, f in turns if p == "scenes"][:2] == [
        ("build-s1-01", False),
        ("build-s1-01", True),
    ]
    assert ("corrected", "decide") in log and [p for _, p, f in turns if p == "tasks"] == ["tasks", "tasks"]
    events = [json.loads(line) for line in (store.root / "progress.jsonl").read_text().splitlines()]
    routed = [(e["attempt"], e["route"]) for e in events if e["event"] == "attempt_routed"]
    assert routed[:2] == [("build-s1-01", "plan"), ("build-s1-02", "repair")], routed
    assert not any(e["event"] == "approvals_invalidated" for e in events), (
        "a tasks-only ledger repair keeps the world"
    )
    assert any(e["event"] == "interrupted" for e in events)
    rewritten = [e[1] for e in log if e[0] == "write"]
    assert rewritten.count("wrap up") == 2 and rewritten.count("decide") == 1, (
        "the repair rewrote only the named scene"
    )
    reviewed = {tuple(e[3]) for e in log if e[0] == "review" and e[2] == "tasks"}
    assert all(len(ids) for ids in reviewed) and set(state.task_reviews) == {
        t for (t,) in store.world.db.execute("SELECT id FROM tasks")
    }
    assert store.world.violations(complete=True) == []
    assert not state.notes, "every note was consumed by writing its scene"
    difficulty = store.summary("complete")["difficulty"]
    assert set(difficulty) == set(state.task_reviews) and all(
        d["level_fit"] == 4 and "evidence_pages" in d and d["concept"] for d in difficulty.values()
    ), "the summary reports each task's measured difficulty beside its level"
    assert set(state.frozen) == {
        s for (s,) in store.world.db.execute("SELECT id FROM scenes WHERE storyline IS NOT NULL")
    }, "approved scenes are frozen"
    ledger_ids = {t for (t,) in store.world.db.execute("SELECT id FROM tasks WHERE gold_source = 'ledger'")}
    final = next(e[3] for e in log if e[0] == "review" and e[1] == "final-01" and e[2] == "tasks")
    assert set(final) == set(state.task_reviews) - ledger_ids, "the final review re-reviews only stale tasks"
    saved = state.model_copy(deep=True)
    sql_task = next(iter(set(state.task_reviews) - ledger_ids))
    for issue, phase in (
        (
            Issue(artifact="tasks", task_ids=[sorted(ledger_ids)[0]], defect="d", requested_change="c"),
            "ledger",
        ),
        (Issue(artifact="tasks", task_ids=[sql_task], defect="d", requested_change="c"), "tasks"),
        (Issue(artifact="ledger", defect="d", requested_change="c"), "ledger"),
    ):
        env.route_rejection(Verdict(approved=False, tasks=[], issues=[issue], summary="s"))
        assert state.phase == phase and not env.repairs("s1"), (issue, state.phase)
    sc3 = [
        m for (m,) in store.world.db.execute("SELECT message_id FROM scene_messages WHERE scene_id = 'sc3'")
    ]
    env.route_rejection(Verdict(approved=False, tasks=[], issues=[Issue(artifact="workspace", message_ids=sc3[:1], defect="d", requested_change="c")], summary="s"))  # fmt: skip
    assert state.phase == "build" and state.built == ["s2"] and set(env.repairs("s1")) == {"sc3"}
    state.notes = {}
    ledger_issue = Issue(artifact="ledger", defect="d", requested_change="c")
    sc3_issue = Issue(artifact="workspace", message_ids=sc3[:1], defect="stiff", requested_change="looser")
    env.route_rejection(Verdict(approved=False, tasks=[], issues=[ledger_issue, sc3_issue], summary="s"))
    assert state.phase == "ledger" and env.repairs("s1") == {"sc3": "stiff Requested change: looser"}, (
        "a workspace issue behind a ledger repair waits for its scene"
    )
    assert "s1" not in state.built, "its storyline is built again after the ledger"
    used = state.rounds["build:s1"]
    state.refunded = {}
    try:
        with (
            store.world.trial() as copy
        ):  # a ledger whose facts changed clears the world and refunds the builds
            changed = Ledger.model_validate_json(json.dumps(state.drafts["ledger"]))
            changed.facts[0] = changed.facts[0].model_copy(update={"value": "roll forward", "anchor": None})
            env.replace_ledger(copy, changed)
            raise LookupError("drop the trial")
    except LookupError:
        pass
    assert state.refunded["build:s1"] == used and store.reserve("build:s1", used).endswith(
        f"-{used + 1:02d}"
    ), "a cleared world gives its storylines their rounds back, under new attempt ids"
    store.state = state = saved
    review = state.task_reviews.pop(sql_task)
    fails(store.publish)
    state.task_reviews[sql_task] = review
    store.world.insert("facts", [dict(id="f9", storyline="s2", subject="x", attribute="y", value="z", channel_id=channel_id("public", "ops", []), author_id=crash["reader"], day=3, summary="s")])  # fmt: skip
    store.world.insert("task_facts", [dict(task_id=sql_task, fact_id="f9")])
    fails(store.publish)
    store.world.db.execute("DELETE FROM task_facts WHERE fact_id = 'f9'")
    store.world.db.execute("DELETE FROM facts WHERE id = 'f9'")
    store.publish()
    world, rows, answers = load_release(store.root / "release")
    assert {r.task_id for r in rows} == set(state.task_reviews) and World(world).db.execute(
        "SELECT COUNT(*) FROM messages"
    ).fetchone()[0]
    assert all(a.rows or a.answer_type == "refusal" for a in answers.values())
    ledger = next(r for r in rows if settings.taxonomy[r.category].gold == "ledger")
    assert answers[ledger.task_id].rows == [{"answer": "rollback"}] and answers[ledger.task_id].messages
    store.publish()  # idempotent
    store.close()
    print(
        "PASS flow: corrections, ledger review and tasks-only repair, scene repair, crash and resume, tasks, final, publish"
    )


class FakeRuntime:
    """The author's VM as files: what the env writes, the author reads and writes, and rm."""

    def __init__(self):
        self.files = {}

    async def write(self, path, data):
        self.files[path] = data

    async def read(self, path, max_bytes=None):
        if path not in self.files:
            raise SandboxError(f"no {path}")
        return self.files[path]

    async def run(self, argv, env):
        for path in argv[2:] if argv[:2] == ["rm", "-f"] else []:
            self.files.pop(path, None)
        return SimpleNamespace(exit_code=0, stdout="", stderr="")


class ScriptedAuthor(ScriptedAgent):
    """A world author played by a script: each turn runs it against the turn's own world tools and VM."""

    def __init__(self, script):
        self.script = script

    @asynccontextmanager
    async def provision(self, task):
        yield FakeRuntime()

    @asynccontextmanager
    async def interaction(self, task, runtime):
        tools = None
        if isinstance(task, WorldAuthorTask):
            (tools,) = task.toolsets(task.config)
            await tools.setup()
            with_state(tools)

        async def turn(prompt):
            await self.script(prompt, tools, runtime)
            return SimpleNamespace(terminated=False)

        yield SimpleNamespace(trace=SimpleNamespace(ok=True, info={}), turn=turn)


async def check_author(root):
    """One world written by a scripted author through the v7 driver: setup, the plan, days that close only when done,
    a review after day 2 whose issues open day 3, a crash in day 3 and a resume from day 2's world and notes, tasks,
    probes within their budget and a hardening turn, a final rejection fixed in the same session, publish."""
    root.mkdir(parents=True)
    base = flow_settings(root, 3)
    settings = base.model_copy(
        update={
            "calendar": base.calendar.model_copy(update={"days": 4}),
            "activity": base.activity.model_copy(update={"messages": 16}),
            "author": base.author.model_copy(
                update={
                    "enabled": True,
                    "tolerance": 1.0,
                    "share_tolerance": 1.0,
                    "review_days": [2],
                    "probe_solves": 2,
                    "probe_rounds": 1,
                    "probe_budget": 10,
                }  # fmt: skip
            ),
        }
    )
    seen, crash = [], {"armed": True}

    class AuthorEnv(GenerationEnv):
        async def author_turn(self, interaction, runtime, task_cls, context, attempt, first):
            if context["phase"] == "premise":
                return Premises(premises=[Premise(company=f"{w} Software", niche="n", region="r", size="s", culture="c", cast="x", staffing={"software_developer": 4}) for w in ("Lattice", "Birch")]).model_dump_json()  # fmt: skip
            ids = [c["user_id"] for c in context["candidates"]]
            kinds = [Routine(kind=k, probability=0.5) for k in ("standup", "handoff", "lunch")]
            channels = [Channel(name="ops", type="public", members=ids, routines=kinds), Channel(name="leads", type="private", members=ids[:2], routines=kinds)]  # fmt: skip
            return Organization(people=[Person(user_id=u, title="Engineer", team="Platform") for u in ids], channels=channels, dm_routines=kinds).model_dump_json()  # fmt: skip

        async def solve(self, agents, task):
            return {"semantic_correctness": 0.5, "correct": True, "grounded": True, "execution_ok": True, "calls": 3, "reason": "r"}, SimpleNamespace(to_record=lambda: {"nodes": []})  # fmt: skip

        async def review(self, agents, payload, attempt, files=None, label=""):
            seen.append(
                ("review", attempt, payload["phase"], "written_through" in payload, "ledger" in payload)
            )
            issues = []
            if payload["phase"] == "world" and attempt in ("review-01", "final-01"):
                first, second = [
                    m for (m,) in self.world.db.execute("SELECT id FROM messages ORDER BY id LIMIT 2")
                ]
                issues = [Issue(artifact="workspace", message_ids=[second], evidence_message_ids=[first], defect="stiff", requested_change="looser")]  # fmt: skip
            criteria = dict.fromkeys(PHASE_CRITERIA.get(payload["phase"], ()), 1.0)
            verdict = Verdict(approved=not issues, tasks=[TaskReview(task_id=t["id"], valid=True, reason="r", level_fit=3 if payload["phase"] == "task" else None) for t in payload["tasks"]], issues=issues, criteria=criteria, summary="s")  # fmt: skip
            validate_verdict(verdict, payload)
            return verdict

    async def script(prompt, tools, runtime):
        """The author: plans, writes each day, writes tasks, hardens two of them, fixes what the review names."""
        call = lambda name, **kwargs: tools._with_state(getattr(tools, name))(**kwargs)  # noqa: E731
        world = World(tools.config.db_path)
        a, b, c, d = [r[0] for r in world.db.execute("SELECT id FROM users ORDER BY id")]
        ops, leads = channel_id("public", "ops", []), channel_id("private", "leads", [])
        assert runtime.files["/task/memory/now.md"].startswith(b"# Now: "), (
            "memory is rendered before every turn"
        )
        seen.append(("turn", tools.config.mode, tools.config.day, prompt.split(":")[0][:40]))

        def talk(channel, *lines):
            lines = [PostLine(**line) for line in lines]
            return call("post", conversation=Conversation(channel_id=channel, about="work", lines=lines))

        async def close():
            while today(world) == tools.config.day:
                await call("advance")

        if prompt.startswith("Before day 1"):
            events = [Event(id="e1", storyline="s2", title="audit window", day=3, time="10:00"), Event(id="e2", storyline="s1", title="service restored", day=2, time="14:00")]  # fmt: skip
            facts = [
                PlanFact(id="f1", storyline="s1", subject="Release 4.2", attribute="decision", value="rollback", anchor="rollback", channel_id=ops, author_id=a, day=1, summary="s"),
                PlanFact(id="f2", storyline="s1", subject="Release 4.2", attribute="state", value="service back", channel_id=ops, author_id=b, day=2, after=["f1"], event="e2", kind="happened", summary="s"),
                PlanFact(id="f3", storyline="s2", subject="Audit", attribute="window", value="dry run", anchor="dry run", channel_id=leads, author_id=a, day=3, event="e1", kind="scheduled", summary="s"),
                PlanFact(id="f4", storyline="s2", subject="Audit", attribute="owner", value="Owen", channel_id=ops, author_id=a, day=3, event="e1", kind="scheduled", summary="s"),
            ]  # fmt: skip
            await call("plan", ledger=Plan(storylines=[Storyline(id="s1", summary="the release"), Storyline(id="s2", summary="the audit")], events=events, facts=facts))  # fmt: skip
            await runtime.write("/task/notes/plan.md", b"arcs; board: semantic and search rest on f3 and f4")
        elif tools.config.mode == "day":
            day = tools.config.day
            if day == 1 and prompt.startswith(
                "It is"
            ):  # first turn: a recap is written, but the day is left open
                await talk(ops, dict(author_id=a, text="we go with the rollback", conveys=["f1"]), dict(author_id=b, text="ok, notes tomorrow", reply_to=0, commits=[Commit(id="c1", text="post the notes", due_day=2)]))  # fmt: skip
                await runtime.write("/task/notes/recap.md", b"day 1: decided, still open")
                return
            if day == 2:
                await call("advance", to="evening")
                await talk(ops, dict(author_id=b, text="service back at {at:e2}; notes are up", conveys=["f2"], closes=[Close(id="c1", status="kept")]))  # fmt: skip
            if day == 3:
                assert "message_ids" in prompt and "evidence_message_ids" in prompt, (
                    "the review's issues open day 3"
                )
                if crash["armed"]:
                    await talk(ops, dict(author_id=c, text="morning"))
                    crash["armed"] = False
                    raise KeyboardInterrupt("simulated crash")
                assert b"board" in runtime.files["/task/notes/plan.md"], (
                    "the author's notes come back with its world"
                )
                issue = json.loads(prompt[prompt.index("[{") : prompt.index("}]") + 2])[0]
                for message in issue["message_ids"]:
                    await call("revise", message_id=message, text="ok, notes tomorrow then")
                seen.append(("revised", issue["message_ids"]))
                await talk(
                    leads, dict(author_id=a, text="dry run for the audit window at {at:e1}", conveys=["f3"])
                )
                await talk(ops, dict(author_id=a, text="Owen owns the audit at {at:e1}", conveys=["f4"]), dict(author_id=d, text="thanks"))  # fmt: skip
            if day == 4:
                assert "message_ids" not in prompt, "a review's issues are delivered once"
            if day == 4 and prompt.startswith(
                "It is"
            ):  # first turn: the day is closed, but no recap is written
                await talk(ops, dict(author_id=d, text="quiet day"))
                await close()
                return
            await close()
            await runtime.write("/task/notes/recap.md", f"day {day}: done".encode())
        elif prompt.startswith("The last day is closed"):
            for i, (category, level, *_) in enumerate(json.loads(tools.config.context)["cells"]):
                await call("add_task", task=scripted_task(settings, i, category, level, a))
        elif prompt.startswith("The solver's tries"):
            for i, (category, level, *_) in list(enumerate(json.loads(tools.config.context)["cells"]))[1:3]:
                task = scripted_task(settings, i, category, level, a)
                await call(
                    "add_task", task=task.model_copy(update={"question": "Asked again: " + task.question})
                )
        elif prompt.startswith("The review rejected"):
            issues = json.loads(prompt[prompt.index("[") : prompt.index("]\n") + 1])
            for issue in issues:
                for message in issue.get("message_ids", []):
                    await call("revise", message_id=message, text="ok, the notes come tomorrow")
            seen.append(("fixed", [m for i in issues for m in i.get("message_ids", [])]))

    agents = SimpleNamespace(
        **{name: ScriptedAgent() for name in ("synthesizer", "builder", "writer", "judge", "solver")},
        author=ScriptedAuthor(script),
    )
    manifest = provenance(settings)
    store = TestStore(settings.output, manifest)
    env = AuthorEnv(settings, store)
    await env.setup(agents)
    try:
        await env.run(None, agents)
        raise AssertionError("the simulated crash did not happen")
    except KeyboardInterrupt:
        pass
    after_day_2 = World(store.root / "attempts" / store.state.restore_point / "world.sqlite").db.execute("SELECT COUNT(*) FROM messages").fetchone()[0]  # fmt: skip
    assert store.world.db.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == after_day_2 + 1, "the crash left a partial day"  # fmt: skip
    store.close()
    store = TestStore(settings.output, manifest)
    assert store.state.phase == "day" and store.state.day == 3 and store.state.restore_point == "day-02-01"
    env = AuthorEnv(settings, store)
    await env.run(None, agents)
    state = store.state
    assert state.phase == "done" and state.rounds["day-03"] == 2, state.rounds
    assert not store.world.db.execute("SELECT 1 FROM messages WHERE text = 'morning'").fetchone(), (
        "the crashed day's start is gone"
    )
    assert [t[3][:6] for t in [s for s in seen if s[0] == "turn"] if t[1] == "day" and t[2] == 4] == [
        "It is ",
        "Day 4 ",
    ], "a day needs its recap"
    turns = [s for s in seen if s[0] == "turn"]
    assert [t[3][:6] for t in turns if t[1] == "day" and t[2] == 1] == ["It is ", "Day 1 "], "an open day is not accepted"  # fmt: skip
    first, second = [m for (m,) in store.world.db.execute("SELECT id FROM messages ORDER BY id LIMIT 2")]
    assert ("revised", [second]) in seen and ("fixed", [second]) in seen, (
        "only the messages to change are revised"
    )
    assert store.world.db.execute("SELECT text FROM messages WHERE id = ?", (second,)).fetchone()[0] == "ok, the notes come tomorrow"  # fmt: skip
    assert "rollback" in store.world.db.execute("SELECT text FROM messages WHERE id = ?", (first,)).fetchone()[0], "the evidence stays"  # fmt: skip
    assert state.probe_solves == 10 and len(state.probed) == 4, (state.probe_solves, state.probed)
    stale = [
        t for (t,) in store.world.db.execute("SELECT id FROM tasks") if state.probed.get(t) != env.task_key(t)
    ]
    assert stale == ["t2"], ("the budget left the second hardened task untried", stale)
    assert ("review", "review-01", "world", True, True) in seen and (
        "review",
        "final-01",
        "world",
        False,
        True,
    ) in seen
    assert (store.root / "attempts" / "day-03-02" / "notes" / "recap.md").read_text() == "day 3: done"
    days = [json.loads(line) for line in (store.root / "progress.jsonl").read_text().splitlines()]
    assert [e["day"] for e in days if e["event"] == "day_closed"] == [1, 2, 3, 4]
    assert store.world.violations(complete=True) == [] and set(state.task_reviews) == {t for (t,) in store.world.db.execute("SELECT id FROM tasks")}  # fmt: skip
    ids = [r[0] for r in store.world.db.execute("SELECT id FROM messages ORDER BY ts_us")]
    assert ids == sorted(ids), "the world was written in time order"
    store.publish()
    world, rows, answers = load_release(store.root / "release")
    assert len(rows) == 4 and World(world).db.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    assert (store.root / "world-calls.jsonl").exists()
    store.close()
    print(
        "PASS author: plan, days closed only when done, review issues delivered, crash and resume, tasks, probes, fix"
    )


def scripted_task(settings, i, category, level, actor):
    """A valid task for any cell of the scripted world: its gold follows the category's source."""
    gold, spec = settings.taxonomy[category].gold, settings.taxonomy[category]
    question = f"Question {i} about the {category} level {level} work?"
    if category == "robustness":
        sql, kind, facts = "SELECT id AS answer FROM users WHERE real_name = 'Nobody Here'", "refusal", []
    elif gold == "ledger":
        sql, kind, facts = "SELECT value AS answer FROM facts WHERE id = 'f4'", "text", ["f3", "f4"]
    elif gold == "hybrid":
        sql, kind, facts = "SELECT COUNT(*) AS answer FROM messages m JOIN evidence e ON e.message_id = m.id WHERE e.fact_id = 'f1'", "number", ["f1"]  # fmt: skip
    else:
        sql, kind, facts = "SELECT COUNT(*) AS answer FROM messages", "number", []
    assert kind in spec.answer_types
    return Task(id=f"t{i}", category=category, level=level, actor_id=actor, question=question, answer_type=kind, gold_sql=sql, facts=facts)  # fmt: skip


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as directory:
        asyncio.run(check_world(Path(directory) / "world"))
        check_tasks(Path(directory) / "tasks")
        check_contracts(Path(directory) / "contracts")
        check_clock(Path(directory) / "clock")
        asyncio.run(check_tools(Path(directory) / "tools"))
        asyncio.run(check_reviews(Path(directory) / "reviews"))
        asyncio.run(check_flow(Path(directory) / "flow"))
        asyncio.run(check_author(Path(directory) / "author"))
    print("All generation checks passed.")
