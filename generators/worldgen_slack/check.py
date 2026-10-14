"""Offline invariant checks, one per rule group the pipeline relies on: uv run --frozen python -m generators.worldgen_slack.check"""

import ast
import asyncio
import inspect
import io
import json
import math
import os
import shutil
import sqlite3
import statistics
import subprocess
import sys
import tarfile
import time
from collections import Counter
from contextlib import asynccontextmanager, contextmanager
from types import SimpleNamespace
import random
import re
import tempfile
import tomllib
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import verifiers.v1 as vf
import verifiers.v1.utils.interrupt
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from verifiers.v1.dialects.base import append_user_notice
from verifiers.v1.errors import SandboxError
from verifiers.v1.mcp.launch import serve
from worldgen_slack.dataset import PrivateAnswer, PublicTask, load_release, sha256, write_release
from worldgen_slack import surface
from worldgen_slack.db import ANSWER_KEY, SHADOWED, SHOWN, World
from worldgen_slack.surface import PAGE, briefing
from worldgen_slack.taskset import EMPTY, AnswerGrade, AnswerJudge, SolverTask, compact, people_shown, shown
from worldgen_slack.tools import SlackTools, WorldToolsConfig, file_hash, stage_world, watch_parent
from .agents.inspection import ReviewState
from .agents.judge import PHASE_GUIDES, JudgeTask, review_payload
from .agents.synthesizer import parse_premise
from .agents.world import AUTHOR_GUIDE, CHAPTER_GUIDE, CRAFT_GUIDE, WorldAuthorTask, context_of, files, forge_guide, ledger_digest, ledger_document
from .chronicle import (
    share_band,
    Move,
    Artifact,
    edit,
    move_moment,
    promises_made,
    tally,
    Annotation,
    forge_task,
    BoardEntry,
    Close,
    Commit,
    Conversation,
    Event,
    Plan,
    PlanFact,
    PostLine,
    add_task,
    advance,
    agenda_budget,
    band,
    bounds,
    close_day,
    front_loaded,
    opening,
    part_of,
    post,
    posted,
    present,
    quotas,
    record_plan,
    revise,
    room,
    start_clock,
    today,
    today_line,
    unplanned,
)
from .chapter import ChapterConfig, ChapterEnv, Material, adopt, brief_slots, chapter_settings, same_schema, version_of
from .chapter import provenance as chapter_provenance
from .env import GenerationEnv, ended, keep_awake, rates, read_one_way, try_digest
from .generate import failure, provenance, run_label
from .generate import run as generate_run
from .store import ReviewLimit, SessionLost, Store
from .forge import ForgeConfig, ForgeEnv, bucket, measured_level
from .forge import measured as forge_measured
from .profile import profile
from .forge import provenance as forge_provenance
from .forge import run as forge_run
from .config import ROOT, Acceptance, Category, Config, Needs
from .contracts import (
    READING,
    Integration,
    UserGroup,
    bot_id,
    miss_class,
    band_move,
    DAY,
    PARTS,
    PHASE_CRITERIA,
    IDENTIFIER,
    TIME_LITERAL,
    Channel,
    Gaps,
    Issue,
    Moment,
    Organization,
    Person,
    Reaction,
    Routine,
    Premise,
    Premises,
    SeedPersona,
    Slot,
    Storyline,
    Task,
    TaskReview,
    Typing,
    Verdict,
    accepted_task,
    at,
    chapter_start,
    frozen_before,
    frozen_digest,
    meta,
    style,
    background_plan,
    failure_cases,
    features,
    panel,
    channel_id,
    check_task,
    deciding,
    direct_messages,
    measures,
    organize,
    pick_cast,
    quota,
    record_task,
    render,
    reread,
    cards,
    habit,
    HABITS,
    user_id,
    validate_verdict,
    verdict_schema,
    window,
    world_meta,
)


async def offline(*args, **kwargs):
    raise AssertionError("the offline checks call no model: script the agent or the judge")




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
    row = dict(id="t4", category="temporal", actor_id="U2", question="When?", answer_type="text", gold_source="sql", gold_sql="SELECT 1 AS answer")  # fmt: skip
    try:  # the taxonomy is data: any category, any level from 1
        with world.trial() as copy:
            copy.insert("tasks", [row | {"level": 4}])
            raise LookupError("drop the trial")
    except LookupError:
        pass
    try:
        with world.trial() as copy:
            copy.insert("tasks", [row | {"level": 0}])
        raise AssertionError("a level-0 task was inserted")
    except sqlite3.IntegrityError:
        pass
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
        "told_late",
        facts=[
            fact
            | dict(
                id="f2",
                subject="a",
                attribute="b",
                day=1,
                moment_us=at(1, 9),
                moment_zone="UTC",
                moment_kind="scheduled",
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

    def read(reader, tool, **arguments):
        return surface.read(reader, tool, arguments)

    def texts(reader, *keywords, filters="", **kwargs):
        found = read(reader, "search_public_and_private", keywords=list(keywords), filters=filters, **kwargs)
        return [i["text"] for i in found["items"]]

    u1, u3 = World(root / "solver.sqlite", actor="U1"), World(root / "solver.sqlite", actor="U3")
    fails(read, solver, "read_channel", channel_id="G1")
    fails(read, u3, "read_channel", channel_id="G1")
    assert read(u3, "list_user_channels")["items"] == [] and texts(solver, "deleted") == []
    assert texts(solver, "rollback", "budget") == [] and texts(u1, "rollback", "budget") == ["private: rollback budget approved"]  # fmt: skip
    assert "can you check the rollback" in texts(solver, "rollback") and "can you check the rollback" not in texts(u3, "rollback")  # fmt: skip
    assert texts(solver, "rollback", filters="is:dm") == ["can you check the rollback"] and texts(u1, "rollback", filters="in:@u2") == ["can you check the rollback"]  # fmt: skip
    assert [i["text"] for i in read(solver, "search_public", keywords=["rollback"])["items"]] == texts(solver, "rollback", channel_types="public_channel"), "the public search reads public channels only"  # fmt: skip
    every = "public_channel,private_channel,mpim,im"
    assert [c["id"] for c in read(solver, "list_user_channels", types=every)["items"]] == ["C1", "D1"]
    assert [c["id"] for c in read(solver, "list_user_channels")["items"]] == ["C1"], "direct conversations only when asked"  # fmt: skip
    assert read(solver, "list_user_channels", types="im")["items"][0]["users"] == ["U1", "U2"], "a DM names both members"  # fmt: skip
    assert read(solver, "read_channel", channel_id="U1")["channel"] == "D1", "a person's id reads the DM with them"
    # The client says whom the assistant acts as and when it is; the tools show every time on the user's own clock.
    assert briefing(solver) == "You are using Slack as the user with user_id U2. It is Thursday, June 18, 2026, 00:00 on your clock (America/Chicago)."  # fmt: skip
    first = read(solver, "read_channel", channel_id="C1")["items"][-1]
    assert first["time_local"] == datetime.fromtimestamp(float(first["ts"]), ZoneInfo("America/Chicago")).strftime("%Y-%m-%d %H:%M:%S -05"), first  # fmt: skip
    fall = int(datetime(2026, 11, 1, 6, 30, tzinfo=ZoneInfo("UTC")).timestamp() * 1e6)  # 1:30 CDT, then CST
    assert [surface.clock(solver, fall + h * 3_600_000_000)[-12:] for h in (0, 1)] == ["01:30:00 -05", "01:30:00 -06"]  # fmt: skip
    assert [m["text"] for m in read(u3, "read_channel", channel_id="C1")["items"]] == ["notes posted for release 4.2", texts(u3, "blocked")[0]]  # fmt: skip
    root_message = read(u3, "read_channel", channel_id="C1")["items"][1]
    assert root_message["reply_count"] == 1 and root_message["reactions"] == [{"name": "eyes", "count": 1}], "counts, not who"  # fmt: skip
    root_ts = root_message["ts"]
    assert [m["text"] for m in read(u3, "read_thread", channel_id="C1", message_ts=root_ts)["items"]][1] == "ack, rolling back now"  # fmt: skip
    assert texts(solver, "release")[0].startswith("Release 4.2 rollback"), "BM25 ranks the denser, older match first"
    assert texts(solver, "release", sort="timestamp")[0] == "notes posted for release 4.2"
    assert texts(solver, "release", sort="timestamp", sort_dir="asc")[0].startswith("Release 4.2 rollback")
    assert texts(solver, '"rolling back"') == ["ack, rolling back now"]
    assert texts(solver, "release", filters="from:@u2") == ["notes posted for release 4.2"] == texts(solver, "4.2", filters="in:#releases on:2026-06-16")  # fmt: skip
    assert texts(solver, "4.2", filters="before:2026-06-16") == texts(solver, "blocked") and texts(solver, "4.2", filters="after:2026-06-15") == texts(solver, "notes")  # fmt: skip
    assert texts(solver, "release", filters="in:#releases in:#exec") == texts(solver, "release"), "a repeated modifier is OR"  # fmt: skip
    assert texts(solver, "rollback", filters="in:#exec") == [] and texts(solver, "release", "-notes") == texts(solver, "blocked")  # fmt: skip
    assert texts(solver, filters="has:reaction") == [texts(solver, "blocked")[0]] and texts(solver, filters="is:thread") == ["ack, rolling back now", texts(solver, "blocked")[0]]  # fmt: skip
    context = read(solver, "search_public_and_private", keywords=["ack"])["items"][0]["context"]
    assert [m["ts"] for m in context["before"]] == [root_ts] and context["after"] == [], "a reply's context is its thread"  # fmt: skip
    page = read(solver, "search_public_and_private", keywords=["release"], limit=1)
    assert read(solver, "search_public_and_private", keywords=["release"], cursor=page["next_cursor"], limit=1)["items"][0]["text"] == "notes posted for release 4.2"  # fmt: skip
    fails(read, solver, "search_public_and_private", keywords=["rollback"], cursor=page["next_cursor"])
    fails(read, solver, "search_public_and_private", keywords=["release"], limit=21)
    fails(read, solver, "search_public_and_private", keywords=[])
    assert read(solver, "get_reactions", channel_id="C1", message_ts=root_ts)["reactions"] == [{"name": "eyes", "count": 1, "users": ["U2"], "names": ["Owen Sato"]}]  # fmt: skip
    assert read(solver, "read_user_profile", user_id="U1")["handle"] == "u1" and read(solver, "read_user_profile")["id"] == "U2"  # fmt: skip
    assert [p["id"] for p in read(solver, "search_users", keywords=["ali"])["items"]] == ["U1"], "a partial name finds its person"  # fmt: skip
    assert [c["name"] for c in read(solver, "search_channels", keywords=["rel"], channel_types="public_channel,private_channel")["items"]] == ["releases"], "private channels stay unlisted"  # fmt: skip
    assert [p["id"] for p in read(u1, "list_channel_members", channel_id="G1")["items"]] == ["U1"] and read(u1, "list_channel_members", channel_id="G1", response_format="ids_only")["format"] == "ids_only"  # fmt: skip
    fails(read, solver, "list_channel_members", channel_id="D1")  # not a direct conversation
    # What the solver reads is the real server's text (tests/fixtures/slack-mcp): its shapes, rendered from records.
    fixture = Path(__file__).resolve().parents[2] / "environments/worldgen_slack/tests/fixtures/slack-mcp"
    served = {t["name"]: t for t in json.loads((fixture / "tools.json").read_text())["tools"]}
    shown_ = surface.render(solver, "read_channel", read(solver, "read_channel", channel_id="C1"))
    assert shown_["messages"].startswith("Channel: #releases (C1)\n\n=== Message from Owen Sato <U2@example.com> (U2) at ") and "\nMessage TS: " in shown_["messages"] and "\nThread: 1 replies (latest: " in shown_["messages"] and "\nReactions: eyes (1)" in shown_["messages"], shown_  # fmt: skip
    assert shown_["pagination_info"] == "There are no more messages available.\n"
    hit = surface.render(solver, "search_public_and_private", read(solver, "search_public_and_private", keywords=["ack"]))
    assert hit["results"].startswith("# Search Results for: \n\n## Messages (1 results)\n### Result 1 of 1\nChannel: #releases (ID: C1)\nFrom: Owen Sato <U2@example.com> (ID: U2) \n") and "\nContext before: \n- From: Alicia Rao" in hit["results"] and hit["pagination_info"] == "End of results - No more pages available.\\n", hit  # fmt: skip
    assert surface.render(solver, "get_reactions", read(solver, "get_reactions", channel_id="C1", message_ts=root_ts)) == {"result": "Reactions on message:\n\n:eyes: × 1 — Owen Sato (U2)"}  # fmt: skip
    assert surface.render(solver, "search_public_and_private", read(solver, "search_public_and_private", keywords=["nothinghere"])) == json.loads((fixture / "responses.json").read_text())[16]["response"], "an empty search reads as the real one"  # fmt: skip
    assert texts(solver, "4.2", filters="on:2026-06-15") == texts(solver, "blocked"), "on: is one day"
    assert surface.text_of(solver, "hi <@U1>, see <#C1>") == "hi <@U1|Alicia>, see <#C1|releases>", "names, as Slack's API gives them"  # fmt: skip
    try:
        with world.trial() as copy:  # a bot's post, a second reply: searched only with include_bots; a thread oldest first
            copy.db.execute("UPDATE users SET is_bot = 1 WHERE id = 'U2'")
            copy.insert("messages", [dict(channel_id="C1", user_id="U1", ts_us=int(float(root_ts) * 1e6) + 9_000_000_123, text="all clear now", parent_id=1)])  # fmt: skip
            bots = World(copy.path, actor="U1")
            assert texts(bots, "notes") == [] and texts(bots, "notes", include_bots=True) == ["notes posted for release 4.2"]  # fmt: skip
            assert [m["text"] for m in read(bots, "read_thread", channel_id="C1", message_ts=root_ts)["items"]][1:] == ["ack, rolling back now", "all clear now"]  # fmt: skip
            bots.close()
            raise LookupError("drop the trial")
    except LookupError:
        pass
    fails(read, solver, "read_user_profile", user_id="U9")
    try:
        read(solver, "read_channel", channel_id="C9")
    except ValueError as error:
        assert str(error) == "execution_failed: channel_not_found", error

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
    search = toolset._with_state(toolset.search_public_and_private)
    answers_ = await asyncio.gather(*(search(keywords=["rollback"]) for _ in range(10)))
    assert len(stored["state"].calls) == 10 and stored["state"].calls[0].tool == "search_public_and_private"
    assert answers_[0]["results"].startswith("# Search Results for: ") and stored["state"].calls[0].output["items"], "the solver reads text; grading keeps the record"  # fmt: skip
    async with (
        asyncio.timeout(30),
        serve(toolset) as url,
        streamable_http_client(url) as (reader, writer_),
        ClientSession(reader, writer_) as client,
    ):
        await client.initialize()
        # The tool server adds its prefix as the solver's harness names a server's tools: slack_read_channel.
        listed = {f"{SlackTools.TOOL_PREFIX}_{t.name}": t for t in (await client.list_tools()).tools}
        assert set(listed) == set(served), sorted(listed)
        for name, tool in listed.items():  # every parameter the real server takes, by its name and kind
            want = served[name]["input_schema"]["properties"]
            assert set(tool.input_schema["properties"]) == set(want), (name, set(tool.input_schema["properties"]) ^ set(want))  # fmt: skip
            assert set(tool.input_schema.get("required", [])) == set(served[name]["input_schema"]["required"]), name
    bad = WorldToolsConfig(actor_id="U2", db_path=config.db_path, db_hash="0" * 64)
    try:
        await SlackTools(bad).setup()
        raise AssertionError("a changed world file was served")
    except ValueError:
        pass
    # A tool server exits with the process that started it: killed while the server watches, or gone before it could.
    watcher = "import os, threading, time\n" + inspect.getsource(watch_parent)
    for late, then in ((False, "watch_parent(); print('watching', flush=True)"), (True, "time.sleep(0.5); watch_parent()")):  # fmt: skip
        server_code = f"{watcher}\n{then}\ntime.sleep(60)"
        wait = "" if late else "s.stdout.readline(); "
        parent_code = f"import subprocess, sys, time; s = subprocess.Popen([sys.executable, '-c', {server_code!r}], stdout=subprocess.PIPE, text=True); {wait}print(s.pid, flush=True); time.sleep(0 if {late} else 60)"  # fmt: skip
        parent = subprocess.Popen([sys.executable, "-c", parent_code], stdout=subprocess.PIPE, text=True)
        server = int(parent.stdout.readline())
        parent.kill()
        parent.wait()
        for _ in range(40):
            try:
                os.kill(server, 0)
            except ProcessLookupError:
                break
            time.sleep(0.1)
        else:
            os.kill(server, 9)
            raise AssertionError(f"a tool server outlived the process that started it (gone before: {late})")
    print(
        "PASS world: triggers, rules with rollback, completion, solver copy, visibility, search, paging, the clock, tools as Slack's own MCP serves them, "
        "servers end with their parent"
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
    assert answers("U1", "SELECT text AS answer FROM commitments WHERE status = 'open'") == [], "what was promised is gold's to read"
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
    assert set(SHOWN) == {*SHADOWED, "users"} and all(shown <= {r[1] for r in world.db.execute(f"PRAGMA table_xinfo({t})")} for t, shown in SHOWN.items()), "SHOWN names the workspace's own columns"  # fmt: skip
    assert world.gold("U2", "SELECT emoji AS answer FROM reactions WHERE created_us > 0")["reads"] == ["reactions.created_us", "reactions.emoji"], "a gold query's own columns are read off"  # fmt: skip
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
        ):  # a page of newer top-level messages pushes the evidence to read_channel's second page
            with copy.batch():
                copy.insert("messages", [dict(channel_id="C1", ts_us=at(1, 11, 7) + 1009 * (i + 1), user_id="U1", text=f"note {i}") for i in range(PAGE)])  # fmt: skip
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
    print("PASS tasks: gold as its actor, reads only, readable evidence, time limit, rank, trial")


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
    spec = tomllib.loads((Path(__file__).parent / "tests/fixtures/taxonomy.toml").read_text())
    return Config.model_validate(
        {
            "sector": "software",
            "seed": 3,
            "output": str(root / "run"),
            "personas": {"path": str(root / "p.jsonl"), "typing": str(root / "t.jsonl"), "gaps": str(gaps)},
            "storylines": 2,
            "tasks": {
                "per_100": 1,
                "bands": spec["tasks"]["bands"],
                "max_answer_rows": 2,
                "styles": spec["tasks"]["styles"],
            },  # 4 tasks  # fmt: skip
            "taxonomy": spec["taxonomy"],
        }
    )


def needing(settings, category, level, **needs):
    """The settings with one level of a category needing only these."""
    spec = settings.taxonomy[category]
    tables = [Needs()] * len(spec.levels)
    tables[level - 1] = Needs(**needs)
    return settings.model_copy(update={"taxonomy": settings.taxonomy | {category: spec.model_copy(update={"needs": tables})}})  # fmt: skip


def needless(taxonomy):
    """The taxonomy without its levels' needs, for checks whose tasks are not built to them."""
    return {name: c.model_copy(update={"needs": None}) for name, c in taxonomy.items()}


def ledger_plan(ops, leads, a, b, **changes):
    """Two storylines on an organized world: a rollback decided on day 1 (f1), service back at e2 on day 2 (f2), and
    an audit's dry run and owner at e1 on day 3 (f3, f4)."""
    events = [
        Event(id="e1", storyline="s2", title="release window", day=3, time="10:00"),
        Event(id="e2", storyline="s1", title="service restored", day=2, time="14:00"),
    ]
    facts = [
        PlanFact(id="f1", storyline="s1", subject="Release 4.2", attribute="decision", value="rollback", anchor="rollback", channel_id=ops, author_id=a, day=1, summary="s"),
        PlanFact(id="f2", storyline="s1", subject="Release 4.2", attribute="state", value="service back", channel_id=ops, author_id=b, day=2, after=["f1"], event="e2", kind="happened", summary="s"),
        PlanFact(id="f3", storyline="s2", subject="Audit", attribute="window", value="dry run", anchor="dry run", channel_id=leads, author_id=a, day=3, event="e1", kind="scheduled", summary="s"),
        PlanFact(id="f4", storyline="s2", subject="Audit", attribute="owner", value="Owen", channel_id=ops, author_id=a, day=3, after=["f3"], supersedes="f3", event="e1", kind="scheduled", summary="s"),
    ]  # fmt: skip
    document = dict(storylines=[Storyline(id="s1", summary="the 4.2 release"), Storyline(id="s2", summary="the audit")], events=events, facts=facts)  # fmt: skip
    return Plan(**document | changes)


def check_contracts(root):
    """Documents become rows only through deterministic rules: the cast, the premise, the organization, the tasks
    on a planned ledger (T2-T7), the agenda of everyday conversations; no literal times."""
    root.mkdir(parents=True)
    settings = contracts_settings(root)
    styles = settings.tasks.styles
    slots, cells = (
        quota(settings.taxonomy, styles, 3, 40),
        sum(len(c.levels) for c in settings.taxonomy.values()),
    )
    assert slots == quota(settings.taxonomy, styles, 3, 40) and [s.id for s in slots] != [s.id for s in quota(settings.taxonomy, styles, 4, 40)]  # fmt: skip
    assert (
        len({s.id for s in slots}) == 40 and len({(s.category, s.level) for s in slots[:cells]}) == cells
    ), "every cell once before any twice, each slot its own id"
    assert all(s.concept in settings.taxonomy[s.category].concepts[s.level - 1] and s.style in styles for s in slots), "concepts come from their own level"  # fmt: skip
    repeats = [(slots[i], slots[i + cells]) for i in range(min(cells, 40 - cells))]
    assert all(r.id == f"{s.category}-l{s.level}-2" and (r.concept != s.concept or len(settings.taxonomy[s.category].concepts[s.level - 1]) == 1) for s, r in repeats), "a repeat requires another concept while one is left"  # fmt: skip
    for wrong in ({"concepts": [["x"]]}, {"needs": [{}]}, {"concepts": [["x"], [], ["y"]]}):
        fails(Category.model_validate, settings.taxonomy["semantic"].model_dump() | wrong)
    whole = settings.model_dump(mode="json")
    assert Config.model_validate(whole | {"tasks": whole["tasks"] | {"per_100": 10}}).task_count == 40, (
        "10 tasks per 100 messages, of 400"
    )
    fails(
        Config.model_validate, whole | {"tasks": whole["tasks"] | {"bands": whole["tasks"]["bands"][:2]}}
    )  # 3 levels
    fails(Config.model_validate, whole | {"tasks": whole["tasks"] | {"bands": [[0.5, 0.25]] * len(whole["tasks"]["bands"])}})  # fmt: skip
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
    assert [card["timezone"] for card in cards(cast).values()] == [p.timezone for p in cast], "the judge reads each clock"  # fmt: skip
    assert {habit(f"U{i:03d}") for i in range(40)} == set(HABITS), "people write times in different ways"
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
    with world.trial() as copy:  # the ledger as the author plans it; tasks read its facts, not its messages
        start_clock(copy)
        record_plan(copy, ledger_plan(ops, leads, a, b), settings)

    def slot(category, level, id="t1"):
        return Slot(id=id, category=category, level=level, concept="a concept", style="a style")

    def task(
        id="t1",
        category="search",
        sql="SELECT value AS answer FROM facts WHERE id = 'f1'",
        facts=("f1",),
        **extra,
    ):
        return Task(id=id, category=category, level=1, actor_id=c, question="Which way did the release go?", answer_type="text", gold_sql=sql, facts=list(facts)) .model_copy(update=extra)  # fmt: skip

    def recorded(task_, slot_, settings_=None, setup=lambda copy: None):
        """The tasks once this one is recorded into its slot, on a dropped trial."""
        try:
            with world.trial() as copy:
                setup(copy)
                record_task(copy, task_, settings_ or settings, slot_)
                raise LookupError([r[0] for r in copy.db.execute("SELECT id FROM tasks ORDER BY id")])
        except LookupError as out:
            return out.args[0]

    search, found = task(), slot("search", 1)
    rejected = {
        "T6 its slot's cell": (search, slot("search", 2)),
        "T6 its slot's id": (search, slot("search", 1, id="t2")),
        "T2 sql reads facts": (task(category="lookup"), slot("lookup", 1)),
        "T2 ledger reads no facts": (task(sql="SELECT real_name AS answer FROM users LIMIT 1"), found),
        "T2 hybrid reads no workspace": (task(category="hybrid"), slot("hybrid", 1)),
        "T4 no facts": (task(facts=()), found),
        "T3 refusal with rows": (task(category="robustness", answer_type="refusal", sql="SELECT 1 AS answer", facts=()), slot("robustness", 1)),
        "T3 too many rows": (task(answer_type="set", sql="SELECT value AS answer FROM facts"), found),
        "T3 text answer of 2 rows": (task(sql="SELECT value AS answer FROM facts LIMIT 2"), found),
        "T3 no answer column": (task(sql="SELECT value FROM facts WHERE id = 'f1'"), found),
        "T5 giveaway": (task(question="Was it a rollback for the release?"), found),
        "unreadable fact": (task(sql="SELECT value AS answer FROM facts WHERE id = 'f3'", facts=("f3",)), found),
    }  # fmt: skip
    for name, (task_, slot_) in rejected.items():
        try:
            recorded(task_, slot_)
            raise AssertionError(f"task accepted: {name}")
        except ValueError:
            pass
    directory = task(
        category="lookup", sql=f"SELECT real_name AS answer FROM users WHERE id = '{d}'", facts=()
    )
    assert "t1" in recorded(directory, slot("lookup", 1)), (
        "a task with no facts meets a level that needs none"
    )
    # A gold query asks only about what the tools show: who reacted, never when (S3's reactions "made on Aug 18").
    for unshown in ("SELECT u.real_name AS answer FROM reactions r JOIN users u ON u.id = r.user_id WHERE r.created_us > 0", "SELECT real_name AS answer FROM users WHERE created_us > 0 LIMIT 1", "SELECT name AS answer FROM channels WHERE creator_id IS NOT NULL LIMIT 1"):  # fmt: skip
        refused("which no Slack tool shows", recorded, task(category="lookup", sql=unshown, facts=()), slot("lookup", 1))  # fmt: skip
    nobody = task(category="robustness", answer_type="refusal", sql="SELECT id AS answer FROM users WHERE real_name = 'Nobody Here'", facts=())  # fmt: skip
    assert "t1" in recorded(nobody, slot("robustness", 1)), "nor one whose level's needs are empty"
    with world.trial() as copy:
        record_task(copy, search, settings, found)
    assert json.loads(world.db.execute("SELECT gold_json FROM tasks").fetchone()[0]) == [
        {"answer": "rollback"}
    ]
    assert world.db.execute("SELECT concept FROM tasks").fetchone()[0] == "a concept", (
        "the slot's concept is stored"
    )
    fails(record_task, world, task(id="t2"), settings, slot("search", 1, id="t2"))  # t1 asks it
    refused("its answer differs for them", record_task, world, task(id="t2", actor_id=d), settings, slot("search", 1, id="t2"))  # fmt: skip  # d asks it too, and gets t1's answer
    count = lambda id, actor: Task(id=id, category="lookup", level=1, actor_id=actor, question="How many private channels am I in?", answer_type="number", gold_sql="SELECT COUNT(*) AS answer FROM channels WHERE type = 'private'")  # noqa: E731  # fmt: skip
    assert "t3" in recorded(count("t3", c), slot("lookup", 1, id="t3"), setup=lambda copy: record_task(copy, count("t2", a), settings, slot("lookup", 1, id="t2"))), "a perspective twin: another asker, another answer"  # fmt: skip

    def settled(facts):  # a level-3 semantic task needs its facts first stated in 2 channels
        sql = f"SELECT value AS answer FROM facts WHERE id = '{facts[-1]}'"
        return Task(id="t9", category="semantic", level=3, actor_id=a, question="What did the review settle?", answer_type="text", gold_sql=sql, facts=facts)  # fmt: skip

    # A level's needs, those its facts give: f1, f2 and f4 are first stated in #ops, f3 in #leads; f2 comes after f1,
    # f4 after and over f3; a decoy counts where the task's actor can read it.
    nine = slot("semantic", 3, id="t9")

    def decoy(channel, id="f5", day=3, attribute="owner"):
        return lambda copy: copy.insert("facts", [dict(id=id, storyline="s2", subject="Audit", attribute=attribute, value="Ines", channel_id=channel, author_id=a, day=day, summary="s", is_decoy=1)])  # fmt: skip

    def retracted(channel, by="f4"):  # a decoy that a fact stated in `by`'s channel supersedes
        def setup(copy):
            decoy(channel)(copy)
            copy.insert("fact_relations", [dict(src_fact=by, dst_fact="f5", kind="supersedes")])

        return setup

    def corrected_unseen(
        copy,
    ):  # the decoy's correction, another decoy, is only in #leads, which c cannot read
        decoy(ops)(copy)
        decoy(leads, id="f6")(copy)
        copy.insert("fact_relations", [dict(src_fact="f6", dst_fact="f5", kind="supersedes")])

    for facts, actor, needs, setup, refusal in (
        (["f1", "f4"], a, {"channels": 2}, None, "first stated in at least 2 channels (they are in 1)"),
        (["f3", "f4"], a, {"relations": 3}, None, "at least 3 supersedes or after relations among its facts (it has 2)"),
        (["f2", "f4"], c, {"decoys": 1}, decoy(leads), "at least 1 near-misses"),  # c is not in #leads
        (["f2", "f4"], c, {"decoys": 1}, retracted(ops), "at least 1 near-misses"),  # f4 corrects it in plain sight
        (["f2", "f4"], c, {"decoys": 1}, decoy(ops, attribute="budget"), "at least 1 near-misses"),  # another attribute
        (["f4", "f3"], a, {"decoys": 1}, decoy(ops), "at least 1 near-misses"),  # it answers f3's window, not the owner
    ):  # fmt: skip
        asked = settled(facts).model_copy(update={"actor_id": actor})
        refused(refusal, recorded, asked, nine, needing(settings, "semantic", 3, **needs), setup or (lambda copy: None))  # fmt: skip
    assert "t9" in recorded(settled(["f1", "f3"]), nine, needing(settings, "semantic", 3, channels=2))
    assert "t9" in recorded(settled(["f3", "f4"]), nine, needing(settings, "semantic", 3, relations=2))
    asked = settled(["f2", "f4"]).model_copy(update={"actor_id": c})
    assert "t9" in recorded(asked, nine, needing(settings, "semantic", 3, decoys=1), decoy(ops))
    assert "t9" in recorded(asked, nine, needing(settings, "semantic", 3, decoys=1), corrected_unseen), (
        "a near-miss whose correction its actor cannot see still misleads"
    )
    # A refusal rests on the truth its actor cannot see (f3, in #leads) beside the stale value it can (f7, in #ops).
    stale = lambda copy: (copy.insert("facts", [dict(id="f7", storyline="s2", subject="Audit", attribute="window", value="full test", channel_id=ops, author_id=a, day=3, summary="s")]), copy.insert("fact_relations", [dict(src_fact="f3", dst_fact="f7", kind="supersedes")]))  # noqa: E731  # fmt: skip
    unseen = Task(id="t9", category="robustness", level=4, actor_id=c, question="When is the audit's full test?", answer_type="refusal", gold_sql="SELECT id AS answer FROM users WHERE real_name = 'Nobody Here'", facts=["f3", "f7"])  # fmt: skip
    four = slot("robustness", 4, id="t9")
    assert "t9" in recorded(unseen, four, needing(settings, "robustness", 4, decoys=1), stale), (
        "its near-miss misleads"
    )
    refused("at least 1 near-misses", recorded, unseen.model_copy(update={"facts": ["f3"]}), four, needing(settings, "robustness", 4, decoys=1))  # fmt: skip  # no stale value in sight

    # A status answer gives the latest value its actor can see (f7, in #ops), which a fact out of their sight settles
    # (f3, in #leads): the useful answer gives it as not settled, where a refusal said nothing.
    status = Task(id="t9", category="robustness", level=3, actor_id=c, question="What is the audit's window?", answer_type="status", gold_sql="SELECT value AS answer FROM facts WHERE id = 'f7'", facts=["f3", "f7"])  # fmt: skip
    three = slot("robustness", 3, id="t9")
    assert "t9" in recorded(status, three, settings, stale)
    refused("out of its actor's sight, that supersedes the value it gives", recorded, status.model_copy(update={"facts": ["f7"]}), three, settings, stale)  # fmt: skip  # nothing settles it
    refused("which supersede the value it gives", recorded, status.model_copy(update={"actor_id": a}), three, settings, stale)  # fmt: skip  # a reads #leads: f3 is the answer
    refused("gold gives the value of one of its facts its actor can read", recorded, status.model_copy(update={"gold_sql": "SELECT value AS answer FROM facts WHERE id = 'f3'"}), three, settings, stale)  # fmt: skip
    refused("gold query reads the facts it answers from", recorded, status.model_copy(update={"gold_sql": "SELECT real_name AS answer FROM users LIMIT 1"}), three, settings, stale)  # fmt: skip

    # The wrong answers a task's grade must reject: the near-miss its actor can read, and a hedge with it.
    try:
        with world.trial() as copy:
            decoy(ops)(copy)
            copy.insert("facts", [dict(id="f6", storyline="s2", subject="Audit", attribute="owner", value="Mia", channel_id=leads, author_id=a, day=3, summary="s", is_decoy=1)])  # fmt: skip  # out of c's sight
            record_task(copy, asked, needing(settings, "semantic", 3, decoys=1), nine)
            raise LookupError(failure_cases(copy, "t9", [{"answer": "Owen"}]))
    except LookupError as out:
        gold, cases = out.args[0]
    assert gold == "Owen." and ("the near-miss 'Ines'", "Ines.") in cases and ("a hedge", "Either Owen or Ines; I can't tell which.") in cases, cases  # fmt: skip
    assert not [c for c in cases if "Mia" in c[1]], "a value its actor cannot see is no near-miss"

    # Lazy solvers over a chain: "full test" (f7, #ops) is settled as "dry run" (f3, #leads), then a decoy repeats
    # "rehearsal" in #ops. Taking the first or the latest value is wrong; the truth sits in a private channel, later.
    def chain(copy):
        stale(copy)
        copy.insert("facts", [dict(id="f8", storyline="s2", subject="Audit", attribute="window", value="rehearsal", channel_id=ops, author_id=b, day=3, summary="s", is_decoy=1)])  # fmt: skip
        (day3,) = copy.db.execute("SELECT start_us FROM calendar WHERE day = 3").fetchone()
        copy.db.execute("UPDATE world_meta SET value = ? WHERE key = 'now_us'", (str(day3 + 14_400_000_000),))
        said = copy.insert("messages", [dict(channel_id=ch, ts_us=day3 + h * 3_600_000_000, user_id=u, text=t) for h, ch, u, t in ((1, ops, a, "full test it is"), (2, leads, a, "dry run it is"), (3, ops, b, "rehearsal it is"))])  # fmt: skip
        copy.insert("evidence", [dict(fact_id=f, message_id=m, role="anchor") for f, m in zip(("f7", "f3", "f8"), said)])  # fmt: skip

    window = Task(id="t9", category="semantic", level=3, actor_id=a, question="What is the audit's window?", answer_type="text", gold_sql="SELECT value AS answer FROM facts WHERE id = 'f3'", facts=["f3", "f7"])  # fmt: skip
    try:
        with world.trial() as copy:
            chain(copy)
            record_task(copy, window, needing(settings, "semantic", 3), nine)
            count = Task(id="t8", category="lookup", level=1, actor_id=a, question="How many messages are there?", answer_type="number", gold_sql="SELECT COUNT(*) AS answer FROM messages")  # fmt: skip
            counted = record_task(copy, count, settings, slot("lookup", 1, id="t8"))["gold"]
            who = window.model_copy(update={"id": "t7", "question": "Who settled the audit's window?", "gold_sql": "SELECT u.real_name AS answer FROM facts f JOIN users u ON u.id = f.author_id WHERE f.id = 'f3'"})  # fmt: skip
            settler = record_task(copy, who, needing(settings, "semantic", 3), slot("semantic", 3, id="t7"))[
                "gold"
            ]
            raise LookupError((panel(copy, "t9", [{"answer": "dry run"}]), features(copy, "t9", [{"answer": "dry run"}]), features(copy, "t8", counted), features(copy, "t7", settler)))  # fmt: skip
    except LookupError as out:
        lazy, shape, counting, settled_by = out.args[0]
    assert settled_by["last_link"] == "private·later", "a person's answer is placed by the fact it is about"
    assert {k: lazy["guesses"][k] for k in ("first", "latest", "top")} == {"first": "full test", "latest": "rehearsal", "top": "rehearsal"} and {"first", "latest", "top"} <= set(lazy["traps"]) and not lazy["shortcut"], lazy  # fmt: skip
    assert (shape["changes"], shape["last_link"], shape["perspective"]) == (3, "private·later", False), shape
    assert counting["perspective"] and counting["derived"], (
        "a count differs by who asks, and no message states it"
    )

    def stated_unseen(copy):  # the truth, f3, is stated in #leads on day 3, where c cannot read it
        stale(copy)
        (day3,) = copy.db.execute("SELECT start_us FROM calendar WHERE day = 3").fetchone()
        copy.db.execute("UPDATE world_meta SET value = ? WHERE key = 'now_us'", (str(day3 + 7_200_000_000),))
        (truth,) = copy.insert("messages", [dict(channel_id=leads, ts_us=day3 + 3_600_000_000, user_id=a, text="dry run it is")])  # fmt: skip
        copy.insert("evidence", [dict(fact_id="f3", message_id=truth, role="anchor", anchor_token="dry run")])

    assert "t9" in recorded(status, three, settings, stated_unseen), (
        "a status rests on its truth stated out of sight"
    )
    try:
        with world.trial() as copy:
            stated_unseen(copy)
            raise LookupError(record_task(copy, unseen, needing(settings, "robustness", 4, decoys=1), four))
    except LookupError as out:
        measured = out.args[0]
    assert (measured["unseen"], measured["evidence_pages"], measured["decoys"]) == (1, None, 1), (
        "evidence its actor cannot read is counted, and measured from no view"
    )
    hybrid = dict(category="hybrid", level=1, actor_id=c, answer_type="number", gold_sql="SELECT COUNT(*) AS answer FROM messages m, facts f WHERE f.id = 'f1'")  # fmt: skip
    unread = Task(
        id="h1", question="How many messages surround the decision?", facts=["f3"], **hybrid
    )  # in leads
    fails(record_task, world, unread, settings, slot("hybrid", 1, id="h1"))  # its actor cannot read its fact
    read = Task(id="h1", question="How many messages surround the rollback?", facts=["f1"], **hybrid)
    assert "h1" in recorded(read, slot("hybrid", 1, id="h1"))
    small = settings.activity.model_copy(update={"conversation_lines": 4})
    plan = background_plan(world, org(), small, 7, 2, 12)  # a day's agenda: 12 messages of everyday talk
    members = {(r[0], r[1]) for r in world.db.execute("SELECT channel_id, user_id FROM members")}
    assert plan == background_plan(world, org(), small, 7, 2, 12) and sum(s.length for s in plan) in (11, 12)
    drawn = lambda scenes: [(x.channel_id, x.part, x.situation, x.length) for x in scenes]  # noqa: E731
    assert drawn(plan) != drawn(background_plan(world, org(), small, 7, 3, 12)), "each day draws its own"
    assert all(s.situation in {r.kind for r in routines} and s.day == 2 for s in plan)
    assert all((s.channel_id, p) in members for s in plan for p in s.participants)
    assert background_plan(world, org(), small, 7, 2, 0) == []
    direct = background_plan(world, org(), small.model_copy(update={"dm_share": 1.0}), 7, 2, 12)
    kinds = {r[0]: r[1] for r in world.db.execute("SELECT id, type FROM channels")}
    assert {kinds[s.channel_id] for s in direct} == {"im"}, "dm_share places conversations in DMs"
    skewed = background_plan(world, org(), small.model_copy(update={"dm_share": 0.0, "channel_skew": 4.0}), 7, 2, 60)  # fmt: skip
    places = Counter(kinds[s.channel_id] for s in skewed)
    assert places["public"] > 3 * places["private"], "busier channels draw more conversations"
    assert TIME_LITERAL.findall("at 3pm, 15:00, 2026-06-15 or Jun 15") == [
        "3pm",
        "15:00",
        "2026-06-15",
        "Jun 15",
    ]
    assert TIME_LITERAL.findall("release 8.14 on v2.10, maybe 2 people") == []
    assert IDENTIFIER.findall("PWSQL-03 to CU16 under CHG-2291, v2; release 4.2 at 10:00") == ["PWSQL-03", "CU16", "CHG-2291", "v2"]  # fmt: skip
    print(
        "PASS contracts: cast, premise, organization, task gold (T2-T7) on a planned ledger, agenda, no literal times"
    )


def organized(root, settings, apps=(), groups=()):
    """An organized world on Chicago time: people a, b, c (and d, in New York), #ops (public: everyone), #leads
    (private: a, b), DMs a-c and code's own, a 10-day calendar from a Monday; `apps`, its integrations."""
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
        integrations=list(apps),
        usergroups=list(groups),
    )
    premise = Premise(
        company="Lattice Creek", niche="n", region="r", size="s", culture="c", cast="x", staffing={}
    )
    root.mkdir(parents=True, exist_ok=True)
    world = World.create(root / "world.sqlite")
    with world.trial() as copy:
        organize(copy, org, cast, premise, settings)
    return world, cast, org, (a, b, c, d)


def check_apps(root):
    """Apps post by code (2026-10-14): the organization lists them, each a bot that is a member of its channel; an
    event one reports is posted at its moment with a title that states no fact; routine posts come on workdays, from
    the queue, in time order with everyone's messages; no quota, share, scene or task counts a bot."""
    root.mkdir(parents=True)
    base = contracts_settings(root)
    settings = base.model_copy(
        update={
            "activity": base.activity.model_copy(update={"messages": 12, "bot_posts": 3}),
            "author": base.author.model_copy(update={"tolerance": 1.0, "share_tolerance": 1.0, "mention_tolerance": 1.0}),  # fmt: skip
            "taxonomy": needless(base.taxonomy),
        }
    )
    deploys = Integration(name="Deploy Bot", kind="deploys", channel="ops", subjects=["web", "api", "worker"])
    builds = Integration(name="CI", kind="ci", channel="ops", subjects=["main", "release", "hotfix"])
    for name, apps, why in (
        ("one", [deploys], "2 to 5 apps"),
        ("lost", [deploys, builds.model_copy(update={"channel": "nowhere"})], "one of the organization's channels"),
        ("timed", [deploys, builds.model_copy(update={"subjects": ["main", "the 5pm build", "x"]})], "names no time"),
    ):  # fmt: skip
        refused(why, organized, root / name, settings, apps)
    world, cast, org, (a, b, c, d) = organized(root / "world", settings, [deploys, builds])
    ops, leads, bot = channel_id("public", "ops", []), channel_id("private", "leads", []), bot_id("Deploy Bot")
    assert world.db.execute("SELECT u.is_bot, m.channel_id FROM users u JOIN members m ON m.user_id = u.id WHERE u.id = ?", (bot,)).fetchone()[:] == (1, ops)  # fmt: skip
    with world.trial() as copy:
        start_clock(copy)
    released = Event(id="e3", storyline="s1", title="production release", day=2, time="16:00", post_by="Deploy Bot")
    plan = ledger_plan(ops, leads, a, b)

    def recorded(document):
        with world.trial() as copy:
            record_plan(copy, document, settings)

    refused("post_by names an app of the workspace", recorded, plan.model_copy(update={"events": [*plan.events, released.model_copy(update={"post_by": "Nobody"})]}))  # fmt: skip
    refused("states no fact or time: ['rollback']", recorded, plan.model_copy(update={"events": [*plan.events, released.model_copy(update={"title": "rollback shipped"})]}))  # fmt: skip
    recorded(plan.model_copy(update={"events": [*plan.events, released]}))
    assert world.db.execute("SELECT COUNT(*) FROM queue WHERE message_id IS NULL").fetchone()[0] > 1, "posts wait for their moment"  # fmt: skip
    assert not world.db.execute("SELECT 1 FROM messages").fetchone(), "nothing is written before the present reaches it"  # fmt: skip
    rng, gaps = random.Random(5), Gaps(settings.personas.gaps)
    lines = {1: [(ops, a, "we roll back, rollback decided", ["f1"])], 2: [(ops, b, "service back at {at:e2}", ["f2"])], 3: [(leads, a, "the dry run is set", ["f3"]), (ops, a, "Owen owns it from {at:e1}", ["f4"])]}  # fmt: skip
    while today(world) is not None and today(world) <= 7:
        day = today(world)
        if day in lines and part_of(world, present(world)) == ("evening" if day == 2 else "early"):
            for channel, author, text, conveys in lines.pop(day):
                post(world, Conversation(channel_id=channel, about="work", lines=[PostLine(author_id=author, text=text, conveys=conveys)]), settings, rng, gaps)  # fmt: skip
        advance(world, settings, onward(world))
    assert world.violations() == [], "code's posts went in in time order"
    assert not world.db.execute("SELECT 1 FROM queue WHERE message_id IS NULL AND due_us < ?", (present(world),)).fetchone(), "the present wrote what came due"  # fmt: skip
    apps_ = [r[0] for r in world.db.execute("SELECT user_id FROM integrations")]
    marks = ", ".join("?" * len(apps_))
    posts = world.db.execute(f"SELECT m.ts_us, m.text, m.subtype, c.date FROM messages m JOIN calendar c ON m.ts_us >= c.start_us AND m.ts_us < c.end_us WHERE m.user_id IN ({marks})", apps_).fetchall()  # fmt: skip
    e3 = world.db.execute("SELECT moment_us FROM events WHERE id = 'e3'").fetchone()[0]
    assert [(p[0], p[1]) for p in posts if "production release" in p[1]] == [(e3, "✅ production release: deployed to production")], "an event's post, at its moment"  # fmt: skip
    assert posts and {p[2] for p in posts} == {"bot_message"} and not {date.fromisoformat(p[3]).weekday() for p in posts} & {5, 6}, "weekends are quiet"  # fmt: skip
    people = world.db.execute("SELECT COUNT(*) FROM messages WHERE user_id NOT IN (" + marks + ")", apps_).fetchone()[0]  # fmt: skip
    assert people == sum(posted(world, *bounds(world, d)) for d in range(1, 8)) == sum(tally(world, d)["messages"] for d in range(1, 8)), "no quota counts a bot"  # fmt: skip
    drawn = background_plan(world, org, settings.activity, 3, 2, 40)
    assert not {p for s in drawn for p in s.participants} & set(apps_), "no scene draws a bot"
    fails(record_task, world, Task(id="t9", category="lookup", level=1, actor_id=bot, question="q?", answer_type="number", gold_sql="SELECT COUNT(*) AS answer FROM messages"), settings, Slot(id="t9", category="lookup", level=1, concept="c", style="s"))  # fmt: skip
    reader = World(world.path, actor=c)
    assert not surface.read(reader, "search_public_and_private", {"keywords": ["production"]})["items"] and surface.read(reader, "search_public_and_private", {"keywords": ["production"], "include_bots": True})["items"], "a search leaves apps out unless asked"  # fmt: skip
    assert "✅ production release: deployed to production" in surface.render(reader, "read_channel", surface.read(reader, "read_channel", {"channel_id": ops}))["messages"]  # fmt: skip
    reader.close()
    # A world whose forge writes its tasks: no slots, and its last day is followed by its final review.
    forged_later = settings.model_copy(update={"tasks": settings.tasks.model_copy(update={"in_world": False})})
    assert (forged_later.task_count, settings.task_count > 0) == (0, True)
    assert GenerationEnv(forged_later, TestStore(root / "later", {"seed": 1})).after_days == "final" and GenerationEnv(settings, TestStore(root / "own", {"seed": 1})).after_days == "tasks"  # fmt: skip
    print("PASS apps: listed, members, an event's post at its moment, routine ones on workdays, in time order, counted by no one")


def check_churn(root):
    """People come and go (2026-10-14): a plan's moves join and leave channels (Slack's posts of them in time order),
    take someone out of the company, archive a channel; recorded moves stay; no one posts where they are not, or in an
    archived channel; a fact's author is in its channel all its day; a status names no time."""
    root.mkdir(parents=True)
    base = contracts_settings(root)
    settings = base.model_copy(
        update={
            "activity": base.activity.model_copy(update={"messages": 12}),
            "author": base.author.model_copy(update={"tolerance": 1.0, "share_tolerance": 1.0, "mention_tolerance": 1.0}),  # fmt: skip
            "taxonomy": needless(base.taxonomy),
        }
    )
    world, cast, org, (a, b, c, d) = organized(root / "world", settings)
    ops, leads = channel_id("public", "ops", []), channel_id("private", "leads", [])
    with world.trial() as copy:
        start_clock(copy)
    plan = ledger_plan(ops, leads, a, b)
    moves = [Move(kind="join", day=2, user_id=c, channel_id=leads), Move(kind="leave", day=4, user_id=b, channel_id=leads), Move(kind="depart", day=4, user_id=d), Move(kind="archive", day=5, channel_id=leads)]  # fmt: skip

    def recorded(document):
        with world.trial() as copy:
            record_plan(copy, document, settings)

    for bad, why in (
        ([Move(kind="join", day=2, user_id=a, channel_id=leads)], "is a member of"),
        ([Move(kind="leave", day=2, user_id=c, channel_id=leads)], "is no member of"),
        ([Move(kind="join", day=2, user_id=c, channel_id="D0000000000")], "a named channel"),
        ([Move(kind="archive", day=3, channel_id=leads)], "is archived by day 3"),  # f3 is stated in #leads on day 3
    ):
        refused(why, recorded, plan.model_copy(update={"moves": bad}))
    refused("a member of", recorded, plan.model_copy(update={"moves": moves[:1], "facts": [*plan.facts[:2], plan.facts[2].model_copy(update={"author_id": c, "day": 2})]}))  # fmt: skip  # c joins #leads on day 2's morning: not all that day
    fails(Move, kind="depart", day=2, user_id=c, channel_id=leads)
    recorded(plan.model_copy(update={"moves": moves}))
    refused("recorded moves stay", recorded, plan.model_copy(update={"moves": moves[1:]}))
    rng, gaps = random.Random(5), Gaps(settings.personas.gaps)
    lines = {1: [(ops, a, "we roll back, rollback decided", ["f1"])], 2: [(ops, b, "service back at {at:e2}", ["f2"])], 3: [(leads, a, "the dry run is set", ["f3"]), (ops, a, "Owen owns it from {at:e1}", ["f4"])]}  # fmt: skip
    while today(world) is not None and today(world) <= 6:
        day = today(world)
        if day in lines and part_of(world, present(world)) == ("evening" if day == 2 else "early"):
            for channel, author, text, conveys in lines.pop(day):
                post(world, Conversation(channel_id=channel, about="work", lines=[PostLine(author_id=author, text=text, conveys=conveys)]), settings, rng, gaps)  # fmt: skip
        if day == 3 and part_of(world, present(world)) == "afternoon":  # c, in #leads since day 2, says hello
            post(world, Conversation(channel_id=leads, about="hi", lines=[PostLine(author_id=c, text="glad to be here")]), settings, rng, gaps)  # fmt: skip
        if day == 5 and part_of(world, present(world)) == "evening":
            refused("archived today", post, world, Conversation(channel_id=leads, about="late", lines=[PostLine(author_id=a, text="anyone?")]), settings, rng, gaps)  # fmt: skip
        advance(world, settings, onward(world))
    assert world.violations() == []
    said = {r["subtype"]: (r["user_id"], r["ts_us"]) for r in world.db.execute("SELECT * FROM messages WHERE subtype IS NOT NULL")}  # fmt: skip
    assert said == {"channel_join": (c, move_moment(world, moves[0])), "channel_leave": (b, move_moment(world, moves[1]) - 1)}, said  # fmt: skip
    assert world.db.execute("SELECT is_deleted, deactivated_us IS NOT NULL FROM users WHERE id = ?", (d,)).fetchone()[:] == (1, 1)  # fmt: skip
    assert not world.db.execute("SELECT 1 FROM members WHERE user_id = ? AND left_us IS NULL", (d,)).fetchone(), "out of every conversation"  # fmt: skip
    assert world.db.execute("SELECT is_archived FROM channels WHERE id = ?", (leads,)).fetchone()[0] == 1
    scenes = background_plan(world, org, settings.activity, 3, 6, 40)
    assert scenes and not {s.channel_id for s in scenes} & {leads} and not {p for s in scenes for p in s.participants} & {d}, "no scene where it cannot be"  # fmt: skip
    person = org.people[0].model_copy(update={"status_emoji": ":palm_tree:", "status_text": "out until 5pm"})
    refused("a status names no time", lambda: organize(World.create(root / "status.sqlite"), org.model_copy(update={"people": [person, *org.people[1:]]}), cast, Premise(company="Lattice Creek", niche="n", region="r", size="s", culture="c", cast="x", staffing={}), settings))  # fmt: skip
    reader = World(world.path, actor=a)
    history = surface.render(reader, "read_channel", surface.read(reader, "read_channel", {"channel_id": leads}))["messages"]
    assert f"<@{c}|{world.db.execute('SELECT display_name FROM users WHERE id = ?', (c,)).fetchone()[0]}> has joined the channel" in history, history  # fmt: skip
    reader.close()
    print("PASS churn: joins and leaves posted in time order, departures, archives; recorded moves stay; no one posts where they cannot")


def check_links(root):
    """Links, groups and edits are code's (2026-10-14): an artifact of the plan has its URL by code and is linked as
    {link:id} once it exists, a ticket app announces a ticket, a group is mentioned as {group:handle}; a raw URL is
    refused; a person edits their own message of today, and Slack marks it."""
    root.mkdir(parents=True)
    base = contracts_settings(root)
    settings = base.model_copy(
        update={
            "activity": base.activity.model_copy(update={"messages": 12}),
            "author": base.author.model_copy(update={"tolerance": 1.0, "share_tolerance": 1.0, "mention_tolerance": 1.0}),  # fmt: skip
            "taxonomy": needless(base.taxonomy),
        }
    )
    tickets = Integration(name="Tracker", kind="tickets", channel="ops", subjects=["login bug", "slow page", "export"], prefix="TKT")  # fmt: skip
    cast_ids = [user_id(p.uuid) for p in [seed_person(i, "America/Chicago") for i in range(2)]]
    world, cast, org, (a, b, c, d) = organized(root / "world", settings, [tickets], [UserGroup(handle="oncall", name="On call", members=cast_ids)])  # fmt: skip
    ops, leads = channel_id("public", "ops", []), channel_id("private", "leads", [])
    with world.trial() as copy:
        start_clock(copy)
    plan = ledger_plan(ops, leads, a, b)
    runbook = Artifact(id="TKT-1", kind="ticket", title="audit runbook", owner_id=b, day=2)

    def recorded(document):
        with world.trial() as copy:
            record_plan(copy, document, settings)

    refused("states no fact or time: ['rollback']", recorded, plan.model_copy(update={"artifacts": [runbook.model_copy(update={"title": "rollback steps"})]}))  # fmt: skip
    refused("a person of the company", recorded, plan.model_copy(update={"artifacts": [runbook.model_copy(update={"owner_id": bot_id("Tracker")})]}))  # fmt: skip
    recorded(plan.model_copy(update={"artifacts": [runbook]}))
    refused("recorded artifacts stay", recorded, plan)
    url = world.db.execute("SELECT url FROM artifacts WHERE id = 'TKT-1'").fetchone()[0]
    assert url == "https://tickets.lattice-creek.example/browse/TKT-1".replace("lattice-creek", "latticecreek"), url
    rng, gaps = random.Random(5), Gaps(settings.personas.gaps)

    def talk(text, author=a, channel=ops, conveys=()):
        return post(world, Conversation(channel_id=channel, about="work", lines=[PostLine(author_id=author, text=text, conveys=list(conveys))]), settings, rng, gaps)  # fmt: skip

    refused("does not exist yet", talk, "see {link:TKT-1}")
    refused("names an artifact of the plan", talk, "see {link:TKT-9}")
    refused("code writes its URL", talk, "see https://tickets.latticecreek.example/browse/TKT-1")
    refused("names a usergroup's handle", talk, "{group:nobody} please look")
    first = talk("{group:oncall} heads up, rollback decided", conveys=["f1"])["messages"][0]
    assert world.db.execute("SELECT text FROM messages WHERE id = ?", (first["id"],)).fetchone()[0].startswith("<!subteam^S"), "a group as Slack writes it"  # fmt: skip
    assert not world.db.execute("SELECT 1 FROM message_mentions WHERE message_id = ?", (first["id"],)).fetchone(), "a group is no person's mention"  # fmt: skip
    edited = edit(world, first["id"], "{group:oncall} heads up: rollback decided")
    assert world.db.execute("SELECT edited_us FROM messages WHERE id = ?", (first["id"],)).fetchone()[0] == present(world) and "rollback decided" in edited["text"]  # fmt: skip
    while today(world) == 1:
        advance(world, settings, onward(world))
    refused("posted earlier today", edit, world, first["id"], "rollback decided, sorry")
    while part_of(world, present(world)) != "afternoon":
        advance(world, settings, onward(world))
    linked_line = talk("runbook is {link:TKT-1}", author=b)["messages"][0]
    assert f"<{url}|TKT-1>" in world.db.execute("SELECT text FROM messages WHERE id = ?", (linked_line["id"],)).fetchone()[0]  # fmt: skip
    announced = world.db.execute("SELECT m.text, m.user_id FROM messages m WHERE m.subtype = 'bot_message' AND m.text LIKE ?", (f"<{url}|TKT-1> created%",)).fetchall()  # fmt: skip
    assert [tuple(r) for r in announced] == [(f"<{url}|TKT-1> created: audit runbook · assigned to <@{b}>", bot_id("Tracker"))], announced  # fmt: skip
    reader = World(world.path, actor=c)
    assert "rollback decided (edited)" in surface.render(reader, "read_channel", surface.read(reader, "read_channel", {"channel_id": ops}))["messages"], "an edit is marked"  # fmt: skip
    reader.close()
    print("PASS links: artifacts by code, linked once they exist, a ticket announced; groups as Slack writes them; edits of today, marked")


def refused(expected, function, *args, **kwargs):
    """The call is refused, for the reason given."""
    try:
        function(*args, **kwargs)
    except ValueError as error:
        assert expected in str(error), (expected, str(error))
        return
    raise AssertionError(f"not refused: {expected}")


def onward(world) -> str:
    """Where advance moves next: the next part of today, or from the night, tomorrow."""
    part, order = part_of(world, present(world)), list(PARTS)
    return "tomorrow" if part == "night" else order[order.index(part) + 1]


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
            "taxonomy": needless(base.taxonomy),
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
    sampler = random.Random(1)
    assert all(gaps.sample(sampler, "") < 3_600_000_000 <= gaps.sample(sampler, "hours") for _ in range(200))

    def plan(**changes):
        return ledger_plan(ops, leads, a, b, **changes)

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
        ("an anchor is words of its value", plan(facts=[facts[0].model_copy(update={"anchor": "Release 4.2"}), *facts[1:]])),
        ("f3 and f4 have one anchor, 'dry run'", plan(facts=[*facts[:3], facts[3].model_copy(update={"value": "dry run", "anchor": "dry run"})])),
        ("contains f3's, 'dry run'", plan(facts=[*facts[:3], facts[3].model_copy(update={"value": "the dry run owner", "anchor": "dry run owner", "after": [], "supersedes": None})])),
        ("f1 comes after itself, through f1 > f2 > f1", plan(facts=[facts[0].model_copy(update={"after": ["f2"]}), *facts[1:]])),
    ):  # fmt: skip
        refused(expected, recorded, bad)
    try:  # an anchor may contain another's when its fact comes after that one
        with world.trial() as copy:
            record_plan(copy, plan(facts=[*facts[:3], facts[3].model_copy(update={"value": "the dry run owner", "anchor": "dry run owner"})]), settings)  # fmt: skip
            raise LookupError("drop the trial")
    except LookupError:
        pass
    try:  # a Saturday with a planned event weighs as a workday; the first plan fixes those days, the total stays
        with world.trial() as copy:
            window_event = Event(id="e3", storyline="s2", title="maintenance window", day=6, time="22:00")
            record_plan(copy, plan(events=[*events, window_event]), settings)
            busy = quotas(copy, settings)
            assert busy[6]["messages"] >= min(busy[d]["messages"] for d in range(1, 6)) > shares[6]["messages"] == busy[7]["messages"], busy  # fmt: skip
            assert sum(q["messages"] for q in busy.values()) == 12
            cutover = Event(id="e4", storyline="s1", title="cutover", day=7, time="09:00")
            record_plan(copy, plan(events=[*events, window_event, cutover]), settings)
            assert quotas(copy, settings) == busy, "a later plan's event days leave the quotas as they were"
            raise LookupError("drop the trial")
    except LookupError:
        pass
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
    # Facts run through the calendar: days 8-10 hold a quarter of the messages, so of 8 facts they first state 1.
    extra = [PlanFact(id=f"f{k}", storyline="s1", subject=f"Topic {k}", attribute="state", value=f"state {k}", channel_id=ops, author_id=a, day=2, summary="s") for k in range(5, 9)]  # fmt: skip
    refused("days 8-10 first state 0 of the plan's 8 facts; they hold 25% of the world's messages, so they first state at least 1", recorded, plan(facts=[*facts, *extra]))  # fmt: skip
    try:
        with world.trial() as copy:
            record_plan(
                copy, plan(facts=[*facts, *extra[:3], extra[3].model_copy(update={"day": 9})]), settings
            )
            raise LookupError("drop the trial")
    except LookupError:
        pass
    late = [f.model_copy(update={"day": 9}) for f in [*facts, *extra]]
    assert front_loaded(world, settings, late, 9) is None, "a third that is over is not held to its share"
    # The board: the planned facts of each slot that rests on them (ledger and hybrid, and those whose level needs
    # something of their facts), as many as its level needs; checked on every plan.
    board_slots = [Slot(id=i, category=c, level=n, concept="a concept", style="a style") for i, c, n in (("t1", "semantic", 3), ("t3", "search", 1), ("t2", "lookup", 1))]  # fmt: skip
    spread = needing(settings, "semantic", 3, channels=2)
    entries = lambda *pairs: [BoardEntry(slot=slot, facts=ids) for slot, ids in pairs]  # noqa: E731

    def boarded(document):
        with world.trial() as copy:
            record_plan(copy, document, spread, board_slots)

    for expected, board in (
        ("each slot that rests on planned facts will rest on; ['t1', 't3'] have none", []),
        ("the slots of now.md that rest on planned facts; ['t2'] do not", entries(("t1", ["f3", "f4"]), ("t3", ["f4"]), ("t2", ["f1"]))),
        ("the board names facts of the plan; ['f9'] are not", entries(("t1", ["f3", "f9"]), ("t3", ["f4"]))),
        ("the board's t1 (level-3 semantic) needs its facts first stated in at least 2 channels (they are in 1)", entries(("t1", ["f1", "f4"]), ("t3", ["f4"]))),
    ):  # fmt: skip
        refused(expected, boarded, plan(board=board))
    # A robustness level-4 slot rests on planned facts too: a near-miss its asker will see, the truth out of sight.
    robust = [
        *board_slots,
        Slot(id="t4", category="robustness", level=4, concept="a concept", style="a style"),
    ]
    near = PlanFact(id="f5", storyline="s2", subject="Audit", attribute="owner", value="Ines", channel_id=ops, author_id=a, day=3, decoy=True, summary="s")  # fmt: skip

    def boarded_robust(document):
        with world.trial() as copy:
            record_plan(copy, document, needing(spread, "robustness", 4, decoys=1), robust)

    usual = entries(("t1", ["f3", "f4"]), ("t3", ["f4"]))
    refused("['t4'] have none", boarded_robust, plan(board=usual))
    refused("the board's t4 (level-4 robustness) needs at least 1 near-misses", boarded_robust, plan(board=[*usual, *entries(("t4", ["f4"]))]))  # fmt: skip
    boarded_robust(plan(facts=[*facts, near], board=[*usual, *entries(("t4", ["f4", "f5"]))]))
    try:  # facts may be shared; a replan's board replaces the last; a fact the board rests on stays planned
        with world.trial() as copy:
            record_plan(copy, plan(board=entries(("t1", ["f3", "f4"]), ("t3", ["f4"]))), spread, board_slots)
            record_plan(copy, plan(board=entries(("t1", ["f1", "f3"]), ("t3", ["f1"]))), spread, board_slots)
            assert [tuple(r) for r in copy.db.execute("SELECT slot, fact_id FROM board ORDER BY 1, 2")] == [("t1", "f1"), ("t1", "f3"), ("t3", "f1")]  # fmt: skip
            dropped = plan(facts=facts[:3], board=entries(("t1", ["f3", "f4"]), ("t3", ["f1"])))
            refused("the board names facts of the plan; ['f4'] are not", record_plan, copy, dropped, spread, board_slots)  # fmt: skip
            raise LookupError("drop the trial")
    except LookupError:
        pass
    rng = random.Random(5)

    slots = [Slot(id="t1", category="semantic", level=3, concept="a concept", style="a style"), Slot(id="t2", category="lookup", level=1, concept="a concept", style="a style")]  # fmt: skip
    owner = Task(id="t1", category="semantic", level=3, actor_id=a, question="Who owns the audit after the dry run?", answer_type="text", gold_sql="SELECT value AS answer FROM facts WHERE id = 'f4'", facts=["f3", "f4"])  # fmt: skip

    def tasked(task, cfg=None, setup=lambda copy: None):
        with world.trial() as copy:
            setup(copy)
            return add_task(copy, task, cfg or settings, slots)

    refused("once the calendar is closed", tasked, owner)

    def posted_(conversation):
        with world.trial() as copy:
            return post(copy, conversation, settings, rng, gaps)

    def talk(*lines, channel=ops, **extra):
        return Conversation(
            channel_id=channel, about="work", lines=[PostLine(**line) for line in lines], **extra
        )

    decided = talk(
        dict(author_id=a, text=f"we go with the rollback, <@{c}> fyi", conveys=["f1"]),
        dict(author_id=b, text="ok, I'll post the notes", reply_to=0, commits=[Commit(id="c1", text="post the notes", due_day=2)], reactions=[Reaction(user_id=a, emoji="eyes")]),
        dict(author_id=b, text="rollback noted", reply_to=0, conveys=["f1"]),
    )  # fmt: skip
    for expected, bad in (
        ("write times and dates only", talk(dict(author_id=a, text="rollback at 3pm", conveys=["f1"]))),
        ("reply_to names an earlier line", talk(dict(author_id=a, text="ok", reply_to=0))),
        ("conveys lists only facts", talk(dict(author_id=a, text="rollback is decided", conveys=["f1", "f9"]))),
        ("{at:...} names a moment", talk(dict(author_id=a, text="see you at {at:lunch}"))),
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
    assert out["conversation"] == "d01-001" and len(out["messages"]) == 3
    roles = [
        r[0] for r in world.db.execute("SELECT role FROM evidence WHERE fact_id = 'f1' ORDER BY message_id")
    ]
    assert roles == ["anchor", "supporting"], (
        "the planned author states it, a second speaker supports it",
        roles,
    )
    assert world.db.execute("SELECT user_id FROM message_mentions").fetchone()[0] == c
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

    assert advanced(to=onward(world))["closed_day"] is None and part_of(world, present(world)) != "early"
    refused("a later part of today", advanced, to="early")
    advanced(to="night")
    held = present(world)
    assert advanced(to="night")["closed_day"] is None and present(world) == held, (
        "the present's own part: no move"
    )
    strict = settings.model_copy(update={"author": settings.author.model_copy(update={"tolerance": 0.0})})
    assert shares[1]["messages"] == 2 and any("messages today" in e for e in close_day(world, strict, 1)), (
        "4 posted"
    )
    assert advanced(to="tomorrow")["closed_day"] == 1 and today(world) == 2
    refused("the day closes from the night", advanced, to="tomorrow")  # a repeat never closes a second day
    advanced(to="morning")
    refused("advance to night first", advanced, to="tomorrow")
    # Shares are checked per day, and a day stays closable: with a day of about 15 messages, f2 still owed.
    assert band(0.3, 0.1, 2) == (0, 1) and band(0.3, 0.1, 40) == (8, 16) and band(0.25, 0.1, 10) == (1, 4)
    big = settings.model_copy(update={"activity": settings.activity.model_copy(update={"messages": 120}), "author": settings.author.model_copy(update={"tolerance": 0.3, "share_tolerance": 0.1})})  # fmt: skip
    assert agenda_budget(world, big, 1) == quotas(world, big)[1]["messages"] - 6, "f1's conversation in #ops"
    ways = room(world, big, 2)
    assert (
        ways["owed"] == 1 and ways["closes"][0] <= quotas(world, big)[2]["messages"] <= ways["closes"][1]
    ), ways
    # A day behind at night is not spaced by the night's own small quota: its next conversation starts while the
    # rest of the day still holds the conversations it needs to close.
    night, end = window(world, 2, "night")[0], bounds(world, 2)[1]
    still = math.ceil(ways["closes"][0] / big.activity.conversation_lines)
    waits = [opening(world, big, random.Random(i), night, 2) for i in range(200)]
    assert max(waits) <= (end - night) // (still + 1) + 60_000_000, (max(waits), still)
    # Four conversations in the morning's first 90 minutes: the next keeps that pace, not half the time left.
    morning, noon = window(world, 2, "morning")
    later = morning + 90 * 60_000_000
    try:
        with world.trial() as copy:
            for k in range(4):
                stamp = morning + k * 20 * 60_000_000
                copy.db.execute("INSERT INTO scenes (id, channel_id, day, part, slot_start_us, slot_end_us, situation) VALUES (?, ?, 2, 'morning', ?, ?, 'work')", (f"p{k}", ops, stamp, stamp + 1))  # fmt: skip
            waits = [opening(copy, big, random.Random(i), later, 2) for i in range(200)]
            assert statistics.mean(waits) < (noon - later) / 4, (statistics.mean(waits), noon - later)
            raise LookupError("drop the trial")
    except LookupError:
        pass
    assert today_line(world, big, 2).endswith("15 h 0 min left today"), today_line(world, big, 2)

    def posted_with(cfg, conversation):
        with world.trial() as copy:
            return post(copy, conversation, cfg, rng, gaps)

    replies = talk(*[dict(author_id=c, text=f"noted {i}") for i in range(12)], thread=rows[0]["id"])
    refused("today could not close", posted_with, big, replies)  # 12 replies: more than any day of 19 holds
    tight = settings.model_copy(update={"author": settings.author.model_copy(update={"tolerance": 0.0})})
    more = [  # with f2, three lines owed in three places today, on a day of at most two messages
        PlanFact(id="f6", storyline="s1", subject="Release 4.2", attribute="owner", value="Mia", channel_id=leads, author_id=a, day=2, summary="s"),
        PlanFact(id="f7", storyline="s1", subject="Release 4.2", attribute="reviewer", value="Raj", channel_id=ops, author_id=c, day=2, summary="s"),
    ]  # fmt: skip

    def recorded_with(cfg, document):
        with world.trial() as copy:
            record_plan(copy, document, cfg)

    refused("leaves today no way to close", recorded_with, tight, plan(facts=[*facts, *more]))
    refused(
        "before_it_happened", posted_, talk(dict(author_id=b, text="service back at {at:e2}", conveys=["f2"]))
    )
    advanced(to="evening")
    assert any(e.startswith("c1 is due") for e in close_day(world, settings, 2)), (
        "the promise due today is open"
    )
    restored = posted_(talk(dict(author_id=b, text="service back at {at:e2}, notes are up", conveys=["f2"], closes=[Close(id="c1", status="kept")])))  # fmt: skip
    moment = at(world, Moment(day=2, time="14:00", zone=zone))
    said = world.db.execute("SELECT ts_us FROM messages WHERE id = ?", (restored["messages"][0]["id"],)).fetchone()[0]  # fmt: skip
    written = {"12h": "2pm", "12h spaced": "2 PM", "24h": "14:00"}[habit(b)]
    assert restored["messages"][0]["text"] == f"service back at {written}, notes are up", ("{at:} in the author's habit, on their clock", restored)  # fmt: skip
    assert reread(written, zone, said) == moment, "it reads back to the moment"
    # A world kept in the clock style (written before 2026-10-14) keeps it: chapters on it read alike.
    assert render(moment, "UTC", moment) == "19:00 UTC" and render(moment, "UTC", moment + DAY).startswith(
        f"{datetime.fromtimestamp(moment / 1e6, ZoneInfo('UTC')):%a}"
    ), "the day is named when it is not the day of writing"
    # Every habit's every form is a literal time (an author cannot type one) and reads back to its exact moment; the
    # zone shows only when the writer's clock is not the company's.
    for way in HABITS:
        for hours in (-24 * 9, -24 * 3, -24, -2, 0, 3, 10, 24 + 2.5, 24 * 4, 24 * 12):  # 12pm and 12am among them
            for minute in (0, 0.5):
                when = moment + int((hours + minute) * 3_600_000_000)
                for place, home in ((zone, zone), ("America/New_York", zone)):
                    text = render(when, place, moment, way, home)
                    assert TIME_LITERAL.search(text) and reread(text, place, moment) == when, (way, hours, text)
                    assert (" E" in text) == (place != home), text
    assert render(moment - 2 * 3_600_000_000, zone, moment, "12h", zone) == "12pm" and render(moment + 24 * 3_600_000_000, zone, moment, "24h", zone) == "tomorrow 14:00"  # fmt: skip
    refused("names its own day", posted_, talk(dict(author_id=b, text="see you tomorrow at {at:e2}", conveys=[])))
    assert not IDENTIFIER.findall("the 5pm sync, 9am standup") and IDENTIFIER.findall("CHG-2291 at 5pm") == ["CHG-2291"]  # fmt: skip
    assert world.db.execute("SELECT status, closed_by FROM commitments").fetchone()[:] == ("kept", restored["messages"][0]["id"])  # fmt: skip
    refused("only its summary may change", recorded, plan(facts=[facts[0].model_copy(update={"value": "roll forward", "anchor": None}), *facts[1:]]))  # fmt: skip
    refused("only its summary may change", recorded, plan(facts=[facts[0], facts[1].model_copy(update={"after": []}), *facts[2:]]))  # fmt: skip
    recorded(plan(facts=[facts[0].model_copy(update={"summary": "decided early"}), *facts[1:]]))
    assert world.db.execute("SELECT summary FROM facts WHERE id = 'f1'").fetchone()[0] == "decided early"
    refused(
        "today (day 2) or later", recorded, plan(facts=[*facts[:3], facts[3].model_copy(update={"day": 1})])
    )
    refused("on or before the day of e1", recorded, plan(facts=[*facts[:2], facts[2].model_copy(update={"day": 4}), facts[3]]))  # fmt: skip
    notes = PlanFact(id="f5", storyline="s1", subject="Release 4.2", attribute="notes", value="notes posted", channel_id=ops, author_id=b, day=2, event="e2", kind="scheduled", summary="s")  # fmt: skip
    refused("e2 is past", recorded, plan(facts=[*facts, notes]))  # it is evening; e2 was at 14:00
    advanced(to="night")
    advanced(to="tomorrow")  # day 3, early: e1 at 10:00 is ahead, and f3 and f4 are to be told before it
    refused("are scheduled before", advanced, to="night")
    assert {e.split()[0] for e in close_day(world, settings, 3)} == {"f3", "f4"}, (
        "planned for day 3, unstated"
    )
    refused("comes after ['f3']", posted_, talk(dict(author_id=a, text="Owen owns the audit at {at:e1}", conveys=["f4"])))  # fmt: skip
    refused("past the moment of ['f3', 'f4']", posted_, talk(*[dict(author_id=a, text="still here", pause="hours")] * 5, channel=leads))  # fmt: skip
    posted_(  # a scheduled fact is told before its moment
        talk(
            dict(author_id=a, text="dry run for the release window at {at:e1}", conveys=["f3"]), channel=leads
        )
    )
    # A fact about an event and with no anchor of its own is stated with the event's time.
    posted_(talk(dict(author_id=a, text="Owen owns the audit at {at:e1}", conveys=["f4"]), dict(author_id=d, text="thanks")))  # fmt: skip
    advanced(to="night")

    def advanced_with(cfg):
        with world.trial() as copy:
            return advance(copy, cfg, "tomorrow")

    refused("day 3 stays open", advanced_with, strict)
    while today(world) is not None:
        advanced(to=onward(world))
    refused("the calendar is closed", posted_, talk(dict(author_id=a, text="late")))
    refused("the calendar is closed", recorded, first_plan)
    measured = close_day(world, settings.model_copy(update={"author": settings.author.model_copy(update={"share_tolerance": 0.0})}), 1)  # fmt: skip
    assert any("3 of today's 4 messages are thread replies" in e for e in measured), measured
    promising = settings.model_copy(update={"activity": settings.activity.model_copy(update={"promise_rate": 0.5})})
    assert any(e == f"{promises_made(world, 1)} promises made today; the day holds at least 2 (a line's commits)" for e in close_day(world, promising, 1)), "a day makes its promises"  # fmt: skip
    refused("t1 is a semantic level 3 task", tasked, owner.model_copy(update={"level": 2}))
    refused("one of the slots you write now: ['t1', 't2']", tasked, owner.model_copy(update={"id": "t9"}))
    level3 = lambda **needs: needing(settings, "semantic", 3, **needs)  # noqa: E731
    result = tasked(owner, level3(channels=2, relations=1))
    assert result["gold"] == [{"answer": "Owen"}] and result["bm25_rank"] and result["channels"] == 2, result
    assert "depth_share" in result and result["hidden"] is False, "rank and depth are measured, and reported"
    # The needs its messages give: where its answer is stated beside its near-misses, what the question names.
    refused("first stated in at least 2 channels", tasked, owner.model_copy(update={"facts": ["f1", "f4"]}), level3(channels=2))  # fmt: skip  # both in #ops
    refused(
        "its answer stated where it is harder to see", tasked, owner, level3(hidden=True)
    )  # f4 is top-level in #ops
    earlier = lambda copy: copy.insert("facts", [dict(id="f8", storyline="s2", subject="Audit", attribute="owner", value="Ines", channel_id=ops, author_id=a, day=2, summary="s", is_decoy=1)])  # noqa: E731  # fmt: skip
    assert tasked(owner, level3(decoys=1, hidden=True), earlier)["hidden"], "stated after every near-miss"
    leads_named = owner.model_copy(update={"question": "Who owns the audit after the dry run in #leads?"})
    refused("naming at most 0 of its evidence's channels and identifiers (it names 1)", tasked, leads_named, level3(named=0))  # fmt: skip
    assert tasked(leads_named, level3(named=1))["named"] == 1
    tasked(owner, level3())
    replier = Task(id="t2", category="lookup", level=1, actor_id=c, question="Who said thanks in #ops?", answer_type="text", gold_sql="SELECT u.real_name AS answer FROM messages m JOIN users u ON u.id = m.user_id WHERE m.text = 'thanks'")  # fmt: skip
    tasked(replier.model_copy(update={"question": "Who said thanks?"}))
    tasked(replier)  # the slot's task is replaced
    assert [tuple(r) for r in world.db.execute("SELECT id, question FROM tasks ORDER BY id")] == [("t1", owner.question), ("t2", replier.question)]  # fmt: skip
    thanks = world.db.execute("SELECT id FROM messages WHERE text = 'thanks'").fetchone()[0]
    stated = world.db.execute("SELECT message_id FROM evidence WHERE fact_id = 'f4'").fetchone()[0]

    def revised(message, text):
        with world.trial() as copy:
            return revise(copy, message, text)

    refused("changes task t2", revised, thanks, "thank you")  # t2's gold answer reads that text
    refused(
        "anchor_token_missing", revised, stated, "the audit has an owner"
    )  # f4's evidence keeps its token
    refused("states nothing new", revised, thanks, "thanks, rollback noted")
    writer, said = world.db.execute("SELECT user_id, ts_us FROM messages WHERE id = ?", (stated,)).fetchone()
    e1 = world.db.execute("SELECT moment_us FROM events WHERE id = 'e1'").fetchone()[0]
    in_habit = render(e1, zone, said, habit(writer), zone)  # a revision writes the time as its writer does
    assert revised(stated, "Owen owns the audit from {at:e1}")["text"] == f"Owen owns the audit from {in_habit}" and reread(in_habit, zone, said) == e1  # fmt: skip
    ids = [r[0] for r in world.db.execute("SELECT id FROM messages ORDER BY ts_us")]
    assert ids == sorted(ids) and world.violations(complete=True) == []
    print(
        "PASS clock: plan and events, posts at the present, tagging, first statements, promises, day close, revise, tasks"
    )


def author_context(settings, cast, org, root, agenda=None):
    """What WorldTools are configured with, as the driver builds it; `agenda` maps a day to its drawn scenes."""
    state = SimpleNamespace(
        quota=[
            Slot(id=f"t{i}", category=c, level=n, concept="a concept", style="a style")
            for i, (c, n) in enumerate((("semantic", 3), ("lookup", 1)), 1)
        ],  # fmt: skip
        cast=cast,
        plans={
            "agenda": {
                str(d): [s.model_dump(mode="json") for s in scenes] for d, scenes in (agenda or {}).items()
            }
        },
    )
    return context_of(settings, state, root, org.model_dump(mode="json"))


async def check_tools(root):
    """The author's tools (v7): ten, served host-side over the world file (annotate is the forge's); each turn's tools act only in their turn
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
            "taxonomy": needless(base.taxonomy),
        }
    )
    world, cast, org, (a, b, c, d) = organized(root, settings)
    ops, leads = channel_id("public", "ops", []), channel_id("private", "leads", [])
    with world.trial() as copy:
        start_clock(copy)
    agenda = {d: background_plan(world, org, settings.activity, settings.seed, d, agenda_budget(world, settings, d)) for d in quotas(world, settings)}  # fmt: skip
    context = author_context(settings, cast, org, root, agenda)
    ledger = Plan(
        storylines=[Storyline(id="s1", summary="the 4.2 release"), Storyline(id="s2", summary="the audit")],
        events=[Event(id="e1", storyline="s2", title="audit window", day=3, time="10:00")],
        facts=[
            PlanFact(id="f1", storyline="s1", subject="Release 4.2", attribute="decision", value="rollback", anchor="rollback", channel_id=ops, author_id=a, day=1, summary="s"),
            PlanFact(id="f2", storyline="s2", subject="Audit", attribute="owner", value="Owen", channel_id=leads, author_id=a, day=3, after=["f1"], event="e1", kind="scheduled", summary="s"),
        ],
        board=[BoardEntry(slot="t1", facts=["f1", "f2"])],
    )  # fmt: skip

    def tools_for(mode, day):
        task = WorldAuthorTask.create(mode, day, world.path, context, f"{mode}-{day:02d}")
        (tools,) = task.toolsets(task.config)
        return tools

    planner = tools_for("plan", 0)
    await planner.setup()
    with_state(planner)
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
        page.startswith("# Now: ") and "day 1 of 10" in page and "[[f1]]" in page and "## Task slots" in page
    )
    assert "everyday conversations code drew for today" in page and agenda[1]
    assert "- today: 0 messages today (thread replies 0, reacted to 0, in DMs 0, mention someone 0); it closes with" in page
    decision = Conversation(channel_id=ops, about="decision", lines=[PostLine(author_id=a, text="rollback it is", conveys=["f1"]), PostLine(author_id=b, text="ok", reply_to=0)])  # fmt: skip
    out = await call(day1, "post", conversation=decision)
    assert len(out["messages"]) == 2 and out["now"].endswith("CDT"), out
    count = world.db.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    assert await call(day1, "post", conversation=decision) == out, (
        "a retry after a lost response returns the post"
    )
    assert world.db.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == count, "and posts nothing more"
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
    try:
        await call(day1, "read", actor_id=c, tool="read_channel", arguments={"nope": 1})
        raise AssertionError("a read with unknown arguments went through")
    except TypeError:
        pass
    while today(world) == 1:
        await call(day1, "advance", to=onward(world))
    try:
        await call(day1, "post", conversation=Conversation(channel_id=ops, about="x", lines=[PostLine(author_id=a, text="hi")]))  # fmt: skip
        raise AssertionError("day 1's tools posted on day 2")
    except ValueError as error:
        assert "day 1 is closed" in str(error), error
    log = [json.loads(line) for line in (root / "world-calls.jsonl").read_text().splitlines()]
    assert {e["tool"] for e in log} >= {"plan", "post", "view", "sql", "read", "advance", "now"}
    assert any(not e["ok"] and "conveys" in e["error"] for e in log)
    assert [e["ok"] for e in log if e["tool"] == "plan"] == [True], "one plan went through"
    assert any(e["tool"] == "read" and not e["ok"] and e["error"].startswith("TypeError") for e in log), (
        "a fault is logged as failed"
    )
    assert any(e["tool"] == "post" and e["args"]["conversation"]["about"] == "decision" for e in log), (
        "with its arguments"
    )
    served = tools_for("day", 2)  # a tool server checks the world file's hash when it starts
    with_state(served)
    described = await served_tools(served)
    assert set(described) == {"now", "view", "sql", "read", "plan", "post", "advance", "revise", "edit", "add_task", "profile", "probe", "annotate"} and served.server_name == "world"  # fmt: skip
    # help() in the author's IPython shows a tool's description alone: it names every field of every document a
    # call takes, nested ones too, with ? when optional.
    for tool, models in {"post": (Conversation, PostLine, Reaction, Commit, Close), "plan": (Plan, PlanFact, Event, Storyline), "add_task": (Task,)}.items():  # fmt: skip
        for model in models:
            for field, info in model.model_fields.items():
                marked = f"{field}{'' if info.is_required() else '?'}: "
                assert marked in described[tool].split("Arguments: ", 1)[1], (tool, marked)
    assert "lines: [1-40 × {author_id: str" in described["post"] and described["now"].endswith(
        "Arguments: none"
    )
    async with (
        asyncio.timeout(30),
        serve(served) as url,
        streamable_http_client(url) as (reader, writer_),
        ClientSession(reader, writer_) as client,
    ):  # nested documents cross the wire as strict models, and a refused write carries its reason
        await client.initialize()
        refused = await client.call_tool("post", {"conversation": {"channel_id": ops, "about": "x", "lines": [{"author_id": a, "text": "hi", "commits": [{"id": "c1", "text": "t", "due_day": 2.5}]}]}})  # fmt: skip
        assert refused.is_error and "due_day" in str(refused.content), refused
    invalid = [json.loads(line) for line in (root / "world-calls.jsonl").read_text().splitlines()][-1]
    assert invalid["tool"] == "post" and invalid["error"].startswith("invalid arguments: conversation.lines.0.commits.0.due_day"), invalid  # fmt: skip
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
                who = [rng.choice([a, b]) for _ in range(4)]  # every other conversation mentions someone once
                post(world, Conversation(channel_id=rng.choice([ops, leads]), about="work", lines=[PostLine(author_id=w, text=f"note {lines + i} on the work" + (f", <@{a if w == b else b}>" if i == 1 and lines % 8 == 0 else ""), reply_to=0 if i else None) for i, w in enumerate(who)]), settings, rng, gaps)  # fmt: skip
                lines += 4
                continue
            except ValueError:
                pass
        advance(world, settings, onward(world))
    assert world.db.execute("SELECT COUNT(*) FROM messages").fetchone()[0] > 300
    # A day holds a share of messages that mention someone, near the measured 11%, in a tighter band than the others.
    days_ = [tally(world, d) for (d,) in world.db.execute("SELECT day FROM calendar")]
    assert share_band(settings, "mention_rate", 52) == (3, 9) and all(share_band(settings, "mention_rate", t["messages"])[0] <= t["mention_rate"] <= share_band(settings, "mention_rate", t["messages"])[1] for t in days_), days_  # fmt: skip
    memory = files(world, settings, context, "tasks")
    # A person's page says who they are beyond their title and how they type, in words; the guide says how Slack sounds.
    page = memory["memory/people/alicia.rao.md"]
    assert "## Who they are" in page and "calm" in page and "midwest" in page and "chess" in page, page
    assert "20% of their messages are short (4 words or fewer)" in page, page
    assert (
        "\nVoice\n- This is Slack"
        in WorldAuthorTask.create("day", 1, world.path, context, "day-01-01").data.system_prompt
    )
    raw = re.compile(r"\b1\d{15}\b")
    assert not [p for p, text in memory.items() if raw.search(text)], "no raw microseconds reach the author"
    assert len(memory["memory/now.md"].encode()) < 20_000, len(memory["memory/now.md"].encode())
    assert Plan.model_validate(json.loads(memory["memory/ledger.json"])) == ledger, (
        "ledger.json is world_plan's input"
    )
    assert not raw.search(json.dumps(ledger_digest(world))), "the judge's ledger is rendered too"
    print(
        "PASS tools: ten served, scoped by turn and day, checked and logged; memory rendered, small, round-trips"
    )


async def served_tools(toolset):
    async with (
        asyncio.timeout(30),
        serve(toolset) as url,
        streamable_http_client(url) as (reader, writer_),
        ClientSession(reader, writer_) as client,
    ):
        await client.initialize()
        return {t.name: t.description for t in (await client.list_tools()).tools}


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
    fails(
        validate_verdict, verdict(approved=False, issues=[workspace]), payload
    )  # a blocking one names its tasks
    validate_verdict(verdict(issues=[workspace.model_copy(update={"blocking": False})]), payload)
    assert "workspace" in str(schema["$defs"]["Issue"]["properties"]["artifact"])
    issue = Issue(artifact="tasks", task_ids=["t2"], defect="d", requested_change="c")
    fails(verdict, issues=[issue])
    fails(verdict, approved=False)
    batch = verdict(approved=False, issues=[issue], tasks=(("t1", True), ("t2", True)))
    strict, lenient = Acceptance(minor_issues_block=True), Acceptance()
    assert accepted_task(batch, lenient, "t1") and not accepted_task(batch, lenient, "t2")
    general = batch.model_copy(update={"issues": [issue.model_copy(update={"task_ids": []})]})
    assert accepted_task(general, lenient, "t1"), "an issue naming no task blocks none: it is the world's"
    minor = batch.model_copy(update={"issues": [issue.model_copy(update={"blocking": False})]})
    assert (
        accepted_task(minor, lenient, "t2")
        and not accepted_task(minor, strict, "t2")
        and not deciding(minor, lenient)
    )

    world.snapshot(root / "review.sqlite")
    judge = JudgeTask.create(payload, root / "review.sqlite", "final-01", 5)
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
    assert set(await served_tools(tools)) == {"check", "read", "sql"} and tools.server_name == "inspect"
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
    _, untasked = await inspected("world", [])
    assert [v.split(":")[0] for v in due["violations"]] == ["unstated_fact"] and "bm25_rank" in due["tasks"][
        "t1"
    ]
    assert untasked["violations"] == [], "completion is due for reviewed tasks"

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
    assert solve.data.system_prompt.startswith("You are using Slack as the user with user_id U2. It is "), "the client says who and when"  # fmt: skip
    assert "rollback" not in solve.data.model_dump_json() + solve.config.tools.model_dump_json()
    assert solve.config.tools.db_hash == sha256(root / "solver.sqlite")
    # The answer judge grades the answer the response commits to and grounds the claims that make it; an unsupported
    # aside is counted, and costs nothing.
    asks = AnswerJudge(solve.config.judge).build_messages(
        question="q", reference={}, response="r", observations=[]
    )
    assert "lists alternatives without settling on one is wrong" in asks[0].content and "do not change grounded" in asks[0].content  # fmt: skip
    assert "never ask for a decoy to be\ncorrected or hedged" in PHASE_GUIDES["world"], (
        "the world judge knows a decoy is planned"
    )
    assert "unsupported_asides" in AnswerGrade.model_json_schema()["required"]
    observed = [{"tool": "read_channel", "arguments": {}, "output": {"items": [{"channel": "C1", "ts": answer.messages[0][1]}]}}]  # fmt: skip
    assert "A status reference is the latest value the asker can see" in asks[0].content and "has abstained" in asks[0].content  # fmt: skip
    # The reward: +1 right and grounded, 0 right but ungrounded or declined, -1 wrong (an aside costs nothing).
    for correct, abstained, grounded, asides, score in ((True, False, True, 2, 1.0), (True, False, False, 0, 0.0), (False, True, False, 0, 0.0), (False, False, False, 0, -1.0), (True, True, True, 0, 1.0), (False, False, True, 0, -1.0)):  # fmt: skip
        grade = AnswerGrade(correct=correct, abstained=abstained, grounded=grounded, unsupported_asides=asides, committed=["x"], missing=[], extra=[], reason="r")  # fmt: skip
        trace = SimpleNamespace(info={"observations": observed}, last_reply="rollback", id="try", record_metrics=lambda m: None)  # fmt: skip
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
        assert trace.info["evaluation"]["unsupported_asides"] == asides and trace.info["evaluation"]["evidence_coverage"] == 0.5  # fmt: skip
    # Grounded by message: tries never shown the gold's messages are ungrounded, whatever the judge says.
    elsewhere = [
        {"tool": "read_channel", "arguments": {}, "output": {"items": [{"channel": "C9", "ts": "1.000001"}]}}
    ]
    grade = AnswerGrade(correct=True, abstained=False, grounded=True, unsupported_asides=0, committed=["x"], missing=[], extra=[], reason="r")  # fmt: skip
    trace = SimpleNamespace(info={"observations": elsewhere}, last_reply="rollback", id="try", record_metrics=lambda m: None)  # fmt: skip
    judged, AnswerJudge.evaluate = AnswerJudge.evaluate, lambda self, **fields: asyncio.sleep(0, SimpleNamespace(parsed=grade))  # fmt: skip
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
        {"tool": "search_messages", "arguments": {"query": "nothing"}, "output": {"items": [], "next_cursor": None}},  # fmt: skip
        {"tool": "read_channel", "arguments": {}, "output": {"items": [{"channel": "C1", "ts": gold_ts, "user": "U1", "text": "a", "time_utc": "z"}]}},  # fmt: skip
        {"tool": "search_messages", "arguments": {"query": "a"}, "output": {"items": [{"channel": "C1", "ts": gold_ts, "user": "U1", "text": "a", "channel_name": "c"}]}},  # fmt: skip
        {"tool": "get_reactions", "arguments": {}, "output": {"channel": "C1", "ts": gold_ts, "reactions": [{"name": "eyes", "count": 1, "users": ["U3"]}]}},  # fmt: skip
        {"tool": "list_channel_members", "arguments": {"channel_id": "C1"}, "output": {"items": ["U4", "U2"], "next_cursor": None}},  # fmt: skip
    ]
    view = compact(calls)
    assert [(m["ts"], m.get("channel_name"), "time_utc" in m) for m in view["messages"]] == [(gold_ts, "c", False)]
    assert view["messages"][0]["reactions"][0]["users"] == ["U3"] and view["members"] == {"C1": ["U2", "U4"]}
    assert [c["results"] for c in view["calls"]] == [0, 1, 1, 1, 2], "an empty search stays in the log"
    assert set().union(*map(people_shown, calls)) == {"U1", "U2", "U3", "U4"}, "authors, reactors, members are seen"
    around = {"tool": "search_public_and_private", "arguments": {}, "output": {"items": [{"channel": "C1", "ts": "5.000001", "user": "U1", "text": "x", "context": {"before": [{"channel": "C1", "ts": "4.000001", "user": "U5", "text": "y"}], "after": []}}]}}  # fmt: skip
    assert shown(around) == {("C1", "5.000001"), ("C1", "4.000001")} and people_shown(around) == {"U1", "U5"}, "a hit's context is shown too"  # fmt: skip
    passed = {}

    async def graded(self, **fields):
        passed.update(fields)
        return SimpleNamespace(parsed=AnswerGrade(correct=True, abstained=False, grounded=True, unsupported_asides=0, committed=["x"], missing=[], extra=[], reason="r"))  # fmt: skip

    trace = SimpleNamespace(info={"observations": calls}, last_reply="rollback", id="try", record_metrics=lambda m: None)  # fmt: skip
    judged, AnswerJudge.evaluate = AnswerJudge.evaluate, graded
    try:
        assert await SolverTask.semantic_correctness(solve, trace) == 1.0
        assert passed["observations"] == view and trace.info["evaluation"]["evidence_coverage"] == 1.0, (
            "U1 is seen as the gold message's author"
        )
        passed.clear()
        trace = SimpleNamespace(info={"observations": calls}, last_reply="  ", id="try", record_metrics=lambda m: None)  # fmt: skip
        assert await SolverTask.semantic_correctness(solve, trace) == -1.0 and not passed, "no grader call"
        empty = trace.info["evaluation"]
        assert empty["reason"] == EMPTY and empty["missing"] == [str(r["answer"]) for r in answer.rows]
    finally:
        AnswerJudge.evaluate = judged
    write_release(root / "release", root / "solver.sqlite", [task], {"t1": answer})
    assert load_release(root / "release")[1:] == ([task], {"t1": answer})
    manifest = json.loads((root / "release" / "manifest.json").read_text())
    assert manifest["format"] == "worldgen-slack.v10"
    for old in ("worldgen-slack.v7", "worldgen-slack.v8"):
        (root / "release" / "manifest.json").write_text(json.dumps(manifest | {"format": old}))
        assert load_release(root / "release")[1:] == ([task], {"t1": answer}), f"a {old} release still loads"
    (root / "release" / "manifest.json").write_text(json.dumps(manifest))
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
    # A refusal's truth out of its actor's sight ships apart, as unseen; evidence its actor can read may not.
    out_of_sight = answer.model_copy(update={"messages": [], "unseen": hidden.messages})
    write_release(root / "release3", root / "solver.sqlite", [task], {"t1": out_of_sight})
    assert load_release(root / "release3")[2]["t1"].unseen == hidden.messages
    fails(write_release, root / "release4", root / "solver.sqlite", [task], {"t1": answer.model_copy(update={"unseen": answer.messages})})  # fmt: skip
    print("PASS reviews: verdict contracts, per-task acceptance, judge tools as actors, solver copy, release")


TestStore = Store  # the flows keep the real store: its traces, events and attempts


class FakeTrace(SimpleNamespace):
    """A scripted seat's trace, with what the store and the env read of a real one."""

    def __init__(self, id, role, record=None, **fields):
        base = dict(ok=True, info={}, errors=[], last_reply="", usage=None, is_completed=True)
        super().__init__(id=id, agent=SimpleNamespace(name=role), **(base | fields))
        self.record = record

    def to_record(self):
        task = getattr(self, "task", None)
        data = task.data.model_dump(mode="json", exclude_none=True) if task is not None else {}
        return self.record or {"id": self.id, "agent": {"name": self.agent.name, "config": {}}, "task": {"data": data}, "ok": self.ok, "nodes": [], "info": self.info}  # fmt: skip

    def record_metric(self, name, value):
        self.info.setdefault("metrics", {})[name] = value


class ScriptedAgent:
    trainable = False
    traced = Counter()  # every scripted interaction's trace has an id of its own

    def trace(self, role, task=None):
        ScriptedAgent.traced[role] += 1
        return FakeTrace(f"{role}-{ScriptedAgent.traced[role]}", role, task=task)

    @asynccontextmanager
    async def provision(self, task):
        yield None

    @asynccontextmanager
    async def interaction(self, task, runtime):
        yield SimpleNamespace(trace=self.trace("agent", task))


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


class FakeRuntime:
    """The author's VM as files: what the env writes, the author reads and writes, and rm."""

    def __init__(self):
        self.files, self.pings = {}, 0

    async def write(self, path, data):
        self.files[path] = data

    async def read(self, path, max_bytes=None):
        if path not in self.files:
            raise SandboxError(f"no {path}")
        return self.files[path]

    async def run(self, argv, env):
        self.pings += argv == ["true"]
        if argv[:2] == ["rm", "-f"]:
            for path in argv[2:]:
                self.files.pop(path, None)
        elif argv[:2] == [
            "sh",
            "-c",
        ]:  # the env's upload: empty a directory, then unpack an archive into /task
            if cleared := re.match(r"rm -rf (\S+) && ", argv[2]):
                for path in [p for p in self.files if p.startswith(cleared.group(1) + "/")]:
                    del self.files[path]
            (archive,) = re.findall(r"tarfile -e (\S+) /task", argv[2])
            with tarfile.open(fileobj=io.BytesIO(self.files.pop(archive)), mode="r:gz") as unpacked:
                for member in unpacked.getmembers():
                    self.files["/task/" + member.name] = unpacked.extractfile(member).read()
        return SimpleNamespace(exit_code=0, stdout="", stderr="")


class ScriptedAuthor(ScriptedAgent):
    """A world author played by a script: each turn runs it against the turn's own world tools and VM."""

    def __init__(self, script):
        self.script = script

    @asynccontextmanager
    async def provision(self, task):
        yield FakeRuntime()  # as verifiers provisions a box: empty, the task's setup runs in its rollout

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

        yield SimpleNamespace(trace=self.trace("author", task), turn=turn)


def setup_document(runtime) -> None:
    """The setup author's turn: the document its phase asks for, written where the env reads it."""
    context = json.loads(runtime.files["/task/input.json"])
    if context["phase"] == "premise":
        document = Premises(premises=[Premise(company=f"{w} Software", niche="n", region="r", size="s", culture="c", cast="x", staffing={"software_developer": 4}) for w in ("Lattice", "Birch")])  # fmt: skip
    else:
        ids = [c["user_id"] for c in context["candidates"]]
        kinds = [Routine(kind=k, probability=0.5) for k in ("standup", "handoff", "lunch")]
        channels = [Channel(name="ops", type="public", members=ids, routines=kinds), Channel(name="leads", type="private", members=ids[:2], routines=kinds)]  # fmt: skip
        document = Organization(people=[Person(user_id=u, title="Engineer", team="Platform") for u in ids], channels=channels, dm_routines=kinds)  # fmt: skip
    runtime.files[f"/task/{context['phase']}.json"] = document.model_dump_json().encode()


async def judged(task, verdict):
    """A review's verdict as a judge seat returns it: the task's own setup lays out its files on a fake box, the verdict
    is written where the task reads it, and the task's finalize checks it."""
    runtime, trace = FakeRuntime(), ScriptedAgent().trace("judge", task)
    trace.state = SimpleNamespace(checked=True, calls=[])
    await task.setup(trace, runtime)
    runtime.files["/task/verdict.json"] = verdict.model_dump_json().encode()
    await task.finalize(trace, runtime)
    return trace


GRADER = {"on": False, "loose": False, "blind": False}  # the flows' answer judge, played by code


async def scripted_complete(self, messages, *, trace=None, schema=None, parse=None, **sampling):
    """The answer judge in the flows: it accepts a task's gold alone; a loose one a bare opener too, a blind one not even
    the gold. Any other judge call fails, as `offline` does."""
    text = messages if isinstance(messages, str) else messages[0].content
    if not GRADER["on"] or not text.startswith("Grade a Slack answer"):
        return await offline()
    fields = json.loads(text[text.rindex("reason.\n") + len("reason.\n") :])
    response, answers = fields["response"], ", ".join(str(r.get("answer")) for r in fields["reference"]["rows"])
    right = response in (f"{answers}.", "I can't answer that from anything you can see: it isn't there.") or response.startswith(f"{answers}, though")  # fmt: skip
    correct = (right and not GRADER["blind"]) or (GRADER["loose"] and response == "Thought process:")
    grade = AnswerGrade(correct=correct, abstained=False, grounded=False, unsupported_asides=0, committed=[], missing=[], extra=[], reason="scripted")  # fmt: skip
    return vf.JudgeResponse(text=grade.model_dump_json(), parsed=parse(vf.JudgeResponse(text="", parsed=grade)) if parse else grade)  # fmt: skip


vf.Judge.complete = scripted_complete  # a judge left unscripted fails here, before it reaches the gateway


def world_seats():
    """The seats of a world with no tasks of its own: an author that sets up, plans, writes four days (revising what a
    review names) and fixes what the final review rejects; a judge with a note after day 2 that rejects the final
    review once. Returns the author, the judge, the messages fixed and the reviews made."""
    fixed, reviewed = [], []

    async def script(prompt, tools, runtime):
        if tools is None:
            return setup_document(runtime)
        call = lambda name, **kwargs: tools._with_state(getattr(tools, name))(**kwargs)  # noqa: E731
        world = World(tools.config.db_path)
        a, b, c, d = [r[0] for r in world.db.execute("SELECT id FROM users ORDER BY id")]
        ops, leads = channel_id("public", "ops", []), channel_id("private", "leads", [])

        def talk(channel, *lines):
            lines = [PostLine(**line) for line in lines]
            return call("post", conversation=Conversation(channel_id=channel, about="work", lines=lines))

        if tools.config.mode == "plan":
            events = [Event(id="e1", storyline="s2", title="audit window", day=3, time="10:00"), Event(id="e2", storyline="s1", title="service restored", day=2, time="14:00")]  # fmt: skip
            facts = [
                PlanFact(id="f1", storyline="s1", subject="Release 4.2", attribute="decision", value="rollback", anchor="rollback", channel_id=ops, author_id=a, day=1, summary="s"),
                PlanFact(id="f2", storyline="s1", subject="Release 4.2", attribute="state", value="service back", channel_id=ops, author_id=b, day=2, after=["f1"], event="e2", kind="happened", summary="s"),
                PlanFact(id="f3", storyline="s2", subject="Audit", attribute="window", value="dry run", anchor="dry run", channel_id=leads, author_id=a, day=3, event="e1", kind="scheduled", summary="s"),
                PlanFact(id="f4", storyline="s2", subject="Audit", attribute="owner", value="Owen", channel_id=ops, author_id=a, day=3, event="e1", kind="scheduled", summary="s"),
            ]  # fmt: skip
            await call("plan", ledger=Plan(storylines=[Storyline(id="s1", summary="the release"), Storyline(id="s2", summary="the audit")], events=events, facts=facts, board=[]))  # fmt: skip
            await runtime.write("/task/notes/plan.md", b"arcs: the release, then the audit")
        elif tools.config.mode == "day":
            day = tools.config.day
            if day == 1:
                await talk(ops, dict(author_id=a, text="we go with the rollback", conveys=["f1"]), dict(author_id=b, text="ok, notes tomorrow", reply_to=0, commits=[Commit(id="c1", text="post the notes", due_day=2)]))  # fmt: skip
            if day == 2:
                await call("advance", to="evening")
                await talk(ops, dict(author_id=b, text="service back at {at:e2}; notes are up", conveys=["f2"], closes=[Close(id="c1", status="kept")]))  # fmt: skip
            if day == 3:
                assert "message_ids" in prompt, "the review's note opens day 3"
                issue = json.loads(prompt[prompt.index("[{") : prompt.index("}]") + 2])[0]
                for message in issue["message_ids"]:
                    await call("revise", message_id=message, text="ok, notes tomorrow then")
                await talk(leads, dict(author_id=a, text="dry run for the audit window at {at:e1}", conveys=["f3"]))
                await talk(ops, dict(author_id=a, text="Owen owns the audit at {at:e1}", conveys=["f4"], reactions=[Reaction(user_id=d, emoji="eyes")]), dict(author_id=d, text=f"thanks <@{a}>"))  # fmt: skip
            if day == 4:
                await talk(ops, dict(author_id=d, text="quiet day"))
            while today(world) == day:
                await call("advance", to=onward(world))
            await runtime.write("/task/notes/recap.md", f"day {day}: done".encode())
        elif prompt.startswith("The review rejected"):
            issues = json.loads(prompt[prompt.index("[") : prompt.index("]\n") + 1])
            for issue in issues:
                for message in issue.get("message_ids", []):
                    await call("revise", message_id=message, text="ok, the notes come tomorrow")
                    fixed.append(message)
        world.close()

    class Judge(ScriptedAgent):  # a note after day 2; the final review rejects once, then approves
        async def run(self, task):
            payload, attempt = json.loads(task.config.payload), task.data.attempt
            reviewed.append(attempt)
            world = World(task.config.tools.db_path)
            first, second = [m for (m,) in world.db.execute("SELECT id FROM messages ORDER BY id LIMIT 2")]
            world.close()
            issues = []
            if attempt in ("review-02-01", "final-01"):
                issues = [Issue(artifact="workspace", message_ids=[second], evidence_message_ids=[first], defect="stiff", requested_change="looser", blocking=attempt == "final-01")]  # fmt: skip
            verdict = Verdict(approved=not any(i.blocking for i in issues), tasks=[], issues=issues, criteria=dict.fromkeys(PHASE_CRITERIA[payload["phase"]], 1.0), summary="s")  # fmt: skip
            return await judged(task, verdict)

    return ScriptedAuthor(script), Judge(), fixed, reviewed


def flow_run_settings(root):
    """The flows' settings: a 4-day world of 16 messages, 4 tasks in 2 batches, a review after day 2."""
    base = flow_settings(root, 3)
    cut = {
        "robustness": 2
    }  # S1's 17 cells, without needs: the flow's scripted tasks are not built to taxonomy v2
    flow = {n: c.model_copy(update={"levels": c.levels[: cut.get(n, 3)], "concepts": c.concepts[: cut.get(n, 3)], "needs": None}) for n, c in base.taxonomy.items()}  # fmt: skip
    return base.model_copy(
        update={
            "taxonomy": flow,
            "calendar": base.calendar.model_copy(update={"days": 4}),
            "activity": base.activity.model_copy(update={"messages": 16}),
            # 4 tasks in 2 batches; level 3's band starts at 0, so the witness tries its task
            "tasks": base.tasks.model_copy(
                update={"per_100": 25, "batch": 2, "bands": [(0.75, 1.0), (0.5, 0.75), (0.0, 0.5)]}
            ),  # fmt: skip
            "author": base.author.model_copy(
                update={
                    "tolerance": 1.0,
                    "share_tolerance": 1.0,
                    "review_days": [2],
                    "tries": 3,
                    "task_rounds": 1,
                    "review_chunk": 1,
                    "solvers": 2,
                    "keepalive": 0.0005,
                }  # fmt: skip
            ),
        }
    )


@contextmanager
def episodes(agents):
    """The CLI's run functions on scripted seats: an episode sets its env up and runs it, with no model, and records a
    failure as verifiers' episodes do."""

    async def run_episode(self, task, context):
        await self.setup(agents)
        errors = []
        try:
            await self.run(task, agents)
        except Exception as error:
            errors.append(SimpleNamespace(type=type(error).__name__, message=str(error)))
        return SimpleNamespace(id="episode", traces=[], errors=errors, to_record=lambda: {"id": "episode", "errors": [vars(e) for e in errors]})  # fmt: skip

    @asynccontextmanager
    async def serving(self):
        yield

    saved = vf.Env.run_episode, vf.Env.serving
    vf.Env.run_episode, vf.Env.serving = run_episode, serving
    try:
        yield
    finally:
        vf.Env.run_episode, vf.Env.serving = saved


def forge_seats(note_on=None) -> dict:
    """The seats of a plain forge: a proposer that writes one lookup task for each open id, a judge that approves every
    task (with a note on message `note_on`, when given), and a solver and a witness that answer right."""

    async def propose(prompt, tools, runtime):
        call = lambda name, **kwargs: tools._with_state(getattr(tools, name))(**kwargs)  # noqa: E731
        world = World(tools.config.db_path)
        (actor,) = world.db.execute("SELECT id FROM users WHERE is_bot = 0 ORDER BY id LIMIT 1").fetchone()
        world.close()
        for open_id in re.findall(r"r\d\d-\d\d", prompt):
            await call("add_task", task=Task(id=open_id, category="lookup", level=1, actor_id=actor, question="Who is the newest person here?", answer_type="text", gold_sql="SELECT real_name AS answer FROM users ORDER BY id DESC LIMIT 1", intent="structure"))  # fmt: skip

    class TaskJudge(ScriptedAgent):
        async def run(self, task):
            payload = json.loads(task.config.payload)
            notes = [Issue(artifact="workspace", message_ids=[note_on], defect="a stiff old line", requested_change="loosen it", blocking=False)] if note_on else []  # fmt: skip
            reviews = [TaskReview(task_id=t["id"], valid=True, reason="r", level_fit=3) for t in payload["tasks"]]
            verdict = Verdict(approved=True, tasks=reviews, issues=notes, criteria=dict.fromkeys(PHASE_CRITERIA[payload["phase"]], 1.0), summary="s")  # fmt: skip
            return await judged(task, verdict)

    class RightSolver(ScriptedAgent):
        async def run(self, task):
            evaluation = {"task_id": task.data.task_id, "execution_ok": True, "semantic_correctness": 1.0, "correct": True, "grounded": True, "calls": 1, "reason": "r", "response": "an answer"}  # fmt: skip
            return FakeTrace(self.trace("solver").id, "solver", info={"evaluation": evaluation}, task=task)

    return dict(judge=TaskJudge(), solver=RightSolver(), witness=RightSolver(), author=ScriptedAuthor(propose))


async def world_run(root, settings, agents_of):
    """A world written with no tasks of its own, as W1 is: setup, the plan, four days with a review after day 2 whose
    note opens day 3, and a final review that rejects once and is fixed in the same session. It ends at the final
    review. The forge, the measure and the chapter below start from it. Returns its run directory and settings."""
    settings = settings.model_copy(update={"output": root / "world", "tasks": settings.tasks.model_copy(update={"in_world": False})})  # fmt: skip
    author, judge, fixed, reviewed = world_seats()

    with episodes(agents_of(judge=judge, author=author)):
        try:
            await generate_run(settings)
            raise AssertionError("a world with no tasks published a release")
        except ValueError as error:  # it publishes after its final review all the same, an empty release
            assert "public tasks and private answers must match exactly" in str(error), error
    store = TestStore(settings.output, provenance(settings))
    state = store.state
    assert state.phase == "done" and GenerationEnv(settings, store).after_days == "final", (state.phase, state.rounds)
    assert reviewed == ["review-02-01", "final-01", "final-02"] and len(fixed) == 1, (reviewed, fixed)
    assert store.world.db.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0 and not state.quota, "no tasks"
    assert [e["day"] for e in map(json.loads, (store.root / "progress.jsonl").read_text().splitlines()) if e["event"] == "day_closed"] == [1, 2, 3, 4]  # fmt: skip
    assert store.world.violations(complete=True) == []
    store.close()
    return settings.output, settings


async def check_author(root):
    """One world written by a scripted author through the v7 driver: setup, the plan, days that close only when done,
    a review after day 2 whose issues open day 3, a crash in that review and in day 3 with resumes from day 2's world
    and notes, tasks, probes and hardening toward each level's band, a final rejection fixed in the same session,
    publish."""
    root.mkdir(parents=True)
    settings = flow_run_settings(root)
    seen, crash, solved, finished = [], {"armed": True, "review": True, "plans": 0, "tasks": True}, Counter(), Counter()  # fmt: skip
    GRADER.update(on=True, loose=False, blind=False)
    t0, t1, t2, t3 = [
        s.id for s in quota(settings.taxonomy, settings.tasks.styles, settings.seed, settings.task_count)
    ]
    # The solver: every other try is right, but t1 and t3 are right every time: t1 (search level 2) is too easy for its
    # band, t3 (join level 1) is in its band. t2's right answers are never grounded, so only its strict rate is 0.
    always = {t1, t3}
    # Runs that crash: t0's first is run again; one try of t2 crashes twice and is left out of its rates.
    crashes = {("tasks-01", t0): {1}, ("tasks-03", t2): {2, 3}}

    class ScriptedSolver(ScriptedAgent):
        async def run(self, task):
            key = (store.state.active_attempt, task.data.task_id)
            solved[key] += 1
            name = "-".join([*key, str(solved[key])])
            if solved[key] in crashes.get(key, ()):
                return FakeTrace(name, "solver", ok=False, errors=[SimpleNamespace(message="model stream ended")], task=task)  # fmt: skip
            finished[key] += 1
            right, grounded = task.data.task_id in always or finished[key] % 2 == 1, task.data.task_id != t2
            if (
                "hard to find" in task.data.prompt
            ):  # a forge candidate it answers only when shown the evidence
                right = "These messages bear on it" in task.data.prompt
            evaluation = {"task_id": task.data.task_id, "execution_ok": True, "semantic_correctness": float(right and grounded), "correct": right, "grounded": grounded, "calls": 2, "reason": "r", "response": "an answer"}  # fmt: skip
            # A search that misses, then a read that shows the task's evidence, if it rests on any.
            found = [{"channel": c, "ts": ts} for c, ts in task.config.reference.messages]
            observations = [{"tool": "search_messages", "arguments": {"query": "release"}, "output": {"items": []}}, {"tool": "read_channel", "arguments": {"channel_id": "C1"}, "output": {"items": found}}]  # fmt: skip
            record = {"id": name, "agent": {"name": "solver", "config": {}}, "task": {"data": {"task_id": task.data.task_id}}, "ok": True, "nodes": [], "info": {"observations": observations, "evaluation": evaluation}}  # fmt: skip
            return FakeTrace(name, "solver", record=record, info={"evaluation": evaluation}, task=task)

    witnessed = Counter()

    class ScriptedWitness(ScriptedAgent):  # the stronger solver answers right
        async def run(self, task):
            key = (store.state.active_attempt, task.data.task_id)
            witnessed[key] += 1
            name = "-".join(["witness", *key, str(witnessed[key])])
            evaluation = {"task_id": task.data.task_id, "execution_ok": True, "semantic_correctness": 1.0, "correct": True, "grounded": True, "calls": 1, "reason": "r", "response": "an answer"}  # fmt: skip
            record = {"id": name, "agent": {"name": "witness", "config": {}}, "task": {"data": {"task_id": task.data.task_id}}, "ok": True, "nodes": [], "info": {"observations": [], "evaluation": evaluation}}  # fmt: skip
            return FakeTrace(name, "witness", record=record, info={"evaluation": evaluation}, task=task)

    class FlowJudge(ScriptedAgent):
        """The judge at its seat: it reads the review's input and writes its verdict; a review after day 2 is cut off
        once; the final review rejects the world once."""

        async def run(self, task):
            payload, attempt, files = json.loads(task.config.payload), task.data.attempt, task.config.files
            seen.append(("review", attempt, payload["phase"], "written_through" in payload, "ledger" in payload, bool(payload.get("previous_issues")), tuple(t["id"] for t in payload["tasks"])))  # fmt: skip
            if payload.get("witness"):
                seen.append(("witnessed", attempt, tuple(t["id"] for t in payload["tasks"]), len(payload["witness"]), tuple(sorted(f for f in files if f.startswith("witness_")))))  # fmt: skip
            if attempt == "review-02-01" and crash["review"]:
                crash["review"] = False
                raise KeyboardInterrupt("simulated crash in a review")
            world = World(task.config.tools.db_path)
            issues = []
            if payload["phase"] == "world" and attempt in ("review-02-02", "final-01"):
                first, second = [m for (m,) in world.db.execute("SELECT id FROM messages ORDER BY id LIMIT 2")]
                # A mid-run review's note blocks nothing, and still opens the next day.
                # The final one's defect touches t0, so its fix may rewrite t0 and no other task.
                issues = [Issue(artifact="workspace", task_ids=[t0] if attempt == "final-01" else [], message_ids=[second], evidence_message_ids=[first], defect="stiff", requested_change="looser", blocking=attempt != "review-02-02")]  # fmt: skip
                if attempt == "final-01":  # a note beside the blocking defect, which the author hears too
                    issues.append(Issue(artifact="workspace", defect="a little formal", requested_change="loosen", blocking=False))  # fmt: skip
            criteria = dict.fromkeys(PHASE_CRITERIA.get(payload["phase"], ()), 1.0)
            # The probe's review finds t3 invalid; unchanged, it is judged again at the final review, on its runs.
            reviews = [TaskReview(task_id=t["id"], valid=not ((attempt.startswith("tasks-") and t["id"] == t3) or "vague" in t["question"]), reason="r", level_fit=3 if payload["phase"] == "task" else None) for t in payload["tasks"]]  # fmt: skip
            last = world.db.execute("SELECT MAX(id) FROM messages").fetchone()[0]
            world.close()
            issues += [Issue(artifact="tasks", task_ids=[r.task_id], message_ids=[last], defect="vague", requested_change="sharpen") for r in reviews if not r.valid]  # fmt: skip
            if attempt.startswith("tasks-") and payload["tasks"] and payload["tasks"][0]["id"] == t3:
                # A defect of the world, naming no task: it blocks none, and the final review hears it.
                issues.append(Issue(artifact="ledger", defect="a summary drifts", requested_change="restate it", blocking=False))  # fmt: skip
            verdict = Verdict(approved=not any(i.blocking for i in issues) and all(r.valid for r in reviews), tasks=reviews, issues=issues, criteria=criteria, summary="s")  # fmt: skip
            return await judged(task, verdict)

    forging = {"vague": False}  # a forge whose every candidate the judge rejects

    async def script(prompt, tools, runtime):
        """The author: sets up, plans, writes each day, writes tasks, hardens two of them, fixes what the review names."""
        if tools is None:
            return setup_document(runtime)
        call = lambda name, **kwargs: tools._with_state(getattr(tools, name))(**kwargs)  # noqa: E731
        world = World(tools.config.db_path)
        a, b, c, d = [r[0] for r in world.db.execute("SELECT id FROM users ORDER BY id")]
        ops, leads = channel_id("public", "ops", []), channel_id("private", "leads", [])
        assert runtime.files["/task/memory/now.md"].startswith(b"# Now: "), (
            "memory is rendered before every turn"
        )
        assert "/task/input.json" not in runtime.files, "the setup files are gone, after a resume too"
        assert "/task/memory/stale.md" not in runtime.files, (
            "memory is rendered whole: a page that is gone is gone"
        )
        runtime.files["/task/memory/stale.md"] = b"a page code no longer renders"
        seen.append(("turn", tools.config.mode, tools.config.day, prompt.split(":")[0][:40]))

        def talk(channel, *lines):
            lines = [PostLine(**line) for line in lines]
            return call("post", conversation=Conversation(channel_id=channel, about="work", lines=lines))

        async def close():
            while today(world) == tools.config.day:
                await call("advance", to=onward(world))

        if tools.config.mode == "plan":
            crash["plans"] += prompt.startswith("Before day 1")
            if crash["plans"] == 1:  # the first attempt never writes plan.md, and is rejected for it
                return
            assert "was rejected: " in prompt and "write /task/notes/plan.md" in prompt, "a retry is told why"
            events = [Event(id="e1", storyline="s2", title="audit window", day=3, time="10:00"), Event(id="e2", storyline="s1", title="service restored", day=2, time="14:00")]  # fmt: skip
            facts = [
                PlanFact(id="f1", storyline="s1", subject="Release 4.2", attribute="decision", value="rollback", anchor="rollback", channel_id=ops, author_id=a, day=1, summary="s"),
                PlanFact(id="f2", storyline="s1", subject="Release 4.2", attribute="state", value="service back", channel_id=ops, author_id=b, day=2, after=["f1"], event="e2", kind="happened", summary="s"),
                PlanFact(id="f3", storyline="s2", subject="Audit", attribute="window", value="dry run", anchor="dry run", channel_id=leads, author_id=a, day=3, event="e1", kind="scheduled", summary="s"),
                PlanFact(id="f4", storyline="s2", subject="Audit", attribute="owner", value="Owen", channel_id=ops, author_id=a, day=3, event="e1", kind="scheduled", summary="s"),
            ]  # fmt: skip
            board = [BoardEntry(slot=t, facts=["f3", "f4"]) for t in (t0, t1, t2)]  # the ledger slots
            await call("plan", ledger=Plan(storylines=[Storyline(id="s1", summary="the release"), Storyline(id="s2", summary="the audit")], events=events, facts=facts, board=board))  # fmt: skip
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
                assert "hedging a decoy where its readers see it voids the near-miss" in prompt, (
                    "a decoy's issue is answered in notes"
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
                await talk(ops, dict(author_id=a, text="Owen owns the audit at {at:e1}", conveys=["f4"], reactions=[Reaction(user_id=d, emoji="eyes")]), dict(author_id=d, text=f"thanks <@{a}>"))  # fmt: skip
            if day == 4:
                assert "message_ids" not in prompt, "a review's issues are delivered once"
            if day == 4 and "was rejected" not in prompt:
                # The first attempt closes the day but never writes its recap, so all 3 of its turns end rejected.
                if prompt.startswith("It is"):
                    await talk(ops, dict(author_id=d, text="quiet day"))
                    await close()
                    await runtime.write("/task/notes/scratch.md", b"day 4, first try")
                return
            if day == 4:
                assert "/task/notes/scratch.md" not in runtime.files, "a rejected attempt's notes are gone"
                assert "back at the start of the day: " in prompt and "recap.md" in prompt, (
                    "a retry is told why"
                )
                await talk(ops, dict(author_id=d, text="quiet day"))
            await close()
            await runtime.write("/task/notes/recap.md", f"day {day}: done".encode())
        elif tools.config.mode == "forge":
            # The proposer: the board and its notes come with the frozen world, which it reads and may not write.
            ids = re.findall(r"r\d\d-\d\d", prompt)
            assert (
                b"Board" in runtime.files["/task/memory/board.md"] and "/task/notes/plan.md" in runtime.files
            ), "the board and the world author's notes come to the proposer"
            assert b"## lookup" in runtime.files["/task/memory/taxonomy.md"] and b"- level 1: one message" in runtime.files["/task/memory/taxonomy.md"], "and the taxonomy"  # fmt: skip
            seen.append(("forge", tuple(ids), "Round" in prompt))
            if ids[0] == "r02-01" and not forging["vague"]:  # fmt: skip  # the archive: what the last round kept, and why it dropped the rest
                archive = runtime.files["/task/memory/archive.md"].decode()
                assert "holds 1 of the 30 tasks it aims for" in prompt and "r01-01" in archive and "kept in lookup·L2 as plain·structure" in archive and "dropped: the judge did not approve it" in archive and "lookup: L1 0/2 · L2 1/2" in archive, archive  # fmt: skip
            try:
                await call("revise", message_id=1, text="rewritten")
                raise AssertionError("the forge wrote the workspace")
            except ValueError as error:
                assert "forge turn" in str(error), error
            try:
                await call(
                    "add_task",
                    task=scripted_task(settings, {"id": "elsewhere", "category": "lookup", "level": 1}, a),
                )
                raise AssertionError("a candidate outside the round's open ids was written")
            except ValueError as error:
                assert "open ids" in str(error), error
            if ids[0] == "r01-01":
                first = world.db.execute("SELECT id FROM messages WHERE text LIKE '%rollback%' ORDER BY ts_us LIMIT 1").fetchone()[0]  # fmt: skip
                await call("annotate", fact=Annotation(id="n1", storyline="s1", subject="Release 4.2", attribute="call", value="roll it back", anchor="rollback", message_ids=[first], summary="s"))  # fmt: skip
                for bad, why in (({"anchor": "nowhere in it"}, "must contain"), ({"subject": "Elsewhere", "supersedes": "nothing"}, "which it supersedes, does not exist")):  # fmt: skip
                    try:
                        await call("annotate", fact=Annotation(id="n2", storyline="s1", subject="Release 4.2", attribute="call", value="other", anchor="rollback", message_ids=[first], summary="s").model_copy(update=bad))  # fmt: skip
                        raise AssertionError(f"an annotation was accepted: {bad}")
                    except ValueError as error:
                        assert why in str(error), error
                # A search for the question's own words reaches the answer first: level 1, whatever it is called.
                notes = Task(id=ids[0], category="search", level=2, actor_id=a, question="When were the notes up for release 4.2?", answer_type="text", gold_sql="SELECT value AS answer FROM facts WHERE id = 'f2'", facts=["f2"], intent="when")  # fmt: skip
                try:
                    await call("add_task", task=notes)
                    raise AssertionError("a shortcut was written above level 1")
                except ValueError as error:
                    assert "so it is a level-1 task" in str(error), error
                assert (await call("add_task", task=notes.model_copy(update={"level": 1})))["panel"][
                    "shortcut"
                ]
                # The proposer reads the world's inventory, and tries a candidate on a copy that writes nothing.
                inventory = await call("profile")
                assert inventory["world"]["messages"] and "material" in inventory, sorted(inventory)
                probed = await call("probe", task=notes.model_copy(update={"id": ids[1], "level": 1, "question": "When did the 4.2 notes go up?"}))
                assert probed["gold"] and "failure_cases" in probed and not world.db.execute("SELECT 1 FROM tasks WHERE id = ?", (ids[1],)).fetchone(), "a probe writes nothing"  # fmt: skip
                try:
                    await call("add_task", task=notes.model_copy(update={"level": 1, "intent": None}))
                    raise AssertionError("a candidate with no intent was written")
                except ValueError as error:
                    assert "name its intent" in str(error), error
                # A task on no facts above level 1 names the answer a hasty reader gives: another one.
                newest = Task(id=ids[0], category="lookup", level=2, actor_id=a, question="Who is the newest person here?", answer_type="text", gold_sql="SELECT real_name AS answer FROM users ORDER BY id DESC LIMIT 1", intent="structure")  # fmt: skip
                for near, why in (([], "near_sql"), ([newest.gold_sql], "returns its gold answer")):
                    try:
                        await call("add_task", task=newest.model_copy(update={"near_sql": near}))
                        raise AssertionError(f"a near-miss query was accepted: {near}")
                    except ValueError as error:
                        assert why in str(error), error
            for n, open_id in enumerate(ids):  # a candidate the judge approves, and a vague one it does not
                question = f"Who is the newest person here, round {open_id}?" + (" vague" if n or forging["vague"] else "")  # fmt: skip
                written = await call("add_task", task=Task(id=open_id, category="lookup", level=2, actor_id=a, question=question, answer_type="text", gold_sql="SELECT real_name AS answer FROM users ORDER BY id DESC LIMIT 1", near_sql=["SELECT real_name AS answer FROM users ORDER BY id LIMIT 1"], intent="structure"))  # fmt: skip
                assert set(written["features"]) == {"changes", "last_link", "rows", "perspective", "derived", "status"} and written["features"]["rows"] == 1, written  # fmt: skip
            if (
                ids[0] == "r02-01" and not forging["vague"]
            ):  # one GLM misses unless shown the evidence; the witness answers
                await call("add_task", task=Task(id=ids[0], category="search", level=1, actor_id=a, question="When were the notes up for release 4.2? it is hard to find", answer_type="text", gold_sql="SELECT value AS answer FROM facts WHERE id = 'f2'", facts=["f2"], intent="when"))  # fmt: skip
        elif prompt.startswith("The last day is closed"):
            context = json.loads(tools.config.context)
            writable, slots = context["writable"], {s["id"]: s for s in context["slots"]}
            seen.append(("tasks", tuple(writable), tuple(t for t in slots if t in prompt)))
            if writable == [t0, t1]:
                try:
                    await call("add_task", task=scripted_task(settings, slots[t2], a))
                    raise AssertionError("a slot of another batch was written")
                except ValueError as error:
                    assert "one of the slots you write now" in str(error), error
                await runtime.write("/task/notes/recap.md", b"tasks: the first batch")
            else:
                assert runtime.files["/task/notes/recap.md"] == b"tasks: the first batch", (
                    "the last batch's notes"
                )
            for t in writable:
                if t in prompt:
                    await call("add_task", task=scripted_task(settings, slots[t], a))
                if t == t2 and crash["tasks"]:  # the second batch is cut off after its first task
                    crash["tasks"] = False
                    raise KeyboardInterrupt("simulated crash in a batch")
        elif prompt.startswith("These tasks come back"):
            back = json.loads(prompt[prompt.index("{") : prompt.index("\nRewrite")])
            assert all(r["level_fit"] == 3 and "evidence_pages" in r and r["concept"] and r["band"] and "right_rate" in r and "strict_rate" in r for r in back.values()), (
                "hardening reads the judge's level fit, code's measures, both rates and the level's band"
            )  # fmt: skip
            assert all(r["question"] and r["gold_sql"] and "gold" in r and r["asked_by"] and r["tries_left"] == 0 and "found_by" in r["solves"][0] and "decoy_reads" in r["solves"][0] for r in back.values()), (
                "a returned task carries what it asks and answers, who asks it, and the route to beat"
            )  # fmt: skip
            assert all(len(json.dumps(r)) < 6000 for r in back.values()), "a returned task is small"
            seen.append(("harden", sorted(back), sorted(r["move"] for r in back.values())))
            seen.append(("pinged", runtime.pings))  # by now the probe ran, with the VM kept awake
            for (
                t,
                r,
            ) in back.items():  # each try as the author reads it: the route, and where the evidence showed
                assert r["solves"] and all(s["steps"][0].startswith('search_messages {"query": "release"} → 0 items') for s in r["solves"]), r["solves"]  # fmt: skip
                assert {s["evidence_at"] for s in r["solves"]} == ({2} if t == t1 else {None}), (
                    t,
                    r["solves"],
                )  # t3 is a join
                assert all(f"/task/memory/solves/{t}/{k}.json" in runtime.files for k in range(1, 4)), "the whole tries"  # fmt: skip
            fixes = sum(1 for x in seen if x[0] == "harden" and t3 in x[1])
            for slot in json.loads(tools.config.context)["slots"]:
                task = scripted_task(settings, slot, a)
                asked = task.model_copy(update={"question": f"Asked again ({len(seen)}): " + task.question})
                if slot["id"] == t3 and slot["id"] in back:
                    # Its first fix revises the message the review names, not the task: it is tried again all the
                    # same. Its second rewrites it with another answer, which a fix may change.
                    if fixes == 1:
                        last = world.db.execute("SELECT MAX(id) FROM messages").fetchone()[0]
                        await call("revise", message_id=last, text="quiet day, nothing to report")
                    else:
                        await call("add_task", task=asked.model_copy(update={"gold_sql": "SELECT COUNT(*) AS answer FROM messages WHERE parent_id IS NULL"}))  # fmt: skip
                elif slot["id"] in back:  # a task that came back is rewritten, so tried again
                    if back[slot["id"]]["move"] == "harder":  # keeping its answer: another answer is refused
                        swapped = asked.model_copy(update={"gold_sql": "SELECT value AS answer FROM facts WHERE id = 'f3'", "facts": ["f3"]})  # fmt: skip
                        try:
                            await call("add_task", task=swapped)
                            raise AssertionError("a harder rewrite changed its answer")
                        except ValueError as error:
                            assert "keeps its answer" in str(error), error
                            seen.append(("answer kept", slot["id"]))
                    await call("add_task", task=asked)
                elif (
                    slot["id"] in json.loads(tools.config.context)["writable"]
                ):  # in this batch, but in its band
                    try:
                        await call("add_task", task=asked)
                        raise AssertionError("a task that did not come back was rewritten")
                    except ValueError as error:
                        assert "one of the slots you write now" in str(error), error
                        seen.append(("kept", slot["id"]))
        elif prompt.startswith("The review rejected"):
            assert "hedging a decoy where its readers see it voids the near-miss" in prompt
            issues = json.loads(prompt[prompt.index("[") : prompt.index("]\n") + 1])
            seen.append(("fix issues", [i.get("blocking") for i in issues]))
            for issue in issues:
                for message in issue.get("message_ids", []):
                    await call("revise", message_id=message, text="ok, the notes come tomorrow")
            seen.append(("fixed", [m for i in issues for m in i.get("message_ids", [])]))
            slots = json.loads(tools.config.context)["slots"]
            task = scripted_task(settings, slots[0], a)  # t0, which the verdict names, rewritten in its slot
            await call("add_task", task=task.model_copy(update={"question": "Asked anew: " + task.question}))
            other = scripted_task(settings, slots[1], a)
            try:
                await call(
                    "add_task", task=other.model_copy(update={"question": "Asked anew: " + other.question})
                )
                raise AssertionError("a fix rewrote a task the verdict does not name")
            except ValueError as error:
                assert "one of the slots you write now: ['" + t0 + "']" in str(error), error

    agents = SimpleNamespace(judge=FlowJudge(), solver=ScriptedSolver(), witness=ScriptedWitness(), author=ScriptedAuthor(script))  # fmt: skip
    manifest = provenance(settings)
    store = TestStore(settings.output, manifest)
    env = GenerationEnv(settings, store)
    await env.setup(agents)
    asked = [{"role": "user", "content": "write today"}]
    append_user_notice(asked)
    assert "call it again" in asked[0]["content"] and "do not retry" not in asked[0]["content"], (
        "a dropped tool call is called again, not given up"
    )
    for _ in range(2):  # the review after day 2 is cut off, then day 3
        try:
            await GenerationEnv(settings, store).run(None, agents)
            raise AssertionError("the simulated crash did not happen")
        except KeyboardInterrupt:
            pass
        if store.state.phase == "review":
            assert store.state.day == 2 and store.state.rounds["day-02"] == 1, (
                "a cut-off review is redone, not its day"
            )
            store.close()
            store = TestStore(settings.output, manifest)
    after_day_2 = World(store.root / "attempts" / store.state.restore_point / "world.sqlite").db.execute("SELECT COUNT(*) FROM messages").fetchone()[0]  # fmt: skip
    assert store.world.db.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == after_day_2 + 1, "the crash left a partial day"  # fmt: skip
    store.close()
    fails(
        Store, settings.output, manifest | {"lock_hash": "changed"}
    )  # a run resumes only with its own config
    store = TestStore(settings.output, manifest)
    assert store.state.phase == "day" and store.state.day == 3 and store.state.restore_point == "day-02-01"
    try:
        await GenerationEnv(settings, store).run(None, agents)
        raise AssertionError("the simulated crash did not happen")
    except KeyboardInterrupt:
        pass
    store.close()
    store = TestStore(settings.output, manifest)
    assert store.state.phase == "tasks" and store.state.batch == 1, "a batch cut off is written again"
    env = GenerationEnv(settings, store)
    await env.run(None, agents)
    state = store.state
    assert state.phase == "done" and state.rounds["day-03"] == 2 and state.rounds["review-02"] == 2, (
        state.rounds
    )
    assert state.rounds["plan"] == 2 and not state.feedback
    assert not store.world.db.execute("SELECT 1 FROM messages WHERE text = 'morning'").fetchone(), (
        "the crashed day's start is gone"
    )
    assert [t[3][:6] for t in [s for s in seen if s[0] == "turn"] if t[1] == "day" and t[2] == 4] == [
        "It is ",
        "Day 4 ",
        "Day 4 ",
        "It is ",
    ], "a day needs its recap, and one that never gets it is written again"
    assert state.rounds["day-04"] == 2, state.rounds
    turns = [s for s in seen if s[0] == "turn"]
    assert [t[3][:6] for t in turns if t[1] == "day" and t[2] == 1] == ["It is ", "Day 1 "], "an open day is not accepted"  # fmt: skip
    first, second = [m for (m,) in store.world.db.execute("SELECT id FROM messages ORDER BY id LIMIT 2")]
    assert ("revised", [second]) in seen and ("fixed", [second]) in seen, (
        "only the messages to change are revised"
    )
    assert store.world.db.execute("SELECT text FROM messages WHERE id = ?", (second,)).fetchone()[0] == "ok, the notes come tomorrow"  # fmt: skip
    assert "rollback" in store.world.db.execute("SELECT text FROM messages WHERE id = ?", (first,)).fetchone()[0], "the evidence stays"  # fmt: skip
    # Too easy, t1 comes back for its one round; t3, which the judge does not approve, comes back after it too. The
    # tasks in their bands never come back. A round counts what changed: t3's revised message, then t3 itself.
    assert [s for s in seen if s[0] == "harden"] == [("harden", [t1], ["harder"]), ("harden", [t3], ["fix"]), ("harden", [t3], ["fix"])]  # fmt: skip
    assert state.task_rounds == {t1: 1, t3: 2}, state.task_rounds
    assert [s for s in seen if s[0] == "answer kept"] == [("answer kept", t1)], (
        "a harder t1 keeps its answer; t3's fix need not"
    )
    # A hardening turn rewrites only what came back: t0 (in its band) is refused in t1's turn, t2 in t3's two turns.
    assert [s for s in seen if s[0] == "kept"] == [("kept", t0), ("kept", t2), ("kept", t2)], [s for s in seen if s[0] == "kept"]  # fmt: skip
    # Two batches, each in its own interaction; the second, cut off after t2, is written again: only t3 is asked for.
    assert [s for s in seen if s[0] == "tasks"] == [("tasks", (t0, t1), (t0, t1)), ("tasks", (t2, t3), (t2, t3)), ("tasks", (t2, t3), (t3,))]  # fmt: skip
    assert state.rounds["tasks"] == 3, state.rounds
    # Each probe tries what changed; the final reviews solve only what changed since: t0, which a fix rewrote.
    assert solved == {("tasks-01", t0): 4, ("tasks-01", t1): 6, ("tasks-03", t2): 4, ("tasks-03", t3): 9, ("final-02", t0): 3}, solved  # fmt: skip
    assert set(state.solves) == {t0, t1, t2, t3}, "a rewritten task's runs are its new ones"
    assert ("review", "review-02-02", "world", True, True, False, ()) in seen
    assert ("review", "final-01", "world", False, True, True, ()) in seen, "the world issue a probe raised"
    assert state.open_issues == [], "the final review has heard them"
    assert ("review", "final-02", "world", False, True, True, ()) in seen, (
        "the final review checks the last issues"
    )
    assert ("review", "final-02", "task", False, False, True, (t0,)) in seen, (
        "only the changed task is reviewed again, with the issue that named it"
    )
    probed = {a: [s[6] for s in seen if s[:3] == ("review", a, "task")] for a in ("tasks-01", "tasks-03")}
    assert probed == {"tasks-01": [(t0,), (t1,), (t1,)], "tasks-03": [(t2,), (t3,), (t3,), (t3,)]}, (
        "the judge reviews each probe's tasks, one chunk at a time; each passes on its own"
    )  # fmt: skip
    assert [s[6] for s in seen if s[:3] == ("review", "final-01", "task")] == [(t3,)], (
        "the final review judges only what changed or failed since its probe's review"
    )
    assert ("review", "final-01", "task", False, False, True, (t3,)) in seen, (
        "with the issues its probe raised"
    )
    assert ("fix issues", [True, False]) in seen, (
        "a fix hears the review's notes as well as its blocking issues"
    )
    assert {e["attempt"] for e in map(json.loads, (store.root / "progress.jsonl").read_text().splitlines()) if e["event"] == "candidate_finished" and e["approved"]} >= {"tasks-01", "tasks-03"}, "each batch's attempt is closed"  # fmt: skip
    assert (store.root / "attempts" / "day-03-02" / "notes" / "recap.md").read_text() == "day 3: done"
    days = [json.loads(line) for line in (store.root / "progress.jsonl").read_text().splitlines()]
    assert [e["day"] for e in days if e["event"] == "day_closed"] == [1, 2, 3, 4]
    assert any(e["event"] == "interrupted" for e in days)
    assert store.world.violations(complete=True) == [] and set(state.task_reviews) == {t for (t,) in store.world.db.execute("SELECT id FROM tasks")}  # fmt: skip
    ids = [r[0] for r in store.world.db.execute("SELECT id FROM messages ORDER BY ts_us")]
    assert ids == sorted(ids), "the world was written in time order"
    summary = store.summary("complete")
    assert summary["rates"][t2] == {"right_rate": 0.5, "strict_rate": 0.0, "coverage": None, "tries": 2, "crashed": 1, "witness_right": 1.0, "level_fit": 3, "level": 3, "band": [0.0, 0.5], "in_band": True, "rounds": 0, "kept": False, "unanswered": 0}, (
        "difficulty is the right-answer rate, the strict rate reported beside it; a try that crashed twice is counted, "
        "and left out of the rates; the band it landed in"
    )  # fmt: skip
    # t1 (level 2) stayed too easy; t0 was rewritten at the fix, to 2 of 3 right; t3 is in its level-1 band
    assert summary["in_band"] == {"1": [1, 1], "2": [1, 2], "3": [1, 1]}, summary["in_band"]
    assert [t for t, r in summary["rates"].items() if r["kept"]] == [t1], (
        "t1 stayed too easy with no rewrite left"
    )
    assert summary["crashed_solves"] == 1 and summary["witness_tries"] == 2, summary
    assert abs(summary["mean_learnability"] - 17 / 36) < 1e-9, summary["mean_learnability"]  # t0 2/3, t2 1/2

    # The witness tries only the task whose band starts at 0, twice; the judge reads its tries; they count in no rate.
    assert witnessed == {("tasks-03", t2): 2}, witnessed
    assert ("witnessed", "tasks-03", (t2,), 2, (f"witness_{t2}_1.json", f"witness_{t2}_2.json")) in seen
    assert state.probes[t2]["witness_right"] == 1.0 and state.probes[t0]["witness_right"] is None
    assert (await env.witness(agents, [t2], {t2: env.task_key(t2)}, "final-02"))[t2]["results"] and witnessed == {("tasks-03", t2): 2}, "an unchanged task keeps its witness"  # fmt: skip
    probe = {"approved": True, "band": (0.0, 0.25), "right_rate": 0.0}
    assert [band_move(probe | {"witness_right": w}) for w in (0.0, 0.5, None)] == ["easier", None, None], (
        "a task whose band starts at 0 lands in it only if the witness answers it"
    )
    assert (
        band_move(probe | {"approved": False}) == "fix" and band_move(probe | {"right_rate": 0.5}) == "harder"
    )

    # The author's VM is pinged while only the env works, and no more once the block ends.
    assert next(s[1] for s in seen if s[0] == "pinged") > 0, "the VM is kept awake through the probes"
    awake = FakeRuntime()
    async with keep_awake(awake, 0.01):
        await asyncio.sleep(0.05)
    pinged = awake.pings
    await asyncio.sleep(0.03)
    assert pinged >= 2 and awake.pings == pinged, (pinged, awake.pings)

    class Slow(ScriptedAgent):  # GLM's gate holds its solves to [author] solvers, whatever the episode allows
        active, peak = 0, 0

        async def run(self, task):
            Slow.active += 1
            Slow.peak = max(Slow.peak, Slow.active)
            await asyncio.sleep(0.01)
            Slow.active -= 1
            evaluation = {"task_id": task.data.task_id, "execution_ok": True, "semantic_correctness": 1.0, "correct": True, "grounded": True, "calls": 1, "reason": "r", "response": "an answer"}  # fmt: skip
            return FakeTrace("slow", "solver", info={"evaluation": evaluation}, task=task)

    await env.solves(SimpleNamespace(solver=Slow()), [t0, t1], "final-02", 3)
    assert Slow.peak == settings.author.solvers == 2, Slow.peak

    class Crashing(ScriptedAgent):
        async def run(self, task):
            return FakeTrace("crash", "solver", ok=False, info={"grading_started": True}, errors=[SimpleNamespace(message="judge timeout")], task=task)  # fmt: skip

    try:
        await env.solves(SimpleNamespace(solver=Crashing()), [t1], "final-02", 2)
        raise AssertionError("a task whose every try crashed was kept")
    except RuntimeError as error:
        assert "crashed twice" in str(error), error
    difficulty = summary["difficulty"]
    assert set(difficulty) == set(state.task_reviews) and all(
        d["level_fit"] == 3 and "evidence_pages" in d and d["concept"] for d in difficulty.values()
    ), "the summary reports each task's measured difficulty beside its level"
    some = sorted(state.task_reviews)[0]
    review = state.task_reviews.pop(some)
    fails(store.publish)  # a task nobody reviewed
    state.task_reviews[some] = review
    store.world.insert("facts", [dict(id="f9", storyline="s2", subject="x", attribute="y", value="z", channel_id=channel_id("public", "ops", []), author_id=store.world.db.execute("SELECT MIN(id) FROM users").fetchone()[0], day=3, summary="s")])  # fmt: skip
    store.world.insert("task_facts", [dict(task_id=some, fact_id="f9")])
    fails(store.publish)  # a task resting on a fact no message states
    store.world.db.execute("DELETE FROM task_facts WHERE fact_id = 'f9'")
    store.world.db.execute("DELETE FROM facts WHERE id = 'f9'")
    # A try that finished without an answer, an empty last message or the turn cap mid-search, is wrong and counted.
    graded = {"task_id": t1, "correct": False, "semantic_correctness": 0.0, "response": "an answer"}
    ends = [("an answer", "agent_completed"), ("", "agent_completed"), ("let me look further", "max_turns")]
    silent = [SolverTask.outcome(SimpleNamespace(ok=True, info={"evaluation": graded | {"response": r}}, stop_condition=stop))["unanswered"] for r, stop in ends]  # fmt: skip
    assert silent == [False, True, True] and rates([{"correct": False, "semantic_correctness": 0.0, "unanswered": u} for u in silent])["unanswered"] == 2  # fmt: skip
    # A try as the author reads it: the call that first showed the evidence, how many showed a decoy, the route up to
    # the evidence and then its last three calls, and a short answer.
    read = lambda tool, ts=None: {"tool": tool, "arguments": {}, "output": {"items": [{"channel": "C1", "ts": ts}] if ts else []}}  # noqa: E731  # fmt: skip
    route = [
        read("read_channel", "2"),
        read("read_thread", "1"),
        *(read("search_messages") for _ in range(6)),
    ]
    digest_ = try_digest({"correct": True, "response": "x" * 500}, {"info": {"observations": route}}, {("C1", "1")}, {("C1", "2")})  # fmt: skip
    assert (digest_["evidence_at"], digest_["decoy_reads"], digest_["calls"], len(digest_["answer"])) == (2, 1, 8, 300), digest_  # fmt: skip
    assert digest_["found_by"].startswith("read_thread") and digest_["steps"][2] == "… 3 more calls …" and len(digest_["steps"]) == 6, digest_  # fmt: skip
    # A hardening turn that changes nothing ends the loop, rounds to spare or not: the same tries would come back.
    idle_turns = []

    async def idle(prompt, extra=None):
        idle_turns.append(prompt)

    state.task_rounds[t1] = 0  # t1, too easy, has its round again
    await env.harden(agents, idle, "final-02", [t1])
    assert len(idle_turns) == 1 and state.task_rounds[t1] == 0, (len(idle_turns), state.task_rounds)
    again = json.loads(idle_turns[0][idle_turns[0].index("{") : idle_turns[0].index("\nRewrite")])[t1]
    assert [h["right_rate"] for h in again["history"]] == [1.0] and not again["history"][0]["question"].startswith("Asked again"), (
        "a task comes back with its earlier versions and how often each was answered"
    )  # fmt: skip
    state.task_rounds[t1] = 1
    # A task's key covers the decoys on its facts' subjects: a revised decoy is a changed task, tried again.
    resting = next(
        t for (t,) in store.world.db.execute("SELECT task_id FROM task_facts WHERE fact_id = 'f4'")
    )
    keyed, lure = env.task_key(resting), store.world.db.execute("SELECT MAX(id) FROM messages").fetchone()[0]
    store.world.insert("facts", [dict(id="f8", storyline="s2", subject="Audit", attribute="owner", value="Mia", is_decoy=1, channel_id=channel_id("public", "ops", []), author_id=store.world.db.execute("SELECT MIN(id) FROM users").fetchone()[0], day=3, summary="s")])  # fmt: skip
    store.world.db.execute(
        "INSERT INTO evidence (fact_id, message_id, role) VALUES ('f8', ?, 'supporting')", (lure,)
    )
    decoyed = env.task_key(resting)
    store.world.db.execute("UPDATE messages SET text = 'Mia owns it now, I think' WHERE id = ?", (lure,))
    assert len({keyed, decoyed, env.task_key(resting)}) == 3, "the key follows the task's decoys"
    store.world.db.execute("DELETE FROM evidence WHERE fact_id = 'f8'")
    store.world.db.execute("DELETE FROM facts WHERE id = 'f8'")
    store.world.db.execute("UPDATE messages SET text = 'quiet day, nothing to report' WHERE id = ?", (lure,))
    # A refusal resting on the truth out of its actor's sight ships that truth apart, as unseen.
    leads = channel_id("private", "leads", [])
    (outsider,) = store.world.db.execute("SELECT id FROM users WHERE id NOT IN (SELECT user_id FROM members WHERE channel_id = ?) ORDER BY id LIMIT 1", (leads,)).fetchone()  # fmt: skip
    (asker,) = store.world.db.execute("SELECT actor_id FROM tasks WHERE id = ?", (some,)).fetchone()
    added = store.world.db.execute("INSERT OR IGNORE INTO task_facts (task_id, fact_id) VALUES (?, 'f3')", (some,)).rowcount  # fmt: skip
    store.world.db.execute("UPDATE tasks SET actor_id = ? WHERE id = ?", (outsider, some))
    truth = [list(r) for r in store.world.db.execute("SELECT m.channel_id, m.ts FROM evidence e JOIN messages m ON m.id = e.message_id WHERE e.fact_id = 'f3' AND e.role = 'anchor'")]  # fmt: skip
    private = store.release_rows("h")[1][some]
    assert truth and private.unseen == truth and not [m for m in private.messages if m in truth], private
    store.world.db.execute("UPDATE tasks SET actor_id = ? WHERE id = ?", (asker, some))
    if added:
        store.world.db.execute("DELETE FROM task_facts WHERE task_id = ? AND fact_id = 'f3'", (some,))
    store.publish()
    world, rows, answers = load_release(store.root / "release")
    assert len(rows) == 4 and World(world).db.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    assert json.loads((store.root / "release" / "manifest.json").read_text())["format"] == "worldgen-slack.v10"
    assert all((r.surface, r.turns, type(r.miss_classes)) == (surface.VERSION, 40, dict) for r in rows), "each task's surface, turns and misses"  # fmt: skip
    assert all((r.right_rate, r.strict_rate, r.tries, r.solver) == (state.task_reviews[r.task_id]["right_rate"], state.task_reviews[r.task_id]["strict_rate"], state.task_reviews[r.task_id]["tries"], settings.env.solver.model) for r in rows), "each task's rates are published"  # fmt: skip
    assert {r.task_id: (r.witness, r.witness_right) for r in rows if r.witness} == {
        t2: (settings.env.witness.model, 1.0)
    }
    assert all(a.rows or a.answer_type == "refusal" for a in answers.values())
    ledger = [r for r in rows if settings.taxonomy[r.category].gold == "ledger"]
    assert ledger and all(answers[r.task_id].rows == [{"answer": "Owen"}] and answers[r.task_id].messages for r in ledger)  # fmt: skip
    # Its evidence is what states its answer, not a value that answer supersedes: that one is stale.
    stating = lambda fact: {tuple(r) for r in store.world.db.execute("SELECT m.channel_id, m.ts FROM evidence e JOIN messages m ON m.id = e.message_id WHERE e.fact_id = ?", (fact,))}  # noqa: E731  # fmt: skip
    assert all({tuple(m) for m in answers[r.task_id].messages} == stating("f3") | stating("f4") for r in ledger), "two current values"  # fmt: skip
    store.world.db.execute("INSERT INTO fact_relations (src_fact, dst_fact, kind) VALUES ('f4', 'f3', 'supersedes')")
    released = store.release_rows("h")[1]
    store.world.db.execute("DELETE FROM fact_relations WHERE src_fact = 'f4' AND dst_fact = 'f3'")
    assert stating("f3") and all({tuple(m) for m in released[r.task_id].messages} == stating("f4") for r in ledger), "a stale value is no evidence"  # fmt: skip
    store.world.db.execute("UPDATE facts SET is_decoy = 1 WHERE id = 'f3'")
    released = store.release_rows("h")[1]
    store.world.db.execute("UPDATE facts SET is_decoy = 0 WHERE id = 'f3'")
    assert all({tuple(m) for m in released[r.task_id].messages} == stating("f4") for r in ledger), "a decoy is no evidence"  # fmt: skip
    # An answer fact restated where its asker cannot read is not evidence they can be shown.
    (asker,) = store.world.db.execute("SELECT actor_id FROM tasks WHERE id = ?", (ledger[0].task_id,)).fetchone()
    hidden = store.world.db.execute("SELECT id, channel_id, ts FROM messages WHERE channel_id = ? ORDER BY id LIMIT 1", (leads,)).fetchone()  # fmt: skip
    store.world.db.execute("UPDATE tasks SET actor_id = ? WHERE id = ?", (outsider, ledger[0].task_id))
    store.world.db.execute("INSERT INTO evidence (fact_id, message_id, role) VALUES ('f4', ?, 'supporting')", (hidden[0],))  # fmt: skip
    released = store.release_rows("h")[1]
    store.world.db.execute("DELETE FROM evidence WHERE fact_id = 'f4' AND message_id = ?", (hidden[0],))
    store.world.db.execute("UPDATE tasks SET actor_id = ? WHERE id = ?", (asker, ledger[0].task_id))
    assert released[ledger[0].task_id].messages and [hidden[1], hidden[2]] not in released[ledger[0].task_id].messages, "out of sight, no evidence"  # fmt: skip
    store.publish()  # idempotent
    assert (store.root / "world-calls.jsonl").exists()
    assert run_label(Path("/x/data/v7-01/software")) == "worldgen-v7-01-software", (
        "a run's sandboxes are findable"
    )
    # A solver samples as a policy in training does, so its tries differ; a rate comes with its 90% interval, wide
    # at a few tries.
    assert settings.env.solver.sampling.temperature == 1.0 == settings.env.witness.sampling.temperature and settings.env.judge.sampling.temperature == 0.0  # fmt: skip
    assert rates([{"correct": i < 6, "semantic_correctness": 0.0} for i in range(8)])["right_interval"] == [0.46, 0.913]  # fmt: skip
    assert rates([{"correct": True, "semantic_correctness": 1.0}] * 4)["right_interval"] == [0.596, 1.0], "4 of 4 still allows 0.6"  # fmt: skip
    graded = [{"correct": True, "grounded": True, "semantic_correctness": 1.0}, {"correct": False, "grounded": False, "semantic_correctness": -1.0}, {"correct": False, "abstained": True, "grounded": False, "semantic_correctness": 0.0}, {"correct": True, "grounded": False, "semantic_correctness": 0.0}]  # fmt: skip
    assert (rates(graded)["strict_rate"], rates(graded)["abstain_rate"]) == (0.25, 0.25), "strict is the share right and grounded, not the mean reward"  # fmt: skip
    assert all(r.right_interval == state.task_reviews[r.task_id]["right_interval"] for r in rows), "intervals are published"  # fmt: skip
    # A task ships with the wrong answers its grade must reject: one the answer judge accepts makes it ambiguous as
    # asked, and it is not approved, whatever its tries.
    assert store.state.screens and all(not s["failed"] for s in store.state.screens.values()), "every task was screened"  # fmt: skip
    gold, cases = failure_cases(store.world, t1, json.loads(store.world.db.execute("SELECT gold_json FROM tasks WHERE id = ?", (t1,)).fetchone()[0]))  # fmt: skip
    assert gold.endswith(".") and ("a bare opener", "Thought process:") in cases and any(n == "declining to answer" for n, _ in cases), cases  # fmt: skip
    lax = GenerationEnv(settings, store)
    GRADER["loose"], store.state.active_attempt = True, "final-02"
    _, verdict = await lax.judge_tasks(agents, [t1], {t1: "screened loose"}, "final-02", 3)
    assert not verdict.approved and any("accepts a wrong answer to it: a bare opener" in i.defect for i in store.state.task_issues[t1]), verdict  # fmt: skip
    assert store.state.screens[t1]["failed"] == ["a bare opener ('Thought process:')"]
    GRADER["loose"], GRADER["blind"] = False, True
    assert (await lax.screen([t1], {t1: "screened blind"}, "final-02"))[t1] == [
        "its gold answer, judged wrong"
    ]
    GRADER["blind"] = False
    # A session lost to its sandbox is none of the author's limits (S3 stopped three times on Prime's dropped process
    # stream, each reported as the author's limits): its block is written again once and spends no attempt, and a
    # second loss in a row stops the run.
    rpc = SimpleNamespace(type="HarnessError", message="harness 'rlm': APIError: process stream RPC failed (unavailable)")  # fmt: skip
    budget = SimpleNamespace(type="HarnessError", message="harness 'rlm': RuntimeError: ACP agent completed without committing a model turn: [token budget reached]")  # fmt: skip
    assert isinstance(ended(SimpleNamespace(errors=[rpc])), SessionLost)
    assert all(isinstance(ended(t), ReviewLimit) for t in (SimpleNamespace(errors=[budget]), SimpleNamespace(errors=[rpc], is_truncated=True), SimpleNamespace(errors=[]))), "the author's own limits"  # fmt: skip

    async def cut_off(prompt):
        return SimpleNamespace(terminated=True)

    try:
        await env.author_step(SimpleNamespace(trace=FakeTrace("cut-off", "author", ok=False, errors=[rpc]), turn=cut_off), FakeRuntime(), "write", "final-02", "tasks")  # fmt: skip
        raise AssertionError("a dropped session was reported as the author's limits")
    except SessionLost:
        pass
    blocks = []

    class Dropping(GenerationEnv):  # the run's blocks, of which the first `losses` lose their session
        def __init__(self, key, losses):
            super().__init__(settings, store)
            self.key, self.losses = key, losses

        async def blocks(self, agents_, runtime, setup):
            blocks.append(store.reserve(self.key, 1))
            if len([b for b in blocks if b.startswith(self.key)]) <= self.losses:
                raise ended(SimpleNamespace(errors=[rpc]))
            store.finish_attempt(True)

    stopping = verifiers.v1.utils.interrupt
    stopping._cleaning_up = True  # a stop cancels the session's streams: it is not retried
    try:
        await Dropping("stop", 1).author_world(agents)
        raise AssertionError("a stopped run was taken up again")
    except SessionLost:
        assert blocks == ["stop-01"] and "stop" not in store.state.lost, blocks
    finally:
        stopping._cleaning_up = False
    blocks.clear()
    await Dropping("block", 1).author_world(agents)
    assert blocks == ["block-01", "block-02"] and store.state.lost == {"block": 1}, (blocks, store.state.lost)
    try:
        await Dropping("gone", 2).author_world(agents)
        raise AssertionError("a block that lost its session twice went on")
    except SessionLost:
        assert blocks[2:] == ["gone-01", "gone-02"] and store.state.lost["gone"] == 2, blocks
    # The world the forge, the measure and the chapter start from: one with no tasks of its own.
    world_dir, world_settings = await world_run(root, settings, lambda **seats: SimpleNamespace(**(vars(agents) | seats)))  # fmt: skip
    source_world = World(world_dir / "world.sqlite")
    # The forge: tasks proposed on the finished world, which stays frozen; what the judge approves is kept.
    plan = ForgeConfig(base=Path("worldgen.toml"), world=world_dir, output=root / "forge", rounds=2, candidates=2, tasks=30)  # fmt: skip
    workspace = source_world.db.execute(
        "SELECT group_concat(id || ':' || text, '|') FROM messages"
    ).fetchone()[0]
    forged = TestStore(plan.output, forge_provenance(plan, settings))

    Forging = ForgeEnv

    store.state.active_attempt = "forge"  # the scripted solver names its tries by the run it reads
    await Forging(plan, settings, forged).run(None, agents)
    assert [x[1] for x in seen if x[0] == "forge"] == [("r01-01", "r01-02"), ("r02-01", "r02-02")]
    assert forged.state.phase == "done" and sorted(t for (t,) in forged.world.db.execute("SELECT id FROM tasks")) == ["r01-01", "r02-01"], "the vague candidates are dropped"  # fmt: skip
    hard = forged.state.forged["r02-01"]
    assert (hard["right"], hard["witness"], hard["hint"], hard["level"], hard["verdict"]) == (0.0, 1.0, 1.0, 3, "kept in search·L3 as k1·when"), hard  # fmt: skip
    assert dict(forged.world.db.execute("SELECT id, level FROM tasks").fetchall()) == {"r01-01": 2, "r02-01": 3}, "its tries give its level"  # fmt: skip
    assert forged.world.db.execute("SELECT group_concat(id || ':' || text, '|') FROM messages").fetchone()[0] == workspace, "the workspace is frozen"  # fmt: skip
    assert tuple(forged.world.db.execute("SELECT channel_id, author_id FROM facts WHERE id = 'n1'").fetchone()) == tuple(source_world.db.execute("SELECT channel_id, user_id FROM messages WHERE text LIKE '%rollback%' ORDER BY ts_us LIMIT 1").fetchone()), "an annotation is placed by its first message"  # fmt: skip
    source_world.close()
    forged.publish()
    # The forge's output, as a measure starts from it: a finished run with tasks of its own.
    source = root / "measure-source"
    source.mkdir()
    forged.world.snapshot(source / "world.sqlite")
    (source / "state.json").write_bytes((forged.root / "state.json").read_bytes())
    assert {r.task_id for r in load_release(plan.output / "release")[1]} == {"r01-01", "r02-01"}
    _, cases = failure_cases(forged.world, "r01-01", json.loads(forged.world.db.execute("SELECT gold_json FROM tasks WHERE id = 'r01-01'").fetchone()[0]))  # fmt: skip
    assert any(name.startswith("the near-miss query's answer") for name, _ in cases), cases
    # A cell keeps the candidate GLM's tries are most mixed on; the lowest band also needs the witness to answer it.
    assert [measured_level(r, settings.tasks.bands) for r in (1.0, 0.75, 0.6, 0.5, 0.25, 0.0)] == [
        1,
        1,
        2,
        2,
        3,
        3,
    ]
    assert bucket({"changes": 3, "last_link": "dm·later", "rows": 2, "perspective": True, "derived": False, "status": False}) == "k3+dm+set+perspective"  # fmt: skip
    # A place holds two tasks. A newcomer to a full one replaces a holder if GLM's tries are more mixed on it, or if
    # it brings a structure the place lacks while two holders share one; else it is dropped.
    weigher, actor = (
        Forging(plan, settings, forged),
        forged.world.db.execute("SELECT MIN(id) FROM users").fetchone()[0],
    )
    newest = Task(id="r09-01", category="lookup", level=2, actor_id=actor, question="Who is the newest person here?", answer_type="text", gold_sql="SELECT real_name AS answer FROM users ORDER BY id DESC LIMIT 1", near_sql=["SELECT real_name AS answer FROM users ORDER BY id LIMIT 1"], intent="structure")  # fmt: skip
    newest_two = newest.model_copy(update={"answer_type": "set", "gold_sql": "SELECT real_name AS answer FROM users ORDER BY id DESC LIMIT 2", "near_sql": ["SELECT real_name AS answer FROM users ORDER BY id LIMIT 2"]})  # fmt: skip
    for task_, open_id, question, right, witness, verdict in (
        (newest, "r09-01", "Who is the newest person here, r09-01?", 0.0, 0.0, "dropped: the witness does not answer it either: hard, or broken"),
        (newest, "r09-01", "Who is the newest person here, r09-01?", 0.5, None, "kept in lookup·L2 as plain·structure"),  # the place had room
        (newest, "r09-02", "Who is the newest person here, r09-02?", 0.625, None, "kept in lookup·L2 as plain·structure"),  # more mixed than r01-01
        (newest, "r09-03", "Who is the newest person here, r09-03, by when they joined?", 0.625, None, "dropped: lookup·L2 is full (r09-01, r09-02), with tasks as mixed or more and its structure"),  # fmt: skip
        (newest_two, "r09-04", "Who are the two newest people here?", 0.625, None, "kept in lookup·L2 as set·structure"),  # a new structure
    ):  # fmt: skip
        with forged.world.trial() as copy:
            forge_task(
                copy, task_.model_copy(update={"id": open_id, "question": question}), settings, [open_id]
            )
        forged.state.solves[open_id] = {"key": "k", "results": [], "traces": []}
        forged.state.forged[open_id] = weigher.place(open_id, {"right_rate": right, "witness_right": witness}, None, "x")  # fmt: skip  # as weigh records it
        assert forged.state.forged[open_id]["verdict"] == verdict, (open_id, right)
    assert [h["task"] for h in forged.state.archive["lookup·L2"]] == ["r09-01", "r09-04"], (
        forged.state.archive
    )
    # The proposer steers by this forge's own numbers, by category and intent.
    steering = forge_measured(forged.state.forged)
    assert any(line.startswith("- lookup·structure: ") and " kept" in line for line in steering) and any(line.startswith("- search·when: ") for line in steering), steering  # fmt: skip
    pages_ = Forging(plan, settings, forged).pages()
    assert "## What this forge has measured" in pages_["memory/archive.md"] and steering[0] in pages_["memory/archive.md"] and "- promised: what someone promised" in pages_["memory/taxonomy.md"]  # fmt: skip
    assert settings.env.witness.sampling.reasoning_effort == "high", "the witness thinks hard"
    assert (forged.state.forged["r01-01"]["verdict"], forged.state.forged["r09-02"]["verdict"]) == ("replaced by r09-02", "replaced by r09-04")  # fmt: skip
    # A wrong answer another reading of its question makes right: the judge's ruling is code's blocking issue, the task
    # is sent back and never kept as hard, and a rewording keeps its asker and gold answer.
    fails(TaskReview, task_id="t", valid=True, reason="r", misread="'Ann', if 'here' is the channel")
    ruled = read_one_way(Verdict(approved=False, tasks=[TaskReview(task_id="r09-05", valid=False, reason="r", level_fit=3, misread="'Ann', if 'here' is the channel")], issues=[Issue(artifact="tasks", task_ids=["r09-05"], defect="two readings", requested_change="r")], criteria={}, summary="s"))  # fmt: skip
    assert [i.defect for i in ruled.issues if i.blocking] == ["two readings", READING + "'Ann', if 'here' is the channel"]  # fmt: skip
    readable = newest.model_copy(update={"id": "r09-05", "question": "Who is the newest person here, r09-05?"})
    with forged.world.trial() as copy:
        forge_task(copy, readable, settings, ["r09-05"])
    forged.state.task_issues["r09-05"] = [i for i in ruled.issues if "r09-05" in i.task_ids]
    sent = weigher.place("r09-05", None, None, "x")
    gold_answer = [r["answer"] for r in forged.world.db.execute("SELECT real_name AS answer FROM users ORDER BY id DESC LIMIT 1")]  # fmt: skip
    assert sent["verdict"] == "sent back: its question can be read another way" and sent["rework"] == {"answer": gold_answer, "actor": actor, "reading": "'Ann', if 'here' is the channel"}, sent  # fmt: skip
    assert not forged.world.db.execute("SELECT 1 FROM tasks WHERE id = 'r09-05'").fetchone(), "never kept"
    rework, other = {"r09-05": sent["rework"]}, forged.world.db.execute("SELECT MAX(id) FROM users").fetchone()[0]
    again = readable.model_copy(update={"id": "r09-06", "question": "Who joined the workspace last?", "rewords": "r09-05"})  # fmt: skip
    for bad, why in (
        (again.model_copy(update={"rewords": "r09-01"}), "sent back for another reading"),
        (again.model_copy(update={"actor_id": other}), "the same asker"),
        (newest_two.model_copy(update={"id": "r09-06", "question": "Who are the two who joined last?", "rewords": "r09-05"}), "keeps its gold answer"),  # fmt: skip
    ):
        try:
            with forged.world.trial() as copy:
                forge_task(copy, bad, settings, ["r09-06"], rework=rework)
            raise AssertionError(f"a rewording was accepted: {bad}")
        except ValueError as error:
            assert why in str(error), error
    with forged.world.trial() as copy:
        forge_task(copy, again, settings, ["r09-06"], rework=rework)
    # Code classes each wrong try: out of turns, evidence never shown, or all of it shown and answered otherwise.
    tried = [{"correct": False, "unanswered": True, "reason": EMPTY}, {"correct": False, "abstained": False, "evidence_coverage": 0.5, "missing": ["x"], "extra": []}, {"correct": False, "abstained": False, "evidence_coverage": 1.0, "missing": ["x"], "extra": ["y"]}, {"correct": True}]  # fmt: skip
    forged.state.solves["r09-06"] = {"key": "k", "results": tried, "traces": []}
    assert weigher.place("r09-06", {"right_rate": 0.25, "witness_right": 1.0}, None, "x")["miss_classes"] == {"budget": 1, "search": 1, "saw_all": 1}  # fmt: skip
    assert [miss_class(r, t) for r, t in (({"abstained": True}, "text"), ({"missing": [], "extra": []}, "status"), ({"committed": ["Ann"], "missing": [], "extra": []}, "refusal"), ({"evidence_coverage": None, "missing": ["a"], "extra": []}, "number"))] == ["abstained", "overclaim", "trap", "saw_all"]  # fmt: skip
    forged.close()
    fails(Store, plan.output, forge_provenance(plan.model_copy(update={"rounds": 3}), settings))  # fmt: skip  # its own config only
    # A forge stops early: at its goal (here, one task and no board), or after two rounds that place nothing. A goal
    # the archive's places cannot hold is refused.
    fails(Forging, plan.model_copy(update={"tasks": 1000}), settings, forged)
    for name, changes, vague, rounds in (
        ("forge-full", {"tasks": 1}, False, 1),
        ("forge-idle", {}, True, 2),
    ):
        early = plan.model_copy(update={"output": root / name, "rounds": 4} | changes)
        forging["vague"], early_store = vague, TestStore(early.output, forge_provenance(early, settings))
        env_ = Forging(early, settings, early_store)
        env_.adopt()
        early_store.world.db.execute("DELETE FROM board")
        await env_.run(None, agents)
        assert (early_store.state.phase, early_store.state.batch) == ("done", rounds), (
            name,
            early_store.state.batch,
        )
        early_store.close()
    # A forge that only measures: the finished run's own tasks, imported as written (past the forge's own gates; a
    # level-2 count with no near-miss query here), then measured like any candidate, with no proposer turn.
    extra = World(source / "world.sqlite", writable=True)  # the forge's output, with one more task of its own
    with extra.trial() as copy:
        record_task(copy, Task(id="extra-l2", category="join", level=2, actor_id=store.world.db.execute("SELECT MIN(id) FROM users").fetchone()[0], question="How many messages are there in all?", answer_type="number", gold_sql="SELECT COUNT(*) AS answer FROM messages"), settings, Slot(id="extra-l2", category="join", level=2, concept="c", style="s"))  # fmt: skip
    source_ids = sorted(t for (t,) in extra.db.execute("SELECT id FROM tasks"))
    extra.close()
    measuring = plan.model_copy(update={"world": source, "output": root / "measured", "rounds": 0, "measure": True, "target": 30})  # fmt: skip
    measured, turns = TestStore(measuring.output, forge_provenance(measuring, settings)), len([x for x in seen if x[0] == "forge"])  # fmt: skip
    await Forging(measuring, settings, measured).run(None, agents)
    assert "extra-l2" in source_ids and sorted(measured.state.forged) == source_ids and not [o for o in measured.state.forged.values() if o["verdict"].startswith("not imported")], measured.state.forged  # fmt: skip
    assert measured.state.phase == "done" and len([x for x in seen if x[0] == "forge"]) == turns, (
        "no proposer turn"
    )
    assert any(o["verdict"].startswith("kept") for o in measured.state.forged.values())
    measured.publish()
    measured.close()
    fails(
        ForgeConfig, base=Path("w.toml"), world=root, output=root, rounds=0
    )  # no rounds and nothing to measure
    store.close()
    retried = SimpleNamespace(
        ok=False, errors=[SimpleNamespace(type="TaskError", message="malformed verdict")]
    )
    limit = SimpleNamespace(type="ReviewLimit", message="no valid ledger")
    assert failure("done", SimpleNamespace(errors=[], traces=[retried])) is None, (
        "a judge failure the review loop retried does not fail a finished run"
    )
    assert failure("day", SimpleNamespace(errors=[], traces=[retried])) == "TaskError: malformed verdict"
    assert failure("done", SimpleNamespace(errors=[limit], traces=[])) == "ReviewLimit: no valid ledger"
    # A chapter continues a world written with tasks of its own, as S3 was (tests/fixtures/parent, written by the flow
    # above as it ran before worlds left their tasks to the forge): its board and its notes are the parent's.
    parent = root / "parent"
    shutil.copytree(Path(__file__).parent / "tests/fixtures/parent", parent)
    await check_chapter(root / "chapter", parent, settings)
    GRADER["on"] = False
    print(
        "PASS author: plan, days closed only when done, review issues delivered, crash and resume, tasks, probes, fix"
    )


async def check_chapter(root, parent, settings):
    """A chapter of the scripted world: its settings and brief; the parent adopted, frozen, with no tasks and its
    last day's notes; a plan that keeps the earlier storylines first, leaves the earlier facts, events and board as they
    are, plants the brief fresh and spreads only its own facts; the past never revised; a brief answer that needs
    hiding posted only where it is hidden; then the chapter written by a scripted author, its reviews reading the past
    as frozen (an issue on it blocks nothing and reaches no one), and a forge on the version it leaves."""
    root.mkdir(parents=True)
    # A world of the first schema (before 2026-10-14) comes forward in place, once: its rows as they were.
    old = root / "schema-1.sqlite"
    shutil.copy(parent / "world.sqlite", old)
    aged = World(old, writable=True)
    for sql in ("DROP TABLE queue", "DROP TABLE integrations", "DROP TABLE artifacts", "DROP TABLE usergroup_members", "DROP TABLE usergroups", "ALTER TABLE messages DROP COLUMN subtype", "ALTER TABLE users DROP COLUMN deactivated_us", "ALTER TABLE channels DROP COLUMN archived_us", "UPDATE world_meta SET value = '1' WHERE key = 'schema_version'"):  # fmt: skip
        aged.db.execute(sql)
    rows = aged.db.execute("SELECT COUNT(*), group_concat(text) FROM messages").fetchone()[:]
    assert set(same_schema(aged)) == {"artifacts", "channels", "integrations", "messages", "queue", "usergroup_members", "usergroups", "users"}  # fmt: skip
    assert (aged.migrate(), same_schema(aged), aged.migrate(), aged.violations()) == (1, [], 2, []), "forward, once"
    assert aged.db.execute("SELECT COUNT(*), group_concat(text) FROM messages").fetchone()[:] == rows
    aged.close()
    chapter = ChapterConfig(base=Path("worldgen.toml"), world=parent, output=root / "unit", seed=11, days=2, messages=6, storylines=1, review_days=[1], material=[Material(category="semantic", level=3, needs={"values": 2, "hidden": True})])  # fmt: skip
    cfg = chapter_settings(chapter, settings)
    assert (cfg.storylines, cfg.author.review_days, cfg.activity.messages, cfg.seed) == (3, [5], 6, 11), cfg
    fails(chapter_settings, chapter.model_copy(update={"material": [Material(category="nowhere", level=1)]}), settings)  # fmt: skip
    fails(ChapterConfig, **chapter.model_dump() | {"review_days": [2]})  # a review after the last day
    before = World(parent / "world.sqlite")
    slots, parent_quotas, parent_hash = brief_slots(chapter, cfg, version_of(before) + 1), quotas(before, settings), sha256(parent / "world.sqlite")  # fmt: skip
    board_rows = before.db.execute("SELECT COUNT(*) FROM board").fetchone()[0]
    before.close()
    assert [s.id for s in slots] == ["semantic-l3-v2-1"] and slots[0].needs.fresh and slots[0].needs.values == 2
    # A chapter on a world of the first schema adopts it brought forward.
    elder = root / "elder"
    shutil.copytree(parent, elder)
    shutil.copy(old, elder / "world.sqlite")
    aged = World(elder / "world.sqlite", writable=True)
    for sql in ("DROP TABLE queue", "ALTER TABLE messages DROP COLUMN subtype", "UPDATE world_meta SET value = '1' WHERE key = 'schema_version'"):  # fmt: skip
        aged.db.execute(sql)
    aged.close()
    on_elder = chapter.model_copy(update={"world": elder, "output": root / "on-elder"})
    elder_store = TestStore(on_elder.output, chapter_provenance(on_elder, cfg))
    adopt(elder_store, on_elder, cfg, slots)
    assert (meta(elder_store.world, "schema_version"), same_schema(elder_store.world)) == ("2", []), "brought forward"
    elder_store.close()
    store = TestStore(chapter.output, chapter_provenance(chapter, cfg))
    adopt(store, chapter, cfg, slots)
    world = store.world
    first = chapter_start(world)
    assert (first, today(world), present(world), meta(world, "version"), meta(world, "parent_hash")) == (5, 5, window(world, 5, "early")[0], "2", parent_hash)  # fmt: skip
    assert {d: q for d, q in quotas(world, cfg).items() if d < first} == parent_quotas, "closed days keep their quotas"
    assert sum(q["messages"] for d, q in quotas(world, cfg).items() if d >= first) == 6
    assert world.db.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0, "a chapter carries no tasks"
    assert (store.root / "attempts/source/notes/recap.md").read_text() == "day 4: done", "the last day's notes"
    frozen = meta(world, "frozen_digest")
    assert frozen == frozen_digest(world)
    ops, leads = channel_id("public", "ops", []), channel_id("private", "leads", [])
    a, other = [r[0] for r in world.db.execute("SELECT user_id FROM members WHERE channel_id = ? ORDER BY user_id", (leads,))][:2]  # fmt: skip
    owner = world.db.execute("SELECT author_id FROM facts WHERE id = 'f3'").fetchone()[0]
    a, other = (owner, other if other != owner else a)
    document = ledger_document(world, {s.id for s in slots})
    assert document["board"] == [], "the earlier board stays out of the chapter's plan"
    plan = Plan.model_validate(document)
    mine = Storyline(id="s3", summary="the next audit")
    later = PlanFact(id="g1", storyline="s3", subject="Audit", attribute="owner", value="Mia", anchor="Mia", channel_id=leads, author_id=a, day=5, supersedes="f4", summary="s")  # fmt: skip
    lure = PlanFact(id="d1", storyline="s3", subject="Audit", attribute="owner", value="Raj", anchor="Raj", channel_id=ops, author_id=a, day=5, decoy=True, summary="s")  # fmt: skip
    entry = [BoardEntry(slot=slots[0].id, facts=["f4", "g1"])]

    def planned(**changes):
        return plan.model_copy(update={"storylines": [*plan.storylines, mine], "facts": [*plan.facts, later, lure], "board": entry} | changes)  # fmt: skip

    def tried(act):
        with world.trial() as copy:
            act(copy)

    def recorded(document, brief=slots):
        tried(lambda copy: record_plan(copy, document, cfg, brief))

    unfinished = unplanned(world, cfg, slots)
    assert any("3 storylines" in e for e in unfinished) and any("none is planned for day 5" in e for e in unfinished) and any("['semantic-l3-v2-1']" in e for e in unfinished), unfinished  # fmt: skip
    old_facts = planned().facts
    for expected, bad, brief in (
        ("keep ['s1', 's2'] in this order, first", planned(storylines=[mine, *plan.storylines]), slots),
        ("plan exactly 3 storylines", planned(storylines=plan.storylines), slots),
        ("f1 is an earlier chapter's, written and frozen", planned(facts=[old_facts[0].model_copy(update={"summary": "said again"}), *old_facts[1:]]), slots),
        ("an earlier chapter's days are written and frozen", planned(events=[*plan.events, Event(id="e9", storyline="s3", title="look back", day=3, time="10:00")]), slots),
        ("planned for today (day 5) or later", planned(facts=[*plan.facts, later.model_copy(update={"day": 4}), lure]), slots),
        ("taking at least 4 values", planned(), [s.model_copy(update={"needs": s.needs.model_copy(update={"values": 4})}) for s in slots]),
        ("first stated in this chapter", planned(board=[BoardEntry(slot=slots[0].id, facts=["f4"])]), slots),
    ):  # fmt: skip
        refused(expected, recorded, bad, brief)
    # The chapter's facts run through its own days (Friday 5 messages, Saturday 1): twelve on its first day leave its
    # second short, and an earlier chapter's facts are not spread over them.
    many = [later.model_copy(update={"id": f"g{k}"}) for k in range(12)]
    assert "days 6-6 first state 0 of the plan's 12 facts; they hold 17% of this chapter's messages" in (front_loaded(world, cfg, many, 5) or "")  # fmt: skip
    assert front_loaded(world, cfg, [*[plan.facts[0]] * 10, later, lure], 5) is None
    recorded(planned())
    assert unplanned(world, cfg, slots) == []
    assert world.db.execute("SELECT COUNT(*) FROM board").fetchone()[0] == board_rows + 2, "the earlier board stays"
    assert [r[0] for r in world.db.execute("SELECT id FROM storylines ORDER BY position")] == ["s1", "s2", "s3"]
    assert json.loads(meta(world, "segments"))[-1]["busy"] == [], "the chapter's first plan marks its own event days"
    old = world.db.execute("SELECT MIN(id) FROM messages").fetchone()[0]
    refused("earlier chapter, written and frozen", tried, lambda copy: revise(copy, old, "edited"))
    gaps, rng = Gaps(cfg.personas.gaps), random.Random(3)

    def said(*lines, channel=leads):
        tried(lambda copy: post(copy, Conversation(channel_id=channel, about="audit", lines=[PostLine(**line) for line in lines]), cfg, rng, gaps))  # fmt: skip

    said(dict(author_id=a, text="I heard Raj owns it", conveys=["d1"]), channel=ops)
    refused("answers the brief's ['semantic-l3-v2-1']", said, dict(author_id=a, text="Mia has the audit now", conveys=["g1"]))  # fmt: skip
    said(dict(author_id=other, text="who has the audit now?"), dict(author_id=a, text="Mia has it", reply_to=0, conveys=["g1"]))  # fmt: skip
    assert frozen_digest(world) == frozen, "the chapter wrote nothing before its first day"
    assert style(world, frozen_before(world))["messages"] == 3, "the chapter's style is its own messages'"
    store.close()

    # The chapter written through the driver: a scripted author, and a judge whose issues name the frozen past.
    run = chapter.model_copy(update={"output": root / "run"})
    turns = []

    async def script(prompt, tools, runtime):
        call = lambda name, **kwargs: tools._with_state(getattr(tools, name))(**kwargs)  # noqa: E731
        world = World(tools.config.db_path)
        turns.append((tools.config.mode, tools.config.day, prompt))
        if tools.config.mode == "plan":
            if prompt.startswith("Before day 5"):
                assert "days 1-4 are written and frozen" in prompt and b"Brief: what this chapter plants" in runtime.files["/task/memory/now.md"] and b"semantic-l3-v2-1" in runtime.files["/task/memory/now.md"]  # fmt: skip
                assert json.loads(runtime.files["/task/memory/ledger.json"])["board"] == [] and runtime.files["/task/notes/recap.md"] == b"day 4: done"  # fmt: skip
                return  # nothing planned: the turn is told what is owed
            if "plan this chapter's facts" in prompt:
                await call("plan", ledger=planned())
                return  # plan.md left as the parent's: the turn is told to write it
            assert "write /task/notes/plan.md" in prompt, prompt
            await runtime.write("/task/notes/plan.md", b"the chapter: the audit changes hands, said in a thread")
            return
        day = tools.config.day
        assert f"day {day} of 6 (this chapter writes days 5-6)" in prompt, prompt
        try:
            await call("revise", message_id=old, text="edited")
            raise AssertionError("a frozen message was revised")
        except ValueError as error:
            assert "earlier chapter" in str(error), error
        if day == 5:
            await call("post", conversation=Conversation(channel_id=ops, about="audit", lines=[PostLine(author_id=a, text="I heard Raj owns it", conveys=["d1"])]))  # fmt: skip
            await call("post", conversation=Conversation(channel_id=leads, about="audit", lines=[PostLine(author_id=other, text="who has the audit now?"), PostLine(author_id=a, text="Mia has it", reply_to=0, conveys=["g1"])]))  # fmt: skip
        while today(world) == day:
            await call("advance", to=onward(world))
        await runtime.write("/task/notes/recap.md", f"day {day}: done".encode())

    reviewed = []

    class ScriptedJudge(ScriptedAgent):
        async def run(self, task):
            payload, attempt = json.loads(task.config.payload), task.data.attempt
            reviewed.append((attempt, "written_from" in payload, "written_through" in payload))
            judged_world = World(task.config.tools.db_path)
            old_id = judged_world.db.execute("SELECT MIN(id) FROM messages").fetchone()[0]
            new_id = judged_world.db.execute("SELECT MAX(id) FROM messages").fetchone()[0]
            issues = [Issue(artifact="workspace", message_ids=[old_id], defect="a stiff old line", requested_change="loosen it", blocking=True)]  # fmt: skip
            if attempt.startswith("review-"):
                issues.append(Issue(artifact="workspace", message_ids=[new_id], defect="a chapter line reads flat", requested_change="loosen it", blocking=False))  # fmt: skip
            judged_world.close()
            verdict = Verdict(approved=False, tasks=[], issues=issues, criteria=dict.fromkeys(PHASE_CRITERIA["world"], 1.0), summary="s")  # fmt: skip
            return await judged(task, verdict)

    agents = SimpleNamespace(judge=ScriptedJudge(), solver=ScriptedAgent(), witness=ScriptedAgent(), author=ScriptedAuthor(script))  # fmt: skip
    manifest = chapter_provenance(run, cfg)
    store = TestStore(run.output, manifest)
    env = ChapterEnv(run, cfg, store, slots)
    await env.setup(agents)
    await env.run(None, agents)
    state = store.state
    assert state.phase == "done" and state.rounds["final"] == 1 and state.rounds["plan"] == 1, state.rounds
    assert [t[2][:20] for t in turns if t[0] == "plan"] == ["Before day 5 (Fri 20", "The plan is not done", "The plan is not done"], [t[2][:20] for t in turns]  # fmt: skip
    day6 = next(t[2] for t in turns if t[:2] == ("day", 6))
    assert "a chapter line reads flat" in day6 and "a stiff old line" not in day6, "a frozen issue reaches no one"
    assert reviewed == [("review-05-01", True, True), ("final-01", True, False)], reviewed
    assert (store.root / "attempts/final-01/frozen_issues-world.json").exists(), "it is kept apart"
    world = store.world
    assert meta(world, "frozen_digest") == frozen_digest(world) == frozen and world.violations(complete=True) == []
    assert {d: q for d, q in quotas(world, cfg).items() if d < 5} == parent_quotas
    assert world.db.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0 and sha256(parent / "world.sqlite") == parent_hash  # fmt: skip
    # Its profile: the group the brief planted, in the version it was planted in, its current value where few read it.
    found = profile(world)
    owners = next(g for g in found["groups"] if (g["subject"], g["attribute"]) == ("Audit", "owner"))
    assert (owners["values"], owners["current"], owners["version"], owners["current_where"]) == (3, ["g1"], 2, ["thread reply in private"]) and "semantic-l3-v2-1" in owners["board"], owners  # fmt: skip
    leads_members = world.db.execute("SELECT COUNT(*) FROM members WHERE channel_id = ?", (leads,)).fetchone()[0]
    assert owners["readers"] == [leads_members, 4] and found["world"]["version"] == 2 and found["material"]["groups_3_values"] >= 1, owners  # fmt: skip
    tools_task = WorldAuthorTask.create("tasks", 0, world.path, env.author_context(), "check")
    (tools,) = tools_task.toolsets(tools_task.config)
    await tools.setup()
    with_state(tools)
    try:
        await tools._with_state(tools.add_task)(task=Task(id=slots[0].id, category="semantic", level=3, actor_id=a, question="Who owns the audit now?", answer_type="text", gold_sql="SELECT value AS answer FROM facts WHERE id = 'g1'", facts=["g1"]))  # fmt: skip
        raise AssertionError("a chapter wrote a task")
    except ValueError as error:
        assert "a chapter writes no tasks" in str(error), error
    world.db.execute("UPDATE messages SET text = text || ', edited' WHERE id = ?", (old,))  # the past, changed behind code
    try:
        await env.final_world(agents)
        raise AssertionError("a changed past passed the final check")
    except ValueError as error:
        assert "an earlier chapter's world changed" in str(error), error
    store.close()
    fails(Store, run.output, chapter_provenance(run.model_copy(update={"days": 3}), cfg))  # its own config only
    # A forge on the version the chapter leaves: the brief is its board; the earlier entries were covered before.
    forge = ForgeConfig(base=Path("worldgen.toml"), world=run.output, output=root / "forge", rounds=1, candidates=1, tasks=1)  # fmt: skip
    forging = TestStore(forge.output, forge_provenance(forge, cfg))
    on_chapter = ForgeEnv(forge, cfg, forging)
    on_chapter.adopt()
    assert on_chapter.covered()[1] == ["semantic-l3-v2-1"], on_chapter.covered()
    # One round on it, whose task review notes a message of the frozen past: thawed, the note is set apart, and the
    # candidate is kept.
    forging.close()
    with episodes(SimpleNamespace(**forge_seats(note_on=old))):
        summary = await forge_run(forge, cfg)
    forging = TestStore(forge.output, forge_provenance(forge, cfg))
    assert summary["status"] == "complete" and forging.state.forged["r01-01"]["verdict"].startswith("kept"), forging.state.forged  # fmt: skip
    assert (forging.root / "attempts/round-01/frozen_issues-tasks.json").exists(), "the frozen issue is set apart"
    assert [r.task_id for r in load_release(forge.output / "release")[1]] == ["r01-01"], "the forge publishes"
    forging.close()
    print("PASS chapter: adopted frozen, plan and brief checked, the past never written, reviews thawed, forged on")


def scripted_task(settings, slot, actor):
    """A valid task for any slot of the scripted world: its gold follows the category's source."""
    category, level = slot["category"], slot["level"]
    gold, spec = settings.taxonomy[category].gold, settings.taxonomy[category]
    question = f"Question {slot['id']} about the {category} level {level} work?"
    if category == "robustness":
        sql, kind, facts = "SELECT id AS answer FROM users WHERE real_name = 'Nobody Here'", "refusal", []
    elif gold == "ledger":
        sql, kind, facts = "SELECT value AS answer FROM facts WHERE id = 'f4'", "text", ["f3", "f4"]
    elif gold == "hybrid":
        sql, kind, facts = "SELECT COUNT(*) AS answer FROM messages m JOIN evidence e ON e.message_id = m.id WHERE e.fact_id = 'f1'", "number", ["f1"]  # fmt: skip
    else:
        sql, kind, facts = "SELECT COUNT(*) AS answer FROM messages", "number", []
    assert kind in spec.answer_types
    return Task(id=slot["id"], category=category, level=level, actor_id=actor, question=question, answer_type=kind, gold_sql=sql, facts=facts)  # fmt: skip


# The files whose strings a model reads: guides, turn prompts, refusals, memory pages and tool descriptions.
PROMPT_SOURCES = [
    *(ROOT / "generators/worldgen_slack" / n for n in ("agents/world.py", "agents/judge.py", "agents/synthesizer.py", "agents/author.py", "agents/inspection.py", "chronicle.py", "contracts.py", "env.py", "forge.py", "chapter.py")),
    *(ROOT / "environments/worldgen_slack/worldgen_slack" / n for n in ("taskset.py", "tools.py", "db.py")),
]  # fmt: skip


def prompt_strings(path: Path) -> list[tuple[int, str]]:
    """A source file's strings that reach a model: every literal (f-strings' parts too) but docstrings, which no model
    reads unless they describe a tool, and model ids (a provider/model path)."""
    tree, skip = ast.parse(path.read_text()), set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):  # fmt: skip
            if not any(getattr(d, "attr", getattr(d, "id", None)) == "tool" for d in getattr(node, "decorator_list", [])):
                skip.add(id(body[0].value))
    return [(n.lineno, n.value) for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in skip and not re.fullmatch(r"[\w.-]+/[\w.-]+", n.value)]  # fmt: skip


def check_prompts(root):
    """What a model reads names no model and no earlier run: a proposer knows the solver by its tries, an author the
    world by its memory. Guides render their numbers from code: the forge's tries and middle rates, the parts of the
    day, the near-miss queries a task may name."""
    root.mkdir(parents=True)
    settings = contracts_settings(root)
    models = [settings.env.author.model, settings.env.judge.model, settings.env.solver.model, settings.env.witness.model, settings.answer_judge.model, "deepseek/deepseek-v4.1-flash"]  # fmt: skip
    names = {"glm", "gpt", "claude", "opus", "sonnet", "haiku", "gemini", "deepseek", "qwen", "kimi", "llama", "openai", "anthropic"}  # fmt: skip
    names |= {w for m in models for w in (m.split("/")[0], re.match(r"[a-z]+", m.split("/")[-1]).group()) if len(w) > 2}
    banned = re.compile(r"\b(?:" + "|".join(sorted(names)) + r")\b|\b[SFM]\d\b|\bforge-t\d", re.IGNORECASE)
    found = [f"{path.relative_to(ROOT)}:{line}: {m.group(0)!r}" for path in PROMPT_SOURCES for line, text in prompt_strings(path) for m in banned.finditer(text)]  # fmt: skip
    configured = [settings.tasks.styles, *([c.definition, *c.levels, *(x for level in c.concepts for x in level)] for c in settings.taxonomy.values())]  # fmt: skip
    found += [f"config: {m.group(0)!r} in {text[:60]!r}" for texts in configured for text in texts for m in banned.finditer(text)]  # fmt: skip
    assert not found, "model or run names in what a model reads: " + "; ".join(found)
    guide = forge_guide(settings.model_dump(mode="json"))
    assert "tries each one 8 times" in guide and "right in 25%-75% of its tries" in guide, "the forge's own numbers"
    other = settings.model_copy(update={"author": settings.author.model_copy(update={"tries": 3}), "tasks": settings.tasks.model_copy(update={"bands": [(0.8, 1.0), (0.2, 0.8), (0.0, 0.2)]})})  # fmt: skip
    guide = forge_guide(other.model_dump(mode="json"))
    assert "tries each one 3 times" in guide and "right in 20%-80% of its tries" in guide, guide[:600]
    parts = ", ".join(f"{p} {a:02d}-{b:02d}" for p, (a, b) in PARTS.items())
    assert parts in " ".join(AUTHOR_GUIDE.split()) and "PARTS_OF_THE_DAY" not in AUTHOR_GUIDE and "NEAR_SQL" not in AUTHOR_GUIDE  # fmt: skip
    assert f"up to {Task.model_fields['near_sql'].metadata[0].max_length} queries" in AUTHOR_GUIDE
    assert "written and frozen" in CHAPTER_GUIDE and "No tasks are written in a chapter" in CHAPTER_GUIDE
    # The world's author writes with the craft page (voice lab 02); the proposer has its own guide, and a chapter's
    # author has the chapter's too.
    world = World.create(root / "w.sqlite")
    context = {"settings": settings.model_dump(mode="json")}
    systems = {name: WorldAuthorTask.create(mode, 1, world.path, context | extra, "x").data.system_prompt for name, mode, extra in (("plan", "plan", {}), ("day", "day", {}), ("forge", "forge", {}), ("chapter", "day", {"chapter": {"first": 5}}))}  # fmt: skip
    assert all("Nobody states the lesson" in systems[m] and "Stakes move" in systems[m] for m in ("plan", "day", "chapter")) and "Stakes move" not in systems["forge"]  # fmt: skip
    assert CHAPTER_GUIDE in systems["chapter"] and CHAPTER_GUIDE not in systems["day"] and CRAFT_GUIDE.strip() in systems["day"]  # fmt: skip
    world.close()
    print("PASS prompts: no model or run names in what models read; guides render their numbers from code")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as directory:
        asyncio.run(check_world(Path(directory) / "world"))
        check_tasks(Path(directory) / "tasks")
        check_contracts(Path(directory) / "contracts")
        check_clock(Path(directory) / "clock")
        check_apps(Path(directory) / "apps")
        check_churn(Path(directory) / "churn")
        check_links(Path(directory) / "links")
        asyncio.run(check_tools(Path(directory) / "tools"))
        asyncio.run(check_reviews(Path(directory) / "reviews"))
        asyncio.run(check_author(Path(directory) / "author"))
        check_prompts(Path(directory) / "prompts")
    print("All generation checks passed.")
