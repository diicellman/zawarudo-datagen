"""The env: the world file as one person reads it, Slack's reads and their server."""

import asyncio
import inspect
import json
import os
import sqlite3
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from worldgen_slack import slack
from worldgen_slack.taskset import system_prompt
from worldgen_slack.toolset import SlackToolset, WorldToolsConfig, stage_world, watch_parent

from ..workspace import ANSWER_KEY, Workspace
from .support import fails, mcp_client, world_fixture


async def test_world(tmp_path):
    """The world file: triggers and rules reject defects at insert and roll back; reads are Slack as one actor."""
    world, at, message = world_fixture(tmp_path / "world.sqlite")
    assert world.violations(complete=True) == []
    row = dict(
        id="t4",
        category="temporal",
        actor_id="U2",
        question="When?",
        answer_type="text",
        gold_source="sql",
        gold_sql="SELECT 1 AS answer",
    )
    try:  # the taxonomy is data: any category, any level from 1
        with world.trial() as copy:
            copy.insert("tasks", [row | {"level": 4}])
            raise LookupError("drop the trial")
    except LookupError:
        pass
    # else a level-0 task was inserted
    with pytest.raises(sqlite3.IntegrityError):
        with world.trial() as copy:
            copy.insert("tasks", [row | {"level": 0}])
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
    on_event = fact | dict(
        subject="a",
        attribute="b",
        day=1,
        event_id="e1",
        moment_zone="America/Chicago",
        moment_kind="scheduled",
    )
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
        ],
    )
    with world.batch():  # facts about one event share its moment; a promise is closed after it is made
        world.insert("events", [window_event])
        world.insert("facts", [on_event | dict(id="f8", moment_us=at(2, 10))])
        world.insert(
            "commitments",
            [
                dict(
                    id="c1",
                    owner_id="U1",
                    text="post notes",
                    message_id=1,
                    due_us=at(3, 0),
                    status="kept",
                    closed_by=3,
                )
            ],
        )
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

    world.solver_copy(tmp_path / "solver.sqlite")
    solver = Workspace(tmp_path / "solver.sqlite", actor="U2")
    names = {r[0] for r in solver.db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert not names & set(ANSWER_KEY) and {"messages", "messages_fts"} <= names

    def read(reader, tool, **arguments):
        return slack.read(reader, tool, arguments)

    def texts(reader, *keywords, filters="", **kwargs):
        found = read(reader, "search_public_and_private", keywords=list(keywords), filters=filters, **kwargs)
        return [i["text"] for i in found["items"]]

    u1, u3 = (
        Workspace(tmp_path / "solver.sqlite", actor="U1"),
        Workspace(tmp_path / "solver.sqlite", actor="U3"),
    )
    fails(read, solver, "read_channel", channel_id="G1")
    fails(read, u3, "read_channel", channel_id="G1")
    assert read(u3, "list_user_channels")["items"] == [] and texts(solver, "deleted") == []
    assert texts(solver, "rollback", "budget") == [] and texts(u1, "rollback", "budget") == [
        "private: rollback budget approved"
    ]
    assert "can you check the rollback" in texts(
        solver, "rollback"
    ) and "can you check the rollback" not in texts(u3, "rollback")
    assert texts(solver, "rollback", filters="is:dm") == ["can you check the rollback"] and texts(
        u1, "rollback", filters="in:@u2"
    ) == ["can you check the rollback"]
    assert [i["text"] for i in read(solver, "search_public", keywords=["rollback"])["items"]] == texts(
        solver, "rollback", channel_types="public_channel"
    ), "the public search reads public channels only"
    every = "public_channel,private_channel,mpim,im"
    assert [c["id"] for c in read(solver, "list_user_channels", types=every)["items"]] == ["C1", "D1"]
    assert [c["id"] for c in read(solver, "list_user_channels")["items"]] == ["C1"], (
        "direct conversations only when asked"
    )
    assert read(solver, "list_user_channels", types="im")["items"][0]["users"] == ["U1", "U2"], (
        "a DM names both members"
    )
    assert read(solver, "read_channel", channel_id="U1")["channel"] == "D1", (
        "a person's id reads the DM with them"
    )
    # The client says whom the assistant acts as and when it is; the tools show every time on the user's own clock.
    assert (
        system_prompt(solver)
        == "You are using Slack as the user with user_id U2. It is Thursday, June 18, 2026, 00:00 on your clock (America/Chicago)."
    )
    first = read(solver, "read_channel", channel_id="C1")["items"][-1]
    assert first["time_local"] == datetime.fromtimestamp(
        float(first["ts"]), ZoneInfo("America/Chicago")
    ).strftime("%Y-%m-%d %H:%M:%S -05"), first
    fall = int(datetime(2026, 11, 1, 6, 30, tzinfo=ZoneInfo("UTC")).timestamp() * 1e6)  # 1:30 CDT, then CST
    assert [slack.clock(solver, fall + h * 3_600_000_000)[-12:] for h in (0, 1)] == [
        "01:30:00 -05",
        "01:30:00 -06",
    ]
    assert [m["text"] for m in read(u3, "read_channel", channel_id="C1")["items"]] == [
        "notes posted for release 4.2",
        texts(u3, "blocked")[0],
    ]
    root_message = read(u3, "read_channel", channel_id="C1")["items"][1]
    assert root_message["reply_count"] == 1 and root_message["reactions"] == [{"name": "eyes", "count": 1}], (
        "counts, not who"
    )
    root_ts = root_message["ts"]
    assert [m["text"] for m in read(u3, "read_thread", channel_id="C1", message_ts=root_ts)["items"]][
        1
    ] == "ack, rolling back now"
    assert texts(solver, "release")[0].startswith("Release 4.2 rollback"), (
        "BM25 ranks the denser, older match first"
    )
    assert texts(solver, "release", sort="timestamp")[0] == "notes posted for release 4.2"
    assert texts(solver, "release", sort="timestamp", sort_dir="asc")[0].startswith("Release 4.2 rollback")
    assert texts(solver, '"rolling back"') == ["ack, rolling back now"]
    assert (
        texts(solver, "release", filters="from:@u2")
        == ["notes posted for release 4.2"]
        == texts(solver, "4.2", filters="in:#releases on:2026-06-16")
    )
    assert texts(solver, "4.2", filters="before:2026-06-16") == texts(solver, "blocked") and texts(
        solver, "4.2", filters="after:2026-06-15"
    ) == texts(solver, "notes")
    assert texts(solver, "release", filters="in:#releases in:#exec") == texts(solver, "release"), (
        "a repeated modifier is OR"
    )
    assert texts(solver, "rollback", filters="in:#exec") == [] and texts(
        solver, "release", "-notes"
    ) == texts(solver, "blocked")
    assert texts(solver, filters="has:reaction") == [texts(solver, "blocked")[0]] and texts(
        solver, filters="is:thread"
    ) == ["ack, rolling back now", texts(solver, "blocked")[0]]
    context = read(solver, "search_public_and_private", keywords=["ack"])["items"][0]["context"]
    assert [m["ts"] for m in context["before"]] == [root_ts] and context["after"] == [], (
        "a reply's context is its thread"
    )
    page = read(solver, "search_public_and_private", keywords=["release"], limit=1)
    assert (
        read(solver, "search_public_and_private", keywords=["release"], cursor=page["next_cursor"], limit=1)[
            "items"
        ][0]["text"]
        == "notes posted for release 4.2"
    )
    assert (
        read(solver, "search_public_and_private", keywords=["rollback"], cursor=page["next_cursor"])["items"]
        == []
    ), "a cursor names a page, not a request, as the server's do"
    fails(read, solver, "search_public_and_private", keywords=["release"], cursor=slack._cursor("next_ts", 1))
    fails(read, solver, "search_public_and_private", keywords=["release"], limit=21)
    fails(read, solver, "search_public_and_private", keywords=[])
    assert read(solver, "get_reactions", channel_id="C1", message_ts=root_ts)["reactions"] == [
        {"name": "eyes", "count": 1, "users": ["U2"], "names": ["Owen Sato"]}
    ]
    assert (
        read(solver, "read_user_profile", user_id="U1")["handle"] == "u1"
        and read(solver, "read_user_profile")["id"] == "U2"
    )
    assert [p["id"] for p in read(solver, "search_users", keywords=["ali"])["items"]] == ["U1"], (
        "a partial name finds its person"
    )
    assert [
        c["name"]
        for c in read(
            solver, "search_channels", keywords=["rel"], channel_types="public_channel,private_channel"
        )["items"]
    ] == ["releases"], "private channels stay unlisted"
    assert [p["id"] for p in read(u1, "list_channel_members", channel_id="G1")["items"]] == ["U1"] and read(
        u1, "list_channel_members", channel_id="G1", response_format="ids_only"
    )["format"] == "ids_only"
    fails(read, solver, "list_channel_members", channel_id="D1")  # not a direct conversation
    # What the solver reads is the real server's text (capture/slack-mcp): its shapes, rendered from records.
    fixture = Path(__file__).resolve().parents[3] / "environments/worldgen_slack/capture/slack-mcp"
    served = {t["name"]: t for t in json.loads((fixture / "tools.json").read_text())["tools"]}
    shown_ = slack.render(solver, "read_channel", read(solver, "read_channel", channel_id="C1"))
    assert (
        shown_["messages"].startswith(
            "Channel: #releases (C1)\n\n=== Message from Owen Sato <U2@example.com> (U2) at "
        )
        and "\nMessage TS: " in shown_["messages"]
        and "\nThread: 1 replies (latest: " in shown_["messages"]
        and "\nReactions: eyes (1)" in shown_["messages"]
    ), shown_
    assert shown_["pagination_info"] == "There are no more messages available.\n"
    hit = slack.render(
        solver, "search_public_and_private", read(solver, "search_public_and_private", keywords=["ack"])
    )
    assert (
        hit["results"].startswith(
            "# Search Results for: \n\n## Messages (1 results)\n### Result 1 of 1\nChannel: #releases (ID: C1)\nFrom: Owen Sato <U2@example.com> (ID: U2) \n"
        )
        and "\nContext before: \n- From: Alicia Rao" in hit["results"]
        and hit["pagination_info"] == "End of results - No more pages available.\\n"
    ), hit
    assert slack.render(
        solver, "get_reactions", read(solver, "get_reactions", channel_id="C1", message_ts=root_ts)
    ) == {"result": "Reactions on message:\n\n:eyes: × 1 — Owen Sato (U2)"}
    assert (
        slack.render(
            solver,
            "search_public_and_private",
            read(solver, "search_public_and_private", keywords=["nothinghere"]),
        )
        == json.loads((fixture / "responses.json").read_text())[16]["response"]
    ), "an empty search reads as the real one"
    assert texts(solver, "4.2", filters="on:2026-06-15") == texts(solver, "blocked"), "on: is one day"
    assert slack.text_of(solver, "hi <@U1>, see <#C1>") == "hi <@U1|Alicia>, see <#C1|releases>", (
        "names, as Slack's API gives them"
    )
    try:
        with (
            world.trial() as copy
        ):  # a bot's post, a second reply: searched only with include_bots; a thread oldest first
            copy.db.execute("UPDATE users SET is_bot = 1 WHERE id = 'U2'")
            copy.insert(
                "messages",
                [
                    dict(
                        channel_id="C1",
                        user_id="U1",
                        ts_us=int(float(root_ts) * 1e6) + 9_000_000_123,
                        text="all clear now",
                        parent_id=1,
                    )
                ],
            )
            bots = Workspace(copy.path, actor="U1")
            assert texts(bots, "notes") == [] and texts(bots, "notes", include_bots=True) == [
                "notes posted for release 4.2"
            ]
            assert [
                m["text"] for m in read(bots, "read_thread", channel_id="C1", message_ts=root_ts)["items"]
            ][1:] == ["ack, rolling back now", "all clear now"]
            bots.close()
            raise LookupError("drop the trial")
    except LookupError:
        pass
    fails(read, solver, "read_user_profile", user_id="U9")
    try:
        read(solver, "read_channel", channel_id="C9")
    except ValueError as error:
        assert (
            str(error) == "execution_failed: channel_not_found\n\n" + slack.EXPLAINED["channel_not_found"]
        ), error

    fails(stage_world, WorldToolsConfig(actor_id="U2", colocated=True), tmp_path / "solver.sqlite")
    config = WorldToolsConfig(actor_id="U2")
    stage_world(config, tmp_path / "solver.sqlite")
    toolset = SlackToolset(config)
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
    assert (
        answers_[0]["results"].startswith("# Search Results for: ")
        and stored["state"].calls[0].output["items"]
    ), "the solver reads text; grading keeps the record"
    async with mcp_client(toolset) as client:
        # The tool server adds its prefix as the solver's harness names a server's tools: slack_read_channel.
        listed = {f"{SlackToolset.TOOL_PREFIX}_{t.name}": t for t in (await client.list_tools()).tools}
        assert set(listed) == set(served), sorted(listed)
        for name, tool in listed.items():  # every parameter the real server takes, by its name and kind
            want = served[name]["input_schema"]["properties"]
            assert set(tool.input_schema["properties"]) == set(want), (
                name,
                set(tool.input_schema["properties"]) ^ set(want),
            )
            assert set(tool.input_schema.get("required", [])) == set(
                served[name]["input_schema"]["required"]
            ), name
    bad = WorldToolsConfig(actor_id="U2", db_path=config.db_path, db_hash="0" * 64)
    # else a changed world file was served
    with pytest.raises(ValueError):
        await SlackToolset(bad).setup()
    # A tool server exits with the process that started it: killed while the server watches, or gone before it could.
    watcher = "import os, threading, time\n" + inspect.getsource(watch_parent)
    for late, then in (
        (False, "watch_parent(); print('watching', flush=True)"),
        (True, "time.sleep(0.5); watch_parent()"),
    ):
        server_code = f"{watcher}\n{then}\ntime.sleep(60)"
        wait = "" if late else "s.stdout.readline(); "
        parent_code = f"import subprocess, sys, time; s = subprocess.Popen([sys.executable, '-c', {server_code!r}], stdout=subprocess.PIPE, text=True); {wait}print(s.pid, flush=True); time.sleep(0 if {late} else 60)"
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
