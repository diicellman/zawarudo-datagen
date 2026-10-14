"""The solver's tools read as the official Slack MCP server does, call for call. The workspace the capture
(environments/worldgen_slack/capture/slack-mcp) was taken in, rebuilt as a world, answers each captured call as the
server did, in every tool and format, but where a world differs from that workspace or the env from the server (each
listed below, with why)."""

import json
import re

import pytest
from worldgen_slack import slack

from ..config import ROOT
from ..workspace import Workspace

CAPTURE = json.loads((ROOT / "environments/worldgen_slack/capture/slack-mcp/responses.json").read_text())
ME, CAPTURED_IN = "U0EXAMPLE01", "C0EXAMPLE01"
# Where a world, or the env, reads otherwise than the server did. A strict expectation: if one comes to read the same,
# its entry goes.
DIFFERENT = {
    4: "the capture's last reply was also sent to the channel; a world's replies stay in their threads",
    5: "the reply sent to the channel too, so the page and its cursor differ",
    7: "the reply sent to the channel too, inside the window",
    27: "the reply sent to the channel too, beside the hit in its channel",
}
# The capture's messages were posted through the server, which signs each; a world's messages are not signed.
SIGNED = re.compile(r"(\n| )\*Sent using\* Claude")


def us(ts: str) -> int:
    seconds, micros = ts.split(".")
    return int(seconds) * 1_000_000 + int(micros)


@pytest.fixture(scope="module")
def captured_in(tmp_path_factory) -> Workspace:
    """Example Co's workspace as the capture saw it: Alex Rivera (Etc/GMT-5), a colleague and Slackbot; #general and
    #banter, made in 2023 with Slack's default purposes; a DM with each person; and the private #zawarudo-capture: a
    join, a post with a thread of three replies, a checklist, and reactions."""
    world = Workspace.create(tmp_path_factory.mktemp("capture") / "world.sqlite")
    join, root, made = us("1791967174.350069"), us("1791967180.718159"), us("1686119850.000000")
    people = [
        ("U0EXAMPLE01", "alex.rivera", "Alex Rivera", 0),
        ("U0EXAMPLE00", "jordan.lee", "Jordan Lee", 0),
        ("USLACKBOT", "slackbot", "Slackbot", 1),
    ]
    # id, name, type, members, creator, made, purpose
    channels = [
        (CAPTURED_IN, "zawarudo-capture", "private", ["U0EXAMPLE01"], ME, join, ""),
        (
            "C0EXAMPLE00",
            "general",
            "public",
            ["U0EXAMPLE01", "U0EXAMPLE00"],
            None,
            made - 1_000_000,
            "This is the one channel that will always include everyone. It’s a great spot for announcements and "
            "team-wide conversations.",
        ),
        (
            "C0EXAMPLE02",
            "banter",
            "public",
            ["U0EXAMPLE01", "U0EXAMPLE00"],
            "U0EXAMPLE00",
            made,
            "This channel is for... well, everything else. It’s a place for team jokes, spur-of-the-moment ideas, and "
            "funny GIFs. Go wild!",
        ),
        ("D0EXAMPLE01", None, "im", ["U0EXAMPLE01", "U0EXAMPLE00"], None, join - 1_000_000, ""),
        ("D0EXAMPLE02", None, "im", ["U0EXAMPLE01", "USLACKBOT"], None, join - 1_000_000, ""),
    ]
    messages = [
        ("1791967174.350069", None, f"<@{ME}> has joined the channel", "channel_join"),
        (
            "1791967180.718159",
            None,
            "Heads up: the 2026.08 prod deploy window moves to Tue 23:00 ET. Details in TKT-4471: "
            "<https://example.com/tickets/TKT-4471|example.com/tickets/TKT-4471>",
            None,
        ),
        ("1791967191.066279", 2, "ack, pager is mine for that night", None),
        ("1791967192.254159", 2, f"<@{ME}> can you confirm the rollback plan is the same as July?", None),
        ("1791967193.334969", 2, "Correction: it's Wed 01:00 ET now, freeze stays Tue 17:00.", None),
        (
            "1791967194.627559",
            None,
            "*Release checklist* for 2026.08:\n• run `make migrate` on staging\n• confirm INC-0812 is closed\n"
            f"• post in <#{CAPTURED_IN}> when done",
            None,
        ),
    ]
    with world.batch():
        world.insert(
            "world_meta",
            [
                {"key": "company", "value": "Example Co"},
                {"key": "now_us", "value": str(us("1791967200.000000"))},
            ],
        )
        world.insert(
            "calendar",
            [
                dict(
                    day=1, date="2026-10-14", start_us=us("1791925200.000000"), end_us=us("1792011600.000000")
                )
            ],
        )
        world.insert(
            "users",
            [
                dict(
                    id=id,
                    handle=handle,
                    real_name=name,
                    display_name="",
                    email=f"{handle}@example.com",
                    title="",
                    tz="Etc/GMT-5",
                    is_bot=bot,
                    created_us=join - 1_000_000,
                )
                for id, handle, name, bot in people
            ],
        )
        world.insert(
            "channels",
            [
                dict(id=id, name=name, type=kind, creator_id=creator, created_us=at, purpose=purpose)
                for id, name, kind, _, creator, at, purpose in channels
            ],
        )
        world.insert(
            "members",
            [
                dict(channel_id=id, user_id=user, joined_us=join if id == CAPTURED_IN else join - 1_000_000)
                for id, _, _, users, *_ in channels
                for user in users
            ],
        )
        world.insert(
            "messages",
            [
                dict(
                    id=n,
                    channel_id=CAPTURED_IN,
                    user_id=ME,
                    ts_us=us(ts),
                    text=text,
                    parent_id=parent,
                    subtype=sub,
                )
                for n, (ts, parent, text, sub) in enumerate(messages, 1)
            ],
        )
        world.insert(
            "reactions",
            [
                dict(message_id=2, user_id=ME, emoji="eyes", created_us=root + 1_000_000),
                dict(message_id=2, user_id=ME, emoji="white_check_mark", created_us=root + 2_000_000),
                dict(message_id=3, user_id=ME, emoji="+1", created_us=us("1791967192.000000")),
            ],
        )
    world.close()
    return Workspace(world.path, actor=ME)


def unsigned(value):
    if isinstance(value, dict):
        return {k: unsigned(v) for k, v in value.items()}
    return SIGNED.sub("", value) if isinstance(value, str) else value


@pytest.mark.parametrize(
    "entry",
    [
        pytest.param(entry, marks=pytest.mark.xfail(strict=True, reason=DIFFERENT[n]))
        if n in DIFFERENT
        else entry
        for n, entry in enumerate(CAPTURE)
    ],
    ids=[f"{n:02d}-{entry['tool'].removeprefix('slack_')}" for n, entry in enumerate(CAPTURE)],
)
def test_reads_as_the_server(captured_in, entry):
    tool = entry["tool"].removeprefix("slack_")
    if "error" in entry:
        with pytest.raises(ValueError) as refused:
            slack.read(captured_in, tool, entry["arguments"])
        assert str(refused.value) == entry["error"]
    else:
        assert slack.render(captured_in, tool, slack.read(captured_in, tool, entry["arguments"])) == unsigned(
            entry["response"]
        )


def test_a_hit_reads_as_the_server(captured_in):
    """A hit listed above for what is around it (a reply sent to the channel too) reads as the server's up to its
    context: its text without bold and code marks."""
    entry = CAPTURE[27]
    have = slack.render(
        captured_in,
        "search_public_and_private",
        slack.read(captured_in, "search_public_and_private", entry["arguments"]),
    )
    assert (
        "*Release checklist*"
        in slack.read(captured_in, "read_channel", {"channel_id": CAPTURED_IN})["items"][0]["text"]
    )
    assert (
        have["results"].split("Context before")[0]
        == unsigned(entry["response"])["results"].split("Context before")[0]
    )


def test_what_the_capture_covers():
    """The capture holds a call of each read tool in each of its formats; search_public's only as an empty result (the
    capture's messages are in a private channel), which shares its hits' text with search_public_and_private."""
    served = json.loads((ROOT / "environments/worldgen_slack/capture/slack-mcp/tools.json").read_text())[
        "tools"
    ]
    formats = {
        (t["name"], option)
        for t in served
        for option in t["input_schema"]["properties"].get("response_format", {}).get("enum") or ["detailed"]
    }
    captured = {(e["tool"], e["arguments"].get("response_format", "detailed")) for e in CAPTURE}
    assert formats <= captured, sorted(formats - captured)
