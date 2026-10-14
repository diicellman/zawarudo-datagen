"""The task contract: gold as its actor, the reads it may make, its rank and its trial."""

from worldgen_slack.slack import PAGE
from worldgen_slack.world import SHADOWED, SHOWN

from ..evidence import measures
from ..forge.tasks import Task, check_task
from ..workspace import Workspace
from .support import contracts_settings, fails, forge_of, world_fixture


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
