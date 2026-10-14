"""The world: cast, premise and organization; apps, churn and links; the clock and a day's writing; the author's tools
and memory."""

import json
import math
import random
import re
import statistics
from collections import Counter
from datetime import date, datetime
from zoneinfo import ZoneInfo

import pytest
from worldgen_slack import slack

from ..config import Category, Needs
from ..evidence import IDENTIFIER
from ..forge.config import ForgeConfig, load_forge
from ..forge.difficulty import failure_cases, features, panel
from ..forge.proposer.task import ProposerTask
from ..forge.tasks import Task, record_task
from ..workspace import Workspace
from ..world.author.memory import files, ledger_digest
from ..world.author.setup import parse_premise
from ..world.author.task import AuthorTask
from ..world.board import Slot, needs_text
from ..world.cast import Gaps, Premise, cards, pick_cast, user_id
from ..world.clock import (
    DAY,
    HABITS,
    Moment,
    at,
    bounds,
    habit,
    part_of,
    present,
    render,
    reread,
    start_clock,
    today,
    window,
    world_meta,
)
from ..world.ledger import (
    Artifact,
    BoardEntry,
    Event,
    Move,
    Plan,
    PlanFact,
    Storyline,
    front_loaded,
    move_moment,
    record_plan,
)
from ..world.organization import (
    Channel,
    Integration,
    Organization,
    Person,
    Routine,
    UserGroup,
    background_plan,
    bot_id,
    channel_id,
    direct_messages,
    organize,
)
from ..world.text import TIME_LITERAL
from ..world.writing import (
    Close,
    Commit,
    Conversation,
    PostLine,
    Reaction,
    advance,
    agenda_budget,
    band,
    close_day,
    edit,
    opening,
    post,
    posted,
    promises_made,
    quotas,
    revise,
    room,
    share_band,
    tally,
    today_line,
)
from .support import (
    BANDS,
    TAXONOMY,
    author_context,
    contracts_settings,
    fails,
    forge_of,
    ledger_plan,
    mcp_client,
    needing,
    onward,
    organized,
    refused,
    seed_person,
    served_tools,
    with_state,
    world_settings,
)


def test_contracts(tmp_path):
    """Documents become rows only through deterministic rules: the cast, the premise, the organization, the tasks
    on a planned ledger (T2-T7), the agenda of everyday conversations; no literal times."""
    settings = contracts_settings(tmp_path)
    tasks = forge_of(settings)
    for wrong in ({"concepts": [["x"]]}, {"needs": [{}]}, {"concepts": [["x"], [], ["y"]]}):
        fails(Category.model_validate, settings.taxonomy["semantic"].model_dump() | wrong)
    toml = tmp_path / "forge.toml"
    toml.write_text(
        f'world = "{tmp_path}"\noutput = "{tmp_path / "forge"}"\ntaxonomy = "{TAXONOMY}"\nbands = {json.dumps(BANDS)}\n'
    )
    assert load_forge(toml).taxonomy == tasks.taxonomy, "a forge reads the taxonomy its config names"
    whole = tasks.model_dump(mode="json")
    fails(ForgeConfig.model_validate, whole | {"bands": whole["bands"][:2]})  # 3 levels
    fails(ForgeConfig.model_validate, whole | {"bands": [[0.5, 0.25]] * len(whole["bands"])})
    jobs = ["software_developer"] * 3 + ["accountant_or_auditor"] * 2 + ["manager"]
    people = [seed_person(i).model_copy(update={"occupation": job}) for i, job in enumerate(jobs)]
    (tmp_path / "p.jsonl").write_text("".join(p.model_dump_json() + "\n" for p in people))
    (tmp_path / "t.jsonl").write_text("".join(p.typing.model_dump_json() + "\n" for p in people))
    personas = settings.personas.model_copy(update={"pool": 3})
    staffing = {"software_developer": 2, "accountant_or_auditor": 1}
    drawn = pick_cast(personas, 3, ["Owen Sato"], staffing)
    assert Counter(p.occupation for p in drawn) == staffing and "Owen Sato" not in {p.name for p in drawn}
    assert drawn == pick_cast(personas, 3, ["Owen Sato"], staffing), "the seed decides the cast"
    fails(pick_cast, personas, 3, ["Owen Sato", "Alicia Rao"], staffing)  # one unused developer is left

    def premises(staffing):  # raw, as the synthesizer writes it
        return json.dumps(
            {
                "premises": [
                    dict(
                        company=f"{w} Co", niche="n", region="r", size="s", culture="c", cast="x", staffing=s
                    )
                    for w, s in (("Birch", {"software_developer": 2, "manager": 1}), ("Cedar", staffing))
                ]
            }
        )

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
    assert [card["timezone"] for card in cards(cast).values()] == [p.timezone for p in cast], (
        "the judge reads each clock"
    )
    assert {habit(f"U{i:03d}") for i in range(40)} == set(HABITS), "people write times in different ways"
    premise = Premise(
        company="Lattice Creek", niche="n", region="r", size="s", culture="c", cast="x", staffing={}
    )
    world = Workspace.create(tmp_path / "world.sqlite")

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
        ),  # d left out
        org(channels=[Channel(name="ops", type="public", members=[a, b, c, d], routines=routines[:2])]),
        org(dm_routines=routines[:2]),
        org(channels=[Channel(type="im", members=[a, b, c])]),
    ):
        # else organization accepted
        with pytest.raises(ValueError):
            with world.trial() as copy:
                organize(copy, bad, cast, premise, settings)
    assert world.db.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0, (
        "a failed trial leaves the world untouched"
    )
    with world.trial() as copy:
        organize(copy, org(), cast, premise, settings)
    dms = [sorted(m.members) for m in direct_messages(org(), settings)]
    assert dms == [sorted(m.members) for m in direct_messages(org(), settings)] and len(dms) == 1, (
        "code tops the author's DM up to dms_per_person, drawn by the seed"
    )
    assert (
        sorted([a, c]) not in dms
        and world.db.execute("SELECT COUNT(*) FROM channels WHERE type = 'im'").fetchone()[0] == 2
    )
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
        return Task(
            id=id,
            category=category,
            level=1,
            actor_id=c,
            question="Which way did the release go?",
            answer_type="text",
            gold_sql=sql,
            facts=list(facts),
        ).model_copy(update=extra)

    def recorded(task_, slot_, settings_=None, setup=lambda copy: None):
        """The tasks once this one is recorded into its slot, on a dropped trial."""
        try:
            with world.trial() as copy:
                setup(copy)
                record_task(copy, task_, settings_ or tasks, slot_)
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
        "T3 refusal with rows": (
            task(category="robustness", answer_type="refusal", sql="SELECT 1 AS answer", facts=()),
            slot("robustness", 1),
        ),
        "T3 too many rows": (task(answer_type="set", sql="SELECT value AS answer FROM facts"), found),
        "T3 text answer of 2 rows": (task(sql="SELECT value AS answer FROM facts LIMIT 2"), found),
        "T3 no answer column": (task(sql="SELECT value FROM facts WHERE id = 'f1'"), found),
        "T5 giveaway": (task(question="Was it a rollback for the release?"), found),
        "unreadable fact": (
            task(sql="SELECT value AS answer FROM facts WHERE id = 'f3'", facts=("f3",)),
            found,
        ),
    }
    for name, (task_, slot_) in rejected.items():
        # else task accepted: {name}
        with pytest.raises(ValueError):
            recorded(task_, slot_)
    directory = task(
        category="lookup", sql=f"SELECT real_name AS answer FROM users WHERE id = '{d}'", facts=()
    )
    assert "t1" in recorded(directory, slot("lookup", 1)), "a task may rest on no facts"
    # A gold query asks only about what the tools show: who reacted, never when (S3's reactions "made on Aug 18").
    for unshown in (
        "SELECT u.real_name AS answer FROM reactions r JOIN users u ON u.id = r.user_id WHERE r.created_us > 0",
        "SELECT real_name AS answer FROM users WHERE created_us > 0 LIMIT 1",
        "SELECT name AS answer FROM channels WHERE creator_id IS NOT NULL LIMIT 1",
    ):
        refused(
            "which no Slack tool shows",
            recorded,
            task(category="lookup", sql=unshown, facts=()),
            slot("lookup", 1),
        )
    nobody = task(
        category="robustness",
        answer_type="refusal",
        sql="SELECT id AS answer FROM users WHERE real_name = 'Nobody Here'",
        facts=(),
    )
    assert "t1" in recorded(nobody, slot("robustness", 1)), "nor a refusal"
    with world.trial() as copy:
        record_task(copy, search, tasks, found)
    assert json.loads(world.db.execute("SELECT gold_json FROM tasks").fetchone()[0]) == [
        {"answer": "rollback"}
    ]
    assert world.db.execute("SELECT concept FROM tasks").fetchone()[0] == "a concept", (
        "the slot's concept is stored"
    )
    fails(record_task, world, task(id="t2"), tasks, slot("search", 1, id="t2"))  # t1 asks it
    refused(
        "its answer differs for them",
        record_task,
        world,
        task(id="t2", actor_id=d),
        tasks,
        slot("search", 1, id="t2"),
    )  # d asks it too, and gets t1's answer

    def count(id, actor):
        return Task(
            id=id,
            category="lookup",
            level=1,
            actor_id=actor,
            question="How many private channels am I in?",
            answer_type="number",
            gold_sql="SELECT COUNT(*) AS answer FROM channels WHERE type = 'private'",
        )

    assert "t3" in recorded(
        count("t3", c),
        slot("lookup", 1, id="t3"),
        setup=lambda copy: record_task(copy, count("t2", a), tasks, slot("lookup", 1, id="t2")),
    ), "a perspective twin: another asker, another answer"

    def settled(facts):  # a level-3 semantic task needs its facts first stated in 2 channels
        sql = f"SELECT value AS answer FROM facts WHERE id = '{facts[-1]}'"
        return Task(
            id="t9",
            category="semantic",
            level=3,
            actor_id=a,
            question="What did the review settle?",
            answer_type="text",
            gold_sql=sql,
            facts=facts,
        )

    # A level's needs, those its facts give: f1, f2 and f4 are first stated in #ops, f3 in #leads; f2 comes after f1,
    # f4 after and over f3; a decoy counts where the task's actor can read it.
    nine = slot("semantic", 3, id="t9")

    def decoy(channel, id="f5", day=3, attribute="owner"):
        return lambda copy: copy.insert(
            "facts",
            [
                dict(
                    id=id,
                    storyline="s2",
                    subject="Audit",
                    attribute=attribute,
                    value="Ines",
                    channel_id=channel,
                    author_id=a,
                    day=day,
                    summary="s",
                    is_decoy=1,
                )
            ],
        )

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

    def measured_of(task_, slot_, setup=lambda copy: None):
        """A task's measures once it is recorded into its slot, on a dropped trial."""
        try:
            with world.trial() as copy:
                setup(copy)
                raise LookupError(record_task(copy, task_, tasks, slot_))
        except LookupError as out:
            return out.args[0]

    for facts, actor, setup, measure, want in (
        (["f1", "f4"], a, None, "channels", 1),
        (["f1", "f3"], a, None, "channels", 2),
        (["f3", "f4"], a, None, "relations", 2),
        (["f2", "f4"], c, decoy(ops), "decoys", 1),
        (
            ["f2", "f4"],
            c,
            corrected_unseen,
            "decoys",
            1,
        ),  # its correction is out of c's sight: it still misleads
        (["f2", "f4"], c, decoy(leads), "decoys", 0),  # c is not in #leads
        (["f2", "f4"], c, retracted(ops), "decoys", 0),  # f4 corrects it in plain sight
        (["f2", "f4"], c, decoy(ops, attribute="budget"), "decoys", 0),  # another attribute
        (["f4", "f3"], a, decoy(ops), "decoys", 0),  # it answers f3's window, not the owner
    ):
        asked = settled(facts).model_copy(update={"actor_id": actor})
        assert measured_of(asked, nine, setup or (lambda copy: None))[measure] == want, (facts, measure, want)
    asked = settled(["f2", "f4"]).model_copy(update={"actor_id": c})

    # A refusal rests on the truth its actor cannot see (f3, in #leads) beside the stale value it can (f7, in #ops).
    def stale(copy):
        return (
            copy.insert(
                "facts",
                [
                    dict(
                        id="f7",
                        storyline="s2",
                        subject="Audit",
                        attribute="window",
                        value="full test",
                        channel_id=ops,
                        author_id=a,
                        day=3,
                        summary="s",
                    )
                ],
            ),
            copy.insert("fact_relations", [dict(src_fact="f3", dst_fact="f7", kind="supersedes")]),
        )

    unseen = Task(
        id="t9",
        category="robustness",
        level=4,
        actor_id=c,
        question="When is the audit's full test?",
        answer_type="refusal",
        gold_sql="SELECT id AS answer FROM users WHERE real_name = 'Nobody Here'",
        facts=["f3", "f7"],
    )
    four = slot("robustness", 4, id="t9")
    assert measured_of(unseen, four, stale)["decoys"] == 1, "its near-miss misleads"
    assert measured_of(unseen.model_copy(update={"facts": ["f3"]}), four)["decoys"] == 0, (
        "no stale value in sight"
    )

    # A status answer gives the latest value its actor can see (f7, in #ops), which a fact out of their sight settles
    # (f3, in #leads): the useful answer gives it as not settled, where a refusal said nothing.
    status = Task(
        id="t9",
        category="robustness",
        level=3,
        actor_id=c,
        question="What is the audit's window?",
        answer_type="status",
        gold_sql="SELECT value AS answer FROM facts WHERE id = 'f7'",
        facts=["f3", "f7"],
    )
    three = slot("robustness", 3, id="t9")
    assert "t9" in recorded(status, three, tasks, stale)
    refused(
        "out of its actor's sight, that supersedes the value it gives",
        recorded,
        status.model_copy(update={"facts": ["f7"]}),
        three,
        tasks,
        stale,
    )  # nothing settles it
    refused(
        "which supersede the value it gives",
        recorded,
        status.model_copy(update={"actor_id": a}),
        three,
        tasks,
        stale,
    )  # a reads #leads: f3 is the answer
    refused(
        "gold gives the value of one of its facts its actor can read",
        recorded,
        status.model_copy(update={"gold_sql": "SELECT value AS answer FROM facts WHERE id = 'f3'"}),
        three,
        tasks,
        stale,
    )
    refused(
        "gold query reads the facts it answers from",
        recorded,
        status.model_copy(update={"gold_sql": "SELECT real_name AS answer FROM users LIMIT 1"}),
        three,
        tasks,
        stale,
    )

    # The wrong answers a task's grade must reject: the near-miss its actor can read, and a hedge with it.
    try:
        with world.trial() as copy:
            decoy(ops)(copy)
            copy.insert(
                "facts",
                [
                    dict(
                        id="f6",
                        storyline="s2",
                        subject="Audit",
                        attribute="owner",
                        value="Mia",
                        channel_id=leads,
                        author_id=a,
                        day=3,
                        summary="s",
                        is_decoy=1,
                    )
                ],
            )  # out of c's sight
            record_task(copy, asked, tasks, nine)
            raise LookupError(failure_cases(copy, "t9", [{"answer": "Owen"}]))
    except LookupError as out:
        gold, cases = out.args[0]
    assert (
        gold == "Owen."
        and ("the near-miss 'Ines'", "Ines.") in cases
        and ("a hedge", "Either Owen or Ines; I can't tell which.") in cases
    ), cases
    assert not [c for c in cases if "Mia" in c[1]], "a value its actor cannot see is no near-miss"

    # Lazy solvers over a chain: "full test" (f7, #ops) is settled as "dry run" (f3, #leads), then a decoy repeats
    # "rehearsal" in #ops. Taking the first or the latest value is wrong; the truth sits in a private channel, later.
    def chain(copy):
        stale(copy)
        copy.insert(
            "facts",
            [
                dict(
                    id="f8",
                    storyline="s2",
                    subject="Audit",
                    attribute="window",
                    value="rehearsal",
                    channel_id=ops,
                    author_id=b,
                    day=3,
                    summary="s",
                    is_decoy=1,
                )
            ],
        )
        (day3,) = copy.db.execute("SELECT start_us FROM calendar WHERE day = 3").fetchone()
        copy.db.execute("UPDATE world_meta SET value = ? WHERE key = 'now_us'", (str(day3 + 14_400_000_000),))
        said = copy.insert(
            "messages",
            [
                dict(channel_id=ch, ts_us=day3 + h * 3_600_000_000, user_id=u, text=t)
                for h, ch, u, t in (
                    (1, ops, a, "full test it is"),
                    (2, leads, a, "dry run it is"),
                    (3, ops, b, "rehearsal it is"),
                )
            ],
        )
        copy.insert(
            "evidence",
            [dict(fact_id=f, message_id=m, role="anchor") for f, m in zip(("f7", "f3", "f8"), said)],
        )

    window = Task(
        id="t9",
        category="semantic",
        level=3,
        actor_id=a,
        question="What is the audit's window?",
        answer_type="text",
        gold_sql="SELECT value AS answer FROM facts WHERE id = 'f3'",
        facts=["f3", "f7"],
    )
    try:
        with world.trial() as copy:
            chain(copy)
            record_task(copy, window, tasks, nine)
            count = Task(
                id="t8",
                category="lookup",
                level=1,
                actor_id=a,
                question="How many messages are there?",
                answer_type="number",
                gold_sql="SELECT COUNT(*) AS answer FROM messages",
            )
            counted = record_task(copy, count, tasks, slot("lookup", 1, id="t8"))["gold"]
            who = window.model_copy(
                update={
                    "id": "t7",
                    "question": "Who settled the audit's window?",
                    "gold_sql": "SELECT u.real_name AS answer FROM facts f JOIN users u ON u.id = f.author_id WHERE f.id = 'f3'",
                }
            )
            settler = record_task(copy, who, tasks, slot("semantic", 3, id="t7"))["gold"]
            raise LookupError(
                (
                    panel(copy, "t9", [{"answer": "dry run"}]),
                    features(copy, "t9", [{"answer": "dry run"}]),
                    features(copy, "t8", counted),
                    features(copy, "t7", settler),
                )
            )
    except LookupError as out:
        lazy, shape, counting, settled_by = out.args[0]
    assert settled_by["last_link"] == "private·later", "a person's answer is placed by the fact it is about"
    assert (
        {k: lazy["guesses"][k] for k in ("first", "latest", "top")}
        == {"first": "full test", "latest": "rehearsal", "top": "rehearsal"}
        and {"first", "latest", "top"} <= set(lazy["traps"])
        and not lazy["shortcut"]
    ), lazy
    assert (shape["changes"], shape["last_link"], shape["perspective"]) == (3, "private·later", False), shape
    assert counting["perspective"] and counting["derived"], (
        "a count differs by who asks, and no message states it"
    )

    def stated_unseen(copy):  # the truth, f3, is stated in #leads on day 3, where c cannot read it
        stale(copy)
        (day3,) = copy.db.execute("SELECT start_us FROM calendar WHERE day = 3").fetchone()
        copy.db.execute("UPDATE world_meta SET value = ? WHERE key = 'now_us'", (str(day3 + 7_200_000_000),))
        (truth,) = copy.insert(
            "messages", [dict(channel_id=leads, ts_us=day3 + 3_600_000_000, user_id=a, text="dry run it is")]
        )
        copy.insert("evidence", [dict(fact_id="f3", message_id=truth, role="anchor", anchor_token="dry run")])

    assert "t9" in recorded(status, three, tasks, stated_unseen), (
        "a status rests on its truth stated out of sight"
    )
    try:
        with world.trial() as copy:
            stated_unseen(copy)
            raise LookupError(record_task(copy, unseen, tasks, four))
    except LookupError as out:
        measured = out.args[0]
    assert (measured["unseen"], measured["evidence_pages"], measured["decoys"]) == (1, None, 1), (
        "evidence its actor cannot read is counted, and measured from no view"
    )
    hybrid = dict(
        category="hybrid",
        level=1,
        actor_id=c,
        answer_type="number",
        gold_sql="SELECT COUNT(*) AS answer FROM messages m, facts f WHERE f.id = 'f1'",
    )
    unread = Task(
        id="h1", question="How many messages surround the decision?", facts=["f3"], **hybrid
    )  # in leads
    fails(record_task, world, unread, tasks, slot("hybrid", 1, id="h1"))  # its actor cannot read its fact
    read = Task(id="h1", question="How many messages surround the rollback?", facts=["f1"], **hybrid)
    assert "h1" in recorded(read, slot("hybrid", 1, id="h1"))
    small = settings.activity.model_copy(update={"conversation_lines": 4})
    plan = background_plan(world, org(), small, 7, 2, 12)  # a day's agenda: 12 messages of everyday talk
    members = {(r[0], r[1]) for r in world.db.execute("SELECT channel_id, user_id FROM members")}
    assert plan == background_plan(world, org(), small, 7, 2, 12) and sum(s.length for s in plan) in (11, 12)

    def drawn(scenes):
        return [(x.channel_id, x.part, x.situation, x.length) for x in scenes]

    assert drawn(plan) != drawn(background_plan(world, org(), small, 7, 3, 12)), "each day draws its own"
    assert all(s.situation in {r.kind for r in routines} and s.day == 2 for s in plan)
    assert all((s.channel_id, p) in members for s in plan for p in s.participants)
    assert background_plan(world, org(), small, 7, 2, 0) == []
    direct = background_plan(world, org(), small.model_copy(update={"dm_share": 1.0}), 7, 2, 12)
    kinds = {r[0]: r[1] for r in world.db.execute("SELECT id, type FROM channels")}
    assert {kinds[s.channel_id] for s in direct} == {"im"}, "dm_share places conversations in DMs"
    skewed = background_plan(
        world, org(), small.model_copy(update={"dm_share": 0.0, "channel_skew": 4.0}), 7, 2, 60
    )
    places = Counter(kinds[s.channel_id] for s in skewed)
    assert places["public"] > 3 * places["private"], "busier channels draw more conversations"
    assert TIME_LITERAL.findall("at 3pm, 15:00, 2026-06-15 or Jun 15") == [
        "3pm",
        "15:00",
        "2026-06-15",
        "Jun 15",
    ]
    assert TIME_LITERAL.findall("release 8.14 on v2.10, maybe 2 people") == []
    assert IDENTIFIER.findall("PWSQL-03 to CU16 under CHG-2291, v2; release 4.2 at 10:00") == [
        "PWSQL-03",
        "CU16",
        "CHG-2291",
        "v2",
    ]
    # Every seat that runs code needs an immutable VM with no egress: the world's author, and the forge's proposer.
    for config, seat in ((settings, "author"), (tasks, "proposer")):
        agent = getattr(config.env, seat)
        loose = agent.model_copy(
            update={"runtime": agent.runtime.model_copy(update={"image": "python:3.12-slim"})}
        )
        refused(
            f"{seat} executes code, so it requires an immutable Prime VM",
            type(config).model_validate,
            config.model_dump(mode="json")
            | {"env": config.env.model_dump(mode="json") | {seat: loose.model_dump(mode="json")}},
        )


def test_apps(tmp_path):
    """Apps post by code (2026-10-14): the organization lists them, each a bot that is a member of its channel; an
    event one reports is posted at its moment with a title that states no fact; routine posts come on workdays, from
    the queue, in time order with everyone's messages; no quota, share, scene or task counts a bot."""
    settings = world_settings(
        tmp_path,
        activity={"messages": 12, "bot_posts": 3},
        author={"tolerance": 1.0, "share_tolerance": 1.0, "mention_tolerance": 1.0},
    )
    deploys = Integration(name="Deploy Bot", kind="deploys", channel="ops", subjects=["web", "api", "worker"])
    builds = Integration(name="CI", kind="ci", channel="ops", subjects=["main", "release", "hotfix"])
    for name, apps, why in (
        ("one", [deploys], "2 to 5 apps"),
        (
            "lost",
            [deploys, builds.model_copy(update={"channel": "nowhere"})],
            "one of the organization's channels",
        ),
        (
            "timed",
            [deploys, builds.model_copy(update={"subjects": ["main", "the 5pm build", "x"]})],
            "names no time",
        ),
    ):
        refused(why, organized, tmp_path / name, settings, apps)
    world, cast, org, (a, b, c, d) = organized(tmp_path / "world", settings, [deploys, builds])
    ops, leads, bot = (
        channel_id("public", "ops", []),
        channel_id("private", "leads", []),
        bot_id("Deploy Bot"),
    )
    assert world.db.execute(
        "SELECT u.is_bot, m.channel_id FROM users u JOIN members m ON m.user_id = u.id WHERE u.id = ?", (bot,)
    ).fetchone()[:] == (1, ops)
    with world.trial() as copy:
        start_clock(copy)
    released = Event(
        id="e3", storyline="s1", title="production release", day=2, time="16:00", post_by="Deploy Bot"
    )
    plan = ledger_plan(ops, leads, a, b)

    def recorded(document):
        with world.trial() as copy:
            record_plan(copy, document, settings)

    refused(
        "post_by names an app of the workspace",
        recorded,
        plan.model_copy(update={"events": [*plan.events, released.model_copy(update={"post_by": "Nobody"})]}),
    )
    refused(
        "states no fact or time: ['rollback']",
        recorded,
        plan.model_copy(
            update={"events": [*plan.events, released.model_copy(update={"title": "rollback shipped"})]}
        ),
    )
    recorded(plan.model_copy(update={"events": [*plan.events, released]}))
    assert world.db.execute("SELECT COUNT(*) FROM queue WHERE message_id IS NULL").fetchone()[0] > 1, (
        "posts wait for their moment"
    )
    assert not world.db.execute("SELECT 1 FROM messages").fetchone(), (
        "nothing is written before the present reaches it"
    )
    rng, gaps = random.Random(5), Gaps(settings.personas.gaps)
    lines = {
        1: [(ops, a, "we roll back, rollback decided", ["f1"])],
        2: [(ops, b, "service back at {at:e2}", ["f2"])],
        3: [(leads, a, "the dry run is set", ["f3"]), (ops, a, "Owen owns it from {at:e1}", ["f4"])],
    }
    while today(world) is not None and today(world) <= 7:
        day = today(world)
        if day in lines and part_of(world, present(world)) == ("evening" if day == 2 else "early"):
            for channel, author, text, conveys in lines.pop(day):
                post(
                    world,
                    Conversation(
                        channel_id=channel,
                        about="work",
                        lines=[PostLine(author_id=author, text=text, conveys=conveys)],
                    ),
                    settings,
                    rng,
                    gaps,
                )
        advance(world, settings, onward(world))
    assert world.violations() == [], "code's posts went in in time order"
    assert not world.db.execute(
        "SELECT 1 FROM queue WHERE message_id IS NULL AND due_us < ?", (present(world),)
    ).fetchone(), "the present wrote what came due"
    apps_ = [r[0] for r in world.db.execute("SELECT user_id FROM integrations")]
    marks = ", ".join("?" * len(apps_))
    posts = world.db.execute(
        f"SELECT m.ts_us, m.text, m.subtype, c.date FROM messages m JOIN calendar c ON m.ts_us >= c.start_us AND m.ts_us < c.end_us WHERE m.user_id IN ({marks})",
        apps_,
    ).fetchall()
    e3 = world.db.execute("SELECT moment_us FROM events WHERE id = 'e3'").fetchone()[0]
    assert [(p[0], p[1]) for p in posts if "production release" in p[1]] == [
        (e3, "✅ production release: deployed to production")
    ], "an event's post, at its moment"
    assert (
        posts
        and {p[2] for p in posts} == {"bot_message"}
        and not {date.fromisoformat(p[3]).weekday() for p in posts} & {5, 6}
    ), "weekends are quiet"
    people = world.db.execute(
        "SELECT COUNT(*) FROM messages WHERE user_id NOT IN (" + marks + ")", apps_
    ).fetchone()[0]
    assert (
        people
        == sum(posted(world, *bounds(world, d)) for d in range(1, 8))
        == sum(tally(world, d)["messages"] for d in range(1, 8))
    ), "no quota counts a bot"
    drawn = background_plan(world, org, settings.activity, 3, 2, 40)
    assert not {p for s in drawn for p in s.participants} & set(apps_), "no scene draws a bot"
    fails(
        record_task,
        world,
        Task(
            id="t9",
            category="lookup",
            level=1,
            actor_id=bot,
            question="q?",
            answer_type="number",
            gold_sql="SELECT COUNT(*) AS answer FROM messages",
        ),
        settings,
        Slot(id="t9", category="lookup", level=1, concept="c", style="s"),
    )
    reader = Workspace(world.path, actor=c)
    assert (
        not slack.read(reader, "search_public_and_private", {"keywords": ["production"]})["items"]
        and slack.read(
            reader, "search_public_and_private", {"keywords": ["production"], "include_bots": True}
        )["items"]
    ), "a search leaves apps out unless asked"
    assert (
        "✅ production release: deployed to production"
        in slack.render(reader, "read_channel", slack.read(reader, "read_channel", {"channel_id": ops}))[
            "messages"
        ]
    )
    reader.close()


def test_churn(tmp_path):
    """People come and go (2026-10-14): a plan's moves join and leave channels (Slack's posts of them in time order),
    take someone out of the company, archive a channel; recorded moves stay; no one posts where they are not, or in an
    archived channel; a fact's author is in its channel all its day; a status names no time."""
    settings = world_settings(
        tmp_path,
        activity={"messages": 12},
        author={"tolerance": 1.0, "share_tolerance": 1.0, "mention_tolerance": 1.0},
    )
    world, cast, org, (a, b, c, d) = organized(tmp_path / "world", settings)
    ops, leads = channel_id("public", "ops", []), channel_id("private", "leads", [])
    with world.trial() as copy:
        start_clock(copy)
    plan = ledger_plan(ops, leads, a, b)
    moves = [
        Move(kind="join", day=2, user_id=c, channel_id=leads),
        Move(kind="leave", day=4, user_id=b, channel_id=leads),
        Move(kind="depart", day=4, user_id=d),
        Move(kind="archive", day=5, channel_id=leads),
    ]

    def recorded(document):
        with world.trial() as copy:
            record_plan(copy, document, settings)

    for bad, why in (
        ([Move(kind="join", day=2, user_id=a, channel_id=leads)], "is a member of"),
        ([Move(kind="leave", day=2, user_id=c, channel_id=leads)], "is no member of"),
        ([Move(kind="join", day=2, user_id=c, channel_id="D0000000000")], "a named channel"),
        (
            [Move(kind="archive", day=3, channel_id=leads)],
            "is archived by day 3",
        ),  # f3 is stated in #leads on day 3
    ):
        refused(why, recorded, plan.model_copy(update={"moves": bad}))
    refused(
        "a member of",
        recorded,
        plan.model_copy(
            update={
                "moves": moves[:1],
                "facts": [*plan.facts[:2], plan.facts[2].model_copy(update={"author_id": c, "day": 2})],
            }
        ),
    )  # c joins #leads on day 2's morning: not all that day
    fails(Move, kind="depart", day=2, user_id=c, channel_id=leads)
    recorded(plan.model_copy(update={"moves": moves}))
    refused("recorded moves stay", recorded, plan.model_copy(update={"moves": moves[1:]}))
    rng, gaps = random.Random(5), Gaps(settings.personas.gaps)
    lines = {
        1: [(ops, a, "we roll back, rollback decided", ["f1"])],
        2: [(ops, b, "service back at {at:e2}", ["f2"])],
        3: [(leads, a, "the dry run is set", ["f3"]), (ops, a, "Owen owns it from {at:e1}", ["f4"])],
    }
    while today(world) is not None and today(world) <= 6:
        day = today(world)
        if day in lines and part_of(world, present(world)) == ("evening" if day == 2 else "early"):
            for channel, author, text, conveys in lines.pop(day):
                post(
                    world,
                    Conversation(
                        channel_id=channel,
                        about="work",
                        lines=[PostLine(author_id=author, text=text, conveys=conveys)],
                    ),
                    settings,
                    rng,
                    gaps,
                )
        if day == 3 and part_of(world, present(world)) == "afternoon":  # c, in #leads since day 2, says hello
            post(
                world,
                Conversation(
                    channel_id=leads, about="hi", lines=[PostLine(author_id=c, text="glad to be here")]
                ),
                settings,
                rng,
                gaps,
            )
        if day == 5 and part_of(world, present(world)) == "evening":
            refused(
                "archived today",
                post,
                world,
                Conversation(channel_id=leads, about="late", lines=[PostLine(author_id=a, text="anyone?")]),
                settings,
                rng,
                gaps,
            )
        advance(world, settings, onward(world))
    assert world.violations() == []
    said = {
        r["subtype"]: (r["user_id"], r["ts_us"])
        for r in world.db.execute("SELECT * FROM messages WHERE subtype IS NOT NULL")
    }
    assert said == {
        "channel_join": (c, move_moment(world, moves[0])),
        "channel_leave": (b, move_moment(world, moves[1]) - 1),
    }, said
    assert world.db.execute(
        "SELECT is_deleted, deactivated_us IS NOT NULL FROM users WHERE id = ?", (d,)
    ).fetchone()[:] == (1, 1)
    assert not world.db.execute(
        "SELECT 1 FROM members WHERE user_id = ? AND left_us IS NULL", (d,)
    ).fetchone(), "out of every conversation"
    assert world.db.execute("SELECT is_archived FROM channels WHERE id = ?", (leads,)).fetchone()[0] == 1
    scenes = background_plan(world, org, settings.activity, 3, 6, 40)
    assert (
        scenes
        and not {s.channel_id for s in scenes} & {leads}
        and not {p for s in scenes for p in s.participants} & {d}
    ), "no scene where it cannot be"
    person = org.people[0].model_copy(update={"status_emoji": ":palm_tree:", "status_text": "out until 5pm"})
    refused(
        "a status names no time",
        lambda: organize(
            Workspace.create(tmp_path / "status.sqlite"),
            org.model_copy(update={"people": [person, *org.people[1:]]}),
            cast,
            Premise(
                company="Lattice Creek", niche="n", region="r", size="s", culture="c", cast="x", staffing={}
            ),
            settings,
        ),
    )
    reader = Workspace(world.path, actor=a)
    history = slack.render(reader, "read_channel", slack.read(reader, "read_channel", {"channel_id": leads}))[
        "messages"
    ]
    assert (
        f"<@{c}|{world.db.execute('SELECT display_name FROM users WHERE id = ?', (c,)).fetchone()[0]}> has joined the channel"
        in history
    ), history
    reader.close()


def test_links(tmp_path):
    """Links, groups and edits are code's (2026-10-14): an artifact of the plan has its URL by code and is linked as
    {link:id} once it exists, a ticket app announces a ticket, a group is mentioned as {group:handle}; a raw URL is
    refused; a person edits their own message of today, and Slack marks it."""
    settings = world_settings(
        tmp_path,
        activity={"messages": 12},
        author={"tolerance": 1.0, "share_tolerance": 1.0, "mention_tolerance": 1.0},
    )
    tickets = Integration(
        name="Tracker",
        kind="tickets",
        channel="ops",
        subjects=["login bug", "slow page", "export"],
        prefix="TKT",
    )
    cast_ids = [user_id(p.uuid) for p in [seed_person(i, "America/Chicago") for i in range(2)]]
    world, cast, org, (a, b, c, d) = organized(
        tmp_path / "world",
        settings,
        [tickets],
        [UserGroup(handle="oncall", name="On call", members=cast_ids)],
    )
    ops, leads = channel_id("public", "ops", []), channel_id("private", "leads", [])
    with world.trial() as copy:
        start_clock(copy)
    plan = ledger_plan(ops, leads, a, b)
    runbook = Artifact(id="TKT-1", kind="ticket", title="audit runbook", owner_id=b, day=2)

    def recorded(document):
        with world.trial() as copy:
            record_plan(copy, document, settings)

    refused(
        "states no fact or time: ['rollback']",
        recorded,
        plan.model_copy(update={"artifacts": [runbook.model_copy(update={"title": "rollback steps"})]}),
    )
    refused(
        "a person of the company",
        recorded,
        plan.model_copy(update={"artifacts": [runbook.model_copy(update={"owner_id": bot_id("Tracker")})]}),
    )
    recorded(plan.model_copy(update={"artifacts": [runbook]}))
    refused("recorded artifacts stay", recorded, plan)
    url = world.db.execute("SELECT url FROM artifacts WHERE id = 'TKT-1'").fetchone()[0]
    assert url == "https://tickets.lattice-creek.example/browse/TKT-1".replace(
        "lattice-creek", "latticecreek"
    ), url
    rng, gaps = random.Random(5), Gaps(settings.personas.gaps)

    def talk(text, author=a, channel=ops, conveys=()):
        return post(
            world,
            Conversation(
                channel_id=channel,
                about="work",
                lines=[PostLine(author_id=author, text=text, conveys=list(conveys))],
            ),
            settings,
            rng,
            gaps,
        )

    refused("does not exist yet", talk, "see {link:TKT-1}")
    refused("names an artifact of the plan", talk, "see {link:TKT-9}")
    refused("code writes its URL", talk, "see https://tickets.latticecreek.example/browse/TKT-1")
    refused("names a usergroup's handle", talk, "{group:nobody} please look")
    first = talk("{group:oncall} heads up, rollback decided", conveys=["f1"])["messages"][0]
    assert (
        world.db.execute("SELECT text FROM messages WHERE id = ?", (first["id"],))
        .fetchone()[0]
        .startswith("<!subteam^S")
    ), "a group as Slack writes it"
    assert not world.db.execute(
        "SELECT 1 FROM message_mentions WHERE message_id = ?", (first["id"],)
    ).fetchone(), "a group is no person's mention"
    edited = edit(world, first["id"], "{group:oncall} heads up: rollback decided")
    assert (
        world.db.execute("SELECT edited_us FROM messages WHERE id = ?", (first["id"],)).fetchone()[0]
        == present(world)
        and "rollback decided" in edited["text"]
    )
    while today(world) == 1:
        advance(world, settings, onward(world))
    refused("posted earlier today", edit, world, first["id"], "rollback decided, sorry")
    while part_of(world, present(world)) != "afternoon":
        advance(world, settings, onward(world))
    linked_line = talk("runbook is {link:TKT-1}", author=b)["messages"][0]
    assert (
        f"<{url}|TKT-1>"
        in world.db.execute("SELECT text FROM messages WHERE id = ?", (linked_line["id"],)).fetchone()[0]
    )
    announced = world.db.execute(
        "SELECT m.text, m.user_id FROM messages m WHERE m.subtype = 'bot_message' AND m.text LIKE ?",
        (f"<{url}|TKT-1> created%",),
    ).fetchall()
    assert [tuple(r) for r in announced] == [
        (f"<{url}|TKT-1> created: audit runbook · assigned to <@{b}>", bot_id("Tracker"))
    ], announced
    reader = Workspace(world.path, actor=c)
    assert (
        "rollback decided (edited)"
        in slack.render(reader, "read_channel", slack.read(reader, "read_channel", {"channel_id": ops}))[
            "messages"
        ]
    ), "an edit is marked"
    reader.close()


def test_clock(tmp_path):
    """The world in time order (v7): a plan whose events never move and whose facts share their event's moment;
    conversations posted only at the present, their statements tagged and first made where the plan says; promises
    opened and closed; days closed only when their quota, facts and promises are settled; fixes that change no task;
    tasks written once the calendar is closed."""
    settings = world_settings(
        tmp_path, activity={"messages": 12}, author={"tolerance": 1.0, "share_tolerance": 1.0}
    )
    gaps = Gaps(settings.personas.gaps)
    world, cast, org, (a, b, c, d) = organized(tmp_path, settings)
    ops, leads = channel_id("public", "ops", []), channel_id("private", "leads", [])
    with world.trial() as copy:
        start_clock(copy)
    assert today(world) == 1 and present(world) == window(world, 1, "early")[0]
    shares = quotas(world, settings)
    assert sum(q["messages"] for q in shares.values()) == 12 and all(
        sum(q["parts"].values()) == q["messages"] for q in shares.values()
    )
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
        (
            "repeated ['f1']",
            plan(events=[*events, Event(id="f1", storyline="s1", title="clash", day=1, time="09:00")]),
        ),
        (
            "a title has no time",
            plan(events=[events[0].model_copy(update={"title": "window at 10:00"}), events[1]]),
        ),
        ("event e9 is not in events", plan(facts=[*facts[:3], facts[3].model_copy(update={"event": "e9"})])),
        (
            f"is a member of {leads}",
            plan(facts=[*facts[:2], facts[2].model_copy(update={"author_id": c}), facts[3]]),
        ),
        (
            "on or after the day of e2",
            plan(facts=[facts[0], facts[1].model_copy(update={"day": 1}), *facts[2:]]),
        ),
        ("on a later day", plan(facts=[facts[0].model_copy(update={"day": 3}), *facts[1:]])),  # f2 follows f1
        (
            "an anchor is words of its value",
            plan(facts=[facts[0].model_copy(update={"anchor": "Release 4.2"}), *facts[1:]]),
        ),
        (
            "f3 and f4 have one anchor, 'dry run'",
            plan(facts=[*facts[:3], facts[3].model_copy(update={"value": "dry run", "anchor": "dry run"})]),
        ),
        (
            "contains f3's, 'dry run'",
            plan(
                facts=[
                    *facts[:3],
                    facts[3].model_copy(
                        update={
                            "value": "the dry run owner",
                            "anchor": "dry run owner",
                            "after": [],
                            "supersedes": None,
                        }
                    ),
                ]
            ),
        ),
        (
            "f1 comes after itself, through f1 > f2 > f1",
            plan(facts=[facts[0].model_copy(update={"after": ["f2"]}), *facts[1:]]),
        ),
    ):
        refused(expected, recorded, bad)
    try:  # an anchor may contain another's when its fact comes after that one
        with world.trial() as copy:
            record_plan(
                copy,
                plan(
                    facts=[
                        *facts[:3],
                        facts[3].model_copy(update={"value": "the dry run owner", "anchor": "dry run owner"}),
                    ]
                ),
                settings,
            )
            raise LookupError("drop the trial")
    except LookupError:
        pass
    try:  # a Saturday with a planned event weighs as a workday; the first plan fixes those days, the total stays
        with world.trial() as copy:
            window_event = Event(id="e3", storyline="s2", title="maintenance window", day=6, time="22:00")
            record_plan(copy, plan(events=[*events, window_event]), settings)
            busy = quotas(copy, settings)
            assert (
                busy[6]["messages"]
                >= min(busy[d]["messages"] for d in range(1, 6))
                > shares[6]["messages"]
                == busy[7]["messages"]
            ), busy
            assert sum(q["messages"] for q in busy.values()) == 12
            cutover = Event(id="e4", storyline="s1", title="cutover", day=7, time="09:00")
            record_plan(copy, plan(events=[*events, window_event, cutover]), settings)
            assert quotas(copy, settings) == busy, "a later plan's event days leave the quotas as they were"
            raise LookupError("drop the trial")
    except LookupError:
        pass
    recorded(first_plan)
    shared = world.db.execute(
        "SELECT DISTINCT f.moment_us, e.moment_us FROM facts f JOIN events e ON e.id = f.event_id WHERE e.id = 'e1'"
    ).fetchall()
    assert len(shared) == 1 and shared[0][0] == shared[0][1] == at(
        world, Moment(day=3, time="10:00", zone=zone)
    ), "facts share their event's moment"
    refused(
        "an event never moves",
        recorded,
        plan(events=[events[0].model_copy(update={"time": "11:00"}), events[1]]),
    )
    refused("planned events stay", recorded, plan(events=events[:1]))
    refused(
        "keep ['s1', 's2']",
        recorded,
        plan(storylines=[Storyline(id="s3", summary="x"), Storyline(id="s2", summary="y")]),
    )
    # Facts run through the calendar: days 8-10 hold a quarter of the messages, so of 8 facts they first state 1.
    extra = [
        PlanFact(
            id=f"f{k}",
            storyline="s1",
            subject=f"Topic {k}",
            attribute="state",
            value=f"state {k}",
            channel_id=ops,
            author_id=a,
            day=2,
            summary="s",
        )
        for k in range(5, 9)
    ]
    refused(
        "days 8-10 first state 0 of the plan's 8 facts; they hold 25% of the world's messages, so they first state at least 1",
        recorded,
        plan(facts=[*facts, *extra]),
    )
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
    board_slots = [
        Slot(id=i, category=c, level=n, concept="a concept", style="a style")
        for i, c, n in (("t1", "semantic", 3), ("t3", "search", 1), ("t2", "lookup", 1))
    ]
    spread = needing(settings, "semantic", 3, channels=2)

    def entries(*pairs):
        return [BoardEntry(slot=slot, facts=ids) for slot, ids in pairs]

    def boarded(document):
        with world.trial() as copy:
            record_plan(copy, document, spread, board_slots)

    for expected, board in (
        ("each slot that rests on planned facts will rest on; ['t1', 't3'] have none", []),
        (
            "the slots of now.md that rest on planned facts; ['t2'] do not",
            entries(("t1", ["f3", "f4"]), ("t3", ["f4"]), ("t2", ["f1"])),
        ),
        ("the board names facts of the plan; ['f9'] are not", entries(("t1", ["f3", "f9"]), ("t3", ["f4"]))),
        (
            "the board's t1 (level-3 semantic) needs its facts first stated in at least 2 channels (they are in 1)",
            entries(("t1", ["f1", "f4"]), ("t3", ["f4"])),
        ),
    ):
        refused(expected, boarded, plan(board=board))
    # A robustness level-4 slot rests on planned facts too: a near-miss its asker will see, the truth out of sight.
    robust = [
        *board_slots,
        Slot(id="t4", category="robustness", level=4, concept="a concept", style="a style"),
    ]
    near = PlanFact(
        id="f5",
        storyline="s2",
        subject="Audit",
        attribute="owner",
        value="Ines",
        channel_id=ops,
        author_id=a,
        day=3,
        decoy=True,
        summary="s",
    )

    def boarded_robust(document):
        with world.trial() as copy:
            record_plan(copy, document, needing(spread, "robustness", 4, decoys=1), robust)

    usual = entries(("t1", ["f3", "f4"]), ("t3", ["f4"]))
    refused("['t4'] have none", boarded_robust, plan(board=usual))
    refused(
        "the board's t4 (level-4 robustness) needs at least 1 near-misses",
        boarded_robust,
        plan(board=[*usual, *entries(("t4", ["f4"]))]),
    )
    boarded_robust(plan(facts=[*facts, near], board=[*usual, *entries(("t4", ["f4", "f5"]))]))
    # Relations among a slot's facts: f2 comes after f1; f4 comes after f3 and supersedes it.
    related = needing(settings, "semantic", 3, relations=2)

    def boarded_related(document):
        with world.trial() as copy:
            record_plan(copy, document, related, board_slots)

    refused(
        "the board's t1 (level-3 semantic) needs at least 2 supersedes or after relations among its facts (it has 1)",
        boarded_related,
        plan(board=entries(("t1", ["f1", "f2"]), ("t3", ["f4"]))),
    )
    boarded_related(plan(board=usual))
    # The needs the author reads for a slot are only those its level sets.
    assert (
        needs_text(Needs(values=2, hidden=True))
        == "its answer in a thread reply, a DM, or later than its near-misses; its answer's subject and attribute taking 2 or more values, its answer among them"
    )
    try:  # facts may be shared; a replan's board replaces the last; a fact the board rests on stays planned
        with world.trial() as copy:
            record_plan(copy, plan(board=entries(("t1", ["f3", "f4"]), ("t3", ["f4"]))), spread, board_slots)
            record_plan(copy, plan(board=entries(("t1", ["f1", "f3"]), ("t3", ["f1"]))), spread, board_slots)
            assert [tuple(r) for r in copy.db.execute("SELECT slot, fact_id FROM board ORDER BY 1, 2")] == [
                ("t1", "f1"),
                ("t1", "f3"),
                ("t3", "f1"),
            ]
            dropped = plan(facts=facts[:3], board=entries(("t1", ["f3", "f4"]), ("t3", ["f1"])))
            refused(
                "the board names facts of the plan; ['f4'] are not",
                record_plan,
                copy,
                dropped,
                spread,
                board_slots,
            )
            raise LookupError("drop the trial")
    except LookupError:
        pass
    rng = random.Random(5)

    slots = [
        Slot(id="t1", category="semantic", level=3, concept="a concept", style="a style"),
        Slot(id="t2", category="lookup", level=1, concept="a concept", style="a style"),
    ]
    owner = Task(
        id="t1",
        category="semantic",
        level=3,
        actor_id=a,
        question="Who owns the audit after the dry run?",
        answer_type="text",
        gold_sql="SELECT value AS answer FROM facts WHERE id = 'f4'",
        facts=["f3", "f4"],
    )

    def tasked(task, setup=lambda copy: None):
        """A task written into its slot of the closed world, as the forge writes one: it replaces the slot's task."""
        slot = next(s for s in slots if s.id == task.id)
        with world.trial() as copy:
            setup(copy)
            copy.db.execute("DELETE FROM task_facts WHERE task_id = ?", (task.id,))
            copy.db.execute("DELETE FROM tasks WHERE id = ?", (task.id,))
            return record_task(copy, task, forge_of(settings), slot)

    def posted_(conversation):
        with world.trial() as copy:
            return post(copy, conversation, settings, rng, gaps)

    def talk(*lines, channel=ops, **extra):
        return Conversation(
            channel_id=channel, about="work", lines=[PostLine(**line) for line in lines], **extra
        )

    decided = talk(
        dict(author_id=a, text=f"we go with the rollback, <@{c}> fyi", conveys=["f1"]),
        dict(
            author_id=b,
            text="ok, I'll post the notes",
            reply_to=0,
            commits=[Commit(id="c1", text="post the notes", due_day=2)],
            reactions=[Reaction(user_id=a, emoji="eyes")],
        ),
        dict(author_id=b, text="rollback noted", reply_to=0, conveys=["f1"]),
    )
    for expected, bad in (
        ("write times and dates only", talk(dict(author_id=a, text="rollback at 3pm", conveys=["f1"]))),
        ("reply_to names an earlier line", talk(dict(author_id=a, text="ok", reply_to=0))),
        (
            "conveys lists only facts",
            talk(dict(author_id=a, text="rollback is decided", conveys=["f1", "f9"])),
        ),
        ("{at:...} names a moment", talk(dict(author_id=a, text="see you at {at:lunch}"))),
        ("list them in conveys", talk(dict(author_id=a, text="rollback is decided"))),
        ("f1 is first stated by", talk(dict(author_id=b, text="rollback is decided", conveys=["f1"]))),
        (
            "f3 is first stated by",
            talk(dict(author_id=a, text="dry run it is", conveys=["f3"]), channel=leads),
        ),
        (
            "nobody reacts to their own line",
            talk(dict(author_id=a, text="ok", reactions=[Reaction(user_id=a, emoji="eyes")])),
        ),
        (
            "from a member of its channel",
            talk(dict(author_id=a, text="ok", reactions=[Reaction(user_id=c, emoji="eyes")]), channel=leads),
        ),
        ("not a member of the channel", talk(dict(author_id=c, text="hi"), channel=leads)),
        (
            "c9 is no open commitment",
            talk(dict(author_id=a, text="ok", closes=[Close(id="c9", status="kept")])),
        ),
        (
            "due today or on a later day",
            talk(dict(author_id=a, text="ok", commits=[Commit(id="c2", text="t", due_day=11)])),
        ),
    ):
        refused(expected, posted_, bad)
    before = present(world)
    out = posted_(decided)
    rows = world.db.execute("SELECT id, ts_us, parent_id FROM messages ORDER BY id").fetchall()
    assert (
        rows[0]["ts_us"] > before
        and present(world) == rows[-1]["ts_us"]
        and rows[1]["parent_id"] == rows[0]["id"]
    )
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
    noted = world.db.execute(
        "SELECT ts_us, parent_id FROM messages WHERE id = ?", (again["messages"][0]["id"],)
    ).fetchone()
    assert noted["ts_us"] > rows[-1]["ts_us"] and noted["parent_id"] == rows[0]["id"], (
        "a later post comes later"
    )
    refused("earlier top-level message", posted_, talk(dict(author_id=c, text="noted"), thread=rows[1]["id"]))
    refused(
        "leave reply_to empty",
        posted_,
        talk(dict(author_id=c, text="a"), dict(author_id=a, text="b", reply_to=0), thread=rows[0]["id"]),
    )
    refused("past midnight", posted_, talk(*[dict(author_id=a, text="still here", pause="hours")] * 20))
    # else a message was written into the past
    with pytest.raises(ValueError, match=re.escape("written_out_of_order")):
        with world.batch():
            world.insert(
                "messages", [dict(channel_id=ops, ts_us=rows[0]["ts_us"] - 1, user_id=a, text="late")]
            )

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
    big = settings.model_copy(
        update={
            "activity": settings.activity.model_copy(update={"messages": 120}),
            "author": settings.author.model_copy(update={"tolerance": 0.3, "share_tolerance": 0.1}),
        }
    )
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
                copy.db.execute(
                    "INSERT INTO scenes (id, channel_id, day, part, slot_start_us, slot_end_us, situation) VALUES (?, ?, 2, 'morning', ?, ?, 'work')",
                    (f"p{k}", ops, stamp, stamp + 1),
                )
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
        PlanFact(
            id="f6",
            storyline="s1",
            subject="Release 4.2",
            attribute="owner",
            value="Mia",
            channel_id=leads,
            author_id=a,
            day=2,
            summary="s",
        ),
        PlanFact(
            id="f7",
            storyline="s1",
            subject="Release 4.2",
            attribute="reviewer",
            value="Raj",
            channel_id=ops,
            author_id=c,
            day=2,
            summary="s",
        ),
    ]

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
    restored = posted_(
        talk(
            dict(
                author_id=b,
                text="service back at {at:e2}, notes are up",
                conveys=["f2"],
                closes=[Close(id="c1", status="kept")],
            )
        )
    )
    moment = at(world, Moment(day=2, time="14:00", zone=zone))
    said = world.db.execute(
        "SELECT ts_us FROM messages WHERE id = ?", (restored["messages"][0]["id"],)
    ).fetchone()[0]
    written = {"12h": "2pm", "12h spaced": "2 PM", "24h": "14:00"}[habit(b)]
    assert restored["messages"][0]["text"] == f"service back at {written}, notes are up", (
        "{at:} in the author's habit, on their clock",
        restored,
    )
    assert reread(written, zone, said) == moment, "it reads back to the moment"
    # A world kept in the clock style (written before 2026-10-14) keeps it: chapters on it read alike.
    assert render(moment, "UTC", moment) == "19:00 UTC" and render(moment, "UTC", moment + DAY).startswith(
        f"{datetime.fromtimestamp(moment / 1e6, ZoneInfo('UTC')):%a}"
    ), "the day is named when it is not the day of writing"
    # Every habit's every form is a literal time (an author cannot type one) and reads back to its exact moment; the
    # zone shows only when the writer's clock is not the company's.
    for way in HABITS:
        for hours in (
            -24 * 9,
            -24 * 3,
            -24,
            -2,
            0,
            3,
            10,
            24 + 2.5,
            24 * 4,
            24 * 12,
        ):  # 12pm and 12am among them
            for minute in (0, 0.5):
                when = moment + int((hours + minute) * 3_600_000_000)
                for place, home in ((zone, zone), ("America/New_York", zone)):
                    text = render(when, place, moment, way, home)
                    assert TIME_LITERAL.search(text) and reread(text, place, moment) == when, (
                        way,
                        hours,
                        text,
                    )
                    assert (" E" in text) == (place != home), text
    assert (
        render(moment - 2 * 3_600_000_000, zone, moment, "12h", zone) == "12pm"
        and render(moment + 24 * 3_600_000_000, zone, moment, "24h", zone) == "tomorrow 14:00"
    )
    refused(
        "names its own day", posted_, talk(dict(author_id=b, text="see you tomorrow at {at:e2}", conveys=[]))
    )
    assert not IDENTIFIER.findall("the 5pm sync, 9am standup") and IDENTIFIER.findall("CHG-2291 at 5pm") == [
        "CHG-2291"
    ]
    assert world.db.execute("SELECT status, closed_by FROM commitments").fetchone()[:] == (
        "kept",
        restored["messages"][0]["id"],
    )
    refused(
        "only its summary may change",
        recorded,
        plan(facts=[facts[0].model_copy(update={"value": "roll forward", "anchor": None}), *facts[1:]]),
    )
    refused(
        "only its summary may change",
        recorded,
        plan(facts=[facts[0], facts[1].model_copy(update={"after": []}), *facts[2:]]),
    )
    recorded(plan(facts=[facts[0].model_copy(update={"summary": "decided early"}), *facts[1:]]))
    assert world.db.execute("SELECT summary FROM facts WHERE id = 'f1'").fetchone()[0] == "decided early"
    refused(
        "today (day 2) or later", recorded, plan(facts=[*facts[:3], facts[3].model_copy(update={"day": 1})])
    )
    refused(
        "on or before the day of e1",
        recorded,
        plan(facts=[*facts[:2], facts[2].model_copy(update={"day": 4}), facts[3]]),
    )
    notes = PlanFact(
        id="f5",
        storyline="s1",
        subject="Release 4.2",
        attribute="notes",
        value="notes posted",
        channel_id=ops,
        author_id=b,
        day=2,
        event="e2",
        kind="scheduled",
        summary="s",
    )
    refused("e2 is past", recorded, plan(facts=[*facts, notes]))  # it is evening; e2 was at 14:00
    advanced(to="night")
    advanced(to="tomorrow")  # day 3, early: e1 at 10:00 is ahead, and f3 and f4 are to be told before it
    refused("are scheduled before", advanced, to="night")
    assert {e.split()[0] for e in close_day(world, settings, 3)} == {"f3", "f4"}, (
        "planned for day 3, unstated"
    )
    refused(
        "comes after ['f3']",
        posted_,
        talk(dict(author_id=a, text="Owen owns the audit at {at:e1}", conveys=["f4"])),
    )
    refused(
        "past the moment of ['f3', 'f4']",
        posted_,
        talk(*[dict(author_id=a, text="still here", pause="hours")] * 5, channel=leads),
    )
    posted_(  # a scheduled fact is told before its moment
        talk(
            dict(author_id=a, text="dry run for the release window at {at:e1}", conveys=["f3"]), channel=leads
        )
    )
    # A fact about an event and with no anchor of its own is stated with the event's time.
    posted_(
        talk(
            dict(author_id=a, text="Owen owns the audit at {at:e1}", conveys=["f4"]),
            dict(author_id=d, text="thanks"),
        )
    )
    advanced(to="night")

    def advanced_with(cfg):
        with world.trial() as copy:
            return advance(copy, cfg, "tomorrow")

    refused("day 3 stays open", advanced_with, strict)
    while today(world) is not None:
        advanced(to=onward(world))
    refused("the calendar is closed", posted_, talk(dict(author_id=a, text="late")))
    refused("the calendar is closed", recorded, first_plan)
    measured = close_day(
        world,
        settings.model_copy(update={"author": settings.author.model_copy(update={"share_tolerance": 0.0})}),
        1,
    )
    assert any("3 of today's 4 messages are thread replies" in e for e in measured), measured
    promising = settings.model_copy(
        update={"activity": settings.activity.model_copy(update={"promise_rate": 0.5})}
    )
    assert any(
        e == f"{promises_made(world, 1)} promises made today; the day holds at least 2 (a line's commits)"
        for e in close_day(world, promising, 1)
    ), "a day makes its promises"
    refused("t1 is a semantic level 3 task", tasked, owner.model_copy(update={"level": 2}))
    result = tasked(owner)
    assert result["gold"] == [{"answer": "Owen"}] and result["bm25_rank"] and result["channels"] == 2, result
    assert "depth_share" in result and result["hidden"] is False, "rank and depth are measured, and reported"
    # What its messages give: where its answer is stated beside its near-misses, what the question names.
    assert tasked(owner.model_copy(update={"facts": ["f1", "f4"]}))["channels"] == 1, "both in #ops"

    def earlier(copy):
        return copy.insert(
            "facts",
            [
                dict(
                    id="f8",
                    storyline="s2",
                    subject="Audit",
                    attribute="owner",
                    value="Ines",
                    channel_id=ops,
                    author_id=a,
                    day=2,
                    summary="s",
                    is_decoy=1,
                )
            ],
        )

    assert tasked(owner, earlier)["hidden"], "stated after every near-miss"
    leads_named = owner.model_copy(update={"question": "Who owns the audit after the dry run in #leads?"})
    assert tasked(leads_named)["named"] == 1
    tasked(owner)
    replier = Task(
        id="t2",
        category="lookup",
        level=1,
        actor_id=c,
        question="Who said thanks in #ops?",
        answer_type="text",
        gold_sql="SELECT u.real_name AS answer FROM messages m JOIN users u ON u.id = m.user_id WHERE m.text = 'thanks'",
    )
    tasked(replier.model_copy(update={"question": "Who said thanks?"}))
    tasked(replier)  # the slot's task is replaced
    assert [tuple(r) for r in world.db.execute("SELECT id, question FROM tasks ORDER BY id")] == [
        ("t1", owner.question),
        ("t2", replier.question),
    ]
    thanks = world.db.execute("SELECT id FROM messages WHERE text = 'thanks'").fetchone()[0]
    stated = world.db.execute("SELECT message_id FROM evidence WHERE fact_id = 'f4'").fetchone()[0]

    def revised(message, text):
        with world.trial() as copy:
            return revise(copy, message, text)

    refused(
        "anchor_token_missing", revised, stated, "the audit has an owner"
    )  # f4's evidence keeps its token
    refused("states nothing new", revised, thanks, "thanks, rollback noted")
    writer, said = world.db.execute("SELECT user_id, ts_us FROM messages WHERE id = ?", (stated,)).fetchone()
    e1 = world.db.execute("SELECT moment_us FROM events WHERE id = 'e1'").fetchone()[0]
    in_habit = render(e1, zone, said, habit(writer), zone)  # a revision writes the time as its writer does
    assert (
        revised(stated, "Owen owns the audit from {at:e1}")["text"] == f"Owen owns the audit from {in_habit}"
        and reread(in_habit, zone, said) == e1
    )
    ids = [r[0] for r in world.db.execute("SELECT id FROM messages ORDER BY ts_us")]
    assert ids == sorted(ids) and world.violations(complete=True) == []


async def test_tools(tmp_path):
    """The author's tools (v7): ten, served host-side over the world file (annotate is the forge's); each turn's tools act only in their turn
    and on their day; writes are checked and logged, reads are live; the memory renders every time on the company
    clock, never as raw microseconds, and the one-pager stays small on a full world."""
    quick = tmp_path / "quick-gaps.json"  # most replies within ten minutes, as in real channels
    quick.write_text(
        json.dumps({"quantiles": [i * 0.6 for i in range(900)] + [600 + i * 138 for i in range(101)]}) + "\n"
    )
    settings = world_settings(
        tmp_path,
        activity={"messages": 400},
        author={"tolerance": 1.0, "share_tolerance": 1.0},
        personas={"gaps": quick},
    )
    world, cast, org, (a, b, c, d) = organized(tmp_path, settings)
    ops, leads = channel_id("public", "ops", []), channel_id("private", "leads", [])
    with world.trial() as copy:
        start_clock(copy)
    agenda = {
        d: background_plan(world, org, settings.activity, settings.seed, d, agenda_budget(world, settings, d))
        for d in quotas(world, settings)
    }
    context = author_context(settings, cast, org, tmp_path, agenda)
    ledger = Plan(
        storylines=[Storyline(id="s1", summary="the 4.2 release"), Storyline(id="s2", summary="the audit")],
        events=[Event(id="e1", storyline="s2", title="audit window", day=3, time="10:00")],
        facts=[
            PlanFact(
                id="f1",
                storyline="s1",
                subject="Release 4.2",
                attribute="decision",
                value="rollback",
                anchor="rollback",
                channel_id=ops,
                author_id=a,
                day=1,
                summary="s",
            ),
            PlanFact(
                id="f2",
                storyline="s2",
                subject="Audit",
                attribute="owner",
                value="Owen",
                channel_id=leads,
                author_id=a,
                day=3,
                after=["f1"],
                event="e1",
                kind="scheduled",
                summary="s",
            ),
        ],
        board=[BoardEntry(slot="t1", facts=["f1", "f2"])],
    )

    def tools_for(mode, day):
        task = AuthorTask.create(mode, day, world.path, context, f"{mode}-{day:02d}")
        (tools,) = task.toolsets(task.config)
        return tools

    planner = tools_for("plan", 0)
    await planner.setup()
    with_state(planner)
    assert "ledger holds 2 storylines, 1 events and 2 facts" in await planner._with_state(planner.plan)(
        ledger=ledger
    )
    # else post ran in the plan turn
    with pytest.raises(ValueError, match=re.escape("post is for the day turn; this is the plan turn")):
        await planner._with_state(planner.post)(
            conversation=Conversation(channel_id=ops, about="x", lines=[PostLine(author_id=a, text="hi")])
        )
    assert not hasattr(planner, "add_task"), "the author is not served the proposer's tools"
    day1 = tools_for("day", 1)
    await day1.setup()
    with_state(day1)

    def call(tools, name, **kwargs):
        return tools._with_state(getattr(tools, name))(**kwargs)

    page = await call(day1, "now")
    assert page.startswith("# Now: ") and "day 1 of 10" in page and "[[f1]]" in page and "## Brief" in page
    assert "everyday conversations code drew for today" in page and agenda[1]
    assert (
        "- today: 0 messages today (thread replies 0, reacted to 0, in DMs 0, mention someone 0); it closes with"
        in page
    )
    decision = Conversation(
        channel_id=ops,
        about="decision",
        lines=[
            PostLine(author_id=a, text="rollback it is", conveys=["f1"]),
            PostLine(author_id=b, text="ok", reply_to=0),
        ],
    )
    out = await call(day1, "post", conversation=decision)
    assert len(out["messages"]) == 2 and out["now"].endswith("CDT"), out
    count = world.db.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    assert await call(day1, "post", conversation=decision) == out, (
        "a retry after a lost response returns the post"
    )
    assert world.db.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == count, "and posts nothing more"
    # else an untagged anchor was posted
    with pytest.raises(ValueError, match=re.escape("conveys")):
        await call(
            day1,
            "post",
            conversation=Conversation(
                channel_id=ops, about="x", lines=[PostLine(author_id=c, text="rollback again")]
            ),
        )
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
    # else an unknown page was shown
    with pytest.raises(ValueError):
        await call(day1, "view", ref="nobody")
    assert (await call(day1, "sql", sql="SELECT COUNT(*) AS n FROM facts"))["rows"] == [{"n": 2}]
    assert (await call(day1, "sql", sql="SELECT COUNT(*) AS n FROM messages", actor_id=c))["rows"] == [
        {"n": 2}
    ]
    for bad in ({"sql": "DELETE FROM facts"}, {"sql": "SELECT * FROM tasks", "actor_id": c}):
        # else sql ran {bad}
        with pytest.raises(ValueError):
            await call(day1, "sql", **bad)
    assert (await call(day1, "read", actor_id=c, tool="read_channel", arguments={"channel_id": ops}))["items"]
    # else a read with unknown arguments went through
    with pytest.raises(TypeError):
        await call(day1, "read", actor_id=c, tool="read_channel", arguments={"nope": 1})
    while today(world) == 1:
        await call(day1, "advance", to=onward(world))
    # else day 1's tools posted on day 2
    with pytest.raises(ValueError, match=re.escape("day 1 is closed")):
        await call(
            day1,
            "post",
            conversation=Conversation(channel_id=ops, about="x", lines=[PostLine(author_id=a, text="hi")]),
        )
    log = [json.loads(line) for line in (tmp_path / "world-calls.jsonl").read_text().splitlines()]
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
    # Each agent is served its own tools, and the reads they share, described once.
    reads = {"now", "view", "sql", "read"}
    assert (
        set(described) == reads | {"plan", "post", "advance", "revise", "edit"}
        and served.server_name == "world"
    )
    forge = {
        "settings": json.loads(forge_of(settings).model_dump_json())
    }  # a forge's settings in its context
    proposer = ProposerTask.create(world.path, context | forge, "forge-check")
    (proposing,) = proposer.toolsets(proposer.config)
    with_state(proposing)
    proposed = await served_tools(proposing)
    assert (
        set(proposed) == reads | {"add_task", "profile", "probe", "annotate"}
        and proposing.server_name == "world"
    )
    assert {t: described[t] for t in reads} == {t: proposed[t] for t in reads}
    # help() in an agent's IPython shows a tool's description alone: it names every field of every document a call
    # takes, nested ones too, with ? when optional.
    for tools, tool, models in (
        (described, "post", (Conversation, PostLine, Reaction, Commit, Close)),
        (described, "plan", (Plan, PlanFact, Event, Storyline)),
        (proposed, "add_task", (Task,)),
    ):
        for model in models:
            for field, info in model.model_fields.items():
                marked = f"{field}{'' if info.is_required() else '?'}: "
                assert marked in tools[tool].split("Arguments: ", 1)[1], (tool, marked)
    assert "lines: [1-40 × {author_id: str" in described["post"] and described["now"].endswith(
        "Arguments: none"
    )
    # nested documents cross the wire as strict models, and a refused write carries its reason
    async with mcp_client(served) as client:
        refused = await client.call_tool(
            "post",
            {
                "conversation": {
                    "channel_id": ops,
                    "about": "x",
                    "lines": [
                        {"author_id": a, "text": "hi", "commits": [{"id": "c1", "text": "t", "due_day": 2.5}]}
                    ],
                }
            },
        )
        assert refused.is_error and "due_day" in str(refused.content), refused
    invalid = [json.loads(line) for line in (tmp_path / "world-calls.jsonl").read_text().splitlines()][-1]
    assert invalid["tool"] == "post" and invalid["error"].startswith(
        "invalid arguments: conversation.lines.0.commits.0.due_day"
    ), invalid
    bad = AuthorTask.create("day", 2, world.path, context, "day-02-01")
    world.db.execute("UPDATE storylines SET summary = 'changed' WHERE id = 's1'")
    # else a changed world file was served
    with pytest.raises(ValueError):
        await bad.toolsets(bad.config)[0].setup()
    world.db.execute("UPDATE storylines SET summary = 'the 4.2 release' WHERE id = 's1'")
    rng, gaps, lines = random.Random(3), Gaps(settings.personas.gaps), 0
    while today(world) is not None:  # a full world: about 400 messages over ten days, written at the present
        day = today(world)
        quota = quotas(world, settings)[day]["messages"]
        if day == 3 and not world.db.execute("SELECT 1 FROM evidence WHERE fact_id = 'f2'").fetchone():
            post(
                world,
                Conversation(
                    channel_id=leads,
                    about="audit",
                    lines=[PostLine(author_id=a, text="Owen owns it from {at:e1}", conveys=["f2"])],
                ),
                settings,
                rng,
                gaps,
            )
        if posted(world, *bounds(world, day)) < quota and part_of(world, present(world)) != "night":
            try:
                who = [rng.choice([a, b]) for _ in range(4)]  # every other conversation mentions someone once
                post(
                    world,
                    Conversation(
                        channel_id=rng.choice([ops, leads]),
                        about="work",
                        lines=[
                            PostLine(
                                author_id=w,
                                text=f"note {lines + i} on the work"
                                + (f", <@{a if w == b else b}>" if i == 1 and lines % 8 == 0 else ""),
                                reply_to=0 if i else None,
                            )
                            for i, w in enumerate(who)
                        ],
                    ),
                    settings,
                    rng,
                    gaps,
                )
                lines += 4
                continue
            except ValueError:
                pass
        advance(world, settings, onward(world))
    assert world.db.execute("SELECT COUNT(*) FROM messages").fetchone()[0] > 300
    # A day holds a share of messages that mention someone, near the measured 11%, in a tighter band than the others.
    days_ = [tally(world, d) for (d,) in world.db.execute("SELECT day FROM calendar")]
    assert share_band(settings, "mention_rate", 52) == (3, 9) and all(
        share_band(settings, "mention_rate", t["messages"])[0]
        <= t["mention_rate"]
        <= share_band(settings, "mention_rate", t["messages"])[1]
        for t in days_
    ), days_
    memory = files(world, settings, context)
    # A person's page says who they are beyond their title and how they type, in words; the guide says how Slack sounds.
    page = memory["memory/people/alicia.rao.md"]
    assert "## Who they are" in page and "calm" in page and "midwest" in page and "chess" in page, page
    assert "20% of their messages are short (4 words or fewer)" in page, page
    assert (
        "\nVoice\n- This is Slack"
        in AuthorTask.create("day", 1, world.path, context, "day-01-01").data.system_prompt
    )
    raw = re.compile(r"\b1\d{15}\b")
    assert not [p for p, text in memory.items() if raw.search(text)], "no raw microseconds reach the author"
    assert len(memory["memory/now.md"].encode()) < 20_000, len(memory["memory/now.md"].encode())
    assert Plan.model_validate(json.loads(memory["memory/ledger.json"])) == ledger, (
        "ledger.json is world_plan's input"
    )
    assert not raw.search(json.dumps(ledger_digest(world))), "the judge's ledger is rendered too"
