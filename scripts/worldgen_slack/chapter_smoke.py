"""A chapter on a finished world, offline: uv run --frozen python scripts/worldgen_slack/chapter_smoke.py <run dir>

No model is called and nothing under the run directory is written (the parent is read only; the chapter lives in a
temporary directory). The chapter appends 2 days and a brief of one contested value, plans a new value of the world's
busiest value group (superseding its current value, in a DM), posts the days as a scripted author would, closes them,
and checks what a chapter promises: the past unchanged, its days' quotas unchanged, the brief met, the lineage kept."""

import json
import random
import sys
import tempfile
from pathlib import Path

from generators.worldgen_slack.agents.world import ledger_document
from generators.worldgen_slack.chapter import adopt, brief_slots, load_chapter, provenance, version_of
from generators.worldgen_slack.chronicle import (
    Conversation,
    Event,
    Plan,
    PlanFact,
    PostLine,
    advance,
    part_of,
    post,
    present,
    quotas,
    record_plan,
    revise,
    spread,
    today,
)
from generators.worldgen_slack.config import ROOT
from generators.worldgen_slack.contracts import PARTS, Gaps, answer_facts, frozen_digest, meta
from generators.worldgen_slack.store import Store
from worldgen_slack.world import file_hash


def refused(expected: str, act) -> None:
    try:
        act()
    except ValueError as error:
        assert expected in str(error), (expected, str(error))
        return
    raise AssertionError(f"not refused: {expected}")


def busiest(world) -> tuple[str, str, str]:
    """The value group with the most values, and its current value (no decoy, superseded by nothing)."""
    subject, attribute, _ = world.db.execute(
        "SELECT subject, attribute, COUNT(DISTINCT lower(value)) n FROM facts GROUP BY 1, 2 ORDER BY n DESC, 1, 2 LIMIT 1"
    ).fetchone()
    ids = [
        r[0]
        for r in world.db.execute(
            "SELECT id FROM facts WHERE subject = ? AND attribute = ?", (subject, attribute)
        )
    ]
    return subject, attribute, answer_facts(world, ids)[0]


def main(parent: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="chapter-smoke-") as directory:
        root = Path(directory)
        config = root / "chapter.toml"
        config.write_text(
            f'base = "{ROOT / "configs/worldgen_slack/worldgen.toml"}"\nworld = "{parent.resolve()}"\n'
            f'output = "{root / "out"}"\nseed = 7\ndays = 2\nmessages = 8\n'
            '[[material]]\ncategory = "semantic"\nlevel = 4\nneeds = { values = 6 }\n'
            "[settings.author]\ntolerance = 1.0\nshare_tolerance = 1.0\n"
        )
        before = file_hash(parent / "world.sqlite")
        chapter, settings = load_chapter(config)
        store = Store(chapter.output, provenance(chapter, settings))
        world = store.world
        slots = brief_slots(chapter, settings, version_of(world) + 1)
        adopt(store, chapter, settings, slots)
        first, frozen = int(meta(world, "chapter_start_day")), meta(world, "frozen_digest")
        assert first == 11 and meta(world, "version") == "2" and meta(world, "parent_hash") == before
        assert today(world) == first and part_of(world, present(world)) == "early", (
            "the present: the chapter's first morning"
        )
        assert world.db.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0, "a chapter carries no tasks"
        assert (
            store.state.restore_point == "source" and (store.root / "attempts/source/notes/recap.md").exists()
        )
        # The parent's days keep the quotas they were written to: rebuilt from its configuration, and each day's
        # messages within the tolerance it closed in.
        run = json.loads((parent / "run.json").read_text())["config"]
        dates = [
            d for (d,) in world.db.execute("SELECT date FROM calendar WHERE day < ? ORDER BY day", (first,))
        ]
        busy = json.loads(
            world.db.execute("SELECT value FROM world_meta WHERE key = 'busy_days'").fetchone()[0]
        )
        rebuilt = spread(
            dates, 1, run["activity"]["messages"], run["author"]["weekend"], run["activity"]["parts"], busy
        )
        parent_quotas = {d: q for d, q in quotas(world, settings).items() if d < first}
        assert parent_quotas == dict(enumerate(rebuilt, 1)), "the parent's days keep their quotas"
        for day, (start, end) in enumerate(
            world.db.execute(
                "SELECT start_us, end_us FROM calendar WHERE day < ? ORDER BY day", (first,)
            ).fetchall(),
            1,
        ):
            count = world.db.execute(
                "SELECT COUNT(*) FROM messages WHERE ts_us >= ? AND ts_us < ?", (start, end)
            ).fetchone()[0]
            assert (
                abs(count - parent_quotas[day]["messages"])
                <= run["author"]["tolerance"] * parent_quotas[day]["messages"]
            ), (day, count)
        assert sum(q["messages"] for d, q in quotas(world, settings).items() if d >= first) == 8

        # The plan: the ledger as the author gets it, going on with a new value of the busiest group, said in a DM.
        subject, attribute, current = busiest(world)
        (author,) = world.db.execute("SELECT author_id FROM facts WHERE id = ?", (current,)).fetchone()
        dm = world.db.execute(
            "SELECT c.id FROM channels c JOIN members m ON m.channel_id = c.id WHERE c.type = 'im' AND m.user_id = ? ORDER BY c.id LIMIT 1",
            (author,),
        ).fetchone()[0]
        storyline = world.db.execute("SELECT storyline FROM facts WHERE id = ?", (current,)).fetchone()[0]
        document = ledger_document(world, {s.id for s in slots})
        assert document["board"] == [], "the parent's board stays out of the chapter's plan"
        plan = Plan.model_validate(document)
        later = PlanFact(
            id="ch2-value",
            storyline=storyline,
            subject=subject,
            attribute=attribute,
            value="held until the vendor patch lands",
            anchor="until the vendor patch",
            channel_id=dm,
            author_id=author,
            day=first,
            supersedes=current,
            summary="the window moves again, said only in a DM",
        )
        entry = [
            f
            for f in world.db.execute(
                "SELECT id FROM facts WHERE subject = ? AND attribute = ? AND is_decoy = 0 ORDER BY day DESC, id",
                (subject, attribute),
            ).fetchall()
        ][:2]
        board = [{"slot": slots[0].id, "facts": [entry[1][0], current, "ch2-value"]}]

        def planned(**changes):
            return Plan.model_validate(
                plan.model_dump(mode="json")
                | {
                    "facts": [*plan.model_dump(mode="json")["facts"], later.model_dump(mode="json")],
                    "board": board,
                }
                | changes
            )

        def tried(act):
            with world.trial() as copy:
                act(copy)

        def recorded(document, brief=slots):
            tried(lambda copy: record_plan(copy, document, settings, brief))

        needy = [s.model_copy(update={"needs": s.needs.model_copy(update={"values": 7})}) for s in slots]
        refused("taking at least 7 values", lambda: recorded(planned(), needy))
        refused(
            "first stated in this chapter",
            lambda: recorded(planned(board=[{"slot": slots[0].id, "facts": [entry[1][0], current]}])),
        )
        refused(
            "planned for today (day 11) or later",
            lambda: recorded(
                planned(
                    facts=[
                        *plan.model_dump(mode="json")["facts"],
                        later.model_copy(update={"day": first - 1}).model_dump(mode="json"),
                    ]
                )
            ),
        )
        refused(
            "written and frozen; a new event",
            lambda: recorded(
                planned(
                    events=[
                        *plan.model_dump(mode="json")["events"],
                        Event(
                            id="ch2-past",
                            storyline=storyline,
                            title="a look back",
                            day=first - 2,
                            time="10:00",
                        ).model_dump(mode="json"),
                    ]
                )
            ),
        )
        frozen_facts = plan.model_dump(mode="json")["facts"]
        frozen_facts[0] = frozen_facts[0] | {"summary": "rewritten"}
        refused(
            "an earlier chapter's, written and frozen",
            lambda: recorded(planned(facts=[*frozen_facts, later.model_dump(mode="json")])),
        )
        refused(
            "in this order, first",
            lambda: recorded(planned(storylines=list(reversed(plan.model_dump(mode="json")["storylines"])))),
        )
        recorded(planned())
        assert world.db.execute("SELECT COUNT(*) FROM board").fetchone()[0] == 45 + 3, (
            "the parent's board stays"
        )
        old = world.db.execute("SELECT id FROM messages ORDER BY id LIMIT 1").fetchone()[0]
        refused(
            "earlier chapter, written and frozen", lambda: tried(lambda copy: revise(copy, old, "edited"))
        )

        # The days, written as a scripted author would: the new value in its DM, a line of chatter, the days closed.
        gaps, rng = Gaps(settings.personas.gaps), random.Random(1)
        talker = world.db.execute(
            "SELECT user_id FROM members WHERE channel_id = ? AND user_id != ?", (dm, author)
        ).fetchone()[0]

        def say(channel, *lines):
            with world.trial() as copy:
                post(
                    copy,
                    Conversation(
                        channel_id=channel, about="work", lines=[PostLine(**line) for line in lines]
                    ),
                    settings,
                    rng,
                    gaps,
                )

        def close():
            with world.trial() as copy:
                for part in [p for p in PARTS if p != "early"] + ["tomorrow"]:
                    advance(copy, settings, part)

        say(
            dm,
            dict(
                author_id=author,
                text="heads up, held until the vendor patch lands, not tuesday",
                conveys=["ch2-value"],
            ),
            dict(author_id=talker, text="ok thx"),
        )
        close()
        say(dm, dict(author_id=talker, text="any news on the patch?"))
        close()
        assert today(world) is None, "the chapter's days are closed"
        assert frozen_digest(world) == frozen, "nothing before the chapter changed"
        assert {d: q for d, q in quotas(world, settings).items() if d < first} == parent_quotas
        assert world.violations(complete=True) == [], world.violations(complete=True)
        assert file_hash(parent / "world.sqlite") == before, "the parent is read only"
        store.close()
    print(
        f"PASS chapter smoke on {parent}: version 2 from day {first}, the past unchanged, the brief planted"
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]))
