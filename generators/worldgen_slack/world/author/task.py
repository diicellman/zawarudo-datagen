"""The world author's task: one block of its work in its VM (the plan, a day, a fix of the final review), opened with
the turn's prompt; the guide it works by, and the prompts of its turns."""

import json
from pathlib import Path
from typing import Self

import verifiers.v1 as vf
from worldgen_slack.taskset import WorldTaskData

from ...prompt import prompt, render
from ...session import AuthoringToolsConfig, tools_config
from ..clock import PARTS, bounds, chapter_start, clock, present
from .tools import AuthorTools

AUTHOR_GUIDE = render(
    prompt(__file__, "guide.md"),
    parts=", ".join(f"{part} {start:02d}-{end:02d}" for part, (start, end) in PARTS.items()),
)
# How a real workspace's days go, beyond how each line is typed: discourse-level craft from StoryScope (arXiv
# 2604.03136). Tested after the guide, blind judges of two model families preferred the days written with it 72% and
# 66% of the time, and its plans escalated more and left more open.
CRAFT_GUIDE = prompt(__file__, "craft.md")
CHAPTER_GUIDE = prompt(__file__, "chapter.md")


class AuthorConfig(vf.TaskConfig):
    tools: AuthoringToolsConfig


class AuthorTask(vf.Task[WorldTaskData, vf.State, AuthorConfig]):
    """One block of the author's work in its VM (the plan, a day, a fix of the final review): the env opens it with the
    turn's prompt, and the author works through its world tools."""

    NEEDS_CONTAINER = True

    @classmethod
    def toolsets(cls, config: AuthorConfig) -> list[vf.Toolset]:
        return [AuthorTools(config.tools)]

    @classmethod
    def create(cls, mode: str, day: int, world: Path, context: dict, attempt: str) -> Self:
        tools = tools_config(mode, day, world, context)
        data = WorldTaskData(
            prompt=None,
            system_prompt=AUTHOR_GUIDE + CRAFT_GUIDE + (CHAPTER_GUIDE if context.get("chapter") else ""),
            attempt=attempt,
            world_hash=tools.db_hash,
            network_allow=[],
            network_block=["*"],
        )
        return cls(data, AuthorConfig(tools=tools))


def rejected(feedback: str, restored: str = "") -> str:
    """Why the last attempt at this block was rejected, and what was put back."""
    return f"\nYour last attempt at this was rejected{restored}: {feedback}" if feedback else ""


def plan_prompt(world, settings, feedback: str = "") -> str:
    count, first = settings.storylines, chapter_start(world)
    total = world.db.execute("SELECT COUNT(*) FROM calendar").fetchone()[0]
    if first > 1:
        return (
            f"Before day {first} ({clock(world, bounds(world, first)[0])[:14]}): days 1-{first - 1} are written and "
            f"frozen, and this chapter writes days {first}-{total}. Plan its ledger with world_plan: the ledger.json "
            f"you have, going on, with {count} storylines in all (the earlier ones first), and the chapter's events "
            "and facts over its days, and the board: the facts each entry of now.md's brief will rest on, as it "
            "needs. Then rewrite /task/notes/plan.md for the chapter: each storyline's arc day by day, and how each "
            f"brief entry is planted. End your turn when the ledger is planned.{rejected(feedback)}"
        )
    return (
        f"Before day 1 ({clock(world, bounds(world, 1)[0])[:14]}): plan the ledger with world_plan: exactly {count} "
        "storylines, their events and their facts, over the calendar in now.md, and its board: the facts each slot "
        "now.md marks for the board will rest on, as many channels, relations and near-misses as its level needs. Then write "
        "/task/notes/plan.md: each storyline's arc day by day, and how each slot will be asked. End your turn when the "
        f"ledger is planned.{rejected(feedback)}"
    )


# A judge flags planned decoys as defects, and an author told nothing corrects them where their readers see them: a
# near-miss corrected in plain sight misleads no one.
DECOY_ISSUES = (
    "An issue about a message that states a decoy is answered in /task/notes/plan.md, not in the world: correcting or "
    "hedging a decoy where its readers see it voids the near-miss its tasks rest on."
)


def day_prompt(world, day: int, issues: list | None = None, feedback: str = "") -> str:
    total = world.db.execute("SELECT COUNT(*) FROM calendar").fetchone()[0]
    review = (
        "\nThe judge reviewed the world so far. Fix what you can in what you write next (world_revise rewrites a "
        f"message where it stands): {json.dumps(issues, ensure_ascii=False)}\n{DECOY_ISSUES}"
        if issues
        else ""
    )
    first = chapter_start(world)
    span = f" (this chapter writes days {first}-{total})" if first > 1 else ""
    return (
        f"It is {clock(world, present(world))}, day {day} of {total}{span}.{review}"
        f"{rejected(feedback, ', and the world and your notes are back at the start of the day')}\nRead world_now() and your notes, and "
        "write today: post its conversations in the order they happen with world_post, and move on with "
        'world_advance(to=part); from the night, to="tomorrow" closes the day once today\'s checks pass. Then write '
        "/task/notes/recap.md "
        "(what happened today, what is open), update plan.md, and end your turn."
    )


def fix_prompt(issues: list) -> str:
    return (
        f"The review rejected the world: {json.dumps(issues, ensure_ascii=False)}\nFix each blocking issue with "
        "world_revise: message_ids are the messages to change, evidence_message_ids "
        "show the defect. Fix the others too, or write in /task/notes/plan.md why you leave one; if you judge an issue "
        f"mistaken, write why there. {DECOY_ISSUES} End your turn when done."
    )
