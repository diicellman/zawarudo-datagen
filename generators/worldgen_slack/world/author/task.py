"""The world author's task: one block of its work in its VM (the plan, a day, a fix of the final review), opened with
the turn's prompt; the guide it works by, and the prompts of its turns."""

import json
from pathlib import Path
from typing import Self

import verifiers.v1 as vf
from worldgen_slack.taskset import WorldTaskData

from ...session import AuthoringToolsConfig, tools_config
from ..clock import PARTS, bounds, chapter_start, clock, present
from .tools import AuthorTools

AUTHOR_GUIDE = """You write one company's Slack workspace, as its people live it, in time order: first the company and its
people, then the ledger of what will be stated, then each day's conversations.
The workspace is a SQLite file that only code writes. You change it with the world_* tools: async functions in your
IPython (`r = await world_post(conversation={...})`) that return JSON text and raise with the reason when a write
breaks a rule; fix and call again. They are live: world_now(), world_view(ref), world_sql(sql, actor_id=None) and
world_read(actor_id, tool, arguments) read the world as it is now.

Memory and notes
- /task/memory/ is code's, rendered from the world before every turn: now.md (the present, today's quota, the
  conversations code drew for today, each storyline's facts, events, open promises, a chapter's brief, style drift),
  ledger.json (the ledger as world_plan takes it), and a page per person, channel, storyline and event. Pages name
  each other as [[id]]; grep them. Every time in them is on the company clock, as a weekday and a date.
- /task/notes/ is yours: plan.md (the story's arcs day by day, and what each brief entry will rest on) and
  recap.md (what happened today, what is open). Besides the world, they are all you keep from one turn to the next.

Time
- The world has a present. world_post writes a conversation at the present and moves it to its last line;
  world_advance(to=part) moves it to a later part of the day: PARTS_OF_THE_DAY. From the night,
  world_advance(to="tomorrow") closes the day, once the day's checks pass, and moves to the next day's early part.
  Nothing is ever written before the present, so write the day in the order it happens.
- People come and go: the plan's moves have someone join or leave a channel, leave the company, or a channel archived,
  on a day; Slack posts the joins and leaves, and from then the conversations are theirs, or not.
- The company's apps post by themselves (code writes them): builds, alerts, deploys, tickets, reminders. People read
  them, react, and reply in their threads; an event in the ledger may name the app that reports it (post_by), and
  then its title is all the app says, so a title names no fact's value and no time.
- A message never contains a clock time or a date. It writes {at:id} for an event, or for a fact about one; code
  renders it on the author's clock, the way that person writes times, with its day when it is not today ("tomorrow
  5pm", "Tue 17:00"): no day word right before it. Relative words (tomorrow, Friday, this morning) are fine elsewhere
  when true at the present.

The ledger (world_plan)
- storylines: exactly the number the plan's prompt gives, the company's workstreams. Fixed once planned; summaries
  may change.
- events: the story's moments: a day of the calendar, a time and, if not the company's, a clock (IANA zone). An event
  never moves once planned: a change of plan is a new event, and a fact that supersedes the old one.
- facts: what someone states in Slack: subject, attribute and value (no time in a value); its author, its channel and
  the day it is first stated; event and kind (happened or scheduled) when it is about an event, and it then carries
  the event's time; after and supersedes, facts it follows or replaces; decoy, a value that is wrong; summary, what it
  means and on whose authority. anchor: distinctive words of the value. Every line that contains a fact's anchor
  states that fact, so each fact has words of its own: no two facts share an anchor, and a fact whose anchor contains
  another's comes after it. A value said again is the same fact conveyed again, not a new fact. A fact about an event
  with no anchor is stated with its time, {at:...}.
- A stated fact never changes; an unstated one may be re-planned for today or later.
- The facts run through the days being written: each third of them first states at least half its share of the
  facts first stated in those days, as it holds of their messages, so the storylines unfold to the last day.

Conversations (world_post)
- One channel; lines by its members, each in its author's way (their typing card is on their page). reply_to is the
  index of an earlier line whose thread the line joins; thread continues an earlier thread of the channel.
  reactions come from other members. pause is "hours" when an hour or more passes before the line.
- conveys lists the facts a line states. A fact's first statement is by its author, in its channel, on its day; a
  happened fact is stated only after it happens, a scheduled one before it happens; after and supersedes keep their
  order. Anyone may repeat a stated
  fact later, listing it in conveys.
- commits opens a promise with the day it is due; closes ends one: kept, changed or dropped. Every promise ends by
  its due day.
- Each day holds about the number of messages now.md gives, spread by the parts' rhythm, and its own shares of thread
  replies, reacted messages and DMs. A weekend day is quiet, unless the first plan of its days puts an event on it: a day with
  an event gets a workday's messages; now.md's today line shows them, and what closing the day still takes. A
  conversation or a plan after which the day could no longer close is refused. now.md lists the everyday
  conversations code drew for the day from the channels' routines; the storylines happen among them. Each person
  writes like their typing card; now.md reports who drifts from it.

Voice
- This is Slack, not email and not a status report. People type quickly: fragments, contractions (I'll, can't,
  it's), dropped subjects ("on it", "looks good"), lowercase starts, the odd typo, a question when they don't know
  something, a one-word reply when that is all it takes.
- Each person sounds like themselves. Their page says who they are and how they type: how long they write, how often
  they ask, start lowercase or use emoji. A director and a new engineer, a careful analyst and a chatty support lead,
  do not sound alike.
- Address people as <@USERID> when a line is aimed at them; a reaction often replaces a "thanks" line; emoji in text
  like :tada: or 🙏 where the person would use them.
- Work talk is concrete and uneven: names of systems, versions, ticket numbers, links, half-formed ideas, mild
  complaints, jokes, side topics, disagreement. Not every line moves the story; not every line is an update.
- A line that states a fact contains the fact's anchor words; the rest of the line is still in its author's voice.

Reviews
- An independent judge reviews the world after some days and at the end. Its issues come back to you: message_ids
  are the messages to change (world_revise rewrites one where it stands), evidence_message_ids show the defect.
""".replace(
    "PARTS_OF_THE_DAY", ", ".join(f"{part} {start:02d}-{end:02d}" for part, (start, end) in PARTS.items())
)


# How a real workspace's days go, beyond how each line is typed: discourse-level craft from StoryScope (arXiv
# 2604.03136). Voice lab 02 (2026-10-12) tested it after the guide: blind judges of two families preferred the days it
# wrote 72% and 66% of the time, and its plans escalated more and left more open.
CRAFT_GUIDE = """
Story and talk (how a real workspace's days go, beyond how each line is typed)
- Stakes move. A storyline grows, stalls, flares or fizzles from day to day: a small issue turns serious, a scare
  turns out to be nothing, a quiet week gets one bad night. It is not a steady run of status updates.
- More than one track. Concerns that have nothing to do with the storylines run alongside them, and some echo a
  storyline without serving it: another team's outage, a contract question, a hiring loop, a flaky test nobody owns.
- Not everything closes. Some questions get no answer; some promises slip or are quietly dropped; some outcomes stay
  partial or unclear. Nobody writes the epilogue.
- Some outcomes come from outside. A vendor ships a fix, a customer changes their mind, a problem stops on its own:
  not every resolution is someone's decision or insight.
- Late and elsewhere. What matters often surfaces later, or in a DM or a thread, and changes how earlier talk reads
  ("oh, so that's why the import failed"). People refer back to old incidents and earlier days, by name.
- Defensible calls go wrong. People make reasonable calls that turn out badly, blame stays unclear, and some
  disagreements are never settled.
- Nobody states the lesson. No line spells out the takeaway, the moral or what the episode shows; wrap-ups are rare
  and short, and most threads just stop.
- Feelings are named plainly when people show them ("ugh", "kinda nervous about tonight", "annoyed tbh"), not acted
  out through the body or the room.
- People show who they are by what they do and say, not by introducing their role.
- Registers differ: from person to person, and for one person from a calm morning to a bad night.
"""


CHAPTER_GUIDE = """
This run is a chapter
- The workspace was written before: its days up to the chapter's first are written and frozen. Nothing in them is
  posted, revised or re-planned; world_revise refuses their messages, and their facts and events stay as they are.
  You write the chapter's days, from its first, in time order, as the same company and people, going on from where
  the last day left off (your notes are where it ended).
- The ledger goes on: the earlier storylines stay first and may carry on; the chapter adds its own after them. A later
  value of an earlier fact is a new fact that supersedes it, stated in the chapter.
- No tasks are written in a chapter. Its brief, in now.md, is the board for a later forge: each entry names what its
  facts need (values, near-misses, channels, relations, where the answer is stated), all of it planted in the chapter.
"""


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


# S3's judge flagged its planned decoys as defects, and the author corrected them where their readers see them: a
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
