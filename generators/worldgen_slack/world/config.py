"""A world's settings, configured in one file: the company and its people, the calendar, the workspace's shape, how
the author writes and the judge reviews, and their two seats. A chapter runs under its world's, with the taxonomy its
brief is drawn from."""

import tomllib
from pathlib import Path
from typing import Literal

import verifiers.v1 as vf
from pydantic import Field

from ..config import Categories, RootedPath, Seats, Section, Text, rooted, vm_seat


class PersonasConfig(Section):
    """The seeded cast and how real Slack users type and reply; all three files come from
    scripts/worldgen_slack/personas.sql."""

    path: RootedPath
    typing: RootedPath
    gaps: RootedPath
    pool: int = Field(default=36, ge=4, le=200)


class CalendarConfig(Section):
    """The world's days on the company clock; the run seed picks a Monday of `year` as day 1."""

    days: int = Field(default=10, ge=1, le=60)
    year: int = Field(default=2026, ge=2000, le=2100)


class ActivityConfig(Section):
    """Reference targets for the workspace's shape. One table drives where code places conversations and
    measure.py's scorecard; each value's provenance is noted in the run config."""

    messages: int = Field(default=400, ge=0, le=5000)
    channel_skew: float = Field(default=1.1, ge=0, le=4)
    dms_per_person: float = Field(default=0.4, ge=0, le=5)
    dm_share: float = Field(default=0.25, ge=0, le=1)
    conversation_lines: float = Field(default=6, ge=2, le=30)
    reply_share: float = Field(default=0.3, ge=0, le=0.95)
    reaction_rate: float = Field(default=0.2, ge=0, le=1)
    parts: dict[Literal["early", "morning", "afternoon", "evening", "night"], float] = Field(
        default_factory=lambda: {
            "early": 0.1,
            "morning": 0.35,
            "afternoon": 0.4,
            "evening": 0.12,
            "night": 0.03,
        }
    )
    mention_rate: float = Field(default=0.11, ge=0, le=1)
    emoji_rate: float = Field(default=0.08, ge=0, le=1)
    # Promises made per message, at least, each day ("I'll have it by Friday"): what was promised and is still open is
    # a request people bring to an assistant. Off unless a world's config sets it.
    promise_rate: float = Field(default=0.0, ge=0, le=0.2)
    # The share of promises that slip (their due day passes with nothing said), change or are dropped, at least, of
    # those made so far, checked as each is made: not every promise is kept, and "was it done?" is asked of those too.
    promise_slips: float = Field(default=0.0, ge=0, le=1)
    # Each app's routine posts per workday (builds, alerts, staging deploys, tickets; a reminder posts once): code's,
    # never a person's. Off unless a world's config sets it, with the apps its organization lists.
    bot_posts: float = Field(default=0.0, ge=0, le=30)


class ReviewRounds(Section):
    """Bounded allowances: final reviews per run."""

    final: int = Field(default=4, ge=1, le=20)


class AuthorSettings(Section):
    """The world author: one agent writes the world in time order."""

    # how far a day's message count may be from its quota
    tolerance: float = Field(default=0.3, ge=0, le=1)
    # a weekend day's share of a workday's messages
    weekend: float = Field(default=0.1, ge=0, le=1)
    # reply, reaction and DM shares, checked at the end
    share_tolerance: float = Field(default=0.1, ge=0, le=1)
    # the share of messages mentioning someone, per day
    mention_tolerance: float = Field(default=0.05, ge=0, le=1)
    # the judge reviews the world after these days
    review_days: list[int] = Field(default_factory=lambda: [5])
    # seconds between pings of the author's VM while only the judge works (it idles out after 30 minutes)
    keepalive: float = Field(default=600, gt=0, le=1700)
    # a day that cannot close is written again from its start
    day_attempts: int = Field(default=2, ge=1, le=5)
    # a plan that cannot be recorded is written again
    plan_attempts: int = Field(default=3, ge=1, le=10)


class LedgerTargets(Section):
    """What the ledger holds at least, counted off the plan (world/profile.py's counts): the material a forge's tasks
    rest on, planned before the day it is written. 0 holds the ledger to nothing."""

    facts: int = Field(default=0, ge=0, le=5000)
    chains: int = Field(default=0, ge=0, le=1000)  # values superseded 3 deep, across 2 or more channels
    deep_chains: int = Field(default=0, ge=0, le=1000)  # 4 deep
    decoys: int = Field(default=0, ge=0, le=1000)  # beside a real value of their subject and attribute
    private_truths: int = Field(
        default=0, ge=0, le=1000
    )  # a changed value, its current one in a private place
    collections: int = Field(
        default=0, ge=0, le=1000
    )  # 4 or more items across 3 or more channels, one changed
    archives: int = Field(default=0, ge=0, le=100)  # channels a move archives
    departures: int = Field(default=0, ge=0, le=100)  # people a move has leave the company


class WorldSeats(Seats):
    author: vf.AgentConfig = vm_seat("z-ai/glm-5.3")


class WorldConfig(Section):
    """One world run, configured in one file."""

    sector: Text
    seed: int = Field(default=0, ge=0)
    output: Path
    corpus: Path = Path("data")
    language: Text = "English"
    premise_count: int = Field(default=12, ge=2, le=30)
    personas: PersonasConfig
    calendar: CalendarConfig = Field(default_factory=CalendarConfig)
    activity: ActivityConfig = Field(default_factory=ActivityConfig)
    storylines: int = Field(default=2, ge=1, le=20)  # the plot the ledger plans before day 1
    ledger: LedgerTargets = Field(default_factory=LedgerTargets)
    # lost sessions a block of work may take before the run stops (each a fresh VM after a pause): a long world loses
    # more (W1 lost 4 in 2.9 hours)
    max_losses: int = Field(default=1, ge=0, le=20)
    review_rounds: ReviewRounds = Field(default_factory=ReviewRounds)
    author: AuthorSettings = Field(default_factory=AuthorSettings)
    env: WorldSeats = Field(default_factory=WorldSeats)
    # A chapter's: the taxonomy its brief's entries are cells of, and the question styles it draws for them
    # (taxonomy.toml). A world run plants no brief.
    taxonomy: Categories = Field(default_factory=dict)
    styles: list[Text] = Field(default_factory=list)


def load_world(path: Path) -> WorldConfig:
    config = WorldConfig.model_validate(tomllib.loads(path.read_text()))
    config.output, config.corpus = rooted(config.output), rooted(config.corpus)
    return config
