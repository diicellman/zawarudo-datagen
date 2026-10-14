"""Validated run settings; all role overrides use native Verifiers configs."""

import hashlib
import json
import tomllib
from pathlib import Path
from typing import Annotated, Literal
import verifiers.v1 as vf
from pydantic import AfterValidator, BaseModel, BeforeValidator, ConfigDict, Field, model_validator
from verifiers.v1.configs.retries import RetryConfig
from verifiers.v1.harnesses.null.harness import NullHarnessConfig
from worldgen_slack.config import role
from worldgen_slack.release import SafeId

ROOT = Path(__file__).resolve().parents[2]


def merged(base: dict, changes: dict) -> dict:
    return {
        k: merged(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
        for k, v in changes.items()
    } | {k: v for k, v in base.items() if k not in changes}


def pinned() -> dict:
    """What every run's provenance pins beside its own settings: the verifiers revision pyproject.toml installs, and
    the lock file."""
    revision = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["uv"]["sources"]["verifiers"][
        "rev"
    ]
    return {
        "verifiers_revision": revision,
        "lock_hash": hashlib.sha256((ROOT / "uv.lock").read_bytes()).hexdigest(),
    }


def world_author(model: str) -> vf.AgentConfig:
    """The world author: an rlm coding agent in one VM for the whole world, with one interaction per block of work
    (setup, plan, each day, tasks); each interaction gets these turn, token and time budgets."""
    base = role(model, author=True)
    return base.model_copy(
        update={
            "harness": base.harness.model_copy(update={"max_total_tokens": 600_000}),
            # Idle between interactions only while the judge or the solver works; a VM left behind ends in 30 min.
            "runtime": base.runtime.model_copy(update={"idle_timeout": 1800}),
            "max_turns": 200,
            "timeout": base.timeout.model_copy(update={"rollout": 5400}),
        }
    )


class PipelineConfig(vf.EnvConfig):
    taskset: vf.TasksetConfig = vf.TasksetConfig(id="worldgen-slack")
    author: vf.AgentConfig = world_author("z-ai/glm-5.3")
    judge: vf.AgentConfig = role("openai/gpt-6-sol")
    # The solver executes no code (null harness; its Slack tools are host-side servers), so a local process
    # replaces the VM.
    solver: vf.AgentConfig = role("z-ai/glm-5.2", solver=True).model_copy(
        update={"runtime": vf.SubprocessConfig()}
    )
    # A stronger solver with the same tools, for the tasks GLM rarely answers: hard, or broken?
    # It thinks hard (FC1's took 2-5 calls and lost tasks the solver answered): it shows a task can be answered.
    witness: vf.AgentConfig = role("openai/gpt-6-sol", solver=True).model_copy(
        update={
            "runtime": vf.SubprocessConfig(),
            "sampling": role("openai/gpt-6-sol", solver=True).sampling.model_copy(
                update={"reasoning_effort": "high"}
            ),
        }
    )
    retries: RetryConfig = RetryConfig(max_retries=0)
    # The episode's agents at once. GLM, whose account allows 8 concurrent requests, is gated on its own by
    # [author] solvers, so the judge's reviews run beside its solves.
    max_concurrent_agents: int | None = 16


def rooted(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


RootedPath = Annotated[Path, AfterValidator(rooted)]
Text = Annotated[str, Field(min_length=1)]


def listed(value):
    """Models often write null for an empty list, a bare string for a one-item list, or a list as a string."""
    if value is None:
        return []
    if isinstance(value, str):
        return json.loads(value) if value.startswith("[") else [value]
    return value


Ids = Annotated[list[SafeId], BeforeValidator(listed)]


class Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


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


class TasksConfig(Section):
    # Per level, the band of the solver's right-answer rate a task of that level measures in.
    bands: list[tuple[float, float]] = Field(min_length=1)
    max_answer_rows: int = Field(default=5, ge=1, le=50)
    styles: list[Text] = Field(min_length=1)  # how a question is asked: a chapter's brief draws one per entry

    @model_validator(mode="after")
    def ordered(self):
        if any(not 0 <= lo <= hi <= 1 for lo, hi in self.bands):
            raise ValueError("a band is [low, high] within 0 to 1")
        return self


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
    # a request people bring to an assistant. Off unless a world's config sets it (S3 and its chapters left it out).
    promise_rate: float = Field(default=0.0, ge=0, le=0.2)
    # Each app's routine posts per workday (builds, alerts, staging deploys, tickets; a reminder posts once): code's,
    # never a person's. Off unless a world's config sets it, with the apps its organization lists.
    bot_posts: float = Field(default=0.0, ge=0, le=30)


class Needs(Section):
    """What a level needs of the facts a board entry rests on, checked when the plan is written: the channels its facts
    are first stated in; the relations among them; its near-misses (decoys, or values whose change its actor cannot
    see, on its answer's subject and attribute, readable and unretracted in its actor's sight:
    `contracts.fact_measures`); and, for a chapter's brief, `values`, how many values its answer's subject and
    attribute takes (the answer's among them), and `fresh`, an answer first stated in the chapter, so that what it
    plants is new. `hidden`, an answer stated where it is harder to see than its near-misses (a thread reply, a DM, or
    a later day), is checked when the answer is posted. `named`, at most how many of its evidence's channel names and
    identifiers (words with digits, as PWSQL-03) a question names, is shown to the author and checked nowhere. Search
    rank and depth are measured and reported, not needed: at a world's size every channel reads in one page."""

    channels: int = Field(default=0, ge=0)
    relations: int = Field(default=0, ge=0)
    decoys: int = Field(default=0, ge=0)
    hidden: bool = False
    named: int | None = Field(default=None, ge=0)
    values: int = Field(default=0, ge=0)
    fresh: bool = False

    def facts(self) -> bool:
        """Whether a level needs anything of its facts, so its slots are planned on the board before day 1."""
        return bool(
            self.channels or self.relations or self.decoys or self.hidden or self.values or self.fresh
        )


class Category(Section):
    """One task category: where its gold answer comes from, the answers it may have, what each level means, the
    concrete concepts a task of each level may require (the seed draws one per task), and what code checks a task of
    each level for."""

    gold: Literal["sql", "ledger", "hybrid"]
    answer_types: list[Literal["text", "set", "number", "refusal", "status"]] = Field(min_length=1)
    definition: Text
    levels: list[Text] = Field(min_length=1)
    concepts: list[list[Text]]
    needs: list[Needs] | None = None

    @model_validator(mode="after")
    def per_level(self):
        if len(self.concepts) != len(self.levels) or not all(self.concepts):
            raise ValueError("concepts has a non-empty list for each level")
        if self.needs is not None and len(self.needs) != len(self.levels):
            raise ValueError("needs has a table for each level")
        return self


class ReviewRounds(Section):
    """Bounded allowances: final reviews per run."""

    final: int = Field(default=4, ge=1, le=20)


class AuthorSettings(Section):
    """The world author: one agent writes the world in time order."""

    tolerance: float = Field(default=0.3, ge=0, le=1)  # how far a day's message count may be from its quota
    weekend: float = Field(default=0.1, ge=0, le=1)  # a weekend day's share of a workday's messages
    share_tolerance: float = Field(
        default=0.1, ge=0, le=1
    )  # reply, reaction and DM shares, checked at the end
    mention_tolerance: float = Field(
        default=0.05, ge=0, le=1
    )  # the share of messages mentioning someone, per day
    review_days: list[int] = Field(
        default_factory=lambda: [5]
    )  # the judge reviews the world after these days
    # The solver's tries of a task (8: at 4, a task right 80% of the time shows 4 of 4 in 41% of probes); the tasks
    # in one judge review (a round's reviews run side by side); the solver's tries at once; the witness's tries of a
    # task below its band's floor.
    tries: int = Field(default=8, ge=1, le=16)
    review_chunk: int = Field(default=5, ge=1, le=50)
    solvers: int = Field(default=8, ge=1, le=64)
    witness_tries: int = Field(default=2, ge=0, le=8)
    keepalive: float = Field(
        default=600, gt=0, le=1700
    )  # seconds between pings of the author's idle VM (30-min timeout)
    day_attempts: int = Field(
        default=2, ge=1, le=5
    )  # a day that cannot close is written again from its start
    plan_attempts: int = Field(default=3, ge=1, le=10)  # a plan that cannot be recorded is written again


class Config(Section):
    """One run, configured in one file."""

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
    tasks: TasksConfig
    taxonomy: dict[Annotated[str, Field(pattern=r"^[a-z_]+$")], Category] = Field(min_length=1)
    review_rounds: ReviewRounds = Field(default_factory=ReviewRounds)
    author: AuthorSettings = Field(default_factory=AuthorSettings)
    env: PipelineConfig = Field(default_factory=PipelineConfig)
    answer_judge: vf.JudgeConfig = vf.JudgeConfig(model="openai/gpt-6-sol")

    @model_validator(mode="after")
    def banded(self):
        if short := [n for n, c in self.taxonomy.items() if len(c.levels) > len(self.tasks.bands)]:
            raise ValueError(f"[tasks] bands has one band for each level of {short}")
        return self

    @model_validator(mode="after")
    def secure(self):
        if self.env.retries.max_retries:
            raise ValueError("whole-pipeline retries would replay mutable work")
        for name in ("author", "judge", "solver", "witness"):
            agent = getattr(self.env, name)
            runtime = agent.runtime
            if not isinstance(agent.harness, NullHarnessConfig) and (
                not isinstance(runtime, vf.PrimeConfig)
                or not runtime.vm
                or runtime.allow
                or runtime.block != ["*"]
                or "@sha256:" not in runtime.image
            ):
                raise ValueError(
                    f"{name} executes code, so it requires an immutable Prime VM with framework-only egress"
                )
            if not agent.model:
                raise ValueError(f"{name} needs an explicit model")
        return self


def load_config(path: Path) -> Config:
    config = Config.model_validate(tomllib.loads(path.read_text()))
    config.output, config.corpus = rooted(config.output), rooted(config.corpus)
    return config
