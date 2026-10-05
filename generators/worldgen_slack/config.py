"""Validated run settings; all role overrides use native Verifiers configs."""

import tomllib
from pathlib import Path
from typing import Annotated, Literal
import verifiers.v1 as vf
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator
from verifiers.v1.configs.retries import RetryConfig
from verifiers.v1.harnesses.null.harness import NullHarnessConfig
from worldgen_slack.config import role

ROOT = Path(__file__).resolve().parents[2]


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
    witness: vf.AgentConfig = role("openai/gpt-6-sol", solver=True).model_copy(
        update={"runtime": vf.SubprocessConfig()}
    )
    retries: RetryConfig = RetryConfig(max_retries=0)
    # The episode's agents at once. GLM, whose account allows 8 concurrent requests, is gated on its own by
    # [author] solvers, so the judge's reviews run beside its solves.
    max_concurrent_agents: int | None = 16


def rooted(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


RootedPath = Annotated[Path, AfterValidator(rooted)]
Text = Annotated[str, Field(min_length=1)]


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
    per_100: float = Field(gt=0, le=100)  # tasks per 100 messages: a bigger world holds more tasks
    # Per level, the band of the solver's right-answer rate a task of that level aims for; hardening steers to it.
    bands: list[tuple[float, float]] = Field(min_length=1)
    batch: int = Field(default=10, ge=1, le=50)  # slots written and hardened in one author interaction
    max_answer_rows: int = Field(default=5, ge=1, le=50)
    styles: list[Text] = Field(min_length=1)

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


class Needs(Section):
    """What code checks a task of one level for when it is written: its evidence's best search rank for the
    question's own words (evidence the words never find passes); how deep its evidence sits (newer top-level messages
    in its channel, and earlier replies in its thread), as a share of the busiest channel its actor can read, so the
    need scales with the world; the channels its facts are first stated in; the relations
    among its facts; the decoys its actor can read on their subjects; and at most how many of its evidence's channel
    names and identifiers (words with digits, as PWSQL-03) the question names."""

    rank: int = Field(default=1, ge=1)
    depth: float = Field(default=0, ge=0, lt=1)
    channels: int = Field(default=0, ge=0)
    relations: int = Field(default=0, ge=0)
    decoys: int = Field(default=0, ge=0)
    named: int | None = Field(default=None, ge=0)


class Category(Section):
    """One task category: where its gold answer comes from, the answers it may have, what each level means, the
    concrete concepts a task of each level may require (the seed draws one per task), and what code checks a task of
    each level for."""

    gold: Literal["sql", "ledger", "hybrid"]
    answer_types: list[Literal["text", "set", "number", "refusal"]] = Field(min_length=1)
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


class Acceptance(Section):
    """A review accepts when no issue blocks; minor issues block only if configured."""

    minor_issues_block: bool = False


class AuthorSettings(Section):
    """The world author: one agent writes the world in time order."""

    tolerance: float = Field(default=0.3, ge=0, le=1)  # how far a day's message count may be from its quota
    weekend: float = Field(default=0.1, ge=0, le=1)  # a weekend day's share of a workday's messages
    share_tolerance: float = Field(
        default=0.1, ge=0, le=1
    )  # reply, reaction and DM shares, checked at the end
    review_days: list[int] = Field(
        default_factory=lambda: [5]
    )  # the judge reviews the world after these days
    # The solver's tries of a task, each time it is probed or reviewed; the turns a task outside its band comes back
    # to the author; the tasks in one judge review (a batch's reviews run side by side); GLM solves at once, as the
    # account allows 8 concurrent requests; the witness's tries of a task below its band's floor.
    tries: int = Field(default=4, ge=1, le=16)
    task_rounds: int = Field(default=3, ge=0, le=10)
    review_chunk: int = Field(default=5, ge=1, le=50)
    solvers: int = Field(default=8, ge=1, le=64)
    witness_tries: int = Field(default=2, ge=0, le=8)
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
    acceptance: Acceptance = Field(default_factory=Acceptance)
    author: AuthorSettings = Field(default_factory=AuthorSettings)
    env: PipelineConfig = Field(default_factory=PipelineConfig)
    answer_judge: vf.JudgeConfig = vf.JudgeConfig(model="openai/gpt-6-sol")

    @model_validator(mode="after")
    def banded(self):
        if short := [n for n, c in self.taxonomy.items() if len(c.levels) > len(self.tasks.bands)]:
            raise ValueError(f"[tasks] bands has one band for each level of {short}")
        return self

    @property
    def task_count(self) -> int:
        """How many tasks the world holds: `[tasks].per_100` for every 100 messages, and at least one."""
        return max(1, round(self.activity.messages * self.tasks.per_100 / 100))

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
    for field in ("output", "corpus"):
        if not getattr(config, field).is_absolute():
            setattr(config, field, (ROOT / getattr(config, field)).resolve())
    return config
