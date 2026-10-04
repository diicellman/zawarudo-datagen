"""Validated run settings; all role overrides use native Verifiers configs."""

import tomllib
from pathlib import Path
from typing import Annotated, Literal
import verifiers.v1 as vf
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator
from verifiers.v1.configs.agent import TimeoutConfig
from verifiers.v1.configs.retries import RetryConfig
from verifiers.v1.harnesses.null.harness import NullHarnessConfig
from worldgen_slack.config import role

ROOT = Path(__file__).resolve().parents[2]


def world_author(model: str) -> vf.AgentConfig:
    """The world author (v7): an rlm coding agent in one VM for the whole world, with one interaction per block of
    work (setup, plan, each day, tasks); each interaction gets these turn, token and time budgets."""
    base = role(model, author=True)
    return base.model_copy(
        update={
            "harness": base.harness.model_copy(update={"max_total_tokens": 600_000}),
            "runtime": base.runtime.model_copy(update={"idle_timeout": 7200}),
            "max_turns": 200,
            "timeout": base.timeout.model_copy(update={"rollout": 5400}),
        }
    )


class PipelineConfig(vf.EnvConfig):
    taskset: vf.TasksetConfig = vf.TasksetConfig(id="worldgen-slack")
    author: vf.AgentConfig = world_author("z-ai/glm-5.3")
    synthesizer: vf.AgentConfig = role("z-ai/glm-5.3", author=True)
    builder: vf.AgentConfig = role("z-ai/glm-5.3", author=True)
    judge: vf.AgentConfig = role("openai/gpt-6-sol")
    # The solver and the writer execute no code (null harness; the solver's Slack tools are host-side servers),
    # so a local process replaces the VM.
    solver: vf.AgentConfig = role("z-ai/glm-5.2", solver=True).model_copy(
        update={"runtime": vf.SubprocessConfig()}
    )
    writer: vf.AgentConfig = vf.AgentConfig(
        model="z-ai/glm-5.3",
        harness=NullHarnessConfig(id="null"),
        runtime=vf.SubprocessConfig(),
        # Low effort is ~12x faster for scene writing (probe: 6 s vs 70-90 s) at similar length.
        sampling=vf.Sampling(temperature=0.9, max_tokens=32_000, reasoning_effort="low"),
        max_turns=3,
        timeout=TimeoutConfig(setup=300, rollout=900, finalize=60),
        retries=RetryConfig(max_retries=0),
    )
    retries: RetryConfig = RetryConfig(max_retries=0)
    # The GLM account allows 8 concurrent requests; GLM writing and GLM solving never overlap.
    max_concurrent_agents: int | None = 8


def rooted(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


RootedPath = Annotated[Path, AfterValidator(rooted)]
Text = Annotated[str, Field(min_length=1)]


class Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SeedDataConfig(Section):
    path: RootedPath


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
    count: int = Field(gt=0, le=100)
    per_storyline: int = Field(default=5, gt=0, le=20)
    messages_per_task: int = Field(default=15, ge=1, le=200)
    max_answer_rows: int = Field(default=5, ge=1, le=50)
    styles: list[Text] = Field(min_length=1)


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


class Category(Section):
    """One task category: where its gold answer comes from, the answers it may have, what each level means, the
    concrete concepts a task of each level may require (the seed draws one per task), and, for categories whose
    answers rest on several facts, how many channels a level's facts are spread over."""

    gold: Literal["sql", "ledger", "hybrid"]
    answer_types: list[Literal["text", "set", "number", "refusal"]] = Field(min_length=1)
    definition: Text
    levels: list[Text] = Field(min_length=1)
    concepts: list[list[Text]]
    spread: list[int] | None = None

    @model_validator(mode="after")
    def per_level(self):
        if len(self.concepts) != len(self.levels) or not all(self.concepts):
            raise ValueError("concepts has a non-empty list for each level")
        if self.spread is not None and (len(self.spread) != len(self.levels) or min(self.spread) < 1):
            raise ValueError("spread has a channel count of at least 1 for each level")
        return self


class ReviewRounds(Section):
    """Separate, bounded allowances: ledger and final attempts per run; build attempts per storyline."""

    ledger: int = Field(default=8, ge=1, le=20)
    build: int = Field(default=8, ge=1, le=20)
    final: int = Field(default=4, ge=1, le=20)


class Acceptance(Section):
    """A review accepts when no issue blocks; minor issues block only if configured."""

    minor_issues_block: bool = False


class AuthorSettings(Section):
    """The world author (v7): one agent writes the world in time order. Off, the v6 pipeline runs."""

    enabled: bool = False
    tolerance: float = Field(default=0.3, ge=0, le=1)  # how far a day's message count may be from its quota
    weekend: float = Field(default=0.1, ge=0, le=1)  # a weekend day's share of a workday's messages
    share_tolerance: float = Field(
        default=0.1, ge=0, le=1
    )  # reply, reaction and DM shares, checked at the end
    review_days: list[int] = Field(
        default_factory=lambda: [5]
    )  # the judge reviews the world after these days
    probe_solves: int = Field(default=4, ge=1, le=16)  # solver runs per task per probe
    probe_rounds: int = Field(default=2, ge=0, le=5)  # turns to harden tasks after their probes
    probe_budget: int = Field(default=120, ge=0, le=2000)  # solver runs for probing, per world
    day_attempts: int = Field(
        default=2, ge=1, le=5
    )  # a day that cannot close is written again from its start


class Config(Section):
    """One run, configured in one file."""

    sector: Text
    seed: int = Field(default=0, ge=0)
    output: Path
    corpus: Path = Path("data")
    language: Text = "English"
    premise_count: int = Field(default=12, ge=2, le=30)
    personas: PersonasConfig
    seed_data: SeedDataConfig | None = None
    calendar: CalendarConfig = Field(default_factory=CalendarConfig)
    activity: ActivityConfig = Field(default_factory=ActivityConfig)
    tasks: TasksConfig
    taxonomy: dict[Annotated[str, Field(pattern=r"^[a-z_]+$")], Category] = Field(min_length=1)
    review_rounds: ReviewRounds = Field(default_factory=ReviewRounds)
    acceptance: Acceptance = Field(default_factory=Acceptance)
    author: AuthorSettings = Field(default_factory=AuthorSettings)
    solves_per_task: int = Field(default=4, ge=1, le=16)
    research_budget_usd: float = Field(default=900.0, gt=0)
    env: PipelineConfig = Field(default_factory=PipelineConfig)
    answer_judge: vf.JudgeConfig = vf.JudgeConfig(model="openai/gpt-6-sol")

    @model_validator(mode="after")
    def secure(self):
        if self.env.retries.max_retries:
            raise ValueError("whole-pipeline retries would replay mutable work")
        for name in ("author", "synthesizer", "builder", "judge", "solver"):
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
        if not isinstance(self.env.writer.harness, NullHarnessConfig) or not self.env.writer.model:
            raise ValueError("writer must use the tool-less null harness with an explicit model")
        return self


def load_config(path: Path) -> Config:
    config = Config.model_validate(tomllib.loads(path.read_text()))
    for field in ("output", "corpus"):
        if not getattr(config, field).is_absolute():
            setattr(config, field, (ROOT / getattr(config, field)).resolve())
    return config
