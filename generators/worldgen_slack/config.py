"""Validated run settings; all role overrides use native Verifiers configs."""

import tomllib
from pathlib import Path
from typing import Annotated
import verifiers.v1 as vf
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator
from verifiers.v1.configs.agent import TimeoutConfig
from verifiers.v1.configs.retries import RetryConfig
from verifiers.v1.harnesses.null.harness import NullHarnessConfig
from worldgen_slack.config import role

ROOT = Path(__file__).resolve().parents[2]


class PipelineConfig(vf.EnvConfig):
    taskset: vf.TasksetConfig = vf.TasksetConfig(id="worldgen-slack")
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


class SeedDataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: RootedPath


class PersonasConfig(BaseModel):
    """A seeded cast: the run seed draws `pool` candidates from a persona file and pairs each with a real Slack
    user's typing statistics. Both files come from scripts/worldgen_slack/personas.sql."""

    model_config = ConfigDict(extra="forbid")
    path: RootedPath
    typing: RootedPath
    pool: int = Field(default=36, ge=4, le=200)


class ReviewRounds(BaseModel):
    """Separate, bounded allowances: catalog and final attempts per run; build attempts per group."""

    model_config = ConfigDict(extra="forbid")
    catalog: int = Field(default=8, ge=1, le=20)
    build: int = Field(default=8, ge=1, le=20)
    final: int = Field(default=4, ge=1, le=20)


class Acceptance(BaseModel):
    """A review accepts when no issue blocks; minor issues block only if configured."""

    model_config = ConfigDict(extra="forbid")
    minor_issues_block: bool = False


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sector: str = Field(min_length=1)
    task_count: int = Field(gt=0, le=100)
    group_size: int = Field(default=10, gt=0, le=10)
    seed: int = Field(default=0, ge=0)
    seed_data: SeedDataConfig | None = None
    personas: PersonasConfig | None = None
    corpus: Path = Path("data")
    language: str = Field(default="English", min_length=1)
    premise_count: int = Field(default=12, ge=2, le=30)
    output: Path
    review_rounds: ReviewRounds = Field(default_factory=ReviewRounds)
    acceptance: Acceptance = Field(default_factory=Acceptance)
    messages_per_task: int = Field(default=15, ge=1, le=200)
    solves_per_task: int = Field(default=4, ge=1, le=16)
    research_budget_usd: float = Field(default=900.0, gt=0)
    env: PipelineConfig = Field(default_factory=PipelineConfig)
    answer_judge: vf.JudgeConfig = vf.JudgeConfig(model="openai/gpt-6-sol")

    @model_validator(mode="after")
    def secure(self):
        if self.env.retries.max_retries:
            raise ValueError("whole-pipeline retries would replay mutable work")
        for name in ("synthesizer", "builder", "judge", "solver"):
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
