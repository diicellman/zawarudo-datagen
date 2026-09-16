"""Validated run settings; all role overrides use native Verifiers configs."""

import tomllib
from pathlib import Path

import verifiers.v1 as vf
from pydantic import BaseModel, ConfigDict, Field, model_validator
from verifiers.v1.configs.retries import RetryConfig
from worldgen_slack.config import role

ROOT = Path(__file__).resolve().parents[2]


class PipelineConfig(vf.EnvConfig):
    taskset: vf.TasksetConfig = vf.TasksetConfig(id="worldgen-slack")
    synthesizer: vf.AgentConfig = role("openai/gpt-5.6-terra", author=True)
    builder: vf.AgentConfig = role("openai/gpt-5.6-terra", author=True)
    judge: vf.AgentConfig = role("openai/gpt-5.6-sol")
    solver: vf.AgentConfig = role("z-ai/glm-5.2", solver=True)
    retries: RetryConfig = RetryConfig(max_retries=0)
    max_concurrent_agents: int | None = 2


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sector: str = Field(min_length=1)
    task_count: int = Field(gt=0, le=100)
    group_size: int = Field(default=10, gt=0, le=10)
    seed: int = Field(default=0, ge=0)
    output: Path
    max_review_rounds: int = Field(default=5, ge=1, le=20)
    target_messages: int = Field(default=2000, ge=1, le=10_000)
    research_budget_usd: float = Field(default=900.0, gt=0)
    env: PipelineConfig = Field(default_factory=PipelineConfig)
    answer_judge: vf.JudgeConfig = vf.JudgeConfig(model="openai/gpt-5.6-sol")

    @model_validator(mode="after")
    def secure(self):
        if self.env.retries.max_retries:
            raise ValueError("whole-pipeline retries would replay mutable work")
        for name in ("synthesizer", "builder", "judge", "solver"):
            agent = getattr(self.env, name)
            runtime = agent.runtime
            if (
                not isinstance(runtime, vf.PrimeConfig)
                or not runtime.vm
                or runtime.allow
                or runtime.block != ["*"]
                or "@sha256:" not in runtime.image
            ):
                raise ValueError(f"{name} requires an immutable Prime VM with framework-only egress")
            if not agent.model:
                raise ValueError(f"{name} needs an explicit model")
        return self


def load_config(path: Path) -> Config:
    config = Config.model_validate(tomllib.loads(path.read_text()))
    if not config.output.is_absolute():
        config.output = (ROOT / config.output).resolve()
    return config
