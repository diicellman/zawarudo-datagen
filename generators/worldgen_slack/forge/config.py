"""A forge's settings, configured in one file: the finished world it writes tasks on, the size of its initial set and
how a task is labelled (verified, measured, judged) and adapted, its seats, and how far a run carries on by itself. A
forge inherits nothing from the configuration its world was written with."""

import tomllib
from pathlib import Path

import verifiers.v1 as vf
from pydantic import Field
from verifiers.v1.configs.agent import TimeoutConfig
from worldgen_slack.config import role

from ..config import RootedPath, Seats, Section, vm_seat

# A try's time. Tries take seconds (FC1's 293: 25 s at the median, 7 min at most); the model client has no read timeout,
# so a request that hangs holds its try to the rollout's end. At 15 min a hung try is cut and tried again; its grade
# gets 5 min.
TRY_TIMEOUT = TimeoutConfig(setup=600, rollout=900, finalize=120, scoring=300)
GLM = "z-ai/glm-5.3"


def thinking(model: str) -> vf.AgentConfig:
    """A seat that answers with the Slack tools and no code (null harness, a local process), thinking hard."""
    seat = role(model, solver=True)
    return seat.model_copy(
        update={
            "runtime": vf.SubprocessConfig(),
            "sampling": seat.sampling.model_copy(update={"reasoning_effort": "high"}),
            "timeout": TRY_TIMEOUT,
        }
    )


class ForgeSeats(Seats):
    # The proposer: an rlm agent in a VM, one interaction per turn. A smaller model than the judges', so no model
    # judges its own writing. It reasons at length and reads whole query results: an author's 600k tokens ran out in 6
    # minutes on 4 versions, before it wrote one (the W1 smoke, 2026-10-16); 1.5M costs it about $0.50.
    proposer: vf.AgentConfig = vm_seat("z-ai/glm-5.3-flash", tokens=1_500_000)
    # The student whose pass rates select the tasks: no code (its Slack tools are host-side servers), a local process.
    solver: vf.AgentConfig = role("deepseek/deepseek-v4.1-flash", solver=True).model_copy(
        update={"runtime": vf.SubprocessConfig(), "timeout": TRY_TIMEOUT}
    )
    # The verifier: a stronger model with the same tools, whose tries must reach a task's gold for it to count. Never
    # the student's family: it would drop tasks exactly where the student is blind.
    verifier: vf.AgentConfig = thinking(GLM)
    # The world runs' agentic judge seat, which a forge never runs (its task judge is one call: `judge` below).
    judge: vf.AgentConfig = role(GLM)


def high(model: str = GLM) -> vf.JudgeConfig:
    return vf.JudgeConfig(model=model, sampling=vf.SamplingConfig(reasoning_effort="high"))


class ForgeConfig(Section):
    """One forge, as the paper's three sets (Synthetic Data RL, 2505.17063, 4.1-4.3): `tasks` is the size of the initial
    set the proposer writes, `batch` at a time; each task is labelled (verified, then measured by the student's `tries`,
    then judged), and `adapt` passes write a harder version of each solved task, an easier one of each unsolved task
    and a repaired one of each verified task the judge refused. A run carries on by itself through `max_losses` lost
    sessions a turn and `max_resumes` stops, until it has spent `spend_limit` dollars."""

    world: RootedPath  # the finished run: its final world.sqlite and its notes
    output: RootedPath
    tasks: int = Field(default=36, ge=1, le=2_000)
    batch: int = Field(default=12, ge=1, le=50)
    adapt: int = Field(default=1, ge=0, le=3)
    tries: int = Field(
        default=8, ge=2, le=16
    )  # the student's (at 8, a task right half the time rarely shows 0 or 8)
    verify: int = Field(default=2, ge=1, le=5)  # the verifier's; a third on a split
    solved: float = Field(
        default=0.875, gt=0, le=1
    )  # a pass rate at or above it is solved: write a harder version
    solvers: int = Field(default=8, ge=1, le=64)  # tries at once
    max_losses: int = Field(default=5, ge=0, le=20)
    max_resumes: int = Field(default=3, ge=0, le=20)
    spend_limit: float = Field(default=50.0, gt=0)
    env: ForgeSeats = Field(default_factory=ForgeSeats)
    judge: vf.JudgeConfig = Field(default_factory=high)  # the task judge: one structured call a task
    answer_judge: vf.JudgeConfig = Field(default_factory=high)


def load_forge(path: Path) -> ForgeConfig:
    return ForgeConfig.model_validate(tomllib.loads(path.read_text()))
