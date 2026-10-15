"""A forge's settings, configured in one file: the finished world it writes tasks on, the taxonomy its tasks are
placed in, its goal and rounds, how a candidate is tried and kept, and its seats. A forge inherits nothing from the
configuration its world was written with."""

import tomllib
from pathlib import Path
from typing import Annotated

import verifiers.v1 as vf
from pydantic import BeforeValidator, Field, model_validator
from verifiers.v1.configs.agent import TimeoutConfig
from worldgen_slack.config import role

from ..config import Categories, RootedPath, Seats, Section, load_taxonomy, vm_seat


def categories(value):
    """The taxonomy a config names by its file, read into its categories; categories as they are."""
    return load_taxonomy(Path(value)).categories if isinstance(value, str | Path) else value


# A try's time. Tries take seconds (FC1's 293: 25 s at the median, 7 min at most; the witness's 42 s at most); the
# model client has no read timeout, so a request that hangs holds its try to the rollout's end, and the round with it.
# At 15 min a hung try is cut and tried once more, and its grade gets 5 min.
TRY_TIMEOUT = TimeoutConfig(setup=600, rollout=900, finalize=120, scoring=300)


class ForgeSeats(Seats):
    # The proposer: an rlm agent in one VM, one interaction per round, as a world's author is.
    proposer: vf.AgentConfig = vm_seat("z-ai/glm-5.3")
    # The solver executes no code (null harness; its Slack tools are host-side servers), so a local process
    # replaces the VM.
    solver: vf.AgentConfig = role("z-ai/glm-5.2", solver=True).model_copy(
        update={"runtime": vf.SubprocessConfig(), "timeout": TRY_TIMEOUT}
    )
    # A stronger solver with the same tools, for the tasks GLM rarely answers: hard, or broken?
    # It thinks hard (at its default effort it took 2-5 calls and lost tasks the solver answered): it shows a task
    # can be answered.
    witness: vf.AgentConfig = role("openai/gpt-6-sol", solver=True).model_copy(
        update={
            "runtime": vf.SubprocessConfig(),
            "sampling": role("openai/gpt-6-sol", solver=True).sampling.model_copy(
                update={"reasoning_effort": "high"}
            ),
            "timeout": TRY_TIMEOUT,
        }
    )


class ForgeConfig(Section):
    """One forge. `tasks` is its goal, the tasks the archive holds when it is done; `rounds` the most rounds it runs
    toward it; `candidates` the candidates one round writes; `target` the tasks a place (a category and a level)
    holds; `measure` has round 0 measure the finished run's own tasks, as written, before any proposer round."""

    world: RootedPath  # the finished run: its final world.sqlite, with the ledger and the board
    output: RootedPath
    taxonomy: Annotated[Categories, BeforeValidator(categories)]  # taxonomy.toml, by its path
    tasks: int = Field(default=40, ge=1, le=10_000)
    rounds: int = Field(default=12, ge=0, le=100)
    candidates: int = Field(default=12, ge=1, le=50)
    target: int = Field(default=2, ge=1, le=100)
    measure: bool = False
    # Per level, the band of the solver's right-answer rate a task of that level measures in.
    bands: list[tuple[float, float]] = Field(min_length=1)
    max_answer_rows: int = Field(default=5, ge=1, le=50)
    # The solver's tries of a task (8: at 4, a task right 80% of the time shows 4 of 4 in 41% of probes); the tasks
    # in one judge review (a round's reviews run side by side); the solver's tries at once; the witness's tries of a
    # task below its band's floor; seconds between pings of the proposer's VM while the solver and the judge work.
    tries: int = Field(default=8, ge=1, le=16)
    review_chunk: int = Field(default=5, ge=1, le=50)
    solvers: int = Field(default=8, ge=1, le=64)
    witness_tries: int = Field(default=2, ge=0, le=8)
    keepalive: float = Field(default=600, gt=0, le=1700)
    env: ForgeSeats = Field(default_factory=ForgeSeats)
    answer_judge: vf.JudgeConfig = vf.JudgeConfig(model="openai/gpt-6-sol")

    @model_validator(mode="after")
    def works(self):
        if self.rounds == 0 and not self.measure:
            raise ValueError("a forge with no rounds only measures: set measure = true")
        if any(not 0 <= lo <= hi <= 1 for lo, hi in self.bands):
            raise ValueError("a band is [low, high] within 0 to 1")
        if short := [n for n, c in self.taxonomy.items() if len(c.levels) > len(self.bands)]:
            raise ValueError(f"bands has one band for each level of {short}")
        return self


def load_forge(path: Path) -> ForgeConfig:
    return ForgeConfig.model_validate(tomllib.loads(path.read_text()))
