"""The proposer's task: one round of its work in its VM, opened with the round's prompt; the guide it works by, with
the forge's own numbers."""

from pathlib import Path
from typing import Self

import verifiers.v1 as vf
from worldgen_slack.taskset import WorldTaskData

from ...prompt import prompt, render
from ...session import AuthoringToolsConfig, tools_config
from ..tasks import Task
from .tools import ProposerTools

# the most near-miss queries a task may name
NEAR_SQL = next(m.max_length for m in Task.model_fields["near_sql"].metadata if hasattr(m, "max_length"))
GOLD_SQL = render(prompt(__file__, "gold.md"), near_sql=str(NEAR_SQL))
GUIDE = prompt(__file__, "guide.md")


def forge_guide(settings: dict) -> str:
    """The proposer's guide, with the forge's own numbers: the solver's tries, and the right-answer rates of the
    middle levels (the bands between the first and the last). It names no model: the proposer knows the solver by
    its tries."""
    tries, bands = settings["tries"], settings["bands"]
    middle = bands[1:-1] or bands
    low, high = min(b[0] for b in middle), max(b[1] for b in middle)
    return render(GUIDE, tries=str(tries), low=f"{low:.0%}", high=f"{high:.0%}") + GOLD_SQL


class ProposerConfig(vf.TaskConfig):
    tools: AuthoringToolsConfig


class ProposerTask(vf.Task[WorldTaskData, vf.State, ProposerConfig]):
    """One round of the proposer's work in its VM: the env opens it with the round's prompt, and the proposer works
    through its world tools."""

    NEEDS_CONTAINER = True

    @classmethod
    def toolsets(cls, config: ProposerConfig) -> list[vf.Toolset]:
        return [ProposerTools(config.tools)]

    @classmethod
    def create(cls, world: Path, context: dict, attempt: str) -> Self:
        tools = tools_config("forge", 0, world, context)
        data = WorldTaskData(
            prompt=None,
            system_prompt=forge_guide(context["settings"]),
            attempt=attempt,
            world_hash=tools.db_hash,
            network_allow=[],
            network_block=["*"],
        )
        return cls(data, ProposerConfig(tools=tools))


def forge_prompt(round_: int, rounds: int, ids: list[str], held: int, goal: int) -> str:
    return (
        f"Round {round_}, of at most {rounds}. The archive holds {held} of the {goal} tasks it aims for; a full place "
        "takes a task only if the solver's tries are more mixed on it, or it brings a structure the place lacks. Read "
        "/task/memory/archive.md and board.md, and your notes. Then write, for each open id, one request its asker "
        f"would really make, with world_add_task: {', '.join(ids)}. Walk each one's route as its asker first. "
        "Write in /task/notes/plan.md what you tried and why, then end your turn."
    )
