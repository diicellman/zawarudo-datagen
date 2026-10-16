"""The proposer's task: one turn of its work in its VM, opened with the turn's prompt; the guide it works by, with the
forge's own numbers."""

from pathlib import Path
from typing import Self

import verifiers.v1 as vf
from worldgen_slack.taskset import WorldTaskData

from ...prompt import prompt, render
from ...session import AuthoringToolsConfig, tools_config
from .tools import ProposerTools

GUIDE = prompt(__file__, "guide.md")


def forge_guide(settings: dict) -> str:
    """The proposer's guide, with the forge's own numbers. It names no model: the proposer knows the student by its
    tries."""
    return render(GUIDE, tries=str(settings["tries"]))


class ProposerConfig(vf.TaskConfig):
    tools: AuthoringToolsConfig


class ProposerTask(vf.Task[WorldTaskData, vf.State, ProposerConfig]):
    """One turn of the proposer's work in its VM: the env opens it with the turn's prompt, and the proposer works
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


def initial_prompt(ids: list[str], written: int, goal: int, family: str) -> str:
    return (
        f"The initial set holds {written} of its {goal} tasks. Write one task with world_add_task for each open id: "
        f"{', '.join(ids)}. Lean this turn toward the {family} family, and spread the asks across people and "
        "storylines; /task/memory/tasks.md lists what is written, so write new requests, never a repeat. Walk each "
        "one's route as its asker first. Write in /task/notes/plan.md what you looked for and could not find, then end "
        "your turn."
    )


def adapt_prompt(ids: list[str]) -> str:
    return (
        "/task/memory/adapt.md gives, for each open id, the task it is a version of, how it went, and the version to "
        f"write: harder, easier or repaired. Write each with world_add_task, its variant_of and why as the page says: "
        f"{', '.join(ids)}. A version stays a request a colleague would make, of its family. Walk each one's route as "
        "its asker first. Write in /task/notes/plan.md what you looked for and could not find, then end your turn."
    )
