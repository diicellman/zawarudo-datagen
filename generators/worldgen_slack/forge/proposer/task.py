"""The proposer's task: one turn of its work in its VM, opened with the turn's prompt; the guide it works by, with the
forge's own numbers."""

from pathlib import Path
from typing import Self

import verifiers.v1 as vf
from worldgen_slack.taskset import WorldTaskData

from ...prompt import prompt, render
from ...session import AuthoringToolsConfig, tools_config
from ..tasks import SKILLS
from .tools import ProposerTools

GUIDE = prompt(__file__, "guide.md")


def forge_guide(settings: dict) -> str:
    """The proposer's guide, with the forge's own numbers: the student's tries and its own turn's budget. It names no
    model: the proposer knows the student by its tries."""
    budget = settings["env"]["proposer"]["harness"]["max_total_tokens"]
    return render(GUIDE, tries=str(settings["tries"]), budget=f"{budget:,}")


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


def aims(skills: dict[str, str | None]) -> str:
    """What the turn's skills ask, each once, for a prompt; empty when no id aims at one."""
    named = sorted({s for s in skills.values() if s})
    return (
        " Each id that names a skill needs it of whoever answers: "
        + "; ".join(f"{s}, {SKILLS[s]}" for s in named)
        + "."
        if named
        else ""
    )


def initial_prompt(
    families: dict[str, str | None], skills: dict[str, str | None], written: int, goal: int
) -> str:
    ids = ", ".join(
        f"{t} ({', '.join(x for x in (f, skills.get(t)) if x)})" if f or skills.get(t) else t
        for t, f in families.items()
    )
    return (
        f"The initial set holds {written} of its {goal} tasks. Write one task with world_add_task for each open id, of "
        f"the family it names: {ids}.{aims(skills)} Spread the asks across people and storylines; /task/memory/tasks.md "
        "shows what is written and what material no task rests on yet, so write new requests, never a repeat. Walk "
        "each one's route as its asker first. Write in /task/notes/forge.md what you looked for and could not find, "
        "then end your turn."
    )


def adapt_prompt(ids: list[str], skills: dict[str, str | None] | None = None) -> str:
    skills = skills or {}
    named = ", ".join(f"{t} ({skills[t]})" if skills.get(t) else t for t in ids)
    return (
        "/task/memory/adapt.md gives, for each open id, the task it is a version of, how it went, and the version to "
        f"write: harder, easier or repaired. Write each with world_add_task, its variant_of and why as the page says: "
        f"{named}.{aims(skills)} A version stays a request a colleague would make, of its family. Walk each one's route "
        "as its asker first. Write in /task/notes/forge.md what you looked for and could not find, then end your turn."
    )
