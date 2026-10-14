"""The setup documents the world author writes first, in its VM's first block: the premise (the company) and the
organization (its people and conversations). Each is written by the file protocol: the phase's inputs, the world and
the phase's contract in, one document out."""

import json
from pathlib import Path
from typing import ClassVar, Self

import verifiers.v1 as vf
from pydantic import BaseModel, Field
from worldgen_slack.taskset import WorldTaskData
from worldgen_slack.world import SCHEMA, file_hash

from ...prompt import prompt
from ..cast import Premises, census, pick_premise, user_id
from ..organization import Organization

FILE_GUIDE = prompt(__file__, "files.md")
PREMISE_GUIDE = prompt(__file__, "premise.md")
ORGANIZATION_GUIDE = prompt(__file__, "organization.md")


CARD = set(
    "name sex age education_level bachelors_field occupation city state timezone professional_persona".split()
)


class SetupConfig(vf.TaskConfig):
    files: dict[str, str] = Field(default_factory=dict, exclude=True, repr=False)
    world: str = Field(default="", exclude=True, repr=False)


class SetupTask(vf.Task[WorldTaskData, vf.State, SetupConfig]):
    """The setup block: each phase's document (the premise, then the organization) written to /task/<phase>.json from
    the phase's files."""

    NEEDS_CONTAINER = True
    outputs: ClassVar[dict[str, type[BaseModel]]] = {"premise": Premises, "organization": Organization}
    guides: ClassVar[dict[str, str]] = {"premise": PREMISE_GUIDE, "organization": ORGANIZATION_GUIDE}

    @classmethod
    def create(cls, context: dict, attempt: str, world: Path) -> Self:
        return cls(
            WorldTaskData(
                prompt="Read /task/guide.md and /task/input.json. Write the current phase's document to /task/<phase>.json.",
                attempt=attempt,
                world_hash=file_hash(world),
                network_allow=[],
                network_block=["*"],
            ),
            SetupConfig(files=cls.files(context), world=str(world)),
        )

    @classmethod
    def files(cls, context: dict) -> dict[str, str]:
        """What a phase's agent reads: its inputs, its contract, its output schema and the world's schema."""
        phase = context["phase"]
        return {
            "input.json": json.dumps(context, ensure_ascii=False, indent=1),
            "guide.md": FILE_GUIDE + cls.guides[phase],
            "schemas.json": json.dumps(cls.outputs[phase].model_json_schema()),
            "schema.sql": SCHEMA.read_text(),
        }

    async def setup(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        for name, text in self.config.files.items():
            await runtime.write("/task/" + name, text.encode())
        await runtime.write("/task/world.sqlite", Path(self.config.world).read_bytes())


def premise_context(settings, used) -> dict:
    countries, occupations = census(settings.personas, used["people"])
    return {
        "phase": "premise",
        "sector": settings.sector,
        "language": settings.language,
        "premise_count": settings.premise_count,
        "used_names": used,
        "country": ", ".join(countries),
        "occupations": dict(occupations.most_common()),
        "pool": settings.personas.pool,
    }


def parse_premise(raw: str, settings, used):
    """The author proposes distinct premises; the run seed, not the model, picks one."""
    premises = Premises.model_validate_json(raw)
    available, pool = census(settings.personas, used["people"])[1], settings.personas.pool
    for p in premises.premises:
        if sum(p.staffing.values()) != pool:
            raise ValueError(
                f"{p.company}: staffing counts add up to exactly {pool}, not {sum(p.staffing.values())}"
            )
        for occupation, count in p.staffing.items():
            if count > available[occupation]:
                raise ValueError(f"{p.company}: only {available[occupation]} unused people are {occupation}")
    return pick_premise(premises, settings.premise_count, used["companies"], settings.seed)


def candidates(cast) -> list[dict]:
    """The cards people are chosen by; code keeps the full profile and pairs it with the chosen person."""
    return [
        {"user_id": user_id(p.uuid)}
        | p.model_dump(include=CARD)
        | {"typing": p.typing.model_dump(exclude={"id", "messages"})}
        for p in cast
    ]


def organization_context(settings, state, used) -> dict:
    return {
        "phase": "organization",
        "premise": state.premise.model_dump(),
        "language": settings.language,
        "candidates": candidates(state.cast),
        "used_names": used["people"],
        "apps": settings.activity.bot_posts > 0,
        "feedback": state.feedback,
    }
