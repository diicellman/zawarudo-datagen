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

from ..cast import Premises, census, pick_premise, user_id
from ..organization import Organization

FILE_GUIDE = """/task/input.json holds this phase's inputs; when it has feedback, that is what code or a reviewer
rejected in your last document. /task/world.sqlite is a read-only SQLite copy of the workspace built so far, and
/task/schema.sql is its schema. /task/schemas.json is the JSON schema of the document to write. Write the document
to /task/<phase>.json, where <phase> is input.json's phase.

"""


PREMISE_GUIDE = """Phase premise: propose exactly premise_count companies in `sector`; the run seed picks one, and the
rest of the run builds that company's Slack workspace. A premise has:
- company: its name, whose first word differs from the other premises' first words and from every name in used_names;
- niche, region, size and culture: what it does, where in `country` it is, how big it is (it employs `pool` people)
  and how it works;
- cast: the kinds of people who work there;
- staffing: how many people of each occupation in `occupations` it employs. The counts add up to exactly `pool`, each
  at most that occupation's count in `occupations`; code hires exactly these people.
Write in `language`.
"""


ORGANIZATION_GUIDE = """Phase organization: give the company's people their jobs, and choose its conversations.
- people: every candidate in `candidates` once, by user_id. A person keeps the candidate's name and timezone. title and
  team are the job they hold here; a few set a Slack status (status_emoji like ":palm_tree:", status_text like "out
  until Monday"; no times).
- channels: the conversations the company works in. A public channel is readable by every person, a private channel
  only by its members; an im is a direct message between exactly 2 people, an mpim a group DM of 3 or more. Public and
  private channels have a name of lowercase letters, digits, - and _, and a topic and purpose in `language`; DMs have
  no name. members are user_ids of chosen people. Code adds more direct messages between people who work together.
- routines: for each public and private channel, 3 to 5 recurring kinds of conversation people have there, each with
  the probability that one of the channel's everyday conversations is of that kind. dm_routines: the same for direct
  messages. Code draws each everyday conversation's kind from them; the storylines are written apart.
- integrations: the apps the company runs in Slack (when `apps` is true, 2 to 5 of them; otherwise none). Each has a
  name (its bot's), a kind (deploys, alerts, tickets, ci or reminders), the channel it posts in, 3 to 12 subjects it
  posts about (services, checks, ticket subjects, reminders; no times), and for tickets the prefix of their numbers.
  Code writes every bot post: routine ones on workdays, and an event the ledger says an app reports.
"""


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
