"""The setup documents the world author writes first: the premise (the company) and the organization (its people
and conversations)."""

from ..contracts import Organization, Premises, census, pick_premise, user_id
from .author import AuthorTask

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
  team are the job they hold here.
- channels: the conversations the company works in. A public channel is readable by every person, a private channel
  only by its members; an im is a direct message between exactly 2 people, an mpim a group DM of 3 or more. Public and
  private channels have a name of lowercase letters, digits, - and _, and a topic and purpose in `language`; DMs have
  no name. members are user_ids of chosen people. Code adds more direct messages between people who work together.
- routines: for each public and private channel, 3 to 5 recurring kinds of conversation people have there, each with
  the probability that one of the channel's everyday conversations is of that kind. dm_routines: the same for direct
  messages. Code draws each everyday conversation's kind from them; the storylines are written apart.
"""

CARD = set(
    "name sex age education_level bachelors_field occupation city state timezone professional_persona".split()
)


class SynthesizerTask(AuthorTask):
    outputs = {"premise": Premises, "organization": Organization}
    guides = {"premise": PREMISE_GUIDE, "organization": ORGANIZATION_GUIDE}


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
        "feedback": state.feedback,
    }
