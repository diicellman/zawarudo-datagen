"""The company and its people: premises, the personas drawn for the cast, their typing cards and their ids."""

import bisect
import json
import random
from collections import Counter
from pathlib import Path
from typing import Annotated

from pydantic import AfterValidator, Field
from worldgen_slack.release import NonEmptyText, SafeId, StrictModel
from worldgen_slack.world import digest

from .clock import zone
from .text import normalized


class Premise(StrictModel):
    company: NonEmptyText
    niche: NonEmptyText
    region: NonEmptyText
    size: NonEmptyText
    culture: NonEmptyText
    cast: NonEmptyText
    staffing: dict[NonEmptyText, Annotated[int, Field(ge=1)]]


class Premises(StrictModel):
    premises: list[Premise] = Field(min_length=2)


def pick_premise(premises: Premises, count: int, used: list[str], seed: int) -> Premise:
    """Entropy comes from code: the model proposes distinct options, the run seed chooses one."""
    names = [normalized(p.company).split()[0] for p in premises.premises]
    taken = {normalized(name).split()[0] for name in used if normalized(name)}
    if len(names) != count or len(set(names)) != count:
        raise ValueError(f"propose exactly {count} premises whose company names start with distinct words")
    if reused := sorted(set(names) & taken):
        raise ValueError(f"company names reuse names from used_names: {reused}")
    return random.Random(seed).choice(premises.premises)


class Typing(StrictModel):
    """How one real Slack user types: medians and shares over their messages."""

    id: SafeId
    messages: int
    median_words: float
    short_share: float
    long_share: float
    question_share: float
    lowercase_share: float
    emoji_share: float


class SeedPersona(StrictModel):
    """A person from the seed persona file (scripts/worldgen_slack/personas.sql); `typing` is paired by the seed."""

    uuid: SafeId
    name: NonEmptyText
    sex: NonEmptyText
    age: int
    marital_status: NonEmptyText
    education_level: NonEmptyText
    bachelors_field: str | None
    occupation: NonEmptyText
    city: NonEmptyText
    state: NonEmptyText
    country: NonEmptyText
    timezone: Annotated[str, AfterValidator(zone)]
    persona: NonEmptyText
    professional_persona: NonEmptyText
    cultural_background: NonEmptyText
    skills: list[str]
    hobbies: list[str]
    typing: Typing | None = None


def census(personas, used=()) -> tuple[list[str], Counter]:
    """The persona file's countries, and how many people unused elsewhere in the corpus hold each occupation."""
    people, taken = [json.loads(line) for line in personas.path.read_text().splitlines()], set(used)
    supply = Counter(p["occupation"] for p in people if p["name"] not in taken)
    return sorted({p["country"] for p in people}), supply


def pick_cast(personas, seed: int, used: list[str], staffing: dict[str, int]) -> list[SeedPersona]:
    """Entropy comes from code: the run seed draws the premise's staffing, occupation by occupation, from people
    unused elsewhere in the corpus, and pairs each candidate with a real user's typing."""
    rng, taken, cast, roles = random.Random(seed), set(used), [], Counter(staffing)
    lines = personas.path.read_text().splitlines()
    for line in rng.sample(lines, len(lines)):
        if not +roles:
            break
        person = json.loads(line)
        if roles[person["occupation"]] > 0 and person["name"] not in taken:
            roles[person["occupation"]] -= 1
            cast.append(SeedPersona.model_validate(person))
    if short := +roles:
        raise ValueError(f"the persona file has too few people unused in the corpus for {dict(short)}")
    typing = rng.sample(personas.typing.read_text().splitlines(), len(cast))
    return [p.model_copy(update={"typing": Typing.model_validate_json(t)}) for p, t in zip(cast, typing)]


def cards(cast: list) -> dict[str, dict]:
    """Each person's typing card, by user id: how they write, as the author and the judge read it."""
    return {
        user_id(p.uuid): p.typing.model_dump(exclude={"id", "messages"}) | {"timezone": p.timezone}
        for p in cast
        if p.typing
    }


def selves(cast: list) -> dict[str, dict]:
    """Each person as their seed draws them, by user id: who they are beyond their title, as the author reads it. The
    seed's own job was drawn for another occupation, so its professional persona is left out."""
    return {
        user_id(p.uuid): {
            "age": p.age,
            "home": f"{p.city}, {p.state}",
            "education": p.education_level.replace("_", " "),
            "persona": p.persona,
            "background": p.cultural_background,
            "hobbies": p.hobbies[:6],
        }
        for p in cast
    }


def user_id(seed_uuid: str) -> str:
    return "U" + digest(["user", seed_uuid])[:10].upper()


def company_zone(cast: list[SeedPersona], chosen: list[str]) -> str:
    """The company clock: the timezone most of its people live in (alphabetical on a tie)."""
    counts = Counter(p.timezone for p in cast if user_id(p.uuid) in chosen)
    return min(counts, key=lambda z: (-counts[z], z))


class Gaps:
    """How long real Slack users take to answer: seconds between consecutive messages of a channel, as quantiles
    (scripts/worldgen_slack/personas.sql). A line with no pause takes a gap under an hour; "hours" takes one above."""

    def __init__(self, path: Path):
        self.quantiles = json.loads(path.read_text().splitlines()[0])["quantiles"]
        self.hour = bisect.bisect_left(self.quantiles, 3600) / (len(self.quantiles) - 1)

    def sample(self, rng: random.Random, pause: str) -> int:
        u = rng.random() * self.hour if pause == "" else self.hour + rng.random() * (1 - self.hour)
        position = u * (len(self.quantiles) - 1)
        low = int(position)
        high = min(low + 1, len(self.quantiles) - 1)
        seconds = self.quantiles[low] + (self.quantiles[high] - self.quantiles[low]) * (position - low)
        return int(seconds * 1_000_000) + rng.randrange(1, 1_000_000)
