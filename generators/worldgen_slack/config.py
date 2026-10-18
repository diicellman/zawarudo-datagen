"""What both parts' settings are written in: the types a config's fields take, the task taxonomy, and the seats every
run has, held to code's rules. A world's settings are in world/config.py, a forge's in forge/config.py."""

import hashlib
import json
import tomllib
from pathlib import Path
from typing import Annotated, Literal

import verifiers.v1 as vf
from pydantic import AfterValidator, BaseModel, BeforeValidator, ConfigDict, Field, model_validator
from verifiers.v1.configs.retries import RetryConfig
from verifiers.v1.harnesses.null.harness import NullHarnessConfig
from worldgen_slack.config import role
from worldgen_slack.release import SafeId

ROOT = Path(__file__).resolve().parents[2]


def merged(base: dict, changes: dict) -> dict:
    return {
        k: merged(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
        for k, v in changes.items()
    } | {k: v for k, v in base.items() if k not in changes}


def pinned() -> dict:
    """What every run's provenance pins beside its own settings: the verifiers revision pyproject.toml installs, and
    the lock file."""
    revision = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["uv"]["sources"]["verifiers"][
        "rev"
    ]
    return {
        "verifiers_revision": revision,
        "lock_hash": hashlib.sha256((ROOT / "uv.lock").read_bytes()).hexdigest(),
    }


def rooted(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


RootedPath = Annotated[Path, AfterValidator(rooted)]
Text = Annotated[str, Field(min_length=1)]


def listed(value):
    """Models often write null for an empty list, a bare string for a one-item list, or a list as a string."""
    if value is None:
        return []
    if isinstance(value, str):
        return json.loads(value) if value.startswith("[") else [value]
    return value


Ids = Annotated[list[SafeId], BeforeValidator(listed)]


class Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------- the task taxonomy


class Needs(Section):
    """What a level needs of the facts a board entry rests on, checked when the plan is written: the channels its facts
    are first stated in; the relations among them; its near-misses (decoys, or values whose change its actor cannot
    see, on its answer's subject and attribute, readable and unretracted in its actor's sight:
    `world.board.fact_measures`); and, for a chapter's brief, `values`, how many values its answer's subject and
    attribute takes (the answer's among them), and `fresh`, an answer first stated in the chapter, so that what it
    plants is new. `hidden`, an answer stated where it is harder to see than its near-misses (a thread reply, a DM, or
    a later day), is checked when the answer is posted. `named`, at most how many of its evidence's channel names and
    identifiers (words with digits, as PWSQL-03) a question names, is shown to the author and checked nowhere. Search
    rank and a channel's page depth are measured and reported (`world.profile`), not needed."""

    channels: int = Field(default=0, ge=0)
    relations: int = Field(default=0, ge=0)
    decoys: int = Field(default=0, ge=0)
    hidden: bool = False
    named: int | None = Field(default=None, ge=0)
    values: int = Field(default=0, ge=0)
    fresh: bool = False

    def facts(self) -> bool:
        """Whether a level needs anything of its facts, so its slots are planned on the board before day 1."""
        return bool(
            self.channels or self.relations or self.decoys or self.hidden or self.values or self.fresh
        )


class Category(Section):
    """One task category: where its gold answer comes from, the answers it may have, what each level means, the
    concrete concepts a task of each level may require (the seed draws one per task), and what code checks a task of
    each level for."""

    gold: Literal["sql", "ledger", "hybrid"]
    answer_types: list[Literal["text", "set", "number", "refusal", "status"]] = Field(min_length=1)
    definition: Text
    levels: list[Text] = Field(min_length=1)
    concepts: list[list[Text]]
    needs: list[Needs] | None = None

    @model_validator(mode="after")
    def per_level(self):
        if len(self.concepts) != len(self.levels) or not all(self.concepts):
            raise ValueError("concepts has a non-empty list for each level")
        if self.needs is not None and len(self.needs) != len(self.levels):
            raise ValueError("needs has a table for each level")
        return self


Categories = dict[Annotated[str, Field(pattern=r"^[a-z_]+$")], Category]


class Taxonomy(Section):
    """The task taxonomy (configs/worldgen_slack/taxonomy.toml): how a question may be asked (a chapter's brief draws
    a style for each entry), and the categories a task is written in, each with its levels."""

    styles: list[Text] = Field(min_length=1)
    categories: Categories = Field(min_length=1)


def load_taxonomy(path: Path) -> Taxonomy:
    return Taxonomy.model_validate(tomllib.loads(rooted(path).read_text()))


# ---------------------------------------------------------------------- seats


def vm_seat(model: str, tokens: int = 600_000) -> vf.AgentConfig:
    """An rlm coding agent in one VM for the whole run (the world's author, the forge's proposer), with one interaction
    per block of work; each interaction gets these turn, token (new tokens: completion and uncached prompt) and time
    budgets."""
    base = role(model, author=True)
    return base.model_copy(
        update={
            "harness": base.harness.model_copy(update={"max_total_tokens": tokens}),
            # Idle between interactions only while the judge or the solver works; a VM left behind ends in 30 min.
            "runtime": base.runtime.model_copy(update={"idle_timeout": 1800}),
            "max_turns": 200,
            "timeout": base.timeout.model_copy(update={"rollout": 5400}),
        }
    )


class Seats(vf.EnvConfig):
    """The seats every run has (a part adds its own), held to code's rules: an agent that runs code works in an
    immutable Prime VM with framework-only egress, and every seat names its model."""

    taskset: vf.TasksetConfig = vf.TasksetConfig(id="worldgen-slack")
    judge: vf.AgentConfig = role("openai/gpt-6-sol")
    retries: RetryConfig = RetryConfig(max_retries=0)
    # The episode's agents at once. GLM, whose account allows 8 concurrent requests, is gated on its own by a forge's
    # `solvers`, so the judge's reviews run beside its solves.
    max_concurrent_agents: int | None = 16

    @model_validator(mode="after")
    def secure(self):
        if self.retries.max_retries:
            raise ValueError("whole-pipeline retries would replay mutable work")
        for name in type(self).model_fields:
            agent = getattr(self, name)
            if not isinstance(agent, vf.AgentConfig):
                continue
            runtime = agent.runtime
            if not isinstance(agent.harness, NullHarnessConfig) and (
                not isinstance(runtime, vf.PrimeConfig)
                or not runtime.vm
                or runtime.allow
                or runtime.block != ["*"]
                or "@sha256:" not in runtime.image
            ):
                raise ValueError(
                    f"{name} executes code, so it requires an immutable Prime VM with framework-only egress"
                )
            if not agent.model:
                raise ValueError(f"{name} needs an explicit model")
        return self
