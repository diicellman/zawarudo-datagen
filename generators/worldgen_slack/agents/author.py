"""The file protocol of the authoring agents: inputs, the world and the phase's contract in, one document out."""

import json
from pathlib import Path
from typing import ClassVar, Self

import verifiers.v1 as vf
from pydantic import BaseModel, Field
from worldgen_slack.db import SCHEMA
from worldgen_slack.tools import WorldTaskData, file_hash

FILE_GUIDE = """/task/input.json holds this phase's inputs; when it has feedback, that is what code or a reviewer
rejected in your last document. /task/world.sqlite is a read-only SQLite copy of the workspace built so far, and
/task/schema.sql is its schema. /task/schemas.json is the JSON schema of the document to write. Write the document
to /task/<phase>.json, where <phase> is input.json's phase.

"""


class AuthorConfig(vf.TaskConfig):
    files: dict[str, str] = Field(default_factory=dict, exclude=True, repr=False)
    world: str = Field(default="", exclude=True, repr=False)


class AuthorTask(vf.Task[WorldTaskData, vf.State, AuthorConfig]):
    NEEDS_CONTAINER = True
    outputs: ClassVar[dict[str, type[BaseModel]]]
    guides: ClassVar[dict[str, str]]

    @classmethod
    def create(cls, context: dict, attempt: str, world: Path) -> Self:
        return cls(
            WorldTaskData(
                prompt="Read /task/guide.md and /task/input.json. Write the current phase's document to /task/<phase>.json.",
                attempt=attempt,
                storyline=context.get("storyline", ""),
                world_hash=file_hash(world),
                network_allow=[],
                network_block=["*"],
            ),
            AuthorConfig(files=cls.files(context), world=str(world)),
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
