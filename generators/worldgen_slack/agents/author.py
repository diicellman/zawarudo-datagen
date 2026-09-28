import json
from typing import ClassVar, Self
import verifiers.v1 as vf
from pydantic import BaseModel, Field
from worldgen_slack.slack.tools import SlackTaskData

FILE_GUIDE = """Create structured synthetic Slack data. Work with files through the coding harness.
There is one shared company/workspace and multiple related questions. No generated source is released.
/task/input.json names the current "phase"; /task/schemas.json maps each phase to its output schema.
Write the current phase's document to /task/<phase>.json, e.g. /task/plan.json. You can inspect existing files
selectively and write the document incrementally. Do not paste a large JSON document into the final reply.
Never change trusted schemas or input files.

"""


class AuthorConfig(vf.TaskConfig):
    files: dict[str, str] = Field(default_factory=dict, exclude=True, repr=False)


class AuthorTask(vf.Task[SlackTaskData, vf.State, AuthorConfig]):
    NEEDS_CONTAINER = True
    outputs: ClassVar[dict[str, type[BaseModel]]]
    instructions: ClassVar[str]

    @classmethod
    def create(cls, context: dict, attempt: str) -> Self:
        files = {
            "input.json": json.dumps(context, ensure_ascii=False, indent=2),
            "schemas.json": json.dumps(
                {phase: model.model_json_schema() for phase, model in cls.outputs.items()}
            ),
            "guide.md": FILE_GUIDE + cls.instructions,
        }
        if context.get("previous_output") is not None:
            files[context["phase"] + ".json"] = json.dumps(
                context["previous_output"], ensure_ascii=False, indent=2
            )
        return cls(
            SlackTaskData(
                prompt="Read /task/guide.md and /task/input.json. Write the current phase's document to /task/<phase>.json.",
                workspace_id=context["workspace_id"],
                group_id=context.get("group_id", ""),
                candidate_id=attempt,
                network_allow=[],
                network_block=["*"],
            ),
            AuthorConfig(files=files),
        )

    async def setup(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        for name, text in self.config.files.items():
            await runtime.write("/task/" + name, text.encode())
