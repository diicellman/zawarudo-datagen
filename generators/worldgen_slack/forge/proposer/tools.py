"""The proposer's tools: the reads it shares with the world author, and its own on the frozen world: a candidate
checked and written (add_task) or tried on a copy that writes nothing (probe), the world's inventory (profile), and a
fact the world already states registered in the ledger (annotate). The workspace is never written."""

import json
from pathlib import Path

import verifiers.v1 as vf
from worldgen_slack.toolset import watch_parent

from ...session import AuthoringToolsConfig, WorldReads
from ..config import ForgeConfig
from ..difficulty import failure_cases
from ..profile import profile
from ..tasks import Annotation, Task, annotate, forge_task


class ProposerTools(WorldReads, vf.Toolset[AuthoringToolsConfig, vf.State]):
    """The proposer's tools. It works in one turn, the forge's, writing the ids its round allows."""

    TOOL_PREFIX = "world"
    settings_model = ForgeConfig

    def allowed(self) -> dict:
        """What the env allows the proposer's turn at hand, read per call: the ids it may write, and the tasks sent
        back for another reading."""
        path = Path(self.context.get("writable_file") or "")
        return json.loads(path.read_text()) if path.is_file() else {"ids": []}

    @vf.tool
    async def add_task(self, task: Task) -> dict:
        """Write a candidate for one of the round's open ids, of the category and level you choose: checked, then its
        gold rows and code's measures returned."""

        def act(copy):
            return forge_task(
                copy, task, self.settings, self.allowed().get("ids", []), rework=self.allowed().get("rework")
            )

        return self._logged("add_task", {"task": task.model_dump(mode="json")}, lambda: self._write(act))

    @vf.tool
    async def profile(self) -> dict:
        """The finished world's inventory, by code: its value groups (values, chains, decoys, where each value sits and
        who can read it), threads, recurring conversations, people, its apps, promises, artifacts and groups, and what
        kept tasks and the board rest on. Read it to find where a hard, fair task can be."""
        return self._logged("profile", {}, lambda: profile(self._world()))

    @vf.tool
    async def probe(self, task: Task) -> dict:
        """A candidate tried on a copy and written nowhere: its gold rows, code's measures, the lazy panel, its features
        and the wrong answers the screen will grade. Shape a task with it, then write it with world_add_task."""

        def act():
            world, out = self._world(), {}
            try:
                with world.trial() as copy:
                    result = forge_task(
                        copy,
                        task,
                        self.settings,
                        self.allowed().get("ids", []),
                        rework=self.allowed().get("rework"),
                    )
                    _, cases = failure_cases(copy, task.id, result["gold"])
                    out = result | {"failure_cases": [name for name, _ in cases]}
                    raise LookupError("a probe writes nothing")
            except LookupError as dropped:
                if out:
                    return out
                raise ValueError(str(dropped)) from None

        return self._logged("probe", {"task": task.model_dump(mode="json")}, act)

    @vf.tool
    async def annotate(self, fact: Annotation) -> dict:
        """Register a fact the world already states, on the messages that state it (each contains its anchor; the
        first is its first statement): the ledger grows, the workspace stays as it is. supersedes names a fact it
        replaces, stated earlier."""
        return self._logged(
            "annotate",
            {"fact": fact.model_dump(mode="json")},
            lambda: self._write(lambda copy: annotate(copy, fact)),
        )


if __name__ == "__main__":
    watch_parent()  # the server process ends with the generator that started it
    ProposerTools.run()
