"""The proposer's tools: the reads it shares with the world author, the world's inventory (profile), and add_task, a
task checked by code and kept for the forge to label. The world is never written: a task is kept as a file in the
run (candidates/<id>.json), which the forge reads when the turn ends."""

import json
from pathlib import Path

import verifiers.v1 as vf
from worldgen_slack.release import atomic_json
from worldgen_slack.toolset import watch_parent

from ...session import AuthoringToolsConfig, WorldReads
from ..config import ForgeConfig
from ..profile import profile
from ..tasks import Task, answer_text, check


class ProposerTools(WorldReads, vf.Toolset[AuthoringToolsConfig, vf.State]):
    """The proposer's tools, for the ids its turn opens."""

    TOOL_PREFIX = "world"
    settings_model = ForgeConfig

    def allowed(self) -> dict:
        """What the turn at hand opens, read per call: its ids, and for a version of a task, which task and why."""
        path = Path(self.context.get("writable_file") or "")
        return json.loads(path.read_text()) if path.is_file() else {"ids": [], "versions": {}}

    def _kept(self) -> Path:
        return Path(self.context["writable_file"]).parent / "candidates"

    def _written(self) -> list[Task]:
        return [Task.model_validate_json(p.read_bytes()) for p in sorted(self._kept().glob("*.json"))]

    @vf.tool
    async def add_task(self, task: Task) -> dict:
        """Write one task under one of this turn's open ids, of the family the id asks for. Code checks its asker, that
        each answer's messages exist and its asker can read them, each answer's shape, that the request gives no answer
        away and repeats no request of its asker, and keeps it, or refuses it with every reason. Writing an id again
        replaces its task."""

        def act():
            allowed = self.allowed()
            if task.id not in allowed.get("ids", []):
                raise ValueError(
                    f"{task.id} is not open this turn; the open ids are {allowed.get('ids', [])}"
                )
            version = allowed.get("versions", {}).get(task.id)
            wanted = (version["variant_of"], version["why"]) if version else (None, None)
            if (task.variant_of, task.why) != wanted:
                raise ValueError(
                    f"{task.id} is a {wanted[1]} version of {wanted[0]}: set variant_of and why so"
                    if version
                    else f"{task.id} is a task of the initial set, a version of none: leave variant_of and why empty"
                )
            family = allowed.get("families", {}).get(task.id)
            if family and task.family != family:
                raise ValueError(f"{task.id} asks for a {family} request: write one of that family")
            world = self._world()
            try:
                check(world, task, self._written())
            finally:
                world.close()
            atomic_json(self._kept() / f"{task.id}.json", task.model_dump(mode="json"))
            return {"task": task.id, "kept": True, "answer": answer_text(task)}

        return self._logged("add_task", {"task": task.model_dump(mode="json")}, act)

    @vf.tool
    async def profile(self) -> dict:
        """The finished world's inventory, by code: its value groups (values, chains, decoys, where each value sits and
        who can read it), threads, recurring conversations, people, its apps, promises, artifacts and groups. Read it
        to find where a request's answers are."""
        return self._logged("profile", {}, lambda: profile(self._world()))


if __name__ == "__main__":
    watch_parent()  # the server process ends with the generator that started it
    ProposerTools.run()
