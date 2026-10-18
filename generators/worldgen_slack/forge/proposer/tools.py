"""The proposer's tools: the reads it shares with the world author, the world's task material, and add_task, a
task checked by code and kept for the forge to label. The world is never written: a task is kept as a file in the
run (candidates/<id>.json), which the forge reads when the turn ends."""

import json
from pathlib import Path

import verifiers.v1 as vf
from worldgen_slack.release import atomic_json
from worldgen_slack.toolset import watch_parent

from ...session import AuthoringToolsConfig, WorldReads
from ..config import ForgeConfig
from ..material import material
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
        """Write one task under one of this turn's open ids, of the family the id asks for, aiming at the skill it names.
        Code checks its asker, that each answer's messages exist and its asker can read them, each answer's shape, that
        the request gives no answer away and repeats no request of its asker, and the shape of its skill (aggregation:
        a set part of 3 or more answers from 2 or more places; abstention: a status or refusal part; synthesis: 2 or
        more parts), and keeps it, or refuses it with every reason. Writing an id again replaces its task."""

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
                check(world, task, self._written(), allowed.get("skills", {}).get(task.id))
            finally:
                world.close()
            atomic_json(self._kept() / f"{task.id}.json", task.model_dump(mode="json"))
            return {"task": task.id, "kept": True, "answer": answer_text(task)}

        return self._logged("add_task", {"task": task.model_dump(mode="json")}, act)

    @vf.tool
    async def material(self) -> dict:
        """The world's task material, by code: a summary (the material counted, by storyline, and how the promises
        went); each value group (a subject and attribute) with every value it took in order, each value's day, place,
        read_channel page and messages (m123, the evidence an answer names), and where so that it is a decoy, what it
        supersedes and that it is current; the collections, sets of items each stated where it came up; every promise,
        its owner, due day and how it went (kept, kept late, changed, dropped, slipped); who can read each private
        channel (a public one, everyone; a DM, the people it names); and the people who could ask (name, id, title,
        clock). A dict: keep it in a variable and look at the part you need."""
        return self._logged("material", {}, lambda: material(self._world()))


if __name__ == "__main__":
    watch_parent()  # the server process ends with the generator that started it
    ProposerTools.run()
