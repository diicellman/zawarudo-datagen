import json
from typing import Literal
import verifiers.v1 as vf
from pydantic import Field, JsonValue
from worldgen_slack.slack.api import ActionName, SlackAPI, ReadCall
from worldgen_slack.slack.models import SlackWorld
from worldgen_slack.slack.tools import RecordedTools, ReadRecord, read_tool_data
from ..contracts import Candidate, Catalog, validate_candidate


def missing_evidence(payload, reads):
    if "candidate" not in payload:
        return {}
    candidate = Candidate.model_validate(payload["candidate"])
    tasks = {t["id"]: t for t in payload["tasks"]}
    missing = {}
    for binding in candidate.bindings:
        if binding.task_id not in tasks:
            continue
        actor = tasks[binding.task_id]["actor_id"]
        observations = [r.output for r in reads if r.actor_id == actor]
        seen_messages = {
            item["message_id"]
            for output in observations
            for item in output.get("items", [])
            if "message_id" in item
        }
        seen_users = {output["user_id"] for output in observations if "user_id" in output}
        messages = {m for c in binding.claims for m in c.message_ids} - seen_messages
        users = {u for c in binding.claims for u in c.user_ids} - seen_users
        if messages or users:
            missing[binding.task_id] = {
                "actor_id": actor,
                "message_ids": sorted(messages),
                "user_ids": sorted(users),
            }
    return missing


class ReviewToolsConfig(vf.ToolsetConfig):
    kind: Literal["review"] = "review"
    payload_json: str = Field(default="", exclude=True, repr=False)
    data_path: str = ""
    data_hash: str = ""


class ReviewTools(RecordedTools[ReviewToolsConfig]):
    TOOL_PREFIX = "inspect"

    async def setup(self) -> None:
        await super().setup()
        self.payload = read_tool_data(self.config)

    @vf.tool
    async def check(self) -> dict:
        """Check schema, references, and evidence replay; report evidence not yet inspected by its reader."""
        catalog = Catalog.model_validate_json(json.dumps(self.payload["catalog"]))
        if "candidate" in self.payload:
            candidate = Candidate.model_validate_json(json.dumps(self.payload["candidate"]))
            result = validate_candidate(catalog, candidate, [t["id"] for t in self.payload["tasks"]])
            result.pop("gold_outputs")
        else:
            result = {"ok": True, "task_count": len(catalog.tasks), "fact_count": len(catalog.facts)}
        result["uninspected_evidence"] = missing_evidence(self.payload, self.state.reads)
        self.state.checked = True
        return result

    @vf.tool
    async def read(self, actor_id: str, action: ActionName, arguments: dict[str, JsonValue]) -> dict:
        """Execute a slack.readonly.v2 action as a task's reader; returns actor-visible observations."""
        if actor_id not in {t["actor_id"] for t in self.payload["tasks"]}:
            raise ValueError("actor is outside this review's task scope")
        if len(self.state.reads) >= 4096:
            raise ValueError("review read history exhausted")
        world = SlackWorld.model_validate_json(json.dumps(self.payload["candidate"]["snapshot"]))
        api = SlackAPI(world, actor_id)
        call = ReadCall(action=action, arguments=arguments)
        output = api.execute(call)
        self.state.reads.append(ReadRecord(actor_id=actor_id, call=call, output=output))
        return output


if __name__ == "__main__":
    ReviewTools.run()
