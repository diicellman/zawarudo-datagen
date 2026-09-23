"""Independent agentic catalog/world review in a fresh runtime."""

import json
from typing import Any, Self

import verifiers.v1 as vf
from pydantic import Field

from worldgen_slack.slack.api import digest
from ..contracts import Candidate, Catalog, Verdict, quality, validate_verdict, verdict_schema
from worldgen_slack.slack.tools import ReadState, SlackTaskData, stage_tool_data
from .inspection import ReviewTools, ReviewToolsConfig, missing_evidence

REVIEW_PROMPT = """Review the artifacts in /task/input.json using /task/schemas.json and /task/review.md.
Inspect the files selectively with your coding harness, run inspect_check, and investigate defects.
For a built workspace, use inspect_read to empirically inspect all bound evidence as each task's reader.
Before approving, call inspect_check again and resolve every entry in uninspected_evidence.
Write your structured verdict to /task/verdict.json. The reviewed_hash is in /task/review.md.
Private facts, gold routes, author statements, and earlier verdicts are context, not proof.
When present, /task/author_trace.json contains the author's observable trace for investigation.
"""

REVIEW_GUIDE = """Approve only after investigating the actual artifacts. Every requested task needs one review.
Catalog: check full-answer completeness, distinct questions, coherent company/people/projects,
shared facts and dates, scope/authority, requested output slots and ordering, and plausible read-only work.
Output restrictions must appear in the public question; private answer metadata cannot impose them.
Detect semantic duplicates even when people or dates differ. shared fact intervals are [from, until).
World: every canonical answer claim must follow from actor-visible messages/directory data.
Read all bound evidence with inspect_read. Examine surrounding threads and competing records.
Test the supplied gold route: its queries/IDs must come from the public question or earlier outputs.
inspect_check verifies identifier discovery and replay, not the semantic provenance of search terms.
A query is not valid merely because SOME terms are public: investigate added restrictive terms.
Unobserved exact cohort names, image labels, identifiers, or answer values are private clues even when
mixed with question-derived words. Ordinary paraphrases of the question do not require literal overlap.
Check stale decisions, unsupported approvals/completions, missing list members, conflicting time scope,
answer-bearing metadata, and accidental shortcuts introduced by OTHER task groups.
Evidence can be shared across tasks. A legitimate lookup may take one read.
All task groups inhabit ONE shared workspace. Content and bindings supporting sibling or future groups
are allowed, including evidence written before that group's build step. Never reject their mere presence
as leakage or demand group-isolated worlds. A shortcut defect needs a concrete artificial answer cache,
question leakage, or an invalid bypass of the question's intended evidence; ordinary supporting facts
for another task are not a shortcut. The requested task list scopes required checks, not allowed content.
Natural surrounding work matters; volume, extra tool calls, and random chatter are not quality. Facts in the private catalog
alone cannot support a public answer. Inspect the full world beyond the bound evidence where relevant.
Inspect the activity statistics returned by inspect_check. Distinct text alone does not establish realism:
repeated two-message exchanges, fixed reply delays, and mechanical posting schedules need investigation.
Give every world criterion a score in [0, 1]: scenario_alignment, world_coherence,
professional_realism, discoverability, shortcut_free, evidence_composition.
Approval requires each >= .75 and weighted quality >= .8 (weights 1, 1.25, 1.25, 1, 1, 1.5).
All blocking issues need artifact and its responsible owner: catalog (questions, answers, shared facts)
belongs to synthesizer; workspace (people/channels/messages) and bindings (evidence/gold routes) belong
to builder. A route using undiscovered IDs is a bindings defect, never a catalog defect.
Include
affected IDs, defect, and requested_change. No approval with unresolved issues.
Do not rewrite inputs. Final prose is not a substitute for the verdict file.
"""


class JudgeConfig(vf.TaskConfig):
    tools: ReviewToolsConfig
    author_trace: str = Field(default="", exclude=True, repr=False)
    review_guide: str | None = Field(default=None, exclude=True, repr=False)


class JudgeTask(vf.Task[SlackTaskData, ReadState, JudgeConfig]):
    NEEDS_CONTAINER = True

    @classmethod
    def toolsets(cls, config: JudgeConfig) -> list[vf.Toolset]:
        return [ReviewTools(config.tools)]

    @classmethod
    def create(
        cls, payload: dict, *, group_id: str = "", candidate_id: str = "", review_guide: str | None = None
    ) -> Self:
        groups = {t["group_id"] for t in payload["tasks"]}
        return cls(
            SlackTaskData(
                prompt=REVIEW_PROMPT,
                workspace_id=payload["catalog"]["workspace_id"],
                group_id=group_id or (next(iter(groups)) if len(groups) == 1 else "all"),
                task_id=payload["tasks"][0]["id"] if len(payload["tasks"]) == 1 else "",
                candidate_id=candidate_id,
                snapshot_hash=digest(payload.get("candidate", {}).get("snapshot", {})),
                network_allow=[],
                network_block=["*"],
            ),
            JudgeConfig(tools=ReviewToolsConfig(payload_json=json.dumps(payload)), review_guide=review_guide),
        )

    async def setup(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        if self.config.author_trace:
            await runtime.write("/task/author_trace.json", self.config.author_trace.encode())
        payload = json.loads(self.config.tools.payload_json)
        stage_tool_data(self, self.config.tools, payload)
        await runtime.write("/task/input.json", self.config.tools.payload_json.encode())
        await runtime.write("/task/schemas.json", json.dumps(verdict_schema(payload["phase"])).encode())
        await runtime.write(
            "/task/review.md",
            (
                (self.config.review_guide if self.config.review_guide is not None else REVIEW_GUIDE)
                + "\nreviewed_hash: "
                + digest(payload)
            ).encode(),
        )

    async def finalize(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        self._tool_directory.cleanup()
        payload = json.loads(self.config.tools.payload_json)
        raw = await runtime.read("/task/verdict.json", max_bytes=2_000_000)
        trace.info["verdict_candidate"] = raw.decode()
        trace.info["observations"] = [r.model_dump(mode="json") for r in trace.state.reads]
        verdict = Verdict.model_validate_json(raw)
        validate_verdict(verdict, payload)
        if not trace.state.checked:
            raise ValueError("reviewer did not run trusted checks")
        if verdict.approved and (missing := missing_evidence(payload, trace.state.reads)):
            raise ValueError("reviewer did not inspect all evidence: " + json.dumps(missing))
        trace.info["verdict"] = verdict.model_dump(mode="json")
        trace.record_metric("approved", float(verdict.approved))
        if payload["phase"] == "world":
            trace.record_reward("world_quality", quality(verdict))
        elif payload["phase"] == "catalog":
            trace.record_reward("catalog_quality", float(verdict.approved))


def review_payload(
    catalog: Catalog, candidate: Candidate | None, task_ids: list[str], phase: str, **extra
) -> dict:
    payload: dict[str, Any] = {
        "phase": phase,
        "catalog": catalog.model_dump(mode="json"),
        "tasks": [t.model_dump(mode="json") for t in catalog.tasks if t.id in task_ids],
        **extra,
    }
    if candidate is not None:
        payload["candidate"] = candidate.model_dump(mode="json")
    return payload
