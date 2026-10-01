"""Independent agentic catalog, task and world review in a fresh runtime."""

import json
from typing import Any, Self

import verifiers.v1 as vf
from pydantic import Field

from worldgen_slack.slack.api import digest
from ..contracts import (
    Candidate,
    Catalog,
    PHASE_CRITERIA,
    Verdict,
    quality,
    timed,
    to_local,
    validate_verdict,
    verdict_schema,
)
from worldgen_slack.slack.tools import ReadState, SlackTaskData, stage_tool_data
from .inspection import ReviewTools, ReviewToolsConfig, missing_evidence

REVIEW_PROMPT = """Review the artifacts in /task/input.json using /task/schemas.json and /task/review.md.
input.json can be large: inspect it selectively with your coding harness. Run inspect_check and investigate defects.
When input.json has tasks and a candidate, use inspect_read to inspect all bound evidence as each task's reader.
Before approving, call inspect_check again and resolve every entry in uninspected_evidence.
Write your structured verdict to /task/verdict.json. The reviewed_hash is in /task/review.md.
Private facts, gold routes, author statements, solver answers, and earlier verdicts are context, not proof.
Other JSON files in /task are observable traces for investigation: author_trace.json is the author's, and
solver_<task_id>_<n>.json are independent solvers' attempts at that task.
inspect_check and inspect_read are brokered network calls; if one fails as unavailable, call it again.
"""

COMMON_GUIDE = """Approve only after investigating the actual artifacts. Every requested task needs one review.
Timed facts (fact_local_times) are first stated in the minute of their valid_from by construction; other facts'
valid_from only orders values. A timed moment that is implausible in itself is a catalog defect.
An issue is blocking when it changes, contradicts, hides or shortcuts evidence a task's answer relies on, or is a
catalog or bindings defect; report other realism and chronology flaws with blocking false.
All blocking issues need artifact and its responsible owner: catalog (questions, answers, shared facts)
belongs to synthesizer; workspace (people/channels/messages) and bindings (evidence/gold routes) belong
to builder. A route using undiscovered IDs is a bindings defect, never a catalog defect.
Include affected IDs, defect, and requested_change. No approval with blocking issues.
When input.json has later_groups, those workstreams are built after this review: their missing tasks, facts and
messages are not defects. When input.json has previous_issues, verify each is resolved. When it has changed_messages, review those messages
with their surroundings; in unchanged content already reviewed, report only blocking issues.
Do not rewrite inputs. Final prose is not a substitute for the verdict file.
"""

PHASE_GUIDES = {
    "catalog": """Catalog: check full-answer completeness, distinct questions, coherent company/people/projects, the
selected premise, distinct plausible personas (a persona's profile fits its role), shared facts and dates (fact_local_times shows timed facts on the
company's local clocks), scope/authority, requested output slots and ordering, and plausible read-only work.
Output restrictions must appear in the public question; private answer metadata cannot impose them.
Detect semantic duplicates even when people or dates differ. shared fact intervals are [from, until).
""",
    "task": """Task: every canonical answer claim must follow from actor-visible messages/directory data.
Facts in the private catalog alone cannot support a public answer. Read all bound evidence with inspect_read.
Examine surrounding threads and competing records.
Test the supplied gold route: its queries/IDs must come from the public question or earlier outputs.
inspect_check verifies identifier discovery, replay, and that queries use no undiscovered answer words; it does not
judge other semantic provenance of search terms.
A query is not valid merely because SOME terms are public: investigate added restrictive terms.
Unobserved exact cohort names, image labels, identifiers, or answer values are private clues even when
mixed with question-derived words. Ordinary paraphrases of the question do not require literal overlap.
Check stale decisions, unsupported approvals/completions, missing list members, conflicting time scope,
answer-bearing metadata, and accidental shortcuts. Evidence can be shared across tasks. A legitimate lookup may
take one read. The workspace is shared by many tasks: content supporting other tasks is allowed. A shortcut defect
needs a concrete artificial answer cache, question leakage, or an invalid bypass of the question's intended
evidence; ordinary supporting facts for another task are not a shortcut.
solves lists independent attempts at each task, each graded for correctness and grounding; their traces are in
solver_<task_id>_<n>.json. Solvers that answer differently or cannot find the evidence can reveal ambiguity, missing time
scope, or undiscoverable evidence. A mistake the evidence clearly rules out is the solver's, not a defect.
Give every criterion in schemas.json a score in [0, 1].
""",
    "world": """World: review the workspace text as a record of real work. The catalog lists the tasks built so far;
separate task reviews verify each task's evidence and bindings. Natural surrounding work matters; volume and random
chatter are not quality. Check chronology across conversations: commitments are kept or explicitly
changed, and nobody reports work as done, or an outcome as known, before it happens.
Inspect the activity statistics returned by inspect_check. Distinct text alone does not establish realism:
repeated two-message exchanges, fixed reply delays, and mechanical posting schedules need investigation.
activity.style reports voice and timing overall and per author: compare each author with their catalog
persona. Different personas writing in one uniform register is a realism defect. Deleting activity or
perturbing timestamps does not repair realism; request that the affected scenes be rewritten.
Give every criterion in schemas.json a score in [0, 1].
""",
}


class JudgeConfig(vf.TaskConfig):
    tools: ReviewToolsConfig
    files: dict[str, str] = Field(default_factory=dict, exclude=True, repr=False)


class JudgeTask(vf.Task[SlackTaskData, ReadState, JudgeConfig]):
    NEEDS_CONTAINER = True

    @classmethod
    def toolsets(cls, config: JudgeConfig) -> list[vf.Toolset]:
        return [ReviewTools(config.tools)]

    @classmethod
    def create(cls, payload: dict, *, group_id: str = "", candidate_id: str = "") -> Self:
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
            JudgeConfig(tools=ReviewToolsConfig(payload_json=json.dumps(payload))),
        )

    async def setup(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        for name, text in self.config.files.items():
            await runtime.write("/task/" + name, text.encode())
        payload = json.loads(self.config.tools.payload_json)
        stage_tool_data(self, self.config.tools, payload)
        await runtime.write("/task/input.json", self.config.tools.payload_json.encode())
        await runtime.write(
            "/task/schemas.json",
            json.dumps(verdict_schema(payload["phase"], [t["id"] for t in payload["tasks"]])).encode(),
        )
        await runtime.write(
            "/task/review.md",
            (COMMON_GUIDE + PHASE_GUIDES[payload["phase"]] + "\nreviewed_hash: " + digest(payload)).encode(),
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
        if payload["phase"] in PHASE_CRITERIA:
            trace.record_metric(payload["phase"] + "_quality", quality(verdict))


def review_payload(
    catalog: Catalog, candidate: Candidate | None, task_ids: list[str], phase: str, **extra
) -> dict:
    zones, minutes = sorted({p.timezone for p in catalog.personas}), timed(catalog)
    payload: dict[str, Any] = {
        "phase": phase,
        "catalog": catalog.model_dump(mode="json"),
        # Derived view: when each timed fact is stated, on the wall clocks of the catalog's people.
        "fact_local_times": {
            f: {z: to_local(m, z, "%A %Y-%m-%d %H:%M") for z in zones} for f, m in minutes.items()
        },
        "tasks": [t.model_dump(mode="json") for t in catalog.tasks if t.id in task_ids],
        **extra,
    }
    if candidate is not None:
        payload["candidate"] = candidate.model_dump(mode="json")
    return payload
