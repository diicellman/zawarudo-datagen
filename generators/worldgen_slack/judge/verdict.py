"""The judge's verdict: what a review reports (issues, task reviews, criteria scores), its schema for one review,
and how verdicts decide: blocking issues decide; scores are reported."""

from typing import Literal, Self

from pydantic import Field, model_validator
from worldgen_slack.release import NonEmptyText, SafeId, StrictModel

from ..config import Ids

# A task the judge finds a second reading of: one of its wrong answers is right as its question can also be read.
READING = "another reading of its question makes a wrong answer right: "


class Issue(StrictModel):
    """A defect a judge reports, on the ledger, a task or the workspace (messages); the author repairs it."""

    artifact: Literal["ledger", "tasks", "workspace"]
    task_ids: Ids = Field(default_factory=list)
    fact_ids: Ids = Field(default_factory=list)
    message_ids: list[int] = Field(default_factory=list)  # the messages to change
    evidence_message_ids: list[int] = Field(
        default_factory=list
    )  # the messages that show the defect; they stay
    defect: NonEmptyText
    requested_change: NonEmptyText
    blocking: bool = True


class TaskReview(StrictModel):
    """One task's review. `misread` quotes a wrong solve's answer that a reading the question's own words allow makes
    right, with that reading: such a task is not valid, and goes back to be reworded, not kept as hard."""

    task_id: SafeId
    valid: bool
    reason: NonEmptyText
    level_fit: int | None = Field(default=None, ge=0, le=4)
    misread: NonEmptyText | None = None

    @model_validator(mode="after")
    def read_one_way(self) -> Self:
        if self.misread and self.valid:
            raise ValueError(
                "a task whose wrong answer another reading of its question makes right is not valid"
            )
        return self


# A task review scores one task; a world review scores the workspace text.
PHASE_CRITERIA = {
    "task": ("question_fit", "discoverability", "shortcut_free"),
    "world": ("scenario_alignment", "world_coherence", "professional_realism"),
}
QUALITY_CRITERIA = tuple(name for names in PHASE_CRITERIA.values() for name in names)


class Verdict(StrictModel):
    approved: bool
    tasks: list[TaskReview]
    issues: list[Issue]
    criteria: dict[str, float] = Field(default_factory=dict)
    summary: NonEmptyText

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if set(self.criteria) - set(QUALITY_CRITERIA):
            raise ValueError("unknown quality criterion")
        if len({t.task_id for t in self.tasks}) != len(self.tasks):
            raise ValueError("duplicate task review")
        if any(not 0 <= value <= 1 for value in self.criteria.values()):
            raise ValueError("criteria must be finite scores from zero to one")
        if self.approved and (any(i.blocking for i in self.issues) or any(not t.valid for t in self.tasks)):
            raise ValueError("approval contradicts blocking defects")
        if not self.approved and not self.issues:
            raise ValueError("rejection must include actionable issues")
        return self


def verdict_schema(phase: str, task_ids: list[str]) -> dict:
    """The contract `validate_verdict` enforces: exactly one review per requested task, and the phase's criteria."""
    schema = Verdict.model_json_schema()
    schema["properties"]["tasks"] |= {"minItems": len(task_ids), "maxItems": len(task_ids)}
    if task_ids:
        schema["$defs"]["TaskReview"]["properties"]["task_id"]["enum"] = task_ids
    names = PHASE_CRITERIA.get(phase, ())
    schema["properties"]["criteria"] = {
        "type": "object",
        "properties": {name: {"type": "number", "minimum": 0, "maximum": 1} for name in names},
        "additionalProperties": False,
    }
    if phase in PHASE_CRITERIA:
        schema["properties"]["criteria"]["required"] = list(names)
        schema["required"].append("criteria")
    if phase == "task":
        schema["$defs"]["TaskReview"]["required"].append("level_fit")
    return schema


def quality(verdict: Verdict) -> float:
    if not verdict.criteria:
        raise ValueError("a task or world review must score its criteria")
    return sum(verdict.criteria.values()) / len(verdict.criteria)


def validate_verdict(verdict: Verdict, payload: dict) -> None:
    if payload["phase"] in PHASE_CRITERIA and set(verdict.criteria) != set(PHASE_CRITERIA[payload["phase"]]):
        raise ValueError(
            f"a {payload['phase']} review scores exactly {list(PHASE_CRITERIA[payload['phase']])}"
        )
    if {r.task_id for r in verdict.tasks} != {t["id"] for t in payload["tasks"]}:
        raise ValueError("judge must review every requested task exactly once")
    if payload["phase"] == "task" and any(r.level_fit is None for r in verdict.tasks):
        raise ValueError("a task review scores each task's level_fit from 0 to 4")
    if payload["phase"] == "task" and any(i.blocking and not i.task_ids for i in verdict.issues):
        raise ValueError(
            "a blocking issue in a task review names the tasks it blocks (task_ids); a defect that blocks no task "
            "is not blocking"
        )


def accepted(verdict: Verdict) -> bool:
    """Blocking issues decide; the judge's scores are reported, not thresholds."""
    return all(t.valid for t in verdict.tasks) and not any(i.blocking for i in verdict.issues)


def accepted_task(verdict: Verdict, task_id: str) -> bool:
    """One task of a batched review passes on its own review and the issues that name it: an issue naming no task is
    the world's, and blocks none of them."""
    return accepted(
        verdict.model_copy(
            update={
                "tasks": [r for r in verdict.tasks if r.task_id == task_id],
                "issues": [i for i in verdict.issues if task_id in i.task_ids],
            }
        )
    )


def read_one_way(verdict: Verdict) -> Verdict:
    """A second reading the judge found is code's issue too: it blocks its task, and sends it back to be reworded."""
    return verdict.model_copy(
        update={
            "issues": verdict.issues
            + [
                Issue(
                    artifact="tasks",
                    task_ids=[r.task_id],
                    defect=READING + r.misread,
                    requested_change="word the question so it rules that reading out, for the same asker and gold answer",
                    blocking=True,
                )
                for r in verdict.tasks
                if r.misread
            ]
        }
    )


def combined(verdicts: list[Verdict]) -> Verdict:
    """Several reviews as one verdict: approved when each is, with every task review and issue; a rejection's feedback
    carries only the rejecting reviews' words."""
    rejected = [v for v in verdicts if not v.approved]
    return Verdict.model_construct(
        approved=not rejected,
        tasks=[r for v in verdicts for r in v.tasks],
        issues=[i for v in verdicts for i in v.issues],
        criteria={},
        summary=" ".join(v.summary for v in rejected or verdicts),
    )
