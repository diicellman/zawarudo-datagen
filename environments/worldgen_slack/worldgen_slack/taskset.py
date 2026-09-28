"""Load Slack QA tasks; solve with scoped tools and grade once with a native LLM judge."""

import json
from pathlib import Path
from typing import Self
import verifiers.v1 as vf
from pydantic import Field
from .config import role
from .dataset import PrivateAnswer, load_release
from .slack.api import digest
from .slack.models import SlackWorld, StrictModel, NonEmptyText
from .slack.tools import ReadState, SlackTaskData, SlackToolset, SlackToolsetConfig, stage_tool_data


class AnswerGrade(StrictModel):
    correct: bool
    grounded: bool
    reason: NonEmptyText


class AnswerJudge(vf.Judge[AnswerGrade]):
    schema = AnswerGrade

    def build_messages(self, **fields):
        return [
            vf.UserMessage(
                content=(
                    "Grade the Slack answer against the question and approved reference specification. "
                    "Check every requested claim and explicit output constraint; accept equivalent wording. "
                    "Output constraints must be stated in the PUBLIC question. Private answer-kind or "
                    "list-order metadata cannot impose hidden formatting requirements. In particular, "
                    "exact_string identifies an exact value, not a demand to output only that value. "
                    "Accept supported explanation unless the question explicitly prohibits it. "
                    "Separately decide whether the solver's RECORDED observations support every material claim "
                    "in its response. A reference answer is not an observation. No observations means ungrounded. "
                    "A factually correct answer can be ungrounded. Do not judge workspace style or sibling tasks. "
                    "Return correct, grounded, and a concise reason citing missing or unsupported claims.\n"
                    + json.dumps(fields, ensure_ascii=False)
                )
            )
        ]


class SolverConfig(vf.TaskConfig):
    tools: SlackToolsetConfig
    reference: PrivateAnswer = Field(exclude=True, repr=False)
    judge: vf.JudgeConfig = vf.JudgeConfig(model="openai/gpt-6-sol")


class SolverTask(vf.Task[SlackTaskData, ReadState, SolverConfig]):
    @classmethod
    def toolsets(cls, config):
        return [SlackToolset(config.tools)]

    @classmethod
    def create(cls, task, world: SlackWorld, workspace_id: str, reference: PrivateAnswer, judge=None) -> Self:
        config = SolverConfig(
            tools=SlackToolsetConfig(snapshot_json=world.model_dump_json(), actor_id=task.actor_id),
            reference=reference,
        )
        if judge is not None:
            config.judge = judge
        return cls(
            SlackTaskData(
                name=task.id if hasattr(task, "id") else task.task_id,
                task_id=task.id if hasattr(task, "id") else task.task_id,
                workspace_id=workspace_id,
                snapshot_hash=digest(world.model_dump(mode="json")),
                prompt=task.question,
                network_allow=[],
                network_block=["*"],
            ),
            config,
        )

    async def setup(self, trace, runtime):
        stage_tool_data(self, self.config.tools, json.loads(self.config.tools.snapshot_json))

    async def finalize(self, trace):
        self._tool_directory.cleanup()
        trace.info["observations"] = [r.model_dump(mode="json") for r in trace.state.reads]

    @vf.reward
    async def semantic_correctness(self, trace: vf.Trace) -> float:
        trace.info["grading_started"] = True
        observations = trace.info.get("observations", [])
        result = await AnswerJudge(self.config.judge).evaluate(
            trace=trace,
            question=self.data.prompt,
            answer=self.config.reference.answer.model_dump(mode="json"),
            response=trace.last_reply,
            observations=observations,
        )
        grade = result.parsed
        if grade.grounded and not observations:
            raise ValueError("judge claimed grounding without observations")
        seen_messages = {
            item["message_id"]
            for r in observations
            for item in r["output"].get("items", [])
            if "message_id" in item
        }
        seen_users = {r["output"]["user_id"] for r in observations if "user_id" in r["output"]}
        reference = self.config.reference
        coverage = (
            len(set(reference.message_ids) & seen_messages) + len(set(reference.user_ids) & seen_users)
        ) / (len(reference.message_ids) + len(reference.user_ids))
        score = float(grade.correct and grade.grounded)
        trace.info["evaluation"] = {
            "task_id": self.data.task_id,
            "snapshot_hash": self.data.snapshot_hash,
            "execution_ok": True,
            "semantic_correctness": score,
            **grade.model_dump(mode="json"),
            "response": trace.last_reply,
            "reference_evidence_coverage": coverage,
            "read_count": len(observations),
            "solver_trace_id": trace.id,
        }
        trace.record_metrics(
            {
                "correct": float(grade.correct),
                "grounded": float(grade.grounded),
                "read_count": float(len(observations)),
                "reference_evidence_coverage": coverage,
            }
        )
        return score


class EvaluationConfig(vf.TaskConfig):
    release_dir: Path = Path("data/milestone/release")
    judge: vf.JudgeConfig = vf.JudgeConfig(model="openai/gpt-6-sol")


class EvaluationTask(vf.Task[SlackTaskData, vf.State, EvaluationConfig]):
    def solver_task(self):
        world, rows, answers = load_release(self.config.release_dir)
        row = next(r for r in rows if r.task_id == self.data.task_id)
        if (row.question, row.snapshot_hash, row.workspace_id) != (
            self.data.prompt,
            self.data.snapshot_hash,
            self.data.workspace_id,
        ):
            raise ValueError("task differs from the immutable release")
        return SolverTask.create(row, world, row.workspace_id, answers[row.task_id], self.config.judge)

    async def validate(self, runtime):
        self.solver_task()


class SlackTasksetConfig(vf.TasksetConfig):
    task: EvaluationConfig = EvaluationConfig()


class SlackTaskset(vf.Taskset[EvaluationTask, SlackTasksetConfig]):
    def load(self):
        _, rows, _ = load_release(self.config.task.release_dir)
        return [
            EvaluationTask(
                SlackTaskData(
                    idx=i,
                    name=row.task_id,
                    prompt=row.question,
                    task_id=row.task_id,
                    workspace_id=row.workspace_id,
                    snapshot_hash=row.snapshot_hash,
                ),
                self.config.task,
            )
            for i, row in enumerate(rows)
        ]


class SlackEnvConfig(vf.EnvConfig):
    taskset: SlackTasksetConfig = SlackTasksetConfig(id="worldgen-slack")
    solver: vf.AgentConfig = role("z-ai/glm-5.2", solver=True)


class SlackEnv(vf.Env[SlackEnvConfig]):
    async def run(self, task: EvaluationTask, agents: vf.Agents):
        await agents.solver.run(task.solver_task())
