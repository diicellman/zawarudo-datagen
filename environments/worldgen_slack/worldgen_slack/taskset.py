"""Slack QA tasks over one world file: solve with the actor's Slack tools, grade once with a native LLM judge."""

import json
from pathlib import Path
from typing import Self

import verifiers.v1 as vf
from pydantic import Field

from .config import role
from .dataset import NonEmptyText, PrivateAnswer, PublicTask, StrictModel, load_release
from .tools import CallState, SlackTools, WorldTaskData, WorldToolsConfig, stage_world


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
                    "Grade a Slack answer against the question and its reference. The reference's rows are the "
                    "answer: for a set every row is required and order does not matter; text and number are one "
                    "value; a refusal is right when the response says the request cannot be fulfilled from what the "
                    "asker can see. Accept equivalent wording and supported explanation. Separately decide whether "
                    "the solver's RECORDED observations support every material claim of its response; the reference "
                    "is not an observation, and no observations means ungrounded. Return correct, grounded and a "
                    "concise reason.\n" + json.dumps(fields, ensure_ascii=False)
                )
            )
        ]


class SolverConfig(vf.TaskConfig):
    tools: WorldToolsConfig
    reference: PrivateAnswer = Field(exclude=True, repr=False)
    judge: vf.JudgeConfig = vf.JudgeConfig(model="openai/gpt-6-sol")


class SolverTask(vf.Task[WorldTaskData, CallState, SolverConfig]):
    @classmethod
    def toolsets(cls, config):
        return [SlackTools(config.tools)]

    @classmethod
    def create(
        cls, task: PublicTask, world: Path, reference: PrivateAnswer, judge=None, network_policy=True
    ) -> Self:
        """`world` is a solver copy (no answer key); `network_policy` blocks egress where the runtime enforces it."""
        config = SolverConfig(tools=WorldToolsConfig(actor_id=task.actor_id), reference=reference)
        stage_world(config.tools, world)
        if judge is not None:
            config.judge = judge
        return cls(
            WorldTaskData(
                name=task.task_id,
                task_id=task.task_id,
                world_hash=config.tools.db_hash,
                prompt=task.question,
                **({"network_allow": [], "network_block": ["*"]} if network_policy else {}),
            ),
            config,
        )

    @staticmethod
    def outcome(trace: vf.Trace) -> dict:
        """A solve's graded result. One that crashed before its grade (its harness, its model stream or its grading
        failed) scores zero and is marked crashed: its try tells nothing about the task."""
        result = trace.info.get("evaluation")
        crashed = result is None or not trace.ok
        if result is None:
            failed = "grading failed: " if trace.info.get("grading_started") else ""
            result = {
                "task_id": trace.task.data.task_id,
                "execution_ok": False,
                "semantic_correctness": 0.0,
                "correct": False,
                "grounded": False,
                "calls": len(trace.info.get("observations", [])),
                "solver_trace_id": trace.id,
                "reason": failed + "; ".join(e.message for e in trace.errors),
                "response": trace.last_reply,
            }
        if not trace.ok:
            result.update(execution_ok=False, semantic_correctness=0.0)
        return result | {"crashed": crashed}

    async def finalize(self, trace):
        trace.info["observations"] = [c.model_dump(mode="json") for c in trace.state.calls]

    @vf.reward
    async def semantic_correctness(self, trace: vf.Trace) -> float:
        trace.info["grading_started"] = True
        observations = trace.info.get("observations", [])
        reference = self.config.reference
        result = await AnswerJudge(self.config.judge).evaluate(
            trace=trace,
            question=self.data.prompt,
            reference={"answer_type": reference.answer_type, "rows": reference.rows},
            response=trace.last_reply,
            observations=observations,
        )
        grade = result.parsed
        if grade.grounded and not observations:
            raise ValueError("judge claimed grounding without observations")
        seen = {
            (item["channel"], item["ts"])
            for c in observations
            for item in c["output"].get("items", [])
            if isinstance(item, dict) and "ts" in item
        }
        users = {c["output"].get("id") for c in observations if c["tool"] == "get_user"}
        needed = len(reference.messages) + len(reference.users)
        found = len({tuple(m) for m in reference.messages} & seen) + len(set(reference.users) & users)
        coverage = found / needed if needed else None
        score = float(grade.correct and grade.grounded)
        trace.info["evaluation"] = {
            "task_id": self.data.task_id,
            "execution_ok": True,
            "semantic_correctness": score,
            **grade.model_dump(mode="json"),
            "response": trace.last_reply,
            "evidence_coverage": coverage,
            "calls": len(observations),
            "solver_trace_id": trace.id,
        }
        trace.record_metrics(
            {
                "correct": float(grade.correct),
                "grounded": float(grade.grounded),
                "calls": float(len(observations)),
            }
        )
        return score


class EvaluationConfig(vf.TaskConfig):
    release_dir: Path = Path("data/v6-01/software/release")
    judge: vf.JudgeConfig = vf.JudgeConfig(model="openai/gpt-6-sol")


class EvaluationTask(vf.Task[WorldTaskData, vf.State, EvaluationConfig]):
    def solver_task(self):
        world, rows, answers = load_release(self.config.release_dir)
        row = next(r for r in rows if r.task_id == self.data.task_id)
        if (row.question, row.world_hash) != (self.data.prompt, self.data.world_hash):
            raise ValueError("task differs from the immutable release")
        return SolverTask.create(row, world, answers[row.task_id], self.config.judge)

    async def validate(self, runtime):
        self.solver_task()


class SlackTasksetConfig(vf.TasksetConfig):
    task: EvaluationConfig = EvaluationConfig()


class SlackTaskset(vf.Taskset[EvaluationTask, SlackTasksetConfig]):
    def load(self):
        _, rows, _ = load_release(self.config.task.release_dir)
        return [
            EvaluationTask(
                WorldTaskData(
                    idx=i, name=r.task_id, prompt=r.question, task_id=r.task_id, world_hash=r.world_hash
                ),
                self.config.task,
            )
            for i, r in enumerate(rows)
        ]


class SlackEnvConfig(vf.EnvConfig):
    taskset: SlackTasksetConfig = SlackTasksetConfig(id="worldgen-slack")
    solver: vf.AgentConfig = role("z-ai/glm-5.2", solver=True)


class SlackEnv(vf.Env[SlackEnvConfig]):
    async def run(self, task: EvaluationTask, agents: vf.Agents):
        await agents.solver.run(task.solver_task())
