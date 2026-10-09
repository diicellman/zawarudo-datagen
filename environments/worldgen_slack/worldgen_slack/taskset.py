"""Slack QA tasks over one world file: solve with the actor's Slack tools, grade once with a native LLM judge."""

import json
from pathlib import Path
from typing import Self

import verifiers.v1 as vf
from pydantic import Field

from .config import role
from .dataset import NonEmptyText, PrivateAnswer, PublicTask, StrictModel, load_release
from .tools import CallState, SlackTools, WorldTaskData, WorldToolsConfig, stage_world


def shown(call: dict) -> set[tuple[str, str]]:
    """The messages one recorded call showed the solver, as (channel, ts)."""
    return {
        (item["channel"], item["ts"])
        for item in call["output"].get("items", [])
        if isinstance(item, dict) and "ts" in item
    }


class AnswerGrade(StrictModel):
    """The committed answer against the reference (correct), whether the response declined to give one (abstained),
    whether the observations support the claims that make its answer (grounded), and how many of the response's other
    material claims they don't (counted, not scored)."""

    correct: bool
    abstained: bool
    grounded: bool
    unsupported_asides: int = Field(ge=0)
    reason: NonEmptyText


class AnswerJudge(vf.Judge[AnswerGrade]):
    schema = AnswerGrade

    def build_messages(self, **fields):
        return [
            vf.UserMessage(
                content=(
                    "Grade a Slack answer against the question and its reference. First find the answer the response "
                    "commits to: one that lists alternatives without settling on one is wrong, while side options it "
                    "labels as such do not count against an answer it commits to. A response that gives no answer and "
                    "says the request cannot be answered or determined has abstained, whether or not that is right. "
                    "The reference's rows are the answer: for a set every row is required and order does not matter; "
                    "text and number are one value; a time asked for on the asker's clock is given on that clock; a "
                    "refusal is right when the response says the request cannot be fulfilled as asked from what the "
                    "asker can see: what it asks about does not exist, is out of their sight, or rests on a premise "
                    "the workspace contradicts. A status reference is the latest value the asker can see, which is "
                    "not settled: the response is right when it gives that value and says it is not final or "
                    "confirmed; giving it as final, or another value, is wrong; saying only that it cannot be "
                    "determined abstains. Accept equivalent wording. Then decide grounded on the answer's own claims: "
                    "the solver's RECORDED observations support the claims that make its answer (the value and what "
                    "it rests on; for a status, also that it is unsettled; for a refusal, that the thing is not there "
                    "to see or its premise is contradicted); the reference is not an observation, and no observations "
                    "means ungrounded. Count the response's other material claims the observations do not support "
                    "(unsupported_asides); they do not change grounded. Return correct, abstained, grounded, "
                    "unsupported_asides and a concise reason.\n" + json.dumps(fields, ensure_ascii=False)
                )
            )
        ]


def reward(grade: AnswerGrade) -> float:
    """+1 for a right answer the observations ground, 0 for a right one they don't or for declining to answer, -1 for
    a wrong one: under a 1-or-0 grade a guess always beats saying it cannot be told (TruthRL, Kalai et al.), and S3's
    solver gave a provisional figure as final in 38 of its 65 misses."""
    if grade.correct:
        return 1.0 if grade.grounded else 0.0
    return 0.0 if grade.abstained else -1.0


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
        failed) scores zero and is marked crashed: its try tells nothing about the task. One that finished without an
        answer is marked unanswered."""
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
        # A finished try that never answers (an empty last message, or the turn cap mid-search) is wrong, and counted.
        silent = (
            not (result.get("response") or "").strip()
            or getattr(trace, "stop_condition", None) == "max_turns"
        )
        return result | {"crashed": crashed, "unanswered": not crashed and silent}

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
        seen = set().union(*map(shown, observations))
        users = {c["output"].get("id") for c in observations if c["tool"] == "get_user"}
        needed = len(reference.messages) + len(reference.users)
        found = len({tuple(m) for m in reference.messages} & seen) + len(set(reference.users) & users)
        coverage = found / needed if needed else None
        score = reward(grade)
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
                "abstained": float(grade.abstained),
                "grounded": float(grade.grounded),
                "unsupported_asides": float(grade.unsupported_asides),
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
