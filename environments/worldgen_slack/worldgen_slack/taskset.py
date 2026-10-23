"""Slack QA tasks over one world file: solve with the actor's Slack tools, grade once with a native LLM judge."""

from datetime import datetime
from pathlib import Path
from typing import Self
from zoneinfo import ZoneInfo

import verifiers.v1 as vf
from pydantic import Field
from verifiers.v1.configs.runtime import NetworkPolicyConfig

from .config import role
from .judge import READINGS, AnswerJudge, Misread, empty_grade, fenced, gold, graded, reward
from .release import PrivateAnswer, PublicTask, load_release
from .slack import people_shown, shown
from .toolset import CallState, SlackToolset, WorldToolsConfig, stage_world
from .world import World


class WorldTaskData(vf.TaskData):
    """Which task and attempt a run belongs to, and which world file it was given."""

    task_id: str = ""
    attempt: str = ""
    world_hash: str = ""


def system_prompt(world) -> str:
    """What a Slack client tells its assistant before the first call: who it acts as (the real server writes the
    caller's id into its tools' descriptions), and the date and time (the client's own)."""
    row = world.db.execute("SELECT value FROM world_meta WHERE key = 'now_us'").fetchone()
    now = datetime.fromtimestamp(int(row[0]) / 1e6, ZoneInfo(world.actor["tz"])) if row else None
    when = (
        f" It is {now:%A, %B} {now.day}, {now:%Y}, {now:%H:%M} on your clock ({world.actor['tz']})."
        if now
        else ""
    )
    return f"You are using Slack as the user with user_id {world.actor['id']}." + when


def out_of_sight(path: str, actor_id: str, reference: PrivateAnswer) -> list[list[dict]]:
    """Each part's messages out of its asker's sight, as the grader reads them: who wrote what, where and when (on the
    asker's clock). They show how things really stand, against which a flat claim is judged."""
    world = World(path)
    try:
        zone = ZoneInfo(world.db.execute("SELECT tz FROM users WHERE id = ?", (actor_id,)).fetchone()[0])
        hidden = []
        for answer in reference.answers():
            said = []
            for channel, ts in answer.unseen:
                row = world.db.execute(
                    """SELECT m.ts_us, m.text, m.parent_id, c.name, c.type, u.real_name FROM messages m
                    JOIN channels c ON c.id = m.channel_id JOIN users u ON u.id = m.user_id
                    WHERE m.channel_id = ? AND m.ts = ?""",
                    (channel, ts),
                ).fetchone()
                if row is None:
                    raise ValueError(f"a message out of sight, {channel} {ts}, is not in the world")
                where = (
                    f"#{row['name']}" if row["name"] else ("a DM" if row["type"] == "im" else "a group DM")
                )
                when = datetime.fromtimestamp(row["ts_us"] / 1e6, zone)
                said.append(
                    {
                        "from": row["real_name"],
                        "where": where + (" (thread reply)" if row["parent_id"] else ""),
                        "when": f"{when:%a %b} {when.day}, {when:%H:%M}",
                        "text": row["text"],
                    }
                )
            hidden.append(said)
        return hidden
    finally:
        world.close()


class SolverConfig(vf.TaskConfig):
    tools: WorldToolsConfig
    reference: PrivateAnswer = Field(exclude=True, repr=False)
    judge: vf.JudgeConfig = vf.JudgeConfig(
        model="z-ai/glm-5.3", sampling=vf.SamplingConfig(reasoning_effort="high")
    )


class SolverTask(vf.Task[WorldTaskData, CallState, SolverConfig]):
    @classmethod
    def toolsets(cls, config):
        return [SlackToolset(config.tools)]

    @classmethod
    def create(
        cls,
        task: PublicTask,
        world: Path,
        reference: PrivateAnswer,
        judge=None,
        network_policy=True,
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
                system_prompt=system_prompt(World(world, actor=task.actor_id)),
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
        outcome, reading = "error", {}
        if not (trace.last_reply or "").strip():
            grade = empty_grade(reference)
        else:
            # The model reads what the answer commits to, criterion by criterion; code grades it. A reading that
            # misreads the gold's parts or items, or a reply that does not parse, is asked again (each call recorded on
            # the trace once it parses): a grade costs a reading, where its try's crash would cost the whole try again.
            hidden = out_of_sight(self.config.tools.db_path, self.config.tools.actor_id, reference)
            for n in range(1, READINGS + 1):
                try:
                    result = await AnswerJudge(self.config.judge).evaluate(
                        trace=trace,
                        request=self.data.prompt,
                        reference=gold(reference, hidden),
                        response=fenced(trace.last_reply),
                    )
                    grade, outcome = graded(result.parsed, reference, bool(observations))
                except Misread:
                    if n == READINGS:
                        raise
                else:
                    reading = {"reading": result.parsed.model_dump(mode="json"), "readings": n}
                    break
        seen = set().union(*map(shown, observations))
        users = set().union(*map(people_shown, observations))

        def covered(answer) -> float | None:  # the share of what its answer rests on that its calls showed
            needed = len(answer.messages) + len(answer.users)
            found = len({tuple(m) for m in answer.messages} & seen) + len(set(answer.users) & users)
            return found / needed if needed else None

        coverage = covered(reference)
        # Grounded by message, not by topic alone: an answer whose gold rests on messages is grounded only if its
        # tries were shown one of them (a stale message is topically just as close); a composite request's, only if
        # each part's was.
        grade = grade.model_copy(
            update={"grounded": grade.grounded and all(covered(a) != 0 for a in reference.answers())}
        )
        score = reward(grade)
        trace.info["evaluation"] = {
            "task_id": self.data.task_id,
            "execution_ok": True,
            "semantic_correctness": score,
            **grade.model_dump(mode="json"),
            "outcome": outcome,
            **reading,
            "response": trace.last_reply,
            "evidence_coverage": coverage,
            **({"part_coverage": [covered(p) for p in reference.parts]} if reference.parts else {}),
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
    release_dir: Path | None = None  # the release to evaluate: a generated world's release/ directory
    judge: vf.JudgeConfig = vf.JudgeConfig(
        model="z-ai/glm-5.3", sampling=vf.SamplingConfig(reasoning_effort="high")
    )


def release(config: EvaluationConfig):
    if config.release_dir is None:
        raise ValueError("set the release to evaluate: --env.taskset.task.release_dir")
    return load_release(config.release_dir)


class EvaluationTask(vf.Task[WorldTaskData, vf.State, EvaluationConfig]):
    def solver_task(self, network_policy: bool = True):
        """The release's task as its solver gets it; `network_policy` where the solver's runtime enforces one (a VM's,
        not a local process's)."""
        world, rows, answers = release(self.config)
        row = next(r for r in rows if r.task_id == self.data.task_id)
        if (row.question, row.world_hash) != (self.data.prompt, self.data.world_hash):
            raise ValueError("task differs from the immutable release")
        return SolverTask.create(row, world, answers[row.task_id], self.config.judge, network_policy)

    async def validate(self, runtime):
        self.solver_task()


class SlackTasksetConfig(vf.TasksetConfig):
    task: EvaluationConfig = EvaluationConfig()


class SlackTaskset(vf.Taskset[EvaluationTask, SlackTasksetConfig]):
    def load(self):
        _, rows, _ = release(self.config.task)
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
    # The model is the run's; the solver runs no code, so no VM.
    solver: vf.AgentConfig = role(None, solver=True).model_copy(update={"runtime": vf.SubprocessConfig()})


class SlackEnv(vf.Env[SlackEnvConfig]):
    async def run(self, task: EvaluationTask, agents: vf.Agents):
        # a local solver (the default) has no egress to block; one in a VM gets the task's network policy
        await agents.solver.run(task.solver_task(isinstance(self.config.solver.runtime, NetworkPolicyConfig)))
