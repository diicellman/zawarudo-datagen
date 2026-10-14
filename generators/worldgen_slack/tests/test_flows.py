"""The scripted runs end to end: a world with no tasks of its own through the CLI, the forge on it, a measure of a
finished run's tasks, and a chapter of a world with tasks of its own, with a forge on the version it leaves; then the
forge's release run through the env as an eval runs it."""

from types import SimpleNamespace

from worldgen_slack.taskset import EvaluationConfig, SlackEnv, SlackEnvConfig, SlackTasksetConfig
from worldgen_slack.toolset import SlackToolset

from .flows import GRADER, run_flows
from .support import with_state


async def test_flows(tmp_path):
    await run_flows(tmp_path / "author")
    await evaluated(tmp_path / "author" / "forge" / "release")


class Solver:
    """A solver played by code: it reads, as the task's asker, its own profile and the channel its answer is stated
    in, and answers right (the gold rows) or wrong; each try's trace is finalized and rewarded as the env's task
    does."""

    def __init__(self, right: bool):
        self.right, self.scores = right, []

    async def run(self, solve):
        tools = SlackToolset(solve.config.tools)
        await tools.setup()
        stored = with_state(tools)
        reference = solve.config.reference
        await tools._with_state(
            tools.read_user_profile
        )()  # its own profile: an answer rests on what was read
        for channel, _ in reference.messages[:1]:
            await tools._with_state(tools.read_channel)(channel_id=channel)
        answer = ", ".join(str(r.get("answer")) for r in reference.rows) if self.right else "nobody knows"
        trace = SimpleNamespace(
            id="solve", info={}, state=stored["state"], last_reply=answer + ".", record_metrics=lambda m: None
        )
        await solve.finalize(trace)
        self.scores.append(await solve.semantic_correctness(trace))


async def evaluated(release):
    """The release through the env with no model: SlackTaskset loads its tasks, each validates against the release,
    SlackEnv runs it with the solver seat, and the answer judge (played by code, grounding what the solver read)
    rewards a right answer +1 and a wrong one -1."""
    env = SlackEnv(
        SlackEnvConfig(
            taskset=SlackTasksetConfig(id="worldgen-slack", task=EvaluationConfig(release_dir=release))
        )
    )
    tasks = env.taskset.load()
    assert tasks, "the release holds the forge's tasks"
    GRADER.update(on=True, loose=False, blind=False, grounded=True)
    try:
        for right, score in ((True, 1.0), (False, -1.0)):
            solver = Solver(right)
            for task in tasks:
                await task.validate(None)
                await env.run(task, SimpleNamespace(solver=solver))
            assert solver.scores == [score] * len(tasks), solver.scores
    finally:
        GRADER.update(on=False, grounded=False)
