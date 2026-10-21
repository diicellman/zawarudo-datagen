"""A forge's tries: the verifier's, which must reach a task's gold for it to count, and the student's, whose pass rate
says how hard it is. Each try runs as the solver does in training (the task's asker, the Slack tools, the answer
judge's grade), on one copy of the frozen world, and is kept the moment it lands under the task's key."""

import asyncio
import math

import httpx
import openai
from verifiers.v1.configs.runtime import NetworkPolicyConfig
from worldgen_slack.release import PublicTask
from worldgen_slack.taskset import SolverTask
from worldgen_slack.world import file_hash

from .tasks import Task, private_answer


def wilson(right: int, n: int, z: float = 1.645) -> list[float] | None:
    """The 90% Wilson interval of a share of right answers."""
    if not n:
        return None
    p = right / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(max(0.0, centre - spread), 3), round(min(1.0, centre + spread), 3)]


def rates(results: list[dict]) -> dict:
    """A task's tries in numbers: how often the answer was right (its difficulty) and its 90% interval, how often it
    was right and grounded (the reward's +1), how often the solver declined, and how much of the gold evidence the
    tries saw. A crashed try tells nothing about the task: it is counted, and left out."""
    finished = [r for r in results if r and not r.get("crashed")]
    seen = [r["evidence_coverage"] for r in finished if r.get("evidence_coverage") is not None]
    right = sum(bool(r.get("correct")) for r in finished)
    n = len(finished)
    return {
        "right_rate": right / n if n else None,
        "right_interval": wilson(right, n),
        "strict_rate": sum(bool(r.get("correct")) and bool(r.get("grounded")) for r in finished) / n
        if n
        else None,
        "abstain_rate": sum(bool(r.get("abstained")) for r in finished) / n if n else None,
        "coverage": sum(seen) / len(seen) if seen else None,
        "tries": n,
        "crashed": len([r for r in results if r]) - n,
        "unanswered": sum(bool(r.get("unanswered")) for r in finished),
    }


def by_part(task: Task, results: list[dict]) -> dict[str, float] | None:
    """A composite request's share of right answers by part, over its graded tries; None for a single question."""
    if len(task.parts) < 2:
        return None
    graded = [
        r["parts"]
        for r in results
        if r and not r.get("crashed") and len(r.get("parts") or []) == len(task.parts)
    ]
    if not graded:
        return None
    return {p.label: sum(g[n]["correct"] for g in graded) / len(graded) for n, p in enumerate(task.parts)}


def reached(task: Task, results: list[dict]) -> list[int]:
    """How many of the verifier's tries reached each part's answer: right on that part (a single question: right on
    the whole). An answer holds when each of its parts is reached twice, though no one try reaches them all: a part is
    a claim of its own, and the student is graded on the request whole."""
    counts = [0] * len(task.parts)
    for r in results:
        if not r or r.get("crashed"):
            continue
        if len(task.parts) == 1:
            counts[0] += bool(r.get("correct"))
        elif len(r.get("parts") or []) == len(task.parts):
            for n, part in enumerate(r["parts"]):
                counts[n] += bool(part.get("correct"))
    return counts


def agreeing(results: list[dict]) -> list[dict]:
    """The verifier's tries that reached the gold: right on every part. A claim of its own that no message states makes
    a try ungrounded, not the gold wrong, so grounding is the student's measure, not the verifier's."""
    return [r for r in results if r and not r.get("crashed") and r.get("correct")]


WAITS = (5, 30, 120)  # seconds before a call is asked again


RETRIED = (408, 409, 425, 429, 500, 502, 503, 504)  # a provider's statuses a call asked again may not meet


def transient(error: BaseException) -> bool:
    """An error of a provider's or the network's, which the same call asked again may not meet: a timeout, a lost
    connection, a rate limit, a server's error. Anything else (an answer that does not parse, a bug in our own code)
    fails at once: asked again, it is paid for again and fails again (forge-w2's regrade paid each of 1,610 grades
    four times over a missing trace method, $166, 2026-10-21)."""
    if isinstance(
        error, TimeoutError | httpx.TransportError | openai.APIConnectionError | openai.RateLimitError
    ):
        return True
    return isinstance(error, openai.APIStatusError) and error.status_code in RETRIED


async def patient(call, timeout: float):
    """A model call cut at `timeout` and asked again after each of WAITS when its error is transient (the model client
    has no read timeout, and Prime's errors come in bursts); any other error, and the last, is raised."""
    for wait in (*WAITS, None):
        try:
            async with asyncio.timeout(timeout):
                return await call()
        except Exception as error:
            if wait is None or not transient(error):
                raise
            await asyncio.sleep(wait)


class Tries:
    """The forge's tries of its tasks, on one solver copy of the world. The student's run `solvers` at a time (its
    account's limit), the verifier's beside them."""

    ATTEMPTS = (
        3  # a try that crashed (a hung request cut at its rollout time among them) runs again, up to 3 times
    )

    def __init__(self, settings, store):
        self.settings, self.store = settings, store
        self.gates = {
            "solver": asyncio.Semaphore(settings.solvers),
            "verifier": asyncio.Semaphore(settings.solvers),
        }

    def copy(self):
        """The world as the solver gets it, made once: the world is frozen, so every try of a run reads one copy."""
        path = self.store.root / "solver.sqlite"
        if not path.exists():
            staged = path.with_suffix(".tmp")
            staged.unlink(missing_ok=True)
            self.store.world.solver_copy(staged)
            staged.replace(path)
        return path

    def solver_task(self, task: Task, world) -> SolverTask:
        """A task as a solver gets it: its question, as its asker, on the solver copy; its grade against its answer."""
        row = PublicTask(
            task_id=task.id,
            question=task.question,
            actor_id=task.actor_id,
            category=task.family,
            level=1,
            answer_type="composite" if len(task.parts) > 1 else task.parts[0].answer_type,
            world_hash=file_hash(world),
        )
        return SolverTask.create(
            row,
            world,
            private_answer(self.store.world, task),
            self.settings.answer_judge,
            isinstance(self.settings.env.solver.runtime, NetworkPolicyConfig),
            self.settings.grading,
        )

    async def solve(self, agents, task: SolverTask, seat: str):
        """One independent solve by `seat`, graded for correctness and grounding. One that crashed runs again, and
        says why in `rerun`; crashed every time, it is kept as crashed, and the task's rates leave it out. So does a
        verifier's that ends with no answer (stuck at its turn cap on calls that lost their arguments): it says nothing
        of whether the answer can be reached. A student's unanswered try is its own failure, and counted."""
        first = None
        for attempt in range(self.ATTEMPTS):
            if attempt and (wait := WAITS[min(attempt - 1, len(WAITS) - 1)]):
                await asyncio.sleep(
                    wait
                )  # a provider's burst of refusals (429: too many concurrent requests) passes
            async with self.gates[seat]:
                trace = await getattr(agents, seat).run(task)
            self.store.trace(trace)
            outcome = SolverTask.outcome(trace)
            if not (outcome["crashed"] or (seat == "verifier" and outcome["unanswered"])):
                break
            if first is None:
                first = outcome.get("reason") or ("crashed" if outcome["crashed"] else "no answer")
        return (outcome | {"rerun": first[:200]} if first is not None else outcome), trace

    async def tried(self, agents, book: dict, task: Task, key: str, n: int, seat: str) -> list[dict]:
        """`n` tries of a task by `seat`, kept in `book` (the state's solves or verifier) under the task's key, each in
        its own place and saved the moment it lands: a run cut off tops the task up to `n` when it resumes, and a try
        that crashed is tried again, since it tells nothing of the task. A changed task (another key) is tried
        afresh."""
        entry = book.get(task.id)
        if entry is None or entry.get("key") != key:
            entry = book[task.id] = {"key": key, "results": [], "traces": []}
        while len(entry["results"]) < n:
            entry["results"].append(None)
            entry["traces"].append(None)
        world = self.copy()

        async def one(i: int) -> None:
            outcome, trace = await self.solve(agents, self.solver_task(task, world), seat)
            entry["results"][i], entry["traces"][i] = outcome, trace.id
            self.store.save()

        await asyncio.gather(
            *(one(i) for i, r in enumerate(entry["results"]) if r is None or r.get("crashed"))
        )
        return entry["results"]

    async def verify(self, agents, task: Task, key: str) -> list[dict]:
        """The verifier's tries: `verify` of them, and one more when a part is reached once; its answer holds when
        each part is reached twice. Tries that crashed reach nothing and refute nothing: short of `verify` that
        answered, the task fails (and is labelled again on a resume), rather than its answer being unreached."""
        n = self.settings.verify
        results = await self.tried(agents, self.store.state.verifier, task, key, n, "verifier")
        if crashed := [r for r in results if r and r.get("crashed")]:
            if len(results) - len(crashed) < n:
                raise RuntimeError(
                    f"the verifier's tries crashed: {crashed[0].get('rerun') or crashed[0].get('reason') or ''}"[
                        :300
                    ]
                )
        if min(reached(task, results)) == 1:  # a split on some part
            results = await self.tried(agents, self.store.state.verifier, task, key, n + 1, "verifier")
        return results

    async def measure(self, agents, task: Task, key: str) -> list[dict]:
        return await self.tried(agents, self.store.state.solves, task, key, self.settings.tries, "solver")
