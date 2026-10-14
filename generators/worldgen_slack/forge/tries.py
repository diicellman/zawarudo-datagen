"""A forge's tries of its candidates: the screen, the solver's runs, the witness and the judge's review on them, and
the rates a task's tries come to."""

import asyncio
import contextlib
import json
import math

from verifiers.v1.configs.runtime import NetworkPolicyConfig
from worldgen_slack.judge import AnswerJudge
from worldgen_slack.taskset import SolverTask
from worldgen_slack.world import digest, file_hash

from ..judge.task import review, review_payload
from ..judge.verdict import Issue, Verdict, accepted_task, combined, read_one_way
from ..session import observable
from .difficulty import failure_cases
from .release import release_rows


def wilson(right: int, n: int, z: float = 1.645) -> list[float]:
    """The 90% interval of a share of right answers out of n tries: how little a few tries pin a rate down."""
    p, spread = right / n, z * z / n
    centre, half = (
        (p + spread / 2) / (1 + spread),
        z * math.sqrt(p * (1 - p) / n + spread / (4 * n)) / (1 + spread),
    )
    return [round(max(0.0, centre - half), 3), round(min(1.0, centre + half), 3)]


def rates(results: list[dict]) -> dict:
    """A task's tries in numbers: how often the answer was right (the task's difficulty) and its 90% interval, how
    often it was right and grounded (the reward's +1), how often the solver declined to answer, and how much of the
    gold evidence the tries saw. A crashed try tells nothing about the task: it is counted, and left out."""
    finished = [r for r in results if not r.get("crashed")]
    seen = [r["evidence_coverage"] for r in finished if r.get("evidence_coverage") is not None]
    right = sum(bool(r["correct"]) for r in finished)
    return {
        "right_rate": right / len(finished),
        "right_interval": wilson(right, len(finished)),
        "strict_rate": sum(bool(r["correct"]) and bool(r.get("grounded")) for r in finished) / len(finished),
        "abstain_rate": sum(bool(r.get("abstained")) for r in finished) / len(finished),
        "coverage": sum(seen) / len(seen) if seen else None,
        "tries": len(finished),
        "crashed": len(results) - len(finished),
        "unanswered": sum(bool(r.get("unanswered")) for r in finished),
    }


# The anchor messages of the decoys on a task's facts' subjects: the near-misses a task rests beside.
DECOYS = """SELECT m.id, m.channel_id, m.ts, m.text FROM task_facts tf JOIN facts f ON f.id = tf.fact_id
JOIN facts d ON d.is_decoy = 1 AND lower(trim(d.subject)) = lower(trim(f.subject))
JOIN evidence e ON e.fact_id = d.id JOIN messages m ON m.id = e.message_id
WHERE tf.task_id = ? GROUP BY m.id ORDER BY m.ts_us, m.id"""


class Tries:
    """A forge's tries of its candidates: the screen of wrong answers, the solver's runs (kept with what the task was
    when they ran), the witness, and the judge's review on them."""

    def __init__(self, settings, store):
        self.settings, self.store = settings, store
        # GLM has its own gate, under the episode's: judges and the witness are not held up behind its solves.
        self.glm = asyncio.Semaphore(settings.author.solvers)

    @property
    def world(self):
        return self.store.world

    def task_key(self, task_id: str) -> str:
        """What a task review depends on: the task, its gold rows, the text of the messages they rest on, and the text
        of the decoys on its facts' subjects; a revised decoy is a changed task."""
        db = self.world.db
        task = dict(
            db.execute(
                "SELECT id, actor_id, question, answer_type, gold_sql, gold_json FROM tasks WHERE id = ?",
                (task_id,),
            ).fetchone()
        )
        texts = [
            r[0]
            for r in db.execute(
                """SELECT m.text FROM task_facts tf JOIN evidence e ON e.fact_id = tf.fact_id
                JOIN messages m ON m.id = e.message_id WHERE tf.task_id = ? ORDER BY m.ts_us""",
                (task_id,),
            )
        ]
        decoys = [r["text"] for r in db.execute(DECOYS, (task_id,))]
        return digest([task, texts, decoys])

    def refresh_gold(self):
        """Gold rows follow the world: every task's query runs again as its actor after the world changes."""
        with self.world.trial() as copy:
            for task_id, actor, sql in copy.db.execute("SELECT id, actor_id, gold_sql FROM tasks").fetchall():
                rows = copy.gold(actor, sql, max_rows=self.settings.tasks.max_answer_rows)["rows"]
                copy.db.execute("UPDATE tasks SET gold_json = ? WHERE id = ?", (json.dumps(rows), task_id))

    def solver_task(self, task_id, world, hint: bool = False):
        """The task as the solver gets it; with `hint`, its question is followed by the messages its answer rests on
        (SPADE's privileged hint: a task the solver misses even so is hard to reason, or broken, not hard to find)."""
        row, answer = next((r, a) for r, a in zip(*self.answers(world)) if r.task_id == task_id)
        if hint:
            texts = [
                self.world.db.execute(
                    "SELECT text FROM messages WHERE channel_id = ? AND ts = ?", (c, ts)
                ).fetchone()[0]
                for c, ts in answer.messages
            ]
            row = row.model_copy(
                update={
                    "question": row.question
                    + "\n\nThese messages bear on it:\n"
                    + "\n".join(f"- {t}" for t in texts)
                }
            )
        return SolverTask.create(
            row,
            world,
            answer,
            self.settings.answer_judge,
            isinstance(self.settings.env.solver.runtime, NetworkPolicyConfig),
        )

    def answers(self, world):
        rows, answers = release_rows(self.store, file_hash(world))
        return rows, [answers[r.task_id] for r in rows]

    async def solve(self, agents, task, seat: str = "solver"):
        """One independent solve of a task by `seat` (GLM, or the witness), graded for correctness and grounding. A
        solve that crashed runs once more; crashed again, it is kept as crashed, and the task's rates leave it out."""
        for _ in range(2):
            async with self.glm if seat == "solver" else contextlib.nullcontext():
                trace = await getattr(agents, seat).run(task)
            self.store.trace(trace)
            if not (outcome := SolverTask.outcome(trace))["crashed"]:
                break
        return outcome, trace

    async def solves(self, agents, task_ids, attempt, n, seat: str = "solver", hint: bool = False) -> dict:
        """`n` independent solves of each task by `seat`, on a solver copy of the world kept with the attempt."""
        copy = self.store.path(attempt, f"{seat}.sqlite")
        copy.unlink(missing_ok=True)
        self.world.solver_copy(copy)
        runs = await asyncio.gather(
            *(
                asyncio.gather(*(self.solve(agents, self.solver_task(t, copy, hint), seat) for _ in range(n)))
                for t in task_ids
            )
        )
        if dead := [t for t, tries in zip(task_ids, runs) if all(o["crashed"] for o, _ in tries)]:
            raise RuntimeError(f"every try of {dead} crashed twice: the solver or its grading is failing")
        return dict(zip(task_ids, runs))

    async def judge_tasks(self, agents, due, keys, attempt, n, previous=()) -> tuple[dict, Verdict]:
        """The judge's review of the `due` tasks, on the solver's runs of each as it is now: runs kept from earlier
        are reused, and only changed tasks are solved. A valid review is kept with its task's key, so a later review
        skips the task until it changes."""
        state = self.store.state
        loose = await self.screen(due, keys, attempt)
        fresh = [t for t in due if state.solves.get(t, {}).get("key") != keys[t]]
        solved = await self.solves(agents, fresh, attempt, n)
        for task_id, task_runs in solved.items():
            self.keep_solves(task_id, keys[task_id], task_runs)
        runs = {t: state.solves[t] for t in due}
        witnessed = await self.witness(agents, due, keys, attempt)
        size = self.settings.author.review_chunk
        chunks = [due[i : i + size] for i in range(0, len(due), size)]

        async def judged(k: int, chunk: list[str]) -> Verdict:
            witness = [result for t in chunk for result in witnessed.get(t, {}).get("results", [])]
            payload = review_payload(
                self.world,
                "task",
                chunk,
                self.settings.taxonomy,
                solves=[result for t in chunk for result in runs[t]["results"]],
                previous_issues=[i.model_dump(mode="json") for i in previous if set(i.task_ids) & set(chunk)],
                **({"witness": witness} if witness else {}),
            )
            files = {
                f"{seat}_{t}_{k}.json": record
                for seat, tried in (("solver", runs), ("witness", witnessed))
                for t in chunk
                if t in tried
                for k, trace_id in enumerate(tried[t]["traces"], 1)
                if (record := self.saved_trace(trace_id))
            }
            return await review(
                self.store,
                agents,
                payload,
                attempt,
                self.settings.tasks.max_answer_rows,
                files,
                "tasks" if len(chunks) == 1 else f"tasks-{k}",
            )

        # A review of a few tasks at a time, the reviews side by side: each task passes on its own chunk's verdict,
        # and on its failure cases: a task whose grade accepts a wrong answer is ambiguous as asked.
        verdicts = await asyncio.gather(*(judged(k, chunk) for k, chunk in enumerate(chunks, 1)))
        issues = {
            t: Issue(
                artifact="tasks",
                task_ids=[t],
                defect="the answer judge accepts a wrong answer to it: " + "; ".join(failed),
                requested_change="ask it so that its gold, and only its gold, is the right answer: tell it apart from what the judge accepted",
                blocking=True,
            )
            for t, failed in loose.items()
            if failed
        }
        verdicts = [
            v.model_copy(
                update={
                    "issues": v.issues + [issues[t] for t in chunk if t in issues],
                    "approved": v.approved and not any(t in issues for t in chunk),
                }
            )
            for chunk, v in zip(chunks, verdicts)
        ]
        verdicts = [read_one_way(v) for v in verdicts]
        verdict_of = {t: v for chunk, v in zip(chunks, verdicts) for t in chunk}
        for (
            task_id
        ) in due:  # the issues each task's latest review raised: why a candidate is dropped or sent back
            state.task_issues[task_id] = [i for i in verdict_of[task_id].issues if task_id in i.task_ids]
        for task_id in due:
            results = runs[task_id]["results"]
            rate = rates(results)
            approved = accepted_task(verdict_of[task_id], task_id)
            fit = next(r.level_fit for r in verdict_of[task_id].tasks if r.task_id == task_id)
            seen = rates(witnessed[task_id]["results"])["right_rate"] if task_id in witnessed else None
            if approved:
                state.task_reviews[task_id] = {
                    "key": keys[task_id],
                    **rate,
                    "witness_right": seen,
                    "level_fit": fit,
                    "results": results,
                }
            right = rate["right_rate"]
            self.store.event(
                "task_reviewed",
                attempt=attempt,
                task_id=task_id,
                approved=approved,
                **rate,
                learnability=4 * right * (1 - right),
            )
        return runs, combined(verdicts)

    async def grade(self, question: str, reference: dict, response: str):
        """The answer judge's grade of one response, with no observations: its `correct` is what a screen reads."""
        for again in (True, False):
            try:
                return await AnswerJudge(self.settings.answer_judge).evaluate(
                    question=question, reference=reference, response=response, observations=[]
                )
            except Exception:
                if not again:
                    raise

    async def screen(self, due, keys, attempt) -> dict[str, list[str]]:
        """Each task's failure cases (`failure_cases`), graded by the answer judge before any try: returns, per task,
        those it accepted, and its gold if it rejected that. Kept with the task's key, so a task is screened once."""
        state = self.store.state
        for t in [t for t in due if state.screens.get(t, {}).get("key") != keys[t]]:
            row = self.world.db.execute(
                "SELECT question, answer_type, gold_json FROM tasks WHERE id = ?", (t,)
            ).fetchone()
            rows = json.loads(row["gold_json"])
            gold, cases = failure_cases(self.world, t, rows)
            reference = {"answer_type": row["answer_type"], "rows": rows}
            graded = await asyncio.gather(
                *(self.grade(row["question"], reference, r) for r in [gold, *(r for _, r in cases)])
            )
            failed = (["its gold answer, judged wrong"] if not graded[0].parsed.correct else []) + [
                f"{name} ({response!r})" for (name, response), g in zip(cases, graded[1:]) if g.parsed.correct
            ]
            usage = [g.usage.model_dump(mode="json") for g in graded if getattr(g, "usage", None) is not None]
            state.screens[t] = {"key": keys[t], "failed": failed, "cases": len(cases), "usage": usage}
            self.store.event("task_screened", attempt=attempt, task_id=t, cases=len(cases), failed=failed)
        return {t: state.screens[t]["failed"] for t in due}

    async def witness(self, agents, due, keys, attempt) -> dict:
        """A stronger solver's tries of each task GLM answers right less often than its band's floor, and of each task
        whose band starts at 0: a task it answers is hard, not broken. Its tries are kept with the task's key and count
        in no rate. Returns the tries of each low task."""
        state, cfg = self.store.state, self.settings
        level = dict(self.world.db.execute("SELECT id, level FROM tasks").fetchall())

        def low(t: str) -> bool:
            floor = cfg.tasks.bands[level[t] - 1][0]
            return floor == 0 or rates(state.solves[t]["results"])["right_rate"] < floor

        if cfg.author.witness_tries and (
            fresh := [t for t in due if low(t) and state.witness.get(t, {}).get("key") != keys[t]]
        ):
            for t, runs in (
                await self.solves(agents, fresh, attempt, cfg.author.witness_tries, "witness")
            ).items():
                state.witness[t] = {
                    "key": keys[t],
                    "results": [r for r, _ in runs],
                    "traces": [x.id for _, x in runs],
                }
        return {t: state.witness[t] for t in due if low(t) and state.witness.get(t, {}).get("key") == keys[t]}

    def keep_solves(self, task_id: str, key: str, runs: list) -> None:
        """A task's solver runs, kept with what the task was when they ran."""
        self.store.state.solves[task_id] = {
            "key": key,
            "results": [r for r, _ in runs],
            "traces": [t.id for _, t in runs],
        }

    def saved_trace(self, trace_id: str) -> dict | None:
        """A saved solver trace, as the judge may read it."""
        path = self.store.root / "traces" / f"{trace_id}.json"
        return observable(json.loads(path.read_text())) if path.exists() else None
