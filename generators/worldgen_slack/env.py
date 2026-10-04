"""The generation episode: one author agent writes the world in time order (premise, organization, the ledger,
every day's conversations, the tasks); the judge reviews it mid-run and at the end; the GLM solver tries the tasks.

The rules live in `chronicle` and `contracts`, the author's tools in `agents/world`; the world itself is one SQLite
file, written only through `World.trial()`."""

import asyncio
import json

import verifiers.v1 as vf
from verifiers.v1.configs.runtime import NetworkPolicyConfig
from verifiers.v1.errors import SandboxError
from worldgen_slack.db import digest
from worldgen_slack.taskset import SolverTask

from .agents import synthesizer
from .agents.judge import JudgeTask, review_payload
from .agents.synthesizer import SynthesizerTask
from .agents.world import (
    WorldAuthorTask,
    context_of,
    day_prompt,
    files,
    fix_prompt,
    harden_prompt,
    ledger_digest,
    plan_prompt,
    tasks_prompt,
)
from .chronicle import bounds, close_day, posted, present, start_clock, today
from .config import Config, PipelineConfig
from .contracts import (
    PHASE_CRITERIA,
    Organization,
    Verdict,
    accepted,
    accepted_task,
    background_plan,
    clock,
    deciding,
    organize,
    pick_cast,
    quality,
    quota,
    user_id,
)
from .store import ReviewLimit, Store, used_names

NOTES = ("plan.md", "recap.md")  # the world author's own files in /task/notes
SETUP_FILES = ("input.json", "guide.md", "schemas.json", "schema.sql", "world.sqlite", "premise.json", "organization.json")  # fmt: skip


def require_trace(trace):
    if not trace.ok:
        raise RuntimeError("agent execution failed: " + "; ".join(e.message for e in trace.errors))


def observable(record: dict) -> dict:
    """A trace record without hidden reasoning or provider state, for another agent to investigate."""
    for node in record["nodes"]:
        node["message"].pop("reasoning_content", None)
        node["message"].pop("provider_state", None)
    return record


class GenerationEnv(vf.Env[PipelineConfig]):
    def __init__(self, settings: Config, store: Store):
        self.settings, self.store = settings, store
        self.used = used_names(settings.corpus, store.root)
        super().__init__(settings.env)

    @property
    def world(self):
        return self.store.world

    async def setup(self, agents):
        for name in ("author", "judge", "solver"):
            getattr(agents, name).trainable = False

    def budget_check(self):
        if self.store.summary("running")["reported_model_cost"] >= self.settings.research_budget_usd:
            raise ReviewLimit("reported model spend reached the budget")

    # ------------------------------------------------------------------ authors
    async def author_turn(self, interaction, runtime, task_cls, context, attempt, first):
        self.budget_check()
        self.store.event("author_started", attempt=attempt, phase_document=context["phase"])
        for name, text in task_cls.files(context).items():
            await runtime.write("/task/" + name, text.encode())
        await runtime.write("/task/world.sqlite", self.world.path.read_bytes())
        segment = await interaction.turn(
            None
            if first
            else f"/task/input.json is updated for phase {context['phase']}. "
            + ("Address its feedback. " if context.get("feedback") else "")
            + f"Write that phase's document to /task/{context['phase']}.json."
        )
        if segment.terminated:
            raise ReviewLimit("author exhausted its native interaction limits")
        try:
            raw = (await runtime.read(f"/task/{context['phase']}.json", max_bytes=24_000_000)).decode()
        except SandboxError:
            raw = ""  # the author ended its turn without writing the file; `author` asks again
        self.store.artifact(attempt, "author_" + context["phase"], {"text": raw})
        self.store.trace(interaction.trace)
        return raw

    async def author(self, interaction, runtime, task_cls, context, attempt, first, parse):
        """Code's rejections go back to the author twice within the attempt before the attempt is rejected."""
        errors = []
        for _ in range(3):
            raw = await self.author_turn(interaction, runtime, task_cls, context, attempt, first)
            first = False
            try:
                if not raw.strip():
                    raise ValueError(
                        f"/task/{context['phase']}.json was not written; write the complete document"
                    )
                return parse(raw)
            except ValueError as error:
                errors.append(str(error)[:8000])
                context = {**context, "feedback": errors[-1]}
                self.store.artifact(attempt, "corrections", errors)
        raise ValueError(errors[-1])

    def apply(self, apply, model, phase):
        """A parse function for `author`: validate the document, apply it to a trial copy of the world, keep it."""

        def parse(raw):
            document = model.model_validate_json(raw)
            with self.world.trial() as copy:
                apply(copy, document)
            self.store.state.drafts[phase] = document.model_dump(mode="json")
            return document

        return parse

    # ------------------------------------------------------------------ setup: the premise and the organization
    async def premise_and_organization(self, interaction, runtime, then: str) -> bool:
        """S0-S1 in one author session: premises (the seed picks one), then the cast and the task quota, then the
        organization; the phase moves to `then`. Returns whether the session's next turn is its first."""
        state, cfg = self.store.state, self.settings
        first = True
        if state.premise is None:
            try:
                state.premise = await self.author(
                    interaction, runtime, SynthesizerTask, synthesizer.premise_context(cfg, self.used), "premise", True,
                    lambda raw: synthesizer.parse_premise(raw, cfg, self.used),
                )  # fmt: skip
            except ValueError as error:
                raise ReviewLimit(f"the author proposed no valid premise: {error}") from None
            first = False
            state.cast = pick_cast(cfg.personas, cfg.seed, self.used["people"], state.premise.staffing)
            state.quota = quota(cfg.taxonomy, cfg.tasks.styles, cfg.seed, cfg.tasks.count)
            state.phase = "organization"
            self.store.save()
            self.store.event("premise_selected", company=state.premise.company, quota=state.quota)
        if state.phase == "organization":
            try:
                await self.author(
                    interaction, runtime, SynthesizerTask, synthesizer.organization_context(cfg, state, self.used),
                    "organization", first,
                    self.apply(lambda w, d: organize(w, d, state.cast, state.premise, cfg), Organization, "organization"),
                )  # fmt: skip
            except ValueError as error:
                raise ReviewLimit(f"the author built no valid organization: {error}") from None
            first = False
            state.organization = [r[0] for r in self.world.db.execute("SELECT id FROM channels")]
            state.phase = then
            self.store.save()
            self.store.event("organization_built", channels=len(state.organization))
        return first

    def reject_structure(self, attempt, error):
        self.store.state.feedback, self.store.state.last_verdict = error[:8000], None
        self.store.artifact(attempt, "validation", {"ok": False, "errors": [error]})
        self.store.finish_attempt(False)

    # ------------------------------------------------------------------ reviews
    async def review(self, agents, payload, attempt, files=None, label=""):
        """`files` maps /task file names to trace records the judge may investigate; `label` names one of an
        attempt's several reviews."""
        suffix, files = ("-" + label if label else ""), files or {}
        self.budget_check()
        self.store.artifact(attempt, "review_input" + suffix, payload)
        self.store.event(
            "review_started", attempt=attempt, label=label, task_ids=[t["id"] for t in payload["tasks"]]
        )
        world = self.store.snapshot(attempt, f"review{suffix}.sqlite")
        # One fresh rerun absorbs a malformed verdict; the review round is spent once.
        for _ in range(2):
            task = JudgeTask.create(
                payload,
                world,
                attempt,
                self.settings.tasks.max_answer_rows,
                {n: json.dumps(r) for n, r in files.items()},
            )
            trace = await agents.judge.run(task)
            self.store.trace(trace)
            if trace.ok:
                break
        require_trace(trace)
        verdict = Verdict.model_validate(trace.info["verdict"])
        self.store.artifact(attempt, "verdict" + suffix, verdict.model_dump(mode="json"))
        verdict = verdict.model_copy(update={"approved": accepted(verdict, self.settings.acceptance)})
        if payload["phase"] in PHASE_CRITERIA:
            self.store.artifact(
                attempt, "acceptance" + suffix, {"approved": verdict.approved, "quality": quality(verdict)}
            )
        return verdict

    def task_key(self, task_id: str) -> str:
        """What a task review depends on: the task, its gold rows, and the text of the messages they rest on."""
        db = self.world.db
        task = dict(db.execute("SELECT id, actor_id, question, answer_type, gold_sql, gold_json FROM tasks WHERE id = ?", (task_id,)).fetchone())  # fmt: skip
        texts = [
            r[0]
            for r in db.execute(
                """SELECT m.text FROM task_facts tf JOIN evidence e ON e.fact_id = tf.fact_id
                JOIN messages m ON m.id = e.message_id WHERE tf.task_id = ? ORDER BY m.ts_us""",
                (task_id,),
            )
        ]
        return digest([task, texts])

    def refresh_gold(self):
        """Gold rows follow the world: every task's query runs again as its actor after the world changes."""
        with self.world.trial() as copy:
            for task_id, actor, sql in copy.db.execute("SELECT id, actor_id, gold_sql FROM tasks").fetchall():
                rows = copy.gold(actor, sql, max_rows=self.settings.tasks.max_answer_rows)["rows"]
                copy.db.execute("UPDATE tasks SET gold_json = ? WHERE id = ?", (json.dumps(rows), task_id))

    def solver_task(self, task_id, world):
        row, answer = next((r, a) for r, a in zip(*self.answers(world)) if r.task_id == task_id)
        return SolverTask.create(
            row,
            world,
            answer,
            self.settings.answer_judge,
            isinstance(self.settings.env.solver.runtime, NetworkPolicyConfig),
        )

    def answers(self, world):
        from worldgen_slack.dataset import sha256

        rows, answers = self.store.release_rows(sha256(world))
        return rows, [answers[r.task_id] for r in rows]

    async def solve(self, agents, task):
        """One independent solve of a task, graded for correctness and grounding."""
        self.budget_check()
        trace = await agents.solver.run(task)
        self.store.trace(trace)
        return SolverTask.outcome(trace), trace

    async def solves(self, agents, task_ids, attempt, n) -> dict:
        """`n` independent solves of each task, on a solver copy of the world kept with the attempt."""
        copy = self.store.path(attempt, "solver.sqlite")
        copy.unlink(missing_ok=True)
        self.world.solver_copy(copy)
        runs = await asyncio.gather(*(asyncio.gather(*(self.solve(agents, self.solver_task(t, copy)) for _ in range(n))) for t in task_ids))  # fmt: skip
        return dict(zip(task_ids, runs))

    async def assess(self, agents, attempt):
        """The final review: every task whose review is stale is solved and judged, and the whole world is judged."""
        state, n = self.store.state, self.settings.solves_per_task
        self.refresh_gold()
        keys = {t: self.task_key(t) for (t,) in self.world.db.execute("SELECT id FROM tasks ORDER BY id")}
        # A task replaced since its review is gone from the world, and so are its review and its runs.
        state.task_reviews = {t: r for t, r in state.task_reviews.items() if t in keys}
        state.solves = {t: r for t, r in state.solves.items() if t in keys}
        due = [t for t, key in keys.items() if state.task_reviews.get(t, {}).get("key") != key]
        previous = state.last_verdict.issues if state.last_verdict else []
        people = {user_id(p.uuid): p.typing.model_dump(exclude={"id", "messages"}) for p in state.cast}

        async def check():
            # The solver's runs of a task as it is now are reused, a probe's included; only changed tasks are solved.
            fresh = [t for t in due if state.solves.get(t, {}).get("key") != keys[t]]
            for task_id, task_runs in (await self.solves(agents, fresh, attempt, n)).items():
                self.keep_solves(task_id, keys[task_id], task_runs)
            runs = {t: state.solves[t] for t in due}
            payload = review_payload(
                self.world, "task", due, self.settings.taxonomy,
                solves=[result for t in due for result in runs[t]["results"]],
                previous_issues=[i.model_dump(mode="json") for i in previous if set(i.task_ids) & runs.keys()],
            )  # fmt: skip
            files = {f"solver_{t}_{k}.json": record for t in due for k, trace_id in enumerate(runs[t]["traces"], 1) if (record := self.saved_trace(trace_id))}  # fmt: skip
            return runs, payload, await self.review(agents, payload, attempt, files, "tasks")

        workspace = [i.model_dump(mode="json") for i in previous if i.artifact == "workspace"]
        world_payload = review_payload(self.world, "world", [], people=people, ledger=ledger_digest(self.world), **({"previous_issues": workspace} if workspace else {}))  # fmt: skip
        async with asyncio.TaskGroup() as group:
            world = group.create_task(self.review(agents, world_payload, attempt, {}, "world"))
            checked = group.create_task(check()) if due else None
        runs, payload, judged = checked.result() if checked else ({}, None, None)
        verdicts = [v for v in (judged, world.result()) if v is not None]
        rejected = [v for v in verdicts if not v.approved]
        for task_id, cached in runs.items():
            results = cached["results"]
            rate = sum(r["semantic_correctness"] for r in results) / len(results)
            approved = accepted_task(judged, self.settings.acceptance, task_id)
            fit = next(r.level_fit for r in judged.tasks if r.task_id == task_id)
            if approved:
                state.task_reviews[task_id] = {"key": keys[task_id], "solve_rate": rate, "level_fit": fit, "results": results}  # fmt: skip
            self.store.event("task_reviewed", attempt=attempt, task_id=task_id, approved=approved, solve_rate=rate, learnability=4 * rate * (1 - rate))  # fmt: skip
        merged = Verdict.model_construct(
            approved=not rejected,
            tasks=[r for v in verdicts for r in v.tasks],
            issues=[i for v in verdicts for i in v.issues],
            criteria={},
            # A rejection's feedback carries only the rejecting reviews' words.
            summary=" ".join(v.summary for v in rejected or verdicts),
        )
        self.store.artifact(attempt, "verdict", merged.model_dump(mode="json"))
        self.store.artifact(
            attempt, "acceptance", {"approved": merged.approved, "reviewed_tasks": list(runs)}
        )
        return merged

    # ------------------------------------------------------------------ the world, written in time order
    def author_context(self) -> dict:
        state = self.store.state
        return context_of(self.settings, state, self.store.root, state.drafts.get("organization"))

    async def author_step(self, interaction, runtime, prompt: str, attempt: str, mode: str) -> None:
        """One author turn: code's memory of the world, rendered fresh, then the turn; the author works through its
        tools, and the world file is code's again when the turn ends."""
        self.budget_check()
        for name, text in files(self.world, self.settings, self.author_context(), mode).items():
            await runtime.write("/task/" + name, text.encode())
        self.store.event("author_turn", attempt=attempt, mode=mode, day=self.store.state.day)
        segment = await interaction.turn(prompt)
        self.store.trace(interaction.trace)
        if segment.terminated:
            raise ReviewLimit("the author exhausted its interaction's limits")

    async def note(self, runtime, name: str) -> str | None:
        try:
            return (await runtime.read("/task/notes/" + name, max_bytes=2_000_000)).decode()
        except SandboxError:
            return None

    async def keep_notes(self, runtime, attempt: str) -> None:
        """The author's notes as the attempt left them, kept with its world."""
        for name in NOTES:
            if (text := await self.note(runtime, name)) is not None:
                self.store.path(attempt, "notes/" + name).write_text(text)

    async def bring_notes(self, runtime, attempt: str) -> None:
        """The notes an attempt kept, back in the author's workspace: a day starts from where the last one closed."""
        for name in NOTES:
            path = self.store.root / "attempts" / attempt / "notes" / name
            if path.exists():
                await runtime.write("/task/notes/" + name, path.read_bytes())

    async def setup_world(self, agents, runtime, task) -> None:
        """The premise and the organization, as documents, in the author's own session."""
        async with agents.author.interaction(task, runtime=runtime) as interaction:
            await self.premise_and_organization(interaction, runtime, "plan")
        self.store.trace(interaction.trace)
        require_trace(interaction.trace)

    async def plan_world(self, agents, runtime) -> None:
        """The ledger, before day 1: storylines, events and facts, and the author's plan of them."""
        state, cfg = self.store.state, self.settings
        if not self.world.db.execute("SELECT 1 FROM world_meta WHERE key = 'chronological'").fetchone():
            with self.world.trial() as copy:
                start_clock(copy)
            organization = Organization.model_validate_json(json.dumps(state.drafts["organization"]))
            agenda = background_plan(self.world, organization, cfg.activity, cfg.seed)
            state.plans["agenda"] = {"scenes": [s.model_dump(mode="json") for s in agenda]}
            self.store.save()
        attempt = self.store.reserve("plan", cfg.author.plan_attempts)
        task = WorldAuthorTask.create("plan", 0, self.world.path, self.author_context(), attempt)
        count = -(-cfg.tasks.count // cfg.tasks.per_storyline)
        async with agents.author.interaction(task, runtime=runtime) as interaction:
            prompt, errors = plan_prompt(self.world, cfg, state.feedback), []
            for _ in range(3):
                await self.author_step(interaction, runtime, prompt, attempt, "plan")
                planned = self.world.db.execute("SELECT COUNT(*) FROM storylines").fetchone()[0]
                errors = [f"plan the ledger with world_plan ({count} storylines)"] * (planned != count)
                errors += ["write /task/notes/plan.md"] * (
                    not (await self.note(runtime, "plan.md") or "").strip()
                )
                if not errors:
                    break
                prompt = "The plan is not done: " + "; ".join(errors) + ". Finish it, then end your turn."
        self.store.trace(interaction.trace)
        require_trace(interaction.trace)
        if errors:
            return self.reject_structure(attempt, "; ".join(errors))
        await self.keep_notes(runtime, attempt)
        state.phase, state.day, state.restore_point, state.feedback = "day", 1, attempt, ""
        self.store.finish_attempt(True)

    async def write_day(self, agents, runtime) -> None:
        """One calendar day, in a fresh interaction that starts from where the last day closed: the world and the
        author's notes. The day is done when it is closed and the author's recap is written."""
        state, cfg = self.store.state, self.settings
        day, total = state.day, self.world.db.execute("SELECT COUNT(*) FROM calendar").fetchone()[0]
        attempt = self.store.reserve(f"day-{day:02d}", cfg.author.day_attempts)
        self.store.restore(state.restore_point)
        await self.bring_notes(runtime, state.restore_point)
        before = await self.note(runtime, "recap.md")
        issues = [i.model_dump(mode="json") for i in state.issues]
        if issues:
            self.store.event("issues_delivered", attempt=attempt, issues=len(issues))
        task = WorldAuthorTask.create("day", day, self.world.path, self.author_context(), attempt)
        async with agents.author.interaction(task, runtime=runtime) as interaction:
            prompt, errors = day_prompt(self.world, day, issues, state.feedback), []
            for _ in range(3):
                await self.author_step(interaction, runtime, prompt, attempt, "day")
                errors = []
                if today(self.world) == day:
                    errors = close_day(self.world, cfg, day) or ["advance through the night to close it"]
                if await self.note(runtime, "recap.md") in (None, before):
                    errors.append("write /task/notes/recap.md: what happened today and what is open")
                if not errors:
                    break
                prompt = f"Day {day} is not done: " + "; ".join(errors) + ". Finish it, then end your turn."
        self.store.trace(interaction.trace)
        require_trace(interaction.trace)
        if errors:
            return self.reject_structure(attempt, "; ".join(errors))
        await self.keep_notes(runtime, attempt)
        start, end = bounds(self.world, day)
        self.store.event("day_closed", attempt=attempt, day=day, messages=posted(self.world, start, end))
        state.issues, state.restore_point, state.feedback = [], attempt, ""
        self.store.finish_attempt(True)
        if day in cfg.author.review_days and day < total:
            state.phase = "review"  # the closed day's review; a resume runs it, not the day again
        else:
            state.day, state.phase = (day + 1, "day") if day < total else (day, "tasks")
        self.store.save()

    def people_typing(self) -> dict:
        return {
            user_id(p.uuid): p.typing.model_dump(exclude={"id", "messages"}) for p in self.store.state.cast
        }

    async def review_so_far(self, agents) -> None:
        """The judge reviews the world written so far; its issues open the author's next day."""
        state, cfg = self.store.state, self.settings
        attempt = self.store.reserve(f"review-{state.day:02d}", 2)
        payload = review_payload(
            self.world,
            "world",
            [],
            people=self.people_typing(),
            written_through=clock(self.world, present(self.world)),
            ledger=ledger_digest(self.world),
        )
        verdict = await self.review(agents, payload, attempt, {}, "world")
        state.issues = deciding(verdict, cfg.acceptance)
        state.day, state.phase = state.day + 1, "day"
        self.store.finish_attempt(verdict.approved)

    def keep_solves(self, task_id: str, key: str, runs: list) -> None:
        """A task's solver runs, kept with what the task was when they ran."""
        self.store.state.solves[task_id] = {"key": key, "results": [r for r, _ in runs], "traces": [t.id for _, t in runs]}  # fmt: skip

    def saved_trace(self, trace_id: str) -> dict | None:
        """A saved solver trace, as the judge may read it."""
        path = self.store.root / "traces" / f"{trace_id}.json"
        return observable(json.loads(path.read_text())) if path.exists() else None

    async def probe(self, agents, attempt: str) -> dict:
        """The solver tries each task that changed since its last runs, within the world's probe budget; each task's
        solve rate and fewest calls are kept on it, and its runs are kept for the final review."""
        state, cfg = self.store.state, self.settings
        self.refresh_gold()
        keys = {t: self.task_key(t) for (t,) in self.world.db.execute("SELECT id FROM tasks ORDER BY id")}
        n = cfg.author.probe_solves
        due = [t for t, key in keys.items() if state.solves.get(t, {}).get("key") != key]
        due = due[: max(0, cfg.author.probe_budget - state.probe_solves) // n]
        if not due:
            return {}
        results = {}
        for task_id, runs in (await self.solves(agents, due, attempt, n)).items():
            self.keep_solves(task_id, keys[task_id], runs)
            outcomes = [outcome for outcome, _ in runs]
            rate = sum(o["semantic_correctness"] for o in outcomes) / n
            fewest = min((o["calls"] for o in outcomes if o["correct"]), default=None)
            self.world.db.execute(
                "UPDATE tasks SET solve_rate = ?, min_calls = ? WHERE id = ?", (rate, fewest, task_id)
            )
            results[task_id] = {"solve_rate": rate, "tries": [{k: o.get(k) for k in ("correct", "calls", "reason")} for o in outcomes]}  # fmt: skip
        state.probe_solves += n * len(due)
        self.store.event("probe", attempt=attempt, rates={t: r["solve_rate"] for t, r in results.items()})
        self.store.save()
        return results

    async def finish_world(self, agents, runtime) -> None:
        """After the last day, in one interaction: the tasks, the solver's tries and the author's hardening, then the
        final review, whose issues the author fixes until the judge approves or the rounds run out."""
        state, cfg = self.store.state, self.settings
        await self.bring_notes(runtime, state.restore_point)
        final = lambda: f"final-{state.rounds.get('final', 0):02d}"  # noqa: E731
        attempt = self.store.reserve("tasks", cfg.review_rounds.final) if state.phase == "tasks" else final()
        task = WorldAuthorTask.create("tasks", 0, self.world.path, self.author_context(), attempt)
        cells = {tuple(c[:2]) for c in state.quota}
        async with agents.author.interaction(task, runtime=runtime) as interaction:

            async def turn(prompt, label=attempt):
                await self.author_step(interaction, runtime, prompt, label, "tasks")

            if state.phase == "tasks":
                prompt = tasks_prompt(cfg)
                for _ in range(3):
                    await turn(prompt)
                    written = {tuple(r) for r in self.world.db.execute("SELECT category, level FROM tasks")}
                    if not (missing := sorted(cells - written)):
                        break
                    prompt = f"These cells have no task yet: {missing}. Write them with world_add_task, then end your turn."
                if missing:
                    raise ReviewLimit(f"the author wrote no task for {missing}")
                for left in range(cfg.author.probe_rounds, -1, -1):
                    results = await self.probe(agents, attempt)
                    if not results or not left:
                        break
                    await turn(harden_prompt(results, (cfg.author.probe_budget - state.probe_solves) // cfg.author.probe_solves))  # fmt: skip
                await self.keep_notes(runtime, attempt)
                state.phase, state.restore_point = "final", attempt
                self.store.finish_attempt(True)
            while state.phase == "final":
                if (verdict := await self.final_world(agents)).approved:
                    break
                invalid = [{"task_id": r.task_id, "invalid": r.reason} for r in verdict.tasks if not r.valid]
                await turn(
                    fix_prompt(
                        [i.model_dump(mode="json") for i in deciding(verdict, cfg.acceptance)] + invalid
                    ),
                    final(),
                )
        self.store.trace(interaction.trace)
        require_trace(interaction.trace)

    async def final_world(self, agents) -> Verdict:
        """The final review: every stale task solved and judged, and the whole world judged."""
        state, cfg = self.store.state, self.settings
        if broken := self.world.violations(complete=True):
            raise ValueError("final deterministic check failed: " + "; ".join(broken))
        attempt = self.store.reserve("final", cfg.review_rounds.final)
        verdict = await self.assess(agents, attempt)
        if verdict.approved:
            state.reviews["final"] = verdict.model_dump(mode="json")
            state.phase = "done"
            self.store.event("workspace_frozen", tasks=len(state.task_reviews))
        state.last_verdict = None if verdict.approved else verdict  # the next task review checks its issues
        self.store.finish_attempt(verdict.approved)
        return verdict

    async def author_world(self, agents) -> None:
        """One author agent and one VM for the whole world; one interaction per block of work."""
        state = self.store.state
        setup = SynthesizerTask.create(
            synthesizer.premise_context(self.settings, self.used), "setup", self.world.path
        )
        async with agents.author.provision(setup) as runtime:
            if state.phase in ("premise", "organization"):
                await self.setup_world(agents, runtime, setup)
            # Provisioning lays out the setup files again on every resume; the author's later turns never see them.
            await runtime.run(["rm", "-f", *(f"/task/{name}" for name in SETUP_FILES)], {})
            while state.phase == "plan":
                await self.plan_world(agents, runtime)
            while state.phase in ("day", "review"):
                await (
                    self.review_so_far(agents) if state.phase == "review" else self.write_day(agents, runtime)
                )
            if state.phase in ("tasks", "final"):
                await self.finish_world(agents, runtime)

    async def run(self, task, agents):
        return await self.author_world(agents)
