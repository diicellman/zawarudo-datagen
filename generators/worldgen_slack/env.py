"""The generation episode: premise → organization → ledger → storylines built and reviewed → tasks → final review.

What each role reads and how its document is checked lives with the role in `agents/` and in `contracts`; the world
itself is one SQLite file, written only here, through `World.trial()`."""

import asyncio
import json
import random
from collections import defaultdict

import verifiers.v1 as vf
from verifiers.v1.configs.runtime import NetworkPolicyConfig
from verifiers.v1.errors import SandboxError
from worldgen_slack.db import digest
from worldgen_slack.taskset import SolverTask

from .agents import builder, synthesizer, writer
from .agents.builder import BuilderTask
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
    PART_ORDER,
    PHASE_CRITERIA,
    PlanError,
    Gaps,
    Ledger,
    Organization,
    Scene,
    ScenePlan,
    TaskSet,
    Verdict,
    accepted,
    accepted_task,
    background_plan,
    check_plan,
    clock,
    deciding,
    layout,
    organize,
    pick_cast,
    quality,
    quota,
    record_ledger,
    record_tasks,
    user_id,
    write_scene,
)
from .store import ReviewLimit, Store, used_names

LEDGER_ROWS = ("task_facts", "tasks", "fact_relations", "facts", "storylines")
WORLD_ROWS = ("evidence", "message_mentions", "reactions", "scene_messages", "messages", "scenes")
NOTES = ("plan.md", "recap.md")  # the world author's own files in /task/notes
SETUP_FILES = ("input.json", "guide.md", "schemas.json", "schema.sql", "world.sqlite", "premise.json", "organization.json")  # fmt: skip


def require_trace(trace):
    if not trace.ok:
        raise RuntimeError("agent execution failed: " + "; ".join(e.message for e in trace.errors))


def observable(trace):
    """A trace record without hidden reasoning or provider state, for another agent to investigate."""
    record = trace.to_record()
    for node in record["nodes"]:
        node["message"].pop("reasoning_content", None)
        node["message"].pop("provider_state", None)
    return record


class GenerationEnv(vf.Env[PipelineConfig]):
    def __init__(self, settings: Config, store: Store, seeds=None):
        self.settings, self.store, self.seeds = settings, store, seeds
        self.used = used_names(settings.corpus, store.root)
        self.gaps = Gaps(settings.personas.gaps)
        super().__init__(settings.env)

    @property
    def world(self):
        return self.store.world

    async def setup(self, agents):
        for name in ("author", "synthesizer", "builder", "writer", "judge", "solver"):
            getattr(agents, name).trainable = False

    def budget_check(self):
        if self.store.summary("running")["reported_model_cost"] >= self.settings.research_budget_usd:
            raise ReviewLimit("reported model spend reached the budget")

    def calendar(self) -> list[dict]:
        rows = self.world.db.execute("SELECT day, date FROM calendar ORDER BY day").fetchall()
        return [{"day": d, "date": date} for d, date in rows]

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

    # ------------------------------------------------------------------ S0-S2: premise, organization, ledger
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
                raise ReviewLimit(f"the synthesizer proposed no valid premise: {error}") from None
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
                raise ReviewLimit(f"the synthesizer built no valid organization: {error}") from None
            first = False
            state.organization = [r[0] for r in self.world.db.execute("SELECT id FROM channels")]
            state.phase = then
            self.store.save()
            self.store.event("organization_built", channels=len(state.organization))
        return first

    async def synthesize(self, agents):
        state, cfg = self.store.state, self.settings
        context = synthesizer.premise_context(cfg, self.used)
        task = SynthesizerTask.create(context, "synthesis", self.world.path)
        async with agents.synthesizer.provision(task) as runtime:
            async with agents.synthesizer.interaction(task, runtime=runtime) as interaction:
                first = await self.premise_and_organization(interaction, runtime, "ledger")
                while state.phase == "ledger":
                    attempt = self.store.reserve("ledger", cfg.review_rounds.ledger)
                    try:
                        await self.author(
                            interaction, runtime, SynthesizerTask, synthesizer.ledger_context(cfg, state, self.calendar()),
                            attempt, first, self.apply(self.replace_ledger, Ledger, "ledger"),
                        )  # fmt: skip
                    except ValueError as error:
                        self.reject_structure(attempt, str(error))
                        continue
                    finally:
                        first = False
                    verdict = await self.review(
                        agents,
                        self.ledger_payload(),
                        attempt,
                        {"author_trace.json": observable(interaction.trace)},
                    )
                    if verdict.approved:
                        state.reviews["ledger"] = verdict.model_dump(mode="json")
                        state.phase, state.feedback, state.last_verdict = "build", "", None
                    else:
                        state.feedback, state.last_verdict = verdict.model_dump_json(), verdict
                    self.store.finish_attempt(verdict.approved)
            self.store.trace(interaction.trace)
            require_trace(interaction.trace)

    def replace_ledger(self, copy, ledger: Ledger):
        """A new ledger replaces the old. If its storylines, channels and facts are unchanged, only its tasks are
        replaced and the written world stays; otherwise the world is cleared back to the organization."""
        state, old = self.store.state, self.store.state.drafts.get("ledger")
        same = old is not None and {k: old[k] for k in ("storylines", "channels", "facts")} == {
            k: ledger.model_dump(mode="json")[k] for k in ("storylines", "channels", "facts")
        }
        ledger_cells = [c for c in state.quota if self.settings.taxonomy[c[0]].gold == "ledger"]
        ledger_tasks = [t for (t,) in copy.db.execute("SELECT id FROM tasks WHERE gold_source = 'ledger'")]
        marks = ", ".join("?" * len(ledger_tasks))
        if same:
            copy.db.execute(f"DELETE FROM task_facts WHERE task_id IN ({marks})", ledger_tasks)
            copy.db.execute(f"DELETE FROM tasks WHERE id IN ({marks})", ledger_tasks)
            record_tasks(copy, ledger.tasks, self.settings, ledger_cells)
            return
        for table in WORLD_ROWS + LEDGER_ROWS:
            copy.db.execute(f"DELETE FROM {table}")
        marks = ", ".join("?" * len(state.organization))
        copy.db.execute(f"DELETE FROM members WHERE channel_id NOT IN ({marks})", state.organization)
        copy.db.execute(f"DELETE FROM channels WHERE id NOT IN ({marks})", state.organization)
        state.plans, state.built, state.frozen, state.notes, state.task_reviews = {}, [], {}, {}, {}
        # A new ledger makes new storylines: their build allowance starts again.
        state.refunded |= {k: n for k, n in state.rounds.items() if k.startswith("build:")}
        state.reviews = {k: v for k, v in state.reviews.items() if k == "ledger"}
        if old is not None:
            self.store.event("approvals_invalidated", reason="the ledger's facts changed")
        record_ledger(copy, ledger, self.settings, ledger_cells)

    def ledger_payload(self) -> dict:
        state = self.store.state
        people = {user_id(p.uuid): p for p in state.cast}
        rows = self.world.db.execute("SELECT id, real_name, title, profile_json FROM users").fetchall()
        return review_payload(
            self.world,
            "ledger",
            [t for t, in self.world.db.execute("SELECT id FROM tasks WHERE gold_source = 'ledger' ORDER BY id")],
            self.settings.taxonomy,
            premise=state.premise.model_dump(),
            people=[
                {"user_id": r["id"], "name": r["real_name"], "title": r["title"], "team": json.loads(r["profile_json"]).get("Team")}
                | people[r["id"]].model_dump(include={"occupation", "education_level", "bachelors_field", "age", "professional_persona"})
                for r in rows
            ],
            storylines=[dict(r) for r in self.world.db.execute("SELECT * FROM storylines ORDER BY position")],
        )  # fmt: skip

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

    def scope(self, storyline: str | None) -> list[str]:
        """The tasks a review covers: in a storyline's build, the ledger tasks whose facts are all written; in the
        final review, every task."""
        if storyline is None:
            return [t for (t,) in self.world.db.execute("SELECT id FROM tasks ORDER BY id")]
        built = {*self.store.state.built, storyline}
        return [
            t
            for (t,) in self.world.db.execute("SELECT id FROM tasks WHERE gold_source = 'ledger' ORDER BY id")
            if all(
                s in built
                for (s,) in self.world.db.execute(
                    "SELECT f.storyline FROM task_facts tf JOIN facts f ON f.id = tf.fact_id WHERE tf.task_id = ?",
                    (t,),
                )
            )  # fmt: skip
        ]

    async def assess(self, agents, storyline, attempt, changed, author=None):
        """Solve and judge every task in scope whose review is stale, and judge the scenes written since the last
        approval. `storyline=None` is the final review: every task and the whole world, unfocused."""
        state, n = self.store.state, self.settings.solves_per_task
        self.refresh_gold()
        keys = {t: self.task_key(t) for t in self.scope(storyline)}
        due = [t for t, key in keys.items() if state.task_reviews.get(t, {}).get("key") != key]
        previous = state.last_verdict.issues if state.last_verdict else []
        people = {user_id(p.uuid): p.typing.model_dump(exclude={"id", "messages"}) for p in state.cast}
        extra = {"storyline": storyline} if storyline else {}

        async def check():
            runs = await self.solves(agents, due, attempt, n)
            payload = review_payload(
                self.world, "task", due, self.settings.taxonomy, **extra,
                solves=[result for task_runs in runs.values() for result, _ in task_runs],
                previous_issues=[i.model_dump(mode="json") for i in previous if set(i.task_ids) & runs.keys()],
            )  # fmt: skip
            files = {f"solver_{t}_{k}.json": observable(trace) for t, task_runs in runs.items() for k, (_, trace) in enumerate(task_runs, 1)}  # fmt: skip
            return runs, payload, await self.review(agents, payload, attempt, files, "tasks")

        workspace = [i.model_dump(mode="json") for i in previous if i.artifact == "workspace"]
        world_payload = None
        if storyline is None:
            rendered = {"ledger": ledger_digest(self.world)} if self.settings.author.enabled else {}
            world_payload = review_payload(self.world, "world", [], people=people, **rendered)
        elif changed or workspace:
            marks = ", ".join("?" * len(changed))
            messages = [m for m, in self.world.db.execute(f"SELECT message_id FROM scene_messages WHERE scene_id IN ({marks}) ORDER BY message_id", changed)]  # fmt: skip
            world_payload = review_payload(self.world, "world", [], **extra, people=people, previous_issues=workspace, changed_messages=messages)  # fmt: skip
        files = {"author_trace.json": observable(author)} if author is not None else {}
        async with asyncio.TaskGroup() as group:
            world = (
                group.create_task(self.review(agents, world_payload, attempt, files, "world"))
                if world_payload
                else None
            )
            checked = group.create_task(check()) if due else None
        runs, payload, judged = checked.result() if checked else ({}, None, None)
        verdicts = [v for v in (judged, world.result() if world else None) if v is not None]
        rejected = [v for v in verdicts if not v.approved]
        for task_id, task_runs in runs.items():
            results = [result for result, _ in task_runs]
            rate = sum(r["semantic_correctness"] for r in results) / n
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
            summary=" ".join(v.summary for v in rejected or verdicts) or "No task or scene needed review.",
        )
        self.store.artifact(attempt, "verdict", merged.model_dump(mode="json"))
        self.store.artifact(
            attempt,
            "acceptance",
            {"approved": merged.approved, "reviewed_tasks": list(runs), "changed_scenes": changed},
        )
        return merged

    # ------------------------------------------------------------------ S3: storylines
    def scene_key(self, storyline: str, scene: Scene) -> str:
        """What a scene is written from: its plan and its beats' facts."""
        facts = [
            dict(self.world.db.execute("SELECT * FROM facts WHERE id = ?", (b.fact,)).fetchone())
            for b in scene.beats
        ]
        return digest([storyline, scene.model_dump(mode="json"), facts])

    async def compose(self, agents, scene_id, prompt, accept):
        """The writer's chat; a seam for the scripted check."""
        written, errors, trace = await writer.compose(agents.writer, prompt, accept)
        trace.info["scene_id"] = scene_id
        self.store.trace(trace)
        return written, errors

    async def write(
        self, agents, storyline: str | None, plan: ScenePlan, dropped: list | None = None
    ) -> list[str]:
        """Write the plan's changed scenes in time order, each into a trial of the world; returns their ids. With
        `dropped` (background conversations), a scene its writer cannot get right is listed there and skipped."""
        state, changed = self.store.state, []
        order = sorted(enumerate(plan.scenes), key=lambda p: (p[1].day, PART_ORDER.index(p[1].part), p[0]))
        kept = {s.id for s in plan.scenes}
        with self.world.trial() as copy:
            for (scene_id,) in copy.db.execute(
                "SELECT id FROM scenes WHERE storyline IS ?", (storyline,)
            ).fetchall():
                if scene_id not in kept:
                    copy.clear_scene(scene_id)
        profiles = {user_id(p.uuid): p for p in state.cast}
        for _, scene in order:
            key = self.scene_key(storyline, scene)
            row = self.world.db.execute(
                "SELECT key, plan_json FROM scenes WHERE id = ?", (scene.id,)
            ).fetchone()
            if row is not None and row["key"] == key:
                continue
            previous = None
            if row is not None:
                previous = [m for m, in self.world.db.execute("SELECT m.text FROM scene_messages sm JOIN messages m ON m.id = sm.message_id WHERE sm.scene_id = ? ORDER BY m.ts_us", (scene.id,))]  # fmt: skip
            with self.world.trial() as copy:
                copy.clear_scene(scene.id)
            typing = {u: p.typing for u, p in profiles.items()}
            shape = layout(self.world, scene, self.settings.activity, random.Random(digest([self.settings.seed, scene.id, "layout"])), typing)  # fmt: skip
            prompt = writer.brief(
                self.world, scene, state.premise.model_dump(), profiles,
                writer.excerpts(self.seeds, f"{self.settings.seed}:{scene.id}"), self.settings.language, previous,
                shape, storyline is None,
            )  # fmt: skip
            rng = random.Random(digest([self.settings.seed, scene.id, key]))

            def accept(written, scene=scene, key=key, rng=rng, shape=shape):
                with self.world.trial() as copy:
                    write_scene(
                        copy, storyline, scene, written, key, random.Random(rng.random()), self.gaps, shape
                    )

            try:
                written, errors = await self.compose(agents, scene.id, prompt, accept)
            except PlanError as error:
                if dropped is None:
                    raise
                errors = [str(error)]
            if errors and dropped is not None:
                dropped.append(scene.id)
                continue
            if errors:
                raise ValueError(f"scene {scene.id} could not be written: {'; '.join(errors)}")
            state.notes.pop(scene.id, None)
            changed.append(scene.id)
        return changed

    async def write_background(self, agents):
        """S3.5: the company's everyday conversations. Code places them by the activity targets and draws each
        one's kind from its conversation's routines; the writer writes them like scenes without beats. One its
        writer cannot get right is dropped; the final review judges the rest with the whole world."""
        state, cfg = self.store.state, self.settings
        if "background" not in state.plans:
            organization = Organization.model_validate_json(json.dumps(state.drafts["organization"]))
            scenes = background_plan(self.world, organization, cfg.activity, cfg.seed)
            state.plans["background"] = {"scenes": [s.model_dump(mode="json") for s in scenes]}
            self.store.save()
            self.store.event(
                "background_planned",
                conversations=len(scenes),
                messages=sum(s.length for s in scenes),
                places=len({s.channel_id for s in scenes}),
            )
        notes = self.repairs("background")
        scenes = [Scene.model_validate_json(json.dumps(s)) for s in state.plans["background"]["scenes"]]
        scenes = [s.model_copy(update={"revision_note": notes.get(s.id, s.revision_note)}) for s in scenes]
        state.plans["background"]["scenes"] = [s.model_dump(mode="json") for s in scenes]
        dropped = []
        if scenes:
            await self.write(agents, None, ScenePlan(scenes=scenes), dropped)
        if dropped:
            self.store.event("background_dropped", scene_ids=dropped)
        workspace_tasks = self.world.db.execute(
            "SELECT 1 FROM tasks WHERE gold_source <> 'ledger'"
        ).fetchone()
        state.phase = "final" if workspace_tasks else "tasks"
        self.store.save()

    def scenes_of(self, issue) -> set[str]:
        marks = ", ".join("?" * len(issue.message_ids))
        return {s for s, in self.world.db.execute(f"SELECT scene_id FROM scene_messages WHERE message_id IN ({marks})", issue.message_ids)}  # fmt: skip

    def notes(self, issues) -> dict[str, str]:
        notes = defaultdict(list)
        for issue in issues:
            for scene_id in self.scenes_of(issue):
                notes[scene_id].append(f"{issue.defect} Requested change: {issue.requested_change}")
        return {scene_id: " ".join(n) for scene_id, n in notes.items()}

    def repairs(self, plan: str) -> dict[str, str]:
        """Scene → the judge's words still waiting for this plan's scenes. Every routed rejection notes its workspace
        issues on the scenes they name; writing a scene consumes its note."""
        planned = {s["id"] for s in (self.store.state.plans.get(plan) or {}).get("scenes", [])}
        return {s: n for s, n in self.store.state.notes.items() if s in planned}

    async def build_storyline(self, agents, storyline: dict):
        state, cfg = self.store.state, self.settings
        budget = "build:" + storyline["id"]
        context = builder.scenes_context(cfg, state, storyline, self.calendar(), sorted(state.frozen))
        async with agents.builder.provision(BuilderTask.create(context, budget, self.world.path)) as runtime:
            while state.phase == "build" and storyline["id"] not in state.built:
                attempt = self.store.reserve(budget, cfg.review_rounds.build)
                notes = self.repairs(storyline["id"])
                frozen = {s["id"]: Scene.model_validate_json(json.dumps(s)) for s in (state.plans.get(storyline["id"]) or {}).get("scenes", []) if s["id"] in state.frozen}  # fmt: skip
                route = "repair" if notes else "plan"
                self.store.event("attempt_routed", attempt=attempt, route=route, scene_ids=sorted(notes))
                context = builder.scenes_context(cfg, state, storyline, self.calendar(), sorted(frozen))
                task = BuilderTask.create(context, attempt, self.world.path)
                async with agents.builder.interaction(task, runtime=runtime) as interaction:
                    try:
                        if notes:
                            plan = ScenePlan.model_validate(state.plans[storyline["id"]])
                            for scene in plan.scenes:
                                scene.revision_note = notes.get(scene.id, scene.revision_note)
                        else:
                            plan = await self.author(
                                interaction, runtime, BuilderTask, context, attempt, True,
                                lambda raw: check_plan(self.world, ScenePlan.model_validate_json(raw), storyline["id"], frozen),
                            )  # fmt: skip
                        state.plans[storyline["id"]] = plan.model_dump(mode="json")
                        changed = await self.write(agents, storyline["id"], plan)
                        if broken := self.world.violations():
                            raise ValueError("; ".join(broken))
                    except ValueError as error:  # PlanError included: the builder re-plans with the reason
                        self.reject_structure(attempt, str(error))
                    else:
                        verdict = await self.assess(
                            agents, storyline["id"], attempt, changed, interaction.trace
                        )
                        self.settle(storyline["id"], verdict)
                self.store.trace(interaction.trace)
                require_trace(interaction.trace)

    def settle(self, storyline: str | None, verdict: Verdict):
        state = self.store.state
        if verdict.approved:
            if storyline is not None:
                state.built.append(storyline)
                for scene_id, key in self.world.db.execute(
                    "SELECT id, key FROM scenes WHERE storyline = ?", (storyline,)
                ).fetchall():
                    state.frozen[scene_id] = key
            state.feedback, state.last_verdict = "", None
        else:
            self.route_rejection(verdict)
        self.store.finish_attempt(verdict.approved)

    def route_rejection(self, verdict: Verdict):
        """Ledger issues, and invalid or faulted ledger tasks, send the run back to the ledger; other task issues to
        the tasks phase; the rest to the build. Whatever the route, workspace issues become notes on the scenes they
        name, whose storylines are built again; one that names no message sends the last storyline back."""
        state = self.store.state
        state.feedback, state.last_verdict = verdict.model_dump_json(), verdict
        issues = deciding(verdict, self.settings.acceptance)
        workspace = [i for i in issues if i.artifact == "workspace"]
        state.notes |= self.notes(workspace)
        marks = ", ".join("?" * len(state.notes))
        rebuilt = {s for s, in self.world.db.execute(f"SELECT storyline FROM scenes WHERE id IN ({marks})", list(state.notes))}  # fmt: skip
        if any(not self.scenes_of(i) for i in workspace) and state.built:
            rebuilt.add(state.built[-1])
        state.built = [s for s in state.built if s not in rebuilt]
        ledger_tasks = {
            t for (t,) in self.world.db.execute("SELECT id FROM tasks WHERE gold_source = 'ledger'")
        }
        faulted = {r.task_id for r in verdict.tasks if not r.valid}
        faulted |= {t for i in issues if i.artifact == "tasks" for t in i.task_ids}
        if any(i.artifact == "ledger" for i in issues) or faulted & ledger_tasks:
            state.phase = "ledger"
        elif faulted or any(i.artifact == "tasks" for i in issues):
            state.phase = "tasks"
        else:
            state.phase = "build"
        self.store.save()

    # ------------------------------------------------------------------ S4: tasks on the finished world, final review
    async def author_tasks(self, agents):
        state, cfg = self.store.state, self.settings
        cells = [c for c in state.quota if cfg.taxonomy[c[0]].gold != "ledger"]
        questions = [
            q for (q,) in self.world.db.execute("SELECT question FROM tasks WHERE gold_source = 'ledger'")
        ]
        context = synthesizer.tasks_context(cfg, state, questions)
        attempt = self.store.reserve("tasks", cfg.review_rounds.final)

        def replace(copy, document: TaskSet):
            stale = [t for (t,) in copy.db.execute("SELECT id FROM tasks WHERE gold_source <> 'ledger'")]
            marks = ", ".join("?" * len(stale))
            copy.db.execute(f"DELETE FROM task_facts WHERE task_id IN ({marks})", stale)
            copy.db.execute(f"DELETE FROM tasks WHERE id IN ({marks})", stale)
            record_tasks(copy, document.tasks, cfg, cells)

        task = SynthesizerTask.create(context, attempt, self.world.path)
        async with agents.synthesizer.provision(task) as runtime:
            async with agents.synthesizer.interaction(task, runtime=runtime) as interaction:
                try:
                    await self.author(
                        interaction,
                        runtime,
                        SynthesizerTask,
                        context,
                        attempt,
                        True,
                        self.apply(replace, TaskSet, "tasks"),
                    )
                except ValueError as error:
                    self.reject_structure(attempt, str(error))
                else:
                    state.phase, state.feedback = "build", ""  # pending scene notes, then the final review
                    self.store.finish_attempt(True)
            self.store.trace(interaction.trace)
            require_trace(interaction.trace)

    async def final_review(self, agents):
        state, cfg = self.store.state, self.settings
        if broken := self.world.violations(complete=True):
            raise ValueError("final deterministic check failed: " + "; ".join(broken))
        attempt = self.store.reserve("final", cfg.review_rounds.final)
        verdict = await self.assess(agents, None, attempt, [])
        if verdict.approved:
            state.reviews["final"] = verdict.model_dump(mode="json")
            state.phase = "done"
            self.store.event("workspace_frozen", tasks=len(state.task_reviews))
        self.settle(None, verdict)

    # ------------------------------------------------------------------ v7: one author writes the world in time order
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
        """S0-S1, as v6 writes them: the premise and the organization, in the author's own session."""
        async with agents.author.interaction(task, runtime=runtime) as interaction:
            await self.premise_and_organization(interaction, runtime, "plan")
        self.store.trace(interaction.trace)
        require_trace(interaction.trace)
        await runtime.run(["rm", "-f", *(f"/task/{name}" for name in SETUP_FILES)], {})

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
        attempt = self.store.reserve("plan", cfg.review_rounds.ledger)
        task = WorldAuthorTask.create("plan", 0, self.world.path, self.author_context(), attempt)
        count = -(-cfg.tasks.count // cfg.tasks.per_storyline)
        async with agents.author.interaction(task, runtime=runtime) as interaction:
            prompt, errors = plan_prompt(self.world, cfg), []
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
        state.phase, state.day, state.restore_point = "day", 1, attempt
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
            prompt, errors = day_prompt(self.world, day, issues), []
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
        state.issues, state.restore_point = [], attempt
        self.store.finish_attempt(True)
        if day in cfg.author.review_days and day < total:
            await self.review_so_far(agents)
        state.day, state.phase = (day + 1, "day") if day < total else (day, "tasks")
        self.store.save()

    def people_typing(self) -> dict:
        return {
            user_id(p.uuid): p.typing.model_dump(exclude={"id", "messages"}) for p in self.store.state.cast
        }

    async def review_so_far(self, agents) -> None:
        """The judge reviews the world written so far; its issues open the author's next day."""
        state, cfg = self.store.state, self.settings
        attempt = self.store.reserve("review", len(cfg.author.review_days))
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
        self.store.finish_attempt(verdict.approved)

    async def probe(self, agents, attempt: str) -> dict:
        """The solver tries each task that changed since its last try, within the world's probe budget; each task's
        solve rate and fewest calls are kept on it."""
        state, cfg = self.store.state, self.settings
        self.refresh_gold()
        keys = {t: self.task_key(t) for (t,) in self.world.db.execute("SELECT id FROM tasks ORDER BY id")}
        n = cfg.author.probe_solves
        due = [t for t, key in keys.items() if state.probed.get(t) != key]
        due = due[: max(0, cfg.author.probe_budget - state.probe_solves) // n]
        if not due:
            return {}
        results = {}
        for task_id, runs in (await self.solves(agents, due, attempt, n)).items():
            outcomes = [outcome for outcome, _ in runs]
            rate = sum(o["semantic_correctness"] for o in outcomes) / n
            fewest = min((o["calls"] for o in outcomes if o["correct"]), default=None)
            self.world.db.execute(
                "UPDATE tasks SET solve_rate = ?, min_calls = ? WHERE id = ?", (rate, fewest, task_id)
            )
            results[task_id] = {"solve_rate": rate, "tries": [{k: o.get(k) for k in ("correct", "calls", "reason")} for o in outcomes]}  # fmt: skip
            state.probed[task_id] = keys[task_id]
        state.probe_solves += n * len(due)
        self.store.event("probe", attempt=attempt, rates={t: r["solve_rate"] for t, r in results.items()})
        self.store.save()
        return results

    async def finish_world(self, agents, runtime) -> None:
        """After the last day, in one interaction: the tasks, the solver's tries and the author's hardening, then the
        final review, whose issues the author fixes until the judge approves or the rounds run out."""
        state, cfg = self.store.state, self.settings
        attempt = self.store.reserve("tasks", cfg.review_rounds.final)
        await self.bring_notes(runtime, state.restore_point)
        task = WorldAuthorTask.create("tasks", 0, self.world.path, self.author_context(), attempt)
        cells = {tuple(c[:2]) for c in state.quota}
        async with agents.author.interaction(task, runtime=runtime) as interaction:

            async def turn(prompt):
                await self.author_step(interaction, runtime, prompt, attempt, "tasks")

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
                state.phase = "final"
                self.store.save()
            while state.phase == "final":
                if (verdict := await self.final_world(agents)).approved:
                    break
                invalid = [{"task_id": r.task_id, "invalid": r.reason} for r in verdict.tasks if not r.valid]
                await turn(
                    fix_prompt(
                        [i.model_dump(mode="json") for i in deciding(verdict, cfg.acceptance)] + invalid
                    )
                )
        self.store.trace(interaction.trace)
        require_trace(interaction.trace)

    async def final_world(self, agents) -> Verdict:
        """The final review: every stale task solved and judged, and the whole world judged."""
        state, cfg = self.store.state, self.settings
        if broken := self.world.violations(complete=True):
            raise ValueError("final deterministic check failed: " + "; ".join(broken))
        attempt = self.store.reserve("final", cfg.review_rounds.final)
        verdict = await self.assess(agents, None, attempt, [])
        if verdict.approved:
            state.reviews["final"] = verdict.model_dump(mode="json")
            state.phase = "done"
            self.store.event("workspace_frozen", tasks=len(state.task_reviews))
        state.last_verdict = None if verdict.approved else verdict  # the next task review checks its issues
        self.store.finish_attempt(verdict.approved)
        return verdict

    async def author_world(self, agents) -> None:
        """v7: one author agent and one VM for the whole world; one interaction per block of work."""
        state = self.store.state
        setup = SynthesizerTask.create(
            synthesizer.premise_context(self.settings, self.used), "setup", self.world.path
        )
        async with agents.author.provision(setup) as runtime:
            if state.phase in ("premise", "organization"):
                await self.setup_world(agents, runtime, setup)
            while state.phase == "plan":
                await self.plan_world(agents, runtime)
            while state.phase == "day":
                await self.write_day(agents, runtime)
            if state.phase in ("tasks", "final"):
                await self.finish_world(agents, runtime)

    async def run(self, task, agents):
        if self.settings.author.enabled:
            return await self.author_world(agents)
        state = self.store.state
        while state.phase != "done":
            self.budget_check()
            if state.phase in ("premise", "organization", "ledger"):
                await self.synthesize(agents)
            elif state.phase == "build":
                storylines = [
                    dict(r) for r in self.world.db.execute("SELECT * FROM storylines ORDER BY position")
                ]
                storyline = next((s for s in storylines if s["id"] not in state.built), None)
                if storyline is None:
                    state.phase = "background"
                    self.store.save()
                else:
                    await self.build_storyline(agents, storyline)
            elif state.phase == "background":
                await self.write_background(agents)
            elif state.phase == "tasks":
                await self.author_tasks(agents)
            elif state.phase == "final":
                await self.final_review(agents)
