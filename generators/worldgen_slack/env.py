"""Catalog → reviewed build groups → final review → independent solves."""

import asyncio
import json
from collections import defaultdict
import verifiers.v1 as vf
from verifiers.v1.errors import SandboxError
from worldgen_slack.dataset import PrivateAnswer, atomic_json
from worldgen_slack.slack.api import digest
from worldgen_slack.taskset import SolverTask
from .agents.synthesizer import SynthesizerTask
from .agents.builder import BuilderTask
from .agents.judge import JudgeTask, review_payload
from .agents.writer import brief, compose, excerpts, observed, visible
from .config import PipelineConfig
from .contracts import (
    BindOutput,
    Candidate,
    Catalog,
    Plan,
    Premises,
    SceneRecord,
    Verdict,
    accepted,
    assemble,
    check_plan,
    first_mentions,
    pick_premise,
    quality,
    timed,
    validate_candidate,
)
from .store import ReviewLimit, used_names


def require_trace(trace):
    if not trace.ok:
        raise RuntimeError("agent execution failed: " + "; ".join(e.message for e in trace.errors))


def reference_for(task, candidate):
    binding = next(b for b in candidate.bindings if b.task_id == task.id)
    return PrivateAnswer(
        answer=task.answer,
        message_ids=sorted({m for c in binding.claims for m in c.message_ids}),
        user_ids=sorted({u for c in binding.claims for u in c.user_ids}),
    )


class GenerationEnv(vf.Env[PipelineConfig]):
    def __init__(self, settings, store, seeds=None):
        self.settings, self.store = settings, store
        if (settings.seed_data is None) != (seeds is None):
            raise ValueError("seed configuration and loaded packet must agree")
        self.seeds = seeds
        self.used = used_names(settings.corpus, store.root)
        super().__init__(settings.env)

    async def setup(self, agents):
        for name in ("synthesizer", "builder", "writer", "judge", "solver"):
            getattr(agents, name).trainable = False

    def budget_check(self):
        if self.store.summary("running")["reported_model_cost"] >= self.settings.research_budget_usd:
            raise ReviewLimit("reported model spend reached the budget")

    async def author_turn(self, interaction, runtime, context, attempt, first):
        self.budget_check()
        self.store.event("author_started", attempt=attempt)
        await runtime.write("/task/input.json", json.dumps(context).encode())
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
        interaction.trace.info.setdefault("candidates", []).append(
            {"candidate_id": attempt, "input_hash": digest(context)}
        )
        self.store.trace(interaction.trace)
        return raw

    async def author(self, interaction, runtime, context, attempt, first, parse):
        """Deterministic errors go back to the author twice within the attempt before it is rejected."""
        errors = []
        for _ in range(3):
            raw = await self.author_turn(interaction, runtime, context, attempt, first)
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

    async def review(self, agents, payload, attempt, author_trace=None):
        self.budget_check()
        self.store.artifact(attempt, "review_input", payload)
        self.store.event("review_started", attempt=attempt, task_ids=[t["id"] for t in payload["tasks"]])
        record = None
        if author_trace is not None:
            record = author_trace.to_record()
            for node in record["nodes"]:
                node["message"].pop("reasoning_content", None)
                node["message"].pop("provider_state", None)
            self.store.artifact(attempt, "author_trace", record)
        # One fresh rerun absorbs a malformed verdict; the review round is still spent only once.
        for _ in range(2):
            task = JudgeTask.create(payload, candidate_id=attempt)
            if record is not None:
                task.config.author_trace = json.dumps(record)
            trace = await agents.judge.run(task)
            self.store.trace(trace)
            if trace.ok:
                break
        require_trace(trace)
        verdict = Verdict.model_validate(trace.info["verdict"])
        self.store.artifact(attempt, "verdict", verdict.model_dump(mode="json"))
        if payload["phase"] == "world":
            verdict = verdict.model_copy(
                update={"approved": accepted(verdict, payload, self.settings.acceptance)}
            )
            self.store.artifact(
                attempt, "acceptance", {"approved": verdict.approved, "quality": quality(verdict)}
            )
        return verdict

    def record_quality(self, trace, verdict, name):
        value = float(verdict.approved) if name == "catalog_quality" else quality(verdict)
        trace.record_reward(name, value)
        trace.info.setdefault("quality_reviews", []).append(
            {"candidate_id": self.store.state.active_attempt, "value": value}
        )
        self.store.trace(trace)

    def route_rejection(self, verdict, budget):
        state = self.store.state
        state.feedback, state.last_verdict = verdict.model_dump_json(), verdict
        group_id = budget.partition(":")[2]
        if group_id in state.built_groups:
            state.built_groups.remove(group_id)
        if any(issue.owner == "synthesizer" for issue in verdict.issues):
            self.store.invalidate("catalog repair requested")
            state.phase = "catalog"
        else:
            state.phase = "build"

    def catalog_context(self):
        state, cfg = self.store.state, self.settings
        return {
            "phase": "catalog",
            "sector": cfg.sector,
            "task_count": cfg.task_count,
            "group_size": cfg.group_size,
            "language": cfg.language,
            "premise": state.premise.model_dump() if state.premise else None,
            "used_names": self.used,
            "workspace_id": state.catalog.workspace_id if state.catalog else f"workspace-{cfg.seed}",
            "feedback": state.feedback,
            "previous_output": state.catalog.model_dump(mode="json") if state.catalog else None,
            "instructions": "Create the whole catalog. On repair preserve task IDs, group IDs, and assignments.",
        }

    async def choose_premise(self, interaction, runtime):
        """The synthesizer proposes distinct premises; the run seed, not the model, picks one."""
        state, cfg = self.store.state, self.settings
        context = {
            "phase": "premise",
            "sector": cfg.sector,
            "language": cfg.language,
            "premise_count": cfg.premise_count,
            "used_names": self.used,
            "workspace_id": f"workspace-{cfg.seed}",
            "feedback": "",
        }

        def parse(raw):
            premises = Premises.model_validate_json(raw)
            return pick_premise(premises, cfg.premise_count, self.used["companies"], cfg.seed)

        try:
            state.premise = await self.author(interaction, runtime, context, "premise", True, parse)
        except ValueError as error:
            raise ReviewLimit(f"synthesizer proposed no valid premise set: {error}") from None
        self.store.save()
        self.store.event("premise_selected", company=state.premise.company)

    def check_catalog(self, raw, context):
        state, cfg = self.store.state, self.settings
        catalog = Catalog.model_validate_json(raw)
        if len(catalog.tasks) != cfg.task_count or catalog.sector.casefold() != cfg.sector.casefold():
            raise ValueError("catalog must match requested task count and sector")
        if catalog.workspace_id != context["workspace_id"]:
            raise ValueError("workspace ID must remain stable")
        if len(catalog.groups) != (cfg.task_count + cfg.group_size - 1) // cfg.group_size:
            raise ValueError("catalog has the wrong number of groups")
        if any(sum(t.group_id == g.id for t in catalog.tasks) > cfg.group_size for g in catalog.groups):
            raise ValueError("group exceeds configured task count")
        if state.catalog and {(t.id, t.group_id) for t in catalog.tasks} != {
            (t.id, t.group_id) for t in state.catalog.tasks
        }:
            raise ValueError("catalog repair must preserve task IDs and group assignments")
        if not catalog.personas or catalog.company != state.premise.company:
            raise ValueError("catalog must use premise.company and give every person a persona")
        if reused := {p.name for p in catalog.people} & set(self.used["people"]):
            raise ValueError(f"people reuse names from used_names: {sorted(reused)}")
        return catalog

    async def synthesize(self, agents):
        state, cfg = self.store.state, self.settings
        task = SynthesizerTask.create(self.catalog_context(), "catalog")
        async with agents.synthesizer.provision(task) as runtime:
            async with agents.synthesizer.interaction(task, runtime=runtime) as interaction:
                first = True
                if state.premise is None:
                    await self.choose_premise(interaction, runtime)
                    first = False
                while state.phase == "catalog":
                    attempt = self.store.reserve("catalog", cfg.review_rounds.catalog)
                    context = self.catalog_context()
                    try:
                        catalog = await self.author(
                            interaction,
                            runtime,
                            context,
                            attempt,
                            first,
                            lambda raw, context=context: self.check_catalog(raw, context),
                        )
                    except ValueError as error:
                        self.reject_structure(attempt, str(error))
                        continue
                    finally:
                        first = False
                    verdict = await self.review(
                        agents,
                        review_payload(catalog, None, [t.id for t in catalog.tasks], "catalog"),
                        attempt,
                        interaction.trace,
                    )
                    self.record_quality(interaction.trace, verdict, "catalog_quality")
                    state.catalog = catalog
                    state.feedback = verdict.model_dump_json()
                    if verdict.approved:
                        if state.candidate is not None:
                            self.store.invalidate("catalog changed; dependent reviews must be repeated")
                        state.reviews["catalog"] = verdict.model_dump(mode="json")
                        state.phase, state.feedback, state.last_verdict = "build", "", None
                    self.store.finish_attempt(verdict.approved)
            self.store.trace(interaction.trace)
            require_trace(interaction.trace)

    def reject_structure(self, attempt, error):
        self.store.state.feedback, self.store.state.last_verdict = error[:8000], None
        self.store.artifact(attempt, "validation", {"ok": False, "errors": [error]})
        self.store.finish_attempt(False)

    def group_scope(self, group):
        state = self.store.state
        included = {*state.built_groups, group.id}
        return {
            "workspace_id": state.catalog.workspace_id,
            "group_id": group.id,
            "catalog": state.catalog.model_dump(mode="json"),
            "required_task_ids": [t.id for t in state.catalog.tasks if t.group_id in included],
            "feedback": state.feedback,
        }

    def plan_context(self, group):
        state, cfg = self.store.state, self.settings
        per_group = cfg.messages_per_task * sum(t.group_id == group.id for t in state.catalog.tasks)
        return {
            "phase": "plan",
            **self.group_scope(group),
            "language": cfg.language,
            "premise": state.premise.model_dump(),
            "new_messages_hint": f"{per_group * 6 // 10}-{per_group * 13 // 10}",
            "timed_facts": sorted(timed(state.catalog)),
            "previous_output": state.plan.model_dump(mode="json") if state.plan else None,
        }

    def scene_key(self, plan, scene):
        """Digest of what a scene is written from, plus the minutes its first-stated timed facts must land in."""
        catalog = self.store.state.catalog
        facts = {f.id: f for f in catalog.facts}
        personas = {p.id: p for p in catalog.personas}
        first = first_mentions(plan)
        minutes = timed(catalog)
        timely = {f: minutes[f] for f, where in first.items() if where == scene.id and f in minutes}
        key = digest(
            [
                scene.model_dump(mode="json"),
                [facts[b.fact_id].model_dump(mode="json") for b in scene.beats],
                [personas[i].model_dump(mode="json") for i in scene.participant_ids],
                timely,
            ]
        )
        return key, timely

    def current(self):
        """Whether every written scene still matches the catalog, so the candidate can be reviewed again."""
        state = self.store.state
        return all(
            (record := state.scenes.get(s.id)) and record.key == self.scene_key(state.plan, s)[0]
            for s in state.plan.scenes
        )

    async def write_scene(self, agents, plan, scene, seen, previous, timely):
        state = self.store.state
        prompt = brief(
            state.catalog,
            state.premise,
            plan,
            scene,
            seen,
            excerpts(self.seeds, f"{self.settings.seed}:{scene.id}"),
            previous,
            self.settings.language,
            timely,
        )
        written, errors, trace = await compose(agents.writer, prompt, scene, state.catalog, timely)
        self.store.trace(trace)
        if errors:
            raise ValueError(f"scene {scene.id} could not be written: {'; '.join(errors)}")
        return written

    async def write(self, agents, plan):
        """Write changed scenes in waves: each waits only for the earlier scenes its participants could have seen."""
        state = self.store.state
        conversations = {c.id: c for c in plan.conversations}
        done = {s.id: asyncio.Event() for s in plan.scenes}

        async def one(scene):
            for other in plan.scenes:
                if visible(other, scene, conversations):
                    await done[other.id].wait()
            (key, timely), previous = self.scene_key(plan, scene), state.scenes.get(scene.id)
            if previous is None or previous.key != key:
                seen = observed(state.catalog, plan, {k: v.scene for k, v in state.scenes.items()}, scene)
                written = await self.write_scene(agents, plan, scene, seen, previous, timely)
                state.scenes[scene.id] = SceneRecord(key=key, scene=written)
                self.store.save()
            done[scene.id].set()

        self.budget_check()
        try:
            async with asyncio.TaskGroup() as group:
                for scene in sorted(plan.scenes, key=lambda s: (s.start, s.id)):
                    group.create_task(one(scene))
        except* ValueError as failures:
            raise ValueError("; ".join(str(e) for e in failures.exceptions)) from None
        state.scenes = {s.id: state.scenes[s.id] for s in plan.scenes}
        self.store.save()
        return assemble(state.catalog, plan, {k: v.scene for k, v in state.scenes.items()})

    def repairs(self):
        """Scene → the judge's words, when every issue of the last rejection points at messages of known scenes."""
        state = self.store.state
        last = state.last_verdict
        if (
            last is None
            or state.plan is None
            or any(i.artifact != "workspace" or not i.message_ids for i in last.issues)
        ):
            return {}
        _, index = assemble(state.catalog, state.plan, {k: v.scene for k, v in state.scenes.items()})
        owner = {m: s for s, ids in index["scene_messages"].items() for m in ids}
        notes = defaultdict(list)
        for issue in last.issues:
            if not (scenes := {owner[m] for m in issue.message_ids if m in owner}):
                return {}
            for scene_id in scenes:
                notes[scene_id].append(f"{issue.defect} Requested change: {issue.requested_change}")
        return {scene_id: " ".join(n) for scene_id, n in notes.items()}

    async def construct(self, agents, interaction, runtime, group, attempt, first, plan=None):
        """Plan (unless a repaired plan is given), write, assemble, bind. Returns a checked candidate, or None
        after a structural rejection."""
        state = self.store.state
        required = self.group_scope(group)["required_task_ids"]
        if plan is None:
            try:
                plan = await self.author(
                    interaction,
                    runtime,
                    self.plan_context(group),
                    attempt,
                    first,
                    lambda raw: check_plan(state.catalog, Plan.model_validate_json(raw)),
                )
            except ValueError as error:
                self.reject_structure(attempt, str(error))
                return None
            first = False
        state.plan = plan
        scenes = {s.id: s for s in plan.scenes}
        for round in range(3):
            try:
                world, index = await self.write(agents, plan)
            except ValueError as error:
                self.reject_structure(attempt, str(error))
                return None

            def bind(raw, world=world, last=round == 2):
                bound = BindOutput.model_validate_json(raw)
                if last or not any(r.scene_id in scenes for r in bound.rewrites):
                    report = validate_candidate(
                        state.catalog, Candidate(snapshot=world, bindings=bound.bindings), required
                    )
                    if not report["ok"]:
                        raise ValueError(json.dumps(report["errors"]))
                return bound

            context = {
                "phase": "bind",
                **self.group_scope(group),
                "workspace": world.model_dump(mode="json"),
                **index,
                "previous_bindings": [b.model_dump(mode="json") for b in state.candidate.bindings]
                if state.candidate
                else [],
            }
            try:
                bound = await self.author(interaction, runtime, context, attempt, first, bind)
                first = False
            except ValueError as error:
                self.reject_structure(attempt, str(error))
                return None
            rewrites = [r for r in bound.rewrites if r.scene_id in scenes]
            if not rewrites or round == 2:
                break
            for rewrite in rewrites:
                scenes[rewrite.scene_id].revision_note = rewrite.note
            self.store.event("scenes_rewritten", attempt=attempt, scene_ids=[r.scene_id for r in rewrites])
        candidate = Candidate(snapshot=world, bindings=bound.bindings)
        state.candidate = candidate
        report = validate_candidate(state.catalog, candidate, required)
        self.store.artifact(attempt, "validation", report)
        if not report["ok"]:
            self.reject_structure(attempt, json.dumps(report["errors"]))
            return None
        return candidate

    async def build_group(self, agents, group):
        state, cfg = self.store.state, self.settings
        budget = "build:" + group.id
        if state.candidate is not None and not state.feedback and self.current():
            context = self.group_scope(group)
            report = validate_candidate(state.catalog, state.candidate, context["required_task_ids"])
            if report["ok"]:
                attempt = self.store.reserve(budget, cfg.review_rounds.build)
                self.store.artifact(attempt, "validation", report)
                self.store.event("candidate_reused", attempt=attempt, group_id=group.id)
                verdict = await self.review(
                    agents,
                    review_payload(state.catalog, state.candidate, context["required_task_ids"], "world"),
                    attempt,
                )
                if verdict.approved:
                    state.built_groups.append(group.id)
                    state.reviews[budget] = verdict.model_dump(mode="json")
                else:
                    self.route_rejection(verdict, budget)
                self.store.finish_attempt(verdict.approved)
                if verdict.approved or state.phase == "catalog":
                    return
        # One VM per group keeps the builder's files; a fresh conversation per attempt keeps its context bounded.
        async with agents.builder.provision(
            BuilderTask.create(self.plan_context(group), group.id)
        ) as runtime:
            while state.phase == "build" and group.id not in state.built_groups:
                attempt = self.store.reserve(budget, cfg.review_rounds.build)
                task = BuilderTask.create(self.plan_context(group), group.id)
                # Issues that name messages repair those scenes in the judge's words; others re-plan.
                notes, focus = self.repairs(), {}
                plan = None
                if notes:
                    plan = state.plan.model_copy(deep=True)
                    for scene in plan.scenes:
                        scene.revision_note = notes.get(scene.id, scene.revision_note)
                    self.store.event("scenes_repaired", attempt=attempt, scene_ids=sorted(notes))
                async with agents.builder.interaction(task, runtime=runtime) as interaction:
                    candidate = await self.construct(agents, interaction, runtime, group, attempt, True, plan)
                    if candidate is not None and notes:
                        _, index = assemble(
                            state.catalog, state.plan, {k: v.scene for k, v in state.scenes.items()}
                        )
                        focus = {
                            "previous_issues": [i.model_dump(mode="json") for i in state.last_verdict.issues],
                            "changed_messages": sorted(m for s in notes for m in index["scene_messages"][s]),
                        }
                    if candidate is not None:
                        # At 100 tasks, reviewing all built tasks is simpler and safer than a dependency engine.
                        verdict = await self.review(
                            agents,
                            review_payload(
                                state.catalog,
                                candidate,
                                self.group_scope(group)["required_task_ids"],
                                "world",
                                **focus,
                            ),
                            attempt,
                            interaction.trace,
                        )
                        self.record_quality(interaction.trace, verdict, "world_quality")
                        if verdict.approved:
                            state.built_groups.append(group.id)
                            state.reviews[budget] = verdict.model_dump(mode="json")
                            state.feedback, state.last_verdict = "", None
                        else:
                            self.route_rejection(verdict, budget)
                        self.store.finish_attempt(verdict.approved)
                self.store.trace(interaction.trace)
                require_trace(interaction.trace)

    async def final_review(self, agents):
        state, cfg = self.store.state, self.settings
        report = validate_candidate(state.catalog, state.candidate, [t.id for t in state.catalog.tasks])
        if not report["ok"]:
            raise ValueError("final deterministic regression failed: " + "; ".join(report["errors"]))
        for group in state.catalog.groups:
            if group.id in state.final_groups:
                continue
            budget = "final:" + group.id
            attempt = self.store.reserve(budget, cfg.review_rounds.final)
            payload = review_payload(
                state.catalog,
                state.candidate,
                [t.id for t in state.catalog.tasks if t.group_id == group.id],
                "world",
            )
            verdict = await self.review(agents, payload, attempt)
            if verdict.approved:
                state.final_groups.append(group.id)
                state.reviews[budget] = verdict.model_dump(mode="json")
            else:
                self.store.invalidate("final review found a regression", catalog=False)
                self.route_rejection(verdict, budget)
            self.store.finish_attempt(verdict.approved)
            if not verdict.approved:
                return
        atomic_json(
            self.store.root / "frozen.json",
            {
                "catalog_hash": digest(state.catalog.model_dump(mode="json")),
                "candidate_hash": digest(state.candidate.model_dump(mode="json")),
            },
        )
        state.phase = "evaluate"
        self.store.save(approved=True)
        self.store.event(
            "workspace_frozen",
            tasks=len(state.catalog.tasks),
            messages=len(state.candidate.snapshot.messages),
        )

    async def evaluate(self, agents):
        state = self.store.state

        async def solve(task):
            self.budget_check()
            self.store.event("solver_started", task_id=task.id)
            trace = await agents.solver.run(
                SolverTask.create(
                    task,
                    state.candidate.snapshot,
                    state.catalog.workspace_id,
                    reference_for(task, state.candidate),
                    self.settings.answer_judge,
                )
            )
            self.store.trace(trace)
            result = trace.info.get("evaluation")
            if result is None and trace.info.get("grading_started"):
                raise RuntimeError("LLM answer grading failed: " + "; ".join(e.message for e in trace.errors))
            if result is None:
                result = {
                    "task_id": task.id,
                    "snapshot_hash": trace.task.data.snapshot_hash,
                    "execution_ok": False,
                    "semantic_correctness": 0.0,
                    "correct": False,
                    "grounded": False,
                    "read_count": len(trace.info.get("observations", [])),
                    "solver_trace_id": trace.id,
                    "reason": "; ".join(e.message for e in trace.errors),
                    "response": trace.last_reply,
                }
            if not trace.ok:
                result.update(execution_ok=False, semantic_correctness=0.0)
            state.evaluation[task.id] = result
            self.store.save()
            self.store.event("task_evaluated", task_id=task.id, score=result["semantic_correctness"])

        pending = [t for t in state.catalog.tasks if t.id not in state.evaluation]
        width = self.config.max_concurrent_agents or 2
        for start in range(0, len(pending), width):
            results = await asyncio.gather(
                *(solve(t) for t in pending[start : start + width]), return_exceptions=True
            )
            for result in results:
                if isinstance(result, BaseException):
                    raise result
        state.phase = "done"
        self.store.save(approved=True)

    async def run(self, task, agents):
        state = self.store.state
        while state.phase != "done":
            self.budget_check()
            if state.phase == "catalog":
                await self.synthesize(agents)
            elif state.phase == "build":
                group = next((g for g in state.catalog.groups if g.id not in state.built_groups), None)
                if group is None:
                    state.phase = "final"
                    self.store.save()
                else:
                    await self.build_group(agents, group)
            elif state.phase == "final":
                await self.final_review(agents)
            elif state.phase == "evaluate":
                await self.evaluate(agents)
