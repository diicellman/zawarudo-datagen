"""The generation episode's control flow: catalog → incrementally built, reviewed groups → final review.

What each role sees and how its output is parsed lives with the role in `agents/`; the rules are in `contracts`."""

import asyncio
import json
from collections import defaultdict
import verifiers.v1 as vf
from verifiers.v1.configs.runtime import NetworkPolicyConfig
from verifiers.v1.errors import SandboxError
from worldgen_slack.dataset import PrivateAnswer, atomic_json
from worldgen_slack.slack.api import digest
from worldgen_slack.taskset import SolverTask
from .agents import builder, synthesizer, writer
from .agents.builder import BuilderTask
from .agents.judge import JudgeTask, review_payload
from .agents.synthesizer import SynthesizerTask
from .config import PipelineConfig
from .contracts import (
    PHASE_CRITERIA,
    BindOutput,
    Candidate,
    Plan,
    SceneRecord,
    Verdict,
    accepted,
    accepted_task,
    assemble,
    deciding,
    check_plan,
    pick_cast,
    quality,
    review_key,
    validate_candidate,
)
from .store import ReviewLimit, used_names


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

    async def review(self, agents, payload, attempt, files=None, label=""):
        """`files` maps /task file names to trace records the judge may investigate; `label` names one of an
        attempt's several reviews."""
        suffix, files = ("-" + label if label else ""), files or {}
        self.budget_check()
        self.store.artifact(attempt, "review_input" + suffix, payload)
        self.store.event(
            "review_started", attempt=attempt, label=label, task_ids=[t["id"] for t in payload["tasks"]]
        )
        if "author_trace.json" in files:
            self.store.artifact(attempt, "author_trace", files["author_trace.json"])
        # One fresh rerun absorbs a malformed verdict; the review round is still spent only once.
        for _ in range(2):
            task = JudgeTask.create(payload, candidate_id=attempt)
            task.config.files = {name: json.dumps(record) for name, record in files.items()}
            trace = await agents.judge.run(task)
            self.store.trace(trace)
            if trace.ok:
                break
        require_trace(trace)
        verdict = Verdict.model_validate(trace.info["verdict"])
        self.store.artifact(attempt, "verdict" + suffix, verdict.model_dump(mode="json"))
        if payload["phase"] in PHASE_CRITERIA:
            verdict = verdict.model_copy(
                update={"approved": accepted(verdict, payload, self.settings.acceptance)}
            )
            self.store.artifact(
                attempt, "acceptance" + suffix, {"approved": verdict.approved, "quality": quality(verdict)}
            )
        return verdict

    def route_rejection(self, verdict, budget):
        state = self.store.state
        state.feedback, state.last_verdict = verdict.model_dump_json(), verdict
        group_id = budget.partition(":")[2]
        if group_id in state.built_groups:
            state.built_groups.remove(group_id)
        if any(issue.owner == "synthesizer" for issue in deciding(verdict, self.settings.acceptance)):
            self.store.invalidate("catalog repair requested")
            state.phase = "catalog"
        else:
            state.phase = "build"

    def reject_structure(self, attempt, error):
        self.store.state.feedback, self.store.state.last_verdict = error[:8000], None
        self.store.artifact(attempt, "validation", {"ok": False, "errors": [error]})
        self.store.finish_attempt(False)

    async def synthesize(self, agents):
        state, cfg = self.store.state, self.settings
        task = SynthesizerTask.create(synthesizer.catalog_context(cfg, state, self.used), "catalog")
        async with agents.synthesizer.provision(task) as runtime:
            async with agents.synthesizer.interaction(task, runtime=runtime) as interaction:
                first = True
                if state.premise is None:
                    try:
                        state.premise = await self.author(
                            interaction,
                            runtime,
                            synthesizer.premise_context(cfg, self.used),
                            "premise",
                            True,
                            lambda raw: synthesizer.parse_premise(raw, cfg, self.used),
                        )
                    except ValueError as error:
                        raise ReviewLimit(f"synthesizer proposed no valid premise set: {error}") from None
                    if cfg.personas:
                        occupations = state.premise.occupations
                        state.cast = pick_cast(cfg.personas, cfg.seed, self.used["people"], occupations)
                    first = False
                    self.store.save()
                    self.store.event("premise_selected", company=state.premise.company)
                while state.phase == "catalog":
                    attempt = self.store.reserve("catalog", cfg.review_rounds.catalog)
                    try:
                        catalog = await self.author(
                            interaction,
                            runtime,
                            synthesizer.catalog_context(cfg, state, self.used),
                            attempt,
                            first,
                            lambda raw: synthesizer.check_catalog(raw, cfg, state, self.used),
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
                        {"author_trace.json": observable(interaction.trace)},
                    )
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

    async def write_scene(self, agents, plan, scene, seen, previous, timely):
        state = self.store.state
        prompt = writer.brief(
            state.catalog,
            state.premise,
            plan,
            scene,
            seen,
            writer.excerpts(self.seeds, f"{self.settings.seed}:{scene.id}"),
            previous,
            self.settings.language,
            timely,
        )
        written, errors, trace = await writer.compose(agents.writer, prompt, scene, state.catalog, timely)
        trace.info["scene_id"] = scene.id
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
                if writer.visible(other, scene, conversations):
                    await done[other.id].wait()
            (key, timely), previous = writer.scene_key(state.catalog, plan, scene), state.scenes.get(scene.id)
            if previous is None or previous.key != key:
                scenes = {k: v.scene for k, v in state.scenes.items()}
                seen = writer.observed(state.catalog, plan, scenes, scene)
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
        """Scene → the judge's words, when every deciding issue of the last rejection points at messages of known
        scenes."""
        state = self.store.state
        issues = deciding(state.last_verdict, self.settings.acceptance) if state.last_verdict else []
        if (
            not issues
            or state.plan is None
            or any(i.artifact != "workspace" or not i.message_ids for i in issues)
        ):
            return {}
        _, index = assemble(state.catalog, state.plan, {k: v.scene for k, v in state.scenes.items()})
        owner = {m: s for s, ids in index["scene_messages"].items() for m in ids}
        notes = defaultdict(list)
        for issue in issues:
            if not (scenes := {owner[m] for m in issue.message_ids if m in owner}):
                return {}
            for scene_id in scenes:
                notes[scene_id].append(f"{issue.defect} Requested change: {issue.requested_change}")
        return {scene_id: " ".join(n) for scene_id, n in notes.items()}

    async def construct(self, agents, interaction, runtime, group, attempt, first, plan=None):
        """Plan (unless a repaired plan is given), write, assemble, bind. Returns a checked candidate, or None
        after a structural rejection."""
        state = self.store.state
        scope = builder.group_scope(state, group.id)
        try:
            if plan is None:
                plan = await self.author(
                    interaction,
                    runtime,
                    builder.plan_context(self.settings, state, group.id),
                    attempt,
                    first,
                    lambda raw: check_plan(
                        builder.scoped(state, group.id), Plan.model_validate_json(raw), writer.frozen(state)
                    ),
                )
                first = False
            state.plan = plan
            world, index = await self.write(agents, plan)

            def bind(raw):
                bound = BindOutput.model_validate_json(raw)
                candidate = Candidate(snapshot=world, bindings=bound.bindings)
                if not (report := validate_candidate(state.catalog, candidate, scope["required_task_ids"]))[
                    "ok"
                ]:
                    raise ValueError(json.dumps(report["errors"]))
                return bound

            context = {
                "phase": "bind",
                **scope,
                "workspace": world.model_dump(mode="json"),
                **index,
                "previous_bindings": [b.model_dump(mode="json") for b in state.candidate.bindings]
                if state.candidate
                else [],
            }
            bound = await self.author(interaction, runtime, context, attempt, first, bind)
        except ValueError as error:
            self.reject_structure(attempt, str(error))
            return None
        state.candidate = Candidate(snapshot=world, bindings=bound.bindings)
        report = validate_candidate(state.catalog, state.candidate, scope["required_task_ids"])
        self.store.artifact(attempt, "validation", report)
        return state.candidate

    async def build_group(self, agents, group):
        state, cfg = self.store.state, self.settings
        budget = "build:" + group.id
        if state.candidate is not None and not state.feedback and writer.current(state):
            required = builder.group_scope(state, group.id)["required_task_ids"]
            report = validate_candidate(state.catalog, state.candidate, required)
            if report["ok"]:
                attempt = self.store.reserve(budget, cfg.review_rounds.build)
                self.store.artifact(attempt, "validation", report)
                self.store.event("candidate_reused", attempt=attempt, group_id=group.id)
                verdict, _ = await self.assess(agents, group, state.candidate, attempt, writer.changed(state))
                self.settle(group, verdict, budget)
                if verdict.approved or state.phase == "catalog":
                    return
        # One VM per group keeps the builder's files; a fresh conversation per attempt keeps its context bounded.
        async with agents.builder.provision(
            BuilderTask.create(builder.plan_context(cfg, state, group.id), group.id)
        ) as runtime:
            while state.phase == "build" and group.id not in state.built_groups:
                attempt = self.store.reserve(budget, cfg.review_rounds.build)
                task = BuilderTask.create(builder.plan_context(cfg, state, group.id), group.id)
                # Issues that name messages repair those scenes in the judge's words; bindings-only issues
                # bind again on the same scenes; others re-plan.
                notes = self.repairs()
                issues = deciding(state.last_verdict, cfg.acceptance) if state.last_verdict else []
                rebind = bool(issues) and all(i.artifact == "bindings" for i in issues)
                plan = None
                if notes or rebind:
                    plan = state.plan.model_copy(deep=True)
                    for scene in plan.scenes:
                        scene.revision_note = notes.get(scene.id, scene.revision_note)
                route = "repair" if notes else "rebind" if rebind else "plan"
                self.store.event("attempt_routed", attempt=attempt, route=route, scene_ids=sorted(notes))
                async with agents.builder.interaction(task, runtime=runtime) as interaction:
                    candidate = await self.construct(agents, interaction, runtime, group, attempt, True, plan)
                    if candidate is not None:
                        verdict, _ = await self.assess(
                            agents, group, candidate, attempt, writer.changed(state), interaction.trace
                        )
                        self.settle(group, verdict, budget)
                self.store.trace(interaction.trace)
                require_trace(interaction.trace)

    def settle(self, group, verdict, budget):
        state = self.store.state
        if verdict.approved:
            state.built_groups.append(group.id)
            state.frozen_scenes = {s.id: state.scenes[s.id].key for s in state.plan.scenes}
            state.feedback, state.last_verdict = "", None
        else:
            self.route_rejection(verdict, budget)
        self.store.finish_attempt(verdict.approved)

    def solver_task(self, task, candidate):
        """A solver outside a sandbox executes no code (`config.secure`), so it gets no network policy to enforce."""
        return SolverTask.create(
            task,
            candidate.snapshot,
            self.store.state.catalog.workspace_id,
            reference_for(task, candidate),
            self.settings.answer_judge,
            network_policy=isinstance(self.settings.env.solver.runtime, NetworkPolicyConfig),
        )

    async def attempt(self, agents, task, candidate):
        """One independent solve of a task, graded for correctness and grounding."""
        self.budget_check()
        trace = await agents.solver.run(self.solver_task(task, candidate))
        self.store.trace(trace)
        return SolverTask.outcome(trace), trace

    async def assess(self, agents, group, candidate, attempt, changed, author=None):
        """Solve and judge every task of `group`'s scope whose review is stale, and judge the scenes written since
        the last approval. `group=None` is the final review: the whole catalog and the whole world without focus.
        Returns the combined verdict and the world verdict."""
        state, n = self.store.state, self.settings.solves_per_task
        catalog = state.catalog if group is None else builder.scoped(state, group.id)
        later = {"later_groups": groups} if (groups := builder.later_groups(state, catalog)) else {}
        previous = state.last_verdict.issues if state.last_verdict else []
        task_ids = [t.id for t in catalog.tasks]
        outputs = validate_candidate(catalog, candidate, task_ids)["gold_outputs"]
        keys = {t: review_key(catalog, candidate, t, outputs[t]) for t in task_ids}
        due = [t for t in catalog.tasks if state.task_reviews.get(t.id, {}).get("key") != keys[t.id]]

        async def check():
            """The proposer–solver fan-out: n independent solves of every due task, then one judge that
            investigates them together."""
            solves = await asyncio.gather(
                *(asyncio.gather(*(self.attempt(agents, t, candidate) for _ in range(n))) for t in due)
            )
            runs = {t.id: runs for t, runs in zip(due, solves)}
            payload = review_payload(
                catalog,
                candidate,
                list(runs),
                "task",
                solves=[result for task_runs in runs.values() for result, _ in task_runs],
                previous_issues=[
                    i.model_dump(mode="json") for i in previous if set(i.task_ids) & runs.keys()
                ],
                **later,
            )
            files = {
                f"solver_{t}_{k}.json": observable(trace)
                for t, task_runs in runs.items()
                for k, (_, trace) in enumerate(task_runs, 1)
            }
            return runs, payload, await self.review(agents, payload, attempt, files, "tasks")

        workspace = [i.model_dump(mode="json") for i in previous if i.artifact == "workspace"]
        world_payload = None
        if group is None:
            world_payload = review_payload(catalog, candidate, [], "world")
        elif changed or workspace:
            _, index = assemble(catalog, state.plan, {k: v.scene for k, v in state.scenes.items()})
            world_payload = review_payload(
                catalog,
                candidate,
                [],
                "world",
                previous_issues=workspace,
                changed_messages=sorted(m for s in changed for m in index["scene_messages"][s]),
                **later,
            )
        files = {"author_trace.json": observable(author)} if author is not None else {}
        async with asyncio.TaskGroup() as tasks:
            world = (
                tasks.create_task(self.review(agents, world_payload, attempt, files, "world"))
                if world_payload is not None
                else None
            )
            checked = tasks.create_task(check()) if due else None
        runs, payload, judged = checked.result() if checked else ({}, None, None)
        verdicts = [v for v in (judged, world.result() if world else None) if v is not None]
        rejected = [v for v in verdicts if not v.approved]
        for task_id, task_runs in runs.items():
            results = [result for result, _ in task_runs]
            rate = sum(r["semantic_correctness"] for r in results) / n
            approved = accepted_task(judged, payload, self.settings.acceptance, task_id)
            if approved:
                state.task_reviews[task_id] = {"key": keys[task_id], "solve_rate": rate, "results": results}
            self.store.event(
                "task_reviewed",
                attempt=attempt,
                task_id=task_id,
                approved=approved,
                solve_rate=rate,
                learnability=4 * rate * (1 - rate),
            )
        merged = Verdict.model_construct(
            reviewed_hash="",
            approved=not rejected,
            tasks=[r for v in verdicts for r in v.tasks],
            issues=[i for v in verdicts for i in v.issues],
            criteria={},
            # A rejection's feedback carries only the rejecting reviews' words.
            summary=" ".join(v.summary for v in rejected or verdicts) or "No task or scene needed review.",
        )
        self.store.artifact(attempt, "review_input", review_payload(catalog, candidate, list(runs), "world"))
        self.store.artifact(attempt, "verdict", merged.model_dump(mode="json"))
        self.store.artifact(
            attempt,
            "acceptance",
            {
                "approved": merged.approved,
                "quality": sum(quality(v) for v in verdicts) / len(verdicts) if verdicts else None,
                "reviewed_tasks": list(runs),
                "changed_scenes": changed,
            },
        )
        return merged, world.result() if world else None

    async def final_review(self, agents):
        state, cfg = self.store.state, self.settings
        report = validate_candidate(state.catalog, state.candidate, [t.id for t in state.catalog.tasks])
        if not report["ok"]:
            raise ValueError("final deterministic regression failed: " + "; ".join(report["errors"]))
        attempt = self.store.reserve("final", cfg.review_rounds.final)
        verdict, world = await self.assess(agents, None, state.candidate, attempt, None)
        if verdict.approved:
            state.reviews["final"] = world.model_dump(mode="json")
        else:
            # Repairs of any scene run in the last group's build; its review covers every scene that changes.
            self.store.invalidate("final review found a regression", catalog=False)
            self.route_rejection(verdict, "final:" + state.built_groups[-1])
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
        state.phase = "done"
        self.store.save()
        self.store.event(
            "workspace_frozen",
            tasks=len(state.catalog.tasks),
            messages=len(state.candidate.snapshot.messages),
        )

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
