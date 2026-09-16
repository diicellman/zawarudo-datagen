"""Catalog → reviewed build groups → final review → independent solves."""

import asyncio
import json
import verifiers.v1 as vf
from worldgen_slack.dataset import PrivateAnswer, atomic_json
from worldgen_slack.slack.api import digest
from worldgen_slack.taskset import SolverTask
from .agents.synthesizer import SynthesizerTask
from .agents.builder import BuilderTask
from .agents.judge import JudgeTask, review_payload
from .config import PipelineConfig
from .contracts import Catalog, Candidate, Verdict, validate_candidate, quality
from .store import ReviewLimit


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
    def __init__(self, settings, store):
        self.settings, self.store = settings, store
        super().__init__(settings.env)

    async def setup(self, agents):
        for name in ("synthesizer", "builder", "judge", "solver"):
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
            else "Read the updated /task/input.json. Address the reviewer feedback and revise /task/output.json."
        )
        if segment.terminated:
            raise ReviewLimit("author exhausted its native interaction limits")
        raw = (await runtime.read("/task/output.json", max_bytes=24_000_000)).decode()
        self.store.artifact(attempt, "author_output", {"text": raw})
        interaction.trace.info.setdefault("candidates", []).append(
            {"candidate_id": attempt, "input_hash": digest(context)}
        )
        self.store.trace(interaction.trace)
        return raw

    async def review(self, agents, payload, attempt, author_trace=None):
        self.budget_check()
        self.store.artifact(attempt, "review_input", payload)
        self.store.event("review_started", attempt=attempt, task_ids=[t["id"] for t in payload["tasks"]])
        task = JudgeTask.create(payload, candidate_id=attempt)
        if author_trace is not None:
            record = author_trace.to_record()
            for node in record["nodes"]:
                node["message"].pop("reasoning_content", None)
                node["message"].pop("provider_state", None)
            self.store.artifact(attempt, "author_trace", record)
            task.config.author_trace = json.dumps(record)
        trace = await agents.judge.run(task)
        self.store.trace(trace)
        require_trace(trace)
        verdict = Verdict.model_validate(trace.info["verdict"])
        self.store.artifact(attempt, "verdict", verdict.model_dump(mode="json"))
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
        state.feedback = verdict.model_dump_json()
        group_id = budget.partition(":")[2]
        if group_id in state.built_groups:
            state.built_groups.remove(group_id)
        if any(issue.owner == "synthesizer" for issue in verdict.issues):
            self.store.invalidate("catalog repair requested")
            state.catalog_budget = budget
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
            "seed": cfg.seed,
            "workspace_id": state.catalog.workspace_id if state.catalog else f"workspace-{cfg.seed}",
            "feedback": state.feedback,
            "previous_output": state.catalog.model_dump(mode="json") if state.catalog else None,
            "instructions": "Create the whole catalog. On repair preserve task IDs, group IDs, and assignments.",
        }

    async def synthesize(self, agents):
        state, cfg = self.store.state, self.settings
        initial = self.catalog_context()
        task = SynthesizerTask.create(initial, "catalog")
        async with agents.synthesizer.provision(task) as runtime:
            async with agents.synthesizer.interaction(task, runtime=runtime) as interaction:
                first = True
                while state.phase == "catalog":
                    attempt = self.store.reserve(state.catalog_budget, cfg.max_review_rounds)
                    context = self.catalog_context()
                    raw = await self.author_turn(interaction, runtime, context, attempt, first)
                    first = False
                    try:
                        catalog = Catalog.model_validate_json(raw)
                        if (
                            len(catalog.tasks) != cfg.task_count
                            or catalog.sector.casefold() != cfg.sector.casefold()
                        ):
                            raise ValueError("catalog must match requested task count and sector")
                        if catalog.workspace_id != context["workspace_id"]:
                            raise ValueError("workspace ID must remain stable")
                        if len(catalog.groups) != (cfg.task_count + cfg.group_size - 1) // cfg.group_size:
                            raise ValueError("catalog has the wrong number of groups")
                        if any(
                            sum(t.group_id == g.id for t in catalog.tasks) > cfg.group_size
                            for g in catalog.groups
                        ):
                            raise ValueError("group exceeds configured task count")
                        if state.catalog and {(t.id, t.group_id) for t in catalog.tasks} != {
                            (t.id, t.group_id) for t in state.catalog.tasks
                        }:
                            raise ValueError("catalog repair must preserve task IDs and group assignments")
                    except ValueError as error:
                        self.reject_structure(attempt, str(error))
                        continue
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
                        state.catalog_budget, state.phase, state.feedback = "catalog", "build", ""
                    self.store.finish_attempt(verdict.approved)
            self.store.trace(interaction.trace)
            require_trace(interaction.trace)

    def reject_structure(self, attempt, error):
        self.store.state.feedback = error[:8000]
        self.store.artifact(attempt, "validation", {"ok": False, "errors": [error]})
        self.store.finish_attempt(False)

    def build_context(self, group):
        state, cfg = self.store.state, self.settings
        included = {*state.built_groups, group.id}
        return {
            "phase": "world",
            "workspace_id": state.catalog.workspace_id,
            "group_id": group.id,
            "catalog": state.catalog.model_dump(mode="json"),
            "required_task_ids": [t.id for t in state.catalog.tasks if t.group_id in included],
            "target_messages_for_this_group": max(1, cfg.target_messages // len(state.catalog.groups)),
            "target_total_messages_after_this_group": max(
                1, cfg.target_messages * len(included) // len(state.catalog.groups)
            ),
            "feedback": state.feedback,
            "previous_output": state.candidate.model_dump(mode="json") if state.candidate else None,
        }

    async def build_group(self, agents, group):
        state, cfg = self.store.state, self.settings
        budget = "build:" + group.id
        if state.candidate is not None and not state.feedback:
            context = self.build_context(group)
            report = validate_candidate(state.catalog, state.candidate, context["required_task_ids"])
            if report["ok"]:
                attempt = self.store.reserve(budget, cfg.max_review_rounds)
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
        task = BuilderTask.create(self.build_context(group), group.id)
        async with agents.builder.provision(task) as runtime:
            async with agents.builder.interaction(task, runtime=runtime) as interaction:
                first = True
                while state.phase == "build" and group.id not in state.built_groups:
                    attempt = self.store.reserve(budget, cfg.max_review_rounds)
                    context = self.build_context(group)
                    raw = await self.author_turn(interaction, runtime, context, attempt, first)
                    first = False
                    try:
                        candidate = Candidate.model_validate_json(raw)
                    except ValueError as error:
                        self.reject_structure(attempt, str(error))
                        continue
                    state.candidate = candidate
                    report = validate_candidate(state.catalog, candidate, context["required_task_ids"])
                    self.store.artifact(attempt, "validation", report)
                    if not report["ok"]:
                        self.reject_structure(attempt, json.dumps(report["errors"]))
                        continue
                    # At 100 tasks, reviewing all built tasks is simpler and safer than a dependency engine.
                    verdict = await self.review(
                        agents,
                        review_payload(state.catalog, candidate, context["required_task_ids"], "world"),
                        attempt,
                        interaction.trace,
                    )
                    self.record_quality(interaction.trace, verdict, "world_quality")
                    if verdict.approved:
                        state.built_groups.append(group.id)
                        state.reviews[budget] = verdict.model_dump(mode="json")
                        state.feedback = ""
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
            attempt = self.store.reserve(budget, cfg.max_review_rounds)
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
