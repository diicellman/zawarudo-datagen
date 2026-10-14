"""A world run: one author agent writes the world in time order (premise, organization, the ledger, every day's
conversations); the judge reviews it mid-run and at the end. It writes no tasks: the forge writes them on the finished
world.

The rules live in the world's modules (ledger, writing, clock, board), the author's task, tools and memory in
`author/`; the world itself is one SQLite file, written only through `Workspace.trial()`."""

import contextlib
import json

from verifiers.v1.errors import SandboxError

from ..config import pinned
from ..judge.task import review, review_payload
from ..judge.verdict import Verdict, combined
from ..session import (
    SETUP_FILES,
    SessionEnv,
    ended,
    keep_awake,
    require_session,
)
from ..store import ReviewLimit, Store, used_names
from .author.memory import ledger_digest
from .author.setup import SetupTask, organization_context, parse_premise, premise_context
from .author.task import AuthorTask, day_prompt, fix_prompt, plan_prompt
from .cast import cards, census, pick_cast
from .clock import (
    bounds,
    calendar,
    clock,
    daily,
    frozen_before,
    frozen_digest,
    meta,
    present,
    start_clock,
    today,
)
from .config import WorldConfig
from .ledger import unplanned
from .organization import Organization, background_plan, organize
from .writing import agenda_budget, close_day, posted


def provenance(config: WorldConfig) -> dict:
    """Everything a resumed world run must share with its start: its configuration, and what every run pins."""
    return {"config": config.model_dump(mode="json")} | pinned()


def dry_run(config: WorldConfig) -> dict:
    """What a world run would start from, spending nothing: its configuration, the cast drawn, and each date's messages
    and their parts."""
    (occupation, _), *_ = census(config.personas)[1].most_common(1)
    pick_cast(config.personas, config.seed, [], {occupation: config.personas.pool})
    plan = {"config": config.model_dump(mode="json")}
    dates = [d["date"] for d in calendar(config.seed, config.calendar, "UTC")]
    plan["daily"] = dict(zip(dates, daily(dates, config)))  # each date's messages and their parts
    return plan


class WorldEnv(SessionEnv):
    """One world written in time order by one author: the premise and the organization, the plan, each day with
    reviews on the way, and a final review the author fixes until the judge approves."""

    def __init__(self, settings: WorldConfig, store: Store):
        super().__init__(settings, store)
        self.used = used_names(settings.corpus, store.root)

    def provision_task(self) -> SetupTask:
        """The setup author's task: the premise and the organization are written in the VM's first block."""
        return SetupTask.create(premise_context(self.settings, self.used), "setup", self.world.path)

    # ------------------------------------------------------------------ authors
    async def author_turn(self, interaction, runtime, context, attempt, first):
        self.store.event("author_started", attempt=attempt, phase_document=context["phase"])
        for name, text in SetupTask.files(context).items():
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
            raise ended(interaction.trace)
        try:
            raw = (await runtime.read(f"/task/{context['phase']}.json", max_bytes=24_000_000)).decode()
        except SandboxError:
            raw = ""  # the author ended its turn without writing the file; `author` asks again
        self.store.artifact(attempt, "author_" + context["phase"], {"text": raw})
        self.store.trace(interaction.trace)
        return raw

    async def author(self, interaction, runtime, context, attempt, first, parse):
        """Code's rejections go back to the author twice within the attempt before the attempt is rejected."""
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
        """The setup in one author session: premises (the seed picks one), then the cast, then the organization; the phase
        moves to `then`. Returns whether the session's next turn is its first."""
        state, cfg = self.store.state, self.settings
        first = True
        if state.premise is None:
            try:
                state.premise = await self.author(
                    interaction,
                    runtime,
                    premise_context(cfg, self.used),
                    "premise",
                    True,
                    lambda raw: parse_premise(raw, cfg, self.used),
                )
            except ValueError as error:
                raise ReviewLimit(f"the author proposed no valid premise: {error}") from None
            first = False
            state.cast = pick_cast(cfg.personas, cfg.seed, self.used["people"], state.premise.staffing)
            state.phase = "organization"
            self.store.save()
            self.store.event("premise_selected", company=state.premise.company)
        if state.phase == "organization":
            try:
                await self.author(
                    interaction,
                    runtime,
                    organization_context(cfg, state, self.used),
                    "organization",
                    first,
                    self.apply(
                        lambda w, d: organize(w, d, state.cast, state.premise, cfg),
                        Organization,
                        "organization",
                    ),
                )
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

    async def assess(self, agents, attempt):
        """The final review: the whole world judged, with the workspace issues the last one raised."""
        state = self.store.state
        previous = list(state.last_verdict.issues) if state.last_verdict else []
        workspace = [i.model_dump(mode="json") for i in previous if i.artifact == "workspace"]
        payload = review_payload(
            self.world,
            "world",
            [],
            people=cards(state.cast),
            ledger=ledger_digest(self.world),
            **({"previous_issues": workspace} if workspace else {}),
            **self.written_from(),
        )
        verdict = combined([await review(self.store, agents, payload, attempt, files={}, label="world")])
        self.store.artifact(attempt, "verdict", verdict.model_dump(mode="json"))
        self.store.artifact(attempt, "acceptance", {"approved": verdict.approved})
        return verdict

    def written_from(self) -> dict:
        """A chapter's reviews: the moment its own days start, before which the world is written and frozen."""
        start = frozen_before(self.world)
        return {"written_from": clock(self.world, start)} if start else {}

    # ------------------------------------------------------------------ the world, written in time order

    async def setup_world(self, agents, runtime, task) -> None:
        """The premise and the organization, as documents, in the author's own session."""
        async with agents.author.interaction(task, runtime=runtime) as interaction:
            await self.premise_and_organization(interaction, runtime, "plan")
        self.store.trace(interaction.trace)
        require_session(interaction.trace)

    async def plan_world(self, agents, runtime) -> None:
        """The ledger, before the first day it plans (day 1, or a chapter's first): storylines, events and facts, the
        board, and the author's plan of them. A chapter starts from its parent's notes, and its plan.md must change."""
        state, cfg = self.store.state, self.settings
        if not self.world.db.execute("SELECT 1 FROM world_meta WHERE key = 'chronological'").fetchone():
            with self.world.trial() as copy:
                start_clock(copy)
        if state.restore_point:  # a chapter: the notes its parent's last day left
            await self.bring_notes(runtime, state.restore_point)
        imported = await self.note(runtime, "plan.md") if state.restore_point else None
        attempt = self.store.reserve("plan", cfg.author.plan_attempts)
        task = AuthorTask.create("plan", 0, self.world.path, self.tools_context(), attempt)
        async with agents.author.interaction(task, runtime=runtime) as interaction:
            prompt, errors = plan_prompt(self.world, cfg, state.feedback), []
            for _ in range(3):
                await self.take_turn(interaction, runtime, prompt, attempt, "plan")
                errors = unplanned(self.world, cfg, state.quota)
                note = await self.note(runtime, "plan.md")
                errors += ["write /task/notes/plan.md"] * (not (note or "").strip() or note == imported)
                if not errors:
                    break
                prompt = "The plan is not done: " + "; ".join(errors) + ". Finish it, then end your turn."
        self.store.trace(interaction.trace)
        require_session(interaction.trace)
        if errors:
            return self.reject_structure(attempt, "; ".join(errors))
        await self.keep_notes(runtime, attempt)
        state.phase, state.day, state.restore_point, state.feedback = "day", today(self.world), attempt, ""
        self.store.finish_attempt(True)

    async def write_day(self, agents, runtime) -> None:
        """One calendar day, in a fresh interaction that starts from where the last day closed: the world and the
        author's notes. The day is done when it is closed and the author's recap is written."""
        state, cfg = self.store.state, self.settings
        day, total = state.day, self.world.db.execute("SELECT COUNT(*) FROM calendar").fetchone()[0]
        attempt = self.store.reserve(f"day-{day:02d}", cfg.author.day_attempts)
        self.store.restore(state.restore_point)
        await self.bring_notes(runtime, state.restore_point)
        agenda = state.plans.setdefault("agenda", {})
        if str(day) not in agenda:  # drawn once, around the plan as it stands; a retried day keeps its draw
            organization = Organization.model_validate_json(json.dumps(state.drafts["organization"]))
            budget = agenda_budget(self.world, cfg, day)
            drawn = background_plan(self.world, organization, cfg.activity, cfg.seed, day, budget)
            agenda[str(day)] = [s.model_dump(mode="json") for s in drawn]
            self.store.save()
        before = await self.note(runtime, "recap.md")
        issues = [i.model_dump(mode="json") for i in state.issues]
        if issues:
            self.store.event("issues_delivered", attempt=attempt, issues=len(issues))
        task = AuthorTask.create("day", day, self.world.path, self.tools_context(), attempt)
        async with agents.author.interaction(task, runtime=runtime) as interaction:
            prompt, errors = day_prompt(self.world, day, issues, state.feedback), []
            for _ in range(3):
                await self.take_turn(interaction, runtime, prompt, attempt, "day")
                errors = []
                if today(self.world) == day:
                    errors = close_day(self.world, cfg, day) or [
                        'advance to the night, then to "tomorrow", to close it'
                    ]
                if await self.note(runtime, "recap.md") in (None, before):
                    errors.append("write /task/notes/recap.md: what happened today and what is open")
                if not errors:
                    break
                prompt = f"Day {day} is not done: " + "; ".join(errors) + ". Finish it, then end your turn."
        self.store.trace(interaction.trace)
        require_session(interaction.trace)
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
            state.day, state.phase = (day + 1, "day") if day < total else (day, "final")
        self.store.save()

    async def review_so_far(self, agents) -> None:
        """The judge reviews the world written so far; its issues open the author's next day."""
        state = self.store.state
        attempt = self.store.reserve(f"review-{state.day:02d}", 2)
        payload = review_payload(
            self.world,
            "world",
            [],
            people=cards(state.cast),
            written_through=clock(self.world, present(self.world)),
            ledger=ledger_digest(self.world),
            **self.written_from(),
        )
        verdict = await review(self.store, agents, payload, attempt, files={}, label="world")
        # It rejects nothing, so every issue is a note for what is written next, blocking or not.
        state.issues = verdict.issues
        state.day, state.phase = state.day + 1, "day"
        self.store.finish_attempt(verdict.approved)

    async def fix_world(self, agents, runtime) -> None:
        """The final review, until the judge approves or its rounds run out. Its issues go to the author in one more
        interaction, opened at the first rejection."""
        state, cfg = self.store.state, self.settings
        interaction = None
        async with contextlib.AsyncExitStack() as stack:
            await stack.enter_async_context(keep_awake(runtime, cfg.author.keepalive))
            while state.phase == "final":
                if (verdict := await self.final_world(agents)).approved:
                    break
                attempt = f"final-{state.rounds.get('final', 0):02d}"
                # Every issue goes back, the non-blocking ones too: the author fixes them or says why not.
                issues = [i.model_dump(mode="json") for i in verdict.issues]
                if interaction is None:
                    await self.bring_notes(runtime, state.restore_point)
                    task = AuthorTask.create("fix", 0, self.world.path, self.tools_context(), attempt)
                    interaction = await stack.enter_async_context(
                        agents.author.interaction(task, runtime=runtime)
                    )
                await self.take_turn(interaction, runtime, fix_prompt(issues), attempt, "fix")
        if interaction is not None:
            self.store.trace(interaction.trace)
            require_session(interaction.trace)

    async def final_world(self, agents) -> Verdict:
        """The final review: every stale task solved and judged, and the whole world judged."""
        state, cfg = self.store.state, self.settings
        if broken := self.world.violations(complete=True):
            raise ValueError("final deterministic check failed: " + "; ".join(broken))
        if (kept := meta(self.world, "frozen_digest")) is not None and frozen_digest(self.world) != kept:
            raise ValueError("final deterministic check failed: an earlier chapter's world changed")
        attempt = self.store.reserve("final", cfg.review_rounds.final)
        verdict = await self.assess(agents, attempt)
        if verdict.approved:
            state.reviews["final"] = verdict.model_dump(mode="json")
            state.phase = "done"
            self.store.event("workspace_frozen", tasks=len(state.task_reviews))
        state.last_verdict = None if verdict.approved else verdict  # the next final review hears its issues
        self.store.finish_attempt(verdict.approved)
        return verdict

    async def blocks(self, agents, runtime, setup) -> None:
        state = self.store.state
        if state.phase in ("premise", "organization"):
            await self.setup_world(agents, runtime, setup)
        # Provisioning lays out the setup files again on every resume; the author's later turns never see them.
        await runtime.run(["rm", "-f", *(f"/task/{name}" for name in SETUP_FILES)], {})
        while state.phase == "plan":
            await self.plan_world(agents, runtime)
        while state.phase in ("day", "review"):
            await (self.review_so_far(agents) if state.phase == "review" else self.write_day(agents, runtime))
        if state.phase == "final":
            await self.fix_world(agents, runtime)
