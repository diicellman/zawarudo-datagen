"""The generation episode: one author agent writes the world in time order (premise, organization, the ledger,
every day's conversations, the tasks); the judge reviews it mid-run and at the end; the GLM solver tries the tasks.

The rules live in `chronicle` and `contracts`, the author's tools in `agents/world`; the world itself is one SQLite
file, written only through `World.trial()`."""

import asyncio
import contextlib
import io
import itertools
import json
import math
import tarfile
import time

import verifiers.v1 as vf
from verifiers.v1.configs.runtime import NetworkPolicyConfig
from verifiers.v1.dialects import base as dialects
from verifiers.v1.errors import SandboxError
from verifiers.v1.utils.interrupt import cleaning_up
from worldgen_slack.dataset import atomic_json
from worldgen_slack.db import digest
from worldgen_slack.taskset import AnswerJudge, SolverTask, shown

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
from .chronicle import agenda_budget, answer_of, bounds, close_day, posted, present, start_clock, today
from .config import Config, PipelineConfig
from .contracts import (
    PHASE_CRITERIA,
    Organization,
    Verdict,
    Issue,
    accepted,
    accepted_task,
    background_plan,
    band_move,
    cards,
    clock,
    failure_cases,
    organize,
    pick_cast,
    quality,
    quota,
)
from .store import ReviewLimit, SessionLost, Store, used_names

NOTES = ("plan.md", "recap.md")  # the world author's own files in /task/notes
SETUP_FILES = ("input.json", "guide.md", "schemas.json", "schema.sql", "world.sqlite", "premise.json", "organization.json")  # fmt: skip


ARCHIVE = "/tmp/task-files.tgz"
# What verifiers tells every agent on a restricted network, on the first user message of each request: by default,
# "do not retry the blocked provider-side operation", even when nothing was removed (to fix upstream: only when a
# capability was). Our VMs block every host and reach their tools through a tunnel, where the true thing to say is
# that a dropped call never arrived; the authors stopped their turns on the default instead of calling again.
NETWORK_NOTICE = (
    "This machine has no network. Its world_* and inspect_* functions reach the host through a tunnel; one that "
    'raises "is unavailable" never arrived: call it again.'
)


async def upload(runtime, files: dict[str, str | bytes], clear: str) -> None:
    """Files into /task in one archive: one upload and one command, not one upload per file. `clear` is emptied
    first, so a file that is gone from the set is gone from the VM too."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, data in files.items():
            data = data.encode() if isinstance(data, str) else data
            info = tarfile.TarInfo(name)
            info.size, info.mtime, info.mode = len(data), int(time.time()), 0o644
            archive.addfile(info, io.BytesIO(data))
    await runtime.write(ARCHIVE, buffer.getvalue())
    script = f"rm -rf {clear} && mkdir -p /task && python3 -m tarfile -e {ARCHIVE} /task && rm -f {ARCHIVE}"
    if (result := await runtime.run(["sh", "-c", script], {})).exit_code:
        raise SandboxError(f"the author's files could not be unpacked: {result.stderr}")


@contextlib.asynccontextmanager
async def keep_awake(runtime, every: float):
    """The author's VM kept from idling out while only the env works (the solver's tries, the judge's reviews): a
    no-op command every `every` seconds. The pings end with the block, so a VM left behind still ends on its idle
    timeout."""

    async def ping():
        while True:
            await asyncio.sleep(every)
            await runtime.run(["true"], {})

    pinger = asyncio.create_task(ping())
    try:
        yield
    finally:
        pinger.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await pinger


def require_trace(trace):
    if not trace.ok:
        raise RuntimeError("agent execution failed: " + "; ".join(e.message for e in trace.errors))


def ended(trace) -> RuntimeError:
    """Why an author's session ended before its work did: its own limits (the turn cap, the token budget), or an
    error of its sandbox, tunnel or provider, which is none of the author's (S3's three stops: Prime's process stream
    dropped, and each was reported as the author's limits)."""
    errors = "; ".join(f"{e.type}: {e.message}" for e in getattr(trace, "errors", None) or [])
    if not errors or getattr(trace, "is_truncated", False) or "budget reached" in errors:
        return ReviewLimit(
            "the author exhausted its interaction's limits" + (f": {errors}" if errors else "")
        )
    return SessionLost("the author's session was lost: " + errors)


def require_session(trace):
    """An author's interaction that failed, for the reason `ended` gives."""
    if not trace.ok:
        raise ended(trace)


def observable(record: dict) -> dict:
    """A trace record without hidden reasoning or provider state, for another agent to investigate."""
    for node in record["nodes"]:
        node["message"].pop("reasoning_content", None)
        node["message"].pop("provider_state", None)
    return record


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


def merged(verdicts: list[Verdict]) -> Verdict:
    """Several reviews as one verdict: approved when each is, with every task review and issue; a rejection's feedback
    carries only the rejecting reviews' words."""
    rejected = [v for v in verdicts if not v.approved]
    return Verdict.model_construct(
        approved=not rejected,
        tasks=[r for v in verdicts for r in v.tasks],
        issues=[i for v in verdicts for i in v.issues],
        criteria={},
        summary=" ".join(v.summary for v in rejected or verdicts),
    )


def try_digest(result: dict, record: dict | None, evidence: set, decoys: set = frozenset()) -> dict:
    """One solver try as the author reads it: its grade, the step at which the gold evidence first showed and the call
    that showed it, how many calls showed a decoy, its route (the calls up to the evidence, then its last three) and
    its answer. The whole try is a file of its own."""
    steps, first, lured = [], None, 0
    for k, call in enumerate(((record or {}).get("info") or {}).get("observations") or [], 1):
        seen = shown(call)
        hits, lures = seen & evidence, seen & decoys
        first, lured = first or (k if hits else None), lured + bool(lures)
        items = call["output"].get("items")
        returned = f"{len(items)} items" if isinstance(items, list) else "one result"
        steps.append(f"{call['tool']} {json.dumps(call['arguments'], ensure_ascii=False)[:160]} → {returned}" + (", evidence" if hits else "") + (", a decoy" if lures else ""))  # fmt: skip
    head = steps[: first or 3]
    rest = steps[len(head) :]
    route = head + ([f"… {len(rest) - 3} more calls …"] if len(rest) > 3 else []) + rest[-3:]
    return {k: result.get(k) for k in ("correct", "grounded", "crashed")} | {
        "reason": (result.get("reason") or "")[:240],
        "calls": len(steps),
        "evidence_at": first,
        "found_by": steps[first - 1] if first else None,
        "decoy_reads": lured,
        "steps": route,
        "answer": (result.get("response") or "")[:300],
    }


# The anchor messages of the decoys on a task's facts' subjects: the near-misses a task rests beside.
DECOYS = """SELECT m.id, m.channel_id, m.ts, m.text FROM task_facts tf JOIN facts f ON f.id = tf.fact_id
JOIN facts d ON d.is_decoy = 1 AND lower(trim(d.subject)) = lower(trim(f.subject))
JOIN evidence e ON e.fact_id = d.id JOIN messages m ON m.id = e.message_id
WHERE tf.task_id = ? GROUP BY m.id ORDER BY m.ts_us, m.id"""


class GenerationEnv(vf.Env[PipelineConfig]):
    def __init__(self, settings: Config, store: Store):
        self.settings, self.store = settings, store
        self.used = used_names(settings.corpus, store.root)
        self.writable: list[str] = []  # the slots the author's tasks turn may write
        # GLM has its own gate, under the episode's: judges and the witness are not held up behind its solves.
        self.glm = asyncio.Semaphore(settings.author.solvers)
        dialects.CAPABILITY_NOTICE = NETWORK_NOTICE  # read by append_user_notice on every request
        super().__init__(settings.env)

    @property
    def world(self):
        return self.store.world

    async def setup(self, agents):
        for name in ("author", "judge", "solver", "witness"):
            getattr(agents, name).trainable = False

    # ------------------------------------------------------------------ authors
    async def author_turn(self, interaction, runtime, task_cls, context, attempt, first):
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
            raise ended(interaction.trace)
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
            state.quota = quota(cfg.taxonomy, cfg.tasks.styles, cfg.seed, cfg.task_count)
            state.phase = "organization"
            self.store.save()
            self.store.event(
                "premise_selected", company=state.premise.company, quota=[s.id for s in state.quota]
            )
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
        """What a task review depends on: the task, its gold rows, the text of the messages they rest on, and the text
        of the decoys on its facts' subjects; a revised decoy is a changed task."""
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
        decoys = [r["text"] for r in db.execute(DECOYS, (task_id,))]
        return digest([task, texts, decoys])

    def returned(self, task_id: str, probe: dict) -> dict:
        """A task as it comes back to the author: what it asks and answers and who asks it, its earlier versions with
        their rates, how it fared, code's measures, and the tries (GLM's, condensed; the witness's)."""
        db = self.world.db
        row = db.execute("SELECT t.question, t.answer_type, t.gold_sql, t.gold_json, u.real_name, u.title FROM tasks t JOIN users u ON u.id = t.actor_id WHERE t.id = ?", (task_id,)).fetchone()  # fmt: skip
        witness = self.store.state.witness.get(task_id, {})
        current = witness.get("key") == self.store.state.probes.get(task_id, {}).get("key")
        return {
            "question": row["question"], "answer_type": row["answer_type"], "gold": json.loads(row["gold_json"]),
            "gold_sql": row["gold_sql"], "asked_by": f"{row['real_name']} ({row['title']})",
            "facts": [f for (f,) in db.execute("SELECT fact_id FROM task_facts WHERE task_id = ? ORDER BY fact_id", (task_id,))],
        } | {k: v for k, v in probe.items() if k not in ("question", "issue_messages")} | {
            "witness_tries": [{"correct": w.get("correct"), "calls": w.get("calls"), "answer": (w.get("response") or "")[:200]} for w in witness.get("results", [])] if current else [],
        }  # fmt: skip

    def texts(self, message_ids) -> list:
        """The current text of these messages, in order."""
        marks = ", ".join("?" * len(message_ids))
        return [r[0] for r in self.world.db.execute(f"SELECT text FROM messages WHERE id IN ({marks}) ORDER BY id", list(message_ids))]  # fmt: skip

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
            texts = [self.world.db.execute("SELECT text FROM messages WHERE channel_id = ? AND ts = ?", (c, ts)).fetchone()[0] for c, ts in answer.messages]  # fmt: skip
            row = row.model_copy(update={"question": row.question + "\n\nThese messages bear on it:\n" + "\n".join(f"- {t}" for t in texts)})  # fmt: skip
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
        runs = await asyncio.gather(*(asyncio.gather(*(self.solve(agents, self.solver_task(t, copy, hint), seat) for _ in range(n))) for t in task_ids))  # fmt: skip
        if dead := [t for t, tries in zip(task_ids, runs) if all(o["crashed"] for o, _ in tries)]:
            raise RuntimeError(f"every try of {dead} crashed twice: the solver or its grading is failing")
        return dict(zip(task_ids, runs))

    async def assess(self, agents, attempt):
        """The final review: every task whose review is stale is solved and judged, and the whole world is judged."""
        state, n = self.store.state, self.settings.author.tries
        self.refresh_gold()
        keys = {t: self.task_key(t) for (t,) in self.world.db.execute("SELECT id FROM tasks ORDER BY id")}
        due = [t for t, key in keys.items() if state.task_reviews.get(t, {}).get("key") != key]
        previous = list(state.last_verdict.issues) if state.last_verdict else []
        # A task whose last probe the judge did not approve is reviewed with the issues that probe raised.
        previous += [i for t, raised in state.task_issues.items() if not state.probes.get(t, {}).get("approved", True) for i in raised if i not in previous]  # fmt: skip
        people = cards(state.cast)

        # The world's open issues: the last final review's, and those task reviews raised that name no task.
        workspace = [i.model_dump(mode="json") for i in previous if i.artifact == "workspace"]
        workspace += [i.model_dump(mode="json") for i in state.open_issues]
        world_payload = review_payload(self.world, "world", [], people=people, ledger=ledger_digest(self.world), **({"previous_issues": workspace} if workspace else {}))  # fmt: skip
        async with asyncio.TaskGroup() as group:
            world = group.create_task(self.review(agents, world_payload, attempt, {}, "world"))
            checked = group.create_task(self.judge_tasks(agents, due, keys, attempt, n, previous)) if due else None  # fmt: skip
        runs, judged = checked.result() if checked else ({}, None)
        verdict = merged([v for v in (judged, world.result()) if v is not None])
        state.open_issues = []  # the final review has seen them; what remains is in its verdict
        self.store.artifact(attempt, "verdict", verdict.model_dump(mode="json"))
        self.store.artifact(
            attempt, "acceptance", {"approved": verdict.approved, "reviewed_tasks": list(runs)}
        )
        return verdict

    # ------------------------------------------------------------------ the world, written in time order
    def author_context(self) -> dict:
        state = self.store.state
        return context_of(
            self.settings, state, self.store.root, state.drafts.get("organization"), self.writable
        )

    async def author_step(
        self, interaction, runtime, prompt: str, attempt: str, mode: str, extra=None
    ) -> None:
        """One author turn: code's memory of the world, rendered fresh (with `extra` memory files, such as the solver's
        tries), then the turn; the author works through its tools, and the world file is code's again when the turn
        ends."""
        memory = files(self.world, self.settings, self.author_context(), mode) | (extra or {})
        await upload(runtime, memory, "/task/memory")
        self.store.event("author_turn", attempt=attempt, mode=mode, day=self.store.state.day)
        segment = await interaction.turn(prompt)
        self.store.trace(interaction.trace)
        if segment.terminated:
            raise ended(interaction.trace)

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
        """The notes an attempt kept, and only those, back in the author's workspace: a day starts from where the
        last one closed, and a failed attempt's notes are gone."""
        kept = self.store.root / "attempts" / attempt / "notes"
        notes = {f"notes/{name}": (kept / name).read_bytes() for name in NOTES if (kept / name).exists()}
        await upload(runtime, notes, "/task/notes")

    async def setup_world(self, agents, runtime, task) -> None:
        """The premise and the organization, as documents, in the author's own session."""
        async with agents.author.interaction(task, runtime=runtime) as interaction:
            await self.premise_and_organization(interaction, runtime, "plan")
        self.store.trace(interaction.trace)
        require_session(interaction.trace)

    async def plan_world(self, agents, runtime) -> None:
        """The ledger, before day 1: storylines, events and facts, and the author's plan of them."""
        state, cfg = self.store.state, self.settings
        if not self.world.db.execute("SELECT 1 FROM world_meta WHERE key = 'chronological'").fetchone():
            with self.world.trial() as copy:
                start_clock(copy)
        attempt = self.store.reserve("plan", cfg.author.plan_attempts)
        task = WorldAuthorTask.create("plan", 0, self.world.path, self.author_context(), attempt)
        count = cfg.storylines
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
        require_session(interaction.trace)
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
        task = WorldAuthorTask.create("day", day, self.world.path, self.author_context(), attempt)
        async with agents.author.interaction(task, runtime=runtime) as interaction:
            prompt, errors = day_prompt(self.world, day, issues, state.feedback), []
            for _ in range(3):
                await self.author_step(interaction, runtime, prompt, attempt, "day")
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
            state.day, state.phase = (day + 1, "day") if day < total else (day, "tasks")
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
        )
        verdict = await self.review(agents, payload, attempt, {}, "world")
        # It rejects nothing, so every issue is a note for what is written next, blocking or not.
        state.issues = verdict.issues
        state.day, state.phase = state.day + 1, "day"
        self.store.finish_attempt(verdict.approved)

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
                self.world, "task", chunk, self.settings.taxonomy,
                solves=[result for t in chunk for result in runs[t]["results"]],
                previous_issues=[i.model_dump(mode="json") for i in previous if set(i.task_ids) & set(chunk)],
                **({"witness": witness} if witness else {}),
            )  # fmt: skip
            files = {f"{seat}_{t}_{k}.json": record for seat, tried in (("solver", runs), ("witness", witnessed)) for t in chunk if t in tried for k, trace_id in enumerate(tried[t]["traces"], 1) if (record := self.saved_trace(trace_id))}  # fmt: skip
            return await self.review(
                agents, payload, attempt, files, "tasks" if len(chunks) == 1 else f"tasks-{k}"
            )

        # A review of a few tasks at a time, the reviews side by side: each task passes on its own chunk's verdict,
        # and on its failure cases: a task whose grade accepts a wrong answer is ambiguous as asked.
        verdicts = await asyncio.gather(*(judged(k, chunk) for k, chunk in enumerate(chunks, 1)))
        issues = {t: Issue(artifact="tasks", task_ids=[t], defect="the answer judge accepts a wrong answer to it: " + "; ".join(failed), requested_change="ask it so that its gold, and only its gold, is the right answer: tell it apart from what the judge accepted", blocking=True) for t, failed in loose.items() if failed}  # fmt: skip
        verdicts = [v.model_copy(update={"issues": v.issues + [issues[t] for t in chunk if t in issues], "approved": v.approved and not any(t in issues for t in chunk)}) for chunk, v in zip(chunks, verdicts)]  # fmt: skip
        verdict_of = {t: v for chunk, v in zip(chunks, verdicts) for t in chunk}
        for task_id in due:  # the issues each task's latest review raised, for the final review
            state.task_issues[task_id] = [i for i in verdict_of[task_id].issues if task_id in i.task_ids]
        # An issue that names no task blocks none of them: it is the world's, and waits for the final review.
        seen_issues = {(i.artifact, i.defect) for i in state.open_issues}
        for issue in (i for v in verdicts for i in v.issues if not i.task_ids):
            if (issue.artifact, issue.defect) not in seen_issues:
                state.open_issues.append(issue)
                seen_issues.add((issue.artifact, issue.defect))
        for task_id in due:
            results = runs[task_id]["results"]
            rate = rates(results)
            approved = accepted_task(verdict_of[task_id], self.settings.acceptance, task_id)
            fit = next(r.level_fit for r in verdict_of[task_id].tasks if r.task_id == task_id)
            seen = rates(witnessed[task_id]["results"])["right_rate"] if task_id in witnessed else None
            if approved:
                state.task_reviews[task_id] = {"key": keys[task_id], **rate, "witness_right": seen, "level_fit": fit, "results": results}  # fmt: skip
            right = rate["right_rate"]
            self.store.event("task_reviewed", attempt=attempt, task_id=task_id, approved=approved, **rate, learnability=4 * right * (1 - right))  # fmt: skip
        return runs, merged(verdicts)

    async def grade(self, question: str, reference: dict, response: str):
        """The answer judge's grade of one response, with no observations: its `correct` is what a screen reads."""
        for again in (True, False):
            try:
                return await AnswerJudge(self.settings.answer_judge).evaluate(question=question, reference=reference, response=response, observations=[])  # fmt: skip
            except Exception:
                if not again:
                    raise

    async def screen(self, due, keys, attempt) -> dict[str, list[str]]:
        """Each task's failure cases (`failure_cases`), graded by the answer judge before any try: returns, per task,
        those it accepted, and its gold if it rejected that. Kept with the task's key, so a task is screened once."""
        state = self.store.state
        for t in [t for t in due if state.screens.get(t, {}).get("key") != keys[t]]:
            row = self.world.db.execute("SELECT question, answer_type, gold_json FROM tasks WHERE id = ?", (t,)).fetchone()  # fmt: skip
            rows = json.loads(row["gold_json"])
            gold, cases = failure_cases(self.world, t, rows)
            reference = {"answer_type": row["answer_type"], "rows": rows}
            graded = await asyncio.gather(*(self.grade(row["question"], reference, r) for r in [gold, *(r for _, r in cases)]))  # fmt: skip
            failed = (["its gold answer, judged wrong"] if not graded[0].parsed.correct else []) + [f"{name} ({response!r})" for (name, response), g in zip(cases, graded[1:]) if g.parsed.correct]  # fmt: skip
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

        if cfg.author.witness_tries and (fresh := [t for t in due if low(t) and state.witness.get(t, {}).get("key") != keys[t]]):  # fmt: skip
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
        self.store.state.solves[task_id] = {"key": key, "results": [r for r, _ in runs], "traces": [t.id for _, t in runs]}  # fmt: skip

    def saved_trace(self, trace_id: str) -> dict | None:
        """A saved solver trace, as the judge may read it."""
        path = self.store.root / "traces" / f"{trace_id}.json"
        return observable(json.loads(path.read_text())) if path.exists() else None

    async def probe(self, agents, attempt: str, ids: list[str]) -> dict:
        """The solver tries each of `ids` that changed since its last probe, and the judge reviews it on those runs; a
        task's latest probe is kept with its key, so a kept task, or a resume, is not probed again. A task's rates and
        fewest calls are kept on it, its runs and a valid review for the final review. Returns, per task, what
        hardening reads: its slot's cell, the tries, the judge's review, code's measures, whether the judge approved
        it, and its level's band of right-answer rates."""
        state, cfg = self.store.state, self.settings
        self.refresh_gold()
        keys = {t: self.task_key(t) for t in ids}
        if due := [t for t in ids if state.probes.get(t, {}).get("key") != keys[t]]:
            runs, judged = await self.judge_tasks(agents, due, keys, attempt, cfg.author.tries)
            measured, answers = self.store.difficulty(), self.store.release_rows("")[1]
            for task_id in due:
                outcomes = runs[task_id]["results"]
                rate = rates(outcomes)
                fewest = min((o["calls"] for o in outcomes if o["correct"]), default=None)
                self.world.db.execute(
                    "UPDATE tasks SET right_rate = ?, strict_rate = ?, min_calls = ? WHERE id = ?",
                    (rate["right_rate"], rate["strict_rate"], fewest, task_id),
                )
                review = next(r for r in judged.tasks if r.task_id == task_id)
                decoys = {(r["channel_id"], r["ts"]) for r in self.world.db.execute(DECOYS, (task_id,))}
                question = self.world.db.execute(
                    "SELECT question FROM tasks WHERE id = ?", (task_id,)
                ).fetchone()[0]
                before = state.probes.get(task_id, {})
                # The versions the task had before, each with how often GLM was right: the moves already tried.
                history = before.get("history", []) + (
                    [{"question": before["question"], "right_rate": before["right_rate"]}]
                    if before.get("question") not in (None, question)
                    else []
                )
                state.probes[task_id] = (
                    {"key": keys[task_id], "question": question, "history": history}
                    | {k: v for k, v in measured[task_id].items() if k not in ("level_fit", "right_rate", "strict_rate")}
                    | rate | {"solves": [try_digest(o, self.saved_trace(i), {tuple(m) for m in answers[task_id].messages}, decoys) for o, i in zip(outcomes, runs[task_id]["traces"])]}
                    | {"level_fit": review.level_fit, "valid": review.valid, "review": review.reason}
                    | {"issues": [f"{i.defect} {i.requested_change}" for i in judged.issues if task_id in i.task_ids]}
                    | {"issue_messages": sorted({m for i in judged.issues if task_id in i.task_ids for m in (*i.message_ids, *i.evidence_message_ids)})}
                    | {"approved": accepted_task(judged, cfg.acceptance, task_id), "band": cfg.tasks.bands[measured[task_id]["level"] - 1]}
                    | {"witness_right": rates(state.witness[task_id]["results"])["right_rate"] if state.witness.get(task_id, {}).get("key") == keys[task_id] else None}
                )  # fmt: skip
            self.store.event("probe", attempt=attempt, rates={t: state.probes[t]["right_rate"] for t in due})
            self.store.save()
        return {t: {k: v for k, v in state.probes[t].items() if k != "key"} for t in ids}

    async def harden(self, agents, turn, attempt: str, ids: list[str]) -> None:
        """The proposer and the solver: GLM tries each task and the judge reviews it; the author gets back each task
        whose right-answer rate misses its level's band while it has rounds left, and each task the judge did not
        approve, rewrites what it will, and the changed tasks are tried again, until none comes back. A task the judge
        did not approve comes back after its last round too; the rounds bound the loop. A round is a rewrite: it counts
        when the task changes (or a message its review names does), and a turn that changes none of them ends the loop,
        since the same tries would come back."""
        state, cfg = self.store.state, self.settings
        for turns in itertools.count():
            back = {}
            for t, r in (await self.probe(agents, attempt, ids)).items():
                move = band_move(r)
                given = state.task_rounds.get(t, 0)
                if move == "fix" or (move and given < cfg.author.task_rounds):
                    back[t] = self.returned(t, r) | {"move": move, "tries_left": max(0, cfg.author.task_rounds - given - 1)}  # fmt: skip
            if not back or turns > cfg.author.task_rounds:
                return
            self.allow(
                back,
                {t: answer_of(self.world, t) for t, r in back.items() if r["move"] in ("harder", "easier")},
            )
            named = {t: state.probes[t].get("issue_messages", []) for t in back}
            before = {t: (self.task_key(t), self.texts(named[t])) for t in back}
            tries = {f"memory/solves/{t}/{k}.json": json.dumps(record, ensure_ascii=False) for t in back for k, i in enumerate(state.solves[t]["traces"], 1) if (record := self.saved_trace(i))}  # fmt: skip
            await turn(harden_prompt(back), tries)
            changed = [t for t in back if (self.task_key(t), self.texts(named[t])) != before[t]]
            for t in changed:
                state.task_rounds[t] = state.task_rounds.get(t, 0) + 1
                if (
                    self.task_key(t) == before[t][0]
                ):  # only the world around it changed: try it afresh all the same
                    for cache in (state.probes, state.solves, state.witness):
                        if t in cache:
                            cache[t]["key"] = None
            self.store.save()
            if not changed:
                return

    def allow(self, ids, keep: dict | None = None) -> None:
        """The slots the author's next turn may write, and the answers those that came back to be harder or easier
        keep: its memory marks them and its tools read them per call."""
        self.writable = list(ids)
        atomic_json(self.store.root / "writable.json", {"ids": self.writable, "keep": keep or {}})

    def batches(self) -> list[list[str]]:
        """The world's slots in batches of `[tasks] batch`, each written and hardened in an interaction of its own."""
        ids, size = [s.id for s in self.store.state.quota], self.settings.tasks.batch
        return [ids[i : i + size] for i in range(0, len(ids), size)]

    async def write_tasks(self, agents, runtime) -> None:
        """One batch of tasks after the last day, in a fresh interaction that starts from the last block's notes: the
        author writes the batch's slots (after a resume, only those still empty), and hardens them against the
        solver's tries. A batch keeps one interaction within its limits however many tasks the world holds."""
        state, cfg = self.store.state, self.settings
        ids, total = self.batches()[state.batch], len(self.batches())
        await self.bring_notes(runtime, state.restore_point)
        attempt = self.store.reserve("tasks", 2 * total)  # a batch cut off is written again once
        self.allow(ids)
        task = WorldAuthorTask.create("tasks", 0, self.world.path, self.author_context(), attempt)
        empty = lambda: [t for t in ids if not self.world.db.execute("SELECT 1 FROM tasks WHERE id = ?", (t,)).fetchone()]  # noqa: E731  # fmt: skip
        async with agents.author.interaction(task, runtime=runtime) as interaction:

            async def turn(prompt, extra=None):
                await self.author_step(interaction, runtime, prompt, attempt, "tasks", extra)

            for n in range(3):
                if not (missing := empty()):
                    break
                self.allow(missing)
                await turn(tasks_prompt(cfg, missing) if n == 0 else f"These slots have no task yet: {missing}. Write them with world_add_task, then end your turn.")  # fmt: skip
            if missing := empty():
                raise ReviewLimit(f"the author wrote no task for {missing}")
            # A probe round can outlast the VM's idle timeout: the solver and the judge work, the author waits.
            async with keep_awake(runtime, cfg.author.keepalive):
                await self.harden(agents, turn, attempt, ids)
        self.store.trace(interaction.trace)
        require_session(interaction.trace)
        await self.keep_notes(runtime, attempt)
        state.batch, state.restore_point = state.batch + 1, attempt
        if state.batch == total:
            state.phase = "final"
        self.store.finish_attempt(True)

    async def fix_world(self, agents, runtime) -> None:
        """The final review, until the judge approves or its rounds run out. Its issues go to the author in one more
        interaction, opened at the first rejection; a fix may rewrite only the tasks the verdict names."""
        state, cfg = self.store.state, self.settings
        interaction = None
        async with contextlib.AsyncExitStack() as stack:
            await stack.enter_async_context(keep_awake(runtime, cfg.author.keepalive))
            while state.phase == "final":
                if (verdict := await self.final_world(agents)).approved:
                    break
                attempt = f"final-{state.rounds.get('final', 0):02d}"
                invalid = [{"task_id": r.task_id, "invalid": r.reason} for r in verdict.tasks if not r.valid]
                # Every issue goes back, the non-blocking ones too: the author fixes them or says why not.
                issues = [i.model_dump(mode="json") for i in verdict.issues] + invalid
                named = [s.id for s in state.quota if any(s.id in i.get("task_ids", [i.get("task_id")]) for i in issues)]  # fmt: skip
                self.allow(named)
                if interaction is None:
                    await self.bring_notes(runtime, state.restore_point)
                    task = WorldAuthorTask.create("tasks", 0, self.world.path, self.author_context(), attempt)
                    interaction = await stack.enter_async_context(
                        agents.author.interaction(task, runtime=runtime)
                    )
                await self.author_step(interaction, runtime, fix_prompt(issues, named), attempt, "tasks")
        if interaction is not None:
            self.store.trace(interaction.trace)
            require_session(interaction.trace)

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
        """One author agent and one VM for the whole world; one interaction per block of work. A session lost to its
        sandbox, tunnel or provider (or a VM gone mid-run) is none of the author's limits: a fresh VM takes the run up
        where its state says, as a resume does, and the attempt it cut off spends no round. A second loss in the same
        block stops the run."""
        state, last = self.store.state, None
        setup = SynthesizerTask.create(
            synthesizer.premise_context(self.settings, self.used), "setup", self.world.path
        )
        while True:
            try:
                async with agents.author.provision(setup) as runtime:
                    return await self.blocks(agents, runtime, setup)
            except (SessionLost, SandboxError) as error:
                if cleaning_up():  # a stop cancels the session's streams: it is a stop, not a loss
                    raise
                where = (state.phase, state.day, state.batch)
                attempt = self.store.lose()
                self.store.event(
                    "session_lost", attempt=attempt, error=str(error)[:2000], again=where != last
                )
                if where == last:
                    raise
                last = where

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
        while state.phase == "tasks":
            await self.write_tasks(agents, runtime)
        if state.phase == "final":
            await self.fix_world(agents, runtime)

    async def run(self, task, agents):
        return await self.author_world(agents)
