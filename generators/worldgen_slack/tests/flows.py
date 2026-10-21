"""The scripted runs: seats played by code (the authors, the proposer, the judge, the solver, the witness and the
answer judge), the agent's VM as files, and a world, the forge, a measure and a chapter run through the envs and the
CLI with them. The behaviour lock and the viewer's check run them too."""

import asyncio
import contextvars
import io
import json
import random
import re
import shutil
import tarfile
from collections import Counter
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from types import SimpleNamespace

import httpx
import openai
import pytest
import verifiers.v1 as vf
import verifiers.v1.judge as vf_judge
import verifiers.v1.utils.interrupt
from verifiers.v1.dialects.base import append_user_notice
from verifiers.v1.errors import SandboxError
from worldgen_slack.judge import AnswerGrade, Grading, PartGrade, PartReading, Reading
from worldgen_slack.judge import Commit as ReadCommit
from worldgen_slack.release import load_release
from worldgen_slack.taskset import SolverTask
from worldgen_slack.world import file_hash

from .. import session
from ..cli import failure, run_forge, run_label, run_world
from ..forge import tries as forge_tries
from ..forge.config import ForgeConfig
from ..forge.env import provenance as forge_provenance
from ..forge.judge import Ruling
from ..forge.proposer.task import ProposerTask
from ..forge.regrade import FAIL_FAST, regrade
from ..forge.tasks import Part, Task
from ..forge.tries import rates
from ..judge.verdict import PHASE_CRITERIA, Issue, TaskReview, Verdict
from ..session import ended, keep_awake
from ..store import ReviewLimit, SessionLost, Store
from ..workspace import Workspace
from ..world.author.memory import ledger_document
from ..world.author.task import AuthorTask
from ..world.cast import Gaps, Premise, Premises
from ..world.chapter import (
    ChapterConfig,
    ChapterEnv,
    Material,
    adopt,
    brief_slots,
    chapter_settings,
    same_schema,
    version_of,
)
from ..world.chapter import provenance as chapter_provenance
from ..world.clock import chapter_start, frozen_before, frozen_digest, meta, present, today, window
from ..world.config import WorldConfig
from ..world.env import WorldEnv, provenance
from ..world.ledger import BoardEntry, Event, Plan, PlanFact, Storyline, front_loaded, record_plan, unplanned
from ..world.organization import Channel, Organization, Person, Routine, channel_id
from ..world.profile import page as profile_page
from ..world.profile import profile
from ..world.style import style
from ..world.writing import Close, Commit, Conversation, PostLine, Reaction, post, quotas, revise
from .support import (
    contracts_settings,
    fails,
    forge_of,
    onward,
    refused,
    seed_person,
    with_state,
)


async def offline(*args, **kwargs):
    raise AssertionError("the offline checks call no model: script the agent or the judge")


# The flows' answer judge, played by code; `grounded` has it ground every answer in what the solver read.
GRADER = {"on": False, "loose": False, "blind": False, "grounded": False, "judge_fails": 0}


def scripted_reading(fields: dict) -> Reading:
    """Grader v2's reader, played by code: a right answer ("the answer") states every gold item of every part; any
    other commits a value that states none."""
    right = fields["response"] == "the answer"
    return Reading(
        parts=[
            PartReading(
                label=part["label"],
                committed=[ReadCommit(value=i["value"], item=i["item"]) for i in part["answer"]]
                if right
                else [ReadCommit(value="another answer")],
                hedged=False,
                holds_back=False,
                claims="unsettled" if part["answer_type"] == "status" else "none",
            )
            for part in fields["reference"]
        ],
        reason="scripted",
    )


def misreading_first(answer):
    """A reader that misreads each try's first reading (one part too many), then reads as `answer` does. Each try's
    grade runs in its own task, so a context variable is its own."""
    first = contextvars.ContextVar("first", default=True)

    def reading(fields: dict) -> Reading:
        read = answer(fields)
        if first.get():
            first.set(False)
            return read.model_copy(update={"parts": [*read.parts, read.parts[0]]})
        return read

    return reading


REAL_COMPLETE = vf.Judge.complete  # the judges' own call, before the flows script it


class FakeInference:
    """The model provider, played by code: each structured call answered by `answer(fields)` (a parsed answer, or None
    for one that does not parse) at a reported cost. Only the network is faked: the judge's own call runs whole,
    recording each paid call on its trace."""

    def __init__(self, answer, cost: float = 0.01):
        self.answer, self.cost, self.calls = answer, cost, 0
        self.beta = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=self.parse)))

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def parse(self, *, response_format, model, messages, **sampling):
        self.calls += 1
        parsed = self.answer(json.loads(messages[0]["content"].rsplit("\n", 1)[1]))
        message = SimpleNamespace(
            content=parsed.model_dump_json() if parsed else "", parsed=parsed, refusal=None
        )
        usage = SimpleNamespace(
            prompt_tokens=1200,
            completion_tokens=300,
            prompt_tokens_details=None,
            completion_tokens_details=None,
            cost=self.cost,
        )
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")], usage=usage)


@contextmanager
def live_judging(network: FakeInference):
    """The judges' own call, on a faked network."""
    scripted, client = vf.Judge.complete, vf_judge.build_async_openai
    vf.Judge.complete, vf_judge.build_async_openai = REAL_COMPLETE, (lambda config: network)
    try:
        yield network
    finally:
        vf.Judge.complete, vf_judge.build_async_openai = scripted, client


def gold_text(reference: dict) -> str:
    """A reference's gold as a right answer says it: its rows, or each part's under its label."""
    if "parts" in reference:
        return " | ".join(f"{p['label']}: " + gold_text(p) for p in reference["parts"])
    return ", ".join(str(r.get("answer")) for r in reference["rows"])


async def scripted_complete(self, messages, *, trace=None, schema=None, parse=None, **sampling):
    """The answer judge in the flows: it accepts a task's gold alone; a loose one a bare opener too, a blind one not even
    the gold. The forge's task judge approves a task unless its request asks something "precisely" (more than a part
    asks), and fails while `judge_fails` lasts. Any other judge call fails, as `offline` does."""
    text = messages if isinstance(messages, str) else messages[0].content
    if text.startswith("You judge one task"):
        if GRADER["judge_fails"]:
            GRADER["judge_fails"] -= 1
            raise openai.APIConnectionError(
                message="the judge's provider is unavailable",
                request=httpx.Request("POST", "https://inference"),
            )
        ruling = Ruling(
            realistic=True, one_reading=True, exact="precisely" not in text, supported=True, reason="r"
        )
        return vf.JudgeResponse(
            text=ruling.model_dump_json(), parsed=parse(vf.JudgeResponse(text="", parsed=ruling))
        )
    if text.startswith("Read a Slack answer"):  # grader v2's reader: a right answer states every gold item
        reading = scripted_reading(json.loads(text.rsplit("\n", 1)[1]))
        return vf.JudgeResponse(
            text=reading.model_dump_json(), parsed=parse(vf.JudgeResponse(text="", parsed=reading))
        )
    if not GRADER["on"] or not text.startswith("Grade a Slack answer"):
        return await offline()
    fields = json.loads(text.rsplit("\n", 1)[1])  # the fields are the prompt's last line
    response, answers = fields["response"], gold_text(fields["reference"])
    right = response in (
        f"{answers}.",
        "I can't answer that from anything you can see: it isn't there.",
    ) or response.startswith(f"{answers}, though")
    correct = (right and not GRADER["blind"]) or (GRADER["loose"] and response == "Thought process:")
    parts = [
        PartGrade(label=p["label"], correct=correct, abstained=False, committed=[], missing=[], extra=[])
        for p in fields["reference"].get("parts", [])
    ]
    grade = AnswerGrade(
        correct=correct,
        abstained=False,
        grounded=GRADER["grounded"],
        unsupported_asides=0,
        committed=[],
        missing=[],
        extra=[],
        reason="scripted",
        parts=parts,
    )
    return vf.JudgeResponse(
        text=grade.model_dump_json(),
        parsed=parse(vf.JudgeResponse(text="", parsed=grade)) if parse else grade,
    )


TestStore = Store  # the flows keep the real store: its traces, events and attempts


class FakeTrace(SimpleNamespace):
    """A scripted seat's trace, with what the store and the env read of a real one."""

    def __init__(self, id, role, record=None, **fields):
        base = dict(ok=True, info={}, errors=[], last_reply="", usage=None, is_completed=True)
        super().__init__(id=id, agent=SimpleNamespace(name=role), **(base | fields))
        self.record = record

    def to_record(self):
        task = getattr(self, "task", None)
        data = task.data.model_dump(mode="json", exclude_none=True) if task is not None else {}
        return self.record or {
            "id": self.id,
            "agent": {"name": self.agent.name, "config": {}},
            "task": {"data": data},
            "ok": self.ok,
            "nodes": [],
            "info": self.info,
        }

    def record_metric(self, name, value):
        self.info.setdefault("metrics", {})[name] = value


class ScriptedAgent:
    trainable = False
    traced = Counter()  # every scripted interaction's trace has an id of its own

    def trace(self, role, task=None):
        ScriptedAgent.traced[role] += 1
        return FakeTrace(f"{role}-{ScriptedAgent.traced[role]}", role, task=task)

    @asynccontextmanager
    async def provision(self, task):
        yield None

    @asynccontextmanager
    async def interaction(self, task, runtime):
        yield SimpleNamespace(trace=self.trace("agent", task))


class FakeRuntime:
    """The author's VM as files: what the env writes, the author reads and writes, and rm."""

    def __init__(self):
        self.files, self.pings = {}, 0

    async def write(self, path, data):
        self.files[path] = data

    async def read(self, path, max_bytes=None):
        if path not in self.files:
            raise SandboxError(f"no {path}")
        return self.files[path]

    async def run(self, argv, env):
        self.pings += argv == ["true"]
        if argv[:2] == ["rm", "-f"]:
            for path in argv[2:]:
                self.files.pop(path, None)
        elif argv[:2] == [
            "sh",
            "-c",
        ]:  # the env's upload: empty a directory, then unpack an archive into /task
            if cleared := re.match(r"rm -rf (\S+) && ", argv[2]):
                for path in [p for p in self.files if p.startswith(cleared.group(1) + "/")]:
                    del self.files[path]
            (archive,) = re.findall(r"tarfile -e (\S+) /task", argv[2])
            with tarfile.open(fileobj=io.BytesIO(self.files.pop(archive)), mode="r:gz") as unpacked:
                for member in unpacked.getmembers():
                    self.files["/task/" + member.name] = unpacked.extractfile(member).read()
        return SimpleNamespace(exit_code=0, stdout="", stderr="")


class ScriptedAuthor(ScriptedAgent):
    """A world author (or the forge's proposer) played by a script: each turn runs it against the turn's own world
    tools and VM. Its traces are named by the seat it plays, as the harness names them."""

    def __init__(self, script, role="author"):
        self.script, self.role = script, role

    @asynccontextmanager
    async def provision(self, task):
        yield FakeRuntime()  # as verifiers provisions a box: empty, the task's setup runs in its rollout

    @asynccontextmanager
    async def interaction(self, task, runtime):
        tools = None
        if isinstance(task, (AuthorTask, ProposerTask)):
            (tools,) = task.toolsets(task.config)
            await tools.setup()
            with_state(tools)

        async def turn(prompt):
            await self.script(prompt, tools, runtime)
            return SimpleNamespace(terminated=False)

        yield SimpleNamespace(trace=self.trace(self.role, task), turn=turn)


def setup_document(runtime) -> None:
    """The setup author's turn: the document its phase asks for, written where the env reads it."""
    context = json.loads(runtime.files["/task/input.json"])
    if context["phase"] == "premise":
        document = Premises(
            premises=[
                Premise(
                    company=f"{w} Software",
                    niche="n",
                    region="r",
                    size="s",
                    culture="c",
                    cast="x",
                    staffing={"software_developer": 4},
                )
                for w in ("Lattice", "Birch")
            ]
        )
    else:
        ids = [c["user_id"] for c in context["candidates"]]
        kinds = [Routine(kind=k, probability=0.5) for k in ("standup", "handoff", "lunch")]
        channels = [
            Channel(name="ops", type="public", members=ids, routines=kinds),
            Channel(name="leads", type="private", members=ids[:2], routines=kinds),
        ]
        document = Organization(
            people=[Person(user_id=u, title="Engineer", team="Platform") for u in ids],
            channels=channels,
            dm_routines=kinds,
        )
    runtime.files[f"/task/{context['phase']}.json"] = document.model_dump_json().encode()


async def judged(task, verdict):
    """A review's verdict as a judge seat returns it: the task's own setup lays out its files on a fake box, the verdict
    is written where the task reads it, and the task's finalize checks it."""
    runtime, trace = FakeRuntime(), ScriptedAgent().trace("judge", task)
    trace.state = SimpleNamespace(checked=True, calls=[])
    await task.setup(trace, runtime)
    runtime.files["/task/verdict.json"] = verdict.model_dump_json().encode()
    await task.finalize(trace, runtime)
    return trace


def flow_settings(root, seed):
    settings = contracts_settings(root)
    (root / "p.jsonl").write_text("".join(seed_person(i).model_dump_json() + "\n" for i in range(6)))
    (root / "t.jsonl").write_text("".join(seed_person(i).typing.model_dump_json() + "\n" for i in range(6)))
    personas = settings.personas.model_copy(update={"pool": 4})
    activity = settings.activity.model_copy(
        update={"messages": 45, "conversation_lines": 3, "dm_share": 0.4, "reaction_rate": 0.3}
    )
    return settings.model_copy(
        update={
            "seed": seed,
            "premise_count": 2,
            "personas": personas,
            "corpus": root / "corpus",
            "activity": activity,
        }
    )


def flow_run_settings(root):
    """The flows' settings: a 4-day world of 16 messages, a review after day 2."""
    base = flow_settings(root, 3)
    # S1's 17 cells, without needs: the flow's scripted tasks are not built to taxonomy v2
    cut = {"robustness": 2}
    flow = {
        n: c.model_copy(
            update={
                "levels": c.levels[: cut.get(n, 3)],
                "concepts": c.concepts[: cut.get(n, 3)],
                "needs": None,
            }
        )
        for n, c in base.taxonomy.items()
    }
    return base.model_copy(
        update={
            "taxonomy": flow,
            "calendar": base.calendar.model_copy(update={"days": 4}),
            "activity": base.activity.model_copy(update={"messages": 16}),
            "author": base.author.model_copy(
                update={"tolerance": 1.0, "share_tolerance": 1.0, "review_days": [2], "keepalive": 0.0005}
            ),
        }
    )


def flow_forge(settings, **plan) -> ForgeConfig:
    """The flows' forge on a world of these settings: 5 initial tasks, 2 a turn, one adapting pass, 4 student tries and
    2 of the verifier's, 2 tries at once."""
    return forge_of(settings, tasks=5, batch=2, adapt=1, tries=4, verify=2, solvers=2, **plan)


def taxonomy_toml(settings: WorldConfig) -> str:
    """A world's taxonomy and question styles as taxonomy.toml holds them (the flows' categories have no needs)."""
    lines = [f"styles = {json.dumps(settings.styles)}"]
    for name, category in settings.taxonomy.items():
        lines.append(f"\n[categories.{name}]")
        lines += [
            f"{k} = {json.dumps(v)}" for k, v in category.model_dump(mode="json", exclude_none=True).items()
        ]
    return "\n".join(lines) + "\n"


def world_seats():
    """The seats of a world with no tasks of its own: an author that sets up, plans, writes four days (revising what a
    review names) and fixes what the final review rejects; a judge with a note after day 2 that rejects the final
    review once. Returns the author, the judge, the messages fixed and the reviews made."""
    fixed, reviewed = [], []

    async def script(prompt, tools, runtime):
        if tools is None:
            return setup_document(runtime)

        def call(name, **kwargs):
            return tools._with_state(getattr(tools, name))(**kwargs)

        world = Workspace(tools.config.db_path)
        a, b, c, d = [r[0] for r in world.db.execute("SELECT id FROM users ORDER BY id")]
        ops, leads = channel_id("public", "ops", []), channel_id("private", "leads", [])

        def talk(channel, *lines):
            lines = [PostLine(**line) for line in lines]
            return call("post", conversation=Conversation(channel_id=channel, about="work", lines=lines))

        if tools.config.mode == "plan":
            events = [
                Event(id="e1", storyline="s2", title="audit window", day=3, time="10:00"),
                Event(id="e2", storyline="s1", title="service restored", day=2, time="14:00"),
            ]
            facts = [
                PlanFact(
                    id="f1",
                    storyline="s1",
                    subject="Release 4.2",
                    attribute="decision",
                    value="rollback",
                    anchor="rollback",
                    channel_id=ops,
                    author_id=a,
                    day=1,
                    summary="s",
                ),
                PlanFact(
                    id="f2",
                    storyline="s1",
                    subject="Release 4.2",
                    attribute="state",
                    value="service back",
                    channel_id=ops,
                    author_id=b,
                    day=2,
                    after=["f1"],
                    event="e2",
                    kind="happened",
                    summary="s",
                ),
                PlanFact(
                    id="f3",
                    storyline="s2",
                    subject="Audit",
                    attribute="window",
                    value="dry run",
                    anchor="dry run",
                    channel_id=leads,
                    author_id=a,
                    day=3,
                    event="e1",
                    kind="scheduled",
                    summary="s",
                ),
                PlanFact(
                    id="f4",
                    storyline="s2",
                    subject="Audit",
                    attribute="owner",
                    value="Owen",
                    channel_id=ops,
                    author_id=a,
                    day=3,
                    event="e1",
                    kind="scheduled",
                    summary="s",
                ),
            ]
            await call(
                "plan",
                ledger=Plan(
                    storylines=[
                        Storyline(id="s1", summary="the release"),
                        Storyline(id="s2", summary="the audit"),
                    ],
                    events=events,
                    facts=facts,
                    board=[],
                ),
            )
            await runtime.write("/task/notes/plan.md", b"arcs: the release, then the audit")
        elif tools.config.mode == "day":
            day = tools.config.day
            if day == 1:
                await talk(
                    ops,
                    dict(author_id=a, text="we go with the rollback", conveys=["f1"]),
                    dict(
                        author_id=b,
                        text="ok, notes tomorrow",
                        reply_to=0,
                        commits=[Commit(id="c1", text="post the notes", due_day=2)],
                    ),
                )
            if day == 2:
                await call("advance", to="evening")
                await talk(
                    ops,
                    dict(
                        author_id=b,
                        text="service back at {at:e2}; notes are up",
                        conveys=["f2"],
                        closes=[Close(id="c1", status="kept")],
                    ),
                )
            if day == 3:
                assert "message_ids" in prompt, "the review's note opens day 3"
                issue = json.loads(prompt[prompt.index("[{") : prompt.index("}]") + 2])[0]
                for message in issue["message_ids"]:
                    await call("revise", message_id=message, text="ok, notes tomorrow then")
                await talk(
                    leads, dict(author_id=a, text="dry run for the audit window at {at:e1}", conveys=["f3"])
                )
                await talk(
                    ops,
                    dict(
                        author_id=a,
                        text="Owen owns the audit at {at:e1}",
                        conveys=["f4"],
                        reactions=[Reaction(user_id=d, emoji="eyes")],
                    ),
                    dict(author_id=d, text=f"thanks <@{a}>"),
                )
            if day == 4:
                await talk(ops, dict(author_id=d, text="quiet day"))
            while today(world) == day:
                await call("advance", to=onward(world))
            await runtime.write("/task/notes/recap.md", f"day {day}: done".encode())
        elif prompt.startswith("The review rejected"):
            pinged = runtime.pings
            await asyncio.sleep(0.01)
            assert runtime.pings > pinged, "the VM is kept awake through the final review and its fix"
            issues = json.loads(prompt[prompt.index("[") : prompt.index("]\n") + 1])
            for issue in issues:
                for message in issue.get("message_ids", []):
                    await call("revise", message_id=message, text="ok, the notes come tomorrow")
                    fixed.append(message)
        world.close()

    class Judge(ScriptedAgent):  # a note after day 2; the final review rejects once, then approves
        async def run(self, task):
            payload, attempt = json.loads(task.config.payload), task.data.attempt
            reviewed.append(attempt)
            world = Workspace(task.config.tools.db_path)
            first, second = [m for (m,) in world.db.execute("SELECT id FROM messages ORDER BY id LIMIT 2")]
            world.close()
            issues = []
            if attempt in ("review-02-01", "final-01"):
                issues = [
                    Issue(
                        artifact="workspace",
                        message_ids=[second],
                        evidence_message_ids=[first],
                        defect="stiff",
                        requested_change="looser",
                        blocking=attempt == "final-01",
                    )
                ]
            verdict = Verdict(
                approved=not any(i.blocking for i in issues),
                tasks=[],
                issues=issues,
                criteria=dict.fromkeys(PHASE_CRITERIA[payload["phase"]], 1.0),
                summary="s",
            )
            return await judged(task, verdict)

    return ScriptedAuthor(script), Judge(), fixed, reviewed


@contextmanager
def episodes(agents):
    """The CLI's run functions on scripted seats: an episode sets its env up and runs it, with no model, and records a
    failure as verifiers' episodes do."""

    async def run_episode(self, task, context):
        await self.setup(agents)
        errors = []
        try:
            await self.run(task, agents)
        except Exception as error:
            errors.append(SimpleNamespace(type=type(error).__name__, message=str(error)))
        return SimpleNamespace(
            id="episode",
            traces=[],
            errors=errors,
            to_record=lambda: {"id": "episode", "errors": [vars(e) for e in errors]},
        )

    @asynccontextmanager
    async def serving(self):
        yield

    saved = vf.Env.run_episode, vf.Env.serving
    vf.Env.run_episode, vf.Env.serving = run_episode, serving
    try:
        yield
    finally:
        vf.Env.run_episode, vf.Env.serving = saved


async def world_run(root, settings, agents_of):
    """A world written through the CLI's own run: setup, the plan, four days with a review after day 2 whose note
    opens day 3, and a final review that rejects once and is fixed in the same session. It ends at the final review,
    and publishes nothing. The forge and the measure below start from it. Returns its run directory and settings."""
    ScriptedAgent.traced.clear()  # each flow numbers its traces from 1
    settings = settings.model_copy(update={"output": root / "world"})
    author, judge, fixed, reviewed = world_seats()

    with episodes(agents_of(judge=judge, author=author)):
        summary = await run_world(settings)
    assert summary["status"] == "complete" and not (settings.output / "release").exists(), (
        "a world run publishes nothing"
    )
    store = TestStore(settings.output, provenance(settings))
    state = store.state
    assert state.phase == "done", (state.phase, state.rounds)
    assert reviewed == ["review-02-01", "final-01", "final-02"] and len(fixed) == 1, (reviewed, fixed)
    assert store.world.db.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0 and not state.quota, (
        "no tasks"
    )
    assert [
        e["day"]
        for e in map(json.loads, (store.root / "progress.jsonl").read_text().splitlines())
        if e["event"] == "day_closed"
    ] == [1, 2, 3, 4]
    assert store.world.violations(complete=True) == []
    store.close()
    # A fix turn on a fresh VM, as after a lost session, starts from the notes of the last day closed: on a copy of
    # the run, its final review reopened and rejected once more.
    again = root / "world-again"
    shutil.copytree(settings.output, again)
    store, noted = TestStore(again, provenance(settings)), []
    store.state.phase = "final"

    async def fix(prompt, tools, runtime):
        noted.append({name: runtime.files.get(f"/task/notes/{name}") for name in ("plan.md", "recap.md")})

    class Rejecting(ScriptedAgent):  # the next final review rejects once, then approves
        async def run(self, task):
            world = Workspace(task.config.tools.db_path)
            first, second = [m for (m,) in world.db.execute("SELECT id FROM messages ORDER BY id LIMIT 2")]
            world.close()
            issues = (
                []
                if noted
                else [
                    Issue(
                        artifact="workspace",
                        message_ids=[second],
                        evidence_message_ids=[first],
                        defect="stiff",
                        requested_change="looser",
                        blocking=True,
                    )
                ]
            )
            return await judged(
                task,
                Verdict(
                    approved=not issues,
                    tasks=[],
                    issues=issues,
                    criteria=dict.fromkeys(PHASE_CRITERIA["world"], 1.0),
                    summary="s",
                ),
            )

    await WorldEnv(settings, store).fix_world(
        SimpleNamespace(author=ScriptedAuthor(fix), judge=Rejecting()), FakeRuntime()
    )
    kept = store.root / "attempts" / store.state.restore_point / "notes"
    assert noted == [{name: (kept / name).read_bytes() for name in ("plan.md", "recap.md")}] and all(
        noted[0].values()
    ), noted
    assert store.state.phase == "done", store.state.phase
    store.close()
    return settings.output, settings


async def run_flows(root):
    """One world written by a scripted author through the driver: setup, the plan, days that close only when done, a
    review after day 2 whose issues open day 3, a crash in that review and in day 3 with resumes from day 2's world
    and notes, a final rejection fixed in the same session. Then the forge on a world with no tasks of its own (its
    rounds, the screen, the tries, the witness, the hint, the archive, the release), a measure on the forge's output,
    and a chapter of a world with tasks of its own."""
    root.mkdir(parents=True)
    settings = flow_run_settings(root)
    seen, crash = [], {"armed": True, "review": True, "plans": 0}
    GRADER.update(on=True, loose=False, blind=False)

    class FlowJudge(ScriptedAgent):
        """The judge at its seat: it reads the review's input and writes its verdict; a review after day 2 is cut off
        once; the final review rejects the world once; a task review finds the vague candidates invalid."""

        async def run(self, task):
            payload, attempt, files = json.loads(task.config.payload), task.data.attempt, task.config.files
            seen.append(
                (
                    "review",
                    attempt,
                    payload["phase"],
                    "written_through" in payload,
                    "ledger" in payload,
                    bool(payload.get("previous_issues")),
                    tuple(t["id"] for t in payload["tasks"]),
                )
            )
            if payload.get("witness"):
                seen.append(
                    (
                        "witnessed",
                        attempt,
                        tuple(t["id"] for t in payload["tasks"]),
                        len(payload["witness"]),
                        tuple(sorted(f for f in files if f.startswith("witness_"))),
                    )
                )
            if attempt == "review-02-01" and crash["review"]:
                crash["review"] = False
                raise KeyboardInterrupt("simulated crash in a review")
            world = Workspace(task.config.tools.db_path)
            issues = []
            if payload["phase"] == "world" and attempt in ("review-02-02", "final-01"):
                first, second = [
                    m for (m,) in world.db.execute("SELECT id FROM messages ORDER BY id LIMIT 2")
                ]
                # A mid-run review's note blocks nothing, and still opens the next day.
                issues = [
                    Issue(
                        artifact="workspace",
                        message_ids=[second],
                        evidence_message_ids=[first],
                        defect="stiff",
                        requested_change="looser",
                        blocking=attempt != "review-02-02",
                    )
                ]
                if attempt == "final-01":  # a note beside the blocking defect, which the author hears too
                    issues.append(
                        Issue(
                            artifact="workspace",
                            defect="a little formal",
                            requested_change="loosen",
                            blocking=False,
                        )
                    )
            criteria = dict.fromkeys(PHASE_CRITERIA.get(payload["phase"], ()), 1.0)
            reviews = [
                TaskReview(
                    task_id=t["id"],
                    valid="vague" not in t["question"],
                    reason="r",
                    level_fit=3 if payload["phase"] == "task" else None,
                )
                for t in payload["tasks"]
            ]
            last = world.db.execute("SELECT MAX(id) FROM messages").fetchone()[0]
            world.close()
            issues += [
                Issue(
                    artifact="tasks",
                    task_ids=[r.task_id],
                    message_ids=[last],
                    defect="vague",
                    requested_change="sharpen",
                )
                for r in reviews
                if not r.valid
            ]
            verdict = Verdict(
                approved=not any(i.blocking for i in issues) and all(r.valid for r in reviews),
                tasks=reviews,
                issues=issues,
                criteria=criteria,
                summary="s",
            )
            return await judged(task, verdict)

    async def script(prompt, tools, runtime):
        """The author: sets up, plans, writes each day, writes tasks, hardens two of them, fixes what the review names."""
        if tools is None:
            return setup_document(runtime)

        def call(name, **kwargs):
            return tools._with_state(getattr(tools, name))(**kwargs)

        world = Workspace(tools.config.db_path)
        a, b, c, d = [r[0] for r in world.db.execute("SELECT id FROM users ORDER BY id")]
        ops, leads = channel_id("public", "ops", []), channel_id("private", "leads", [])
        assert runtime.files["/task/memory/now.md"].startswith(b"# Now: "), (
            "memory is rendered before every turn"
        )
        assert "/task/input.json" not in runtime.files, "the setup files are gone, after a resume too"
        assert "/task/memory/stale.md" not in runtime.files, (
            "memory is rendered whole: a page that is gone is gone"
        )
        runtime.files["/task/memory/stale.md"] = b"a page code no longer renders"
        seen.append(("turn", tools.config.mode, tools.config.day, prompt.split(":")[0][:40]))

        def talk(channel, *lines):
            lines = [PostLine(**line) for line in lines]
            return call("post", conversation=Conversation(channel_id=channel, about="work", lines=lines))

        async def close():
            while today(world) == tools.config.day:
                await call("advance", to=onward(world))

        if tools.config.mode == "plan":
            crash["plans"] += prompt.startswith("Before day 1")
            if crash["plans"] == 1:  # the first attempt never writes plan.md, and is rejected for it
                return
            assert "was rejected: " in prompt and "write /task/notes/plan.md" in prompt, "a retry is told why"
            events = [
                Event(id="e1", storyline="s2", title="audit window", day=3, time="10:00"),
                Event(id="e2", storyline="s1", title="service restored", day=2, time="14:00"),
            ]
            facts = [
                PlanFact(
                    id="f1",
                    storyline="s1",
                    subject="Release 4.2",
                    attribute="decision",
                    value="rollback",
                    anchor="rollback",
                    channel_id=ops,
                    author_id=a,
                    day=1,
                    summary="s",
                ),
                PlanFact(
                    id="f2",
                    storyline="s1",
                    subject="Release 4.2",
                    attribute="state",
                    value="service back",
                    channel_id=ops,
                    author_id=b,
                    day=2,
                    after=["f1"],
                    event="e2",
                    kind="happened",
                    summary="s",
                ),
                PlanFact(
                    id="f3",
                    storyline="s2",
                    subject="Audit",
                    attribute="window",
                    value="dry run",
                    anchor="dry run",
                    channel_id=leads,
                    author_id=a,
                    day=3,
                    event="e1",
                    kind="scheduled",
                    summary="s",
                ),
                PlanFact(
                    id="f4",
                    storyline="s2",
                    subject="Audit",
                    attribute="owner",
                    value="Owen",
                    channel_id=ops,
                    author_id=a,
                    day=3,
                    event="e1",
                    kind="scheduled",
                    summary="s",
                ),
            ]
            await call(
                "plan",
                ledger=Plan(
                    storylines=[
                        Storyline(id="s1", summary="the release"),
                        Storyline(id="s2", summary="the audit"),
                    ],
                    events=events,
                    facts=facts,
                    board=[],
                ),
            )
            await runtime.write("/task/notes/plan.md", b"arcs: the release, then the audit")
        elif tools.config.mode == "day":
            day = tools.config.day
            if day == 1 and prompt.startswith(
                "It is"
            ):  # first turn: a recap is written, but the day is left open
                await talk(
                    ops,
                    dict(author_id=a, text="we go with the rollback", conveys=["f1"]),
                    dict(
                        author_id=b,
                        text="ok, notes tomorrow",
                        reply_to=0,
                        commits=[Commit(id="c1", text="post the notes", due_day=2)],
                    ),
                )
                await runtime.write("/task/notes/recap.md", b"day 1: decided, still open")
                return
            if day == 2:
                await call("advance", to="evening")
                await talk(
                    ops,
                    dict(
                        author_id=b,
                        text="service back at {at:e2}; notes are up",
                        conveys=["f2"],
                        closes=[Close(id="c1", status="kept")],
                    ),
                )
            if day == 3:
                assert "message_ids" in prompt and "evidence_message_ids" in prompt, (
                    "the review's issues open day 3"
                )
                assert "hedging a decoy where its readers see it voids the near-miss" in prompt, (
                    "a decoy's issue is answered in notes"
                )
                if crash["armed"]:
                    await talk(ops, dict(author_id=c, text="morning"))
                    crash["armed"] = False
                    raise KeyboardInterrupt("simulated crash")
                assert b"arcs" in runtime.files["/task/notes/plan.md"], (
                    "the author's notes come back with its world"
                )
                issue = json.loads(prompt[prompt.index("[{") : prompt.index("}]") + 2])[0]
                for message in issue["message_ids"]:
                    await call("revise", message_id=message, text="ok, notes tomorrow then")
                seen.append(("revised", issue["message_ids"]))
                await talk(
                    leads, dict(author_id=a, text="dry run for the audit window at {at:e1}", conveys=["f3"])
                )
                await talk(
                    ops,
                    dict(
                        author_id=a,
                        text="Owen owns the audit at {at:e1}",
                        conveys=["f4"],
                        reactions=[Reaction(user_id=d, emoji="eyes")],
                    ),
                    dict(author_id=d, text=f"thanks <@{a}>"),
                )
            if day == 4:
                assert "message_ids" not in prompt, "a review's issues are delivered once"
            if day == 4 and "was rejected" not in prompt:
                # The first attempt closes the day but never writes its recap, so all 3 of its turns end rejected.
                if prompt.startswith("It is"):
                    await talk(ops, dict(author_id=d, text="quiet day"))
                    await close()
                    await runtime.write("/task/notes/scratch.md", b"day 4, first try")
                return
            if day == 4:
                assert "/task/notes/scratch.md" not in runtime.files, "a rejected attempt's notes are gone"
                assert "back at the start of the day: " in prompt and "recap.md" in prompt, (
                    "a retry is told why"
                )
                await talk(ops, dict(author_id=d, text="quiet day"))
            await close()
            await runtime.write("/task/notes/recap.md", f"day {day}: done".encode())
        elif prompt.startswith("The review rejected"):
            assert "hedging a decoy where its readers see it voids the near-miss" in prompt
            issues = json.loads(prompt[prompt.index("[") : prompt.index("]\n") + 1])
            seen.append(("fix issues", [i.get("blocking") for i in issues]))
            for issue in issues:
                for message in issue.get("message_ids", []):
                    await call("revise", message_id=message, text="ok, the notes come tomorrow")
            seen.append(("fixed", [m for i in issues for m in i.get("message_ids", [])]))

    agents = SimpleNamespace(judge=FlowJudge(), author=ScriptedAuthor(script))
    manifest = provenance(settings)
    store = TestStore(settings.output, manifest)
    env = WorldEnv(settings, store)
    await env.setup(agents)
    asked = [{"role": "user", "content": "write today"}]
    append_user_notice(asked)
    assert "call it again" in asked[0]["content"] and "do not retry" not in asked[0]["content"], (
        "a dropped tool call is called again, not given up"
    )
    for _ in range(2):  # the review after day 2 is cut off, then day 3
        # else the simulated crash did not happen
        with pytest.raises(KeyboardInterrupt):
            await WorldEnv(settings, store).run(None, agents)
        if store.state.phase == "review":
            assert store.state.day == 2 and store.state.rounds["day-02"] == 1, (
                "a cut-off review is redone, not its day"
            )
            store.close()
            store = TestStore(settings.output, manifest)
    after_day_2 = (
        Workspace(store.root / "attempts" / store.state.restore_point / "world.sqlite")
        .db.execute("SELECT COUNT(*) FROM messages")
        .fetchone()[0]
    )
    assert store.world.db.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == after_day_2 + 1, (
        "the crash left a partial day"
    )
    store.close()
    fails(
        Store, settings.output, manifest | {"lock_hash": "changed"}
    )  # a run resumes only with its own config
    store = TestStore(settings.output, manifest)
    assert store.state.phase == "day" and store.state.day == 3 and store.state.restore_point == "day-02-01"
    env = WorldEnv(settings, store)
    await env.run(None, agents)
    state = store.state
    assert state.phase == "done" and state.rounds["day-03"] == 2 and state.rounds["review-02"] == 2, (
        state.rounds
    )
    assert state.rounds["plan"] == 2 and not state.feedback
    assert not store.world.db.execute("SELECT 1 FROM messages WHERE text = 'morning'").fetchone(), (
        "the crashed day's start is gone"
    )
    assert [t[3][:6] for t in [s for s in seen if s[0] == "turn"] if t[1] == "day" and t[2] == 4] == [
        "It is ",
        "Day 4 ",
        "Day 4 ",
        "It is ",
    ], "a day needs its recap, and one that never gets it is written again"
    assert state.rounds["day-04"] == 2, state.rounds
    turns = [s for s in seen if s[0] == "turn"]
    assert [t[3][:6] for t in turns if t[1] == "day" and t[2] == 1] == ["It is ", "Day 1 "], (
        "an open day is not accepted"
    )
    first, second = [m for (m,) in store.world.db.execute("SELECT id FROM messages ORDER BY id LIMIT 2")]
    assert ("revised", [second]) in seen and ("fixed", [second]) in seen, (
        "only the messages to change are revised"
    )
    assert (
        store.world.db.execute("SELECT text FROM messages WHERE id = ?", (second,)).fetchone()[0]
        == "ok, the notes come tomorrow"
    )
    assert (
        "rollback" in store.world.db.execute("SELECT text FROM messages WHERE id = ?", (first,)).fetchone()[0]
    ), "the evidence stays"
    assert ("review", "review-02-02", "world", True, True, False, ()) in seen
    assert ("review", "final-01", "world", False, True, False, ()) in seen
    assert ("review", "final-02", "world", False, True, True, ()) in seen, (
        "the final review checks the last issues"
    )
    assert ("fix issues", [True, False]) in seen, (
        "a fix hears the review's notes as well as its blocking issues"
    )
    assert (store.root / "attempts" / "day-03-02" / "notes" / "recap.md").read_text() == "day 3: done"
    days = [json.loads(line) for line in (store.root / "progress.jsonl").read_text().splitlines()]
    assert [e["day"] for e in days if e["event"] == "day_closed"] == [1, 2, 3, 4]
    assert any(e["event"] == "interrupted" for e in days)
    assert (
        store.world.violations(complete=True) == []
        and not store.world.db.execute("SELECT 1 FROM tasks").fetchone()
    ), "a world run writes no task"
    ids = [r[0] for r in store.world.db.execute("SELECT id FROM messages ORDER BY ts_us")]
    assert ids == sorted(ids), "the world was written in time order"
    summary = store.summary("complete")
    assert (summary["tasks"], summary["rates"], summary["final_approved"]) == (0, {}, True), summary
    assert (store.root / "world-calls.jsonl").exists()
    # The author's VM is pinged while only the env works, and no more once the block ends.
    awake = FakeRuntime()
    async with keep_awake(awake, 0.01):
        await asyncio.sleep(0.05)
    pinged = awake.pings
    await asyncio.sleep(0.03)
    assert pinged >= 2 and awake.pings == pinged, (pinged, awake.pings)
    # A try that finished without an answer, an empty last message or the turn cap mid-search, is wrong and counted.
    graded = {"task_id": "t", "correct": False, "semantic_correctness": 0.0, "response": "an answer"}
    ends = [("an answer", "agent_completed"), ("", "agent_completed"), ("let me look further", "max_turns")]
    silent = [
        SolverTask.outcome(
            SimpleNamespace(ok=True, info={"evaluation": graded | {"response": r}}, stop_condition=stop)
        )["unanswered"]
        for r, stop in ends
    ]
    assert (
        silent == [False, True, True]
        and rates([{"correct": False, "semantic_correctness": 0.0, "unanswered": u} for u in silent])[
            "unanswered"
        ]
        == 2
    )
    assert run_label(Path("/x/data/v7-01/software")) == "worldgen-v7-01-software", (
        "a run's sandboxes are findable"
    )
    # A solver samples as a policy in training does, so its tries differ; a rate comes with its 90% interval, wide
    # at a few tries.
    seats = flow_forge(settings, world=root, output=root).env
    assert (
        seats.solver.sampling.temperature == 1.0 == seats.verifier.sampling.temperature
        and seats.verifier.sampling.reasoning_effort == "high"
    )
    # A judge thinks as long as an author before it writes its verdict (a final review of W2's smoke was cut off at
    # 16k, mid-call); a solver answers in 16k.
    assert (settings.env.judge.sampling.max_tokens, settings.env.author.sampling.max_tokens) == (
        64_000,
        64_000,
    ) and (seats.solver.sampling.max_tokens == 16_000)
    assert rates([{"correct": i < 6, "semantic_correctness": 0.0} for i in range(8)])["right_interval"] == [
        0.46,
        0.913,
    ]
    assert rates([{"correct": True, "semantic_correctness": 1.0}] * 4)["right_interval"] == [0.596, 1.0], (
        "4 of 4 still allows 0.6"
    )
    graded = [
        {"correct": True, "grounded": True, "semantic_correctness": 1.0},
        {"correct": False, "grounded": False, "semantic_correctness": -1.0},
        {"correct": False, "abstained": True, "grounded": False, "semantic_correctness": 0.0},
        {"correct": True, "grounded": False, "semantic_correctness": 0.0},
    ]
    assert (rates(graded)["strict_rate"], rates(graded)["abstain_rate"]) == (0.25, 0.25), (
        "strict is the share right and grounded, not the mean reward"
    )
    # A session lost to its sandbox is none of the author's limits (S3 stopped three times on Prime's dropped process
    # stream, each reported as the author's limits): its block is written again once and spends no attempt, and a
    # second loss in a row stops the run.
    rpc = SimpleNamespace(
        type="HarnessError", message="harness 'rlm': APIError: process stream RPC failed (unavailable)"
    )
    budget = SimpleNamespace(
        type="HarnessError",
        message="harness 'rlm': RuntimeError: ACP agent completed without committing a model turn: [token budget reached]",
    )
    assert isinstance(ended(SimpleNamespace(errors=[rpc])), SessionLost)
    assert all(
        isinstance(ended(t), ReviewLimit)
        for t in (
            SimpleNamespace(errors=[budget]),
            SimpleNamespace(errors=[rpc], is_truncated=True),
            SimpleNamespace(errors=[]),
        )
    ), "the author's own limits"

    async def cut_off(prompt):
        return SimpleNamespace(terminated=True)

    # else a dropped session was reported as the author's limits
    with pytest.raises(SessionLost):
        await env.take_turn(
            SimpleNamespace(trace=FakeTrace("cut-off", "author", ok=False, errors=[rpc]), turn=cut_off),
            FakeRuntime(),
            "write",
            "final-02",
            "fix",
        )
    blocks = []

    class Dropping(WorldEnv):  # the run's blocks, of which the first `losses` lose their session
        def __init__(self, key, losses):
            super().__init__(settings, store)
            self.key, self.losses = key, losses

        async def blocks(self, agents_, runtime, setup):
            blocks.append(store.reserve(self.key, 1))
            if len([b for b in blocks if b.startswith(self.key)]) <= self.losses:
                raise ended(SimpleNamespace(errors=[rpc]))
            store.finish_attempt(True)

    stopping = verifiers.v1.utils.interrupt
    stopping._cleaning_up = True  # a stop cancels the session's streams: it is not retried
    try:
        await Dropping("stop", 1).work_in_vm(agents)
        raise AssertionError("a stopped run was taken up again")
    except SessionLost:
        assert blocks == ["stop-01"] and "stop" not in store.state.lost, blocks
    finally:
        stopping._cleaning_up = False
    blocks.clear()
    await Dropping("block", 1).work_in_vm(agents)
    assert blocks == ["block-01", "block-02"] and store.state.lost == {"block": 1}, (blocks, store.state.lost)
    try:
        await Dropping("gone", 2).work_in_vm(agents)
        raise AssertionError("a block that lost its session twice went on")
    except SessionLost:
        assert blocks[2:] == ["gone-01", "gone-02"] and store.state.lost["gone"] == 2, blocks
    settings_ = (
        env.settings
    )  # a run that allows more losses (a long world's, a forge's 5) takes a block up again
    env.settings = settings_.model_copy(update={"max_losses": 3})
    blocks.clear()
    dropping = Dropping("many", 3)
    dropping.settings = env.settings
    await dropping.work_in_vm(agents)
    assert blocks == ["many-01", "many-02", "many-03", "many-04"] and store.state.lost["many"] == 3, blocks
    env.settings = settings_
    # The world the forge, the measure and the chapter start from: one with no tasks of its own.
    world_dir, world_settings = await world_run(
        root, settings, lambda **seats: SimpleNamespace(**(vars(agents) | seats))
    )
    await forge_flow(root / "forge", world_dir, settings)
    store.close()
    retried = SimpleNamespace(
        ok=False, errors=[SimpleNamespace(type="TaskError", message="malformed verdict")]
    )
    limit = SimpleNamespace(type="ReviewLimit", message="no valid ledger")
    assert failure("done", SimpleNamespace(errors=[], traces=[retried])) is None, (
        "a judge failure the review loop retried does not fail a finished run"
    )
    assert failure("day", SimpleNamespace(errors=[], traces=[retried])) == "TaskError: malformed verdict"
    assert failure("done", SimpleNamespace(errors=[limit], traces=[])) == "ReviewLimit: no valid ledger"
    # A chapter continues a world written with tasks of its own, as S3 was (tests/fixtures/parent, written by the flow
    # above as it ran before worlds left their tasks to the forge): its board and its notes are the parent's.
    parent = root / "parent"
    shutil.copytree(Path(__file__).parent / "fixtures/parent", parent)
    await chapter_flow(root / "chapter", parent, settings)
    GRADER["on"] = False


async def forge_flow(root, world_dir, settings):
    """The forge on a finished world, through the CLI as a person runs it: five initial tasks, labelled (verified,
    measured, judged), then one adapting pass that writes a harder, an easier and a repaired version, labelled too, and
    the release of every approved task. Along the way: a task code refuses and the proposer fixes; a verifier split
    tried a third time; a task whose student tries all crash, labelled again after a wait and then failed while the
    others go on, taken up by the next pass and, once its provider is back, by the forge run again, which publishes its
    release anew; a proposer session lost and taken up again; and a judge down for a whole pass, which stops the run,
    and the CLI takes it up again."""
    plan = flow_forge(settings, world=world_dir, output=root, skills=["temporal"])
    world = Workspace(world_dir / "world.sqlite")
    people = [r[0] for r in world.db.execute("SELECT id FROM users WHERE is_bot = 0 ORDER BY id")]

    def readable(user):
        reader = Workspace(world.path, actor=user)
        ids = [r[0] for r in reader.db.execute("SELECT id FROM messages WHERE is_deleted = 0 ORDER BY ts_us")]
        reader.close()
        return ids

    everything = [r[0] for r in world.db.execute("SELECT id FROM messages WHERE is_deleted = 0")]
    asker = next(u for u in people if set(everything) - set(readable(u)))
    seen, hidden = readable(asker), sorted(set(everything) - set(readable(asker)))
    posts = world.db.execute(
        f"""SELECT m.id, u.real_name, m.text FROM messages m JOIN users u ON u.id = m.user_id
        WHERE m.id IN ({", ".join(map(str, seen))}) AND u.is_bot = 0 AND m.user_id <> ? AND m.parent_id IS NULL
        ORDER BY m.ts_us""",
        (asker,),
    ).fetchall()
    names = list(dict.fromkeys(r["real_name"] for r in posts))
    first, second = posts[0], next(r for r in posts if r["real_name"] != posts[0]["real_name"])
    world.close()

    def one(label, answer, evidence, kind="text"):
        return Part(label=label, answer_type=kind, answer=answer, evidence=evidence)

    def written(task_id, version):
        """Each open id's task: a request of a family, its answer and the messages it rests on."""
        who = one("who posted it", [first["real_name"]], [first["id"]])
        tasks = {
            "i001": ("catch-up", "Who posted the first update I can see?", [who]),
            "i002": (
                "decisions",
                "Catch me up: who posted the first update, and who answered?",
                [who, one("who answered", [second["real_name"]], [second["id"]])],
            ),
            "i003": (
                "commitments",
                "Who has posted where I can read it?",
                [one("who posted", names[:2], [first["id"], second["id"]], "set")],
            ),
            "i004": (
                "status",
                "Who precisely wrote the second update?",
                [one("who wrote it", [second["real_name"]], [second["id"]])],
            ),
            "i005": (
                "catch-up",
                "Who wrote the second update I can see?",
                [one("who wrote it", [second["real_name"]], [second["id"]])],
            ),
            "h101": (
                "catch-up",
                "Before I report: who posted the first update, and who answered it?",
                [who, one("who answered it", [second["real_name"]], [second["id"]])],
            ),
            "e102": (
                "decisions",
                "Who answered the first update?",
                [one("who answered it", [second["real_name"]], [second["id"]])],
            ),
            "r103": (
                "status",
                "Who wrote the update after the first?",
                [one("who wrote it", [second["real_name"]], [second["id"]])],
            ),
        }
        family, question, parts = tasks[task_id]
        return Task(
            id=task_id, family=family, actor_id=asker, question=question, parts=parts, **(version or {})
        )

    turns = Counter()

    async def propose(prompt, tools, runtime):
        def call(name, **kwargs):
            return tools._with_state(getattr(tools, name))(**kwargs)

        assert "/task/memory/tasks.md" in runtime.files and "/task/memory/families.md" in runtime.files
        assert "## Coverage: " in runtime.files["/task/memory/tasks.md"].decode(), (
            "what the tasks so far cover"
        )
        # The world's task material: each value with the messages an answer names, the current one marked.
        found = await call("material")
        assert all(v["messages"] for g in found["groups"] for v in g["values"]) and any(
            v.get("current") for g in found["groups"] for v in g["values"]
        ), found
        assert any(first["real_name"] in person for person in found["people"]), found["people"]
        # Where the world's material is, by storyline, how its promises went, and how deep each value sits.
        assert {"chains", "collections", "by_storyline", "promises"} <= set(found["summary"]) and all(
            v["page"] >= 1 for g in found["groups"] for v in g["values"]
        ), found["summary"]
        allowed = tools.allowed()
        assert all(allowed["skills"][t] == "temporal" for t in allowed["ids"]) and (
            "temporal, the answer as of a moment" in prompt
        ), (
            "each id aims at the skill code dealt it, a version at its source's, and the prompt says what it needs"
        )
        if FAULTS["stop"]:  # a Ctrl-C mid-turn reaches the run as the proposer's stream cancelled
            FAULTS["stop"] -= 1
            verifiers.v1.utils.interrupt._cleaning_up = True
            raise SessionLost("process stream RPC failed (canceled): Request was cancelled")
        if allowed["versions"]:  # an adapting turn reads what to write
            page = runtime.files["/task/memory/adapt.md"].decode()
            assert all(f"## {t}: " in page for t in allowed["ids"]), page
            assert ("a harder version of i001" in page) == ("h101" in allowed["ids"]) and (
                "The judge refused it" in page
            ) == ("r103" in allowed["ids"]), page
            turns["adapt"] += 1
        for task_id in allowed["ids"]:
            task = written(task_id, allowed["versions"].get(task_id))
            if task_id == "i002":  # first as a request of another family than its id asks for
                with pytest.raises(ValueError, match="i002 asks for a decisions request"):
                    await call("add_task", task=task.model_copy(update={"family": "status"}))
            if task_id == "i001" and hidden:  # its first try rests on a message out of its asker's sight
                wrong = task.model_copy(
                    update={"parts": [one("who posted it", [first["real_name"]], [hidden[0]])]}
                )
                with pytest.raises(ValueError, match="its asker cannot read message"):
                    await call("add_task", task=wrong)
            await call("add_task", task=task)
        await runtime.write("/task/notes/forge.md", b"looked for a date the audit closes: none in Slack")
        if allowed["versions"] and FAULTS["session"]:  # cut off after it wrote its versions
            FAULTS["session"] -= 1
            raise SessionLost("the proposer's sandbox dropped its stream")

    # The student's tries, right by a schedule per task; i005's crash while its provider is down. The verifier reaches
    # every answer, but splits on i003 (its second try wrong), so a third decides.
    schedule = {
        "i001": "1111",
        "i005": "1111",
        "i002": "0000",
        "i003": "1010",
        "i004": "1111",
        "h101": "1100",
        "e102": "1010",
        "r103": "0110",
    }
    counts = Counter()

    def tries_of(seat):
        class Trying(ScriptedAgent):
            async def run(self, task):
                task_id = task.data.task_id
                counts[seat, task_id] += 1
                n = counts[seat, task_id]
                trace = self.trace(seat, task)
                if seat == "solver" and task_id == "i005" and FAULTS["crash"]:
                    trace.ok, trace.errors = (
                        False,
                        [SimpleNamespace(type="Crash", message="model stream ended")],
                    )
                    return trace
                right = (
                    (n != 2 or task_id != "i003")
                    if seat == "verifier"
                    else schedule[task_id][(n - 1) % 4] == "1"
                )
                reference = task.config.reference
                parts = [
                    {
                        "label": p.label,
                        "correct": right,
                        "abstained": False,
                        "committed": [],
                        "missing": [],
                        "extra": [],
                    }
                    for p in reference.parts
                ]
                trace.info = {
                    "evaluation": {
                        "task_id": task_id,
                        "execution_ok": True,
                        "semantic_correctness": 1.0 if right else -1.0,
                        "correct": right,
                        "grounded": right,
                        "calls": 1,
                        "reason": "r",
                        "response": "the answer" if right else "another answer",
                        "parts": parts,
                    },
                    "observations": [
                        {"tool": "read_channel", "arguments": {"channel_id": "C1"}, "output": {}}
                    ],
                }
                return trace

        return Trying()

    agents = SimpleNamespace(
        proposer=ScriptedAuthor(propose, role="proposer"),
        solver=tries_of("solver"),
        verifier=tries_of("verifier"),
        judge=ScriptedAgent(),
    )
    FAULTS["session"], GRADER["judge_fails"] = (
        1,
        16,
    )  # the judge fails all 4 attempts of the first pass's 4 rulings
    FAULTS["stop"], FAULTS["crash"] = 1, 1
    with episodes(agents):
        try:
            summary = await run_forge(plan)
        finally:
            verifiers.v1.utils.interrupt._cleaning_up = False
        assert summary["status"] == "interrupted", (
            "a stop is never resumed, though its stream ended as a loss"
        )
        summary = await run_forge(plan)  # rerun, as its person would
    assert summary["status"] == "complete", summary
    events = [json.loads(line) for line in (root / "progress.jsonl").read_text().splitlines()]
    kinds = Counter(e["event"] for e in events)
    assert kinds["session_lost"] == 1, "a lost session is taken up again"
    assert turns["adapt"] == 2, (
        "the versions a turn cut off had kept are taken, not written again: r103 alone is left for a turn"
    )
    assert (kinds["task_failed"], kinds["labels_retried"]) == (5 + 1 + 1, 2 + 2), (
        "the first pass failed on the judge (and i005 on its tries), then went on; i005 is labelled again twice after a"
        " wait, in its own pass and in the next, and fails both times"
    )
    store = TestStore(plan.output, forge_provenance(plan))
    forged = store.state.forged
    assert {t: r.get("label") for t, r in forged.items()} == {
        "i001": "solved",
        "i002": "unsolved",
        "i003": "window",
        "i004": "rejected",
        "i005": None,
        "h101": "window",
        "e102": "window",
        "r103": "window",
    }, forged
    assert "crashed" in forged["i005"]["failed"], "its tries all crashed: it fails, the others go on"
    first_release = load_release(root / "release")[1]
    assert len(store.state.verifier["i003"]["results"]) == 3, "a split is decided by a third try"
    assert forged["i004"]["verified"] and not forged["i004"]["ruling"]["exact"], "verified, but over-specific"
    assert {r.get("skill") for r in forged.values()} == {"temporal"}, "each kept with the skill it aimed at"
    assert [(forged[t]["set"], forged[t]["task"]["variant_of"]) for t in ("h101", "e102", "r103")] == [
        ("harder", "i001"),
        ("easier", "i002"),
        ("repaired", "i004"),
    ]
    calls = [json.loads(line) for line in (root / "world-calls.jsonl").read_text().splitlines()]
    assert any(c["tool"] == "add_task" and "cannot read" in (c["error"] or "") for c in calls) or not hidden
    _, rows, answers = load_release(root / "release")
    assert {r.task_id: r.set for r in rows} == {
        "i001": "initial",
        "i002": "initial",
        "i003": "initial",
        "h101": "harder",
        "e102": "easier",
        "r103": "repaired",
    }, "every approved task, with the set it came from"
    assert all(a.solution.seat == "verifier" and a.solution.calls for a in answers.values())
    assert next(r for r in rows if r.task_id == "i002").part_rates == {
        "who posted it": 0.0,
        "who answered": 0.0,
    }
    report = json.loads((root / "report.json").read_text())
    assert report["families"]["status"]["written"] == 2 and report["families"]["status"]["repaired"] == 1, (
        "each initial id asks for its family in turn, and a version keeps its source's"
    )
    assert report["proposer_notes"].startswith("looked for a date"), (
        "the proposer's notes, as its last turn kept them"
    )
    assert summary["labels"]["window"] == 4
    store.close()
    # Its provider back, the forge is run again: the task its labelling left failed is labelled, and the release is
    # published anew, the earlier one kept beside it.
    FAULTS["crash"] = 0
    with episodes(agents):
        summary = await run_forge(plan)
    assert summary["status"] == "complete" and summary["labels"]["solved"] == 2, summary
    events = [json.loads(line) for line in (root / "progress.jsonl").read_text().splitlines()]
    assert [e["tasks"] for e in events if e["event"] == "forge_reopened"] == [["i005"]]
    store = TestStore(plan.output, forge_provenance(plan))
    assert (store.state.phase, store.state.forged["i005"]["label"], store.state.batch) == (
        "done",
        "solved",
        1,
    ), "labelled in the last pass, with no version of it: the adapting passes are spent"
    store.close()
    _, rows, _ = load_release(root / "release")
    assert {r.task_id for r in rows} == {r.task_id for r in first_release} | {"i005"}
    assert load_release(root / "release.superseded")[1] == first_release
    # A regrade: every stored try graded again by grader v2 (its reader scripted to read a right answer as stating the
    # gold), with one gold split into answer and context. No try runs again, the labels stand, the release is anew.
    splits = root / "splits"
    splits.mkdir()
    (splits / "batch-01.json").write_text(
        json.dumps(
            [
                {
                    "task_id": "i001",
                    "split": [
                        {
                            "part": "who posted it",
                            "answer": [first["real_name"]],
                            "context": ["the first update"],
                            "changed": True,
                        }
                    ],
                }
            ]
        )
    )
    tried = sum(counts.values())
    v2 = plan.model_copy(update={"grading": Grading(version=2, ground=False)})
    # A regrade grades on the judges' own call, only the network faked, so what the call does once it is paid is tested
    # too: forge-w2's first regrade failed every grade there, after paying for it, four times over ($166, 2026-10-21).
    # A grader whose answers never parse: the regrade stops after its first FAIL_FAST failures, each paid once.
    with live_judging(FakeInference(lambda fields: None)) as broken:
        with pytest.raises(RuntimeError, match=f"its first {FAIL_FAST} grades all failed"):
            await regrade(v2, splits)
    assert broken.calls <= FAIL_FAST + plan.solvers, ("it stops, not grading every try", broken.calls)
    # A live sample: five tries graded and their cost measured, nothing changed. Its reader misreads each try's first
    # reading: the grade asks again, and both readings are paid and counted.
    with live_judging(FakeInference(misreading_first(scripted_reading))) as sampler:
        sampled = await regrade(v2, splits, sample=5)
    assert (sampled["status"], sampled["graded"], sampled["failed"], sampled["reread"]) == (
        "sampled",
        5,
        0,
        5,
    ), sampled
    assert sampler.calls == 10 and sampled["per_grade"] == 0.02, ("two readings a try, both counted", sampled)
    # The regrade: every try graded once, paid once, its cost measured; the labels stand.
    with live_judging(FakeInference(scripted_reading)) as network:
        summary = await regrade(v2, splits)
    assert (summary["status"], summary["moved"], summary["split_tasks"]) == ("complete", {}, 1), summary
    assert network.calls == summary["tries"] == summary["graded"] and summary["spent"] == pytest.approx(
        0.01 * summary["tries"]
    ), ("each grade paid once", network.calls, summary)
    assert summary["reread"] == 0, summary
    assert sum(counts.values()) == tried, "no try runs again"
    store = TestStore(plan.output, forge_provenance(plan))
    i001 = store.state.forged["i001"]
    assert i001["task"]["parts"][0]["context"] == ["the first update"] and not i001["gold_before_split"][
        0
    ].get("context")
    assert store.state.plans["forge"]["regraded"]["grading"]["version"] == 2
    assert all(r.get("regraded") and r.get("outcome") for r in store.state.solves["i001"]["results"])
    store.close()
    _, rows, answers = load_release(root / "release")
    assert answers["i001"].context == ["the first update"], "the release carries the split gold"


# The forge flow's faults, each spent once.
FAULTS = {"session": 0, "stop": 0, "crash": 0}


async def chapter_flow(root, parent, settings):
    """A chapter of the scripted world: its settings and brief; the parent adopted, frozen, with no tasks and its
    last day's notes; a plan that keeps the earlier storylines first, leaves the earlier facts, events and board as they
    are, plants the brief fresh and spreads only its own facts; the past never revised; a brief answer that needs
    hiding posted only where it is hidden; then the chapter written by a scripted author, its reviews reading the past
    as frozen (an issue on it blocks nothing and reaches no one), and a forge on the version it leaves."""
    ScriptedAgent.traced.clear()
    root.mkdir(parents=True)
    # A world of the first schema (before 2026-10-14) comes forward in place, once: its rows as they were.
    old = root / "schema-1.sqlite"
    shutil.copy(parent / "world.sqlite", old)
    aged = Workspace(old, writable=True)
    for sql in (
        "DROP TABLE queue",
        "DROP TABLE integrations",
        "DROP TABLE artifacts",
        "DROP TABLE usergroup_members",
        "DROP TABLE usergroups",
        "ALTER TABLE messages DROP COLUMN subtype",
        "ALTER TABLE users DROP COLUMN deactivated_us",
        "ALTER TABLE channels DROP COLUMN archived_us",
        "UPDATE world_meta SET value = '1' WHERE key = 'schema_version'",
    ):
        aged.db.execute(sql)
    rows = aged.db.execute("SELECT COUNT(*), group_concat(text) FROM messages").fetchone()[:]
    tasks = aged.db.execute("SELECT * FROM tasks ORDER BY id").fetchall()
    assert set(same_schema(aged)) == {
        "artifacts",
        "channels",
        "commitments",
        "facts",
        "integrations",
        "messages",
        "queue",
        "usergroup_members",
        "usergroups",
        "users",
    }
    assert (aged.migrate(), same_schema(aged), aged.migrate(), aged.violations()) == (1, [], 4, []), (
        "forward, once"
    )
    assert aged.db.execute("SELECT COUNT(*), group_concat(text) FROM messages").fetchone()[:] == rows
    # v3 remade the tasks table (a task may be composite): its rows as they were, the new columns empty.
    assert [tuple(t)[: len(tasks[0])] for t in aged.db.execute("SELECT * FROM tasks ORDER BY id")] == [
        tuple(t) for t in tasks
    ] and aged.db.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert {"family", "parent", "part", "label"} <= {
        r[1] for r in aged.db.execute("PRAGMA table_info(tasks)")
    }
    aged.close()
    taxonomy = root / "taxonomy.toml"  # the flows' own taxonomy, as a chapter's config names one
    taxonomy.write_text(taxonomy_toml(settings))
    chapter = ChapterConfig(
        base=Path("worldgen.toml"),
        world=parent,
        output=root / "unit",
        taxonomy=taxonomy,
        seed=11,
        days=2,
        messages=6,
        storylines=1,
        review_days=[1],
        material=[Material(category="semantic", level=3, needs={"values": 2, "hidden": True})],
    )
    world_config = settings.model_copy(update={"taxonomy": {}, "styles": []})  # a world's names no taxonomy
    cfg = chapter_settings(chapter, world_config)
    assert (cfg.storylines, cfg.author.review_days, cfg.activity.messages, cfg.seed) == (3, [5], 6, 11), cfg
    assert (cfg.taxonomy, cfg.styles) == (settings.taxonomy, settings.styles), "the chapter's own taxonomy"
    fails(
        chapter_settings,
        chapter.model_copy(update={"material": [Material(category="nowhere", level=1)]}),
        world_config,
    )
    fails(ChapterConfig, **chapter.model_dump() | {"review_days": [2]})  # a review after the last day
    before = Workspace(parent / "world.sqlite")
    slots, parent_quotas, parent_hash = (
        brief_slots(chapter, cfg, version_of(before) + 1),
        quotas(before, settings),
        file_hash(parent / "world.sqlite"),
    )
    board_rows = before.db.execute("SELECT COUNT(*) FROM board").fetchone()[0]
    before.close()
    assert (
        [s.id for s in slots] == ["semantic-l3-v2-1"] and slots[0].needs.fresh and slots[0].needs.values == 2
    )
    # A chapter on a world of the first schema adopts it brought forward.
    elder = root / "elder"
    shutil.copytree(parent, elder)
    shutil.copy(old, elder / "world.sqlite")
    aged = Workspace(elder / "world.sqlite", writable=True)
    for sql in (
        "DROP TABLE queue",
        "ALTER TABLE messages DROP COLUMN subtype",
        "UPDATE world_meta SET value = '1' WHERE key = 'schema_version'",
    ):
        aged.db.execute(sql)
    aged.close()
    on_elder = chapter.model_copy(update={"world": elder, "output": root / "on-elder"})
    elder_store = TestStore(on_elder.output, chapter_provenance(on_elder, cfg))
    adopt(elder_store, on_elder, cfg, slots)
    assert (meta(elder_store.world, "schema_version"), same_schema(elder_store.world)) == ("4", []), (
        "brought forward"
    )
    elder_store.close()
    store = TestStore(chapter.output, chapter_provenance(chapter, cfg))
    adopt(store, chapter, cfg, slots)
    world = store.world
    first = chapter_start(world)
    assert (first, today(world), present(world), meta(world, "version"), meta(world, "parent_hash")) == (
        5,
        5,
        window(world, 5, "early")[0],
        "2",
        parent_hash,
    )
    assert {d: q for d, q in quotas(world, cfg).items() if d < first} == parent_quotas, (
        "closed days keep their quotas"
    )
    assert sum(q["messages"] for d, q in quotas(world, cfg).items() if d >= first) == 6
    assert world.db.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0, "a chapter carries no tasks"
    assert (store.root / "attempts/source/notes/recap.md").read_text() == "day 4: done", (
        "the last day's notes"
    )
    frozen = meta(world, "frozen_digest")
    assert frozen == frozen_digest(world)
    ops, leads = channel_id("public", "ops", []), channel_id("private", "leads", [])
    a, other = [
        r[0]
        for r in world.db.execute(
            "SELECT user_id FROM members WHERE channel_id = ? ORDER BY user_id", (leads,)
        )
    ][:2]
    owner = world.db.execute("SELECT author_id FROM facts WHERE id = 'f3'").fetchone()[0]
    a, other = (owner, other if other != owner else a)
    document = ledger_document(world, {s.id for s in slots})
    assert document["board"] == [], "the earlier board stays out of the chapter's plan"
    plan = Plan.model_validate(document)
    mine = Storyline(id="s3", summary="the next audit")
    later = PlanFact(
        id="g1",
        storyline="s3",
        subject="Audit",
        attribute="owner",
        value="Mia",
        anchor="Mia",
        channel_id=leads,
        author_id=a,
        day=5,
        supersedes="f4",
        summary="s",
    )
    lure = PlanFact(
        id="d1",
        storyline="s3",
        subject="Audit",
        attribute="owner",
        value="Raj",
        anchor="Raj",
        channel_id=ops,
        author_id=a,
        day=5,
        decoy=True,
        summary="s",
    )
    entry = [BoardEntry(slot=slots[0].id, facts=["f4", "g1"])]

    def planned(**changes):
        return plan.model_copy(
            update={
                "storylines": [*plan.storylines, mine],
                "facts": [*plan.facts, later, lure],
                "board": entry,
            }
            | changes
        )

    def tried(act):
        with world.trial() as copy:
            act(copy)

    def recorded(document, brief=slots):
        tried(lambda copy: record_plan(copy, document, cfg, brief))

    unfinished = unplanned(world, cfg, slots)
    assert (
        any("3 storylines" in e for e in unfinished)
        and any("none is planned for day 5" in e for e in unfinished)
        and any("['semantic-l3-v2-1']" in e for e in unfinished)
    ), unfinished
    old_facts = planned().facts
    for expected, bad, brief in (
        ("keep ['s1', 's2'] in this order, first", planned(storylines=[mine, *plan.storylines]), slots),
        ("plan exactly 3 storylines", planned(storylines=plan.storylines), slots),
        (
            "f1 is an earlier chapter's, written and frozen",
            planned(facts=[old_facts[0].model_copy(update={"summary": "said again"}), *old_facts[1:]]),
            slots,
        ),
        (
            "an earlier chapter's days are written and frozen",
            planned(
                events=[*plan.events, Event(id="e9", storyline="s3", title="look back", day=3, time="10:00")]
            ),
            slots,
        ),
        (
            "planned for today (day 5) or later",
            planned(facts=[*plan.facts, later.model_copy(update={"day": 4}), lure]),
            slots,
        ),
        (
            "taking at least 4 values",
            planned(),
            [s.model_copy(update={"needs": s.needs.model_copy(update={"values": 4})}) for s in slots],
        ),
        ("first stated in this chapter", planned(board=[BoardEntry(slot=slots[0].id, facts=["f4"])]), slots),
    ):
        refused(expected, recorded, bad, brief)
    # The chapter's facts run through its own days (Friday 5 messages, Saturday 1): twelve on its first day leave its
    # second short, and an earlier chapter's facts are not spread over them.
    many = [later.model_copy(update={"id": f"g{k}"}) for k in range(12)]
    assert "days 6-6 first state 0 of the plan's 12 facts; they hold 17% of this chapter's messages" in (
        front_loaded(world, cfg, many, 5) or ""
    )
    assert front_loaded(world, cfg, [*[plan.facts[0]] * 10, later, lure], 5) is None
    recorded(planned())
    assert unplanned(world, cfg, slots) == []
    assert world.db.execute("SELECT COUNT(*) FROM board").fetchone()[0] == board_rows + 2, (
        "the earlier board stays"
    )
    assert [r[0] for r in world.db.execute("SELECT id FROM storylines ORDER BY position")] == [
        "s1",
        "s2",
        "s3",
    ]
    assert json.loads(meta(world, "segments"))[-1]["busy"] == [], (
        "the chapter's first plan marks its own event days"
    )
    old = world.db.execute("SELECT MIN(id) FROM messages").fetchone()[0]
    refused("earlier chapter, written and frozen", tried, lambda copy: revise(copy, old, "edited"))
    gaps, rng = Gaps(cfg.personas.gaps), random.Random(3)

    def said(*lines, channel=leads):
        tried(
            lambda copy: post(
                copy,
                Conversation(channel_id=channel, about="audit", lines=[PostLine(**line) for line in lines]),
                cfg,
                rng,
                gaps,
            )
        )

    said(dict(author_id=a, text="I heard Raj owns it", conveys=["d1"]), channel=ops)
    refused(
        "answers the brief's ['semantic-l3-v2-1']",
        said,
        dict(author_id=a, text="Mia has the audit now", conveys=["g1"]),
    )
    said(
        dict(author_id=other, text="who has the audit now?"),
        dict(author_id=a, text="Mia has it", reply_to=0, conveys=["g1"]),
    )
    assert frozen_digest(world) == frozen, "the chapter wrote nothing before its first day"
    assert style(world, frozen_before(world))["messages"] == 3, "the chapter's style is its own messages'"
    store.close()

    # The chapter written through the driver: a scripted author, and a judge whose issues name the frozen past.
    run = chapter.model_copy(update={"output": root / "run"})
    turns = []

    async def script(prompt, tools, runtime):
        def call(name, **kwargs):
            return tools._with_state(getattr(tools, name))(**kwargs)

        world = Workspace(tools.config.db_path)
        turns.append((tools.config.mode, tools.config.day, prompt))
        if tools.config.mode == "plan":
            if prompt.startswith("Before day 5"):
                assert (
                    "days 1-4 are written and frozen" in prompt
                    and b"Brief: what this chapter plants" in runtime.files["/task/memory/now.md"]
                    and b"semantic-l3-v2-1" in runtime.files["/task/memory/now.md"]
                )
                assert (
                    json.loads(runtime.files["/task/memory/ledger.json"])["board"] == []
                    and runtime.files["/task/notes/recap.md"] == b"day 4: done"
                )
                return  # nothing planned: the turn is told what is owed
            if "plan this chapter's facts" in prompt:
                await call("plan", ledger=planned())
                return  # plan.md left as the parent's: the turn is told to write it
            assert "write /task/notes/plan.md" in prompt, prompt
            await runtime.write(
                "/task/notes/plan.md", b"the chapter: the audit changes hands, said in a thread"
            )
            return
        day = tools.config.day
        assert f"day {day} of 6 (this chapter writes days 5-6)" in prompt, prompt
        # else a frozen message was revised
        with pytest.raises(ValueError, match=re.escape("earlier chapter")):
            await call("revise", message_id=old, text="edited")
        if day == 5:
            await call(
                "post",
                conversation=Conversation(
                    channel_id=ops,
                    about="audit",
                    lines=[PostLine(author_id=a, text="I heard Raj owns it", conveys=["d1"])],
                ),
            )
            await call(
                "post",
                conversation=Conversation(
                    channel_id=leads,
                    about="audit",
                    lines=[
                        PostLine(author_id=other, text="who has the audit now?"),
                        PostLine(author_id=a, text="Mia has it", reply_to=0, conveys=["g1"]),
                    ],
                ),
            )
        while today(world) == day:
            await call("advance", to=onward(world))
        await runtime.write("/task/notes/recap.md", f"day {day}: done".encode())

    reviewed = []

    class ScriptedJudge(ScriptedAgent):
        async def run(self, task):
            payload, attempt = json.loads(task.config.payload), task.data.attempt
            reviewed.append((attempt, "written_from" in payload, "written_through" in payload))
            judged_world = Workspace(task.config.tools.db_path)
            old_id = judged_world.db.execute("SELECT MIN(id) FROM messages").fetchone()[0]
            new_id = judged_world.db.execute("SELECT MAX(id) FROM messages").fetchone()[0]
            issues = [
                Issue(
                    artifact="workspace",
                    message_ids=[old_id],
                    defect="a stiff old line",
                    requested_change="loosen it",
                    blocking=True,
                )
            ]
            if attempt.startswith("review-"):
                issues.append(
                    Issue(
                        artifact="workspace",
                        message_ids=[new_id],
                        defect="a chapter line reads flat",
                        requested_change="loosen it",
                        blocking=False,
                    )
                )
            judged_world.close()
            verdict = Verdict(
                approved=False,
                tasks=[],
                issues=issues,
                criteria=dict.fromkeys(PHASE_CRITERIA["world"], 1.0),
                summary="s",
            )
            return await judged(task, verdict)

    agents = SimpleNamespace(
        judge=ScriptedJudge(), solver=ScriptedAgent(), witness=ScriptedAgent(), author=ScriptedAuthor(script)
    )
    manifest = chapter_provenance(run, cfg)
    store = TestStore(run.output, manifest)
    env = ChapterEnv(run, cfg, store, slots)
    await env.setup(agents)
    await env.run(None, agents)
    state = store.state
    assert state.phase == "done" and state.rounds["final"] == 1 and state.rounds["plan"] == 1, state.rounds
    assert [t[2][:20] for t in turns if t[0] == "plan"] == [
        "Before day 5 (Fri 20",
        "The plan is not done",
        "The plan is not done",
    ], [t[2][:20] for t in turns]
    day6 = next(t[2] for t in turns if t[:2] == ("day", 6))
    assert "a chapter line reads flat" in day6 and "a stiff old line" not in day6, (
        "a frozen issue reaches no one"
    )
    assert reviewed == [("review-05-01", True, True), ("final-01", True, False)], reviewed
    assert (store.root / "attempts/final-01/frozen_issues-world.json").exists(), "it is kept apart"
    world = store.world
    assert (
        meta(world, "frozen_digest") == frozen_digest(world) == frozen
        and world.violations(complete=True) == []
    )
    assert {d: q for d, q in quotas(world, cfg).items() if d < 5} == parent_quotas
    assert (
        world.db.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0
        and file_hash(parent / "world.sqlite") == parent_hash
    )
    # Its profile: the group the brief planted, in the version it was planted in, its current value where few read it.
    found = profile(world)
    owners = next(g for g in found["groups"] if (g["subject"], g["attribute"]) == ("Audit", "owner"))
    assert (owners["values"], owners["current"], owners["version"], owners["current_where"]) == (
        3,
        ["g1"],
        2,
        ["thread reply in private"],
    ) and "semantic-l3-v2-1" in owners["board"], owners
    leads_members = world.db.execute(
        "SELECT COUNT(*) FROM members WHERE channel_id = ?", (leads,)
    ).fetchone()[0]
    assert (
        owners["readers"] == [leads_members, 4]
        and found["world"]["version"] == 2
        and found["material"]["groups_3_values"] >= 1
    ), owners
    assert found["counts"]["private_truths"] >= 1 and "Audit" in profile_page(found), (
        "the owner changed, its current value in a private channel: a private truth, on the page"
    )
    tools_task = AuthorTask.create("fix", 0, world.path, env.tools_context(), "check")
    (tools,) = tools_task.toolsets(tools_task.config)
    assert not any(hasattr(tools, t) for t in ("add_task", "probe", "annotate")), (
        "a chapter's author writes no task"
    )
    world.db.execute(
        "UPDATE messages SET text = text || ', edited' WHERE id = ?", (old,)
    )  # the past, changed behind code
    # else a changed past passed the final check
    with pytest.raises(ValueError, match=re.escape("an earlier chapter's world changed")):
        await env.final_world(agents)
    store.close()
    fails(
        Store, run.output, chapter_provenance(run.model_copy(update={"days": 3}), cfg)
    )  # its own config only


@contextmanager
def scripted_judges():
    """No judge reaches a model while the checks run: the answer judge is played by code (GRADER), and any other judge
    call fails; and nothing waits for a provider or a sandbox to come back."""
    saved = vf.Judge.complete, session.LOSS_WAITS, forge_tries.WAITS
    vf.Judge.complete, session.LOSS_WAITS, forge_tries.WAITS = scripted_complete, (0,), (0, 0, 0)
    try:
        yield
    finally:
        vf.Judge.complete, session.LOSS_WAITS, forge_tries.WAITS = saved
