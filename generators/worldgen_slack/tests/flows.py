"""The scripted runs: seats played by code (the authors, the proposer, the judge, the solver, the witness and the
answer judge), the agent's VM as files, and a world, the forge, a measure and a chapter run through the envs and the
CLI with them. The behaviour lock and the viewer's check run them too."""

import asyncio
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

import pytest
import verifiers.v1 as vf
import verifiers.v1.utils.interrupt
from verifiers.v1.dialects.base import append_user_notice
from verifiers.v1.errors import SandboxError
from worldgen_slack import slack
from worldgen_slack.judge import EMPTY, AnswerGrade
from worldgen_slack.release import load_release
from worldgen_slack.taskset import SolverTask
from worldgen_slack.world import file_hash

from ..cli import failure, run_forge, run_label, run_world
from ..forge.config import ForgeConfig
from ..forge.difficulty import failure_cases
from ..forge.env import ForgeEnv, bucket, measured_level
from ..forge.env import measured as forge_measured
from ..forge.env import provenance as forge_provenance
from ..forge.profile import profile
from ..forge.proposer.task import ProposerTask
from ..forge.release import miss_class, publish, release_rows
from ..forge.tasks import Annotation, Task, forge_task, record_task
from ..forge.tries import rates
from ..judge.verdict import PHASE_CRITERIA, READING, Issue, TaskReview, Verdict, read_one_way
from ..session import ended, keep_awake
from ..store import ReviewLimit, SessionLost, Store
from ..workspace import Workspace
from ..world.author.memory import ledger_document
from ..world.author.task import AuthorTask
from ..world.board import Slot
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
from ..world.style import style
from ..world.writing import Close, Commit, Conversation, PostLine, Reaction, post, quotas, revise
from .support import (
    contracts_settings,
    fails,
    forge_of,
    onward,
    refused,
    scripted_task,
    seed_person,
    with_state,
)


async def offline(*args, **kwargs):
    raise AssertionError("the offline checks call no model: script the agent or the judge")


# The flows' answer judge, played by code; `grounded` has it ground every answer in what the solver read.
GRADER = {"on": False, "loose": False, "blind": False, "grounded": False}


async def scripted_complete(self, messages, *, trace=None, schema=None, parse=None, **sampling):
    """The answer judge in the flows: it accepts a task's gold alone; a loose one a bare opener too, a blind one not even
    the gold. Any other judge call fails, as `offline` does."""
    text = messages if isinstance(messages, str) else messages[0].content
    if not GRADER["on"] or not text.startswith("Grade a Slack answer"):
        return await offline()
    fields = json.loads(text[text.rindex("reason.\n") + len("reason.\n") :])
    response, answers = (
        fields["response"],
        ", ".join(str(r.get("answer")) for r in fields["reference"]["rows"]),
    )
    right = response in (
        f"{answers}.",
        "I can't answer that from anything you can see: it isn't there.",
    ) or response.startswith(f"{answers}, though")
    correct = (right and not GRADER["blind"]) or (GRADER["loose"] and response == "Thought process:")
    grade = AnswerGrade(
        correct=correct,
        abstained=False,
        grounded=GRADER["grounded"],
        unsupported_asides=0,
        committed=[],
        missing=[],
        extra=[],
        reason="scripted",
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
    """The flows' forge on a world of these settings: 3 tries of a candidate, one task in a review, 2 tries at once,
    and level 3's band starting at 0, so the witness tries its tasks."""
    return forge_of(
        settings,
        bands=[(0.75, 1.0), (0.5, 0.75), (0.0, 0.5)],
        tries=3,
        review_chunk=1,
        solvers=2,
        keepalive=0.0005,
        **plan,
    )


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


def forge_seats(note_on=None) -> dict:
    """The seats of a plain forge: a proposer that writes one lookup task for each open id, a judge that approves every
    task (with a note on message `note_on`, when given), and a solver and a witness that answer right."""

    async def propose(prompt, tools, runtime):
        def call(name, **kwargs):
            return tools._with_state(getattr(tools, name))(**kwargs)

        world = Workspace(tools.config.db_path)
        (actor,) = world.db.execute("SELECT id FROM users WHERE is_bot = 0 ORDER BY id LIMIT 1").fetchone()
        world.close()
        for open_id in re.findall(r"r\d\d-\d\d", prompt):
            await call(
                "add_task",
                task=Task(
                    id=open_id,
                    category="lookup",
                    level=1,
                    actor_id=actor,
                    question="Who is the newest person here?",
                    answer_type="text",
                    gold_sql="SELECT real_name AS answer FROM users ORDER BY id DESC LIMIT 1",
                    intent="structure",
                ),
            )

    class TaskJudge(ScriptedAgent):
        async def run(self, task):
            payload = json.loads(task.config.payload)
            notes = (
                [
                    Issue(
                        artifact="workspace",
                        message_ids=[note_on],
                        defect="a stiff old line",
                        requested_change="loosen it",
                        blocking=False,
                    )
                ]
                if note_on
                else []
            )
            reviews = [
                TaskReview(task_id=t["id"], valid=True, reason="r", level_fit=3) for t in payload["tasks"]
            ]
            verdict = Verdict(
                approved=True,
                tasks=reviews,
                issues=notes,
                criteria=dict.fromkeys(PHASE_CRITERIA[payload["phase"]], 1.0),
                summary="s",
            )
            return await judged(task, verdict)

    class RightSolver(ScriptedAgent):
        async def run(self, task):
            evaluation = {
                "task_id": task.data.task_id,
                "execution_ok": True,
                "semantic_correctness": 1.0,
                "correct": True,
                "grounded": True,
                "calls": 1,
                "reason": "r",
                "response": "an answer",
            }
            return FakeTrace(self.trace("solver").id, "solver", info={"evaluation": evaluation}, task=task)

    return dict(
        judge=TaskJudge(),
        solver=RightSolver(),
        witness=RightSolver(),
        proposer=ScriptedAuthor(propose, role="proposer"),
        author=ScriptedAgent(),
    )


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
    seen, crash, solved, finished = [], {"armed": True, "review": True, "plans": 0}, Counter(), Counter()
    GRADER.update(on=True, loose=False, blind=False)
    # The solver: every other try is right, and a candidate hard to find only when shown its evidence. One try of
    # r02-01 crashes twice and is left out of its rates.
    crashes = {("forge", "r02-01"): {2, 3}}

    class ScriptedSolver(ScriptedAgent):
        async def run(self, task):
            key = (store.state.active_attempt, task.data.task_id)
            solved[key] += 1
            name = "-".join([*key, str(solved[key])])
            if solved[key] in crashes.get(key, ()):
                return FakeTrace(
                    name,
                    "solver",
                    ok=False,
                    errors=[SimpleNamespace(message="model stream ended")],
                    task=task,
                )
            finished[key] += 1
            right, grounded = finished[key] % 2 == 1, True
            if (
                "hard to find" in task.data.prompt
            ):  # a forge candidate it answers only when shown the evidence
                right = "These messages bear on it" in task.data.prompt
            evaluation = {
                "task_id": task.data.task_id,
                "execution_ok": True,
                "semantic_correctness": float(right and grounded),
                "correct": right,
                "grounded": grounded,
                "calls": 2,
                "reason": "r",
                "response": "an answer",
            }
            # A search that misses, then a read that shows the task's evidence, if it rests on any.
            found = [{"channel": c, "ts": ts} for c, ts in task.config.reference.messages]
            observations = [
                {"tool": "search_messages", "arguments": {"query": "release"}, "output": {"items": []}},
                {"tool": "read_channel", "arguments": {"channel_id": "C1"}, "output": {"items": found}},
            ]
            record = {
                "id": name,
                "agent": {"name": "solver", "config": {}},
                "task": {"data": {"task_id": task.data.task_id}},
                "ok": True,
                "nodes": [],
                "info": {"observations": observations, "evaluation": evaluation},
            }
            return FakeTrace(name, "solver", record=record, info={"evaluation": evaluation}, task=task)

    witnessed = Counter()

    class ScriptedWitness(ScriptedAgent):  # the stronger solver answers right
        async def run(self, task):
            key = (store.state.active_attempt, task.data.task_id)
            witnessed[key] += 1
            name = "-".join(["witness", *key, str(witnessed[key])])
            evaluation = {
                "task_id": task.data.task_id,
                "execution_ok": True,
                "semantic_correctness": 1.0,
                "correct": True,
                "grounded": True,
                "calls": 1,
                "reason": "r",
                "response": "an answer",
            }
            record = {
                "id": name,
                "agent": {"name": "witness", "config": {}},
                "task": {"data": {"task_id": task.data.task_id}},
                "ok": True,
                "nodes": [],
                "info": {"observations": [], "evaluation": evaluation},
            }
            return FakeTrace(name, "witness", record=record, info={"evaluation": evaluation}, task=task)

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

    forging = {"vague": False}  # a forge whose every candidate the judge rejects

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
        elif tools.config.mode == "forge":
            # The proposer: the board and its notes come with the frozen world, which it reads and may not write.
            ids = re.findall(r"r\d\d-\d\d", prompt)
            assert (
                b"Board" in runtime.files["/task/memory/board.md"] and "/task/notes/plan.md" in runtime.files
            ), "the board and the world author's notes come to the proposer"
            assert (
                b"## lookup" in runtime.files["/task/memory/taxonomy.md"]
                and b"- level 1: one message" in runtime.files["/task/memory/taxonomy.md"]
            ), "and the taxonomy"
            seen.append(("forge", tuple(ids), "Round" in prompt))
            if (
                ids[0] == "r02-01" and not forging["vague"]
            ):  # the archive: what the last round kept, and why it dropped the rest
                archive = runtime.files["/task/memory/archive.md"].decode()
                assert (
                    "holds 1 of the 30 tasks it aims for" in prompt
                    and "r01-01" in archive
                    and "kept in lookup·L2 as plain·structure" in archive
                    and "dropped: the judge did not approve it" in archive
                    and "lookup: L1 0/2 · L2 1/2" in archive
                ), archive
            assert not any(hasattr(tools, w) for w in ("plan", "post", "advance", "revise", "edit")), (
                "the proposer is served no tool that writes the workspace"
            )
            # else a candidate outside the round's open ids was written
            with pytest.raises(ValueError, match=re.escape("open ids")):
                await call(
                    "add_task",
                    task=scripted_task(settings, {"id": "elsewhere", "category": "lookup", "level": 1}, a),
                )
            if ids[0] == "r01-01":
                first = world.db.execute(
                    "SELECT id FROM messages WHERE text LIKE '%rollback%' ORDER BY ts_us LIMIT 1"
                ).fetchone()[0]
                await call(
                    "annotate",
                    fact=Annotation(
                        id="n1",
                        storyline="s1",
                        subject="Release 4.2",
                        attribute="call",
                        value="roll it back",
                        anchor="rollback",
                        message_ids=[first],
                        summary="s",
                    ),
                )
                for bad, why in (
                    ({"anchor": "nowhere in it"}, "must contain"),
                    (
                        {"subject": "Elsewhere", "supersedes": "nothing"},
                        "which it supersedes, does not exist",
                    ),
                ):
                    # else an annotation was accepted: {bad}
                    with pytest.raises(ValueError, match=re.escape(why)):
                        await call(
                            "annotate",
                            fact=Annotation(
                                id="n2",
                                storyline="s1",
                                subject="Release 4.2",
                                attribute="call",
                                value="other",
                                anchor="rollback",
                                message_ids=[first],
                                summary="s",
                            ).model_copy(update=bad),
                        )
                # A search for the question's own words reaches the answer first: level 1, whatever it is called.
                notes = Task(
                    id=ids[0],
                    category="search",
                    level=2,
                    actor_id=a,
                    question="When were the notes up for release 4.2?",
                    answer_type="text",
                    gold_sql="SELECT value AS answer FROM facts WHERE id = 'f2'",
                    facts=["f2"],
                    intent="when",
                )
                # else a shortcut was written above level 1
                with pytest.raises(ValueError, match=re.escape("so it is a level-1 task")):
                    await call("add_task", task=notes)
                assert (await call("add_task", task=notes.model_copy(update={"level": 1})))["panel"][
                    "shortcut"
                ]
                # The proposer reads the world's inventory, and tries a candidate on a copy that writes nothing.
                inventory = await call("profile")
                assert inventory["world"]["messages"] and "material" in inventory, sorted(inventory)
                probed = await call(
                    "probe",
                    task=notes.model_copy(
                        update={"id": ids[1], "level": 1, "question": "When did the 4.2 notes go up?"}
                    ),
                )
                assert (
                    probed["gold"]
                    and "failure_cases" in probed
                    and not world.db.execute("SELECT 1 FROM tasks WHERE id = ?", (ids[1],)).fetchone()
                ), "a probe writes nothing"
                # else a candidate with no intent was written
                with pytest.raises(ValueError, match=re.escape("name its intent")):
                    await call("add_task", task=notes.model_copy(update={"level": 1, "intent": None}))
                # A task on no facts above level 1 names the answer a hasty reader gives: another one.
                newest = Task(
                    id=ids[0],
                    category="lookup",
                    level=2,
                    actor_id=a,
                    question="Who is the newest person here?",
                    answer_type="text",
                    gold_sql="SELECT real_name AS answer FROM users ORDER BY id DESC LIMIT 1",
                    intent="structure",
                )
                for near, why in (([], "near_sql"), ([newest.gold_sql], "returns its gold answer")):
                    # else a near-miss query was accepted: {near}
                    with pytest.raises(ValueError, match=re.escape(why)):
                        await call("add_task", task=newest.model_copy(update={"near_sql": near}))
            for n, open_id in enumerate(ids):  # a candidate the judge approves, and a vague one it does not
                question = f"Who is the newest person here, round {open_id}?" + (
                    " vague" if n or forging["vague"] else ""
                )
                written = await call(
                    "add_task",
                    task=Task(
                        id=open_id,
                        category="lookup",
                        level=2,
                        actor_id=a,
                        question=question,
                        answer_type="text",
                        gold_sql="SELECT real_name AS answer FROM users ORDER BY id DESC LIMIT 1",
                        near_sql=["SELECT real_name AS answer FROM users ORDER BY id LIMIT 1"],
                        intent="structure",
                    ),
                )
                assert (
                    set(written["features"])
                    == {"changes", "last_link", "rows", "perspective", "derived", "status"}
                    and written["features"]["rows"] == 1
                ), written
            if (
                ids[0] == "r02-01" and not forging["vague"]
            ):  # one GLM misses unless shown the evidence; the witness answers
                await call(
                    "add_task",
                    task=Task(
                        id=ids[0],
                        category="search",
                        level=1,
                        actor_id=a,
                        question="When were the notes up for release 4.2? it is hard to find",
                        answer_type="text",
                        gold_sql="SELECT value AS answer FROM facts WHERE id = 'f2'",
                        facts=["f2"],
                        intent="when",
                    ),
                )
        elif prompt.startswith("The review rejected"):
            assert "hedging a decoy where its readers see it voids the near-miss" in prompt
            issues = json.loads(prompt[prompt.index("[") : prompt.index("]\n") + 1])
            seen.append(("fix issues", [i.get("blocking") for i in issues]))
            for issue in issues:
                for message in issue.get("message_ids", []):
                    await call("revise", message_id=message, text="ok, the notes come tomorrow")
            seen.append(("fixed", [m for i in issues for m in i.get("message_ids", [])]))

    agents = SimpleNamespace(
        judge=FlowJudge(),
        solver=ScriptedSolver(),
        witness=ScriptedWitness(),
        author=ScriptedAuthor(script),
        proposer=ScriptedAuthor(script, role="proposer"),
    )
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
        seats.solver.sampling.temperature == 1.0 == seats.witness.sampling.temperature
        and seats.judge.sampling.temperature == 0.0
    )
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
    # The world the forge, the measure and the chapter start from: one with no tasks of its own.
    world_dir, world_settings = await world_run(
        root, settings, lambda **seats: SimpleNamespace(**(vars(agents) | seats))
    )
    source_world = Workspace(world_dir / "world.sqlite")
    # The forge: tasks proposed on the finished world, which stays frozen; what the judge approves is kept.
    plan = flow_forge(settings, world=world_dir, output=root / "forge", rounds=2, candidates=2, tasks=30)
    workspace = source_world.db.execute(
        "SELECT group_concat(id || ':' || text, '|') FROM messages"
    ).fetchone()[0]
    forged = TestStore(plan.output, forge_provenance(plan))

    Forging = ForgeEnv

    ScriptedAgent.traced.clear()
    store.state.active_attempt = "forge"  # the scripted solver names its tries by the run it reads
    await Forging(plan, forged).run(None, agents)
    assert [x[1] for x in seen if x[0] == "forge"] == [("r01-01", "r01-02"), ("r02-01", "r02-02")]
    chunks = {
        a: sorted(s[6] for s in seen if s[:3] == ("review", a, "task")) for a in ("round-01", "round-02")
    }
    assert chunks == {"round-01": [("r01-01",), ("r01-02",)], "round-02": [("r02-01",), ("r02-02",)]}, (
        "the judge reviews a round's candidates one chunk at a time"
    )
    events = [json.loads(line) for line in (forged.root / "progress.jsonl").read_text().splitlines()]
    roles = {path.name.split("-")[0] for path in (forged.root / "traces").glob("*.json")}
    assert (
        "proposer" in roles and "author" not in roles and any(e["event"] == "proposer_turn" for e in events)
    ), "the forge's rounds are the proposer's, in a seat of its own"
    assert forged.state.phase == "done" and sorted(
        t for (t,) in forged.world.db.execute("SELECT id FROM tasks")
    ) == ["r01-01", "r02-01"], "the vague candidates are dropped"
    hard = forged.state.forged["r02-01"]
    assert (hard["right"], hard["witness"], hard["hint"], hard["level"], hard["verdict"]) == (
        0.0,
        1.0,
        1.0,
        3,
        "kept in search·L3 as k1·when",
    ), hard
    assert dict(forged.world.db.execute("SELECT id, level FROM tasks").fetchall()) == {
        "r01-01": 2,
        "r02-01": 3,
    }, "its tries give its level"
    assert (
        forged.world.db.execute("SELECT group_concat(id || ':' || text, '|') FROM messages").fetchone()[0]
        == workspace
    ), "the workspace is frozen"
    assert tuple(
        forged.world.db.execute("SELECT channel_id, author_id FROM facts WHERE id = 'n1'").fetchone()
    ) == tuple(
        source_world.db.execute(
            "SELECT channel_id, user_id FROM messages WHERE text LIKE '%rollback%' ORDER BY ts_us LIMIT 1"
        ).fetchone()
    ), "an annotation is placed by its first message"
    source_world.close()
    publish(forged)
    # The forge's output, as a measure starts from it: a finished run with tasks of its own.
    source = root / "measure-source"
    source.mkdir()
    forged.world.snapshot(source / "world.sqlite")
    (source / "state.json").write_bytes((forged.root / "state.json").read_bytes())
    assert {r.task_id for r in load_release(plan.output / "release")[1]} == {"r01-01", "r02-01"}
    _, cases = failure_cases(
        forged.world,
        "r01-01",
        json.loads(forged.world.db.execute("SELECT gold_json FROM tasks WHERE id = 'r01-01'").fetchone()[0]),
    )
    assert any(name.startswith("the near-miss query's answer") for name, _ in cases), cases
    # The forge's tries, as the forge keeps them. The witness tried the task the solver rarely answers, twice; an
    # unchanged task keeps its witness. The scripted solver counts these tries apart from the forge runs'.
    store.state.active_attempt = "check"
    forge_env = Forging(plan, forged)
    assert len(forged.state.witness["r02-01"]["results"]) == plan.witness_tries
    assert any(x[0] == "witnessed" and "r02-01" in x[2] and x[3] == 2 for x in seen), (
        "the judge reads the witness's tries"
    )
    tried = sum(witnessed.values())
    assert (
        await forge_env.tries.witness(
            agents, ["r02-01"], {"r02-01": forge_env.tries.task_key("r02-01")}, "check"
        )
    )["r02-01"]["results"] and sum(witnessed.values()) == tried, "an unchanged task keeps its witness"

    class Slow(
        ScriptedAgent
    ):  # the solver's gate holds its solves to [author] solvers, whatever the episode allows
        active, peak = 0, 0

        async def run(self, task):
            Slow.active += 1
            Slow.peak = max(Slow.peak, Slow.active)
            await asyncio.sleep(0.01)
            Slow.active -= 1
            evaluation = {
                "task_id": task.data.task_id,
                "execution_ok": True,
                "semantic_correctness": 1.0,
                "correct": True,
                "grounded": True,
                "calls": 1,
                "reason": "r",
                "response": "an answer",
            }
            return FakeTrace("slow", "solver", info={"evaluation": evaluation}, task=task)

    await forge_env.tries.solves(SimpleNamespace(solver=Slow()), ["r01-01", "r02-01"], "check", 3)
    assert Slow.peak == plan.solvers == 2, Slow.peak

    class Crashing(ScriptedAgent):
        async def run(self, task):
            return FakeTrace(
                "crash",
                "solver",
                ok=False,
                info={"grading_started": True},
                errors=[SimpleNamespace(message="judge timeout")],
                task=task,
            )

    # else a task whose every try crashed was kept
    with pytest.raises(RuntimeError, match=re.escape("crashed twice")):
        await forge_env.tries.solves(SimpleNamespace(solver=Crashing()), ["r01-01"], "check", 2)
    # Its summary: each task's rates, a try that crashed twice counted and left out of them, and its difficulty.
    summary = forged.summary("complete")
    assert (
        summary["rates"]["r02-01"]["crashed"] == 1
        and summary["rates"]["r02-01"]["tries"] == plan.tries - 1
        and summary["rates"]["r01-01"]["level"] == 2
    ), summary["rates"]
    assert summary["witness_tries"] == 2 and summary["crashed_solves"] == 1, summary
    difficulty = summary["difficulty"]
    assert set(difficulty) == {t for (t,) in forged.world.db.execute("SELECT id FROM tasks")} and all(
        "evidence_pages" in d and d["concept"] for d in difficulty.values()
    ), "the summary reports each task's measured difficulty beside its level"
    # A release takes every task reviewed, on facts the world states.
    review = forged.state.task_reviews.pop("r01-01")
    refused("an approved review of every task", publish, forged)  # a task nobody reviewed
    forged.state.task_reviews["r01-01"] = review
    forged.world.insert(
        "facts",
        [
            dict(
                id="f9",
                storyline="s2",
                subject="x",
                attribute="y",
                value="z",
                channel_id=channel_id("public", "ops", []),
                author_id=forged.world.db.execute("SELECT MIN(id) FROM users").fetchone()[0],
                day=3,
                summary="s",
            )
        ],
    )
    forged.world.insert("task_facts", [dict(task_id="r01-01", fact_id="f9")])
    refused("a world that breaks no rule", publish, forged)  # a task resting on a fact no message states
    forged.world.db.execute("DELETE FROM task_facts WHERE fact_id = 'f9'")
    forged.world.db.execute("DELETE FROM facts WHERE id = 'f9'")
    # A task's key covers the decoys on its facts' subjects: a revised decoy is a changed task, tried again.
    keyed, lure = (
        forge_env.tries.task_key("r02-01"),
        forged.world.db.execute("SELECT MAX(id) FROM messages").fetchone()[0],
    )
    lured = forged.world.db.execute("SELECT text FROM messages WHERE id = ?", (lure,)).fetchone()[0]
    forged.world.insert(
        "facts",
        [
            dict(
                id="f8",
                storyline="s1",
                subject="Release 4.2",
                attribute="state",
                value="still down",
                is_decoy=1,
                channel_id=channel_id("public", "ops", []),
                author_id=forged.world.db.execute("SELECT MIN(id) FROM users").fetchone()[0],
                day=3,
                summary="s",
            )
        ],
    )
    forged.world.db.execute(
        "INSERT INTO evidence (fact_id, message_id, role) VALUES ('f8', ?, 'supporting')", (lure,)
    )
    decoyed = forge_env.tries.task_key("r02-01")
    forged.world.db.execute("UPDATE messages SET text = 'still down, I think' WHERE id = ?", (lure,))
    assert len({keyed, decoyed, forge_env.tries.task_key("r02-01")}) == 3, "the key follows the task's decoys"
    forged.world.db.execute("DELETE FROM evidence WHERE fact_id = 'f8'")
    forged.world.db.execute("DELETE FROM facts WHERE id = 'f8'")
    forged.world.db.execute("UPDATE messages SET text = ? WHERE id = ?", (lured, lure))

    # A release's evidence is what states its answer: not a value that answer supersedes, not a decoy, not what its
    # asker cannot read; a refusal's truth out of its asker's sight ships apart, as unseen.
    def stating(fact):
        return {
            tuple(r)
            for r in forged.world.db.execute(
                "SELECT m.channel_id, m.ts FROM evidence e JOIN messages m ON m.id = e.message_id WHERE e.fact_id = ?",
                (fact,),
            )
        }

    assert {tuple(m) for m in release_rows(forged, "h")[1]["r02-01"].messages} == stating("f2"), (
        "its answer's own"
    )
    forged.world.db.execute("INSERT INTO task_facts (task_id, fact_id) VALUES ('r02-01', 'f1')")
    forged.world.db.execute(
        "INSERT INTO fact_relations (src_fact, dst_fact, kind) VALUES ('f2', 'f1', 'supersedes')"
    )
    released = release_rows(forged, "h")[1]["r02-01"]
    forged.world.db.execute("DELETE FROM fact_relations WHERE src_fact = 'f2' AND dst_fact = 'f1'")
    forged.world.db.execute("DELETE FROM task_facts WHERE task_id = 'r02-01' AND fact_id = 'f1'")
    assert stating("f1") and {tuple(m) for m in released.messages} == stating("f2"), (
        "a stale value is no evidence"
    )
    forged.world.db.execute("UPDATE facts SET is_decoy = 1 WHERE id = 'f2'")
    released = release_rows(forged, "h")[1]["r02-01"]
    forged.world.db.execute("UPDATE facts SET is_decoy = 0 WHERE id = 'f2'")
    assert not released.messages, "a decoy is no evidence"
    leads = channel_id("private", "leads", [])
    (outsider,) = forged.world.db.execute(
        "SELECT id FROM users WHERE is_bot = 0 AND id NOT IN (SELECT user_id FROM members WHERE channel_id = ?) ORDER BY id LIMIT 1",
        (leads,),
    ).fetchone()
    (asker,) = forged.world.db.execute("SELECT actor_id FROM tasks WHERE id = 'r02-01'").fetchone()
    hidden = forged.world.db.execute(
        "SELECT id, channel_id, ts FROM messages WHERE channel_id = ? ORDER BY id LIMIT 1", (leads,)
    ).fetchone()
    forged.world.db.execute("UPDATE tasks SET actor_id = ? WHERE id = 'r02-01'", (outsider,))
    forged.world.db.execute(
        "INSERT INTO evidence (fact_id, message_id, role) VALUES ('f2', ?, 'supporting')", (hidden[0],)
    )
    released = release_rows(forged, "h")[1]["r02-01"]
    forged.world.db.execute("DELETE FROM evidence WHERE fact_id = 'f2' AND message_id = ?", (hidden[0],))
    forged.world.db.execute("UPDATE tasks SET actor_id = ? WHERE id = 'r02-01'", (asker,))
    assert released.messages and [hidden[1], hidden[2]] not in released.messages, "out of sight, no evidence"
    (asker,) = forged.world.db.execute("SELECT actor_id FROM tasks WHERE id = 'r01-01'").fetchone()
    forged.world.db.execute("INSERT INTO task_facts (task_id, fact_id) VALUES ('r01-01', 'f3')")
    forged.world.db.execute("UPDATE tasks SET actor_id = ? WHERE id = 'r01-01'", (outsider,))
    truth = [
        list(r)
        for r in forged.world.db.execute(
            "SELECT m.channel_id, m.ts FROM evidence e JOIN messages m ON m.id = e.message_id WHERE e.fact_id = 'f3' AND e.role = 'anchor'"
        )
    ]
    private = release_rows(forged, "h")[1]["r01-01"]
    forged.world.db.execute("UPDATE tasks SET actor_id = ? WHERE id = 'r01-01'", (asker,))
    forged.world.db.execute("DELETE FROM task_facts WHERE task_id = 'r01-01' AND fact_id = 'f3'")
    assert truth and private.unseen == truth and not [m for m in private.messages if m in truth], private
    rows = load_release(plan.output / "release")[1]
    assert all(r.right_interval == forged.state.task_reviews[r.task_id]["right_interval"] for r in rows), (
        "intervals are published"
    )
    reviews = forged.state.task_reviews
    assert all(
        (r.right_rate, r.strict_rate, r.tries, r.solver)
        == (
            reviews[r.task_id]["right_rate"],
            reviews[r.task_id]["strict_rate"],
            reviews[r.task_id]["tries"],
            plan.env.solver.model,
        )
        for r in rows
    ), "each task's rates are published"
    assert all((r.surface, r.turns, type(r.miss_classes)) == (slack.VERSION, 40, dict) for r in rows), (
        "each task's surface, turns and misses"
    )
    assert {r.task_id: (r.witness, r.witness_right) for r in rows if r.witness} == {
        "r02-01": (plan.env.witness.model, 1.0)
    }, "the witness, on the task it tried"
    # A task ships with the wrong answers its grade must reject: one the answer judge accepts makes it ambiguous as
    # asked, and it is not approved, whatever its tries.
    assert forged.state.screens and all(not s["failed"] for s in forged.state.screens.values()), (
        "every task was screened"
    )
    gold, cases = failure_cases(
        forged.world,
        "r01-01",
        json.loads(forged.world.db.execute("SELECT gold_json FROM tasks WHERE id = 'r01-01'").fetchone()[0]),
    )
    assert (
        gold.endswith(".")
        and ("a bare opener", "Thought process:") in cases
        and any(n == "declining to answer" for n, _ in cases)
    ), cases
    GRADER["loose"] = True
    _, verdict = await forge_env.tries.judge_tasks(
        agents, ["r01-01"], {"r01-01": "screened loose"}, "check", 3
    )
    assert not verdict.approved and any(
        "accepts a wrong answer to it: a bare opener" in i.defect for i in forged.state.task_issues["r01-01"]
    ), verdict
    assert forged.state.screens["r01-01"]["failed"] == ["a bare opener ('Thought process:')"]
    GRADER["loose"], GRADER["blind"] = False, True
    assert (await forge_env.tries.screen(["r01-01"], {"r01-01": "screened blind"}, "check"))["r01-01"] == [
        "its gold answer, judged wrong"
    ]
    GRADER["blind"] = False
    # An unchanged task keeps its tries: judged again, as a resumed round judges it, it is not solved again.
    tried = sum(solved.values())
    await forge_env.tries.judge_tasks(
        agents, ["r02-01"], {"r02-01": forge_env.tries.task_key("r02-01")}, "check", 3
    )
    assert sum(solved.values()) == tried, "an unchanged task was solved again"
    store.state.active_attempt = "forge"
    # A cell keeps the candidate GLM's tries are most mixed on; the lowest band also needs the witness to answer it.
    assert [measured_level(r, plan.bands) for r in (1.0, 0.75, 0.6, 0.5, 0.25, 0.0)] == [
        1,
        1,
        2,
        2,
        3,
        3,
    ]
    assert (
        bucket(
            {
                "changes": 3,
                "last_link": "dm·later",
                "rows": 2,
                "perspective": True,
                "derived": False,
                "status": False,
            }
        )
        == "k3+dm+set+perspective"
    )
    # A place holds two tasks. A newcomer to a full one replaces a holder if GLM's tries are more mixed on it, or if
    # it brings a structure the place lacks while two holders share one; else it is dropped.
    weigher, actor = (
        Forging(plan, forged),
        forged.world.db.execute("SELECT MIN(id) FROM users").fetchone()[0],
    )
    newest = Task(
        id="r09-01",
        category="lookup",
        level=2,
        actor_id=actor,
        question="Who is the newest person here?",
        answer_type="text",
        gold_sql="SELECT real_name AS answer FROM users ORDER BY id DESC LIMIT 1",
        near_sql=["SELECT real_name AS answer FROM users ORDER BY id LIMIT 1"],
        intent="structure",
    )
    newest_two = newest.model_copy(
        update={
            "answer_type": "set",
            "gold_sql": "SELECT real_name AS answer FROM users ORDER BY id DESC LIMIT 2",
            "near_sql": ["SELECT real_name AS answer FROM users ORDER BY id LIMIT 2"],
        }
    )
    for task_, open_id, question, right, witness, verdict in (
        (
            newest,
            "r09-01",
            "Who is the newest person here, r09-01?",
            0.0,
            0.0,
            "dropped: the witness does not answer it either: hard, or broken",
        ),
        (
            newest,
            "r09-01",
            "Who is the newest person here, r09-01?",
            0.5,
            None,
            "kept in lookup·L2 as plain·structure",
        ),  # the place had room
        (
            newest,
            "r09-02",
            "Who is the newest person here, r09-02?",
            0.625,
            None,
            "kept in lookup·L2 as plain·structure",
        ),  # more mixed than r01-01
        (
            newest,
            "r09-03",
            "Who is the newest person here, r09-03, by when they joined?",
            0.625,
            None,
            "dropped: lookup·L2 is full (r09-01, r09-02), with tasks as mixed or more and its structure",
        ),
        (
            newest_two,
            "r09-04",
            "Who are the two newest people here?",
            0.625,
            None,
            "kept in lookup·L2 as set·structure",
        ),  # a new structure
    ):
        with forged.world.trial() as copy:
            forge_task(copy, task_.model_copy(update={"id": open_id, "question": question}), plan, [open_id])
        forged.state.solves[open_id] = {"key": "k", "results": [], "traces": []}
        forged.state.forged[open_id] = weigher.place(
            open_id, {"right_rate": right, "witness_right": witness}, None, "x"
        )  # as weigh records it
        assert forged.state.forged[open_id]["verdict"] == verdict, (open_id, right)
    assert [h["task"] for h in forged.state.archive["lookup·L2"]] == ["r09-01", "r09-04"], (
        forged.state.archive
    )
    # The proposer steers by this forge's own numbers, by category and intent.
    steering = forge_measured(forged.state.forged)
    assert any(line.startswith("- lookup·structure: ") and " kept" in line for line in steering) and any(
        line.startswith("- search·when: ") for line in steering
    ), steering
    pages_ = Forging(plan, forged).pages()
    assert (
        "## What this forge has measured" in pages_["memory/archive.md"]
        and steering[0] in pages_["memory/archive.md"]
        and "- promised: what someone promised" in pages_["memory/taxonomy.md"]
    )
    assert plan.env.witness.sampling.reasoning_effort == "high", "the witness thinks hard"
    assert (forged.state.forged["r01-01"]["verdict"], forged.state.forged["r09-02"]["verdict"]) == (
        "replaced by r09-02",
        "replaced by r09-04",
    )
    # A wrong answer another reading of its question makes right: the judge's ruling is code's blocking issue, the task
    # is sent back and never kept as hard, and a rewording keeps its asker and gold answer.
    fails(TaskReview, task_id="t", valid=True, reason="r", misread="'Ann', if 'here' is the channel")
    ruled = read_one_way(
        Verdict(
            approved=False,
            tasks=[
                TaskReview(
                    task_id="r09-05",
                    valid=False,
                    reason="r",
                    level_fit=3,
                    misread="'Ann', if 'here' is the channel",
                )
            ],
            issues=[
                Issue(artifact="tasks", task_ids=["r09-05"], defect="two readings", requested_change="r")
            ],
            criteria={},
            summary="s",
        )
    )
    assert [i.defect for i in ruled.issues if i.blocking] == [
        "two readings",
        READING + "'Ann', if 'here' is the channel",
    ]
    readable = newest.model_copy(
        update={"id": "r09-05", "question": "Who is the newest person here, r09-05?"}
    )
    with forged.world.trial() as copy:
        forge_task(copy, readable, plan, ["r09-05"])
    forged.state.task_issues["r09-05"] = [i for i in ruled.issues if "r09-05" in i.task_ids]
    sent = weigher.place("r09-05", None, None, "x")
    gold_answer = [
        r["answer"]
        for r in forged.world.db.execute("SELECT real_name AS answer FROM users ORDER BY id DESC LIMIT 1")
    ]
    assert sent["verdict"] == "sent back: its question can be read another way" and sent["rework"] == {
        "answer": gold_answer,
        "actor": actor,
        "reading": "'Ann', if 'here' is the channel",
    }, sent
    assert not forged.world.db.execute("SELECT 1 FROM tasks WHERE id = 'r09-05'").fetchone(), "never kept"
    rework, other = (
        {"r09-05": sent["rework"]},
        forged.world.db.execute("SELECT MAX(id) FROM users").fetchone()[0],
    )
    again = readable.model_copy(
        update={"id": "r09-06", "question": "Who joined the workspace last?", "rewords": "r09-05"}
    )
    for bad, why in (
        (again.model_copy(update={"rewords": "r09-01"}), "sent back for another reading"),
        (again.model_copy(update={"actor_id": other}), "the same asker"),
        (
            newest_two.model_copy(
                update={"id": "r09-06", "question": "Who are the two who joined last?", "rewords": "r09-05"}
            ),
            "keeps its gold answer",
        ),
    ):
        # else a rewording was accepted: {bad}
        with pytest.raises(ValueError, match=re.escape(why)):
            with forged.world.trial() as copy:
                forge_task(copy, bad, plan, ["r09-06"], rework=rework)
    with forged.world.trial() as copy:
        forge_task(copy, again, plan, ["r09-06"], rework=rework)
    # Code classes each wrong try: out of turns, evidence never shown, or all of it shown and answered otherwise.
    tried = [
        {"correct": False, "unanswered": True, "reason": EMPTY},
        {"correct": False, "abstained": False, "evidence_coverage": 0.5, "missing": ["x"], "extra": []},
        {"correct": False, "abstained": False, "evidence_coverage": 1.0, "missing": ["x"], "extra": ["y"]},
        {"correct": True},
    ]
    forged.state.solves["r09-06"] = {"key": "k", "results": tried, "traces": []}
    assert weigher.place("r09-06", {"right_rate": 0.25, "witness_right": 1.0}, None, "x")["miss_classes"] == {
        "budget": 1,
        "search": 1,
        "saw_all": 1,
    }
    assert [
        miss_class(r, t)
        for r, t in (
            ({"abstained": True}, "text"),
            ({"missing": [], "extra": []}, "status"),
            ({"committed": ["Ann"], "missing": [], "extra": []}, "refusal"),
            ({"evidence_coverage": None, "missing": ["a"], "extra": []}, "number"),
        )
    ] == ["abstained", "overclaim", "trap", "saw_all"]
    forged.close()
    fails(Store, plan.output, forge_provenance(plan.model_copy(update={"rounds": 3})))  # its own config only
    # A forge stops early: at its goal (here, one task and no board), or after two rounds that place nothing. A goal
    # the archive's places cannot hold is refused.
    fails(Forging, plan.model_copy(update={"tasks": 1000}), forged)
    crashes.clear()  # the forge above is the one whose solver crashes
    for name, changes, vague, rounds in (
        ("forge-full", {"tasks": 1}, False, 1),
        ("forge-idle", {}, True, 2),
    ):
        ScriptedAgent.traced.clear()
        solved.clear(), finished.clear(), witnessed.clear()
        early = plan.model_copy(update={"output": root / name, "rounds": 4} | changes)
        forging["vague"], early_store = (
            vague,
            TestStore(early.output, forge_provenance(early)),
        )
        env_ = Forging(early, early_store)
        env_.adopt()
        early_store.world.db.execute("DELETE FROM board")
        await env_.run(None, agents)
        assert (early_store.state.phase, early_store.state.batch) == ("done", rounds), (
            name,
            early_store.state.batch,
        )
        early_store.close()
    # A forge that only measures: the finished run's own tasks, imported as written (past the forge's own gates; a
    # level-2 count with no near-miss query here), then measured like any candidate, with no proposer turn.
    extra = Workspace(
        source / "world.sqlite", writable=True
    )  # the forge's output, with one more task of its own
    with extra.trial() as copy:
        record_task(
            copy,
            Task(
                id="extra-l2",
                category="join",
                level=2,
                actor_id=store.world.db.execute("SELECT MIN(id) FROM users").fetchone()[0],
                question="How many messages are there in all?",
                answer_type="number",
                gold_sql="SELECT COUNT(*) AS answer FROM messages",
            ),
            plan,
            Slot(id="extra-l2", category="join", level=2, concept="c", style="s"),
        )
    source_ids = sorted(t for (t,) in extra.db.execute("SELECT id FROM tasks"))
    extra.close()
    ScriptedAgent.traced.clear()
    solved.clear(), finished.clear(), witnessed.clear()
    measuring = plan.model_copy(
        update={"world": source, "output": root / "measured", "rounds": 0, "measure": True, "target": 30}
    )
    measured, turns = (
        TestStore(measuring.output, forge_provenance(measuring)),
        len([x for x in seen if x[0] == "forge"]),
    )
    await Forging(measuring, measured).run(None, agents)
    assert (
        "extra-l2" in source_ids
        and sorted(measured.state.forged) == source_ids
        and not [o for o in measured.state.forged.values() if o["verdict"].startswith("not imported")]
    ), measured.state.forged
    assert measured.state.phase == "done" and len([x for x in seen if x[0] == "forge"]) == turns, (
        "no proposer turn"
    )
    assert any(o["verdict"].startswith("kept") for o in measured.state.forged.values())
    publish(measured)
    measured.close()
    refused(
        "a forge with no rounds only measures",
        ForgeConfig.model_validate,
        plan.model_dump(mode="json") | {"rounds": 0, "measure": False},
    )
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
    assert set(same_schema(aged)) == {
        "artifacts",
        "channels",
        "integrations",
        "messages",
        "queue",
        "usergroup_members",
        "usergroups",
        "users",
    }
    assert (aged.migrate(), same_schema(aged), aged.migrate(), aged.violations()) == (1, [], 2, []), (
        "forward, once"
    )
    assert aged.db.execute("SELECT COUNT(*), group_concat(text) FROM messages").fetchone()[:] == rows
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
    assert (meta(elder_store.world, "schema_version"), same_schema(elder_store.world)) == ("2", []), (
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
    # A forge on the version the chapter leaves: the brief is its board; the earlier entries were covered before.
    forge = flow_forge(cfg, world=run.output, output=root / "forge", rounds=1, candidates=1, tasks=1)
    forging = TestStore(forge.output, forge_provenance(forge))
    on_chapter = ForgeEnv(forge, forging)
    on_chapter.adopt()
    assert on_chapter.covered()[1] == ["semantic-l3-v2-1"], on_chapter.covered()
    # One round on it, whose task review notes a message of the frozen past: thawed, the note is set apart, and the
    # candidate is kept.
    forging.close()
    with episodes(SimpleNamespace(**forge_seats(note_on=old))):
        summary = await run_forge(forge)
    forging = TestStore(forge.output, forge_provenance(forge))
    assert summary["status"] == "complete" and forging.state.forged["r01-01"]["verdict"].startswith("kept"), (
        forging.state.forged
    )
    assert (forging.root / "attempts/round-01/frozen_issues-tasks.json").exists(), (
        "the frozen issue is set apart"
    )
    assert [r.task_id for r in load_release(forge.output / "release")[1]] == ["r01-01"], "the forge publishes"
    forging.close()


@contextmanager
def scripted_judges():
    """No judge reaches a model while the checks run: the answer judge is played by code (GRADER), and any other judge
    call fails."""
    saved, vf.Judge.complete = vf.Judge.complete, scripted_complete
    try:
        yield
    finally:
        vf.Judge.complete = saved
