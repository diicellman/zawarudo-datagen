"""The task forge: tasks written on a finished, frozen world, as the paper's three sets (Synthetic Data RL, 2505.17063).
uv run --frozen worldgen-slack forge --config configs/worldgen_slack/forge.toml [--dry-run]

The proposer writes the initial set; every task is labelled (the verifier must reach its answer from Slack, the student
tries it, the judge rules on it); then the proposer writes a harder version of each task the student always gets right,
an easier one of each it never gets, and a repaired one of each verified task the judge refused, and those are
labelled too. The release is every approved task, with the set it came from and how hard it measured."""

import asyncio
import json
import shutil

from worldgen_slack.release import atomic_json
from worldgen_slack.world import file_hash

from ..config import pinned
from ..session import SETUP_FILES, SessionEnv, require_session
from ..store import ReviewLimit, Store
from ..workspace import Workspace, reset_tasks
from .config import ForgeConfig
from .judge import rule
from .proposer.task import ProposerTask, adapt_prompt, initial_prompt
from .release import KEPT
from .tasks import FAMILIES, Task, answer_text, check, key
from .tries import Tries, agreeing, by_part, rates, reached

# $ per million fresh input, cached input and output tokens (`prime inference models`, 2026-10-16), for a call its
# provider leaves unpriced; a model not listed is priced as the dearest.
PRICES = {
    "z-ai/glm-5.3": (1.4, 0.26, 4.4),
    "z-ai/glm-5.3-flash": (0.15, 0.15, 0.5),
    "deepseek/deepseek-v4.1-flash": (0.3, 0.03, 1.2),
    "anthropic/claude-opus-5.5": (4.0, 4.0, 20.0),
}
DEAREST = (4.0, 4.0, 20.0)


def priced(usage: dict, model: str | None) -> float:
    if usage.get("cost") is not None:
        return usage["cost"]
    fresh, cached, out = PRICES.get(model or "", DEAREST)
    return (
        (usage.get("prompt_tokens") or 0) * fresh
        + (usage.get("cached_input_tokens") or 0) * cached
        + (usage.get("completion_tokens") or 0) * out
    ) / 1e6


def spent(store) -> float:
    """What a forge has spent: each trace's calls (its own and its answer judge's), and the task judge's rulings, at
    their reported cost or else list prices: an estimate that errs high."""
    env, judges = store.settings.get("env", {}), store.settings.get("answer_judge", {}).get("model")
    total = 0.0
    for path in (store.root / "traces").glob("*.json"):
        try:
            trace = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        total += priced(trace.get("usage") or {}, env.get(trace["agent"]["name"], {}).get("model"))
        total += sum(priced(u, judges) for u in trace.get("extra_usage") or [])
    judge = store.settings.get("judge", {}).get("model")
    return total + sum(priced(r.get("usage") or {}, judge) for r in store.state.forged.values())


def provenance(config: ForgeConfig) -> dict:
    """Everything a resumed forge must share with its start: its configuration, the world it writes on (by hash), the
    pinned framework and the lock file."""
    return {
        "config": config.model_dump(mode="json"),
        "world_hash": file_hash(config.world / "world.sqlite"),
    } | pinned()


def label_of(record: dict, solved: float) -> str:
    """A labelled task's label: rejected (its answer not verified, or the judge refused it), else by its pass rate:
    solved, unsolved (never right), or in the window."""
    if not record.get("verified") or not record.get("approved"):
        return "rejected"
    right = record["rates"]["right_rate"]
    return "solved" if right >= solved else "unsolved" if right == 0 else "window"


class ForgeEnv(SessionEnv):
    seat = "proposer"
    seats = ("proposer", "solver", "verifier", "judge")
    notes = ("plan.md", "recap.md", "forge.md")  # the world author's, then the proposer's own

    def __init__(self, settings: ForgeConfig, store: Store):
        super().__init__(settings, store)
        self.tries = Tries(settings, store)

    @property
    def plan(self) -> dict:
        return self.store.state.plans.setdefault("forge", {})

    def provision_task(self) -> ProposerTask:
        """The proposer's own task: no persona file is read to provision its VM."""
        return ProposerTask.create(self.world.path, self.tools_context(), "setup")

    async def run(self, task, agents):
        state = self.store.state
        if state.phase == "premise":
            self.adopt()
        while state.phase != "done":
            if spent(self.store) >= self.settings.spend_limit:
                raise ReviewLimit(f"the forge has spent its limit, ${self.settings.spend_limit:.2f}")
            if state.phase in ("generate", "adapt"):
                await self.work_in_vm(
                    agents
                )  # the proposer writes the phase's tasks, then the phase is "label"
            elif state.phase == "label":
                await self.label(agents)
                self.next_pass()

    def adopt(self) -> None:
        """The finished world, frozen (its tables as the schema now has them, no tasks), and the world author's last
        notes for the proposer to start from."""
        source = Workspace(self.settings.world / "world.sqlite")
        source.db.backup(self.world.db)
        source.close()
        self.world.migrate()
        reset_tasks(self.world)
        state = json.loads((self.settings.world / "state.json").read_text())
        notes = self.settings.world / "attempts" / state.get("restore_point", "") / "notes"
        if state.get("restore_point") and notes.is_dir():
            shutil.copytree(notes, self.store.path("source", "notes"), dirs_exist_ok=True)
        self.store.state.phase, self.store.state.batch, self.store.state.restore_point = (
            "generate",
            0,
            "source",
        )
        # Each initial id asks for a family in turn, so every family is written whatever the turns' sizes.
        families = list(FAMILIES)
        self.plan["planned"] = {
            f"i{n:03d}": {"family": families[(n - 1) % len(families)]}
            for n in range(1, self.settings.tasks + 1)
        }
        self.store.event(
            "world_adopted",
            world=str(self.settings.world),
            messages=self.world.db.execute("SELECT COUNT(*) FROM messages").fetchone()[0],
        )
        self.store.save()

    # ------------------------------------------------------------------ writing: the initial set, then its versions

    def open_ids(self) -> list[str]:
        return [t for t in self.plan["planned"] if t not in self.store.state.forged]

    async def blocks(self, agents, runtime, setup) -> None:
        """The proposer's turns in one VM until the phase's tasks are written, or two turns in a row write none. A turn
        cut off with its session keeps what it wrote: a fresh VM takes those tasks before it opens a turn."""
        await runtime.run(["rm", "-f", *(f"/task/{name}" for name in SETUP_FILES)], {})
        state = self.store.state
        kept = [t for t in self.open_ids() if (self.store.root / "candidates" / f"{t}.json").exists()]
        if kept and self.take(kept):
            self.plan["empty"] = 0
        while state.phase in ("generate", "adapt"):
            if not (ids := self.open_ids()) or self.plan.get("empty", 0) >= 2:
                state.phase, self.plan["empty"] = "label", 0
                self.store.save()
                return
            await self.turn(agents, runtime, ids[: self.settings.batch])

    async def turn(self, agents, runtime, ids: list[str]) -> None:
        state = self.store.state
        n = self.plan["turn"] = self.plan.get("turn", 0) + 1
        attempt = self.store.begin(f"{'initial' if state.batch == 0 else f'adapt{state.batch}'}-{n:02d}")
        versions = {t: v for t, v in self.plan["planned"].items() if t in ids and "variant_of" in v}
        families = {t: self.family_of(t) for t in ids}
        self.allow(ids, versions, families)
        await self.bring_notes(runtime, state.restore_point)
        task = ProposerTask.create(self.world.path, self.tools_context(), attempt)
        if state.batch == 0:
            written = len(state.forged)
            prompt = initial_prompt(families, written, self.settings.tasks)
        else:
            prompt = adapt_prompt(ids)
        async with agents.proposer.interaction(task, runtime=runtime) as interaction:
            await self.take_turn(interaction, runtime, prompt, attempt, "forge", self.pages(ids))
        self.store.trace(interaction.trace)
        require_session(interaction.trace)
        await self.keep_notes(runtime, attempt)
        added = self.take(ids)
        self.plan["empty"] = 0 if added else self.plan.get("empty", 0) + 1
        state.restore_point = attempt
        self.store.finish_attempt(bool(added))

    def family_of(self, task_id: str) -> str | None:
        """The family an open id asks for: an initial task's by its place, a version's its source's (a run begun
        before ids had families leaves its initial ones free)."""
        planned = self.plan["planned"][task_id]
        if "variant_of" in planned:
            return self.store.state.forged[planned["variant_of"]]["task"]["family"]
        return planned.get("family")

    def allow(self, ids: list[str], versions: dict, families: dict) -> None:
        """The ids the proposer's turn may write, the family each asks for, and which task each version is of and
        why: its tools read them per call."""
        atomic_json(
            self.store.root / "writable.json", {"ids": list(ids), "versions": versions, "families": families}
        )

    def take(self, ids: list[str]) -> int:
        """The turn's tasks into the run: each kept file checked again (the world is the one it was checked on), with
        its set and pass. Returns how many it took."""
        state, taken = self.store.state, 0
        written = [Task.model_validate(r["task"]) for r in state.forged.values()]
        for task_id in ids:
            path = self.store.root / "candidates" / f"{task_id}.json"
            if not path.exists():
                continue
            task = Task.model_validate_json(path.read_bytes())
            try:
                check(self.world, task, written)
            except (
                ValueError
            ) as error:  # its tool checked it: only another task of the turn can make it a repeat
                self.store.event("task_refused", task_id=task_id, error=str(error)[:500])
                continue
            state.forged[task_id] = {
                "task": task.model_dump(mode="json"),
                "set": "initial" if task.why is None else task.why,
                "pass": state.batch,
                "key": key(task),
            }
            written.append(task)
            taken += 1
        self.store.event("tasks_taken", attempt=state.active_attempt, tasks=taken, of=len(ids))
        self.store.save()
        return taken

    def pages(self, ids: list[str]) -> dict[str, str]:
        """The forge's own memory pages: the tasks written, the families, and in an adapting turn what to write."""
        state = self.store.state
        tasks = ["# Tasks written (write new requests, never a repeat)", ""]
        for task_id, record in state.forged.items():
            task = Task.model_validate(record["task"])
            tasks.append(
                f"- {task_id} ({record['set']}, {task.family}, asked by {task.actor_id}): {task.question}\n  answer: "
                + " | ".join(answer_text(task))
            )
        out = {
            "memory/tasks.md": "\n".join(tasks) + "\n",
            "memory/families.md": "# Families: the request a task is\n\n"
            + "\n".join(f"- {k}: {v}" for k, v in FAMILIES.items())
            + "\n",
        }
        if state.batch:
            out["memory/adapt.md"] = self.adapt_page(ids)
        return out

    WHY = {
        "harder": "The student answered it right {right} of {tries}. Write a harder version: add a part the request "
        "needs, ask for the answer as of a moment, ask for every item where a reader would stop at one place, or ask "
        "for what is not settled. Its difficulty comes from what the request needs, not from tool tedium.",
        "easier": "The student never answered it ({right} of {tries}), though the verifier did. Write an easier version: "
        "a sub-question, fewer parts, or the part it missed, asked alone.",
        "repaired": "The judge refused it: {reason} Rewrite it so the judge's point is met: a request a colleague would "
        "make, with one reading, each answer exactly what its part asks, resting on messages that state it.",
    }

    def adapt_page(self, ids: list[str]) -> str:
        """What each open version is of, how its source went, and what to write."""
        state, lines = self.store.state, ["# Versions to write", ""]
        for task_id in ids:
            version = self.plan["planned"][task_id]
            record = state.forged[version["variant_of"]]
            task = Task.model_validate(record["task"])
            measured = record.get("rates") or {}
            wrong = [
                (r.get("response") or "")[:300]
                for r in state.solves.get(task.id, {}).get("results", [])
                if r and not r.get("crashed") and not r.get("correct")
            ]
            lines += [
                f"## {task_id}: {'an' if version['why'] == 'easier' else 'a'} {version['why']} version of {task.id} "
                f"(variant_of={task.id}, why={version['why']})",
                f"- its request ({task.family}, asked by {task.actor_id}): {task.question}",
                "- its answer: " + " | ".join(answer_text(task)),
                f"- the student: right {measured.get('right_rate')} of {measured.get('tries')} tries"
                + (
                    "; by part: " + ", ".join(f"{k} {v:.2f}" for k, v in record["parts"].items())
                    if record.get("parts")
                    else ""
                ),
            ]
            lines += [f"- a wrong answer: {w!r}" for w in wrong[:2]]
            lines.append(
                "- what to write: "
                + self.WHY[version["why"]].format(
                    right=round((measured.get("right_rate") or 0) * (measured.get("tries") or 0)),
                    tries=measured.get("tries"),
                    reason=(record.get("ruling") or {}).get("reason", ""),
                )
            )
            lines.append("")
        return "\n".join(lines)

    # ------------------------------------------------------------------ labelling: verify, measure, judge

    async def label(self, agents) -> None:
        """Every task of the pass labelled side by side, each step kept as it lands. A task whose step fails is marked
        failed and the others go on; most of them failing is the run's stop."""
        state = self.store.state
        due = [t for t, r in state.forged.items() if r["pass"] == state.batch and "label" not in r]
        outcomes = await asyncio.gather(*(self.label_one(agents, t) for t in due), return_exceptions=True)
        failed = {t: o for t, o in zip(due, outcomes) if isinstance(o, BaseException)}
        for task_id, error in failed.items():
            if isinstance(error, asyncio.CancelledError):
                raise error
            state.forged[task_id]["failed"] = f"{type(error).__name__}: {error}"[:500]
            self.store.event("task_failed", task_id=task_id, error=state.forged[task_id]["failed"])
        self.store.save()
        if failed and len(failed) * 2 > len(due):
            raise RuntimeError(
                f"labelling failed for {len(failed)} of {len(due)} tasks: {next(iter(failed.values()))}"
            )

    async def label_one(self, agents, task_id: str) -> None:
        record = self.store.state.forged[task_id]
        record.pop("failed", None)
        task, task_key = Task.model_validate(record["task"]), record["key"]
        verified = await self.tries.verify(agents, task, task_key)
        record["agree"] = len(agreeing(verified))
        record["reached"] = reached(
            task, verified
        )  # by part: an answer holds when each part is reached twice
        record["verified"] = min(record["reached"]) >= 2
        if record["verified"]:
            measured = await self.tries.measure(agents, task, task_key)
            record["rates"] = rates(measured)
            if record["rates"]["right_rate"] is None:
                raise RuntimeError("every one of the student's tries crashed")
            record["parts"] = by_part(task, measured)
            if "ruling" not in record:
                ruling, usage = await rule(self.settings.judge, self.world, task, verified, measured)
                record["ruling"], record["usage"] = ruling.model_dump(), usage
            record["approved"] = all(
                record["ruling"][k] for k in ("realistic", "one_reading", "exact", "supported")
            )
        record["label"] = label_of(record, self.settings.solved)
        self.store.event(
            "task_labelled",
            task_id=task_id,
            label=record["label"],
            agree=record["agree"],
            reached=record["reached"],
        )
        self.store.save()

    # ------------------------------------------------------------------ the next pass

    def next_pass(self) -> None:
        """After a labelled pass: an adapting pass when one is left and it has tasks to write (a harder version of each
        solved task, an easier one of each unsolved task, a repaired one of each verified task the judge refused), else
        done."""
        state = self.store.state
        sources = [
            (t, why)
            for t, r in state.forged.items()
            if r["pass"] == state.batch
            for why in [
                {"solved": "harder", "unsolved": "easier"}.get(r.get("label"))
                or ("repaired" if r.get("label") == "rejected" and r.get("verified") else None)
            ]
            if why
        ]
        if state.batch >= self.settings.adapt or not sources:
            state.phase = "done"
        else:
            state.batch += 1
            self.plan["planned"] = {
                f"{why[0]}{state.batch}{n:02d}": {"variant_of": t, "why": why}
                for n, (t, why) in enumerate(sources, 1)
            }
            self.plan["turn"], state.phase = 0, "adapt"
        self.store.event("pass_labelled", passed=state.batch, next=state.phase, versions=len(sources))
        self.store.save()


def dry_run(config: ForgeConfig) -> dict:
    """What a forge would start from, spending nothing."""
    world = Workspace(config.world / "world.sqlite")

    def count(table):
        return world.db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

    return {
        "world": str(config.world),
        "world_hash": file_hash(config.world / "world.sqlite"),
        "messages": count("messages"),
        "people": count("users"),
        "tasks": config.tasks,
        "batch": config.batch,
        "adapt": config.adapt,
        "tries": config.tries,
        "verify": config.verify,
        "proposer": config.env.proposer.model,
        "verifier": config.env.verifier.model,
        "solver": config.env.solver.model,
        "judge": config.judge.model,
        "spend_limit": config.spend_limit,
        "labels": list(KEPT) + ["rejected"],
    }
