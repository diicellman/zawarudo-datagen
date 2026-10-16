"""An agent's session in a VM: files uploaded in one archive, the VM kept awake while only the env works, the
network notice its requests carry, and how a session that ended early is told from the agent's own limits; and the
world tools both agents in a VM (the world author and the proposer) share."""

import asyncio
import contextlib
import functools
import hashlib
import inspect
import io
import json
import re
import tarfile
import time
import typing
from pathlib import Path
from typing import Literal

import verifiers.v1 as vf
from pydantic import JsonValue, TypeAdapter, ValidationError
from verifiers.v1.dialects import base as dialects
from verifiers.v1.errors import SandboxError
from verifiers.v1.utils.decorators import discover_decorated
from verifiers.v1.utils.interrupt import cleaning_up
from worldgen_slack import slack
from worldgen_slack.toolset import SlackToolset
from worldgen_slack.world import file_hash

from .config import Seats
from .store import ReviewLimit, SessionLost, Store
from .workspace import Workspace
from .world.author.memory import files, now_page, view
from .world.cast import cards, selves
from .world.config import WorldConfig
from .world.organization import channel_id

ARCHIVE = "/tmp/task-files.tgz"
# Seconds to wait before taking a lost session up again, by how many it has lost in the block: Prime's dropped streams
# come in bursts, so a fresh VM at once often meets the same burst.
LOSS_WAITS = (30, 60, 120, 240, 240)
# What verifiers tells every agent on a restricted network, on the first user message of each request: by default,
# "do not retry the blocked provider-side operation", even when nothing was removed (to fix upstream: only when a
# capability was). Our VMs block every host and reach their tools through a tunnel, where the true thing to say is
# that a dropped call never arrived; the authors stopped their turns on the default instead of calling again.
NETWORK_NOTICE = (
    "This machine has no network. Its world_* and inspect_* functions reach the host through a tunnel; one that "
    'raises "is unavailable" never arrived: call it again.'
)


def apply_network_notice() -> None:
    """Every request of every agent carries NETWORK_NOTICE in place of verifiers' default; idempotent."""
    dialects.CAPABILITY_NOTICE = NETWORK_NOTICE  # read by append_user_notice on every request


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
    """Why an agent's session ended before its work did: its own limits (the turn cap, the token budget), or an
    error of its sandbox, tunnel or provider, which is none of the agent's (a dropped process stream once read as the
    agent's own limits)."""
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


SETUP_FILES = (
    "input.json",
    "guide.md",
    "schemas.json",
    "schema.sql",
    "world.sqlite",
    "premise.json",
    "organization.json",
)


class SessionEnv(vf.Env[Seats]):
    """An env whose agent (a world's author, a forge's proposer) works in one VM for the whole run, one interaction
    per block of work: the VM provisioned again after a lost session, code's memory rendered before every turn, and
    the notes an attempt kept brought back. A subclass says which seat works its blocks, what the VM is provisioned
    with, and what its blocks are."""

    seat = "author"  # the seat whose agent works in the VM
    seats = ("author", "judge")  # every seat of the run, none trained
    notes = ("plan.md", "recap.md")  # the agent's own files in /task/notes, kept with each attempt

    def __init__(self, settings, store: Store):
        """`settings` is the run's (a world's or a forge's), whose `env` holds its seats."""
        self.settings, self.store = settings, store
        apply_network_notice()
        super().__init__(settings.env)

    @property
    def world(self):
        return self.store.world

    async def setup(self, agents):
        for name in self.seats:
            getattr(agents, name).trainable = False

    def tools_context(self) -> dict:
        """What the agent's world tools are configured with, from the run's state."""
        state = self.store.state
        return context_of(self.settings, state, self.store.root, state.drafts.get("organization"))

    async def take_turn(self, interaction, runtime, prompt: str, attempt: str, mode: str, extra=None) -> None:
        """One turn of the agent: code's memory of the world, rendered fresh (with `extra` memory files, such as the
        solver's tries), then the turn; the agent works through its tools, and the world file is code's again when the
        turn ends."""
        memory = files(self.world, self.settings, self.tools_context()) | (extra or {})
        await upload(runtime, memory, "/task/memory")
        self.store.event(f"{self.seat}_turn", attempt=attempt, mode=mode, day=self.store.state.day)
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
        """The agent's notes as the attempt left them, kept with its world."""
        for name in self.notes:
            if (text := await self.note(runtime, name)) is not None:
                self.store.path(attempt, "notes/" + name).write_text(text)

    async def bring_notes(self, runtime, attempt: str) -> None:
        """The notes an attempt kept, and only those, back in the agent's VM: a day (or a round) starts from where the
        last one ended, and a failed attempt's notes are gone."""
        kept = self.store.root / "attempts" / attempt / "notes"
        notes = {f"notes/{name}": (kept / name).read_bytes() for name in self.notes if (kept / name).exists()}
        await upload(runtime, notes, "/task/notes")

    async def work_in_vm(self, agents) -> None:
        """One agent and one VM for the whole run; one interaction per block of work. A session lost to its sandbox,
        tunnel or provider (or a VM gone mid-run) is none of the agent's limits: after a pause, a fresh VM takes the
        run up where its state says, as a resume does, and the attempt it cut off spends no round. A block may lose
        its session `max_losses` times (the run's setting; a world's 1); one more stops the run."""
        state, losses = self.store.state, {}
        allowed = getattr(self.settings, "max_losses", 1)
        setup = self.provision_task()
        while True:
            try:
                async with getattr(agents, self.seat).provision(setup) as runtime:
                    return await self.blocks(agents, runtime, setup)
            except (SessionLost, SandboxError) as error:
                if cleaning_up():  # a stop cancels the session's streams: it is a stop, not a loss
                    raise
                where = (state.phase, state.day, state.batch)
                losses[where] = n = losses.get(where, 0) + 1
                attempt = self.store.lose()
                self.store.event("session_lost", attempt=attempt, error=str(error)[:2000], again=n <= allowed)
                if n > allowed:
                    raise
                await asyncio.sleep(LOSS_WAITS[min(n, len(LOSS_WAITS)) - 1])

    async def run(self, task, agents):
        return await self.work_in_vm(agents)

    def provision_task(self) -> vf.Task:
        """The task the VM is provisioned with: provisioning runs none of its setup."""
        raise NotImplementedError

    async def blocks(self, agents, runtime, setup) -> None:
        raise NotImplementedError


# ---------------------------------------------------------------------- the world tools both agents in a VM share


def read_arguments() -> dict[str, dict[str, str]]:
    """Each Slack read tool's arguments and their types, as the solver's own tool takes them (required ones so marked)."""

    def kind(hint) -> str:
        text = (
            hint.__name__
            if isinstance(hint, type)
            else str(hint).replace("typing.", "").removesuffix(" | None")
        )
        return re.sub(r"^Optional\[(.*)\]$", r"\1", text)

    out = {}
    for tool in slack.TOOLS:
        method = getattr(SlackToolset, tool)
        hints = typing.get_type_hints(method)
        out[tool] = {
            name: kind(hints[name]) + (" (required)" if p.default is inspect.Parameter.empty else "")
            for name, p in inspect.signature(method).parameters.items()
            if name != "self"
        }
    return out


READS = read_arguments()


class AuthoringToolsConfig(vf.ToolsetConfig):
    """An agent's world tools for one turn: the turn (mode) and its day, the world file and its hash, and the context
    code renders the memory from, a file and its hash. A toolset's config reaches its server in one environment
    variable, which holds at most 128 KB; the context grows with the cast and the days, so it travels as a file."""

    mode: Literal["plan", "day", "fix", "forge"] = "plan"
    day: int = 0
    db_path: str = ""
    db_hash: str = ""
    # settings, task slots, typing cards, the day's drawn conversations, routines, the call log
    context_path: str = ""
    context_hash: str = ""


def tools_config(mode: str, day: int, world: Path, context: dict) -> AuthoringToolsConfig:
    """An agent's world tools for one turn, served host-side, where the world file is and code writes it. The context
    is kept beside the world file, named by its hash (`contexts/<sha256>.json`), so a turn never rewrites another's."""
    text = json.dumps(context).encode()
    key = hashlib.sha256(text).hexdigest()
    path = world.parent / "contexts" / f"{key}.json"
    if not path.exists():
        path.parent.mkdir(exist_ok=True)
        staged = path.with_suffix(".tmp")
        staged.write_bytes(text)
        staged.replace(path)
    tools = AuthoringToolsConfig(
        mode=mode,
        day=day,
        db_path=str(world),
        db_hash=file_hash(world),
        context_path=str(path),
        context_hash=key,
    )
    if tools.colocated or tools.runtime.type != "subprocess" or tools.url is not None:
        raise ValueError("the world file requires a host-side tool server")
    return tools


def context_of(settings, state, store_root: Path, organization: dict | None) -> dict:
    """What an agent's world tools are configured with: the settings, a chapter's brief, each person's typing card and
    seeded self, the conversations code drew per day, the channels' routines, and the call log."""
    routines = {}
    if organization:
        for c in organization["channels"]:
            if c.get("routines"):
                routines[channel_id(c["type"], c.get("name"), c["members"])] = c["routines"]
    agenda = {
        day: [
            {k: s[k] for k in ("channel_id", "part", "situation", "participants", "length")} for s in scenes
        ]
        for day, scenes in (state.plans.get("agenda") or {}).items()
    }
    return {
        "settings": json.loads(settings.model_dump_json()),
        "slots": [s.model_dump(mode="json") for s in state.quota],
        # the ids the proposer's turn may write, read per call
        "writable_file": str(store_root / "writable.json"),
        "cards": cards(state.cast),
        "selves": selves(state.cast),
        "agenda": agenda,
        "routines": routines,
        "log": str(store_root / "world-calls.jsonl"),
    }


TYPES = {"string": "str", "integer": "int", "number": "float", "boolean": "bool", "object": "object"}


def shape(schema: dict, defs: dict) -> str:
    """One JSON schema, compact: a document's fields with ? when optional, a list's bounds, a choice's values."""
    if "$ref" in schema:
        return shape(defs[schema["$ref"].rsplit("/", 1)[1]], defs)
    if "anyOf" in schema:
        return " | ".join(shape(s, defs) for s in schema["anyOf"] if s.get("type") != "null")
    if "enum" in schema:
        return "|".join(json.dumps(v) for v in schema["enum"])
    if schema.get("type") == "array":
        lo, hi = schema.get("minItems"), schema.get("maxItems")
        bound = f"{lo or 0}-{hi} × " if hi else (f"{lo}+ × " if lo else "")
        return f"[{bound}{shape(schema.get('items', {}), defs)}]"
    if "properties" in schema:
        required = set(schema.get("required", []))
        return (
            "{"
            + ", ".join(
                f"{k}{'' if k in required else '?'}: {shape(v, defs)}"
                for k, v in schema["properties"].items()
            )
            + "}"
        )
    return TYPES.get(schema.get("type"), "any")


def arguments(fn) -> str:
    """A tool's arguments as the agent passes them, rendered from the types that check the call: the agent's
    help() shows only a tool's description."""
    hints, out = typing.get_type_hints(fn), []
    for name, param in inspect.signature(fn).parameters.items():
        schema = TypeAdapter(hints[name]).json_schema()
        optional = "?" if param.default is not inspect.Parameter.empty else ""
        out.append(f"{name}{optional}: {shape(schema, schema.get('$defs', {}))}")
    return ", ".join(out) or "none"


class WorldReads:
    """What both agents in a VM read the world with (world_now, world_view, world_sql, world_read), and how every call
    of their world tools runs: one at a time, refused with its reason outside its turns, logged with its arguments and
    outcome to world-calls.jsonl, and a write applied to a trial of the world file. A plain mixin, put before
    vf.Toolset; the tools keep no state of their own, so with the base state a call makes no state round trip through
    the tunnel."""

    settings_model: type = WorldConfig  # what the context's settings validate as

    async def setup(self) -> None:
        self._lock = asyncio.Lock()
        if file_hash(self.config.db_path) != self.config.db_hash:
            raise ValueError("world file hash mismatch")
        if file_hash(self.config.context_path) != self.config.context_hash:
            raise ValueError("context file hash mismatch")
        self.context = json.loads(Path(self.config.context_path).read_bytes())
        self.settings = self.settings_model.model_validate(self.context["settings"])

    def _with_state(self, fn):
        synced = super()._with_state(fn)

        @functools.wraps(synced)
        async def serialized(*args, **kwargs):
            # One call at a time: each write is applied to a trial of the world file and replaces it whole.
            async with self._lock:
                return await synced(*args, **kwargs)

        return serialized

    def register(self, mcp) -> None:
        """Each tool's description ends with its arguments, so the agent reads what a call takes before making one.
        Arguments the tools' schemas refuse never reach a tool: log them too, with a short reason."""
        for fn in discover_decorated(self, "tool"):
            doc = (fn.__doc__ or "").strip()
            name = getattr(fn, "tool_name", None) or fn.__name__
            mcp.add_tool(self._with_state(fn), name=name, description=f"{doc}\n\nArguments: {arguments(fn)}")
        call = mcp.call_tool

        async def call_tool(name, arguments, context=None):
            try:
                return await call(name, arguments, context)
            except Exception as error:
                if not isinstance(error.__cause__, ValidationError):
                    raise
                reason = "; ".join(
                    f"{'.'.join(map(str, e['loc']))}: {e['msg']}"
                    for e in error.__cause__.errors(include_url=False)
                )
                self._log(name, arguments, f"invalid arguments: {reason}")
                raise ValueError(f"{name}: invalid arguments: {reason}") from None

        mcp.call_tool = call_tool

    def _log(self, tool: str, arguments: dict, error: str | None) -> None:
        if log := self.context.get("log"):
            entry = {"time": time.time(), "mode": self.config.mode, "day": self.config.day, "tool": tool}
            with open(log, "a") as stream:
                stream.write(
                    json.dumps(entry | {"ok": error is None, "error": error, "args": arguments}) + "\n"
                )

    def _logged(self, tool: str, arguments: dict, act, *modes: str):
        """Run one call, refused when it is not this turn's, and log it with its arguments and outcome."""
        error = None
        try:
            if modes:
                self._turn(tool, *modes)
            return act()
        except (ValueError, LookupError) as caught:
            error = str(caught)
            raise ValueError(error) from None
        except Exception as caught:  # a fault, not a refusal: logged as failed, and raised as it is
            error = f"{type(caught).__name__}: {caught}"
            raise
        finally:
            self._log(tool, arguments, error)

    def _world(self, writable: bool = False) -> Workspace:
        return Workspace(self.config.db_path, writable=writable)

    def _turn(self, tool: str, *modes: str) -> None:
        if self.config.mode not in modes:
            raise ValueError(
                f"{tool} is for the {' or '.join(modes)} turn; this is the {self.config.mode} turn"
            )

    def _write(self, act):
        world = self._world(writable=True)
        try:
            with world.trial() as copy:
                return act(copy)
        finally:
            world.close()

    @vf.tool
    async def now(self) -> str:
        """The one-pager: the present, today's quota and drawn conversations, each storyline's facts, events, open
        promises, a chapter's brief and style drift."""
        return self._logged("now", {}, lambda: now_page(self._world(), self.settings, self.context))

    @vf.tool
    async def view(self, ref: str) -> str:
        """One page: a person (id, handle or name), a channel (id or #name), a storyline, an event, a fact, a task, a
        conversation (dNN-NNN) or a message (m123), with its latest messages and links."""
        return self._logged("view", {"ref": ref}, lambda: view(self._world(), ref, self.context))

    @vf.tool
    async def sql(self, sql: str, actor_id: str | None = None) -> dict:
        """One read-only SELECT over the whole world file, or, with actor_id, over the world as that person sees it
        (the way gold queries run). At most 200 rows."""

        def act():
            world = self._world()
            return world.gold(actor_id, sql, max_rows=200) if actor_id else world.select(sql, max_rows=200)

        return self._logged("sql", {"sql": sql, "actor_id": actor_id}, act)

    @vf.tool
    async def read(self, actor_id: str, tool: str, arguments: dict[str, JsonValue]) -> dict:
        def act():
            if tool not in READS:
                raise ValueError(f"no read tool {tool!r}: the tools are {', '.join(READS)}")
            if unknown := sorted(set(arguments) - set(READS[tool])):
                raise TypeError(f"{tool} takes {', '.join(READS[tool])}; not {', '.join(unknown)}")
            return slack.read(Workspace(self.config.db_path, actor=actor_id), tool, arguments)

        return self._logged("read", {"actor_id": actor_id, "tool": tool, "arguments": arguments}, act)

    read.__doc__ = (
        "A Slack read tool as any person, by the solver's own tool names without slack_, with the arguments each "
        "takes as the solver's own tool does: "
        + "; ".join(
            f"{tool}({', '.join(f'{a}: {kind}' for a, kind in takes.items())})"
            for tool, takes in READS.items()
        )
        + ". Returns what it showed, structured."
    )
