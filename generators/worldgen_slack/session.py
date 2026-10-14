"""An agent's session in a VM: files uploaded in one archive, the VM kept awake while only the env works, the
network notice its requests carry, and how a session that ended early is told from the agent's own limits."""

import asyncio
import contextlib
import io
import tarfile
import time

import verifiers.v1 as vf
from verifiers.v1.dialects import base as dialects
from verifiers.v1.errors import SandboxError
from verifiers.v1.utils.interrupt import cleaning_up

from .agents.world import context_of, files
from .config import Config, PipelineConfig
from .store import ReviewLimit, SessionLost, Store

ARCHIVE = "/tmp/task-files.tgz"
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


NOTES = ("plan.md", "recap.md")  # the world author's own files in /task/notes


SETUP_FILES = (
    "input.json",
    "guide.md",
    "schemas.json",
    "schema.sql",
    "world.sqlite",
    "premise.json",
    "organization.json",
)


class SessionEnv(vf.Env[PipelineConfig]):
    """An env whose author works in one VM for the whole run, one interaction per block of work: the VM provisioned
    again after a lost session, code's memory rendered before every turn, and the notes an attempt kept brought back.
    A subclass says which seat works its blocks, what the VM is provisioned with, and what its blocks are."""

    seat = "author"  # the seat whose agent works in the VM
    seats = ("author", "judge", "solver", "witness")  # every seat of the run, none trained

    def __init__(self, settings: Config, store: Store):
        self.settings, self.store = settings, store
        apply_network_notice()
        super().__init__(settings.env)

    @property
    def world(self):
        return self.store.world

    async def setup(self, agents):
        for name in self.seats:
            getattr(agents, name).trainable = False

    def author_context(self) -> dict:
        state = self.store.state
        return context_of(self.settings, state, self.store.root, state.drafts.get("organization"))

    async def author_step(
        self, interaction, runtime, prompt: str, attempt: str, mode: str, extra=None
    ) -> None:
        """One author turn: code's memory of the world, rendered fresh (with `extra` memory files, such as the solver's
        tries), then the turn; the author works through its tools, and the world file is code's again when the turn
        ends."""
        memory = files(self.world, self.settings, self.author_context(), mode) | (extra or {})
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

    async def author_world(self, agents) -> None:
        """One author agent and one VM for the whole world; one interaction per block of work. A session lost to its
        sandbox, tunnel or provider (or a VM gone mid-run) is none of the author's limits: a fresh VM takes the run up
        where its state says, as a resume does, and the attempt it cut off spends no round. A second loss in the same
        block stops the run."""
        state, last = self.store.state, None
        setup = self.provision_task()
        while True:
            try:
                async with getattr(agents, self.seat).provision(setup) as runtime:
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

    async def run(self, task, agents):
        return await self.author_world(agents)

    def provision_task(self) -> vf.Task:
        """The task the VM is provisioned with: provisioning runs none of its setup."""
        raise NotImplementedError

    async def blocks(self, agents, runtime, setup) -> None:
        raise NotImplementedError
