"""An agent's session in a VM: files uploaded in one archive, the VM kept awake while only the env works, the
network notice its requests carry, and how a session that ended early is told from the agent's own limits."""

import asyncio
import contextlib
import io
import tarfile
import time

from verifiers.v1.dialects import base as dialects
from verifiers.v1.errors import SandboxError

from .store import ReviewLimit, SessionLost

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
