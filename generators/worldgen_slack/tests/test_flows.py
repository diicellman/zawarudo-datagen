"""The scripted runs end to end: a world with no tasks of its own through the CLI, the forge on it, a measure of a
finished run's tasks, and a chapter of a world with tasks of its own, with a forge on the version it leaves."""

from .flows import run_flows


async def test_flows(tmp_path):
    await run_flows(tmp_path / "author")
