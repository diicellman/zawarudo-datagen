"""Generate one Slack world with its tasks, on native Verifiers agents: uv run --frozen worldgen-slack --config <toml>
(or `worldgen-slack forge --config <toml>`: tasks on a finished world)."""

import argparse
import asyncio
import hashlib
import json
import re
import sys
from pathlib import Path

import verifiers.v1 as vf
from verifiers.v1.clients import EvalClientConfig, ModelContext
from verifiers.v1.runtimes.prime import set_base_sandbox_labels
from verifiers.v1.utils.interrupt import install_interrupt
from worldgen_slack.dataset import atomic_json

from .chronicle import daily
from .config import ROOT, Config, load_config
from .contracts import calendar, census, pick_cast, quota
from .env import GenerationEnv
from .store import Store


def provenance(config: Config) -> dict:
    """Everything a resumed run must share with its start: the configuration (taxonomy included), the pinned
    framework and the lock file."""
    return {
        "config": config.model_dump(mode="json"),
        "verifiers_revision": "ac2ec29",
        "lock_hash": hashlib.sha256((ROOT / "uv.lock").read_bytes()).hexdigest(),
    }


def failure(phase: str, episode) -> str | None:
    """Why a run failed, or None. A run fails when its control flow did not finish. An agent failure the pipeline
    absorbed (a malformed verdict rerun, a corrected author turn) is counted in the summary, not fatal."""
    if phase == "done" and not episode.errors:
        return None
    reason = "; ".join(f"{e.type}: {e.message}" for e in episode.errors)
    return reason or "; ".join(f"{e.type}: {e.message}" for t in episode.traces for e in t.errors) or phase


def run_label(output: Path) -> str:
    """The label every sandbox of a run carries, so one left behind is found by it: prime sandbox list."""
    return "worldgen-" + re.sub(r"[^a-z0-9]+", "-", f"{output.parent.name}-{output.name}".lower()).strip("-")


async def run(config: Config) -> dict:
    set_base_sandbox_labels([run_label(config.output)])
    store = Store(config.output, provenance(config))
    try:
        if store.state.phase != "done":
            env = GenerationEnv(config, store)
            context = ModelContext(
                model=config.env.solver.model, client=EvalClientConfig(), sampling=vf.Sampling()
            )
            seed = vf.Task(vf.TaskData(idx=config.seed, prompt="Generate the shared Slack workspace."))
            async with env.serving():
                episode = await env.run_episode(seed, context)
            for trace in episode.traces:
                store.trace(trace)
            atomic_json(store.root / f"episode-{episode.id}.json", episode.to_record())
            if (reason := failure(store.state.phase, episode)) is not None:
                return store.summary(
                    "incomplete" if "ReviewLimit" in reason else "infrastructure_error", reason
                )
        summary = store.summary("complete")
        store.publish()
        return summary
    except asyncio.CancelledError:
        store.summary("interrupted", "run interrupted; committed progress and available traces are preserved")
        raise
    finally:
        store.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", nargs="?", choices=["forge"], help="forge: tasks on a finished world (forge.py)"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="validate the configuration; spend nothing")
    args = parser.parse_args()
    if args.command == "forge":
        from . import forge

        plan, settings = forge.load_forge(args.config)
        if args.dry_run:
            print(json.dumps(forge.dry_run(plan, settings), indent=1))
            return 0
        install_interrupt()
        try:
            result = asyncio.run(forge.run(plan, settings))
        except KeyboardInterrupt:
            print(
                "interrupted: the forge is stopped and its sandboxes deleted; rerun to resume",
                file=sys.stderr,
            )
            return 130
        print(json.dumps(result, indent=2))
        return 0 if result["status"] == "complete" else 2
    config = load_config(args.config)
    if args.dry_run:
        (occupation, _), *_ = census(config.personas)[1].most_common(1)
        pick_cast(config.personas, config.seed, [], {occupation: config.personas.pool})
        plan = {
            "config": config.model_dump(mode="json"),
            "quota": [
                s.model_dump()
                for s in quota(config.taxonomy, config.tasks.styles, config.seed, config.task_count)
            ],
        }
        dates = [d["date"] for d in calendar(config.seed, config.calendar, "UTC")]
        plan["daily"] = dict(zip(dates, daily(dates, config)))  # each date's messages and their parts
        print(json.dumps(plan, indent=1))
        return 0
    # The first Ctrl-C or SIGTERM unwinds the run: its interactions close, its tool servers and tunnels stop and its
    # VM is deleted; signals during that cleanup are ignored. The checkpoint stays for a resume.
    install_interrupt()
    try:
        result = asyncio.run(run(config))
    except KeyboardInterrupt:
        print("interrupted: the run is stopped and its sandboxes deleted; rerun to resume", file=sys.stderr)
        return 130
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    sys.exit(main())
