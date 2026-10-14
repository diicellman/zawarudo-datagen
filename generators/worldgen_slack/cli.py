"""worldgen-slack: write one Slack world on native Verifiers agents, continue a finished one, or forge tasks on one.

uv run --frozen worldgen-slack --config <toml>            a world
uv run --frozen worldgen-slack chapter --config <toml>    a finished world continued forward in time, into a new version
uv run --frozen worldgen-slack forge --config <toml>      tasks on a finished world
uv run --frozen worldgen-slack profile --world <run dir>  a finished world's material for tasks"""

import argparse
import asyncio
import json
import re
import sys
from collections.abc import Callable
from pathlib import Path

import verifiers.v1 as vf
from verifiers.v1.clients import EvalClientConfig, ModelContext
from verifiers.v1.runtimes.prime import set_base_sandbox_labels
from verifiers.v1.utils.interrupt import install_interrupt
from worldgen_slack.release import atomic_json

from .world import chapter
from .forge import env as forge
from .forge import profile
from .forge.release import publish
from .config import Config, load_config
from .world.env import WorldEnv, dry_run, provenance
from .store import Store

# Each run's episode prompt, and what its interruption calls it.
PARTS = {
    "world": ("Generate the shared Slack workspace.", "run"),
    "chapter": ("Continue the shared Slack workspace.", "chapter"),
    "forge": ("Forge tasks on the finished workspace.", "forge"),
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


async def run(
    part: str,
    output: Path,
    origin: dict,
    settings: Config,
    start: Callable[[Store], vf.Env],
    finish: Callable[[Store], None] = lambda store: None,
) -> dict:
    """One run: its env (`start`) as one episode on its run directory, from where the directory left off, then
    `finish` once it is complete. Its summary says how it ended; an interrupted run keeps its committed progress."""
    prompt, name = PARTS[part]
    set_base_sandbox_labels([run_label(output)])
    store = Store(output, origin)
    try:
        if store.state.phase != "done":
            env = start(store)
            context = ModelContext(
                model=settings.env.solver.model, client=EvalClientConfig(), sampling=vf.Sampling()
            )
            seed = vf.Task(vf.TaskData(idx=settings.seed, prompt=prompt))
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
        finish(store)
        return summary
    except asyncio.CancelledError:
        store.summary(
            "interrupted", f"{name} interrupted; committed progress and available traces are preserved"
        )
        raise
    finally:
        store.close()


async def run_world(config: Config) -> dict:
    return await run(
        "world", config.output, provenance(config), config, lambda store: WorldEnv(config, store)
    )


async def run_chapter(plan: chapter.ChapterConfig, settings: Config) -> dict:
    def start(store: Store) -> chapter.ChapterEnv:
        parent = chapter.parent_world(plan.world)
        slots = chapter.brief_slots(plan, settings, chapter.version_of(parent) + 1)
        parent.close()
        return chapter.ChapterEnv(plan, settings, store, slots)

    return await run("chapter", plan.output, chapter.provenance(plan, settings), settings, start)


async def run_forge(plan: forge.ForgeConfig, settings: Config) -> dict:
    def release(store: Store) -> None:
        # an archive that kept nothing has no release
        if store.world.db.execute("SELECT 1 FROM tasks").fetchone():
            publish(store)

    origin = forge.provenance(plan, settings)
    return await run(
        "forge", plan.output, origin, settings, lambda store: forge.ForgeEnv(plan, settings, store), release
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "command", nargs="?", choices=["chapter", "forge", "profile"], help="a world when omitted"
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--world", type=Path, help="profile: the run directory whose world.sqlite to profile")
    parser.add_argument("--dry-run", action="store_true", help="validate the configuration; spend nothing")
    args = parser.parse_args()
    if args.command == "profile":
        if args.world is None:
            parser.error("profile takes --world")
        print(json.dumps(profile.write(args.world), indent=1))
        return 0
    if args.config is None:
        parser.error("--config is required")
    if args.command == "forge":
        plan, settings = forge.load_forge(args.config)
        dry, going = (lambda: forge.dry_run(plan, settings)), (lambda: run_forge(plan, settings))
    elif args.command == "chapter":
        plan, settings = chapter.load_chapter(args.config)
        dry, going = (lambda: chapter.dry_run(plan, settings)), (lambda: run_chapter(plan, settings))
    else:
        config = load_config(args.config)
        dry, going = (lambda: dry_run(config)), (lambda: run_world(config))
    if args.dry_run:
        print(json.dumps(dry(), indent=1))
        return 0
    # The first Ctrl-C or SIGTERM unwinds the run: its interactions close, its tool servers and tunnels stop and its
    # VM is deleted; signals during that cleanup are ignored. The checkpoint stays for a resume.
    install_interrupt()
    try:
        result = asyncio.run(going())
    except KeyboardInterrupt:
        name = PARTS[args.command or "world"][1]
        print(
            f"interrupted: the {name} is stopped and its sandboxes deleted; rerun to resume", file=sys.stderr
        )
        return 130
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    sys.exit(main())
