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
from collections.abc import Awaitable, Callable
from pathlib import Path

import verifiers.v1 as vf
from verifiers.v1.clients import EvalClientConfig, ModelContext
from verifiers.v1.runtimes.prime import set_base_sandbox_labels
from verifiers.v1.utils.interrupt import cleaning_up, install_interrupt
from worldgen_slack.release import atomic_json

from .forge import env as forge
from .forge import regrade as forge_regrade
from .forge.config import ForgeConfig, load_forge
from .forge.release import publish
from .store import Store
from .world import chapter, profile
from .world.config import WorldConfig, load_world
from .world.env import WorldEnv, dry_run, provenance

# Each run's episode prompt, and what its interruption calls it.
PARTS = {
    "world": ("Generate the shared Slack workspace.", "run"),
    "chapter": ("Continue the shared Slack workspace.", "chapter"),
    "forge": ("Forge tasks on the finished workspace.", "forge"),
    "regrade": ("", "regrade"),  # no episode: a forge's tries graded again
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
    seed: int,
    start: Callable[[Store], vf.Env],
    finish: Callable[[Store], None] = lambda store: None,
    reopen: Callable[[Store], bool] = lambda store: False,
) -> dict:
    """One run: its env (`start`) as one episode on its run directory, from where the directory left off, then
    `finish` once it is complete; `seed` is the episode's task index. Its summary says how it ended; an interrupted run
    keeps its committed progress."""
    prompt, name = PARTS[part]
    set_base_sandbox_labels([run_label(output)])
    store = Store(output, origin)
    try:
        reopen(store)  # a finished run with work left undone (a forge's failed labels) takes it up
        if store.state.phase != "done":
            env = start(store)
            # Every seat names its model (Seats.secure), so the episode's model is a stand-in that no agent uses: the
            # model of the seat the run's session works in.
            context = ModelContext(
                model=getattr(env.config, env.seat).model, client=EvalClientConfig(), sampling=vf.Sampling()
            )
            task = vf.Task(vf.TaskData(idx=seed, prompt=prompt))
            async with env.serving():
                episode = await env.run_episode(task, context)
            for trace in episode.traces:
                store.trace(trace)
            atomic_json(store.root / f"episode-{episode.id}.json", episode.to_record())
            if (reason := failure(store.state.phase, episode)) is not None:
                # A stop (Ctrl-C, SIGTERM) that reached the run as its streams cancelled is a stop, never resumed.
                return store.summary(
                    "interrupted"
                    if cleaning_up()
                    else "incomplete"
                    if "ReviewLimit" in reason
                    else "infrastructure_error",
                    reason,
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


async def carry_on(kind: str, once: Callable[[], Awaitable[dict]], resumes: int) -> dict:
    """A run that carries on by itself after a stop that is infrastructure's (a provider's, a sandbox's), from where
    its state says, up to `resumes` times; a stop in the same place for the same reason twice is a bug, and ends it."""
    stops = []
    while True:
        summary = await once()
        where = (summary["phase"], summary.get("day"), summary["reason"][:160])
        if summary["status"] != "infrastructure_error" or where in stops or len(stops) == resumes:
            return summary
        stops.append(where)
        print(json.dumps({"event": f"{kind}_resumed", "phase": where[0], "reason": where[2]}), flush=True)


async def run_world(config: WorldConfig) -> dict:
    """A world, carrying on by itself after an infrastructure stop (a judge's sandbox lost twice in one review)."""
    return await carry_on(
        "world",
        lambda: run(
            "world", config.output, provenance(config), config.seed, lambda store: WorldEnv(config, store)
        ),
        config.max_resumes,
    )


async def run_chapter(plan: chapter.ChapterConfig, settings: WorldConfig) -> dict:
    def start(store: Store) -> chapter.ChapterEnv:
        parent = chapter.parent_world(plan.world)
        slots = chapter.brief_slots(plan, settings, chapter.version_of(parent) + 1)
        parent.close()
        return chapter.ChapterEnv(plan, settings, store, slots)

    return await carry_on(
        "chapter",
        lambda: run("chapter", plan.output, chapter.provenance(plan, settings), settings.seed, start),
        settings.max_resumes,
    )


async def run_forge(config: ForgeConfig) -> dict:
    """A forge: its one episode is task 0, as it draws nothing at random. It carries on by itself as a world does."""
    origin = forge.provenance(config)
    return await carry_on(
        "forge",
        lambda: run(
            "forge",
            config.output,
            origin,
            0,
            lambda store: forge.ForgeEnv(config, store),
            publish,
            forge.reopen,
        ),
        config.max_resumes,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "command", nargs="?", choices=["chapter", "forge", "profile", "regrade"], help="a world when omitted"
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--world", type=Path, help="profile: the run directory whose world.sqlite to profile")
    parser.add_argument("--dry-run", action="store_true", help="validate the configuration; spend nothing")
    parser.add_argument(
        "--splits", type=Path, help="regrade: reference batches that split golds into answer and context"
    )
    args = parser.parse_args()
    if args.command == "profile":
        if args.world is None:
            parser.error("profile takes --world")
        print(json.dumps(profile.write(args.world), indent=1))
        return 0
    if args.config is None:
        parser.error("--config is required")
    if args.command == "forge":
        config = load_forge(args.config)
        dry, going = (lambda: forge.dry_run(config)), (lambda: run_forge(config))
    elif args.command == "regrade":
        config = load_forge(args.config)
        dry = lambda: forge_regrade.dry_regrade(config, args.splits)  # noqa: E731
        going = lambda: forge_regrade.regrade(config, args.splits)  # noqa: E731
    elif args.command == "chapter":
        plan, settings = chapter.load_chapter(args.config)
        dry, going = (lambda: chapter.dry_run(plan, settings)), (lambda: run_chapter(plan, settings))
    else:
        config = load_world(args.config)
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
        result = {"status": "interrupted"}
    if result["status"] == "interrupted":
        name = PARTS[args.command or "world"][1]
        print(
            f"interrupted: the {name} is stopped and its sandboxes deleted; rerun to resume", file=sys.stderr
        )
        return 130
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    sys.exit(main())
