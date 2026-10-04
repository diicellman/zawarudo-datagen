"""Generate one Slack world with its tasks, on native Verifiers agents: uv run --frozen worldgen-slack --config <toml>"""

import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path

import verifiers.v1 as vf
from verifiers.v1.clients import EvalClientConfig, ModelContext
from worldgen_slack.dataset import atomic_json
from worldgen_slack.db import digest

from .chronicle import daily
from .config import ROOT, Config, load_config
from .contracts import SeedPacket, calendar, census, load_seed_packet, pick_cast, quota
from .env import GenerationEnv
from .store import Store


def provenance(config: Config, seeds: SeedPacket | None = None) -> dict:
    """Everything a resumed run must share with its start: the configuration (taxonomy included), the pinned
    framework, the lock file and the seed packet."""
    if (config.seed_data is None) != (seeds is None):
        raise ValueError("seed configuration and loaded packet must agree")
    result = {
        "config": config.model_dump(
            mode="json", exclude={"seed_data"} if config.seed_data is None else set()
        ),
        "verifiers_revision": "ac2ec29",
        "lock_hash": hashlib.sha256((ROOT / "uv.lock").read_bytes()).hexdigest(),
    }
    if seeds is not None:
        result["seed_data_hash"] = digest(seeds.model_dump(mode="json"))
    return result


def failure(phase: str, episode) -> str | None:
    """Why a run failed, or None. A run fails when its control flow did not finish. An agent failure the pipeline
    absorbed (a malformed verdict rerun, a corrected author turn) is counted in the summary, not fatal."""
    if phase == "done" and not episode.errors:
        return None
    reason = "; ".join(f"{e.type}: {e.message}" for e in episode.errors)
    return reason or "; ".join(f"{e.type}: {e.message}" for t in episode.traces for e in t.errors) or phase


async def run(config: Config) -> dict:
    seeds = load_seed_packet(config.seed_data.path) if config.seed_data else None
    store = Store(config.output, provenance(config, seeds))
    try:
        if store.state.phase != "done":
            env = GenerationEnv(config, store, seeds)
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
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--dry-run", action="store_true", help="validate the configuration and seed data; spend nothing"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if args.dry_run:
        if config.seed_data:
            load_seed_packet(config.seed_data.path)
        (occupation, _), *_ = census(config.personas)[1].most_common(1)
        pick_cast(config.personas, config.seed, [], {occupation: config.personas.pool})
        plan = {
            "config": config.model_dump(mode="json"),
            "quota": quota(config.taxonomy, config.tasks.styles, config.seed, config.tasks.count),
        }
        if config.author.enabled:  # the world author's days: each date's messages and their parts
            dates = [d["date"] for d in calendar(config.seed, config.calendar, "UTC")]
            plan["daily"] = dict(zip(dates, daily(dates, config)))
        print(json.dumps(plan, indent=1))
        return 0
    result = asyncio.run(run(config))
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    sys.exit(main())
