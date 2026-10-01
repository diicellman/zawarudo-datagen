"""Run shared-workspace backward generation with native Verifiers agents."""

import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path

import verifiers.v1 as vf
from .config import ROOT, Config, load_config
from .env import GenerationEnv
from .store import Store
from .contracts import SeedPacket, census, load_seed_packet, pick_cast
from worldgen_slack.slack.api import digest
from verifiers.v1.clients import EvalClientConfig, ModelContext
from worldgen_slack.dataset import atomic_json


def provenance(config: Config, seeds: SeedPacket | None = None) -> dict:
    if (config.seed_data is None) != (seeds is None):
        raise ValueError("seed configuration and loaded packet must agree")
    result = {
        "config": config.model_dump(
            mode="json", exclude={k for k in ("seed_data", "personas") if getattr(config, k) is None}
        ),
        "verifiers_revision": "ac2ec29",
        "lock_hash": hashlib.sha256((ROOT / "uv.lock").read_bytes()).hexdigest(),
    }
    if seeds is not None:
        result["seed_data_hash"] = digest(seeds.model_dump(mode="json"))
    return result


async def run(config: Config) -> dict:
    seeds = load_seed_packet(config.seed_data.path) if config.seed_data else None
    manifest = provenance(config, seeds)
    store = Store(config.output, manifest)

    try:
        if seeds is not None:
            atomic_json(store.root / "seeds.json", seeds.model_dump(mode="json"))
        if store.state.phase != "done":
            env = GenerationEnv(config, store, seeds)
            context = ModelContext(
                model=config.env.solver.model,
                client=EvalClientConfig(),
                sampling=vf.Sampling(),
            )
            seed = vf.Task(vf.TaskData(idx=config.seed, prompt="Generate the shared Slack workspace."))
            async with env.serving():
                episode = await env.run_episode(seed, context)
            for trace in episode.traces:
                store.trace(trace)
            atomic_json(store.root / f"episode-{episode.id}.json", episode.to_record())
            if not episode.ok and (
                store.state.phase != "done"
                or episode.errors
                or any(
                    not trace.ok and trace.agent.name not in {"solver", "writer"} for trace in episode.traces
                )
            ):
                reason = "; ".join(f"{e.type}: {e.message}" for e in episode.errors)
                reason = reason or "; ".join(
                    f"{e.type}: {e.message}" for t in episode.traces for e in t.errors
                )
                status = "incomplete" if "ReviewLimit" in reason else "infrastructure_error"
                return store.summary(status, reason)
        summary = store.summary("complete")
        store.publish()
        return summary
    except asyncio.CancelledError:
        store.summary(
            "interrupted",
            "run interrupted; committed progress and available traces are preserved",
        )
        raise
    finally:
        store.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.dry_run:
        if config.seed_data:
            load_seed_packet(config.seed_data.path)
        if config.personas:
            pick_cast(config.personas, config.seed, [], list(census(config.personas)[1]))
        print(config.model_dump_json(indent=2))
        return 0
    result = asyncio.run(run(config))
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    sys.exit(main())
