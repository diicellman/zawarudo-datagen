"""Run shared-workspace backward generation with native Verifiers agents."""

import argparse
import asyncio
import contextlib
import hashlib
import json
import sys
from pathlib import Path

import verifiers.v1 as vf
from .config import ROOT, Config, load_config
from .env import GenerationEnv
from .store import Store
from verifiers.v1.clients import EvalClientConfig, ModelContext
from worldgen_slack.dataset import atomic_json


def provenance(config: Config) -> dict:
    return {
        "config": config.model_dump(mode="json"),
        "verifiers_revision": "ac2ec29",
        "lock_hash": hashlib.sha256((ROOT / "uv.lock").read_bytes()).hexdigest(),
    }


async def run(config: Config) -> dict:
    store = Store(config.output, provenance(config))

    async def heartbeat():
        while True:
            await asyncio.sleep(10)
            store.event(
                "heartbeat",
                attempt=store.state.active_attempt,
                built_groups=len(store.state.built_groups),
                evaluated=len(store.state.evaluation),
            )

    pulse = asyncio.create_task(heartbeat())
    try:
        if store.state.phase != "done":
            env = GenerationEnv(config, store)
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
                or any(not trace.ok and trace.agent.name != "solver" for trace in episode.traces)
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
        pulse.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await pulse
        store.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.dry_run:
        print(config.model_dump_json(indent=2))
        return 0
    result = asyncio.run(run(config))
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    sys.exit(main())
