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
from .contracts import Catalog, SeedPacket, Verdict, load_seed_packet, validate_verdict
from .agents.judge import review_payload
from worldgen_slack.slack.api import digest
from verifiers.v1.clients import EvalClientConfig, ModelContext
from worldgen_slack.dataset import atomic_json


def provenance(config: Config, seeds: SeedPacket | None = None) -> dict:
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


async def run(config: Config, *, catalog: Catalog | None = None, approval: Verdict | None = None) -> dict:
    seeds = load_seed_packet(config.seed_data.path) if config.seed_data else None
    manifest = provenance(config, seeds)
    if (catalog is None) != (approval is None):
        raise ValueError("a fixed catalog requires its approval")
    if catalog is not None:
        validate_verdict(
            approval, review_payload(catalog, None, [task.id for task in catalog.tasks], "catalog")
        )
        if (
            not approval.approved
            or len(catalog.tasks) != config.task_count
            or catalog.sector != config.sector
        ):
            raise ValueError("fixed catalog must be approved and match the run configuration")
        if len(catalog.groups) != (config.task_count + config.group_size - 1) // config.group_size or any(
            sum(task.group_id == group.id for task in catalog.tasks) > config.group_size
            for group in catalog.groups
        ):
            raise ValueError("fixed catalog groups must match the run configuration")
        manifest["fixed_catalog_hash"] = digest(catalog.model_dump(mode="json"))
        manifest["catalog_approval_hash"] = digest(approval.model_dump(mode="json"))
    store = Store(config.output, manifest)

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
        if seeds is not None:
            atomic_json(store.root / "seeds.json", seeds.model_dump(mode="json"))
        if catalog is not None and store.state.catalog is None:
            store.state.catalog = catalog.model_copy(deep=True)
            store.state.reviews["catalog"] = approval.model_dump(mode="json")
            store.state.phase = "build"
            store.save(approved=True)
            store.event("fixed_catalog_loaded", catalog_hash=manifest["fixed_catalog_hash"])
        if store.state.phase != "done":
            env = GenerationEnv(config, store, seeds, fixed_catalog=catalog is not None)
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
        if config.seed_data:
            load_seed_packet(config.seed_data.path)
        print(config.model_dump_json(indent=2))
        return 0
    result = asyncio.run(run(config))
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    sys.exit(main())
