"""Run three fixed-catalog pairs through the existing generator; preserve every attempted arm."""

import argparse
import asyncio
import hashlib
import json
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import Field

from generators.worldgen_slack.config import ROOT, SeedDataConfig, load_config
from generators.worldgen_slack.contracts import Catalog, Verdict, load_seed_packet, validate_verdict
from generators.worldgen_slack.agents.judge import review_payload
from generators.worldgen_slack.generate import run
from worldgen_slack.dataset import atomic_json, read_json
from worldgen_slack.slack.api import digest
from worldgen_slack.slack.models import NonEmptyText, StrictModel


class Study(StrictModel):
    output: NonEmptyText
    seed_path: NonEmptyText
    configs: list[NonEmptyText] = Field(min_length=3, max_length=3)
    prior_reported_cost_usd: float = Field(default=0, ge=0, lt=25, allow_inf_nan=False)


def inputs(study):
    output = (ROOT / study.output).resolve()
    if not output.is_relative_to(ROOT / "data"):
        raise ValueError("study output must be inside data/")
    seed_path = (ROOT / study.seed_path).resolve()
    packet = load_seed_packet(seed_path)
    cases = []
    for path in study.configs:
        config = load_config(ROOT / path)
        state = read_json(config.output / "state.json")
        catalog_data = read_json(config.output / "artifacts" / f"{state['catalog']}.json")
        if digest(catalog_data) != state["catalog"]:
            raise ValueError("source catalog hash differs")
        catalog = Catalog.model_validate(catalog_data)
        approval = Verdict.model_validate(state["reviews"]["catalog"])
        validate_verdict(
            approval, review_payload(catalog, None, [task.id for task in catalog.tasks], "catalog")
        )
        if (
            not approval.approved
            or config.task_count != 10
            or config.group_size != 5
            or config.target_messages != 300
        ):
            raise ValueError(
                "study requires approved ten-task catalogs, groups of five, and 300-message targets"
            )
        cases.append((config, catalog, approval))
    if len({catalog.workspace_id for _, catalog, _ in cases}) != 3:
        raise ValueError("study requires three distinct workspaces")
    manifest = {
        "study": study.model_dump(),
        "reported_budget_usd": 30,
        "dispatch_stop_usd": 25,
        "seed_data_hash": digest(packet.model_dump(mode="json")),
        "cases": [
            dict(
                config=config.model_dump(mode="json"),
                catalog=catalog.model_dump(mode="json"),
                approval=approval.model_dump(mode="json"),
            )
            for config, catalog, approval in cases
        ],
        "source_hashes": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((ROOT / "generators/worldgen_slack").rglob("*.py"))
        },
        "runner_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "lock_hash": hashlib.sha256((ROOT / "uv.lock").read_bytes()).hexdigest(),
    }
    return output, seed_path, cases, manifest


async def execute(study, arm: Literal["seeded", "unseeded"] | None = None):
    output, seed_path, cases, manifest = inputs(study)
    budget = 25 - study.prior_reported_cost_usd
    if arm is not None:
        output = output / arm
        budget /= 2
        manifest = {**manifest, "arm": arm, "dispatch_budget_usd": budget}
    manifest_path = output / "study.json"
    if manifest_path.exists():
        if read_json(manifest_path) != manifest:
            raise ValueError("study inputs changed; use a new output directory")
    elif output.exists() and any(output.iterdir()):
        raise ValueError("study output is nonempty without a manifest")
    else:
        atomic_json(manifest_path, manifest)
    results, spent = [], 0.0
    for index, (source, catalog, approval) in enumerate(cases):
        for label, seeded in zip(("a", "b"), (False, True) if index % 2 == 0 else (True, False), strict=True):
            if arm is not None and seeded != (arm == "seeded"):
                continue
            name = f"case-{index + 1}-{label}"
            root = output / name
            if (root / "run.json").exists():
                if not (root / "summary.json").exists():
                    raise ValueError(f"{name} was interrupted without accounting; inspect before continuing")
                result = read_json(root / "summary.json")
                if result["status"] == "running":
                    raise ValueError(f"{name} was interrupted; inspect accounting before continuing")
            elif spent >= budget:
                result = {"status": "budget_stopped", "reported_model_cost": 0}
            else:
                config = source.model_copy(
                    update={
                        "output": root,
                        "seed_data": SeedDataConfig(path=seed_path) if seeded else None,
                        "research_budget_usd": budget - spent,
                    }
                )
                result = await run(config, catalog=catalog, approval=approval)
            spent += result["reported_model_cost"]
            results.append({"run": name, "sector": source.sector, "seeded": seeded, **result})
            atomic_json(output / "results.json", {"reported_model_cost": spent, "arms": results})
            print(json.dumps(dict(run=name, status=result["status"], reported_model_cost=spent)), flush=True)
            if result["status"] == "infrastructure_error":
                raise RuntimeError(f"{name} hit an infrastructure error; inspect before further dispatch")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--arm", choices=("seeded", "unseeded"), help="Run one half in its own directory")
    args = parser.parse_args()
    study = Study.model_validate(tomllib.loads(args.config.read_text()))
    if args.dry_run:
        output, _, cases, manifest = inputs(study)
        print(
            json.dumps(
                dict(
                    output=str(output / args.arm if args.arm else output),
                    pairs=len(cases),
                    arm=args.arm,
                    seed_data_hash=manifest["seed_data_hash"],
                    reported_budget_usd=30,
                    dispatch_stop_usd=25,
                    prior_reported_cost_usd=study.prior_reported_cost_usd,
                    dispatch_budget_usd=(25 - study.prior_reported_cost_usd) / (2 if args.arm else 1),
                ),
                indent=2,
            )
        )
    else:
        asyncio.run(execute(study, args.arm))


if __name__ == "__main__":
    main()
