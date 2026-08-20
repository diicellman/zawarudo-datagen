from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import tomllib
from pathlib import Path
from typing import Any, Self

import verifiers.v1 as vf
from pydantic import BaseModel, ConfigDict, Field, model_validator
from verifiers.v1.clients import EvalClientConfig, ModelContext
from verifiers.v1.configs.retries import RetryConfig

from .contracts import (
    GenerationSeedData,
    QualityFilterConfig,
    authentication_failure,
    redact_secrets,
)
from .env import SlackDataGenerationConfig, SlackDataGenerationEnv

_EPISODE_RETRY_TYPES = [
    "ProviderError",
    "SandboxError",
    "HarnessError",
    "InterceptionError",
    "TunnelError",
    "ToolsetError",
    "EnvError",
]


class GenerationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str = "slack-generator-v0"
    default_model: str = "z-ai/glm-5.2"
    count: int = Field(default=10, ge=1)
    concurrency: int = Field(default=2, ge=1)
    output: Path = Path("data/slack-v0")
    max_item_retries: int = Field(default=1, ge=0, le=3)
    client: EvalClientConfig = EvalClientConfig()
    sampling: vf.Sampling = vf.Sampling(temperature=0.2, max_tokens=16384)
    env: SlackDataGenerationConfig
    quality_filter: QualityFilterConfig

    @model_validator(mode="after")
    def apply_runner_settings(self) -> Self:
        env_values = self.env.model_dump()
        client = self.client.model_dump(mode="json")
        headers = client.pop("headers", {})
        runner_provenance = {
            "default_model": self.default_model,
            "client": client,
            "header_names": sorted(headers),
            "headers_hash": "sha256:"
            + hashlib.sha256(json.dumps(headers, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "sampling": self.sampling.model_dump(mode="json"),
            "count": self.count,
            "concurrency": self.concurrency,
        }
        env_values.update(
            {
                "output_dir": self.output,
                "run_id": self.run_id,
                "quality_filter": self.quality_filter,
                "runner_provenance": runner_provenance,
                "retries": RetryConfig(
                    max_retries=self.max_item_retries,
                    include=_EPISODE_RETRY_TYPES,
                ),
            }
        )
        self.env = SlackDataGenerationConfig.model_validate(env_values)
        return self


def load_generation_config(path: str | Path, overrides: dict[str, Any] | None = None) -> GenerationConfig:
    config_path = Path(path).expanduser().resolve()
    try:
        values = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"could not load generation config: {config_path}") from exc
    values.update({key: value for key, value in (overrides or {}).items() if value is not None})
    return GenerationConfig.model_validate(values)


def _select_tasks(env: SlackDataGenerationEnv, count: int) -> list[vf.Task]:
    completed = env.writer.completed_seeds
    tasks = list(env.taskset.head(count))
    for task in tasks:
        if not isinstance(task.data, GenerationSeedData):
            raise TypeError("generation taskset yielded an unexpected task type")
    return [task for task in tasks if task.data.generation_seed not in completed]


def _authentication_failure(episode: vf.Episode) -> tuple[str, int] | None:
    errors = [*episode.errors]
    for trace in episode.traces:
        errors.extend(trace.errors)
    return authentication_failure(errors)


async def generate(config: GenerationConfig) -> dict[str, Any]:
    env = SlackDataGenerationEnv(config.env)
    context = ModelContext(
        model=config.default_model,
        client=config.client,
        sampling=config.sampling,
    )
    tasks = _select_tasks(env, config.count)
    semaphore = asyncio.Semaphore(config.concurrency)

    async def run_task(task: vf.Task) -> vf.Episode:
        slot = env.slots(task)[0]

        async def complete(episode: vf.Episode) -> None:
            if auth := _authentication_failure(episode):
                auth_error, _ = auth
                safe_auth_error = str(redact_secrets(auth_error))[:1_000]
                env.writer.write_setup_failure(
                    generation_seed=task.data.generation_seed,
                    episode=episode,
                    reason=auth_error,
                    traces=env.pop_auth_failure_traces(task.data.generation_seed),
                )
                raise RuntimeError(
                    "Prime authentication failed. Run `prime login` and verify Prime Inference "
                    f"access. Provider detail: {safe_auth_error}"
                )
            await env.persist_episode(task, episode)

        return await env.run_slot(
            slot,
            context,
            semaphore=semaphore,
            on_complete=complete,
        )

    async with env.serving():
        async with asyncio.TaskGroup() as group:
            futures = [group.create_task(run_task(task)) for task in tasks]
    for future in futures:
        future.result()
    return env.writer.write_summary()


def _print_summary(summary: dict[str, Any]) -> None:
    distribution = summary["quality_score_distribution"]
    semantic = summary["semantic_status_counts"]
    status = summary["item_status_counts"]
    lines = [
        ("run_id", summary["run_id"]),
        ("attempted", summary["attempted"]),
        ("semantically solved", semantic.get("solved", 0)),
        ("semantically challenging", semantic.get("challenging", 0)),
        ("hard-gate rejected", summary["hard_gate_rejected"]),
        ("quality-threshold rejected", summary["quality_threshold_rejected"]),
        ("criterion-floor rejected", summary["criterion_floor_rejected"]),
        ("written to dataset", summary["written_to_dataset"]),
        (
            "judge score mean / median / p10 / p90",
            " / ".join(
                "n/a" if distribution[key] is None else f"{distribution[key]:.3f}"
                for key in ("mean", "median", "p10", "p90")
            ),
        ),
        (
            "configured acceptance threshold",
            summary["quality_filter"]["min_accept_score"],
        ),
        ("synthesizer failures", summary["synthesizer_failures"]),
        ("builder failures", summary["builder_failures"]),
        ("judge protocol failures", summary["judge_protocol_failures"]),
        ("infrastructure errors", summary["infrastructure_errors"]),
        ("other status counts", status),
        ("output directory", summary["output"]),
    ]
    for label, value in lines:
        print(f"{label}: {value}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="worldgen-slack")
    subparsers = parser.add_subparsers(dest="command", required=True)
    command = subparsers.add_parser("generate", help="generate a Slack QA/world dataset")
    command.add_argument("--config", type=Path, required=True)
    command.add_argument("--count", type=int)
    command.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command != "generate":
        raise ValueError(f"unsupported command: {args.command}")
    config = load_generation_config(
        args.config,
        {"count": args.count, "output": args.output},
    )
    summary = asyncio.run(generate(config))
    _print_summary(summary)
    return 0


__all__ = ["GenerationConfig", "generate", "load_generation_config", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
