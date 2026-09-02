from __future__ import annotations

import asyncio
import hashlib
import json
import shlex
from pathlib import Path
from typing import Any

import verifiers.v1 as vf

from ..slack.models import ScenarioSpec, SlackWorld, TaskContract
from ..slack.validate import CHECKER_RUNNER_SOURCE, MAX_SOURCE_BYTES

WORLD_STUB = """from worldgen_slack.slack.models import SlackWorld, TaskContract


def build(seed: int, contract: TaskContract) -> SlackWorld:
    raise NotImplementedError
"""

BUILDER_PROMPT = """Implement `/task/workspace/world.py`.

It must define `build(seed: int, contract: TaskContract) -> SlackWorld`. Use the supplied scenario,
contract, derived schemas, and builder guide. Do not change the contract. Run `/task/check-world`
after editing. When it exits successfully, stop immediately and reply briefly; do not continue
inspecting or editing after a pass. The file is the authoritative output.
"""


class BuilderData(vf.TaskData):
    generation_seed: int
    interface_id: str


class BuilderConfig(vf.TaskConfig):
    scenario: ScenarioSpec
    contract: TaskContract
    public_seed: int = 0


class BuilderTask(vf.Task[BuilderData, vf.State, BuilderConfig]):
    async def setup(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        package = Path(__file__).resolve().parents[1]
        slack = package / "slack"
        files = {
            "/task/scenario.json": self.config.scenario.model_dump_json(indent=2).encode(),
            "/task/task_contract.json": self.config.contract.model_dump_json(indent=2).encode(),
            "/task/runtime/slack_world.schema.json": json.dumps(
                SlackWorld.model_json_schema(), indent=2, sort_keys=True
            ).encode(),
            "/task/runtime/task_contract.schema.json": json.dumps(
                TaskContract.model_json_schema(), indent=2, sort_keys=True
            ).encode(),
            "/task/runtime/builder_guide.md": (slack / "builder_guide.md").read_bytes(),
            "/task/workspace/world.py": WORLD_STUB.encode(),
            "/task/fixed/worldgen_slack/__init__.py": package.joinpath("__init__.py").read_bytes(),
            "/task/fixed/worldgen_slack/contracts.py": package.joinpath("contracts.py").read_bytes(),
            "/task/fixed/worldgen_slack/slack/__init__.py": slack.joinpath("__init__.py").read_bytes(),
            "/task/fixed/worldgen_slack/slack/models.py": slack.joinpath("models.py").read_bytes(),
            "/task/fixed/worldgen_slack/slack/api.py": slack.joinpath("api.py").read_bytes(),
            "/task/fixed/worldgen_slack/slack/validate.py": slack.joinpath("validate.py").read_bytes(),
        }
        for path, content in files.items():
            await runtime.write(path, content)
        program = await runtime.prepare_uv_script(CHECKER_RUNNER_SOURCE)
        command = [
            *program,
            "check-candidate",
            "--candidate",
            "/task/workspace/world.py",
            "--contract",
            "/task/task_contract.json",
            "--output",
            "/task/public-result.json",
            "--seeds",
            str(self.config.public_seed),
        ]
        wrapper = (
            "#!/bin/sh\n"
            "rm -f /task/public-result.json\n"
            "ulimit -f 8192\n"
            "PYTHONPATH=/task/fixed "
            + shlex.join(command)
            + " > /task/public.stdout 2> /task/public.stderr\n"
            "status=$?\n"
            "tail -c 20000 /task/public.stdout 2>/dev/null || true\n"
            "tail -c 20000 /task/public.stderr >&2 2>/dev/null || true\n"
            "exit $status\n"
        )
        await runtime.write("/task/check-world", wrapper.encode())
        result = await runtime.run(["chmod", "+x", "/task/check-world"], {})
        if result.exit_code:
            raise RuntimeError(f"could not prepare public checker: {result.stderr[-500:]}")


def make_builder_task(
    *,
    generation_seed: int,
    interface_id: str,
    scenario: ScenarioSpec,
    contract: TaskContract,
    public_seed: int,
) -> BuilderTask:
    return BuilderTask(
        BuilderData(
            idx=generation_seed,
            name=contract.task_slug or f"slack-builder-{generation_seed:08d}",
            prompt=BUILDER_PROMPT,
            network_allow=[],
            network_block=["*"],
            generation_seed=generation_seed,
            interface_id=interface_id,
        ),
        BuilderConfig(scenario=scenario, contract=contract, public_seed=public_seed),
    )


async def _public_check(runtime: vf.Runtime, timeout_seconds: float) -> dict[str, Any]:
    try:
        async with asyncio.timeout(timeout_seconds):
            process = await runtime.run(["/task/check-world"], {})
    except TimeoutError:
        result = {"ok": False, "errors": ["public checker timed out"]}
    else:
        if process.exit_code not in (0, 2):
            detail = (process.stderr or process.stdout)[-1_000:] or "public checker failed"
            if not await runtime.alive():
                raise vf.SandboxError("builder runtime stopped during public validation")
            raise RuntimeError(detail)
        raw = await runtime.read("/task/public-result.json", max_bytes=1_000_000)
        try:
            result = json.loads(raw)
        except json.JSONDecodeError as exc:
            result = {"ok": False, "errors": [f"malformed public result: {exc}"]}
        if not isinstance(result, dict):
            result = {"ok": False, "errors": ["public result must be a JSON object"]}
    errors = [str(value) for value in result.get("static_errors", [])]
    for report in result.get("reports", []):
        if not isinstance(report, dict) or report.get("ok"):
            continue
        if report.get("error"):
            errors.append(str(report["error"]))
        errors.extend(
            str(check.get("detail"))
            for check in report.get("checks", [])
            if isinstance(check, dict) and not check.get("ok")
        )
    errors.extend(str(value) for value in result.get("errors", []))
    if not result.get("ok") and not errors:
        errors.append("public checker rejected the candidate")
    ok = bool(result.get("ok")) and not errors
    return {
        "ok": ok,
        "report": "\n".join(["RESULT PASS"] if ok else ["RESULT FAIL", *errors]),
        "result": result,
    }


def _metadata(trace: vf.Trace, source: bytes, checks: list[dict[str, Any]]) -> dict[str, Any]:
    runtime_id = trace.agent.runtime.id if trace.agent.runtime else None
    usage = trace.usage
    return {
        "trace_id": trace.id,
        "runtime_id": runtime_id,
        "turns": trace.num_turns,
        "tokens": int(usage.total_tokens or 0) if usage else 0,
        "wall_seconds": trace.timing.agent.duration,
        "source_lines": len(source.splitlines()),
        "source_hash": "sha256:" + hashlib.sha256(source).hexdigest(),
        "public_checker_attempts": len(checks),
    }


async def build_world(
    task: BuilderTask,
    agent: vf.Agent,
    *,
    max_checks: int,
    check_timeout_seconds: float,
    progress: Any | None = None,
) -> tuple[vf.Trace, bytes]:
    attempts: list[dict[str, Any]] = []
    async with agent.provision(task) as runtime:
        async with agent.interaction(task, runtime=runtime) as interaction:
            segment = await interaction.turn()
            for number in range(1, max_checks + 1):
                if progress is not None:
                    progress.stage_updated(
                        task.data.generation_seed,
                        event="check_started",
                        check=number,
                    )
                source = await runtime.read("/task/workspace/world.py", max_bytes=MAX_SOURCE_BYTES)
                checked = await _public_check(runtime, check_timeout_seconds)
                attempts.append({"attempt": number, **checked})
                if progress is not None:
                    progress.stage_updated(
                        task.data.generation_seed,
                        event="check_finished",
                        check=number,
                        check_ok=bool(checked["ok"]),
                    )
                if checked["ok"] or number == max_checks or segment.terminated:
                    break
                if progress is not None:
                    progress.stage_updated(
                        task.data.generation_seed,
                        event="repair",
                        repair_count=number,
                    )
                segment = await interaction.turn(
                    "Public checker feedback:\n"
                    + checked["report"]
                    + "\nRepair only /task/workspace/world.py, then rerun /task/check-world."
                )
            trace = interaction.trace
    if not trace.ok:
        raise RuntimeError("builder rollout failed")
    trace.info["candidate_source"] = source.decode("utf-8", errors="replace")
    trace.info["public_checks"] = attempts
    trace.info["builder_metadata"] = _metadata(trace, source, attempts)
    return trace, source


__all__ = [
    "BuilderData",
    "BuilderTask",
    "WORLD_STUB",
    "build_world",
    "make_builder_task",
]
