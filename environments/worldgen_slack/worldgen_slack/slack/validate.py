from __future__ import annotations

import argparse
import ast
import asyncio
import hashlib
import importlib.util
import inspect
import json
import re
import shlex
import sys
import unicodedata
from pathlib import Path
from typing import Any, Callable, Iterable

import verifiers.v1 as vf

from ..contracts import CheckResult, FailureOwner, ValidationReport
from .api import SlackAPI, SlackNotFoundError
from .models import SlackWorld, TaskContract

SAFE_IMPORT_ROOTS = {
    "__future__",
    "collections",
    "dataclasses",
    "datetime",
    "decimal",
    "enum",
    "functools",
    "hashlib",
    "itertools",
    "math",
    "pydantic",
    "random",
    "string",
    "typing",
    "worldgen_slack.slack.models",
}
UNSAFE_NAMES = {
    "__builtins__",
    "__import__",
    "breakpoint",
    "compile",
    "delattr",
    "dir",
    "eval",
    "exec",
    "getattr",
    "globals",
    "help",
    "input",
    "locals",
    "memoryview",
    "open",
    "setattr",
    "vars",
}
UNSAFE_ATTRIBUTES = {
    "chdir",
    "chmod",
    "connect",
    "fork",
    "kill",
    "open",
    "popen",
    "rename",
    "rmdir",
    "spawn",
    "system",
    "unlink",
    "urlopen",
    "write_bytes",
    "write_text",
}
UNSAFE_IMPORT_MEMBERS = (
    UNSAFE_NAMES
    | UNSAFE_ATTRIBUTES
    | {
        "os",
        "pathlib",
        "requests",
        "shutil",
        "socket",
        "subprocess",
        "sys",
        "tempfile",
        "urllib",
    }
)

PROTECTED_IMPORT_MEMBERS = {
    "__future__": {"annotations"},
    "pydantic": {
        "BaseModel",
        "ConfigDict",
        "Field",
        "RootModel",
        "ValidationError",
        "computed_field",
        "field_serializer",
        "field_validator",
        "model_serializer",
        "model_validator",
    },
    "typing": {
        "Annotated",
        "Any",
        "Callable",
        "ClassVar",
        "Dict",
        "Final",
        "Generic",
        "Iterable",
        "Iterator",
        "List",
        "Literal",
        "Mapping",
        "NamedTuple",
        "Optional",
        "Protocol",
        "Sequence",
        "Set",
        "Self",
        "Tuple",
        "TypeAlias",
        "TypeVar",
        "TypedDict",
        "Union",
        "cast",
    },
    "worldgen_slack.slack.models": {
        "AnswerSpec",
        "Conversation",
        "EvidenceRequirement",
        "GoldCall",
        "Message",
        "Reaction",
        "SlackWorld",
        "StrictModel",
        "TaskContract",
        "User",
    },
}
PRIVATE_FIELD_TERMS = {
    "canonical_answer",
    "forbidden_claims",
    "gold_calls",
    "required_claims",
    "required_evidence",
    "task_contract",
}
PROMPT_INJECTION_TERMS = (
    "ignore previous instructions",
    "ignore all previous",
    "system prompt",
    "call the tool",
    "use the tool",
    "tool instructions",
)
MAX_COUNTS = {
    "users": 40,
    "conversations": 15,
    "messages": 150,
    "reactions": 100,
}
MAX_SOURCE_BYTES = 512_000
MAX_RESULT_BYTES = 3_000_000
MAX_SNAPSHOT_BYTES = 512_000


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    value = re.sub(r"[^\w\s-]", " ", value)
    return " ".join(value.split())


def canonical_world_bytes(world: SlackWorld) -> bytes:
    return json.dumps(
        world.model_dump(mode="json"),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def canonical_world_hash(world: SlackWorld) -> str:
    return hashlib.sha256(canonical_world_bytes(world)).hexdigest()


def static_source_errors(source: str) -> list[str]:
    errors: list[str] = []
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"SYNTAX: {exc.msg} at line {exc.lineno}"]
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [item.name for item in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        for name in names:
            if not any(name == root or name.startswith(root + ".") for root in SAFE_IMPORT_ROOTS):
                errors.append(f"IMPORT: {name!r} is not allowlisted")
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for imported in node.names:
                member = imported.name.rsplit(".", 1)[-1]
                if (
                    member.startswith("_")
                    or member in UNSAFE_IMPORT_MEMBERS
                    or (imported.asname and imported.asname.startswith("_"))
                ):
                    errors.append(f"IMPORT MEMBER: {imported.name!r} is not allowed")
                if isinstance(node, ast.Import) and any(
                    imported.name == module or imported.name.startswith(module + ".")
                    for module in PROTECTED_IMPORT_MEMBERS
                ):
                    errors.append(f"IMPORT: use explicit safe members from {imported.name!r}")
                if isinstance(node, ast.ImportFrom) and node.module:
                    protected_submodule = next(
                        (
                            module
                            for module in PROTECTED_IMPORT_MEMBERS
                            if node.module.startswith(module + ".")
                        ),
                        None,
                    )
                    if protected_submodule is not None:
                        errors.append(f"IMPORT: protected submodule {node.module!r} is not allowed")
                    elif node.module in PROTECTED_IMPORT_MEMBERS and (
                        imported.name not in PROTECTED_IMPORT_MEMBERS[node.module]
                    ):
                        errors.append(
                            f"IMPORT MEMBER: {imported.name!r} is not allowlisted from {node.module!r}"
                        )
        if isinstance(node, ast.Name) and (node.id in UNSAFE_NAMES or node.id.startswith("__")):
            errors.append(f"NAME: {node.id!r} is not allowed")
        if isinstance(node, ast.Attribute) and (
            node.attr.startswith(("_", "exec", "spawn")) or node.attr in UNSAFE_ATTRIBUTES
        ):
            errors.append(f"ATTRIBUTE: {node.attr!r} is not allowed")
    definitions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "build"
    ]
    if len(definitions) != 1 or not isinstance(definitions[0], ast.FunctionDef):
        errors.append("CONTRACT: define exactly one synchronous top-level build()")
    else:
        definition = definitions[0]
        arguments = definition.args
        exact_arguments = (
            not arguments.posonlyargs
            and [argument.arg for argument in arguments.args] == ["seed", "contract"]
            and arguments.vararg is None
            and not arguments.kwonlyargs
            and arguments.kwarg is None
            and not arguments.defaults
            and not arguments.kw_defaults
            and not definition.decorator_list
        )
        if not exact_arguments:
            errors.append("CONTRACT: build signature must be exactly build(seed, contract)")
    return sorted(set(errors))


def _message_ids(value: Any) -> set[str]:
    if isinstance(value, dict):
        found = {item for key, item in value.items() if key == "message_id" and isinstance(item, str)}
        for item in value.values():
            found.update(_message_ids(item))
        return found
    if isinstance(value, list):
        found: set[str] = set()
        for item in value:
            found.update(_message_ids(item))
        return found
    return set()


def _dispatch_gold(api: SlackAPI, tool: str, arguments: dict[str, Any]) -> Any:
    methods = {
        "slack_list_conversations": api.list_conversations,
        "slack_search_messages": api.search_messages,
        "slack_get_conversation_history": api.get_conversation_history,
        "slack_get_thread": api.get_thread,
        "slack_get_user": api.get_user,
    }
    method = methods[tool]
    try:
        inspect.signature(method).bind(**arguments)
    except TypeError as exc:
        raise ValueError(f"invalid arguments for {tool}: {exc}") from exc
    return method(**arguments)


def _check(name: str, problems: list[str]) -> CheckResult:
    return CheckResult(
        name=name,
        ok=not problems,
        detail="PASS" if not problems else "; ".join(problems),
        failure_owner=FailureOwner.NONE if not problems else FailureOwner.BUILDER,
    )


def validate_world(world: SlackWorld, contract: TaskContract) -> dict[str, Any]:
    checks: list[CheckResult] = []
    counts = {name: len(getattr(world, name)) for name in MAX_COUNTS}
    size_problems = [
        f"{name} count {counts[name]} exceeds {maximum}"
        for name, maximum in MAX_COUNTS.items()
        if counts[name] > maximum
    ]
    snapshot_size = len(canonical_world_bytes(world))
    if snapshot_size > MAX_SNAPSHOT_BYTES:
        size_problems.append(f"canonical snapshot exceeds {MAX_SNAPSHOT_BYTES} bytes")
    checks.append(_check("world_size_and_bounds", size_problems))

    actor_problems: list[str] = []
    api: SlackAPI | None = None
    try:
        api = SlackAPI(world, contract.actor_id)
    except ValueError as exc:
        actor_problems.append(str(exc))
    checks.append(_check("actor_visibility", actor_problems))

    messages = {message.id: message for message in world.messages}
    evidence_results: list[dict[str, Any]] = []
    evidence_problems: list[str] = []
    for requirement in contract.required_evidence:
        problems: list[str] = []
        message = messages.get(requirement.message_id)
        if message is None:
            problems.append("message missing")
        else:
            if message.conversation_id != requirement.conversation_id:
                problems.append("conversation mismatch")
            if requirement.author_id is not None and message.author_id != requirement.author_id:
                problems.append("author mismatch")
            if requirement.thread_root_id is not None and (
                message.thread_root_id != requirement.thread_root_id
            ):
                problems.append("thread root mismatch")
            missing = [
                term
                for term in requirement.required_terms
                if normalize_text(term) not in normalize_text(message.text)
            ]
            if missing:
                problems.append(f"missing terms {missing}")
            if message.deleted:
                problems.append("message is deleted")
            if api is None or not api.is_conversation_visible(message.conversation_id):
                problems.append("message is invisible to actor")
        evidence_results.append(
            {
                "evidence_id": requirement.evidence_id,
                "ok": not problems,
                "detail": "; ".join(problems) or "PASS",
            }
        )
        evidence_problems.extend(f"{requirement.evidence_id}: {problem}" for problem in problems)
    checks.append(_check("required_evidence", evidence_problems))

    gold_outputs: list[dict[str, Any]] = []
    gold_problems: list[str] = []
    if api is not None:
        for number, call in enumerate(contract.gold_calls, 1):
            try:
                output = _dispatch_gold(api, call.tool, call.arguments)
            except (SlackNotFoundError, TypeError, ValueError) as exc:
                gold_problems.append(f"call {number}: {type(exc).__name__}: {exc}")
                break
            gold_outputs.append(
                {
                    "index": number,
                    "tool": call.tool,
                    "arguments": call.arguments,
                    "output": output,
                }
            )
    exposed_ids = _message_ids([item["output"] for item in gold_outputs])
    missing_evidence = [
        requirement.evidence_id
        for requirement in contract.required_evidence
        if requirement.message_id not in exposed_ids
    ]
    if missing_evidence:
        gold_problems.append(f"gold path did not expose evidence {missing_evidence}")
    if len(json.dumps(gold_outputs, default=str).encode()) > MAX_SNAPSHOT_BYTES:
        gold_problems.append("gold call log exceeds the output bound")
    checks.append(_check("gold_replay", gold_problems))

    public_strings = [
        *(user.name for user in world.users),
        *(user.display_name or "" for user in world.users),
        *(conversation.name or "" for conversation in world.conversations),
        *(conversation.topic for conversation in world.conversations),
        *(conversation.purpose for conversation in world.conversations),
        *(message.text for message in world.messages),
    ]
    public_blob = "\n".join(public_strings)
    normalized_blob = normalize_text(public_blob)
    leakage: list[str] = []
    if task_slug := contract.task_slug:
        if normalize_text(task_slug) in normalized_blob:
            leakage.append("model-proposed task slug appears in public content")
    if len(normalize_text(contract.question)) >= 12 and (
        normalize_text(contract.question) in normalized_blob
    ):
        leakage.append("literal question was copied into public content")
    for conversation in world.conversations:
        name = normalize_text(conversation.name or "")
        if any(term == name or name.startswith(term + " ") for term in ("answer", "gold", "oracle")):
            leakage.append(f"degenerate channel name {conversation.name!r}")
    for phrase in PROMPT_INJECTION_TERMS:
        if phrase in public_blob.casefold():
            leakage.append(f"prompt-injection phrase {phrase!r}")
    if any(f'"{field}"' in public_blob.casefold() for field in PRIVATE_FIELD_TERMS):
        leakage.append("serialized private contract field appears in public content")
    checks.append(_check("private_data_and_answer_cache", leakage))

    return {
        "ok": all(check.ok for check in checks),
        "checks": [check.model_dump(mode="json") for check in checks],
        "counts": counts,
        "snapshot_bytes": snapshot_size,
        "evidence": evidence_results,
        "gold_call_log": gold_outputs,
    }


def _load_build(candidate: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, candidate)
    if spec is None or spec.loader is None:
        raise ImportError("could not create candidate module spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
    build = getattr(module, "build", None)
    if not callable(build):
        raise TypeError("candidate does not expose callable build")
    return build


def check_candidate(candidate: Path, contract: TaskContract, seeds: Iterable[int]) -> dict[str, Any]:
    source = candidate.read_text(encoding="utf-8")
    source_errors = static_source_errors(source)
    if source_errors:
        return {"ok": False, "static_errors": source_errors, "snapshots": {}, "reports": []}
    seed_list = list(dict.fromkeys(seeds))
    snapshots: dict[str, dict[str, Any]] = {}
    reports: list[dict[str, Any]] = []
    for index, seed in enumerate(seed_list):
        try:
            build = _load_build(candidate, f"candidate_world_{index}")
            world = SlackWorld.model_validate(build(seed, contract.model_copy(deep=True)))
            snapshots[str(seed)] = world.model_dump(mode="json")
            reports.append({"seed": seed, **validate_world(world, contract)})
        except BaseException as exc:  # Generated candidate failure boundary.
            reports.append(
                {
                    "seed": seed,
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    if seed_list:
        try:
            build = _load_build(candidate, "candidate_world_repeat")
            repeat = SlackWorld.model_validate(build(seed_list[0], contract.model_copy(deep=True)))
            snapshots[f"repeat_{seed_list[0]}"] = repeat.model_dump(mode="json")
        except BaseException as exc:  # Generated candidate failure boundary.
            reports.append(
                {
                    "seed": f"repeat_{seed_list[0]}",
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return {
        "ok": len(reports) == len(seed_list) and all(report.get("ok") for report in reports),
        "static_errors": [],
        "reports": reports,
        "snapshots": snapshots,
    }


def validate_compiled_snapshots(
    snapshots: dict[str, Any],
    contract: TaskContract,
    seeds: Iterable[int],
    *,
    runtime: dict[str, Any] | None = None,
) -> ValidationReport:
    seed_list = list(seeds)
    worlds: dict[int, SlackWorld] = {}
    checks: list[CheckResult] = []
    public_gold_log: list[dict[str, Any]] = []
    for seed in seed_list:
        try:
            world = SlackWorld.model_validate(snapshots[str(seed)])
        except Exception as exc:
            checks.append(
                CheckResult(
                    name=f"seed_{seed}:snapshot_schema",
                    ok=False,
                    detail=f"{type(exc).__name__}: {exc}",
                    failure_owner=FailureOwner.BUILDER,
                )
            )
            continue
        worlds[seed] = world
        report = validate_world(world, contract)
        if seed == seed_list[0]:
            public_gold_log = report["gold_call_log"]
        checks.extend(
            CheckResult(
                name=f"seed_{seed}:{item['name']}",
                ok=item["ok"],
                detail=item["detail"],
                failure_owner=FailureOwner(item["failure_owner"]),
            )
            for item in report["checks"]
        )

    public_seed = seed_list[0] if seed_list else None
    deterministic = False
    if public_seed is not None and public_seed in worlds:
        try:
            repeat = SlackWorld.model_validate(snapshots[f"repeat_{public_seed}"])
            deterministic = canonical_world_hash(repeat) == canonical_world_hash(worlds[public_seed])
        except Exception:
            deterministic = False
    checks.append(
        _check(
            "same_seed_determinism",
            [] if deterministic else ["repeated public seed changed its canonical snapshot"],
        )
    )

    hashes = {seed: canonical_world_hash(world) for seed, world in worlds.items()}
    variation = len(seed_list) < 2 or len(set(hashes.values())) > 1
    checks.append(
        _check(
            "hidden_seed_variation",
            [] if variation else ["hidden seeds did not vary nonessential world content"],
        )
    )

    evidence_signatures: list[list[tuple[Any, ...]]] = []
    for seed in seed_list:
        world = worlds.get(seed)
        if world is None:
            continue
        by_id = {message.id: message for message in world.messages}
        evidence_signatures.append(
            [
                (
                    requirement.message_id,
                    requirement.conversation_id,
                    requirement.author_id,
                    requirement.thread_root_id,
                    normalize_text(by_id[requirement.message_id].text)
                    if requirement.message_id in by_id
                    else None,
                )
                for requirement in contract.required_evidence
            ]
        )
    evidence_stable = bool(evidence_signatures) and all(
        signature == evidence_signatures[0] for signature in evidence_signatures
    )
    checks.append(
        _check(
            "hidden_seed_answer_stability",
            [] if evidence_stable else ["answer-bearing evidence changed across seeds"],
        )
    )

    ok = bool(seed_list) and len(worlds) == len(seed_list) and all(check.ok for check in checks)
    return ValidationReport(
        ok=ok,
        failure_owner=FailureOwner.NONE if ok else FailureOwner.BUILDER,
        checks=checks,
        public_snapshot=worlds.get(public_seed) if ok and public_seed is not None else None,
        hidden_snapshot_hashes=[f"sha256:{hashes[seed]}" for seed in seed_list[1:] if seed in hashes],
        gold_call_log=public_gold_log,
        runtime=runtime or {},
    )


CHECKER_RUNNER_SOURCE = """# /// script
# requires-python = ">=3.11"
# dependencies = ["pydantic==2.13.4", "verifiers==0.3.1"]
# ///
import sys
sys.path.insert(0, "/task/fixed")
from worldgen_slack.slack.validate import main
raise SystemExit(main())
"""


def _failed_report(
    owner: FailureOwner,
    name: str,
    detail: str,
    runtime: dict[str, Any] | None = None,
) -> ValidationReport:
    return ValidationReport(
        ok=False,
        failure_owner=owner,
        checks=[CheckResult(name=name, ok=False, detail=detail, failure_owner=owner)],
        runtime=runtime or {},
    )


async def evaluate_candidate_in_runtime(
    source: bytes,
    contract: TaskContract,
    runtime_config: vf.PrimeConfig,
    *,
    seeds: tuple[int, ...] = (0, 101, 202),
    timeout_seconds: float = 90.0,
    on_seed_complete: Callable[[int, bool], None] | None = None,
) -> ValidationReport:
    from verifiers.v1.runtimes import provision_runtime

    if not isinstance(runtime_config, vf.PrimeConfig):
        raise TypeError("candidate runtime must be PrimeConfig")
    if not runtime_config.vm:
        raise ValueError("candidate runtime must use vm=true")
    if runtime_config.allow or runtime_config.block != ["*"]:
        raise ValueError("candidate runtime must use framework-only egress")
    if len(source) > MAX_SOURCE_BYTES:
        return _failed_report(FailureOwner.BUILDER, "source_size", "candidate source is too large")
    try:
        source_text = source.decode("utf-8")
    except UnicodeDecodeError as exc:
        return _failed_report(FailureOwner.BUILDER, "source_encoding", str(exc))
    source_errors = static_source_errors(source_text)
    if source_errors:
        return _failed_report(
            FailureOwner.BUILDER,
            "static_source",
            "; ".join(source_errors),
        )

    package_dir = Path(__file__).resolve().parents[1]
    async with provision_runtime(runtime_config) as runtime:
        runtime_info = runtime.info.model_dump(mode="json")
        await runtime.prepare_setup()
        fixed_files = {
            "/task/fixed/worldgen_slack/__init__.py": package_dir.joinpath("__init__.py").read_bytes(),
            "/task/fixed/worldgen_slack/contracts.py": package_dir.joinpath("contracts.py").read_bytes(),
            "/task/fixed/worldgen_slack/slack/__init__.py": package_dir.joinpath(
                "slack", "__init__.py"
            ).read_bytes(),
            "/task/fixed/worldgen_slack/slack/models.py": package_dir.joinpath(
                "slack", "models.py"
            ).read_bytes(),
            "/task/fixed/worldgen_slack/slack/api.py": package_dir.joinpath("slack", "api.py").read_bytes(),
            "/task/fixed/worldgen_slack/slack/validate.py": Path(__file__).read_bytes(),
            "/task/candidate/world.py": source,
            "/task/contract.json": contract.model_dump_json(indent=2).encode(),
        }
        for path, content in fixed_files.items():
            await runtime.write(path, content)
        program = await runtime.prepare_uv_script(CHECKER_RUNNER_SOURCE)
        await runtime.prepare_execution([])
        probe = await runtime.run(
            [
                "python",
                "-c",
                "import urllib.request; urllib.request.urlopen('https://example.com', timeout=2)",
            ],
            {},
        )
        if probe.exit_code == 0:
            raise RuntimeError("candidate runtime allowed external network")
        snapshots: dict[str, Any] = {}
        current_seed: int | None = None
        try:
            async with asyncio.timeout(timeout_seconds):
                for current_seed in seeds:
                    output_path = f"/task/result-{current_seed}.json"
                    argv = [
                        *program,
                        "check-candidate",
                        "--candidate",
                        "/task/candidate/world.py",
                        "--contract",
                        "/task/contract.json",
                        "--output",
                        output_path,
                        "--seeds",
                        str(current_seed),
                    ]
                    wrapper = (
                        "#!/bin/sh\n"
                        "ulimit -f 8192\n"
                        + "PYTHONPATH=/task/fixed "
                        + shlex.join(argv)
                        + " > /task/candidate.stdout 2> /task/candidate.stderr\n"
                        "status=$?\n"
                        "tail -c 2000 /task/candidate.stdout 2>/dev/null || true\n"
                        "tail -c 2000 /task/candidate.stderr >&2 2>/dev/null || true\n"
                        "exit $status\n"
                    )
                    await runtime.write("/task/run-candidate", wrapper.encode())
                    process = await runtime.run(["/bin/sh", "/task/run-candidate"], {})
                    if process.exit_code not in (0, 2):
                        detail = (process.stderr or process.stdout)[
                            -2_000:
                        ] or "fixed candidate runner failed"
                        if not await runtime.alive():
                            raise vf.SandboxError("candidate runtime stopped during execution")
                        raise RuntimeError(detail)
                    raw = await runtime.read(output_path, max_bytes=MAX_RESULT_BYTES)
                    try:
                        result = json.loads(raw)
                    except json.JSONDecodeError as exc:
                        raise RuntimeError("fixed candidate runner returned invalid JSON") from exc
                    seed_snapshots = result.get("snapshots")
                    if not isinstance(seed_snapshots, dict):
                        raise RuntimeError("fixed candidate runner produced no snapshot map")
                    snapshots.update(seed_snapshots)
                    if on_seed_complete is not None:
                        on_seed_complete(current_seed, process.exit_code == 0)
        except TimeoutError:
            if current_seed is not None and on_seed_complete is not None:
                on_seed_complete(current_seed, False)
            return _failed_report(
                FailureOwner.BUILDER,
                "candidate_timeout",
                "candidate validation timed out",
                runtime_info,
            )
        return validate_compiled_snapshots(
            snapshots,
            contract,
            seeds,
            runtime=runtime_info,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    check = subparsers.add_parser("check-candidate")
    check.add_argument("--candidate", type=Path, required=True)
    check.add_argument("--contract", type=Path, required=True)
    check.add_argument("--output", type=Path, required=True)
    check.add_argument("--seeds", nargs="+", type=int, required=True)
    args = parser.parse_args(argv)
    contract = TaskContract.model_validate_json(args.contract.read_text(encoding="utf-8"))
    result = check_candidate(args.candidate, contract, args.seeds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True),
        encoding="utf-8",
    )
    return 0 if result["ok"] else 2


__all__ = [
    "CHECKER_RUNNER_SOURCE",
    "MAX_SOURCE_BYTES",
    "canonical_world_hash",
    "evaluate_candidate_in_runtime",
    "normalize_text",
    "static_source_errors",
    "validate_compiled_snapshots",
    "validate_world",
]


if __name__ == "__main__":
    raise SystemExit(main())
