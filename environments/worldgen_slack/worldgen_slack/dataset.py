from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import threading
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from enum import Enum
from importlib.metadata import version
from pathlib import Path
from typing import Any

import verifiers.v1 as vf

from .contracts import (
    INTERFACE_ID,
    FailureOwner,
    GenerationResult,
    ItemStatus,
    JudgeVerdict,
    QualityFilterConfig,
    ScenarioSpec,
    SynthesizedItem,
    ValidationReport,
    synthesized_signature,
    synthesized_signature_text,
    token_jaccard,
    redact_secrets,
)
from .slack.models import SlackWorld, TaskContract
from .slack.validation import canonical_world_hash

GENERATOR_SCHEMA_VERSION = "worldgen-slack.dataset.v1"


def _generator_source_hash() -> str:
    package = Path(__file__).resolve().parent
    base = package.parent
    files = [path for path in package.rglob("*") if path.is_file() and path.suffix in {".py", ".json", ".md"}]
    files.extend(
        path
        for path in (base / "worldgen_slack_generation.py", base / "worldgen_slack_generated.py")
        if path.is_file()
    )
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda item: item.relative_to(base).as_posix()):
        relative = path.relative_to(base).as_posix().encode()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        content = path.read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return "sha256:" + digest.hexdigest()


_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,159}$")
_SAFE_ATTEMPT_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,239}$")
_SECRET_KEY = re.compile(r"(?i)(?:api[_-]?key|authorization|password|secret|token)$")
_SECRET_VALUE = re.compile(
    r"(?i)(bearer\s+|(?:api[_-]?key|password|secret|token)\s*[=:]\s*)"
    r"[A-Za-z0-9_./+\-=]{8,}"
)
JSONL_FILES = (
    "dataset.jsonl",
    "public_tasks.jsonl",
    "private_oracles.jsonl",
    "attempts.jsonl",
)
CATALOG_FIELDS = {
    "instance_id",
    "domain",
    "scenario",
    "question",
    "answer",
    "required_claims",
    "status",
    "quality_score",
    "judge_scores",
    "world_ref",
    "snapshot_ref",
    "oracle_ref",
    "solver_trace_ref",
    "judge_ref",
    "interface_id",
    "interface_ref",
    "world_hash",
    "contract_hash",
}
PUBLIC_FIELDS = {"instance_id", "scenario", "question", "snapshot_ref", "interface_id"}
PRIVATE_FIELDS = {
    "instance_id",
    "answer",
    "required_claims",
    "forbidden_claims",
    "task_contract_ref",
    "gold_ref",
}


def _jsonable(value: Any, *, redact: bool = False) -> Any:
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError("non-finite floats are not valid dataset JSON")
        return value
    if isinstance(value, str):
        return _SECRET_VALUE.sub(lambda match: match.group(1) + "[REDACTED]", value) if redact else value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"dataset JSON key is not a string: {key!r}")
            output[key] = (
                "[REDACTED]" if redact and _SECRET_KEY.search(key) else _jsonable(item, redact=redact)
            )
        return output
    if isinstance(value, set):
        items = [_jsonable(item, redact=redact) for item in value]
        return sorted(
            items,
            key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")),
        )
    if isinstance(value, (list, tuple)):
        return [_jsonable(item, redact=redact) for item in value]
    raise TypeError(f"unsupported dataset JSON value: {type(value).__name__}")


def _dumps(value: Any, *, pretty: bool = False, redact: bool = False) -> str:
    serializable = _jsonable(value, redact=redact)
    if redact:
        serializable = redact_secrets(serializable)
    return (
        json.dumps(
            serializable,
            ensure_ascii=False,
            allow_nan=False,
            indent=2 if pretty else None,
            separators=None if pretty else (",", ":"),
            sort_keys=True,
        )
        + "\n"
    )


def _atomic_write(path: Path, content: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    binary = isinstance(content, bytes)
    mode = "wb" if binary else "w"
    kwargs = {} if binary else {"encoding": "utf-8"}
    handle = tempfile.NamedTemporaryFile(
        mode,
        prefix=f".{path.name}.",
        dir=path.parent,
        delete=False,
        **kwargs,
    )
    try:
        with handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(handle.name, path)
        _fsync_directory(path.parent)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    line = _dumps(row)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def _read_json_value(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid JSON at {path}") from exc


def _read_json(path: Path) -> dict[str, Any]:
    value = _read_json_value(path)
    if not isinstance(value, dict):
        raise TypeError(f"JSON at {path} must be an object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw, parse_constant=_reject_constant)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"invalid JSONL at {path}:{number}") from exc
        if not isinstance(row, dict):
            raise TypeError(f"JSONL row at {path}:{number} must be an object")
        rows.append(row)
    return rows


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value!r} is not supported")


def _sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    return slug[:80].rstrip("-") or "slack-item"


def _reject_symlink(path: Path, label: str) -> None:
    if path.is_symlink():
        raise ValueError(f"managed {label} must not be a symlink: {path}")


def _inside(root: Path, reference: str) -> Path:
    if not isinstance(reference, str) or not reference.strip():
        raise ValueError("dataset reference must be a nonempty relative path")
    relative = Path(reference)
    if (
        relative.is_absolute()
        or any(part in {".", ".."} for part in relative.parts)
        or relative.as_posix() != reference
    ):
        raise ValueError(f"dataset reference must be canonical and relative: {reference!r}")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"dataset reference escapes output root: {reference!r}") from exc
    return resolved


def _row_index(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for number, row in enumerate(rows, 1):
        instance_id = row.get("instance_id")
        if not isinstance(instance_id, str) or not _SAFE_COMPONENT.fullmatch(instance_id):
            raise ValueError(f"{label} row {number} has an unsafe instance_id")
        if instance_id in index:
            raise ValueError(f"{label} dataset has duplicate instance IDs")
        index[instance_id] = row
    return index


def validate_release_integrity(
    root: str | Path,
    *,
    expected_run_id: str | None = None,
    require_tasks: bool = False,
) -> None:
    release = Path(root).expanduser().resolve()
    for name in (*JSONL_FILES, "setup_failures.jsonl", "run_manifest.json"):
        path = release / name
        if path.exists() or path.is_symlink():
            _reject_symlink(path, name)
    _reject_symlink(release / "worlds", "worlds directory")
    _reject_symlink(release / "interfaces", "interfaces directory")
    _reject_symlink(release / ".transactions", "transaction directory")
    catalog_rows = _read_jsonl(release / "dataset.jsonl")
    public_rows = _read_jsonl(release / "public_tasks.jsonl")
    private_rows = _read_jsonl(release / "private_oracles.jsonl")
    attempt_rows = _read_jsonl(release / "attempts.jsonl")
    run_manifest_path = release / "run_manifest.json"
    if not run_manifest_path.is_file():
        raise ValueError("release has no immutable run_manifest.json")
    run_manifest = _read_json(run_manifest_path)
    manifest_run_id = run_manifest.get("run_id")
    if not isinstance(manifest_run_id, str) or not _SAFE_COMPONENT.fullmatch(manifest_run_id):
        raise ValueError("run manifest has an unsafe run_id")
    if expected_run_id is not None and manifest_run_id != expected_run_id:
        raise ValueError("run manifest belongs to another run")
    if run_manifest.get("schema_version") != 1:
        raise ValueError("run manifest schema mismatch")
    if run_manifest.get("interface_id") != INTERFACE_ID:
        raise ValueError("run manifest interface mismatch")
    if run_manifest.get("generator_schema_version") != GENERATOR_SCHEMA_VERSION:
        raise ValueError("unsupported generator dataset schema")
    if not isinstance(run_manifest.get("generator_source_hash"), str) or not run_manifest[
        "generator_source_hash"
    ].startswith("sha256:"):
        raise ValueError("run manifest has no generator source hash")
    catalog = _row_index(catalog_rows, "catalog")
    public = _row_index(public_rows, "public")
    private = _row_index(private_rows, "private")
    if set(catalog) != set(public) or set(catalog) != set(private):
        raise ValueError("catalog/public/private instance IDs are not one-to-one")
    if require_tasks and not catalog:
        raise ValueError(f"release contains no accepted tasks: {release}")

    attempt_ids: set[str] = set()
    terminal_seeds: set[int] = set()
    accepted_attempts: dict[str, int] = {}
    retained_rejected: set[str] = set()
    for number, row in enumerate(attempt_rows, 1):
        attempt_id = row.get("attempt_id")
        if not isinstance(attempt_id, str) or not _SAFE_ATTEMPT_COMPONENT.fullmatch(attempt_id):
            raise ValueError(f"attempt row {number} has an unsafe attempt_id")
        if attempt_id in attempt_ids:
            raise ValueError("attempts.jsonl has duplicate attempt IDs")
        attempt_ids.add(attempt_id)
        if row.get("run_id") != manifest_run_id:
            raise ValueError(f"attempt {attempt_id!r} belongs to another run")
        instance_id = row.get("instance_id")
        if instance_id is not None and (
            not isinstance(instance_id, str) or not _SAFE_COMPONENT.fullmatch(instance_id)
        ):
            raise ValueError(f"attempt {attempt_id!r} has an unsafe instance_id")
        written = row.get("written_to_dataset")
        if not isinstance(written, bool):
            raise ValueError(f"attempt {attempt_id!r} has no boolean persistence decision")
        retry_discarded = row.get("retry_discarded")
        if not isinstance(retry_discarded, bool):
            raise ValueError(f"attempt {attempt_id!r} has no retry_discarded flag")
        generation_seed = row.get("generation_seed")
        if isinstance(generation_seed, bool) or not isinstance(generation_seed, int) or generation_seed < 0:
            raise ValueError(f"attempt {attempt_id!r} has an invalid generation seed")
        if retry_discarded:
            if (
                written
                or row.get("failure_owner") != "infrastructure"
                or row.get("item_status") != ItemStatus.INFRASTRUCTURE_ERROR.value
            ):
                raise ValueError(f"retry-discarded attempt {attempt_id!r} is not infrastructure-owned")
        elif generation_seed in terminal_seeds:
            raise ValueError(f"generation seed {generation_seed} has multiple terminal attempts")
        else:
            terminal_seeds.add(generation_seed)
        artifact_ref = row.get("artifact_ref")
        if written:
            if not isinstance(instance_id, str) or instance_id not in catalog:
                raise ValueError(f"accepted attempt {attempt_id!r} has no accepted dataset row")
            accepted_attempts[instance_id] = accepted_attempts.get(instance_id, 0) + 1
            expected_artifact = f"worlds/{instance_id}"
            if artifact_ref != expected_artifact:
                raise ValueError(f"accepted attempt {attempt_id!r} has a cross-world artifact")
        elif artifact_ref is not None:
            expected_artifact = f"rejected/{attempt_id}"
            if artifact_ref != expected_artifact:
                raise ValueError(f"rejected attempt {attempt_id!r} has an invalid artifact_ref")
            retained_rejected.add(attempt_id)
        if artifact_ref is not None and not _inside(release, artifact_ref).is_dir():
            raise ValueError(f"attempt {attempt_id!r} has a dangling artifact_ref")
        trace_refs = row.get("trace_refs")
        if not isinstance(trace_refs, dict):
            raise ValueError(f"attempt {attempt_id!r} has invalid trace_refs")
        for references in trace_refs.values():
            if not isinstance(references, list) or not all(
                isinstance(reference, str) for reference in references
            ):
                raise ValueError(f"attempt {attempt_id!r} has invalid trace references")
            for reference in references:
                target = _inside(release, reference)
                if not target.is_file():
                    raise ValueError(f"attempt {attempt_id!r} has a dangling trace reference")
                if written and not reference.startswith(f"worlds/{instance_id}/"):
                    raise ValueError(f"attempt {attempt_id!r} has a cross-world trace reference")
                if (
                    not written
                    and artifact_ref is not None
                    and not reference.startswith(f"rejected/{attempt_id}/")
                ):
                    raise ValueError(f"attempt {attempt_id!r} has a cross-attempt trace reference")
    if set(accepted_attempts) != set(catalog) or any(count != 1 for count in accepted_attempts.values()):
        raise ValueError("accepted dataset rows must have exactly one accepted attempt")

    interface_root = release / "interfaces" / INTERFACE_ID
    _reject_symlink(interface_root, "interface version directory")
    interface_hashes = {}
    for name in ("interface.json", "models.schema.json", "behavior.md"):
        path = interface_root / name
        _reject_symlink(path, f"interface file {name}")
        if not path.is_file():
            raise ValueError(f"release interface file is missing: {path}")
        interface_hashes[name] = _sha256(path.read_bytes())
    if run_manifest.get("interface_hashes") != interface_hashes:
        raise ValueError("run manifest interface hashes mismatch")

    for instance_id, catalog_row in catalog.items():
        public_row = public[instance_id]
        private_row = private[instance_id]
        if set(catalog_row) != CATALOG_FIELDS:
            raise ValueError(f"catalog row {instance_id!r} has unexpected fields")
        if set(public_row) != PUBLIC_FIELDS:
            raise ValueError(f"public row {instance_id!r} has missing or private fields")
        if set(private_row) != PRIVATE_FIELDS:
            raise ValueError(f"private row {instance_id!r} has unexpected fields")
        base = f"worlds/{instance_id}"
        expected_catalog_refs = {
            "world_ref": f"{base}/world.py",
            "snapshot_ref": f"{base}/snapshot.json",
            "oracle_ref": f"{base}/task_contract.json",
            "solver_trace_ref": f"{base}/solver_trace.json",
            "judge_ref": f"{base}/judge_verdict.json",
            "interface_ref": f"interfaces/{INTERFACE_ID}/interface.json",
        }
        for key, reference in expected_catalog_refs.items():
            if catalog_row.get(key) != reference:
                raise ValueError(f"cross-world {key} for {instance_id}")
            if not _inside(release, reference).is_file():
                raise ValueError(f"dangling {key} for {instance_id}")
        if public_row.get("snapshot_ref") != expected_catalog_refs["snapshot_ref"]:
            raise ValueError(f"public row {instance_id!r} references another world")
        if private_row.get("task_contract_ref") != expected_catalog_refs["oracle_ref"]:
            raise ValueError(f"private row {instance_id!r} references another contract")
        if private_row.get("gold_ref") != f"{base}/gold.json":
            raise ValueError(f"private row {instance_id!r} references another gold file")
        gold_path = _inside(release, private_row["gold_ref"])
        if not gold_path.is_file():
            raise ValueError(f"dangling gold file for {instance_id}")
        world_dir = release / "worlds" / instance_id
        _reject_symlink(world_dir, f"world directory {instance_id}")
        if any(path.is_symlink() for path in world_dir.iterdir()):
            raise ValueError(f"accepted world {instance_id!r} contains a symlink")
        required_files = {
            "scenario.json",
            "qa.json",
            "task_contract.json",
            "world.py",
            "snapshot.json",
            "hidden_snapshot_hashes.json",
            "gold.json",
            "validation.json",
            "synthesizer_trace.json",
            "builder_trace.json",
            "solver_trace.json",
            "judge_trace.json",
            "judge_verdict.json",
            "manifest.json",
            "public_checks.json",
        }
        actual_files = {path.name for path in world_dir.iterdir() if path.is_file()}
        if actual_files != required_files:
            raise ValueError(
                f"accepted world {instance_id!r} file set differs: {sorted(actual_files ^ required_files)}"
            )
        manifest_path = world_dir / "manifest.json"
        manifest = _read_json(manifest_path)
        if manifest.get("schema_version") != 1:
            raise ValueError(f"manifest schema mismatch for {instance_id}")
        artifact_hashes = manifest.get("artifact_hashes")
        expected_artifacts = required_files - {"manifest.json"}
        if not isinstance(artifact_hashes, dict) or set(artifact_hashes) != expected_artifacts:
            raise ValueError(f"manifest artifact hash set mismatch for {instance_id}")
        for name in expected_artifacts:
            if artifact_hashes[name] != _sha256((world_dir / name).read_bytes()):
                raise ValueError(f"artifact hash mismatch for {instance_id}: {name}")
        if manifest.get("instance_id") != instance_id:
            raise ValueError(f"manifest identity mismatch for {instance_id}")
        matching_attempts = [
            row
            for row in attempt_rows
            if row.get("attempt_id") == manifest.get("attempt_id")
            and row.get("instance_id") == instance_id
            and row.get("written_to_dataset") is True
        ]
        if len(matching_attempts) != 1:
            raise ValueError(f"manifest attempt mismatch for {instance_id}")
        matching_attempt = matching_attempts[0]
        if manifest.get("runtime_records") != matching_attempt.get("runtime_records"):
            raise ValueError(f"manifest runtime records mismatch for {instance_id}")
        if manifest.get("builder") != matching_attempt.get("builder"):
            raise ValueError(f"manifest builder metadata mismatch for {instance_id}")
        if manifest.get("generation_seed") != matching_attempt.get("generation_seed"):
            raise ValueError(f"manifest generation seed mismatch for {instance_id}")
        if manifest.get("run_id") != manifest_run_id:
            raise ValueError(f"manifest run mismatch for {instance_id}")
        if manifest.get("interface_id") != INTERFACE_ID:
            raise ValueError(f"manifest interface mismatch for {instance_id}")
        for key in (
            "generator_schema_version",
            "generator_source_hash",
            "worldgen_version",
            "verifiers_version",
            "pydantic_version",
            "prime_image",
            "rlm_revision",
        ):
            if manifest.get(key) != run_manifest.get(key):
                raise ValueError(f"manifest {key} mismatch for {instance_id}")
        if manifest.get("interface_hashes") != interface_hashes:
            raise ValueError(f"manifest interface hashes mismatch for {instance_id}")
        world_path = _inside(release, expected_catalog_refs["world_ref"])
        snapshot_path = _inside(release, expected_catalog_refs["snapshot_ref"])
        contract_path = _inside(release, expected_catalog_refs["oracle_ref"])
        world = SlackWorld.model_validate(_read_json(snapshot_path))
        contract = TaskContract.model_validate(_read_json(contract_path))
        scenario = ScenarioSpec.model_validate(_read_json(world_dir / "scenario.json"))
        synthesized = SynthesizedItem(scenario=scenario, task=contract)
        if manifest.get("synthesized_item_hash") != _sha256(_canonical_bytes(synthesized)):
            raise ValueError(f"synthesized item hash mismatch for {instance_id}")
        qa = _read_json(world_dir / "qa.json")
        hidden_hashes = _read_json_value(world_dir / "hidden_snapshot_hashes.json")
        if hidden_hashes != manifest.get("hidden_snapshot_hashes"):
            raise ValueError(f"hidden snapshot hashes mismatch for {instance_id}")
        if not isinstance(hidden_hashes, list) or not all(
            isinstance(value, str) and value.startswith("sha256:") for value in hidden_hashes
        ):
            raise ValueError(f"invalid hidden snapshot hashes for {instance_id}")
        gold = _read_json(world_dir / "gold.json")
        expected_calls = [call.model_dump(mode="json") for call in contract.gold_calls]
        if gold.get("calls") != expected_calls or not isinstance(gold.get("call_log"), list):
            raise ValueError(f"gold artifact mismatch for {instance_id}")
        validation_data = _read_json(world_dir / "validation.json")
        if matching_attempt.get("validation") != validation_data:
            raise ValueError(f"attempt validation mismatch for {instance_id}")
        validation = ValidationReport.model_validate_json(
            json.dumps({**validation_data, "public_snapshot": world.model_dump(mode="json")})
        )
        if not validation.ok or validation.gold_call_log != gold["call_log"]:
            raise ValueError(f"validation/gold mismatch for {instance_id}")
        verdict_data = _read_json(world_dir / "judge_verdict.json")
        verdict = JudgeVerdict.model_validate_json(json.dumps(verdict_data))
        if matching_attempt.get("verdict") != verdict_data:
            raise ValueError(f"attempt verdict mismatch for {instance_id}")
        if matching_attempt.get("judge_scores") != verdict.scores:
            raise ValueError(f"attempt judge scores mismatch for {instance_id}")
        if catalog_row.get("judge_scores") != verdict.scores:
            raise ValueError(f"catalog judge scores mismatch for {instance_id}")
        if catalog_row.get("status") != matching_attempt.get("semantic_status"):
            raise ValueError(f"semantic status mismatch for {instance_id}")
        if catalog_row.get("quality_score") != matching_attempt.get("quality_score"):
            raise ValueError(f"quality score mismatch for {instance_id}")
        public_checks = _read_json_value(world_dir / "public_checks.json")
        if not isinstance(public_checks, list):
            raise ValueError(f"invalid public checks for {instance_id}")
        trace_ids = matching_attempt.get("trace_ids")
        if not isinstance(trace_ids, dict):
            raise ValueError(f"accepted attempt has invalid trace IDs for {instance_id}")
        for role in ("synthesizer", "builder", "solver", "judge"):
            trace_record = _read_json(world_dir / f"{role}_trace.json")
            ids = trace_ids.get(role)
            if (
                not isinstance(trace_record.get("id"), str)
                or not isinstance(ids, list)
                or trace_record["id"] not in ids
            ):
                raise ValueError(f"{role} trace identity mismatch for {instance_id}")
        expected_hashes = {
            "world_hash": _sha256(world_path.read_bytes()),
            "snapshot_hash": _sha256(snapshot_path.read_bytes()),
            "canonical_snapshot_hash": "sha256:" + canonical_world_hash(world),
            "contract_hash": _sha256(_canonical_bytes(contract)),
        }
        for key, expected in expected_hashes.items():
            if manifest.get(key) != expected:
                raise ValueError(f"{key.replace('_', ' ')} mismatch for {instance_id}")
        if catalog_row.get("world_hash") != expected_hashes["world_hash"]:
            raise ValueError(f"catalog world hash mismatch for {instance_id}")
        if catalog_row.get("contract_hash") != expected_hashes["contract_hash"]:
            raise ValueError(f"catalog contract hash mismatch for {instance_id}")
        if catalog_row.get("interface_id") != INTERFACE_ID or public_row.get("interface_id") != INTERFACE_ID:
            raise ValueError(f"dataset interface mismatch for {instance_id}")
        if (
            catalog_row.get("scenario") != scenario.description
            or public_row.get("scenario") != scenario.description
        ):
            raise ValueError(f"scenario mismatch for {instance_id}")
        expected_qa = {
            "question": contract.question,
            "answer": contract.answer.canonical_answer,
            "required_claims": contract.answer.required_claims,
            "forbidden_claims": contract.answer.forbidden_claims,
        }
        if qa != expected_qa:
            raise ValueError(f"qa.json mismatch for {instance_id}")
        if (
            public_row.get("question") != contract.question
            or catalog_row.get("question") != contract.question
        ):
            raise ValueError(f"question mismatch for {instance_id}")
        if (
            private_row.get("answer") != contract.answer.canonical_answer
            or catalog_row.get("answer") != contract.answer.canonical_answer
        ):
            raise ValueError(f"answer mismatch for {instance_id}")
        if (
            private_row.get("required_claims") != contract.answer.required_claims
            or catalog_row.get("required_claims") != contract.answer.required_claims
        ):
            raise ValueError(f"required claims mismatch for {instance_id}")
        if private_row.get("forbidden_claims") != contract.answer.forbidden_claims:
            raise ValueError(f"forbidden claims mismatch for {instance_id}")

    world_ids = {
        path.name
        for path in (release / "worlds").iterdir()
        if path.is_dir() and not path.name.startswith(".")
    }
    if world_ids != set(catalog):
        raise ValueError("world directories and accepted dataset rows differ")
    rejected_root = release / "rejected"
    _reject_symlink(rejected_root, "rejected directory")
    if rejected_root.is_dir() and any(path.is_symlink() for path in rejected_root.iterdir()):
        raise ValueError("rejected directory contains a symlink")
    rejected_ids = (
        {path.name for path in rejected_root.iterdir() if path.is_dir() and not path.name.startswith(".")}
        if rejected_root.is_dir()
        else set()
    )
    if rejected_ids != retained_rejected:
        raise ValueError("rejected directories and retained attempt rows differ")
    for attempt_id in rejected_ids:
        directory = rejected_root / attempt_id
        if any(path.is_symlink() for path in directory.rglob("*")):
            raise ValueError(f"rejected artifact {attempt_id!r} contains a symlink")
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError(f"rejected artifact {attempt_id!r} has no manifest")
        manifest = _read_json(manifest_path)
        if manifest.get("attempt_id") != attempt_id:
            raise ValueError(f"rejected artifact {attempt_id!r} has a manifest mismatch")
        files = {
            path.relative_to(directory).as_posix(): path
            for path in directory.rglob("*")
            if path.is_file() and path != manifest_path
        }
        hashes = manifest.get("artifact_hashes")
        if not isinstance(hashes, dict) or set(hashes) != set(files):
            raise ValueError(f"rejected artifact {attempt_id!r} hash set mismatch")
        for name, path in files.items():
            if hashes[name] != _sha256(path.read_bytes()):
                raise ValueError(f"rejected artifact {attempt_id!r} hash mismatch: {name}")


class DatasetWriter:
    def __init__(
        self,
        output: str | Path,
        *,
        run_id: str,
        quality_filter: QualityFilterConfig,
        prime_image: str,
        rlm_revision: str,
        environment_config: Mapping[str, Any],
    ) -> None:
        if not _SAFE_COMPONENT.fullmatch(run_id):
            raise ValueError(f"unsafe run_id: {run_id!r}")
        self.root = Path(output).expanduser().resolve()
        self.run_id = run_id
        self.quality_filter = quality_filter
        self.prime_image = prime_image
        self.rlm_revision = rlm_revision
        raw_environment_config = _jsonable(environment_config)
        self.environment_config_hash = _sha256(_canonical_bytes(raw_environment_config))
        self.environment_config = redact_secrets(raw_environment_config)
        self.generator_source_hash = _generator_source_hash()
        self.worldgen_version = version("worldgen-slack")
        self.verifiers_version = version("verifiers")
        self.pydantic_version = version("pydantic")
        self._lock = threading.RLock()
        self.root.mkdir(parents=True, exist_ok=True)
        worlds = self.root / "worlds"
        _reject_symlink(worlds, "worlds directory")
        worlds.mkdir(exist_ok=True)
        for name in (*JSONL_FILES, "setup_failures.jsonl"):
            path = self.root / name
            _reject_symlink(path, name)
            path.touch(exist_ok=True)
        self._recover_transactions()
        self.interface_hashes = self._install_interface()
        self._install_run_manifest()
        self._validate_resume()
        self._load_signature_index()
        self.write_summary()

    @property
    def attempts_path(self) -> Path:
        return self.root / "attempts.jsonl"

    @property
    def completed_seeds(self) -> set[int]:
        return {
            int(row["generation_seed"])
            for row in _read_jsonl(self.attempts_path)
            if isinstance(row.get("generation_seed"), int) and row.get("retry_discarded") is not True
        }

    @property
    def transaction_root(self) -> Path:
        return self.root / ".transactions"

    def _recover_transactions(self) -> None:
        _reject_symlink(self.transaction_root, "transaction directory")
        self.transaction_root.mkdir(exist_ok=True)
        for journal in sorted(self.transaction_root.glob("*.json")):
            _reject_symlink(journal, "transaction journal")
            record = _read_json(journal)
            sizes = record.get("jsonl_sizes")
            if not isinstance(sizes, dict) or set(sizes) != set(JSONL_FILES):
                raise ValueError(f"invalid transaction journal: {journal}")
            for name, size in sizes.items():
                if not isinstance(size, int) or size < 0:
                    raise ValueError(f"invalid transaction size in {journal}: {name}")
                path = self.root / name
                if path.stat().st_size < size:
                    raise ValueError(f"transaction journal exceeds current file size: {journal}")
                with path.open("r+b") as handle:
                    handle.truncate(size)
                    handle.flush()
                    os.fsync(handle.fileno())
            instance_id = record.get("instance_id")
            if instance_id is not None and (
                not isinstance(instance_id, str) or not _SAFE_COMPONENT.fullmatch(instance_id)
            ):
                raise ValueError(f"unsafe instance_id in transaction journal: {journal}")
            if isinstance(instance_id, str) and not record.get("world_existed", False):
                shutil.rmtree(self.root / "worlds" / instance_id, ignore_errors=True)
            attempt_id = record.get("attempt_id")
            if not isinstance(attempt_id, str) or not _SAFE_ATTEMPT_COMPONENT.fullmatch(attempt_id):
                raise ValueError(f"unsafe attempt_id in transaction journal: {journal}")
            if not record.get("rejected_existed", False):
                shutil.rmtree(self.root / "rejected" / attempt_id, ignore_errors=True)
            journal.unlink()
        for temporary in (self.root / "worlds").glob(".*"):
            if temporary.is_dir():
                shutil.rmtree(temporary, ignore_errors=True)
        rejected = self.root / "rejected"
        if rejected.is_dir():
            for temporary in rejected.glob(".*"):
                if temporary.is_dir():
                    shutil.rmtree(temporary, ignore_errors=True)

    def _install_interface(self) -> dict[str, str]:
        source = Path(__file__).resolve().parent / "slack"
        interfaces = self.root / "interfaces"
        destination = interfaces / INTERFACE_ID
        _reject_symlink(interfaces, "interfaces directory")
        _reject_symlink(destination, "interface version directory")
        destination.mkdir(parents=True, exist_ok=True)
        hashes: dict[str, str] = {}
        for source_name, target_name in (
            ("interface.json", "interface.json"),
            ("models.schema.json", "models.schema.json"),
            ("behavior.md", "behavior.md"),
        ):
            data = (source / source_name).read_bytes()
            target = destination / target_name
            _reject_symlink(target, f"interface file {target_name}")
            if target.exists() and target.read_bytes() != data:
                raise ValueError(f"existing interface artifact differs: {target}")
            if not target.exists():
                _atomic_write(target, data)
            hashes[target_name] = _sha256(data)
        return hashes

    def _install_run_manifest(self) -> None:
        manifest = {
            "schema_version": 1,
            "generator_schema_version": GENERATOR_SCHEMA_VERSION,
            "interface_id": INTERFACE_ID,
            "run_id": self.run_id,
            "worldgen_version": self.worldgen_version,
            "verifiers_version": self.verifiers_version,
            "pydantic_version": self.pydantic_version,
            "generator_source_hash": self.generator_source_hash,
            "prime_image": self.prime_image,
            "rlm_revision": self.rlm_revision,
            "interface_hashes": self.interface_hashes,
            "environment_config_hash": self.environment_config_hash,
            "environment_config": self.environment_config,
        }
        path = self.root / "run_manifest.json"
        _reject_symlink(path, "run manifest")
        if path.exists():
            if _read_json(path) != _jsonable(manifest):
                raise ValueError("output directory belongs to a different immutable run config")
            return
        has_existing_attempts = bool(_read_jsonl(self.root / "attempts.jsonl"))
        has_existing_worlds = any((self.root / "worlds").iterdir())
        if has_existing_attempts or has_existing_worlds:
            raise ValueError("existing output has no immutable run_manifest.json")
        _atomic_write(path, _dumps(manifest, pretty=True))

    def _validate_resume(self) -> None:
        validate_release_integrity(self.root, expected_run_id=self.run_id)

    def _load_signature_index(self) -> None:
        self._signatures: dict[str, str] = {}
        self._signature_seeds: dict[str, int] = {}
        self._committed_signatures: set[str] = set()
        for row in _read_jsonl(self.attempts_path):
            signature = row.get("signature")
            text = row.get("signature_text")
            seed = row.get("generation_seed")
            if isinstance(signature, str) and isinstance(text, str):
                self._signatures[signature] = text
                self._committed_signatures.add(signature)
                if isinstance(seed, int):
                    self._signature_seeds[signature] = seed

    def release_uncommitted_signatures(self, generation_seed: int) -> None:
        with self._lock:
            stale = [
                signature
                for signature, seed in self._signature_seeds.items()
                if seed == generation_seed and signature not in self._committed_signatures
            ]
            for signature in stale:
                self._signatures.pop(signature, None)
                self._signature_seeds.pop(signature, None)

    def reserve_signature(
        self,
        item: SynthesizedItem,
        *,
        generation_seed: int,
        near_duplicate_threshold: float,
    ) -> tuple[bool, str, str, str | None]:
        signature = synthesized_signature(item)
        text = synthesized_signature_text(item)
        with self._lock:
            if signature in self._signatures:
                if self._signature_seeds.get(signature) == generation_seed:
                    return True, signature, text, None
                return False, signature, text, "exact_duplicate"
            closest = max(
                (
                    (token_jaccard(text, other), key)
                    for key, other in self._signatures.items()
                    if self._signature_seeds.get(key) != generation_seed
                ),
                default=(0.0, ""),
            )
            if closest[0] >= near_duplicate_threshold:
                return False, signature, text, f"near_duplicate:{closest[0]:.3f}:{closest[1]}"
            self._signatures[signature] = text
            self._signature_seeds[signature] = generation_seed
        return True, signature, text, None

    def recent_summaries(self, limit: int = 10) -> list[dict[str, object]]:
        rows = _read_jsonl(self.root / "dataset.jsonl")[-limit:]
        output: list[dict[str, object]] = []
        for row in rows:
            scenario_path = self.root / "worlds" / row["instance_id"] / "scenario.json"
            scenario = json.loads(scenario_path.read_text(encoding="utf-8"))
            output.append(
                {
                    "organization": scenario["organization"],
                    "workflow": scenario["workflow"],
                    "question": row["question"],
                    "required_claims": row.get("required_claims", []),
                }
            )
        return output

    @staticmethod
    def identity(item: SynthesizedItem) -> tuple[str, str]:
        contract_hash = hashlib.sha256(_canonical_bytes(item.task)).hexdigest()
        source = item.task.task_slug or f"{item.scenario.organization}-{item.scenario.workflow}"
        instance_id = f"{_safe_slug(source)}--{contract_hash[:12]}"
        if not _SAFE_COMPONENT.fullmatch(instance_id):
            raise ValueError(f"unsafe generated instance ID: {instance_id!r}")
        return instance_id, contract_hash

    def _next_attempt_id(self, generation_seed: int) -> str:
        count = sum(row.get("generation_seed") == generation_seed for row in _read_jsonl(self.attempts_path))
        return f"{self.run_id}--seed-{generation_seed:08d}--attempt-{count + 1:04d}"

    @staticmethod
    def _runtime_records(traces: Sequence[vf.Trace]) -> list[dict[str, Any]]:
        output = []
        for trace in traces:
            if not trace.agent or not trace.agent.runtime:
                continue
            record = trace.agent.runtime.model_dump(mode="json")
            record["role"] = trace.agent.name
            record["trace_id"] = trace.id
            output.append(record)
        return output

    def write_setup_failure(
        self,
        *,
        generation_seed: int,
        episode: vf.Episode,
        reason: str,
        traces: Sequence[vf.Trace] = (),
    ) -> dict[str, Any]:
        diagnostic_traces = list(traces) or list(episode.traces)
        errors = [*episode.errors]
        for trace in diagnostic_traces:
            errors.extend(trace.errors)
        row = redact_secrets(
            {
                "run_id": self.run_id,
                "created_at": datetime.now(UTC).isoformat(),
                "generation_seed": generation_seed,
                "reason": reason,
                "episode_id": episode.id,
                "errors": [{"type": error.type, "message": error.message} for error in errors],
                "trace_ids": [trace.id for trace in diagnostic_traces],
                "runtime_records": self._runtime_records(diagnostic_traces),
            }
        )
        with self._lock:
            _append_jsonl(self.root / "setup_failures.jsonl", row)
        return row

    def write_attempt(
        self,
        result: GenerationResult,
        *,
        signature_text: str | None,
        source: str | None,
        traces: Sequence[vf.Trace],
        public_checks: Sequence[Mapping[str, Any]] = (),
        builder_metadata: Mapping[str, Any] | None = None,
        retry_discarded: bool = False,
    ) -> dict[str, Any]:
        with self._lock:
            attempt_id = self._next_attempt_id(result.generation_seed)
            trace_ids: dict[str, list[str]] = {}
            for trace in traces:
                role = trace.agent.name if trace.agent and trace.agent.name else "unknown"
                trace_ids.setdefault(role, []).append(trace.id)
            result = result.model_copy(update={"trace_ids": trace_ids})
            sizes = {self.root / name: (self.root / name).stat().st_size for name in JSONL_FILES}
            world_path = self.root / "worlds" / result.instance_id if result.instance_id is not None else None
            rejected_path = self.root / "rejected" / attempt_id
            world_existed = bool(world_path and world_path.exists())
            rejected_existed = rejected_path.exists()
            journal = self.transaction_root / f"{attempt_id}.json"
            if journal.exists():
                raise FileExistsError(f"transaction already exists: {journal}")
            _atomic_write(
                journal,
                _dumps(
                    {
                        "attempt_id": attempt_id,
                        "instance_id": result.instance_id,
                        "world_existed": world_existed,
                        "rejected_existed": rejected_existed,
                        "jsonl_sizes": {path.name: size for path, size in sizes.items()},
                    },
                    pretty=True,
                ),
            )
            _fsync_directory(self.transaction_root)
            try:
                written = False
                artifact_ref = None
                trace_refs: dict[str, list[str]] = {}
                if result.decision and result.decision.write_to_dataset:
                    artifact_ref = self._write_accepted(
                        result,
                        source=source,
                        traces=traces,
                        public_checks=public_checks,
                        builder_metadata=builder_metadata or {},
                        attempt_id=attempt_id,
                    )
                    written = True
                    for role in trace_ids:
                        reference = f"{artifact_ref}/{role}_trace.json"
                        if (self.root / reference).is_file():
                            trace_refs[role] = [reference]
                elif self.quality_filter.retain_rejected_artifacts:
                    artifact_ref, trace_refs = self._write_rejected(
                        attempt_id,
                        result,
                        source=source,
                        traces=traces,
                        public_checks=public_checks,
                        builder_metadata=builder_metadata or {},
                    )

                attempt_reason = result.reason
                if result.failure_owner == FailureOwner.INFRASTRUCTURE and attempt_reason:
                    attempt_reason = str(redact_secrets(attempt_reason))[:4_000]
                safe_builder = dict(builder_metadata or {})
                episode_errors = safe_builder.get("episode_errors")
                if isinstance(episode_errors, list):
                    safe_builder["episode_errors"] = [
                        {
                            "type": str(item.get("type", ""))[:128],
                            "message": str(redact_secrets(item.get("message", "")))[:1_000],
                        }
                        for item in episode_errors[:20]
                        if isinstance(item, dict)
                    ]
                safe_solver = result.solver.model_dump(mode="json") if result.solver else None
                if safe_solver is not None:
                    safe_solver["visible_errors"] = [
                        str(redact_secrets(value))[:1_000]
                        for value in safe_solver.get("visible_errors", [])[:20]
                    ]
                row = {
                    "attempt_id": attempt_id,
                    "run_id": self.run_id,
                    "created_at": datetime.now(UTC).isoformat(),
                    "generation_seed": result.generation_seed,
                    "instance_id": result.instance_id,
                    "item_status": result.status.value,
                    "failure_owner": result.failure_owner.value,
                    "reason": attempt_reason,
                    "signature": result.signature,
                    "signature_text": signature_text,
                    "semantic_status": result.decision.status if result.decision else None,
                    "judge_scores": result.verdict.scores if result.verdict else None,
                    "quality_score": result.decision.quality_score if result.decision else None,
                    "criterion_failures": (result.decision.criterion_failures if result.decision else []),
                    "rejection_reason": (result.decision.rejection_reason if result.decision else None),
                    "written_to_dataset": written,
                    "retry_discarded": retry_discarded,
                    "artifact_ref": artifact_ref,
                    "trace_ids": trace_ids,
                    "trace_refs": trace_refs,
                    "runtime_records": self._runtime_records(traces),
                    "candidate_runtime": (result.validation.runtime if result.validation else None),
                    "builder": safe_builder,
                    "solver": safe_solver,
                    "verdict": (result.verdict.model_dump(mode="json") if result.verdict else None),
                    "validation": (
                        result.validation.model_dump(mode="json", exclude={"public_snapshot"})
                        if result.validation
                        else None
                    ),
                }
                _append_jsonl(self.attempts_path, row)
                validate_release_integrity(self.root, expected_run_id=self.run_id)
                self.write_summary()
                if result.signature is not None:
                    self._committed_signatures.add(result.signature)
                journal.unlink()
                _fsync_directory(self.transaction_root)
                return row
            except BaseException:
                for path, size in sizes.items():
                    with path.open("r+b") as handle:
                        handle.truncate(size)
                        handle.flush()
                        os.fsync(handle.fileno())
                if world_path is not None and not world_existed:
                    shutil.rmtree(world_path, ignore_errors=True)
                if not rejected_existed:
                    shutil.rmtree(rejected_path, ignore_errors=True)
                journal.unlink(missing_ok=True)
                _fsync_directory(self.transaction_root)
                raise

    def _write_accepted(
        self,
        result: GenerationResult,
        *,
        source: str | None,
        traces: Sequence[vf.Trace],
        public_checks: Sequence[Mapping[str, Any]],
        builder_metadata: Mapping[str, Any],
        attempt_id: str,
    ) -> str:
        if not all(
            (
                result.instance_id,
                result.synthesized,
                result.validation,
                result.validation and result.validation.public_snapshot,
                result.solver,
                result.verdict,
                result.decision,
                source,
            )
        ):
            raise ValueError("accepted item is missing required artifacts")
        item = result.synthesized
        validation = result.validation
        verdict = result.verdict
        decision = result.decision
        instance_id = result.instance_id
        assert item is not None and validation is not None and verdict is not None and decision is not None
        assert instance_id is not None and source is not None and validation.public_snapshot is not None
        computed_id, contract_hash = self.identity(item)
        if computed_id != instance_id:
            raise ValueError("accepted instance ID does not match its frozen contract")
        destination = self.root / "worlds" / instance_id
        if destination.exists():
            raise FileExistsError(f"accepted item already exists: {instance_id}")
        temporary = Path(tempfile.mkdtemp(prefix=f".{instance_id}.", dir=self.root / "worlds"))
        canonical_snapshot_bytes = _canonical_bytes(validation.public_snapshot)
        snapshot_bytes = canonical_snapshot_bytes + b"\n"
        source_bytes = source.encode()
        trace_by_role = {
            role: [trace for trace in traces if trace.agent and trace.agent.name == role]
            for role in ("synthesizer", "builder", "solver", "judge")
        }
        missing_traces = [role for role, role_traces in trace_by_role.items() if not role_traces]
        if missing_traces:
            raise ValueError(f"accepted item is missing role traces: {missing_traces}")
        runtime_records = self._runtime_records(traces)
        manifest = {
            "schema_version": 1,
            "generator_schema_version": GENERATOR_SCHEMA_VERSION,
            "generator_source_hash": self.generator_source_hash,
            "worldgen_version": self.worldgen_version,
            "verifiers_version": self.verifiers_version,
            "pydantic_version": self.pydantic_version,
            "run_id": self.run_id,
            "attempt_id": attempt_id,
            "instance_id": instance_id,
            "generation_seed": result.generation_seed,
            "interface_id": INTERFACE_ID,
            "interface_hashes": self.interface_hashes,
            "prime_image": self.prime_image,
            "rlm_revision": self.rlm_revision,
            "world_hash": _sha256(source_bytes),
            "snapshot_hash": _sha256(snapshot_bytes),
            "canonical_snapshot_hash": "sha256:" + canonical_world_hash(validation.public_snapshot),
            "contract_hash": "sha256:" + contract_hash,
            "synthesized_item_hash": _sha256(_canonical_bytes(item)),
            "hidden_snapshot_hashes": validation.hidden_snapshot_hashes,
            "runtime_records": runtime_records,
            "builder": dict(builder_metadata),
            "created_at": datetime.now(UTC).isoformat(),
        }
        qa = {
            "question": item.task.question,
            "answer": item.task.answer.canonical_answer,
            "required_claims": item.task.answer.required_claims,
            "forbidden_claims": item.task.answer.forbidden_claims,
        }
        gold = {
            "calls": [call.model_dump(mode="json") for call in item.task.gold_calls],
            "call_log": validation.gold_call_log,
        }
        try:
            files: dict[str, str | bytes] = {
                "scenario.json": _dumps(item.scenario, pretty=True),
                "qa.json": _dumps(qa, pretty=True),
                "task_contract.json": _dumps(item.task, pretty=True),
                "world.py": source_bytes,
                "snapshot.json": snapshot_bytes,
                "hidden_snapshot_hashes.json": _dumps(validation.hidden_snapshot_hashes, pretty=True),
                "gold.json": _dumps(gold, pretty=True),
                "validation.json": _dumps(
                    validation.model_dump(mode="json", exclude={"public_snapshot"}),
                    pretty=True,
                ),
                "judge_verdict.json": _dumps(verdict, pretty=True),
                "public_checks.json": _dumps(list(public_checks), pretty=True),
            }
            for role, role_traces in trace_by_role.items():
                if role_traces:
                    files[f"{role}_trace.json"] = _dumps(
                        role_traces[-1].to_record(), pretty=True, redact=True
                    )
            manifest["artifact_hashes"] = {
                name: _sha256(content if isinstance(content, bytes) else content.encode())
                for name, content in files.items()
            }
            files["manifest.json"] = _dumps(manifest, pretty=True)
            for name, content in files.items():
                _atomic_write(temporary / name, content)
            _fsync_directory(temporary)
            os.replace(temporary, destination)
            _fsync_directory(destination.parent)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

        base = f"worlds/{instance_id}"
        catalog = {
            "instance_id": instance_id,
            "domain": "slack",
            "scenario": item.scenario.description,
            "question": item.task.question,
            "answer": item.task.answer.canonical_answer,
            "required_claims": item.task.answer.required_claims,
            "status": decision.status,
            "quality_score": decision.quality_score,
            "judge_scores": verdict.scores,
            "world_ref": f"{base}/world.py",
            "snapshot_ref": f"{base}/snapshot.json",
            "oracle_ref": f"{base}/task_contract.json",
            "solver_trace_ref": f"{base}/solver_trace.json",
            "judge_ref": f"{base}/judge_verdict.json",
            "interface_id": INTERFACE_ID,
            "interface_ref": f"interfaces/{INTERFACE_ID}/interface.json",
            "world_hash": manifest["world_hash"],
            "contract_hash": manifest["contract_hash"],
        }
        public = {
            "instance_id": instance_id,
            "scenario": item.scenario.description,
            "question": item.task.question,
            "snapshot_ref": f"{base}/snapshot.json",
            "interface_id": INTERFACE_ID,
        }
        private = {
            "instance_id": instance_id,
            "answer": item.task.answer.canonical_answer,
            "required_claims": item.task.answer.required_claims,
            "forbidden_claims": item.task.answer.forbidden_claims,
            "task_contract_ref": f"{base}/task_contract.json",
            "gold_ref": f"{base}/gold.json",
        }
        _append_jsonl(self.root / "dataset.jsonl", catalog)
        _append_jsonl(self.root / "public_tasks.jsonl", public)
        _append_jsonl(self.root / "private_oracles.jsonl", private)
        return base

    def _write_rejected(
        self,
        attempt_id: str,
        result: GenerationResult,
        *,
        source: str | None,
        traces: Sequence[vf.Trace],
        public_checks: Sequence[Mapping[str, Any]],
        builder_metadata: Mapping[str, Any],
    ) -> tuple[str, dict[str, list[str]]]:
        destination = self.root / "rejected" / attempt_id
        if destination.exists():
            raise FileExistsError(f"rejected artifact already exists: {attempt_id}")
        rejected_root = self.root / "rejected"
        _reject_symlink(rejected_root, "rejected directory")
        rejected_root.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{attempt_id}.", dir=rejected_root))
        base = f"rejected/{attempt_id}"
        trace_refs: dict[str, list[str]] = {}
        files: dict[str, str | bytes] = {
            "result.json": _dumps(result, pretty=True, redact=True),
            "public_checks.json": _dumps(list(public_checks), pretty=True, redact=True),
            "builder.json": _dumps(dict(builder_metadata), pretty=True, redact=True),
        }
        if source is not None:
            files["world.py"] = source
        if result.validation and result.validation.public_snapshot:
            files["snapshot.json"] = _canonical_bytes(result.validation.public_snapshot) + b"\n"
        counters: Counter[str] = Counter()
        for trace in traces:
            role = trace.agent.name if trace.agent and trace.agent.name else "unknown"
            counters[role] += 1
            suffix = "" if counters[role] == 1 else f"-{counters[role]}"
            relative = f"traces/{role}{suffix}.json"
            files[relative] = _dumps(trace.to_record(), pretty=True, redact=True)
            trace_refs.setdefault(role, []).append(f"{base}/{relative}")
        manifest = {
            "schema_version": 1,
            "attempt_id": attempt_id,
            "artifact_hashes": {
                name: _sha256(content if isinstance(content, bytes) else content.encode())
                for name, content in files.items()
            },
        }
        files["manifest.json"] = _dumps(manifest, pretty=True)
        try:
            for name, content in files.items():
                _atomic_write(temporary / name, content)
            _fsync_directory(temporary)
            os.replace(temporary, destination)
            _fsync_directory(destination.parent)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return base, trace_refs

    def write_summary(self) -> dict[str, Any]:
        with self._lock:
            attempts = _read_jsonl(self.attempts_path)
            final_attempts = [row for row in attempts if row.get("retry_discarded") is not True]
            scores = [
                float(row["quality_score"])
                for row in attempts
                if isinstance(row.get("quality_score"), (int, float))
            ]
            statuses = Counter(str(row.get("item_status")) for row in final_attempts)
            owners = Counter(str(row.get("failure_owner")) for row in final_attempts)
            semantic = Counter(
                str(row.get("semantic_status")) for row in final_attempts if row.get("semantic_status")
            )
            floor_counts: Counter[str] = Counter()
            for row in attempts:
                floor_counts.update(row.get("criterion_failures") or [])
            written = [row for row in attempts if row.get("written_to_dataset")]
            before_quality = [
                row for row in attempts if row.get("semantic_status") in self.quality_filter.accepted_statuses
            ]
            builder_rows = [row.get("builder") or {} for row in attempts]
            solver_rows = [row.get("solver") or {} for row in attempts]
            sandbox_ids = {
                record.get("id")
                for row in attempts
                for record in [*(row.get("runtime_records") or []), row.get("candidate_runtime") or {}]
                if isinstance(record, dict) and record.get("id")
            }
            summary = {
                "run_id": self.run_id,
                "output": str(self.root),
                "attempted": len(final_attempts),
                "attempt_records": len(attempts),
                "retry_discarded_attempts": len(attempts) - len(final_attempts),
                "judge_complete_attempts": len(scores),
                "written_to_dataset": len(written),
                "semantic_status_counts": dict(sorted(semantic.items())),
                "item_status_counts": dict(sorted(statuses.items())),
                "failure_owner_counts": dict(sorted(owners.items())),
                "hard_gate_rejected": statuses[ItemStatus.HARD_GATE_REJECTED.value],
                "quality_threshold_rejected": statuses[ItemStatus.QUALITY_THRESHOLD_REJECTED.value],
                "criterion_floor_rejected": statuses[ItemStatus.CRITERION_FLOOR_REJECTED.value],
                "criterion_floor_counts": dict(sorted(floor_counts.items())),
                "duplicate_contracts_rejected": statuses[ItemStatus.DUPLICATE.value],
                "synthesizer_failures": statuses[ItemStatus.SYNTHESIZER_FAILURE.value],
                "builder_failures": statuses[ItemStatus.BUILDER_FAILURE.value],
                "judge_protocol_failures": statuses[ItemStatus.JUDGE_PROTOCOL_FAILURE.value],
                "infrastructure_errors": statuses[ItemStatus.INFRASTRUCTURE_ERROR.value],
                "quality_score_distribution": _distribution(scores),
                "quality_filter": self.quality_filter.model_dump(mode="json"),
                "acceptance_rate_before_quality_filter": (
                    len(before_quality) / len(scores) if scores else 0.0
                ),
                "acceptance_rate_after_quality_filter": (len(written) / len(scores) if scores else 0.0),
                "written_by_semantic_status": dict(
                    Counter(str(row.get("semantic_status")) for row in written)
                ),
                "average_builder_turns": _mean_values(builder_rows, "turns"),
                "average_builder_tokens": _mean_values(builder_rows, "tokens"),
                "average_builder_wall_seconds": _mean_values(builder_rows, "wall_seconds"),
                "average_solver_tool_calls": _mean_values(solver_rows, "tool_call_count"),
                "prime_sandbox_provisioning_count": len(sandbox_ids),
                "prime_sandbox_ids": sorted(sandbox_ids),
                "prime_image": self.prime_image,
                "rlm_revision": self.rlm_revision,
                "interface_hashes": self.interface_hashes,
            }
            _atomic_write(self.root / "summary.json", _dumps(summary, pretty=True))
            return summary


def _mean_values(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
    return sum(values) / len(values) if values else 0.0


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _distribution(values: Sequence[float]) -> dict[str, float | None]:
    return {
        "mean": sum(values) / len(values) if values else None,
        "median": _percentile(values, 0.5),
        "p10": _percentile(values, 0.1),
        "p90": _percentile(values, 0.9),
    }


__all__ = ["DatasetWriter"]
