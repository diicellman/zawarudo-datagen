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
    FailedStage,
    FailureKind,
    FailureOwner,
    GenerationResult,
    ItemStatus,
    ReleaseAcceptanceConfig,
    ValidationReport,
    WorldJudgeVerdict,
    item_identity,
    normalize_signature_text,
    redact_secrets,
)
from .slack.models import (
    INTERFACE_ID,
    QUALITY_CRITERIA,
    ScenarioSpec,
    SlackWorld,
    SynthesizedItem,
    TaskContract,
)
from .slack.validate import canonical_world_hash

GENERATOR_SCHEMA_VERSION = "worldgen-slack.dataset.v3"
RELEASE_TABLE_SCHEMA_VERSION = 2


def synthesized_signature_text(item: SynthesizedItem) -> str:
    fields = [
        item.scenario.organization,
        item.scenario.workflow,
        item.task.question,
        *item.task.answer.required_claims,
    ]
    return " | ".join(normalize_signature_text(value) for value in fields)


def synthesized_signature(item: SynthesizedItem) -> str:
    return hashlib.sha256(synthesized_signature_text(item).encode()).hexdigest()


def token_jaccard(left: str, right: str) -> float:
    left_tokens = set(normalize_signature_text(left).split())
    right_tokens = set(normalize_signature_text(right).split())
    if not left_tokens and not right_tokens:
        return 1.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _generator_source_hash() -> str:
    package = Path(__file__).resolve().parent
    base = package.parent
    files = [
        path
        for path in package.rglob("*")
        if path.is_file() and path.suffix in {".py", ".json", ".md", ".toml"}
    ]
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
    "artifacts.jsonl",
)
ROLE_NAMES = ("synthesizer", "builder", "solver", "judge")
ATTEMPT_VIEWER_FIELDS = {
    "schema_version",
    "run_id",
    "attempt_id",
    "created_at",
    "instance_id",
    "generation_seed",
    "status",
    "item_status",
    "failure_owner",
    "failed_stage",
    "failure_kind",
    "synthesis_ok",
    "validation_ok",
    "solver_score",
    "builder_score",
    "builder_raw_score",
    "task_unambiguous",
    "world_supports_task",
    "world_criteria",
    "synthesizer_tokens",
    "builder_tokens",
    "solver_tokens",
    "judge_tokens",
    "synthesizer_duration_ms",
    "builder_duration_ms",
    "solver_duration_ms",
    "judge_duration_ms",
    "synthesizer_cost",
    "builder_cost",
    "solver_cost",
    "judge_cost",
    "solver_judge_tokens",
    "solver_judge_cost",
    "solver_judge_model",
    "world_judge_completed_action_count",
    "world_judge_retry_count",
    "whole_episode_retry_count",
    "scenario_path",
    "contract_path",
    "source_path",
    "snapshot_path",
    "validation_path",
    "solver_trace_path",
    "world_judge_trace_path",
    "solver_verdict_path",
    "world_verdict_path",
    "progress_path",
}
ARTIFACT_VIEWER_FIELDS = {
    "schema_version",
    "run_id",
    "attempt_id",
    "instance_id",
    "path",
    "sha256",
    "size_bytes",
}
CATALOG_FIELDS = {
    "instance_id",
    "domain",
    "scenario",
    "question",
    "answer",
    "required_claims",
    "status",
    "solver_score",
    "builder_score",
    "builder_raw_score",
    "world_criteria",
    "world_hard_gates",
    "world_ref",
    "snapshot_ref",
    "oracle_ref",
    "solver_trace_ref",
    "world_judge_trace_ref",
    "solver_verdict_ref",
    "world_verdict_ref",
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
    for name in (*JSONL_FILES, "progress.jsonl", "manifest.json", "run_manifest.json"):
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
    artifact_rows = _read_jsonl(release / "artifacts.jsonl")
    viewer_manifest_path = release / "manifest.json"
    if not viewer_manifest_path.is_file():
        raise ValueError("release has no viewer manifest.json")
    viewer_manifest = _read_json(viewer_manifest_path)
    expected_viewer_manifest = {
        "schema_version": RELEASE_TABLE_SCHEMA_VERSION,
        "tables": {
            "attempts": {"path": "attempts.jsonl", "format": "jsonl"},
            "artifacts": {"path": "artifacts.jsonl", "format": "jsonl"},
            "progress": {"path": "progress.jsonl", "format": "jsonl"},
        },
    }
    if viewer_manifest != expected_viewer_manifest:
        raise ValueError("viewer manifest contract mismatch")
    run_manifest_path = release / "run_manifest.json"
    if not run_manifest_path.is_file():
        raise ValueError("release has no immutable run_manifest.json")
    run_manifest = _read_json(run_manifest_path)
    manifest_run_id = run_manifest.get("run_id")
    if not isinstance(manifest_run_id, str) or not _SAFE_COMPONENT.fullmatch(manifest_run_id):
        raise ValueError("run manifest has an unsafe run_id")
    if expected_run_id is not None and manifest_run_id != expected_run_id:
        raise ValueError("run manifest belongs to another run")
    if run_manifest.get("schema_version") != 2:
        raise ValueError("run manifest schema mismatch")
    if run_manifest.get("interface_id") != INTERFACE_ID:
        raise ValueError("run manifest interface mismatch")
    if run_manifest.get("generator_schema_version") != GENERATOR_SCHEMA_VERSION:
        raise ValueError("unsupported generator dataset schema")
    if not isinstance(run_manifest.get("generator_source_hash"), str) or not run_manifest[
        "generator_source_hash"
    ].startswith("sha256:"):
        raise ValueError("run manifest has no generator source hash")
    if run_manifest.get("progress_path") != "progress.jsonl":
        raise ValueError("run manifest has an invalid progress path")
    if not isinstance(run_manifest.get("acceptance"), dict):
        raise ValueError("run manifest has no acceptance policy")
    ReleaseAcceptanceConfig.model_validate(run_manifest["acceptance"])
    for field in ("target_accepted", "max_attempts", "concurrency"):
        if isinstance(run_manifest.get(field), bool) or not isinstance(run_manifest.get(field), int):
            raise ValueError(f"run manifest has invalid {field}")
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
        if not ATTEMPT_VIEWER_FIELDS <= set(row):
            missing = sorted(ATTEMPT_VIEWER_FIELDS - set(row))
            raise ValueError(f"attempt {attempt_id!r} is missing viewer fields: {missing}")
        if row.get("schema_version") != RELEASE_TABLE_SCHEMA_VERSION:
            raise ValueError(f"attempt {attempt_id!r} has an unsupported schema version")
        created_at = row.get("created_at")
        try:
            timestamp = datetime.fromisoformat(created_at)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"attempt {attempt_id!r} has an invalid UTC timestamp") from exc
        if timestamp.tzinfo is None or timestamp.utcoffset() != UTC.utcoffset(timestamp):
            raise ValueError(f"attempt {attempt_id!r} timestamp is not UTC")
        for role in ROLE_NAMES:
            for suffix in ("tokens", "duration_ms"):
                field = f"{role}_{suffix}"
                value = row.get(field)
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError(f"attempt {attempt_id!r} has invalid {field}")
            cost = row.get(f"{role}_cost")
            if cost is not None and (
                isinstance(cost, bool) or not isinstance(cost, (int, float)) or cost < 0
            ):
                raise ValueError(f"attempt {attempt_id!r} has invalid {role}_cost")
        for field in ("solver_score", "builder_score", "builder_raw_score"):
            value = row.get(field)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1
            ):
                raise ValueError(f"attempt {attempt_id!r} has invalid {field}")
        criteria = row.get("world_criteria")
        if criteria is not None and (
            not isinstance(criteria, dict) or set(criteria) != set(QUALITY_CRITERIA)
        ):
            raise ValueError(f"attempt {attempt_id!r} has invalid world criteria")
        if row.get("progress_path") != "progress.jsonl":
            raise ValueError(f"attempt {attempt_id!r} has an invalid progress path")
        instance_id = row.get("instance_id")
        if instance_id is not None and (
            not isinstance(instance_id, str) or not _SAFE_COMPONENT.fullmatch(instance_id)
        ):
            raise ValueError(f"attempt {attempt_id!r} has an unsafe instance_id")
        written = row.get("written_to_dataset")
        if not isinstance(written, bool):
            raise ValueError(f"attempt {attempt_id!r} has no boolean persistence decision")
        expected_status = "accepted" if written else "rejected"
        if row.get("status") != expected_status:
            raise ValueError(f"attempt {attempt_id!r} status disagrees with persistence")
        generation_seed = row.get("generation_seed")
        if isinstance(generation_seed, bool) or not isinstance(generation_seed, int) or generation_seed < 0:
            raise ValueError(f"attempt {attempt_id!r} has an invalid generation seed")
        if generation_seed in terminal_seeds:
            raise ValueError(f"generation seed {generation_seed} has multiple terminal attempts")
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
        for field in (
            "scenario_path",
            "contract_path",
            "source_path",
            "snapshot_path",
            "validation_path",
            "solver_trace_path",
            "world_judge_trace_path",
            "solver_verdict_path",
            "world_verdict_path",
        ):
            reference = row.get(field)
            if reference is None:
                continue
            target = _inside(release, reference)
            if not target.is_file():
                raise ValueError(f"attempt {attempt_id!r} has a dangling {field}")
            if artifact_ref is None or not reference.startswith(f"{artifact_ref}/"):
                raise ValueError(f"attempt {attempt_id!r} has a cross-attempt {field}")
    if set(accepted_attempts) != set(catalog) or any(count != 1 for count in accepted_attempts.values()):
        raise ValueError("accepted dataset rows must have exactly one accepted attempt")

    attempts_by_id = {row["attempt_id"]: row for row in attempt_rows}
    artifact_paths: set[str] = set()
    for number, row in enumerate(artifact_rows, 1):
        if set(row) != ARTIFACT_VIEWER_FIELDS:
            raise ValueError(f"artifact row {number} has unexpected fields")
        if row.get("schema_version") != RELEASE_TABLE_SCHEMA_VERSION:
            raise ValueError(f"artifact row {number} has an unsupported schema version")
        if row.get("run_id") != manifest_run_id:
            raise ValueError(f"artifact row {number} belongs to another run")
        attempt = attempts_by_id.get(row.get("attempt_id"))
        if attempt is None or row.get("instance_id") != attempt.get("instance_id"):
            raise ValueError(f"artifact row {number} has no matching attempt")
        reference = row.get("path")
        target = _inside(release, reference)
        if reference in artifact_paths:
            raise ValueError("artifacts.jsonl has duplicate paths")
        artifact_paths.add(reference)
        if not target.is_file() or target.is_symlink():
            raise ValueError(f"artifact row {number} has a dangling or symlinked path")
        digest = row.get("sha256")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError(f"artifact row {number} has an invalid SHA-256 hash")
        if digest != hashlib.sha256(target.read_bytes()).hexdigest():
            raise ValueError(f"artifact hash mismatch: {reference}")
        if row.get("size_bytes") != target.stat().st_size:
            raise ValueError(f"artifact size mismatch: {reference}")

    interface_root = release / "interfaces" / INTERFACE_ID
    _reject_symlink(interface_root, "interface version directory")
    interface_hashes = {}
    for name in (
        "interface.json",
        "slack_world.schema.json",
        "task_contract.schema.json",
        "builder_guide.md",
        "world_rubric.toml",
    ):
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
            "world_judge_trace_ref": f"{base}/judge_trace.json",
            "solver_verdict_ref": f"{base}/solver_verdict.json",
            "world_verdict_ref": f"{base}/world_verdict.json",
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
            "solver_verdict.json",
            "world_verdict.json",
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
        if manifest.get("schema_version") != 2:
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
        expected_attempt_paths = {
            "scenario_path": f"{base}/scenario.json",
            "contract_path": f"{base}/task_contract.json",
            "source_path": f"{base}/world.py",
            "snapshot_path": f"{base}/snapshot.json",
            "validation_path": f"{base}/validation.json",
            "solver_trace_path": f"{base}/solver_trace.json",
            "world_judge_trace_path": f"{base}/judge_trace.json",
            "solver_verdict_path": f"{base}/solver_verdict.json",
            "world_verdict_path": f"{base}/world_verdict.json",
        }
        for field, reference in expected_attempt_paths.items():
            if matching_attempt.get(field) != reference:
                raise ValueError(f"attempt {field} mismatch for {instance_id}")
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
            "world_rubric_hash",
        ):
            if manifest.get(key) != run_manifest.get(key):
                raise ValueError(f"manifest {key} mismatch for {instance_id}")
        if manifest.get("interface_hashes") != interface_hashes:
            raise ValueError(f"manifest interface hashes mismatch for {instance_id}")
        if manifest.get("acceptance") != run_manifest.get("acceptance"):
            raise ValueError(f"manifest acceptance policy mismatch for {instance_id}")
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
        validation = ValidationReport.model_validate_json(
            json.dumps({**validation_data, "public_snapshot": world.model_dump(mode="json")})
        )
        if not validation.ok or validation.gold_call_log != gold["call_log"]:
            raise ValueError(f"validation/gold mismatch for {instance_id}")
        from .agents.judge import world_reward_scores
        from .agents.solver import SolverAnswerVerdict, solver_verdict_scores

        world_verdict = WorldJudgeVerdict.model_validate_json(
            json.dumps(_read_json(world_dir / "world_verdict.json"))
        )
        solver_verdict = SolverAnswerVerdict.model_validate(_read_json(world_dir / "solver_verdict.json"))
        solver_scores = solver_verdict_scores(solver_verdict, contract.answer)
        world_scores = world_reward_scores(True, world_verdict)
        if matching_attempt.get("solver_score") != solver_scores["semantic_correctness"]:
            raise ValueError(f"attempt solver score mismatch for {instance_id}")
        if matching_attempt.get("builder_score") != world_scores["world_quality"]:
            raise ValueError(f"attempt builder score mismatch for {instance_id}")
        if matching_attempt.get("builder_raw_score") != world_scores["world_quality_raw"]:
            raise ValueError(f"attempt raw builder score mismatch for {instance_id}")
        if matching_attempt.get("world_criteria") != world_scores["criteria"]:
            raise ValueError(f"attempt world criteria mismatch for {instance_id}")
        expected_hard = world_scores["hard_gates"]
        if catalog_row.get("world_hard_gates") != expected_hard:
            raise ValueError(f"catalog world hard gates mismatch for {instance_id}")
        for key in ("solver_score", "builder_score", "builder_raw_score", "world_criteria"):
            if catalog_row.get(key) != matching_attempt.get(key):
                raise ValueError(f"{key} mismatch for {instance_id}")
        if catalog_row.get("status") != matching_attempt.get("semantic_status"):
            raise ValueError(f"semantic status mismatch for {instance_id}")
        public_checks = _read_json_value(world_dir / "public_checks.json")
        if not isinstance(public_checks, list):
            raise ValueError(f"invalid public checks for {instance_id}")
        trace_records: dict[str, dict[str, Any]] = {}
        for role in ("synthesizer", "builder", "solver", "judge"):
            trace_record = _read_json(world_dir / f"{role}_trace.json")
            trace_records[role] = trace_record
            agent = trace_record.get("agent")
            if (
                not isinstance(trace_record.get("id"), str)
                or not isinstance(agent, dict)
                or agent.get("name") != role
            ):
                raise ValueError(f"{role} trace identity mismatch for {instance_id}")

        solver_trace = trace_records["solver"]
        solver_rewards = solver_trace.get("rewards")
        solver_info = solver_trace.get("info")
        if not isinstance(solver_rewards, dict) or not isinstance(solver_info, dict):
            raise ValueError(f"solver trace reward data missing for {instance_id}")
        semantic_reward = solver_rewards.get("semantic_correctness")
        if (
            not isinstance(semantic_reward, dict)
            or semantic_reward.get("score") != solver_scores["semantic_correctness"]
            or semantic_reward.get("weight") != 1.0
        ):
            raise ValueError(f"solver named reward mismatch for {instance_id}")
        if solver_info.get("solver_semantic_verdict") != solver_verdict.model_dump(mode="json"):
            raise ValueError(f"solver trace verdict mismatch for {instance_id}")
        for reward_name in (
            "required_claim_coverage",
            "contradiction_free",
            "forbidden_claim_count",
        ):
            reward = solver_rewards.get(reward_name)
            if (
                not isinstance(reward, dict)
                or reward.get("score") != solver_scores[reward_name]
                or reward.get("weight") != 0.0
            ):
                raise ValueError(f"solver diagnostic reward {reward_name!r} mismatch for {instance_id}")
        semantic_details = solver_info.get("solver_semantic_scores")
        if (
            not isinstance(semantic_details, dict)
            or semantic_details.get("semantic_correctness") != solver_scores["semantic_correctness"]
        ):
            raise ValueError(f"solver trace score details mismatch for {instance_id}")

        builder_trace = trace_records["builder"]
        builder_rewards = builder_trace.get("rewards")
        builder_info = builder_trace.get("info")
        if not isinstance(builder_rewards, dict) or not isinstance(builder_info, dict):
            raise ValueError(f"builder trace reward data missing for {instance_id}")
        world_reward = builder_rewards.get("world_quality")
        raw_reward = builder_rewards.get("world_quality_raw")
        if (
            not isinstance(world_reward, dict)
            or world_reward.get("score") != world_scores["world_quality"]
            or world_reward.get("weight") != 1.0
            or not isinstance(raw_reward, dict)
            or raw_reward.get("score") != world_scores["world_quality_raw"]
            or raw_reward.get("weight") != 0.0
            or builder_info.get("world_reward") != world_scores
        ):
            raise ValueError(f"builder named reward mismatch for {instance_id}")
        expected_builder_diagnostics = {
            "deterministic_validation": world_scores["deterministic_validation"],
            "task_unambiguous": float(world_scores["hard_gates"]["task_unambiguous"]),
            "world_supports_task": float(world_scores["hard_gates"]["world_supports_task"]),
            **world_scores["criteria"],
        }
        for reward_name, expected_score in expected_builder_diagnostics.items():
            reward = builder_rewards.get(reward_name)
            if (
                not isinstance(reward, dict)
                or reward.get("score") != expected_score
                or reward.get("weight") != 0.0
            ):
                raise ValueError(f"builder diagnostic reward {reward_name!r} mismatch for {instance_id}")

        judge_info = trace_records["judge"].get("info")
        if not isinstance(judge_info, dict) or judge_info.get("world_verdict") != world_verdict.model_dump(
            mode="json"
        ):
            raise ValueError(f"world judge trace verdict mismatch for {instance_id}")
        completed_actions = judge_info.get("completed_actions")
        if not isinstance(completed_actions, list) or not completed_actions:
            raise ValueError(f"world judge trace has no empirical Slack actions for {instance_id}")
        from .slack.tools import SlackActionRecord

        for action in completed_actions:
            SlackActionRecord.model_validate(action)
        if matching_attempt.get("world_judge_completed_action_count") != len(completed_actions):
            raise ValueError(f"world judge action count mismatch for {instance_id}")
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

    expected_artifact_paths = {
        path.relative_to(release).as_posix()
        for root_name in ("worlds", "rejected")
        for path in (release / root_name).rglob("*")
        if path.is_file()
    }
    if artifact_paths != expected_artifact_paths:
        difference = sorted(artifact_paths ^ expected_artifact_paths)
        raise ValueError(f"artifacts.jsonl and canonical files differ: {difference}")


class DatasetWriter:
    def __init__(
        self,
        output: str | Path,
        *,
        run_id: str,
        acceptance: ReleaseAcceptanceConfig,
        prime_image: str,
        rlm_revision: str,
        environment_config: Mapping[str, Any],
        target_accepted: int,
        max_attempts: int,
        concurrency: int,
        progress_path: str = "progress.jsonl",
    ) -> None:
        if not _SAFE_COMPONENT.fullmatch(run_id):
            raise ValueError(f"unsafe run_id: {run_id!r}")
        if target_accepted < 1 or max_attempts < target_accepted or concurrency < 1:
            raise ValueError("invalid target, attempt guard, or concurrency")
        if progress_path != "progress.jsonl":
            raise ValueError("progress path must be the canonical progress.jsonl")
        self.root = Path(output).expanduser().resolve()
        self.run_id = run_id
        self.acceptance = acceptance
        self.target_accepted = target_accepted
        self.max_attempts = max_attempts
        self.concurrency = concurrency
        self.progress_path = progress_path
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
        self.process_started_at = datetime.now(UTC)
        self.root.mkdir(parents=True, exist_ok=True)
        previous_summary_path = self.root / "summary.json"
        previous_summary = (
            _read_json(previous_summary_path)
            if previous_summary_path.is_file() and not previous_summary_path.is_symlink()
            else {}
        )
        previous_active = previous_summary.get("active_wall_time_seconds", 0.0)
        self.prior_active_seconds = (
            float(previous_active)
            if isinstance(previous_active, (int, float)) and not isinstance(previous_active, bool)
            else 0.0
        )
        existing_manifest = self.root / "run_manifest.json"
        _reject_symlink(existing_manifest, "run manifest")
        if existing_manifest.is_file() and (
            _read_json(existing_manifest).get("generator_schema_version") != GENERATOR_SCHEMA_VERSION
        ):
            raise ValueError("release uses an incompatible dataset schema; create a new v3 output directory")
        worlds = self.root / "worlds"
        _reject_symlink(worlds, "worlds directory")
        worlds.mkdir(exist_ok=True)
        for name in (*JSONL_FILES, self.progress_path):
            path = self.root / name
            _reject_symlink(path, name)
            path.touch(exist_ok=True)
        self._recover_transactions()
        self.interface_hashes = self._install_interface()
        self._install_viewer_manifest()
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
            if isinstance(row.get("generation_seed"), int)
        }

    @property
    def attempted_count(self) -> int:
        return len(_read_jsonl(self.attempts_path))

    @property
    def accepted_count(self) -> int:
        return sum(bool(row.get("written_to_dataset")) for row in _read_jsonl(self.attempts_path))

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
        artifacts = {
            "interface.json": (source / "interface.json").read_bytes(),
            "slack_world.schema.json": _dumps(SlackWorld.model_json_schema(), pretty=True),
            "task_contract.schema.json": _dumps(TaskContract.model_json_schema(), pretty=True),
            "builder_guide.md": (source / "builder_guide.md").read_bytes(),
            "world_rubric.toml": (source / "world_rubric.toml").read_bytes(),
        }
        hashes: dict[str, str] = {}
        for name, data in artifacts.items():
            target = destination / name
            _reject_symlink(target, f"interface file {name}")
            content = data if isinstance(data, bytes) else data.encode()
            if target.exists() and target.read_bytes() != content:
                raise ValueError(f"existing interface artifact differs: {target}")
            if not target.exists():
                _atomic_write(target, content)
            hashes[name] = _sha256(content)
        return hashes

    def _install_viewer_manifest(self) -> None:
        manifest = {
            "schema_version": RELEASE_TABLE_SCHEMA_VERSION,
            "tables": {
                "attempts": {"path": "attempts.jsonl", "format": "jsonl"},
                "artifacts": {"path": "artifacts.jsonl", "format": "jsonl"},
                "progress": {"path": "progress.jsonl", "format": "jsonl"},
            },
        }
        path = self.root / "manifest.json"
        _reject_symlink(path, "viewer manifest")
        if path.exists() and _read_json(path) != manifest:
            raise ValueError("release manifest differs from the supported viewer contract")
        if not path.exists():
            _atomic_write(path, _dumps(manifest, pretty=True))

    def _install_run_manifest(self) -> None:
        path = self.root / "run_manifest.json"
        _reject_symlink(path, "run manifest")
        existing = _read_json(path) if path.exists() else None
        created_at = existing.get("created_at") if existing is not None else datetime.now(UTC).isoformat()
        manifest = {
            "schema_version": 2,
            "generator_schema_version": GENERATOR_SCHEMA_VERSION,
            "interface_id": INTERFACE_ID,
            "run_id": self.run_id,
            "created_at": created_at,
            "worldgen_version": self.worldgen_version,
            "verifiers_version": self.verifiers_version,
            "pydantic_version": self.pydantic_version,
            "generator_source_hash": self.generator_source_hash,
            "prime_image": self.prime_image,
            "rlm_revision": self.rlm_revision,
            "interface_hashes": self.interface_hashes,
            "world_rubric_hash": self.interface_hashes["world_rubric.toml"],
            "environment_config_hash": self.environment_config_hash,
            "environment_config": self.environment_config,
            "acceptance": self.acceptance.model_dump(mode="json"),
            "target_accepted": self.target_accepted,
            "max_attempts": self.max_attempts,
            "concurrency": self.concurrency,
            "progress_path": self.progress_path,
        }
        if existing is not None:
            if existing != _jsonable(manifest):
                raise ValueError("output directory belongs to a different immutable run config")
        else:
            has_existing_attempts = bool(_read_jsonl(self.root / "attempts.jsonl"))
            has_existing_worlds = any((self.root / "worlds").iterdir())
            if has_existing_attempts or has_existing_worlds:
                raise ValueError("existing output has no immutable run_manifest.json")
            _atomic_write(path, _dumps(manifest, pretty=True))
        self.run_started_at = datetime.fromisoformat(str(created_at))

    def _validate_resume(self) -> None:
        validate_release_integrity(self.root, expected_run_id=self.run_id)

    def _load_signature_index(self) -> None:
        self._signatures: dict[str, str] = {}
        self._signature_seeds: dict[str, int] = {}
        self._committed_signatures: set[str] = set()
        for row in _read_jsonl(self.attempts_path):
            if row.get("item_status") == ItemStatus.DUPLICATE.value:
                continue
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

    @staticmethod
    def identity(item: SynthesizedItem) -> tuple[str, str]:
        instance_id, contract_hash = item_identity(item)
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

    @staticmethod
    def _role_stats(traces: Sequence[vf.Trace], role: str) -> dict[str, Any]:
        selected = [trace for trace in traces if trace.agent and trace.agent.name == role]
        discarded = [
            record
            for trace in selected
            for record in trace.info.get("discarded_attempt_usage", [])
            if isinstance(record, dict)
        ]
        usages = [usage for trace in selected if (usage := getattr(trace, "usage", None))]
        usages.extend(
            vf.Usage.model_validate(record["usage"])
            for record in discarded
            if isinstance(record.get("usage"), dict)
        )
        usage = vf.Usage.aggregate(usages)
        auxiliary_values = [item for trace in selected for item in getattr(trace, "extra_usage", [])]
        auxiliary_values.extend(
            vf.Usage.model_validate(item)
            for record in discarded
            for item in record.get("extra_usage", [])
            if isinstance(item, dict)
        )
        auxiliary = vf.Usage.aggregate(auxiliary_values)
        seconds = 0.0
        for trace in selected:
            timing = getattr(trace, "timing", None)
            if timing is None:
                continue
            seconds += sum(
                float(getattr(getattr(timing, name, None), "duration", 0.0))
                for name in ("boot", "setup", "agent", "finalize", "scoring")
            )
        seconds += sum(
            float(record.get("duration_ms", 0)) / 1_000
            for record in discarded
            if isinstance(record.get("duration_ms"), (int, float))
        )
        models = {trace.agent.config.model for trace in selected if trace.agent and trace.agent.config.model}
        models.update(record["model"] for record in discarded if isinstance(record.get("model"), str))
        return {
            "tokens": int(usage.total_tokens or 0)
            if usage is not None
            else sum(int(getattr(trace, "num_total_tokens", 0)) for trace in selected),
            "input_tokens": usage.input_tokens if usage is not None else None,
            "output_tokens": usage.completion_tokens if usage is not None else None,
            "cached_input_tokens": usage.cached_input_tokens if usage is not None else None,
            "reasoning_tokens": usage.reasoning_tokens if usage is not None else None,
            "cost": usage.cost if usage is not None else None,
            "auxiliary_tokens": int(auxiliary.total_tokens or 0) if auxiliary is not None else 0,
            "auxiliary_cost": auxiliary.cost if auxiliary is not None else None,
            "duration_ms": int(round(seconds * 1_000)),
            "model": next(iter(models)) if len(models) == 1 else None,
            "retry_count": sum(
                int(trace.info.get("retry_count", 0))
                for trace in selected
                if isinstance(trace.info.get("retry_count", 0), int)
            ),
        }

    def _artifact_rows(
        self,
        *,
        attempt_id: str,
        instance_id: str | None,
        artifact_ref: str | None,
    ) -> list[dict[str, Any]]:
        if artifact_ref is None:
            return []
        directory = _inside(self.root, artifact_ref)
        rows = []
        for path in sorted(directory.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(self.root).as_posix()
            rows.append(
                {
                    "schema_version": RELEASE_TABLE_SCHEMA_VERSION,
                    "run_id": self.run_id,
                    "attempt_id": attempt_id,
                    "instance_id": instance_id,
                    "path": relative,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "size_bytes": path.stat().st_size,
                }
            )
        return rows

    def write_failed_episode(
        self,
        *,
        generation_seed: int,
        episode: vf.Episode,
        reason: str,
        failed_stage: FailedStage,
        failure_kind: FailureKind,
        failure_owner: FailureOwner,
        traces: Sequence[vf.Trace] = (),
    ) -> dict[str, Any]:
        diagnostic_traces = list(traces) or list(episode.traces)
        errors = [*episode.errors]
        for trace in diagnostic_traces:
            errors.extend(trace.errors)
        status = (
            ItemStatus.INFRASTRUCTURE_ERROR
            if failure_owner == FailureOwner.INFRASTRUCTURE
            else ItemStatus.PROTOCOL_ERROR
        )
        result = GenerationResult(
            generation_seed=generation_seed,
            status=status,
            failure_owner=failure_owner,
            failed_stage=failed_stage,
            failure_kind=failure_kind,
            reason=str(redact_secrets(reason))[:4_000],
        )
        return self.write_attempt(
            result,
            signature_text=None,
            source=None,
            traces=diagnostic_traces,
            builder_metadata={
                "episode_id": episode.id,
                "episode_errors": [
                    {
                        "type": error.type,
                        "message": str(redact_secrets(error.message))[:1_000],
                    }
                    for error in errors[:20]
                ],
            },
        )

    def write_attempt(
        self,
        result: GenerationResult,
        *,
        signature_text: str | None,
        source: str | None,
        traces: Sequence[vf.Trace],
        public_checks: Sequence[Mapping[str, Any]] = (),
        builder_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        result = GenerationResult.model_validate_json(result.model_dump_json())
        if result.synthesized is not None and result.signature != synthesized_signature(result.synthesized):
            raise ValueError("persisted signature does not match the synthesized item")
        with self._lock:
            attempt_id = self._next_attempt_id(result.generation_seed)
            trace_ids: dict[str, list[str]] = {}
            for trace in traces:
                role = trace.agent.name if trace.agent and trace.agent.name else "unknown"
                trace_ids.setdefault(role, []).append(trace.id)
            if result.trace_ids and result.trace_ids != trace_ids:
                raise ValueError("generation result trace IDs do not match persisted traces")
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
                elif self.acceptance.retain_rejected_artifacts:
                    artifact_ref, trace_refs = self._write_rejected(
                        attempt_id,
                        result,
                        source=source,
                        traces=traces,
                        public_checks=public_checks,
                        builder_metadata=builder_metadata or {},
                    )

                attempt_reason = str(redact_secrets(result.reason))[:4_000] if result.reason else None
                role_stats = {role: self._role_stats(traces, role) for role in ROLE_NAMES}
                synthesis_ok = result.synthesized is not None or any(
                    trace.agent
                    and trace.agent.name == "synthesizer"
                    and isinstance(trace.info.get("synthesized_item"), dict)
                    for trace in traces
                )
                validation_ok = (
                    result.validation.ok
                    if result.validation is not None
                    else any(
                        trace.agent
                        and trace.agent.name == "builder"
                        and isinstance(trace.info.get("worldgen_validation"), dict)
                        and trace.info["worldgen_validation"].get("ok") is True
                        for trace in traces
                    )
                )
                accepted_base = artifact_ref if written else None
                row = {
                    "schema_version": RELEASE_TABLE_SCHEMA_VERSION,
                    "attempt_id": attempt_id,
                    "run_id": self.run_id,
                    "created_at": datetime.now(UTC).isoformat(),
                    "generation_seed": result.generation_seed,
                    "instance_id": result.instance_id,
                    "status": "accepted" if written else "rejected",
                    "item_status": result.status.value,
                    "failure_owner": result.failure_owner.value,
                    "failed_stage": result.failed_stage.value if result.failed_stage else None,
                    "failure_kind": result.failure_kind.value if result.failure_kind else None,
                    "synthesis_ok": synthesis_ok,
                    "validation_ok": validation_ok,
                    "solver_score": result.solver_score,
                    "solver_metrics": result.solver_metrics,
                    "builder_score": result.builder_score,
                    "builder_raw_score": result.builder_raw_score,
                    "task_unambiguous": (
                        result.world_hard_gates["task_unambiguous"] if result.world_hard_gates else None
                    ),
                    "world_supports_task": (
                        result.world_hard_gates["world_supports_task"] if result.world_hard_gates else None
                    ),
                    "world_criteria": result.world_criteria,
                    "scenario_path": f"{accepted_base}/scenario.json" if accepted_base else None,
                    "contract_path": f"{accepted_base}/task_contract.json" if accepted_base else None,
                    "source_path": (
                        f"{artifact_ref}/world.py"
                        if artifact_ref and (self.root / artifact_ref / "world.py").is_file()
                        else None
                    ),
                    "snapshot_path": (
                        f"{artifact_ref}/snapshot.json"
                        if artifact_ref and (self.root / artifact_ref / "snapshot.json").is_file()
                        else None
                    ),
                    "validation_path": f"{accepted_base}/validation.json" if accepted_base else None,
                    "solver_trace_path": (
                        trace_refs.get("solver", [None])[-1] if trace_refs.get("solver") else None
                    ),
                    "world_judge_trace_path": (
                        trace_refs.get("judge", [None])[-1] if trace_refs.get("judge") else None
                    ),
                    "solver_verdict_path": (
                        f"{accepted_base}/solver_verdict.json" if accepted_base else None
                    ),
                    "world_verdict_path": (f"{accepted_base}/world_verdict.json" if accepted_base else None),
                    "progress_path": self.progress_path,
                    "reason": attempt_reason,
                    "signature": result.signature,
                    "signature_text": signature_text,
                    "semantic_status": result.decision.status if result.decision else None,
                    "criterion_failures": result.decision.criterion_failures if result.decision else [],
                    "rejection_reason": (
                        result.decision.rejection_reason
                        if result.decision
                        else result.failure_kind.value
                        if result.failure_kind
                        else None
                    ),
                    "written_to_dataset": written,
                    "artifact_ref": artifact_ref,
                    "builder_turns": int((builder_metadata or {}).get("turns", 0)),
                    "solver_completed_action_count": (
                        len(result.solver.completed_actions) if result.solver else 0
                    ),
                    "world_judge_retry_count": role_stats["judge"]["retry_count"],
                    "whole_episode_retry_count": 0,
                    "retry_counts": {role: role_stats[role]["retry_count"] for role in ROLE_NAMES},
                    "models": {role: role_stats[role]["model"] for role in ROLE_NAMES},
                    "role_outcomes": {
                        role: {
                            "present": any(trace.agent and trace.agent.name == role for trace in traces),
                            "ok": any(
                                trace.agent and trace.agent.name == role and trace.ok for trace in traces
                            ),
                        }
                        for role in ROLE_NAMES
                    },
                    "usage_by_role": role_stats,
                    "solver_judge_tokens": role_stats["solver"]["auxiliary_tokens"],
                    "solver_judge_cost": role_stats["solver"]["auxiliary_cost"],
                    "solver_judge_model": next(
                        (
                            trace.info.get("solver_answer_judge_model")
                            for trace in traces
                            if trace.agent
                            and trace.agent.name == "solver"
                            and isinstance(trace.info.get("solver_answer_judge_model"), str)
                        ),
                        None,
                    ),
                    "world_judge_completed_action_count": next(
                        (
                            len(trace.info.get("completed_actions", []))
                            for trace in traces
                            if trace.agent
                            and trace.agent.name == "judge"
                            and isinstance(trace.info.get("completed_actions"), list)
                        ),
                        0,
                    ),
                }
                for role in ROLE_NAMES:
                    row[f"{role}_tokens"] = role_stats[role]["tokens"]
                    row[f"{role}_duration_ms"] = role_stats[role]["duration_ms"]
                    row[f"{role}_cost"] = role_stats[role]["cost"]
                artifact_rows = self._artifact_rows(
                    attempt_id=attempt_id,
                    instance_id=result.instance_id,
                    artifact_ref=artifact_ref,
                )
                for artifact_row in artifact_rows:
                    _append_jsonl(self.root / "artifacts.jsonl", artifact_row)

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
                result.solver.semantic_verdict if result.solver else None,
                result.world_verdict,
                result.decision,
                source,
            )
        ):
            raise ValueError("accepted item is missing required artifacts")
        item = result.synthesized
        validation = result.validation
        verdict = result.world_verdict
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
            "schema_version": 2,
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
            "world_rubric_hash": self.interface_hashes["world_rubric.toml"],
            "acceptance": self.acceptance.model_dump(mode="json"),
            "prime_image": self.prime_image,
            "rlm_revision": self.rlm_revision,
            "solver_score": result.solver_score,
            "solver_metrics": result.solver_metrics,
            "builder_score": result.builder_score,
            "builder_raw_score": result.builder_raw_score,
            "world_criteria": result.world_criteria,
            "world_hard_gates": result.world_hard_gates,
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
                "solver_verdict.json": _dumps(result.solver.semantic_verdict, pretty=True),
                "world_verdict.json": _dumps(verdict, pretty=True),
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
            "solver_score": result.solver_score,
            "builder_score": result.builder_score,
            "builder_raw_score": result.builder_raw_score,
            "world_criteria": result.world_criteria,
            "world_hard_gates": result.world_hard_gates,
            "world_ref": f"{base}/world.py",
            "snapshot_ref": f"{base}/snapshot.json",
            "oracle_ref": f"{base}/task_contract.json",
            "solver_trace_ref": f"{base}/solver_trace.json",
            "world_judge_trace_ref": f"{base}/judge_trace.json",
            "solver_verdict_ref": f"{base}/solver_verdict.json",
            "world_verdict_ref": f"{base}/world_verdict.json",
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

    def write_summary(
        self,
        *,
        exit_reason: str | None = None,
        release_integrity_ok: bool | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            attempts = _read_jsonl(self.attempts_path)
            written = [row for row in attempts if row.get("written_to_dataset")]
            statuses = Counter(str(row.get("item_status")) for row in attempts)
            owners = Counter(str(row.get("failure_owner")) for row in attempts)
            stages = Counter(str(row.get("failed_stage")) for row in attempts if row.get("failed_stage"))
            kinds = Counter(str(row.get("failure_kind")) for row in attempts if row.get("failure_kind"))
            semantic = Counter(
                str(row.get("semantic_status")) for row in attempts if row.get("semantic_status")
            )
            floor_counts: Counter[str] = Counter()
            for row in attempts:
                floor_counts.update(row.get("criterion_failures") or [])

            role_usage: dict[str, Any] = {}
            reported_costs: list[float] = []
            for role in ROLE_NAMES:
                durations = [
                    float(row[f"{role}_duration_ms"])
                    for row in attempts
                    if isinstance(row.get(f"{role}_duration_ms"), (int, float))
                    and row[f"{role}_duration_ms"] > 0
                ]
                costs = [
                    float(row[f"{role}_cost"])
                    for row in attempts
                    if isinstance(row.get(f"{role}_cost"), (int, float))
                ]
                reported_costs.extend(costs)
                role_usage[role] = {
                    "tokens": sum(int(row.get(f"{role}_tokens", 0)) for row in attempts),
                    "cost": sum(costs) if costs else None,
                    "duration_ms": _distribution(durations),
                    "models": sorted(
                        {
                            model
                            for row in attempts
                            if isinstance((model := (row.get("models") or {}).get(role)), str)
                        }
                    ),
                    "retries": sum(int((row.get("retry_counts") or {}).get(role, 0)) for row in attempts),
                }
            role_yield: dict[str, Any] = {}
            for role in ROLE_NAMES:
                started = sum(
                    bool((row.get("role_outcomes") or {}).get(role, {}).get("present")) for row in attempts
                )
                succeeded = sum(
                    bool((row.get("role_outcomes") or {}).get(role, {}).get("ok")) for row in attempts
                )
                role_yield[role] = {
                    "started": started,
                    "succeeded": succeeded,
                    "yield": succeeded / started if started else None,
                }

            solver_judge_costs = [
                float(row["solver_judge_cost"])
                for row in attempts
                if isinstance(row.get("solver_judge_cost"), (int, float))
            ]
            reported_costs.extend(solver_judge_costs)
            solver_judge = {
                "tokens": sum(int(row.get("solver_judge_tokens", 0)) for row in attempts),
                "cost": sum(solver_judge_costs) if solver_judge_costs else None,
                "models": sorted(
                    {
                        row["solver_judge_model"]
                        for row in attempts
                        if isinstance(row.get("solver_judge_model"), str)
                    }
                ),
            }

            progress = _read_jsonl(self.root / self.progress_path)
            progress_times: list[datetime] = []
            for event in progress:
                timestamp = event.get("timestamp_utc")
                if not isinstance(timestamp, str):
                    raise ValueError("progress event has an invalid timestamp_utc")
                try:
                    progress_times.append(datetime.fromisoformat(timestamp.replace("Z", "+00:00")))
                except ValueError as exc:
                    raise ValueError("progress event has an invalid timestamp_utc") from exc
            progress_gaps = [
                (right - left).total_seconds()
                for left, right in zip(progress_times, progress_times[1:], strict=False)
            ]
            max_progress_gap = max(progress_gaps, default=0.0) if progress_times else None
            overlap_by_seed: dict[int, bool] = {}
            for row in written:
                seed = row.get("generation_seed")
                if not isinstance(seed, int):
                    continue
                role_events = {
                    (event.get("stage"), event.get("event")): event.get("timestamp_utc")
                    for event in progress
                    if event.get("seed") == seed
                    and event.get("stage") in {"solver", "world_judge"}
                    and event.get("event") in {"started", "finished"}
                }
                try:
                    solver_start = datetime.fromisoformat(
                        str(role_events[("solver", "started")]).replace("Z", "+00:00")
                    )
                    solver_end = datetime.fromisoformat(
                        str(role_events[("solver", "finished")]).replace("Z", "+00:00")
                    )
                    judge_start = datetime.fromisoformat(
                        str(role_events[("world_judge", "started")]).replace("Z", "+00:00")
                    )
                    judge_end = datetime.fromisoformat(
                        str(role_events[("world_judge", "finished")]).replace("Z", "+00:00")
                    )
                    overlap_by_seed[seed] = max(solver_start, judge_start) <= min(solver_end, judge_end)
                except (KeyError, ValueError):
                    overlap_by_seed[seed] = False
            stage_values: dict[str, list[float]] = {}
            for event in progress:
                duration = event.get("duration_ms")
                if event.get("event") == "finished" and isinstance(duration, (int, float)):
                    stage_values.setdefault(str(event.get("stage")), []).append(float(duration))
            criterion_distributions = {
                criterion: _distribution(
                    [
                        float(row["world_criteria"][criterion])
                        for row in attempts
                        if isinstance(row.get("world_criteria"), dict)
                    ]
                )
                for criterion in QUALITY_CRITERIA
            }
            solver_scores = [
                float(row["solver_score"])
                for row in attempts
                if isinstance(row.get("solver_score"), (int, float))
            ]
            builder_scores = [
                float(row["builder_score"])
                for row in attempts
                if isinstance(row.get("builder_score"), (int, float))
            ]
            total_cost = sum(reported_costs) if reported_costs else None
            now = datetime.now(UTC)
            elapsed = max(0.0, (now - self.run_started_at).total_seconds())
            active_elapsed = self.prior_active_seconds + max(
                0.0, (now - self.process_started_at).total_seconds()
            )
            summary = {
                "run_id": self.run_id,
                "output": str(self.root),
                "target_accepted": self.target_accepted,
                "actual_accepted": len(written),
                "written_to_dataset": len(written),
                "attempted": len(attempts),
                "qualification_passed": False,
                "exit_reason": exit_reason,
                "attempt_cap_exhausted": (
                    len(attempts) >= self.max_attempts and len(written) < self.target_accepted
                ),
                "max_attempts": self.max_attempts,
                "concurrency": self.concurrency,
                "wall_time_seconds": elapsed,
                "active_wall_time_seconds": active_elapsed,
                "accepted_worlds_per_hour": (
                    len(written) / (active_elapsed / 3_600) if active_elapsed > 0 else None
                ),
                "attempts_per_accepted_world": len(attempts) / len(written) if written else None,
                "item_status_counts": dict(sorted(statuses.items())),
                "semantic_status_counts": dict(sorted(semantic.items())),
                "failure_owner_counts": dict(sorted(owners.items())),
                "rejections_by_stage": dict(sorted(stages.items())),
                "failure_kind_counts": dict(sorted(kinds.items())),
                "criterion_floor_counts": dict(sorted(floor_counts.items())),
                "duplicate_contracts_rejected": statuses[ItemStatus.DUPLICATE.value],
                "synthesis_yield": (
                    sum(row.get("synthesis_ok") is True for row in attempts) / len(attempts)
                    if attempts
                    else None
                ),
                "validation_yield": (
                    sum(row.get("validation_ok") is True for row in attempts)
                    / sum(row.get("synthesis_ok") is True for row in attempts)
                    if any(row.get("synthesis_ok") is True for row in attempts)
                    else None
                ),
                "solver_score_distribution": _distribution(solver_scores),
                "builder_score_distribution": _distribution(builder_scores),
                "world_criterion_distributions": criterion_distributions,
                "role_usage": role_usage,
                "role_yield": role_yield,
                "solver_answer_judge_usage": solver_judge,
                "total_reported_cost": total_cost,
                "cost_per_accepted_world": total_cost / len(written)
                if total_cost is not None and written
                else None,
                "stage_duration_ms": {
                    stage: _distribution(values) for stage, values in sorted(stage_values.items())
                },
                "max_progress_gap_seconds": max_progress_gap,
                "solver_world_judge_overlap": {
                    str(seed): value for seed, value in sorted(overlap_by_seed.items())
                },
                "release_integrity_ok": release_integrity_ok,
                "retry_counts": {
                    "whole_episode": sum(int(row.get("whole_episode_retry_count", 0)) for row in attempts),
                    "world_judge": sum(int(row.get("world_judge_retry_count", 0)) for row in attempts),
                },
                "acceptance": self.acceptance.model_dump(mode="json"),
                "prime_image": self.prime_image,
                "rlm_revision": self.rlm_revision,
                "interface_hashes": self.interface_hashes,
                "progress_path": self.progress_path,
            }
            accepted_signatures = [row.get("signature") for row in written]
            end_conditions = {
                "accepted_target_reached": len(written) >= self.target_accepted,
                "attempt_cap_respected": len(attempts) <= self.max_attempts,
                "accepted_rows_are_solved": all(row.get("semantic_status") == "solved" for row in written),
                "solver_threshold_met": all(
                    isinstance(row.get("solver_score"), (int, float))
                    and row["solver_score"] >= self.acceptance.min_solver_score
                    for row in written
                ),
                "builder_threshold_met": all(
                    isinstance(row.get("builder_score"), (int, float))
                    and row["builder_score"] >= self.acceptance.min_world_score
                    for row in written
                ),
                "world_hard_gates_met": all(
                    row.get("task_unambiguous") is True and row.get("world_supports_task") is True
                    for row in written
                ),
                "criterion_floors_met": all(
                    isinstance(row.get("world_criteria"), dict)
                    and all(
                        row["world_criteria"].get(name, -1) >= floor
                        for name, floor in self.acceptance.minimum_world_scores.items()
                    )
                    for row in written
                ),
                "accepted_signatures_unique": (
                    len(accepted_signatures) == len(set(accepted_signatures)) and all(accepted_signatures)
                ),
                "whole_episode_retries_zero": all(
                    row.get("whole_episode_retry_count") == 0 for row in attempts
                ),
                "structured_solver_verdicts_present": all(
                    isinstance(row.get("solver_verdict_path"), str) for row in written
                ),
                "empirical_world_judge_actions_present": all(
                    isinstance(row.get("world_verdict_path"), str)
                    and row.get("world_judge_completed_action_count", 0) > 0
                    for row in written
                ),
                "solver_world_judge_overlapped": (
                    len(overlap_by_seed) == len(written) and all(overlap_by_seed.values())
                ),
                "active_wall_time_at_most_20_minutes": active_elapsed <= 1_200,
                "progress_heartbeat_within_interval": (
                    bool(progress_times) and max_progress_gap is not None and max_progress_gap <= 10.0
                ),
                "release_integrity_passed": release_integrity_ok is True,
            }
            summary["qualification_passed"] = all(end_conditions.values())
            _atomic_write(self.root / "summary.json", _dumps(summary, pretty=True))
            report = {
                "schema_version": 1,
                "run_id": self.run_id,
                "generated_at": datetime.now(UTC).isoformat(),
                "passed": all(end_conditions.values()),
                "target_accepted": self.target_accepted,
                "actual_accepted": len(written),
                "attempted": len(attempts),
                "max_attempts": self.max_attempts,
                "exit_reason": exit_reason,
                "end_conditions": end_conditions,
                "solver_score_distribution": summary["solver_score_distribution"],
                "builder_score_distribution": summary["builder_score_distribution"],
                "world_criterion_distributions": summary["world_criterion_distributions"],
                "role_usage": role_usage,
                "role_yield": role_yield,
                "solver_answer_judge_usage": solver_judge,
                "rejections_by_stage": summary["rejections_by_stage"],
                "failure_kind_counts": summary["failure_kind_counts"],
                "synthesis_yield": summary["synthesis_yield"],
                "validation_yield": summary["validation_yield"],
                "duplicate_contracts_rejected": summary["duplicate_contracts_rejected"],
                "attempts_per_accepted_world": summary["attempts_per_accepted_world"],
                "accepted_worlds_per_hour": summary["accepted_worlds_per_hour"],
                "concurrency": self.concurrency,
                "stage_duration_ms": summary["stage_duration_ms"],
                "retry_counts": summary["retry_counts"],
                "max_progress_gap_seconds": max_progress_gap,
                "solver_world_judge_overlap": summary["solver_world_judge_overlap"],
                "total_reported_cost": total_cost,
                "cost_per_accepted_world": summary["cost_per_accepted_world"],
                "wall_time_seconds": elapsed,
                "active_wall_time_seconds": active_elapsed,
                "acceptance": self.acceptance.model_dump(mode="json"),
                "generator_source_hash": self.generator_source_hash,
                "interface_hashes": self.interface_hashes,
                "world_rubric_hash": self.interface_hashes["world_rubric.toml"],
            }
            _atomic_write(self.root / "qualification_report.json", _dumps(report, pretty=True))
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
        "p95": _percentile(values, 0.95),
    }


__all__ = [
    "DatasetWriter",
    "synthesized_signature",
    "synthesized_signature_text",
    "validate_release_integrity",
]
