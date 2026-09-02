from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import verifiers.v1 as vf
from pydantic import Field

from ..agents.solver import SolverConfig, SolverTask
from ..dataset import validate_release_integrity
from ..slack.models import INTERFACE_ID, SlackWorld, TaskContract, validate_safe_identifier
from ..slack.validate import validate_world

_PUBLIC_FIELDS = {"instance_id", "scenario", "question", "snapshot_ref", "interface_id"}
_PRIVATE_FIELDS = {
    "instance_id",
    "answer",
    "required_claims",
    "forbidden_claims",
    "task_contract_ref",
    "gold_ref",
}


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value!r} is not supported")


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    except (OSError, ValueError, UnicodeError) as exc:
        raise ValueError(f"could not read {label} JSON at {path}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"{label} JSON at {path} must be an object")
    return value


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"{label} JSONL does not exist: {path}")
    rows: list[dict[str, Any]] = []
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw, parse_constant=_reject_constant)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"invalid {label} JSONL line {number} in {path}") from exc
        if not isinstance(row, dict):
            raise TypeError(f"{label} JSONL line {number} must be an object")
        rows.append(row)
    if not rows:
        raise ValueError(f"{label} JSONL is empty: {path}")
    return rows


def _inside(root: Path, reference: str) -> Path:
    if not isinstance(reference, str) or not reference.strip():
        raise ValueError("dataset reference must be a nonempty relative path")
    relative = Path(reference)
    if relative.is_absolute():
        raise ValueError(f"dataset reference must be relative: {reference!r}")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"dataset reference escapes release root: {reference!r}") from exc
    return resolved


def _index(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for number, row in enumerate(rows, 1):
        try:
            instance_id = validate_safe_identifier(row["instance_id"], "instance_id")
        except KeyError as exc:
            raise ValueError(f"{label} row {number} has no instance_id") from exc
        if instance_id in output:
            raise ValueError(f"duplicate instance_id {instance_id!r} in {label}")
        output[instance_id] = row
    return output


def _canonical_contract_hash(contract: TaskContract) -> str:
    raw = json.dumps(
        contract.model_dump(mode="json"),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return "sha256:" + hashlib.sha256(raw).hexdigest()


class GeneratedSlackTasksetConfig(vf.TasksetConfig):
    release_dir: Path = Path("data/slack-qualification-v1")
    task: SolverConfig = Field(default_factory=SolverConfig)


class GeneratedSlackTaskset(vf.Taskset[SolverTask, GeneratedSlackTasksetConfig]):
    def load(self) -> list[SolverTask]:
        root = self.config.release_dir.expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"release directory does not exist: {root}")
        validate_release_integrity(root, require_tasks=True)
        public_rows = _read_jsonl(root / "public_tasks.jsonl", "public tasks")
        private_rows = _read_jsonl(root / "private_oracles.jsonl", "private oracles")
        public = _index(public_rows, "public tasks")
        private = _index(private_rows, "private oracles")
        if set(public) != set(private):
            raise ValueError("public and private instance IDs are not one-to-one")

        tasks: list[SolverTask] = []
        for index, row in enumerate(public_rows):
            instance_id = row["instance_id"]
            if set(row) != _PUBLIC_FIELDS:
                raise ValueError(
                    f"public row {instance_id!r} has missing or private fields: "
                    f"{sorted(set(row) ^ _PUBLIC_FIELDS)}"
                )
            oracle = private[instance_id]
            if set(oracle) != _PRIVATE_FIELDS:
                raise ValueError(
                    f"private row {instance_id!r} has unexpected fields: "
                    f"{sorted(set(oracle) ^ _PRIVATE_FIELDS)}"
                )
            if row["interface_id"] != INTERFACE_ID:
                raise ValueError(f"task {instance_id!r} requests an unsupported interface")
            expected_snapshot = f"worlds/{instance_id}/snapshot.json"
            expected_contract = f"worlds/{instance_id}/task_contract.json"
            expected_gold = f"worlds/{instance_id}/gold.json"
            if row["snapshot_ref"] != expected_snapshot:
                raise ValueError(f"task {instance_id!r} references another world snapshot")
            if oracle["task_contract_ref"] != expected_contract:
                raise ValueError(f"task {instance_id!r} references another task contract")
            if oracle["gold_ref"] != expected_gold:
                raise ValueError(f"task {instance_id!r} references another gold file")

            snapshot_path = _inside(root, row["snapshot_ref"])
            contract_path = _inside(root, oracle["task_contract_ref"])
            gold_path = _inside(root, oracle["gold_ref"])
            manifest_path = snapshot_path.parent / "manifest.json"
            for path in (snapshot_path, contract_path, gold_path, manifest_path):
                if not path.is_file():
                    raise ValueError(f"task {instance_id!r} has a dangling reference: {path}")
            world = SlackWorld.model_validate(_read_json(snapshot_path, "world snapshot"))
            contract = TaskContract.model_validate(_read_json(contract_path, "task contract"))
            manifest = _read_json(manifest_path, "world manifest")
            snapshot_hash = "sha256:" + hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
            if snapshot_hash != manifest.get("snapshot_hash"):
                raise ValueError(f"snapshot hash mismatch for {instance_id!r}")
            if _canonical_contract_hash(contract) != manifest.get("contract_hash"):
                raise ValueError(f"contract hash mismatch for {instance_id!r}")
            if manifest.get("instance_id") != instance_id:
                raise ValueError(f"manifest identity mismatch for {instance_id!r}")
            if manifest.get("interface_id") != INTERFACE_ID:
                raise ValueError(f"manifest interface mismatch for {instance_id!r}")
            if contract.question != row["question"]:
                raise ValueError(f"public/private question mismatch for {instance_id!r}")
            if contract.answer.canonical_answer != oracle["answer"]:
                raise ValueError(f"private answer mismatch for {instance_id!r}")
            if contract.answer.required_claims != oracle["required_claims"]:
                raise ValueError(f"private required claims mismatch for {instance_id!r}")
            if contract.answer.forbidden_claims != oracle["forbidden_claims"]:
                raise ValueError(f"private forbidden claims mismatch for {instance_id!r}")
            report = validate_world(world, contract)
            if not report["ok"]:
                failures = [item["detail"] for item in report["checks"] if not item["ok"]]
                raise ValueError(
                    f"accepted snapshot {instance_id!r} fails deterministic validation: "
                    + "; ".join(failures[:5])
                )
            tasks.append(
                SolverTask.from_snapshot(
                    instance_id=instance_id,
                    contract=contract,
                    world=world,
                    answer_judge=self.config.task.answer_judge,
                    idx=index,
                )
            )
        return tasks


__all__ = ["GeneratedSlackTaskset", "GeneratedSlackTasksetConfig"]
