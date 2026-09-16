"""One shared snapshot, public questions, and private answer references."""

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Literal
from .slack.models import SafeId, SlackWorld, StrictModel, AnswerSpec
from .slack.api import INTERFACE_ID, canonical, digest


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".pending-")
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(canonical(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path: Path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f"invalid JSON constant: {value}")

    return json.loads(path.read_bytes(), object_pairs_hook=unique, parse_constant=invalid)


class PublicTask(StrictModel):
    task_id: SafeId
    workspace_id: SafeId
    question: str
    actor_id: SafeId
    interface_id: Literal["slack.readonly.v2"] = INTERFACE_ID
    snapshot_ref: str
    snapshot_hash: str


class PrivateAnswer(StrictModel):
    answer: AnswerSpec
    message_ids: list[SafeId]
    user_ids: list[SafeId]


class Manifest(StrictModel):
    format: Literal["worldgen-slack.v1"] = "worldgen-slack.v1"
    workspace_id: SafeId
    snapshot_hash: str
    files: dict[str, str]


def load_release(root: Path):
    from .slack.api import SlackAPI

    root = root.resolve()
    manifest = Manifest.model_validate_json((root / "manifest.json").read_bytes())
    if set(manifest.files) != {"tasks.json", "snapshot.json", "answers.json"}:
        raise ValueError("release must contain public tasks, snapshot, and private answers")
    for name, expected in manifest.files.items():
        path = root / name
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"release artifact hash mismatch: {name}")
    world = SlackWorld.model_validate_json((root / "snapshot.json").read_bytes())
    if digest(world.model_dump(mode="json")) != manifest.snapshot_hash:
        raise ValueError("snapshot identity mismatch")
    rows = [PublicTask.model_validate(row) for row in read_json(root / "tasks.json")]
    answers = {
        key: PrivateAnswer.model_validate(value) for key, value in read_json(root / "answers.json").items()
    }
    if (
        not rows
        or len({row.task_id for row in rows}) != len(rows)
        or set(answers) != {r.task_id for r in rows}
    ):
        raise ValueError("public tasks and private answers must match exactly")
    messages = {m.id: m for m in world.messages}
    for row in rows:
        if (row.workspace_id, row.snapshot_hash, row.snapshot_ref) != (
            manifest.workspace_id,
            manifest.snapshot_hash,
            "snapshot.json",
        ):
            raise ValueError("task references a different snapshot")
        api = SlackAPI(world, row.actor_id)
        answer = answers[row.task_id]
        if not answer.message_ids and not answer.user_ids:
            raise ValueError("answer has no evidence")
        for identifier in answer.message_ids:
            message = messages.get(identifier)
            if message is None or message.deleted or not api.is_conversation_visible(message.conversation_id):
                raise ValueError("answer references missing or inaccessible evidence")
        for identifier in answer.user_ids:
            api.get_user(identifier)
    return world, rows, answers


def write_release(root: Path, world: SlackWorld, rows: list[PublicTask], answers: dict[str, PrivateAnswer]):
    expected = (world, rows, answers)
    if root.exists():
        if load_release(root) != expected:
            raise ValueError("refusing to overwrite a different release")
        return
    root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root.parent, prefix=".release-") as temporary:
        stage = Path(temporary) / "release"
        files = {
            "tasks.json": [row.model_dump(mode="json") for row in rows],
            "snapshot.json": world.model_dump(mode="json"),
            "answers.json": {key: value.model_dump(mode="json") for key, value in answers.items()},
        }
        for name, content in files.items():
            atomic_json(stage / name, content)
        manifest = Manifest(
            workspace_id=rows[0].workspace_id,
            snapshot_hash=digest(world.model_dump(mode="json")),
            files={name: hashlib.sha256((stage / name).read_bytes()).hexdigest() for name in files},
        )
        atomic_json(stage / "manifest.json", manifest.model_dump(mode="json"))
        if load_release(stage) != expected:
            raise ValueError("published data differs from approved inputs")
        os.rename(stage, root)
