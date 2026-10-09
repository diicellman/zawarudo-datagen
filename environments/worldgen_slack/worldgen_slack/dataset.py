"""The release: one solver copy of the world, public questions, and private gold answers."""

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from .db import World, canonical


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


SafeId = Annotated[str, Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")]
NonEmptyText = Annotated[str, Field(min_length=1)]


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".pending-")
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(canonical(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
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

    return json.loads(path.read_bytes(), object_pairs_hook=unique)


def sha256(path: Path) -> str:
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class PublicTask(StrictModel):
    """A task as a solver gets it, with how hard it measured: the solver model's tries at the release (those that
    crashed, and those that never answered), its share of right answers (with its 90% interval), of right and
    grounded ones, and of those that declined to answer, the gold evidence the tries saw, and the witness's share of
    right answers when it tried. Training can filter a curriculum on them, as prime-envs filters on avg@k columns."""

    task_id: SafeId
    question: NonEmptyText
    actor_id: SafeId
    category: NonEmptyText
    level: int
    answer_type: Literal["text", "set", "number", "refusal", "status"]
    world_hash: str
    solver: str | None = None
    tries: int | None = None
    crashed: int | None = None
    unanswered: int | None = None
    right_rate: float | None = None
    right_interval: Annotated[list[float], Field(min_length=2, max_length=2)] | None = (
        None  # its 90% interval
    )
    strict_rate: float | None = None
    abstain_rate: float | None = None
    coverage: float | None = None
    witness: str | None = None
    witness_right: float | None = None


class PrivateAnswer(StrictModel):
    """The gold rows of a task, the messages (channel, ts) and people its answer rests on, and, for a refusal, the
    truth out of its actor's sight (`unseen`)."""

    answer_type: Literal["text", "set", "number", "refusal", "status"]
    rows: list[dict[str, JsonValue]]
    gold_sql: NonEmptyText
    messages: list[Annotated[list[str], Field(min_length=2, max_length=2)]]  # [channel, ts]
    unseen: list[Annotated[list[str], Field(min_length=2, max_length=2)]] = []
    users: list[str]


class Manifest(StrictModel):
    format: Literal["worldgen-slack.v6", "worldgen-slack.v7", "worldgen-slack.v8", "worldgen-slack.v9"] = (
        # v7: tasks carry their rates; v8: strict grounds the answer's claims, unanswered counted; v9: status answers,
        # the abstain rate and the right rate's interval, and a reward of +1, 0 or -1
        "worldgen-slack.v9"
    )
    world_hash: str
    files: dict[str, str]


FILES = {"world.sqlite", "tasks.json", "answers.json"}


def load_release(root: Path) -> tuple[Path, list[PublicTask], dict[str, PrivateAnswer]]:
    """Verify a release before use: file hashes, one world, matching tasks and answers, readable evidence."""
    root = root.resolve()
    manifest = Manifest.model_validate_json((root / "manifest.json").read_bytes())
    if set(manifest.files) != FILES:
        raise ValueError("a release holds world.sqlite, tasks.json and answers.json")
    for name, expected in manifest.files.items():
        if (root / name).is_symlink() or sha256(root / name) != expected:
            raise ValueError(f"release file hash mismatch: {name}")
    rows = [PublicTask.model_validate(r) for r in read_json(root / "tasks.json")]
    answers = {k: PrivateAnswer.model_validate(v) for k, v in read_json(root / "answers.json").items()}
    if not rows or len({r.task_id for r in rows}) != len(rows) or set(answers) != {r.task_id for r in rows}:
        raise ValueError("public tasks and private answers must match exactly")
    for row in rows:
        if row.world_hash != manifest.world_hash:
            raise ValueError("a task references another world")
        reader, answer = World(root / "world.sqlite", actor=row.actor_id), answers[row.task_id]
        for channel, ts in answer.messages + answer.unseen:
            readable = reader.db.execute(
                "SELECT 1 FROM messages WHERE channel_id = ? AND ts = ?", (channel, ts)
            ).fetchone()
            if not readable and [channel, ts] in answer.messages:
                raise ValueError(f"{row.task_id}: evidence its actor cannot read")
            if readable and [channel, ts] in answer.unseen:
                raise ValueError(f"{row.task_id}: evidence out of sight that its actor can read")
        for user in answers[row.task_id].users:
            reader.get_user(user)
        reader.close()
    return root / "world.sqlite", rows, answers


def write_release(root: Path, world: Path, rows: list[PublicTask], answers: dict[str, PrivateAnswer]) -> None:
    """Publish atomically; an existing release must be the same one."""
    if root.exists():
        _, old_rows, old_answers = load_release(root)
        if (old_rows, old_answers) != (rows, answers) or sha256(root / "world.sqlite") != sha256(world):
            raise ValueError("refusing to overwrite a different release")
        return
    root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root.parent, prefix=".release-") as temporary:
        stage = Path(temporary) / "release"
        stage.mkdir()
        shutil.copyfile(world, stage / "world.sqlite")
        atomic_json(stage / "tasks.json", [r.model_dump(mode="json") for r in rows])
        atomic_json(stage / "answers.json", {k: v.model_dump(mode="json") for k, v in answers.items()})
        files = {name: sha256(stage / name) for name in sorted(FILES)}
        atomic_json(
            stage / "manifest.json", Manifest(world_hash=files["world.sqlite"], files=files).model_dump()
        )
        if load_release(stage)[1:] != (rows, answers):
            raise ValueError("published data differs from approved inputs")
        os.rename(stage, root)
