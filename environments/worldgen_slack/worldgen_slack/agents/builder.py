from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any

import verifiers.v1 as vf
from pydantic import Field

from ..contracts import ScenarioSpec
from ..slack.models import TaskContract

WORLD_STUB = """from worldgen_slack.slack.models import SlackWorld, TaskContract


def build(seed: int, contract: TaskContract) -> SlackWorld:
    raise NotImplementedError
"""

BUILDER_PROMPT = """Build the task-specific Slack world in `/task/workspace/world.py`.
Read `/task/scenario.json`, `/task/task_contract.json`, `/task/runtime/MODELS.md`,
`/task/runtime/API.md`, and `/task/runtime/builder_guide.md`. The starter is empty. Keep the
contract frozen and the fixed Slack interface unchanged. Run `/task/check-world` after edits.
The source artifact is authoritative, so finish with a valid file rather than a status token.
"""

_PUBLIC_RUNNER_SOURCE = r"""# /// script
# requires-python = ">=3.11"
# dependencies = ["pydantic==2.13.4"]
# ///
import ast
import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, "/task/.public")
from worldgen_slack.slack.api import SlackAPI
from worldgen_slack.slack.models import SlackWorld, TaskContract

ALLOWED = {
    "__future__", "collections", "dataclasses", "datetime", "decimal", "enum",
    "functools", "hashlib", "itertools", "math", "pydantic", "random", "string",
    "typing", "worldgen_slack.slack.models",
}
UNSAFE_NAMES = {
    "__builtins__", "__import__", "breakpoint", "compile", "delattr", "dir",
    "eval", "exec", "getattr", "globals", "help", "input", "locals",
    "memoryview", "open", "setattr", "vars",
}
UNSAFE_ATTRIBUTES = {
    "chdir", "chmod", "connect", "fork", "kill", "open", "popen",
    "rename", "rmdir", "spawn", "system", "unlink", "urlopen", "write_bytes",
    "write_text",
}
UNSAFE_IMPORT_MEMBERS = UNSAFE_NAMES | UNSAFE_ATTRIBUTES | {
    "os", "pathlib", "requests", "shutil", "socket", "subprocess", "sys",
    "tempfile", "urllib",
}
PROTECTED_IMPORT_MEMBERS = {
    "__future__": {"annotations"},
    "pydantic": {
        "BaseModel", "ConfigDict", "Field", "RootModel", "ValidationError",
        "computed_field", "field_serializer", "field_validator", "model_serializer",
        "model_validator",
    },
    "typing": {
        "Annotated", "Any", "Callable", "ClassVar", "Dict", "Final", "Generic", "Iterable", "Iterator", "List", "Literal",
        "Mapping", "NamedTuple", "Optional", "Protocol", "Sequence", "Set", "Self",
        "Tuple", "TypeAlias", "TypeVar", "TypedDict", "Union", "cast",
    },
    "worldgen_slack.slack.models": {
        "AnswerSpec", "Conversation", "EvidenceRequirement", "GoldCall", "Message",
        "Reaction", "SlackWorld", "StrictModel", "TaskContract", "User",
    },
}


def static_errors(source):
    errors = []
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"syntax: {exc}"]
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [item.name for item in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        for name in names:
            if not any(name == root or name.startswith(root + ".") for root in ALLOWED):
                errors.append(f"import not allowed: {name}")
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for imported in node.names:
                member = imported.name.rsplit(".", 1)[-1]
                if (
                    member.startswith("_")
                    or member in UNSAFE_IMPORT_MEMBERS
                    or (imported.asname and imported.asname.startswith("_"))
                ):
                    errors.append(f"import member not allowed: {imported.name}")
                if isinstance(node, ast.Import) and any(
                    imported.name == module or imported.name.startswith(module + ".")
                    for module in PROTECTED_IMPORT_MEMBERS
                ):
                    errors.append(f"use explicit safe imports from: {imported.name}")
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
                        errors.append(f"protected submodule not allowed: {node.module}")
                    elif node.module in PROTECTED_IMPORT_MEMBERS and (
                        imported.name not in PROTECTED_IMPORT_MEMBERS[node.module]
                    ):
                        errors.append(f"import member not allowlisted: {imported.name}")
        if isinstance(node, ast.Name) and (
            node.id in UNSAFE_NAMES or node.id.startswith("__")
        ):
            errors.append(f"name not allowed: {node.id}")
        if isinstance(node, ast.Attribute) and (
            node.attr.startswith(("_", "exec", "spawn"))
            or node.attr in UNSAFE_ATTRIBUTES
        ):
            errors.append(f"attribute not allowed: {node.attr}")
    builds = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "build"
    ]
    if len(builds) != 1 or [arg.arg for arg in builds[0].args.args] != ["seed", "contract"]:
        errors.append("define exactly build(seed, contract)")
    return sorted(set(errors))


def dispatch(api, tool, arguments):
    methods = {
        "slack_list_conversations": api.list_conversations,
        "slack_search_messages": api.search_messages,
        "slack_get_conversation_history": api.get_conversation_history,
        "slack_get_thread": api.get_thread,
        "slack_get_user": api.get_user,
    }
    return methods[tool](**arguments)


def main():
    candidate = Path(sys.argv[1])
    contract_path = Path(sys.argv[2])
    output_path = Path(sys.argv[3])
    seed = int(sys.argv[4])
    source = candidate.read_text(encoding="utf-8")
    errors = static_errors(source)
    result = {"ok": False, "errors": errors, "counts": {}, "gold_call_log": []}
    if not errors:
        try:
            contract = TaskContract.model_validate_json(contract_path.read_text(encoding="utf-8"))
            spec = importlib.util.spec_from_file_location("candidate_public", candidate)
            if spec is None or spec.loader is None:
                raise ImportError("could not load candidate")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            world = SlackWorld.model_validate(module.build(seed, contract.model_copy(deep=True)))
            counts = {
                name: len(getattr(world, name))
                for name in ("users", "conversations", "messages", "reactions")
            }
            result["counts"] = counts
            if len(world.model_dump_json().encode()) > 512_000:
                errors.append("snapshot exceeds 512000 bytes")
            limits = {
                "users": 40,
                "conversations": 15,
                "messages": 150,
                "reactions": 100,
            }
            errors.extend(
                f"{name} count exceeds {maximum}"
                for name, maximum in limits.items()
                if counts[name] > maximum
            )
            api = SlackAPI(world, contract.actor_id)
            messages = {message.id: message for message in world.messages}
            for evidence in contract.required_evidence:
                message = messages.get(evidence.message_id)
                if message is None:
                    errors.append(f"evidence {evidence.evidence_id}: missing message")
                    continue
                if message.conversation_id != evidence.conversation_id:
                    errors.append(f"evidence {evidence.evidence_id}: conversation mismatch")
                if evidence.author_id is not None and message.author_id != evidence.author_id:
                    errors.append(f"evidence {evidence.evidence_id}: author mismatch")
                if evidence.thread_root_id is not None and message.thread_root_id != evidence.thread_root_id:
                    errors.append(f"evidence {evidence.evidence_id}: thread mismatch")
                for term in evidence.required_terms:
                    if term.casefold() not in message.text.casefold():
                        errors.append(f"evidence {evidence.evidence_id}: missing term {term!r}")
                if message.deleted or not api.is_conversation_visible(message.conversation_id):
                    errors.append(f"evidence {evidence.evidence_id}: not actor-visible")
            outputs = []
            for index, call in enumerate(contract.gold_calls, 1):
                value = dispatch(api, call.tool, call.arguments)
                outputs.append(
                    {
                        "index": index,
                        "tool": call.tool,
                        "arguments": call.arguments,
                        "output": value,
                    }
                )
            result["gold_call_log"] = outputs
            exposed = json.dumps([item["output"] for item in outputs], sort_keys=True).casefold()
            for evidence in contract.required_evidence:
                by_id = evidence.message_id.casefold() in exposed
                by_terms = all(term.casefold() in exposed for term in evidence.required_terms)
                if not (by_id or by_terms):
                    errors.append(f"gold calls do not recover {evidence.evidence_id}")
            public = "\n".join(
                [
                    *(conversation.name or "" for conversation in world.conversations),
                    *(message.text for message in world.messages),
                ]
            ).casefold()
            if contract.question.casefold() in public:
                errors.append("literal question appears in public content")
            if any(
                f'"{name}"' in public
                for name in ("canonical_answer", "required_evidence", "gold_calls")
            ):
                errors.append("serialized private contract field appears in public content")
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
    result["errors"] = errors
    result["ok"] = not errors
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    print("RESULT PASS" if result["ok"] else "RESULT FAIL")
    for error in errors[:20]:
        print(f"- {error}")
    return 0 if result["ok"] else 2


raise SystemExit(main())
"""


class BuilderData(vf.TaskData):
    generation_seed: int
    interface_id: str


class BuilderState(vf.State):
    candidate_source: str = ""
    public_attempts: list[dict[str, Any]] = Field(default_factory=list)
    conversational_completion: str = ""
    classification: str = ""


class BuilderConfig(vf.TaskConfig):
    scenario: ScenarioSpec
    contract: TaskContract
    public_seed: int = 0


class BuilderTask(vf.Task[BuilderData, BuilderState, BuilderConfig]):
    async def setup(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        package = Path(__file__).resolve().parents[1]
        slack = package / "slack"
        files = {
            "/task/scenario.json": self.config.scenario.model_dump_json(indent=2).encode(),
            "/task/task_contract.json": self.config.contract.model_dump_json(indent=2).encode(),
            "/task/runtime/MODELS.md": (slack / "models.schema.json").read_bytes(),
            "/task/runtime/API.md": (slack / "behavior.md").read_bytes(),
            "/task/runtime/builder_guide.md": (slack / "builder_guide.md").read_bytes(),
            "/task/workspace/world.py": WORLD_STUB.encode(),
            "/task/.public/worldgen_slack/__init__.py": b"",
            "/task/.public/worldgen_slack/slack/__init__.py": b"",
            "/task/.public/worldgen_slack/slack/models.py": (slack / "models.py").read_bytes(),
            "/task/.public/worldgen_slack/slack/api.py": (slack / "api.py").read_bytes(),
        }
        for path, content in files.items():
            await runtime.write(path, content)
        program = await runtime.prepare_uv_script(_PUBLIC_RUNNER_SOURCE)
        command = [
            *program,
            "/task/workspace/world.py",
            "/task/task_contract.json",
            "/task/public-result.json",
            str(self.config.public_seed),
        ]
        wrapper = (
            "#!/bin/sh\n"
            "rm -f /task/public-result.json\n"
            "ulimit -f 8192\n"
            "PYTHONPATH=/task/.public "
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
    name = contract.task_slug or f"slack-builder-{generation_seed:08d}"
    data = BuilderData(
        idx=generation_seed,
        name=name,
        prompt=BUILDER_PROMPT,
        network_allow=[],
        network_block=["*"],
        generation_seed=generation_seed,
        interface_id=interface_id,
    )
    return BuilderTask(
        data,
        BuilderConfig(scenario=scenario, contract=contract, public_seed=public_seed),
    )


__all__ = [
    "BuilderData",
    "BuilderState",
    "BuilderTask",
    "WORLD_STUB",
    "make_builder_task",
]
