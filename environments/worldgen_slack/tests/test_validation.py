from __future__ import annotations

from pathlib import Path
import os
import subprocess
import sys

import pytest
import verifiers.v1 as vf

from worldgen_slack.contracts import parse_synthesized_item, preflight_synthesized_item
from worldgen_slack.env import SlackDataGenerationConfig
from worldgen_slack.slack.validate import (
    check_candidate,
    evaluate_candidate_in_runtime,
    static_source_errors,
    validate_compiled_snapshots,
    validate_world,
)

from conftest import make_world


def test_evidence_is_reachable_only_through_declared_message_rows(world, contract) -> None:
    report = validate_world(world, contract)
    assert report["ok"]
    assert {row["output"][0]["message_id"] for row in report["gold_call_log"] if row["output"]} >= {
        "newer",
        "root",
    }

    broken = contract.model_copy(
        update={
            "gold_calls": [
                contract.gold_calls[0].model_copy(update={"arguments": {"query": "unrelated rehearsal"}})
            ]
        }
    )
    failed = validate_world(world, broken)
    assert not failed["ok"]
    assert any(check["name"] == "gold_replay" and not check["ok"] for check in failed["checks"])


def test_source_policy_rejects_escape_and_nonexact_build_signatures() -> None:
    invalid_sources = [
        "import os\ndef build(seed, contract): return None\n",
        "async def build(seed, contract): return None\n",
        "def build(seed, contract=None): return None\n",
        "def build(seed, contract, *args): return None\n",
        "@staticmethod\ndef build(seed, contract): return None\n",
        "def build(seed, contract):\n    return open('/etc/passwd')\n",
    ]
    assert all(static_source_errors(source) for source in invalid_sources)
    assert not static_source_errors("def build(seed, contract):\n    return None\n")


def test_generated_program_failure_is_a_bounded_candidate_rejection(
    tmp_path: Path, contract, monkeypatch
) -> None:
    candidate = tmp_path / "world.py"
    candidate.write_text("def build(seed, contract):\n    return None\n")

    def generated_failure(*_args):
        raise SystemExit("candidate tried to exit")

    monkeypatch.setattr(
        "worldgen_slack.slack.validate._load_build",
        lambda *_args: generated_failure,
    )
    result = check_candidate(candidate, contract, [0])
    assert not result["ok"]
    assert result["snapshots"] == {}
    assert all("SystemExit" in report["error"] for report in result["reports"])


def test_seed_family_requires_determinism_variation_and_answer_stability(contract) -> None:
    snapshots = {
        "0": make_world(0).model_dump(mode="json"),
        "101": make_world(101).model_dump(mode="json"),
        "202": make_world(202).model_dump(mode="json"),
        "repeat_0": make_world(0).model_dump(mode="json"),
    }
    assert validate_compiled_snapshots(snapshots, contract, [0, 101, 202]).ok

    changed = make_world(101).model_dump(mode="json")
    next(message for message in changed["messages"] if message["id"] == "msg_fix")["text"] += " Later."
    snapshots["101"] = changed
    report = validate_compiled_snapshots(snapshots, contract, [0, 101, 202])
    assert not report.ok
    assert any(check.name == "hidden_seed_answer_stability" and not check.ok for check in report.checks)


@pytest.mark.asyncio
async def test_candidate_execution_accepts_only_default_deny_prime_vm(contract) -> None:
    with pytest.raises(TypeError, match="PrimeConfig"):
        await evaluate_candidate_in_runtime(b"", contract, vf.SubprocessConfig())
    with pytest.raises(ValueError, match="framework-only egress"):
        await evaluate_candidate_in_runtime(
            b"",
            contract,
            vf.PrimeConfig(vm=True, allow=["*"], block=[]),
        )

    config = SlackDataGenerationConfig(taskset=vf.TasksetConfig(id="worldgen-slack-generation"))
    values = config.model_dump(mode="python")
    values["solver"]["runtime"] = vf.SubprocessConfig()
    with pytest.raises(TypeError, match="solver runtime must be PrimeConfig"):
        SlackDataGenerationConfig.model_validate(values)


def test_synthesized_json_and_preflight_are_strict(synthesized) -> None:
    parsed = parse_synthesized_item(synthesized.model_dump_json())
    preflight_synthesized_item(parsed)
    fenced = parse_synthesized_item(f"prefix ```json\n{synthesized.model_dump_json()}\n``` suffix")
    assert fenced == synthesized
    with pytest.raises(ValueError, match="strict JSON"):
        parse_synthesized_item('{"scenario": null, "scenario": null}')
    with pytest.raises(ValueError, match="strict JSON"):
        parse_synthesized_item('{"value": NaN}')
    with pytest.raises(ValueError, match="exceeds 65536 bytes"):
        parse_synthesized_item("x" * 65_537 + synthesized.model_dump_json())
    with pytest.raises(ValueError):
        parse_synthesized_item("{} " + synthesized.model_dump_json())
    invalid = parsed.model_copy(
        update={"task": parsed.task.model_copy(update={"question": "Post the answer?"})}
    )
    with pytest.raises(ValueError, match="write action"):
        preflight_synthesized_item(invalid)


def test_fixed_checker_projection_imports_without_unprojected_tool_modules(tmp_path) -> None:
    package = Path(__file__).resolve().parents[1] / "worldgen_slack"
    fixed = tmp_path / "fixed" / "worldgen_slack"
    (fixed / "slack").mkdir(parents=True)
    for relative in (
        "__init__.py",
        "contracts.py",
        "slack/__init__.py",
        "slack/models.py",
        "slack/api.py",
        "slack/validate.py",
    ):
        target = fixed / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((package / relative).read_bytes())
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import worldgen_slack.slack.validate as value; "
                "assert value.__file__.startswith(%r)" % str(tmp_path)
            ),
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(tmp_path / "fixed")},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
