from __future__ import annotations

import json
from pathlib import Path

import verifiers.v1 as vf

from ..contracts import GenerationSeedData, SynthesizedItem


class SynthesizerData(vf.TaskData):
    generation_seed: int
    interface_id: str


class SynthesizerState(vf.State):
    response_json: str = ""
    repair_attempts: int = 0
    classification: str = ""


class SynthesizerTask(vf.Task[SynthesizerData, SynthesizerState]):
    pass


def make_synthesizer_task(
    seed: GenerationSeedData,
    recent_summaries: list[dict[str, object]],
) -> SynthesizerTask:
    slack_dir = Path(__file__).resolve().parents[1] / "slack"
    interface = (slack_dir / "interface.json").read_text(encoding="utf-8")
    ontology_schema = (slack_dir / "models.schema.json").read_text(encoding="utf-8")
    output_schema = json.dumps(SynthesizedItem.model_json_schema(), indent=2, sort_keys=True)
    recent = json.dumps(recent_summaries, ensure_ascii=False, indent=2, sort_keys=True)
    prompt = f"""Synthesize one new read-only Slack workplace scenario and one concrete QA contract.
Generation seed: {seed.generation_seed}

The question must be answerable only from facts exposed by the fixed actions. Use a concise private
answer, auditable required claims, concrete evidence bindings, and a valid gold read path. Ask one
question. Do not request writes, unauthorized private facts, self-rooted evidence, contradictory
message bindings, or state the answer in the question. IDs are symbolic and will be materialized by
a later builder. Return exactly one JSON object matching the output schema, with no fence or prose.

FIXED INTERFACE:
{interface}

FIXED PUBLIC ONTOLOGY:
{ontology_schema}

OUTPUT SCHEMA:
{output_schema}

RECENT ACCEPTED ITEMS TO AVOID DUPLICATING:
{recent}
"""
    return SynthesizerTask(
        SynthesizerData(
            idx=seed.idx,
            name=seed.name,
            prompt=prompt,
            network_allow=[],
            network_block=["*"],
            generation_seed=seed.generation_seed,
            interface_id=seed.interface_id,
        )
    )


def repair_prompt(error: Exception) -> str:
    return (
        "The JSON object failed strict parsing or semantic preflight: "
        f"{type(error).__name__}: {error}. Return one corrected JSON object only."
    )


__all__ = [
    "SynthesizerData",
    "SynthesizerState",
    "SynthesizerTask",
    "make_synthesizer_task",
    "repair_prompt",
]
