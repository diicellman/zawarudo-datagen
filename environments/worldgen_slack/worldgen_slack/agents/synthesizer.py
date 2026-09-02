from __future__ import annotations

import hashlib
import json
import random
from enum import StrEnum
from pathlib import Path
from typing import Any, TypeVar

import verifiers.v1 as vf
from pydantic import ValidationError

from ..contracts import (
    AnswerKind,
    EvidenceLayout,
    GenerationSeedData,
    ReasoningShape,
    SynthesisBrief,
    Workflow,
    parse_synthesized_item,
    preflight_synthesized_item,
)
from ..slack.models import SynthesizedItem

EnumT = TypeVar("EnumT", bound=StrEnum)


class SynthesizerData(vf.TaskData):
    generation_seed: int
    interface_id: str
    brief: SynthesisBrief


class SynthesizerTask(vf.Task[SynthesizerData]):
    pass


def _rng(schedule_seed: int, label: str, block: int) -> random.Random:
    digest = hashlib.sha256(f"worldgen-slack-brief-v1:{schedule_seed}:{label}:{block}".encode()).digest()
    return random.Random(int.from_bytes(digest[:16], "big"))


def _balanced(enum_type: type[EnumT], count: int, schedule_seed: int, label: str) -> list[EnumT]:
    values = list(enum_type)
    output: list[EnumT] = []
    block = 0
    while len(output) < count:
        shuffled = values.copy()
        _rng(schedule_seed, label, block).shuffle(shuffled)
        output.extend(shuffled)
        block += 1
    return output[:count]


def _compatible(reasoning: ReasoningShape, layout: EvidenceLayout) -> bool:
    if reasoning == ReasoningShape.CROSS_CHANNEL:
        return layout == EvidenceLayout.MULTIPLE_CHANNELS
    if reasoning == ReasoningShape.CROSS_THREAD:
        return layout != EvidenceLayout.ONE_THREAD
    return True


def _repair_layouts(
    reasoning: list[ReasoningShape],
    layouts: list[EvidenceLayout],
) -> None:
    for index, shape in enumerate(reasoning):
        if _compatible(shape, layouts[index]):
            continue
        swap = next(
            (
                other
                for other in range(len(layouts))
                if _compatible(shape, layouts[other]) and _compatible(reasoning[other], layouts[index])
            ),
            None,
        )
        if swap is not None:
            layouts[index], layouts[swap] = layouts[swap], layouts[index]
        elif shape == ReasoningShape.CROSS_CHANNEL:
            layouts[index] = EvidenceLayout.MULTIPLE_CHANNELS
        else:
            layouts[index] = EvidenceLayout.MULTIPLE_THREADS


def _build_synthesis_block(schedule_seed: int) -> tuple[SynthesisBrief, ...]:
    count = 10
    workflows = _balanced(Workflow, count, schedule_seed, "workflow")
    reasoning = _balanced(ReasoningShape, count, schedule_seed, "reasoning")
    layouts = _balanced(EvidenceLayout, count, schedule_seed, "evidence-layout")
    answers = _balanced(AnswerKind, count, schedule_seed, "answer-kind")
    _repair_layouts(reasoning, layouts)

    def assemble(answer_order: list[AnswerKind]) -> tuple[SynthesisBrief, ...]:
        return tuple(
            SynthesisBrief(
                workflow=workflows[index],
                reasoning_shape=reasoning[index],
                evidence_layout=layouts[index],
                answer_kind=answer_order[index],
            )
            for index in range(count)
        )

    candidates = [answers[index:] + answers[:index] for index in range(count)]
    for attempt in range(32):
        shuffled = answers.copy()
        _rng(schedule_seed, "answer-dedup", attempt).shuffle(shuffled)
        candidates.append(shuffled)
    for answer_order in candidates:
        briefs = assemble(answer_order)
        if len(set(brief.model_dump_json() for brief in briefs)) == len(briefs):
            return briefs
    raise RuntimeError("deterministic synthesis block could not avoid duplicate briefs")


def _block_seed(schedule_seed: int, block: int, nonce: int) -> int:
    if block == 0 and nonce == 0:
        return schedule_seed
    digest = hashlib.sha256(
        f"worldgen-slack-brief-block-v1:{schedule_seed}:{block}:{nonce}".encode()
    ).digest()
    return int.from_bytes(digest[:8], "big")


def build_synthesis_schedule(schedule_seed: int, count: int) -> tuple[SynthesisBrief, ...]:
    if schedule_seed < 0 or count < 0:
        raise ValueError("schedule seed and count must be non-negative")
    output: list[SynthesisBrief] = []
    seen: set[str] = set()
    for block in range((count + 9) // 10):
        for nonce in range(1_000):
            candidate = _build_synthesis_block(_block_seed(schedule_seed, block, nonce))
            rendered = {brief.model_dump_json() for brief in candidate}
            if not rendered & seen:
                output.extend(candidate)
                seen.update(rendered)
                break
        else:
            raise RuntimeError("deterministic synthesis schedule exhausted unique brief blocks")
    return tuple(output[:count])


def synthesis_brief_at(schedule_seed: int, index: int) -> SynthesisBrief:
    if index < 0:
        raise ValueError("synthesis brief index must be non-negative")
    return build_synthesis_schedule(schedule_seed, index + 1)[index]


def make_synthesizer_task(seed: GenerationSeedData, brief: SynthesisBrief) -> SynthesizerTask:
    interface = (Path(__file__).resolve().parents[1] / "slack" / "interface.json").read_text()
    schema = json.dumps(SynthesizedItem.model_json_schema(), indent=2, sort_keys=True)
    prompt = f"""Create one Slack QA contract for the supplied synthesis brief.

Return one JSON object matching the supplied proposal schema. The scenario controls organization and
workflow context; the task is the private QA contract. Invent compact realistic content and symbolic
IDs that a builder can materialize. The task must be answerable through the listed read-only actions.
Require genuine evidence composition: use at least two distinct evidence messages, and do not repeat
one scalar canonical answer across evidence items or let one message restate the full conclusion.
A root message has no thread_root_id. A reply's thread_root_id references a different message_id.

SYNTHESIS BRIEF:
{brief.model_dump_json(indent=2)}

SLACK ACTIONS:
{interface}

PROPOSAL SCHEMA:
{schema}
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
            brief=brief,
        )
    )


def _bounded_error(error: ValueError | ValidationError) -> str:
    if isinstance(error, ValidationError):
        detail = json.dumps(
            error.errors(include_input=False, include_url=False)[:12],
            ensure_ascii=False,
            sort_keys=True,
            default=lambda value: f"{type(value).__name__}: {value}",
        )
    else:
        detail = str(error)
    encoded = detail.encode("utf-8", errors="replace")[:3_500]
    return encoded.decode("utf-8", errors="ignore")


def repair_prompt(error: ValueError | ValidationError) -> str:
    return (
        "The proposal failed strict JSON, schema, or semantic preflight. Correct it once. "
        "Return one JSON object only. Bounded validation error: " + _bounded_error(error)
    )


def bounded_response(text: str, max_bytes: int = 65_536) -> dict[str, object]:
    raw = text.encode("utf-8", errors="replace")
    prefix = raw[:max_bytes]
    return {
        "text": prefix.decode("utf-8", errors="ignore"),
        "utf8_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "truncated": len(raw) > max_bytes,
    }


async def synthesize(
    seed: GenerationSeedData,
    agent: vf.Agent,
    brief: SynthesisBrief,
    *,
    progress: Any | None = None,
) -> SynthesizedItem | None:
    item: SynthesizedItem | None = None
    error: ValueError | ValidationError | None = None
    repairs = 0
    final_response = ""
    async with agent.interaction(make_synthesizer_task(seed, brief)) as interaction:
        segment = await interaction.turn()
        for attempt in range(2):
            final_response = segment.last_reply
            try:
                item = parse_synthesized_item(final_response)
                preflight_synthesized_item(item, brief=brief)
                error = None
                break
            except (ValueError, ValidationError) as exc:
                item = None
                error = exc
                if attempt or segment.terminated:
                    break
                repairs += 1
                if progress is not None:
                    progress.stage_updated(
                        seed.generation_seed,
                        event="repair",
                        repair_count=repairs,
                        error_type=type(exc).__name__,
                    )
                segment = await interaction.turn(repair_prompt(exc))
        trace = interaction.trace
    trace.info["synthesis_brief"] = brief.model_dump(mode="json")
    trace.info["synthesis_repairs"] = repairs
    if not trace.ok:
        trace.info["synthesizer_error"] = (
            "; ".join(f"{entry.type}: {entry.message}" for entry in trace.errors[:20])[:4_000]
            or "synthesizer rollout failed without a recorded error"
        )
        trace.info["synthesizer_raw_response"] = bounded_response(final_response)
        raise RuntimeError("synthesizer rollout failed")
    if item is not None:
        trace.info["synthesized_item"] = item.model_dump(mode="json")
    elif error is not None:
        trace.info["synthesizer_error"] = f"{type(error).__name__}: {_bounded_error(error)}"
        trace.info["synthesizer_raw_response"] = bounded_response(final_response)
    else:
        raise RuntimeError("synthesizer ended without an item or parse error")
    return item


__all__ = [
    "SynthesizerData",
    "SynthesizerTask",
    "bounded_response",
    "build_synthesis_schedule",
    "make_synthesizer_task",
    "synthesis_brief_at",
    "synthesize",
]
