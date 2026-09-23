"""Prepare bounded, inspectable Slack seed excerpts without running generation."""

import argparse
import json
import tomllib
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlencode
from urllib.request import urlopen

from pydantic import Field, model_validator

from generators.worldgen_slack.contracts import SeedExample, SeedMessage, SeedPacket, SEED_MAX_BYTES
from worldgen_slack.slack.api import digest
from worldgen_slack.slack.models import NonEmptyText, SafeId, StrictModel
from worldgen_slack.dataset import atomic_json

ROOT = Path(__file__).resolve().parents[2]
MAX_RESPONSE_BYTES = 1_048_576


class Selection(StrictModel):
    id: SafeId
    dataset: Literal["spencer/software_slacks", "unionai/flyte-slack-data"]
    revision: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    notes: NonEmptyText
    join_pairs: bool = False

    @model_validator(mode="after")
    def bounded_range(self):
        if not 0 < self.end - self.start <= 100:
            raise ValueError("select 1–100 rows using an exclusive end")
        if self.join_pairs and self.dataset != "unionai/flyte-slack-data":
            raise ValueError("join_pairs only applies to manually checked Flyte sequences")
        return self


class Preparation(StrictModel):
    output: NonEmptyText
    examples: list[Selection] = Field(min_length=1, max_length=48)

    @model_validator(mode="after")
    def unique_ids(self):
        if len({example.id for example in self.examples}) != len(self.examples):
            raise ValueError("selection IDs must be unique")
        return self


def fetch_json(url):
    with urlopen(url, timeout=30) as response:
        raw = response.read(MAX_RESPONSE_BYTES + 1)
        revision = response.headers.get("X-Revision")
    if len(raw) > MAX_RESPONSE_BYTES:
        raise ValueError("HTTP response exceeds 1 MiB")
    return json.loads(raw), revision


def normalize_software(rows):
    if len({(row["workspace"], row["channel"]) for row in rows}) != 1:
        raise ValueError("a software excerpt must stay in one workspace/channel")
    return [SeedMessage(text=row["text"], speaker=row["user"], timestamp=row["ts"]) for row in rows]


def normalize_flyte(rows, *, join_pairs):
    if len(rows) > 1 and not join_pairs:
        raise ValueError("multi-row Flyte excerpts require manually checked join_pairs=true")
    messages = [SeedMessage(text=rows[0]["input"])]
    for index, row in enumerate(rows):
        if index and rows[index - 1]["output"] != row["input"]:
            raise ValueError("Flyte sequence crosses a pair boundary; select separate excerpts")
        messages.append(SeedMessage(text=row["output"]))
    return messages


def normalize(selection, response):
    rows = response["rows"]
    indices = [row["row_idx"] for row in rows]
    if indices != list(range(selection.start, selection.end)):
        raise ValueError("viewer returned incomplete or unexpected row indices")
    if any(row["truncated_cells"] for row in rows):
        raise ValueError("viewer truncated source cells")
    records = [row["row"] for row in rows]
    messages = (
        normalize_software(records)
        if selection.dataset == "spencer/software_slacks"
        else normalize_flyte(records, join_pairs=selection.join_pairs)
    )
    return SeedExample(
        id=selection.id,
        dataset=selection.dataset,
        revision=selection.revision,
        rows=indices,
        source_sha256=digest(response),
        notes=selection.notes,
        messages=messages,
    )


def prepare(config):
    output = (ROOT / config.output).resolve()
    if not output.is_relative_to(ROOT / "data/seeds"):
        raise ValueError("preparation output must be inside data/seeds/")
    examples, sources = [], []
    for selection in config.examples:
        metadata_url = "https://huggingface.co/api/datasets/" + selection.dataset
        metadata, _ = fetch_json(metadata_url)
        if metadata["sha"] != selection.revision:
            raise ValueError("dataset revision changed; review the selection before updating it")
        url = "https://datasets-server.huggingface.co/rows?" + urlencode(
            dict(
                dataset=selection.dataset,
                config="default",
                split="train",
                offset=selection.start,
                length=selection.end - selection.start,
            )
        )
        response, revision = fetch_json(url)
        # The viewer is asynchronous: matching repository HEAD alone cannot establish its cache revision.
        if revision != selection.revision:
            raise ValueError("viewer revision is missing or differs from the pinned source")
        examples.append(normalize(selection, response))
        sources.append((selection.id, dict(url=url, revision=revision, metadata=metadata, response=response)))
    packet = SeedPacket(examples=examples)
    if len(packet.model_dump_json(indent=2).encode()) > SEED_MAX_BYTES:
        raise ValueError("formatted packet exceeds 256 KiB")
    for identifier, source in sources:
        atomic_json(output.parent / "raw" / f"{identifier}.json", source)
    atomic_json(output, packet.model_dump(mode="json"))
    print(f"Draft packet: {output}; inspect boundaries, identities, context, and instructions before use.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = Preparation.model_validate(tomllib.loads(args.config.read_text()))
    if args.dry_run:
        print(config.model_dump_json(indent=2))
    else:
        prepare(config)


if __name__ == "__main__":
    main()
