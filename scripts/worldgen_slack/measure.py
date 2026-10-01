"""Compare generated runs with baselines: name diversity, voice/timing statistics, repairs, cost, solving.

uv run --frozen python scripts/worldgen_slack/measure.py data/v2-01/* --baseline data/study_22_09/*/case-*
"""

import argparse
import json
import statistics
from pathlib import Path
from types import SimpleNamespace

from generators.worldgen_slack.config import ROOT
from generators.worldgen_slack.contracts import Persona, Typing, load_seed_packet, style
from generators.worldgen_slack.store import used_names
from worldgen_slack.slack.models import Message, SlackWorld, User

STYLE = ("median_words", "lowercase_start", "period_end", "question", "short", "seconds_zero", "off_hours")


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def catalog_of(run):
    """The catalog parts measured here; older catalogs predate later catalog fields."""
    catalog = read(run / "artifacts" / f"{read(run / 'state.json')['catalog']}.json")
    return SimpleNamespace(
        company=catalog["company"],
        people=[User.model_validate(p) for p in catalog["people"]],
        personas=[Persona.model_validate(p) for p in catalog["personas"]],
    )


def candidates(run):
    """Every reviewed or authored world in attempt order, as (attempt, snapshot)."""
    for folder in sorted((run / "attempts").iterdir(), key=lambda p: p.stat().st_mtime):
        payload = read(folder / "review_input.json") or {}
        raw = read(folder / "author_output.json")
        world = payload.get("candidate") or (
            json.loads(raw["text"]) if raw and '"snapshot"' in raw["text"] else None
        )
        if world and folder.name.startswith("build"):
            yield folder.name, world["snapshot"]


def revisions(run):
    rows, previous = [], {}
    for attempt, snapshot in candidates(run):
        current = {m["id"]: m for m in snapshot["messages"]}
        shared = current.keys() & previous.keys()
        text = sum(current[k]["text"] != previous[k]["text"] for k in shared)
        timing = sum(
            current[k]["text"] == previous[k]["text"] and current[k]["timestamp"] != previous[k]["timestamp"]
            for k in shared
        )
        rows.append(
            f"{attempt}: {len(current)} msgs, +{len(current.keys() - previous.keys())} -{len(previous.keys() - current.keys())} text~{text} timing-only~{timing}"
        )
        previous = current
    return rows


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(str(v) for v in row) + " |" for row in rows])


def world_row(name, world, personas=()):
    measured = style(world, list(personas))
    return [name, measured["messages"], *(measured[k] if measured[k] is not None else "–" for k in STYLE)]


def target(typing):
    if typing is None:
        return "–"
    return (
        f"{typing.median_words:g}w lc {typing.lowercase_share:.2f} q {typing.question_share:.2f} "
        f"short {typing.short_share:.2f}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--baseline", nargs="*", type=Path, default=[])
    parser.add_argument("--legacy", type=Path, default=ROOT / "data_intermidiate")
    parser.add_argument(
        "--seeds", type=Path, default=ROOT / "generators/worldgen_slack/seeds/slack-examples.json"
    )
    parser.add_argument("--typing", type=Path, default=ROOT / "data/seeds/typing-profiles.jsonl")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    sections, overview, names, authors, styles = [], [], [], [], []
    for run in args.runs + args.baseline:
        summary, catalog, state = read(run / "summary.json") or {}, catalog_of(run), read(run / "state.json")
        checkpoint = read(run / "artifacts" / f"{state['candidate']}.json") if state["candidate"] else None
        snapshot = read(run / "release/snapshot.json") or (checkpoint or {}).get("snapshot")
        label = f"{run.parent.name}/{run.name}"
        cost = sum(r["reported_cost"] for r in summary.get("usage_by_role", {}).values())
        writers = summary.get("usage_by_role", {}).get("writer", {}).get("traces", 0)
        # v2 runs evaluated once per task; v3 runs keep every build solve of each approved task review.
        results = state.get("evaluation", {}).values() or [
            r for review in state.get("task_reviews", {}).values() for r in review.get("results", [])
        ]
        corrections = sum(len(read(path)) for path in run.glob("attempts/*/corrections.json"))
        overview.append(
            [
                label,
                summary.get("status"),
                catalog.company,
                summary.get("message_count"),
                summary.get("rejected_candidates"),
                corrections,
                writers,
                f"${cost:.2f}",
                f"{summary.get('solved_tasks')}/{summary.get('task_count')}",
                sum(r["correct"] for r in results),
                sum(r["grounded"] for r in results),
            ]
        )
        corpus = used_names(ROOT / "data", run)
        people = {p.name for p in catalog.people}
        firsts = {name.split()[0] for name in corpus["people"]}
        reused_company = catalog.company.split()[0].casefold() in {
            c.split()[0].casefold() for c in corpus["companies"]
        }
        names.append(
            [
                label,
                catalog.company,
                reused_company,
                len(people & set(corpus["people"])),
                f"{len({n.split()[0] for n in people} & firsts)}/{len(people)}",
            ]
        )
        if snapshot:
            world = SlackWorld.model_validate(snapshot)
            styles.append(world_row(label, world, catalog.personas))
            if run in args.runs:
                voices = {p.id: p.voice for p in catalog.personas}
                targets = {p.id: p.profile.typing for p in catalog.personas if p.profile}
                for author, measured in style(world, catalog.personas)["authors"].items():
                    person = next(p.name for p in catalog.people if p.id == author)
                    authors.append(
                        [
                            label,
                            person,
                            voices.get(author, "–")[:60],
                            measured["messages"],
                            *(measured[k] for k in STYLE[:5]),
                            target(targets.get(author)),
                        ]
                    )
        sections.append(f"### {label}\n\n" + "\n".join(f"- {row}" for row in revisions(run)))
    legacy = [SlackWorld.model_validate(read(p)) for p in sorted(args.legacy.glob("worlds/*/snapshot.json"))]
    if legacy:
        messages = [m for w in legacy for m in w.messages]
        styles.append(
            [
                "legacy (pooled)",
                *world_row("", SlackWorld.model_construct(users=[], conversations=[], messages=messages))[1:],
            ]
        )
    if args.seeds.exists():
        seed_messages = [
            Message(
                id=f"s{i}", conversation_id="c", author_id="u", text=m.text, timestamp="2000-01-01T00:00:01Z"
            )
            for i, m in enumerate(m for e in load_seed_packet(args.seeds).examples for m in e.messages)
        ]
        row = world_row(
            "real seeds",
            SlackWorld.model_construct(
                users=[User(id="u", name="u")], conversations=[], messages=seed_messages
            ),
        )
        styles.append(row[:7] + ["–", "–"])
    if args.typing.exists():
        users = [Typing.model_validate_json(line) for line in args.typing.read_text().splitlines()]
        middle = {
            k: statistics.median(getattr(u, k) for u in users) for k in Typing.model_fields if k != "id"
        }
        styles.append(
            [
                f"real users (median of {len(users)})",
                "–",
                middle["median_words"],
                middle["lowercase_share"],
                "–",
            ]
            + [middle["question_share"], middle["short_share"], "–", "–"]
        )
    report = "\n\n".join(
        [
            "## Runs\n\n"
            + table(
                [
                    "run",
                    "status",
                    "company",
                    "messages",
                    "rejected",
                    "corrected in-attempt",
                    "writer calls",
                    "cost",
                    "solved",
                    "correct",
                    "grounded",
                ],
                overview,
            ),
            "## Name reuse against the rest of data/\n\n"
            + table(
                ["run", "company", "company word reused", "full names reused", "first names reused"], names
            ),
            "## Voice and timing\n\n" + table(["world", "messages", *STYLE], styles),
            "## Per author (new runs)\n\n"
            + table(["run", "person", "persona voice", "messages", *STYLE[:5], "typing target"], authors),
            "## Revisions per build attempt\n\n" + "\n\n".join(sections),
        ]
    )
    if args.output:
        args.output.write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
