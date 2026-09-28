# Slack generation

The repository [README](../../README.md) contains the command, code map, output layout,
and evaluation instructions.

`GenerationEnv.run()` is the entry to the algorithm. Generation is backward: the catalog of
questions and facts comes first, and the world is built around it.

1. **Premise.** The synthesizer proposes `premise_count` distinct companies. Code picks one
   with the run seed. Names used by other catalogs under `corpus` (default `data/`) are
   passed in as data to avoid, so variety comes from code rather than model defaults.
2. **Catalog.** The synthesizer writes people, a persona per person (role, seniority,
   timezone, voice), facts and tasks. The judge reviews it.
3. **Plan.** Per group, the builder plans conversations and scenes: participants, start and end,
   situation, length, and beats that place catalog facts. It also declares recurring `details`
   as `{value, since, at}`: `since` is when the detail becomes known, and `at` is a scheduled
   moment people will mention. Scenes that depend on each other do not overlap. It writes no
   message text.
4. **Write.** One writer agent per changed scene writes the messages in the participants'
   voices. Scenes are written in waves: a scene waits only for earlier scenes its participants could
   have seen, and up to six writers run at once. Each writer sees earlier messages in its conversation,
   the last day of messages its participants saw elsewhere, the plan details already known, every
   concrete detail and open commitment from earlier visible scenes, and the facts true at its start.
   Every time in the brief and every line is on one scene clock (the first participant's timezone);
   code converts to UTC. `valid_from` is when a fact is first stated in Slack: a fact with a time of
   day is stated in exactly that minute. A reply to a reply joins its thread. Writers run at low
   reasoning effort.
5. **Assemble and bind.** Code turns scenes into messages with deterministic IDs. The builder
   binds evidence and gold routes, or requests scene rewrites. Each attempt uses a fresh builder
   conversation on the group's VM, so context stays bounded while files persist.
6. **Review.** An independent judge reviews every built task against the whole world and marks
   each issue blocking or minor. Code accepts the world from the judge's findings and the
   `[acceptance]` floors. A rejection whose issues name messages repairs only those scenes, from
   their previous version and the judge's words, and the re-review focuses on them. Other
   rejections return to the plan. Changing a scene or its `revision_note` rewrites only that scene;
   unchanged scenes are cached in `state.json`.

Authors write one file per phase (`/task/premise.json`, `catalog.json`, `plan.json`, `bind.json`).
Deterministic errors (schema, plan references, gold-route replay) go back to the author up to twice
within the same attempt and are logged in `attempts/*/corrections.json`; only then does the attempt
count as rejected. A judge run that fails, such as with a malformed verdict, is rerun once.
`language` (default `English`) sets the working language of the premise, catalog, questions
and messages; the company itself can be set anywhere.

The synthesizer owns QA and facts. The builder owns the plan and bindings. Writers own the
text. Contracts use half-open fact intervals `[valid_from, valid_until)`. Reviews must cover
all requested claims against the exact candidate. Private facts guide generation but do not
prove that the Slack world supports an answer. `inspect_check` reports `activity.style`:
voice and timing statistics overall and per author, which the judge compares with the personas.

At the intended 100-task scale, each build review checks all built tasks. This favors
simple, conservative regression coverage over a separate dependency-tracking system.
Catalog edits invalidate approvals; consumed review allowances are kept.

The configured budgets stop new calls after reported model spend reaches the limit.
In-flight calls, sandbox charges, and unpriced calls can exceed reported totals. Failed
attempts remain part of research accounting. No solver-success signal rewards the authors.

## Measuring runs

```bash
uv run --frozen python scripts/worldgen_slack/measure.py data/v2-09/software --baseline data/v2-01/software
```

The report covers name reuse against the rest of `data/`, voice and timing statistics
(including pooled legacy worlds and the real seed excerpts), revisions per build attempt,
writer calls, cost and solving.

## Conversation seeds

Prepare reference examples separately from world generation:

```bash
uv run --frozen python scripts/worldgen_slack/prepare_seeds.py --config configs/seed-preparation-expanded.toml --dry-run
uv run --frozen python scripts/worldgen_slack/prepare_seeds.py --config configs/seed-preparation-expanded.toml
```

The selection TOML specifies dataset revisions and row ranges (`start` inclusive, `end`
exclusive). Requests are capped at 100 rows and 1 MiB, and pinned revisions must still be
current. Raw responses stay under ignored `data/seeds/raw/`. Inspect and redact the draft,
then save the reviewed packet. The reviewed packet used by the v2 configs is committed at
`generators/worldgen_slack/seeds/slack-examples.json`. Packets are capped at 48 excerpts, 512 messages,
128,000 text characters and 256 KiB. Both source datasets have unspecified licenses.

Enable seeds in a generation TOML:

```toml
[seed_data]
path = "generators/worldgen_slack/seeds/slack-examples.json"
```

Each writer call receives two compact `speaker: text` excerpts, chosen deterministically
from the run seed and scene ID, as style references. They are never instructions or facts.
Authors, judges and solvers never receive the packet. The packet is hashed into `run.json`,
and resume rejects a changed packet.

