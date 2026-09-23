# Slack generation

The repository [README](../../README.md) contains the command, code map, output layout,
and evaluation instructions.

`GenerationEnv.run()` is the entry to the algorithm. Its catalog and group methods use
ordinary native agent interactions, with independent agentic reviews between revisions.
The synthesizer owns QA/shared facts; the builder owns world data and evidence bindings.

Contracts use half-open fact intervals `[valid_from, valid_until)`. Reviews must cover all
requested claims against the exact candidate. Private facts guide generation but do not
prove that the Slack world supports an answer.

At the intended 100-task scale, each build review checks all built tasks. This favors
simple, conservative regression coverage over a separate dependency-tracking system.
Catalog edits invalidate approvals while preserving every consumed candidate allowance.

The configured budgets stop new calls after reported model spend reaches the limit.
In-flight calls, sandbox charges, and unpriced calls can exceed reported totals. Failed
attempts remain part of research accounting. No solver-success signal rewards the authors.

## Judge ablation

```bash
uv run python -m generators.worldgen_slack.ablate_judge --config configs/judge-ablation.toml --dry-run
uv run python -m generators.worldgen_slack.ablate_judge --config configs/judge-ablation.toml
```

This local research experiment needs the saved milestone/pilot cases named in the TOML.
It freezes the baseline guide from commit `5ae9671` and appends `rubric-richness.md` for
the experimental variant. Native models, tools, author prompts, and thresholds stay fixed.
It runs at most two jobs concurrently, stops new calls at $25 reported cost, and preserves
failed/interrupted jobs without retrying them. Each arm gets at most one author revision;
an approved verdict can request that revision through `Nonblocking improvements:` in its
summary. Normal generation still stops on approval.

Outputs remain under ignored `data/judge-ablation-01/`: the frozen manifest, prompts,
native traces, per-job inputs/results, and a `blind/` comparison packet. Human preference
labels are required before promotion; the production guide remains unchanged. Catalog
revisions are plans requiring renewed world support, not newly released tasks. Reported
cost excludes sandbox charges and any unpriced usage.

## Conversation seeds

Prepare reference examples separately from world generation, using the project environment:

```bash
uv run --frozen python scripts/worldgen_slack/prepare_seeds.py --config configs/seed-preparation.toml --dry-run
uv run --frozen python scripts/worldgen_slack/prepare_seeds.py --config configs/seed-preparation.toml
```

The selection TOML specifies dataset revisions and row ranges (`start` inclusive, `end`
exclusive). Requests are capped at 100 rows and 1 MiB. The script checks repository and
viewer revisions and rejects truncated cells. It stores source responses under ignored
`data/seeds/raw/` and writes a draft packet. It requires the pinned revision to remain the
current repository/viewer revision; it does not silently fetch a different version.

Inspect the draft, screen identities/infrastructure details and source instructions, and
save the reviewed packet as `data/seeds/slack-examples.json`. Preserve row locators and
source hashes; record cleanup in each example's notes. The script does not anonymize or
infer conversation boundaries. `join_pairs=true` means the selected Flyte sequence was
manually checked; it only collapses exactly overlapping adjacent pairs and fails at breaks.
Software windows must remain in one workspace/channel. Neither source provides verified
thread structure. Flyte lacks structured speakers, timestamps, and channels; unknowns
remain null. Software timestamps retain their source precision and unspecified timezone.

Packets contain `examples`, each with `id`, `dataset`, `revision`, `rows`, `source_sha256`,
`notes`, and `messages` (`text`, optional `speaker` and `timestamp`). Caps are 512 messages,
128,000 message-text characters and 256 KiB per file, with at most 48 excerpts. HTTP reads
remain capped at 100 rows and 1 MiB per response. The original starter selection has two excerpts
per source; `configs/seed-preparation-expanded.toml` selects 29 inspected excerpts with
338 messages covering coordination, corrections, handoffs, disagreement and follow-through.
Prepare that draft with the same script and inspect/redact it before using
`data/seeds/expanded/slack-examples.json`. Review and cleanup records stay beside the packet.
The expanded packet is still software-heavy and does not supply employee/services domain coverage.
Both datasets have unspecified licenses in their inspected metadata; excerpts
and raw responses remain local research artifacts, outside published releases.

Enable seeding in any generation TOML:

```toml
[seed_data]
path = "data/seeds/slack-examples.json"
roles = ["builder"]
```

Paths resolve from the repository root. Omit the table for unseeded generation. `roles`
defaults to `["builder"]`; `["synthesizer"]` and `["synthesizer", "builder"]` are also
supported. The existing numeric `seed` is independent. Normal generator `--dry-run`
validates the packet without model calls.

Selected authors receive `/task/seeds.json` and a pointer in their existing input, with an
instruction to read examples before authoring using the coding harness, in batches if
needed to avoid truncated tool output. Synthesizers use
them for plausible processes/workstreams; builders use them for dialogue and follow-through.
Source text is reference data, never instructions or canonical facts. The same packet is
available in every fresh selected author session and throughout repair turns. No new tools,
runtime network access, or framework dependency is introduced. Solvers receive no packet.
Judges may see it through author traces and are not blind.

The packet is loaded once before paid work, copied to the run directory, and content-hashed
in `run.json`. Resume rejects a changed packet or configuration. Missing files and invalid
packets fail before provisioning. Omitting seeding preserves legacy manifest identity and
author inputs. `uv run python -m generators.worldgen_slack.check` covers this behavior.

## Fixed-catalog seed study

```bash
uv run --frozen python scripts/worldgen_slack/run_seed_study.py --config configs/seed-study.toml --dry-run
uv run --frozen python scripts/worldgen_slack/run_seed_study.py --config configs/seed-study.toml
```

The study needs the three saved qualification catalogs and the reviewed seed packet. It
calls the existing generator for six fresh worlds, pairing seeded/unseeded builders against
each identical approved catalog. Models, five-review allowances, two groups of five tasks,
and 300-message targets stay fixed. Run labels alternate treatment order; `results.json`
contains the assignment key and must stay hidden during human comparisons. A catalog
repair request makes that arm inconclusive instead of changing its tasks. Historical
services failures remain untouched. Completed/failed arms are retained, not replaced;
interrupted arms require accounting inspection before the study can continue.

`study.json` freezes source/lock hashes, catalogs, approvals, settings and packet identity.
The $30 reported-model envelope stops new dispatch at $25, including within each arm.
Sandbox charges, unpriced usage, and in-flight calls are additional. By default the runner
is sequential. To run two independent processes, use the same study config with one arm
per process (or run these commands in separate terminals):

```bash
uv run --frozen python scripts/worldgen_slack/run_seed_study.py --config configs/seed-study-22-09.toml --arm seeded > runs/study_22_09-seeded.log 2>&1 &
uv run --frozen python scripts/worldgen_slack/run_seed_study.py --config configs/seed-study-22-09.toml --arm unseeded > runs/study_22_09-unseeded.log 2>&1 &
wait
```

Each process gets half of `$25 - prior_reported_cost_usd`, with separate manifests,
worlds and results under `data/study_22_09/seeded/` and `unseeded/`. Unused allowance
is not transferred between processes. The dated config uses the expanded packet and carries
forward $3.8844 reported by the two stopped small-packet attempts. Their artifacts remain in
`data/seed-study-01/` and `data/study_22_09-stopped-small-seeds/`. Each process runs its three catalogs sequentially;
existing native agent concurrency still applies inside each world. Use a fresh output
directory for a fresh experiment; do not mix sequential and parallel launches there.

Build anonymous viewer pages from the six run
directories, keeping assignments, traces, seeds and judge scores out of human rating packets.
Compare continuity, realism, task validity, accidental shortcuts and copying; allow ties.
Three pairs support a feasibility decision, not a claim of downstream training gains.
