# Zawarudo Data Generation

Prime-native synthetic data generation for read-only Slack question-answering worlds. The generator
creates a QA contract, writes and validates one `world.py`, probes it with a blind solver, filters it
with an empirical agentic judge, and exports accepted snapshots as a normal Verifiers taskset.

Implementation details and trust boundaries are documented in
[`environments/worldgen_slack/README.md`](environments/worldgen_slack/README.md).

## Prerequisites

- Python 3.12 or newer
- [`uv`](https://docs.astral.sh/uv/)
- Prime CLI access and a Prime account with Inference and Sandbox access

## Setup

From the repository root:

```bash
prime lab setup
uv sync --all-packages --all-groups
prime login
prime lab doctor
```

Confirm that the local command and plugins resolve:

```bash
uv run worldgen-slack --help
uv run eval worldgen-slack-generated --dry-run True --no-push --plain
```

Prime credentials stay in the Prime CLI credential store. Do not add API keys to TOML files or
forward them into model sandboxes.

## Configure generation

The checked-in configuration is [`configs/generation.toml`](configs/generation.toml). Its main
settings are:

- `run_id`: internal manifest and attempt-ID namespace;
- `count`: number of seed tasks in the run;
- `concurrency`: items generated at the same time;
- `output`: release directory used when `--output` is absent;
- `max_item_retries`: whole-item operational retries;
- role models, turn limits, immutable Prime image, and RLM revision;
- quality weights, criterion floors, acceptance threshold, and accepted semantic statuses.

The CLI can override `count` and `output`. Change other values in a copied TOML configuration.

## Generate data

Run a one-item smoke:

```bash
uv run worldgen-slack generate \
  --config configs/generation.toml \
  --count 1 \
  --output runs/prime-smoke
```

Run ten seeds into the default release location:

```bash
uv run worldgen-slack generate \
  --config configs/generation.toml \
  --count 10 \
  --output data/slack-v0
```

To resume, run the exact same command with the same source, configuration, count, run ID, and output
path. Completed terminal seeds are skipped. A source or provenance mismatch fails instead of mixing
releases.

Inspect the result:

```bash
cat data/slack-v0/summary.json
cat data/slack-v0/attempts.jsonl
find data/slack-v0/worlds -maxdepth 2 -type f | sort
```

## Output names

`runs/` and `data/` have no special naming behavior. The release root is exactly the CLI `--output`
path, or the TOML `output` value when the CLI flag is absent. Names such as `prime-smoke` are chosen
by the caller.

`run_id` does not name the directory. It appears in manifest and attempt IDs such as:

```text
slack-generator-v0--seed-00000003--attempt-0001
```

Accepted world directories use a host-owned identity:

```text
<sanitized-task-slug>--<first-12-characters-of-contract-hash>
```

A release contains `run_manifest.json`, `summary.json`, all-attempt and public/private JSONL files,
the fixed interface assets, and `worlds/<instance_id>/` directories with source, snapshots,
validation, verdicts, manifests, and four role traces.

Local `runs/`, `data/`, and `references/` trees are ignored by Git. The `.log` and `.pid` files used
for long local runs are operator conveniences, not release artifacts.

## Evaluate generated tasks

Evaluate three accepted snapshots with Prime Inference and Prime VMs:

```bash
uv run eval @ configs/eval-generated-prime.toml \
  --env.taskset.release-dir data/slack-v0 \
  -n 3 -r 1 -c 2 \
  --no-push --rich False --plain
```

Validate configuration without running models:

```bash
uv run eval @ configs/eval-generated-prime.toml \
  --env.taskset.release-dir data/slack-v0 \
  --dry-run True --no-push --plain
```

`worldgen-slack-generated` evaluates already-built snapshots and never executes `world.py`.
`worldgen-slack-generation` can be driven by `uv run eval` for timing and trace inspection, but the
standard eval runner does not call the standalone atomic dataset-commit callback. Use
`worldgen-slack generate` to create a release.

## Development checks

```bash
uv run --project environments/worldgen_slack pytest -q environments/worldgen_slack/tests
uv run --project environments/worldgen_slack ruff check environments/worldgen_slack
uv run --project environments/worldgen_slack ruff format --check environments/worldgen_slack
uv lock --check
```
