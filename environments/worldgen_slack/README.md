# WorldGen Slack

A standalone Prime-native generator for read-only Slack question-answering worlds. One synthesized
scenario and QA contract maps to one model-written `world.py`. Accepted items can be loaded as a
normal `verifiers.v1` taskset without executing generated Python.

## Setup

From the repository root:

```bash
prime lab setup
uv sync --all-packages --all-groups
prime login
prime lab doctor
```

Prime credentials are read through `EvalClientConfig`; do not put credentials in the generation
configuration or forward them into a model harness.

## Generate data

From the repository root:

```bash
uv run worldgen-slack generate \
  --config configs/generation.toml \
  --count 10 \
  --output data/slack-v0
```

The command uses `SlackDataGenerationEnv` through `Env.serving()` and `Env.run_slot()`. It does not
shell out to `eval`. The synthesizer, builder, candidate validator, solver, and judge all use fresh
`PrimeConfig(vm=True)` sandboxes with default-deny external egress. Model calls use
`EvalClientConfig` and the active Prime credentials. The fixed, trusted Slack MCP service stays
host-local and Verifiers framework-tunnels it to each isolated solver or judge VM; this is not a
model-execution or generated-code fallback.

`configs/generation.toml` controls model settings, concurrency, whole-item retries, judge score
weights, criterion floors, the inclusive aggregate threshold, accepted semantic statuses, and
optional rejected-artifact retention. The judge has one internal retry for verdict protocol errors.
Operational failures use the item retry policy.

## Output

The output directory contains:

- `attempts.jsonl`: every completed item and its typed outcome, trace references, runtime records,
  raw judge scores, normalized score, and persistence decision.
- `dataset.jsonl`: accepted catalog rows.
- `public_tasks.jsonl`: solver-safe rows with no oracle fields.
- `private_oracles.jsonl`: host-side answers, claims, and gold references.
- `worlds/<instance_id>/`: source, validated snapshot, contract, traces, verdict, and manifest.
- `interfaces/slack.readonly.v1/`: the fixed interface definition and hashes.
- `summary.json`: acceptance and failure counts plus judge-score distributions.
- `rejected/`: optional rejected artifacts, kept separate from accepted worlds.

The release root is exactly the CLI `--output` path, or the TOML `output` value when the flag is
absent. `run_id` is an internal namespace, not the directory name. Attempt IDs use
`<run_id>--seed-<8 digits>--attempt-<4 digits>`. Accepted world directories use the sanitized
synthesized `task_slug` plus the first 12 characters of the host-computed canonical contract hash.

Accepted directories are built in temporary sibling directories and atomically renamed. Writes are
insert-only. Resume validates all accepted references and skips seeds already recorded in
`attempts.jsonl`.

## Evaluate generated tasks

The generated taskset loads and hashes `snapshot.json`; it never imports `world.py`.

```bash
uv run eval @ configs/eval-generated-prime.toml \
  --env.taskset.release-dir data/slack-v0 \
  -n 3 -r 1 -c 2 \
  --no-push --rich False --plain
```

For a configuration-only check, add `--dry-run True`. The `worldgen-slack-generation` plugin can be
used with standard eval for timing traces, but eval does not call the standalone atomic release
commit callback. Use `uv run worldgen-slack generate ...` to write datasets.

The plugin IDs are:

- `worldgen-slack-generation`
- `worldgen-slack-generated`

## Trust boundary

The builder receives only the frozen scenario, QA contract, fixed model/API documentation, an empty
`world.py` stub, and a reduced public checker. Generated source executes only in a separate fresh
Prime VM. The host downloads bounded JSON snapshots and revalidates the fixed Pydantic ontology,
references, visibility, evidence, gold calls, determinism, hidden-seed stability, size bounds, and
anti-leakage checks.

The blind generation solver receives only the question and a validated Slack snapshot through the
fixed read-only Toolset. The judge receives private audit context in its own VM. Its chat reply is not
authoritative: `JudgeTask.finalize()` reads and validates `/tmp/worldgen_judge_verdict.json` while the
runtime is alive and requires at least one successful Slack call.

## Development checks

```bash
uv sync --project environments/worldgen_slack --all-groups
uv run --project environments/worldgen_slack pytest -q environments/worldgen_slack/tests
uv run --project environments/worldgen_slack ruff check environments/worldgen_slack
uv run --project environments/worldgen_slack ruff format --check environments/worldgen_slack
```

The Prime VM family is pinned to `python:3.12-slim@sha256:2c941e860699f878900b0edc2403613c234d4b32eda3cc9fa7036991a2a63c4a`. The package pins `verifiers==0.3.0`, `pydantic==2.13.4`, and the RLM revision
`e0080b25afddbc71ecf5476e4f50c3c63edddedb`.
