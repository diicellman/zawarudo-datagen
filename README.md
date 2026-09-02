# Zawarudo Data Generation

Prime-native synthetic data generation for read-only Slack question-answering worlds.
The fixed protocol is:

```text
synthesize → build → validate → solve → judge
```

## Ownership

- **Env owns sequence.** `SlackDataGenerationEnv` contains the five explicit stages and short circuits.
- **Slack owns meaning.** `slack/models.py`, `api.py`, `tools.py`, and `validate.py` define data, behavior, native actions, and hard gates.
- **Verifiers owns execution.** Native agents, retries, traces, toolsets, runtimes, rewards, and metrics drive every episode.
- **Generator owns persistence.** `generate.py` applies deduplication and quality policy; `dataset.py` commits and audits releases.
- **Janitor owns observation.** It opens a completed release without affecting canonical artifacts.

No workflow engine, registry, plugin layer, or project-owned MCP server is used.

## Setup

Requirements: Python 3.12+, `uv`, and Prime Inference/Sandbox access.

```bash
prime lab setup
uv sync --all-packages --all-groups
prime login
prime lab doctor
```

Do not put credentials in TOML files or forward them into model sandboxes.

## Generate

```bash
uv run --frozen worldgen-slack generate --config configs/generation.toml
```

The config targets accepted worlds, not attempted seeds. Optional CLI overrides are
`--target-accepted`, `--max-attempts`, `--concurrency`, `--run-id`, and `--output`. The generator
keeps a fixed worker pool active until the accepted target or exact attempt cap is reached, then
drains work already in flight.

Whole-episode retries are zero. Only the world judge has one narrowly scoped native retry for
operational failures. Expected synthesis, deterministic, semantic, and quality rejections return
normally and are recorded with exact taxonomy. Provider, runtime, tunnel, and trusted-code failures
remain native failed episodes. Progress is fsynced to `progress.jsonl` and printed at least every ten
seconds.

Resume with the same command and immutable run configuration. Recorded seeds are skipped.
Configuration, source, provenance, path, or hash mismatches fail loudly.

## Release layout

The current generator contract is `worldgen-slack.dataset.v3` with release table schema 2. Older,
append-incompatible generator releases are rejected rather than silently mixed.

```text
manifest.json                 # Janitor table discovery
run_manifest.json             # immutable generator provenance
attempts.jsonl                # one stable scalar row per terminal attempt
artifacts.jsonl               # one path and SHA-256 row per artifact
dataset.jsonl                 # accepted private catalog
public_tasks.jsonl            # solver-safe snapshot rows
private_oracles.jsonl         # host-only answer and gold references
interfaces/slack.readonly.v1/ # action contract, derived schemas, builder guide
worlds/<instance_id>/         # accepted source, snapshot, validation, traces, verdict
rejected/<attempt_id>/        # optional, always separate
summary.json
qualification_report.json       # independently gated qualification result
progress.jsonl                  # durable stage, branch, retry, and heartbeat events
```

Accepted commits are insert-only and transactional. Public rows never contain answers, claims,
evidence requirements, or gold calls. Generated evaluation hashes and loads `snapshot.json`; it
never imports or executes `world.py`.

Inspect a release with Janitor:

```bash
(cd janitor && cargo run -- release ../data/slack-qualification-v1)
```

Janitor registers the manifest-declared `attempts` and `artifacts` JSONL files as read-only DuckDB
views. Deleting viewer state cannot change the release.

## Rewards and acceptance

These are separate:

- deterministic validation is a builder metric and a hard gate;
- a narrow host-side semantic judge scores solver claims and accepts equivalent paraphrases;
- a world-only agentic judge uses empirical Slack reads to score six weighted quality criteria and
  the hard gates `task_unambiguous` and `world_supports_task`;
- exact/date/list matching remains diagnostic and cannot override semantic correctness;
- deterministic or world hard-gate failure zeros final world quality while retaining raw criteria.

The authoritative named rewards are `semantic_correctness` and `world_quality`. Release acceptance
also requires the configured solver threshold, world threshold, every per-criterion floor,
deduplication, and a solved semantic status. Challenging and rejected rows never count toward the
accepted target.

## Security boundary

Generated `world.py` runs only in fresh Prime VMs. Every model and candidate runtime uses an
immutable image, `vm=true`, `allow=[]`, and `block=["*"]`. Public feedback is non-authoritative.
The host downloads bounded snapshots, rebuilds strict Pydantic models, and reruns hard gates.
Solvers and judges receive validated snapshots through a thin native `vf.Toolset`; they never
execute generated source. Private oracle data is supplied only to the judge and host-side release.

## Evaluate accepted snapshots

```bash
uv run eval @ configs/eval-generated-prime.toml --env.taskset.release-dir data/slack-qualification-v1 -n 3 -r 1 -c 2 --no-push --rich False --plain
```

Use `--dry-run True` for configuration-only validation.

## Checks

```bash
uv run pytest -q
uv run --project environments/worldgen_slack ruff check   environments/worldgen_slack/worldgen_slack environments/worldgen_slack/tests
uv run --project environments/worldgen_slack ruff format --check   environments/worldgen_slack/worldgen_slack environments/worldgen_slack/tests
uv lock --check
(cd janitor && cargo fmt --all --check && cargo clippy --all-targets --all-features -- -D warnings)
(cd janitor && cargo test --all-targets --all-features)
```

The live Prime test is skipped unless `WORLDGEN_SLACK_EXTERNAL_SMOKE=1`. It also requires Prime
Inference, Sandbox provisioning, `z-ai/glm-5.2`, tunneling, and access to the pinned RLM revision.
