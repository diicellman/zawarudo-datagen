# WorldGen Slack

A native `verifiers.v1` environment for generating read-only Slack QA worlds:

```text
synthesize → build → validate → solve → judge
```

`SlackDataGenerationEnv` owns only this sequence. Slack models/API/validation own semantics.
Verifiers owns execution, native retries, traces, runtimes, scoring hooks, and the `vf.Toolset`
transport. The standalone generator owns policy, resume, and atomic release writes. Janitor owns
post-run viewing.

## Run

From the repository root:

```bash
uv run --frozen worldgen-slack generate --config configs/generation.toml
```

The CLI schedules a target number of accepted worlds with a fixed worker pool, an exact attempt cap,
durable resume, deduplication, and transactional release writes. The environment uses native
Verifiers execution and trace ownership. Whole-episode retries are zero; only the world judge has
one narrowly scoped native operational retry. No custom retry state or trace-status mutation exists.

## Trust and correctness

All roles and generated-code checks use immutable Prime VMs with default-deny egress. The public
builder check and private candidate check share `slack/validate.py`, but public feedback is never
authoritative. The host accepts only bounded snapshots and reruns source, ontology, visibility,
evidence, gold-call, determinism, variation, stability, size, and leakage gates.

The solver sees only its question, actor, validated snapshot, and the read-only Slack Toolset. Its
natural last reply is scored by a host-side semantic claim judge with the private oracle. The
world-only judge receives no solver response, trace, reward, or verdict; it must make an empirically
recorded Slack call and write a fresh bounded rubric verdict. Exact/date/list matching is diagnostic
only and cannot override the semantic reward.

## Release

Generator schema `worldgen-slack.dataset.v3` and release table schema 2 provide:

- `manifest.json`, `attempts.jsonl`, and `artifacts.jsonl` for Janitor;
- `run_manifest.json`, `summary.json`, `qualification_report.json`, and durable `progress.jsonl`;
- `dataset.jsonl`, solver-safe `public_tasks.jsonl`, and host-only `private_oracles.jsonl`;
- derived schemas plus `interface.json` and `builder_guide.md`;
- accepted `worlds/<instance_id>/` and optional separate `rejected/<attempt_id>/` artifacts.

Accepted writes use a transaction journal and atomic directory replacement. Resume validates paths,
symlinks, hashes, identities, public/private correspondence, and immutable provenance. The generated
taskset loads `snapshot.json` only and never executes `world.py`.

```bash
uv run eval @ configs/eval-generated-prime.toml --env.taskset.release-dir data/slack-qualification-v1 --dry-run True --no-push --plain

(cd janitor && cargo run -- release ../data/slack-qualification-v1)
```

## Development

```bash
uv run --project environments/worldgen_slack pytest -q environments/worldgen_slack/tests
uv run --project environments/worldgen_slack ruff check environments/worldgen_slack/worldgen_slack environments/worldgen_slack/tests
uv run --project environments/worldgen_slack ruff format --check environments/worldgen_slack/worldgen_slack environments/worldgen_slack/tests
```

Set `WORLDGEN_SLACK_EXTERNAL_SMOKE=1` only when Prime Inference, Sandbox, tunneling, model, and pinned
RLM access are available.
