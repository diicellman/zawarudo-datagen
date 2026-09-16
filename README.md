# Zawarudo

Generate related QA tasks first, build one supporting Slack world in reviewed groups,
then solve each task independently. Generation reviews are agentic; solver grading uses
one native LLM judge call.

## Run

```bash
uv sync --all-extras --all-groups
uv run worldgen-slack --config configs/generation.toml --dry-run
uv run worldgen-slack --config configs/generation.toml
```

Prime Inference and Sandbox access must be configured. The default config runs three
related tasks in two groups. `generation-pilot.toml` requests ten tasks;
`generation-production-100.toml` requests 100 tasks and about 2,000 messages.

## Read the code in this order

| Location | Responsibility |
|---|---|
| `generators/worldgen_slack/env.py` | The pipeline: synthesize, build groups, final review, evaluate |
| `generators/worldgen_slack/agents/synthesizer.py` | Complete QA catalog and company facts |
| `generators/worldgen_slack/agents/builder.py` | Structured world data and evidence bindings |
| `generators/worldgen_slack/agents/judge.py` | Independent agentic catalog/world review |
| `generators/worldgen_slack/contracts.py` | Generation contracts and deterministic checks |
| `generators/worldgen_slack/store.py` | Artifacts, one restart checkpoint, summaries, and publication |
| `environments/worldgen_slack/worldgen_slack/slack/` | Slack data models, paginated reads, and actor visibility |
| `environments/worldgen_slack/worldgen_slack/taskset.py` | Finished taskset, fresh solver sessions, and `vf.Judge` answer grading |
| `environments/worldgen_slack/worldgen_slack/dataset.py` | Shared snapshot format and integrity-checked loading |

`generate.py` starts the native Env from validated TOML. Verifiers owns model execution,
harnesses, runtimes, traces, and episodes. Task hooks prepare files and capture results.
The small author task handles file setup; the inspection toolset runs trusted generation checks.
There is one pipeline, one Slack API, one taskset format, and one root uv lockfile.

The control-flow examples are Verifiers' [ProposerSolverEnv](references/verifiers/environments/proposer_solver/proposer_solver/taskset.py)
and [IsolatedAgenticJudgeEnv](references/verifiers/verifiers/v1/envs/agentic_judge/env.py).
Authors use native `Agent.interaction()` for feedback. Every generation review uses a fresh
`Agent.run()` with captured artifacts and the observable author trace. The judge cannot edit
committed candidates. Solver grading calls `vf.Judge` from a task reward; it starts no judge agent.

## Outputs and feedback

```text
data/<run>/
  run.json               # configuration and dependency identity
  state.json             # atomic restart checkpoint and remaining allowances
  artifacts/             # catalog and candidate versions, addressed by hash
  attempts/              # author outputs, checks, critiques, observable author traces
  traces/                # native agent traces, including failures and judge-call usage
  progress.jsonl         # live stages, repairs, and ten-second heartbeats
  frozen.json            # final reviewed catalog/candidate identities
  summary.json           # outcomes, sizes, repair counts, latency, and costs
  release/
    tasks.json           # public questions, actors, and shared snapshot references
    snapshot.json        # one Slack world
    answers.json         # private answers and reference evidence
    manifest.json        # identity and file hashes
```

Rerun the same command to resume. Five candidates are allowed per group, including the
initial one; build, upstream repair, and final review share the group allowance.
Catalog defects return to the synthesizer and invalidate dependent reviews;
world/evidence defects return to the builder. Upstream repair does not reset allowances.
Each build review checks all built tasks, and every task gets final review against the
complete world. Exhaustion preserves an incomplete run.
An existing candidate that already supports a group goes directly to independent review
when no repair feedback is pending. It consumes a review round without unnecessary rebuilding.
Reviewers can inspect missing evidence reads through `inspect_check`; finalization enforces
the same check before accepting an approval.

Catalog quality, world quality, and solver performance are separate signals. A valid task
stays in the dataset when its solver fails. Correctness and grounding must both pass for
solver success; private reference answers cannot substitute for observed evidence.

`answers.json` and generation artifacts are private. Solvers receive one question and
actor-scoped tools, never the release directory or sibling catalog.

## Check and evaluate

```bash
uv run python -m generators.worldgen_slack.check
uv run python -m generators.worldgen_slack.check --live-feedback
uv run eval @ configs/eval-generated.toml -n 1 -r 2 --no-push --plain
```

The single generator check includes retained Slack domain checks, native tool startup,
repair routing, bounded retries, restart, stale reviews, regression, and oracle separation.
The optional live check spends model/sandbox usage and retains clearly marked controlled
defects and real author/reviewer repair exchanges under `data/feedback-check/`.

Historical runs and `data_intermidiate/` are preserved. The retired source is captured in
`data/code-before-consolidation/source.tar.gz`; it is not an active implementation.
Local research reports and plans live in ignored `local-docs/`; generated runs remain
under ignored `data/` and `outputs/` directories.
