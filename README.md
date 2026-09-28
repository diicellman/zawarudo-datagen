# Zawarudo

Generate related QA tasks first, build one supporting Slack world in reviewed groups,
then solve each task independently. Generation reviews are agentic; solver grading uses
one native LLM judge call.

## Run on a fresh machine

Requirements: [uv](https://docs.astral.sh/uv/), git, and a Prime account with Inference and Sandbox
access. Authenticate with `prime login` or set `PRIME_API_KEY`.

```bash
git clone <repo> && cd zawarudo-datagen && git checkout feat/scene-writer
uv sync --all-extras --all-groups
uv run --frozen python -m generators.worldgen_slack.check        # offline checks, no paid calls
uv run --frozen worldgen-slack --config configs/worldgen_slack/software.toml --dry-run
mkdir -p runs
uv run --frozen worldgen-slack --config configs/worldgen_slack/software.toml > runs/software.log 2>&1   # paid
```

- **Configs** (`configs/worldgen_slack/`): `software.toml`, `employee.toml` and `services.toml` each generate one
  10-task world (two groups of 5). These are the v2-09 run configs. `seeds.toml` rebuilds the seed packet, and
  `eval.toml` evaluates a released world. `configs/{eval,gepa,rl}` are Prime Lab templates.
- **Change `output` and `seed` for each new world.** A rerun of the same command resumes from `state.json`.
  Resume is refused if the config differs from the run's `run.json`.
- **Models:** per agent in `[env.<role>]` (synthesizer, builder, writer, judge, solver) and `[answer_judge]`.
- **Acceptance:** `[acceptance]` defaults to criterion floor .65, weighted floor .75, minor issues non-blocking.
- **Density:** `messages_per_task` (default 15).
- **Reference numbers (v2-09):** a world takes 30–40 min and about $1.6–2.2 in priced calls. Writer and solver
  calls (GLM) are unpriced.
- **GLM limit:** the account allows 8 concurrent GLM requests, so run **one world at a time**.
- **Follow a run:**
  `tail -F runs/software.log | grep -E '"event": "(premise_selected|candidate_finished|scenes_repaired|workspace_frozen)"|Traceback|exit='`.
  Stop with Ctrl-C (or SIGINT); the checkpoint is kept.
- **Inspect:**
  - `uv run --frozen python scripts/worldgen_slack/measure.py data/<run>` for style, revisions, cost and
    correctness;
  - `uv run --frozen python viewer/build.py data/<run> --output viewer/artifacts-<name>.html` builds an offline
    page. It contains private answers, so share it only with authorized people.
- **Seed excerpts:** `generators/worldgen_slack/seeds/slack-examples.json` holds 29 reviewed excerpts from
  `unionai/flyte-slack-data` and `spencer/software_slacks` (source licenses unspecified). They set the style
  texture only.

## Read the code in this order

| Location | Responsibility |
|---|---|
| `generators/worldgen_slack/env.py` | The pipeline: premise, catalog, plan → write → bind per group, final review, evaluate |
| `generators/worldgen_slack/agents/synthesizer.py` | Premise proposals, then the QA catalog, facts and personas |
| `generators/worldgen_slack/agents/builder.py` | Scene plan and evidence bindings (no message text) |
| `generators/worldgen_slack/agents/writer.py` | One tool-less writer chat per scene, in the personas' voices |
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

Rerun the same command to resume. Review attempts have separate bounded allowances
(`review_rounds`: 8 catalog attempts per run, 8 build and 4 final attempts per group), and
`research_budget_usd` stops new calls. Catalog defects return to the synthesizer and use the
catalog allowance; world/evidence defects return to the builder.
Each build review checks all built tasks, and every task gets final review against the
complete world. Exhaustion preserves an incomplete run.
World acceptance is decided in code from the judge's findings (`[acceptance]`). The judge marks
issues blocking or minor. Rejections whose issues name messages repair only those scenes, in the
judge's words, and are re-reviewed with focus. Other rejections re-plan. After a catalog repair that
leaves every scene's facts unchanged, the existing candidate is reviewed again without rebuilding.
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
uv run eval @ configs/worldgen_slack/eval.toml -n 1 -r 2 --no-push --plain
```

The single generator check includes retained Slack domain checks, native tool startup,
repair routing, bounded retries, restart, stale reviews, regression, and oracle separation.
The optional live check spends model/sandbox usage and retains clearly marked controlled
defects and real author/reviewer repair exchanges under `data/feedback-check/`.

Historical runs and `data_intermidiate/` are preserved. The retired source is captured in
`data/code-before-consolidation/source.tar.gz`; it is not an active implementation.
Local research reports and plans live in ignored `local-docs/`; generated runs remain
under ignored `data/` and `outputs/` directories.
