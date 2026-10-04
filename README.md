# Zawarudo

Generate a Slack workspace with related QA tasks. One author agent writes the world in time order: the company,
its people and channels, a ledger of facts and events, every conversation day by day, then the tasks. An agentic
judge reviews the world mid-run and at the end, and a GLM solver tries each task. The world is one SQLite file;
code is its only writer and checks every write.

## Run on a fresh machine

Requirements: [uv](https://docs.astral.sh/uv/), git, and a Prime account with Inference and Sandbox access.
Authenticate with `prime login` or set `PRIME_API_KEY`.

```bash
uv sync --all-extras --all-groups
uv run --frozen python -m generators.worldgen_slack.check        # offline checks, no paid calls
uv run --frozen worldgen-slack --config configs/worldgen_slack/worldgen.toml --dry-run
mkdir -p runs
uv run --frozen worldgen-slack --config configs/worldgen_slack/worldgen.toml > runs/world.log 2>&1   # paid
```

- **Config** (`configs/worldgen_slack/worldgen.toml`): the sector, the seed, the calendar, the workspace's shape
  (`[activity]`), the task count and taxonomy, and one model per agent (`[env.author]`, `[env.judge]`,
  `[env.solver]`, `[answer_judge]`). `eval.toml` evaluates a released world. `configs/{eval,gepa,rl}` are Prime Lab
  templates.
- **Change `output` and `seed` for each new world.** Rerunning the same command resumes from `state.json`; resume is
  refused if the config differs from the run's `run.json`.
- **Spend:** `research_budget_usd` stops new calls once the reported spend reaches it.
- **GLM limit:** the account allows 8 concurrent GLM requests, so run **one world at a time**.
- **Follow a run:** `tail -F runs/world.log | grep -E '"event": "(day_closed|probe|candidate_finished|workspace_frozen)"|Traceback'`.
- **Stop a run** with Ctrl-C or `kill <pid>`: it unwinds, deletes its VM and keeps its checkpoint; rerun to resume.
  Its sandboxes carry the label `worldgen-<run>-<world>` (`prime sandbox list`), in case one is left behind.
- **Inspect:**
  - `uv run --frozen python scripts/worldgen_slack/measure.py data/<run>` for the scorecard, style, the author's
    days and tool calls, probes and cost;
  - `uv run --frozen python viewer/build.py data/<run> --output viewer/artifacts-<name>.html` builds an offline
    page. It contains private answers, so share it only with authorized people.

## Read the code in this order

| Location | Responsibility |
|---|---|
| `generators/worldgen_slack/env.py` | The episode: setup, the plan, each day, mid-run reviews, tasks, probes, the final review |
| `generators/worldgen_slack/chronicle.py` | The rules of a world written in time order: plan, post, advance, close a day, revise, add a task |
| `generators/worldgen_slack/agents/world.py` | The author's memory (rendered by code), its world tools, its turns |
| `generators/worldgen_slack/agents/synthesizer.py` | The setup documents: premise and organization |
| `generators/worldgen_slack/agents/judge.py` | The agentic judge: task and world reviews, with inspection tools |
| `generators/worldgen_slack/contracts.py` | Cast, organization, agenda, task checks (T1-T7), text rules, verdicts |
| `generators/worldgen_slack/store.py` | The run directory: checkpoint, attempts, traces, summary, release |
| `environments/worldgen_slack/worldgen_slack/db.py` | The world file: schema, rules checked on every write, actor-scoped reads |
| `environments/worldgen_slack/worldgen_slack/taskset.py` | The released taskset: fresh solver sessions and `vf.Judge` grading |

`generate.py` starts the native Env from validated TOML. Verifiers owns model execution, harnesses, runtimes,
traces and episodes. The author is an rlm coding agent in one Prime VM for the whole world, with one interaction
per block of work; its world tools run on the host. The judge cannot edit the world; its issues go back to the
author.

## Outputs

```text
data/<run>/
  run.json               # configuration and dependency identity
  state.json             # atomic restart checkpoint and remaining allowances
  world.sqlite           # the world as written so far
  attempts/<attempt>/    # each block's world snapshot, the author's notes, reviews and verdicts
  traces/                # native agent traces, with usage
  progress.jsonl         # events: turns, closed days, probes, reviews
  world-calls.jsonl      # every world tool call and its outcome
  summary.json           # outcome, task mix, solve rates, difficulty, cost
  release/
    world.sqlite         # the world, without the answer key
    tasks.json           # public questions and actors
    answers.json         # private answers and reference evidence
    manifest.json        # identity and file hashes
```

World and task quality and solver performance are separate signals. A valid task stays in the dataset when its
solver fails. `answers.json` and generation artifacts are private; solvers receive one question and actor-scoped
tools, never the release directory.

## Check and evaluate

```bash
uv run --frozen python -m generators.worldgen_slack.check
uv run --frozen python viewer/check.py
uv run eval @ configs/worldgen_slack/eval.toml -n 1 -r 2 --no-push --plain
```

Local research reports and plans live in ignored `local-docs/`; generated runs live under ignored `data/`.
