# Zawarudo

Generate a Slack workspace, then QA tasks on it, for training agents that answer colleagues' questions from Slack.
The pipeline has two parts:

1. **The world run** (`worldgen-slack`). One author agent writes the world in time order: the company, its people
   and channels, a ledger of facts and events (with the board: the facts each planned task will rest on), then every
   conversation day by day. An agentic judge reviews the world mid-run and at the end. The world is one SQLite
   file; code is its only writer and checks every write. The run also writes tasks today, in its own task phase.
2. **The forge** (`worldgen-slack forge`). It writes tasks on a finished world, which it never changes. Round by
   round, a proposer writes candidate tasks; code checks them, the answer judge grades the wrong answers code
   builds for each, a GLM solver tries each 8 times, a stronger witness tries the hard ones, and the judge reviews
   them. An archive keeps what lands, and tells the proposer how the rest went wrong.

## Run on a fresh machine

Requirements: [uv](https://docs.astral.sh/uv/), git, the [DuckDB CLI](https://duckdb.org/docs/installation/)
(1.4 or later) to build the seed files, and a Prime account with Inference and Sandbox access. Authenticate with
`prime login` or set `PRIME_API_KEY` (a `.env` file in the repository root works too).

```bash
uv sync --all-extras --all-groups
uv run --frozen python -m generators.worldgen_slack.check        # offline checks, no model calls
```

### 1. Build the seed files (once)

The cast and the way people type come from two public Hugging Face datasets, read at pinned revisions:
[Nemotron-Personas-USA](https://huggingface.co/datasets/nvidia/Nemotron-Personas-USA) (CC-BY-4.0) for the people,
and [spencer/software_slacks](https://huggingface.co/datasets/spencer/software_slacks) for how real Slack users
type and how long they take to reply. They are not in the repository; one script builds them into `data/seeds/`:

```bash
mkdir -p data/seeds
duckdb < scripts/worldgen_slack/personas.sql      # needs network; a few minutes
```

| File | What it holds |
|---|---|
| `personas-usa.jsonl` | 25,248 office personas over 79 occupations: name, age, city, state and home timezone, background, skills, hobbies |
| `typing-profiles.jsonl` | 2,445 real users' typing statistics (message length, short and long shares, questions, lowercase starts, emoji); users are keyed by a digest |
| `reply-gaps.json` | 1,001 quantiles of the seconds between consecutive messages in a channel |

The script is deterministic (pinned revisions, every selection ordered by a hash). Built with DuckDB 1.4.4, the
files' sha256 are `68300a9a…9016`, `1fc89dce…fe41` and `57eb5581…aace`.

### 2. Write a world

```bash
uv run --frozen worldgen-slack --config configs/worldgen_slack/worldgen.toml --dry-run   # spends nothing
mkdir -p runs
uv run --frozen worldgen-slack --config configs/worldgen_slack/worldgen.toml > runs/world.log 2>&1   # paid
```

`worldgen.toml` is S3's configuration: 1,000 messages over 10 days and 40 tasks, about 6.5 hours and $75. For a
first run, copy it, set a new `seed` and `output`, and make it a smoke: `[calendar] days = 2`,
`[activity] messages = 80`, `[tasks] per_100 = 10` and `batch = 4` (8 tasks). An earlier smoke of that size took
69 minutes and about $7.

### 3. Forge tasks on it

```bash
uv run --frozen worldgen-slack forge --config configs/worldgen_slack/forge.toml --dry-run   # spends nothing
uv run --frozen worldgen-slack forge --config configs/worldgen_slack/forge.toml > runs/forge.log 2>&1   # paid
```

In `forge.toml`, `world` names the finished run to write on (its `data/<run>` directory), `base` the configuration
it was written with, `output` a new directory, and `rounds`, `candidates` and `target` how much to forge. A one-round
test of 10 candidates took 21 minutes and about $7.

- **Config** (`configs/worldgen_slack/worldgen.toml`): the sector, the seed, the calendar, the workspace's shape
  (`[activity]`), the tasks per 100 messages, each level's band of solver right-answer rates, the taxonomy with each
  level's needs, and one model per agent (`[env.author]`, `[env.judge]`, `[env.solver]`, `[env.witness]`,
  `[answer_judge]`). `forge.toml` configures a forge; its `[settings.*]` tables replace sections of its base.
  `eval.toml` evaluates a released world. `configs/{eval,gepa,rl}` are Prime Lab templates.
- **Change `output` and `seed` for each new world.** Rerunning the same command resumes from `state.json`; resume is
  refused if the config differs from the run's `run.json`.
- **Spend:** reported per role in `summary.json` and by `measure.py`; nothing caps it.
- **GLM limit:** the account allows 8 concurrent GLM requests, so run **one world at a time**.
- **Follow a run:** `tail -F runs/world.log | grep -E '"event": "(day_closed|probe|candidate_finished|workspace_frozen)"|Traceback'`.
  For a forge: `grep -E '"event": "(task_screened|task_reviewed|candidate_dropped|candidate_finished)"'`.
- **Stop a run** with Ctrl-C or `kill <pid>`: it unwinds, deletes its VM and keeps its checkpoint; rerun to resume.
  A second Ctrl-C is ignored while it cleans up. Its sandboxes carry the label `worldgen-<run>-<world>`
  (`prime sandbox list`), in case one is left behind.
- **A lost sandbox** (its process stream dropped) is not the author's fault: the run takes it up again on a fresh VM
  once per block, and a second loss in the same block stops it.
- **Inspect:**
  - `uv run --frozen python scripts/worldgen_slack/measure.py data/<run>` for the scorecard, style, the author's
    days and tool calls, probes and cost;
  - `uv run --frozen python viewer/build.py data/<run> --output viewer/artifacts-<name>.html` builds an offline
    page. It contains private answers, so share it only with authorized people.

## Read the code in this order

| Location | Responsibility |
|---|---|
| `generators/worldgen_slack/env.py` | The episode: setup, the plan, each day, mid-run reviews, tasks, probes, the final review |
| `generators/worldgen_slack/forge.py` | The forge: a finished world adopted frozen, rounds of candidates, the archive, its release |
| `generators/worldgen_slack/chronicle.py` | The rules of a world written in time order: plan, post, advance, close a day, revise, add a task |
| `generators/worldgen_slack/agents/world.py` | The author's memory (rendered by code), its world tools, its turns |
| `generators/worldgen_slack/agents/synthesizer.py` | The setup documents: premise and organization |
| `generators/worldgen_slack/agents/judge.py` | The agentic judge: task and world reviews, with inspection tools |
| `generators/worldgen_slack/contracts.py` | Cast, organization, agenda, task checks (T1-T7), the lazy-solver panel, failure cases, text rules, verdicts |
| `generators/worldgen_slack/store.py` | The run directory: checkpoint, attempts, traces, summary, release |
| `environments/worldgen_slack/worldgen_slack/db.py` | The world file: schema, rules checked on every write, actor-scoped reads |
| `environments/worldgen_slack/worldgen_slack/taskset.py` | The released taskset: fresh solver sessions, `vf.Judge` grading and the +1 / 0 / -1 reward |

`generate.py` starts the native Env from validated TOML. Verifiers owns model execution, harnesses, runtimes,
traces and episodes. The author is an rlm agent in one Prime VM for the whole world, with one interaction per block
of work; its world tools run on the host. The forge's proposer sits in the same seat, with the world frozen. The
judge cannot edit the world; its issues go back to the author.

**Agents and models** (one per seat, set in the config): the author and the proposer, Claude Opus 5.5; the judge,
the witness and the answer judge, gpt-6-sol; the solver, GLM 5.3 flash, with the asker's 10 Slack tools and no SQL,
sampling at temperature 1. The reward, the rubrics and the release format are in the
[taskset README](environments/worldgen_slack/README.md).

**Ground truth.** No model writes an answer. A task's gold is the result of its gold query over the stored world,
run as the asker sees it: from the ledger planned before day 1 (backward tasks), from the workspace's own rows
(forward tasks: who posted, replied, reacted), or both. Code checks that the query finds messages by what they say
and not by id, reads only what the solver's tools show, and returns only what the asker can read. The judge tests
the query against the question; the solver, which has no SQL, and the witness reach the same answer by reading
Slack.

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
    tasks.json           # public questions and actors, with each task's measured rates
    answers.json         # private answers and reference evidence
    manifest.json        # identity and file hashes (format worldgen-slack.v9)
```

A forge's directory has the same layout, with the world it adopted and the archive it kept.

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
