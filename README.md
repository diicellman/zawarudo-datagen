# Zawarudo

Generate a Slack workspace, then QA tasks on it, for training agents that answer colleagues' questions from Slack.
Three parts, each in a directory of its own:

1. **The world** (`worldgen-slack`, `generators/worldgen_slack/world/`). One author agent writes a company's
   workspace in time order: the company, its people and channels, a ledger of facts and events, then every
   conversation day by day. An agentic judge reviews it mid-run and at the end. The world is one SQLite file; code
   is its only writer and checks every write. A **chapter** (`worldgen-slack chapter`) continues a finished world by
   some days, into a new version, and plants a brief: the facts a later forge's tasks will rest on.
2. **The forge** (`worldgen-slack forge`, `generators/worldgen_slack/forge/`). It writes tasks on a finished world,
   which it never changes. Round by round, a proposer agent writes candidates; code checks each and grades the wrong
   answers it builds for it, a solver tries it, a stronger witness tries the hard ones, and the judge reviews them. An
   archive keeps what lands and tells the proposer how the rest went wrong. Its release is the taskset.
3. **The env** (`environments/worldgen_slack/`). The released taskset as a Verifiers env: each task a fresh solver
   session with the read tools of Slack's own MCP server, as the task's asker, graded by a native judge.

## Run on a fresh machine

Requirements: [uv](https://docs.astral.sh/uv/), git, the [DuckDB CLI](https://duckdb.org/docs/installation/)
(1.4 or later) to build the seed files, and a Prime account with Inference and Sandbox access. Authenticate with
`prime login` or set `PRIME_API_KEY` (a `.env` file in the repository root works too).

```bash
uv sync --all-extras --all-groups
uv run --frozen pytest        # offline, no model calls
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
uv run --frozen worldgen-slack --config configs/worldgen_slack/smoke-w1.toml --dry-run   # spends nothing
mkdir -p runs
uv run --frozen worldgen-slack --config configs/worldgen_slack/smoke-w1.toml > runs/world.log 2>&1   # paid
```

`smoke-w1.toml` is two days of W1's world, about $5; `w1.toml` is W1 itself and `worldgen.toml` S3's world. For a new
world, copy one and set a new `seed` and `output`.

### 3. Continue it with a chapter (optional)

```bash
uv run --frozen worldgen-slack chapter --config configs/worldgen_slack/chapter-c1.toml --dry-run
```

A chapter names the finished world it continues (`world`), the configuration that world was written with (`base`),
the taxonomy its brief's entries are cells of, its days and messages, and its brief (`[[material]]`). Nothing before
its first day is written again.

### 4. Forge tasks on it

```bash
uv run --frozen worldgen-slack forge --config configs/worldgen_slack/forge.toml --dry-run   # spends nothing
uv run --frozen worldgen-slack forge --config configs/worldgen_slack/forge.toml > runs/forge.log 2>&1   # paid
```

A forge config is self-contained: the finished world (`world`), a new `output`, the initial set's size and turn
(`tasks`, `batch`), its adapting passes (`adapt`), how a task is labelled (`verify`, `tries`, `solved`), how far a
run carries on by itself (`max_losses`, `max_resumes`, `spend_limit`), and one model per seat (`[env.proposer]`,
`[env.verifier]`, `[env.solver]`, the task `[judge]`, `[answer_judge]`).
`uv run --frozen worldgen-slack profile --world data/<run>` measures a world for a person (`world/profile.py`):
its value groups, threads and people, as counts, for whoever writes its next chapter.

### Running notes

- **Configs** (`configs/worldgen_slack/`): world configs (`worldgen.toml`, `w1.toml`, `smoke-w1.toml`) hold a world's
  sections and two seats (`[env.author]`, `[env.judge]`); chapter and forge configs are described above;
  `taxonomy.toml` holds the task categories and the question styles; `eval.toml` evaluates a release.
  `configs/{eval,gepa,rl}` are Prime Lab templates.
- **Resume:** rerunning the same command resumes from `state.json`; it is refused if the config differs from the
  run's `run.json`. Change `output` (and a world's `seed`) for each new run.
- **Spend:** reported per role in `summary.json` and by `measure.py`; nothing caps it.
- **GLM limit:** the account allows 8 concurrent GLM requests, so run one GLM run at a time.
- **Follow a run:** `tail -F runs/world.log | grep -E '"event": "(day_closed|review_started|candidate_finished|workspace_frozen)"|Traceback'`.
  For a forge: `grep -E '"event": "(task_screened|task_reviewed|candidate_dropped|candidate_finished)"'`.
- **Stop a run** with Ctrl-C or `kill <pid>`: it unwinds, deletes its VM and keeps its checkpoint; rerun to resume.
  A second Ctrl-C is ignored while it cleans up. Its sandboxes carry the label `worldgen-<run>-<world>`
  (`prime sandbox list`), in case one is left behind.
- **A lost sandbox** (its process stream dropped) is not the agent's fault: the run takes it up again on a fresh VM
  once per block, and a second loss in the same block stops it.
- **Inspect:** `uv run --frozen python scripts/worldgen_slack/measure.py data/<run>` for a world's scorecard, style,
  the author's days and tool calls, and cost; `uv run --frozen python viewer/build.py data/<run> --output
  viewer/artifacts-<name>.html` builds an offline page. It contains private answers, so share it only with authorized
  people.

## The code

| Location | What it holds |
|---|---|
| `generators/worldgen_slack/cli.py` | `worldgen-slack`: one run function for a world, a chapter and a forge, and their dry runs |
| `generators/worldgen_slack/world/` | The world in time order: its env (`env.py`), chapters, config, and the rules by topic (cast, organization, clock, ledger, board, writing, style, apps) |
| `generators/worldgen_slack/world/author/` | The world author: its setup documents, its task and turn prompts, its tools and the memory code renders for it |
| `generators/worldgen_slack/forge/` | The forge: its env (rounds, the archive), config, the task contract, difficulty, tries and the release |
| `generators/worldgen_slack/forge/proposer/` | The proposer: its task and guide, and its tools |
| `generators/worldgen_slack/judge/` | The agentic judge both parts use: its task and review, its inspection tools, its verdict |
| `generators/worldgen_slack/session.py` | An agent's session in a VM, shared by the world and the forge, and the world tools both agents read with |
| `generators/worldgen_slack/workspace.py`, `store.py`, `config.py` | The world file as code writes it; the run directory; the settings both parts share |
| `environments/worldgen_slack/worldgen_slack/` | The env: the taskset, the solver's Slack tools and their server, the answer judge, the release format |
| `generators/worldgen_slack/tests/` | The tests: the scripted runs (`flows.py`), each part's rules, Slack's reads against the official server's capture, the layout |

Each agent's prompts are files in a `prompts/` folder beside its code. The world part never loads the forge's code;
the forge builds on the world's. Verifiers owns model execution, harnesses, runtimes, traces and episodes. The
author and the proposer are rlm agents, each in one Prime VM for the whole run, with one interaction per block of
work; their world tools run on the host. The judge cannot edit the world; its issues go back to the author.

**Agents and models** (one per seat, set in the config): the author and the proposer, Claude Opus 5.5; the judge,
the witness and the answer judge, gpt-6-sol; the solver, GLM 5.3 flash (or the student, DeepSeek V4.1 flash), with the
asker's 10 Slack tools and no SQL, sampling at temperature 1. The reward, the rubrics and the release format are in
the [env's README](environments/worldgen_slack/README.md).

**Ground truth.** No model writes an answer. A task's gold is the result of its gold query over the stored world,
run as the asker sees it: from the ledger planned before day 1, from the workspace's own rows (who posted, replied,
reacted), or both. Code checks that the query finds messages by what they say and not by id, reads only what the
solver's tools show, and returns only what the asker can read. The judge tests the query against the question; the
solver, which has no SQL, and the witness reach the same answer by reading Slack.

## Outputs

```text
data/<run>/
  run.json               # the configuration and the pinned framework and lock file
  state.json             # the atomic restart checkpoint
  world.sqlite           # the world as written so far
  attempts/<attempt>/    # each block's world snapshot, the agent's notes, reviews and verdicts
  traces/                # native agent traces, with usage
  progress.jsonl         # events: turns, closed days, reviews, candidates
  world-calls.jsonl      # every world tool call and its outcome
  summary.json           # outcome, task mix, solve rates, difficulty, cost
  release/               # a forge's, when its archive kept tasks
    world.sqlite         # the world, without the answer key
    tasks.json           # public questions and askers, with each task's measured rates
    answers.json         # private answers and reference evidence
    manifest.json        # identity and file hashes (format worldgen-slack.v10)
```

A world run and a chapter end at their final review and publish no release; a forge publishes its archive. World
and task quality and solver performance are separate signals: a valid task stays in the dataset when its solver
fails. `answers.json` and generation artifacts are private; solvers receive one question and actor-scoped tools,
never the release directory.

## Test and evaluate

```bash
uv run --frozen pytest
uv run --frozen python viewer/check.py
uv run --frozen ruff check && uv run --frozen ruff format --check
uv run eval @ configs/worldgen_slack/eval.toml -n 1 -r 2 --no-push --plain
```

Local research reports and plans live in ignored `local-docs/`; generated runs live under ignored `data/`.
