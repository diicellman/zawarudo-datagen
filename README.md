# Zawarudo

Generate a Slack workspace, then QA tasks on it, for training agents that answer colleagues' questions from Slack.
Three parts, each in a directory of its own:

1. **The world** (`worldgen-slack`, `generators/worldgen_slack/world/`). One author agent writes a company's
   workspace in time order: the company, its people and channels, a ledger of facts and events, then every
   conversation day by day. An agentic judge reviews it mid-run and at the end. The world is one SQLite file; code
   is its only writer and checks every write. A **chapter** (`worldgen-slack chapter`) continues a finished world by
   some days, into a new version, and plants a brief: the facts a later forge's tasks will rest on.
2. **The forge** (`worldgen-slack forge`, `generators/worldgen_slack/forge/`). It writes tasks on a finished world,
   which it never changes, as three sets (Synthetic Data RL, arXiv 2505.17063): a proposer agent writes the initial
   set, each request aimed at a skill code deals it; each task is labelled (a verifier must reach its answer from
   Slack, the student tries it, a judge rules on it); then the proposer writes a harder version of each task the
   student always solves, an easier one of each it never does, and a repaired one of each the judge refused. Its
   release is the taskset, every task with its measured pass rate.
3. **The env** (`environments/worldgen_slack/`). The released taskset as a Verifiers env: each task a fresh solver
   session with the read tools of Slack's own MCP server, as the task's asker, graded by a native judge.

## Run on a fresh machine

Requirements: [uv](https://docs.astral.sh/uv/), git, the [DuckDB CLI](https://duckdb.org/docs/installation/)
(1.4 or later) to build the seed files, and a Prime account with Inference access (and Sandbox access, unless the
agents run in Docker: below). Authenticate with `prime login`, or export `PRIME_API_KEY` in the shell that starts a
run: nothing reads a `.env` file by itself (`set -a; . ./.env; set +a` loads one that holds only `KEY=value` lines).

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

`smoke-w1.toml` is two days of W1's world, about $5; `w1.toml` is W1 itself and `worldgen.toml` S3's world. `w2.toml`
is W2: 40 days and 5,000 messages, its ledger held to `[ledger]` targets and its promises slipping (`promise_slips`);
`smoke-w2.toml` is four of its days (its first runs found what W2 needed). For a new world, copy one and set a new
`seed` (it picks day 1) and `output`. W2 took 4.5 hours and $86; a run that long goes in a detached session, e.g.
`tmux new-session -d -s w2 "uv run --frozen worldgen-slack --config configs/worldgen_slack/w2.toml >> runs/w2.log 2>&1"`.

### Run the agents in Docker instead of Prime VMs (optional)

The agents that run code (the world's author and judge, the forge's proposer) work in Prime VMs by default. With
`[env] sandbox = "docker"` they work in Docker containers on the run's machine: the same pinned image, `/task` and
framework-only egress. The solvers already run as local processes either way. It needs Docker, and the user starting
the run in the `docker` group (in a session started after joining it). A container of the plain base installs the rlm
harness first, minutes on a slow link; the agents' image has it installed:

```bash
uv run --frozen python scripts/worldgen_slack/agent_image.py   # builds it, prints the config lines to use
```

It prints `sandbox = "docker"` and `docker_image = "sha256:..."`, the image's content id, for the config's `[env]`.
The id pins a run to the image as a digest pins a Prime VM, and it is the machine's own: `w2.toml` and `forge-w2.toml`
name the image built on the research VM where W2 ran, so on another machine build it and put its id there (or drop
both lines to use Prime VMs). W2 and its forge ran this way with no lost session, where Prime's sandbox streams
dropped six times on W2's smoke.

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
(`tasks`, `batch`), the skills code deals its ids (`skills`) and how often an id is offered (`offers`), its adapting
passes (`adapt`), how a task is labelled (`verify`, `tries`, `solved`), how far a run carries on by itself
(`max_losses`, `max_resumes`, `spend_limit`), and one model per seat (`[env.proposer]`, `[env.verifier]`,
`[env.solver]`, the task `[judge]`, `[answer_judge]`). `forge-w2-smoke.toml` forges 8 on W2 (about $10) and
`forge-w2.toml` 150, about 270 with their versions. A forge's `report.json` says what it covered: by family and
skill, its askers and openings, the material no task rests on yet.
`uv run --frozen worldgen-slack profile --world data/<run>` measures a world for a person (`world/profile.py`):
its ledger's counts by storyline (what `[ledger]` holds a world to), its value groups, promises, threads, channel
depth and people, for whoever writes its next chapter.

### Running notes

- **Configs** (`configs/worldgen_slack/`): world configs (`worldgen.toml`, `w1.toml`, `w2.toml` and their smokes) hold a world's
  sections and two seats (`[env.author]`, `[env.judge]`); chapter and forge configs are described above;
  `taxonomy.toml` holds the task categories and the question styles; `eval.toml` evaluates a release.
  `configs/{eval,gepa,rl}` are Prime Lab templates.
- **Resume:** rerunning the same command resumes from `state.json`; it is refused if the config differs from the
  run's `run.json`. Change `output` (and a world's `seed`) for each new run.
- **Spend:** reported per role in `summary.json` and by `measure.py`; nothing caps it.
- **GLM limit:** the account allows 32 concurrent GLM requests (a 429 past it), so run one GLM run at a time; a
  forge runs `solvers` verifier tries at once (GLM), the grades of its tries, and its task judge at a quarter of
  `solvers`.
- **Follow a run:** `tail -F runs/world.log | grep -E '"event": "(day_closed|review_started|candidate_finished|session_lost|world_resumed|workspace_frozen)"|"status"'`.
  For a forge: `grep -E '"event": "(tasks_taken|task_labelled|pass_labelled|ids_dropped|session_lost|forge_resumed)"|"status"'`.
- **Stop a run** with Ctrl-C or `kill <pid>`: it unwinds, deletes its VM and keeps its checkpoint; rerun to resume.
  A second Ctrl-C is ignored while it cleans up. Its sandboxes carry the label `worldgen-<run>-<world>`
  (`prime sandbox list`), in case one is left behind.
- **A run carries on by itself.** A lost sandbox (its process stream dropped) is taken up again in a fresh one after
  a pause, `max_losses` times a block (a world's day, a forge's turn); a stop that is infrastructure's (a provider's,
  a judge's sandbox lost twice) is resumed in place, `max_resumes` times; the same stop in the same place twice is a
  bug and ends the run. No wrapper script is needed: the command alone runs to the end or says why it stopped.
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
| `generators/worldgen_slack/forge/` | The forge: its env (the three sets), config, the task contract, the material and coverage, tries, the task judge and the release |
| `generators/worldgen_slack/forge/proposer/` | The proposer: its task and guide, and its tools |
| `generators/worldgen_slack/judge/` | The agentic judge both parts use: its task and review, its inspection tools, its verdict |
| `generators/worldgen_slack/session.py` | An agent's session in a VM, shared by the world and the forge, and the world tools both agents read with |
| `generators/worldgen_slack/workspace.py`, `store.py`, `config.py` | The world file as code writes it; the run directory; the settings both parts share |
| `environments/worldgen_slack/worldgen_slack/` | The env: the taskset, the solver's Slack tools and their server, the answer judge, the release format |
| `generators/worldgen_slack/tests/` | The tests: the scripted runs (`flows.py`), each part's rules, Slack's reads against the official server's capture, the layout |

Each agent's prompts are files in a `prompts/` folder beside its code. The world part never loads the forge's code;
the forge builds on the world's. Verifiers owns model execution, harnesses, runtimes, traces and episodes. The
author and the proposer are rlm agents, each in one Prime VM for the whole run, with one interaction per block of
work; their world tools run on the host. With `[env] sandbox = "docker"` every agent that runs code (author,
proposer, judge) works in a Docker container on the run's machine instead: the same pinned image, `/task` and
framework-only egress, for when Prime's sandbox streams drop (the machine needs Docker, and its user the `docker`
group). The judge cannot edit the world; its issues go back to the author.

**Agents and models** (one per seat, set in the config): the world's author and the forge's proposer, Claude Opus
5.5; the world's judge, gpt-6-sol; the forge's verifier, task judge and answer judge, GLM 5.3 (the verifier at max
effort in `forge-w2.toml`); the student, DeepSeek V4.1 flash, with the asker's 10 Slack tools and no SQL, sampling at
temperature 1. The proposer, the models that check its tasks and the student are of three families, so none checks
its own family's writing. The reward, the rubrics and the release format are in the
[env's README](environments/worldgen_slack/README.md).

**Ground truth.** A task's answer is text the proposer writes, part by part, with the messages it rests on, which
its asker can read. Code checks what it can (the asker, the messages in their sight, each answer's shape, a skill's
shape, no answer given away, no repeat); a verifier, a stronger model with the student's tools, must reach each
part's answer from Slack twice, or the task is dropped; the judge rules whether a colleague would ask it, whether its
words allow one reading, whether each answer is exactly what its part asks, and whether its messages state it.

## Outputs

```text
data/<run>/
  run.json               # the configuration and the pinned framework and lock file
  state.json             # the atomic restart checkpoint
  world.sqlite           # the world as written so far
  attempts/<attempt>/    # each block's world snapshot, the agent's notes, reviews and verdicts
  traces/                # native agent traces, with usage
  progress.jsonl         # events: turns, closed days, reviews, labels
  world-calls.jsonl      # every world tool call and its outcome
  summary.json           # outcome, labels, cost per role
  profile.json, .md      # a world's material, written by `worldgen-slack profile`
  report.json            # a forge's: what it covered, by family and skill, and where the world ran short
  release/               # a forge's approved tasks
    world.sqlite         # the world, without the answer key
    tasks.json           # public questions and askers, with each task's set and measured rates
    answers.json         # private answers, the messages they rest on, and the verifier's try that reached them
    manifest.json        # identity and file hashes (format worldgen-slack.v12)
```

A world run and a chapter end at their final review and publish no release; a forge publishes its approved tasks. World
and task quality and solver performance are separate signals: a valid task stays in the dataset when its solver
fails. `answers.json` and generation artifacts are private; solvers receive one question and actor-scoped tools,
never the release directory.

## Test and evaluate

```bash
uv run --frozen pytest
uv run --frozen python viewer/check.py
uv run --frozen ruff check && uv run --frozen ruff format --check
uv run --frozen eval @ configs/worldgen_slack/eval.toml --env.taskset.task.release_dir data/<forge>/release -r 2 --no-push
```

Local research reports and plans live in ignored `local-docs/`; generated runs live under ignored `data/`.
