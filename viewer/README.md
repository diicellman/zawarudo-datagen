# Worldgen viewer

An offline HTML page per run: a world, a chapter or a forge. It reads the run's saved files and embeds them, with
no model calls, no server, and no changes to run data.

```bash
uv run --frozen python viewer/build.py data/v7-05/software --output viewer/artifacts-v7-05.html
open viewer/artifacts-v7-05.html
```

- Pass several run directories to compare them on one page.
- `--traces` embeds every agent transcript, which makes the page several MB larger.
- Generated pages are ignored by Git.
- **A page contains private answers, facts and full snapshots.** The "Answers shown" toggle only hides them on
  screen; it is not redaction.
- It reads runs written since v7 (S3, its chapter C1, and their forges) and later.
- An attempt's world that an earlier attempt already holds (a forge's every turn, on its frozen world) is embedded
  once.

## Labelling a forge's tasks

A forge's page labels its tasks. Each task shows:
- its label, set, dealt skill, rates (by part), verification and the task judge's ruling;
- the messages its answer rests on, and those out of the asker's sight;
- the near-misses beside it (earlier or later values and decoys, marked when the asker cannot read them);
- every student and verifier try, with its grade by part and its calls;
- the QC reading, when the run holds a `qc.json`.

The label panel beside it asks five questions: fair, one reading, the kind of failure, realistic, and the skill
exercised. Each answer saves at once, and j / k move between tasks.

```bash
uv run --frozen python viewer/build.py data/v8-05/forge-w2 --output viewer/artifacts-forge-w2.html
uv run --frozen python viewer/serve.py viewer/artifacts-forge-w2.html --labels local-docs/labels/labels.json
```

- **Served** (`serve.py`, on http://127.0.0.1:8766), labels are saved to the labels file as `<run>/<task id>`.
- **Opened as a file**, they stay in the browser; Export and Import move them as JSON.

## Tabs

| tab | shows | built from |
|---|---|---|
| Overview | the company and its premise, the storylines, counts, cost and wall time, each task's share of right answers, usage by role, and a chapter's brief with the facts its board entries rest on | `summary.json`, `state.json`, the world file |
| Lineage | every attempt in the order it ran; selecting one shows its reviews, criteria and issues (linked to the flagged messages), validation errors, corrections, and how the world changed since the previous snapshot. Below it, a timeline of every agent run, a lane per role | `attempts/*`, `progress.jsonl`, `traces/*` |
| World | a Slack-style view of the final world or of any attempt's, by day, with each channel's members, and a reader filter (only what one person can see). Each message shows its reactions, mentions as names, the facts it states (decoys marked) and the conversation that wrote it; the thread pane shows the conversation's situation, beats and open promises | the world file's `messages`, `reactions`, `members` and `evidence` |
| Tasks | the question and its asker, the answer, the board entry and facts it rests on, its gold query, each solve with what it read, the witness's tries and the measured rates. A forge's tasks are filtered by label and by your labels, each with its tries, near-misses, ruling, QC reading and label panel | the world file's tasks, `summary.json`, solver and witness traces; a forge's `state.json`, `release/` and `qc.json` |
| Cast | people with their persona, seeded profile, and typing target next to what they actually wrote | the cast, the world file |
| Traces | every agent run, filtered by role (author, proposer, judge, solver, witness): attempt, task or conversation, duration, tokens and cost. Transcripts appear only with `--traces` | `traces/*.json` |

The page reads saved outputs; it never imports or runs the generator. It is an omniscient reviewer's view, not a
replay of what a solver could access: the reader filter shows the conversations a person can see; it does not re-run
the tools.

## Check

```bash
uv run --frozen python viewer/check.py
```

The check builds a page from the tests' scripted runs (`generators/worldgen_slack/tests/flows.py`): a world, its
forge, and a chapter. It asserts the attempts' order, the snapshots, each conversation by its lines, the cast,
reactions and mentions, the facts each message states, every task's gold query and rates, the probes and the
witness, the chapter and its board, and HTML escaping.
