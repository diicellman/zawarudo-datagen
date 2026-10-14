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

## Tabs

| tab | shows | built from |
|---|---|---|
| Overview | the company and its premise, the storylines, counts, cost and wall time, each task's share of right answers, usage by role, and a chapter's brief with the facts its board entries rest on | `summary.json`, `state.json`, the world file |
| Lineage | every attempt in the order it ran; selecting one shows its reviews, criteria and issues (linked to the flagged messages), validation errors, corrections, and how the world changed since the previous snapshot. Below it, a timeline of every agent run, a lane per role | `attempts/*`, `progress.jsonl`, `traces/*` |
| World | a Slack-style view of the final world or of any attempt's, by day, with each channel's members, and a reader filter (only what one person can see). Each message shows its reactions, mentions as names, the facts it states (decoys marked) and the conversation that wrote it; the thread pane shows the conversation's situation, beats and open promises | the world file's `messages`, `reactions`, `members` and `evidence` |
| Tasks | the question and its asker, the answer, the board entry and facts it rests on, its gold query, each solve with what it read, the witness's tries and the measured rates | the world file's tasks, `summary.json`, solver and witness traces |
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
