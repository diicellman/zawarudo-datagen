# Worldgen viewer

An offline HTML page per generation run. It reads a run's saved files and embeds them, with no model calls, no
server, and no changes to run data.

```bash
uv run --frozen python viewer/build.py data/v3-03/software --output viewer/artifacts-v3-03.html
open viewer/artifacts-v3-03.html
```

- Pass several run directories to compare them on one page.
- `--traces` embeds every agent transcript, which makes the page several MB larger.
- Generated pages are ignored by Git.
- **A page contains private answers, facts and full snapshots.** The "Answers shown" toggle only hides them on
  screen; it is not redaction.

## Tabs

| tab | shows | built from |
|---|---|---|
| Overview | company, premise, counts, cost, wall time, solve rate per task, usage by role | `summary.json`, `state.json` |
| Lineage | the attempt graph (premise → catalog → groups → final → publish). Each edge is labelled with the route taken next: repair, rebind, re-plan, catalog repair, resume. Selecting an attempt shows its reviews, criteria, issues (with links to the flagged messages), validation errors, in-attempt corrections, and world changes since the previous snapshot. Below it, a timeline of every agent run by role | `attempts/*`, `progress.jsonl` (`attempt_routed`, `candidate_*`), `traces/*` |
| World | a Slack-style view of the final world or of any attempt's world. It has a reader filter (only what one person can see). Each message shows its evidence chips and the scene that wrote it; the thread pane shows the scene's situation, beats and open promises | the candidate snapshot, `state.plan`, `state.scenes` |
| Tasks | question, reader, canonical answer, claims → evidence messages; the gold route step by step (results, evidence hits, which IDs feed the next call); the independent solves with their read routes; the task judge's verdict | bindings, `validation.json` `gold_outputs`, `review_input-<task>.json` solves, solver traces |
| Cast | people with their persona, seeded profile, and typing target next to what they actually wrote | catalog personas, snapshot |
| Traces | every agent run: role, attempt, task or scene, duration, tokens and cost. Transcripts appear only with `--traces` | `traces/*.json` |

## Notes

- The page reads saved outputs; it never imports or runs the generator.
  - The gold chain uses the gold outputs saved with the latest candidate, so it describes the current bindings.
  - Message → scene mapping uses the generator's message-id rule (`"m" + digest([scene_id, line])[:10]`).
- **Older runs:**
  - v2 runs predate per-review files and `attempt_routed`, so the lineage derives their routes from attempt
    order.
  - Writer traces before v3-03 have no scene id.
- This is an omniscient reviewer view, not a replay of what a solver could access. The reader filter shows the
  conversations a person can see; it does not re-run the tools.

## Check

```bash
uv run --frozen python viewer/check.py
uv run --frozen ruff check viewer && uv run --frozen ruff format --check viewer
```

The check builds a page from the scripted generation run. It asserts:
- attempt order;
- that route events reach the page;
- that every task's gold chain reaches all its bound evidence;
- that every message maps to its scene;
- HTML escaping.
