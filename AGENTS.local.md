# AGENTS.local.md

Project guidance for agents in this workspace, read after `AGENTS.md`.

## Testing

The time goes to runs, not ceremony. One runnable check per non-trivial piece of logic, as `AGENTS.md` asks.

- **Each change:** run the test file of the module it touches, and ruff on the changed files. That takes seconds.
- **Before a paid run:** the whole `uv run --frozen pytest` (about 15 s), `viewer/check.py`, and the run's
  `--dry-run`, on the machine that will run it.
- **The behaviour lock (`local-docs/tools-v8/lock.py`) and the mutation suite (`local-docs/tools-v8/mutate.py`):** only
  for a refactor of code that world generation and the forge share, or before merging to `main`. Never per commit.
