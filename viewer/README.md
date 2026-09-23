# Local artifact viewer

Read-only human inspection of current worldgen artifacts. All new code lives here;
no generator imports, model calls, server, dependencies, or changes to run data.

## Open the qualification study

From the repository root:

```bash
uv run --frozen python viewer/build.py
open viewer/artifacts.html
```

This exports the software, employee, and incomplete services runs into one offline
HTML file. On other platforms, open the file using your browser's file menu.

To select runs explicitly:

```bash
uv run --frozen python viewer/build.py data/qualification-01/software --output viewer/artifacts-software.html
```

Rebuild after artifacts change. Generated pages are ignored by Git. **The page
contains private answers, planning data, and full snapshots.** Share only with
people authorized to inspect the dataset; hiding labels is not access control.

## Seed study (22 September)

```bash
uv run --frozen python viewer/build.py data/study_22_09/unseeded/case-1-a data/study_22_09/seeded/case-1-b data/study_22_09/seeded/case-2-a data/study_22_09/unseeded/case-2-b data/study_22_09/unseeded/case-3-a data/study_22_09/seeded/case-3-b --output viewer/artifacts-study-22-09.html
open viewer/artifacts-study-22-09.html
```

Case 1 = software; case 2 = employee; case 3 = services. Run labels identify
seeded versus unseeded explicitly. Seeded services is incomplete: its last
candidate is not an approved release. All six arms are included. The original
qualification page remains `viewer/artifacts.html`. Open two browser windows to
compare arms; the built-in before/after selector compares attempts within a run.
This is not a blinded export. Source seed licensing is unresolved; keep this
research export local rather than publishing it.

## Review workflow

1. Select a workspace, attempt, and task.
2. Read the question; optionally hide private labels before exploring.
3. Follow evidence links into full threads, or browse/search the whole workspace.
4. Read the saved judge findings. They are allegations, not ground truth.
5. Compare two attempts. Inspect text changes separately from metadata changes.
   The earlier verdict describes the earlier artifact; the later verdict reviews
   the later artifact. Adjacent attempts may add a group, not just repair defects.

Attempt order comes from progress events, with name ordering only as fallback.
Catalog attempts and rejected structural outputs remain visible. Missing verdicts
are not approval. A snapshot is not necessarily a frozen release. This is an
omniscient reviewer view, **not a solver-access simulation** or a replay of tools.
The first version supports current `attempts/` artifacts, not legacy independent
world formats. It does not execute traces, validate release integrity, or grade
answers. Original artifacts remain the source of truth.

## Check

```bash
uv run --frozen python viewer/check.py
uv run ruff check viewer
uv run ruff format --check viewer
```

The small check uses the saved software qualification run and temporary malformed
input. No calls to models or sandboxes are made.
