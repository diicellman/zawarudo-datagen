# Slack generation

The repository [README](../../README.md) contains the command, code map, output layout,
and evaluation instructions.

`GenerationEnv.run()` is the entry to the algorithm. Its catalog and group methods use
ordinary native agent interactions, with independent agentic reviews between revisions.
The synthesizer owns QA/shared facts; the builder owns world data and evidence bindings.

Contracts use half-open fact intervals `[valid_from, valid_until)`. Reviews must cover all
requested claims against the exact candidate. Private facts guide generation but do not
prove that the Slack world supports an answer.

At the intended 100-task scale, each build review checks all built tasks. This favors
simple, conservative regression coverage over a separate dependency-tracking system.
Catalog edits invalidate approvals while preserving every consumed candidate allowance.

The configured budgets stop new calls after reported model spend reaches the limit.
In-flight calls, sandbox charges, and unpriced calls can exceed reported totals. Failed
attempts remain part of research accounting. No solver-success signal rewards the authors.
