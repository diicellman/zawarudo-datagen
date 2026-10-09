# Slack generation

The repository [README](../../README.md) contains the command, code map, output layout, and evaluation
instructions.

`GenerationEnv.author_world()` is the entry to the algorithm. One author agent, an rlm agent, works in one
Prime VM for the whole world, with one interaction per block of work. The world is written forward in time: code
holds the present (`world_meta.now_us`), and nothing is written before it.

1. **Setup.** The author proposes `premise_count` companies as a document; code picks one with the run seed and
   draws the cast from the persona file for the occupations it names. The author then writes the organization:
   titles, teams, channels with members and their everyday routines. Code adds direct messages and the calendar.
2. **Plan.** With `world_plan`, the author writes the ledger: storylines, events (a moment each, never moved once
   planned) and facts (what is stated, by whom, in which channel, on which day, about which event), and the board:
   the facts each slot that rests on planned facts (ledger and hybrid, and those whose level needs a near-miss) will
   rest on, checked against its level's needs (channels, relations, near-misses: decoys, or values whose change the
   asker cannot see, that nothing in their sight retracts), so a hard backward task's material exists while it can
   still be posted. It also writes `/task/notes/plan.md`: the arcs per day.
3. **Days.** Each calendar day is one interaction. The author reads `/task/memory` (rendered by code before every
   turn: `now.md`, the ledger, a page per person, channel, storyline and event) and its own notes, then posts the
   day's conversations in time order with `world_post` and moves on with `world_advance`. Code times every line
   from real reply gaps, tags every line that contains a fact's anchor as a statement of it, and refuses what
   breaks a rule: a first statement by someone other than its planned author, a fact before what it follows, a
   time written as a literal, a promise left open past its day. A day closes only when its quota, its facts and
   its promises are settled. The author then writes `/task/notes/recap.md`; the world and the notes are kept with
   the day, and a day that fails is written again from where the last one closed.
4. **Reviews.** After the days in `[author] review_days`, the judge reviews the world written so far; its issues
   open the author's next day, where `world_revise` rewrites a message in place.
5. **Tasks.** After the last day, the author writes one task per slot with `world_add_task`: the seed draws
   `[tasks] per_100` slots per 100 messages over the taxonomy's cells. Its gold query is checked (T1-T7, readable
   facts, the level's needs, among them `hidden`: its answer stated where it is harder to see than its near-misses)
   and code's measures of its difficulty come back. The GLM solver tries each task `[author] tries` times (8, sampling
   at temperature 1, so that its tries differ); the judge reviews it. A task whose share of right
   answers misses its level's `[tasks] bands`, or that the judge does not approve, comes back to the author with its
   question, gold and the solver's route (where the evidence first showed, which decoys it read), for up to
   `task_rounds` rewrites: a rewrite keeps the answer and changes the route, and a turn that changes nothing ends the
   loop.
   The slots are written in batches of `[tasks] batch`, each in an author interaction of its own. A stronger solver,
   the witness (`[env.witness]`), tries each task GLM answers right less often than its band's floor: one it answers
   is hard, not broken. Its tries go to the judge and count in no rate.
6. **Final review.** The judge reviews every task (with the solver's runs) and the whole world. Issues go back to
   the same author session until the judge approves or `review_rounds.final` runs out. Then the release is
   published.

The judge cannot edit the world. Its scores are reported; blocking issues decide acceptance (`[acceptance]`). An
interrupted run resumes from its checkpoint and the last closed block. Spend is reported, never capped: the summary
gives it per role.

## The forge: tasks on a finished world

`uv run --frozen worldgen-slack forge --config configs/worldgen_slack/forge.toml` writes tasks on a world a run has
finished, and leaves the world as it is (`forge.py`). It starts from the run's final `world.sqlite` with its tasks
emptied, its board and ledger kept, and the world author's last notes; then, round by round, a proposer in the
author's seat writes candidates for the round's open ids, of the category and level it chooses, and the solver,
the witness and the judge try and review each as a probe does. A candidate's tries give its level (the first
band whose floor its right rate reaches; the lowest band also needs the witness to answer it), and those GLM
misses half the time or more are tried twice more with the messages their answer rests on, which tells a task hard
to find from one hard to reason or broken. What the judge approves goes into an archive cell (its category, its
measured level and its structure: values in sight, where its answer sits, a set, another asker's other answer, an
answer to work out, a status), which keeps the candidate GLM's tries are most mixed on. `memory/archive.md` tells
the proposer, each round, what each category and level holds against `target`, which board entries no kept task
rests on, and how each recent candidate fared and how GLM went wrong. The forge stops when its rounds run out, the
archive is full, or two rounds keep nothing; the release is the archive. The proposer reads the board (the tasks the world was planned for: backward generation stays), and may
register with `world_annotate` a fact the world already states, on its messages. Nothing it does posts, revises or
moves the workspace, so no task goes stale under it. The forge resumes like a run, with its own configuration and
the world's hash.
