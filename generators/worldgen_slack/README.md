# Slack generation

The repository [README](../../README.md) contains the command, code map, output layout, and evaluation
instructions.

`GenerationEnv.author_world()` is the entry to the algorithm. One author agent, an rlm coding agent, works in one
Prime VM for the whole world, with one interaction per block of work. The world is written forward in time: code
holds the present (`world_meta.now_us`), and nothing is written before it.

1. **Setup.** The author proposes `premise_count` companies as a document; code picks one with the run seed and
   draws the cast from the persona file for the occupations it names. The author then writes the organization:
   titles, teams, channels with members and their everyday routines. Code adds direct messages and the calendar.
2. **Plan.** With `world_plan`, the author writes the ledger: storylines, events (a moment each, never moved once
   planned) and facts (what is stated, by whom, in which channel, on which day, about which event). It also writes
   `/task/notes/plan.md`: the arcs per day and a board of what each task cell will rest on.
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
   facts, the level's spread) and code's measures of its difficulty come back. The GLM solver tries each task
   `[author] tries` times and the judge reviews it. A task whose share of right answers misses its level's
   `[tasks] bands`, or that the judge does not approve, comes back to the author, for up to `task_rounds` turns.
6. **Final review.** The judge reviews every task (with the solver's runs) and the whole world. Issues go back to
   the same author session until the judge approves or `review_rounds.final` runs out. Then the release is
   published.

The judge cannot edit the world. Its scores are reported; blocking issues decide acceptance (`[acceptance]`). An
interrupted run resumes from its checkpoint and the last closed block. Spend is reported, never capped: the summary
gives it per role.
