# Slack generation

The repository [README](../../README.md) has the commands, the code map, the output layout, and how to evaluate.
This page is the algorithm.

## The world

`WorldEnv` (`world/env.py`) runs it. One author, an rlm agent, works in one Prime VM for the whole world, with one
interaction per block of work (`SessionEnv.work_in_vm`). The world is written forward in time: code holds the
present (`world_meta.now_us`), and nothing is written before it.

1. **Setup.** The author proposes `premise_count` companies as a document; code picks one with the run seed and
   draws the cast from the persona file for the occupations it names. The author then writes the organization:
   titles, teams, channels with their members and everyday routines, and the apps that post in them. Code adds
   direct messages and the calendar.
2. **Plan.** With `world_plan`, the author writes the ledger: storylines, events (a moment each, never moved once
   planned) and facts (what is stated, by whom, in which channel, on which day, about which event). It also writes
   `/task/notes/plan.md`: the arcs, day by day.
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
5. **Final review.** The judge reviews the whole world. Its issues go back to the same author session, in fix
   turns, until the judge approves or `review_rounds.final` runs out. The run then ends: it writes no tasks.

The judge cannot edit the world. Its scores are reported; blocking issues decide approval. An interrupted run
resumes from its checkpoint and the last closed block. Spend is reported, never capped: the summary gives it per
role.

## A chapter

`ChapterEnv` (`world/chapter.py`) continues a finished world (a run's or an earlier chapter's) into a new version.
It adopts the world frozen, extends its calendar by `days`, and plans what they hold: the earlier storylines going
on, its own after them, and its brief. Each brief entry is a cell of the taxonomy whose facts code checks on the
board when the ledger is planned (its level's needs, the material's on top, and always first stated in the
chapter), so a forge on the new version has hard material to rest tasks on. The same author writes the days as a
world's are written. Nothing before the chapter's first day is posted, revised or re-planned, and its reviews read
the earlier days as written and frozen.

## The forge

`ForgeEnv` (`forge/env.py`) writes tasks on a finished world and leaves the world as it is. It starts from the
world's final file with its tasks emptied, its board and ledger kept, and the author's last notes.

- **Round 0, when `measure` is set:** the world's own tasks, imported as written and measured as any candidate is.
- **Each round:** the proposer (`forge/proposer/`), an rlm agent in its own VM and seat, writes a candidate for each
  of the round's open ids with `world_add_task`, of the category and level it chooses. It reads the world as any
  person sees it, the world's inventory (`world_profile`), and tries a candidate on a copy that writes nothing
  (`world_probe`); `world_annotate` registers a fact the world already states, on its messages. It never posts,
  revises or moves the workspace, so no task goes stale under it.
- **Each candidate:** code checks its gold query and builds the wrong answers its grade must reject (the screen: a
  task whose grade accepts one is dropped); the solver tries it `tries` times; the witness tries one the solver
  answers right less often than its band's floor (one it answers is hard, not broken); one the solver misses half the
  time or more is tried twice more with the messages its answer rests on, which tells a task hard to find from one
  hard to reason or broken; the judge reviews it, with the tries.
- **The archive:** a candidate's tries give its level (the first band whose floor its right rate reaches; the lowest
  band also needs the witness to answer it). What the judge approves takes a place: its category and measured
  level, which holds up to `target` tasks (MAP-Elites over task features). A newcomer to a full place replaces a
  holder when it brings a structure the place lacks while two holders share one, or when the solver's tries are more
  mixed on it. `memory/archive.md` tells the proposer, each round, what each place holds, which board entries no kept
  task rests on, and how each recent candidate fared and how the solver went wrong. A candidate whose question the
  judge reads two ways is sent back, for the next round to reword with the same asker and gold.
- **The end:** the forge stops at its goal (`tasks` kept and every board entry covered), when its rounds run out,
  or after two rounds that keep nothing. Its archive is the release, when it kept any task.

A forge resumes like a run, with its own configuration and the world's hash.
