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

`ForgeEnv` (`forge/env.py`) writes tasks on a finished world and leaves the world as it is, as the paper's three
sets (Synthetic Data RL, arXiv 2505.17063, 4.1-4.3). It starts from the world's final file and the author's last
notes; its tasks live in its own state, never in the world file.

- **The initial set:** the proposer (`forge/proposer/`), an rlm agent in its own VM, writes `tasks` requests with
  `world_add_task`, `batch` a turn: each of a family (catch-up, decisions, commitments, status), in 1 to 4 parts, each
  part with its answer and the messages it rests on. Code refuses at once, with every reason, what it can tell is wrong:
  an asker who is no person, a message out of the asker's sight, an answer of the wrong shape, two parts alike, a
  question that gives its answer away, a repeat.
- **Labelling** (no VM; every task side by side, each step kept as it lands): the verifier, a stronger model with the
  student's Slack tools, must reach the task's answer in 2 tries (a third on a split), or the task is dropped; the
  student tries it `tries` times; the judge (`forge/judge.py`, one structured call) rules whether a colleague would
  ask it, whether its words allow another reading, whether each answer is exactly what its part asks, and whether its
  messages state it. A task is rejected, solved (right `solved` of the time or more), unsolved (never), or in the
  window.
- **Adapting** (`adapt` passes): the proposer writes a harder version of each solved task, an easier one of each
  unsolved task, and a repaired one of each verified task the judge refused, which are labelled in turn.
- **The release:** every approved task, with its set, its rates (by part), and the verifier's try that reached its
  answer; `report.json` says where the world ran short, for whoever writes the next chapter.

A forge carries on by itself: a lost proposer session is taken up again after a pause (`max_losses` times a turn), a
failed model call is asked again, a task whose steps fail is marked failed while the others go on, and the CLI resumes
the run after an infrastructure stop (`max_resumes`), until it has spent `spend_limit`. A turn cut off keeps the tasks
it wrote. A Ctrl-C is a stop, never resumed, even when it reaches the run as a cancelled stream. It resumes like a run,
with its own configuration and the world's hash.
