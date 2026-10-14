"""The proposer's task: one round of its work in its VM, opened with the round's prompt; the guide it works by, with
the forge's own numbers."""

from pathlib import Path
from typing import Self

import verifiers.v1 as vf
from worldgen_slack.taskset import WorldTaskData

from ...session import AuthoringToolsConfig, tools_config
from ..tasks import Task
from .tools import ProposerTools

GOLD_SQL = """- gold_sql is one SELECT over world.sqlite as the task's actor sees it: channels, members, messages,
  message_mentions, reactions and thread_stats hold only what the actor can read; users, calendar, storylines, facts,
  fact_relations and evidence are whole. local(us) writes a time on the actor's clock, local(us, zone) on another.
  It returns the answer in a column named answer, and may name the evidence in message_id and user_id columns.
- What the gold query reads follows the category's gold in `taxonomy`: a sql query reads only the workspace, a
  ledger query reads the facts it answers from, a hybrid query reads both.
- The gold query finds messages by what the question names (words, people, channels, threads, reactions, times),
  never by a message id or by id order: code checks that it answers the same with the messages renumbered.
- It asks only about what the solver's Slack tools show: who reacted, not when; who is a member now, not when they
  joined; nothing of a channel's creator or a person's account date. Code refuses a column no tool shows.
- answer_type: text or number is one row; set is 1 to max_answer_rows rows; refusal is no rows as the actor (nothing
  to find); status is one row, the latest value the actor can see, which a fact out of their sight settles: its gold
  reads that value from the ledger, and its facts name the settling fact. A status is answered right by giving the
  value as not settled.
- near_sql, optional: up to NEAR_SQL queries, the gold with one condition relaxed, each answering otherwise (the answer
  a hasty reader gives).
- question is what actor_id asks, in `language`; it does not contain its answer.
""".replace(
    "NEAR_SQL",
    str(next(m.max_length for m in Task.model_fields["near_sql"].metadata if hasattr(m, "max_length"))),
)


def forge_guide(settings: dict) -> str:
    """The proposer's guide, with the forge's own numbers: the solver's tries, and the right-answer rates of the
    middle levels (the bands between the first and the last). It names no model: the proposer knows the solver by
    its tries."""
    tries, bands = settings["author"]["tries"], settings["tasks"]["bands"]
    middle = bands[1:-1] or bands
    low, high = min(b[0] for b in middle), max(b[1] for b in middle)
    return (
        f"""You propose tasks on one company's finished Slack workspace, for training agents that answer colleagues'
questions from Slack. The workspace is written and frozen: nothing is posted, revised or moved. Each round you write
candidate tasks; then a solver model tries each one {tries} times with the asker's Slack tools, a stronger model tries
the ones the solver rarely answers, and a judge reviews each. What lands is kept in an archive by category, by the
level its tries measure, and by its structure; the rest is dropped. Your aim is tasks of every level, and above all
the middle ones (the solver right in {low:.0%}-{high:.0%} of its tries) that are fair: a careful colleague with the
same access answers them right.

Tools and memory
- world_now(), world_view(ref), world_sql(sql, actor_id=None) and world_read(actor_id, tool, arguments) read the
  world: the whole file, or what one person sees, through the solver's own tools. Walk a task's route as its asker
  before you write it: search with the question's words, read what comes up, and see what a hasty reader would
  answer.
- world_profile() is the world's inventory by code; world_probe(task) tries a candidate on a copy and writes nothing.
- world_add_task(task) writes a candidate for one of the round's open ids, with the category, level and intent you
  choose (taxonomy.md lists the intents: the request it is, a person's to an assistant); code checks it and returns
  its gold rows, its measures and its traps. Ask what people ask: balance the intents. world_annotate(fact) registers a fact the
  world already states, on its messages, when a task needs one the ledger lacks; the ledger is the world's own
  record of who stated what, with its decoys (planned wrong claims) and its changes (supersedes).
- /task/memory/ is code's: taxonomy.md (each category, what its levels ask for, its gold and answer types),
  archive.md (what each round kept and dropped, how often the solver was right, how it went wrong, and the cells
  still empty), board.md (the tasks the world was planned for, and the facts each rests on: write these first),
  ledger.json, and a page per person (its id, U..., is what actor_id and world_read take), channel, storyline and
  event. /task/notes/ is yours; the world's author left its plan.md there.

What makes a task hard, and what does not
- A solver that searches well and reads threads answers wrong when the cheapest route ends at a wrong value: the
  first value its search for the question's own words finds is stale, a near-miss or a decoy. A near-miss read beside
  its correction misleads no one; one whose correction sits where that search does not reach (a thread under another
  post, a DM, a later day, other words) does.
- What makes a solver miss: a correction that is itself out of date; a value that changed more than once across
  channels; a false premise matching a near-miss (a choice between two people when the answer is a third); a set or
  a count that is complete only when every place is read (each missed place costs it); an answer as of a moment, not
  the latest; who asks (the same question has another answer for someone who cannot read the DM that settles it);
  answers to work out, not to copy (durations across zones, counts nobody wrote).
- What does not: rewording; a long question that spells out the route; naming a near-miss and calling it wrong ("the
  count going around is wrong, so go by QA's post"): that hands over the answer. Ask as a colleague would, briefly.
- Middle rates come from two or three soft obstacles stacked, each of which the solver gets past most of the time,
  not from one trap it always or never falls into.
- A task is fair: its gold is the complete, correct answer from what its asker can read, and nothing else in their
  sight is as good an answer. Code grades a task's wrong answers before the solver tries it: a near-miss, a hedge, a
  set missing a row; a task whose grade accepts one is dropped.
- A task has one reading. Say what counts where a reader could take it two ways: whether thread replies count as
  posting, whether the poster counts among those who replied, whether a reaction breaks a quiet spell, what "then"
  orders, whose clock and which days. A task the judge finds a second reading of is sent back, not kept: the next
  round may reword it (world_add_task with rewords set to its id, the same asker and the same gold answer).

Tasks
- category and level from taxonomy.md; the level is your guess, and the solver's tries decide.
- A task above level 1 that rests on no facts names a near_sql: the answer a hasty reader gives.
- A status task (robustness) is for a value its asker can see that a fact out of their sight settles: its gold is
  that visible value from the ledger, its facts name the settling fact, and the right answer gives it as not settled.
  A refusal is for what is not there to see at all.
"""
        + GOLD_SQL
    )


class ProposerConfig(vf.TaskConfig):
    tools: AuthoringToolsConfig


class ProposerTask(vf.Task[WorldTaskData, vf.State, ProposerConfig]):
    """One round of the proposer's work in its VM: the env opens it with the round's prompt, and the proposer works
    through its world tools."""

    NEEDS_CONTAINER = True

    @classmethod
    def toolsets(cls, config: ProposerConfig) -> list[vf.Toolset]:
        return [ProposerTools(config.tools)]

    @classmethod
    def create(cls, world: Path, context: dict, attempt: str) -> Self:
        tools = tools_config("forge", 0, world, context)
        data = WorldTaskData(
            prompt=None,
            system_prompt=forge_guide(context["settings"]),
            attempt=attempt,
            world_hash=tools.db_hash,
            network_allow=[],
            network_block=["*"],
        )
        return cls(data, ProposerConfig(tools=tools))


def forge_prompt(round_: int, rounds: int, ids: list[str], held: int, goal: int) -> str:
    return (
        f"Round {round_}, of at most {rounds}. The archive holds {held} of the {goal} tasks it aims for; a full place "
        "takes a task only if the solver's tries are more mixed on it, or it brings a structure the place lacks. Read "
        "/task/memory/archive.md and board.md, and your notes. Then write one "
        f"candidate with world_add_task for each open id: {', '.join(ids)}. Walk each one's route as its asker first. "
        "Write in /task/notes/plan.md what you tried and why, then end your turn."
    )
