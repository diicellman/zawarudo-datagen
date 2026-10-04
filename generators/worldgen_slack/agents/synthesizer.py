"""The synthesizer: the company, its people and conversations, the fact ledger, and the tasks."""

from ..contracts import Ledger, Organization, Premises, TaskSet, census, pick_premise, user_id
from .author import AuthorTask

PREMISE_GUIDE = """Phase premise: propose exactly premise_count companies in `sector`; the run seed picks one, and the
rest of the run builds that company's Slack workspace. A premise has:
- company: its name, whose first word differs from the other premises' first words and from every name in used_names;
- niche, region, size and culture: what it does, where in `country` it is, how big it is (it employs `pool` people)
  and how it works;
- cast: the kinds of people who work there;
- staffing: how many people of each occupation in `occupations` it employs. The counts add up to exactly `pool`, each
  at most that occupation's count in `occupations`; code hires exactly these people.
Write in `language`.
"""

ORGANIZATION_GUIDE = """Phase organization: give the company's people their jobs, and choose its conversations.
- people: every candidate in `candidates` once, by user_id. A person keeps the candidate's name and timezone. title and
  team are the job they hold here.
- channels: the conversations the company works in. A public channel is readable by every person, a private channel
  only by its members; an im is a direct message between exactly 2 people, an mpim a group DM of 3 or more. Public and
  private channels have a name of lowercase letters, digits, - and _, and a topic and purpose in `language`; DMs have
  no name. members are user_ids of chosen people. Code adds more direct messages between people who work together.
- routines: for each public and private channel, 3 to 5 recurring kinds of conversation people have there, each with
  the probability that one of the channel's everyday conversations is of that kind. dm_routines: the same for direct
  messages. Code draws each everyday conversation's kind from them; the storylines are written apart.
The ledger review checks that each title and team fit the person's occupation, education and age.
"""

GOLD_SQL = """- gold_sql is one SELECT over world.sqlite as the task's actor sees it: channels, members, messages,
  message_mentions, reactions and thread_stats hold only what the actor can read; users, calendar, storylines, facts,
  fact_relations and evidence are whole. local(us) writes a time on the actor's clock, local(us, zone) on another.
  It returns the answer in a column named answer, and may name the evidence in message_id and user_id columns.
- What the gold query reads follows the category's gold in `taxonomy`: a sql query reads only the workspace, a
  ledger query reads the facts it answers from, a hybrid query reads both.
- The gold query finds messages by what the question names (words, people, channels, threads, reactions, times),
  never by a message id or by id order: code checks that it answers the same with the messages renumbered.
- answer_type: text or number is one row; set is 1 to max_answer_rows rows; refusal is no rows as the actor.
- question is what actor_id asks, in `language`; it does not contain its answer.
"""

GOLD = (
    GOLD_SQL
    + """- Each cell of `cells` gives a category, a level, the concept its task requires, and the style its question is asked
  in. Write 1 to 3 candidate tasks for each cell, with distinct ids. probability is how likely a
  task of that cell is to be like this one. Code checks every candidate and the run seed picks a valid one per cell,
  favoring the less likely.
"""
)

LEDGER_GUIDE = (
    """Phase ledger: plan what the company's Slack will state, before any message is written, and the tasks whose
answers come from it. world.sqlite holds the people (users) and conversations (channels, members).
- storylines: exactly storyline_count workstreams, in the order they will be built.
- facts: what someone states in Slack.
  - subject, attribute and value say what is stated. value is the answer it gives, with no time or date in it.
  - anchor: words of value that every message stating the fact contains exactly, or null.
  - places: 1 to 3 places where it could first be stated, each a channel_id, an author_id who is a member of it, and
    the probability of that place. Code draws one with the run seed, among places every actor of a task resting on
    the fact can read; a level with a spread in `taxonomy` has its task's facts drawn into that many channels.
  - day: on which day of `calendar` it is first stated.
  - after: facts first stated before this one. supersedes: the earlier fact whose value this one replaces. Both name
    facts of the same or an earlier storyline.
  - Facts with the same subject and attribute and different values form a supersedes chain, or all but one are decoy.
  - happened_at: when the event it reports happened; it is stated only after that moment. scheduled_for: the moment it
    plans. A moment is a day of `calendar`, a time "HH:MM" and an IANA zone; code writes it into each message on its
    reader's clock.
  - summary: what the fact means and on whose authority.
- tasks: for the cells in `cells`; `taxonomy` defines each category and level. Messages do not exist yet, so a gold
  query reads facts. facts names the facts a task's answer rests on.
"""
    + GOLD
    + """The ledger review checks that the facts fit the premise and the people's roles, that each fact's authority is
plausible, and that each task asks one clear question its gold query answers exactly.
"""
)

TASKS_GUIDE = (
    """Phase tasks: the workspace is written. Add tasks for the cells in `cells`; `taxonomy` defines
each category and level. world.sqlite holds the whole workspace, and the facts with the messages stating them
(evidence).
"""
    + GOLD
    + """- facts: for a hybrid task, the facts its answer starts from; for the others, none.
- A task asks a question no task in existing_questions asks.
The task review checks that each gold query answers its question as asked, that the actor can find the answer with
Slack's read tools, that nothing hands the answer over outside its evidence, and that each task is as hard as its
level says, measured: how deep its evidence sits, the tables its query reads, its evidence's search rank.
"""
)

CARD = set(
    "name sex age education_level bachelors_field occupation city state timezone professional_persona".split()
)


class SynthesizerTask(AuthorTask):
    outputs = {"premise": Premises, "organization": Organization, "ledger": Ledger, "tasks": TaskSet}
    guides = {
        "premise": PREMISE_GUIDE,
        "organization": ORGANIZATION_GUIDE,
        "ledger": LEDGER_GUIDE,
        "tasks": TASKS_GUIDE,
    }


def premise_context(settings, used) -> dict:
    countries, occupations = census(settings.personas, used["people"])
    return {
        "phase": "premise",
        "sector": settings.sector,
        "language": settings.language,
        "premise_count": settings.premise_count,
        "used_names": used,
        "country": ", ".join(countries),
        "occupations": dict(occupations.most_common()),
        "pool": settings.personas.pool,
    }


def parse_premise(raw: str, settings, used):
    """The synthesizer proposes distinct premises; the run seed, not the model, picks one."""
    premises = Premises.model_validate_json(raw)
    available, pool = census(settings.personas, used["people"])[1], settings.personas.pool
    for p in premises.premises:
        if sum(p.staffing.values()) != pool:
            raise ValueError(
                f"{p.company}: staffing counts add up to exactly {pool}, not {sum(p.staffing.values())}"
            )
        for occupation, count in p.staffing.items():
            if count > available[occupation]:
                raise ValueError(f"{p.company}: only {available[occupation]} unused people are {occupation}")
    return pick_premise(premises, settings.premise_count, used["companies"], settings.seed)


def candidates(cast) -> list[dict]:
    """The cards people are chosen by; code keeps the full profile and pairs it with the chosen person."""
    return [
        {"user_id": user_id(p.uuid)}
        | p.model_dump(include=CARD)
        | {"typing": p.typing.model_dump(exclude={"id", "messages"})}
        for p in cast
    ]


def organization_context(settings, state, used) -> dict:
    return {
        "phase": "organization",
        "premise": state.premise.model_dump(),
        "language": settings.language,
        "candidates": candidates(state.cast),
        "used_names": used["people"],
        "feedback": state.feedback,
    }


def definitions(settings, cells) -> dict:
    """The taxonomy as data: each requested category's definition, and the meaning and fact spread of its requested
    levels."""
    out = {}
    for category, level, *_ in cells:
        spec = settings.taxonomy[category]
        entry = out.setdefault(category, {"gold": spec.gold, "definition": spec.definition, "levels": {}})
        entry["levels"][level] = spec.levels[level - 1]
        if spec.spread:
            entry.setdefault("spread", {})[level] = spec.spread[level - 1]
    return out


def requested(cells) -> list[dict]:
    """The cells to write tasks for: each with the concept its task requires and the style its question is in."""
    return [dict(zip(("category", "level", "concept", "style"), c)) for c in cells]


def ledger_context(settings, state, calendar: list[dict]) -> dict:
    cells = [c for c in state.quota if settings.taxonomy[c[0]].gold == "ledger"]
    return {
        "phase": "ledger",
        "premise": state.premise.model_dump(),
        "language": settings.language,
        "storyline_count": -(-settings.tasks.count // settings.tasks.per_storyline),
        "calendar": calendar,
        "cells": requested(cells),
        "taxonomy": definitions(settings, cells),
        "max_answer_rows": settings.tasks.max_answer_rows,
        "feedback": state.feedback,
        "previous_output": state.drafts.get("ledger"),
    }


def tasks_context(settings, state, questions: list[str]) -> dict:
    cells = [c for c in state.quota if settings.taxonomy[c[0]].gold != "ledger"]
    return {
        "phase": "tasks",
        "language": settings.language,
        "cells": requested(cells),
        "taxonomy": definitions(settings, cells),
        "max_answer_rows": settings.tasks.max_answer_rows,
        "existing_questions": questions,
        "feedback": state.feedback,
        "previous_output": state.drafts.get("tasks"),
    }
