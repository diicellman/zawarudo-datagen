from .author import AuthorTask
from .writer import frozen
from ..contracts import BindOutput, Plan, first_mentions, timed

BUILDER_GUIDE = """You plan the workspace and bind evidence. You never write message text: a separate writer turns
each scene into messages in the voices of the catalog personas.

Phase plan: the catalog holds the tasks built so far, this group's tasks, and their workstreams' facts.
Write the Plan with every conversation and scene from previous_output plus the scenes this group's facts need.
Scenes in frozen_scene_ids belong to the approved world: keep them unchanged except for their revision_note. Other
earlier scenes may change when feedback requires it.
Conversations: channels, private channels, DMs and group DMs the company would really use. Public channels
are readable by all users; private channels/DMs only by members. Archived content cannot support answers.
A scene is one stretch of talk in one conversation: a thread or a run of top-level messages. Give its
participants (members), UTC start and end, expected length, and situation: what is going on, what each participant
wants, and what gets decided or stays open. Beats place catalog facts: {fact_id, author_id}. The author is
someone the fact description makes plausible; the writer has them state it. Every fact that supports a
required task needs a beat in a scene its reader can see. A superseded value needs its own earlier beat.
Stage facts where work produces them: preparation, negotiation, corrections, decisions, and follow-up.
Each fact in timed_facts is first stated in the minute of its valid_from, so its first scene starts by then.
Surround evidence with the rest of the work: parallel workstreams, blockers, handoffs, unanswered
questions, and ordinary coordination. Reuse people and channels where natural. Starts and lengths follow
the work and people's timezones; avoid fixed schedules and uniform scene sizes. Scenes that depend on each
other do not overlap: one ends before the other starts. new_messages_hint is a
rough range for this group's added messages, never a quota. To fix the writing of an existing scene, set
its revision_note to a concrete instruction; changing situation, participants or beats rewrites it.
details declares concrete details that recur across scenes (systems, versions, environment or document names, IDs,
schedules) once as {value, since, at}. since is the UTC time the detail becomes known (null if always known); at is
the UTC moment of a scheduled meeting, deadline or window. Declare every time people will schedule, promise or wait
for, consistent with the catalog; writers use only declared times. Scene and conversation IDs are opaque: no dates, names,
or answer values. Visible conversation names,
topics and purposes use `language`.
Facts and QA labels belong to the synthesizer: never change them. Report catalog defects instead.
The final workspace supports every task simultaneously. Later scenes must not silently invalidate earlier
answers. Questions about the past need explicit time scope. Avoid a full-answer summary that bypasses
intended composition, but ordinary authoritative single-message lookups are valid.
No answer caches, benchmark questions, private contract fields, or generation markers.

Phase bind: input.json holds the assembled workspace, scene_messages (scene → message IDs) and conveyed
(fact → messages stating it). Write bindings for every required task; keep earlier bindings valid.
Read the messages you bind: each claim needs messages that really state it.
Each Binding maps every canonical claim index to message_ids and/or user_ids, and has replayable gold_calls.
Actions: list_conversations(cursor?,limit?), search_messages(query,conversation_id?,author_id?,after?,before?,
cursor?,limit?), get_conversation_history(conversation_id,cursor?,limit?),
get_thread(conversation_id,root_message_id,cursor?,limit?), get_user(user_id).
Reads return {items,next_cursor}, except get_user. Page limits are 1–100. Gold calls use {action,arguments}.
Gold routes must begin with public-question clues and discover subsequent IDs through prior results.
Do not mix question words with undiscovered private answer labels in a search. Discover exact cohort
names, image labels, and answer values from an earlier read before using them to narrow later queries.
The first call must not contain an internal conversation/user/message ID absent from the question.
Use global search or list_conversations first. Bindings and gold routes are builder-owned artifacts.
"""


class BuilderTask(AuthorTask):
    outputs = {"plan": Plan, "bind": BindOutput}
    instructions = BUILDER_GUIDE


def scoped(state, group_id):
    """The catalog as a group sees it: built and current tasks, and the facts of their workstreams."""
    included = {*state.built_groups, group_id}
    tasks = [t for t in state.catalog.tasks if t.group_id in included]
    shown = {f for t in tasks for f in t.fact_ids} | (
        set(first_mentions(state.plan)) if state.plan else set()
    )
    facts = [f for f in state.catalog.facts if f.group_id in included or f.id in shown]
    groups = [g for g in state.catalog.groups if g.id in included | {f.group_id for f in facts}]
    return state.catalog.model_copy(update={"tasks": tasks, "facts": facts, "groups": groups})


def later_groups(state, catalog) -> list[dict]:
    """Workstreams a scoped view leaves out because they are built later."""
    return [g.model_dump() for g in state.catalog.groups if g not in catalog.groups]


def group_scope(state, group_id) -> dict:
    catalog = scoped(state, group_id)
    return {
        "workspace_id": state.catalog.workspace_id,
        "group_id": group_id,
        "catalog": catalog.model_dump(mode="json"),
        "required_task_ids": [t.id for t in catalog.tasks],
        "later_groups": later_groups(state, catalog),
        "feedback": state.feedback,
    }


def plan_context(settings, state, group_id) -> dict:
    per_group = settings.messages_per_task * sum(t.group_id == group_id for t in state.catalog.tasks)
    return {
        "phase": "plan",
        **group_scope(state, group_id),
        "language": settings.language,
        "premise": state.premise.model_dump(),
        "new_messages_hint": f"{per_group * 6 // 10}-{per_group * 13 // 10}",
        "timed_facts": sorted(timed(scoped(state, group_id))),
        "frozen_scene_ids": sorted(s.id for s in frozen(state)),
        "previous_output": state.plan.model_dump(mode="json") if state.plan else None,
    }
