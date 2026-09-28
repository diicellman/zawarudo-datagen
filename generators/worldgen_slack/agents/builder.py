from .author import AuthorTask
from ..contracts import BindOutput, Plan

BUILDER_GUIDE = """You plan the workspace and bind evidence. You never write message text: a separate writer turns
each scene into messages in the voices of the catalog personas.

Phase plan: write the Plan for the WHOLE workspace, including every earlier conversation and scene from
previous_output. Preserve earlier scenes and their IDs unless feedback requires a change.
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
Read the messages you bind. If a message does not really support its claim or a scene contradicts the
catalog, add a rewrite {scene_id, note} instead; the scene is rewritten and you bind again. open_promises
lists commitments made in scenes; when a required answer relies on one, a later scene must keep it.
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
