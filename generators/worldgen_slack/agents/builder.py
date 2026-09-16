from .author import AuthorTask
from ..contracts import Candidate

BUILDER_GUIDE = """Output Candidate with snapshot and bindings for every built task. Preserve all earlier content
unless a repair requires changing it. Materialize the catalog people exactly. Authors must be members;
replies follow their root in the same conversation. Public channels are readable by all users, private
channels/DMs only by members. Archived/deleted content cannot support answers. Use canonical UTC dates.
Facts and QA labels belong to the synthesizer: never change them while building. Report defects instead.
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
Build activity around the evidence: work preparation, negotiation, corrections, decisions, follow-up,
parallel work, ordinary coordination, and different writing styles. Reuse people/channels/evidence where
natural. Volume is guidance, not a reward; do not pad with repetitive templates or random chatter.
No answer caches, benchmark questions, private contract fields, generation markers, or answer-encoding IDs.
Do not cycle generic status texts, fill reusable reply templates, or use arithmetic posting schedules.
Do not give every thread one acknowledgement at the same delay. Timing and conversation length should
follow the actual work represented. The cumulative message target includes the existing workspace;
the group target estimates additional activity. Both are guidance, never reasons to pad.
Use opaque message IDs without calendar dates.
The final workspace supports every task simultaneously. Later records must not silently invalidate earlier
answers. Questions about the past need explicit time scope. Avoid adding a full-answer summary that bypasses
intended composition, but ordinary authoritative single-message lookups are valid.
"""


class BuilderTask(AuthorTask):
    output_type = Candidate
    instructions = BUILDER_GUIDE
