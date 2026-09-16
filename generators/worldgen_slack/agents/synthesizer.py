from .author import AuthorTask
from ..contracts import Catalog

CATALOG_GUIDE = """Plan the whole task collection before Slack messages are built. Use stable opaque identifiers.
Use opaque person IDs such as u_001, not names such as usr_maya. Update actor references consistently.
Create people (User schema), workstream groups, and facts with subject/predicate/value and UTC validity
intervals [valid_from, valid_until). Give conflicting values non-overlapping intervals. A fact description
explains the event and authority. Tasks specify a complete canonical answer, only requested answer claims,
fact_ids, reader actor_id, group_id, and reasoning. Every actor must be in people.
Set list_order_matters=false unless the question explicitly requires an ordered answer.
Tasks can share facts but must ask meaningfully different questions. Cover a natural mix of lookups, temporal reasoning,
cross-thread/channel joins, identity resolution, scoped lists, comparisons, and exceptions.
Group questions around connected work. Do not invent 100 unrelated answer snippets. Give public questions
enough clues for discovery without leaking the answer. Resolve time scope, authority, and list scope.
Avoid repeatedly reskinning one question. A plan, request, or constraint does not establish a decision.
"""


class SynthesizerTask(AuthorTask):
    output_type = Catalog
    instructions = CATALOG_GUIDE
