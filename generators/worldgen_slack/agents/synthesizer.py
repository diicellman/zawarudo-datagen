from .author import AuthorTask
from ..contracts import Catalog, Premises

SYNTHESIZER_GUIDE = """Phase premise: propose exactly premise_count companies for the requested sector. One is chosen at
random, so each must support the whole task collection. Make them genuinely different from each other:
company identity, niche within the sector, country/region and working language, size and maturity, working
culture (formality, pace, remote/office), and the cast (who works there, backgrounds fitting the region).
used_names lists companies and people already used elsewhere in this corpus: do not reuse or echo them.
The company can be anywhere, but its working language is `language`, and all text you write uses it.

Phase catalog: plan the whole task collection for the selected premise before Slack messages are built.
The company is exactly premise.company; people fit its region, size and culture and never reuse used_names.
Use stable opaque identifiers. Use opaque person IDs such as u_001, not names such as usr_maya. Update actor
references consistently. Give every person one persona: role, seniority, IANA timezone, and voice. Voice
describes how that person actually writes in Slack (length, formality, punctuation, habits) and differs
between people as it would in that company, from terse or casual to careful, as its culture allows.
Create people (User schema), workstream groups, and facts with subject/predicate/value and UTC validity
intervals [valid_from, valid_until). valid_from is when someone first states the fact in Slack; an event time
an answer needs belongs in the value. Choose moments as the people involved live them on their own local clocks.
Give conflicting values non-overlapping intervals. A fact description
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
    outputs = {"premise": Premises, "catalog": Catalog}
    instructions = SYNTHESIZER_GUIDE
