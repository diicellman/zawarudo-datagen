from .author import AuthorTask
from ..contracts import Catalog, Premises, census, pick_premise

SYNTHESIZER_GUIDE = """Phase premise: propose exactly premise_count companies for the requested sector. One is chosen at
random, so each must support the whole task collection. Make them genuinely different from each other:
company identity, niche within the sector, country/region and working language, size and maturity, working
culture (formality, pace, remote/office), and the cast (who works there, backgrounds fitting the region).
used_names lists companies and people already used elsewhere in this corpus: do not reuse or echo them.
The company can be anywhere, but its working language is `language`, and all text you write uses it.
When input.json has country, every company is based in that country and its people may live anywhere in it.
When input.json has occupations (people available per occupation), each company lists in occupations the ones
it employs; its people are drawn from them.

Phase catalog: plan the whole task collection for the selected premise before Slack messages are built.
The company is exactly premise.company; people fit its region, size and culture and never reuse used_names.
Use stable opaque identifiers. Use opaque person IDs such as u_001, not names such as usr_maya. Update actor
references consistently. Give every person one persona: role, seniority, IANA timezone, and voice. Voice
describes how that person actually writes in Slack (length, formality, punctuation, habits) and differs
between people as it would in that company, from terse or casual to careful, as its culture allows.
When input.json has candidates, every person is a distinct candidate: the persona's seed_id is the candidate's,
and the person keeps the candidate's name and timezone. Role, team and seniority fit the candidate's occupation,
education and age; the voice follows from their profile and their typing (how they type in Slack).
Create people (User schema), workstream groups, and facts, each in the group of the workstream it belongs to,
with subject/predicate/value and UTC validity intervals [valid_from, valid_until). valid_from is when someone
first states the fact in Slack; an event time an answer needs belongs in the value. Choose moments as the people
involved live them on their own local clocks. Give conflicting values non-overlapping intervals. A fact description
explains the event and authority. Tasks specify a complete canonical answer, only requested answer claims,
fact_ids, reader actor_id, group_id, and reasoning. Every actor must be in people.
Set list_order_matters=false unless the question explicitly requires an ordered answer.
Tasks can share facts but must ask meaningfully different questions. Cover a natural mix of lookups, temporal reasoning,
cross-thread/channel joins, identity resolution, scoped lists, comparisons, and exceptions.
Group questions around connected work. Do not invent 100 unrelated answer snippets. Give public questions
enough clues for discovery without leaking the answer. Resolve time scope, authority, and list scope.
Avoid repeatedly reskinning one question. A plan, request, or constraint does not establish a decision.
"""


CARD = set(
    "name sex age education_level bachelors_field occupation city state timezone professional_persona".split()
)


class SynthesizerTask(AuthorTask):
    outputs = {"premise": Premises, "catalog": Catalog}
    instructions = SYNTHESIZER_GUIDE


def workspace_id(settings, state) -> str:
    return state.catalog.workspace_id if state.catalog else f"workspace-{settings.seed}"


def premise_context(settings, used) -> dict:
    return {
        "phase": "premise",
        "sector": settings.sector,
        "language": settings.language,
        "premise_count": settings.premise_count,
        "used_names": used,
        "workspace_id": f"workspace-{settings.seed}",
        "feedback": "",
    } | (supply(settings.personas) if settings.personas else {})


def supply(personas) -> dict:
    countries, occupations = census(personas)
    return {"country": ", ".join(countries), "occupations": dict(occupations.most_common())}


def candidates(cast) -> list[dict]:
    """The cards people are chosen by; code attaches the full profile to each chosen persona."""
    return [
        {"seed_id": p.uuid}
        | p.model_dump(include=CARD)
        | {"typing": p.typing.model_dump(exclude={"id", "messages"})}
        for p in cast
    ]


def parse_premise(raw, settings, used):
    """The synthesizer proposes distinct premises; the run seed, not the model, picks one."""
    premises = Premises.model_validate_json(raw)
    if settings.personas:
        available = census(settings.personas)[1]
        for p in premises.premises:
            if sum(available[o] for o in set(p.occupations)) < settings.personas.pool:
                raise ValueError(
                    f"{p.company}: list occupations from input.json whose people total {settings.personas.pool} or more"
                )
    return pick_premise(premises, settings.premise_count, used["companies"], settings.seed)


def catalog_context(settings, state, used) -> dict:
    return {
        "phase": "catalog",
        "sector": settings.sector,
        "task_count": settings.task_count,
        "group_size": settings.group_size,
        "language": settings.language,
        "premise": state.premise.model_dump() if state.premise else None,
        "used_names": used,
        "workspace_id": workspace_id(settings, state),
        "feedback": state.feedback,
        "previous_output": state.catalog.model_dump(mode="json") if state.catalog else None,
        "instructions": "Create the whole catalog. On repair preserve task IDs, group IDs, and assignments.",
    } | ({"candidates": candidates(state.cast)} if state.cast else {})


def cast_personas(catalog, cast) -> Catalog:
    """Each human persona names a distinct candidate and keeps its name and timezone; its profile is attached."""
    pool, people = {p.uuid: p for p in cast}, {p.id: p for p in catalog.people}
    humans = [p for p in catalog.personas if not people[p.id].is_bot]
    chosen = [p.seed_id for p in humans]
    if set(chosen) - pool.keys() or len(set(chosen)) != len(chosen):
        raise ValueError("every person needs a distinct seed_id from candidates")
    if mismatched := [
        f"{p.id} is {pool[p.seed_id].name} ({pool[p.seed_id].timezone})"
        for p in humans
        if (people[p.id].name, p.timezone) != (pool[p.seed_id].name, pool[p.seed_id].timezone)
    ]:
        raise ValueError("people keep their candidate's name and timezone: " + "; ".join(mismatched))
    profiles = {p.id: pool[p.seed_id] for p in humans}
    personas = [p.model_copy(update={"profile": profiles.get(p.id)}) for p in catalog.personas]
    return catalog.model_copy(update={"personas": personas})


def check_catalog(raw, settings, state, used) -> Catalog:
    catalog = Catalog.model_validate_json(raw)
    if len(catalog.tasks) != settings.task_count or catalog.sector.casefold() != settings.sector.casefold():
        raise ValueError("catalog must match requested task count and sector")
    if catalog.workspace_id != workspace_id(settings, state):
        raise ValueError("workspace ID must remain stable")
    if len(catalog.groups) != (settings.task_count + settings.group_size - 1) // settings.group_size:
        raise ValueError("catalog has the wrong number of groups")
    if any(sum(t.group_id == g.id for t in catalog.tasks) > settings.group_size for g in catalog.groups):
        raise ValueError("group exceeds configured task count")
    if state.catalog and {(t.id, t.group_id) for t in catalog.tasks} != {
        (t.id, t.group_id) for t in state.catalog.tasks
    }:
        raise ValueError("catalog repair must preserve task IDs and group assignments")
    if not catalog.personas or catalog.company != state.premise.company:
        raise ValueError("catalog must use premise.company and give every person a persona")
    if reused := {p.name for p in catalog.people} & set(used["people"]):
        raise ValueError(f"people reuse names from used_names: {sorted(reused)}")
    return cast_personas(catalog, state.cast) if state.cast else catalog
