"""Scene writer: one conversation scene per fresh chat, without tools or code execution."""

import json
import random
import verifiers.v1 as vf
from datetime import datetime, timedelta
from worldgen_slack.slack.api import digest
from ..contracts import (
    Catalog,
    Plan,
    Premise,
    SceneRecord,
    ScenePlan,
    SeedPacket,
    SeedPersona,
    WrittenScene,
    check_scene,
    clock,
    first_mentions,
    timed,
    to_local,
    to_utc,
)
from worldgen_slack.slack.models import Conversation

WRITER_GUIDE = """Write one scene of a company's Slack workspace: the messages coworkers post while they work.
The brief gives the company, the conversation, the participants with personas, the situation, recent
messages, and beats. recent_messages come earlier in this conversation; seen_elsewhere are recent messages
the participants saw in other conversations; established_facts are true at the scene start; world_details and established_details are fixed facts of this
workspace. Never contradict any of them, and restate them only when the work calls for it. commitments are
promises made earlier: keep them, or have someone explicitly change them. A participant's profile is who they are
outside their role; typing is how they type in Slack, measured on a real person: median words per message and
the shares of their messages with at most four words, a question, a lowercase start, or an emoji. Their lines
follow those numbers. Each person writes in their own voice as their persona describes; people differ in
length, capitalization, punctuation, formality, abbreviations and emoji. Slack is chat, not email: most
messages are one line, many are a few words (acks, quick questions, "on it", "hmm"), people split a thought
across several messages, and multi-paragraph posts are rare. Write in the brief's language. Continue the
recent messages naturally without repeating them. Write about `length` messages, fewer or more if the
situation calls for it.
Beats are facts that must come up. The beat's author states the fact in their own words as part of the work,
explicitly enough that a reader of that message alone learns the value (names, numbers, dates as people
write them). A beat with state_at happens right then: the line stating it is written in that minute.
Mark each such message with conveys: [fact_id]. Do not contradict any beat or state other
decisions as settled. Never mention fact IDs, evidence, questions, or that this is generated.
Every time in the brief and every local_time you write is on the scene clock (scene.clock); people's own timezones
are in their personas. local_time is YYYY-MM-DD HH:MM:SS with irregular seconds; the scene runs from scene.start to scene.end.
Times people schedule or promise come from world_details, beats and earlier messages; do not invent new ones.
Messages follow each other in time. Gaps follow the work: quick
exchanges take seconds or minutes; waiting can take hours or the next working day.
The first message is a top-level post. reply_to is the 0-based index of an earlier line of this scene; the message
joins that line's thread. null posts a new top-level message.
style_references are real Slack excerpts from other workplaces. Borrow their texture, such as fragments,
follow-ups, hedges and corrections; never copy their wording, names, or topics.
If the brief has previous_version and revision_note, revise that version to satisfy the note.
In introduces, list each new concrete detail the scene invents (names, versions, times, numbers, document or ticket
IDs); in promises, list each commitment still open when the scene ends (who, what, by when).
Reply with JSON only: {"lines": [{"author_id", "text", "local_time", "reply_to", "conveys"}], "introduces": [],
"promises": []}.
"""


def excerpts(seeds: SeedPacket | None, key: str, count: int = 2) -> list[str]:
    if seeds is None:
        return []
    chosen = random.Random(key).sample(seeds.examples, min(count, len(seeds.examples)))
    return ["\n".join(f"{m.speaker or 'someone'}: {m.text}" for m in e.messages[:14]) for e in chosen]


def portrait(profile: SeedPersona | None) -> dict:
    """A seeded person beyond their role, and how they type in Slack."""
    if profile is None:
        return {}
    return {
        "profile": {
            "age": profile.age,
            "home": f"{profile.city}, {profile.state}",
            "background": profile.cultural_background,
            "personality": profile.persona,
            "work_style": profile.professional_persona,
            "interests": profile.hobbies,
        },
        "typing": profile.typing.model_dump(exclude={"id", "messages"}),
    }


def brief(
    catalog: Catalog,
    premise: Premise,
    plan: Plan,
    scene: ScenePlan,
    observed: dict[str, list],
    references: list[str],
    previous: SceneRecord | None,
    language: str,
    timely: dict[str, str],
) -> str:
    """`observed` holds (utc, conversation name, line) for this conversation ("recent") and others ("elsewhere"),
    plus the details and commitments of earlier scenes the participants could see."""
    conversation = next(c for c in plan.conversations if c.id == scene.conversation_id)
    people = {p.id: p for p in catalog.people}
    personas = {p.id: p for p in catalog.personas}
    facts = {f.id: f for f in catalog.facts}
    zone = personas[scene.participant_ids[0]].timezone

    def on(utc: str, form: str = "%A %Y-%m-%d %H:%M") -> str:
        """Every time in the brief is on the scene clock."""
        return to_local(utc, zone, form)

    def shown(entries):
        return [
            {
                "conversation": c,
                "author": people[x.author_id].name,
                "time": on(u, "%A %Y-%m-%d %H:%M:%S"),
                "text": x.text,
            }
            for u, c, x in entries
        ]

    value = {
        "language": language,
        "company": premise.model_dump() | {"overview": catalog.overview},
        "conversation": conversation.model_dump(include={"name", "kind", "topic", "purpose"}),
        "participants": [
            {"author_id": i, "name": people[i].name, "team": people[i].team}
            | personas[i].model_dump(exclude={"id", "seed_id", "profile"})
            | portrait(personas[i].profile)
            for i in scene.participant_ids
        ],
        "scene": {
            "situation": scene.situation,
            "clock": zone,
            "start": on(scene.start, "%Y-%m-%d %H:%M:%S"),
            "weekday": on(scene.start, "%A"),
            "end": on(scene.end, "%Y-%m-%d %H:%M:%S"),
            "length": scene.length,
            "beats": [
                {"author_id": b.author_id}
                | facts[b.fact_id].model_dump(exclude={"id", "valid_from", "valid_until"})
                | {"fact_id": b.fact_id}
                | ({"state_at": on(timely[b.fact_id], "%Y-%m-%d %H:%M")} if b.fact_id in timely else {})
                for b in scene.beats
            ],
        },
        # ponytail: every currently valid fact; filter by workstream once catalogs reach hundreds of facts.
        "established_facts": [
            f.model_dump(include={"subject", "predicate", "value"}) | {"since": on(f.valid_from)}
            for f in catalog.facts
            if f.valid_from <= scene.start and (f.valid_until is None or f.valid_until > scene.start)
        ],
        "world_details": {
            name: d.value if d.at is None else {"value": d.value, "at": on(d.at)}
            for name, d in plan.details.items()
            if d.since is None or d.since <= scene.start
        },
        "established_details": observed["established"],
        "commitments": observed["commitments"],
        "recent_messages": shown(observed["recent"]),
        "seen_elsewhere": shown(observed["elsewhere"]),
        "style_references": references,
    }
    if previous is not None and scene.revision_note:
        value |= {
            "previous_version": previous.scene.model_dump()["lines"],
            "revision_note": scene.revision_note,
        }
    return json.dumps(value, ensure_ascii=False, indent=1)


def visible(other: ScenePlan, scene: ScenePlan, conversations: dict[str, Conversation]) -> bool:
    """Whether `scene` may depend on `other`: it comes earlier and the participants could have seen it."""
    if (other.start, other.id) >= (scene.start, scene.id):
        return False
    shared = set(conversations[other.conversation_id].member_ids) & set(scene.participant_ids)
    return other.conversation_id == scene.conversation_id or bool(shared)


def observed(catalog: Catalog, plan: Plan, written: dict[str, WrittenScene], scene: ScenePlan) -> dict:
    """What the participants saw before the scene: its own conversation, the last day elsewhere, and every
    concrete detail earlier visible scenes established."""
    zones = {p.id: p.timezone for p in catalog.personas}
    conversations = {c.id: c for c in plan.conversations}
    day_before = (datetime.fromisoformat(scene.start) - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    recent, elsewhere, established, commitments = [], [], [], []
    for other in plan.scenes:
        if other.id not in written or not visible(other, scene, conversations):
            continue
        conversation = conversations[other.conversation_id]
        same = other.conversation_id == scene.conversation_id
        established += written[other.id].introduces
        commitments += written[other.id].promises
        for line in written[other.id].lines:
            utc = to_utc(line.local_time, clock(other, zones))
            if utc <= scene.start and (same or utc >= day_before):
                (recent if same else elsewhere).append((utc, conversation.name or conversation.kind, line))
    return {
        "recent": sorted(recent, key=lambda e: e[0])[-15:],
        "elsewhere": sorted(elsewhere, key=lambda e: e[0])[-20:],
        "established": established,
        "commitments": commitments,
    }


def writer_task() -> vf.Task:
    return vf.Task(vf.TaskData(prompt=None, system_prompt=WRITER_GUIDE))


def reply_json(text: str) -> str:
    return text[text.find("{") : text.rfind("}") + 1]


async def compose(
    agent, prompt: str, scene: ScenePlan, catalog: Catalog, timely: dict[str, str]
) -> tuple[WrittenScene | None, list[str], vf.Trace]:
    """One fresh writer chat; up to two correction turns when the reply breaks the scene contract."""
    zones = {p.id: p.timezone for p in catalog.personas}
    written, errors = None, ["the writer ended without a reply"]
    async with agent.interaction(writer_task()) as interaction:
        for _ in range(3):
            segment = await interaction.turn(prompt)
            if segment.terminated:
                break
            try:
                written = WrittenScene.model_validate_json(reply_json(segment.last_reply))
                errors = check_scene(scene, written, zones, timely)
            except ValueError as error:
                written, errors = None, [str(error)[:4000]]
            if not errors:
                break
            prompt = "Fix these problems and reply with the complete corrected JSON only:\n" + "\n".join(
                errors
            )
    return written, errors, interaction.trace


def scene_key(catalog: Catalog, plan: Plan, scene: ScenePlan) -> tuple[str, dict[str, str]]:
    """Digest of what a scene is written from, plus the minutes its first-stated timed facts must land in."""
    facts = {f.id: f for f in catalog.facts}
    personas = {p.id: p for p in catalog.personas}
    minutes = timed(catalog)
    timely = {
        f: minutes[f] for f, where in first_mentions(plan).items() if where == scene.id and f in minutes
    }
    key = digest(
        [
            scene.model_dump(mode="json"),
            [facts[b.fact_id].model_dump(mode="json") for b in scene.beats],
            [personas[i].model_dump(mode="json") for i in scene.participant_ids],
            timely,
        ]
    )
    return key, timely


def current(state) -> bool:
    """Whether every written scene still matches the catalog, so the candidate can be reviewed again."""
    return all(
        (record := state.scenes.get(s.id)) and record.key == scene_key(state.catalog, state.plan, s)[0]
        for s in state.plan.scenes
    )


def frozen(state) -> list[ScenePlan]:
    """Scenes exactly as approved: a later plan keeps them and may only change their revision_note."""
    if state.plan is None:
        return []
    return [
        s
        for s in state.plan.scenes
        if state.frozen_scenes.get(s.id) == scene_key(state.catalog, state.plan, s)[0]
    ]


def changed(state) -> list[str]:
    """Scenes written since the world was last approved."""
    return [s.id for s in state.plan.scenes if state.frozen_scenes.get(s.id) != state.scenes[s.id].key]
