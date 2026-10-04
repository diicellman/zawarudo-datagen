"""The scene writer: one scene's messages per fresh chat, without tools; code checks and places them."""

import json
import random
from datetime import datetime
from zoneinfo import ZoneInfo

import verifiers.v1 as vf

from ..contracts import Layout, PlanError, Scene, SeedPacket, SeedPersona, WrittenScene, window, world_meta

WRITER_GUIDE = """Write one scene of a company's Slack: the messages its participants post. The brief gives the
company, the conversation, the participants, the scene and what came before it.
- participants may post. profile is who a person is outside their job; typing is how they type in Slack, measured on
  a real person: median words per message, and the shares of messages with at most four words, with a question,
  starting lowercase, and with an emoji. The world review compares each person's messages with their typing.
- recent_messages came earlier in this conversation; seen_elsewhere are messages the participants saw elsewhere in the
  day before; established and commitments are details and open promises of earlier scenes they saw. The world review
  checks that no message contradicts them and that promises are kept or explicitly changed.
- beats are facts the scene states: the beat's author_id states it in a line whose conveys lists its fact_id, and
  when the fact has an anchor, that line contains the anchor exactly.
- A time or a date appears only as {at:fact_id} for a fact in `moments`; code writes it on each reader's clock.
  Mentions are <@user_id>. No line names a fact id.
- reply_to is the index of an earlier line of this scene whose thread the line joins; null posts a new message.
  layout.shape says how the conversation is threaded: "flat" posts every line as a new message, "thread" makes every
  line after the first a reply in its thread, "free" leaves it to you.
- reactions are emoji reactions to a line: {"user_id", "emoji"} with an emoji name such as "eyes". Exactly
  layout.reactions lines carry reactions, from participants or layout.audience, never from the line's author.
- layout.short: at least this many lines are short, at most 4 words. layout.long, when set: at most this many lines
  are long, more than 20 words. Which lines, and whose, is yours.
- pause is "hours" when an hour or more passes before the line, and "" otherwise.
- style_references are real Slack excerpts from other workplaces: borrow their texture, never their wording, names or
  topics.
- previous_version and revision_note, when present: revise that version as the note asks.
- introduces lists each new concrete detail the scene invents (a name, a version, a number, a document); promises
  lists each commitment still open when the scene ends.
Write in `language`, about `length` lines. Reply with JSON only:
{"lines": [{"author_id", "text", "reply_to", "conveys", "pause", "reactions"}], "introduces": [], "promises": []}.
"""


def portrait(profile: SeedPersona) -> dict:
    """A seeded person beyond their job, and how they type in Slack."""
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


def excerpts(seeds: SeedPacket | None, key: str, count: int = 2) -> list[str]:
    if seeds is None:
        return []
    chosen = random.Random(key).sample(seeds.examples, min(count, len(seeds.examples)))
    return ["\n".join(f"{m.speaker or 'someone'}: {m.text}" for m in e.messages[:14]) for e in chosen]


def brief(
    world,
    scene: Scene,
    premise: dict,
    profiles: dict[str, SeedPersona],
    references,
    language,
    previous=None,
    layout: Layout | None = None,
    background: bool = False,
):
    """Everything one scene is written from, read from the world: who, where, which facts, and what came before."""
    zone = ZoneInfo(world_meta(world, "zone"))
    start = window(world, scene.day, scene.part)[0]
    if scene.during:
        start = world.db.execute("SELECT moment_us FROM facts WHERE id = ?", (scene.during,)).fetchone()[0]
    channel = dict(
        world.db.execute(
            "SELECT name, type, topic, purpose FROM channels WHERE id = ?", (scene.channel_id,)
        ).fetchone()
    )
    users = {r["id"]: r for r in world.db.execute("SELECT * FROM users")}
    seen = {r[0] for r in world.db.execute(f"SELECT DISTINCT channel_id FROM members WHERE user_id IN ({', '.join('?' * len(scene.participants))})", scene.participants)}  # fmt: skip

    def shown(rows):
        return [
            {"author": users[r["user_id"]]["real_name"], "time": f"{datetime.fromtimestamp(r['ts_us'] / 1e6, zone):%a %H:%M}", "text": r["text"]}
            for r in rows
        ]  # fmt: skip

    recent = world.db.execute(
        "SELECT * FROM messages WHERE channel_id = ? AND ts_us < ? AND is_deleted = 0 ORDER BY ts_us DESC LIMIT 15",
        (scene.channel_id, start),
    ).fetchall()[::-1]
    elsewhere = [
        r
        for r in world.db.execute(
            "SELECT * FROM messages WHERE channel_id <> ? AND ts_us BETWEEN ? AND ? AND is_deleted = 0 ORDER BY ts_us",
            (scene.channel_id, start - 86_400_000_000, start),
        ).fetchall()
        if r["channel_id"] in seen
    ][-20:]
    earlier = [
        json.loads(r[0])
        for r in world.db.execute(
            "SELECT plan_json FROM scenes WHERE slot_start_us < ? ORDER BY slot_start_us", (start,)
        )
        if json.loads(r[0]).get("channel_id") in seen
    ]
    facts = {r["id"]: dict(r) for r in world.db.execute("SELECT * FROM facts")}
    moments = [
        {"fact_id": f, "about": f"{x['subject']}: {x['attribute']}"}
        for f, x in facts.items()
        if x["moment_us"] is not None
        and not background
        and (x["day"] <= scene.day or f in {b.fact for b in scene.beats})
    ]
    value = {
        "language": language,
        "company": premise,
        "conversation": channel,
        "participants": [
            {
                "user_id": u,
                "name": users[u]["real_name"],
                "title": users[u]["title"],
                "team": json.loads(users[u]["profile_json"]).get("Team"),
            }
            | portrait(profiles[u])
            for u in scene.participants
        ],  # fmt: skip
        "scene": {
            "situation": scene.situation,
            "day": scene.day,
            "weekday": f"{datetime.fromtimestamp(start / 1e6, zone):%A}",
            "part": scene.part,
            "length": scene.length,
            "beats": [
                {"fact_id": b.fact, "author_id": b.author_id}
                | {k: facts[b.fact][k] for k in ("subject", "attribute", "value", "anchor", "summary")}
                for b in scene.beats
            ],
        },
        "layout": (layout or Layout()).model_dump()
        | {
            "audience": [{"user_id": u, "name": users[u]["real_name"]} for u in (layout or Layout()).audience]
        },
        "moments": moments,
        "recent_messages": shown(recent),
        "seen_elsewhere": shown(elsewhere),
        "established": [d for e in earlier for d in e.get("introduces", [])],
        "commitments": [p for e in earlier for p in e.get("promises", [])],
        "style_references": references,
    }
    if previous is not None and scene.revision_note:
        value |= {"previous_version": previous, "revision_note": scene.revision_note}
    return json.dumps(value, ensure_ascii=False, indent=1)


def writer_task() -> vf.Task:
    return vf.Task(vf.TaskData(prompt=None, system_prompt=WRITER_GUIDE))


def reply_json(text: str) -> str:
    return text[text.find("{") : text.rfind("}") + 1]


async def compose(agent, prompt: str, accept) -> tuple[WrittenScene | None, list[str], vf.Trace]:
    """One fresh writer chat. `accept` checks and stores a reply; what it rejects goes back as a correction turn,
    twice at most. A PlanError is the plan's to fix and ends the chat."""
    written, errors = None, ["the writer ended without a reply"]
    async with agent.interaction(writer_task()) as interaction:
        for _ in range(3):
            segment = await interaction.turn(prompt)
            if segment.terminated:
                break
            try:
                written = WrittenScene.model_validate_json(reply_json(segment.last_reply))
                accept(written)
                errors = []
            except PlanError:
                raise
            except ValueError as error:
                written, errors = None, [str(error)[:4000]]
            if not errors:
                break
            prompt = "Fix these problems and reply with the complete corrected JSON only:\n" + "\n".join(
                errors
            )
    return written, errors, interaction.trace
