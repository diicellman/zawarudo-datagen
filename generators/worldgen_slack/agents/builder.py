"""The builder: one storyline's scenes, placed on the organization and the calendar."""

from ..contracts import PARTS, ScenePlan
from .author import AuthorTask

PART_HOURS = ", ".join(f"{part} {start:02d}:00-{end:02d}:00" for part, (start, end) in PARTS.items())

SCENES_GUIDE = f"""Phase scenes: plan the scenes of one storyline. A writer turns each scene into messages; code places
them in time. world.sqlite holds the people, conversations and facts, and every message written so far.
- A scene is one stretch of talk in one conversation (channel_id), among participants who are its members.
- day is a day of `calendar`; part is when on that day it starts, on the company clock: {PART_HOURS}. during names a
  fact with happened_at or scheduled_for: the scene then starts at that moment, on its day.
- situation: what is going on, what each participant wants, and what gets decided or stays open. length: about how
  many messages.
- beats: the facts of `storyline` the scene states, each by a participant (author_id). Every fact of the storyline
  has a beat. A fact's first beat is in its channel, by its author, on its day (as the facts table says).
- A fact that happened is first stated in a part that starts after its moment, or in a scene during it. A fact with
  after or supersedes is first stated in a later part or day than those facts, or in the same scene.
- Scenes without beats carry the rest of the work around the facts. new_messages_hint is a rough range of messages
  for this storyline, not a quota.
- Scenes in frozen_scene_ids are approved: keep them unchanged except revision_note, a concrete instruction to
  rewrite that scene's text. Changing anything else of a scene rewrites it.
The world review checks that the workspace reads as real work and tells its storylines.
"""


class BuilderTask(AuthorTask):
    outputs = {"scenes": ScenePlan}
    guides = {"scenes": SCENES_GUIDE}


def scenes_context(settings, state, storyline: dict, calendar: list[dict], frozen: list[str]) -> dict:
    per_storyline = settings.tasks.messages_per_task * settings.tasks.per_storyline
    return {
        "phase": "scenes",
        "storyline": storyline["id"],
        "summary": storyline["summary"],
        "language": settings.language,
        "calendar": calendar,
        "new_messages_hint": f"{per_storyline * 6 // 10}-{per_storyline * 13 // 10}",
        "frozen_scene_ids": frozen,
        "feedback": state.feedback,
        "previous_output": state.plans.get(storyline["id"]),
    }
