"""The world author's tools: the reads it shares with the proposer, and the writes of its turns: the ledger (plan), a
day's conversations (post, advance, edit), and a message rewritten where it stands (revise). They are served host-side
over the world file; code is the only writer, owns every time, and checks every write."""

import random
from typing import Literal

import verifiers.v1 as vf
from worldgen_slack.toolset import watch_parent
from worldgen_slack.world import digest

from ...session import AuthoringToolsConfig, WorldReads
from ..board import Slot
from ..cast import Gaps
from ..clock import Part, bounds, present, today
from ..ledger import Plan, record_plan
from ..writing import Conversation, advance, drift, edit, post, revise
from .memory import people


class AuthorTools(WorldReads, vf.Toolset[AuthoringToolsConfig, vf.State]):
    """The world author's tools. Each write acts only in its turns; post and advance only on the day the turn writes."""

    TOOL_PREFIX = "world"

    async def setup(self) -> None:
        await super().setup()
        self.every_slot = [Slot.model_validate(s) for s in self.context.get("slots", [])]
        self.gaps = Gaps(self.settings.personas.gaps)
        self.posted = None  # the last post: its conversation, the present it left, and its result

    def _turn(self, tool: str, *modes: str) -> None:
        super()._turn(tool, *modes)
        if (
            self.config.mode == "day"
            and tool in ("post", "advance")
            and today(self._world()) != self.config.day
        ):
            raise ValueError(
                f"day {self.config.day} is closed: write recap.md, update plan.md and end your turn"
            )

    @vf.tool
    async def plan(self, ledger: Plan) -> str:
        """Plan the ledger, whole: storylines, events, facts, and the board (the facts each entry of a chapter's brief
        will rest on, as its level needs). Planned events and stated facts stay as they are; the rest may change.
        Returns what the ledger now holds."""

        def act():
            self._write(lambda copy: record_plan(copy, ledger, self.settings, self.every_slot))
            world = self._world()
            counts = [
                world.db.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                for t in ("storylines", "events", "facts")
            ]
            return f"the ledger holds {counts[0]} storylines, {counts[1]} events and {counts[2]} facts"

        return self._logged("plan", {"ledger": ledger.model_dump(mode="json")}, act, "plan", "day")

    @vf.tool
    async def post(self, conversation: Conversation) -> dict:
        """Post one conversation at the present: its lines, timed by code after the present, the facts they state,
        their reactions and promises. Returns the messages as written and the new present."""

        document = conversation.model_dump(mode="json")

        def act():
            now = present(self._world())
            if self.posted and self.posted[:2] == (document, now):
                # The same conversation again, and nothing written since: a retry after a lost response, not a
                # second conversation.
                return self.posted[2]
            seed = digest([self.settings.seed, "post", now, document])
            result = self._write(
                lambda copy: post(copy, conversation, self.settings, random.Random(seed), self.gaps)
            )
            self.posted = (document, present(self._world()), result)
            return result

        return self._logged("post", {"conversation": document}, act, "day")

    @vf.tool
    async def advance(self, to: Part | Literal["tomorrow"]) -> dict:
        """Move the present to the part `to` of today; from the night, "tomorrow" closes the day and moves to the
        next day's early part. The present's own part leaves it where it is. Reports what keeps a day open, and the day's style
        drift once it closes."""

        def act():
            day = today(self._world())
            moved = self._write(lambda copy: advance(copy, self.settings, to))
            if moved["closed_day"]:
                world = self._world()
                who, notes = people(world), drift(world, self.context.get("cards", {}), bounds(world, day)[0])
                moved["drift"] = {who[user]["real_name"]: note for user, note in notes.items()}
            return moved

        return self._logged("advance", {"to": to}, act, "day")

    @vf.tool
    async def revise(self, message_id: int, text: str) -> dict:
        """Rewrite one message's text where it stands: same author, time, thread and statements."""
        return self._logged(
            "revise",
            {"message_id": message_id, "text": text},
            lambda: self._write(lambda copy: revise(copy, message_id, text)),
            "day",
            "fix",
        )

    @vf.tool
    async def edit(self, message_id: int, text: str) -> dict:
        """Its author edits a message posted earlier today, now: the same statements, as a person fixes a typo or
        adds a word. Slack marks it edited; what it said before is gone."""
        return self._logged(
            "edit",
            {"message_id": message_id, "text": text},
            lambda: self._write(lambda copy: edit(copy, message_id, text)),
            "day",
        )


if __name__ == "__main__":
    watch_parent()  # the server process ends with the generator that started it
    AuthorTools.run()
