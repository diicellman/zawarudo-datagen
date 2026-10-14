"""How a world's people write: the activity and style measured on their messages."""

import statistics
from collections import defaultdict
from datetime import datetime
from zoneinfo import ZoneInfo

from .text import EMOJI, PEOPLE

# Words: a short line has at most SHORT, a long one more than LONG, as the typing profiles measure.
SHORT, LONG = 4, 20


def activity(world, start_us: int | None = None, end_us: int | None = None) -> dict:
    """The workspace's shape as `[activity]` measures it, over all messages or those in [start_us, end_us): how many,
    and the shares that are thread replies, carry a reaction, sit in DMs, mention someone or carry an emoji."""
    span = (start_us if start_us is not None else -1 << 62, end_us if end_us is not None else 1 << 62)
    rows = world.db.execute(
        """SELECT m.id, m.parent_id IS NOT NULL AS reply, c.type IN ('im', 'mpim') AS dm, m.text,
        EXISTS (SELECT 1 FROM reactions r WHERE r.message_id = m.id) AS reacted,
        EXISTS (SELECT 1 FROM message_mentions x WHERE x.message_id = m.id) AS mentions
        FROM messages m JOIN channels c ON c.id = m.channel_id JOIN users u ON u.id = m.user_id
        WHERE {PEOPLE} AND m.ts_us >= ? AND m.ts_us < ?""".format(PEOPLE=PEOPLE),
        span,
    ).fetchall()
    share = lambda values: round(sum(values) / len(rows), 3) if rows else 0.0  # noqa: E731
    return {
        "messages": len(rows),
        "reply_share": share([r["reply"] for r in rows]),
        "reaction_rate": share([r["reacted"] for r in rows]),
        "dm_share": share([r["dm"] for r in rows]),
        "mention_rate": share([r["mentions"] for r in rows]),
        "emoji_rate": share([bool(EMOJI.search(r["text"])) for r in rows]),
    }


def style(world, since_us: int = 0) -> dict:
    """Surface statistics of voice and timing, overall and per author, of the messages since `since_us` (a chapter's
    own); the world judge compares them with people's typing."""
    rows = world.db.execute(
        f"SELECT m.user_id, m.text, m.ts_us, u.tz FROM messages m JOIN users u ON u.id = m.user_id WHERE {PEOPLE} AND m.is_deleted = 0 AND m.ts_us >= ?",
        (since_us,),
    ).fetchall()

    def measure(items):
        texts = [r["text"].strip() for r in items]
        local = [datetime.fromtimestamp(r["ts_us"] / 1e6, ZoneInfo(r["tz"])) for r in items]
        share = lambda values: round(sum(values) / len(values), 2)  # noqa: E731
        return {
            "messages": len(texts),
            "median_words": statistics.median(len(t.split()) for t in texts),
            "short": share([len(t.split()) <= SHORT for t in texts]),
            "long": share([len(t.split()) > LONG for t in texts]),
            "question": share(["?" in t for t in texts]),
            "lowercase_start": share([t[:1].islower() for t in texts]),
            "off_hours": share([t.hour < 7 or t.hour >= 20 or t.weekday() >= 5 for t in local]),
        }

    if not rows:
        return {}
    by_author = defaultdict(list)
    for r in rows:
        by_author[r["user_id"]].append(r)
    return measure(rows) | {"authors": {a: measure(items) for a, items in sorted(by_author.items())}}
