"""One Slack world in one SQLite file, opened as one person: what they can see, through views that hide the rest."""

import hashlib
import json
import math
import re
import sqlite3
import struct
import unicodedata
from pathlib import Path
from urllib.request import pathname2url

SCHEMA = Path(__file__).with_name("schema.sql")


def canonical(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).encode()


def digest(value) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path: Path | str) -> str:
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


# Per connection, TEMP views under the real names hide what one actor cannot read; unqualified names resolve to them.
SHADOWED = ("channels", "members", "messages", "message_mentions", "reactions", "thread_stats")
SCOPE = """
DROP VIEW IF EXISTS temp.thread_stats; DROP VIEW IF EXISTS temp.reactions; DROP VIEW IF EXISTS temp.message_mentions;
DROP VIEW IF EXISTS temp.messages; DROP VIEW IF EXISTS temp.members; DROP VIEW IF EXISTS temp.channels;
DROP TABLE IF EXISTS temp.actor; CREATE TEMP TABLE actor (id TEXT NOT NULL);
CREATE TEMP VIEW channels AS SELECT c.* FROM main.channels c
  WHERE c.type = 'public' OR EXISTS (SELECT 1 FROM main.members m, temp.actor a
    WHERE m.channel_id = c.id AND m.user_id = a.id AND m.left_us IS NULL);
CREATE TEMP VIEW members AS SELECT m.* FROM main.members m JOIN temp.channels c ON c.id = m.channel_id;
CREATE TEMP VIEW messages AS SELECT m.* FROM main.messages m
  JOIN temp.channels c ON c.id = m.channel_id WHERE m.is_deleted = 0;
CREATE TEMP VIEW message_mentions AS SELECT x.* FROM main.message_mentions x JOIN temp.messages m ON m.id = x.message_id;
CREATE TEMP VIEW reactions AS SELECT x.* FROM main.reactions x JOIN temp.messages m ON m.id = x.message_id;
CREATE TEMP VIEW thread_stats AS SELECT p.id AS root_id, COUNT(r.id) AS reply_count, MAX(r.ts_us) AS latest_reply_us
  FROM temp.messages p JOIN temp.messages r ON r.parent_id = p.id GROUP BY p.id;
"""
# Of the workspace, the columns the solver's tools show (slack.py), or let it work out (a mention is <@U..> in a
# message's text; a thread's latest reply is the last one it reads; a channel's members are those who have not left): a gold
# query filters on and returns these alone. S3's gold read reactions.created_us, when the tools say who reacted but
# never when.
SHOWN = {
    "users": {
        "id",
        "handle",
        "real_name",
        "display_name",
        "email",
        "title",
        "tz",
        "status_text",
        "status_emoji",
        "is_bot",
        "is_admin",
        "is_deleted",
        "profile_json",
    },
    "channels": {"id", "name", "type", "topic", "purpose", "is_archived"},
    "members": {"channel_id", "user_id", "left_us"},
    "messages": {"id", "channel_id", "ts_us", "ts", "user_id", "parent_id", "text", "is_deleted", "subtype"},
    "message_mentions": {"message_id", "user_id"},
    "reactions": {"message_id", "user_id", "emoji"},
    "thread_stats": {"root_id", "reply_count", "latest_reply_us"},
}


def bm25(info: bytes, k1: float = 1.2, b: float = 0.75) -> float:
    """Okapi BM25 of one row from FTS4 matchinfo(..., 'pcnalx')."""
    values = struct.unpack(f"={len(info) // 4}I", info)
    phrases, columns, rows = values[:3]
    average, length = values[3 : 3 + columns], values[3 + columns : 3 + 2 * columns]
    hits = values[3 + 2 * columns :]
    score = 0.0
    for i in range(phrases * columns):
        here, _, docs = hits[3 * i : 3 * i + 3]
        if here:
            column = i % columns
            idf = math.log(1 + (rows - docs + 0.5) / (docs + 0.5))
            norm = 1 - b + b * length[column] / max(average[column], 1)
            score += idf * here * (k1 + 1) / (here + k1 * norm)
    return score


def words(text: str) -> list[str]:
    return re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", text).casefold())


class World:
    """Code is the only writer. Reads go through `scope(actor)` and return what that actor can see in Slack:
    public channels to everyone, private channels and DMs to their current members."""

    def __init__(self, path: Path | str, actor: str | None = None, writable: bool = False) -> None:
        self.path = Path(path).resolve()
        uri = f"file:{pathname2url(str(self.path))}?mode={'rw' if writable else 'ro'}"
        self.db = sqlite3.connect(uri, uri=True, isolation_level=None, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.create_function("bm25", -1, bm25, deterministic=True)
        self.actor: sqlite3.Row | None = None
        if actor is not None:
            self.scope(actor)

    def close(self) -> None:
        self.db.close()

    def zone(self) -> str:
        """The company clock, or UTC for a world without one."""
        row = self.db.execute("SELECT value FROM world_meta WHERE key = 'zone'").fetchone()
        return row[0] if row else "UTC"

    def scope(self, actor: str) -> None:
        self.actor = self.db.execute("SELECT * FROM users WHERE id = ?", (actor,)).fetchone()
        if self.actor is None:
            raise LookupError("user_not_found")
        self.db.executescript(SCOPE)
        self.db.execute("INSERT INTO temp.actor VALUES (?)", (actor,))
