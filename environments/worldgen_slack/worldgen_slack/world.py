"""One Slack world in one SQLite file: code-only writes checked at insert time, and actor-scoped Slack reads."""

import hashlib
import json
import math
import re
import sqlite3
import struct
import tempfile
import time
import unicodedata
from collections.abc import Collection
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from urllib.request import pathname2url
from zoneinfo import ZoneInfo


SCHEMA = Path(__file__).with_name("schema.sql")
# The schema's version, and how a world of an older one comes forward to it: the columns each version added (the
# tables it added are made from schema.sql). A world adopted by a chapter or a forge is brought forward first.
SCHEMA_VERSION = 2
COLUMNS = {
    2: [
        ("users", "deactivated_us", "ALTER TABLE users ADD COLUMN deactivated_us INTEGER"),
        ("channels", "archived_us", "ALTER TABLE channels ADD COLUMN archived_us INTEGER"),
        (
            "messages",
            "subtype",
            "ALTER TABLE messages ADD COLUMN subtype TEXT CHECK (subtype IS NULL OR subtype IN ('bot_message', 'channel_join', 'channel_leave'))",
        ),
        ("events", "post_by", "ALTER TABLE events ADD COLUMN post_by TEXT REFERENCES users(id)"),
        ("tasks", "intent", "ALTER TABLE tasks ADD COLUMN intent TEXT"),
    ],
}


def canonical(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).encode()


def digest(value) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path: Path | str) -> str:
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


ANSWER_KEY = (
    "queue",
    "integrations",
    "artifacts",
    "commitments",
    "task_facts",
    "board",
    "tasks",
    "evidence",
    "fact_relations",
    "facts",
    "events",
    "scene_messages",
    "scenes",
    "storylines",
    "calendar",
)
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
# What a gold query may read: the world as its actor sees it, the directory, the calendar, the fact ledger and the
# promises people made (what was promised, by whom, and how it ended: kept, changed or dropped).
GOLD_READS = {
    *SHADOWED,
    "users",
    "calendar",
    "storylines",
    "facts",
    "fact_relations",
    "evidence",
    "commitments",
    "artifacts",
}
# Of the workspace, the columns the tools below show, or let the solver work out (a mention is <@U..> in a message's
# text; a thread's latest reply is the last one it reads; a channel's members are those who have not left): a gold
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

FIRST = """first AS (
  SELECT e.fact_id, MIN(m.ts_us) AS ts_us FROM evidence e JOIN messages m ON m.id = e.message_id
  WHERE e.role IN ('anchor', 'supporting') GROUP BY e.fact_id)"""

# Each query returns the rows that break one rule; the message is the correction its author receives.
RULES = {
    "stated_off_day": (
        f"""WITH {FIRST} SELECT f.id AS fact, f.day FROM facts f JOIN first ON first.fact_id = f.id
        JOIN calendar c ON c.day = f.day WHERE first.ts_us < c.start_us OR first.ts_us >= c.end_us""",
        "fact {fact} is planned for day {day} but is first stated on another day",
    ),
    "before_it_happened": (
        """SELECT e.fact_id AS fact, e.message_id AS message FROM evidence e JOIN facts f ON f.id = e.fact_id
        JOIN messages m ON m.id = e.message_id
        WHERE f.moment_kind = 'happened' AND e.role IN ('anchor', 'supporting') AND m.ts_us < f.moment_us""",
        "message {message} states fact {fact} before it happened",
    ),
    "told_late": (
        f"""WITH {FIRST} SELECT f.id AS fact FROM facts f JOIN first ON first.fact_id = f.id
        WHERE f.moment_kind = 'scheduled' AND first.ts_us >= f.moment_us""",
        "fact {fact} is scheduled: it is first stated before its moment",
    ),
    "out_of_order": (
        f"""WITH {FIRST} SELECT r.src_fact AS fact, r.dst_fact AS other, r.kind FROM fact_relations r
        JOIN first a ON a.fact_id = r.src_fact JOIN first b ON b.fact_id = r.dst_fact
        WHERE r.kind IN ('after', 'supersedes') AND a.ts_us <= b.ts_us""",
        "fact {fact} must be first stated after fact {other} ({kind})",
    ),
    "anchor_token_missing": (
        """SELECT e.fact_id AS fact, e.message_id AS message, e.anchor_token AS token FROM evidence e
        JOIN messages m ON m.id = e.message_id
        WHERE e.anchor_token IS NOT NULL AND instr(lower(m.text), lower(e.anchor_token)) = 0""",
        "message {message} must contain {token!r} to state fact {fact}",
    ),
    "ambiguous_value": (
        """WITH RECURSIVE chain(src, dst) AS (
          SELECT src_fact, dst_fact FROM fact_relations WHERE kind = 'supersedes'
          UNION SELECT c.src, r.dst_fact FROM chain c JOIN fact_relations r ON r.src_fact = c.dst AND r.kind = 'supersedes')
        SELECT f.id AS fact, g.id AS other FROM facts f JOIN facts g ON g.id > f.id
          AND lower(trim(g.subject)) = lower(trim(f.subject)) AND lower(trim(g.attribute)) = lower(trim(f.attribute))
        WHERE f.is_decoy = 0 AND g.is_decoy = 0 AND lower(trim(f.value)) <> lower(trim(g.value))
          AND NOT EXISTS (SELECT 1 FROM chain WHERE (src = f.id AND dst = g.id) OR (src = g.id AND dst = f.id))""",
        "facts {fact} and {other} give different values for one subject and attribute; one must supersede the "
        "other, or one is a decoy",
    ),
    "unreadable_evidence": (
        """SELECT t.id AS task, tf.fact_id AS fact FROM tasks t JOIN task_facts tf ON tf.task_id = t.id
        WHERE t.answer_type NOT IN ('refusal', 'status')
          AND EXISTS (SELECT 1 FROM evidence e WHERE e.fact_id = tf.fact_id AND e.role = 'anchor')
          AND NOT EXISTS (
            SELECT 1 FROM evidence e JOIN messages m ON m.id = e.message_id JOIN channels c ON c.id = m.channel_id
            WHERE e.fact_id = tf.fact_id AND e.role = 'anchor' AND m.is_deleted = 0
              AND (c.type = 'public' OR EXISTS (SELECT 1 FROM members mb WHERE mb.channel_id = c.id
                                                AND mb.user_id = t.actor_id AND mb.left_us IS NULL)))""",
        "task {task}: its actor cannot read any message that states fact {fact}",
    ),
    "future_message": (
        """SELECT m.id AS message FROM messages m
        WHERE m.ts_us > (SELECT CAST(value AS INTEGER) FROM world_meta WHERE key = 'now_us')""",
        "message {message} is later than the world's present",
    ),
    "outside_slot": (
        """SELECT sm.message_id AS message, sm.scene_id AS scene FROM scene_messages sm
        JOIN scenes s ON s.id = sm.scene_id JOIN messages m ON m.id = sm.message_id
        WHERE m.ts_us < s.slot_start_us OR m.ts_us >= s.slot_end_us""",
        "message {message} falls outside the slot of scene {scene}",
    ),
    "event_moment": (
        """SELECT f.id AS fact, f.event_id AS event FROM facts f JOIN events e ON e.id = f.event_id
        WHERE f.moment_us IS NOT e.moment_us OR f.moment_zone IS NOT e.zone""",
        "fact {fact} is about event {event}, so it keeps the event's moment",
    ),
    # A world written in time order only ever adds at its present: message ids grow with time.
    "written_out_of_order": (
        """SELECT id AS message FROM (SELECT id, ts_us,
          MAX(ts_us) OVER (ORDER BY id ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS latest FROM messages)
        WHERE ts_us < latest AND EXISTS (SELECT 1 FROM world_meta WHERE key = 'chronological')""",
        "message {message} is earlier than a message written before it; nothing is written into the past",
    ),
    "commitment_order": (
        """SELECT c.id AS commitment FROM commitments c JOIN messages made ON made.id = c.message_id
        JOIN messages closed ON closed.id = c.closed_by WHERE closed.ts_us <= made.ts_us""",
        "commitment {commitment} is closed before it is made",
    ),
}
# Hold once a stage is complete, not after every insert: tasks are planned before their facts are written.
COMPLETION = {
    "unstated_fact": (
        """SELECT tf.task_id AS task, tf.fact_id AS fact FROM task_facts tf
        WHERE NOT EXISTS (SELECT 1 FROM evidence e WHERE e.fact_id = tf.fact_id AND e.role = 'anchor')""",
        "fact {fact}, which task {task} needs, is never stated",
    ),
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

    @classmethod
    def create(cls, path: Path | str) -> "World":
        if Path(path).exists():
            raise ValueError(f"{path} already exists")
        with sqlite3.connect(path) as db:
            db.executescript(SCHEMA.read_text())
        return cls(path, writable=True)

    def close(self) -> None:
        self.db.close()

    def migrate(self) -> int:
        """Bring a world of an older schema forward, in place and idempotently: each version's columns, then every
        table, index, view and trigger of schema.sql it lacks. Returns the version it was at."""
        row = self.db.execute("SELECT value FROM world_meta WHERE key = 'schema_version'").fetchone()
        was = int(row[0]) if row else 1
        fresh = sqlite3.connect(":memory:")
        fresh.executescript(SCHEMA.read_text())
        objects = fresh.execute(
            "SELECT type, name, sql FROM sqlite_master WHERE sql IS NOT NULL AND name NOT LIKE 'sqlite_%' ORDER BY rowid"
        ).fetchall()
        fresh.close()
        have = {r[0] for r in self.db.execute("SELECT name FROM sqlite_master")}
        with self.batch():
            for version in range(was + 1, SCHEMA_VERSION + 1):
                for table, column, sql in COLUMNS.get(version, []):
                    if column not in {r[1] for r in self.db.execute(f"PRAGMA table_info({table})")}:
                        self.db.execute(sql)
            for kind, name, sql in objects:
                if name not in have and not name.startswith("messages_fts_"):
                    self.db.execute(sql)
            self.db.execute(
                "INSERT OR REPLACE INTO world_meta (key, value) VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )
        return was

    # ------------------------------------------------------------------ writes (generation code only)
    def insert(self, table: str, rows: list[dict]) -> list[int]:
        ids = []
        for row in rows:
            if not re.fullmatch(r"\w+", table) or not all(re.fullmatch(r"\w+", c) for c in row):
                raise ValueError("invalid table or column name")
            sql = f"INSERT INTO {table} ({', '.join(row)}) VALUES ({', '.join('?' * len(row))})"
            ids.append(self.db.execute(sql, list(row.values())).lastrowid)
        return ids

    @contextmanager
    def batch(self):
        """All or nothing: a trigger failure or a broken rule rolls the batch back and raises ValueError whose text
        is the correction for the batch's author."""
        self.db.execute("SAVEPOINT batch")
        try:
            yield self
            if broken := self.violations():
                raise ValueError("; ".join(broken))
        except BaseException as error:
            self.db.execute("ROLLBACK TO batch")
            self.db.execute("RELEASE batch")
            if isinstance(error, sqlite3.IntegrityError):
                raise ValueError(str(error)) from None
            raise
        self.db.execute("RELEASE batch")

    @contextmanager
    def renumbered(self):
        """A copy of the world whose message ids are reversed and moved past the highest one, with the map back to
        the world's ids. A query that finds messages by what they say answers the same on it; one that names a
        message by its id, or orders by id instead of time, does not."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "renumbered.sqlite"
            self.snapshot(path)
            db = sqlite3.connect(path)
            (top,) = db.execute("SELECT COALESCE(MAX(id), 0) FROM messages").fetchone()
            mirror = 2 * top + 1
            with db:
                db.execute(f"UPDATE messages SET id = {mirror} - id, parent_id = {mirror} - parent_id")
                for table in ("message_mentions", "reactions", "scene_messages", "evidence"):
                    db.execute(f"UPDATE {table} SET message_id = {mirror} - message_id")
            db.close()
            yield World(path), lambda message_id: mirror - message_id

    @contextmanager
    def trial(self):
        """A writable copy to apply and check a whole document on, gold queries included (they read through their
        own connection, so they need committed rows). On success the copy becomes the world; on failure it is
        dropped and the world is untouched."""
        with tempfile.TemporaryDirectory(prefix="world-trial-") as directory:
            self.snapshot(Path(directory) / "trial.sqlite")
            copy = World(Path(directory) / "trial.sqlite", writable=True)
            try:
                yield copy
                copy.db.backup(self.db)
            finally:
                copy.close()

    def checks(self, complete: bool | Collection[str] = False) -> list[tuple[str, dict]]:
        """`complete` adds the completion rules: True for every task, or the ids of the tasks they are due for."""
        if self.actor is not None:
            raise ValueError("rules run on the whole world, not on one actor's view")
        rules = RULES | (COMPLETION if complete else {})
        rows = [(name, dict(row)) for name, (sql, _) in rules.items() for row in self.db.execute(sql)]
        return [(n, r) for n, r in rows if complete is True or n not in COMPLETION or r["task"] in complete]

    def violations(self, complete: bool | Collection[str] = False) -> list[str]:
        messages = {name: text for name, (_, text) in (RULES | COMPLETION).items()}
        return [f"{name}: " + messages[name].format(**row) for name, row in self.checks(complete)]

    def snapshot(self, path: Path | str) -> None:
        with sqlite3.connect(path) as out:
            self.db.backup(out)
        out.close()

    def solver_copy(self, path: Path | str) -> None:
        """The world as solvers get it: the answer key is dropped, not hidden."""
        self.snapshot(path)
        with sqlite3.connect(path) as out:
            for table in ANSWER_KEY:
                out.execute(f"DROP TABLE IF EXISTS {table}")
        out.execute("VACUUM")
        out.close()

    # ------------------------------------------------------------------ gold answers
    def gold(self, actor: str, sql: str, max_rows: int = 100, seconds: float = 2.0) -> dict:
        """Run a task's gold query exactly as `actor` sees the world: one read-only SELECT over the shadow views,
        the directory, the calendar and the fact ledger. Returns its columns, rows and the tables it read; every
        message_id or user_id it returns must be one the actor can see."""
        reader = World(self.path, actor=actor)
        try:
            return reader._gold(sql, max_rows, seconds)
        finally:
            reader.close()

    def _gold(self, sql: str, max_rows: int, seconds: float) -> dict:
        actor = self.actor["id"]
        out = self.select(
            sql,
            max_rows,
            seconds,
            lambda table, source: table in GOLD_READS and not (table in SHADOWED and source is None),
        )
        messages = {r["message_id"] for r in out["rows"] if r.get("message_id") is not None}
        if hidden := messages - {r[0] for r in self.db.execute("SELECT id FROM messages")}:
            raise ValueError(f"the gold query returns messages {sorted(hidden)} that {actor} cannot read")
        users = {r["user_id"] for r in out["rows"] if r.get("user_id") is not None}
        if unknown := users - {r[0] for r in self.db.execute("SELECT id FROM users")}:
            raise ValueError(f"the gold query returns unknown users {sorted(unknown)}")
        return out

    def select(
        self, sql: str, max_rows: int = 100, seconds: float = 2.0, readable=lambda table, source: True
    ) -> dict:
        """One read-only SELECT within a time limit: its columns, at most `max_rows` + 1 rows (one past the cap shows
        the cap was hit), the tables it read, and the columns it names itself (`reads`, as table.column: a view's
        own reads are the view's). `readable(table, source)` decides which tables of the file it may read;
        `local(us)` writes a time on the reader's clock (the actor's, else the company's)."""
        reader = self.actor["id"] if self.actor else "the world"
        tz = self.actor["tz"] if self.actor else self.zone()
        zone, tables, reads = ZoneInfo(tz), set(), set()

        def authorize(action, table, column, schema, source):
            if action == sqlite3.SQLITE_READ and source is None and column:
                reads.add(f"{table}.{column}")
            if action == sqlite3.SQLITE_READ and schema == "main":
                if not readable(table, source):
                    return sqlite3.SQLITE_DENY
                tables.add(table)
            allowed = (
                sqlite3.SQLITE_SELECT,
                sqlite3.SQLITE_READ,
                sqlite3.SQLITE_FUNCTION,
                sqlite3.SQLITE_RECURSIVE,
            )
            return sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY

        def local(us, place=None):
            moment = datetime.fromtimestamp(us / 1e6, ZoneInfo(place) if place else zone)
            return moment.strftime("%Y-%m-%d %H:%M %Z")

        deadline = time.monotonic() + seconds
        self.db.create_function("local", -1, local, deterministic=True)
        self.db.set_progress_handler(lambda: time.monotonic() > deadline, 10_000)
        self.db.set_authorizer(authorize)
        try:
            cursor = self.db.execute(sql)
            columns = [c[0] for c in cursor.description or ()]
            rows = [dict(zip(columns, r)) for r in cursor.fetchmany(max_rows + 1)]
        except (sqlite3.Error, sqlite3.Warning) as error:
            raise ValueError(f"the query fails as {reader}: {error}") from None
        finally:
            self.db.set_authorizer(None)
            self.db.set_progress_handler(None, 0)
        return {"columns": columns, "rows": rows, "tables": sorted(tables), "reads": sorted(reads)}

    def zone(self) -> str:
        """The company clock, or UTC for a world without one."""
        row = self.db.execute("SELECT value FROM world_meta WHERE key = 'zone'").fetchone()
        return row[0] if row else "UTC"

    def rank(self, question: str, message_ids: list[int]) -> int | None:
        """The best rank of these messages when the actor searches the question's own words (any word, BM25)."""
        if not message_ids or not (terms := list(dict.fromkeys(words(question)))):
            return None
        rows = self.db.execute(
            """SELECT m.id FROM main.messages_fts JOIN messages m ON m.id = messages_fts.docid
            WHERE messages_fts MATCH ? ORDER BY bm25(matchinfo(messages_fts, 'pcnalx')) DESC, m.ts_us DESC, m.id""",
            (" OR ".join(terms),),
        )
        hits = [r[0] for r in rows]
        return min((hits.index(m) + 1 for m in message_ids if m in hits), default=None)

    # ------------------------------------------------------------------ reads (Slack, as one actor)
    def scope(self, actor: str) -> None:
        self.actor = self.db.execute("SELECT * FROM users WHERE id = ?", (actor,)).fetchone()
        if self.actor is None:
            raise LookupError("user_not_found")
        self.db.executescript(SCOPE)
        self.db.execute("INSERT INTO temp.actor VALUES (?)", (actor,))
