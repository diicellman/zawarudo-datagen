"""One Slack world in one SQLite file: code-only writes checked at insert time, and actor-scoped Slack reads."""

import base64
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


def canonical(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).encode()


def digest(value) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


ANSWER_KEY = (
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
# What a gold query may read: the world as its actor sees it, the directory, the calendar and the fact ledger.
GOLD_READS = {*SHADOWED, "users", "calendar", "storylines", "facts", "fact_relations", "evidence"}
TOOLS = (
    "search_messages",
    "search_users",
    "search_channels",
    "list_user_channels",
    "read_channel",
    "read_thread",
    "get_user",
    "list_channel_members",
    "get_reactions",
    "whoami",
)

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
        WHERE t.answer_type <> 'refusal'
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

MESSAGE = """SELECT m.id, m.channel_id, m.ts, m.ts_us, m.user_id, m.text, p.ts AS parent_ts, s.reply_count
FROM messages m LEFT JOIN main.messages p ON p.id = m.parent_id
LEFT JOIN thread_stats s ON s.root_id = m.id"""
MODIFIER = re.compile(r"(?<!\S)(in|from|before|after|on):(\S+)", re.IGNORECASE)


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


def _limit(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 100:
        raise ValueError("limit must be an integer from 1 to 100")
    return value


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
        the cap was hit) and the tables it read. `readable(table, source)` decides which tables of the file it may
        read; `local(us)` writes a time on the reader's clock (the actor's, else the company's)."""
        reader = self.actor["id"] if self.actor else "the world"
        tz = self.actor["tz"] if self.actor else self.zone()
        zone, tables = ZoneInfo(tz), set()

        def authorize(action, table, column, schema, source):
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
        return {"columns": columns, "rows": rows, "tables": sorted(tables)}

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

    def call(self, tool: str, arguments: dict) -> dict:
        if tool not in TOOLS:
            raise ValueError(f"unknown tool {tool!r}")
        return getattr(self, tool)(**arguments)

    def _page(self, items: list, scope: list, cursor: str | None, limit: int) -> dict:
        """ponytail: queries load every match, then slice; fine for worlds of thousands of messages."""
        key, offset = digest([self.actor["id"], scope]), 0
        if cursor is not None:
            try:
                saved, offset = json.loads(base64.b64decode(cursor, validate=True))
            except (ValueError, TypeError):
                raise ValueError("invalid_cursor") from None
            if saved != key or type(offset) is not int or not 0 <= offset <= len(items):
                raise ValueError("invalid_cursor")
        end = offset + _limit(limit)
        more = base64.b64encode(canonical([key, end])).decode() if end < len(items) else None
        return {"items": items[offset:end], "next_cursor": more}

    def _times(self, us: int, prefix: str = "time") -> dict:
        """A moment as a Slack client shows it to the actor: in UTC, and on the actor's own clock."""
        moment = datetime.fromtimestamp(us / 1e6, ZoneInfo(self.actor["tz"]))
        return {
            f"{prefix}_utc": datetime.fromtimestamp(us / 1e6, ZoneInfo("UTC")).strftime("%Y-%m-%dT%H:%M:%SZ"),
            f"{prefix}_local": moment.strftime("%a %Y-%m-%d %H:%M %Z"),
        }

    def _message(self, row: sqlite3.Row) -> dict:
        out = {"channel": row["channel_id"], "ts": row["ts"], "user": row["user_id"], "text": row["text"]}
        out |= self._times(row["ts_us"])
        if row["parent_ts"]:
            out["thread_ts"] = row["parent_ts"]
        elif row["reply_count"]:
            out |= {"thread_ts": row["ts"], "reply_count": row["reply_count"]}
        return out

    def _channel(self, channel_id: str) -> sqlite3.Row:
        row = self.db.execute("SELECT * FROM channels WHERE id = ?", (channel_id,)).fetchone()
        if row is None:
            raise LookupError("channel_not_found")
        return row

    def _day_us(self, value: str, days: int = 0) -> int:
        """Midnight starting the given YYYY-MM-DD (plus `days`) on the actor's clock."""
        try:
            start = datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=ZoneInfo(self.actor["tz"]))
        except ValueError:
            raise ValueError(f"dates are YYYY-MM-DD, not {value!r}") from None
        return int(
            datetime.fromordinal(start.toordinal() + days).replace(tzinfo=start.tzinfo).timestamp() * 1e6
        )

    def search_messages(
        self, query: str, sort: str = "score", cursor: str | None = None, limit: int = 20
    ) -> dict:
        if not isinstance(query, str) or not query.strip() or len(query) > 512:
            raise ValueError("query must contain 1-512 characters")
        if sort not in ("score", "timestamp"):
            raise ValueError("sort is score or timestamp")
        where, args = [], []
        for name, value in MODIFIER.findall(query):
            name, value = name.casefold(), value.strip("<>")
            if name == "in":
                where.append("m.channel_id IN (SELECT id FROM main.channels WHERE id = ? OR name = ?)")
                args += [value.split("|")[0], value.split("|")[-1].lstrip("#")]
            elif name == "from":
                where.append("m.user_id IN (SELECT id FROM main.users WHERE id = ? OR handle = ?)")
                args += [value.lstrip("@"), value.lstrip("@")]
            else:
                bounds = {"before": ("<", 0), "after": (">=", 1), "on": (">=", 0)}[name]
                where.append(f"m.ts_us {bounds[0]} ?")
                args.append(self._day_us(value, bounds[1]))
                if name == "on":
                    where.append("m.ts_us < ?")
                    args.append(self._day_us(value, 1))
        text = MODIFIER.sub(" ", query)
        phrases = re.findall(r'"([^"]+)"', text)
        terms = [f'"{" ".join(words(p))}"' for p in phrases if words(p)] + words(
            re.sub(r'"[^"]*"', " ", text)
        )
        if terms:
            sql = MESSAGE.replace(
                "FROM messages m",
                "FROM main.messages_fts JOIN messages m ON m.id = messages_fts.docid",
            ).replace("SELECT m.id,", "SELECT bm25(matchinfo(messages_fts, 'pcnalx')) AS score, m.id,")
            where.insert(0, "messages_fts MATCH ?")
            args.insert(0, " ".join(terms))
        else:
            sql = MESSAGE.replace("SELECT m.id,", "SELECT 0 AS score, m.id,")
        order = "score DESC, m.ts_us DESC" if sort == "score" and terms else "m.ts_us DESC"
        rows = self.db.execute(f"{sql} WHERE {' AND '.join(where) or '1'} ORDER BY {order}, m.id", args)
        items = [self._message(r) | {"channel_name": self._channel(r["channel_id"])["name"]} for r in rows]
        return self._page(items, ["search", query, sort], cursor, limit)

    def search_users(self, query: str, cursor: str | None = None, limit: int = 20) -> dict:
        pattern = f"%{query.strip()}%"
        rows = self.db.execute(
            """SELECT * FROM users WHERE id = ? OR real_name LIKE ? OR display_name LIKE ? OR handle LIKE ?
            OR email LIKE ? ORDER BY real_name, id""",
            (query.strip(), pattern, pattern, pattern, pattern),
        )
        items = [
            {k: r[k] for k in ("id", "real_name", "display_name", "title")} | {"name": r["handle"]}
            for r in rows
        ]
        return self._page(items, ["users", query], cursor, limit)

    def search_channels(self, query: str, cursor: str | None = None, limit: int = 20) -> dict:
        pattern = f"%{query.strip()}%"
        rows = self.db.execute(
            """SELECT c.*, (SELECT COUNT(*) FROM main.members m WHERE m.channel_id = c.id AND m.left_us IS NULL) AS n
            FROM channels c WHERE c.name IS NOT NULL AND (c.name LIKE ? OR c.topic LIKE ? OR c.purpose LIKE ?)
            ORDER BY c.name""",
            (pattern, pattern, pattern),
        )
        items = [self._channel_info(r) for r in rows]
        return self._page(items, ["channels", query], cursor, limit)

    @staticmethod
    def _channel_info(row: sqlite3.Row) -> dict:
        return {k: row[k] for k in ("id", "name", "type", "topic", "purpose")} | {
            "is_archived": bool(row["is_archived"]),
            "num_members": row["n"],
        }

    def list_user_channels(
        self, types: str = "public,private,mpim,im", cursor: str | None = None, limit: int = 50
    ) -> dict:
        kinds = [t.strip() for t in types.split(",") if t.strip()]
        if not kinds or set(kinds) - {"public", "private", "mpim", "im"}:
            raise ValueError("types is a comma-separated subset of public, private, mpim, im")
        # A direct conversation lists all its members, the user among them, as Slack's conversations.members does.
        rows = self.db.execute(
            f"""SELECT c.*, (SELECT COUNT(*) FROM main.members m WHERE m.channel_id = c.id AND m.left_us IS NULL) AS n,
            (SELECT group_concat(m.user_id) FROM main.members m WHERE m.channel_id = c.id AND m.left_us IS NULL) AS everyone
            FROM main.channels c JOIN main.members mine ON mine.channel_id = c.id AND mine.user_id = ?
              AND mine.left_us IS NULL
            WHERE c.type IN ({", ".join("?" * len(kinds))}) ORDER BY c.type, c.name, c.id""",
            (self.actor["id"], *kinds),
        )
        items = [
            self._channel_info(r)
            | ({"users": sorted((r["everyone"] or "").split(","))} if r["name"] is None else {})
            for r in rows
        ]
        return self._page(items, ["mine", kinds], cursor, limit)

    def read_channel(
        self,
        channel_id: str,
        oldest: str | None = None,
        latest: str | None = None,
        cursor: str | None = None,
        limit: int = 50,
    ) -> dict:
        self._channel(channel_id)
        where, args = ["m.channel_id = ?", "m.parent_id IS NULL"], [channel_id]
        for op, bound in ((">", oldest), ("<", latest)):
            if bound is not None:
                where.append(f"m.ts {op} ?")
                args.append(bound)
        rows = self.db.execute(f"{MESSAGE} WHERE {' AND '.join(where)} ORDER BY m.ts_us DESC", args)
        return self._page(
            [self._message(r) for r in rows], ["history", channel_id, oldest, latest], cursor, limit
        )

    def read_thread(
        self, channel_id: str, thread_ts: str, cursor: str | None = None, limit: int = 50
    ) -> dict:
        self._channel(channel_id)
        root = self.db.execute(
            "SELECT id FROM messages WHERE channel_id = ? AND ts = ? AND parent_id IS NULL",
            (channel_id, thread_ts),
        ).fetchone()
        if root is None:
            raise LookupError("thread_not_found")
        rows = self.db.execute(
            f"{MESSAGE} WHERE m.id = ? OR m.parent_id = ? ORDER BY m.ts_us", (root[0], root[0])
        )
        return self._page([self._message(r) for r in rows], ["thread", channel_id, thread_ts], cursor, limit)

    def get_user(self, user_id: str) -> dict:
        row = self.db.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        if row is None:
            raise LookupError("user_not_found")
        keys = ("id", "real_name", "display_name", "title", "tz", "status_text", "status_emoji", "email")
        return {k: row[k] for k in keys} | {
            "name": row["handle"],
            "is_bot": bool(row["is_bot"]),
            "is_admin": bool(row["is_admin"]),
            "deleted": bool(row["is_deleted"]),
            "profile": json.loads(row["profile_json"]),
        }

    def list_channel_members(self, channel_id: str, cursor: str | None = None, limit: int = 50) -> dict:
        self._channel(channel_id)
        rows = self.db.execute(
            "SELECT user_id FROM members WHERE channel_id = ? AND left_us IS NULL ORDER BY user_id",
            (channel_id,),
        )
        return self._page([r[0] for r in rows], ["members", channel_id], cursor, limit)

    def get_reactions(self, channel_id: str, ts: str) -> dict:
        self._channel(channel_id)
        message = self.db.execute(
            "SELECT id FROM messages WHERE channel_id = ? AND ts = ?", (channel_id, ts)
        ).fetchone()
        if message is None:
            raise LookupError("message_not_found")
        rows = self.db.execute(
            """SELECT emoji, COUNT(*) AS n, group_concat(user_id) AS users FROM
            (SELECT * FROM reactions WHERE message_id = ? ORDER BY created_us) GROUP BY emoji ORDER BY MIN(created_us)""",
            (message[0],),
        )
        reactions = [{"name": r["emoji"], "count": r["n"], "users": r["users"].split(",")} for r in rows]
        return {"channel": channel_id, "ts": ts, "reactions": reactions}

    def whoami(self) -> dict:
        """The user the tools act as, and the present on their clock (Slack's auth.test)."""
        me, now = self.actor, self.db.execute("SELECT value FROM world_meta WHERE key = 'now_us'").fetchone()
        team = self.db.execute("SELECT value FROM world_meta WHERE key = 'company'").fetchone()
        out = {"user_id": me["id"], "name": me["handle"], "real_name": me["real_name"], "title": me["title"]}
        out |= {"tz": me["tz"], "team": team[0] if team else None}
        return out | (self._times(int(now[0]), "now") if now else {})
