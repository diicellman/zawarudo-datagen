"""The solver's Slack surface: the read tools of Slack's own MCP server, as captured on 2026-10-14
(`tests/fixtures/slack-mcp`), over one world as one actor.

Each read returns a record: what it showed, structured, which grading reads (the messages, people and reactions it
showed). `render` writes a record as the real server's text, which is what the solver reads. Shapes the capture did not
show (a direct conversation's header, a long reaction list, a members page cursor) follow its nearest neighbours and are
marked `reconstructed`."""

import base64
import json
import re
from datetime import datetime
from zoneinfo import ZoneInfo

VERSION = "slack-mcp-2026-10-14"  # the capture this surface follows
# The read tools, by the real server's names without its `slack_` prefix (the tool server adds it).
TOOLS = (
    "search_public",
    "search_public_and_private",
    "search_channels",
    "search_users",
    "read_channel",
    "read_thread",
    "read_user_profile",
    "get_reactions",
    "list_channel_members",
    "list_user_channels",
)
KINDS = {"public_channel": "public", "private_channel": "private", "mpim": "mpim", "im": "im"}
NAMES = {v: k for k, v in KINDS.items()}
TITLES = {"public": "Public Channel", "private": "Private Channel", "mpim": "Group DM", "im": "DM"}
FILTER = re.compile(r"(?<!\S)(in|from|with|has|is|before|after|on|during|creator):(\S+)", re.IGNORECASE)
MONTHS = {m: i for i, m in enumerate(("january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"), 1)}  # fmt: skip
MENTION, CHANNEL_REF = re.compile(r"<@(U[A-Z0-9]+)>"), re.compile(r"<#(C[A-Z0-9]+|G[A-Z0-9]+)>")
CONTEXT = 3  # messages of context on each side of a search hit
PAGE = 100  # read_channel's default page: how deep a message sits is counted in these


class Failed(ValueError):
    """A tool error as the real server words it: `execution_failed: <code>`."""

    def __init__(self, code: str):
        super().__init__(f"execution_failed: {code}")


def words(text: str) -> list[str]:
    from .db import words as tokens

    return tokens(text)


def _page(world, items: list, scope: list, cursor: str | None, limit: int, top: int) -> tuple[list, str | None]:
    """A page of items; the cursor is opaque, bound to the actor and the request, as the real server's are."""
    from .db import canonical, digest

    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= top:
        raise ValueError(f"limit must be an integer from 1 to {top}")
    key, offset = digest([world.actor["id"], scope]), 0
    if cursor:
        try:
            saved, offset = json.loads(base64.b64decode(cursor, validate=True))
        except (ValueError, TypeError):
            raise Failed("invalid_cursor") from None
        if saved != key or type(offset) is not int or not 0 <= offset <= len(items):
            raise Failed("invalid_cursor")
    end = offset + limit
    return items[offset:end], (base64.b64encode(canonical([key, end])).decode() if end < len(items) else None)


def clock(world, us: int) -> str:
    """A moment as the server shows it: on the caller's clock, with its offset (`2026-10-14 13:39:40 +05`)."""
    moment = datetime.fromtimestamp(us / 1e6, ZoneInfo(world.actor["tz"]))
    offset = moment.strftime("%z")
    return f"{moment:%Y-%m-%d %H:%M:%S} {offset[:3]}" + (f":{offset[3:]}" if offset[3:] != "00" else "")


def domain(world) -> str:
    row = world.db.execute("SELECT value FROM world_meta WHERE key = 'company'").fetchone()
    return re.sub(r"[^a-z0-9]+", "-", (row[0] if row else "workspace").lower()).strip("-") + ".slack.com"


def label(user) -> str:
    return user["display_name"] or user["real_name"]


def person(world, user_id: str):
    row = world.db.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    if row is None:
        raise Failed("user_not_found")
    return row


def channel(world, channel_id: str):
    """A conversation the actor can read, by id; a user id names the direct conversation with that person."""
    if channel_id.startswith("U"):
        row = world.db.execute(
            """SELECT c.* FROM channels c WHERE c.type = 'im' AND EXISTS (SELECT 1 FROM members m WHERE m.channel_id
            = c.id AND m.user_id = ?) AND EXISTS (SELECT 1 FROM members m WHERE m.channel_id = c.id AND m.user_id = ?)""",
            (world.actor["id"], channel_id),
        ).fetchone()
    else:
        row = world.db.execute("SELECT * FROM channels WHERE id = ?", (channel_id,)).fetchone()
    if row is None:
        raise Failed("channel_not_found")
    return row


def channel_name(world, row) -> str:
    """How a conversation is named: `#name`, or for a direct one the other people's names (reconstructed)."""
    if row["name"]:
        return "#" + row["name"]
    others = [label(u) for u in world.db.execute("SELECT u.* FROM members m JOIN users u ON u.id = m.user_id WHERE m.channel_id = ? AND m.user_id <> ? AND m.left_us IS NULL ORDER BY u.real_name", (row["id"], world.actor["id"]))]  # fmt: skip
    return "DM: " + ", ".join(others) if row["type"] == "im" else "Group DM: " + ", ".join(others)


def text_of(world, text: str) -> str:
    """A message's text as Slack's API gives it: a mention carries its person's name, a channel link its name."""

    def mentioned(m):
        row = world.db.execute("SELECT * FROM users WHERE id = ?", (m.group(1),)).fetchone()
        return f"<@{m.group(1)}|{label(row)}>" if row else m.group(0)

    def linked(m):
        row = world.db.execute("SELECT name FROM main.channels WHERE id = ?", (m.group(1),)).fetchone()
        return f"<#{m.group(1)}|{row[0]}>" if row and row[0] else m.group(0)

    return CHANNEL_REF.sub(linked, MENTION.sub(mentioned, text))


MESSAGE = """SELECT m.id, m.channel_id, m.ts, m.ts_us, m.user_id, m.text, m.parent_id, p.ts AS parent_ts,
s.reply_count, s.latest_reply_us, u.real_name, u.display_name, u.email, u.is_bot, c.name AS channel_name, c.type,
m.subtype, m.edited_us
FROM messages m LEFT JOIN main.messages p ON p.id = m.parent_id LEFT JOIN thread_stats s ON s.root_id = m.id
JOIN users u ON u.id = m.user_id JOIN channels c ON c.id = m.channel_id"""


def message(world, row) -> dict:
    """One message as a record: where, who, when, its text, its thread, and its reactions' counts (not who)."""
    out = {
        "channel": row["channel_id"],
        "channel_name": row["channel_name"],
        "channel_type": row["type"],
        "ts": row["ts"],
        "user": row["user_id"],
        "user_name": row["real_name"],
        "user_email": row["email"],
        "text": text_of(world, row["text"]),
        "time_local": clock(world, row["ts_us"]),
    }
    if row["subtype"]:  # a join, a leave, or an app's post
        out["subtype"] = row["subtype"]
    if row["edited_us"]:
        out["edited"] = True
    if row["parent_ts"]:
        out["thread_ts"] = row["parent_ts"]
    elif row["reply_count"]:
        out |= {"thread_ts": row["ts"], "reply_count": row["reply_count"], "latest_reply": clock(world, row["latest_reply_us"])}  # fmt: skip
    counts = world.db.execute(
        "SELECT emoji, COUNT(*) FROM reactions WHERE message_id = ? GROUP BY emoji ORDER BY MIN(created_us)",
        (row["id"],),
    ).fetchall()
    if counts:
        out["reactions"] = [{"name": e, "count": n} for e, n in counts]
    return out


def messages(world, where: str, args: list, order: str) -> list:
    return world.db.execute(f"{MESSAGE} WHERE {where} ORDER BY {order}", args).fetchall()


def stamp(value: str | None) -> int | None:
    """A Slack ts or a Unix time, in microseconds."""
    if value in (None, ""):
        return None
    try:
        return round(float(value) * 1e6)
    except ValueError:
        raise Failed("invalid_ts_oldest" if value else "invalid_ts") from None


def briefing(world) -> str:
    """What a Slack client tells its assistant before the first call: who it acts as (the real server writes the
    caller's id into its tools' descriptions), and the date and time (the client's own)."""
    row = world.db.execute("SELECT value FROM world_meta WHERE key = 'now_us'").fetchone()
    now = datetime.fromtimestamp(int(row[0]) / 1e6, ZoneInfo(world.actor["tz"])) if row else None
    when = f" It is {now:%A, %B} {now.day}, {now:%Y}, {now:%H:%M} on your clock ({world.actor['tz']})." if now else ""
    return f"You are using Slack as the user with user_id {world.actor['id']}." + when


# ---------------------------------------------------------------------- search


def day_us(world, value: str, days: int = 0) -> int:
    """Midnight starting the given YYYY-MM-DD (plus `days`) on the actor's clock."""
    try:
        start = datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=ZoneInfo(world.actor["tz"]))
    except ValueError:
        raise Failed(f"invalid_date: dates are YYYY-MM-DD, not {value!r}") from None
    return int(datetime.fromordinal(start.toordinal() + days).replace(tzinfo=start.tzinfo).timestamp() * 1e6)


def during(world, value: str) -> tuple[int, int]:
    """`during:` a month (august, or 2026-08) or a day: its start and end on the actor's clock."""
    row = world.db.execute("SELECT value FROM world_meta WHERE key = 'now_us'").fetchone()
    now = datetime.fromtimestamp(int(row[0]) / 1e6 if row else 0, ZoneInfo(world.actor["tz"]))
    if (month := MONTHS.get(value.lower())) or re.fullmatch(r"\d{4}-\d{2}", value):
        year, month = (now.year, month) if month else map(int, value.split("-"))
        first = f"{year:04d}-{month:02d}-01"
        after = f"{year + month // 12:04d}-{month % 12 + 1:02d}-01"
        return day_us(world, first), day_us(world, after)
    return day_us(world, value), day_us(world, value, 1)


def terms(keywords: list[str]) -> tuple[list[str], list[str]]:
    """Keywords as full-text terms, every one required: a quoted phrase, or a word with punctuation in it (`2026.08`,
    `TKT-4471`), is a phrase of its parts; `-word` excludes."""
    wanted, unwanted = [], []
    for keyword in keywords or []:
        negative = keyword.startswith("-") and len(keyword) > 1
        parts = words(keyword.strip("-").strip('"'))
        if parts:
            (unwanted if negative else wanted).append(f'"{" ".join(parts)}"' if len(parts) > 1 else parts[0])
    return wanted, unwanted


def search(
    world,
    private: bool,
    keywords: list[str] | None = None,
    filters: str = "",
    natural_language_query: str = "",
    after: str | None = None,
    before: str | None = None,
    sort: str = "score",
    sort_dir: str = "desc",
    limit: int = 20,
    cursor: str | None = None,
    include_context: bool = True,
    include_bots: bool = False,
    channel_types: str | None = None,
    only_my_channels: bool = False,
    content_types: str | None = None,
    context_channel_id: str | None = None,
    max_context_length: int | None = None,
    response_format: str = "detailed",
) -> dict:
    """Messages matching every keyword and filter, in channels the actor can read (public ones only, for the public
    search), by relevance (BM25) or time. Semantic re-ranking is off, as on the captured account."""
    wanted, unwanted = terms(keywords or [])
    found = FILTER.findall(filters or "")
    if not wanted and not found:
        raise Failed("invalid_arguments: give keywords or filters")
    if sort not in ("score", "timestamp") or sort_dir not in ("asc", "desc"):
        raise Failed("invalid_arguments: sort is score or timestamp, sort_dir asc or desc")
    kinds = ["public"] if not private else [KINDS[t.strip()] for t in (channel_types or "public_channel,private_channel,mpim,im").split(",") if t.strip() in KINDS]  # fmt: skip
    where, args = [f"c.type IN ({', '.join('?' * len(kinds))})"], list(kinds)
    if only_my_channels:
        where.append("EXISTS (SELECT 1 FROM members x WHERE x.channel_id = m.channel_id AND x.user_id = ? AND x.left_us IS NULL)")  # fmt: skip
        args.append(world.actor["id"])
    if not include_bots:
        where.append("u.is_bot = 0")
    grouped: dict[str, list] = {}
    for name, value in found:
        grouped.setdefault(name.lower(), []).append(value)
    for name, values in grouped.items():
        either, both = [], []
        for value in values:
            clause, more = modifier(world, name, value)
            (both if name in ("with", "has") else either).append((clause, more))
        if either:
            where.append("(" + " OR ".join(c for c, _ in either) + ")")
            args += [a for _, more in either for a in more]
        for clause, more in both:
            where.append(clause)
            args += more
    for bound, op in ((after, ">="), (before, "<=")):
        if (us := stamp(bound)) is not None:
            where.append(f"m.ts_us {op} ?")
            args.append(us)
    for term in unwanted:
        where.append("m.id NOT IN (SELECT docid FROM main.messages_fts WHERE messages_fts MATCH ?)")
        args.append(term)
    score = "0"
    if wanted:
        where.append("m.id IN (SELECT docid FROM main.messages_fts WHERE messages_fts MATCH ?)")
        args.append(" ".join(wanted))
        score = "(SELECT bm25(matchinfo(messages_fts, 'pcnalx')) FROM main.messages_fts WHERE messages_fts MATCH ? AND docid = m.id)"  # fmt: skip
    direction = "ASC" if sort_dir == "asc" else "DESC"
    order = f"m.ts_us {direction}, m.id" if sort == "timestamp" or not wanted else "score DESC, m.ts_us DESC, m.id"
    sql = MESSAGE.replace("SELECT m.id,", f"SELECT {score} AS score, m.id,") + f" WHERE {' AND '.join(where)} ORDER BY {order}"  # fmt: skip
    rows = world.db.execute(sql, ([" ".join(wanted)] if wanted else []) + args).fetchall()
    shown, more = _page(world, rows, ["search", private, keywords, filters, after, before, sort, sort_dir, channel_types, only_my_channels, include_bots], cursor, limit, 20)  # fmt: skip
    items = []
    for row in shown:
        hit = message(world, row)
        if include_context:
            hit["context"] = context(world, row, max_context_length)
        items.append(hit)
    return {"items": items, "next_cursor": more, "format": response_format}


def modifier(world, name: str, value: str) -> tuple[str, list]:
    """One search modifier as SQL over the actor's view."""
    bare = value.strip("<>").split("|")[0]
    if name == "in":
        if bare.startswith("@") or bare.startswith("U"):
            who = user_ref(world, bare)
            return "(c.type = 'im' AND EXISTS (SELECT 1 FROM members x WHERE x.channel_id = c.id AND x.user_id = ?))", [who]  # fmt: skip
        return "(c.id = ? OR c.name = ?)", [bare.lstrip("#"), bare.lstrip("#")]
    if name == "from":
        return "m.user_id = ?", [user_ref(world, bare)]
    if name == "with":  # a thread or a direct conversation the person is in
        who = user_ref(world, bare)
        return """(EXISTS (SELECT 1 FROM messages t WHERE t.user_id = ? AND (t.id = COALESCE(m.parent_id, m.id)
            OR t.parent_id = COALESCE(m.parent_id, m.id))) OR (c.type IN ('im', 'mpim') AND EXISTS (SELECT 1 FROM
            members x WHERE x.channel_id = c.id AND x.user_id = ?)))""", [who, who]
    if name == "has":
        if value.lower() == "link":
            return "(m.text LIKE '%http://%' OR m.text LIKE '%https://%')", []
        if value.lower() == "reaction":
            return "EXISTS (SELECT 1 FROM reactions r WHERE r.message_id = m.id)", []
        if value.startswith(":") and value.endswith(":") and len(value) > 2:
            return "EXISTS (SELECT 1 FROM reactions r WHERE r.message_id = m.id AND r.emoji = ?)", [value.strip(":")]
        return "0", []  # pins, files and saved items: the world has none
    if name == "is":
        if value.lower() == "thread":
            return "(m.parent_id IS NOT NULL OR EXISTS (SELECT 1 FROM messages r WHERE r.parent_id = m.id))", []
        if value.lower() == "dm":
            return "c.type IN ('im', 'mpim')", []
        return "0", []
    if name in ("before", "after", "on"):
        if name == "on":
            return "(m.ts_us >= ? AND m.ts_us < ?)", [day_us(world, value), day_us(world, value, 1)]
        return ("m.ts_us < ?", [day_us(world, value)]) if name == "before" else ("m.ts_us >= ?", [day_us(world, value, 1)])  # fmt: skip
    if name == "during":
        start, end = during(world, value)
        return "(m.ts_us >= ? AND m.ts_us < ?)", [start, end]
    return "0", []  # creator: is a canvas's


def user_ref(world, value: str) -> str:
    """A person named in a filter: <@U…>, U…, @handle or handle."""
    bare = value.strip("<>@").split("|")[0]
    row = world.db.execute("SELECT id FROM users WHERE id = ? OR handle = ?", (bare, bare)).fetchone()
    return row[0] if row else bare


def context(world, row, length: int | None) -> dict:
    """The messages around a hit, as the real server gives them: a reply's thread (before and after it); a root's
    replies after it; a top-level message's channel around it."""
    root = row["parent_id"]
    if root:
        before = messages(world, "(m.id = ? OR m.parent_id = ?) AND m.ts_us < ?", [root, root, row["ts_us"]], "m.ts_us DESC")[:CONTEXT][::-1]  # fmt: skip
        after = messages(world, "m.parent_id = ? AND m.ts_us > ?", [root, row["ts_us"]], "m.ts_us")[:CONTEXT]
    elif row["reply_count"]:
        before, after = [], messages(world, "m.parent_id = ?", [row["id"]], "m.ts_us")[:CONTEXT]
    else:
        near = "m.channel_id = ? AND m.parent_id IS NULL AND m.ts_us {} ?"
        before = messages(world, near.format("<"), [row["channel_id"], row["ts_us"]], "m.ts_us DESC")[:2][::-1]
        after = messages(world, near.format(">"), [row["channel_id"], row["ts_us"]], "m.ts_us")[:2]
    cut = lambda m: m | {"text": m["text"][:length]} if length else m  # noqa: E731
    return {"before": [cut(message(world, r)) for r in before], "after": [cut(message(world, r)) for r in after]}


# ---------------------------------------------------------------------- reads


def read_channel(world, channel_id: str, oldest: str | None = None, latest: str | None = None, limit: int = PAGE, cursor: str | None = None, response_format: str = "detailed") -> dict:  # fmt: skip
    """A conversation's top-level messages, newest first, between `oldest` and `latest`."""
    row = channel(world, channel_id)
    where, args = ["m.channel_id = ?", "m.parent_id IS NULL"], [row["id"]]
    for bound, op in ((oldest, ">="), (latest, "<=")):
        if (us := stamp(bound)) is not None:
            where.append(f"m.ts_us {op} ?")
            args.append(us)
    rows = messages(world, " AND ".join(where), args, "m.ts_us DESC, m.id DESC")
    shown, more = _page(world, rows, ["history", row["id"], oldest, latest], cursor, limit, 100)
    return {"channel": row["id"], "channel_name": channel_name(world, row), "items": [message(world, r) for r in shown], "next_cursor": more, "format": response_format}  # fmt: skip


def read_thread(world, channel_id: str, message_ts: str, oldest: str | None = None, latest: str | None = None, limit: int = 100, cursor: str | None = None, response_format: str = "detailed") -> dict:  # fmt: skip
    """A thread: its parent message, then its replies oldest first."""
    row = channel(world, channel_id)
    parent = messages(world, "m.channel_id = ? AND m.ts = ?", [row["id"], message_ts], "m.id")
    if not parent:
        raise Failed("thread_not_found")
    if parent[0]["parent_id"]:  # a reply names its thread
        parent = messages(world, "m.id = ?", [parent[0]["parent_id"]], "m.id")
    where, args = ["m.parent_id = ?"], [parent[0]["id"]]
    for bound, op in ((oldest, ">="), (latest, "<=")):
        if (us := stamp(bound)) is not None:
            where.append(f"m.ts_us {op} ?")
            args.append(us)
    replies = messages(world, " AND ".join(where), args, "m.ts_us, m.id")
    shown, more = _page(world, replies, ["thread", row["id"], message_ts, oldest, latest], cursor, limit, 1000)
    items = [message(world, parent[0])] + [message(world, r) for r in shown]
    return {"channel": row["id"], "parent": parent[0]["ts"], "replies": len(replies), "items": items, "next_cursor": more, "format": response_format}  # fmt: skip


def profile(row) -> dict:
    return {
        "id": row["id"],
        "handle": row["handle"],
        "real_name": row["real_name"],
        "display_name": row["display_name"],
        "title": row["title"] or "",
        "email": row["email"],
        "tz": row["tz"],
        "status_text": row["status_text"] or "",
        "status_emoji": row["status_emoji"] or "",
        "is_bot": bool(row["is_bot"]),
        "is_admin": bool(row["is_admin"]),
        "deleted": bool(row["is_deleted"]),
        "fields": json.loads(row["profile_json"]),
    }


def read_user_profile(world, user_id: str | None = None, include_locale: bool = False, response_format: str = "detailed") -> dict:  # fmt: skip
    """A person's profile; with no user_id, the caller's own."""
    row = person(world, user_id or world.actor["id"])
    company = world.db.execute("SELECT value FROM world_meta WHERE key = 'company'").fetchone()
    return profile(row) | {"organization": company[0] if company else "", "format": response_format}


def get_reactions(world, channel_id: str, message_ts: str) -> dict:
    """Each reaction on a message, with who added it."""
    row = channel(world, channel_id)
    found = world.db.execute("SELECT id FROM messages WHERE channel_id = ? AND ts = ?", (row["id"], message_ts)).fetchone()  # fmt: skip
    if found is None:
        raise Failed("message_not_found")
    rows = world.db.execute(
        """SELECT r.emoji, r.user_id, u.real_name FROM reactions r JOIN users u ON u.id = r.user_id
        WHERE r.message_id = ? ORDER BY r.created_us""",
        (found[0],),
    ).fetchall()
    reactions: dict[str, dict] = {}
    for emoji, user_id, name in rows:
        entry = reactions.setdefault(emoji, {"name": emoji, "count": 0, "users": [], "names": []})
        entry["count"] += 1
        entry["users"].append(user_id)
        entry["names"].append(name)
    return {"channel": row["id"], "ts": message_ts, "reactions": list(reactions.values())}


def list_channel_members(world, channel_id: str, cursor: str | None = None, limit: int = 30, response_format: str = "detailed", include_bots: bool = False, include_deleted: bool = False) -> dict:  # fmt: skip
    """A conversation's current members (not a direct one's): profiles, or ids only, which are not filtered."""
    row = channel(world, channel_id)
    if row["type"] == "im":
        raise Failed("method_not_supported_for_channel_type")
    if response_format not in ("detailed", "concise", "ids_only"):
        raise Failed("invalid_arguments: response_format is detailed, concise or ids_only")
    filtered = response_format != "ids_only"
    rows = world.db.execute(
        "SELECT u.* FROM members m JOIN users u ON u.id = m.user_id WHERE m.channel_id = ? AND m.left_us IS NULL"
        + (" AND u.is_bot = 0" if filtered and not include_bots else "")
        + (" AND u.is_deleted = 0" if filtered and not include_deleted else "")
        + " ORDER BY u.id",
        (row["id"],),
    ).fetchall()
    shown, more = _page(world, rows, ["members", row["id"], response_format, include_bots, include_deleted], cursor, limit, 30)  # fmt: skip
    return {"channel": row["id"], "total": len(rows), "items": [profile(r) for r in shown], "next_cursor": more, "format": response_format}  # fmt: skip


def list_user_channels(world, types: str = "public_channel,private_channel", format: str = "full", limit: int = 50, cursor: str | None = None, name_prefix: str | None = None, exclude_archived: bool = False, team_id: str | None = None) -> dict:  # fmt: skip
    """The conversations the caller is a member of, archived ones included unless excluded."""
    kinds = [KINDS.get(t.strip()) for t in types.split(",") if t.strip()]
    if not kinds or None in kinds:
        raise Failed("invalid_arguments: types is a comma-separated subset of public_channel, private_channel, mpim, im")  # fmt: skip
    rows = world.db.execute(
        f"""SELECT c.* FROM channels c JOIN members mine ON mine.channel_id = c.id AND mine.user_id = ?
        AND mine.left_us IS NULL WHERE c.type IN ({", ".join("?" * len(kinds))})""" + (" AND c.is_archived = 0" if exclude_archived else "") + " ORDER BY CASE c.type WHEN 'public' THEN 0 WHEN 'private' THEN 1 WHEN 'mpim' THEN 2 ELSE 3 END, c.name, c.id",  # fmt: skip
        (world.actor["id"], *kinds),
    ).fetchall()
    if name_prefix:
        rows = [r for r in rows if (r["name"] or "").lower().startswith(name_prefix.lower())]
    shown, more = _page(world, rows, ["mine", types, name_prefix, exclude_archived], None if name_prefix else cursor, limit, 200)  # fmt: skip
    items = []
    for r in shown:
        people = [u for (u,) in world.db.execute("SELECT user_id FROM members WHERE channel_id = ? AND left_us IS NULL ORDER BY user_id", (r["id"],))]  # fmt: skip
        items.append({"id": r["id"], "name": channel_name(world, r), "type": r["type"], "is_archived": bool(r["is_archived"])} | ({"users": people} if r["name"] is None else {}))  # fmt: skip
    return {"total": len(rows), "items": items, "next_cursor": more, "format": format}


def matches(texts: list[str], keywords: list[str]) -> bool:
    """Every keyword in the texts (`-word` in none), word by word from a word's start: a partial name finds its
    person, as the real server's people and channel search does."""
    bag = [w for t in texts if t for w in words(t)]
    for keyword in keywords or []:
        negative = keyword.startswith("-") and len(keyword) > 1
        parts = words(keyword.strip("-").strip('"'))
        if not parts:
            continue
        hit = any(all(bag[i + j].startswith(w) for j, w in enumerate(parts)) for i in range(len(bag) - len(parts) + 1))  # fmt: skip
        if hit == negative:
            return False
    return True


def search_users(world, keywords: list[str] | None = None, natural_language_query: str = "", limit: int = 20, cursor: str | None = None, response_format: str = "detailed") -> dict:  # fmt: skip
    """People whose name, handle, email, title or profile fields have every keyword."""
    if not keywords:
        raise Failed("invalid_arguments: give keywords")
    rows = [r for r in world.db.execute("SELECT * FROM users WHERE is_deleted = 0 ORDER BY real_name, id") if matches([r["real_name"], r["display_name"], r["handle"], r["email"], r["title"] or "", r["id"], *map(str, json.loads(r["profile_json"]).values())], keywords)]  # fmt: skip
    shown, more = _page(world, rows, ["users", keywords], cursor, limit, 20)
    return {"items": [profile(r) for r in shown], "next_cursor": more, "format": response_format}


def search_channels(world, keywords: list[str] | None = None, natural_language_query: str = "", channel_types: str = "public_channel", include_archived: bool = False, limit: int = 20, cursor: str | None = None, response_format: str = "detailed") -> dict:  # fmt: skip
    """Named channels the caller can see whose name, topic or purpose has every keyword."""
    if not keywords:
        raise Failed("invalid_arguments: give keywords")
    kinds = [KINDS.get(t.strip()) for t in channel_types.split(",") if t.strip()]
    if not kinds or set(kinds) - {"public", "private"}:
        raise Failed("invalid_arguments: channel_types is public_channel, private_channel or both")
    rows = world.db.execute(
        f"""SELECT c.*, u.real_name AS creator_name FROM channels c LEFT JOIN users u ON u.id = c.creator_id
        WHERE c.name IS NOT NULL AND c.type IN ({", ".join("?" * len(kinds))})""" + ("" if include_archived else " AND c.is_archived = 0") + " ORDER BY c.name",  # fmt: skip
        kinds,
    ).fetchall()
    rows = [r for r in rows if matches([r["name"].replace("-", " "), r["topic"], r["purpose"]], keywords)]
    shown, more = _page(world, rows, ["channels", keywords, channel_types, include_archived], cursor, limit, 20)
    items = [{"id": r["id"], "name": r["name"], "type": r["type"], "topic": r["topic"], "purpose": r["purpose"], "is_archived": bool(r["is_archived"]), "creator": r["creator_id"], "creator_name": r["creator_name"], "created": clock(world, r["created_us"])} for r in shown]  # fmt: skip
    return {"items": items, "next_cursor": more, "format": response_format}


def read(world, tool: str, arguments: dict) -> dict:
    """One read tool's record, as the actor."""
    if tool not in TOOLS:
        raise ValueError(f"unknown tool {tool!r}")
    if tool in ("search_public", "search_public_and_private"):
        if tool == "search_public":
            arguments = {k: v for k, v in arguments.items() if k != "channel_types"}
        return search(world, tool == "search_public_and_private", **arguments)
    return globals()[tool](world, **arguments)


# ---------------------------------------------------------------------- the text the solver reads


def who(m: dict) -> str:
    return f"{m['user_name']} <{m['user_email']}>"


def permalink(world, m: dict) -> str:
    base = f"https://{domain(world)}/archives/{m['channel']}/p{m['ts'].replace('.', '')}"
    return base + (f"?thread_ts={m['thread_ts']}&cid={m['channel']}" if m.get("thread_ts") else "")


def tail(m: dict) -> str:
    out = " (edited)" if m.get("edited") else ""  # reconstructed: Slack shows an edit's mark after its text
    if m.get("reply_count"):
        out += f"\nThread: {m['reply_count']} replies (latest: {m['latest_reply']})"
    if m.get("reactions"):
        out += "\nReactions: " + ", ".join(f"{r['name']} ({r['count']})" for r in m["reactions"])
    return out


def render(world, tool: str, record: dict) -> dict:
    """A record as the real server's response: JSON whose values are text."""
    concise = record.get("format") == "concise"
    if tool == "read_channel":
        head = f"Channel: {record['channel_name']} ({record['channel']})\n\n"
        if concise:
            body = "\n\n".join(f"{who(m)}: {m['text']} [{m['time_local']}]" for m in record["items"])
        else:
            body = "\n\n".join(f"=== Message from {who(m)} ({m['user']}) at {m['time_local']} === \nMessage TS: {m['ts']}\n{m['text']}{tail(m)}" for m in record["items"])  # fmt: skip
        more = f"There are more messages available. To view the next page, use cursor: `{record['next_cursor']}`\n" if record["next_cursor"] else "There are no more messages available.\n"  # fmt: skip
        return {"messages": head + body, "pagination_info": more}
    if tool == "read_thread":
        parent, replies = record["items"][0], record["items"][1:]
        more = f"There are more messages in this thread. To view the next page, use cursor: `{record['next_cursor']}`\n" if record["next_cursor"] else "There are no more messages in this thread.\n"  # fmt: skip  # reconstructed
        if concise:
            text = f"THREAD: {parent['text']} [{who(parent)}]\n\n" + "".join(f"> {who(m)}: {m['text']}\n" for m in replies)  # fmt: skip
            return {"messages": text, "pagination_info": more}
        text = f"=== THREAD PARENT MESSAGE ===\nFrom: {who(parent)} ({parent['user']})\nTime: {parent['time_local']}\nMessage TS: {parent['ts']}\n{parent['text']}" + ("\nReactions: " + ", ".join(f"{r['name']} ({r['count']})" for r in parent["reactions"]) if parent.get("reactions") else "") + "\n\n"  # fmt: skip
        if not replies:
            return {"messages": text + "No thread messsages\n", "pagination_info": more}
        text += f"=== THREAD REPLIES ({record['replies']} total) ===\n\n"
        text += "\n".join(f"--- Reply {k} of {record['replies']} ---\nFrom: {who(m)} ({m['user']})\nTime: {m['time_local']}\nMessage TS: {m['ts']}\n{m['text']}" + ("\nReactions: " + ", ".join(f"{r['name']} ({r['count']})" for r in m["reactions"]) if m.get("reactions") else "") + "\n" for k, m in enumerate(replies, 1))  # fmt: skip
        return {"messages": text, "pagination_info": more}
    if tool in ("search_public", "search_public_and_private"):
        items = record["items"]
        more = f"For the next page of results use cursor `{record['next_cursor']}`\n" if record["next_cursor"] else "End of results - No more pages available.\\n"  # fmt: skip
        if not items:
            return {"results": "# Search Results for: \n\nNo results found.\n", "pagination_info": more}
        if concise:
            lines = "".join(f"{k}. {('#' + m['channel_name']) if m['channel_name'] else 'DM'} - {m['user_name']}: {m['text']} {m['time_local']}\n" for k, m in enumerate(items, 1))  # fmt: skip
            return {"results": f"# Search Results for: \n\n## Messages ({len(items)} results)\n({len(items)} results)\n{lines}", "pagination_info": more}  # fmt: skip
        out = f"# Search Results for: \n\n## Messages ({len(items)} results)\n"
        for k, m in enumerate(items, 1):
            where = f"#{m['channel_name']}" if m["channel_name"] else "DM"  # reconstructed for direct conversations
            out += f"### Result {k} of {len(items)}\nChannel: {where} (ID: {m['channel']})\nFrom: {who(m)} (ID: {m['user']}) \nTime: {m['time_local']}\nMessage_ts: {m['ts']}\n"  # fmt: skip
            out += (f"Reply count: {m['reply_count']}\n" if m.get("reply_count") else "") + f"Permalink: [link]({permalink(world, m)})\nText: \n{m['text']}\n"  # fmt: skip
            for side in ("before", "after"):
                if near := m.get("context", {}).get(side):
                    out += f"Context {side}: \n" + "".join(f"- From: {'' if c.get('subtype') in ('channel_join', 'channel_leave') else who(c)} (ID: {'U00' if c.get('subtype') in ('channel_join', 'channel_leave') else c['user']}) \n  Message_ts: {c['ts']}\n  {c['text']}\n" for c in near)  # fmt: skip  # a join's author shows as the capture does
            out += "\n---\n\n"
        return {"results": out, "pagination_info": more}
    if tool == "get_reactions":
        if not record["reactions"]:
            return {"result": "No reactions found on this message."}
        return {"result": "Reactions on message:\n\n" + "\n".join(f":{r['name']}: × {r['count']} — " + ", ".join(f"{n} ({u})" for n, u in zip(r["names"], r["users"])) for r in record["reactions"])}  # fmt: skip
    if tool == "read_user_profile":
        p = record
        if concise:
            return {"result": f"{p['handle']} ({p['display_name']})\n{p['organization']}\n{p['title']}\n{p['email']}\nStatus: {p['status_emoji']} {p['status_text']}\n"}  # fmt: skip
        fields = "".join(f"{k}: {v}\n" for k, v in p["fields"].items())  # reconstructed: custom profile fields
        return {"result": f"User ID: {p['id']}\nUsername: {p['handle']}\nDisplay Name: {p['display_name']}\nReal Name: {p['real_name']}\nPronouns: \nTitle: {p['title']}\nEmail: {p['email']}\nOrganization Name: {p['organization']}\nPhone: \nStatus: {p['status_emoji']} {p['status_text']}\nTimezone: {p['tz']}\n{fields}Admin: {'Yes' if p['is_admin'] else 'No'}\nOwner: No\nBot: {'Yes' if p['is_bot'] else 'No'}\nRestricted: No\n"}  # fmt: skip
    if tool == "list_channel_members":
        more = f"More members available. Use cursor: `{record['next_cursor']}`\n" if record["next_cursor"] else "No more members available.\n"  # fmt: skip  # reconstructed
        head = f"# Channel Members ({record['total']})\n\n"
        if record["format"] == "ids_only":
            return {"members": head + "".join(f"{p['id']}\n" for p in record["items"]), "pagination_info": more}
        if record["format"] == "concise":
            return {"members": head + "".join(f"@{p['handle']} ({p['display_name']})\n" for p in record["items"]), "pagination_info": more}  # fmt: skip
        body = "".join(f"## Member {k}\nUser ID: {p['id']}\nUsername: @{p['handle']}\nDisplay Name: {p['display_name']}\nReal Name: {p['real_name']}\nEmail: {p['email']}\n\n" for k, p in enumerate(record["items"], 1))  # fmt: skip
        return {"members": head + body, "pagination_info": more}
    if tool == "list_user_channels":
        if not record["items"]:
            return {"result": "You are not a member of any channels matching that request."}  # reconstructed
        head = f"## My Channels (showing {len(record['items'])} of {record['total']} total)\n\n"
        if record["format"] == "ids_only":
            body = "".join(f"{c['id']}\n" for c in record["items"])
        elif record["format"] == "names_only":
            body = "".join(f"{c['name']}\n" for c in record["items"])
        else:
            body = "".join(f"### {c['name']}\n- **ID:** {c['id']}\n- **Type:** {TITLES[c['type']]}\n- **Archived:** {'Yes' if c['is_archived'] else 'No'}\n\n" for c in record["items"])  # fmt: skip
        more = f"\nMore channels available. Use cursor: `{record['next_cursor']}`\n" if record["next_cursor"] else ""  # reconstructed
        return {"result": head + body + more}
    if tool == "search_users":
        items, more = record["items"], f"For the next page of results use cursor `{record['next_cursor']}`\n" if record["next_cursor"] else "End of results - No more pages available.\\n"  # fmt: skip
        if not items:
            return {"results": "# Search Results for: \n\nNo results found.\n", "pagination_info": more}
        body = "".join(f"### Result {k} of {len(items)}\nName: {p['real_name']}\nUser ID: {p['id']}\nTitle: {p['title']}\nEmail: {p['email']}\nTimezone: {p['tz']}\nPermalink: [link](https://{domain(world)}/team/{p['id']})\n\n---\n\n" for k, p in enumerate(items, 1))  # fmt: skip
        return {"results": f"# Search Results for: \n\n## Users ({len(items)} results)\n" + body, "pagination_info": more}  # fmt: skip
    if tool == "search_channels":
        items, more = record["items"], f"For the next page of results use cursor `{record['next_cursor']}`\n" if record["next_cursor"] else "End of results - No more pages available.\\n"  # fmt: skip
        if not items:
            return {"results": "# Search Results for: \n\nNo results found.\n", "pagination_info": more}
        body = ""
        for k, c in enumerate(items, 1):
            body += f"### Result {k} of {len(items)}\nName: #{c['name']}\n"
            body += f"Topic: {c['topic']}\n" if c["topic"] else ""  # reconstructed: the captured channel had none
            body += f"Purpose: {c['purpose']}\n" if c["purpose"] else ""
            body += f"Creator: {c['creator_name']} (<@{c['creator']})\nCreated: {c['created']}\nPermalink: [link](https://{domain(world)}/archives/{c['id']})\nIs Archived: {'true' if c['is_archived'] else 'false'}\nChannel Type: {NAMES[c['type']]}\n\n---\n\n"  # fmt: skip
        return {"results": f"# Search Results for: \n\n## Channels ({len(items)} results)\n" + body, "pagination_info": more}  # fmt: skip
    raise ValueError(f"unknown tool {tool!r}")
