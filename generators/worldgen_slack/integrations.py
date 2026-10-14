"""Apps in the workspace, posting by code: deploys, alerts, tickets, builds and reminders.

The author picks a workspace's integrations in its organization (each with the channel it posts in and the subjects it
posts about); code writes every bot post. A post is queued at its moment and written when the present reaches it, in
time order with everyone's messages (`drain`), so the world stays one written forward. A post's text comes from its
kind's template and its subject, or from a ledger event that *happened*, whose title states no fact: a bot never
states an answer first. Bots and joins are no one's messages: no quota, share or task counts them (`PEOPLE`)."""

import datetime as dt
import json
import random
import re

from worldgen_slack.world import digest

from .contracts import APP_KINDS, TIME_LITERAL, queue

# What each kind of app posts: about one of its subjects (noise, drawn by the seed on workdays), and when a ledger
# event it reports happens. {n} is a running number, {who} a member of its channel, {subject} or {title} the matter.
KINDS = {
    "deploys": {
        "noise": "Deployed {subject} to staging · build #{n}",
        "event": "✅ {title}: deployed to production",
    },
    "alerts": {
        "noise": "🔴 Triggered: {subject}",
        "resolved": "✅ Resolved: {subject}",
        "event": "🔴 Triggered: {title}",
    },
    "tickets": {
        "noise": "{prefix}-{n} created: {subject} · assigned to <@{who}>",
        "event": "{prefix}-{n} created: {title} · assigned to <@{who}>",
    },
    "ci": {
        "noise": "Build #{n} on main {outcome} · {minutes} min",
        "event": "Build #{n} for {title} passed ✅ · {minutes} min",
    },
    "reminders": {"noise": "Reminder: {subject}", "event": "Reminder: {title}"},
}
FAILS = 0.15  # a build's chance to fail
assert tuple(KINDS) == APP_KINDS


def apps(world) -> list[dict]:
    """The workspace's integrations: each bot's id, name, kind, channel and subjects."""
    return [
        dict(r)
        for r in world.db.execute(
            """SELECT i.user_id, i.kind, i.channel_id, u.real_name AS name, u.profile_json FROM integrations i
            JOIN users u ON u.id = i.user_id ORDER BY i.user_id"""
        )
    ]


def subjects_of(app: dict) -> list[str]:
    return json.loads(app["profile_json"]).get("subjects", [])


def prefix_of(app: dict) -> str:
    return json.loads(app["profile_json"]).get("prefix", "TKT")


def number(world, app: dict, start: int = 1000) -> int:
    """An app's next running number (builds, tickets), counted from what it has queued."""
    n = world.db.execute("SELECT COUNT(*) FROM queue WHERE user_id = ?", (app["user_id"],)).fetchone()[0]
    return start + int(digest([app["user_id"]])[:3], 16) % 4000 + n


def member(world, channel_id: str, rng, at_us: int) -> str:
    """A person in the channel at that moment."""
    people = [
        r[0]
        for r in world.db.execute(
            "SELECT m.user_id FROM members m JOIN users u ON u.id = m.user_id WHERE m.channel_id = ? AND m.joined_us <= ? AND (m.left_us IS NULL OR m.left_us > ?) AND u.is_bot = 0 ORDER BY m.user_id",
            (channel_id, at_us, at_us),
        )
    ]
    return rng.choice(people)


def text(world, app: dict, template: str, rng, at_us: int, **fields) -> str:
    fields.setdefault("n", number(world, app))
    fields.setdefault("prefix", prefix_of(app))
    fields.setdefault("outcome", "failed ❌" if rng.random() < FAILS else "passed ✅")
    fields.setdefault("minutes", rng.randint(4, 28))
    if "{who}" in template:
        fields.setdefault("who", member(world, app["channel_id"], rng, at_us))
    return template.format(**fields)


def open_at(world, channel_id: str, at_us: int) -> bool:
    row = world.db.execute("SELECT archived_us FROM channels WHERE id = ?", (channel_id,)).fetchone()
    return row[0] is None or row[0] > at_us


def queue_day(world, settings, day: int) -> int:
    """A workday's routine posts of each app, drawn by the seed: `bot_posts` of them, about its subjects, in the
    working hours; an alert resolves itself within the hour; reminders keep their morning. Weekends are quiet."""
    rate = settings.activity.bot_posts
    date, start, end = world.db.execute(
        "SELECT date, start_us, end_us FROM calendar WHERE day = ?", (day,)
    ).fetchone()
    if rate <= 0 or dt.date.fromisoformat(date).weekday() >= 5:
        return 0
    queued = 0
    for app in apps(world):
        rng = random.Random(digest([settings.seed, "bots", app["user_id"], day]))
        subjects = subjects_of(app)
        count = 1 if app["kind"] == "reminders" else max(0, round(rng.gauss(rate, rate / 3)))
        for k in range(count):
            hour = 9 + (0 if app["kind"] == "reminders" else rng.random() * 9)
            due = start + int(hour * 3_600_000_000) + rng.randint(0, 59_000_000)
            subject = rng.choice(subjects) if subjects else ""
            if not open_at(world, app["channel_id"], due):
                continue
            post = text(world, app, KINDS[app["kind"]]["noise"], rng, due, subject=subject)
            if queue(
                world,
                due,
                app["channel_id"],
                app["user_id"],
                post,
                f"noise:{app['user_id']}:{day}:{k}",
                "bot_message",
            ):
                queued += 1
            if app["kind"] == "alerts":
                later = due + rng.randint(5, 55) * 60_000_000
                if later < end:
                    queue(
                        world,
                        later,
                        app["channel_id"],
                        app["user_id"],
                        KINDS["alerts"]["resolved"].format(subject=subject),
                        f"noise:{app['user_id']}:{day}:{k}:resolved",
                        "bot_message",
                    )
    return queued


def queue_events(world, settings, after_us: int) -> int:
    """Each ledger event an app reports, posted at its moment by that app: the events after `after_us` (the present,
    or a chapter's first day), whose moments are still to come."""
    queued = 0
    for event in world.db.execute(
        "SELECT * FROM events WHERE post_by IS NOT NULL AND moment_us >= ? ORDER BY moment_us", (after_us,)
    ).fetchall():
        app = next(a for a in apps(world) if a["user_id"] == event["post_by"])
        rng = random.Random(digest([settings.seed, "event", event["id"]]))
        if not open_at(world, app["channel_id"], event["moment_us"]):
            continue
        post = text(world, app, KINDS[app["kind"]]["event"], rng, event["moment_us"], title=event["title"])
        queued += queue(
            world,
            event["moment_us"],
            app["channel_id"],
            app["user_id"],
            post,
            f"event:{event['id']}",
            "bot_message",
        )
    return queued


def event_errors(world, events, anchors: list[str]) -> list[str]:
    """An event an app reports names an app of the workspace, and its title states no fact (no fact's anchor, no
    time): a bot's post never tells an answer first."""
    names = {a["name"] for a in apps(world)}
    errors = []
    for e in events:
        if e.post_by is None:
            continue
        if e.post_by not in names:
            errors.append(f"event {e.id}: post_by names an app of the workspace, one of {sorted(names)}")
        found = [a for a in anchors if re.search(rf"(?<!\w){re.escape(a)}(?!\w)", e.title, re.IGNORECASE)]
        if TIME_LITERAL.search(e.title) or found:
            errors.append(
                f"event {e.id}: an app posts its title, which states no fact or time: {found or 'a time'}"
            )
    return errors


def app_id(world, name: str) -> str | None:
    """A bot's user id, by its app's name."""
    row = world.db.execute(
        "SELECT i.user_id FROM integrations i JOIN users u ON u.id = i.user_id WHERE u.real_name = ?", (name,)
    ).fetchone()
    return row[0] if row else None
