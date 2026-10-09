-- One Slack world. Times are integer microseconds since the epoch (UTC); code assigns every one of them.
-- Load order: users, channels, members, messages, then the rest (triggers read members and messages).

CREATE TABLE world_meta (
  key   TEXT PRIMARY KEY,                            -- schema_version, workspace_id, company, now_us, ...
  value TEXT NOT NULL
);

-- ===================================================================== solver-visible, through the tools
CREATE TABLE users (
  id           TEXT PRIMARY KEY,                     -- 'U' + base36
  handle       TEXT NOT NULL UNIQUE,
  real_name    TEXT NOT NULL,
  display_name TEXT NOT NULL,                        -- not unique on purpose
  email        TEXT NOT NULL UNIQUE,
  title        TEXT,
  tz           TEXT NOT NULL,                        -- IANA zone
  status_text  TEXT,
  status_emoji TEXT,
  is_bot       INTEGER NOT NULL DEFAULT 0 CHECK (is_bot IN (0, 1)),
  is_admin     INTEGER NOT NULL DEFAULT 0 CHECK (is_admin IN (0, 1)),
  is_deleted   INTEGER NOT NULL DEFAULT 0 CHECK (is_deleted IN (0, 1)),
  created_us   INTEGER NOT NULL,
  profile_json TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(profile_json))  -- custom profile fields
);

CREATE TABLE channels (
  id          TEXT PRIMARY KEY,                      -- C.. public, G.. private and mpim, D.. im
  name        TEXT,                                  -- NULL for im and mpim
  type        TEXT NOT NULL CHECK (type IN ('public', 'private', 'im', 'mpim')),
  topic       TEXT NOT NULL DEFAULT '',
  purpose     TEXT NOT NULL DEFAULT '',
  creator_id  TEXT REFERENCES users(id),
  is_archived INTEGER NOT NULL DEFAULT 0 CHECK (is_archived IN (0, 1)),
  created_us  INTEGER NOT NULL,
  CHECK ((type IN ('im', 'mpim')) = (name IS NULL))
);
CREATE UNIQUE INDEX channels_name ON channels(name) WHERE name IS NOT NULL;

CREATE TABLE members (
  channel_id TEXT NOT NULL REFERENCES channels(id),
  user_id    TEXT NOT NULL REFERENCES users(id),
  joined_us  INTEGER NOT NULL,
  left_us    INTEGER,                                -- NULL while still a member
  PRIMARY KEY (channel_id, user_id, joined_us),
  CHECK (left_us IS NULL OR left_us > joined_us)
);
CREATE INDEX members_user ON members(user_id);

CREATE TABLE messages (
  id         INTEGER PRIMARY KEY,                    -- internal; tools address a message by (channel, ts)
  channel_id TEXT NOT NULL REFERENCES channels(id),
  ts_us      INTEGER NOT NULL,
  ts         TEXT GENERATED ALWAYS AS (printf('%d.%06d', ts_us / 1000000, ts_us % 1000000)) VIRTUAL,
  user_id    TEXT NOT NULL REFERENCES users(id),
  parent_id  INTEGER REFERENCES messages(id),        -- NULL = top level; else the thread root
  text       TEXT NOT NULL,
  is_deleted INTEGER NOT NULL DEFAULT 0 CHECK (is_deleted IN (0, 1)),
  edited_us  INTEGER,
  UNIQUE (channel_id, ts_us)
);
CREATE INDEX messages_parent ON messages(parent_id);
CREATE INDEX messages_user ON messages(user_id);

CREATE TABLE message_mentions (                      -- derived by code from <@U..> in the text
  message_id INTEGER NOT NULL REFERENCES messages(id),
  user_id    TEXT NOT NULL REFERENCES users(id),
  PRIMARY KEY (message_id, user_id)
);

CREATE TABLE reactions (
  message_id INTEGER NOT NULL REFERENCES messages(id),
  user_id    TEXT NOT NULL REFERENCES users(id),
  emoji      TEXT NOT NULL,
  created_us INTEGER NOT NULL,
  PRIMARY KEY (message_id, user_id, emoji)
);

-- Keyword search. FTS4 because the uv Python's SQLite has no FTS5; bm25() is registered by db.py.
CREATE VIRTUAL TABLE messages_fts USING fts4(content="messages", text, tokenize=porter);
CREATE TRIGGER messages_fts_bd BEFORE DELETE ON messages BEGIN
  DELETE FROM messages_fts WHERE docid = OLD.id;
END;
CREATE TRIGGER messages_fts_bu BEFORE UPDATE OF text ON messages BEGIN
  DELETE FROM messages_fts WHERE docid = OLD.id;
END;
CREATE TRIGGER messages_fts_au AFTER UPDATE OF text ON messages BEGIN
  INSERT INTO messages_fts(docid, text) VALUES (NEW.id, NEW.text);
END;
CREATE TRIGGER messages_fts_ai AFTER INSERT ON messages BEGIN
  INSERT INTO messages_fts(docid, text) VALUES (NEW.id, NEW.text);
END;

CREATE VIEW thread_stats AS
SELECT p.id AS root_id, COUNT(r.id) AS reply_count, MAX(r.ts_us) AS latest_reply_us
FROM messages p JOIN messages r ON r.parent_id = p.id AND r.is_deleted = 0
GROUP BY p.id;

CREATE TRIGGER messages_author_member BEFORE INSERT ON messages BEGIN
  SELECT RAISE(ABORT, 'the author was not a member of the channel when posting')
  WHERE NOT EXISTS (
    SELECT 1 FROM members m
    WHERE m.channel_id = NEW.channel_id AND m.user_id = NEW.user_id
      AND m.joined_us <= NEW.ts_us AND (m.left_us IS NULL OR m.left_us > NEW.ts_us));
END;

CREATE TRIGGER messages_thread_valid BEFORE INSERT ON messages WHEN NEW.parent_id IS NOT NULL BEGIN
  SELECT RAISE(ABORT, 'a reply must point at an earlier top-level message of the same channel')
  WHERE NOT EXISTS (
    SELECT 1 FROM messages p
    WHERE p.id = NEW.parent_id AND p.parent_id IS NULL
      AND p.channel_id = NEW.channel_id AND p.ts_us < NEW.ts_us);
END;

CREATE TRIGGER reactions_valid BEFORE INSERT ON reactions BEGIN
  SELECT RAISE(ABORT, 'a reaction must come after its message, from a member of its channel')
  WHERE NOT EXISTS (
    SELECT 1
    FROM messages m JOIN members mb ON mb.channel_id = m.channel_id AND mb.user_id = NEW.user_id
    WHERE m.id = NEW.message_id AND NEW.created_us > m.ts_us
      AND mb.joined_us <= NEW.created_us AND (mb.left_us IS NULL OR mb.left_us > NEW.created_us));
END;

-- ===================================================================== answer key: dropped from the solver copy
CREATE TABLE calendar (                              -- the company clock's days, written by code
  day      INTEGER PRIMARY KEY CHECK (day >= 1),
  date     TEXT NOT NULL UNIQUE,                     -- YYYY-MM-DD on the company clock
  start_us INTEGER NOT NULL,
  end_us   INTEGER NOT NULL,
  CHECK (end_us > start_us)
);

CREATE TABLE storylines (
  id       TEXT PRIMARY KEY,
  summary  TEXT NOT NULL,
  position INTEGER NOT NULL UNIQUE                   -- build order
);

CREATE TABLE scenes (
  id            TEXT PRIMARY KEY,
  channel_id    TEXT NOT NULL REFERENCES channels(id),
  storyline     TEXT REFERENCES storylines(id),   -- NULL: a background conversation of no storyline
  day           INTEGER NOT NULL REFERENCES calendar(day),
  part          TEXT NOT NULL CHECK (part IN ('early', 'morning', 'afternoon', 'evening', 'night')),
  slot_start_us INTEGER NOT NULL,                    -- code places the scene; its messages stay inside
  slot_end_us   INTEGER NOT NULL,
  situation     TEXT NOT NULL,
  plan_json     TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(plan_json)),  -- participants, beats, notes
  key           TEXT NOT NULL DEFAULT '',            -- digest of what the scene was written from
  CHECK (slot_end_us > slot_start_us)
);

CREATE TABLE scene_messages (
  message_id INTEGER PRIMARY KEY REFERENCES messages(id),
  scene_id   TEXT NOT NULL REFERENCES scenes(id)
);

CREATE TABLE events (                                -- a story moment: one time that every fact about it shares
  id        TEXT PRIMARY KEY,
  storyline TEXT REFERENCES storylines(id),
  title     TEXT NOT NULL,
  moment_us INTEGER NOT NULL,
  zone      TEXT NOT NULL                            -- the clock the moment is told on
);

CREATE TABLE facts (
  id          TEXT PRIMARY KEY,
  storyline   TEXT NOT NULL REFERENCES storylines(id),
  subject     TEXT NOT NULL,                         -- 'Release 8.14'
  attribute   TEXT NOT NULL,                         -- 'launch time'
  value       TEXT NOT NULL,                         -- canonical value
  anchor      TEXT,                                  -- words every message stating it must contain
  channel_id  TEXT NOT NULL REFERENCES channels(id), -- where it is first stated
  author_id   TEXT NOT NULL REFERENCES users(id),    -- who first states it
  day         INTEGER NOT NULL REFERENCES calendar(day),  -- the day it is first stated
  moment_us   INTEGER,                               -- the story moment the fact is about, if any
  moment_zone TEXT,                                  -- the clock the moment is told on
  moment_kind TEXT CHECK (moment_kind IN ('happened', 'scheduled')),
  event_id    TEXT REFERENCES events(id),            -- the event whose moment it carries, if any
  is_decoy    INTEGER NOT NULL DEFAULT 0 CHECK (is_decoy IN (0, 1)),
  summary     TEXT NOT NULL,
  CHECK ((moment_us IS NULL) = (moment_kind IS NULL) AND (moment_us IS NULL) = (moment_zone IS NULL)),
  CHECK (event_id IS NULL OR moment_us IS NOT NULL)
);

CREATE TABLE fact_relations (                        -- src <kind> dst: src comes after / supersedes dst
  src_fact TEXT NOT NULL REFERENCES facts(id),
  dst_fact TEXT NOT NULL REFERENCES facts(id),
  kind     TEXT NOT NULL CHECK (kind IN ('after', 'supersedes')),
  PRIMARY KEY (src_fact, dst_fact, kind),
  CHECK (src_fact <> dst_fact)
);

CREATE TABLE evidence (
  fact_id      TEXT NOT NULL REFERENCES facts(id),
  message_id   INTEGER NOT NULL REFERENCES messages(id),
  role         TEXT NOT NULL CHECK (role IN ('anchor', 'supporting')),
  anchor_token TEXT,                                 -- a surface form the text must contain, if any
  PRIMARY KEY (fact_id, message_id)
);

CREATE TABLE tasks (
  id           TEXT PRIMARY KEY,
  category     TEXT NOT NULL,                        -- a category of the run's taxonomy, which is data
  level        INTEGER NOT NULL CHECK (level >= 1),
  concept      TEXT NOT NULL DEFAULT '',             -- what a task of its level requires, drawn by the seed
  actor_id     TEXT NOT NULL REFERENCES users(id),
  question     TEXT NOT NULL,
  answer_type  TEXT NOT NULL CHECK (answer_type IN ('text', 'set', 'number', 'refusal', 'status')),
  gold_source  TEXT NOT NULL CHECK (gold_source IN ('sql', 'ledger', 'hybrid')),
  gold_sql     TEXT NOT NULL,                        -- run as the actor; returns answer (+ message_id, user_id)
  gold_json    TEXT NOT NULL DEFAULT '[]' CHECK (json_valid(gold_json)),  -- its rows
  min_calls    INTEGER,                              -- the fewest tool calls of a right probe
  right_rate   REAL,                                 -- the probes' share of right answers: the task's difficulty
  strict_rate  REAL                                  -- the probes' share right and grounded: the released reward
);

CREATE TABLE task_facts (
  task_id TEXT NOT NULL REFERENCES tasks(id),
  fact_id TEXT NOT NULL REFERENCES facts(id),
  PRIMARY KEY (task_id, fact_id)
);

CREATE TABLE board (                                 -- the plan's facts for each ledger and hybrid task slot
  slot    TEXT NOT NULL,
  fact_id TEXT NOT NULL REFERENCES facts(id),
  PRIMARY KEY (slot, fact_id)
);

CREATE TABLE commitments (                           -- what someone promised in a message, and how it ended
  id         TEXT PRIMARY KEY,
  owner_id   TEXT NOT NULL REFERENCES users(id),
  text       TEXT NOT NULL,
  message_id INTEGER NOT NULL REFERENCES messages(id),
  due_us     INTEGER NOT NULL,
  status     TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'kept', 'changed', 'dropped')),
  closed_by  INTEGER REFERENCES messages(id),        -- the message that kept, changed or dropped it
  CHECK ((status = 'open') = (closed_by IS NULL))
);
