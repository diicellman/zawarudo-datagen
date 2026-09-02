# Slack world builder guide

Implement only `/task/workspace/world.py`:

```python
from worldgen_slack.slack.models import SlackWorld, TaskContract


def build(seed: int, contract: TaskContract) -> SlackWorld:
    raise NotImplementedError
```

The contract is frozen. Materialize every symbolic ID it names. Public, non-archived channels are
visible to all users. Private channels and direct messages require actor membership. Reads exclude
deleted messages and archived conversations. History returns roots newest first. Threads return the
root and replies oldest first. Search ranks exact phrases, token overlap, timestamp, and message ID.

The actor must recover all required evidence through the read-only actions. Preserve answer-bearing
evidence for every seed. Keep each evidence message focused on its assigned fact; do not add a single
message that restates the full conclusion or repeats every answer claim. Add a small amount of natural,
non-answer-bearing workplace context so the world is compact but not a staged minimal fixture.

The seed must not be a no-op. Use it to vary at least one realistic, nonessential message, reaction, or
other irrelevant field so canonical snapshots differ across seeds while required evidence stays stable.
Authors must be conversation members. Replies must follow their roots chronologically. Stay within 40
users, 15 conversations, 150 messages, 100 reactions, and a 512,000-byte canonical snapshot.

Do not serialize private contract fields, copy the literal question into Slack, create answer/oracle
channels, add prompt instructions, access the network, install packages, or alter the fixed models.
Run `/task/check-world` after edits. Hidden seeds run later in a separate Prime VM.
