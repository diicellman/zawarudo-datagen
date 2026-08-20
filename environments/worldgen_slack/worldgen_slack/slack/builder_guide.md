# Slack world builder guide

Implement only `/task/workspace/world.py`.

```python
from worldgen_slack.slack.models import SlackWorld, TaskContract


def build(seed: int, contract: TaskContract) -> SlackWorld:
    raise NotImplementedError
```

The contract is frozen. Materialize every symbolic user, conversation, message, and thread ID it names. The actor must be able to discover all required evidence through the fixed read-only actions. Preserve answer-bearing evidence for every seed. Vary only nonessential users, messages, ordering, or distractors.

Use 5–40 users, 3–15 conversations, 15–150 messages, and no more than 100 reactions. Keep each canonical snapshot below 512,000 bytes. Authors must be conversation members. Replies must follow their root chronologically. Do not serialize the private contract, copy the literal question into Slack, create an answer/oracle channel, add prompt instructions, access the network, install packages, or alter fixed models and validation.

Run `/task/check-world` after edits. Its report is public guidance. The host later executes the source in a separate fresh Prime VM with hidden seeds and stricter validation.
