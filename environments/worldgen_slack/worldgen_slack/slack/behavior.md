# Slack read-only interface v1

- Public, non-archived channels are visible to every workspace user.
- Private channels, direct messages, and group direct messages require actor membership.
- Deleted messages and archived conversations never appear in reads or search.
- Conversation history returns root messages newest first.
- Threads return the root followed by replies oldest first.
- Search ranks exact phrase match, token overlap, timestamp, then message ID.
- Search returns at most 10 records. History and threads return at most 50 records.
- User lookup reads the workspace directory and never changes state.
- Every timestamp is strict UTC in `YYYY-MM-DDTHH:MM:SSZ` form.
