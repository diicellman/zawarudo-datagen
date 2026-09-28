# Slack QA taskset

`worldgen-slack` loads related questions against one immutable Slack snapshot. Each task
starts a fresh solver session with actor-scoped tools. A native `vf.Judge` call grades
correctness and grounding in the solver's own observations. It does not launch a judge agent.

```bash
uv run eval @ configs/worldgen_slack/eval.toml -n 1 -r 2 --no-push --plain
```

Set `--env.taskset.task.release_dir` to select another generated release. Public rows and
snapshot references are validated before execution; private answers stay host-side.
The full-eval convention is 100 tasks, one rollout each.

## Supported Slack subset: `slack.readonly.v2`

- `list_conversations(cursor?, limit?)`
- `search_messages(query, conversation_id?, author_id?, after?, before?, cursor?, limit?)`
- `get_conversation_history(conversation_id, cursor?, limit?)`
- `get_thread(conversation_id, root_message_id, cursor?, limit?)`
- `get_user(user_id)`

Pages contain `items` and `next_cursor`; directory lookup returns one record. Page size is
1–100 (default 50). History returns roots newest first; threads return root and replies
oldest first. Search uses deterministic phrase/token overlap ranking and exclusive UTC
time bounds. It is a Slack-like lexical subset, not Slack's production search parser.
Cursors belong to one actor, snapshot, and query.

Public channels are readable by all workspace users; private channels and DMs require
membership. Archived conversations and deleted messages are excluded. The interface does
not implement writes, attachments, Slack OAuth, or production MCP compatibility.

Native subprocess tool servers hold the snapshot on the host. A small file reference
avoids serializing thousands of messages into the tool-server environment variable.
Recorded reads are serialized per server because the pinned Verifiers state endpoint
replaces the whole state on each update. These constraints are covered by the runnable
check in the generator.
