# Slack QA taskset

`worldgen-slack` loads related questions against one released Slack world: a SQLite file without the answer key.
Each task starts a fresh solver session with actor-scoped Slack tools. A native `vf.Judge` call grades correctness
and grounding in the solver's own observations; it does not launch a judge agent.

```bash
uv run eval @ configs/worldgen_slack/eval.toml -n 1 -r 2 --no-push --plain
```

Set `--env.taskset.task.release_dir` to select another generated release. Public rows and the world's hash are
validated before execution; private answers stay host-side.

## The solver's tools

`search_messages`, `search_users`, `search_channels`, `list_user_channels`, `read_channel`, `read_thread`,
`get_user`, `list_channel_members` and `get_reactions`, as the task's actor. Pages hold `items` and a cursor bound to
one actor, world and query; a page holds 1-100 items. Search ranks by BM25.

Public channels are readable by everyone in the workspace; private channels and DMs need membership, which a
member who left no longer has. Deleted messages are excluded. There are no writes.

The tool server runs on the host and opens the world file by path, after checking its hash. Each call is recorded
for grading.
