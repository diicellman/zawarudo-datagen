# Slack QA taskset

`worldgen-slack` loads related questions against one released Slack world: a SQLite file without the answer key.
Each task starts a fresh solver session with actor-scoped Slack tools. A native `vf.Judge` call grades the answer the
response commits to (one that lists alternatives without settling on one is wrong) and whether the solver's own
observations support the claims that make it; other unsupported claims are counted (`unsupported_asides`) but cost
nothing. It does not launch a judge agent.

```bash
uv run eval @ configs/worldgen_slack/eval.toml -n 1 -r 2 --no-push --plain
```

Set `--env.taskset.task.release_dir` to select another generated release. Public rows and the world's hash are
validated before execution; private answers stay host-side.

A release (format `worldgen-slack.v8`) carries, with each public task, how hard it measured when it was generated:
the solver model and its tries (`tries`; `crashed`, and `unanswered` for those that ended without an answer), its
share of right answers (`right_rate`, the difficulty) and of right and grounded ones (`strict_rate`, the reward), the
share of the gold evidence its tries saw (`coverage`), and, for a task the solver rarely answered, a stronger
witness's share of right answers (`witness`, `witness_right`). A curriculum can filter on them. `worldgen-slack.v7`
releases (whose strict rate needed every claim grounded) and v6 releases (without rates) still load.

## The solver's tools

`search_messages`, `search_users`, `search_channels`, `list_user_channels`, `read_channel`, `read_thread`,
`get_user`, `list_channel_members`, `get_reactions` and `whoami`, as the task's actor. `whoami` (Slack's auth.test)
says who the actor is, with their timezone and the present on their clock; every message carries its time in UTC
(`time_utc`) and on the actor's clock (`time_local`), and a DM lists all its members. Pages hold `items` and a cursor
bound to one actor, world and query; a page holds 1-100 items. Search ranks by BM25.

Public channels are readable by everyone in the workspace; private channels and DMs need membership, which a
member who left no longer has. Deleted messages are excluded. There are no writes.

The tool server runs on the host and opens the world file by path, after checking its hash. Each call is recorded
for grading.
