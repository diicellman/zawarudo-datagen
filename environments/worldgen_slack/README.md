# Slack QA taskset

`worldgen-slack` loads related questions against one released Slack world: a SQLite file without the answer key.
Each task starts a fresh solver session with actor-scoped Slack tools. A native `vf.Judge` call grades the answer the
response commits to (one that lists alternatives without settling on one is wrong), whether it declined to answer,
and whether the solver's own observations support the claims that make its answer; other unsupported claims are
counted (`unsupported_asides`) but cost nothing. The reward is +1 for a right, grounded answer, 0 for a right one the
observations don't ground or for declining to answer, and -1 for a wrong one: under a 1-or-0 grade a guess always
beats saying it cannot be told. A `status` task's right answer is the latest value its asker can see, given as not
settled; a `refusal` is right when nothing is there to find. It does not launch a judge agent.

```bash
uv run eval @ configs/worldgen_slack/eval.toml -n 1 -r 2 --no-push --plain
```

Set `--env.taskset.task.release_dir` to the generated release to evaluate (it is required). Public rows and the
world's hash are validated before execution; private answers stay host-side.

A release (format `worldgen-slack.v12`; v11 still loads) carries, with each public task, how hard it measured when it was generated:
the solver model and its tries (`tries`; `crashed`, and `unanswered` for those that ended without an answer), its
share of right answers (`right_rate`, the difficulty, with its 90% interval `right_interval`), of right and grounded
ones (`strict_rate`, the reward's +1) and of answers it declined (`abstain_rate`), the
share of the gold evidence its tries saw (`coverage`), and, for a task the solver rarely answered, a stronger
witness's share of right answers (`witness`, `witness_right`). A curriculum can filter on them. Each task names its
`family`, the request it is: catch-up, decisions & why, commitments, or who-knows & status. A `composite` task is one
request in 2-4 parts, each with its own gold rows (its private answer's `parts`, each under its `label`); it is graded
part by part and settled by code, all or nothing: +1 only when every part is right and grounded, -1 when any part is
wrong, 0 when it declined what it could not tell. Each private answer carries its `solution`: a try that reached
its gold from Slack alone, right and grounded (the solver's, or else the witness's), with its answer and its calls;
a task no try reached is not released. A forged task's gold (v12) is the answer the forge's verifier reached
from Slack, with no query; its public row names the set it came from (initial, or a harder, easier or repaired
version of a task) and the solver's rate by part. Older releases are read from git.

## The solver's tools

The read tools of Slack's own MCP server, as captured in two sessions on 2026-10-14 (`capture/slack-mcp`):
`search_public`, `search_public_and_private`, `search_channels`, `search_users`, `read_channel`, `read_thread`,
`read_user_profile`, `get_reactions`, `list_channel_members` and `list_user_channels` (served as `slack_*`), with
the real server's parameters and descriptions, as the task's actor. Each returns the real server's text: names
filled in, times on the actor's clock with their offset, reactions as counts in a channel's history (`get_reactions`
names who). Search takes `keywords` (all required) and `filters` (Slack's modifiers) and ranks by BM25; a hit
carries the messages around it. Cursors are the server's and name a place, not a request: a channel's history and a
thread page back from a message (`next_ts`), a search by page number (`CURRENT_PAGE`), a list by its last item. The
system prompt says who the actor is and when it is, as a Slack client does; a profile with no `user_id` is the
actor's own. Each call keeps a record of what it showed (`slack.py`), which grading reads; the solver reads the
text. A release records the surface and turn limit each task was measured on.

Public channels are readable by everyone in the workspace; private channels and DMs need membership, which a
member who left no longer has. Deleted messages are excluded. There are no writes.

The tool server runs on the host and opens the world file by path, after checking its hash. Each call is recorded
for grading. `generators/worldgen_slack/tests/test_capture.py` replays the capture against the workspace it was taken
in, rebuilt as a world, and lists where the env reads otherwise than the server: only around a reply that was also
sent to the channel, which worlds do not have.
