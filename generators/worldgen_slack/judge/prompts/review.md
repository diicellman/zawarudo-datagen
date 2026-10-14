Review what /task/input.json names, as /task/review.md says. /task/world.sqlite is the workspace
with its facts and tasks; /task/schema.sql is its schema. inspect_check runs code's checks and re-runs the reviewed
tasks' gold queries; inspect_read reads Slack as a task's actor; inspect_sql runs SQL as a task's actor. Other JSON
files in /task are agents' traces: solver_<task_id>_<n>.json are independent solves of that task, and
witness_<task_id>_<n>.json a stronger solver's tries of a task the solves rarely answer. Write the verdict to
/task/verdict.json following /task/schemas.json.
