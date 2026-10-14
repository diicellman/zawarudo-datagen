Approve only after investigating. Call inspect_check before writing the verdict.
- tasks: one review per task in input.json; when a task is valid depends on the review, below.
- issues: each defect, with its artifact, the ids it concerns, the defect and the requested_change. artifact is ledger
  for facts, storylines, people's titles and channels; tasks for a task's question, actor, answer type or gold query;
  workspace for messages: message_ids are the messages to change, evidence_message_ids the messages that show the
  defect and stay as they are.
- blocking is true when a defect changes, contradicts, hides or hands over a task's answer, or breaks the ledger;
  otherwise it is false.
- approved is true only when every task is valid and no issue blocks.
- When input.json has previous_issues, check that each is resolved.
