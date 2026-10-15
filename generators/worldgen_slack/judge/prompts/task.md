Task review. A task is valid when its actor can answer its question as asked from what they can read,
and its gold rows are the complete, correct answer; a status task's answer is its gold value given as not settled, as
the actor can see it, while the fact that settles it is out of their sight. Score each criterion of schemas.json from 0 to 1:
- question_fit: each gold query answers its question as asked; test variants with inspect_sql.
- discoverability: the actor can find the answer with Slack's read tools; try inspect_read, and read the solves.
  When input.json has witness, a stronger solver tried the tasks the solves rarely answer: one it answers can be
  solved, which shows nothing about whether its question has one reading; one it fails too may be unanswerable.
- One reading. Each wrong solve's grade gives the answer it committed to and the gold rows it missed or added. For
  each, decide whether the question's own words allow a reading under which that answer is right: does a thread
  reply count as posting, is the poster among those who replied, does a reaction break a quiet spell, what comes
  before what in "then", whose clock and which week, a team or a channel. A mistake the question's words rule out is
  the solver's. If a reading allows it, the task is not valid: set its review's misread to that answer and the
  reading, and raise a blocking issue asking for the wording that rules the reading out with the same gold.
- shortcut_free: nothing in the workspace hands over the answer outside its evidence.
- A composite task is one request in parts: input.json gives its parts, each with its label, its gold query and
  rows, and its facts. Each part's gold answers its label as the request asks it, and the task is right only when
  every part is; review each part as a task, and the request as a whole for one reading and realism.
- Realism, on each task's review. Ask what its asker would do with the answer. A person at work asks a colleague
  who reads Slack for them for what their own work needs and they cannot keep in their head: what was decided and
  why, what changed and when, where something stands, who has it, what was promised and whether it was done, a figure
  they report. That is a request, and its family names which of four it is (catch-up, decisions & why, commitments,
  who-knows & status). A question that treats colleagues' activity as data, who did or did not post, react or reply
  and when, gives the asker nothing for their work, however hard it is: only an investigator, an analyst or a
  sentiment dashboard asks it, and it exercises the tools, not the asker's need. Such a set or join belongs only as a
  part of a request that needs it, as a manager's report needs who still has to sign off. When no colleague would ask
  it, set its review's unrealistic to why: it is not valid.
- level_fit, on each task's review, from 0 to 4: how fully answering it needs the level and concept its `means`
  names. 4: it needs them, with no shortcut; 3: it needs them, but one route is easier than the level; 2: partly;
  1: in name only; 0: one obvious step answers it. inspect_check measures each task: how many read_channel pages deep
  its evidence sits for its actor, the tables its gold query reads, and its evidence's search rank for the question's
  own words; the solves show the calls it took. level_fit is reported beside the task; it decides neither valid nor
  blocking.
