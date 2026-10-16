You judge one task written for training an assistant that reads a company's Slack for a person at work. The task is
a request its asker makes of that assistant, in 1 to 4 parts, each with the answer its writer gave and the messages
that answer rests on. A verifier, another model with the same Slack tools, has already reached that answer from
Slack; its answers are below. A student model tried it too, and its wrong answers are below. Rule on four things,
each true or false, and give one short reason that names what decided any false.

- realistic. Ask what its asker would do with the answer. A person at work asks a colleague who reads Slack for them
  for what their own work needs and they cannot keep in their head: what was decided and why, what changed and when,
  where something stands, who has it, what was promised and whether it was done, a figure they report. That is a
  request, and its family names which of four it is (catch-up, decisions & why, commitments, who-knows & status). A
  question that treats colleagues' activity as data, who did or did not post, react or reply and when, gives the asker
  nothing for their work, however hard it is: only an investigator, an analyst or a sentiment dashboard asks it, and
  it exercises the tools, not the asker's need. Such a set or join belongs only as a part of a request that needs it,
  as a manager's report needs who still has to sign off. The asker must be someone whose work needs the answer.
- one_reading. For each wrong answer the student gave, decide whether the request's own words allow a reading under
  which that answer is right: does a thread reply count as posting, whose clock and which days, what comes before
  what, a team or a channel, the plan as of when. A mistake the words rule out is the student's; a reading the words
  allow is the request's fault, and one_reading is false.
- exact. Each part's answer answers exactly what its label asks, nothing more specific: a part that asks whether a
  writeup was posted is answered by whether and the day, not a clock time; why something was pulled is answered by the
  reason, not a measurement the request never asks for. An answer that demands more than the request asks turns right
  answers wrong, and exact is false.
- supported. The messages a part rests on state its answer, as its asker can read them; for a status, the visible
  value is not settled, as the unseen message shows; for a refusal, what is asked is not there to see, or its premise
  is contradicted.

The task, the messages, the verifier's and the student's answers:
{task}
