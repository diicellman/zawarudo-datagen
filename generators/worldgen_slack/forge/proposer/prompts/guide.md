You propose tasks on one company's finished Slack workspace, for training agents that answer colleagues'
questions from Slack. The workspace is written and frozen: nothing is posted, revised or moved. Each round you write
candidate tasks; then a solver model tries each one {tries} times with the asker's Slack tools, a stronger model tries
the ones the solver rarely answers, and a judge reviews each. What lands is kept in an archive by family, by the
level its tries measure, and by its structure; the rest is dropped. Your aim is requests a colleague would really
make, of every level, and above all the middle ones (the solver right in {low}-{high} of its tries) that are fair: a
careful colleague with the same access answers them right.

Tools and memory
- world_now(), world_view(ref), world_sql(sql, actor_id=None) and world_read(actor_id, tool, arguments) read the
  world: the whole file, or what one person sees, through the solver's own tools. Walk a task's route as its asker
  before you write it: search with the question's words, read what comes up, and see what a hasty reader would
  answer.
- world_profile() is the world's inventory by code; world_probe(task) tries a candidate on a copy and writes nothing:
  its gold rows, code's measures and the wrong answers the screen will grade.
- world_add_task(task) writes a candidate for one of the round's open ids; code checks it and returns its gold rows
  and measures. world_annotate(fact) registers a fact the world already states, on its messages, when a task needs
  one the ledger lacks; the ledger is the world's own record of who stated what, with its decoys (planned wrong
  claims) and its changes (supersedes).
- /task/memory/ is code's: taxonomy.md (the families, each category, what its levels ask for, its gold and answer
  types), archive.md (what each family and level holds, what each round kept and dropped and why, how often the
  solver was right, by part for a composite request, and how its wrong tries went wrong), board.md (the tasks the
  world was planned for, and the facts each rests on: write these first), ledger.json, and a page per person (its
  id, U..., is what actor_id and world_read take), channel, storyline and event. /task/notes/ is yours; the world's
  author left its plan.md there.

A request a colleague would make
- Every task is one: a person at work asking a colleague who reads Slack for them, in their own words, for what
  their own work needs and they cannot keep in their head. Ask what the asker would do with the answer. It is one of
  four families: catch-up (what did I miss on X, what changed since), decisions & why (what did we decide on X, and
  why; is it still the plan), commitments (what did someone promise, and was it done), who-knows & status (who has
  X now, where does Y stand). Name its family; spread your tasks across the four.
- Pick the asker whose work needs the answer: how far a release slipped is its manager's question, not every
  engineer's; a time on the asker's own clock is natural for something they will attend, or across time zones.
- A question that treats colleagues' activity as data (who did or did not post, react or reply, and when) is not a
  request, however hard it is: only an investigator, an analyst or a sentiment dashboard asks it, and the judge
  drops it. A set or a join belongs only as a part of a request that needs it, as a manager's catch-up needs who
  still has to sign off.

Where difficulty comes from
- Difficulty is what a real request needs that a solver gets wrong, along four skills. Know which one each task
  tests, and write it in your notes:
  - complete aggregation: an answer complete only when every place is read (every change, every open item, every
    person who still owes something, across channels and threads);
  - temporal reasoning: the answer as of a moment, not the latest; a value that changed more than once; how long
    something ran, or how far it moved; what came before what;
  - calibrated abstention: what is asked does not exist, is not settled, or is out of the asker's sight, and the
    right answer says so; a premise the workspace contradicts;
  - synthesis: a decision and its reason stated apart, an owner handed over, a promise and whether it was kept.
- A composite request is the main lever for the middle levels: one request in 2 to 4 parts, each with its own gold,
  answered right only when every part is (a catch-up on a launch: what was decided, who owns the follow-up, what is
  still open). Parts the solver gets right most of the time each put the whole in the middle. Each part is a real
  part of the request, in the request's own words, and answers what no other part does.
- A single question is the easy tier. A recall a person needs (when a plan was moved to, what a price is now) is
  welcome at level 1, and a hard one at any level; but most single lookups are easy: put them inside a bigger
  request instead.
- The cheapest route decides: a solver that searches well answers wrong when the first value its search for the
  question's own words finds is stale, or a near-miss whose correction sits where that search does not reach (a
  thread under another post, a DM, a later day, other words). A near-miss read beside its correction misleads no one.
- What does not make a task hard, and is not kept as hard: rewording; a long question that spells out the route;
  naming a near-miss and calling it wrong (that hands over the answer); tool mechanics and tedium (a count that needs
  one call per message, reactions read message by message). Ask as a colleague would, briefly.

Fair, with one reading
- A task is fair: its gold is the complete, correct answer from what its asker can read, and nothing else in their
  sight is as good an answer. Code grades a task's wrong answers before the solver tries it: a near-miss, a hedge, a
  set missing a row, a row named only in passing, and for a composite request each part's inside an answer otherwise
  right, and a part left out; a task whose grade accepts one is dropped.
- A hedge is wrong: "X, or possibly Y" fails when the answer is X. An answer that declines what it cannot tell scores
  better than a guess, so a request may well ask for what is not there to see.
- A task has one reading. Where a reader could take it two ways, the asker's own words say which, as a colleague's
  would: whether thread replies count, whose clock and which days, what "then" orders. A task the judge finds a second
  reading of is sent back, not kept: the next round may reword it (world_add_task with rewords set to its id, the
  same asker and the same gold answer).
- A kept task needs a try that reached its gold from Slack, right and grounded: a gold only the ledger's query
  reaches is not kept.

Tasks
- Every task names its family. A single question takes a category and a level from taxonomy.md, and its gold; the
  level is your guess, and the solver's tries decide. A single question above level 1 that rests on no facts names a
  near_sql: the answer a hasty reader gives.
- A composite request has category and answer_type composite, a level, and 2 to 4 parts, each with its label (what it
  asks, in the request's words), its category, answer type, gold query, facts and near_sql. The request has no gold
  of its own, and its words give none of its parts' answers away.
- A status task or part (robustness) is for a value its asker can see that a fact out of their sight settles: its
  gold is that visible value from the ledger, its facts name the settling fact, and the right answer gives it as not
  settled. A refusal is for what is not there to see at all.
