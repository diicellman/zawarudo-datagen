- gold_sql is one SELECT over world.sqlite as the task's actor sees it: channels, members, messages,
  message_mentions, reactions and thread_stats hold only what the actor can read; users, calendar, storylines, facts,
  fact_relations and evidence are whole. local(us) writes a time on the actor's clock, local(us, zone) on another.
  It returns the answer in a column named answer, and may name the evidence in message_id and user_id columns.
- What the gold query reads follows the category's gold in `taxonomy`: a sql query reads only the workspace, a
  ledger query reads the facts it answers from, a hybrid query reads both.
- The gold query finds messages by what the question names (words, people, channels, threads, reactions, times),
  never by a message id or by id order: code checks that it answers the same with the messages renumbered.
- It asks only about what the solver's Slack tools show: who reacted, not when; who is a member now, not when they
  joined; nothing of a channel's creator or a person's account date. Code refuses a column no tool shows.
- answer_type: text or number is one row; set is 1 to max_answer_rows rows; refusal is no rows as the actor (nothing
  to find); status is one row, the latest value the actor can see, which a fact out of their sight settles: its gold
  reads that value from the ledger, and its facts name the settling fact. A status is answered right by giving the
  value as not settled.
- near_sql, optional: up to {near_sql} queries, the gold with one condition relaxed, each answering otherwise (the answer
  a hasty reader gives).
- question is what actor_id asks, in `language`; it does not contain its answer.
