Phase organization: give the company's people their jobs, and choose its conversations.
- people: every candidate in `candidates` once, by user_id. A person keeps the candidate's name and timezone. title and
  team are the job they hold here; a few set a Slack status (status_emoji like ":palm_tree:", status_text like "out
  until Monday"; no times).
- channels: the conversations the company works in. A public channel is readable by every person, a private channel
  only by its members; an im is a direct message between exactly 2 people, an mpim a group DM of 3 or more. Public and
  private channels have a name of lowercase letters, digits, - and _, and a topic and purpose in `language`; DMs have
  no name. members are user_ids of chosen people. Code adds more direct messages between people who work together.
  Some channels are for work that ends or moves on (a launch room, a project's channel), which the ledger can archive;
  some workstreams have a code name people use in some places and plain words in others.
- routines: for each public and private channel, 3 to 5 recurring kinds of conversation people have there, each with
  the probability that one of the channel's everyday conversations is of that kind. dm_routines: the same for direct
  messages. Code draws each everyday conversation's kind from them; the storylines are written apart.
- integrations: the apps the company runs in Slack (when `apps` is true, 2 to 5 of them; otherwise none). Each has a
  name (its bot's), a kind (deploys, alerts, tickets, ci or reminders), the channel it posts in, 3 to 12 subjects it
  posts about (services, checks, ticket subjects, reminders; no times), and for tickets the prefix of their numbers.
  Code writes every bot post: routine ones on workdays, and an event the ledger says an app reports.
