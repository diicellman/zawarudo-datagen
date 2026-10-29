You grade one answer to a request someone made of an assistant that reads their company's Slack. Grade it part by
part against the gold. For each part the gold gives its answer type, its answer items, its context (details the
answer rests on that the request does not ask for), and for some parts what is out of the asker's sight: messages
the asker can't read, which show how things really stand.

For each part, first report what the answer commits to. Then report each criterion below: one short reason that
quotes or names what decided it, then true or false. Report; do not grade. Code decides the outcome from what you
report.

- **commits.** The words the answer commits to as its answer to this part. A flat headline or opening claim commits,
  even when the body later softens it with a caveat. An explicit correction ("twice, no, three times") replaces the
  value it corrects. A value it settles on commits, though framed as an inference or with a caveat, or derived from a
  date ("the next day after Oct 29, i.e., Friday, October 30"). A value named only in passing does not: an aside, an
  option it labels as a side option, or something it offers while saying it isn't what was asked for ("inferred, not
  from anything he said"). Every criterion below is asked of what it commits to.
- **gives**, one per answer item. It gives the item, in any equivalent wording; a time on the asker's clock counts;
  a date with no weekday gives a date. A plain or flat form of a scoped item gives it ("No, not yet" gives "nothing
  visible says he has loaded them"); whether the flat claim holds is conflicts' question.
- **conflicts.** It commits to something the truth rules out: another value for an answer item (a stale value, a
  decoy); a denial of an answer item ("nobody posted a figure", when the gold's figure was posted); a flat claim that
  what is out of the asker's sight contradicts ("it hasn't come in", when it came in by DM; "yes, still the plan",
  when it slipped), judged by what that message says, not what it is about; for a part that can't be answered, an
  answer anyway: a value it presents as the thing asked for, as if its premise held; an answer item's value credited
  to another person or message than the gold names; a weekday other than the gold's, when the request asks what,
  which or on what day. Name what conflicts. None of these conflict: a claim scoped to what is visible ("not yet, as
  of the last record", "nothing in Slack says so"); a value it corrects; a nearby fact it offers as such, not as the
  answer; other people named beside the gold's own (one who decides, one who delivers). A detail beside what was
  asked doesn't conflict here, however wrong (see false_asides).
- **extra** (a set; null for any other part). It presents as one of the set anything the gold's items don't include.
  Name each.
- **hedged.** It offers two or more values for the part without settling on one.
- **holds_back.** It says it can't find, see or confirm the answer, or it flags its answer as partial, inferred or
  unconfirmed ("these three; there may be more I couldn't confirm").
- **declines** (a part that can't be answered; null for any other part). It says plainly that what is asked isn't
  there to see, can't be answered as asked, or that it couldn't find it.
- **false_asides.** The context details it contradicts, each named; none if it contradicts none.

The request:
{request}

The gold, by part:
{reference}

The answer, between the fences:
{response}
