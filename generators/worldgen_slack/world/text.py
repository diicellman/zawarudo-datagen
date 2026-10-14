"""The text rules a world's lines keep: the patterns code reads in a message, and normalized text."""

import re


def normalized(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.casefold()))


PLACEHOLDER = re.compile(r"\{at:([A-Za-z0-9][A-Za-z0-9_.-]*)\}")


# A person's own message, as SQL over messages m joined to users u: not a bot's post, not a join or a leave.
PEOPLE = "u.is_bot = 0 AND m.subtype IS NULL"


# Links and groups are code's too: {link:TKT-4471} renders as Slack's <url|TKT-4471> for an artifact of the registry,
# {group:oncall} as <!subteam^S..|@oncall>; a raw URL in authored text is refused.
LINK = re.compile(r"\{link:([A-Za-z0-9][A-Za-z0-9#._/-]*)\}")


GROUP = re.compile(r"\{group:([a-z0-9][a-z0-9_-]*)\}")


RAW_URL = re.compile(r"https?://|www\.", re.IGNORECASE)


MENTION = re.compile(r"<@([A-Za-z0-9]+)>")


# ponytail: clock times and calendar dates only; relative words ("tomorrow", "Friday") are left to the judge,
# and "may 2" (the verb) is read as a date.
TIME_LITERAL = re.compile(
    r"\b(?:\d{1,2}:\d{2}|\d{1,2}\s?[ap]\.?m\b\.?|\d{4}-\d{2}-\d{2}|"
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sept?(?:ember)?|"
    r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?\s\d{1,2}(?:st|nd|rd|th)?\b)",
    re.IGNORECASE,
)


# A word that names a day: before {at:id}, which names its own day, it would name it twice.
DAY_WORD = re.compile(
    r"\b(?:today|tonight|tomorrow|tmrw|yesterday|(?:mon|tue|tues|wed|thu|thur|thurs|fri|sat|sun)(?:day)?|"
    r"wednesday|saturday)\.?,?\s*(?:at\s+|@\s*)?$",
    re.IGNORECASE,
)


EMOJI = re.compile(r":[a-z0-9_+\-]+:|[☀-➿\U0001F300-\U0001FAFF]")
