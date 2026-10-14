"""A task's tries: how a wrong try went wrong."""

from worldgen_slack.judge import EMPTY


def miss_class(result: dict, answer_type: str) -> str:
    """How a wrong try went wrong, by code: `budget` (no answer: empty, or out of turns), `abstained` (declined),
    `overclaim` (a status answer's value given as final), `trap` (a refusal answered with a value), `search` (some of
    the evidence its answer rests on never shown) or `saw_all` (shown all of it, and answered otherwise). Whether a
    wrong answer is a reasoning slip or a reading the question allows is the judge's ruling, on every wrong answer."""
    if result.get("unanswered") or result.get("reason") == EMPTY:
        return "budget"
    if result.get("abstained"):
        return "abstained"
    aligned = "missing" in result and not result["missing"] and not result.get("extra")
    if answer_type == "status" and aligned:
        return "overclaim"
    if answer_type == "refusal" and result.get("committed"):
        return "trap"
    coverage = result.get("evidence_coverage")
    return "search" if coverage is not None and coverage < 1 else "saw_all"
