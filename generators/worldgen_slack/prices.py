"""What a model's calls cost when its provider leaves them unpriced: $ per million fresh input, cached input and output
tokens, from `prime inference models` (2026-10-20), except where the bill says otherwise. A model not listed is priced
as the dearest, an estimate that errs high.

GLM 5.3's agent calls (a forge's verifier) report no cost, and its cached input, listed at $0.26, was billed at about
$1.20 on 2026-10-19: Prime's CSV against the day's traces (scripts/worldgen_slack/bill.py), where DeepSeek's cache
discount and Opus's reported costs matched the bill to 1%. It is priced as billed until a probe shows the discount."""

PRICES = {
    "z-ai/glm-5.3": (1.4, 1.2, 4.4),  # cached: billed, not listed (0.26)
    "z-ai/glm-5.3-flash": (0.15, 0.03, 0.5),
    "deepseek/deepseek-v4.1-flash": (0.3, 0.03, 1.2),
    "anthropic/claude-opus-5.5": (4.0, 4.0, 20.0),  # reports its own cost
    "anthropic/claude-sonnet-5.5": (2.0, 2.0, 10.0),
    "openai/gpt-6-sol": (2.0, 2.0, 10.0),
    "moonshotai/kimi-k2.6": (0.95, 0.95, 4.0),
    "qwen/qwen3.7-flash": (0.03, 0.03, 0.13),
}
DEAREST = (4.0, 4.0, 20.0)


def priced(usage: dict, model: str | None) -> float:
    """One call's (or one trace's) cost: as its provider reported it, else its tokens at these prices."""
    if usage.get("cost") is not None:
        return usage["cost"]
    fresh, cached, out = PRICES.get(model or "", DEAREST)
    return (
        (usage.get("prompt_tokens") or 0) * fresh
        + (usage.get("cached_input_tokens") or 0) * cached
        + (usage.get("completion_tokens") or 0) * out
    ) / 1e6
