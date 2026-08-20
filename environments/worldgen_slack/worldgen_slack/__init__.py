"""Prime-native Slack synthetic data generation."""

from typing import Any

__all__ = ["GenerationSeedTaskset", "SlackDataGenerationEnv"]


def __getattr__(name: str) -> Any:
    if name == "GenerationSeedTaskset":
        from .tasksets.generation import GenerationSeedTaskset

        return GenerationSeedTaskset
    if name == "SlackDataGenerationEnv":
        from .env import SlackDataGenerationEnv

        return SlackDataGenerationEnv
    raise AttributeError(name)
