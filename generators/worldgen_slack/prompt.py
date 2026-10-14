"""Prompt text, kept in a prompts/ folder beside the code that sends it: read once, whole (its first and last newlines
included), and filled with code's values through named placeholders alone."""

import re
from pathlib import Path


def prompt(module: str, name: str) -> str:
    """The text of prompts/`name` beside `module` (a module's __file__)."""
    return (Path(module).with_name("prompts") / name).read_text(encoding="utf-8")


def render(template: str, **values: str) -> str:
    """`template` with each {name} of `values` replaced in one pass. Any other brace stays as it is (the {at:id} an
    author writes in a message), and a value with no placeholder is refused."""
    if unused := [name for name in values if "{" + name + "}" not in template]:
        raise KeyError(f"the template has no placeholder for {unused}")
    return re.sub(r"\{(\w+)\}", lambda m: values.get(m.group(1), m.group(0)), template)
