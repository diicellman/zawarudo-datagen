"""What a model reads names no model and no earlier run, and guides render their numbers from code."""

import ast
import re
from pathlib import Path

from ..config import ROOT
from ..forge.judge import PROMPT as JUDGE_PROMPT
from ..forge.proposer.task import ProposerTask, forge_guide
from ..workspace import Workspace
from ..world.author.task import AUTHOR_GUIDE, CHAPTER_GUIDE, CRAFT_GUIDE, AuthorTask, targets_text
from ..world.clock import PARTS
from ..world.config import LedgerTargets
from ..world.profile import MATERIAL
from .support import contracts_settings, forge_of

# The files whose strings a model reads: guides, turn prompts, refusals, memory pages and tool descriptions.
PROMPT_SOURCES = [
    *(
        ROOT / "generators/worldgen_slack" / n
        for n in ("judge/task.py", "judge/tools.py", "judge/verdict.py", "session.py")
    ),
    *sorted((ROOT / "generators/worldgen_slack/world").rglob("*.py")),
    *sorted((ROOT / "generators/worldgen_slack/forge").rglob("*.py")),
    *sorted((ROOT / "generators/worldgen_slack").rglob("prompts/*.md")),
    *(
        ROOT / "environments/worldgen_slack/worldgen_slack" / n
        for n in ("taskset.py", "judge.py", "toolset.py", "slack.py", "world.py")
    ),
]


def prompt_strings(path: Path) -> list[tuple[int, str]]:
    """A source file's strings that reach a model: every literal (f-strings' parts too) but docstrings, which no model
    reads unless they describe a tool, and model ids (a provider/model path)."""
    if path.suffix == ".md":  # a prompt file reaches a model whole
        return list(enumerate(path.read_text().splitlines(), 1))
    tree, skip = ast.parse(path.read_text()), set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (
            isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            and body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
        ):
            if not any(
                getattr(d, "attr", getattr(d, "id", None)) == "tool"
                for d in getattr(node, "decorator_list", [])
            ):
                skip.add(id(body[0].value))
    return [
        (n.lineno, n.value)
        for n in ast.walk(tree)
        if isinstance(n, ast.Constant)
        and isinstance(n.value, str)
        and id(n) not in skip
        and not re.fullmatch(r"[\w.-]+/[\w.-]+", n.value)
    ]


def test_prompts(tmp_path):
    """What a model reads names no model and no earlier run: a proposer knows the solver by its tries, an author the
    world by its memory. Guides render their numbers from code: the forge's tries and middle rates, the parts of the
    day, the near-miss queries a task may name."""
    settings = contracts_settings(tmp_path)
    forge = forge_of(settings)
    models = [
        settings.env.author.model,
        settings.env.judge.model,
        forge.env.solver.model,
        forge.env.verifier.model,
        forge.env.proposer.model,
        forge.judge.model,
        forge.answer_judge.model,
        "deepseek/deepseek-v4.1-flash",
    ]
    names = {
        "glm",
        "gpt",
        "claude",
        "opus",
        "sonnet",
        "haiku",
        "gemini",
        "deepseek",
        "qwen",
        "kimi",
        "llama",
        "openai",
        "anthropic",
    }
    names |= {
        w
        for m in models
        for w in (m.split("/")[0], re.match(r"[a-z]+", m.split("/")[-1]).group())
        if len(w) > 2
    }
    banned = re.compile(r"\b(?:" + "|".join(sorted(names)) + r")\b|\b[SFM]\d\b|\bforge-t\d", re.IGNORECASE)
    found = [
        f"{path.relative_to(ROOT)}:{line}: {m.group(0)!r}"
        for path in PROMPT_SOURCES
        for line, text in prompt_strings(path)
        for m in banned.finditer(text)
    ]
    configured = [
        settings.styles,
        *(
            [c.definition, *c.levels, *(x for level in c.concepts for x in level)]
            for c in settings.taxonomy.values()
        ),
    ]
    found += [
        f"config: {m.group(0)!r} in {text[:60]!r}"
        for texts in configured
        for text in texts
        for m in banned.finditer(text)
    ]
    assert not found, "model or run names in what a model reads: " + "; ".join(found)
    guide = forge_guide(forge.model_dump(mode="json"))
    assert "tries it 8 times" in guide, "the forge's own numbers"
    assert "tries it 3 times" in forge_guide(forge.model_copy(update={"tries": 3}).model_dump(mode="json"))
    # It knows its own turn's budget, and that only what add_task took outlives the turn.
    budget = f"spent its {forge.env.proposer.harness.max_total_tokens:,} new tokens"
    assert budget in " ".join(guide.split()) and "kept the moment" in guide
    # The proposer writes requests a colleague would make, by the judge's own test, and gets its middle rates from
    # composite requests along the four skills, not from tool tedium; its answers are what its parts ask, no more; a
    # hedge is wrong.
    flat, review = " ".join(guide.split()), " ".join(JUDGE_PROMPT.split())
    assert all(
        phrase in flat
        for phrase in (
            "Ask what the asker would do with the answer",
            "Name its family",
            "Pick the asker whose work needs the answer",
            "A composite request is the main lever",
            "calibrated abstention",
            "tool mechanics and tedium",
            "the shortest complete answer to the label",
            "with world_sql as the asker",
            "A hedge is wrong",
        )
    ), "the guide's rules"
    assert (
        "treats colleagues' activity as data" in flat and "treats colleagues' activity as data" in review
    ), "the proposer aims at the judge's own test of a request"
    assert "intent" not in flat, "a task is a family, not one of the seven intents"
    # It reads the material's collections and promises, aims at the skill an id names, and opens requests anew (its
    # one example of a composite no longer opens "before I...", which a quarter of a set's requests echoed).
    assert (
        all(
            phrase in flat
            for phrase in (
                "the collections",
                "slipped: past due",
                "the skill its request needs",
                "openings already used",
            )
        )
        and "before i" not in flat.lower()
    ), "the guide's material, skills and openings"
    # The author plans the ledger as real work leaves it: values that change out of a first search's reach,
    # near-misses beside the real value, collections never listed whole, handoffs, and promises that may slip.
    authoring = " ".join(AUTHOR_GUIDE.split())
    assert all(
        phrase in authoring
        for phrase in (
            "where a search for the first one's words does not reach them",
            "Near-misses (decoy)",
            "never listed whole in one place",
            "Handoffs",
            "unless it slips",
            "Quantities that must add up",
        )
    ), "the guide's fact bundle"
    parts = ", ".join(f"{p} {a:02d}-{b:02d}" for p, (a, b) in PARTS.items())
    assert parts in " ".join(AUTHOR_GUIDE.split()) and "PARTS_OF_THE_DAY" not in AUTHOR_GUIDE
    assert "nothing more specific" in review, "the judge holds each answer to what its part asks"
    # The plan's prompt asks for what the world's ledger holds at least, in numbers from its config, and nothing when
    # the config asks nothing.
    dense = settings.model_copy(update={"ledger": LedgerTargets(facts=240, chains=30, collections=12)})
    assert targets_text(settings) == "" and targets_text(dense) == (
        " The ledger holds at least: 240 facts; 30 "
        + MATERIAL["chains"]
        + "; 12 "
        + MATERIAL["collections"]
        + "."
    )
    assert "written and frozen" in CHAPTER_GUIDE and "No tasks are written in a chapter" in CHAPTER_GUIDE
    # The world's author writes with the craft page (voice lab 02); the proposer has its own guide, and a chapter's
    # author has the chapter's too.
    world = Workspace.create(tmp_path / "w.sqlite")
    context = {"settings": settings.model_dump(mode="json")}
    systems = {
        name: AuthorTask.create(mode, 1, world.path, context | extra, "x").data.system_prompt
        for name, mode, extra in (
            ("plan", "plan", {}),
            ("day", "day", {}),
            ("chapter", "day", {"chapter": {"first": 5}}),
        )
    }
    proposing = {"settings": forge.model_dump(mode="json")}
    systems["forge"] = ProposerTask.create(world.path, proposing, "x").data.system_prompt
    assert (
        all(
            "Nobody states the lesson" in systems[m] and "Stakes move" in systems[m]
            for m in ("plan", "day", "chapter")
        )
        and "Stakes move" not in systems["forge"]
    )
    assert (
        CHAPTER_GUIDE in systems["chapter"]
        and CHAPTER_GUIDE not in systems["day"]
        and CRAFT_GUIDE.strip() in systems["day"]
    )
    world.close()
