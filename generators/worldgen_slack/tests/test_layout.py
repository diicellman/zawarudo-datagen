"""The code's layout: the world part loads no module of the forge, the env imports no generator, no module imports
inside a function or in a cycle, the generators open a world file only as a Workspace, and every subpackage's
__init__ is empty."""

import ast
import subprocess
import sys
from pathlib import Path

from ..config import ROOT

PACKAGE = ROOT / "generators/worldgen_slack"
ENV = ROOT / "environments/worldgen_slack/worldgen_slack"


def sources(root: Path) -> list[Path]:
    return [p for p in sorted(root.rglob("*.py")) if "__pycache__" not in p.parts]


def module_of(path: Path) -> str:
    base = ROOT if path.is_relative_to(PACKAGE) else ENV.parent
    return ".".join(path.relative_to(base).with_suffix("").parts).removesuffix(".__init__")


MODULES = {module_of(p): p for p in sources(PACKAGE) + sources(ENV)}


def imported(path: Path) -> set[str]:
    """The modules of the code base `path` imports at its top level, by their absolute names."""
    package = module_of(path).split(".")
    if path.name != "__init__.py":
        package = package[:-1]
    out = set()
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = package[: len(package) - node.level + 1] if node.level else []
            module = ".".join([*base, *(node.module.split(".") if node.module else [])])
            # `from package import submodule` depends on the submodule; anything else, on the module it names
            names = [f"{module}.{a.name}" if f"{module}.{a.name}" in MODULES else module for a in node.names]
        else:
            continue
        out |= {name for name in names if name in MODULES}
    return out


def test_the_world_loads_no_forge():
    """Everything the world part runs on (its envs, its author, the judge, the session and the store) loads no module
    of the forge, in a fresh interpreter."""
    world = [m for m in MODULES if m.startswith("generators.worldgen_slack.world")]
    code = "import sys\n" + "".join(f"import {m}\n" for m in world)
    code += "print(sorted(m for m in sys.modules if m.startswith('generators.worldgen_slack.forge')))"
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT, check=True)
    assert done.stdout.strip() == "[]", done.stdout


def test_the_env_imports_no_generator():
    leaks = [
        f"{module_of(p)} imports {m}" for p in sources(ENV) for m in imported(p) if m.startswith("generators")
    ]
    assert not leaks, leaks


def test_no_import_inside_a_function():
    lazy = [
        f"{module_of(p)}:{node.lineno}"
        for p in sources(PACKAGE) + sources(ENV)
        for fn in ast.walk(ast.parse(p.read_text()))
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
        for node in ast.walk(fn)
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    assert not lazy, lazy


def test_no_import_cycle():
    graph = {m: imported(p) - {m} for m, p in MODULES.items()}
    state: dict[str, int] = {}  # 1 while on the path, 2 once done

    def visit(module: str, path: list[str]) -> None:
        state[module] = 1
        for nxt in sorted(graph[module]):
            if state.get(nxt) == 1:
                raise AssertionError("import cycle: " + " -> ".join(path[path.index(nxt) :] + [module, nxt]))
            if nxt not in state:
                visit(nxt, path + [module])
        state[module] = 2

    for module in sorted(graph):
        if module not in state:
            visit(module, [])


def test_a_world_file_is_opened_as_a_workspace():
    """The generators write the world file only through Workspace, and open it the same way to read it."""
    opened = [
        f"{module_of(p)}:{node.lineno}"
        for p in sources(PACKAGE)
        for node in ast.walk(ast.parse(p.read_text()))
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "World"
    ]
    assert not opened, opened


def test_subpackage_inits_are_empty():
    full = [str(p.relative_to(ROOT)) for p in PACKAGE.parent.rglob("__init__.py") if p.read_text().strip()]
    assert not full, full
