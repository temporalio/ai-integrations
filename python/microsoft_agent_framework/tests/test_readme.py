"""The README's Python blocks must compile, import real names, and use only defined names."""

from __future__ import annotations

import ast
import builtins
import importlib
import re
import symtable
from pathlib import Path

README = Path(__file__).resolve().parents[1] / "README.md"

_BLOCK = re.compile(r"```python\n(.*?)```", re.DOTALL)

# The quickstart's own file, which its client.py imports.
_LOCAL_MODULES = {"workflow"}


def _python_blocks() -> list[str]:
    return _BLOCK.findall(README.read_text(encoding="utf-8"))


def test_readme_has_python_blocks() -> None:
    assert len(_python_blocks()) >= 3, "expected the quickstart and feature examples"


def test_readme_blocks_compile() -> None:
    for index, block in enumerate(_python_blocks()):
        compile(block, f"<README block {index}>", "exec")


def test_readme_imports_resolve() -> None:
    """Every imported module and name exists in the installed packages."""
    for index, block in enumerate(_python_blocks()):
        for node in ast.walk(ast.parse(block)):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    importlib.import_module(alias.name)
            elif isinstance(node, ast.ImportFrom) and node.module not in _LOCAL_MODULES:
                assert node.module is not None
                module = importlib.import_module(node.module)
                for alias in node.names:
                    assert hasattr(module, alias.name) or _importable(
                        f"{node.module}.{alias.name}"
                    ), f"README block {index}: {node.module} has no {alias.name}"


def _importable(name: str) -> bool:
    try:
        importlib.import_module(name)
    except ImportError:
        return False
    return True


def test_readme_names_are_defined() -> None:
    """Each block uses only builtins and names an earlier block, or itself, defines."""
    defined = set(dir(builtins))
    for index, block in enumerate(_python_blocks()):
        table = symtable.symtable(block, f"<README block {index}>", "exec")
        introduced = _module_definitions(table)
        missing = _referenced_globals(table) - defined - introduced
        assert not missing, (
            f"README block {index} uses undefined names {sorted(missing)}"
        )
        defined |= introduced


def _module_definitions(table: symtable.SymbolTable) -> set[str]:
    return {
        symbol.get_name()
        for symbol in table.get_symbols()
        if symbol.is_assigned() or symbol.is_imported() or symbol.is_namespace()
    }


def _referenced_globals(table: symtable.SymbolTable) -> set[str]:
    names: set[str] = set()
    if table.get_type() == "module":
        names |= {
            symbol.get_name()
            for symbol in table.get_symbols()
            if symbol.is_referenced()
            and not (
                symbol.is_assigned() or symbol.is_imported() or symbol.is_namespace()
            )
        }
    else:
        names |= {
            symbol.get_name()
            for symbol in table.get_symbols()
            if symbol.is_global() and symbol.is_referenced()
        }
    for child in table.get_children():
        names |= _referenced_globals(child)
    return names
