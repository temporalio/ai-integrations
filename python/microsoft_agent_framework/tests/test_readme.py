"""The README's Python blocks must compile, import real names, and use only defined names."""

from __future__ import annotations

import ast
import asyncio
import builtins
import importlib
import re
import symtable
import sys
from pathlib import Path
from typing import Any

import pytest

from temporalio.client import Client
from temporalio.microsoft_agent_framework import MicrosoftAgentFrameworkPlugin
from temporalio.microsoft_agent_framework.testing import fake_chat_client
from temporalio.worker import Worker

README = Path(__file__).resolve().parents[1] / "README.md"

_BLOCK = re.compile(r"```python\n(.*?)```", re.DOTALL)
# A python block introduced by a line that names its file, as the quickstart's are.
_NAMED_BLOCK = re.compile(r"`(\w+\.py)`[^\n]*\n\n```python\n(.*?)```", re.DOTALL)

# The quickstart's own module, which its worker.py and client.py import.
_LOCAL_MODULES = {"workflow"}
_QUICKSTART_FILES = ("workflow.py", "worker.py", "client.py")


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


def _quickstart_files() -> dict[str, str]:
    files: dict[str, str] = {}
    for match in _NAMED_BLOCK.finditer(README.read_text(encoding="utf-8")):
        name, block = match.group(1), match.group(2)
        if name in _QUICKSTART_FILES and name not in files:
            files[name] = block
    return files


async def test_quickstart_runs(
    client: Client,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The quickstart files run as written: worker.py's main() and client.py's main().

    Compiling the blocks does not catch a provider import in ``workflow.py``:
    the sandbox re-imports that file per run, and the OpenAI SDK's import tree
    trips its restrictions, so ``Worker(...)`` raises at validation.
    """
    files = _quickstart_files()
    assert set(files) == set(_QUICKSTART_FILES)
    package = tmp_path / "quickstart"
    package.mkdir()
    for name, block in files.items():
        (package / name).write_text(block, encoding="utf-8")
    monkeypatch.syspath_prepend(str(package))
    names = [name.removesuffix(".py") for name in _QUICKSTART_FILES]
    try:
        worker_module = importlib.import_module("worker")
        client_module = importlib.import_module("client")
        # The files connect to localhost:7233; both get the test server
        # instead, and the worker's plugin gets the fake client in place of
        # OpenAI. Everything else is theirs, the task queue and the workflow id
        # included: a mismatch between the two files leaves the run waiting.
        config = client.config()

        async def connect(target_host: str, *, plugins: Any = (), **_: Any) -> Client:
            assert target_host == "localhost:7233"
            return Client(**{**config, "plugins": list(plugins)})

        monkeypatch.setattr(Client, "connect", connect)
        monkeypatch.setattr(
            worker_module,
            "plugin",
            MicrosoftAgentFrameworkPlugin(
                chat_clients={"gpt-4o": fake_chat_client(["Hi."])}
            ),
        )
        worker_task = asyncio.create_task(worker_module.main())
        try:
            await asyncio.wait_for(client_module.main(), timeout=30)
        finally:
            worker_task.cancel()
            await asyncio.gather(worker_task, return_exceptions=True)
    finally:
        for name in names:
            sys.modules.pop(name, None)
    assert capsys.readouterr().out.strip() == "Hi."
