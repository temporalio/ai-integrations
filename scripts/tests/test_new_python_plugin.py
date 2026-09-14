from __future__ import annotations

import ast
import importlib.util
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]


def _load_scaffolder():  # type: ignore[reportMissingParameterType]
    path = REPO / "scripts/new_python_plugin.py"
    spec = importlib.util.spec_from_file_location("new_python_plugin", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fresh_scaffold_is_owned_locally_and_has_a_runnable_test(
    tmp_path: Path,
) -> None:
    scaffolder = _load_scaffolder()
    root = tmp_path / "repo"
    root.mkdir()
    (root / "LICENSE").write_text("license\n")
    scaffolder.REPO_ROOT = root
    scaffolder.TEMPLATE = REPO / "python/_template"

    assert scaffolder.main(["fresh_integration", "--description", "Fresh plugin"]) == 0

    plugin = root / "python/fresh_integration"
    manifest = (plugin / "plugin.toml").read_text()
    assert "upstream =" not in manifest
    assert "imported-from =" not in manifest
    assert "TRANSITION(sdk-cutover)" not in manifest
    assert "TRANSITION(sdk-cutover)" not in (plugin / "pyproject.toml").read_text()
    assert (plugin / "tests/test_installed_matches_source.py").is_file()
    for source in (plugin / "tests").rglob("*.py"):
        ast.parse(source.read_text(), filename=str(source))

    conftest = (plugin / "tests/conftest.py").read_text()
    assert "import pytest_asyncio\n\nfrom temporalio.client" in conftest
