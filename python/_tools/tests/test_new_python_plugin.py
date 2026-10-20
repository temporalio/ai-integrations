from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from types import ModuleType

import check_conventions
import pytest
import tomllib
from conftest import commit_all, init_repo

REPO = Path(__file__).resolve().parents[3]


def load_scaffolder() -> ModuleType:
    path = REPO / "python/_tools/new_python_plugin.py"
    spec = importlib.util.spec_from_file_location("new_python_plugin", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "maturity", ["pre-release", "public-preview", "generally-available"]
)
def test_new_plugin_is_top_level_and_release_ready(
    tmp_path: Path, maturity: str
) -> None:
    repo = init_repo(tmp_path / "repo")
    scaffolder = load_scaffolder()
    scaffolder.REPO_ROOT = repo
    scaffolder.TEMPLATE = REPO / "python/_template"

    assert (
        scaffolder.main(
            ["fakeplug", "--description", "A fake integration", "--maturity", maturity]
        )
        == 0
    )

    plugin = repo / "python/fakeplug"
    metadata = tomllib.loads((plugin / "plugin.toml").read_text())
    assert metadata["plugin"]["maturity"] == maturity
    assert metadata["plugin"]["root-api"] == "temporalio.fakeplug"
    assert (plugin / "src/temporalio/fakeplug/__init__.py").is_file()
    assert not (plugin / "src/temporalio/contrib").exists()
    assert (plugin / "tests/test_installed_matches_source.py").is_file()
    assert (plugin / "tests/helpers/environment.py").is_file()
    assert (plugin / "tests/test_env.py").is_file()
    for source in (plugin / "tests").rglob("*.py"):
        ast.parse(source.read_text(), filename=str(source))

    conftest = (plugin / "tests/conftest.py").read_text()
    assert "import pytest_asyncio\n\nfrom temporalio.client" in conftest

    (plugin / "uv.lock").write_text("version = 1\n")
    commit_all(repo, "add generated plugin")
    assert check_conventions.Checker(repo).run() == []
