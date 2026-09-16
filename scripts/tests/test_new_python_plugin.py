from __future__ import annotations

import importlib.util
import tomllib
from pathlib import Path
from types import ModuleType

from conftest import commit_all, init_repo

import check_conventions

REPO = Path(__file__).resolve().parents[2]


def load_scaffolder() -> ModuleType:
    path = REPO / "scripts/new_python_plugin.py"
    spec = importlib.util.spec_from_file_location("new_python_plugin", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_new_plugin_is_top_level_and_release_ready(tmp_path: Path) -> None:
    repo = init_repo(tmp_path / "repo")
    scaffolder = load_scaffolder()
    scaffolder.REPO_ROOT = repo
    scaffolder.TEMPLATE = REPO / "python/_template"

    assert (
        scaffolder.main(
            ["fakeplug", "--description", "A fake integration", "--maturity", "preview"]
        )
        == 0
    )

    plugin = repo / "python/fakeplug"
    metadata = tomllib.loads((plugin / "plugin.toml").read_text())
    assert metadata["plugin"]["root-api"] == "temporalio.fakeplug"
    assert "upstream" not in metadata["plugin"]
    assert metadata["release"]["allow-final"] is True
    assert (plugin / "src/temporalio/fakeplug/__init__.py").is_file()
    assert not (plugin / "src/temporalio/contrib").exists()

    (plugin / "uv.lock").write_text("version = 1\n")
    commit_all(repo, "add generated plugin")
    assert check_conventions.Checker(repo).run(nightly=False) == []


def test_upstream_mode_uses_transitional_layout(tmp_path: Path) -> None:
    repo = init_repo(tmp_path / "repo")
    package = repo / "python/fakeplug/src/temporalio/contrib/fakeplug"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text('"""Imported package."""\n')
    scaffolder = load_scaffolder()
    scaffolder.REPO_ROOT = repo
    scaffolder.TEMPLATE = REPO / "python/_template"

    upstream = "temporalio/sdk-python:temporalio/contrib/fakeplug"
    assert (
        scaffolder.main(
            [
                "fakeplug",
                "--description",
                "A migrated integration",
                "--existing",
                "--upstream",
                upstream,
            ]
        )
        == 0
    )

    plugin = repo / "python/fakeplug"
    metadata = tomllib.loads((plugin / "plugin.toml").read_text())
    assert metadata["plugin"]["root-api"] == "temporalio.contrib.fakeplug"
    assert metadata["plugin"]["upstream"] == upstream
    assert metadata["release"]["allow-final"] is False
    assert (package / "py.typed").is_file()
    assert not (plugin / "src/temporalio/fakeplug").exists()
