"""Independent TypeScript packages must keep their metadata and dependencies consistent."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from _common import Plugin
from check_conventions import STANDARD_TYPESCRIPT_TEST_SUPPORT, Checker
from conftest import commit_all


def _plugin(repo: Path) -> Plugin:
    directory = repo / "typescript/fake"
    directory.mkdir(parents=True)
    package = {"name": "@temporalio/fake", "version": "0.0.0", "engines": {"node": ">=22.12.0"},
               "dependencies": {"@temporalio/workflow": "^1.24.0"},
               "devDependencies": {"@temporalio/test-helpers": "link:tests/helpers"}}
    (directory / "package.json").write_text(json.dumps(package))
    (directory / "plugin.toml").write_text('''[plugin]
name = "fake"
language = "typescript"
coordinate = "@temporalio/fake"
root-api = "@temporalio/fake"
registry = "npm"
maturity = "pre-release"
[ci]
runtime-versions = ["22", "24"]
[smoke]
imports = ["@temporalio/fake"]
''')
    for name in ("pnpm-lock.yaml", "pnpm-workspace.yaml", "tsconfig.json", "Makefile", "README.md"):
        (directory / name).touch()
    (directory / "LICENSE").write_bytes((repo / "LICENSE").read_bytes())
    for plugin_rel, template_rel in STANDARD_TYPESCRIPT_TEST_SUPPORT.items():
        canonical = repo / "typescript/_template" / template_rel
        canonical.parent.mkdir(parents=True, exist_ok=True)
        canonical.write_text(f"// canonical {plugin_rel}\n")
        rendered = directory / plugin_rel
        rendered.parent.mkdir(parents=True, exist_ok=True)
        rendered.write_bytes(canonical.read_bytes())
    commit_all(repo, "add TypeScript plugin")
    return Plugin("typescript", "fake", directory)


def test_independent_typescript_plugin_and_private_test_helpers(repo: Path) -> None:
    plugin = _plugin(repo)
    checker = Checker(repo)
    checker.check_typescript_plugin(plugin)
    assert checker.violations == []


@pytest.mark.parametrize("support_file", sorted(STANDARD_TYPESCRIPT_TEST_SUPPORT))
@pytest.mark.parametrize("failure", ["drift", "missing", "missing-template"])
def test_typescript_standard_test_support(repo: Path, support_file: str, failure: str) -> None:
    plugin = _plugin(repo)
    support = plugin.path / support_file
    if failure == "drift":
        support.write_text("// drifted\n")
        expected = "differs from canonical"
    elif failure == "missing":
        support.unlink()
        expected = "missing standard test support file"
    else:
        (repo / "typescript/_template" / STANDARD_TYPESCRIPT_TEST_SUPPORT[support_file]).unlink()
        expected = "missing canonical test support template"
    checker = Checker(repo)
    checker.check_typescript_plugin(plugin)
    assert len(checker.violations) == 1
    assert support_file in checker.violations[0] and expected in checker.violations[0]


@pytest.mark.parametrize("section", ["dependencies", "devDependencies", "peerDependencies"])
def test_typescript_rejects_dependencies_on_other_checkouts(repo: Path, section: str) -> None:
    plugin = _plugin(repo)
    manifest = plugin.path / "package.json"
    package = json.loads(manifest.read_text())
    package.setdefault(section, {})["@temporalio/other"] = "link:../other"
    manifest.write_text(json.dumps(package))
    checker = Checker(repo)
    checker.check_typescript_plugin(plugin)
    assert any("must use a published registry coordinate" in error for error in checker.violations)


def test_typescript_rejects_wrong_runtime_floor_and_missing_lock(repo: Path) -> None:
    plugin = _plugin(repo)
    (plugin.path / "pnpm-lock.yaml").unlink()
    meta = plugin.path / "plugin.toml"
    meta.write_text(meta.read_text().replace('runtime-versions = ["22", "24"]', 'runtime-versions = ["24"]'))
    checker = Checker(repo)
    checker.check_typescript_plugin(plugin)
    assert any("missing pnpm-lock.yaml" in error for error in checker.violations)
    assert any("Node engine floor" in error for error in checker.violations)


def _set_runtime(plugin: Plugin, floor: str, versions: str) -> None:
    manifest = plugin.path / "package.json"
    package = json.loads(manifest.read_text())
    package["engines"] = {"node": floor}
    manifest.write_text(json.dumps(package))
    meta = plugin.path / "plugin.toml"
    meta.write_text(meta.read_text().replace('runtime-versions = ["22", "24"]', f"runtime-versions = {versions}"))


def test_typescript_accepts_the_engine_floor_minor_as_the_minimum_runtime(repo: Path) -> None:
    plugin = _plugin(repo)
    _set_runtime(plugin, ">=22.17.0", '["22.17", "24"]')
    checker = Checker(repo)
    checker.check_typescript_plugin(plugin)
    assert checker.violations == []


@pytest.mark.parametrize("versions", ['["22.12", "24"]', '["22.17.0", "24"]', '["22", "24.1"]', '["23", "24"]'])
def test_typescript_rejects_runtime_versions_off_the_engine_floor(repo: Path, versions: str) -> None:
    plugin = _plugin(repo)
    _set_runtime(plugin, ">=22.17.0", versions)
    checker = Checker(repo)
    checker.check_typescript_plugin(plugin)
    assert any("Node engine floor" in error for error in checker.violations)
