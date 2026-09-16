from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import check_conventions
from conftest import commit_all, make_python_plugin


def run(repo: Path, nightly: bool = False) -> list[str]:
    return check_conventions.Checker(repo).run(nightly=nightly)


def test_empty_repo_passes(repo: Path) -> None:
    assert run(repo) == []


def test_valid_plugin_passes(plugin_repo: Path) -> None:
    assert run(plugin_repo) == []


def test_python_version_is_the_development_placeholder(plugin_repo: Path) -> None:
    pyproject = plugin_repo / "python/fakeplug/pyproject.toml"
    pyproject.write_text(pyproject.read_text().replace('version = "0.0.0"', 'version = "0.1.0"'))
    assert any("tag-authoritative development placeholder" in x for x in run(plugin_repo))


def test_namespace_init_files_are_forbidden(plugin_repo: Path) -> None:
    (plugin_repo / "python/fakeplug/src/temporalio/__init__.py").write_text("")
    (plugin_repo / "python/fakeplug/src/temporalio/contrib/__init__.py").write_text("")
    v = run(plugin_repo)
    assert any("src/temporalio/__init__.py" in x for x in v) and any("src/temporalio/contrib/__init__.py" in x for x in v)


def test_license_is_a_committed_identical_copy(plugin_repo: Path) -> None:
    lic = plugin_repo / "python/fakeplug/LICENSE"
    assert run(plugin_repo) == []  # committed copy identical to the root file
    lic.write_text("MIT\n")
    assert any("differs from the root LICENSE" in x for x in run(plugin_repo))
    lic.unlink()
    lic.mkdir()
    assert any("must be a regular file" in x for x in run(plugin_repo))
    lic.rmdir()
    lic.write_bytes((plugin_repo / "LICENSE").read_bytes())
    subprocess.run(["git", "-C", str(plugin_repo), "rm", "-q", "--cached", "python/fakeplug/LICENSE"], check=True)
    assert any("must be committed" in x for x in run(plugin_repo))


def test_changelog_and_smoke_test_files_are_forbidden(plugin_repo: Path) -> None:
    (plugin_repo / "python/fakeplug/CHANGELOG.md").write_text("# x\n")
    (plugin_repo / "python/fakeplug/smoke_test.py").write_text("")
    commit_all(plugin_repo, "add forbidden files")
    v = run(plugin_repo)
    assert any("CHANGELOG.md" in x for x in v) and any("smoke_test.py" in x for x in v)


def test_relative_readme_links_are_forbidden(plugin_repo: Path) -> None:
    (plugin_repo / "python/fakeplug/README.md").write_text("see [streams](../workflow_streams/README.md)\n")
    assert any("relative link" in x for x in run(plugin_repo))


def test_plugin_folder_suffix_and_language_lockfile(plugin_repo: Path) -> None:
    (plugin_repo / "python/uv.lock").write_text("")
    (plugin_repo / "python/foo_plugin").mkdir()
    (plugin_repo / "python/foo_plugin/pyproject.toml").write_text("[project]\nname='x'\nversion='0'\n")
    v = run(plugin_repo)
    assert any("language-level lockfiles" in x for x in v) and any("must not end in -plugin/_plugin" in x for x in v)


def test_plugin_toml_agreement(plugin_repo: Path) -> None:
    meta = plugin_repo / "python/fakeplug/plugin.toml"
    meta.write_text(meta.read_text().replace('coordinate = "temporalio-fakeplug"', 'coordinate = "temporalio-other"'))
    v = run(plugin_repo)
    assert any("coordinate must be 'temporalio-fakeplug'" in x for x in v)
    assert any("project.name" in x for x in v)


def test_top_level_temporalio_root_api_is_allowed(plugin_repo: Path) -> None:
    meta = plugin_repo / "python/fakeplug/plugin.toml"
    meta.write_text(
        meta.read_text().replace("temporalio.contrib.fakeplug", "temporalio.fakeplug")
    )
    pyproject = plugin_repo / "python/fakeplug/pyproject.toml"
    pyproject.write_text(
        pyproject.read_text().replace(
            "temporalio.contrib.fakeplug", "temporalio.fakeplug"
        )
    )
    old_package = plugin_repo / "python/fakeplug/src/temporalio/contrib/fakeplug"
    new_package = plugin_repo / "python/fakeplug/src/temporalio/fakeplug"
    old_package.rename(new_package)
    assert run(plugin_repo) == []


def test_contrib_root_api_requires_upstream(plugin_repo: Path) -> None:
    meta = plugin_repo / "python/fakeplug/plugin.toml"
    meta.write_text(
        meta.read_text().replace(
            'upstream = "temporalio/sdk-python:temporalio/contrib/fakeplug"\n',
            "",
        )
    )
    assert any("allowed only for an upstream-backed migration" in x for x in run(plugin_repo))


def test_contrib_root_api_requires_final_releases_disabled(plugin_repo: Path) -> None:
    meta = plugin_repo / "python/fakeplug/plugin.toml"
    meta.write_text(meta.read_text().replace("allow-final = false", "allow-final = true"))
    assert any("allowed only for an upstream-backed migration" in x for x in run(plugin_repo))


@pytest.mark.parametrize(
    "malformed",
    ['["temporalio.contrib.fakeplug"]', '{ value = "temporalio.contrib.fakeplug" }'],
)
def test_malformed_root_api_is_reported(plugin_repo: Path, malformed: str) -> None:
    meta = plugin_repo / "python/fakeplug/plugin.toml"
    meta.write_text(
        meta.read_text().replace(
            'root-api = "temporalio.contrib.fakeplug"', f"root-api = {malformed}"
        )
    )
    assert any("root-api must be 'temporalio.fakeplug'" in x for x in run(plugin_repo))


def test_unrelated_root_api_is_rejected(plugin_repo: Path) -> None:
    meta = plugin_repo / "python/fakeplug/plugin.toml"
    meta.write_text(meta.read_text().replace("temporalio.contrib.fakeplug", "other.fakeplug"))
    assert any("root-api must be 'temporalio.fakeplug'" in x for x in run(plugin_repo))


def test_maturity_classifier_must_agree(plugin_repo: Path) -> None:
    meta = plugin_repo / "python/fakeplug/plugin.toml"
    meta.write_text(meta.read_text().replace('maturity = "experimental"', 'maturity = "ga"'))
    assert any("Development Status :: 5 - Production/Stable" in x for x in run(plugin_repo))


def test_no_secrets_or_owners_in_plugin_toml(plugin_repo: Path) -> None:
    meta = plugin_repo / "python/fakeplug/plugin.toml"
    meta.write_text(meta.read_text().replace('[ci]\n', '[ci]\nlive-secrets = ["OPENAI_API_KEY"]\n'))
    assert any("live-secrets" in x for x in run(plugin_repo))


def test_path_sources_and_required_version(plugin_repo: Path) -> None:
    pp = plugin_repo / "python/fakeplug/pyproject.toml"
    pp.write_text(pp.read_text().replace('required-version = ">=0.12.5,<0.13"', 'required-version = ">=0.11"')
                  + '\n[tool.uv.sources]\ntemporalio-mcp = { path = "../mcp" }\n')
    v = run(plugin_repo)
    assert any("path/workspace source" in x for x in v) and any("required-version" in x for x in v)


def test_exclude_newer_requires_temporalio_exemption(plugin_repo: Path) -> None:
    pp = plugin_repo / "python/fakeplug/pyproject.toml"
    pp.write_text(pp.read_text().replace("exclude-newer-package = { temporalio = false }\n", ""))
    assert any("temporalio is not exempted" in x for x in run(plugin_repo))


def test_pr_commit_count_does_not_imply_history_import(plugin_repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")
    monkeypatch.setenv("PR_COMMITS", "25")
    monkeypatch.setenv("PR_LABELS", '["enhancement"]')
    assert run(plugin_repo) == []


def test_cli_exit_codes(plugin_repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert check_conventions.main(["--repo-root", str(plugin_repo)]) == 0
    (plugin_repo / "python/fakeplug/src/temporalio/__init__.py").write_text("")
    assert check_conventions.main(["--repo-root", str(plugin_repo)]) == 1
    assert "FAIL" in capsys.readouterr().out
