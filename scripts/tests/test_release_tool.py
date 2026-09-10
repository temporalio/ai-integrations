from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest
from packaging.version import Version

import release_tool
from conftest import commit_all, git, make_python_plugin


def test_parse_tag_valid() -> None:
    assert release_tool.parse_tag("python/openai_agents/v1.0.0rc1") == {
        "tag": "python/openai_agents/v1.0.0rc1", "language": "python", "plugin": "openai_agents", "version": "1.0.0rc1",
        "prerelease": "true", "plugin_dir": "python/openai_agents",
    }
    assert release_tool.parse_tag("go/googleadk/v0.3.0")["prerelease"] == "false"
    assert release_tool.parse_tag("typescript/vercel-ai-sdk/v1.0.0.dev1")["prerelease"] == "true"


@pytest.mark.parametrize("tag", [
    "v1.0.0", "python/v1.0.0", "rust/foo/v1.0.0", "python/Foo/v1.0.0", "python/foo/1.0.0",
    "python/foo/v1.0.0-rc1", "python/foo/v1.0.0RC1", "python/foo/v01.0.0", "python/foo/v1.0.0+local", "python/foo/vabc",
])
def test_parse_tag_invalid(tag: str) -> None:
    with pytest.raises(release_tool.PolicyError):
        release_tool.parse_tag(tag)


def test_policy_first_release() -> None:
    release_tool.check_policy(Version("1.0.0"), "ga", [])
    release_tool.check_policy(Version("1.0.0rc1"), "ga", [])
    release_tool.check_policy(Version("0.1.0"), "experimental", [])
    release_tool.check_policy(Version("0.1.0a1"), "preview", [])
    for v, m in [("0.1.0", "ga"), ("1.0.0", "experimental"), ("1.1.0", "ga"), ("1.0.0.post1", "ga"), ("2.0.0", "ga")]:
        with pytest.raises(release_tool.PolicyError):
            release_tool.check_policy(Version(v), m, [])


def test_policy_existing_coordinate_moves_strictly_forward() -> None:
    published = [Version("1.0.0"), Version("1.1.0"), Version("0.9.0")]
    release_tool.check_policy(Version("1.1.1"), "ga", published)
    release_tool.check_policy(Version("1.2.0rc1"), "ga", published)
    release_tool.check_policy(Version("2.0.0"), "experimental", published)  # existing coordinates: no major rule
    for v in ("1.1.0", "1.0.5", "0.9.1", "1.1.0rc1"):
        with pytest.raises(release_tool.PolicyError):
            release_tool.check_policy(Version(v), "ga", published)


def _registry(tmp_path: Path, versions: list[str] | None) -> Path:
    f = tmp_path / "registry.json"
    if versions is None:
        return f  # missing file == 404
    f.write_text(json.dumps({"releases": {v: [] for v in versions}}))
    return f


def test_cli_policy_prerelease_with_no_published_versions(plugin_repo: Path, tmp_path: Path) -> None:
    reg = _registry(tmp_path, None)
    rc = release_tool.main(["--repo-root", str(plugin_repo), "check-version-policy", "--plugin-dir", str(plugin_repo / "python/fakeplug"),
                            "--version", "0.1.0rc1", "--registry-json", str(reg), "--testpypi-json", str(tmp_path / "absent.json")])
    assert rc == 0


def test_cli_policy_final_blocked_by_allow_final(plugin_repo: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    reg = _registry(tmp_path, None)
    rc = release_tool.main(["--repo-root", str(plugin_repo), "check-version-policy", "--plugin-dir", str(plugin_repo / "python/fakeplug"),
                            "--version", "0.1.0", "--registry-json", str(reg)])
    assert rc == 1 and "allow-final is false" in capsys.readouterr().out


def test_cli_policy_final_blocked_by_transition_markers(repo: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    d = make_python_plugin(repo, "fakeplug", allow_final=True)
    (d / "src/temporalio/contrib/fakeplug/_impl.py").write_text("# TRANSITION(sdk-cutover): remove\nVALUE = 1\n")
    commit_all(repo, "plugin")
    reg = _registry(tmp_path, None)
    policy = ["--repo-root", str(repo), "check-version-policy", "--plugin-dir", str(d), "--version", "0.1.0", "--registry-json", str(reg),
              "--testpypi-json", str(tmp_path / "absent.json")]
    rc = release_tool.main(policy)
    out = capsys.readouterr().out
    assert rc == 1 and "TRANSITION(sdk-cutover)" in out
    (d / "src/temporalio/contrib/fakeplug/_impl.py").write_text("VALUE = 1\n")
    commit_all(repo, "clean")
    assert release_tool.main(policy) == 0


def test_cli_policy_warns_when_the_version_is_already_on_testpypi(plugin_repo: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    reg = _registry(tmp_path, None)
    staged = tmp_path / "testpypi.json"
    staged.write_text(json.dumps({"releases": {"0.1.0rc1": []}}))
    args = ["--repo-root", str(plugin_repo), "check-version-policy", "--plugin-dir", str(plugin_repo / "python/fakeplug"),
            "--registry-json", str(reg), "--testpypi-json", str(staged)]
    # A re-run after a staged upload is the normal recovery path, so this is a warning, not a failure.
    assert release_tool.main([*args, "--version", "0.1.0rc1"]) == 0
    assert "::warning::" in capsys.readouterr().out
    assert release_tool.main([*args, "--version", "0.1.0rc2"]) == 0
    assert "not yet on TestPyPI" in capsys.readouterr().out


def test_cli_policy_does_not_touch_testpypi_unless_asked(plugin_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_args, **_kwargs):
        raise AssertionError("network access")

    monkeypatch.setattr(release_tool.urllib.request, "urlopen", boom)
    reg = _registry(tmp_path, None)
    assert release_tool.main(["--repo-root", str(plugin_repo), "check-version-policy", "--plugin-dir", str(plugin_repo / "python/fakeplug"),
                              "--version", "0.1.0rc1", "--registry-json", str(reg)]) == 0


def test_transition_markers_fail_closed_on_git_errors(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def broken(*_args, **_kwargs):
        return subprocess.CompletedProcess(args=[], returncode=128, stdout="", stderr="fatal: not a git repository")

    monkeypatch.setattr(release_tool.subprocess, "run", broken)
    with pytest.raises(release_tool.PolicyError, match="git grep"):
        release_tool.transition_markers(tmp_path, "python/fakeplug")

    def missing(*_args, **_kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(release_tool.subprocess, "run", missing)
    with pytest.raises(release_tool.PolicyError, match="git is required"):
        release_tool.transition_markers(tmp_path, "python/fakeplug")


def _dist(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    dist = tmp_path / "dist"
    dist.mkdir()
    digests = {}
    for name, payload in (("fakeplug-0.1.0-py3-none-any.whl", b"wheel"), ("fakeplug-0.1.0.tar.gz", b"sdist")):
        (dist / name).write_bytes(payload)
        digests[name] = hashlib.sha256(payload).hexdigest()
    return dist, digests


def _index_json(tmp_path: Path, digests: dict[str, str]) -> Path:
    f = tmp_path / "release.json"
    f.write_text(json.dumps({"urls": [{"filename": n, "digests": {"sha256": d}} for n, d in digests.items()]}))
    return f


def test_verify_index_files_accepts_identical_digests(tmp_path: Path) -> None:
    dist, digests = _dist(tmp_path)
    assert release_tool.main(["verify-index-files", "--coordinate", "temporalio-fakeplug", "--version", "0.1.0", "--dist", str(dist),
                              "--index", "testpypi", "--attempts", "1", "--delay", "0", "--index-json", str(_index_json(tmp_path, digests))]) == 0


def test_verify_index_files_rejects_different_bytes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    dist, digests = _dist(tmp_path)
    digests["fakeplug-0.1.0.tar.gz"] = hashlib.sha256(b"someone else's sdist").hexdigest()
    assert release_tool.main(["verify-index-files", "--coordinate", "temporalio-fakeplug", "--version", "0.1.0", "--dist", str(dist),
                              "--attempts", "1", "--delay", "0", "--index-json", str(_index_json(tmp_path, digests))]) == 1
    assert "different bytes" in capsys.readouterr().out


def test_verify_index_files_rejects_missing_files(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    dist, digests = _dist(tmp_path)
    digests.pop("fakeplug-0.1.0.tar.gz")
    assert release_tool.main(["verify-index-files", "--coordinate", "temporalio-fakeplug", "--version", "0.1.0", "--dist", str(dist),
                              "--attempts", "2", "--delay", "0", "--index-json", str(_index_json(tmp_path, digests))]) == 1
    assert "does not serve" in capsys.readouterr().out
    # An absent release (404) is reported the same way rather than as a crash.
    assert release_tool.main(["verify-index-files", "--coordinate", "temporalio-fakeplug", "--version", "0.1.0", "--dist", str(dist),
                              "--attempts", "1", "--delay", "0", "--index-json", str(tmp_path / "missing.json")]) == 1


def test_cli_policy_existing_versions(plugin_repo: Path, tmp_path: Path) -> None:
    reg = _registry(tmp_path, ["0.1.0", "0.2.0"])
    args = ["--repo-root", str(plugin_repo), "check-version-policy", "--plugin-dir", str(plugin_repo / "python/fakeplug"), "--registry-json", str(reg),
            "--testpypi-json", str(tmp_path / "absent.json")]
    assert release_tool.main([*args, "--version", "0.3.0rc1"]) == 0
    assert release_tool.main([*args, "--version", "0.2.0"]) == 1
    assert release_tool.main([*args, "--version", "0.1.5"]) == 1


def test_release_notes_initial_and_incremental(plugin_repo: Path, tmp_path: Path) -> None:
    repo = plugin_repo
    git(repo, "tag", "python/fakeplug/v0.1.0rc1")
    notes = release_tool.release_notes(repo, "python/fakeplug", "python/fakeplug/v0.1.0rc1", "temporalio/ai-integrations")
    assert "First standalone release of `temporalio-fakeplug`" in notes
    assert "TestPyPI only" in notes and "temporalio.contrib.fakeplug" in notes
    assert "commits/main/temporalio/contrib/fakeplug" in notes
    assert "tree/refs/tags/python/fakeplug/v0.1.0rc1/python/fakeplug" in notes

    (repo / "python/fakeplug/src/temporalio/contrib/fakeplug/_impl.py").write_text("VALUE = 2\n")
    commit_all(repo, "Fix the thing (#12)")
    (repo / "README.md").write_text("# other\n")
    commit_all(repo, "Unrelated root change (#13)")
    (repo / "python/fakeplug/README.md").write_text("# fake\n\nmore https://example.com\n")
    commit_all(repo, "Docs: mention temporalio/sdk-python#77 (#14)")
    git(repo, "tag", "python/fakeplug/v0.1.0")
    notes = release_tool.release_notes(repo, "python/fakeplug", "python/fakeplug/v0.1.0", "temporalio/ai-integrations")
    assert "[#12](https://github.com/temporalio/ai-integrations/pull/12)" in notes
    assert "[#14](https://github.com/temporalio/ai-integrations/pull/14)" in notes
    assert "#13" not in notes
    assert "temporalio/sdk-python#77" in notes and "pull/77" not in notes
    assert "compare/python/fakeplug/v0.1.0rc1...python/fakeplug/v0.1.0" in notes
    assert "Pre-release notes" not in notes


def test_prerelease_notes_warn_about_sdk_overlap_only_while_the_sdk_bundles_the_plugin(repo: Path) -> None:
    d = make_python_plugin(repo, "fakeplug", allow_final=True)
    commit_all(repo, "plugin")
    git(repo, "tag", "python/fakeplug/v0.1.0rc1")
    notes = release_tool.release_notes(repo, "python/fakeplug", "python/fakeplug/v0.1.0rc1", "temporalio/ai-integrations")
    assert "TestPyPI only" in notes
    assert "still embeds" not in notes and "SDK cutover" not in notes
    assert d.is_dir()


def test_release_notes_ignores_other_plugins_tags(plugin_repo: Path) -> None:
    repo = plugin_repo
    git(repo, "tag", "python/other/v9.0.0")
    git(repo, "tag", "python/fakeplug/v0.1.0")
    assert release_tool.previous_tag(repo, "fakeplug" and "python", "fakeplug", Version("0.1.0")) is None
    notes = release_tool.release_notes(repo, "python/fakeplug", "python/fakeplug/v0.1.0", "temporalio/ai-integrations")
    assert "First standalone release" in notes


def test_json_documents_parses_paginated_output() -> None:
    docs = release_tool._json_documents('[{"a":1}]\n[{"b":2},{"c":3}]')
    assert docs == [[{"a": 1}], [{"b": 2}, {"c": 3}]]
    assert release_tool._json_documents("") == []
