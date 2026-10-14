"""Go release policy and proxy verification without provider or registry credentials."""

from __future__ import annotations

import io
import json
import subprocess
import urllib.error
import zipfile
from pathlib import Path

import pytest

import go_release
import release_tool
from conftest import commit_all, git


@pytest.fixture
def go_repo(repo: Path) -> Path:
    module = repo / "go/fakeplug"
    module.mkdir(parents=True)
    (module / "plugin.toml").write_text('''[plugin]
name = "fakeplug"
language = "go"
coordinate = "go.temporal.io/fakeplug"
registry = "goproxy"
root-api = "fakeplug"
maturity = "public-preview"
[release]
allow-final = false
[ci]
runtime-versions = ["1.26.6", "1.27"]
''')
    (module / "go.mod").write_text("module go.temporal.io/fakeplug\n\ngo 1.26.6\n")
    (module / "go.sum").write_text("")
    (module / "plugin.go").write_text("package fakeplug\n")
    (module / "LICENSE").write_bytes((repo / "LICENSE").read_bytes())
    commit_all(repo, "Add Go plugin")
    return repo


@pytest.mark.parametrize("raw", ["1.0", "01.0.0", "1.0.0rc1", "v1.0.0", "1.0.0+build", "1.0.0-rc.01", "1.0.0-", "1.0.0\n"])
def test_go_tag_rejects_noncanonical_versions(raw: str) -> None:
    with pytest.raises(release_tool.PolicyError):
        release_tool.parse_tag(f"go/fakeplug/v{raw}")


def test_go_semver_precedence() -> None:
    versions = ["0.1.0-alpha", "0.1.0-alpha.1", "0.1.0-alpha.beta", "0.1.0-beta", "0.1.0-beta.2",
                "0.1.0-beta.11", "0.1.0-rc.1", "0.1.0", "0.1.1", "0.10.0", "1.0.0"]
    parsed = [go_release.GoVersion(v) for v in versions]
    assert sorted(reversed(parsed)) == parsed
    assert release_tool.parse_tag("go/fakeplug/v0.1.0-rc.1")["prerelease"] == "true"


def test_go_tag_metadata_does_not_need_a_python_manifest(go_repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert release_tool.main(["--repo-root", str(go_repo), "parse-tag", "go/fakeplug/v0.1.0-rc.1"]) == 0
    outputs = dict(line.split("=", 1) for line in capsys.readouterr().out.splitlines())
    assert outputs["coordinate"] == "go.temporal.io/fakeplug"
    assert outputs["go_runtime"] == "1.27"
    assert outputs["github_prerelease"] == "true"
    assert "requires_python" not in outputs


@pytest.mark.parametrize("change", ["unknown", "module", "registry", "major"])
def test_go_tag_rejects_invalid_metadata(go_repo: Path, change: str) -> None:
    tag = "go/fakeplug/v0.1.0"
    module = go_repo / "go/fakeplug"
    if change == "unknown":
        tag = "go/unknown/v0.1.0"
    elif change == "module":
        (module / "go.mod").write_text("module go.temporal.io/sdk/contrib/fakeplug\n")
    elif change == "registry":
        manifest = module / "plugin.toml"
        manifest.write_text(manifest.read_text().replace('registry = "goproxy"', 'registry = "pypi"'))
    else:
        tag = "go/fakeplug/v2.0.0"
    assert release_tool.main(["--repo-root", str(go_repo), "parse-tag", tag]) == 1


def test_go_registry_ordering_and_first_release_after_tagging(go_repo: Path, tmp_path: Path) -> None:
    registry = tmp_path / "versions.txt"
    args = ["--repo-root", str(go_repo), "check-version-policy", "--plugin-dir", str(go_repo / "go/fakeplug"),
            "--registry-json", str(registry)]
    registry.write_text("")
    assert release_tool.main([*args, "--version", "0.1.0-rc.1"]) == 0
    assert release_tool.main([*args, "--version", "0.2.0-rc.1"]) == 1
    registry.write_text("v0.2.0-rc.1\n")
    assert release_tool.main([*args, "--version", "0.2.0-rc.1"]) == 1
    registry.write_text("v0.1.0-rc.2\nv0.1.0-rc.11\n")
    assert release_tool.main([*args, "--version", "0.1.0-rc.3"]) == 1
    assert release_tool.main([*args, "--version", "0.1.0-rc.11"]) == 0
    assert release_tool.main([*args, "--version", "0.1.0-rc.12"]) == 0
    assert release_tool.main([*args, "--version", "0.1.0"]) == 1  # allow-final gate
    manifest = go_repo / "go/fakeplug/plugin.toml"
    manifest.write_text(manifest.read_text().replace("allow-final = false", "allow-final = true"))
    assert release_tool.main([*args, "--version", "0.1.0"]) == 0
    (go_repo / "go/fakeplug/plugin.go").write_text("// TRANSITION(sdk-cutover)\npackage fakeplug\n")
    assert release_tool.main([*args, "--version", "0.1.0"]) == 1


@pytest.mark.parametrize("code,missing", [(404, True), (410, True), (403, False), (500, False)])
def test_go_proxy_errors_fail_closed(monkeypatch: pytest.MonkeyPatch, code: int, missing: bool) -> None:
    def response(url: str, **_kwargs: object) -> None:
        raise urllib.error.HTTPError(url, code, "error", None, None)

    monkeypatch.setattr(go_release.urllib.request, "urlopen", response)
    if missing:
        assert go_release.fetch_versions("go.temporal.io/fakeplug") == []
    else:
        with pytest.raises(go_release.GoReleaseError, match=f"HTTP {code}"):
            go_release.fetch_versions("go.temporal.io/fakeplug")


@pytest.mark.parametrize("record,accepted", [
    ("go.temporal.io/fakeplug git https://github.com/temporalio/ai-integrations go/fakeplug", True),
    ("go.temporal.io/fakeplug git https://github.com/temporalio/ai-integrations.git go/fakeplug", True),
    ("go.temporal.io/fakeplug git https://github.com/temporalio/sdk-go contrib/fakeplug", False),
    ("go.temporal.io/fakeplug git https://github.com/temporalio/ai-integrations", False),
    ("go.temporal.io/fakeplug git https://github.com/temporalio/ai-integrations go/other", False),
])
def test_vanity_requires_the_exact_repository_and_subdirectory(monkeypatch: pytest.MonkeyPatch, record: str, accepted: bool) -> None:
    monkeypatch.setattr(go_release.urllib.request, "urlopen", lambda *_args, **_kwargs: io.BytesIO(f'<meta name="go-import" content="{record}">'.encode()))
    if accepted:
        go_release.check_vanity("go.temporal.io/fakeplug", "temporalio/ai-integrations", "go/fakeplug")
    else:
        with pytest.raises(go_release.GoReleaseError, match="must serve go-import"):
            go_release.check_vanity("go.temporal.io/fakeplug", "temporalio/ai-integrations", "go/fakeplug")


def module_zip(repo: Path, archive: Path, change: str = "") -> Path:
    module = repo / "go/fakeplug"
    with zipfile.ZipFile(archive, "w") as z:
        for source in sorted(module.iterdir()):
            if source.name == "plugin.go" and change == "missing":
                continue
            content = source.read_bytes()
            if source.name == "plugin.go" and change == "bytes":
                content += b"// different source\n"
            z.writestr(f"go.temporal.io/fakeplug@v0.1.0-rc.1/{source.name}", content)
        if change == "extra":
            z.writestr("go.temporal.io/fakeplug@v0.1.0-rc.1/extra.go", "package fakeplug\n")
    return archive


@pytest.mark.parametrize("change", ["", "missing", "extra", "bytes"])
def test_proxy_archive_must_be_the_tested_file_set_and_bytes(go_repo: Path, tmp_path: Path, change: str) -> None:
    archive = module_zip(go_repo, tmp_path / "module.zip", change)
    if change:
        with pytest.raises(go_release.GoReleaseError, match="differs"):
            go_release.verify_zip(go_repo, "go/fakeplug", "go.temporal.io/fakeplug", "0.1.0-rc.1", archive)
    else:
        go_release.verify_zip(go_repo, "go/fakeplug", "go.temporal.io/fakeplug", "0.1.0-rc.1", archive)


def test_proxy_archive_excludes_nested_modules_and_vendor_directories(go_repo: Path, tmp_path: Path) -> None:
    module = go_repo / "go/fakeplug"
    archive = module_zip(go_repo, tmp_path / "module.zip")
    for name, content in [("vendor/example/example.go", "package example\n"), ("tools/go.mod", "module example.com/tools\n"),
                          ("tools/tool.go", "package tools\n"), ("docs/vendor", "A regular file named vendor.\n")]:
        target = module / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    commit_all(go_repo, "Add vendored and nested modules")
    with zipfile.ZipFile(archive, "a") as z:
        z.writestr("go.temporal.io/fakeplug@v0.1.0-rc.1/docs/vendor", (module / "docs/vendor").read_bytes())
    go_release.verify_zip(go_repo, "go/fakeplug", "go.temporal.io/fakeplug", "0.1.0-rc.1", archive)


@pytest.mark.parametrize("invalid", ["version", "path", "checksum", "error"])
def test_verification_rejects_wrong_resolutions_and_unverified_downloads(go_repo: Path, monkeypatch: pytest.MonkeyPatch, invalid: str) -> None:
    data = {"Path": "go.temporal.io/fakeplug", "Version": "v0.1.0-rc.1", "Sum": "h1:zip", "GoModSum": "h1:mod"}
    if invalid == "version":
        data["Version"] = "v0.1.0-rc.2"
    elif invalid == "path":
        data["Path"] = "go.temporal.io/other"
    elif invalid == "checksum":
        del data["Sum"]
    else:
        data["Error"] = "module unavailable"
    monkeypatch.setattr(go_release.subprocess, "run", lambda command, **_kwargs: subprocess.CompletedProcess(command, 0, stdout=json.dumps(data)))
    with pytest.raises(go_release.GoReleaseError):
        go_release.verify_module(go_repo, "go/fakeplug", "go.temporal.io/fakeplug", "0.1.0-rc.1", attempts=1, delay=0)


def test_verification_uses_public_checksums_and_builds_a_clean_consumer(go_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive = module_zip(go_repo, tmp_path / "module.zip")
    calls = []
    real_run = subprocess.run

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        if command[0] == "git":
            return real_run(command, **kwargs)
        calls.append(command)
        env = kwargs["env"]
        assert env["GOPROXY"] == "https://proxy.golang.org" and env["GOSUMDB"] == "sum.golang.org"
        assert all(env[key] == "" for key in ("GOPRIVATE", "GONOPROXY", "GONOSUMDB", "GOFLAGS"))
        assert env["GOWORK"] == "off" and env["GOTOOLCHAIN"] == "local"
        # Cached dependencies without go.mod must not become consumer packages.
        assert not Path(env["GOMODCACHE"]).is_relative_to(kwargs["cwd"])
        data = {"Path": "go.temporal.io/fakeplug", "Version": "v0.1.0-rc.1", "Sum": "h1:zip", "GoModSum": "h1:mod",
                "Zip": str(archive), "GoMod": str(go_repo / "go/fakeplug/go.mod")}
        if command[1:3] == ["get", "go.temporal.io/fakeplug@v0.1.0-rc.1"]:
            assert 'import _ "go.temporal.io/fakeplug"' in (kwargs["cwd"] / "main.go").read_text()
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(data), stderr="")

    monkeypatch.setattr(go_release.subprocess, "run", run)
    go_release.verify_module(go_repo, "go/fakeplug", "go.temporal.io/fakeplug", "0.1.0-rc.1", attempts=1, delay=0)
    assert [command[1] for command in calls] == ["mod", "mod", "get", "mod", "mod", "build"]
    assert calls[-1] == ["go", "build", "-mod=readonly", "./..."]


def test_go_release_notes_use_semver_boundaries_and_go_installation(go_repo: Path) -> None:
    git(go_repo, "tag", "go/fakeplug/v0.1.0-rc.2")
    (go_repo / "go/fakeplug/plugin.go").write_text("package fakeplug\n// Fix\n")
    commit_all(go_repo, "Fix Go plugin")
    git(go_repo, "tag", "go/fakeplug/v0.1.0-rc.11")
    notes = release_tool.release_notes(go_repo, "go/fakeplug", "go/fakeplug/v0.1.0-rc.11", "temporalio/ai-integrations")
    assert "go get go.temporal.io/fakeplug@v0.1.0-rc.11" in notes
    assert "go/fakeplug/v0.1.0-rc.2...go/fakeplug/v0.1.0-rc.11" in notes
    assert "TestPyPI" not in notes and "SDK cutover" not in notes
    git(go_repo, "tag", "go/fakeplug/v0.1.0")
    final = release_tool.release_notes(go_repo, "go/fakeplug", "go/fakeplug/v0.1.0", "temporalio/ai-integrations")
    assert "First standalone release" in final and "Fix Go plugin" in final and "Add Go plugin" in final
