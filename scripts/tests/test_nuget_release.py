"""NuGet release ordering, tested-package identity, and fail-closed verification against nuget.org."""

from __future__ import annotations

import http.client
import io
import json
import os
import subprocess
import urllib.error
import zipfile
from pathlib import Path

import nuget_release
import pytest
import release_tool
import yaml
from conftest import commit_all, git

COORDINATE = "Temporalio.Extensions.Fakedotnet"
VERSION = "0.0.1-rc.1"
PLUGIN_DIR = "dotnet/fakedotnet"
WORKFLOW = Path(__file__).resolve().parents[2] / ".github/workflows/release-dotnet.yml"


@pytest.mark.parametrize("version", [
    "0.0.1rc1", "01.0.0", "1.0.0+build", "1.0.0.0", "0.0.1-rc.01", "v0.0.1", "0.1", "0.1.0-", "0.1.0\n", " 0.1.0", "0.1.0 ",
    "0.0.1-RC.1", "0.0.1-Rc.1", "0.0.1-rc.1+SHA", "0.0.1-rc1", "0.0.1-rc", "0.0.1-rc.1a", "0.0.1-beta1.2",
])
def test_nuget_rejects_noncanonical_versions(version: str) -> None:
    with pytest.raises(nuget_release.NuGetReleaseError, match="NuGet version"):
        nuget_release.NuGetVersion(version)
    with pytest.raises(release_tool.PolicyError):
        release_tool.parse_tag(f"dotnet/microsoft-agent-framework/v{version}")


def test_uppercase_labels_never_reach_nuget_org() -> None:
    # nuget.org would serve 0.0.1-rc.1 for an uploaded 0.0.1-RC.1 and the flat container membership test would
    # never succeed; rejecting the tag keeps the policy and the verification in one case.
    with pytest.raises(nuget_release.NuGetReleaseError, match="must be lowercase"):
        nuget_release.NuGetVersion("0.0.1-RC.1")
    assert nuget_release.NuGetVersion("0.0.1-rc.1") in [nuget_release.NuGetVersion("0.0.1-rc.1")]


def test_dotted_prerelease_labels_order_numerically() -> None:
    versions = [nuget_release.NuGetVersion(raw) for raw in ("0.0.1-alpha.2", "0.0.1-beta.1", "0.0.1-rc.9", "0.0.1-rc.10", "0.0.1")]
    assert sorted(reversed(versions)) == versions
    assert release_tool.check_policy(versions[3], "pre-release", versions[:3]) is None
    assert nuget_release.NuGetVersion("0.0.1-preview.1").is_prerelease


def test_dotnet_tags_parse_with_the_prerelease_flag() -> None:
    assert release_tool.parse_tag("dotnet/microsoft-agent-framework/v0.0.1-rc.1") == {
        "tag": "dotnet/microsoft-agent-framework/v0.0.1-rc.1", "language": "dotnet", "plugin": "microsoft-agent-framework",
        "version": "0.0.1-rc.1", "prerelease": "true", "plugin_dir": "dotnet/microsoft-agent-framework",
    }
    assert release_tool.parse_tag("dotnet/microsoft-agent-framework/v0.0.1")["prerelease"] == "false"


# dotnet_runtime is the last runtime-versions entry, the SDK channel global.json pins: release-dotnet.yml installs
# only that SDK in the publish and verify jobs and the clean consumer targets its framework.
def test_cli_parse_dotnet_tag_emits_the_sdk_channel(dotnet_repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert release_tool.main(["--repo-root", str(dotnet_repo), "parse-tag", f"dotnet/fakedotnet/v{VERSION}"]) == 0
    outputs = dict(line.split("=", 1) for line in capsys.readouterr().out.splitlines())
    assert outputs["coordinate"] == COORDINATE and outputs["root_api"] == COORDINATE
    assert outputs["dotnet_runtime"] == "10.0"
    assert outputs["smoke_imports"] == f"{COORDINATE}.FakedotnetPlugin"
    assert outputs["github_prerelease"] == "true" and outputs["maturity"] == "pre-release"
    assert "requires_python" not in outputs and "dist_tag" not in outputs and "go_runtime" not in outputs
    assert release_tool.main(["--repo-root", str(dotnet_repo), "parse-tag", "dotnet/unknown/v0.0.1"]) == 1
    assert "unknown .NET plugin" in capsys.readouterr().out


@pytest.mark.parametrize("old,new", [
    ('registry = "nuget"', 'registry = "npm"'),
    ('coordinate = "Temporalio.Extensions.Fakedotnet"', 'coordinate = "Temporalio.Other"'),
    ('root-api = "Temporalio.Extensions.Fakedotnet"', 'root-api = "Temporalio.Other"'),
    ('language = "dotnet"', 'language = "go"'),
    ('name = "fakedotnet"', 'name = "other"'),
])
def test_cli_parse_dotnet_tag_rejects_inconsistent_metadata(dotnet_repo: Path, old: str, new: str) -> None:
    meta = dotnet_repo / PLUGIN_DIR / "plugin.toml"
    meta.write_text(meta.read_text().replace(old, new))
    assert release_tool.main(["--repo-root", str(dotnet_repo), "parse-tag", f"dotnet/fakedotnet/v{VERSION}"]) == 1


def test_flat_container_order_includes_unlisted_versions(tmp_path: Path) -> None:
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"versions": ["0.0.1-rc.1", "0.0.1-rc.2", "0.0.1", "0.0.2-rc.10", "0.0.2-rc.9"]}))
    published = release_tool.fetch_published_versions(COORDINATE, "nuget", index)
    assert [str(v) for v in sorted(published)] == ["0.0.1-rc.1", "0.0.1-rc.2", "0.0.1", "0.0.2-rc.9", "0.0.2-rc.10"]
    with pytest.raises(release_tool.PolicyError, match="newest published"):
        release_tool.check_policy(nuget_release.NuGetVersion("0.0.2-rc.3"), "pre-release", published)
    assert "re-run" in (release_tool.check_policy(nuget_release.NuGetVersion("0.0.2-rc.10"), "pre-release", published) or "")
    assert release_tool.check_policy(nuget_release.NuGetVersion("0.0.2"), "pre-release", published) is None
    assert release_tool.check_policy(nuget_release.NuGetVersion("0.0.1-rc.1"), "pre-release", []) is None
    with pytest.raises(release_tool.PolicyError, match="first release"):
        release_tool.check_policy(nuget_release.NuGetVersion("0.1.0"), "pre-release", [])
    assert release_tool.fetch_published_versions(COORDINATE, "nuget", tmp_path / "absent.json") == []


def test_noncanonical_index_entries_under_our_coordinate_raise(tmp_path: Path) -> None:
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"versions": ["1.0.0.1"]}))
    with pytest.raises(release_tool.PolicyError, match="NuGet version"):
        release_tool.fetch_published_versions(COORDINATE, "nuget", index)
    index.write_text(json.dumps({"items": []}))
    with pytest.raises(release_tool.PolicyError, match="invalid flat container index"):
        release_tool.fetch_published_versions(COORDINATE, "nuget", index)


def test_cli_version_policy_for_a_dotnet_plugin(dotnet_repo: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    index = tmp_path / "index.json"
    args = ["--repo-root", str(dotnet_repo), "check-version-policy", "--plugin-dir", str(dotnet_repo / PLUGIN_DIR),
            "--registry-json", str(index)]
    assert release_tool.main([*args, "--version", "0.0.1-rc.1"]) == 0
    assert release_tool.main([*args, "--version", "0.1.0"]) == 1
    index.write_text(json.dumps({"versions": ["0.0.1-rc.1"]}))
    assert release_tool.main([*args, "--version", "0.0.1-rc.1"]) == 0
    assert "::warning::" in capsys.readouterr().out
    assert release_tool.main([*args, "--version", "0.0.1"]) == 0
    assert release_tool.main([*args, "--version", "0.0.1-rc.0"]) == 1


@pytest.mark.parametrize("status", [401, 403, 500])
def test_registry_errors_are_never_treated_as_unpublished(monkeypatch: pytest.MonkeyPatch, status: int) -> None:
    def fail(*args, **kwargs):
        raise urllib.error.HTTPError("https://api.nuget.org", status, "failed", {}, None)
    monkeypatch.setattr(nuget_release.urllib.request, "urlopen", fail)
    with pytest.raises(nuget_release.NuGetReleaseError, match=f"HTTP {status}"):
        nuget_release.fetch_versions(COORDINATE)


def test_only_registry_404_means_new_coordinate(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(*args, **kwargs):
        raise urllib.error.HTTPError("https://api.nuget.org", 404, "missing", {}, None)
    monkeypatch.setattr(nuget_release.urllib.request, "urlopen", missing)
    assert nuget_release.fetch_versions(COORDINATE) == []


def nupkg(*, version: str = VERSION, package_id: str = COORDINATE, dll: bytes = b"MZ", signature: bytes | None = None,
          stored: bool = True, signature_first: bool = False, extra: bytes | None = None) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        def sign() -> None:
            info = zipfile.ZipInfo(".signature.p7s")
            info.compress_type = zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED
            archive.writestr(info, signature or b"")

        if signature is not None and signature_first:
            sign()
        archive.writestr(f"{package_id}.nuspec", (
            '<package xmlns="http://schemas.microsoft.com/packaging/2013/05/nuspec.xsd"><metadata>'
            f"<id>{package_id}</id><version>{version}</version></metadata></package>"))
        archive.writestr(f"lib/net10.0/{COORDINATE}.dll", dll)
        if extra is not None:
            archive.writestr("extra/README.txt", extra)
        if signature is not None and not signature_first:
            sign()
    return buffer.getvalue()


def dist_with(tmp_path: Path, payload: bytes | None = None, name: str = f"{COORDINATE}.{VERSION}.nupkg") -> Path:
    dist = tmp_path / "dist"
    dist.mkdir(parents=True, exist_ok=True)
    (dist / name).write_bytes(payload or nupkg())
    return dist


@pytest.mark.parametrize("fault", ["version", "id", "name", "two"])
def test_wrong_tested_package_cannot_be_published(tmp_path: Path, fault: str) -> None:
    if fault == "version":
        dist = dist_with(tmp_path, nupkg(version="0.0.0"))
    elif fault == "id":
        dist = dist_with(tmp_path, nupkg(package_id="Temporalio.Other"))
    elif fault == "name":
        dist = dist_with(tmp_path, name="other.nupkg")
    else:
        dist = dist_with(tmp_path)
        (dist / "second.nupkg").write_bytes(nupkg())
    with pytest.raises(nuget_release.NuGetReleaseError, match="does not match|exactly one"):
        nuget_release.artifact(dist, COORDINATE, VERSION)
    assert nuget_release.artifact(dist_with(tmp_path / "ok"), COORDINATE, VERSION).name == f"{COORDINATE}.{VERSION}.nupkg"


def test_unsigned_entries_strip_only_the_trailing_stored_signature() -> None:
    plain = nuget_release.unsigned_entries(nupkg())
    assert [name for name, *_ in plain] == [f"{COORDINATE}.nuspec", f"lib/net10.0/{COORDINATE}.dll"]
    assert nuget_release.served_matches(nupkg(signature=b"repository signature"), nupkg())
    assert not nuget_release.served_matches(nupkg(dll=b"MZ-other", signature=b"sig"), nupkg())
    # A served package with one entry more than the tested one is not the tested package either.
    assert not nuget_release.served_matches(nupkg(extra=b"added by someone", signature=b"sig"), nupkg())
    assert not nuget_release.served_matches(nupkg(signature=b"sig"), nupkg(extra=b"in the tested package"))
    for payload in (nupkg(signature=b"sig", stored=False), nupkg(signature=b"sig", signature_first=True)):
        # Permanent: a signature in the wrong place or compressed never changes on retry.
        with pytest.raises(nuget_release.NuGetMismatchError, match="trailing stored repository signature"):
            nuget_release.unsigned_entries(payload)


def fake_nuget(monkeypatch: pytest.MonkeyPatch, *, index: list[str] | None, served: bytes | None, leaf: dict | None) -> list[str]:
    """Route nuget_release._get by URL: the flat index, the served nupkg and the registration leaf."""
    urls: list[str] = []

    def get(url: str) -> bytes | None:
        urls.append(url)
        if url.endswith("/index.json"):
            return None if index is None else json.dumps({"versions": index}).encode()
        if url.endswith(".nupkg"):
            return served
        if url.endswith(".json"):
            return None if leaf is None else json.dumps(leaf).encode()
        raise AssertionError(url)

    monkeypatch.setattr(nuget_release, "_get", get)
    return urls


def test_registry_state_is_false_until_the_served_package_exists(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dist = dist_with(tmp_path)
    fake_nuget(monkeypatch, index=None, served=None, leaf=None)
    assert nuget_release.registry_state(COORDINATE, VERSION, dist) is False
    fake_nuget(monkeypatch, index=["0.0.1-rc.1"], served=None, leaf=None)
    assert nuget_release.registry_state(COORDINATE, VERSION, dist) is False
    urls = fake_nuget(monkeypatch, index=["0.0.1-rc.1"], served=nupkg(signature=b"sig"), leaf={"listed": True})
    assert nuget_release.registry_state(COORDINATE, VERSION, dist) is True
    assert urls == [
        "https://api.nuget.org/v3-flatcontainer/temporalio.extensions.fakedotnet/index.json",
        "https://api.nuget.org/v3-flatcontainer/temporalio.extensions.fakedotnet/0.0.1-rc.1/temporalio.extensions.fakedotnet.0.0.1-rc.1.nupkg",
        "https://api.nuget.org/v3/registration5-gz-semver2/temporalio.extensions.fakedotnet/0.0.1-rc.1.json",
    ]
    fake_nuget(monkeypatch, index=["0.0.1-rc.1"], served=nupkg(signature=b"sig"), leaf={})
    assert nuget_release.registry_state(COORDINATE, VERSION, dist) is True


def test_registry_state_mismatch_and_unlisted_are_final(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dist = dist_with(tmp_path)
    fake_nuget(monkeypatch, index=["0.0.1-rc.1"], served=nupkg(dll=b"MZ-other", signature=b"sig"), leaf={"listed": True})
    with pytest.raises(nuget_release.NuGetMismatchError, match="different package entries"):
        nuget_release.registry_state(COORDINATE, VERSION, dist)
    fake_nuget(monkeypatch, index=["0.0.1-rc.1"], served=nupkg(signature=b"sig"), leaf={"listed": False})
    with pytest.raises(nuget_release.NuGetMismatchError, match="unlisted"):
        nuget_release.registry_state(COORDINATE, VERSION, dist)
    # The registration leaf lags the flat container for a moment after publication; that is retried, not final.
    fake_nuget(monkeypatch, index=["0.0.1-rc.1"], served=nupkg(signature=b"sig"), leaf=None)
    with pytest.raises(nuget_release.NuGetReleaseError, match="no registration leaf") as excinfo:
        nuget_release.registry_state(COORDINATE, VERSION, dist)
    assert not isinstance(excinfo.value, nuget_release.NuGetMismatchError)


def test_cli_registry_state_outputs_and_retries_propagation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dist = dist_with(tmp_path)
    monkeypatch.setattr(release_tool, "nuget_registry_state", lambda *args: False)
    output = tmp_path / "output"
    args = ["nuget-registry-state", "--coordinate", COORDINATE, "--version", VERSION, "--dist", str(dist), "--github-output", str(output)]
    assert release_tool.main(args) == 0
    assert output.read_text() == "exists=false\n"
    assert release_tool.main([*args, "--require-present"]) == 1
    responses = iter([False, True])
    monkeypatch.setattr(release_tool, "nuget_registry_state", lambda *args: next(responses))
    output.unlink()
    assert release_tool.main([*args, "--require-present", "--attempts", "2", "--delay", "0"]) == 0
    assert output.read_text() == "exists=true\n"


@pytest.mark.parametrize("transient", [ConnectionResetError("reset by peer"), http.client.IncompleteRead(b"half"), TimeoutError("slow")])
def test_cli_registry_state_retries_connection_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, transient: Exception) -> None:
    dist = dist_with(tmp_path)
    responses = iter([transient, True])

    def flaky(*args):
        value = next(responses)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(release_tool, "nuget_registry_state", flaky)
    output = tmp_path / "output"
    args = ["nuget-registry-state", "--coordinate", COORDINATE, "--version", VERSION, "--dist", str(dist),
            "--github-output", str(output), "--require-present", "--attempts", "2", "--delay", "0"]
    assert release_tool.main(args) == 0
    assert output.read_text() == "exists=true\n"


def test_cli_registry_state_never_retries_a_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    calls = []

    def mismatch(*args):
        calls.append(args)
        raise nuget_release.NuGetMismatchError("nuget.org serves different package entries")

    monkeypatch.setattr(release_tool, "nuget_registry_state", mismatch)
    assert release_tool.main(["nuget-registry-state", "--coordinate", COORDINATE, "--version", VERSION, "--dist", str(tmp_path),
                              "--require-present", "--attempts", "5", "--delay", "0"]) == 1
    assert len(calls) == 1
    assert "different package entries" in capsys.readouterr().out


def test_dotnet_prerelease_notes_name_the_explicit_version(dotnet_repo: Path) -> None:
    git(dotnet_repo, "tag", f"dotnet/fakedotnet/v{VERSION}")
    notes = release_tool.release_notes(dotnet_repo, PLUGIN_DIR, f"dotnet/fakedotnet/v{VERSION}", "temporalio/ai-integrations")
    assert f"dotnet add package {COORDINATE} --version {VERSION}" in notes
    assert "TestPyPI" not in notes and "First standalone release of `Temporalio.Extensions.Fakedotnet`" in notes
    git(dotnet_repo, "tag", "dotnet/fakedotnet/v0.0.1")
    final = release_tool.release_notes(dotnet_repo, PLUGIN_DIR, "dotnet/fakedotnet/v0.0.1", "temporalio/ai-integrations")
    assert "Pre-release notes" not in final


# --------------------------------------------------------------------------- recovery


@pytest.fixture
def nuget_recovery_source(dotnet_repo: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, dict, dict, dict]:
    sha = git(dotnet_repo, "rev-parse", "HEAD")
    git(dotnet_repo, "tag", f"dotnet/fakedotnet/v{VERSION}")
    git(dotnet_repo, "update-ref", "refs/remotes/origin/main", sha)
    run = {"repository": {"full_name": "temporalio/ai-integrations"}, "event": "workflow_dispatch", "head_branch": "main",
           "head_sha": sha, "path": ".github/workflows/release-dotnet.yml", "status": "completed"}
    names = ["Validate .NET tag and version policy", "Test / matrix"]
    names.extend(f"Test / fakedotnet ({runner}, dotnet10.0)" for runner in ("ubuntu-latest", "macos-latest", "windows-latest"))
    jobs = {"jobs": [{"name": name, "status": "completed", "conclusion": "success"} for name in names]}
    artifacts = {"artifacts": [{"name": "dist-dotnet-fakedotnet", "expired": False}]}

    def gh(*args: str) -> str:
        endpoint = args[1]
        if "/jobs?" in endpoint:
            return json.dumps([jobs])
        if "/artifacts?" in endpoint:
            return json.dumps(artifacts)
        return json.dumps(run)

    monkeypatch.setattr(release_tool, "_gh", gh)
    return dotnet_repo, run, jobs, artifacts


@pytest.mark.parametrize("event,branch", [
    ("workflow_dispatch", "main"), ("workflow_dispatch", f"dotnet/fakedotnet/v{VERSION}"), ("release", f"dotnet/fakedotnet/v{VERSION}"),
])
def test_nuget_recovery_reuses_tested_dry_run_or_release_artifacts(nuget_recovery_source: tuple, event: str, branch: str) -> None:
    repo, run, *_ = nuget_recovery_source
    run.update(event=event, head_branch=branch)
    output = repo / "outputs"
    assert release_tool.main(["--repo-root", str(repo), "check-recovery-run", "--repo", "temporalio/ai-integrations",
                              "--tag", f"dotnet/fakedotnet/v{VERSION}", "--run-id", "123", "--github-output", str(output)]) == 0
    assert output.read_text() == "recovery=true\nartifact_run_id=123\n"


@pytest.mark.parametrize("key,value", [
    ("event", "pull_request"), ("event", "push"), ("head_branch", "feature"), ("head_sha", "different-commit"),
    ("path", ".github/workflows/release-typescript.yml"), ("status", "in_progress"), ("repository", {"full_name": "other/repo"}),
])
def test_nuget_recovery_rejects_untrusted_or_unfinished_sources(nuget_recovery_source: tuple, key: str, value: object) -> None:
    repo, run, *_ = nuget_recovery_source
    run[key] = value
    with pytest.raises(release_tool.PolicyError, match="completed release-dotnet.yml"):
        release_tool.check_recovery_run(repo, "temporalio/ai-integrations", f"dotnet/fakedotnet/v{VERSION}", 123)


@pytest.mark.parametrize("name", [
    "Validate .NET tag and version policy", "Test / matrix", "Test / fakedotnet (ubuntu-latest, dotnet10.0)",
    "Test / fakedotnet (macos-latest, dotnet10.0)", "Test / fakedotnet (windows-latest, dotnet10.0)",
])
@pytest.mark.parametrize("state", ["missing", "failure", "in_progress"])
def test_nuget_recovery_requires_the_complete_successful_matrix(nuget_recovery_source: tuple, name: str, state: str) -> None:
    repo, _, jobs, _ = nuget_recovery_source
    job = next(job for job in jobs["jobs"] if job["name"] == name)
    if state == "missing":
        jobs["jobs"].remove(job)
    elif state == "in_progress":
        job["status"] = state
    else:
        job["conclusion"] = state
    with pytest.raises(release_tool.PolicyError, match="no successful completed job"):
        release_tool.check_recovery_run(repo, "temporalio/ai-integrations", f"dotnet/fakedotnet/v{VERSION}", 123)


@pytest.mark.parametrize("state", ["missing", "expired", "wrong_name", "duplicate"])
def test_nuget_recovery_requires_one_available_package_artifact(nuget_recovery_source: tuple, state: str) -> None:
    repo, _, _, artifacts = nuget_recovery_source
    if state == "missing":
        artifacts["artifacts"] = []
    elif state == "expired":
        artifacts["artifacts"][0]["expired"] = True
    elif state == "wrong_name":
        artifacts["artifacts"][0]["name"] = "dist-dotnet-other"
    else:
        artifacts["artifacts"].append(dict(artifacts["artifacts"][0]))
    with pytest.raises(release_tool.PolicyError, match="exactly one unexpired dist-dotnet-fakedotnet"):
        release_tool.check_recovery_run(repo, "temporalio/ai-integrations", f"dotnet/fakedotnet/v{VERSION}", 123)


@pytest.mark.parametrize("filename", [
    "plugin.toml", "global.json", "Directory.Build.props", "nuget.config",
    "src/Temporalio.Extensions.Fakedotnet/Temporalio.Extensions.Fakedotnet.csproj", "tests/Temporalio.Extensions.Fakedotnet.Tests/packages.lock.json",
])
def test_nuget_recovery_rejects_changed_package_shaping_files(nuget_recovery_source: tuple, filename: str) -> None:
    repo, *_ = nuget_recovery_source
    path = repo / PLUGIN_DIR / filename
    path.write_text(path.read_text() + "\n")
    with pytest.raises(release_tool.PolicyError, match="match the release tag"):
        release_tool.check_recovery_run(repo, "temporalio/ai-integrations", f"dotnet/fakedotnet/v{VERSION}", 123)


def test_nuget_recovery_ignores_files_that_do_not_shape_the_package(nuget_recovery_source: tuple) -> None:
    repo, *_ = nuget_recovery_source
    (repo / PLUGIN_DIR / "src/Temporalio.Extensions.Fakedotnet/FakedotnetPlugin.cs").write_text("// changed after the tag\n")
    release_tool.check_recovery_run(repo, "temporalio/ai-integrations", f"dotnet/fakedotnet/v{VERSION}", 123)


def test_nuget_recovery_rejects_tags_outside_main(nuget_recovery_source: tuple) -> None:
    repo, *_ = nuget_recovery_source
    (repo / "README.md").write_text("unmerged change")
    commit_all(repo, "Unmerged feature")
    git(repo, "tag", "dotnet/fakedotnet/v0.0.1-rc.2")
    with pytest.raises(release_tool.PolicyError, match="reachable from origin/main"):
        release_tool.check_recovery_run(repo, "temporalio/ai-integrations", "dotnet/fakedotnet/v0.0.1-rc.2", 123)


# --------------------------------------------------------------------------- release-dotnet.yml


@pytest.mark.parametrize("ref_type,skip,expected", [
    ("branch", "false", "false"), ("branch", "true", "false"),
    ("tag", "true", "false"), ("tag", "", "false"), ("tag", "false", "true"),
])
def test_only_tagged_non_dry_runs_publish(tmp_path: Path, ref_type: str, skip: str, expected: str) -> None:
    workflow = yaml.safe_load(WORKFLOW.read_text())
    gate = next(step for step in workflow["jobs"]["prepare"]["steps"] if step.get("id") == "gate")
    output = tmp_path / "output"
    env = dict(os.environ, REF_TYPE=ref_type, SKIP_PUBLISH=skip, GITHUB_OUTPUT=str(output))
    result = subprocess.run(["bash", "-c", gate["run"]], env=env, cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert output.read_text() == f"publish={expected}\n"


def test_nuget_publishes_only_the_full_matrix_artifact_with_oidc() -> None:
    text = WORKFLOW.read_text()
    workflow = yaml.safe_load(text)
    assert workflow["permissions"] == {}
    assert "push" not in workflow[True]
    assert workflow[True]["release"]["types"] == ["published"]
    assert "secrets." not in text, "trusted publishing needs no stored token; NUGET_USER is a public profile name"
    prepare = workflow["jobs"]["prepare"]
    assert prepare["if"] == "github.event_name != 'release' || startsWith(github.event.release.tag_name, 'dotnet/')"
    assert prepare["permissions"]["actions"] == "read"
    assert prepare["outputs"]["artifact_run_id"] == "${{ steps.recovery.outputs.artifact_run_id || github.run_id }}"
    assert prepare["outputs"]["dotnet_runtime"] == "${{ steps.tag.outputs.dotnet_runtime }}"
    recovery = next(step for step in prepare["steps"] if step.get("id") == "recovery")
    assert recovery["if"] == "github.event_name == 'workflow_dispatch' && inputs.recover-run != ''"
    assert 'check-recovery-run --tag "$TAG" --run-id "$RUN_ID"' in recovery["run"]
    test = workflow["jobs"]["test"]
    assert test["uses"] == "./.github/workflows/_dotnet-plugin.yml"
    assert test["with"] == {"plugin": "${{ needs.prepare.outputs.plugin }}", "deps": "locked",
                            "ref": "${{ needs.prepare.outputs.sha }}", "version": "${{ needs.prepare.outputs.version }}"}
    assert test["if"] == "needs.prepare.outputs.recovery != 'true'"
    # Three jobs: the registry check and the consumer smoke run without a token; only the push job has one.
    existing = workflow["jobs"]["existing"]
    assert existing["needs"] == ["prepare", "test"]
    assert "!cancelled() && needs.prepare.result == 'success'" in existing["if"]
    assert "needs.prepare.outputs.publish == 'true'" in existing["if"]
    assert "needs.test.result == 'success' || needs.prepare.outputs.recovery == 'true'" in existing["if"]
    assert existing["permissions"] == {"contents": "read", "actions": "read"}
    check = next(step for step in existing["steps"] if "nuget-registry-state" in step.get("run", ""))
    assert "--require-present" not in check["run"]
    publish = workflow["jobs"]["publish"]
    assert publish["needs"] == ["prepare", "test", "existing"]
    assert "needs.existing.result == 'success'" in publish["if"] and "needs.prepare.outputs.publish == 'true'" in publish["if"]
    assert publish["permissions"] == {"id-token": "write", "actions": "read"}
    assert publish["environment"] == {"name": "nuget", "url": "${{ needs.prepare.outputs.release_url }}"}
    steps = publish["steps"]
    assert not any("checkout" in step.get("uses", "") for step in steps), "the OIDC job downloads the artifact only"
    assert not any(step.get("run") for step in steps if "dotnet nuget push" not in step.get("run", "")), "the OIDC job runs nothing but the push"
    setup = next(step for step in steps if "setup-dotnet@" in step.get("uses", ""))
    assert setup["with"] == {"dotnet-version": "${{ needs.prepare.outputs.dotnet_runtime }}.x"}
    download = next(step for step in steps if "download-artifact@" in step.get("uses", ""))
    assert download["with"]["name"] == "dist-dotnet-${{ needs.prepare.outputs.plugin }}"
    assert download["with"]["run-id"] == "${{ needs.prepare.outputs.artifact_run_id }}"
    assert download["with"]["github-token"] == "${{ github.token }}"
    login = next(step for step in steps if "NuGet/login@" in step.get("uses", ""))
    # Both pushes run on every attempt (--skip-duplicate), so a re-run after a partial upload completes the symbols.
    assert login["id"] == "login" and "if" not in login
    assert login["with"] == {"user": "${{ vars.NUGET_USER }}"}
    push = next(step for step in steps if "dotnet nuget push" in step.get("run", ""))
    assert "if" not in push
    assert push["env"] == {"NUGET_API_KEY": "${{ steps.login.outputs.NUGET_API_KEY }}"}
    assert '"$PLUGIN_DIR"/dist/*.nupkg' in push["run"] and '"$PLUGIN_DIR"/dist/*.snupkg' in push["run"]
    assert push["run"].count("--skip-duplicate") == 2 and "--no-symbols" in push["run"]
    assert steps.index(download) < steps.index(login) < steps.index(push)
    verify = workflow["jobs"]["verify"]
    assert verify["needs"] == ["prepare", "publish"]
    assert "needs.publish.result == 'success'" in verify["if"] and "needs.prepare.outputs.publish == 'true'" in verify["if"]
    assert verify["permissions"] == {"contents": "read", "actions": "read"}
    commands = "\n".join(step.get("run", "") for job in (existing, publish, verify) for step in job["steps"])
    assert "make build" not in commands and "dotnet pack" not in commands, "publication never rebuilds"
    assert "--require-present --attempts 60 --delay 30" in commands
    assert "smoke_dotnet.py" in commands and "--source https://api.nuget.org/v3/index.json" in commands
    assert commands.index("--require-present") < commands.index("smoke_dotnet.py")
    for job in (existing, publish, verify):
        download = next(step for step in job["steps"] if "download-artifact@" in step.get("uses", ""))
        assert download["with"]["run-id"] == "${{ needs.prepare.outputs.artifact_run_id }}", "every job uses the tested artifact"


def test_nuget_recovery_requires_a_tag_ref(tmp_path: Path) -> None:
    recovery = next(step for step in yaml.safe_load(WORKFLOW.read_text())["jobs"]["prepare"]["steps"]
                    if step.get("id") == "recovery")
    env = dict(os.environ, REF_TYPE="branch")
    result = subprocess.run(["bash", "-c", recovery["run"]], env=env, cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode != 0 and "must be dispatched on the release tag" in result.stdout


def test_nuget_tagged_source_must_be_on_main(dotnet_repo: Path, tmp_path: Path) -> None:
    git(dotnet_repo, "update-ref", "refs/remotes/origin/main", "HEAD")
    (dotnet_repo / "README.md").write_text("unmerged change")
    commit_all(dotnet_repo, "Unmerged feature")
    tag = f"dotnet/fakedotnet/v{VERSION}"
    git(dotnet_repo, "tag", "-a", tag, "-m", "unmerged release")
    source = next(step for step in yaml.safe_load(WORKFLOW.read_text())["jobs"]["prepare"]["steps"] if step.get("id") == "source")
    env = dict(os.environ, REF_TYPE="tag", TAG=tag, GITHUB_OUTPUT=str(tmp_path / "output"))
    result = subprocess.run(["bash", "-c", source["run"]], env=env, cwd=dotnet_repo, capture_output=True, text=True)
    assert result.returncode != 0 and "release tags must be on main" in result.stdout
