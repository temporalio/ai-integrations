"""npm release ordering, artifact identity, and fail-closed registry verification."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import subprocess
import tarfile
import urllib.error
from pathlib import Path

import pytest
import yaml

import npm_release
import release_tool
from conftest import commit_all, git


@pytest.mark.parametrize("version", [
    "0.1.0rc1", "0.1.0.dev1", "01.0.0", "0.1.0-rc.01", "0.1.0+build", "v0.1.0", "0.1",
    "0.1.0-", "0.1.0-rc.1+build", "0.1.0\n", " 0.1.0", "0.1.0 ",
])
def test_npm_rejects_noncanonical_versions(version: str) -> None:
    with pytest.raises(npm_release.NpmReleaseError, match="npm version"):
        npm_release.NpmVersion(version)
    with pytest.raises(release_tool.PolicyError):
        release_tool.parse_tag(f"typescript/google-adk/v{version}")


def test_npm_uses_semver_order_including_numeric_and_arbitrary_prereleases(tmp_path: Path) -> None:
    versions = ["0.1.0-rc.10", "0.1.0-rc.2", "0.1.0", "0.1.0-zeta", "0.1.0-1"]
    data = tmp_path / "registry.json"
    data.write_text(json.dumps({"name": "@temporalio/fake", "versions": {v: {"deprecated": "old"} for v in versions}}))
    published = release_tool.fetch_published_versions("@temporalio/fake", "npm", data)
    assert [str(v) for v in sorted(published)] == ["0.1.0-1", "0.1.0-rc.2", "0.1.0-rc.10", "0.1.0-zeta", "0.1.0"]
    with pytest.raises(release_tool.PolicyError, match="newest published"):
        release_tool.check_policy(npm_release.NpmVersion("0.1.0-rc.11"), "public-preview", published)
    assert release_tool.check_policy(npm_release.NpmVersion("0.1.0"), "public-preview", []) is None
    with pytest.raises(release_tool.PolicyError, match="first release"):
        release_tool.check_policy(npm_release.NpmVersion("1.0.0"), "public-preview", [])


@pytest.mark.parametrize("status", [401, 403, 500])
def test_registry_errors_are_never_treated_as_unpublished(monkeypatch: pytest.MonkeyPatch, status: int) -> None:
    def fail(*args, **kwargs):
        raise urllib.error.HTTPError("https://registry.npmjs.org", status, "failed", {}, None)
    monkeypatch.setattr(npm_release.urllib.request, "urlopen", fail)
    with pytest.raises(npm_release.NpmReleaseError, match=f"HTTP {status}"):
        npm_release.fetch_versions("@temporalio/fake")


def test_only_registry_404_means_new_coordinate(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(*args, **kwargs):
        raise urllib.error.HTTPError("https://registry.npmjs.org", 404, "missing", {}, None)
    monkeypatch.setattr(npm_release.urllib.request, "urlopen", missing)
    assert npm_release.fetch_versions("@temporalio/fake") == []


def tarball(dist: Path, **overrides) -> Path:
    dist.mkdir(exist_ok=True)
    path = dist / "fake.tgz"
    package = {"name": "@temporalio/fake", "version": "0.1.0", **overrides}
    payload = json.dumps(package).encode()
    with tarfile.open(path, "w:gz") as archive:
        member = tarfile.TarInfo("package/package.json")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))
    return path


def metadata(path: Path) -> dict:
    payload = path.read_bytes()
    return {"name": "@temporalio/fake", "version": "0.1.0", "dist": {
        "integrity": "sha512-" + base64.b64encode(hashlib.sha512(payload).digest()).decode(),
    }}


@pytest.mark.parametrize("overrides", [{"private": True}, {"version": "0.0.0"}, {"name": "@temporalio/other"}])
def test_wrong_or_private_tarball_cannot_be_published(tmp_path: Path, overrides: dict) -> None:
    tarball(tmp_path, **overrides)
    with pytest.raises(npm_release.NpmReleaseError, match="does not match"):
        npm_release.artifact(tmp_path, "@temporalio/fake", "0.1.0")


def test_rerun_requires_matching_registry_integrity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tarball(tmp_path)
    info = metadata(path)
    monkeypatch.setattr(npm_release, "fetch_package", lambda *args: {"name": "@temporalio/fake", "versions": {"0.1.0": info}})
    monkeypatch.setattr(npm_release.urllib.request, "urlopen", lambda *args, **kwargs: pytest.fail("npm install verifies the download"))
    assert npm_release.registry_state("@temporalio/fake", "0.1.0", tmp_path)
    info["dist"]["integrity"] = "sha512-wrong"
    with pytest.raises(npm_release.NpmReleaseError, match="integrity differs"):
        npm_release.registry_state("@temporalio/fake", "0.1.0", tmp_path)


@pytest.mark.parametrize("override", [{"deprecated": "withdrawn"}, {"version": "0.1.1"}, {"name": "@temporalio/other"}])
def test_registry_identity_and_deprecation_are_verified(tmp_path: Path, override: dict) -> None:
    path = tarball(tmp_path)
    with pytest.raises(npm_release.NpmReleaseError, match="identity differs"):
        npm_release.verify_artifact({**metadata(path), **override}, path, "@temporalio/fake", "0.1.0")


@pytest.mark.parametrize("dist", [{}, {"integrity": "sha1-legacy"}])
def test_registry_integrity_is_required(tmp_path: Path, dist: dict) -> None:
    path = tarball(tmp_path)
    info = metadata(path)
    info["dist"] = dist
    with pytest.raises(npm_release.NpmReleaseError, match="integrity differs"):
        npm_release.verify_artifact(info, path, "@temporalio/fake", "0.1.0")


def test_missing_version_skips_upload_only_when_not_required(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tarball(tmp_path)
    monkeypatch.setattr(npm_release, "fetch_package", lambda *args: None)
    args = ["npm-registry-state", "--coordinate", "@temporalio/fake", "--version", "0.1.0", "--dist", str(tmp_path),
            "--github-output", str(tmp_path / "output")]
    assert release_tool.main(args) == 0
    assert (tmp_path / "output").read_text() == "exists=false\n"
    assert release_tool.main([*args, "--require-present"]) == 1


def test_registry_propagation_is_retried(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    responses = iter([False, True])
    monkeypatch.setattr(release_tool, "registry_state", lambda *args: next(responses))
    args = ["npm-registry-state", "--coordinate", "@temporalio/fake", "--version", "0.1.0", "--dist", str(tmp_path),
            "--require-present", "--attempts", "2", "--delay", "0", "--github-output", str(tmp_path / "output")]
    assert release_tool.main(args) == 0
    assert (tmp_path / "output").read_text() == "exists=true\n"


WORKFLOW = Path(__file__).resolve().parents[2] / ".github/workflows/release-typescript.yml"


@pytest.mark.parametrize("ref_type,skip,expected", [
    ("branch", "false", "false"), ("branch", "true", "false"),
    ("tag", "true", "false"), ("tag", "", "false"), ("tag", "false", "true"),
])
def test_only_tagged_non_dry_runs_publish(tmp_path: Path, ref_type: str, skip: str, expected: str) -> None:
    workflow = yaml.safe_load(WORKFLOW.read_text())
    gate = next(step for step in workflow["jobs"]["prepare"]["steps"] if step.get("id") == "gate")
    (tmp_path / "package.json").write_text('{"private": false}')
    output = tmp_path / "output"
    env = dict(os.environ, REF_TYPE=ref_type, SKIP_PUBLISH=skip, PLUGIN_DIR=".", GITHUB_OUTPUT=str(output))
    result = subprocess.run(["bash", "-c", gate["run"]], env=env, cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert output.read_text() == f"publish={expected}\n"


def test_npm_publishes_only_the_full_matrix_artifact_with_oidc() -> None:
    workflow = yaml.safe_load(WORKFLOW.read_text())
    assert workflow["permissions"] == {}
    assert "push" not in workflow[True]
    assert workflow[True]["release"]["types"] == ["published"]
    test = workflow["jobs"]["test"]
    assert test["uses"] == "./.github/workflows/_typescript-plugin.yml"
    assert test["with"]["deps"] == "locked"
    assert test["with"]["ref"] == "${{ needs.prepare.outputs.sha }}"
    publish = workflow["jobs"]["publish"]
    assert publish["needs"] == ["prepare", "test"]
    assert "!cancelled() && needs.prepare.result == 'success'" in publish["if"]
    assert "needs.prepare.outputs.publish == 'true'" in publish["if"]
    assert "needs.test.result == 'success' || needs.prepare.outputs.recovery == 'true'" in publish["if"]
    assert test["if"] == "needs.prepare.outputs.recovery != 'true'"
    assert publish["permissions"] == {"contents": "read", "actions": "read", "id-token": "write"}
    assert publish["environment"]["name"] == "npm"
    download = next(step for step in publish["steps"] if "download-artifact@" in step.get("uses", ""))
    assert download["with"]["name"] == test["with"]["artifact-name"]
    assert download["with"]["run-id"] == "${{ needs.prepare.outputs.artifact_run_id }}"
    assert download["with"]["github-token"] == "${{ github.token }}"
    prepare = workflow["jobs"]["prepare"]
    assert prepare["permissions"]["actions"] == "read"
    assert prepare["outputs"]["artifact_run_id"] == "${{ steps.recovery.outputs.artifact_run_id || github.run_id }}"
    recovery = next(step for step in prepare["steps"] if step.get("id") == "recovery")
    assert recovery["if"] == "github.event_name == 'workflow_dispatch' && inputs.recover-run != ''"
    assert 'check-recovery-run --tag "$TAG" --run-id "$RUN_ID"' in recovery["run"]
    upload = next(step for step in publish["steps"] if "npm publish" in step.get("run", ""))
    assert upload["if"] == "steps.existing.outputs.exists == 'false'"
    assert '"$PLUGIN_DIR"/dist/*.tgz' in upload["run"]
    assert not any("make build" in step.get("run", "") for step in publish["steps"])
    commands = "\n".join(step.get("run", "") for step in publish["steps"])
    assert "--require-present" in commands and "smoke.mjs" in commands
    reusable = yaml.safe_load(WORKFLOW.with_name("_typescript-plugin.yml").read_text())
    for job in reusable["jobs"].values():
        checkout = next(step for step in job["steps"] if "checkout@" in step.get("uses", ""))
        assert checkout["with"]["ref"] == "${{ inputs.ref || github.sha }}"
    steps = reusable["jobs"]["test"]["steps"]
    inject = next(i for i, step in enumerate(steps) if step.get("name") == "Inject release version")
    lint = next(i for i, step in enumerate(steps) if step.get("name") == "Lint and test")
    assert inject < lint
    save = next(step for step in steps if "upload-artifact@" in step.get("uses", ""))
    assert save["with"]["overwrite"] is True


@pytest.mark.parametrize("language", ["python", "java", "go", "typescript"])
def test_release_events_do_not_share_concurrency_between_workflows(language: str) -> None:
    workflow = yaml.safe_load(WORKFLOW.with_name(f"release-{language}.yml").read_text())
    assert workflow["concurrency"]["group"].startswith("${{ github.workflow }}-")
    assert workflow["concurrency"]["cancel-in-progress"] is False


def test_npm_recovery_requires_a_tag_ref(tmp_path: Path) -> None:
    recovery = next(step for step in yaml.safe_load(WORKFLOW.read_text())["jobs"]["prepare"]["steps"]
                    if step.get("id") == "recovery")
    env = dict(os.environ, REF_TYPE="branch")
    result = subprocess.run(["bash", "-c", recovery["run"]], env=env, cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode != 0 and "must be dispatched on the release tag" in result.stdout


@pytest.fixture
def npm_recovery_source(repo: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, dict, dict, dict]:
    plugin = repo / "typescript/fake"
    plugin.mkdir(parents=True)
    (plugin / "plugin.toml").write_text('[ci]\nruntime-versions = ["22", "24"]\n')
    (plugin / "package.json").write_text('{"name": "@temporalio/fake", "version": "0.0.0", "private": false}')
    (plugin / "pnpm-lock.yaml").write_text("lockfileVersion: '9.0'\n")
    (plugin / "pnpm-workspace.yaml").write_text("minimumReleaseAge: 20160\n")
    sha = commit_all(repo, "Add npm package")
    git(repo, "tag", "typescript/fake/v0.1.0")
    git(repo, "update-ref", "refs/remotes/origin/main", sha)
    run = {"repository": {"full_name": "temporalio/ai-integrations"}, "event": "workflow_dispatch", "head_branch": "main",
           "head_sha": sha, "path": ".github/workflows/release-typescript.yml", "status": "completed"}
    names = ["Validate TypeScript tag and version policy", "Test / matrix"]
    names.extend(f"Test / fake ({runner}, node{version})" for runner, version in (
        ("ubuntu-latest", "22"), ("ubuntu-latest", "24"), ("macos-latest", "24"), ("windows-latest", "24"),
    ))
    jobs = {"jobs": [{"name": name, "status": "completed", "conclusion": "success"} for name in names]}
    artifacts = {"artifacts": [{"name": "npm-dist-fake", "expired": False}]}

    def gh(*args: str) -> str:
        endpoint = args[1]
        if "/jobs?" in endpoint:
            return json.dumps([jobs])
        if "/artifacts?" in endpoint:
            return json.dumps(artifacts)
        return json.dumps(run)

    monkeypatch.setattr(release_tool, "_gh", gh)
    return repo, run, jobs, artifacts


@pytest.mark.parametrize("event,branch", [
    ("workflow_dispatch", "main"), ("workflow_dispatch", "typescript/fake/v0.1.0"), ("release", "typescript/fake/v0.1.0"),
])
def test_npm_recovery_reuses_tested_dry_run_or_release_artifacts(npm_recovery_source: tuple, event: str, branch: str) -> None:
    repo, run, *_ = npm_recovery_source
    run.update(event=event, head_branch=branch)
    output = repo / "outputs"
    assert release_tool.main(["--repo-root", str(repo), "check-recovery-run", "--repo", "temporalio/ai-integrations",
                              "--tag", "typescript/fake/v0.1.0", "--run-id", "123", "--github-output", str(output)]) == 0
    assert output.read_text() == "recovery=true\nartifact_run_id=123\n"


@pytest.mark.parametrize("key,value", [
    ("event", "pull_request"), ("event", "push"), ("head_branch", "feature"), ("head_sha", "different-commit"),
    ("path", ".github/workflows/ci.yml"), ("status", "in_progress"), ("repository", {"full_name": "other/repo"}),
])
def test_npm_recovery_rejects_untrusted_or_unfinished_sources(npm_recovery_source: tuple, key: str, value: object) -> None:
    repo, run, *_ = npm_recovery_source
    run[key] = value
    with pytest.raises(release_tool.PolicyError, match="completed release-typescript.yml"):
        release_tool.check_recovery_run(repo, "temporalio/ai-integrations", "typescript/fake/v0.1.0", 123)


@pytest.mark.parametrize("name", [
    "Validate TypeScript tag and version policy", "Test / matrix", "Test / fake (ubuntu-latest, node22)",
    "Test / fake (ubuntu-latest, node24)", "Test / fake (macos-latest, node24)", "Test / fake (windows-latest, node24)",
])
@pytest.mark.parametrize("state", ["missing", "failure", "in_progress", "skipped"])
def test_npm_recovery_requires_the_complete_successful_matrix(npm_recovery_source: tuple, name: str, state: str) -> None:
    repo, _, jobs, _ = npm_recovery_source
    job = next(job for job in jobs["jobs"] if job["name"] == name)
    if state == "missing":
        jobs["jobs"].remove(job)
    elif state == "in_progress":
        job["status"] = state
    else:
        job["conclusion"] = state
    with pytest.raises(release_tool.PolicyError, match="no successful completed job"):
        release_tool.check_recovery_run(repo, "temporalio/ai-integrations", "typescript/fake/v0.1.0", 123)


@pytest.mark.parametrize("state", ["missing", "expired", "wrong_name", "duplicate"])
def test_npm_recovery_requires_one_available_tarball(npm_recovery_source: tuple, state: str) -> None:
    repo, _, _, artifacts = npm_recovery_source
    if state == "missing":
        artifacts["artifacts"] = []
    elif state == "expired":
        artifacts["artifacts"][0]["expired"] = True
    elif state == "wrong_name":
        artifacts["artifacts"][0]["name"] = "npm-dist-other"
    else:
        artifacts["artifacts"].append(dict(artifacts["artifacts"][0]))
    with pytest.raises(release_tool.PolicyError, match="exactly one unexpired npm-dist-fake"):
        release_tool.check_recovery_run(repo, "temporalio/ai-integrations", "typescript/fake/v0.1.0", 123)


@pytest.mark.parametrize("filename", ["plugin.toml", "package.json", "pnpm-lock.yaml", "pnpm-workspace.yaml"])
def test_npm_recovery_rejects_changed_package_metadata(npm_recovery_source: tuple, filename: str) -> None:
    repo, *_ = npm_recovery_source
    path = repo / "typescript/fake" / filename
    path.write_text(path.read_text() + "\n")
    with pytest.raises(release_tool.PolicyError, match="match the release tag"):
        release_tool.check_recovery_run(repo, "temporalio/ai-integrations", "typescript/fake/v0.1.0", 123)


def test_npm_recovery_rejects_tags_outside_main(npm_recovery_source: tuple) -> None:
    repo, *_ = npm_recovery_source
    (repo / "README.md").write_text("unmerged change")
    commit_all(repo, "Unmerged feature")
    git(repo, "tag", "typescript/fake/v0.1.1")
    with pytest.raises(release_tool.PolicyError, match="reachable from origin/main"):
        release_tool.check_recovery_run(repo, "temporalio/ai-integrations", "typescript/fake/v0.1.1", 123)


def test_private_package_cannot_pass_publication_gate(tmp_path: Path) -> None:
    gate = next(step for step in yaml.safe_load(WORKFLOW.read_text())["jobs"]["prepare"]["steps"] if step.get("id") == "gate")
    (tmp_path / "package.json").write_text('{"private": true}')
    output = tmp_path / "output"
    env = dict(os.environ, REF_TYPE="tag", SKIP_PUBLISH="false", PLUGIN_DIR=".", GITHUB_OUTPUT=str(output))
    result = subprocess.run(["bash", "-c", gate["run"]], env=env, cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode != 0 and not output.exists()


def test_npm_tagged_source_must_be_on_main(plugin_repo: Path, tmp_path: Path) -> None:
    git(plugin_repo, "update-ref", "refs/remotes/origin/main", "HEAD")
    (plugin_repo / "README.md").write_text("unmerged change")
    commit_all(plugin_repo, "Unmerged feature")
    tag = "typescript/fake/v0.1.0"
    git(plugin_repo, "tag", "-a", tag, "-m", "unmerged release")
    source = next(step for step in yaml.safe_load(WORKFLOW.read_text())["jobs"]["prepare"]["steps"] if step.get("id") == "source")
    env = dict(os.environ, REF_TYPE="tag", TAG=tag, GITHUB_OUTPUT=str(tmp_path / "output"))
    result = subprocess.run(["bash", "-c", source["run"]], env=env, cwd=plugin_repo, capture_output=True, text=True)
    assert result.returncode != 0 and "release tags must be on main" in result.stdout
