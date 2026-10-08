"""Static checks on the workflow files: SHA-pinned actions, no secrets, least-privilege permissions."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
WORKFLOWS = sorted((REPO / ".github" / "workflows").glob("*.yml"))
ACTIONS = sorted((REPO / ".github" / "actions").glob("*/action.yml"))
USES_LINE = re.compile(r"^\s*-?\s*uses:\s*(?P<ref>\S+)(?:\s+#\s*(?P<comment>.*))?\s*$")
SHA_REF = re.compile(r"^[\w.-]+/[\w.-]+(?:/[\w./-]+)?@[0-9a-f]{40}$")
ALLOWED_UNPINNED = ("./.github/", "temporalio/public-actions/")


def _uses_lines(path: Path) -> list[tuple[str, str | None]]:
    out = []
    for line in path.read_text().splitlines():
        m = USES_LINE.match(line)
        if m:
            out.append((m.group("ref"), m.group("comment")))
    return out


@pytest.mark.parametrize("path", WORKFLOWS + ACTIONS, ids=lambda p: p.relative_to(REPO).as_posix())
def test_actions_are_sha_pinned_with_version_comments(path: Path) -> None:
    for ref, comment in _uses_lines(path):
        if ref.startswith(ALLOWED_UNPINNED):
            continue
        assert SHA_REF.match(ref), f"{path.name}: {ref} must be pinned to a 40-hex commit SHA"
        assert comment and (comment.startswith("v") or comment.startswith("release/")), f"{path.name}: {ref} needs a '# vN' comment"


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_workflows_parse_and_have_no_secrets(path: Path) -> None:
    doc = yaml.safe_load(path.read_text())
    assert isinstance(doc, dict) and "jobs" in doc
    text = path.read_text()
    assert "secrets." not in text.replace("secrets: inherit", "secrets.INHERIT"), f"{path.name} must not use repository secrets"
    assert "secrets: inherit" not in text, f"{path.name} must not inherit secrets"
    assert "cancel-in-progress: true" not in text, f"{path.name} must not cancel in-progress runs (required checks)"


def test_ci_status_is_the_fan_in() -> None:
    doc = yaml.safe_load((REPO / ".github/workflows/ci.yml").read_text())
    status = doc["jobs"]["ci-status"]
    assert status["if"] == "always()"
    assert set(status["needs"]) == {"changes", "conventions", "python", "java"}
    assert doc["jobs"]["python"]["uses"] == "./.github/workflows/_python-plugin.yml"
    assert "needs.changes.result == 'success'" in doc["jobs"]["python"]["if"]
    assert doc["jobs"]["java"]["uses"] == "./.github/workflows/_java-plugin.yml"
    assert "needs.changes.result == 'success'" in doc["jobs"]["java"]["if"]


def test_release_publish_jobs_are_inline_and_oidc_only() -> None:
    doc = yaml.safe_load((REPO / ".github/workflows/release-python.yml").read_text())
    for job in ("publish-testpypi", "publish-pypi"):
        j = doc["jobs"][job]
        assert j["permissions"] == {"id-token": "write", "actions": "read"}
        assert "uses" not in j, "publish jobs must be inline (PyPI rejects reusable workflows as trusted publishers)"
        steps = [s.get("uses", "") for s in j["steps"]]
        assert not any("checkout" in s for s in steps), "publish jobs download artifacts only"
        assert any(s.startswith("pypa/gh-action-pypi-publish@") for s in steps)
    assert doc["jobs"]["publish-testpypi"]["environment"] == "testpypi"
    assert doc["jobs"]["publish-pypi"]["environment"]["name"] == "pypi"
    # Uploads require a tagged run's tests or validated artifact recovery; PyPI needs its policy gate.
    assert "needs.prepare.outputs.publish == 'true'" in doc["jobs"]["publish-testpypi"]["if"]
    assert "needs.test.result == 'success' || needs.prepare.outputs.recovery == 'true'" in doc["jobs"]["publish-testpypi"]["if"]
    assert "needs.prepare.outputs.publish == 'true' && needs.prepare.outputs.publish_pypi == 'true'" in doc["jobs"]["publish-pypi"]["if"]
    assert doc[True]["release"]["types"] == ["published"]  # PyYAML parses the `on` key as boolean True
    assert "push" not in doc[True], "tag creation must not publish before release notes are reviewed"
    assert doc["jobs"]["prepare"]["if"] == "github.event_name != 'release' || startsWith(github.event.release.tag_name, 'python/')"
    inputs = doc[True]["workflow_dispatch"]["inputs"]
    assert inputs["skip-publish"]["type"] == "boolean" and inputs["tag"]["type"] == "string"
    # A dispatch on a tag ref with a different tag input must be rejected before anything runs.
    prepare_steps = [s.get("name", "") for s in doc["jobs"]["prepare"]["steps"]]
    assert prepare_steps.index("Dispatch inputs are consistent with the ref") < prepare_steps.index("Parse tag")
    gate = next(s for s in doc["jobs"]["prepare"]["steps"] if s.get("id") == "gate")
    assert '[ "$SKIP_PUBLISH" = "false" ]' in gate["run"], "publish only on the literal false"
    tag = next(s for s in doc["jobs"]["prepare"]["steps"] if s.get("id") == "tag")
    assert tag["env"]["TAG"] == "${{ inputs.tag || github.event.release.tag_name || github.ref_name }}"
    # Only prepare may look at the ref name (to check the dispatch inputs against it); every later job
    # works from prepare's parsed outputs.
    for name, job in doc["jobs"].items():
        if name == "prepare":
            continue
        for step in job.get("steps", []):
            assert "github.ref_name" not in yaml.dump(step.get("env", {})), f"{name}: read the parsed tag, not the ref name"
    assert doc["concurrency"]["group"] == "release-${{ inputs.tag || github.event.release.tag_name || github.ref_name }}"


def test_published_release_is_verified_before_uploads_and_linked_from_deployment() -> None:
    jobs = yaml.safe_load((REPO / ".github/workflows/release-python.yml").read_text())["jobs"]
    prepare = jobs["prepare"]
    publish = jobs["publish-pypi"]
    assert set(publish["needs"]) == {"prepare", "smoke-testpypi"}
    assert publish["environment"]["url"] == "${{ needs.prepare.outputs.release_url }}"
    assert prepare["outputs"]["release_url"] == "${{ steps.release.outputs.url }}"
    verify = next(step for step in prepare["steps"] if step.get("id") == "release")
    assert "check-published-release" in verify["run"] and '--github-output "$GITHUB_OUTPUT"' in verify["run"]
    assert verify["if"] == "steps.gate.outputs.publish == 'true'", "dry runs need no existing release"
    assert verify["env"]["TAG"] == "${{ steps.tag.outputs.tag }}"
    assert prepare["steps"].index(next(s for s in prepare["steps"] if s.get("id") == "gate")) < prepare["steps"].index(verify)
    summary = next(step for step in prepare["steps"] if "GITHUB_STEP_SUMMARY" in step.get("run", ""))
    assert summary["env"]["RELEASE_URL"] == "${{ steps.release.outputs.url }}"
    assert summary["if"] == verify["if"]


def test_publication_does_not_modify_the_reviewed_github_release() -> None:
    jobs = yaml.safe_load((REPO / ".github/workflows/release-python.yml").read_text())["jobs"]
    for job in jobs.values():
        assert job.get("permissions", {}).get("contents") != "write"
        for step in job.get("steps", []):
            assert "draft-release" not in step.get("run", "")
            assert "publish-release" not in step.get("run", "")


def test_release_smoke_is_strict_unless_the_sdk_still_bundles_the_plugin() -> None:
    doc = yaml.safe_load((REPO / ".github/workflows/release-python.yml").read_text())
    # smoke.py: "0" is strict; only allow-final = false (the plugin the SDK still ships) may overlap.
    expected = "${{ needs.prepare.outputs.allow_final == 'true' && '0' || '1' }}"
    for job in ("smoke-testpypi", "smoke-pypi"):
        smoke = [s for s in doc["jobs"][job]["steps"] if "smoke_from_index.sh" in s.get("run", "")]
        assert len(smoke) == 1 and smoke[0]["env"]["ALLOW_OVERLAP_WITH_CORE"] == expected, job
        assert smoke[0]["env"]["REQUIRES_PYTHON"] == "${{ needs.prepare.outputs.requires_python }}", job
        verify = [s for s in doc["jobs"][job]["steps"] if "verify-index-files" in s.get("run", "")]
        assert len(verify) == 1, f"{job} must prove the index serves the tested artifacts"
        index = job.split("-", 1)[1]
        assert f"--index {index}" in verify[0]["run"]
    assert doc["jobs"]["prepare"]["outputs"]["allow_final"] == "${{ steps.tag.outputs.allow_final }}"


def test_release_publishes_the_tested_artifacts() -> None:
    doc = yaml.safe_load((REPO / ".github/workflows/release-python.yml").read_text())
    assert "build" not in doc["jobs"], "the test job's dist cell builds the artifacts; do not rebuild for publishing"
    tested = "dist-${{ needs.prepare.outputs.plugin }}-locked"
    for job in ("publish-testpypi", "publish-pypi", "smoke-testpypi", "smoke-pypi"):
        downloads = [s["with"]["name"] for s in doc["jobs"][job]["steps"] if "download-artifact" in s.get("uses", "")]
        assert downloads == [tested], f"{job} must use the artifact the test job produced"
        download = next(s for s in doc["jobs"][job]["steps"] if "download-artifact" in s.get("uses", ""))
        assert download["with"]["run-id"] == "${{ needs.prepare.outputs.artifact_run_id }}"
    assert "test" in doc["jobs"]["publish-testpypi"]["needs"]
    assert doc["jobs"]["test"]["with"]["deps"] == "locked"
    assert doc["jobs"]["test"]["with"]["version"] == "${{ needs.prepare.outputs.version }}"
    # Recovery skips rebuilding. Every dependent job must handle that skip explicitly and still
    # require its own immediate predecessor to succeed.
    for job, predecessor in (("smoke-testpypi", "publish-testpypi"), ("publish-pypi", "smoke-testpypi"), ("smoke-pypi", "publish-pypi")):
        condition = doc["jobs"][job]["if"]
        assert "!cancelled()" in condition and f"needs.{predecessor}.result == 'success'" in condition
    plugin_wf = yaml.safe_load((REPO / ".github/workflows/_python-plugin.yml").read_text())
    assert plugin_wf[True]["workflow_call"]["inputs"]["version"]["default"] == ""
    steps = plugin_wf["jobs"]["test"]["steps"]
    inject = next(s for s in steps if s.get("name") == "Apply release version from tag")
    sync = next(s for s in steps if s.get("name") == "Sync environment")
    assert inject["if"] == "inputs.version != ''"
    assert 'uv version "$RELEASE_VERSION"' in inject["run"]
    assert steps.index(inject) < steps.index(sync), "release version must be injected before sync, test, and build"
    upload = next(s for s in steps if "upload-artifact" in s.get("uses", "") and s["with"]["name"].startswith("dist-"))
    assert upload["with"]["name"] == "dist-${{ inputs.plugin }}-${{ inputs.deps }}"
    assert upload["if"] == "matrix.dist" and upload["with"]["if-no-files-found"] == "error"
    assert upload["with"]["overwrite"] is True, "a re-run of the test job must replace the artifact, not duplicate it"
    junit = next(s for s in steps if "upload-artifact" in s.get("uses", "") and s["with"]["name"].startswith("junit-"))
    assert "${{ inputs.deps }}" in junit["with"]["name"], "ci.yml runs two lanes for one plugin in one run"
    build_check = next(s for s in steps if s.get("uses") == "./.github/actions/python-build-check")
    assert build_check["if"] == "matrix.dist"
    assert steps.index(build_check) < steps.index(upload), "the artifact must be built and checked before it is uploaded"


@pytest.mark.parametrize("ref_type,recovery,skip,publish", [
    ("tag", "", "false", True), ("branch", "", "false", False),
    ("branch", "true", "false", True), ("branch", "false", "false", False),
    ("tag", "", "true", False), ("branch", "true", "true", False),
    ("tag", "", "1", False), ("branch", "true", "True", False),
])
def test_release_publication_gate(ref_type: str, recovery: str, skip: str, publish: bool, tmp_path: Path) -> None:
    doc = yaml.safe_load((REPO / ".github/workflows/release-python.yml").read_text())
    gate = next(s for s in doc["jobs"]["prepare"]["steps"] if s.get("id") == "gate")
    output = tmp_path / "outputs"
    env = dict(os.environ, REF_TYPE=ref_type, RECOVERY=recovery, SKIP_PUBLISH=skip,
               PRERELEASE="false", ALLOW_FINAL="true", OVERRIDE="false", GITHUB_OUTPUT=str(output))
    subprocess.run(["bash", "-c", gate["run"]], env=env, check=True, capture_output=True, text=True)
    assert f"publish={'true' if publish else 'false'}\n" in output.read_text()


@pytest.mark.parametrize("prerelease,allow_final,override,publish_pypi", [
    ("false", "true", "false", True), ("false", "true", "true", True),
    ("true", "true", "false", False), ("true", "true", "true", True),
    ("true", "false", "false", False), ("true", "false", "true", False),
])
def test_registry_routing_uses_version_and_cutover_policy(
    prerelease: str, allow_final: str, override: str, publish_pypi: bool, tmp_path: Path,
) -> None:
    jobs = yaml.safe_load((REPO / ".github/workflows/release-python.yml").read_text())["jobs"]
    gate = next(s for s in jobs["prepare"]["steps"] if s.get("id") == "gate")
    output = tmp_path / "outputs"
    env = dict(os.environ, REF_TYPE="tag", RECOVERY="", SKIP_PUBLISH="false", PRERELEASE=prerelease,
               ALLOW_FINAL=allow_final, OVERRIDE=override, GITHUB_OUTPUT=str(output))
    subprocess.run(["bash", "-c", gate["run"]], env=env, check=True, capture_output=True, text=True)
    assert f"publish_pypi={'true' if publish_pypi else 'false'}\n" in output.read_text()


@pytest.mark.parametrize("ref_type,ref_name", [("tag", "main"), ("branch", "feature"), ("tag", "python/fake/v0.0.1")])
def test_release_recovery_rejects_dispatch_outside_main(ref_type: str, ref_name: str, tmp_path: Path) -> None:
    doc = yaml.safe_load((REPO / ".github/workflows/release-python.yml").read_text())
    recovery = next(s for s in doc["jobs"]["prepare"]["steps"] if s.get("id") == "recovery")
    env = dict(os.environ, REF_TYPE=ref_type, REF_NAME=ref_name, TAG="python/fake/v0.0.1", RUN_ID="123", GITHUB_OUTPUT=str(tmp_path / "outputs"))
    result = subprocess.run(["bash", "-c", recovery["run"]], env=env, capture_output=True, text=True)
    assert result.returncode == 1 and "recovery must be dispatched on main" in result.stdout


def test_release_checkouts_do_not_persist_credentials() -> None:
    doc = yaml.safe_load((REPO / ".github/workflows/release-python.yml").read_text())
    for name, job in doc["jobs"].items():
        for step in job.get("steps", []):
            if "actions/checkout@" in step.get("uses", ""):
                assert step.get("with", {}).get("persist-credentials") is False, f"{name}: checkout must set persist-credentials: false"


def test_top_level_permissions_are_empty_or_read_only() -> None:
    for path in WORKFLOWS:
        doc = yaml.safe_load(path.read_text())
        perms = doc.get("permissions")
        assert perms is not None, f"{path.name} must declare top-level permissions"
        assert all(v == "read" for v in perms.values()) or perms == {} or path.name == "opengrep.yml", path.name


def test_java_matrix_runs_before_artifact_upload() -> None:
    doc = yaml.safe_load((REPO / '.github/workflows/_java-plugin.yml').read_text())
    steps = doc['jobs']['test']['steps']
    test = next(s for s in steps if s.get('name') == 'Lint and test')
    build = next(s for s in steps if s.get('name') == 'Build and verify tested distributions')
    upload = next(s for s in steps if s.get('name') == 'Upload tested distributions')
    assert 'spotlessCheck test' in test['run'] and 'spotlessApply' not in test['run']
    assert '-PreleaseVersion=$RELEASE_VERSION' in test['run']
    assert '-PspringBootVersion=$SPRING_BOOT_VERSION' in test['run']
    assert build['if'] == upload['if'] == 'matrix.dist'
    assert 'check_java_dist.py' in build['run'] and 'smoke_java.py' in build['run']
    assert steps.index(test) < steps.index(build) < steps.index(upload)


def test_java_latest_dependencies_are_selected_and_locked_before_testing() -> None:
    ci = yaml.safe_load((REPO / '.github/workflows/ci.yml').read_text())
    mode = ci['jobs']['java']['with']['deps']
    assert mode == ci['jobs']['python']['with']['deps']
    assert "github.event_name == 'schedule'" in mode
    assert "github.event_name == 'workflow_dispatch' && inputs.latest-deps" in mode
    doc = yaml.safe_load((REPO / '.github/workflows/_java-plugin.yml').read_text())
    assert doc[True]['workflow_call']['inputs']['deps']['default'] == 'locked'
    assert doc['jobs']['test']['env']['DEPS'] == '${{ inputs.deps }}'
    steps = doc['jobs']['test']['steps']
    resolve = next(s for s in steps if s.get('name') == 'Resolve latest dependencies')
    test = next(s for s in steps if s.get('name') == 'Lint and test')
    build = next(s for s in steps if s.get('name') == 'Build and verify tested distributions')
    assert resolve['if'] == "inputs.deps == 'latest'"
    assert 'resolveAndLockAll --write-locks --refresh-dependencies' in resolve['run']
    for step in (resolve, test, build):
        assert '-PdependencyMode=$DEPS' in step['run']
        assert '-PspringBootVersion=$SPRING_BOOT_VERSION' in step['run']
    assert '--write-locks' not in test['run'] + build['run']
    assert steps.index(resolve) < steps.index(test) < steps.index(build)
