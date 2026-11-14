"""Static workflow checks: pinned actions, isolated publishing secrets, least privilege."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import release_tool
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
    if path.name == "release-java.yml":
        allowed = {"stage": {"CENTRAL_USERNAME", "CENTRAL_PASSWORD", "GPG_PRIVATE_KEY", "GPG_PASSPHRASE"},
                   "publish": {"CENTRAL_USERNAME", "CENTRAL_PASSWORD"}}
        for name, job in doc["jobs"].items():
            found = set(re.findall(r"secrets\.([A-Z_]+)", yaml.dump(job)))
            assert found == allowed.get(name, set()), f"{name}: unexpected Maven credentials"
            if found:
                assert job["environment"] == {"stage": "maven-central-staging", "publish": "maven-central"}[name]
                assert "needs.prepare.outputs.publish == 'true'" in job["if"]
    else:
        assert "secrets." not in text, f"{path.name} must not use stored publishing credentials"
    assert "secrets: inherit" not in text, f"{path.name} must not inherit secrets"
    assert "cancel-in-progress: true" not in text, f"{path.name} must not cancel in-progress runs (required checks)"


def test_non_ga_releases_use_environments_without_required_reviewers() -> None:
    java = yaml.safe_load((REPO / ".github/workflows/release-java.yml").read_text())["jobs"]
    assert java["prepare"]["outputs"]["maturity"] == "${{ steps.tag.outputs.maturity }}"
    # The credential environments track staging and publication separately.
    # The approval-only environment protects GA releases without gating non-GA ones.
    assert java["publish"]["environment"] == "maven-central"
    assert java["stage"]["environment"] == "maven-central-staging"
    assert java["approve"]["environment"] == "maven-central-approval"
    assert set(java["approve"]["needs"]) == {"prepare", "stage"}
    assert java["approve"]["permissions"] == {}
    python = yaml.safe_load((REPO / ".github/workflows/release-python.yml").read_text())["jobs"]
    assert python["publish-testpypi"]["environment"] == "testpypi"
    assert python["publish-pypi"]["environment"]["name"] == "pypi"
    go = yaml.safe_load((REPO / ".github/workflows/release-go.yml").read_text())["jobs"]
    assert all("environment" not in job for job in go.values())
    dotnet = yaml.safe_load((REPO / ".github/workflows/release-dotnet.yml").read_text())["jobs"]
    assert dotnet["publish"]["environment"]["name"] == "nuget"
    assert all("environment" not in job for name, job in dotnet.items() if name != "publish")


@pytest.mark.parametrize("maturity,requires_approval", [
    ("pre-release", False), ("public-preview", False), ("generally-available", True),
    ("", True), ("unknown", True),
])
def test_java_publication_requires_successful_staging_and_applicable_approval(maturity: str, requires_approval: bool) -> None:
    jobs = yaml.safe_load((REPO / ".github/workflows/release-java.yml").read_text())["jobs"]

    def allows(job: str, *, approval: str = "skipped", stage: str = "success", prepare: str = "success",
               publish: str = "true", prerelease: str = "false", cancelled: bool = False) -> bool:
        needs = SimpleNamespace(
            prepare=SimpleNamespace(result=prepare, outputs=SimpleNamespace(
                maturity=maturity, publish=publish, prerelease=prerelease,
            )),
            stage=SimpleNamespace(result=stage),
            approve=SimpleNamespace(result=approval),
        )
        # These conditions use comparisons and boolean operators shared by both languages.
        # Evaluate the workflow's actual expressions so skipped dependencies cannot silently
        # block non-GA publication or let failed/rejected approvals through.
        expression = jobs[job]["if"].replace("&&", " and ").replace("||", " or ").replace("!cancelled()", "not cancelled")
        return eval(expression, {"__builtins__": {}}, {"needs": needs, "cancelled": cancelled})

    assert allows("approve") == requires_approval
    assert allows("publish") == (not requires_approval)
    assert allows("publish", approval="success")
    for approval in ("failure", "cancelled"):
        assert not allows("publish", approval=approval)
    for stage in ("failure", "skipped", "cancelled"):
        assert not allows("publish", stage=stage, approval="success")
    assert not allows("publish", prepare="failure", approval="success")
    assert not allows("publish", cancelled=True, approval="success")
    for job in ("approve", "publish"):
        assert not allows(job, publish="false", approval="success")
        assert not allows(job, prerelease="true", approval="success")


def test_ci_status_is_the_fan_in() -> None:
    doc = yaml.safe_load((REPO / ".github/workflows/ci.yml").read_text())
    status = doc["jobs"]["ci-status"]
    assert status["if"] == "always()"
    assert set(status["needs"]) == {"changes", "conventions", "python", "java", "go", "typescript", "dotnet"}
    for language in ("python", "java", "go", "typescript", "dotnet"):
        job = doc["jobs"][language]
        assert job["uses"] == f"./.github/workflows/_{language}-plugin.yml"
        assert "needs.changes.result == 'success'" in job["if"]
        assert f"needs.changes.outputs.{language} != '[]'" in job["if"]


def test_tooling_installs_pnpm_for_cooldown_regressions() -> None:
    workflow = yaml.safe_load((REPO / ".github/workflows/ci.yml").read_text())
    steps = workflow["jobs"]["conventions"]["steps"]
    setup = next(step for step in steps if step.get("uses", "").startswith("pnpm/action-setup@"))
    tests = next(step for step in steps if step.get("name") == "Tooling unit tests")
    assert setup["with"]["version"] == "10.27.0"
    assert steps.index(setup) < steps.index(tests)


def test_typescript_checks_tarball_and_clean_consumer_after_testing() -> None:
    workflow = yaml.safe_load((REPO / ".github/workflows/_typescript-plugin.yml").read_text())
    steps = workflow["jobs"]["test"]["steps"]
    setup_node = next(step for step in steps if step.get("uses", "").startswith("actions/setup-node@"))
    assert setup_node["with"]["check-latest"] is True
    test = next(step for step in steps if step.get("name") == "Lint and test")
    build = next(step for step in steps if step.get("name") == "Build and verify npm package")
    assert steps.index(test) < steps.index(build)
    assert "make lint && make test" in test["run"]
    assert "check_npm_package.py" in build["run"] and "smoke.mjs" in build["run"]
    assert build["if"] == "matrix.dist"
    sync = next(step for step in steps if step.get("name") == "Sync dependencies")
    assert "make sync-latest" in sync["run"] and "make sync" in sync["run"]


def test_dotnet_checks_package_and_clean_consumer_after_testing() -> None:
    ci = yaml.safe_load((REPO / ".github/workflows/ci.yml").read_text())
    assert ci["jobs"]["dotnet"]["with"]["deps"] == ci["jobs"]["python"]["with"]["deps"]
    workflow = yaml.safe_load((REPO / ".github/workflows/_dotnet-plugin.yml").read_text())
    assert workflow[True]["workflow_call"]["inputs"]["version"]["default"] == ""
    for job in workflow["jobs"].values():
        checkout = next(step for step in job["steps"] if step.get("uses", "").startswith("actions/checkout@"))
        assert checkout["with"] == {"ref": "${{ inputs.ref || github.sha }}", "persist-credentials": False}
    test = workflow["jobs"]["test"]
    # The version reaches build, test and pack through dotnet.mk's RELEASE_VERSION; no checkout file changes.
    assert test["env"]["RELEASE_VERSION"] == "${{ inputs.version || '0.0.0' }}"
    steps = test["steps"]
    names = [step.get("name", "") for step in steps]
    assert not any("version" in name.lower() for name in names), "the plugin tree is never modified for a release"
    setup = next(step for step in steps if step.get("uses", "").startswith("actions/setup-dotnet@"))
    # The SDK comes from the plugin's global.json; the cell's runtime is installed next to it for the tests.
    assert setup["with"]["dotnet-version"] == "${{ matrix.runtime }}.x"
    assert setup["with"]["global-json-file"] == "dotnet/${{ inputs.plugin }}/global.json"
    assert setup["with"]["cache"] is True and setup["with"]["cache-dependency-path"].endswith("/**/packages.lock.json")
    sdk = names.index("SDK is the global.json channel and the runtime under test is installed")
    check = names.index("Check dependency resolution under the cooldown")
    sync = names.index("Sync dependencies")
    lint = names.index("Lint")
    run_tests = names.index("Test")
    build = names.index("Build and verify NuGet package")
    reports = names.index("Upload test reports")
    upload = names.index("Upload tested package")
    assert sdk < check < sync < lint < run_tests < build < reports < upload
    assert steps[lint]["run"] == 'cd "$PLUGIN_DIR" && make lint'
    # Each cell runs the target framework of its runtime; the global.json SDK builds every framework.
    assert steps[run_tests]["run"] == 'cd "$PLUGIN_DIR" && make test TEST_FRAMEWORK="net$RUNTIME"'
    assert steps[run_tests]["env"] == {"RUNTIME": "${{ matrix.runtime }}"}
    assert steps[build]["if"] == "matrix.dist"
    assert "make build" in steps[build]["run"]
    assert "check_dotnet_dist.py" in steps[build]["run"] and '--commit "$(git rev-parse HEAD)"' in steps[build]["run"]
    assert "smoke_dotnet.py" in steps[build]["run"] and '--dist "$PLUGIN_DIR/dist"' in steps[build]["run"]
    assert steps[reports]["if"] == "always()"
    assert "${{ inputs.deps }}" in steps[reports]["with"]["name"], "nightly and release runs of one plugin differ in deps"
    assert steps[upload]["if"] == "matrix.dist"
    assert steps[upload]["with"]["name"] == "dist-dotnet-${{ inputs.plugin }}"
    assert steps[upload]["with"]["if-no-files-found"] == "error" and steps[upload]["with"]["overwrite"] is True


@pytest.mark.parametrize("language,sync_name", [
    ("python", "Sync environment"), ("typescript", "Sync dependencies"), ("dotnet", "Sync dependencies"),
])
def test_cooldown_is_checked_before_installing_the_committed_lock(language: str, sync_name: str) -> None:
    workflow = yaml.safe_load((REPO / f".github/workflows/_{language}-plugin.yml").read_text())
    steps = workflow["jobs"]["test"]["steps"]
    check = next(step for step in steps if step.get("name") == "Check dependency resolution under the cooldown")
    sync = next(step for step in steps if step.get("name") == sync_name)
    assert check["if"] == "matrix.dist && inputs.deps == 'locked'"
    assert check["run"] == 'cd "$PLUGIN_DIR" && make check-deps'
    assert steps.index(check) < steps.index(sync)
    assert "locked) make sync ;;" in sync["run"]
    assert "latest) make sync-latest ;;" in sync["run"]


def test_go_release_reuses_the_full_matrix_and_verifies_the_same_commit() -> None:
    doc = yaml.safe_load((REPO / ".github/workflows/release-go.yml").read_text())
    assert doc[True]["push"]["tags"] == ["go/*/v*"]
    assert doc[True]["release"]["types"] == ["published"]
    assert doc["permissions"] == {}
    # prepare also reads the plugin's nightly issue; nothing else in a Go release needs more than contents.
    assert all(job["permissions"] == ({"contents": "read", "issues": "read"} if name == "prepare" else {"contents": "read"})
               for name, job in doc["jobs"].items())
    test = doc["jobs"]["test"]
    assert test["uses"] == "./.github/workflows/_go-plugin.yml"
    assert test["with"] == {"plugin": "${{ needs.prepare.outputs.plugin }}", "deps": "locked", "ref": "${{ needs.prepare.outputs.sha }}"}
    verify = doc["jobs"]["verify"]
    assert verify["needs"] == ["prepare", "test"]
    assert verify["if"] == "needs.prepare.outputs.verify == 'true'"
    checkout = next(step for step in verify["steps"] if step.get("uses", "").startswith("actions/checkout@"))
    assert checkout["with"]["ref"] == "${{ needs.prepare.outputs.sha }}"
    reusable = yaml.safe_load((REPO / ".github/workflows/_go-plugin.yml").read_text())
    for job in reusable["jobs"].values():
        checkout = next(step for step in job["steps"] if step.get("uses", "").startswith("actions/checkout@"))
        assert checkout["with"]["ref"] == "${{ inputs.ref || github.sha }}"


@pytest.mark.parametrize("ref_type,dry_run,verify", [
    ("tag", "false", "true"), ("tag", "true", "false"), ("branch", "false", "false"),
    ("branch", "true", "false"), ("tag", "", "false"), ("tag", "False", "false"),
])
def test_go_release_dry_runs_do_not_verify_a_public_module(tmp_path: Path, ref_type: str, dry_run: str, verify: str) -> None:
    doc = yaml.safe_load((REPO / ".github/workflows/release-go.yml").read_text())
    gate = next(step for step in doc["jobs"]["prepare"]["steps"] if step.get("id") == "gate")
    output = tmp_path / "output"
    env = dict(os.environ, REF_TYPE=ref_type, DRY_RUN=dry_run, GITHUB_OUTPUT=str(output))
    result = subprocess.run(["bash", "-c", gate["run"]], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert output.read_text() == f"verify={verify}\n"


@pytest.mark.parametrize("ref_type,ref_name,tag,allowed", [
    ("tag", "go/fake/v0.1.0", "", True), ("tag", "go/fake/v0.1.0", "go/fake/v0.1.0", True),
    ("tag", "go/fake/v0.1.0", "go/other/v0.1.0", False), ("branch", "main", "", False),
    ("branch", "main", "go/fake/v0.1.0", True),
])
def test_go_release_dispatch_ref_must_match_the_tag(ref_type: str, ref_name: str, tag: str, allowed: bool) -> None:
    doc = yaml.safe_load((REPO / ".github/workflows/release-go.yml").read_text())
    step = next(step for step in doc["jobs"]["prepare"]["steps"] if step.get("name") == "Dispatch inputs are consistent with the ref")
    env = dict(os.environ, REF_TYPE=ref_type, REF_NAME=ref_name, INPUT_TAG=tag)
    result = subprocess.run(["bash", "-c", step["run"]], env=env, capture_output=True, text=True)
    assert (result.returncode == 0) == allowed, result.stdout + result.stderr


@pytest.mark.parametrize("on_main,ref_type,allowed", [(True, "tag", True), (False, "tag", False), (False, "branch", True)])
def test_go_release_source_must_be_main_reachable_for_tag_runs(plugin_repo: Path, tmp_path: Path, on_main: bool, ref_type: str, allowed: bool) -> None:
    from conftest import commit_all, git

    git(plugin_repo, "update-ref", "refs/remotes/origin/main", "HEAD")
    if not on_main:
        (plugin_repo / "README.md").write_text("feature branch\n")
        commit_all(plugin_repo, "Feature")
    tag = "go/fake/v0.1.0"
    git(plugin_repo, "tag", "-a", tag, "-m", "Go release")
    doc = yaml.safe_load((REPO / ".github/workflows/release-go.yml").read_text())
    source = next(step for step in doc["jobs"]["prepare"]["steps"] if step.get("id") == "source")
    output = tmp_path / "output"
    env = dict(os.environ, REF_TYPE=ref_type, TAG=tag, GITHUB_OUTPUT=str(output))
    result = subprocess.run(["bash", "-c", source["run"]], env=env, cwd=plugin_repo, capture_output=True, text=True)
    assert (result.returncode == 0) == allowed, result.stdout + result.stderr
    if allowed:
        assert output.read_text() == f"sha={git(plugin_repo, 'rev-parse', 'HEAD')}\n"


@pytest.mark.parametrize("language,skip_when", [
    ("python", "github.event_name != 'workflow_dispatch' || inputs.recover-run == ''"),
    ("typescript", "github.event_name != 'workflow_dispatch' || inputs.recover-run == ''"),
    ("dotnet", "github.event_name != 'workflow_dispatch' || inputs.recover-run == ''"),
    ("java", "github.event_name != 'workflow_dispatch' || inputs.deployment-id == ''"),
    # A Go tag is public once it exists, so only the branch dry run before tagging can stop it.
    ("go", "github.ref_type != 'tag' && github.event_name == 'workflow_dispatch'"),
])
def test_releases_check_the_nightly_run_after_the_version_policy(language: str, skip_when: str) -> None:
    prepare = yaml.safe_load((REPO / f".github/workflows/release-{language}.yml").read_text())["jobs"]["prepare"]
    assert prepare["permissions"]["issues"] == "read"
    names = [step.get("name", "") for step in prepare["steps"]]
    assert names.index("Nightly run is green") == names.index("Version policy") + 1
    step = prepare["steps"][names.index("Nightly run is green")]
    assert step["if"] == skip_when
    assert step["env"] == {"GH_TOKEN": "${{ github.token }}", "TAG": "${{ steps.tag.outputs.tag }}"}
    assert step["run"].endswith('release_tool.py check-nightly --tag "$TAG"')


# The publish jobs of each release workflow: the ones with an OIDC token, the action that uses it and the
# job whose `if` carries the tests-or-recovery gate (for .NET that is the token-free check before the push).
RELEASE_PUBLISHERS = {
    "python": {"jobs": ("publish-testpypi", "publish-pypi"), "action": "pypa/gh-action-pypi-publish@", "gated": "publish-testpypi",
               "environments": {"publish-testpypi": "testpypi", "publish-pypi": {"name": "pypi", "url": "${{ needs.prepare.outputs.release_url }}"}},
               "final": "publish-pypi", "final_if": "needs.prepare.outputs.publish == 'true' && needs.prepare.outputs.publish_pypi == 'true'",
               "tag_step": "Parse tag", "final_needs": {"prepare", "smoke-testpypi"}},
    "dotnet": {"jobs": ("publish",), "action": "NuGet/login@", "gated": "existing",
               "environments": {"publish": {"name": "nuget", "url": "${{ needs.prepare.outputs.release_url }}"}},
               "final": "publish", "final_if": "needs.existing.result == 'success' && needs.prepare.outputs.publish == 'true'",
               "tag_step": "Parse .NET tag", "final_needs": {"prepare", "test", "existing"}},
}


@pytest.mark.parametrize("language", sorted(RELEASE_PUBLISHERS))
def test_release_publish_jobs_are_inline_and_oidc_only(language: str) -> None:
    spec = RELEASE_PUBLISHERS[language]
    doc = yaml.safe_load((REPO / f".github/workflows/release-{language}.yml").read_text())
    for job in spec["jobs"]:
        j = doc["jobs"][job]
        assert j["permissions"] == {"id-token": "write", "actions": "read"}
        assert "uses" not in j, "publish jobs must be inline (PyPI rejects reusable workflows as trusted publishers)"
        steps = [s.get("uses", "") for s in j["steps"]]
        assert not any("checkout" in s for s in steps), "publish jobs download artifacts only"
        assert any(s.startswith(spec["action"]) for s in steps)
        assert j["environment"] == spec["environments"][job]
    for name, job in doc["jobs"].items():
        if name not in spec["jobs"]:
            assert "id-token" not in job.get("permissions", {}), f"{name}: only the publish jobs hold an OIDC token"
    # Uploads require a tagged run's tests or validated artifact recovery; the final registry needs its policy gate.
    gated = doc["jobs"][spec["gated"]]
    assert "needs.prepare.outputs.publish == 'true'" in gated["if"]
    assert "needs.test.result == 'success' || needs.prepare.outputs.recovery == 'true'" in gated["if"]
    assert spec["final_if"] in doc["jobs"][spec["final"]]["if"]
    assert doc[True]["release"]["types"] == ["published"]  # PyYAML parses the `on` key as boolean True
    assert "push" not in doc[True], "tag creation must not publish before release notes are reviewed"
    assert doc["jobs"]["prepare"]["if"] == f"github.event_name != 'release' || startsWith(github.event.release.tag_name, '{language}/')"
    inputs = doc[True]["workflow_dispatch"]["inputs"]
    assert inputs["skip-publish"]["type"] == "boolean" and inputs["tag"]["type"] == "string"
    # A dispatch on a tag ref with a different tag input must be rejected before anything runs.
    prepare_steps = [s.get("name", "") for s in doc["jobs"]["prepare"]["steps"]]
    assert prepare_steps.index("Dispatch inputs are consistent with the ref") < prepare_steps.index(spec["tag_step"])
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
    assert doc["concurrency"]["group"] == "${{ github.workflow }}-${{ inputs.tag || github.event.release.tag_name || github.ref_name }}"


@pytest.mark.parametrize("language", sorted(RELEASE_PUBLISHERS))
def test_published_release_is_verified_before_uploads_and_linked_from_deployment(language: str) -> None:
    spec = RELEASE_PUBLISHERS[language]
    jobs = yaml.safe_load((REPO / f".github/workflows/release-{language}.yml").read_text())["jobs"]
    prepare = jobs["prepare"]
    publish = jobs[spec["final"]]
    assert set(publish["needs"]) == spec["final_needs"]
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


@pytest.mark.parametrize("language", ["python", "typescript", "java", "go", "dotnet"])
def test_publication_does_not_modify_the_reviewed_github_release(language: str) -> None:
    jobs = yaml.safe_load((REPO / f".github/workflows/release-{language}.yml").read_text())["jobs"]
    for job in jobs.values():
        assert job.get("permissions", {}).get("contents") != "write"
        for step in job.get("steps", []):
            assert "draft-release" not in step.get("run", "")
            assert "publish-release" not in step.get("run", "")


def test_release_smoke_verifies_tested_artifacts() -> None:
    doc = yaml.safe_load((REPO / ".github/workflows/release-python.yml").read_text())
    for job in ("smoke-testpypi", "smoke-pypi"):
        smoke = [s for s in doc["jobs"][job]["steps"] if "smoke_from_index.sh" in s.get("run", "")]
        assert smoke[0]["env"]["REQUIRES_PYTHON"] == "${{ needs.prepare.outputs.requires_python }}", job
        verify = [s for s in doc["jobs"][job]["steps"] if "verify-index-files" in s.get("run", "")]
        assert len(verify) == 1, f"{job} must prove the index serves the tested artifacts"
        index = job.split("-", 1)[1]
        assert f"--index {index}" in verify[0]["run"]


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
               PRERELEASE="false", OVERRIDE="false", GITHUB_OUTPUT=str(output))
    subprocess.run(["bash", "-c", gate["run"]], env=env, check=True, capture_output=True, text=True)
    assert f"publish={'true' if publish else 'false'}\n" in output.read_text()


@pytest.mark.parametrize("prerelease,override,publish_pypi", [
    ("false", "false", True), ("false", "true", True),
    ("true", "false", False), ("true", "true", True),
])
def test_registry_routing_uses_version_and_dispatch_override(
    prerelease: str, override: str, publish_pypi: bool, tmp_path: Path,
) -> None:
    jobs = yaml.safe_load((REPO / ".github/workflows/release-python.yml").read_text())["jobs"]
    gate = next(s for s in jobs["prepare"]["steps"] if s.get("id") == "gate")
    output = tmp_path / "outputs"
    env = dict(os.environ, REF_TYPE="tag", RECOVERY="", SKIP_PUBLISH="false", PRERELEASE=prerelease,
               OVERRIDE=override, GITHUB_OUTPUT=str(output))
    subprocess.run(["bash", "-c", gate["run"]], env=env, check=True, capture_output=True, text=True)
    assert f"publish_pypi={'true' if publish_pypi else 'false'}\n" in output.read_text()


@pytest.mark.parametrize("ref_type,ref_name", [("tag", "main"), ("branch", "feature"), ("tag", "python/fake/v0.0.1")])
def test_release_recovery_rejects_dispatch_outside_main(ref_type: str, ref_name: str, tmp_path: Path) -> None:
    doc = yaml.safe_load((REPO / ".github/workflows/release-python.yml").read_text())
    recovery = next(s for s in doc["jobs"]["prepare"]["steps"] if s.get("id") == "recovery")
    env = dict(os.environ, REF_TYPE=ref_type, REF_NAME=ref_name, TAG="python/fake/v0.0.1", RUN_ID="123", GITHUB_OUTPUT=str(tmp_path / "outputs"))
    result = subprocess.run(["bash", "-c", recovery["run"]], env=env, capture_output=True, text=True)
    assert result.returncode == 1 and "recovery must be dispatched on main" in result.stdout


@pytest.mark.parametrize("language", ["python", "java", "typescript", "go", "dotnet"])
def test_release_checkouts_do_not_persist_credentials(language: str) -> None:
    doc = yaml.safe_load((REPO / f".github/workflows/release-{language}.yml").read_text())
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


def test_java_release_reuses_tested_bytes_and_recovers_deployments() -> None:
    doc = yaml.safe_load((REPO / ".github/workflows/release-java.yml").read_text())
    jobs = doc["jobs"]
    assert doc[True]["release"]["types"] == ["published"]
    assert "push" not in doc[True], "publishing the reviewed release creates the tag and starts staging"
    assert doc["concurrency"]["group"] == "${{ github.workflow }}-${{ inputs.tag || github.event.release.tag_name || github.ref_name }}"
    prepare = jobs["prepare"]
    assert prepare["if"] == "github.event_name != 'release' || startsWith(github.event.release.tag_name, 'java/')"
    tag = next(s for s in prepare["steps"] if s.get("id") == "tag")
    assert tag["env"]["TAG"] == "${{ inputs.tag || github.event.release.tag_name || github.ref_name }}"
    verify = next(s for s in prepare["steps"] if "check-published-release" in s.get("run", ""))
    assert verify["if"] == "steps.gate.outputs.publish == 'true'"
    assert verify["env"]["GH_TOKEN"] == "${{ github.token }}"
    assert verify["env"]["TAG"] == "${{ steps.tag.outputs.tag }}"
    assert jobs["test"]["uses"] == "./.github/workflows/_java-plugin.yml"
    assert jobs["test"]["with"]["version"] == "${{ needs.prepare.outputs.version }}"
    assert set(jobs["stage"]["needs"]) == {"prepare", "test"}
    assert jobs["stage"]["if"] == "needs.prepare.outputs.publish == 'true'"
    assert set(jobs["publish"]["needs"]) == {"prepare", "stage", "approve"}
    assert "!cancelled()" in jobs["publish"]["if"]
    assert "needs.stage.result == 'success'" in jobs["publish"]["if"]
    assert "needs.prepare.outputs.publish == 'true' && needs.prepare.outputs.prerelease == 'false'" in jobs["publish"]["if"]
    for name in ("stage", "publish"):
        steps = jobs[name]["steps"]
        downloads = [s["with"]["name"] for s in steps if "download-artifact@" in s.get("uses", "")]
        assert downloads == ["dist-java-${{ needs.prepare.outputs.plugin }}", "central-deployment-${{ needs.prepare.outputs.plugin }}"]
        assert not any("gradlew" in s.get("run", "") for s in steps), "publish jobs must not rebuild"
    recover = next(s for s in jobs["stage"]["steps"] if s.get("name") == "Recover saved deployment on a failed-job rerun")
    assert "continue-on-error" not in recover
    assert recover["if"] == "steps.recovery.outputs.restore_state == 'true'"
    check = next(s for s in jobs["stage"]["steps"] if s.get("id") == "recovery")
    assert "check-java-stage-recovery" in check["run"]
    assert check["env"]["RUN_ATTEMPT"] == "${{ github.run_attempt }}"
    assert check["env"]["GH_TOKEN"] == "${{ github.token }}"
    assert jobs["stage"]["permissions"]["actions"] == "read"
    command = next(s for s in jobs["stage"]["steps"] if s.get("id") == "staging")
    assert jobs["stage"]["name"] == release_tool.JAVA_STAGE_JOB
    assert command["name"] == release_tool.JAVA_STAGE_STEP
    assert command["env"]["RECOVER_ONLY"] == "${{ steps.recovery.outputs.recover_only }}"
    assert "--recover-only" in command["run"]
    steps = jobs["stage"]["steps"]
    assert steps.index(check) < steps.index(recover) < steps.index(command)
    preserve = next(s for s in jobs["stage"]["steps"] if s.get("name") == "Preserve signed bundle and deployment ID for recovery")
    assert preserve["if"] == "always() && (steps.staging.outcome == 'success' || steps.staging.outcome == 'failure' || steps.staging.outcome == 'cancelled')"
    assert preserve["with"]["overwrite"] is True
    for name in ("stage", "publish"):
        command = next(s for s in jobs[name]["steps"] if "maven_release.py" in s.get("run", ""))
        assert f"maven_release.py {name}" in command["run"] and "--smoke" in command["run"]
    gate = next(s for s in jobs["prepare"]["steps"] if s.get("id") == "gate")
    assert '[ "$REF_TYPE" = "tag" ] && [ "$SKIP_PUBLISH" = "false" ]' in gate["run"]
