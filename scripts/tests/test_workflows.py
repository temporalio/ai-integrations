"""Static checks on the workflow files: SHA-pinned actions, no secrets, least-privilege permissions."""

from __future__ import annotations

import re
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
    assert set(status["needs"]) == {"changes", "conventions", "python", "python-lowest"}
    assert doc["jobs"]["python"]["uses"] == "./.github/workflows/_python-plugin.yml"
    assert "needs.changes.result == 'success'" in doc["jobs"]["python"]["if"]


def test_release_publish_jobs_are_inline_and_oidc_only() -> None:
    doc = yaml.safe_load((REPO / ".github/workflows/release-python.yml").read_text())
    for job in ("publish-testpypi", "publish-pypi"):
        j = doc["jobs"][job]
        assert j["permissions"] == {"id-token": "write"}
        assert "uses" not in j, "publish jobs must be inline (PyPI rejects reusable workflows as trusted publishers)"
        steps = [s.get("uses", "") for s in j["steps"]]
        assert not any("checkout" in s for s in steps), "publish jobs download artifacts only"
        assert any(s.startswith("pypa/gh-action-pypi-publish@") for s in steps)
    assert doc["jobs"]["publish-testpypi"]["environment"] == "testpypi"
    assert doc["jobs"]["publish-pypi"]["environment"] == "pypi"
    # Nothing is uploaded on a dry run or from a non-tag ref; PyPI additionally needs the policy gate.
    assert doc["jobs"]["publish-testpypi"]["if"] == "needs.prepare.outputs.publish == 'true'"
    assert doc["jobs"]["publish-pypi"]["if"] == "needs.prepare.outputs.publish == 'true' && needs.prepare.outputs.publish_pypi == 'true'"
    assert "needs.prepare.outputs.publish == 'true'" in doc["jobs"]["github-release"]["if"]
    assert doc[True]["push"]["tags"] == ["python/*/v*"]  # PyYAML parses the `on` key as boolean True
    inputs = doc[True]["workflow_dispatch"]["inputs"]
    assert inputs["skip-publish"]["type"] == "boolean" and inputs["tag"]["type"] == "string"
    # A dispatch on a tag ref with a different tag input must be rejected before anything runs.
    prepare_steps = [s.get("name", "") for s in doc["jobs"]["prepare"]["steps"]]
    assert prepare_steps.index("Dispatch inputs are consistent with the ref") < prepare_steps.index("Parse tag")
    gate = next(s for s in doc["jobs"]["prepare"]["steps"] if s.get("id") == "gate")
    assert '[ "$REF_TYPE" = "tag" ] && [ "$SKIP_PUBLISH" = "false" ]' in gate["run"], "publish only on the literal false"
    # The release job must act on the parsed tag, never on the ref name (they differ on a dry run).
    release_text = yaml.dump(doc["jobs"]["github-release"])
    assert "github.ref_name" not in release_text and "needs.prepare.outputs.tag" in release_text
    assert doc["concurrency"]["group"] == "release-${{ inputs.tag || github.ref }}"
    assert doc["jobs"]["github-release"]["if"].lstrip().startswith("!cancelled()")


def test_release_smoke_is_strict_unless_the_sdk_still_bundles_the_plugin() -> None:
    doc = yaml.safe_load((REPO / ".github/workflows/release-python.yml").read_text())
    # smoke.py: "0" is strict; only allow-final = false (the plugin the SDK still ships) may overlap.
    expected = "${{ needs.prepare.outputs.allow_final == 'true' && '0' || '1' }}"
    for job in ("smoke-testpypi", "smoke-pypi"):
        smoke = [s for s in doc["jobs"][job]["steps"] if "smoke_from_index.sh" in s.get("run", "")]
        assert len(smoke) == 1 and smoke[0]["env"]["ALLOW_OVERLAP_WITH_CORE"] == expected, job
        verify = [s for s in doc["jobs"][job]["steps"] if "verify-index-files" in s.get("run", "")]
        assert len(verify) == 1, f"{job} must prove the index serves the tested artifacts"
        index = job.split("-", 1)[1]
        assert f"--index {index}" in verify[0]["run"]
    assert doc["jobs"]["prepare"]["outputs"]["allow_final"] == "${{ steps.tag.outputs.allow_final }}"


def test_release_publishes_the_tested_artifacts() -> None:
    doc = yaml.safe_load((REPO / ".github/workflows/release-python.yml").read_text())
    assert "build" not in doc["jobs"], "the test job's dist cell builds the artifacts; do not rebuild for publishing"
    tested = "dist-${{ needs.prepare.outputs.plugin }}-locked"
    for job in ("publish-testpypi", "publish-pypi", "smoke-testpypi", "smoke-pypi", "github-release"):
        downloads = [s["with"]["name"] for s in doc["jobs"][job]["steps"] if "download-artifact" in s.get("uses", "")]
        assert downloads == [tested], f"{job} must use the artifact the test job produced"
    assert "test" in doc["jobs"]["publish-testpypi"]["needs"]
    assert doc["jobs"]["test"]["with"]["deps"] == "locked"
    plugin_wf = yaml.safe_load((REPO / ".github/workflows/_python-plugin.yml").read_text())
    steps = plugin_wf["jobs"]["test"]["steps"]
    upload = next(s for s in steps if "upload-artifact" in s.get("uses", "") and s["with"]["name"].startswith("dist-"))
    assert upload["with"]["name"] == "dist-${{ inputs.plugin }}-${{ inputs.deps }}"
    assert upload["if"] == "matrix.dist" and upload["with"]["if-no-files-found"] == "error"
    build_check = next(s for s in steps if s.get("uses") == "./.github/actions/python-build-check")
    assert build_check["if"] == "matrix.dist"
    assert steps.index(build_check) < steps.index(upload), "the artifact must be built and checked before it is uploaded"


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
