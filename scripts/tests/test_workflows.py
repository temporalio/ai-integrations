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


def test_release_publishes_the_tested_artifacts() -> None:
    doc = yaml.safe_load((REPO / ".github/workflows/release-python.yml").read_text())
    assert "build" not in doc["jobs"], "the test job's dist cell builds the artifacts; do not rebuild for publishing"
    tested = "dist-${{ needs.prepare.outputs.plugin }}"
    for job in ("publish-testpypi", "publish-pypi", "github-release"):
        downloads = [s["with"]["name"] for s in doc["jobs"][job]["steps"] if "download-artifact" in s.get("uses", "")]
        assert downloads == [tested], f"{job} must publish the artifact the test job produced"
    assert "test" in doc["jobs"]["publish-testpypi"]["needs"]
    plugin_wf = yaml.safe_load((REPO / ".github/workflows/_python-plugin.yml").read_text())
    upload = [s for s in plugin_wf["jobs"]["test"]["steps"] if "upload-artifact" in s.get("uses", "") and s["with"]["name"].startswith("dist-")]
    assert upload and upload[0]["with"]["name"] == "dist-${{ inputs.plugin }}"


def test_release_checkouts_do_not_persist_credentials() -> None:
    doc = yaml.safe_load((REPO / ".github/workflows/release-python.yml").read_text())
    for name, job in doc["jobs"].items():
        for step in job.get("steps", []):
            if "actions/checkout@" in step.get("uses", ""):
                assert step["with"].get("persist-credentials") is False, f"{name}: checkout must set persist-credentials: false"


def test_top_level_permissions_are_empty_or_read_only() -> None:
    for path in WORKFLOWS:
        doc = yaml.safe_load(path.read_text())
        perms = doc.get("permissions")
        assert perms is not None, f"{path.name} must declare top-level permissions"
        assert all(v == "read" for v in perms.values()) or perms == {} or path.name == "opengrep.yml", path.name
