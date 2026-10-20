"""Workflow contracts independent of individual shell recipes."""

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]


def test_workflows_use_bazel_and_immutable_actions():
    for path in (ROOT / ".github/workflows").glob("*.yml"):
        text = path.read_text()
        yaml.safe_load(text)
        assert "scripts/" not in text
        assert "make test" not in text
        for owner, name, ref in re.findall(r"uses: ([\w-]+)/([\w-]+)@([^\s]+)", text):
            assert re.fullmatch(r"[0-9a-f]{40}", ref), (path, owner, name, ref)


def test_required_status_and_release_artifact_contracts():
    text = (ROOT / ".github/workflows/ci.yml").read_text()
    assert "name: ci-status" in text
    for language, artifact in [
        ("python", "dist-${{ inputs.plugin }}-${{ inputs.deps }}"),
        ("typescript", "inputs.artifact-name"),
        ("java", "dist-java-${{ inputs.plugin }}"),
    ]:
        assert (
            artifact in (ROOT / f".github/workflows/_{language}-plugin.yml").read_text()
        )
        assert (
            f"//{language}/_tools:release"
            in (ROOT / f".github/workflows/release-{language}.yml").read_text()
        )
