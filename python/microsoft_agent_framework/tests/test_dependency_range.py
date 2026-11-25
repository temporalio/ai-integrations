"""``TESTED_VERSIONS`` is coupled to the lock, so a lock bump moves it in the same PR.

The range runs from the floor the plugin was first tested on through the
``agent-framework-core`` version the committed ``uv.lock`` pins. Locked runs
check the upper bound against that lock. Both dependency modes check that the
README's Installation sentence names the range.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

import pytest
from packaging.version import Version

from temporalio.microsoft_agent_framework._workflow_runtime import TESTED_VERSIONS

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
LOCK = PLUGIN_ROOT / "uv.lock"
README = PLUGIN_ROOT / "README.md"


def _locked_version(package: str) -> str:
    lock = tomllib.loads(LOCK.read_text(encoding="utf-8"))
    for entry in lock.get("package", []):
        if entry.get("name") == package:
            return str(entry["version"])
    raise AssertionError(f"{package} is not in {LOCK}")


@pytest.mark.skipif(
    os.environ.get("DEPS") == "latest",
    reason="Latest dependency runs replace the committed lock with a temporary lock",
)
def test_tested_range_matches_the_lock() -> None:
    if not LOCK.is_file():
        pytest.skip(f"{LOCK} is absent; the range check needs the lock file")
    locked = Version(_locked_version("agent-framework-core"))
    floor, _, upper = TESTED_VERSIONS.partition(" through ")
    assert Version(floor) <= locked
    assert Version(upper) == locked, (
        f"uv.lock pins agent-framework-core {locked} but TESTED_VERSIONS is "
        f"{TESTED_VERSIONS!r}. Move the range in _workflow_runtime.py and the "
        "tested-range sentence in the README's Installation paragraph in the same "
        "PR as the lock bump."
    )


def test_readme_names_the_tested_range() -> None:
    assert TESTED_VERSIONS in README.read_text(encoding="utf-8"), (
        f"the README's Installation paragraph must name the tested range "
        f"{TESTED_VERSIONS!r}; move it with the lock bump."
    )
