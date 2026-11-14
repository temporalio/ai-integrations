"""The newest-dependency lane keeps ``TESTED_VERSIONS`` honest.

CI's ``deps: latest`` lane runs ``make sync-latest`` (``uv lock --upgrade``, lock
not committed) before the suite, so in that lane the lock on disk differs from
the committed one. There, and only there, this test fails when the installed
``agent-framework-core`` is above the tested range: the nightly opens its
issue naming the version, and a maintainer extends the range and the README in
a normal PR once the rest of the suite passed on it.
"""

from __future__ import annotations

import importlib.metadata
import subprocess
from pathlib import Path

import pytest
from packaging.version import Version

from temporalio.microsoft_agent_framework._workflow_runtime import TESTED_VERSIONS

PLUGIN_ROOT = Path(__file__).resolve().parents[1]


def _lock_differs_from_committed() -> bool:
    """Whether ``uv.lock`` on disk is not the committed one (``make sync-latest``)."""
    committed = subprocess.run(
        ["git", "show", "HEAD:./uv.lock"],
        cwd=PLUGIN_ROOT,
        capture_output=True,
        text=True,
    )
    if committed.returncode != 0:
        # Not a git checkout: run the check anyway, it cannot fail on a
        # locked install.
        return True
    return committed.stdout != (PLUGIN_ROOT / "uv.lock").read_text(encoding="utf-8")


def test_tested_range_covers_the_newest_release() -> None:
    if not _lock_differs_from_committed():
        pytest.skip("locked dependencies; the newest-dependency lane checks the range")
    installed = Version(importlib.metadata.version("agent-framework-core"))
    newest = Version(TESTED_VERSIONS.split(" through ")[-1])
    assert installed <= newest, (
        f"agent-framework-core {installed} is above the tested range "
        f"({TESTED_VERSIONS}). This run used the newest dependencies; if the rest "
        f"of the suite passed on {installed}, extend TESTED_VERSIONS in "
        "_workflow_runtime.py and the tested range in the README's Installation "
        f"paragraph to {installed}."
    )
