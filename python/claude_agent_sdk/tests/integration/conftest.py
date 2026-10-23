"""Integration tests: the bundled Claude Code in a local sandbox, with a scripted Anthropic API."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from temporalio.testing import WorkflowEnvironment
from tests.helpers.harness import Harness


@pytest.fixture
def harness(env: WorkflowEnvironment, tmp_path: Path) -> Iterator[Harness]:
    harness = Harness(env.client, tmp_path)
    yield harness
    harness.release.set()
    harness.sandbox.kill_processes()
    harness.api.shutdown()
