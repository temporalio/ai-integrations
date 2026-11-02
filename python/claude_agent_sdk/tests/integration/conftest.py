"""Integration tests: the bundled Claude Code on this host, with a scripted Anthropic API."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from temporalio.testing import WorkflowEnvironment
from tests.helpers.harness import Harness


@pytest.fixture
def harness(
    env: WorkflowEnvironment, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Harness]:
    harness = Harness(env.client, tmp_path, monkeypatch)
    yield harness
    harness.kill_processes()
    harness.api.shutdown()
