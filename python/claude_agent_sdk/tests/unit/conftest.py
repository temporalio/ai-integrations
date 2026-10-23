"""Unit tests: an in-memory sandbox with a scripted Claude Code, and no model."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from temporalio.testing import WorkflowEnvironment
from tests.helpers.memory_sandbox import MemoryHarness


@pytest.fixture
def harness(env: WorkflowEnvironment) -> Iterator[MemoryHarness]:
    harness = MemoryHarness(env.client)
    yield harness
    harness.sandbox.stop()
