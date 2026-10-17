"""Options for the tests' Workers."""

from __future__ import annotations

from typing import Any

FAIL_FAST: dict[str, Any] = {"workflow_failure_exception_types": [Exception]}
"""A bug in Workflow code fails the Workflow, so the test fails in seconds with that
bug as the cause. By default its Workflow task would fail again and again, and the
test would wait for its timeout. Tests that check a failed Workflow task on purpose
leave this out."""
