"""Regression coverage for the shared test server fixture."""

from __future__ import annotations

import pytest

from tests.helpers.environment import create_test_environment


async def test_live_tests_require_a_server_address(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TEMPORAL_TEST_ADDRESS", raising=False)
    with pytest.raises(RuntimeError, match="make test.*TEMPORAL_TEST_ADDRESS"):
        await create_test_environment()
