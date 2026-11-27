"""Connect tests to the dev server owned by the test launcher."""

from __future__ import annotations

import os

from temporalio.client import Client
from temporalio.testing import WorkflowEnvironment


async def create_test_environment() -> WorkflowEnvironment:
    """Return an environment connected to the shared test server."""
    address = os.environ.get("TEMPORAL_TEST_ADDRESS")
    if not address:
        raise RuntimeError(
            "Run live Temporal tests with `make test` or set TEMPORAL_TEST_ADDRESS "
            "to a running server."
        )
    return WorkflowEnvironment.from_client(await Client.connect(address))
