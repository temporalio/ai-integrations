"""Common pytest hooks and fixtures for this integration."""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncGenerator
from functools import partial
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from typesafe_sdk import Choice, JSONContent, Noul, Score

from temporalio.client import Client
from temporalio.testing import WorkflowEnvironment
from temporalio.typesafe._workflow import _questions_payload
from tests import DEV_SERVER_DOWNLOAD_VERSION
from tests.helpers.environment import diagnosed_local_server
from tests.helpers.plugin_meta import load_plugin_meta
from tests.helpers.provenance import ProvenanceError, check_provenance

PLUGIN_ROOT = Path(__file__).resolve().parents[1]


def question_payload(
    question: Choice | Noul | Score | dict[str, JSONContent | None],
) -> dict[str, Any]:
    """Produce the Activity wire payload for one native question.

    Delegates to the production encoder so tests cannot drift from what the
    Activity actually validates and sends.
    """
    return next(iter(_questions_payload({"x": question}).values()))


def pytest_runtest_setup(item):  # type: ignore[reportMissingParameterType]
    """Print a newline so that custom printed output starts on a new line."""
    if item.config.getoption("-s"):
        print()


def pytest_sessionstart(session: pytest.Session) -> None:
    """Abort unless the installed plugin is the non-editable build of this checkout."""
    if hasattr(session.config, "workerinput"):
        return
    plugin = load_plugin_meta(PLUGIN_ROOT)
    try:
        check_provenance(
            plugin.coordinate,
            plugin.package_relpath,
        )
    except ProvenanceError as exc:
        pytest.exit(f"provenance guard failed: {exc}", returncode=1)


@pytest.fixture(scope="session")
def event_loop():
    """Create the session event loop."""
    loop = asyncio.get_event_loop_policy().new_event_loop()  # type: ignore[reportDeprecated]
    yield loop
    try:
        loop.close()
    except TypeError:
        raise


@pytest_asyncio.fixture(scope="session")  # type: ignore[reportUntypedFunctionDecorator]
async def env() -> AsyncGenerator[WorkflowEnvironment, None]:
    """Start the pinned local Temporal development server.

    Attach instead of spawning by setting ``TEMPORAL_TEST_ADDRESS`` (a running
    server's ``host:port``), or select the server binary with
    ``TEMPORAL_DEV_SERVER_PATH`` when the SDK's own download is unreachable.
    """
    address = os.environ.get("TEMPORAL_TEST_ADDRESS")
    if address:
        environment = WorkflowEnvironment.from_client(await Client.connect(address))
    else:
        options: dict[str, Any] = {
            "dev_server_download_version": DEV_SERVER_DOWNLOAD_VERSION,
            "dev_server_log_level": "info",
            "dev_server_existing_path": os.environ.get("TEMPORAL_DEV_SERVER_PATH"),
        }
        ip = os.environ.get("TEMPORAL_TEST_IP")
        if ip:
            options["ip"] = ip
        async with diagnosed_local_server(
            partial(WorkflowEnvironment.start_local, **options),
            server_version=DEV_SERVER_DOWNLOAD_VERSION,
        ) as environment:
            yield environment
        return
    yield environment
    await environment.shutdown()


@pytest_asyncio.fixture  # type: ignore[reportUntypedFunctionDecorator]
async def client(env: WorkflowEnvironment) -> Client:
    """Return the local environment's client."""
    return env.client


# There is an issue in tests sometimes in GitHub Actions where even though all tests
# pass, an unclear outer area is killing the process with a bad exit code. This
# hook forcefully kills the process as success when the exit code from pytest
# is a success.
@pytest.hookimpl(hookwrapper=True, trylast=True)
def pytest_cmdline_main(config):  # type: ignore[reportMissingParameterType, reportUnusedParameter]
    """Preserve the successful exit workaround without disrupting xdist."""
    result = yield
    exit_code = result.get_result()
    numprocesses = getattr(config.option, "numprocesses", None)
    running_with_xdist = hasattr(config, "workerinput") or numprocesses not in (
        None,
        0,
        "0",
    )
    if exit_code == 0 and not running_with_xdist:
        os._exit(0)
    return exit_code
