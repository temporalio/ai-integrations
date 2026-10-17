"""Common pytest hooks and fixtures for this integration."""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import AsyncGenerator, Iterator
from functools import partial
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio

from temporalio.client import Client, WorkflowHandle
from temporalio.testing import WorkflowEnvironment
from tests import DEV_SERVER_DOWNLOAD_VERSION
from tests.helpers.environment import start_local_with_retry
from tests.helpers.plugin_meta import load_plugin_meta
from tests.helpers.provenance import ProvenanceError, check_provenance

PLUGIN_ROOT = Path(__file__).resolve().parents[1]

# The engine inherits this process's environment. Drop any Claude or Anthropic
# settings from the machine running the tests (an API key, a cloud provider, a parent
# Claude Code session's flags), so the engine only ever talks to the local fake API.
for _key in [k for k in os.environ if k.startswith(("CLAUDE", "ANTHROPIC"))]:
    del os.environ[_key]


def pytest_runtest_setup(item):  # type: ignore[reportMissingParameterType]
    """Print a newline so that custom printed output starts on a new line."""
    if item.config.getoption("-s"):
        print()


def pytest_sessionstart(session: pytest.Session) -> None:
    """Abort unless the installed plugin is the non-editable build of this checkout."""
    if hasattr(session.config, "workerinput"):
        return
    plugin = load_plugin_meta(PLUGIN_ROOT)
    allow_overlap = (not plugin.allow_final) or os.environ.get(
        "ALLOW_OVERLAP_WITH_CORE"
    ) == "1"
    try:
        check_provenance(
            plugin.coordinate,
            plugin.package_relpath,
            allow_overlap=allow_overlap,
            warn=lambda message: print(f"provenance: {message}"),
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
    """Start the pinned local Temporal development server."""
    environment = await start_local_with_retry(
        partial(
            WorkflowEnvironment.start_local,
            dev_server_download_version=DEV_SERVER_DOWNLOAD_VERSION,
        )
    )
    yield environment
    await environment.shutdown()


SUGGEST_AT = 200
"""History events at which the ``limited`` server suggests Continue-As-New."""
LIMIT = 600
"""History events at which the ``limited`` server terminates a run."""


@pytest_asyncio.fixture(scope="module")  # type: ignore[reportUntypedFunctionDecorator]
async def limited() -> AsyncGenerator[WorkflowEnvironment, None]:
    """A dev server whose history limits are low, so they are reached in seconds.

    The defaults are 51,200 events, with Continue-As-New suggested from 4,096.
    """
    settings = {
        "limit.historyCount.suggestContinueAsNew": SUGGEST_AT,
        "limit.historyCount.warn": SUGGEST_AT,
        "limit.historyCount.error": LIMIT,
    }
    extra: list[str] = []
    for key, value in settings.items():
        extra += ["--dynamic-config-value", f"{key}={value}"]
    environment = await start_local_with_retry(
        partial(
            WorkflowEnvironment.start_local,
            dev_server_download_version=DEV_SERVER_DOWNLOAD_VERSION,
            dev_server_extra_args=extra,
        )
    )
    yield environment
    await environment.shutdown()


@pytest_asyncio.fixture  # type: ignore[reportUntypedFunctionDecorator]
async def client(env: WorkflowEnvironment) -> Client:
    """Return the local environment's client."""
    return env.client


@pytest.fixture
def address(env: WorkflowEnvironment) -> str:
    """The dev server address, for Worker processes the tests start."""
    return env.client.service_client.config.target_host


@pytest.fixture
def shop_dir(tmp_path: Path) -> Iterator[Path]:
    """Give each test its own shop ledger (and clear the refund test knobs)."""
    saved = {
        k: os.environ.get(k) for k in ("SHOP_DIR", "FAIL_REFUND_TIMES", "REFUND_DELAY")
    }
    os.environ["SHOP_DIR"] = str(tmp_path / "shop")
    os.environ.pop("FAIL_REFUND_TIMES", None)
    os.environ.pop("REFUND_DELAY", None)
    yield tmp_path / "shop"
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


async def wait_for_approval(
    handle: WorkflowHandle[Any, Any], timeout: float = 45
) -> dict[str, Any] | None:
    """Wait until the refund agent asks for approval; None if it finished first."""
    from tests.refund.workflows import RefundAgentWorkflow

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        pending = await handle.query(RefundAgentWorkflow.pending_approvals)
        if pending:
            return pending[0]
        status = (await handle.describe()).status
        if status is not None and status.name != "RUNNING":
            return None
        await asyncio.sleep(0.2)
    raise TimeoutError("no approval request arrived")


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
