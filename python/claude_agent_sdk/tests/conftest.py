"""Common pytest hooks and fixtures for this integration."""

from __future__ import annotations

import asyncio
import contextlib
import faulthandler
import os
import subprocess
import threading
from collections.abc import AsyncGenerator, Iterator
from functools import partial
from pathlib import Path

import pytest
import pytest_asyncio
import pytest_timeout  # pyright: ignore[reportMissingTypeStubs]

from temporalio.testing import WorkflowEnvironment
from tests import DEV_SERVER_DOWNLOAD_VERSION
from tests.helpers.environment import start_local_with_retry
from tests.helpers.harness import Harness
from tests.helpers.plugin_meta import load_plugin_meta
from tests.helpers.provenance import ProvenanceError, check_provenance

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
# The whole run's limit, well past a slow full run: each test has its own.
SESSION_TIMEOUT = 30 * 60
# The time a process's first test that uses the dev server gets to start it, which
# may download it first and try three times, past the test's own limit.
SERVER_START_TIMEOUT = 5 * 60
_server_started = False


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
    faulthandler.dump_traceback_later(SESSION_TIMEOUT, exit=True)


@pytest.hookimpl(tryfirst=True)
def pytest_timeout_set_timer(
    item: pytest.Item, settings: pytest_timeout.Settings
) -> bool:
    """End a test that runs past its timeout, setup and teardown included.

    As pytest-timeout's thread method does: dump every thread's stack and exit
    the process, which xdist reports as the test's failure. Its signal method
    raises in whatever the main thread runs, which an asyncio task can swallow.
    First kill what the test started: its CLIs, and this process's children,
    such as the dev server.
    """
    timeout = settings.timeout
    if "env" in getattr(item, "fixturenames", ()) and not _server_started:
        timeout += SERVER_START_TIMEOUT

    def on_timeout() -> None:
        harness = getattr(item, "funcargs", {}).get("harness")
        if isinstance(harness, Harness):
            harness.kill_processes()
        with contextlib.suppress(OSError):  # no pkill on Windows
            subprocess.run(["pkill", "-KILL", "-P", str(os.getpid())], check=False)
        pytest_timeout.timeout_timer(item, settings)

    timer = threading.Timer(timeout, on_timeout)
    timer.daemon = True

    def cancel() -> None:
        timer.cancel()
        timer.join()

    item.cancel_timeout = cancel  # type: ignore[attr-defined]
    timer.start()
    return True


@pytest.fixture(scope="session")
def event_loop() -> Iterator[asyncio.AbstractEventLoop]:
    loop = asyncio.get_event_loop_policy().new_event_loop()  # type: ignore[reportDeprecated]
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="session")  # type: ignore[reportUntypedFunctionDecorator]
async def env() -> AsyncGenerator[WorkflowEnvironment, None]:
    global _server_started
    environment = await start_local_with_retry(
        partial(
            WorkflowEnvironment.start_local,
            dev_server_download_version=DEV_SERVER_DOWNLOAD_VERSION,
        )
    )
    _server_started = True
    yield environment
    await environment.shutdown()


# There is an issue in tests sometimes in GitHub Actions where even though all tests
# pass, an unclear outer area is killing the process with a bad exit code. This
# hook forcefully kills the process as success when the exit code from pytest
# is a success.
@pytest.hookimpl(hookwrapper=True, trylast=True)
def pytest_cmdline_main(config):  # type: ignore[reportMissingParameterType, reportUnusedParameter]
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
