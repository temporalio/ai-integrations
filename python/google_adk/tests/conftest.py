"""Common pytest hooks and fixtures for this integration."""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncGenerator
from functools import partial
from pathlib import Path

# ADK lazily imports this provider even for local models. Load its large OpenAI
# type tree before workflow tasks start so cold imports cannot trip the SDK's
# two-second deadlock detector (particularly under Python 3.10).
import openai  # noqa: E402,F401  # pyright: ignore[reportUnusedImport]
import opentelemetry._logs._internal
import opentelemetry.metrics
import opentelemetry.metrics._internal
import opentelemetry.trace
import pytest
import pytest_asyncio
from opentelemetry.metrics import NoOpMeterProvider
from opentelemetry.util._once import Once

from temporalio.client import Client
from temporalio.testing import WorkflowEnvironment
from tests import DEV_SERVER_DOWNLOAD_VERSION
from tests.helpers.environment import start_local_with_retry
from tests.helpers.plugin_meta import load_plugin_meta
from tests.helpers.provenance import ProvenanceError, check_provenance

PLUGIN_ROOT = Path(__file__).resolve().parents[1]


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


@pytest.fixture
def reset_otel_tracer_provider():
    """Isolate global OpenTelemetry tracer provider state around a test.

    Proxy tracers bound to a real provider stay bound forever; OTel has no
    rebind mechanism for tracers. Tests must install their provider before
    any span is created through a proxy tracer they care about.
    """
    opentelemetry.trace._TRACER_PROVIDER_SET_ONCE = Once()
    opentelemetry.trace._TRACER_PROVIDER = None
    yield
    opentelemetry.trace._TRACER_PROVIDER_SET_ONCE = Once()
    opentelemetry.trace._TRACER_PROVIDER = None


def _reset_meter_provider_globals() -> None:
    # Reset the set-once latch, then park proxy meters on a no-op provider:
    # set_meter_provider rebinds every proxy meter and its instruments, so
    # instruments bound during an earlier test stop recording into that
    # test's dead reader. Reset the latch again so the next installer wins.
    opentelemetry.metrics._internal._METER_PROVIDER_SET_ONCE = Once()
    opentelemetry.metrics._internal._METER_PROVIDER = None
    opentelemetry.metrics.set_meter_provider(NoOpMeterProvider())
    opentelemetry.metrics._internal._METER_PROVIDER_SET_ONCE = Once()
    opentelemetry.metrics._internal._METER_PROVIDER = None


@pytest.fixture
def reset_otel_meter_provider():
    """Isolate global OpenTelemetry meter provider state around a test.

    Both setup and teardown park proxy meters on a no-op provider, so
    instruments bound by other tests neither record into this test's provider
    unexpectedly nor keep recording into this test's reader afterwards. Any
    set_meter_provider call rebinds proxy meters, so tests that install their
    own provider are unaffected by the parking.
    """
    _reset_meter_provider_globals()
    yield
    _reset_meter_provider_globals()


@pytest.fixture
def reset_otel_logger_provider():
    """Isolate global OpenTelemetry logger provider state around a test.

    Unlike proxy meters, proxy loggers cache their real logger on first use
    and never rebind, even across a later set_logger_provider call. Tests
    exercising a library's module-level logger must clear that cache
    themselves (e.g. Google ADK's telemetry logger).
    """
    opentelemetry._logs._internal._LOGGER_PROVIDER_SET_ONCE = Once()
    opentelemetry._logs._internal._LOGGER_PROVIDER = None
    yield
    opentelemetry._logs._internal._LOGGER_PROVIDER_SET_ONCE = Once()
    opentelemetry._logs._internal._LOGGER_PROVIDER = None
