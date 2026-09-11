"""Shared pytest configuration for this plugin.

The provenance guard aborts the session unless the installed plugin is the non-editable build of
this checkout (tests/helpers/provenance.py).
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncGenerator
from pathlib import Path

import opentelemetry.trace
import pytest
import pytest_asyncio
from agents.tracing import set_trace_processors
from opentelemetry.util._once import Once

from temporalio.client import Client
from temporalio.testing import WorkflowEnvironment
from tests.helpers.plugin_meta import load_plugin_meta
from tests.helpers.provenance import ProvenanceError, check_provenance

from . import DEV_SERVER_DOWNLOAD_VERSION

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
PLUGIN = load_plugin_meta(PLUGIN_ROOT)

# The Agents SDK installs a process-global exporter by default. Tests that
# exercise tracing install their own processors, so begin with an empty set to
# prevent unrelated test spans from being exported from background threads.
set_trace_processors([])

# ---------------------------------------------------------------------------
# pytest hooks and provenance guard
# ---------------------------------------------------------------------------


def pytest_runtest_setup(item):  # type: ignore[reportMissingParameterType]
    """Print a newline so that custom printed output starts on new line."""
    if item.config.getoption("-s"):
        print()


def pytest_sessionstart(session: pytest.Session) -> None:  # type: ignore[reportUnusedParameter]
    """Abort unless the installed plugin is the non-editable build of this checkout."""
    allow_overlap = (not PLUGIN.allow_final) or os.environ.get(
        "ALLOW_OVERLAP_WITH_CORE"
    ) == "1"
    try:
        check_provenance(
            PLUGIN.coordinate,
            PLUGIN.package_relpath,
            allow_overlap=allow_overlap,
            warn=lambda message: print(f"provenance: {message}"),
        )
    except ProvenanceError as exc:
        pytest.exit(f"provenance guard failed: {exc}", returncode=1)


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()  # type: ignore[reportDeprecated]
    yield loop
    try:
        loop.close()
    except TypeError:
        raise


@pytest_asyncio.fixture(scope="session")  # type: ignore[reportUntypedFunctionDecorator]
async def env() -> AsyncGenerator[WorkflowEnvironment, None]:
    # No --dynamic-config-value flags: the dev server's defaults cover everything the plugin
    # suites exercise (verified by running the full suite without any). Add a flag here only when a
    # test needs a server feature that is off by default, and say which test needs it.
    env = await WorkflowEnvironment.start_local(
        dev_server_download_version=DEV_SERVER_DOWNLOAD_VERSION,
    )

    yield env
    await env.shutdown()


@pytest_asyncio.fixture  # type: ignore[reportUntypedFunctionDecorator]
async def client(env: WorkflowEnvironment) -> Client:
    return env.client


# There is an issue in tests sometimes in GitHub actions where even though all tests
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


CONTINUE_AS_NEW_SUGGEST_HISTORY_COUNT = 50


# OpenTelemetry's global providers are set-once per process with no public
# way to unset them, so tests needing their own provider must reset the
# globals directly -- the same isolation pattern OpenTelemetry's own test
# suite uses (opentelemetry.test.globals_test). Isolation cannot be delegated
# to scheduling: CI runs pytest-xdist with --dist=worksteal, which ignores
# xdist_group pinning, so any test in the suite can share a worker process
# with any other. Every test that installs a global provider must therefore
# use these fixtures and leave the globals reset behind it.


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
