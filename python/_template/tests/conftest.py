"""Shared pytest configuration for this plugin.

Vendored from temporalio/sdk-python ``tests/conftest.py`` (origin/main) and pruned to what this
plugin's tests use. Re-sync it by hand when upstream changes (scripts/migrate/README.md).

Added on top of upstream:

* Offline mode. Every test replays recorded HTTP traffic (vcrpy via pytest-recording) from
  ``tests/contrib/<plugin>/cassettes/<test module>/<test node name>.yaml`` next to the test module.
  CI never holds a real API key. Re-record locally with ``make record``, which sets
  ``RECORD_MODE`` (see "Offline tests" in AGENTS.md and python/README.md).
* Provenance guard. The session aborts unless the installed plugin is the non-editable build of
  this checkout (tests/helpers/provenance.py, kept in sync with scripts/ci/smoke.py).
"""

from __future__ import annotations

import asyncio
import os
import re
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import opentelemetry.trace
import pytest
import pytest_asyncio
from opentelemetry.util._once import Once

from temporalio.client import Client
from temporalio.envconfig import ClientConfigProfile
from temporalio.testing import WorkflowEnvironment
from tests.helpers.plugin_meta import load_plugin_meta
from tests.helpers.provenance import ProvenanceError, check_provenance

from . import DEV_SERVER_DOWNLOAD_VERSION

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
PLUGIN = load_plugin_meta(PLUGIN_ROOT)

# ---------------------------------------------------------------------------
# Offline mode: record/replay of HTTP traffic
# ---------------------------------------------------------------------------

# Plugin-specific offline settings live in plugin.toml ``[offline]`` (read by ``PLUGIN`` above), so
# this file stays identical across plugins:
#   dummy-env  placeholder values exported during replay when unset (e.g. OPENAI_API_KEY =
#              "sk-cassette-replay") so upstream ``if not os.environ.get(...)`` skip guards do not
#              skip; never valid credentials, every request is answered from a cassette.
#   skips      tests skipped while replaying (never while recording), test function name -> reason.

#: Request and response headers that must never land in a committed cassette.
_SENSITIVE_HEADERS = (
    "authorization",
    "openai-organization",
    "openai-project",
    "cookie",
    "set-cookie",
    "x-request-id",
)

# Plugins built on the OpenAI Agents SDK: it registers BatchTraceProcessor(BackendSpanExporter) by
# default, and with any OPENAI_API_KEY set (including a dummy replay key) that exporter POSTs traces
# to api.openai.com from a daemon thread outside every cassette. Drop it; tests that assert on traces
# install their own processors. Do NOT set OPENAI_AGENTS_DISABLE_TRACING: that turns every span into a
# no-op and breaks tracing tests. Plugins without the Agents SDK simply skip this.
try:
    from agents.tracing import set_trace_processors
except ImportError:  # not an OpenAI Agents SDK plugin
    pass
else:
    set_trace_processors([])


def _record_mode() -> str:
    """``none`` (replay only, the default and what CI runs) or a vcrpy record mode.

    Driven by the ``RECORD_MODE`` environment variable, which ``make record`` sets. It is placed in
    ``vcr_config`` because that key overrides pytest-recording's ``--record-mode`` flag.
    """
    return os.environ.get("RECORD_MODE") or "none"


def _replaying() -> bool:
    return _record_mode() == "none"


def _scrub_response(response: dict[str, Any]) -> dict[str, Any]:
    headers = response.get("headers") or {}
    for key in list(headers):
        if key.lower() in _SENSITIVE_HEADERS:
            del headers[key]
    return response


@pytest.fixture(scope="session")
def vcr_config() -> dict[str, Any]:
    return {
        "record_mode": _record_mode(),
        # Body matching pairs concurrent and out-of-order requests correctly; request bodies
        # carry no timestamps or uuids, so they are stable across runs.
        "match_on": ["method", "scheme", "host", "port", "path", "query", "body"],
        "filter_headers": list(_SENSITIVE_HEADERS),
        "before_record_response": _scrub_response,
        # Model calls run inside Temporal activities, which retry. With this on, every replay of an
        # identical request returns the first recorded interaction instead of consuming the next one.
        # A test that needs two identical consecutive requests to get different responses would need
        # its own handling.
        "allow_playback_repeats": True,
        "decode_compressed_response": True,
        # Local servers (in-process MCP mocks, tooling on 127.0.0.1) are part of the test itself,
        # not third-party traffic: let them through instead of recording them.
        "ignore_localhost": True,
    }


@pytest.fixture
def default_cassette_name(request: pytest.FixtureRequest) -> str:
    """Cassette file stem: the test node name (including parameters), made filesystem-safe."""
    name = request.node.name
    if request.cls is not None:
        name = f"{request.cls.__name__}.{name}"
    return re.sub(r"[^A-Za-z0-9_.\[\]=,-]", "_", name)


# ---------------------------------------------------------------------------
# pytest hooks (upstream hooks plus the offline and provenance additions)
# ---------------------------------------------------------------------------


def pytest_runtest_setup(item):  # type: ignore[reportMissingParameterType]
    """Print a newline so that custom printed output starts on new line."""
    if item.config.getoption("-s"):
        print()


def pytest_addoption(parser):  # type: ignore[reportMissingParameterType]
    parser.addoption(
        "-E",
        "--workflow-environment",
        default="local",
        help="Which workflow environment to use ('local', 'time-skipping', 'envconfig', or ip:port for existing server)",
    )


def _uses_envconfig_server(env_type: str) -> bool:
    return env_type == "envconfig"


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "requires_local_server: test requires local-server-only behavior and cannot run against an envconfig server",
    )
    if _replaying():
        for name, value in PLUGIN.dummy_env.items():
            os.environ.setdefault(name, value)


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


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    replaying = _replaying()
    skip_local_only = pytest.mark.skip(
        reason="requires a local Temporal server, not the configured envconfig server"
    )
    envconfig = _uses_envconfig_server(config.getoption("--workflow-environment"))
    for item in items:
        if envconfig and item.get_closest_marker("requires_local_server"):
            item.add_marker(skip_local_only)
        base_name = getattr(item, "originalname", None) or item.name.split("[", 1)[0]
        if replaying and base_name in PLUGIN.offline_skips:
            item.add_marker(
                pytest.mark.skip(
                    reason=f"offline replay: {PLUGIN.offline_skips[base_name]}"
                )
            )
            continue
        # pytest-recording only wraps tests carrying the ``vcr`` marker; every test gets one so
        # any HTTP call is either replayed from its cassette or, when recording, captured into it.
        item.add_marker(pytest.mark.vcr)


async def _create_env_from_envconfig() -> WorkflowEnvironment:
    config = ClientConfigProfile.load().to_client_connect_config()
    if not config.get("target_host"):
        raise ValueError(
            "An envconfig workflow environment requires TEMPORAL_ADDRESS or an envconfig profile with an address"
        )
    return WorkflowEnvironment.from_client(await Client.connect(**config))


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()  # type: ignore[reportDeprecated]
    yield loop
    try:
        loop.close()
    except TypeError:
        raise


@pytest.fixture(scope="session")
def env_type(request: pytest.FixtureRequest) -> str:
    return request.config.getoption("--workflow-environment")  # type: ignore[reportReturnType]


@pytest_asyncio.fixture(scope="session")  # type: ignore[reportUntypedFunctionDecorator]
async def env(env_type: str) -> AsyncGenerator[WorkflowEnvironment, None]:
    if _uses_envconfig_server(env_type):
        env = await _create_env_from_envconfig()
    elif env_type == "local":
        env = await WorkflowEnvironment.start_local(
            dev_server_extra_args=[
                "--dynamic-config-value",
                "system.forceSearchAttributesCacheRefreshOnRead=true",
                "--dynamic-config-value",
                f"limit.historyCount.suggestContinueAsNew={CONTINUE_AS_NEW_SUGGEST_HISTORY_COUNT}",
                "--dynamic-config-value",
                "system.enableEagerWorkflowStart=true",
                "--dynamic-config-value",
                "frontend.enableExecuteMultiOperation=true",
                "--dynamic-config-value",
                "frontend.workerVersioningWorkflowAPIs=true",
                "--dynamic-config-value",
                "frontend.workerVersioningDataAPIs=true",
                "--dynamic-config-value",
                "system.enableDeploymentVersions=true",
                "--dynamic-config-value",
                "frontend.activityAPIsEnabled=true",
                "--dynamic-config-value",
                "frontend.enableCancelWorkerPollsOnShutdown=true",
                "--dynamic-config-value",
                "component.nexusoperations.recordCancelRequestCompletionEvents=true",
                "--dynamic-config-value",
                "activity.enableStandalone=true",
                "--dynamic-config-value",
                "activity.startDelayEnabled=true",
                "--dynamic-config-value",
                "history.enableChasm=true",
                "--dynamic-config-value",
                "history.enableTransitionHistory=true",
                "--dynamic-config-value",
                "history.enableCHASMCallbacks=true",
                "--dynamic-config-value",
                "history.enableCHASMSignalBacklinks=true",
                "--dynamic-config-value",
                "nexusoperation.enableStandalone=true",
                "--dynamic-config-value",
                'system.system.refreshNexusEndpointsMinWait="0s"',
                "--dynamic-config-value",
                "history.enableSignalWithStartFromWorkflow=true",
                "--dynamic-config-value",
                "history.enableUpdateCallbacks=true",
                "--dynamic-config-value",
                "activity.enableCallbacks=true",
            ],
            dev_server_download_version=DEV_SERVER_DOWNLOAD_VERSION,
        )
    elif env_type == "time-skipping":
        env = await WorkflowEnvironment.start_time_skipping()
    else:
        env = WorkflowEnvironment.from_client(await Client.connect(env_type))

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
