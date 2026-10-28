"""Common pytest hooks and fixtures for this integration."""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncGenerator, Callable
from functools import partial
from pathlib import Path

import claude_agent_sdk
import pytest
import pytest_asyncio

from temporalio.claude_agent_sdk import _native
from temporalio.claude_agent_sdk.testing import FakeClaudeCLI
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


@pytest.fixture(autouse=True)
def _skip_version_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    """The fake CLI is not a released Claude Code build, so skip the SDK's version probe."""
    monkeypatch.setenv("CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK", "1")


@pytest.fixture
def verified(monkeypatch: pytest.MonkeyPatch) -> None:
    """Treat the installed SDK and the fake CLI as a verified native-tool pair, on a host with a
    sandbox backend and (on Linux) the pinned seccomp helper; the fake CLI uses neither."""
    monkeypatch.setattr(
        _native,
        "NATIVE_TOOLS_VERIFIED",
        frozenset({(claude_agent_sdk.__version__, "2.1.294")}),
    )
    monkeypatch.setattr(_native, "sandbox_backend_available", lambda: True)
    monkeypatch.setattr(_native, "seccomp_requirement", lambda: None)


def scripted_cli(directory: Path, **scenario: object) -> tuple[str, Path]:
    """Script the fake CLI and return ``(cli_path, record_path)``."""
    cli = FakeClaudeCLI(directory, **scenario)
    return cli.cli_path, cli.record_path


@pytest.fixture
def _cli() -> Callable[..., tuple[str, Path]]:
    """The :func:`scripted_cli` factory, for tests that take it as a fixture."""
    return scripted_cli


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


# Pytest discovers fixtures by name. This keeps the checker from calling them unused.
_ = (_skip_version_probe, _cli)
