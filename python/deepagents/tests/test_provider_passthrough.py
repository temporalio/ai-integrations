"""Optional provider packages installed on the Worker do not stop Workflows.

deepagents loads ``langchain_aws`` and ``langchain_fireworks`` into every agent
it builds, and LangChain loads ``langchain_openai`` for provider-native
structured output, each with ``importlib.import_module`` from Workflow code and
only when the package is installed. The sandbox refuses what these packages do
at import, and passes that call through only for a module the process has
already imported. So the plugin passes them through and imports the installed
ones before a Workflow runs.
"""

from __future__ import annotations

import asyncio
import re
import sys
import warnings
from collections.abc import Iterator
from pathlib import Path

import pytest

from temporalio.client import Client
from temporalio.deepagents import _serde

README = Path(__file__).resolve().parents[1] / "README.md"

_COLD_PROCESS = """
import asyncio
import os
import sys
from pathlib import Path

from tests.helpers.cold_provider_agents import execute, install, replay

package, mode, directory = sys.argv[1], sys.argv[2], Path(sys.argv[3])
install(package, directory / "site")
history_path = directory / "history.json"
if mode == "execute":
    asyncio.run(execute(package, sys.argv[4], sys.argv[5], history_path))
else:
    asyncio.run(replay(package, history_path))

# Match conftest's exit workaround only after execution/replay and all assertions.
sys.stdout.flush()
sys.stderr.flush()
os._exit(0)
"""


@pytest.mark.parametrize(
    "package", ["langchain_aws", "langchain_fireworks", "langchain_openai"]
)
async def test_cold_agent_execution_and_replay(
    client: Client, tmp_path: Path, package: str
) -> None:
    """Execute and replay an agent in fresh processes with ``package`` installed."""
    for mode in ("execute", "replay"):
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            _COLD_PROCESS,
            package,
            mode,
            str(tmp_path),
            client.service_client.config.target_host,
            client.namespace,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            output, _ = await asyncio.wait_for(process.communicate(), timeout=45)
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()
        assert process.returncode == 0, output.decode()


def test_optional_provider_modules_are_passed_through() -> None:
    passthrough = _serde.default_passthrough_modules()
    for module in _serde._OPTIONAL_PROVIDER_MODULES:
        assert module.partition(".")[0] in passthrough


def test_readme_lists_the_default_passthrough_modules() -> None:
    section = README.read_text(encoding="utf-8").split("## Sandbox passthrough")[1]
    listed = set(re.findall(r"`(\w+)`", section.split("\n## ")[0]))
    assert listed >= set(_serde.default_passthrough_modules())


@pytest.fixture
def packages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A directory on the path whose ``provider_*`` modules are unloaded afterwards."""
    monkeypatch.syspath_prepend(str(tmp_path))
    yield tmp_path
    for name in [name for name in sys.modules if name.startswith("provider_")]:
        del sys.modules[name]


def _import(monkeypatch: pytest.MonkeyPatch, module: str) -> None:
    monkeypatch.setattr(_serde, "_OPTIONAL_PROVIDER_MODULES", (module,))
    _serde.import_optional_provider_modules()


def test_installed_provider_module_is_imported(
    packages: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (packages / "provider_installed" / "middleware").mkdir(parents=True)
    (packages / "provider_installed" / "__init__.py").write_text("")
    (packages / "provider_installed" / "middleware" / "__init__.py").write_text("")
    (packages / "provider_installed" / "middleware" / "prompt_caching.py").write_text(
        ""
    )

    _import(monkeypatch, "provider_installed.middleware.prompt_caching")

    assert "provider_installed.middleware.prompt_caching" in sys.modules


def test_missing_provider_package_is_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _import(monkeypatch, "provider_missing.middleware.prompt_caching")

    assert "provider_missing" not in sys.modules


def test_provider_release_without_the_module_is_skipped(
    packages: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (packages / "provider_old").mkdir()
    (packages / "provider_old" / "__init__.py").write_text("")

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _import(monkeypatch, "provider_old.middleware.prompt_caching")

    assert "provider_old" in sys.modules


@pytest.mark.parametrize(
    ("source", "error"),
    [
        ("raise RuntimeError('bad install')", "RuntimeError: bad install"),
        ("import provider_dependency", "No module named 'provider_dependency'"),
    ],
    ids=["raises", "missing_dependency"],
)
def test_broken_provider_package_warns_and_does_not_raise(
    packages: Path, monkeypatch: pytest.MonkeyPatch, source: str, error: str
) -> None:
    (packages / "provider_broken").mkdir()
    (packages / "provider_broken" / "__init__.py").write_text(source)

    with pytest.warns(UserWarning, match="could not import provider_broken") as caught:
        _import(monkeypatch, "provider_broken.middleware.prompt_caching")

    assert error in str(caught[0].message)
