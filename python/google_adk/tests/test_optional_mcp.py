"""Core ADK integration remains usable without the optional MCP dependency."""

import asyncio
import subprocess
import sys
from pathlib import Path

import pytest

from temporalio.client import Client


def test_core_plugin_without_mcp() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.abc
import sys

class BlockMcp(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "mcp" or fullname.startswith("mcp."):
            raise ModuleNotFoundError("MCP is unavailable", name="mcp")

sys.meta_path.insert(0, BlockMcp())

from temporalio.google_adk import GoogleAdkPlugin, TemporalModel, pending_hitl_requests
from temporalio.google_adk._plugin import setup_deterministic_runtime

GoogleAdkPlugin()
setup_deterministic_runtime()
assert "temporalio.google_adk._mcp" not in sys.modules
""",
        ],
        check=True,
    )


@pytest.mark.parametrize("without_mcp", [True, False])
@pytest.mark.parametrize("without_model_providers", [True, False])
async def test_cold_agent_execution_and_replay(
    client: Client, tmp_path: Path, without_mcp: bool, without_model_providers: bool
) -> None:
    """Execute and replay cold ADK with a provider import slower than two seconds."""
    history_path = tmp_path / "history.json"
    for mode in ("execute", "replay"):
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            """
import asyncio
import importlib.abc
import os
import sys
import time
from pathlib import Path

class BlockOptionalDependencies(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if sys.argv[1] == "True" and (fullname == "mcp" or fullname.startswith("mcp.")):
            raise ModuleNotFoundError("MCP is unavailable", name=fullname)
        if sys.argv[6] == "True" and fullname.split(".")[0] in {"anthropic", "litellm", "openai"}:
            raise ModuleNotFoundError("Model provider is unavailable", name=fullname)

sys.meta_path.insert(0, BlockOptionalDependencies())

class SlowModelImport(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "google.adk.models.lite_llm":
            # Exceed Temporal's default deadlock deadline. Provider discovery
            # must happen during setup, before workflow activations start.
            time.sleep(2.1)

sys.meta_path.insert(0, SlowModelImport())

from tests.helpers.cold_adk_workflow import execute, replay

assert "google.adk.auth.auth_handler" not in sys.modules
assert "google.adk.models.lite_llm" not in sys.modules
assert "openai" not in sys.modules
history_path = Path(sys.argv[3])
if sys.argv[2] == "execute":
    asyncio.run(execute(sys.argv[4], sys.argv[5], history_path))
else:
    asyncio.run(replay(history_path))

assert not any(name == "mcp" or name.startswith("mcp.") for name in sys.modules)
assert "temporalio.google_adk._mcp" not in sys.modules
if sys.argv[6] == "True":
    assert not any(name.split(".")[0] in {"anthropic", "litellm", "openai"} for name in sys.modules)

# The SDK bridge can abort during interpreter shutdown on Python 3.10.
# Match conftest's exit workaround only after execution/replay and all assertions.
sys.stdout.flush()
sys.stderr.flush()
os._exit(0)
""",
            str(without_mcp),
            mode,
            str(history_path),
            client.service_client.config.target_host,
            client.namespace,
            str(without_model_providers),
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
