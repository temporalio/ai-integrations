"""Core ADK integration remains usable without the optional MCP dependency."""

import subprocess
import sys


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
