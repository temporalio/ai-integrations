"""Core Gemini integration remains usable without the optional MCP dependency."""

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

from google.genai import Client
from temporalio.google_genai import GoogleGenAIPlugin, TemporalAsyncClient, activity_as_tool
from temporalio.google_genai.testing import GeminiTestServer

client = Client(api_key="test-key")
try:
    GoogleGenAIPlugin(client)
    assert "temporalio.google_genai._mcp" not in sys.modules
    assert "temporalio.google_genai._temporal_mcp" not in sys.modules
finally:
    client.close()
""",
        ],
        check=True,
    )
