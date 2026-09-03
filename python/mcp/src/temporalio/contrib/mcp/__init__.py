"""Run MCP Python SDK v2 clients durably from Temporal workflows.

Register named worker-side clients with :class:`MCPPlugin`, then construct a
:class:`TemporalMCPClient` with the same name inside workflow code. MCP
operations execute as Activities; transports and credentials remain on the
worker.

This package is experimental and may change in future versions.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from temporalio.contrib.mcp._plugin import MCPPlugin
    from temporalio.contrib.mcp._workflow import TemporalMCPClient

__all__ = ["MCPPlugin", "TemporalMCPClient"]


def __getattr__(name: str) -> Any:
    """Load each public API symbol without importing worker-only code into workflows."""
    if name == "MCPPlugin":
        from temporalio.contrib.mcp import _plugin

        return getattr(_plugin, name)
    if name == "TemporalMCPClient":
        from temporalio.contrib.mcp._workflow import TemporalMCPClient

        return TemporalMCPClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
