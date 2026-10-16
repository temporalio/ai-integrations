"""Import the Claude Agent SDK at use time.

Callers use :func:`load_sdk` instead of importing ``claude_agent_sdk`` at
module level, so workflow code can import this package without the SDK
installed and no SDK object is created at import.
"""

from __future__ import annotations

import functools
import importlib
from types import ModuleType

INSTALL_MESSAGE = (
    "The Claude turn Activity requires `claude-agent-sdk`; "
    "reinstall `temporalio-claude-agent-sdk` with its dependencies."
)


@functools.cache
def load_sdk() -> ModuleType:
    """Import and return ``claude_agent_sdk``, or raise with install guidance."""
    try:
        return importlib.import_module("claude_agent_sdk")
    except ModuleNotFoundError as exc:
        if (exc.name or "").split(".")[0] == "claude_agent_sdk":
            raise RuntimeError(INSTALL_MESSAGE) from exc
        raise


@functools.cache
def load_mcp() -> tuple[ModuleType, ModuleType]:
    """Import ``(mcp.server, mcp.types)`` for the tool bridge.

    The bridge builds its own lowlevel ``mcp.server.Server`` because it must read each call's
    ``_meta``, which ``create_sdk_mcp_server`` drops. ``mcp`` is a hard dependency.
    """
    return importlib.import_module("mcp.server"), importlib.import_module("mcp.types")
