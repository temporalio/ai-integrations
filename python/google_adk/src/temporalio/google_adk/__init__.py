"""Temporal Integration for ADK.

This module provides the necessary components to run ADK Agents within Temporal Workflows.
"""

from typing import TYPE_CHECKING, Any

from temporalio.google_adk._hitl import (
    HitlRequest,
    hitl_confirmation_response,
    hitl_input_response,
    pending_hitl_requests,
)
from temporalio.google_adk._model import TemporalModel
from temporalio.google_adk._plugin import (
    GoogleAdkPlugin,
)

if TYPE_CHECKING:
    from temporalio.google_adk._mcp import (
        TemporalMcpToolSet,
        TemporalMcpToolSetProvider,
        TemporalStatefulMcpToolSet,
        TemporalStatefulMcpToolSetProvider,
    )

__all__ = [
    "GoogleAdkPlugin",
    "HitlRequest",
    "TemporalMcpToolSet",
    "TemporalMcpToolSetProvider",
    "TemporalStatefulMcpToolSet",
    "TemporalStatefulMcpToolSetProvider",
    "TemporalModel",
    "hitl_confirmation_response",
    "hitl_input_response",
    "pending_hitl_requests",
]


def __getattr__(name: str) -> Any:
    """Load MCP toolsets only when the optional MCP integration is used."""
    if name in {
        "TemporalMcpToolSet",
        "TemporalMcpToolSetProvider",
        "TemporalStatefulMcpToolSet",
        "TemporalStatefulMcpToolSetProvider",
    }:
        from temporalio.google_adk import _mcp

        return getattr(_mcp, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
