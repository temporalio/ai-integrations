"""Temporal integration for Microsoft Agent Framework.

Installed from the ``temporalio-agent-framework`` distribution as
``temporalio.agent_framework``. Build an ``agent_framework.Agent`` inside a
``@workflow.defn`` with ``client=TemporalChatClient("model-name")`` and add
``AgentFrameworkPlugin(chat_clients=...)`` to the Client or Worker: every model
round trip runs as a Temporal Activity while the agent's own loop, tools,
middleware and approvals run in the workflow.

The public names load on first access, so importing this package does not
import the worker-side modules into workflow code.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = [
    "AgentFrameworkPlugin",
    "AgentFrameworkWorkflowError",
    "TemporalChatClient",
    "TemporalMCPToolProvider",
    "workflow",
]

if TYPE_CHECKING:
    from temporalio.agent_framework import workflow
    from temporalio.agent_framework._chat_client import TemporalChatClient
    from temporalio.agent_framework._mcp import TemporalMCPToolProvider
    from temporalio.agent_framework._plugin import AgentFrameworkPlugin
    from temporalio.agent_framework.workflow import AgentFrameworkWorkflowError


def __getattr__(name: str) -> Any:
    if name == "AgentFrameworkPlugin":
        from temporalio.agent_framework._plugin import AgentFrameworkPlugin

        return AgentFrameworkPlugin
    if name == "TemporalChatClient":
        from temporalio.agent_framework._chat_client import TemporalChatClient

        return TemporalChatClient
    if name == "TemporalMCPToolProvider":
        from temporalio.agent_framework._mcp import TemporalMCPToolProvider

        return TemporalMCPToolProvider
    if name == "AgentFrameworkWorkflowError":
        from temporalio.agent_framework.workflow import AgentFrameworkWorkflowError

        return AgentFrameworkWorkflowError
    if name == "workflow":
        import importlib

        return importlib.import_module("temporalio.agent_framework.workflow")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
