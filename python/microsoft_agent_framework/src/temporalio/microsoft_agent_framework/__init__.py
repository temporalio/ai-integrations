"""Temporal integration for Microsoft Agent Framework.

Installed from the ``temporalio-microsoft-agent-framework`` distribution as
``temporalio.microsoft_agent_framework``. Build an ``agent_framework.Agent`` inside a
``@workflow.defn`` with ``client=TemporalChatClient("model-name")`` and add
``MicrosoftAgentFrameworkPlugin(chat_clients=...)`` to the Client or Worker: every model
round trip runs as a Temporal Activity while the agent's own loop, tools,
middleware and approvals run in the workflow.

The public names load on first access, so importing this package does not
import the worker-side modules into workflow code.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = [
    "MicrosoftAgentFrameworkPlugin",
    "MicrosoftAgentFrameworkWorkflowError",
    "TemporalChatClient",
    "TemporalMCPToolProvider",
    "workflow",
]

if TYPE_CHECKING:
    from temporalio.microsoft_agent_framework import workflow
    from temporalio.microsoft_agent_framework._chat_client import TemporalChatClient
    from temporalio.microsoft_agent_framework._mcp import TemporalMCPToolProvider
    from temporalio.microsoft_agent_framework._plugin import (
        MicrosoftAgentFrameworkPlugin,
    )
    from temporalio.microsoft_agent_framework.workflow import (
        MicrosoftAgentFrameworkWorkflowError,
    )


def __dir__() -> list[str]:
    return sorted(set(__all__) | {name for name in globals() if name.startswith("__")})


def __getattr__(name: str) -> Any:
    if name == "MicrosoftAgentFrameworkPlugin":
        from temporalio.microsoft_agent_framework._plugin import (
            MicrosoftAgentFrameworkPlugin,
        )

        return MicrosoftAgentFrameworkPlugin
    if name == "TemporalChatClient":
        from temporalio.microsoft_agent_framework._chat_client import TemporalChatClient

        return TemporalChatClient
    if name == "TemporalMCPToolProvider":
        from temporalio.microsoft_agent_framework._mcp import TemporalMCPToolProvider

        return TemporalMCPToolProvider
    if name == "MicrosoftAgentFrameworkWorkflowError":
        from temporalio.microsoft_agent_framework.workflow import (
            MicrosoftAgentFrameworkWorkflowError,
        )

        return MicrosoftAgentFrameworkWorkflowError
    if name == "workflow":
        import importlib

        return importlib.import_module("temporalio.microsoft_agent_framework.workflow")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
