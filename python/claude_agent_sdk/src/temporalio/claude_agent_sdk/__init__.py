"""Claude Agent SDK integration for Temporal.

Register :class:`ClaudeAgentPlugin` on a worker and call :func:`run_claude_turn` from a
workflow. The Claude Code CLI runs only inside the turn Activity; importing this package
does not import ``claude_agent_sdk``, so workflow code can import it freely.

.. warning:: Experimental. See the README for supported features and limitations.
"""

from temporalio.claude_agent_sdk import testing, workflow
from temporalio.claude_agent_sdk._activity import PartialSink
from temporalio.claude_agent_sdk._lease import FileLeaseStore, LeaseStore
from temporalio.claude_agent_sdk._modal_workspace import ModalVolumeWorkspaceStore
from temporalio.claude_agent_sdk._models import (
    READ_ONLY_NATIVE_TOOLS,
    ClaudeDeferredTool,
    ClaudePartialEvent,
    ClaudeTurnOptions,
    ClaudeTurnResult,
    TokenUsage,
)
from temporalio.claude_agent_sdk._plugin import ClaudeAgentPlugin
from temporalio.claude_agent_sdk._session import (
    ClaudeSession,
    ClaudeSessionState,
    run_claude_turn,
)
from temporalio.claude_agent_sdk._session_store import FileSessionStore
from temporalio.claude_agent_sdk._workspace import (
    SharedDirWorkspaceStore,
    WorkspaceSnapshotError,
    WorkspaceStore,
)
from temporalio.claude_agent_sdk.sandbox import (
    SandboxError,
    SandboxLost,
    SandboxRuntime,
)

__all__ = [
    "READ_ONLY_NATIVE_TOOLS",
    "ClaudeAgentPlugin",
    "ClaudeDeferredTool",
    "ClaudePartialEvent",
    "ClaudeSession",
    "ClaudeSessionState",
    "ClaudeTurnOptions",
    "ClaudeTurnResult",
    "FileLeaseStore",
    "FileSessionStore",
    "LeaseStore",
    "ModalVolumeWorkspaceStore",
    "PartialSink",
    "SandboxError",
    "SandboxLost",
    "SandboxRuntime",
    "SharedDirWorkspaceStore",
    "TokenUsage",
    "WorkspaceSnapshotError",
    "WorkspaceStore",
    "run_claude_turn",
    "testing",
    "workflow",
]
