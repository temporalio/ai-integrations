"""Durable Claude Agent SDK agents on Temporal.

Every durable tool call Claude makes runs as its own Temporal Activity: finished
calls never run again after a crash, retries follow your retry policy, and tools can
wait for a human approval for as long as needed.

Workflow side: :class:`DurableClaudeAgent` and :func:`activity_as_tool`.
Worker side: :class:`ClaudeAgentPlugin` with :class:`ClaudeAgentSdkRunner` (or the
scripted runner in :mod:`temporalio.claude_agent_sdk.testing` for tests).
"""

from temporalio.claude_agent_sdk._activity import (
    SegmentRunner,
    make_segment_activity,
    make_tool_step_activity,
)
from temporalio.claude_agent_sdk._events import follow_agent
from temporalio.claude_agent_sdk._models import (
    TOOL_CALL_INTERRUPTED,
    TOOL_CALL_NOT_RUN,
    WORKSPACE_CORRUPT,
    WORKSPACE_INVALID,
    WORKSPACE_NOT_FOUND,
    WORKSPACE_TOO_LARGE,
    AgentState,
    ConversationRef,
    DeferredCall,
    SegmentInput,
    SegmentOutput,
    ToolOutcome,
    ToolSpec,
    ToolStepInput,
    ToolStepRetry,
    Workspace,
)
from temporalio.claude_agent_sdk._plugin import ClaudeAgentPlugin
from temporalio.claude_agent_sdk._runner import ClaudeAgentSdkRunner
from temporalio.claude_agent_sdk._session_store import FileSessionStore
from temporalio.claude_agent_sdk._workflow import (
    SEGMENT_ACTIVITY_NAME,
    TOOL_STEP_ACTIVITY_NAME,
    DurableClaudeAgent,
    DurableTool,
    activity_as_tool,
)
from temporalio.claude_agent_sdk._workspace import FileWorkspaceStore, WorkspaceStore

from . import testing

__all__ = [
    "SEGMENT_ACTIVITY_NAME",
    "TOOL_CALL_INTERRUPTED",
    "TOOL_CALL_NOT_RUN",
    "TOOL_STEP_ACTIVITY_NAME",
    "WORKSPACE_CORRUPT",
    "WORKSPACE_INVALID",
    "WORKSPACE_NOT_FOUND",
    "WORKSPACE_TOO_LARGE",
    "AgentState",
    "ClaudeAgentPlugin",
    "ClaudeAgentSdkRunner",
    "ConversationRef",
    "DeferredCall",
    "DurableClaudeAgent",
    "DurableTool",
    "FileSessionStore",
    "FileWorkspaceStore",
    "SegmentInput",
    "SegmentOutput",
    "SegmentRunner",
    "ToolOutcome",
    "ToolSpec",
    "ToolStepInput",
    "ToolStepRetry",
    "Workspace",
    "WorkspaceStore",
    "activity_as_tool",
    "follow_agent",
    "make_segment_activity",
    "make_tool_step_activity",
    "testing",
]
