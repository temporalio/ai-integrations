# pyright: reportUnusedFunction=false

"""Checks on the MCPServer that a worker's factory returns.

The agent reads workflow-facing settings from the workflow-side server it
holds, never from the factory's server. A setting there that the plugin can
neither apply nor safely ignore is rejected with a non-retryable error.

This module needs no ``mcp`` extra, so both the MCP v2 backend and the
deprecated providers can use it.
"""

import inspect

from agents.mcp import MCPServer

from temporalio.exceptions import ApplicationError

# How the Agents SDK evaluates an approval policy; a subclass that replaces it
# cannot be evaluated by looking at the policy alone.
_SDK_APPROVAL_EVALUATOR = getattr(MCPServer, "_get_needs_approval_for_tool", None)


def _has_dynamic_tool_filter(server: MCPServer) -> bool:
    """Return whether the server has a callable ``tool_filter``."""
    return callable(getattr(server, "tool_filter", None))


def _reject_dynamic_tool_filter(name: str, server: MCPServer) -> None:
    """Fail fast on a worker-side filter that can never be applied.

    A callable ``tool_filter`` needs the run context and agent, which exist only
    in the workflow, so the Agents SDK rejects one on the worker side and the
    list-tools Activity would otherwise retry forever. A static filter needs
    neither and is applied here as usual.
    """
    if _has_dynamic_tool_filter(server):
        raise ApplicationError(
            f"MCP server {name!r} sets a callable tool_filter on its worker-side "
            "MCPServer, which cannot be applied there because the run context "
            "and agent it receives exist only in the workflow; pass the filter "
            "to temporal_mcp_server() instead",
            non_retryable=True,
        )


def _has_worker_side_require_approval(server: MCPServer) -> bool:
    """Return whether the server's approval policy can ask for approval.

    Only a policy the Agents SDK normalizes to no approval is allowed:
    ``"never"``, ``False``, or a mapping or tool list with only ``"never"``
    entries. A callable policy, or a class or instance that replaces how the
    SDK evaluates the policy, cannot be evaluated on the worker, so it counts
    as asking.
    """
    # The Agents SDK keeps the normalized policy only in this private field:
    # a bool, a dict of tool names to bools, or a callable.
    policy = getattr(server, "_needs_approval_policy", False)
    if isinstance(policy, dict):
        policy = any(policy.values())
    # Find the evaluator the SDK's call would resolve, an instance attribute
    # before the class's method, without running it or any descriptor. A
    # custom __getattribute__ could resolve something else, so it counts too.
    evaluator = inspect.getattr_static(server, "_get_needs_approval_for_tool", None)
    overridden = (
        evaluator is not None and evaluator is not _SDK_APPROVAL_EVALUATOR
    ) or type(server).__getattribute__ is not object.__getattribute__
    return callable(policy) or bool(policy) or overridden


def _reject_worker_side_require_approval(
    name: str, server: MCPServer, route: str
) -> None:
    """Raise when the factory's server has an approval policy; see the message.

    ``route`` names the workflow-side function that takes the policy instead.
    """
    if _has_worker_side_require_approval(server):
        raise ApplicationError(
            f"MCP server {name!r} sets require_approval on its worker-side "
            "MCPServer; the agent never reads that copy, so the policy takes "
            f"effect only when passed to {route}(require_approval=...); remove "
            "it from the factory's server",
            non_retryable=True,
        )
