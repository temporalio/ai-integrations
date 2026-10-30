"""Per-call tool approval held in the workflow, without a graph interrupt.

``interrupt_on=`` pauses the whole graph: LangGraph raises the interrupt at the
end of the step, so when a sub-agent asks for approval while sibling sub-agents
run in the same step (parallel ``task`` calls), nothing is pending until the
slowest sibling finishes, and the resume waits for every decision.
:class:`ToolApprovals` holds one tool call instead: its middleware records the
call as pending and waits in the workflow (``workflow.wait_condition``) for a
decision delivered by an Update, while the rest of the agent keeps running.
"""

from __future__ import annotations

import copy
import dataclasses
from collections.abc import Awaitable, Callable, Iterable
from typing import Any

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from langchain.agents.middleware import AgentMiddleware
    from langchain_core.messages import ToolMessage


@dataclasses.dataclass
class _Decision:
    approved: bool
    reason: str | None


class ToolApprovals:
    """Make chosen tool calls wait for a human decision, one call at a time.

    Create one on the workflow instance, add :meth:`middleware` to each agent
    or sub-agent whose tools need approval (like ``interrupt_on``, by tool
    name), and expose :meth:`pending_approvals` from a Query and
    :meth:`decide` from an Update, with :meth:`validate_decision` as its
    validator::

        @workflow.defn
        class Agent:
            def __init__(self) -> None:
                self.approvals = ToolApprovals()

            @workflow.run
            async def run(self, request: str) -> str:
                agent = create_temporal_deep_agent(
                    model="anthropic:claude-sonnet-4-5",
                    tools=[activity_as_tool(issue_refund, start_to_close_timeout=...)],
                    middleware=[self.approvals.middleware(["issue_refund"])],
                )
                ...

            @workflow.query
            def pending_approvals(self) -> list[dict[str, Any]]:
                return self.approvals.pending_approvals()

            @workflow.update
            def decide(self, request_id: str, approved: bool) -> None:
                self.approvals.decide(request_id, approved)

            @decide.validator
            def check_decide(self, request_id: str, approved: bool) -> None:
                self.approvals.validate_decision(request_id)

    A call is identified by the model's tool call id, which is part of the
    recorded model output and so replays identically (a missing or repeated
    id gets a suffix in arrival order). An approved call runs
    as it would without the middleware; a rejected one returns an error
    ``ToolMessage`` to the model, and the tool does not run. Decisions come
    from history, so a gated call replays without waiting.
    """

    def __init__(self) -> None:
        """Start with no pending requests."""
        self._requests: dict[str, dict[str, Any]] = {}
        self._decisions: dict[str, _Decision] = {}
        self._fallback_ids = 0

    def middleware(self, tools: Iterable[str]) -> AgentMiddleware:
        """Agent middleware that holds calls to the named tools for a decision.

        ``tools`` is a list of tool names; a bare string or an empty list is
        refused, since either would hold nothing.
        """
        if isinstance(tools, (str, bytes)):
            raise TypeError("tools must be a list of tool names, not one string.")
        names = frozenset(tools)
        if not names:
            raise ValueError("tools must name at least one tool.")
        return _ToolApprovalMiddleware(self, names)

    def pending_approvals(self) -> list[dict[str, Any]]:
        """The calls waiting for a decision, oldest first, for a Query.

        Each entry is a copy: ``request_id`` (the tool call id), ``tool`` and
        ``args`` as the model wrote them.
        """
        return [copy.deepcopy(r) for r in self._requests.values()]

    def validate_decision(self, request_id: str) -> None:
        """Raise ``ValueError`` unless ``request_id`` is waiting for a decision.

        Use it as the Update's validator, so a decision for an unknown or
        already decided request is refused and never reaches history.
        """
        if request_id not in self._requests or request_id in self._decisions:
            raise ValueError(f"No pending tool call {request_id!r}.")

    def decide(
        self, request_id: str, approved: bool, reason: str | None = None
    ) -> bool:
        """Approve or reject a pending call; return whether it was recorded.

        A decision for a call that is not pending is not recorded, and not
        kept for later, which is why the Update needs
        :meth:`validate_decision`.
        """
        if request_id not in self._requests or request_id in self._decisions:
            return False
        self._decisions[request_id] = _Decision(approved, reason)
        return True

    def _request_id(self, call: Any) -> str:
        base = call.get("id") or ""
        if not base:
            self._fallback_ids += 1
            base = f"{call['name']}-{self._fallback_ids}"
        request_id, n = base, 1
        while request_id in self._requests:
            n += 1
            request_id = f"{base}#{n}"
        return request_id

    async def _hold(
        self, request: Any, handler: Callable[[Any], Awaitable[Any]]
    ) -> Any:
        """Record ``request``'s tool call as pending and run it once approved."""
        call = request.tool_call
        request_id = self._request_id(call)
        self._requests[request_id] = {
            "request_id": request_id,
            "tool": call["name"],
            "args": copy.deepcopy(call.get("args") or {}),
        }
        try:
            await workflow.wait_condition(lambda: request_id in self._decisions)
            decision = self._decisions[request_id]
        finally:
            # Also on cancellation: nothing stays pending for a call that is gone.
            self._requests.pop(request_id, None)
            self._decisions.pop(request_id, None)
        if decision.approved:
            return await handler(request)
        reason = f": {decision.reason}" if decision.reason else ""
        return ToolMessage(
            content=f"The call to {call['name']} was rejected{reason}. It did not run.",
            tool_call_id=call.get("id") or request_id,
            name=call["name"],
            status="error",
        )


class _ToolApprovalMiddleware(AgentMiddleware):
    """Hold calls to the named tools until :class:`ToolApprovals` has a decision."""

    def __init__(self, approvals: ToolApprovals, tools: frozenset[str]) -> None:
        super().__init__()
        self._approvals = approvals
        self._tools = tools

    async def awrap_tool_call(
        self, request: Any, handler: Callable[[Any], Awaitable[Any]]
    ) -> Any:
        """Wait for a decision on a gated tool call; pass other calls through."""
        if request.tool_call["name"] not in self._tools:
            return await handler(request)
        return await self._approvals._hold(request, handler)

    def wrap_tool_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        """Refuse the sync loop: waiting for a decision needs the workflow's event loop."""
        del request, handler
        raise RuntimeError(
            "ToolApprovals needs the async agent loop (ainvoke); it cannot hold a "
            "tool call in a sync invoke."
        )
