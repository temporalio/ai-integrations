"""Per-call tool approval for ``create_temporal_deep_agent(approval_handler=...)``.

``interrupt_on=`` pauses the whole graph: LangGraph raises the interrupt at the
end of the step, so when a sub-agent asks for approval while sibling sub-agents
run in the same step (parallel ``task`` calls), nothing is pending until the
slowest sibling finishes, and the resume waits for every decision.

With an ``approval_handler``, ``interrupt_on`` stays the approval policy, but
each guarded tool call waits on its own: Deep Agents builds its
``HumanInTheLoopMiddleware`` for the main agent and every declarative sub-agent
as usual, and :class:`_PerCallApprovalMiddleware` takes its place. It keeps the
native configuration (decision options, ``description``, ``when`` predicates)
and request/decision formats, but asks the handler from inside the tool call
instead of raising a graph interrupt after the model call.
"""

from __future__ import annotations

import contextvars
import importlib
from collections.abc import Awaitable, Callable, Generator, Mapping
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from langchain.agents.middleware import AgentMiddleware, HumanInTheLoopMiddleware

if TYPE_CHECKING:
    from langchain.agents.middleware.human_in_the_loop import (
        HITLRequest,
        HITLResponse,
    )

ApprovalHandler = Callable[[str, "HITLRequest"], Awaitable["HITLResponse"]]
"""``approval_handler(request_id, request) -> response``.

``request`` is a native ``HITLRequest`` with one action; the response is a
native ``HITLResponse`` with one decision.
"""

# Deep Agents modules that build HumanInTheLoopMiddleware from interrupt_on.
_HITL_BUILDERS = ("deepagents.graph", "deepagents.middleware.subagents")
# HumanInTheLoopMiddleware helpers the adapter reuses, so a decision is applied
# exactly as the native middleware applies it.
_NATIVE_HELPERS = (
    "_create_action_and_config",
    "_process_decision",
    "_apply_edit",
    "_annotate_edited_result",
)
_SCOPE_ATTR = "_temporal_approval_scope"


class _ApprovalScope:
    """The handler and waiting request ids of one ``create_temporal_deep_agent`` call."""

    def __init__(self, handler: ApprovalHandler) -> None:
        self.handler = handler
        self._unnamed = 0
        self._waiting: set[str] = set()

    def request_id(self, call: Mapping[str, Any]) -> str:
        """The model's tool call id, made unique among the calls now waiting."""
        base = call.get("id") or ""
        if not base:
            self._unnamed += 1
            base = f"{call['name']}-{self._unnamed}"
        request_id, n = base, 1
        while request_id in self._waiting:
            n += 1
            request_id = f"{base}#{n}"
        return request_id

    @contextmanager
    def waiting(self, request_id: str) -> Generator[None]:
        """Mark ``request_id`` as waiting for a decision for the block."""
        self._waiting.add(request_id)
        try:
            yield
        finally:
            self._waiting.discard(request_id)


class _PerCallApprovalMiddleware(HumanInTheLoopMiddleware):
    """``HumanInTheLoopMiddleware`` that asks the approval handler per tool call."""

    # No after_model hook: nothing is batched or interrupted after the model call.
    after_model = AgentMiddleware.after_model  # type: ignore[assignment]
    aafter_model = AgentMiddleware.aafter_model  # type: ignore[assignment]

    def __init__(self, scope: _ApprovalScope, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._scope = scope

    def _config_for(self, request: Any) -> Any:
        """The call's interrupt_on config, or None when the call is not guarded."""
        config = self.interrupt_on.get(request.tool_call["name"])
        if config is None:
            return None
        when = config.get("when")
        if when is not None and not when(request):
            return None
        return config

    async def awrap_tool_call(
        self, request: Any, handler: Callable[[Any], Awaitable[Any]]
    ) -> Any:
        """Ask the approval handler about a guarded call, then apply its decision."""
        config = self._config_for(request)
        if config is None:
            return await handler(request)
        tool_call = request.tool_call
        action, review = self._create_action_and_config(
            tool_call, config, request.state, request.runtime
        )
        hitl_request: HITLRequest = {
            "action_requests": [action],
            "review_configs": [review],
        }
        request_id = self._scope.request_id(tool_call)
        with self._scope.waiting(request_id):
            response = await self._scope.handler(request_id, hitl_request)
        decisions = response.get("decisions") if isinstance(response, Mapping) else None
        if not isinstance(decisions, list) or len(decisions) != 1:
            raise ValueError(
                "approval_handler must return a HITLResponse with exactly one "
                f"decision for tool call {request_id!r}; got {response!r}."
            )
        decision = decisions[0]
        # Raises, as the native middleware does, for a decision type the tool's
        # allowed_decisions does not allow.
        _, tool_message = self._process_decision(decision, tool_call, config)
        if tool_message is not None:  # reject or respond: the tool does not run
            return tool_message
        if decision["type"] == "edit":
            executed = decision["edited_action"]
            request = self._apply_edit(request, executed)
            return self._annotate_edited_result(
                await handler(request), request, executed
            )
        return await handler(request)

    def wrap_tool_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        """Refuse a guarded call in the sync loop: waiting needs the workflow's event loop."""
        if self._config_for(request) is None:
            return handler(request)
        raise RuntimeError(
            "approval_handler needs the async agent loop (ainvoke); a guarded tool "
            "call cannot wait for a decision in a sync invoke."
        )


class _HumanInTheLoopSwitch(HumanInTheLoopMiddleware):
    """Installed in place of ``HumanInTheLoopMiddleware`` in Deep Agents' builders.

    While a ``create_temporal_deep_agent`` call with an ``approval_handler``
    runs on this thread it builds the per-call adapter; otherwise it builds the
    native middleware, so nothing changes for other agents.
    """

    _temporal_approval_scope: contextvars.ContextVar[_ApprovalScope | None] = (
        contextvars.ContextVar("temporal_deepagents_approval_scope", default=None)
    )

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        """Build the per-call adapter inside an approval scope, else the native middleware."""
        scope = cls._temporal_approval_scope.get()
        if scope is None:
            return HumanInTheLoopMiddleware(*args, **kwargs)
        return _PerCallApprovalMiddleware(scope, *args, **kwargs)


def _installed_switch() -> Any:
    """Install the switch in Deep Agents' builders (once per process) and return it.

    A switch that is already installed is reused even when this module was
    imported again (the workflow sandbox re-imports it), so every workflow
    shares one context variable.
    """
    modules = [importlib.import_module(name) for name in _HITL_BUILDERS]
    for module in modules:
        if not hasattr(module, "HumanInTheLoopMiddleware"):
            raise ValueError(
                "approval_handler does not support this deepagents version: "
                f"{module.__name__} does not build HumanInTheLoopMiddleware."
            )
    switch = next(
        (
            module.HumanInTheLoopMiddleware
            for module in modules
            if hasattr(module.HumanInTheLoopMiddleware, _SCOPE_ATTR)
        ),
        _HumanInTheLoopSwitch,
    )
    for module in modules:
        if not hasattr(module.HumanInTheLoopMiddleware, _SCOPE_ATTR):
            setattr(module, "HumanInTheLoopMiddleware", switch)
    return switch


def _has_hitl(middleware: Any) -> bool:
    return any(isinstance(m, HumanInTheLoopMiddleware) for m in middleware or ())


def check_approval_config(kwargs: Mapping[str, Any]) -> None:
    """Fail at construction for a setup the approval handler cannot serve."""
    missing = [h for h in _NATIVE_HELPERS if not hasattr(HumanInTheLoopMiddleware, h)]
    if missing:
        raise ValueError(
            "approval_handler needs a langchain HumanInTheLoopMiddleware that "
            "applies decisions per tool call; this langchain version lacks "
            f"{', '.join(missing)}. Upgrade langchain."
        )
    if _has_hitl(kwargs.get("middleware")):
        raise ValueError(
            "approval_handler cannot be combined with a HumanInTheLoopMiddleware "
            "in middleware=; configure approvals with interrupt_on."
        )
    for spec in kwargs.get("subagents") or ():
        if isinstance(spec, Mapping) and _has_hitl(spec.get("middleware")):
            raise ValueError(
                "approval_handler cannot be combined with a "
                f"HumanInTheLoopMiddleware in sub-agent {spec.get('name')!r}'s "
                "middleware; configure it with the sub-agent's interrupt_on."
            )


@contextmanager
def approval_scope(handler: ApprovalHandler) -> Generator[None]:
    """Build every ``HumanInTheLoopMiddleware`` in this block as a per-call adapter."""
    var = _installed_switch()._temporal_approval_scope
    token = var.set(_ApprovalScope(handler))
    try:
        yield
    finally:
        var.reset(token)
