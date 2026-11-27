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

Deep Agents compiles a declarative sub-agent again when a task call carries a
response schema in its config, long after ``create_temporal_deep_agent`` has
returned. A ``task`` tool built inside the approval scope therefore puts that
one compilation back in the scope, so the sub-agent it compiles then gets the
adapter too; what the sub-agent then runs, including an agent one of its tools
builds, is outside the scope and keeps its own setting.
"""

from __future__ import annotations

import contextvars
import importlib
import inspect
from collections.abc import Awaitable, Callable, Generator, Mapping
from contextlib import contextmanager
from functools import wraps
from typing import TYPE_CHECKING, Any

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from langchain.agents.middleware import AgentMiddleware, HumanInTheLoopMiddleware
    from langchain.agents.structured_output import (
        AutoStrategy,
        ProviderStrategy,
        ToolStrategy,
    )
    from langchain_core.messages import AIMessage, ToolMessage
    from langgraph.runtime import get_runtime

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
# The Deep Agents function that builds the ``task`` tool, which compiles a
# sub-agent again with ``_SUBAGENT_BUILDER`` when a task call carries a
# response schema.
_TASK_TOOL_MODULE = "deepagents.middleware.subagents"
_TASK_TOOL_BUILDER = "_build_task_tool"
_SUBAGENT_BUILDER = "create_sub_agent"
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
    """The handler and the approvals of one ``create_temporal_deep_agent`` call."""

    def __init__(self, handler: ApprovalHandler) -> None:
        self.handler = handler
        self._unnamed = 0
        # Every request id handed out, pending or decided, so a decision can
        # only ever match the action the reviewer saw.
        self._issued: set[str] = set()
        # Decisions by tool call object, so a call that tool middleware retries
        # is not asked again. The call is kept so its id() is not reused.
        self._decided: dict[int, tuple[Mapping[str, Any], str, Any]] = {}

    def request_id(self, call: Mapping[str, Any]) -> str:
        """The model's tool call id, made unique among every id this scope issued.

        A model can reuse a call id in a later turn, so a repeat gets ``#2``,
        ``#3`` and so on. Ids follow the order in which the workflow asks,
        which replays deterministically.
        """
        base = call.get("id") or ""
        if not base:
            self._unnamed += 1
            base = f"{call['name']}-{self._unnamed}"
        request_id, n = base, 1
        while request_id in self._issued:
            n += 1
            request_id = f"{base}#{n}"
        self._issued.add(request_id)
        return request_id

    def decision_for(self, call: Mapping[str, Any]) -> tuple[str, Any] | None:
        """The request id and decision already given for this tool call."""
        known = self._decided.get(id(call))
        if known is None or known[0] is not call:
            return None
        return known[1], known[2]

    def record(self, call: Mapping[str, Any], request_id: str, decision: Any) -> None:
        """Remember the decision for this tool call, for a retry of the call."""
        self._decided[id(call)] = (call, request_id, decision)


class _PerCallApprovalMiddleware(HumanInTheLoopMiddleware):
    """``HumanInTheLoopMiddleware`` that asks the approval handler per tool call."""

    # Nothing is batched or interrupted after the model call; the after-model
    # hooks only refuse a guarded structured-output call (see _refuse_...).
    def after_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        """Refuse a guarded structured-output call: it never reaches a tool call."""
        _refuse_guarded_structured_output(self, state, runtime)
        return None

    async def aafter_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        """Refuse a guarded structured-output call: it never reaches a tool call."""
        _refuse_guarded_structured_output(self, state, runtime)
        return None

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
        if getattr(request.tool, "return_direct", False):
            raise ValueError(_RETURN_DIRECT_REFUSAL.format(name=tool_call["name"]))
        # Tool middleware outside this one (ToolRetryMiddleware) can run the
        # same call again; native HITL reviews a call once, so reuse the decision.
        known = self._scope.decision_for(tool_call)
        if known is None:
            decision = await self._ask(request, tool_call, config)
        else:
            _, decision = known
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

    async def _ask(self, request: Any, tool_call: Any, config: Any) -> Any:
        """Ask the handler about the call and return its one decision."""
        # A description callable gets the graph Runtime, as it does natively.
        action, review = self._create_action_and_config(
            tool_call, config, request.state, _graph_runtime(request)
        )
        hitl_request: HITLRequest = {
            "action_requests": [action],
            "review_configs": [review],
        }
        request_id = self._scope.request_id(tool_call)
        response = await self._scope.handler(request_id, hitl_request)
        decisions = response.get("decisions") if isinstance(response, Mapping) else None
        if not isinstance(decisions, list) or len(decisions) != 1:
            raise ValueError(
                "approval_handler must return a HITLResponse with exactly one "
                f"decision for tool call {request_id!r}; got {response!r}."
            )
        self._scope.record(tool_call, request_id, decisions[0])
        return decisions[0]

    def wrap_tool_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        """Refuse a guarded call in the sync loop: waiting needs the workflow's event loop."""
        if self._config_for(request) is None:
            return handler(request)
        raise RuntimeError(
            "approval_handler needs the async agent loop (ainvoke); a guarded tool "
            "call cannot wait for a decision in a sync invoke."
        )


def _graph_runtime(request: Any) -> Any:
    """The graph ``Runtime`` of the running node, as native HITL passes it.

    Falls back to the tool call's ``ToolRuntime`` outside a graph run.
    """
    try:
        runtime = get_runtime()
    except (RuntimeError, KeyError):
        runtime = None
    return runtime if runtime is not None else request.runtime


_STRUCTURED_OUTPUT_REFUSAL = (
    "approval_handler cannot gate the structured-output tool {name!r}: LangChain "
    "returns structured output from the model call, so the call never reaches "
    "the tool-call step where the handler waits. Remove {name!r} from "
    "interrupt_on, use the native interrupt (no approval_handler), or use a "
    "ProviderStrategy response format."
)


_RETURN_DIRECT_REFUSAL = (
    "approval_handler cannot gate the return_direct tool {name!r}: a decision "
    "that skips the tool (reject or respond) would end the agent as if the "
    "tool had run, where the native interrupt continues to the model. Remove "
    "{name!r} from interrupt_on, use the native interrupt (no "
    "approval_handler), or set return_direct=False."
)

_TOOL_WRAPPER_REFUSAL = (
    "approval_handler cannot be combined with {where} that wraps tool calls "
    "({name}): it may run a tool without calling the next handler, which "
    "would skip the approval. Remove its wrap_tool_call/awrap_tool_call, or "
    "use the native interrupt (no approval_handler)."
)


def _refuse_guarded_structured_output(
    middleware: _PerCallApprovalMiddleware, state: Any, runtime: Any
) -> None:
    """Raise if the model just made a structured-output call that is guarded.

    Catches the response formats chosen at run time (a sub-agent's response
    schema passed with the task); the formats known when the agent is built
    are refused by :func:`check_approval_config`. Only the structured-output
    calls are considered: LangChain answers each of them with a tool message
    in the model step, whether it returns the structured response or asks the
    model to retry a call that failed validation, while any other call in the
    message still goes through the tool node, where the handler asks.
    """
    messages = state.get("messages") or []
    last_ai = next(
        (
            i
            for i in range(len(messages) - 1, -1, -1)
            if isinstance(messages[i], AIMessage)
        ),
        None,
    )
    if last_ai is None:
        return
    answered = {
        (m.tool_call_id, m.name)
        for m in messages[last_ai + 1 :]
        if isinstance(m, ToolMessage)
    }
    for call in getattr(messages[last_ai], "tool_calls", None) or ():
        if (call.get("id"), call.get("name")) not in answered:
            continue
        config = middleware.interrupt_on.get(call["name"])
        if config is not None and middleware._should_interrupt(
            call, config, state, runtime
        ):
            raise ValueError(_STRUCTURED_OUTPUT_REFUSAL.format(name=call["name"]))


def _structured_output_tool_names(response_format: Any) -> set[str]:
    """Names of the tools LangChain may use to return ``response_format``."""
    if response_format is None or isinstance(response_format, ProviderStrategy):
        return set()
    if not isinstance(response_format, ToolStrategy):
        # A bare schema or AutoStrategy becomes a ToolStrategy for models
        # without native structured output.
        schema = (
            response_format.schema
            if isinstance(response_format, AutoStrategy)
            else response_format
        )
        response_format = ToolStrategy(schema=schema)
    return {spec.name for spec in response_format.schema_specs}


def _guarded_names(interrupt_on: Any) -> set[str]:
    return {name for name, config in (interrupt_on or {}).items() if config}


class _HumanInTheLoopSwitch(HumanInTheLoopMiddleware):
    """Installed in place of ``HumanInTheLoopMiddleware`` in Deep Agents' builders.

    While a ``create_temporal_deep_agent`` call with an ``approval_handler``
    runs on this thread it builds the per-call adapter; otherwise it builds the
    native middleware, so nothing changes for other agents.
    """

    _temporal_approval_scope: contextvars.ContextVar[_ApprovalScope | None] = (
        contextvars.ContextVar("temporal_deepagents_approval_scope", default=None)
    )
    # Set while a task tool runs: the scope its one run-time compilation uses.
    _temporal_compile_scope: contextvars.ContextVar[_PendingCompile | None] = (
        contextvars.ContextVar("temporal_deepagents_compile_scope", default=None)
    )

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        """Build the per-call adapter inside an approval scope, else the native middleware."""
        scope = cls._temporal_approval_scope.get()
        if scope is None:
            return HumanInTheLoopMiddleware(*args, **kwargs)
        return _PerCallApprovalMiddleware(scope, *args, **kwargs)


class _PendingCompile:
    """The approval scope for the sub-agent one task call may compile."""

    def __init__(self, scope: _ApprovalScope | None) -> None:
        self.scope = scope


def _with_compile_scope(
    fn: Callable[..., Any] | None,
    var: contextvars.ContextVar[_PendingCompile | None],
    scope: _ApprovalScope | None,
) -> Callable[..., Any] | None:
    """Wrap a task tool function so its run-time compilation uses ``scope``.

    Each run gets a fresh :class:`_PendingCompile`; a task tool built outside
    an approval scope sets none, so it never compiles with an outer scope.
    """
    if fn is None:
        return None
    if inspect.iscoroutinefunction(fn):

        @wraps(fn)
        async def run_async(*args: Any, **kwargs: Any) -> Any:
            token = var.set(_PendingCompile(scope) if scope is not None else None)
            try:
                return await fn(*args, **kwargs)
            finally:
                var.reset(token)

        return run_async

    @wraps(fn)
    def run(*args: Any, **kwargs: Any) -> Any:
        token = var.set(_PendingCompile(scope) if scope is not None else None)
        try:
            return fn(*args, **kwargs)
        finally:
            var.reset(token)

    return run


def _install_task_tool_scope(
    var: contextvars.ContextVar[_ApprovalScope | None],
    compile_var: contextvars.ContextVar[_PendingCompile | None],
) -> None:
    """Make a ``task`` tool built in an approval scope compile in that scope.

    Deep Agents compiles a declarative sub-agent again, building its
    ``HumanInTheLoopMiddleware`` then, when a task call carries a response
    schema in its config. That happens when the tool runs, after
    :func:`approval_scope` has ended. Only that compilation runs in the scope:
    the sub-agent's run, and any agent built while it runs, does not.
    """
    module = importlib.import_module(_TASK_TOOL_MODULE)
    build = getattr(module, _TASK_TOOL_BUILDER, None)
    create = getattr(module, _SUBAGENT_BUILDER, None)
    if build is None or create is None:
        raise ValueError(
            "approval_handler does not support this deepagents version: "
            f"{_TASK_TOOL_MODULE} does not build the task tool with "
            f"{_TASK_TOOL_BUILDER} and sub-agents with {_SUBAGENT_BUILDER}."
        )
    if not getattr(build, _SCOPE_ATTR, False):

        @wraps(build)
        def build_task_tool(*args: Any, **kwargs: Any) -> Any:
            tool = build(*args, **kwargs)
            scope = var.get()
            tool.func = _with_compile_scope(tool.func, compile_var, scope)
            tool.coroutine = _with_compile_scope(tool.coroutine, compile_var, scope)
            return tool

        setattr(build_task_tool, _SCOPE_ATTR, True)
        setattr(module, _TASK_TOOL_BUILDER, build_task_tool)
    if not getattr(create, _SCOPE_ATTR, False):

        @wraps(create)
        def create_sub_agent(*args: Any, **kwargs: Any) -> Any:
            pending = compile_var.get()
            # The task tool compiles at run time only for a response schema,
            # and does so once per call.
            if (
                pending is None
                or pending.scope is None
                or kwargs.get("response_format") is None
            ):
                return create(*args, **kwargs)
            scope, pending.scope = pending.scope, None
            token = var.set(scope)
            try:
                return create(*args, **kwargs)
            finally:
                var.reset(token)

        setattr(create_sub_agent, _SCOPE_ATTR, True)
        setattr(module, _SUBAGENT_BUILDER, create_sub_agent)


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
    _install_task_tool_scope(
        switch._temporal_approval_scope, switch._temporal_compile_scope
    )
    return switch


def _wraps_tool_calls(middleware: Any) -> bool:
    # Look at the instance, not only the class: a hook can be assigned per
    # instance or delegated to another object.
    for hook in ("wrap_tool_call", "awrap_tool_call"):
        impl = getattr(middleware, hook, None)
        if getattr(impl, "__func__", impl) is not getattr(AgentMiddleware, hook):
            return True
    return False


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
    interrupt_on = kwargs.get("interrupt_on")
    tools = kwargs.get("tools")
    agents = [("the agent", kwargs.get("response_format"), interrupt_on, tools)]
    agents += [
        (
            f"sub-agent {spec.get('name')!r}",
            spec.get("response_format"),
            spec.get("interrupt_on", interrupt_on),
            spec.get("tools", tools),
        )
        for spec in kwargs.get("subagents") or ()
        if isinstance(spec, Mapping)
    ]
    for who, response_format, policy, agent_tools in agents:
        for tool in agent_tools or ():
            name = getattr(tool, "name", None)
            if getattr(tool, "return_direct", False) and name in _guarded_names(policy):
                raise ValueError(
                    f"In {who}: " + _RETURN_DIRECT_REFUSAL.format(name=name)
                )
        both = _structured_output_tool_names(response_format) & _guarded_names(policy)
        if both:
            raise ValueError(
                f"In {who}: " + _STRUCTURED_OUTPUT_REFUSAL.format(name=sorted(both)[0])
            )
    if _has_hitl(kwargs.get("middleware")):
        raise ValueError(
            "approval_handler cannot be combined with a HumanInTheLoopMiddleware "
            "in middleware=; configure approvals with interrupt_on."
        )
    # Approval runs inside the tool-call chain, after middleware from
    # middleware=; one that runs a tool itself never calls the next handler.
    for where, wrappers in [("middleware=", kwargs.get("middleware"))] + [
        (f"sub-agent {spec.get('name')!r}'s middleware", spec.get("middleware"))
        for spec in kwargs.get("subagents") or ()
        if isinstance(spec, Mapping)
    ]:
        for wrapper in wrappers or ():
            if _wraps_tool_calls(wrapper):
                raise ValueError(
                    _TOOL_WRAPPER_REFUSAL.format(
                        where=where, name=type(wrapper).__name__
                    )
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
