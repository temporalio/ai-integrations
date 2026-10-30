import asyncio
import contextlib
import contextvars
import sys
from collections.abc import AsyncGenerator, Callable, Generator
from datetime import timedelta
from types import FrameType
from typing import Any

from strands import Agent
from strands.agent._concurrency import _ConcurrencyController, _InflightInvocation
from strands.agent.conversation_manager import NullConversationManager
from strands.hooks import BeforeModelCallEvent, HookCallback
from strands.types.agent import AgentInput
from strands.types.content import Messages

from temporalio import workflow
from temporalio.common import Priority, RetryPolicy
from temporalio.workflow import ActivityCancellationType, VersioningIntent

from ._temporal_mcp_client import TemporalMCPClient
from ._temporal_model import TemporalModel
from ._temporal_sandbox import TemporalSandbox

_SNAPSHOT_DISABLED = (
    "TemporalAgent disables take_snapshot()/load_snapshot(). Temporal "
    "workflows already persist agent state durably via the event history at "
    "a finer granularity than Strands snapshots. Remove the snapshot call "
    "and rely on Temporal's durable execution instead."
)


class _WorkflowInflightInvocation(_InflightInvocation):
    def register_waiter(self) -> asyncio.Future:
        # Strands uses asyncio.wrap_future to notify duplicates across threads.
        # On Python <3.13 that calls loop methods the Workflow loop does not
        # implement. Workflow invocations share one loop, so wait on its state.
        return asyncio.create_task(workflow.wait_condition(lambda: self.settled))


class _WorkflowConcurrencyController(_ConcurrencyController):
    def _check_idempotency(
        self, idempotency_token: Any
    ) -> tuple[_InflightInvocation | None, Any]:
        waiting_on, registered_token = super()._check_idempotency(idempotency_token)
        if registered_token is not None:
            self._inflight = _WorkflowInflightInvocation()
        return waiting_on, registered_token


# Strands runs a sync tool function with asyncio.to_thread in these modules.
_SYNC_TOOL_MODULES = ("strands.tools.decorator", "strands.tools.tools")

# True while a TemporalAgent's event loop, which calls its tools, takes a step.
_in_agent_loop: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_in_agent_loop", default=False
)

# The callers whose run_in_executor calls the Workflow loop runs inline. The
# Deep Agents plugin adds its caller to the same list on the same loop, so with
# both plugins one wrapper checks the original caller against both.
_INLINE_CALLERS = "_temporal_inline_executor_callers"


@contextlib.contextmanager
def _agent_loop_step() -> Generator[None, None, None]:
    """Run the sync tools called in this step directly in the Workflow.

    Strands runs a sync tool function with ``asyncio.to_thread``. The Workflow
    event loop has no threads, so the tool failed before its body ran and the
    model got ``Error: NotImplementedError - ``. Tools are not changed, only how
    the loop runs that call. A step never spans an ``await`` of the caller, so
    it sets and resets the flag in one task even when a stream moves between
    tasks.
    """
    token = _in_agent_loop.set(True)
    try:
        yield
    finally:
        _in_agent_loop.reset(token)


def _run_inline_when_called_by(
    loop: asyncio.AbstractEventLoop, is_caller: Callable[[FrameType], bool]
) -> None:
    """Make the Workflow loop run ``run_in_executor`` inline for this caller."""
    callers: list[Callable[[FrameType], bool]] | None = getattr(
        loop, _INLINE_CALLERS, None
    )
    if callers is None:
        callers = []
        run_in_executor = loop.run_in_executor

        def inline_run_in_executor(
            executor: Any, func: Callable[..., Any], *args: Any
        ) -> asyncio.Future[Any]:
            caller = sys._getframe(1)
            if not any(called_by(caller) for called_by in callers):
                return run_in_executor(executor, func, *args)
            future = loop.create_future()
            future.set_result(func(*args))
            return future

        setattr(loop, "run_in_executor", inline_run_in_executor)
        setattr(loop, _INLINE_CALLERS, callers)
    if is_caller not in callers:
        callers.append(is_caller)


def _called_by_sync_tool(frame: FrameType) -> bool:
    """Whether this is ``asyncio.to_thread`` called by Strands to run a sync tool.

    Those are the only ``asyncio.to_thread`` calls in ``_SYNC_TOOL_MODULES``.
    """
    return (
        _in_agent_loop.get()
        and frame.f_code is asyncio.to_thread.__code__
        and frame.f_back is not None
        and frame.f_back.f_globals.get("__name__") in _SYNC_TOOL_MODULES
    )


class TemporalAgent(Agent):
    """A Strands ``Agent`` that routes model calls through a Temporal activity.

    ``model`` is the name of a factory registered in
    ``StrandsPlugin(models={...})``. The activity options apply to every model
    invocation this agent makes. ``sandbox`` is a workflow-side
    ``TemporalSandbox`` whose name selects a worker-side factory. All other
    keyword arguments are forwarded to Strands' ``Agent`` (``tools``, ``hooks``, ``system_prompt``,
    ``structured_output_model``, ``messages``, etc.).

    Strands' ``retry_strategy`` is disabled; configure retries via
    ``retry_policy`` here and on the activity options accepted by
    ``activity_as_tool``, ``activity_as_hook``, and ``TemporalMCPClient``.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        task_queue: str | None = None,
        schedule_to_close_timeout: timedelta | None = None,
        schedule_to_start_timeout: timedelta | None = None,
        start_to_close_timeout: timedelta | None = None,
        heartbeat_timeout: timedelta | None = None,
        retry_policy: RetryPolicy | None = None,
        cancellation_type: ActivityCancellationType = ActivityCancellationType.TRY_CANCEL,
        versioning_intent: VersioningIntent | None = None,
        summary: str | None = None,
        priority: Priority = Priority.default,
        streaming_topic: str | None = None,
        streaming_batch_interval: timedelta = timedelta(milliseconds=100),
        sandbox: TemporalSandbox | None = None,
        model_state: dict[str, Any] | None = None,
        **agent_kwargs: Any,
    ) -> None:
        """Build a TemporalAgent from a registered model name and activity options."""
        if agent_kwargs.get("retry_strategy") is not None:
            raise ValueError(
                "TemporalAgent disables Strands retries; configure retries via "
                "retry_policy on TemporalAgent and on the activity options "
                "passed to workflow.activity_as_tool, workflow.activity_as_hook, "
                "or TemporalMCPClient. Remove retry_strategy from "
                "TemporalAgent(...) or pass retry_strategy=None."
            )
        agent_kwargs["retry_strategy"] = None
        self._manager_given = agent_kwargs.get(
            "conversation_manager"
        ) is not None or agent_kwargs.get("context_manager") not in (None, False)

        temporal_model = TemporalModel(
            model_name=model,
            task_queue=task_queue,
            schedule_to_close_timeout=schedule_to_close_timeout,
            schedule_to_start_timeout=schedule_to_start_timeout,
            start_to_close_timeout=start_to_close_timeout,
            heartbeat_timeout=heartbeat_timeout,
            retry_policy=retry_policy,
            cancellation_type=cancellation_type,
            versioning_intent=versioning_intent,
            summary=summary,
            priority=priority,
            streaming_topic=streaming_topic,
            streaming_batch_interval=streaming_batch_interval,
        )
        super().__init__(model=temporal_model, sandbox=sandbox, **agent_kwargs)
        self._concurrency = _WorkflowConcurrencyController(self._concurrency.mode)
        self._temporal_model = temporal_model
        # A model state carried from an earlier run, such as a stateful model's
        # last response id across Continue-As-New.
        if model_state:
            self._model_state = dict(model_state)

        # Strands invokes ToolProvider.load_tools() once at construction on a
        # separate run_async thread that has no workflow runtime, so a
        # TemporalMCPClient cannot list its tools there. Instead refresh from a
        # BeforeModelCallEvent hook, which runs on the workflow loop just before
        # the registry is read each turn. cache_tools=True lists once (guarded
        # by _fetched); cache_tools=False re-lists every turn.
        for provider in self.tool_registry._tool_providers:
            if isinstance(provider, TemporalMCPClient):
                self.hooks.add_callback(
                    BeforeModelCallEvent, self._make_mcp_refresh_hook(provider)
                )

    def _make_mcp_refresh_hook(
        self, provider: TemporalMCPClient
    ) -> HookCallback[BeforeModelCallEvent]:
        async def hook(event: BeforeModelCallEvent) -> None:
            if provider._cache_tools and provider._fetched:
                return
            old_names = {tool.tool_name for tool in provider._tools}
            await provider._refresh()
            self._reconcile_mcp_tools(event, provider, old_names)

        return hook

    def _reconcile_mcp_tools(
        self,
        event: BeforeModelCallEvent,
        provider: TemporalMCPClient,
        old_names: set[str],
    ) -> None:
        reg = event.agent.tool_registry
        new = {tool.tool_name: tool for tool in provider._tools}
        # Tools the server dropped or renamed since the last listing. There is
        # no public unregister, so remove them from the registry directly.
        for name in old_names - set(new):
            reg.registry.pop(name, None)
            reg.dynamic_tools.pop(name, None)
        # replace() swaps an existing tool in place (no hot-reload guard);
        # register_tool() adds a newly-discovered one.
        for name, tool in new.items():
            if name in reg.registry:
                reg.replace(tool)
            else:
                reg.register_tool(tool)

    async def _convert_prompt_to_messages(self, prompt: AgentInput) -> Messages:
        """Load the worker-side model's config, then convert the prompt.

        Strands reads the model's context window limit for proactive
        compression and its model id for tracing in workflow code, before the
        first model call. ``stream_async`` awaits this method first after it
        takes the invocation lock and registers the idempotency token, and
        before it starts the agent span. Loading here keeps Strands' rejection,
        deduplication and cancellation of overlapping invocations unchanged.
        """
        await self._temporal_model._load_config()
        if self._temporal_model.stateful:
            self._use_server_side_state()
        return await super()._convert_prompt_to_messages(prompt)

    async def _run_loop(self, *args: Any, **kwargs: Any) -> AsyncGenerator[Any, None]:
        """Run Strands' event loop, with sync tools run directly in the Workflow.

        Every invocation runs its tools in this loop, also when the agent is a
        ``Graph`` or ``Swarm`` node or another agent's tool.
        """
        _run_inline_when_called_by(asyncio.get_running_loop(), _called_by_sync_tool)
        events = super()._run_loop(*args, **kwargs)
        # The caller can take each event in a different task, so each step
        # sets the flag only while it runs.
        try:
            while True:
                with _agent_loop_step():
                    try:
                        event = await events.__anext__()
                    except StopAsyncIteration:
                        return
                yield event
        finally:
            with _agent_loop_step():
                await events.aclose()

    def _use_server_side_state(self) -> None:
        # What Strands' Agent does at construction for a stateful model, which
        # TemporalAgent can only learn once the worker model's config is read.
        if self._manager_given:
            raise ValueError(
                "context_manager and conversation_manager cannot be used with a "
                "stateful model. The model manages conversation state server-side."
            )
        if not isinstance(self.conversation_manager, NullConversationManager):
            self.conversation_manager = NullConversationManager()

    @property
    def model_state(self) -> dict[str, Any]:
        """A copy of the model state, such as a stateful model's last response id.

        Carry it with the messages across Continue-As-New and pass it back as
        ``TemporalAgent(model_state=...)``.
        """
        return dict(self._model_state)

    def take_snapshot(self, *_args: Any, **_kwargs: Any) -> Any:
        """Disabled; Temporal's event history is the source of truth."""
        raise NotImplementedError(_SNAPSHOT_DISABLED)

    def load_snapshot(self, *_args: Any, **_kwargs: Any) -> Any:
        """Disabled; Temporal's event history is the source of truth."""
        raise NotImplementedError(_SNAPSHOT_DISABLED)
