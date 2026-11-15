import asyncio
import contextlib
import contextvars
import dataclasses
import sys
import weakref
from collections.abc import AsyncIterator, Awaitable, Generator
from typing import Any, Callable

from agents import (
    Agent,
    AgentsException,
    FunctionTool,
    Handoff,
    ModelProvider,
    MultiProvider,
    RunConfig,
    RunContextWrapper,
    RunResult,
    RunResultStreaming,
    RunState,
    SQLiteSession,
    TContext,
    TResponseInputItem,
)
from agents.run import DEFAULT_AGENT_RUNNER, AgentRunner, RunOptions
from agents.sandbox import SandboxAgent
from typing_extensions import Unpack

from temporalio import workflow
from temporalio.exceptions import ApplicationError
from temporalio.openai_agents._errors import (
    AgentsWorkflowError,
    NonRetryableAgentsWorkflowError,
)
from temporalio.openai_agents._model_parameters import ModelActivityParameters
from temporalio.openai_agents._temporal_model_provider import (
    _current_run_model_settings,
    _TemporalModelProvider,
    capture_scope,
    install_model_provider,
)
from temporalio.openai_agents._temporal_model_stub import (
    _current_streamed_run,
    _StreamedRun,
    _TemporalModelStub,
)
from temporalio.openai_agents._trace_interceptor import _refuse_tracing_key
from temporalio.openai_agents.sandbox._temporal_sandbox_client import (
    TemporalSandboxClient,
)


@dataclasses.dataclass(eq=False)
class _ToolOperation:
    """A Nexus operation started by a tool, pending until the tool's call ends."""

    task: asyncio.Task[Any] | None
    cancelled: bool = False
    """Set when the run has stopped and waits for the tool, whose task the Agents SDK cancelled."""


@dataclasses.dataclass(eq=False)
class _AgentRun:
    """An agent run in a Workflow, with its tools' pending Nexus operations.

    Also keeps the agent runs started in its tools.
    """

    operations: set[_ToolOperation] = dataclasses.field(default_factory=set)
    nested: set["_AgentRun"] = dataclasses.field(default_factory=set)
    stopping: bool = False
    """Set when the run was cancelled or failed and waits for its operations."""

    def pending(self, *, all_nested: bool) -> list[_ToolOperation]:
        """The pending operations of this run and of its nested runs.

        Only nested runs that are stopping count, unless ``all_nested`` is set.
        """
        pending = list(self.operations)
        for run in self.nested:
            if all_nested or run.stopping:
                pending.extend(run.pending(all_nested=all_nested))
        return pending


# The innermost active agent run in a Workflow. Tasks copy it, so an agent run started in a
# tool, or in a task that a tool starts, is nested in the run of that tool.
_current_agent_run: contextvars.ContextVar[_AgentRun | None] = contextvars.ContextVar(
    "_current_agent_run", default=None
)


@contextlib.contextmanager
def _track_tool_operation() -> Generator[_ToolOperation, None, None]:
    """Keep a tool's Nexus operation pending in its run until the block exits."""
    run = _current_agent_run.get()
    operation = _ToolOperation(task=asyncio.current_task())
    if run is not None:
        run.operations.add(operation)
    try:
        yield operation
    finally:
        if run is not None:
            run.operations.discard(operation)


async def _wait_for_tool_operations(run: _AgentRun) -> bool:
    """Wait for the run's pending Nexus operations after it was cancelled or failed.

    The Agents SDK cancels a run's tool tasks and raises without waiting for them. The
    Workflow would then close in the same Workflow Task, before Temporal delivers the cancel
    requests of the tools' Nexus operations. Waiting keeps their cancellation type, as when
    Workflow code calls ``execute_operation`` itself.
    The run also waits for the operations of its nested runs that stop with it, such as an
    agent that runs as a tool. It checks which ones stop once the cancellation has reached
    them, so an agent run in a task that a tool started, and that nothing cancels, does not
    delay it.
    A further cancel request, such as the Workflow's after Workflow code cancelled the run,
    goes to the tools' tasks, as it would to a direct call: a Workflow cancellation ends an
    operation that is still starting, and a started one keeps its cancellation type. The
    wait goes on until every tool's call has ended. Returns whether a cancel request arrived
    during the wait; the caller then raises.
    """
    run.stopping = True
    for operation in run.operations:
        operation.cancelled = True
    if not run.pending(all_nested=True):
        return False
    cancel_requested = False
    while True:
        try:
            # The condition is checked once no task is ready to run, so the cancellation has
            # reached every nested run that stops with this one.
            await workflow.wait_condition(lambda: not run.pending(all_nested=False))
            return cancel_requested
        except asyncio.CancelledError:
            cancel_requested = True
            for operation in run.pending(all_nested=False):
                if operation.task is not None:
                    operation.task.cancel()


# Recursively replace models in all agents
def _convert_agent(
    model_params: ModelActivityParameters,
    agent: Agent[Any],
    seen: dict[int, Agent] | None,
    *,
    visited: set[int] | None = None,
) -> Agent[Any]:
    if seen is None:
        seen = dict()
    if visited is None:
        visited = set()

    new_agent = seen.get(id(agent))
    if new_agent is not None and id(agent) in visited:
        return new_agent

    # This agent has already been processed in some other run
    if isinstance(agent.model, _TemporalModelStub):
        return agent

    visited.add(id(agent))
    if new_agent is None:
        new_agent = dataclasses.replace(agent)
        seen[id(agent)] = new_agent
    else:
        for field in dataclasses.fields(agent):
            if field.init:
                setattr(new_agent, field.name, getattr(agent, field.name))

    name = _model_name(agent)

    new_handoffs: list[Agent | Handoff] = []
    for handoff in agent.handoffs:
        if isinstance(handoff, Agent):
            new_handoffs.append(
                _convert_agent(model_params, handoff, seen, visited=visited)
            )
        elif isinstance(handoff, Handoff):
            original_invoke = handoff.on_invoke_handoff

            # Use default parameter to capture original_invoke by value, not reference
            async def on_invoke(
                context: RunContextWrapper[Any],
                args: str,
                invoke_func: Callable[
                    [RunContextWrapper[Any], str], Awaitable[Any]
                ] = original_invoke,
            ) -> Agent:
                handoff_agent = await invoke_func(context, args)
                return _convert_agent(model_params, handoff_agent, seen)

            new_handoff = dataclasses.replace(handoff, on_invoke_handoff=on_invoke)
            handoff_agent = (
                handoff._agent_ref() if handoff._agent_ref is not None else None
            )
            if isinstance(handoff_agent, Agent):
                new_handoff._agent_ref = weakref.ref(
                    _convert_agent(model_params, handoff_agent, seen, visited=visited)
                )
            new_handoffs.append(new_handoff)
        else:
            raise TypeError(f"Unknown handoff type: {type(handoff)}")

    new_agent.model = _TemporalModelStub(
        model_name=name,
        model_params=model_params,
        agent=agent,
        current_run_model_settings=_current_run_model_settings.get,
    )
    new_agent.handoffs = new_handoffs
    new_agent.tools = [
        dataclasses.replace(
            tool,
            _agent_instance=_convert_agent(
                model_params, tool._agent_instance, seen, visited=visited
            ),
        )
        if isinstance(tool, FunctionTool) and isinstance(tool._agent_instance, Agent)
        else tool
        for tool in agent.tools
    ]
    return new_agent


# True during an agent run, so the Workflow loop runs sync function tools inline.
_in_agent_run: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_in_agent_run", default=False
)


@contextlib.contextmanager
def _inline_sync_function_tools() -> Generator[None, None, None]:
    """Run the sync function tools of the agent run in this scope in the Workflow.

    The Agents SDK runs a sync function tool with ``asyncio.to_thread``. The Workflow
    event loop has no threads, so the tool failed before its body ran and the model
    got an empty error. Tools are not changed, only how the loop runs that call.
    """
    _run_sync_function_tool_calls_inline(asyncio.get_running_loop())
    token = _in_agent_run.set(True)
    try:
        yield
    finally:
        _in_agent_run.reset(token)


def _run_sync_function_tool_calls_inline(loop: asyncio.AbstractEventLoop) -> None:
    """Make the Workflow loop run a sync function tool's thread call inline."""
    if getattr(loop, "_runs_sync_function_tools_inline", False):
        return
    run_in_executor = loop.run_in_executor

    def inline_run_in_executor(
        executor: Any, func: Callable[..., Any], *args: Any
    ) -> asyncio.Future[Any]:
        if not (
            _in_agent_run.get() and _called_by_sync_function_tool(sys._getframe(1))
        ):
            return run_in_executor(executor, func, *args)
        future = loop.create_future()
        future.set_result(func(*args))
        return future

    setattr(loop, "run_in_executor", inline_run_in_executor)
    setattr(loop, "_runs_sync_function_tools_inline", True)


def _called_by_sync_function_tool(frame: Any) -> bool:
    """Whether this is ``asyncio.to_thread`` called by a sync ``@function_tool``.

    Those are the only ``asyncio.to_thread`` calls in ``agents.tool``.
    """
    return (
        frame.f_code is asyncio.to_thread.__code__
        and frame.f_back is not None
        and frame.f_back.f_globals.get("__name__") == "agents.tool"
    )


def _has_sandbox_agent(agent: Agent[Any], seen: set[int] | None = None) -> bool:
    """Check if any agent in the graph (following direct Agent handoffs) is a SandboxAgent."""
    if seen is None:
        seen = set()
    if id(agent) in seen:
        return False
    seen.add(id(agent))
    if isinstance(agent, SandboxAgent):
        return True
    for handoff in agent.handoffs:
        if isinstance(handoff, Agent) and _has_sandbox_agent(handoff, seen):
            return True
    return False


# Runs without this patch marker started before the plugin refused
# ``RunConfig.model_provider``. They keep the old behavior, the Worker's provider, so that
# they replay.
_REFUSE_RUN_MODEL_PROVIDER_PATCH = "openai_agents.refuse-run-model-provider"
# Runs started without it keep running on copies of the agents, so their histories replay.
_RUN_CALLER_AGENTS_PATCH = "openai_agents.run-caller-agents"


def _is_default_model_provider(provider: ModelProvider) -> bool:
    """Whether ``provider`` is an unconfigured ``MultiProvider``, as ``RunConfig()`` makes."""
    if type(provider) is not MultiProvider:
        return False
    return _multi_provider_settings(provider) == _multi_provider_settings(
        MultiProvider()
    )


def _multi_provider_settings(provider: MultiProvider) -> dict[str, Any]:
    # Caches filled on use are not settings.
    caches = {"_fallback_providers", "_ws_model_cache_by_loop"}
    settings: dict[str, Any] = {
        name: value
        for name, value in vars(provider).items()
        if name not in caches and name != "openai_provider"
    }
    openai_settings = vars(provider.openai_provider)
    if "_stored_api_key" in openai_settings:
        # No client was passed, so ``_client`` caches the one built on first use.
        caches.add("_client")
    settings["openai_provider"] = {
        name: value for name, value in openai_settings.items() if name not in caches
    }
    return settings


def _coerce_run_config(value: object) -> RunConfig:
    """openai-agents >= 0.19 also accepts a plain dict for ``run_config``.

    This function normalizes to a RunConfig instance.
    """
    if isinstance(value, RunConfig):
        return value
    if not isinstance(value, dict):
        raise TypeError(
            f"run_config must be a RunConfig instance or a dict, got {type(value).__name__}"
        )
    field_names = {
        config_field.name
        for config_field in dataclasses.fields(RunConfig)
        if config_field.init
    }
    unknown_fields = sorted(str(name) for name in value if name not in field_names)
    if unknown_fields:
        raise TypeError(f"Unknown run_config settings: {', '.join(unknown_fields)}")
    return RunConfig(**value)


class TemporalOpenAIRunner(AgentRunner):
    """Temporal Runner for OpenAI agents.

    Forwards model calls to a Temporal activity.

    """

    def __init__(
        self,
        model_params: ModelActivityParameters,
    ) -> None:
        """Initialize the Temporal OpenAI Runner."""
        self._runner = DEFAULT_AGENT_RUNNER or AgentRunner()
        self.model_params = model_params

    def _prepare_workflow_run(
        self,
        starting_agent: Agent[TContext],
        kwargs: RunOptions[TContext],
    ) -> tuple[Agent[Any], RunConfig]:
        """Workflow-only validation and ``kwargs`` rewrite shared by ``run()`` and ``run_streamed()``."""
        # Refuse a Model object up front for every agent reachable through public fields.
        # Agents behind handoff(), as_tool() or a custom on_invoke_handoff are only
        # reachable through private SDK fields, so the call_model_input_filter refuses
        # them just before their model call instead.
        pending: list[Agent[Any]] = [starting_agent]
        checked: set[int] = set()
        while pending:
            agent = pending.pop()
            if id(agent) in checked:
                continue
            checked.add(id(agent))
            if agent.model is not None and not isinstance(
                agent.model, (str, _TemporalModelStub)
            ):
                raise ValueError(
                    "Temporal workflows require a model name to be a string in the agent."
                )
            pending.extend(h for h in agent.handoffs if isinstance(h, Agent))

        for t in starting_agent.tools:
            if callable(t):
                raise ValueError(
                    "Provided tool is not a tool type. If using an activity, make sure to wrap it with openai_agents.workflow.activity_as_tool."
                )

        if starting_agent.mcp_servers:
            from temporalio.openai_agents._mcp import (
                _StatefulMCPServerReference,
                _StatelessMCPServerReference,
            )

            for s in starting_agent.mcp_servers:
                if isinstance(
                    s,
                    (
                        _StatelessMCPServerReference,
                        _StatefulMCPServerReference,
                    ),
                ):
                    continue
                # Avoid importing the MCP v2-only proxy while validating a
                # workflow that uses the legacy MCP v1 provider path.
                if (
                    type(s).__module__
                    == "temporalio.openai_agents._temporal_mcp_server"
                    and type(s).__name__ == "_TemporalMCPServer"
                ):
                    continue
                raise ValueError(
                    f"Unknown mcp_server type {type(s)} may not work durably."
                )

        if isinstance(kwargs.get("session"), SQLiteSession):
            raise ValueError("Temporal workflows don't support SQLite sessions.")

        run_config = kwargs.get("run_config")
        run_config = (
            RunConfig() if run_config is None else _coerce_run_config(run_config)
        )

        # The plugin's model provider turns the name into the model Activity. Keeping the
        # name lets the SDK apply that model's default settings, as it does natively.
        if run_config.model and not isinstance(run_config.model, str):
            raise ValueError(
                "Temporal workflows require a model name to be a string in the run config."
            )

        # Model calls run in an Activity on the Worker's provider, so a provider set on the
        # run would be silently ignored. A nested run, such as an agent used as a tool,
        # inherits the plugin's provider.
        if not isinstance(
            run_config.model_provider, _TemporalModelProvider
        ) and not _is_default_model_provider(run_config.model_provider):
            if workflow.patched(_REFUSE_RUN_MODEL_PROVIDER_PATCH):
                raise ValueError(
                    "Temporal workflows do not support RunConfig.model_provider. Model "
                    "calls use the provider set on the Worker with "
                    "OpenAIAgentsPlugin(model_provider=...). To use several providers, "
                    "set a MultiProvider with a prefix map there and choose a provider "
                    "by model name."
                )

        # Model calls run in an Activity, which cannot get the run's own tracing key.
        if run_config.tracing is not None:
            _refuse_tracing_key(run_config.tracing.get("api_key"))

        run_config = install_model_provider(self.model_params, run_config)

        # run_config.sandbox is global for the entire run — configure it if any agent needs it.
        if _has_sandbox_agent(starting_agent) or run_config.sandbox:
            if run_config.sandbox is None:
                raise ValueError(
                    "A SandboxAgent was provided but run_config.sandbox is not configured. "
                    "You must set run_config.sandbox to a SandboxRunConfig. "
                    "For example:\n"
                    "  from temporalio.openai_agents.workflow import temporal_sandbox_client\n"
                    "  run_config = RunConfig(sandbox=SandboxRunConfig(client=temporal_sandbox_client('my-backend')))"
                )
            elif run_config.sandbox.session is not None:
                raise AgentsWorkflowError(
                    "run_config.sandbox.session is not supported by the Temporal OpenAI Agents "
                    "plugin. A live sandbox session is not a durable construct in a workflow. "
                    "Pass run_config.sandbox.client=temporal_sandbox_client(name) instead."
                )
            elif run_config.sandbox.client is None:
                raise ValueError(
                    "run_config.sandbox.client must be set to a temporal sandbox client. "
                    "Use temporalio.openai_agents.workflow.temporal_sandbox_client(name) "
                    "to create one, where name matches a SandboxClientProvider registered on the plugin."
                )
            elif not isinstance(run_config.sandbox.client, TemporalSandboxClient):
                raise ValueError(
                    "run_config.sandbox.client must be created via "
                    "temporalio.openai_agents.workflow.temporal_sandbox_client(name). "
                    "Do not pass a raw sandbox client directly."
                )

        kwargs["run_config"] = run_config
        if not workflow.patched(_RUN_CALLER_AGENTS_PATCH):
            return _convert_agent(self.model_params, starting_agent, None), run_config
        return starting_agent, run_config

    async def run(
        self,
        starting_agent: Agent[TContext],
        input: str | list[TResponseInputItem] | RunState[TContext],
        **kwargs: Unpack[RunOptions[TContext]],
    ) -> RunResult:
        """Run the agent in a Temporal workflow."""
        if not workflow.in_workflow():
            return await self._runner.run(
                starting_agent,
                input,
                **kwargs,
            )

        starting_agent, run_config = self._prepare_workflow_run(starting_agent, kwargs)

        run = _AgentRun()
        parent = _current_agent_run.get()
        if parent is not None:
            parent.nested.add(run)
        token = _current_agent_run.set(run)
        with capture_scope(run_config.model_settings), _inline_sync_function_tools():
            try:
                return await self._runner.run(
                    starting_agent=starting_agent,
                    input=input,
                    **kwargs,
                )
            except asyncio.CancelledError:
                await _wait_for_tool_operations(run)
                raise
            except AgentsException as e:
                # When a tool fails, the Agents SDK also cancels the run's other tools. Wait
                # only when the error fails the Workflow: an error that fails the Workflow
                # Task must leave no new commands, so that a fixed Worker can replay it.
                if (
                    e.__cause__ is not None
                    and workflow.is_failure_exception(e.__cause__)
                ) or workflow.is_failure_exception(e):
                    if await _wait_for_tool_operations(run):
                        # The Workflow was cancelled during the wait.
                        raise asyncio.CancelledError() from e
                # In order for workflow failures to properly fail the workflow, we need to rewrap them in
                # a Temporal error
                _reraise_workflow_failure(e)
                raise e
            finally:
                _current_agent_run.reset(token)
                if parent is not None:
                    parent.nested.discard(run)

    def run_sync(
        self,
        starting_agent: Agent[TContext],
        input: str | list[TResponseInputItem] | RunState[TContext],
        **kwargs: Any,
    ) -> RunResult:
        """Run the agent synchronously (not supported in Temporal workflows)."""
        if not workflow.in_workflow():
            return self._runner.run_sync(
                starting_agent,
                input,
                **kwargs,
            )
        raise RuntimeError("Temporal workflows do not support synchronous model calls.")

    def run_streamed(
        self,
        starting_agent: Agent[TContext],
        input: str | list[TResponseInputItem] | RunState[TContext],
        **kwargs: Unpack[RunOptions[TContext]],
    ) -> RunResultStreaming:
        """Run the agent with streaming responses.

        .. warning::
            Streaming inside Temporal workflows is experimental and may
            change in future versions.

        Inside a workflow, model calls execute as the streaming model
        activity. The workflow consumes events via
        ``RunResultStreaming.stream_events()`` after each activity
        completes; external clients can subscribe to the configured
        stream topic to receive events as they arrive.
        """
        if not workflow.in_workflow():
            return self._runner.run_streamed(
                starting_agent,
                input,
                **kwargs,
            )

        # Fail-fast before the agents framework starts a background task:
        # validation raised inside ``Model.stream_response`` is otherwise
        # captured into ``RunResultStreaming._stored_exception`` and may
        # be silently dropped if the queue completion sentinel is read
        # before the run_loop_task is observed as done.
        if self.model_params.streaming_topic is None:
            raise AgentsWorkflowError(
                "Runner.run_streamed requires "
                "ModelActivityParameters.streaming_topic to be set."
            )
        if self.model_params.use_local_activity:
            raise AgentsWorkflowError(
                "Runner.run_streamed is incompatible with "
                "use_local_activity (local activities do not support "
                "heartbeats or the workflow stream signal channel)."
            )

        starting_agent, run_config = self._prepare_workflow_run(starting_agent, kwargs)

        # The streaming loop runs in a task created inside run_streamed, which copies
        # this context, so the scope only needs to cover the call itself.
        streamed_run = _StreamedRun()
        token = _current_streamed_run.set(streamed_run)
        try:
            with (
                capture_scope(run_config.model_settings),
                _inline_sync_function_tools(),
            ):
                streamed_result = self._runner.run_streamed(
                    starting_agent=starting_agent,
                    input=input,
                    **kwargs,
                )
        finally:
            _current_streamed_run.reset(token)
        # The model stub reads it to watch the run's parallel input guardrails.
        streamed_run.result = streamed_result

        # Mirror the AgentsException -> AgentsWorkflowError rewrap done
        # in run() above. The streaming runner attaches the actual run
        # to ``run_loop_task``; we wrap ``stream_events()`` (rather than
        # the task itself) so the rewrap happens on the consumer's
        # coroutine. Wrapping in a second asyncio task introduces a
        # scheduling gap: ``RunResultStreaming.stream_events()`` reads
        # the queue completion sentinel as soon as the run loop ends,
        # but the wrapper task only resumes its ``await`` after another
        # event-loop tick — between those two points, ``_check_errors``
        # sees no exception and ``_await_task_safely`` later swallows
        # the rewrapped one. Iterating the underlying generator first,
        # then inspecting the finished task on exit, keeps the rewrap
        # race-free without touching ``run_loop_task``.
        original_stream_events = streamed_result.stream_events
        run_loop_task = streamed_result.run_loop_task

        async def _stream_events_with_rewrap() -> AsyncIterator[Any]:
            try:
                async for event in original_stream_events():
                    yield event
            except AgentsException as e:
                _reraise_workflow_failure(e)
                raise
            # The agents framework may have stored the run-loop
            # exception on ``_stored_exception`` (or surfaced it through
            # the iterator) without re-raising it through stream_events.
            # By the time the iterator is exhausted, ``run_loop_task``
            # is done — surface its exception here so a failed run
            # cannot appear successful, applying the workflow-failure
            # rewrap when applicable.
            if run_loop_task is not None and run_loop_task.done():
                exc = run_loop_task.exception()
                if exc is not None:
                    if isinstance(exc, AgentsException):
                        _reraise_workflow_failure(exc)
                    raise exc

        streamed_result.stream_events = _stream_events_with_rewrap  # type: ignore[method-assign]
        return streamed_result


def _reraise_workflow_failure(e: AgentsException) -> None:
    """Rewrap an AgentsException whose cause is a Temporal workflow failure.

    Returns normally when ``e`` is not workflow-failure-bearing so the
    caller can re-raise the original. The error is non-retryable when the
    cause is a non-retryable ``ApplicationError``, so a Workflow retry policy
    does not rerun the whole agent run for it.
    """
    if e.__cause__ and workflow.is_failure_exception(e.__cause__):
        message = f"Workflow failure exception in Agents Framework: {e}"
        reraise = (
            NonRetryableAgentsWorkflowError(message)
            if isinstance(e.__cause__, ApplicationError) and e.__cause__.non_retryable
            else AgentsWorkflowError(message)
        )
        reraise.__traceback__ = e.__traceback__
        raise reraise from e.__cause__


def _model_name(agent: Agent[Any]) -> str | None:
    name = agent.model
    if name is not None and not isinstance(name, str):
        raise ValueError(
            "Temporal workflows require a model name to be a string in the agent."
        )
    return name
