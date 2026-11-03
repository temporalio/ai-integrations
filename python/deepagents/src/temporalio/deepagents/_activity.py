"""The activities that carry every nondeterministic Deep Agents operation.

The Deep Agents control loop runs *inside* the workflow; the operations that
must not run there — talking to an LLM, executing a tool that does real I/O, or
touching a real filesystem / shell backend — are moved out to these activities.

Each activity is a method on :class:`DeepAgentActivities` so the worker-only
dependencies (the ``model_provider`` that builds real chat models from a name,
the streaming batch interval) can be captured on the instance rather than
smuggled through activity inputs. API keys therefore live on the worker, never
in a workflow input or in history.

Every method:

* takes a single serializable dataclass in and returns a single dataclass out
  (LangChain objects travel as their ``dumpd`` JSON form via
  :mod:`temporalio.deepagents._serde`);
* translates the LLM SDK's HTTP error into Temporal's retry contract so a 429
  honors the upstream ``retry-after`` instead of hammering it;
* heartbeats on a background task so a slow (thinking-mode / long-context) call
  is not mistaken for a stuck worker.
"""

from __future__ import annotations

import asyncio
import dataclasses
import importlib
import inspect
from collections.abc import Mapping
from datetime import timedelta
from functools import wraps
from typing import Any, Callable

from temporalio import activity
from temporalio.deepagents import _serde
from temporalio.exceptions import ApplicationError

# Activity type names. The workflow dispatches by these strings, so the
# in-workflow model / tool stubs never import the activity class itself.
INVOKE_MODEL = "deepagents.invoke_model"
INVOKE_MODEL_STREAMING = "deepagents.invoke_model_streaming"
INVOKE_TOOL = "deepagents.invoke_tool"
BACKEND_OP = "deepagents.backend_op"

CONTEXT_OVERFLOW_ERROR_TYPE = "deepagents.ContextOverflowError"
"""``ApplicationError.type`` of a model call the provider rejected as too long."""


# ---------------------------------------------------------------------------
# Boundary payloads
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ModelActivityInput:
    """A single LLM request.

    ``model_name`` is resolved to a real model by the worker's
    ``model_provider``; the workflow never ships credentials.
    """

    model_name: str
    messages: list[Any]
    """Messages in ``langchain_core.load.dumpd`` form."""
    tool_schemas: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    """OpenAI-format tool advertisements (name + description + argument schema)."""
    bind_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)
    config: dict[str, Any] = dataclasses.field(default_factory=dict)
    """A stripped ``RunnableConfig`` (see :func:`_serde.strip_runnable_config`)."""
    streaming_topic: str | None = None


@dataclasses.dataclass
class ModelActivityOutput:
    """The model's reply.

    ``message`` is an ``AIMessage`` in ``dumpd`` form, carrying tool calls,
    usage, and response metadata.
    """

    message: Any


@dataclasses.dataclass
class ToolActivityInput:
    """One tool execution routed to an activity."""

    tool_name: str
    tool_call_id: str
    args: dict[str, Any]
    config: dict[str, Any] = dataclasses.field(default_factory=dict)
    # The path and class name of each LangChain message in ``args``, from
    # ``_serde.find_tool_arg_messages``. Input from older workflow code lacks it
    # and passes through unchanged.
    arg_messages: list[tuple[list[str | int], str]] = dataclasses.field(
        default_factory=list
    )


@dataclasses.dataclass
class ToolActivityOutput:
    """A tool result as a ``ToolMessage`` in ``dumpd`` form."""

    message: Any


@dataclasses.dataclass
class BackendOpInput:
    """A single filesystem / shell / store operation for a wrapped backend."""

    backend_ref: str
    """Backend reference; instance wrappers resolve it in the local registry."""
    op: str
    """Backend method name, e.g. ``ls`` / ``read_file`` / ``write_file`` / ``execute``."""
    args: list[Any] = dataclasses.field(default_factory=list)
    kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)
    backend_factory: str | None = None
    """Worker-configured factory name; when set, bypasses the local registry."""
    factory_argument: Any = None
    """Serializable argument passed to the factory on each activity attempt."""


@dataclasses.dataclass
class BackendOpOutput:
    """The backend operation's return value.

    Deepagents protocol dataclasses (``WriteResult`` / ``ReadResult`` / …)
    ride in the tagged form produced by ``_serde.dump_backend_result`` so the
    in-workflow stub can rebuild the real type; plain JSON values pass
    through unchanged.
    """

    result: Any


# ---------------------------------------------------------------------------
# Heartbeating + error translation
# ---------------------------------------------------------------------------


def _auto_heartbeater(fn: Callable) -> Callable:
    """Heartbeat at half the configured ``heartbeat_timeout`` while ``fn`` runs.

    Long LLM calls (thinking mode, long context, streaming accumulation) can run
    well past a scheduler's patience; without a heartbeat Temporal would cancel
    them and surface a ``HeartbeatTimeoutError`` instead of the real problem.
    """

    @wraps(fn)
    async def wrapped(*args: Any, **kwargs: Any) -> Any:
        heartbeat_timeout = activity.info().heartbeat_timeout
        beat_task: asyncio.Task | None = None
        if heartbeat_timeout:
            interval = heartbeat_timeout.total_seconds() / 2

            async def beat() -> None:
                while True:
                    activity.heartbeat()
                    await asyncio.sleep(interval)

            beat_task = asyncio.create_task(beat())
        try:
            return await fn(*args, **kwargs)
        finally:
            if beat_task is not None:
                beat_task.cancel()
                # Let the cancellation land before returning so no pending task
                # outlives the activity (a bare ``cancel()`` leaves the task to
                # be destroyed while pending if the loop shuts down first).
                # ``asyncio.wait`` never re-raises the task's CancelledError.
                await asyncio.wait([beat_task])

    return wrapped


def _translate_api_error(exc: Exception) -> ApplicationError | None:
    """Map an LLM SDK HTTP error onto Temporal's retry contract.

    Works by duck typing so neither ``openai`` nor ``anthropic`` needs to be
    imported here: both expose ``status_code`` and ``response.headers``. Returns
    ``None`` when ``exc`` is not a recognizable HTTP status error, so the caller
    can fall through to its generic handling.
    """
    status = getattr(exc, "status_code", None)
    if status is None:
        return None
    headers: dict[str, Any] = {}
    response = getattr(exc, "response", None)
    if response is not None:
        headers = dict(getattr(response, "headers", {}) or {})
    # Case-insensitive header access.
    lower = {str(k).lower(): v for k, v in headers.items()}

    retryable = status in (408, 409, 429) or 500 <= status < 600
    should_retry = lower.get("x-should-retry")
    if should_retry == "false":
        retryable = False
    elif should_retry == "true":
        retryable = True

    delay_ms = lower.get("retry-after-ms")
    retry_after = lower.get("retry-after")
    next_delay: timedelta | None = None
    try:
        if delay_ms is not None:
            next_delay = timedelta(milliseconds=int(delay_ms))
        elif retry_after is not None:
            next_delay = timedelta(seconds=int(retry_after))
    except (TypeError, ValueError):
        next_delay = None

    return ApplicationError(
        str(exc),
        type=type(exc).__name__,
        non_retryable=not retryable,
        next_retry_delay=next_delay,
    )


# HTTP statuses and messages that mark a provider's context overflow, as in Deep
# Agents' summarization middleware since 0.7.14. Its check is private and older
# 0.7 releases lack it, so the plugin keeps its own copy.
_CONTEXT_OVERFLOW_STATUSES = frozenset({400, 413, 422})
_CONTEXT_OVERFLOW_MARKERS = (
    "context_length_exceeded",
    "contextwindowexceedederror",
    "maximum context length",
    "exceeds the context window",
    "exceeds the available context size",
    "context window exceeded",
    "context limit exceeded",
    "input tokens exceed the configured limit",
    "prompt is too long",
)


def _is_context_overflow(exc: Exception) -> bool:
    """Return whether the provider rejected the request as too long.

    LangChain's ``ContextOverflowError`` covers the typed overflow errors, with
    or without an HTTP status. An OpenAI-compatible server such as vLLM returns
    an ordinary 400 instead, so an error with one of the statuses above whose
    message names an overflow counts too.
    """
    from langchain_core.exceptions import ContextOverflowError

    if isinstance(exc, ContextOverflowError):
        return True
    if getattr(exc, "status_code", None) not in _CONTEXT_OVERFLOW_STATUSES:
        return False
    text = str(exc).lower()
    return any(marker in text for marker in _CONTEXT_OVERFLOW_MARKERS)


def _context_overflow_error(exc: Exception) -> ApplicationError | None:
    """Mark a provider's context-window overflow so the workflow can recover.

    Deep Agents' summarization middleware summarizes older messages and retries
    when a model call fails with ``ContextOverflowError``. The provider's
    exception is only available here, so classify it here and give the failure
    a type the workflow side rebuilds the overflow from. Retrying the same
    request cannot succeed, so it is non-retryable.
    """
    if not _is_context_overflow(exc):
        return None
    return ApplicationError(
        str(exc), type=CONTEXT_OVERFLOW_ERROR_TYPE, non_retryable=True
    )


# ---------------------------------------------------------------------------
# The activities
# ---------------------------------------------------------------------------


class _BackendFactory:
    """Invoke a worker-side backend factory with zero or one argument."""

    def __init__(self, name: str, factory: Callable[..., Any]) -> None:
        self._name = name
        self._factory = factory
        self._accepts_argument = False
        try:
            signature = inspect.signature(factory)
        except (TypeError, ValueError):
            return
        self._accepts_argument = any(
            parameter.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.VAR_POSITIONAL,
            )
            for parameter in signature.parameters.values()
        )
        try:
            signature.bind(None) if self._accepts_argument else signature.bind()
        except TypeError as exc:
            raise TypeError(
                f"Backend factory {name!r} must accept zero arguments or one "
                "positional factory_argument."
            ) from exc

    def __call__(self, argument: Any) -> Any:
        if self._accepts_argument:
            return self._factory(argument)
        if argument is not None:
            raise ApplicationError(
                f"Backend factory {self._name!r} declares no parameters, so it "
                "cannot receive a factory_argument; declare one positional "
                "parameter to accept it.",
                type="DeepAgentsInvalidBackendFactoryArgument",
                non_retryable=True,
            )
        return self._factory()


# Binding options that a provider's ``bind_tools`` converts to its own format
# (Anthropic turns ``"any"`` into ``{"type": "any"}``, OpenAI into
# ``"required"``). Passing them to ``.bind()`` instead skips that conversion.
_TOOL_BIND_KWARGS = ("tool_choice", "parallel_tool_calls")


def _default_model_provider(model_name: str) -> Any:
    """Build a chat model from a name string with LLM-SDK retries disabled.

    Temporal owns retries; the model client must not also retry, or a single
    logical attempt fans out into nested retry storms that Temporal can neither
    see nor bound.
    """
    # importlib: `langchain` (unlike langchain-core) is absent on Python 3.10
    # environments where the deepagents extra cannot install; a static import
    # here fails type-checking there.
    init_chat_model = importlib.import_module("langchain.chat_models").init_chat_model

    return init_chat_model(model_name, max_retries=0)


class DeepAgentActivities:
    """Holds the worker-side dependencies and exposes the four activities.

    An instance is created by ``DeepAgentsPlugin``
    and its bound methods are registered on the worker.
    """

    def __init__(
        self,
        *,
        model_provider: Callable[[str], Any] | None = None,
        backends: Mapping[str, Callable[..., Any]] | None = None,
        streaming_batch_interval: timedelta = timedelta(milliseconds=100),
    ) -> None:
        """Store worker-side model and backend factories and streaming settings."""
        self._model_provider = model_provider or _default_model_provider
        self._backends = {
            name: _BackendFactory(name, factory)
            for name, factory in (backends or {}).items()
        }
        self._streaming_batch_interval = streaming_batch_interval

    def _build_bound_model(self, input: ModelActivityInput) -> Any:
        model = self._model_provider(input.model_name)
        bind_kwargs = dict(input.bind_kwargs)
        if input.tool_schemas:
            tool_kwargs = {
                k: bind_kwargs.pop(k) for k in _TOOL_BIND_KWARGS if k in bind_kwargs
            }
            model = model.bind_tools(input.tool_schemas, **tool_kwargs)
        if bind_kwargs:
            model = model.bind(**bind_kwargs)
        return model

    @activity.defn(name=INVOKE_MODEL)
    @_auto_heartbeater
    async def invoke_model(self, input: ModelActivityInput) -> ModelActivityOutput:
        """Run exactly one LLM call and return the resulting ``AIMessage``."""
        messages = _serde.load_messages(input.messages)
        config = _serde.rebuild_runnable_config(input.config)
        model = self._build_bound_model(input)
        try:
            message = await model.ainvoke(messages, config=config)
        except Exception as exc:
            overflow = _context_overflow_error(exc)
            if overflow is not None:
                raise overflow from exc
            translated = _translate_api_error(exc)
            if translated is not None:
                activity.logger.warning(
                    "Model call failed with an HTTP status error", exc_info=True
                )
                raise translated from exc
            raise
        return ModelActivityOutput(message=_serde.dump_object(message))

    @activity.defn(name=INVOKE_MODEL_STREAMING)
    @_auto_heartbeater
    async def invoke_model_streaming(
        self, input: ModelActivityInput
    ) -> ModelActivityOutput:
        """Stream one LLM call, publishing chunk batches to ``streaming_topic``.

        Token-level deltas are coalesced at ``streaming_batch_interval`` and
        pushed to external subscribers via the shared workflow-streams topic; the
        aggregated final ``AIMessage`` is returned to the workflow so the
        durable result is identical to the non-streaming path.
        """
        from temporalio.contrib.workflow_streams import WorkflowStreamClient

        messages = _serde.load_messages(input.messages)
        config = _serde.rebuild_runnable_config(input.config)
        model = self._build_bound_model(input)

        final: Any = None
        try:
            async with WorkflowStreamClient.from_within_activity(
                batch_interval=self._streaming_batch_interval
            ) as client:
                topic = (
                    client.topic(input.streaming_topic)
                    if input.streaming_topic
                    else None
                )
                async for chunk in model.astream(messages, config=config):
                    if topic is not None:
                        topic.publish(_serde.dump_object(chunk))
                    final = chunk if final is None else final + chunk
        except Exception as exc:
            overflow = _context_overflow_error(exc)
            if overflow is not None:
                raise overflow from exc
            translated = _translate_api_error(exc)
            if translated is not None:
                activity.logger.warning(
                    "Streaming model call failed with an HTTP status error",
                    exc_info=True,
                )
                raise translated from exc
            raise
        return ModelActivityOutput(message=_serde.dump_object(final))

    @activity.defn(name=INVOKE_TOOL)
    @_auto_heartbeater
    async def invoke_tool(self, input: ToolActivityInput) -> ToolActivityOutput:
        """Execute one registered tool and return its ``ToolMessage``."""
        from temporalio.deepagents._tools import get_registered_tool

        tool = get_registered_tool(input.tool_name)
        if tool is None:
            raise ApplicationError(
                f"Tool {input.tool_name!r} is not registered on this worker. "
                "Call tool_as_activity(...) at startup on every activity worker, "
                "or expose a registered activity with activity_as_tool(...).",
                type="DeepAgentsUnknownTool",
                non_retryable=True,
            )
        config = _serde.rebuild_runnable_config(input.config)
        tool_call = {
            "name": input.tool_name,
            "args": _serde.load_tool_arg_messages(input.args, input.arg_messages),
            "id": input.tool_call_id,
            "type": "tool_call",
        }
        message = await tool.ainvoke(tool_call, config=config)
        return ToolActivityOutput(message=_serde.dump_object(message))

    @activity.defn(name=BACKEND_OP)
    @_auto_heartbeater
    async def backend_op(self, input: BackendOpInput) -> BackendOpOutput:
        """Run one operation against a factory-built or registered backend."""
        from temporalio.deepagents._tools import lookup_backend

        if input.backend_factory is not None:
            factory = self._backends.get(input.backend_factory)
            if factory is None:
                raise ApplicationError(
                    f"Backend factory {input.backend_factory!r} is not configured "
                    "on this worker. Configure DeepAgentsPlugin(backends=...) "
                    "on every activity worker.",
                    type="DeepAgentsUnknownBackendFactory",
                    non_retryable=True,
                )
            # Preserve the activity context without blocking heartbeat tasks.
            backend = await asyncio.to_thread(factory, input.factory_argument)
            if inspect.isawaitable(backend):
                backend = await backend
        else:
            backend = lookup_backend(input.backend_ref)
        if backend is None:
            raise ApplicationError(
                f"Backend {input.backend_ref!r} is not registered on this worker. "
                "Use TemporalBackend with a named worker-side factory for "
                "operations that may run in another process.",
                type="DeepAgentsUnknownBackend",
                non_retryable=True,
            )
        method = getattr(backend, input.op, None)
        if method is None:
            raise ApplicationError(
                f"Backend {input.backend_ref!r} has no operation {input.op!r}.",
                type="DeepAgentsUnknownBackendOp",
                non_retryable=True,
            )
        result = method(
            *_serde.load_backend_result(input.args),
            **_serde.load_backend_result(input.kwargs),
        )
        if asyncio.iscoroutine(result):
            result = await result
        # Protocol results are plain dataclasses whose attributes the
        # middleware reads in-workflow — tag them so the stub can rebuild
        # the real type instead of receiving a decayed dict.
        return BackendOpOutput(result=_serde.dump_backend_result(result))
