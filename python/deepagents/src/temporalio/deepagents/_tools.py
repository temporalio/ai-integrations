"""The tool + backend seams: the explicit per-unit Workflow-vs-Activity choice.

Deep Agents holds its tools and filesystem/shell backends in-workflow. A tool or
backend op that only reads and writes ``DeepAgentState`` is pure and belongs in
the workflow (deterministic, replay-safe). One that does real I/O — a web
search, a shell command, a disk write — must not run there. This module gives
the user three explicit ways to move that work to an activity:

* :func:`activity_as_tool` — expose an existing ``@activity.defn`` as a Deep
  Agents tool (Temporal adopters already have activities; don't make them
  re-declare);
* :func:`tool_as_activity` — wrap a LangChain ``BaseTool`` / callable so its
  execution runs as an activity;
* :class:`TemporalBackend` — wrap a real-I/O backend so each file/exec op runs
  as an activity.

The choice is always explicit: an unwrapped non-builtin tool runs in-workflow,
and the plugin warns at construction so that is a conscious decision, never a
silent one.

Registries here live in a ``temporalio``-namespaced (sandbox-passthrough) module
and are process-local. Tools must be registered at startup in every activity
worker. Instance-backed ``TemporalBackend`` wrappers require activities to run
in the same process; named ``TemporalBackend`` handles send a factory argument
so any activity worker can construct the backend independently.
"""

from __future__ import annotations

import importlib
import inspect
import threading
import uuid as _uuid
import warnings
import weakref
from collections.abc import Awaitable, Mapping
from datetime import timedelta
from functools import wraps
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Callable

from temporalio import activity as activity_mod
from temporalio import workflow
from temporalio.deepagents import _activity, _serde
from temporalio.exceptions import ActivityError, RetryState, is_cancelled_exception
from temporalio.workflow import ActivityConfig

# LangChain is a runtime dependency of the *tool seam*, but importing this module
# must not require it (the plugin imports it just to read tool defaults). So the
# ``langchain_core.tools`` symbols are imported lazily inside the functions that
# actually build tools; ``from __future__ import annotations`` keeps the type
# hints below as strings so they never touch LangChain at import time.
if TYPE_CHECKING:
    from langchain_core.tools import BaseTool


# ---------------------------------------------------------------------------
# Tool registry (worker-side execution targets)
# ---------------------------------------------------------------------------

_TOOL_REGISTRY: dict[str, "BaseTool"] = {}
_BACKEND_REGISTRY: dict[str, Any] = {}
# Recently retired backends (insertion-ordered, oldest evicted first). Keeps a
# GC'd wrapper's backend reachable for the eviction -> activity-start window;
# see _unregister_backend. Bounded so retired backends cannot accumulate.
_RETIRED_BACKENDS: dict[str, Any] = {}
_RETIRED_BACKENDS_MAX = 512
# Serializes registration against the GC-time unregister in
# _unregister_backend, which may run on another thread.
_BACKEND_REGISTRY_LOCK = threading.Lock()

# Worker-wide default activity options for tools wrapped with tool_as_activity,
# set by the plugin. Not per-workflow state: fixed for the worker's lifetime, and
# this module is sandbox-passthrough so the workflow sees the configured value.
_tool_defaults: dict[str, Any] = {}


def set_tool_defaults(options: Any) -> None:
    """Install the plugin's default ``tool_activity_options`` (called by the plugin).

    Accepts a single ``ActivityConfig`` or a ``Mapping[tool_name, ActivityConfig]``.
    """
    _tool_defaults.clear()
    if options:
        _tool_defaults["__value__"] = options


def _resolve_tool_options(
    tool_name: str, instance_options: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Merge plugin tool defaults (possibly per-tool) with a per-call override."""
    opts: dict[str, Any] = {}
    default = _tool_defaults.get("__value__")
    if isinstance(default, Mapping):
        per_tool = default.get(tool_name)
        if isinstance(per_tool, Mapping):
            opts.update(per_tool)
        elif not any(isinstance(v, Mapping) for v in default.values()):
            opts.update(default)
    if instance_options:
        opts.update(instance_options)
    return opts


# Names of tools that route through Temporal (activity_as_tool / tool_as_activity),
# used to warn about unwrapped non-builtin tools at workflow build time.
_ROUTED_TOOL_NAMES: set[str] = set()

# Deep Agents' built-in tools run in-workflow (pure state mutations); they are
# not expected to be wrapped, so they never trigger the unwrapped-tool warning.
# These are the LLM-facing TOOL names (``read_file``, ``write_file``, …) —
# NOT the backend protocol method names in ``_BACKEND_OPS`` (``read``,
# ``aread``, …); the two namespaces intentionally differ.
_BUILTIN_TOOL_NAMES = frozenset(
    {
        "write_todos",
        "ls",
        "read_file",
        "write_file",
        "edit_file",
        "glob",
        "grep",
        "execute",
        "task",
    }
)


def register_tool(tool: BaseTool) -> None:
    """Record ``tool`` so :meth:`DeepAgentActivities.invoke_tool` can run it."""
    _TOOL_REGISTRY[tool.name] = tool


def warn_unwrapped_tools(tools: Any) -> None:
    """Warn once per unwrapped, non-builtin tool passed to ``create_deep_agent``.

    Running a tool in-workflow is only safe if it is pure/deterministic. The
    Workflow-vs-Activity choice must be conscious, so any user tool that was not
    routed through :func:`tool_as_activity` / :func:`activity_as_tool` gets a
    construction-time warning rather than silently executing in the workflow.
    """
    for tool in tools or ():
        name = getattr(tool, "name", getattr(tool, "__name__", None))
        if name is None or name in _BUILTIN_TOOL_NAMES or name in _ROUTED_TOOL_NAMES:
            continue
        warnings.warn(
            f"Tool {name!r} is passed to create_deep_agent unwrapped and will run "
            f"inside the workflow. That is only safe if it is pure/deterministic. "
            f"If it does I/O, wrap it with tool_as_activity(...) or expose an "
            f"existing activity with activity_as_tool(...).",
            stacklevel=3,
        )


def get_registered_tool(name: str) -> BaseTool | None:
    """Look up a tool registered by :func:`register_tool`."""
    return _TOOL_REGISTRY.get(name)


def register_backend(ref: str, backend: Any) -> None:
    """Record a backend so :meth:`DeepAgentActivities.backend_op` can reach it."""
    with _BACKEND_REGISTRY_LOCK:
        _BACKEND_REGISTRY[ref] = backend


def _unregister_backend(ref: str, inner: Any) -> None:
    """Retire ``ref`` from the registry if it still maps to ``inner``.

    GC hook for :class:`TemporalBackend` (via ``weakref.finalize``): a wrapper
    is typically constructed per workflow run, so without cleanup a long-lived
    worker accumulates one registry entry per run. The identity guard is
    load-bearing: refs are deterministic per run, so after a cache eviction a
    replay re-registers the *same* ref with a fresh inner backend — the evicted
    wrapper's finalizer must not remove that live registration.

    Retired entries move to the bounded :data:`_RETIRED_BACKENDS` store instead
    of vanishing: a ``backend_op`` activity scheduled just before a cache
    eviction can be delivered *after* the evicted wrapper is collected, and the
    replay that would re-register the ref only happens once that very activity
    completes. :func:`lookup_backend` still resolves the ref in that window.
    """
    with _BACKEND_REGISTRY_LOCK:
        if _BACKEND_REGISTRY.get(ref) is inner:
            del _BACKEND_REGISTRY[ref]
            _RETIRED_BACKENDS.pop(ref, None)
            _RETIRED_BACKENDS[ref] = inner
            while len(_RETIRED_BACKENDS) > _RETIRED_BACKENDS_MAX:
                _RETIRED_BACKENDS.pop(next(iter(_RETIRED_BACKENDS)))


def registered_backends() -> dict[str, Any]:
    """Return the live backend registry (read by the plugin at worker build)."""
    return _BACKEND_REGISTRY


def lookup_backend(ref: str) -> Any | None:
    """Resolve ``ref`` for a ``backend_op`` activity.

    Prefers the live registry, then falls back to recently retired entries so
    an activity dispatched before a cache eviction still resolves (see
    :func:`_unregister_backend`).
    """
    with _BACKEND_REGISTRY_LOCK:
        backend = _BACKEND_REGISTRY.get(ref)
        if backend is not None:
            return backend
        return _RETIRED_BACKENDS.get(ref)


# ---------------------------------------------------------------------------
# Terminal tool-activity failures returned to the model (opt-in)
# ---------------------------------------------------------------------------

# Prefix on the text a ``return_error_to_model`` wrapper hands the model when
# the tool's activity failed for good, so a failure is recognizable without
# parsing the sentence around it. The literal prefix is part of what the model
# is shown and of recorded history, which makes it a replay contract rather
# than a free wording choice.
_TOOL_ERROR_PREFIX = "[tool error]"

# Bound on the underlying failure's message carried into that text: it is
# recorded in workflow history and repeated into the next model prompt, so an
# arbitrarily long exception message must not bloat either.
_FAILURE_DETAIL_MAX = 300


def _failure_reason(error: ActivityError) -> str:
    """Say why there is no result, in the words the model should act on.

    An :class:`ActivityError` raised into workflow code means Temporal is done
    with that activity, but "non-retryable" and "retries exhausted" tell a model
    different things about whether re-asking could help.
    """
    if error.retry_state == RetryState.NON_RETRYABLE_FAILURE:
        return "failed with a non-retryable error"
    if error.retry_state == RetryState.MAXIMUM_ATTEMPTS_REACHED:
        return "failed after exhausting its retries"
    if error.retry_state == RetryState.TIMEOUT:
        return "timed out"
    return "failed and Temporal will not retry it again"


def _tool_failure_text(tool_name: str, error: ActivityError) -> str:
    """Render a terminal :class:`ActivityError` as text the model can act on.

    The text names the tool, says why there is no result, and carries the
    underlying failure's type and message (truncated at
    :data:`_FAILURE_DETAIL_MAX`) — the model then decides what to do next
    instead of the run dying. Saying "not evidence about the task" is
    deliberate: a bare "failed" reads like an empty result, which invites the
    model to invent one.
    """
    cause = error.cause or error
    kind = getattr(cause, "type", None) or type(cause).__name__
    message = (getattr(cause, "message", None) or str(cause) or "no detail")[
        :_FAILURE_DETAIL_MAX
    ]
    return (
        f"{_TOOL_ERROR_PREFIX} {tool_name} {_failure_reason(error)} "
        f"({kind}: {message}). "
        "This is a tool failure, not evidence about the task: do not invent a "
        "result from it. Try a different tool or arguments, or report the failure."
    )


async def _as_tool_result(awaitable: Awaitable[Any], tool_name: str) -> Any:
    """Await a tool dispatch, turning a terminal activity failure into text.

    Returns the awaited result, or — when the activity failed for good — the
    failure text :func:`_tool_failure_text` builds, which the caller returns as
    the tool's content. ``awaitable`` may cover more than the dispatch itself
    (:func:`tool_as_activity` unpacks the activity output inside it); the
    conversion applies to the whole await. LangGraph's ToolNode re-raises every
    error but an argument error, so an ActivityError that reaches it fails the
    whole agent run and discards the work already recorded in the transcript.
    Callers opt in per tool, and only for this narrow conversion:
    cancellation is re-raised, because an activity cancelled under a workflow
    cancellation is not a tool result, and the workflow must still end up
    cancelled.
    """
    try:
        return await awaitable
    except ActivityError as err:
        if is_cancelled_exception(err):
            raise
        return _tool_failure_text(tool_name, err)


# ---------------------------------------------------------------------------
# activity_as_tool
# ---------------------------------------------------------------------------


def activity_as_tool(
    activity: Callable,
    *,
    start_to_close_timeout: timedelta,
    name: str | None = None,
    description: str | None = None,
    retry_policy: Any = None,
    summary: str | None = None,
    return_error_to_model: bool = False,
) -> BaseTool:
    """Expose an existing Temporal activity as a Deep Agents tool.

    The returned tool advertises the activity's argument schema to the model and,
    when called in-workflow, dispatches to the activity via
    ``workflow.execute_activity`` — Temporal owns its retries and timeout.

    Args:
        activity: A function decorated with ``@activity.defn``.
        start_to_close_timeout: Required per-call timeout for the activity.
        name: Override the tool name advertised to the model (defaults to
            the activity definition name).
        description: Override the tool description advertised to the model
            (defaults to the activity docstring).
        retry_policy: Optional Temporal retry policy for the activity.
        summary: Optional ``summary=`` recorded on each activity invocation.
        return_error_to_model: Hand a terminal activity failure to the model as
            ``[tool error]`` text instead of raising. Off by default: the
            ActivityError then propagates and fails the agent run. The text
            carries the underlying failure's type and message (truncated), so
            opting in sends the activity's error text to the model. Cancellation
            always propagates.
    """
    with workflow.unsafe.imports_passed_through():
        from langchain_core.tools import StructuredTool

    defn = activity_mod._Definition.from_callable(activity)
    if defn is None:
        raise ValueError(
            "activity_as_tool requires a function decorated with @activity.defn; "
            f"{getattr(activity, '__name__', activity)!r} is not an activity."
        )
    tool_name = name or defn.name
    if tool_name is None:
        raise ValueError(
            "activity_as_tool requires a named activity (dynamic activities "
            "have no definition name); pass name= explicitly."
        )
    tool_desc = description or (activity.__doc__ or f"Temporal activity {tool_name}.")
    act_summary = summary or f"tool:{tool_name}"

    # Temporal activities take positional arguments only. StructuredTool infers
    # the model-facing schema from ``_run``'s signature, so we mirror the activity's
    # own parameter names onto ``_run`` (via ``@wraps``) and then map the keyword
    # call the model produces onto the activity's parameters, one argument each.
    import inspect

    params = [
        p
        for p in inspect.signature(activity).parameters.values()
        if p.name not in ("self", "cls")
    ]

    @wraps(activity)
    async def _run(*args: Any, **kwargs: Any) -> Any:
        values = list(args)
        for param in params[len(args) :]:
            if param.name in kwargs:
                values.append(kwargs[param.name])
            elif param.default is not inspect.Parameter.empty:
                # Pass the default too: the worker decodes arguments with the
                # activity's type hints only when every parameter has a value.
                values.append(param.default)
            else:
                raise TypeError(f"{tool_name} missing argument {param.name!r}")
        dispatch = workflow.execute_activity(
            activity,
            args=values,
            start_to_close_timeout=start_to_close_timeout,
            retry_policy=retry_policy,
            summary=act_summary,
        )
        if return_error_to_model:
            return await _as_tool_result(dispatch, tool_name)
        return await dispatch

    _ROUTED_TOOL_NAMES.add(tool_name)
    return StructuredTool.from_function(
        coroutine=_run,
        name=tool_name,
        description=tool_desc,
    )


# ---------------------------------------------------------------------------
# tool_as_activity
# ---------------------------------------------------------------------------


def tool_as_activity(
    tool: BaseTool | Callable,
    *,
    start_to_close_timeout: timedelta,
    activity_options: Mapping[str, Any] | None = None,
    return_error_to_model: bool = False,
) -> BaseTool:
    """Wrap a LangChain tool / callable so its execution runs as an activity.

    The underlying tool is registered on the worker; the returned tool keeps the
    same name and argument schema (so the model's calls are unchanged) but, when
    invoked in-workflow, dispatches ``deepagents.invoke_tool`` instead of running
    the tool body inline.

    Call this at module import or worker startup on every process that can
    execute tool activities. Registration inside a workflow alone does not
    make the tool available in another worker process.

    Args:
        tool: A LangChain ``BaseTool`` or a plain callable (wrapped as a
            ``StructuredTool``).
        start_to_close_timeout: Default ``start_to_close_timeout`` for the
            ``deepagents.invoke_tool`` activity, applied when neither
            ``activity_options`` nor the plugin's ``tool_activity_options``
            sets one.
        activity_options: Extra ``execute_activity`` options (retry policy,
            task queue routing, ...) overriding the plugin's defaults.
        return_error_to_model: Hand a terminal activity failure to the model as
            ``[tool error]`` text instead of raising. Off by default: the
            ActivityError then propagates and fails the agent run. The text
            carries the underlying failure's type and message (truncated), so
            opting in sends the activity's error text to the model. Cancellation
            always propagates.
    """
    with workflow.unsafe.imports_passed_through():
        from langchain_core.tools import BaseTool, StructuredTool

    base_tool: BaseTool
    if isinstance(tool, BaseTool):
        base_tool = tool
    else:
        base_tool = StructuredTool.from_function(
            tool if not _is_coroutine(tool) else None,
            coroutine=tool if _is_coroutine(tool) else None,
        )
    register_tool(base_tool)

    tool_name = base_tool.name
    opts = _resolve_tool_options(tool_name, activity_options)
    opts.setdefault("start_to_close_timeout", start_to_close_timeout)

    async def _run(**kwargs: Any) -> Any:
        from temporalio.deepagents.workflow import call_tool

        tool_call_id = kwargs.pop("__tool_call_id__", None) or workflow.uuid4().hex
        activity_input = _activity.ToolActivityInput(
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            args=kwargs,
        )
        dispatch = call_tool(
            activity_input,
            summary=f"tool:{tool_name}",
            **opts,
        )

        async def _dispatched_content() -> Any:
            output = await dispatch
            message = _serde.load_object(output.message)
            # Return CONTENT, not the pre-built ToolMessage: the activity cannot
            # know the model's real tool_call_id, so a ToolMessage assembled
            # there carries a generated id that a real provider rejects as an
            # unpaired tool_result. Given plain content, the tool node stamps
            # the model's own id — exactly as it does for unwrapped tools.
            with workflow.unsafe.imports_passed_through():
                from langchain_core.messages import ToolMessage

            if isinstance(message, ToolMessage):
                return message.content
            return message

        if return_error_to_model:
            # Only the failure conversion bypasses the unpacking, and only by
            # returning text: a successful call still unwraps its activity
            # output exactly as the default path does.
            return await _as_tool_result(_dispatched_content(), tool_name)
        return await _dispatched_content()

    _ROUTED_TOOL_NAMES.add(tool_name)
    return StructuredTool(
        name=tool_name,
        description=base_tool.description,
        args_schema=base_tool.args_schema,  # type: ignore[arg-type]
        coroutine=_run,
    )


def _is_coroutine(fn: Any) -> bool:
    import inspect

    return inspect.iscoroutinefunction(fn)


# ---------------------------------------------------------------------------
# TemporalBackend
# ---------------------------------------------------------------------------

# Async protocol methods and their sync twins. deepagents' ``BackendProtocol``
# implements each async DEFAULT as ``asyncio.to_thread(sync_twin, ...)``; the
# deterministic workflow event loop has no thread executor, so any built-in
# tool call against an unwrapped in-workflow backend (e.g. the default
# ``StateBackend``) raises ``NotImplementedError``. A real model hits this on
# its first spontaneous ``grep``/``read_file`` call; scripted-model tests that
# never call built-ins sail past it.
_ASYNC_TO_SYNC_OPS: dict[str, str] = {
    "als": "ls",
    "als_info": "ls_info",
    "aread": "read",
    "awrite": "write",
    "aedit": "edit",
    "aglob": "glob",
    "aglob_info": "glob_info",
    "agrep": "grep",
    "agrep_raw": "grep_raw",
    "adownload_files": "download_files",
    "aupload_files": "upload_files",
    "aexecute": "execute",
    "adelete": "delete",
}

_original_backend_async_defaults: dict[str, Any] = {}


def install_backend_async_patch() -> None:
    """Make ``BackendProtocol``'s async defaults workflow-safe.

    Inside a workflow, run the sync twin inline: for state-only backends that
    is deterministic and semantically identical to the upstream default,
    which merely moves the same sync call onto a worker thread. Outside a
    workflow (activities, clients) the upstream default — thread hop plus
    timeout guard — is used unchanged. Subclasses that override an async
    method natively are unaffected; only the protocol defaults are replaced.
    Idempotent.
    """
    # importlib: `deepagents` is absent on Python 3.10 environments (its floor
    # is 3.11), so a static import here fails type-checking there.
    BackendProtocol = importlib.import_module(
        "deepagents.backends.protocol"
    ).BackendProtocol

    if _original_backend_async_defaults:
        return
    for async_name, sync_name in _ASYNC_TO_SYNC_OPS.items():
        original = BackendProtocol.__dict__.get(async_name)
        if original is None:
            continue

        def _make(sync_name: str, original: Any) -> Any:
            async def patched(self: Any, *args: Any, **kwargs: Any) -> Any:
                if workflow.in_workflow():
                    return getattr(self, sync_name)(*args, **kwargs)
                return await original(self, *args, **kwargs)

            return patched

        _original_backend_async_defaults[async_name] = original
        setattr(BackendProtocol, async_name, _make(sync_name, original))


def uninstall_backend_async_patch() -> None:
    """Restore ``BackendProtocol``'s upstream async defaults."""
    if not _original_backend_async_defaults:
        return
    BackendProtocol = importlib.import_module(
        "deepagents.backends.protocol"
    ).BackendProtocol

    for async_name, original in _original_backend_async_defaults.items():
        setattr(BackendProtocol, async_name, original)
    _original_backend_async_defaults.clear()


# Backend METHOD names (deepagents ``BackendProtocol`` +
# ``SandboxBackendProtocol``) whose calls must cross the activity boundary:
# every sync I/O method, its async ``a``-prefixed twin, and the sandbox/shell
# execute pair. The async twins are load-bearing — deepagents' filesystem
# middleware drives backends through ``als``/``aread``/``awrite``/… — so
# intercepting only sync names lets an agent's built-in file tools run I/O
# in-workflow. These are backend PROTOCOL method names, distinct from the
# LLM-facing tool names in ``_BUILTIN_TOOL_NAMES`` (``read_file`` etc.).
_BACKEND_OPS = (
    # Sync protocol surface.
    "ls",
    "ls_info",
    "read",
    "write",
    "edit",
    "glob",
    "glob_info",
    "grep",
    "grep_raw",
    "download_files",
    "upload_files",
    "execute",
    "delete",
    # Async twins (what FilesystemMiddleware actually calls).
    "als",
    "als_info",
    "aread",
    "awrite",
    "aedit",
    "aglob",
    "aglob_info",
    "agrep",
    "agrep_raw",
    "adownload_files",
    "aupload_files",
    "aexecute",
    "adelete",
)

# Capability-gated ops. deepagents decides whether a backend has ``delete`` from
# the CLASS (``type(backend).delete`` against the protocol default) and whether
# it can ``execute`` from a nominal ``isinstance`` check against
# ``SandboxBackendProtocol`` (an ABC subclass, not a structural Protocol).
# Neither can be answered per instance, so the wrapper mirrors the inner
# backend at class level; see :func:`_wrapper_class_for`. Some probes read the
# method's SIGNATURE as well (``execute_accepts_timeout`` looks for a
# ``timeout`` parameter on ``type(backend).execute``), so these dispatchers
# also carry the inner method's signature; see :func:`_make_backend_op`.
_OPTIONAL_OPS = frozenset({"delete", "adelete", "execute", "aexecute"})


class TemporalBackend:
    """Route a real-I/O backend's operations through Temporal activities.

    Wrap a ``FilesystemBackend`` / ``LocalShellBackend`` / ``StoreBackend`` /
    ``CompositeBackend`` so each file or shell operation becomes a durable
    ``deepagents.backend_op`` activity instead of touching disk / shell from the
    workflow. State-only backends (``StateBackend``) need no wrapping — they are
    pure workflow state and run in-workflow.

    Unknown attribute access is forwarded to the inner backend so backend
    metadata / configuration the agent reads (but that does no I/O) still works.

    Wrapping an instance requires the operation's activity to run in the same
    process as the workflow. For multiple worker processes or serverless workers,
    pass the name of a factory registered with
    ``DeepAgentsPlugin(backends=...)`` instead. No backend is constructed
    in the workflow; each activity attempt invokes the worker-side factory.

    Args:
        inner: Backend instance, or the name of a worker-side backend factory.
        activity_options: SDK ``ActivityConfig`` for backend activity timeouts,
            retries and routing.
        factory_argument: Serializable argument passed to a named factory, such
            as a sandbox ID. It is recorded in workflow history; keep credentials
            on the worker. Factories declaring one positional parameter receive
            this value (``None`` by default); parameterless factories are called
            without arguments. Operations and retries must reconnect to the same
            external state; instance memory is not carried between operations.
        backend_type: Required for a named factory. Class used to advertise
            delete and shell capabilities and method signatures, without
            instantiating it. For a composite backend, supply a class exposing
            the default backend's capabilities.
        metadata: For a named factory, attributes read by workflow code, such
            as ``id`` for a sandbox. Values must be deterministic and require no I/O.
    """

    def __new__(
        cls,
        inner: Any,
        *,
        activity_options: ActivityConfig | None = None,
        factory_argument: Any = None,
        backend_type: type | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "TemporalBackend":
        """Pick the wrapper class that mirrors ``inner``'s optional capabilities.

        Applies to subclasses too: a user's ``class MyBackend(TemporalBackend)``
        gets a mirrored subclass of ``MyBackend``, so ``isinstance`` and any
        methods it defines are preserved.
        """
        if isinstance(inner, str):
            if not inner:
                raise ValueError("A backend factory name is required.")
            if backend_type is None:
                raise ValueError(
                    "backend_type is required for a named backend factory."
                )
            mirror = backend_type
        else:
            if (
                factory_argument is not None
                or backend_type is not None
                or metadata is not None
            ):
                raise ValueError(
                    "factory_argument, backend_type and metadata require a named backend factory."
                )
            mirror = inner
        target: type[TemporalBackend] = (
            cls
            if getattr(cls, "_temporal_mirror_of", None) is not None
            else _wrapper_class_for(mirror, cls)
        )
        return object.__new__(target)

    def __init__(
        self,
        inner: Any,
        *,
        activity_options: ActivityConfig | None = None,
        factory_argument: Any = None,
        backend_type: type | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Wrap an instance or reference a named worker-side backend factory."""
        self._backend_factory: str | None = None
        self._factory_argument: Any = None
        self._configure_dispatch(activity_options)
        if isinstance(inner, str):
            self._inner = SimpleNamespace(**dict(metadata or {}))
            self._backend_factory = inner
            self._factory_argument = factory_argument
            return
        self._inner = inner
        register_backend(self._ref, inner)
        # Balance the registration when this wrapper is garbage-collected
        # (workflow completion / cache eviction) so per-run backends do not
        # accumulate in the worker-global registry. The finalizer captures
        # (ref, inner) — not ``self`` — so it cannot keep the wrapper alive,
        # and it only removes its own registration (see _unregister_backend).
        self._finalizer = weakref.finalize(self, _unregister_backend, self._ref, inner)

    def _configure_dispatch(self, activity_options: ActivityConfig | None) -> None:
        # A deterministic id: workflow.uuid4() is seeded per-run, so the ref is
        # identical across replays (unlike id(inner)). Falls back to a plain uuid
        # when a backend is wrapped outside a workflow (e.g. in a plain test).
        if workflow.in_workflow():
            self._ref = f"backend:{workflow.uuid4().hex}"
        else:
            self._ref = f"backend:{_uuid.uuid4().hex}"
        self._opts: dict[str, Any] = dict(activity_options or {})
        self._opts.setdefault("start_to_close_timeout", timedelta(minutes=1))

    async def _dispatch(self, op: str, *args: Any, **kwargs: Any) -> Any:
        from temporalio.deepagents.workflow import call_backend_op

        activity_input = _activity.BackendOpInput(
            backend_ref=self._ref,
            op=op,
            args=_serde.dump_backend_result(list(args)),
            kwargs=_serde.dump_backend_result(dict(kwargs)),
            backend_factory=self._backend_factory,
            factory_argument=self._factory_argument,
        )
        output = await call_backend_op(
            activity_input,
            summary=f"backend:{op}",
            **self._opts,
        )
        return _serde.load_backend_result(output.result)

    def __getattr__(self, name: str) -> Any:
        """Forward anything that is not an I/O op to the inner backend unchanged."""
        return getattr(self._inner, name)


def _make_backend_op(name: str, mirror: Any = None) -> Callable[..., Any]:
    """Build the activity dispatcher for backend op ``name``.

    ``mirror`` is the inner backend's own method for the op. When given, the
    dispatcher advertises that method's signature (through ``__signature__``;
    it still accepts and forwards any arguments). deepagents probes some
    capabilities by signature: ``execute_accepts_timeout`` looks for a
    ``timeout`` parameter on ``type(backend).execute`` and, unlike the
    ``max_count`` probe, does not take ``**kwargs`` as a stand-in, so a bare
    ``(*args, **kwargs)`` dispatcher made a wrapped ``LocalShellBackend``
    refuse the per-command ``timeout`` its unwrapped self accepts. Carrying the
    inner signature keeps every such probe answering exactly as it does for the
    inner backend, including for a sandbox that does *not* accept ``timeout``
    (deepagents then declines the call up front instead of the activity failing
    on an unexpected keyword).
    """

    async def _op(self: TemporalBackend, *args: Any, **kwargs: Any) -> Any:
        return await self._dispatch(name, *args, **kwargs)

    _op.__name__ = _op.__qualname__ = name
    if mirror is not None:
        try:
            signature = inspect.signature(mirror)
        except (TypeError, ValueError):
            pass
        else:
            setattr(_op, "__signature__", signature)
    return _op


# The I/O ops are real class-level methods, not ``__getattr__`` products, so
# class-based capability checks in deepagents see them.
for _op_name in _BACKEND_OPS:
    if _op_name not in _OPTIONAL_OPS:
        setattr(TemporalBackend, _op_name, _make_backend_op(_op_name))

_wrapper_classes: dict[tuple[type, type, bool, bool], type[TemporalBackend]] = {}


def _wrapper_class_for(
    inner: Any, base: type[TemporalBackend]
) -> type[TemporalBackend]:
    """Return the subclass of ``base`` whose optional capabilities mirror ``inner``.

    ``delete`` / ``adelete``: a delete-capable inner backend gets dispatchers
    like every other op; any other inner backend keeps the protocol defaults,
    so deepagents disables its delete tool exactly as it would for the
    unwrapped backend. Execution: deepagents offers its shell tool only to a
    nominal ``SandboxBackendProtocol`` instance, so when deepagents reports the
    inner backend as execution-capable the mirror also derives from that class
    (with ``execute`` / ``aexecute`` dispatchers and ``id`` forwarded); a wrapper
    around a plain filesystem or store backend stays a non-sandbox class. The
    dispatchers built here carry the inner methods' signatures, which deepagents
    reads for finer probes such as ``execute_accepts_timeout``; see
    :func:`_make_backend_op`. An op that ``base`` already defines (a user
    subclass) is left alone. Without deepagents installed there is nothing to
    mirror and ``base`` is used as-is.
    """
    try:
        protocol = importlib.import_module("deepagents.backends.protocol")
        filesystem = importlib.import_module("deepagents.middleware.filesystem")
    except ImportError:
        return base
    inner_cls = inner if isinstance(inner, type) else type(inner)
    default_delete = protocol.BackendProtocol.delete
    has_delete = getattr(inner_cls, "delete", default_delete) is not default_delete
    has_execute = (
        issubclass(inner_cls, protocol.SandboxBackendProtocol)
        if isinstance(inner, type)
        else bool(filesystem.supports_execution(inner))
    )
    # Keyed by the inner CLASS too, since the dispatchers carry its method
    # signatures. ``has_execute`` stays in the key: for a ``CompositeBackend``
    # deepagents answers it per instance, from the default backend.
    key = (base, inner_cls, has_delete, has_execute)
    cls = _wrapper_classes.get(key)
    if cls is None:
        namespace: dict[str, Any] = {
            "__module__": base.__module__,
            "__qualname__": base.__qualname__,
            "__doc__": base.__doc__,
            "_temporal_mirror_of": base,
        }
        for op in ("delete", "adelete"):
            if hasattr(base, op):
                continue
            namespace[op] = (
                _make_backend_op(op, mirror=getattr(inner_cls, op, None))
                if has_delete
                else getattr(protocol.BackendProtocol, op)
            )
        bases: tuple[type, ...] = (base,)
        if has_execute:
            sandbox_cls = protocol.SandboxBackendProtocol
            if not issubclass(base, sandbox_cls):
                bases = (base, sandbox_cls)
            for op in ("execute", "aexecute"):
                if not hasattr(base, op):
                    namespace[op] = _make_backend_op(
                        op, mirror=getattr(inner_cls, op, None)
                    )
            if not hasattr(base, "id"):
                # The sandbox class defines ``id``; keep reading the inner
                # backend's, which ``__getattr__`` would otherwise have served.
                namespace["id"] = property(lambda self: self._inner.id)
        # Workflow tasks run on a thread pool, so two runs can build the same
        # key at once; the first class stored wins for both callers.
        cls = _wrapper_classes.setdefault(key, type(base.__name__, bases, namespace))
    return cls
