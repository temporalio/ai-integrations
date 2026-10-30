"""The plugin object users add to ``plugins=[...]``.

:class:`DeepAgentsPlugin` wires everything together so that existing
``deepagents`` code — ``create_deep_agent(...).ainvoke(...)`` — runs durably
inside a ``@workflow.defn`` with no other changes:

* registers the four activities that carry the nondeterministic work;
* installs the LangChain-aware data converter (composing, never clobbering, a
  user converter);
* passes the LangChain / LangGraph / deepagents import tree through the workflow
  sandbox;
* wraps the worker run in a ``run_context`` that patches Deep Agents' model
  resolution seam (``deepagents.graph.resolve_model``) to auto-route bare
  ``model=`` strings through activities — regardless of how the user imported
  ``create_deep_agent``;
* registers :class:`DeepAgentsWorkflowError` as a workflow-failure type;
* pushes the model / tool dispatch defaults down to the seams that read them.

The plugin auto-propagates from a ``Client`` to any ``Worker`` built from it, so
add it on exactly one side. If one instance is on the Client and one on the
Worker, the Worker instance does not add an activity whose name (or, for a
dynamic activity, the dynamic slot) is already registered, and still
registered after its own step, by the same callable, a wrapper of it
(``__wrapped__``), or, for a second instance, the Client instance's copy of the
same built-in activity. The Worker keeps what the Client side registered,
instead of failing with "More than one activity named ...". Any other repeated
activity name, such as two instances on one side or a caller activity with a
plugin activity's name, and any second dynamic activity, is left to the Worker,
which refuses it.
"""

from __future__ import annotations

import inspect
import sys
import warnings
from collections.abc import AsyncGenerator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import timedelta
from typing import Any, Callable

from temporalio import activity as activity_mod
from temporalio.converter import DataConverter
from temporalio.deepagents import _serde, _tools
from temporalio.deepagents._activity import DeepAgentActivities
from temporalio.deepagents.workflow import DeepAgentsWorkflowError
from temporalio.plugin import SimplePlugin
from temporalio.worker import WorkflowRunner
from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner

# Runtime floor for the Deep Agents control loop. Kept in a constant so
# static checkers do not narrow `sys.version_info` comparisons into
# unreachable-code findings on new interpreters.
_MIN_PYTHON = (3, 11)


class DeepAgentsPlugin(SimplePlugin):
    """Temporal plugin that makes LangChain Deep Agents durable.

    Args:
        model_provider: Builds a real chat model from a name string, worker-side.
            This is where API keys live; only the model name ever crosses the
            workflow boundary. Defaults to LangChain's ``init_chat_model`` with
            LLM-SDK retries disabled (Temporal owns retries).
        model_activity_options: ``ActivityConfig`` (or ``Mapping[model_name,
            ActivityConfig]``) for the model activities — timeouts, retry policy.
        tool_activity_options: ``ActivityConfig`` (or ``Mapping[tool_name,
            ActivityConfig]``) default for tools wrapped with ``tool_as_activity``.
        backends: Map of names to worker-side backend factories used by
            ``TemporalBackend("name", ...)``. Factories accept zero arguments or
            one positional ``factory_argument`` and return a backend, synchronously
            or asynchronously. A one-parameter factory receives ``None`` when the
            workflow supplies no argument.
            Configure the same names on every worker polling backend activities;
            credentials and connections belong in these factories.
        streaming_topic: When set, model calls stream through the streaming
            activity and publish chunk batches to this workflow-streams topic.
        streaming_batch_interval: How long the streaming activity coalesces
            chunks before publishing a batch.
        passthrough_modules: Extra sandbox-passthrough modules, merged with the
            plugin's LangChain/deepagents defaults.
        data_converter: Explicit converter override. ``None`` adapts the client
            or replayer's existing converter, preserving its codec and other
            settings. The SDK default payload converter is upgraded; converters
            using ``DeepAgentsPayloadConverter`` or a subclass are accepted;
            other payload converters raise.
    """

    def __init__(
        self,
        *,
        model_provider: Callable[[str], Any] | None = None,
        model_activity_options: Any = None,
        tool_activity_options: Any = None,
        backends: Mapping[str, Callable[..., Any]] | None = None,
        streaming_topic: str | None = None,
        streaming_batch_interval: timedelta = timedelta(milliseconds=100),
        passthrough_modules: Sequence[str] | None = None,
        data_converter: DataConverter | None = None,
    ) -> None:
        """Configure the plugin; see the class docstring for parameters."""
        if sys.version_info < _MIN_PYTHON:
            warnings.warn(
                "DeepAgentsPlugin requires Python >= 3.11 (deepagents pins "
                ">=3.11); the Deep Agents control loop relies on contextvars "
                "propagation through asyncio that older versions lack.",
                stacklevel=2,
            )

        self._passthrough_modules = passthrough_modules
        # Held on the instance so the wiring is statically traceable: each is
        # passed straight into the call that consumes it (below), not stashed as
        # dead config.
        self._tool_activity_options = tool_activity_options
        self._data_converter = (
            _serde.build_data_converter(data_converter)
            if data_converter is not None
            else None
        )

        # Push dispatch defaults down to the model and tool seams that read them.
        # These live in ``_serde`` / ``_tools`` (langchain-free modules) so
        # constructing the plugin never imports LangChain. ``tool_activity_options``
        # is stored under ``_tools._tool_defaults`` and read back on the tool
        # dispatch path by ``_tools._resolve_tool_options(...)`` — which
        # ``tool_as_activity`` calls to compute each tool activity's timeout/retry
        # when the caller does not override ``activity_options``.
        _serde.set_settings(
            model_activity_options=model_activity_options,
            streaming_topic=streaming_topic,
        )
        # wired-via-composition: consumed on the tool dispatch path in _tools.py
        # (_resolve_tool_options), renamed to ``options`` at the callsite.
        _tools.set_tool_defaults(self._tool_activity_options)

        # The activities that carry the nondeterministic work. model_provider and
        # the batch interval are captured here so API keys never enter an input.
        self._activities = DeepAgentActivities(
            model_provider=model_provider,
            backends=backends,
            streaming_batch_interval=streaming_batch_interval,
        )
        # Kept apart from the public ``activities`` setting, which callers may change.
        self._builtin_activities = (
            self._activities.invoke_model,
            self._activities.invoke_model_streaming,
            self._activities.invoke_tool,
            self._activities.backend_op,
        )

        super().__init__(
            "langchain.DeepAgentsPlugin",
            activities=list(self._builtin_activities),
            data_converter=self._configure_data_converter,
            workflow_runner=self._make_workflow_runner(),
            workflow_failure_exception_types=[DeepAgentsWorkflowError],
            run_context=self._run_context,
        )

    def _configure_data_converter(
        self, converter: DataConverter | None
    ) -> DataConverter:
        return _serde.build_data_converter(
            self._data_converter if self._data_converter is not None else converter
        )

    # -- sandbox passthrough -------------------------------------------------

    def _make_workflow_runner(
        self,
    ) -> Callable[[WorkflowRunner | None], WorkflowRunner]:
        modules = _serde.resolve_passthrough_modules(self._passthrough_modules)

        def workflow_runner(runner: WorkflowRunner | None) -> WorkflowRunner:
            if runner is None:
                raise ValueError("No WorkflowRunner provided to DeepAgentsPlugin.")
            if isinstance(runner, SandboxedWorkflowRunner):
                return replace(
                    runner,
                    restrictions=runner.restrictions.with_passthrough_modules(*modules),
                )
            return runner

        return workflow_runner

    # -- run context ---------------------------------------------------------

    @asynccontextmanager
    async def _run_context(self) -> AsyncGenerator[None, None]:
        """Patch Deep Agents' model resolution seam for the worker's lifetime.

        The patch (on ``deepagents.graph.resolve_model``, the seam
        ``create_deep_agent`` uses for the agent and every sub-agent) only
        rewrites ``model=`` strings when running inside a workflow, so it is inert
        on plain clients / activity workers. Determinism itself
        rides on the same mechanism ``temporalio.langgraph`` uses — the loop is
        in-workflow and every nondeterministic call is an activity — so no
        speculative time/uuid shims are installed here.

        One shim *is* required: LangChain's ``create_agent`` factory wraps every
        model node with LangSmith's ``@traceable``, whose async path hops onto a
        thread via ``asyncio.run_in_executor`` — which the deterministic workflow
        event loop does not implement (it raises ``NotImplementedError``). Tracing
        is an observability concern that must not run in-workflow, so we install
        LangSmith's own Temporal escape hatch (``set_runtime_overrides``) to run
        that setup inline when ``in_workflow()``, and defer to the default thread
        hop everywhere else (activities / clients, where tracing is fine).
        """
        # Imported lazily (not at module top) so plugin construction stays
        # langchain-free; the patch is only needed once the worker is running.
        # The import itself is inside the guard: on a worker without LangChain
        # installed (e.g. one that only runs the plugin's determinism / failure
        # paths) importing ``_model`` raises ``ModuleNotFoundError``, and the
        # worker must still start — it simply runs without the auto-wrap patch.
        patched = False
        try:
            from temporalio.deepagents import _model, _tools

            _model.install_model_patch()
            _tools.install_backend_async_patch()
            patched = True
        except ImportError as exc:  # LangChain / deepagents absent on this worker
            # Deliberately ImportError only: a *missing* optional dependency
            # must not stop the worker, but a genuine patch-installation bug
            # (e.g. upstream renaming the seam raises AttributeError) must
            # surface at startup, not degrade into silent in-workflow calls.
            warnings.warn(
                f"DeepAgentsPlugin could not patch create_deep_agent ({exc}); "
                "use explicit TemporalModel(...) instances to route model calls "
                "through activities.",
                stacklevel=2,
            )
        # Installed for the life of the process, never uninstalled — see
        # _install_langsmith_temporal_override for why.
        _install_langsmith_temporal_override()
        try:
            yield
        finally:
            if patched:
                # Import is cached: patched=True implies the import above succeeded.
                from temporalio.deepagents import _model, _tools

                _model.uninstall_model_patch()
                _tools.uninstall_backend_async_patch()

    # -- worker config -------------------------------------------------------

    def configure_worker(self, config: Any) -> Any:
        """Skip activities the Client's instance already registered."""
        before = list(config.get("activities") or [])
        config = super().configure_worker(config)
        client_instance = self._client_instance(config)
        if client_instance is not None:
            config["activities"] = self._without_repeats(
                before, config.get("activities") or [], client_instance
            )
        return config

    def _client_instance(self, config: Any) -> DeepAgentsPlugin | None:
        """The Client's instance, if this is the Worker's only instance.

        Only one instance on the Client and one on the Worker qualify. They can
        be the same object, which the Worker then calls twice.
        """
        client = config.get("client")
        client_plugins = client.config().get("plugins", []) if client else []
        on_client = [p for p in client_plugins if isinstance(p, DeepAgentsPlugin)]
        on_worker = [
            p for p in config.get("plugins") or [] if isinstance(p, DeepAgentsPlugin)
        ]
        if len(on_client) == 1 and len(on_worker) == 1 and on_worker[0] is self:
            return on_client[0]
        return None

    def _without_repeats(
        self,
        before: Sequence[Any],
        after: Sequence[Any],
        client_instance: DeepAgentsPlugin,
    ) -> list[Any]:
        """Drop the activities this call added that repeat a kept registration.

        An activity registered before this call that is still in ``after``
        keeps its activity name, or the dynamic slot if it is dynamic. An added
        activity with that name or slot repeats it when the kept callable is it
        or wraps it, or, for one of this instance's built-in activities, is or
        wraps the Client instance's copy of it. Dropping such a repeat never
        changes which code runs under the name; any other repeat reaches the
        Worker's duplicate-name or duplicate-dynamic check.
        """
        remaining = list(before)
        added: list[bool] = []
        kept_keys: dict[tuple[str | None], list[Any]] = {}
        for act in after:
            index = _index_of(act, remaining)
            added.append(index is None)
            if index is not None:
                del remaining[index]
                key = _activity_key(act)
                if key is not None:
                    kept_keys.setdefault(key, []).extend(_wrapped_chain(act))
        counterparts = list(
            zip(self._builtin_activities, client_instance._builtin_activities)
        )
        kept: list[Any] = []
        for act, is_added in zip(after, added):
            key = _activity_key(act) if is_added else None
            chain = kept_keys.get(key, []) if key is not None else []
            if _index_of(act, chain) is not None or any(
                _same_callable(act, own) and _index_of(client_copy, chain) is not None
                for own, client_copy in counterparts
            ):
                continue
            kept.append(act)
        return kept


def _same_callable(a: Any, b: Any) -> bool:
    """Whether ``a`` and ``b`` are one callable, without calling ``__eq__``.

    Two bound methods of one function on one object count as one callable;
    any other pair must be the same object. Callables that only compare equal
    are different registrations.
    """
    if a is b:
        return True
    return (
        inspect.ismethod(a)
        and inspect.ismethod(b)
        and a.__self__ is b.__self__
        and a.__func__ is b.__func__
    )


def _index_of(act: Any, acts: Sequence[Any]) -> int | None:
    """The index of the first entry of ``acts`` that is ``act``, or ``None``."""
    return next((i for i, other in enumerate(acts) if _same_callable(act, other)), None)


def _wrapped_chain(act: Any) -> list[Any]:
    """``act`` and the callables it wraps, following ``__wrapped__``."""
    chain: list[Any] = []
    while act is not None and len(chain) < 100:
        chain.append(act)
        act = getattr(act, "__wrapped__", None)
    return chain


def _activity_key(act: Any) -> tuple[str | None] | None:
    """``(name,)`` for the activity ``act`` registers, or ``None`` if it has none.

    A dynamic activity has no name, so its key is ``(None,)``: the one dynamic
    slot a Worker has.
    """
    defn = activity_mod._Definition.from_callable(act)
    return (defn.name,) if defn is not None else None


async def _temporal_aio_to_thread(
    default_aio_to_thread: Callable[..., Any],
    ctx: Any,
    func: Callable[..., Any],
    /,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Run LangSmith's ``aio_to_thread`` seam safely inside a workflow.

    Outside a workflow (activities, clients) we defer to LangSmith's default
    thread hop. Inside a workflow the deterministic event loop cannot spawn a
    thread, so we run the setup inline in the requested context. ``func`` here is
    LangSmith's own run-tree bookkeeping — cheap and side-effect-free with respect
    to workflow determinism.
    """
    from temporalio import workflow

    if not workflow.in_workflow():
        return await default_aio_to_thread(ctx, func, *args, **kwargs)
    with workflow.unsafe.sandbox_unrestricted():
        return ctx.run(func, *args, **kwargs)


def _install_langsmith_temporal_override() -> None:
    """Route LangSmith's thread hop inline while in a workflow.

    ``set_runtime_overrides`` mutates a module global in the sandbox-passthrough
    ``langsmith`` package, so the in-workflow tracer sees it too. No-op (and
    harmless) when LangSmith is not installed or too old to expose
    ``set_runtime_overrides``.

    The override is deliberately installed for the life of the process and
    never uninstalled: LangSmith exposes a single process-wide override slot
    (each ``set_runtime_overrides`` call replaces it wholesale), and
    ``temporalio.langsmith`` installs a behaviorally identical override
    exactly once, without reinstalling. Resetting the slot on this worker's
    shutdown would therefore permanently strip a composed LangSmith plugin's
    workflow-safety override. Leaving it installed is safe: the override defers
    to LangSmith's default thread hop whenever ``workflow.in_workflow()`` is
    false, so it is inert outside workflows.
    """
    try:
        import langsmith

        langsmith.set_runtime_overrides(aio_to_thread=_temporal_aio_to_thread)
    except Exception:
        pass
