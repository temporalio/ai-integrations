"""The model seam: a chat model whose every call becomes a Temporal activity.

:class:`TemporalModel` is a real ``BaseChatModel``. Placed anywhere Deep Agents
expects a model, it routes each generation through the ``deepagents.invoke_model``
activity instead of calling the provider inline. Because Deep Agents' sub-agents
inherit the parent's ``model`` instance by default, substituting this one object
makes every model call in the whole agent tree durable — no middleware injection
that a sub-agent could silently miss.

Two ways to get one:

* explicit — the user writes ``TemporalModel("anthropic:claude-...")`` and hands
  it to ``create_deep_agent(model=...)``; this is the public type users assert
  against;
* implicit — while running inside a workflow, the plugin patches
  ``create_deep_agent`` so a bare ``model="anthropic:..."`` string is wrapped
  automatically (see :func:`install_model_patch`).

Worker-wide dispatch defaults (activity options, streaming topic) live in
:class:`temporalio.deepagents._serde.Settings`, a module global set by
the plugin. They live in ``_serde`` (not here) so the plugin can configure them
without importing this module — which would drag LangChain into plugin
construction. This module is under ``temporalio``, which the workflow sandbox
passes through, so the object the workflow reads is the same one the plugin set.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from datetime import timedelta
from typing import Any

from temporalio import workflow
from temporalio.deepagents import _activity, _serde
from temporalio.exceptions import ActivityError, ApplicationError

with workflow.unsafe.imports_passed_through():
    from langchain_core.callbacks import (
        AsyncCallbackManagerForLLMRun,
        CallbackManagerForLLMRun,
    )
    from langchain_core.exceptions import ContextOverflowError
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import AIMessageChunk, BaseMessage
    from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult


def _as_message_chunk(message: Any) -> Any:
    """Re-shape a finished ``AIMessage`` as the ``AIMessageChunk`` a stream yields."""
    return AIMessageChunk(
        content=getattr(message, "content", ""),
        additional_kwargs=getattr(message, "additional_kwargs", {}) or {},
        response_metadata=getattr(message, "response_metadata", {}) or {},
        id=getattr(message, "id", None),
        tool_calls=getattr(message, "tool_calls", []) or [],
        usage_metadata=getattr(message, "usage_metadata", None),
    )


# Patch id for rebuilding ContextOverflowError from a failed model Activity.
# Never change it: it is recorded in histories as a marker.
_CONTEXT_OVERFLOW_PATCH = "deepagents.context-overflow-recovery"


async def _call_model(
    activity_name: str,
    activity_input: _activity.ModelActivityInput,
    *,
    summary: str,
    **opts: Any,
) -> _activity.ModelActivityOutput:
    """Run the model Activity, raising a provider overflow as ``ContextOverflowError``.

    Deep Agents' summarization middleware summarizes and retries only when the
    model call raises an overflow it recognizes; an ``ActivityError`` is not one.
    Histories recorded before this change let the ``ActivityError`` propagate,
    and the Workflow may have handled it, so they keep that behavior.
    """
    from temporalio.deepagents.workflow import call_model

    try:
        return await call_model(activity_name, activity_input, summary=summary, **opts)
    except ActivityError as error:
        cause = error.cause
        if (
            isinstance(cause, ApplicationError)
            and cause.type == _activity.CONTEXT_OVERFLOW_ERROR_TYPE
            and workflow.patched(_CONTEXT_OVERFLOW_PATCH)
        ):
            raise ContextOverflowError(cause.message) from error
        raise


def _without_id(dumped: Any) -> Any:
    """Drop the message id from a ``dumpd`` message, which is not replay-stable."""
    if isinstance(dumped, dict) and isinstance(dumped.get("kwargs"), dict):
        kwargs = {k: v for k, v in dumped["kwargs"].items() if k != "id"}
        return {**dumped, "kwargs": kwargs}
    return dumped


def _has_repr(dumped: Any) -> bool:
    """Whether ``dumpd`` fell back to a ``repr`` somewhere in this value.

    ``dumpd`` writes a value it cannot serialize as ``not_implemented`` with
    the object's ``repr``, which often holds a memory address and so differs
    between the original run and its replay.
    """
    if isinstance(dumped, dict):
        if dumped.get("type") == "not_implemented":
            return True
        return any(_has_repr(v) for v in dumped.values())
    if isinstance(dumped, list):
        return any(_has_repr(v) for v in dumped)
    return False


# ---------------------------------------------------------------------------
# Activity-option resolution (worker-wide defaults live in _serde.Settings)
# ---------------------------------------------------------------------------


_DEFAULT_MODEL_TIMEOUT = timedelta(minutes=5)


def _resolve_activity_options(
    model_name: str, instance_options: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Merge worker defaults with a per-instance override into execute_activity kwargs."""
    opts: dict[str, Any] = {}
    default = _serde.get_settings().model_activity_options
    if isinstance(default, Mapping) and not _looks_like_activity_config(default):
        # Mapping[model_name, ActivityConfig]: pick this model's entry.
        chosen = default.get(model_name)
        if isinstance(chosen, Mapping):
            opts.update(chosen)
    elif isinstance(default, Mapping):
        opts.update(default)
    if instance_options:
        opts.update(instance_options)
    opts.setdefault("start_to_close_timeout", _DEFAULT_MODEL_TIMEOUT)
    return opts


def _looks_like_activity_config(m: Mapping[str, Any]) -> bool:
    """A single ``ActivityConfig`` has known option keys; a per-model map does not."""
    known = {
        "task_queue",
        "schedule_to_close_timeout",
        "schedule_to_start_timeout",
        "start_to_close_timeout",
        "heartbeat_timeout",
        "retry_policy",
        "cancellation_type",
        "activity_id",
        "versioning_intent",
        "summary",
        "priority",
    }
    return bool(m) and all(k in known for k in m)


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------


class TemporalModel(BaseChatModel):
    """A ``BaseChatModel`` that runs each generation as a Temporal activity.

    Args:
        model: The provider model name resolved worker-side by the plugin's
            ``model_provider`` (e.g. ``"anthropic:claude-sonnet-4-5"``). Only the
            name crosses the workflow boundary; credentials stay on the worker.
        activity_options: Optional per-model ``execute_activity`` overrides
            (timeouts, retry policy). Falls back to the plugin's
            ``model_activity_options``.
    """

    model: str
    activity_options: dict[str, Any] | None = None

    # ``protected_namespaces=()`` silences pydantic's warning about the ``model``
    # field colliding with its ``model_`` namespace.
    model_config = {"arbitrary_types_allowed": True, "protected_namespaces": ()}

    @property
    def _llm_type(self) -> str:
        return "temporal-deepagents"

    def bind_tools(
        self,
        tools: Sequence[Any],
        *,
        tool_choice: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        """Bind tools to the model the way LangChain's ``create_agent`` expects.

        ``BaseChatModel.bind_tools`` is abstract (raises ``NotImplementedError``),
        but the agent factory calls ``model.bind_tools(tools, ...)`` on every model
        node — so a durable model must implement it or the whole loop dies. The
        tools are converted to their JSON schema *now* (at bind time) so the bound
        object is serialization-safe, and carried as the ``tools`` kwarg that
        :meth:`_build_input` already reads and forwards to the activity, where the
        real provider model is what actually binds them.
        """
        schemas = [
            t if isinstance(t, dict) else _serde.tool_to_schema(t) for t in tools
        ]
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        return self.bind(tools=schemas, **kwargs)

    def _summary(self) -> str:
        return f"invoke_model[{self.model}]"

    @staticmethod
    def _call_identity(messages: Sequence[BaseMessage]) -> str | None:
        """A replay-stable name for this call: which agent loop, and its transcript.

        Built only from values that replay identically: the agent's name from
        the run config (``lc_agent_name``, which ``create_agent`` sets for each
        sub-agent) and the non-system messages of the request in ``dumpd`` form
        minus their ids, which LangGraph assigns with ``uuid4`` outside the
        workflow's seeded random. System prompts, tool schemas and the model
        name are left out, so editing them does not fail the replay of open
        runs. Returns None when the identity cannot be built (a value JSON
        cannot hold, one ``dumpd`` could only write as a ``repr``, or any other
        error); the call then keeps its sequence id. Only computed for a call that
        overlaps another one (see ``_model_activity_id`` in ``workflow``).
        """
        with workflow.unsafe.imports_passed_through():
            from langchain_core.runnables.config import ensure_config

        # The identity only adds a check, and the sequence id is always a safe
        # answer, so any failure to build it keeps the sequence id.
        try:
            metadata = ensure_config().get("metadata") or {}
            agent = str(metadata.get("lc_agent_name") or "")
            history = [m for m in messages if m.type != "system"]
            dumped = [_without_id(_serde.dump_object(m)) for m in history]
            if _has_repr(dumped):
                return None
            canonical = json.dumps(
                [agent, dumped], sort_keys=True, separators=(",", ":")
            )
        except Exception:
            return None
        digest = hashlib.sha256(canonical.encode()).hexdigest()[:16]
        turn = sum(1 for m in history if m.type == "ai")
        return f"deepagents.model:{digest}:{turn}"

    def _build_input(
        self,
        messages: Sequence[BaseMessage],
        streaming_topic: str | None,
        **kwargs: Any,
    ) -> _activity.ModelActivityInput:
        tools = kwargs.get("tools") or []
        tool_schemas = [
            t if isinstance(t, dict) else _serde.tool_to_schema(t) for t in tools
        ]
        bind_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k not in ("tools", "config", "run_manager", "stop", "callbacks")
            and _serde._is_jsonish(v)
        }
        if kwargs.get("stop"):
            bind_kwargs["stop"] = kwargs["stop"]
        return _activity.ModelActivityInput(
            model_name=self.model,
            messages=_serde.dump_messages(messages),
            tool_schemas=tool_schemas,
            bind_kwargs=bind_kwargs,
            config=_serde.strip_runnable_config(kwargs.get("config") or {}),
            streaming_topic=streaming_topic,
        )

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise NotImplementedError(
            "TemporalModel is async-only inside a Temporal workflow. Drive the "
            "agent with `await agent.ainvoke(...)` / `.astream(...)`, which uses "
            "the async model path."
        )

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        activity_input = self._build_input(messages, None, stop=stop, **kwargs)
        opts = _resolve_activity_options(self.model, self.activity_options)
        output = await _call_model(
            _activity.INVOKE_MODEL,
            activity_input,
            summary=self._summary(),
            call_identity=lambda: self._call_identity(messages),
            **opts,
        )
        message = _serde.load_object(output.message)
        return ChatResult(generations=[ChatGeneration(message=message)])

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        topic = _serde.get_settings().streaming_topic
        opts = _resolve_activity_options(self.model, self.activity_options)
        if topic:
            activity_input = self._build_input(messages, topic, stop=stop, **kwargs)
            output = await _call_model(
                _activity.INVOKE_MODEL_STREAMING,
                activity_input,
                summary=f"invoke_model_streaming[{self.model}]",
                call_identity=lambda: self._call_identity(messages),
                **opts,
            )
        else:
            # No topic configured: fall back to a single response-level chunk.
            activity_input = self._build_input(messages, None, stop=stop, **kwargs)
            output = await _call_model(
                _activity.INVOKE_MODEL,
                activity_input,
                summary=self._summary(),
                call_identity=lambda: self._call_identity(messages),
                **opts,
            )
        message = _serde.load_object(output.message)
        yield ChatGenerationChunk(message=_as_message_chunk(message))

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        raise NotImplementedError(
            "TemporalModel streaming is async-only; use `agent.astream(...)`."
        )


# ---------------------------------------------------------------------------
# create_deep_agent model patch (implicit wrapping)
# ---------------------------------------------------------------------------
#
# The durability seam is ``resolve_model``, which turns a ``model=`` name
# string into a live ``BaseChatModel``. We patch every binding a string can
# reach in-workflow:
#
# - ``deepagents.graph`` — read afresh by ``create_deep_agent`` for the main
#   agent and every sub-agent (its module-top import froze its own binding).
# - ``deepagents._models`` — the definition site; covers call-time importers
#   such as ``create_summarization_tool_middleware``.
# - ``SummarizationMiddleware.__init__`` — pre-resolves a string summarizer
#   before the middleware delegates to LangChain's ``init_chat_model``.
#
# An unpatched binding means a real provider client constructed (and called)
# inside the workflow: nondeterministic and replay-unsafe. Every patched seam
# is deepagents-internal and covered by this package's deepagents version
# pin, so none is guarded: a missing seam is a broken install and fails the
# worker at startup rather than silently reverting. We do NOT rebind
# ``deepagents.create_deep_agent`` for durability — callers who already did
# ``from deepagents import create_deep_agent`` hold the original object — only
# a best-effort wrap to fire the advisory construction-time warnings.

_original_create_deep_agent: Any = None
_original_resolve_model: Any = None
_original_summarization_init: Any = None


def _wrap_model_arg(model: Any) -> Any:
    """Resolve a ``create_deep_agent`` model argument to a durable model.

    A name string becomes a :class:`TemporalModel`. A :class:`TemporalModel`
    passes through. Any other live ``BaseChatModel`` instance is rejected at the
    workflow boundary: it has no name the activity could rebuild from, so it would
    run its provider call *inside the workflow* — nondeterministic and unsafe.
    """
    from temporalio.deepagents.workflow import DeepAgentsWorkflowError

    if isinstance(model, str):
        return TemporalModel(model=model)
    if isinstance(model, TemporalModel):
        return model
    if isinstance(model, BaseChatModel):
        raise DeepAgentsWorkflowError(
            f"create_deep_agent received a live {type(model).__name__} model "
            f"instance, which would run inside the workflow (nondeterministic). "
            f'Pass model="provider:name" (auto-routed through an activity) or '
            f"wrap it as TemporalModel(...) instead."
        )
    return model


def install_model_patch() -> None:
    """Route Deep Agents' model resolution through :class:`TemporalModel`.

    Patches the model-resolution bindings listed above so a name string
    becomes a durable :class:`TemporalModel` in-workflow, and wraps
    ``deepagents.create_deep_agent`` for the advisory warnings. No effect
    outside workflows. Idempotent.
    """
    global _original_create_deep_agent, _original_resolve_model
    # importlib: `deepagents` is absent on Python 3.10 environments (its floor
    # is 3.11), so static imports here fail type-checking there.
    deepagents = importlib.import_module("deepagents")
    _graph = importlib.import_module("deepagents.graph")
    _models = importlib.import_module("deepagents._models")

    if _original_resolve_model is None:
        # One original serves both bindings (same function object).
        _original_resolve_model = _models.resolve_model

        def patched_resolve_model(model: Any) -> Any:
            if workflow.in_workflow():
                return _wrap_model_arg(model)
            return _original_resolve_model(model)

        setattr(_graph, "resolve_model", patched_resolve_model)
        setattr(_models, "resolve_model", patched_resolve_model)

    global _original_summarization_init
    if _original_summarization_init is None:
        _da_sum = importlib.import_module("deepagents.middleware.summarization")
        summarization_cls = _da_sum.SummarizationMiddleware
        original_init = summarization_cls.__init__
        _original_summarization_init = original_init

        def patched_summarization_init(
            self: Any, model: Any, *args: Any, **kwargs: Any
        ) -> None:
            if workflow.in_workflow() and isinstance(model, str):
                model = _wrap_model_arg(model)
            original_init(self, model, *args, **kwargs)

        summarization_cls.__init__ = patched_summarization_init

    if _original_create_deep_agent is None:
        _original_create_deep_agent = deepagents.create_deep_agent

        def patched(*args: Any, **kwargs: Any) -> Any:
            if workflow.in_workflow():
                from temporalio.deepagents._tools import warn_unwrapped_tools
                from temporalio.deepagents.workflow import (
                    warn_durable_checkpointer,
                )

                warn_unwrapped_tools(kwargs.get("tools"))
                warn_durable_checkpointer(kwargs.get("checkpointer"))
            return _original_create_deep_agent(*args, **kwargs)

        setattr(deepagents, "create_deep_agent", patched)


def uninstall_model_patch() -> None:
    """Restore the original ``resolve_model`` / ``create_deep_agent``."""
    global _original_create_deep_agent, _original_resolve_model
    if _original_resolve_model is not None:
        _graph = importlib.import_module("deepagents.graph")
        _models = importlib.import_module("deepagents._models")

        setattr(_graph, "resolve_model", _original_resolve_model)
        setattr(_models, "resolve_model", _original_resolve_model)
        _original_resolve_model = None
    global _original_summarization_init
    if _original_summarization_init is not None:
        _da_sum = importlib.import_module("deepagents.middleware.summarization")

        _da_sum.SummarizationMiddleware.__init__ = _original_summarization_init
        _original_summarization_init = None
    if _original_create_deep_agent is not None:
        deepagents = importlib.import_module("deepagents")

        setattr(deepagents, "create_deep_agent", _original_create_deep_agent)
        _original_create_deep_agent = None
