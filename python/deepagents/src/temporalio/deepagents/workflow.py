"""Workflow-side surface: the failure type, the dispatch helpers, and the runner.

Everything here runs *inside* the workflow. The dispatch helpers
(:func:`call_model` / :func:`call_tool` / :func:`call_backend_op`) are the single
choke point through which the in-workflow model / tool / backend stubs reach
their activities. Every dispatch runs its own Activity; only executions
recorded before the ``deepagents.retire-result-cache`` patch consult the
legacy continue-as-new result cache during replay.

:func:`run_deep_agent` is the optional driver that adds continue-as-new
state-carry around a native ``agent.ainvoke(...)`` — plain ``agent.ainvoke(...)``
still works without it.
"""

from __future__ import annotations

import dataclasses
import importlib
import warnings
from collections.abc import Callable, Mapping
from typing import Any

from temporalio import workflow
from temporalio.deepagents import _activity, _serde
from temporalio.exceptions import ApplicationError

# Reserved key under which the CAN result cache rides inside a state snapshot.
_CACHE_KEY = "__temporal_cache__"
_INPUT_CARRIED_KEY = "__temporal_input_in_transcript__"

# Patch id for naming overlapping model Activities by the call they answer (see
# _model_activity_id). Runs recorded before it use the SDK's sequence ids.
_CALL_IDENTITY_PATCH = "deepagents.parallel-model-call-ids"


@dataclasses.dataclass
class _ModelDispatches:
    """One run's model dispatch state, for naming overlapping model Activities."""

    in_flight: int = 0
    named: bool | None = None
    """The patch answer, asked once, at the first overlapping dispatch."""
    issued: dict[str, int] = dataclasses.field(default_factory=dict)


# Per-run state lives on the workflow instance itself: it is created and
# dropped with the run (an evicted run replays on a new instance with fresh
# state), and the SDK runs one activation of a run at a time, so it needs no
# lock. A workflow class whose instances take no new attributes (custom
# ``__slots__``) keeps the SDK's sequence ids.
_DISPATCHES_ATTR = "_temporal_deepagents_model_dispatches"


def _run_dispatches() -> _ModelDispatches | None:
    """This run's dispatch state, or None if the instance cannot hold it."""
    instance = workflow.instance()
    state = getattr(instance, _DISPATCHES_ATTR, None)
    if isinstance(state, _ModelDispatches):
        return state
    state = _ModelDispatches()
    try:
        setattr(instance, _DISPATCHES_ATTR, state)
    except (AttributeError, TypeError):
        return None
    return state


# Checkpointer classes that keep their state in the workflow's own memory and are
# therefore rehydrated for free by deterministic replay. Anything else does its
# own I/O and is not replay-safe from inside the workflow.
_IN_WORKFLOW_SAVERS = frozenset({"InMemorySaver", "MemorySaver"})


def warn_durable_checkpointer(checkpointer: Any) -> None:
    """Warn when a user hands ``create_deep_agent`` a durable checkpointer.

    The Deep Agents loop runs inside the workflow, so a checkpointer that does
    its own database / disk I/O would run that I/O from workflow code — not
    replay-safe. We respect the user's choice (a warning, not a hard failure),
    and point them at the durability path that *is* safe: the default in-workflow
    ``InMemorySaver`` rehydrated by replay, plus
    :func:`run_deep_agent` with ``continue_as_new_after`` for long conversations.
    """
    if checkpointer is None:
        return
    if type(checkpointer).__name__ in _IN_WORKFLOW_SAVERS:
        return
    warnings.warn(
        f"create_deep_agent received a durable checkpointer "
        f"{type(checkpointer).__name__!r}. The agent loop runs inside the "
        f"workflow, so this checkpointer's I/O would run from workflow code, "
        f"which is not replay-safe. Prefer the default in-workflow InMemorySaver "
        f"(rehydrated by replay) plus run_deep_agent(continue_as_new_after=...) "
        f"for long-conversation durability.",
        stacklevel=3,
    )


class DeepAgentsWorkflowError(ApplicationError):
    """Raised for non-retryable Deep Agents failures surfaced in the workflow.

    This is the type registered in the plugin's
    ``workflow_failure_exception_types``, so a model / tool failure that Temporal
    has exhausted (or an invalid agent configuration) fails the workflow with a
    stable ``ApplicationError.type`` — never a stringified peer exception.
    """

    TYPE = "deepagents.DeepAgentsWorkflowError"

    def __init__(self, message: str, *, non_retryable: bool = True) -> None:
        """Construct the error with the plugin's stable failure ``type``."""
        super().__init__(message, type=self.TYPE, non_retryable=non_retryable)


# ---------------------------------------------------------------------------
# Dispatch helpers (the model / tool / backend choke point)
# ---------------------------------------------------------------------------


def _legacy_lookup(kind: str, name: str, payload: Any) -> tuple[str | None, bool, Any]:
    """Legacy-cache lookup: ``(key, hit, value)``; key is None on new executions."""
    if not _legacy_result_cache():
        return None, False, None
    key = _serde.cache_key(kind, name, payload)
    hit, value = _serde.cache_lookup(key)
    return key, hit, value


def _legacy_result_cache() -> bool:
    """Whether this execution uses the legacy continue-as-new result cache.

    New executions do not cache at all: repeated identical calls are
    legitimate work (a re-issued tool call, a deliberate model resample, a
    re-read after a write), and under the resume-from-transcript
    continue-as-new semantics a continued run never re-executes prior
    dispatches — so a carried cache entry could only ever serve a stale
    result to a genuinely new call. Replay of a single run needs no cache:
    history supplies recorded activity results.

    Patch-gated because histories recorded under the legacy cache contain
    dedup decisions (a repeated call answered with no activity scheduled);
    replaying them without the cache would emit commands history does not
    have.
    """
    return not workflow.patched("deepagents.retire-result-cache")


def _model_activity_id(
    state: _ModelDispatches | None,
    call_identity: Callable[[], str | None] | None,
    opts: Mapping[str, Any],
) -> str | None:
    """Name a model Activity by the call it answers when it overlaps another one.

    Every model call is the same Activity type, and Temporal does not compare
    Activity inputs on replay, so with the SDK's sequence ids a recorded result
    is delivered by position. Sequential calls cannot change places, but two
    concurrent ones can: when a change in scheduling order (a library upgrade,
    an ``await`` added in middleware) makes two parallel sub-agents schedule in
    the other order, each would silently get the other's result. The SDK does
    compare Activity ids on replay, so an id derived from the call itself
    turns that into a nondeterminism error.

    Only a call dispatched while another model call of the run is in flight is
    named, and the patch is asked at the first such call, so runs that never
    overlap model calls keep sequence ids and record no marker. A run recorded
    before the patch replays it as unset and keeps sequence ids from then on.
    Returns None (keep the SDK default) when the caller set its own
    ``activity_id``.
    """
    if state is None or call_identity is None or "activity_id" in opts:
        return None
    if state.in_flight == 0:
        return None
    if state.named is None:
        state.named = workflow.patched(_CALL_IDENTITY_PATCH)
    if not state.named:
        return None
    identity = call_identity()
    if identity is None:
        return None
    count = state.issued.get(identity, 0)
    state.issued[identity] = count + 1
    return identity if count == 0 else f"{identity}#{count + 1}"


async def call_model(
    activity_name: str,
    activity_input: _activity.ModelActivityInput,
    *,
    summary: str,
    call_identity: Callable[[], str | None] | None = None,
    **opts: Any,
) -> _activity.ModelActivityOutput:
    """Dispatch one model call as its own Activity.

    ``call_identity`` returns a replay-stable name for the call (see
    :meth:`TemporalModel._call_identity`); it is only computed when the call
    overlaps another model call of the run (see :func:`_model_activity_id`).
    """
    legacy_key, hit, cached = _legacy_lookup(
        "model",
        activity_input.model_name,
        [activity_input.messages, activity_input.tool_schemas],
    )
    if hit:
        return _activity.ModelActivityOutput(message=cached)
    state = _run_dispatches()
    activity_id = _model_activity_id(state, call_identity, opts)
    if activity_id is not None:
        opts = {**opts, "activity_id": activity_id}
    if state is not None:
        state.in_flight += 1
    try:
        output = await workflow.execute_activity(
            activity_name,
            activity_input,
            result_type=_activity.ModelActivityOutput,
            summary=summary,
            **opts,
        )
    finally:
        if state is not None:
            state.in_flight -= 1
    if legacy_key is not None:
        _serde.cache_put(legacy_key, output.message)
    return output


async def call_tool(
    activity_input: _activity.ToolActivityInput,
    *,
    summary: str,
    **opts: Any,
) -> _activity.ToolActivityOutput:
    """Dispatch one tool call as its own Activity."""
    legacy_key, hit, cached = _legacy_lookup(
        "tool", activity_input.tool_name, activity_input.args
    )
    if hit:
        return _activity.ToolActivityOutput(message=cached)
    activity_input = dataclasses.replace(
        activity_input,
        arg_messages=_serde.find_tool_arg_messages(activity_input.args),
    )
    output = await workflow.execute_activity(
        _activity.INVOKE_TOOL,
        activity_input,
        result_type=_activity.ToolActivityOutput,
        summary=summary,
        **opts,
    )
    if legacy_key is not None:
        _serde.cache_put(legacy_key, output.message)
    return output


async def call_backend_op(
    activity_input: _activity.BackendOpInput,
    *,
    summary: str,
    **opts: Any,
) -> _activity.BackendOpOutput:
    """Dispatch one backend op as its own Activity."""
    legacy_key, hit, cached = _legacy_lookup(
        f"backend:{activity_input.backend_ref}",
        activity_input.op,
        [activity_input.args, activity_input.kwargs],
    )
    if hit:
        return _activity.BackendOpOutput(result=cached)
    output = await workflow.execute_activity(
        _activity.BACKEND_OP,
        activity_input,
        result_type=_activity.BackendOpOutput,
        summary=summary,
        **opts,
    )
    if legacy_key is not None:
        _serde.cache_put(legacy_key, output.result)
    return output


# ---------------------------------------------------------------------------
# run_deep_agent (continue-as-new state carry)
# ---------------------------------------------------------------------------


def _merge_snapshot(input: Any, snapshot: Mapping[str, Any]) -> Any:
    """Prepend a snapshot's carried messages onto the next turn's input.

    The driver's own continue-as-new re-invocation avoids duplicating the
    original prompt: a Mapping input travels without its "messages" key, and
    a bare (non-Mapping) prompt travels as-is with a snapshot marker telling
    this merge not to re-append it (the type must survive for the user's
    ``@workflow.run`` signature). An externally supplied ``state_snapshot``
    plus a fresh input still composes: carried history first, new input
    after. Agents are expected to return the accumulated transcript in
    ``result["messages"]`` (as deepagents/LangGraph reducers do) — the carry
    only strips input messages when the transcript is non-empty.
    """
    raw_prior: Any = snapshot.get("messages") or []
    prior = list(raw_prior)
    if not prior:
        return input
    if isinstance(input, Mapping):
        merged = dict(input)
        raw_next: Any = input.get("messages") or []
        merged["messages"] = [*prior, *list(raw_next)]
        return merged
    if _INPUT_CARRIED_KEY in snapshot and snapshot[_INPUT_CARRIED_KEY] == input:
        # Internal continue-as-new of a bare prompt: this exact input is
        # already in the transcript; it rode along only to preserve its type.
        # A DIFFERENT bare input (an externally harvested snapshot plus a
        # fresh prompt) falls through and composes as usual.
        return {"messages": prior}
    return {"messages": [*prior, *_as_message_list(input)]}


def _as_message_list(input: Any) -> list[Any]:
    if isinstance(input, (list, tuple)):
        return list(input)
    return [input]


async def run_deep_agent(
    agent: Any,
    input: Any,
    *,
    continue_as_new_after: int | None = None,
    state_snapshot: Mapping[str, Any] | None = None,
) -> Any:
    """Drive ``agent.ainvoke(input)`` with continue-as-new state carry.

    Once a completed turn leaves pending todos and history has grown past the
    limit, its messages are carried into a fresh run via
    ``workflow.continue_as_new``. Only legacy executions carry the result cache.

    By default (``continue_as_new_after=None``) the limit is the server's own
    recommendation — ``workflow.info().is_continue_as_new_suggested()`` — which
    accounts for both history length and size; this is the recommended mode.
    Pass an explicit ``continue_as_new_after=N`` to trigger on a fixed history
    event count instead. To run an agent with NO continue-as-new behavior, call
    ``agent.ainvoke(...)`` directly rather than using this driver.

    The enclosing ``@workflow.run`` method must accept the continued call — i.e.
    its signature is ``(input, state_snapshot=None)`` — because that is how the
    carried state is threaded into the next run.
    """
    # The carry across continue-as-new derives from the ORIGINAL input:
    # re-threading the merged input would hand a dict to str-typed run
    # signatures and re-prepend carried messages on every later boundary.
    original_input = input
    # Resume path: rehydrate the result cache and fold carried messages in.
    if state_snapshot is not None:
        # Only legacy executions consult the carried cache; on new
        # executions the inbound legacy entries are dead weight, and at the
        # upgrade hop a repeated identical call re-executes (conservative
        # direction) rather than being served a possibly-stale carried result.
        if _legacy_result_cache():
            _serde.set_result_cache(dict(state_snapshot.get(_CACHE_KEY) or {}))
        else:
            _serde.set_result_cache({})
        input = _merge_snapshot(input, state_snapshot)
    else:
        _serde.set_result_cache({})

    try:
        result = await agent.ainvoke(input)
    except ApplicationError:
        raise
    except Exception as exc:
        # If the framework wrapped a Temporal failure, surface the registered
        # workflow-failure type rather than the framework's generic exception.
        cause = exc.__cause__
        if cause is not None and workflow.is_failure_exception(cause):
            raise DeepAgentsWorkflowError(f"Deep Agents run failed: {exc}") from cause
        raise

    if continue_as_new_after is None:
        # Default: follow the server's judgement. The suggestion accounts for
        # history count AND size limits, which a fixed event threshold cannot.
        should_continue = workflow.info().is_continue_as_new_suggested()
    else:
        should_continue = (
            workflow.info().get_current_history_length() >= continue_as_new_after
        )
    if should_continue and _has_pending_work(result):
        carried = _extract_messages(result)
        if not carried:
            # A turn may report pending todos with an empty/pruned transcript;
            # the conversation the agent SAW must still cross the boundary.
            if isinstance(input, Mapping):
                carried = _extract_messages(input)
            else:
                carried = _as_message_list(input)
        snapshot: dict[str, Any] = {"messages": carried}
        if _legacy_result_cache():
            snapshot[_CACHE_KEY] = _serde.result_cache_snapshot() or {}
        # ``continue_as_new`` threads positional args into the next run via
        # ``args=``; the enclosing ``@workflow.run`` receives them as
        # ``(input, state_snapshot)``, so the carried input must keep the
        # user's declared input TYPE (a dict cannot decode into a run method
        # typed for a bare-string prompt). When the transcript already
        # carries the conversation (including the original input messages),
        # a Mapping input travels without its "messages" key, and a
        # non-Mapping input travels as-is with a snapshot marker telling
        # _merge_snapshot not to re-append it. An EMPTY transcript re-sends
        # the input unchanged so the original prompt is never lost.
        carry_input: Any = original_input
        if carried:
            if isinstance(original_input, Mapping):
                carry_input = {
                    k: v for k, v in original_input.items() if k != "messages"
                }
            else:
                snapshot[_INPUT_CARRIED_KEY] = original_input
        workflow.continue_as_new(args=[carry_input, snapshot])

    return result


def _extract_messages(result: Any) -> list[Any]:
    if isinstance(result, Mapping):
        raw: Any = result.get("messages") or []
        return list(raw)
    return []


def _has_pending_work(result: Any) -> bool:
    """True when the agent left unfinished todos worth carrying past a CAN.

    A finished single-shot run has no pending todos, so this returns False and the
    driver returns the result instead of looping on continue-as-new forever.
    """
    if isinstance(result, Mapping):
        todos: Any = result.get("todos") or []
        return any(
            isinstance(t, Mapping) and t.get("status") not in ("completed", "done")
            for t in todos
        )
    return False


def create_temporal_deep_agent(
    *args: Any,
    activity_options: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> Any:
    """Build a Deep Agent whose model calls run as durable activities.

    A thin wrapper over ``deepagents.create_deep_agent`` that makes the
    Temporal wiring explicit: a ``model=`` name string is wrapped in
    ``TemporalModel`` carrying this
    agent's ``activity_options`` (``execute_activity`` overrides — timeouts,
    retry policy — for its model calls). Every other argument — tools,
    backend, sub-agents, ``interrupt_on`` — is forwarded unchanged.

    Unmodified ``create_deep_agent(...)`` also works inside a workflow (the
    plugin substitutes the durable model automatically, using the plugin's
    ``model_activity_options``); use this wrapper to scope activity options
    to one agent instead of configuring them plugin-wide.
    """
    with workflow.unsafe.imports_passed_through():
        # importlib keeps this resolution absolute: a static
        # `import deepagents` from inside this same-named package directory
        # is flagged (and on 3.10, mis-resolved) as implicitly relative.
        deepagents_mod: Any = importlib.import_module("deepagents")

    from temporalio.deepagents._model import TemporalModel

    model = args[0] if args else kwargs.pop("model", None)
    if isinstance(model, str):
        model = TemporalModel(
            model=model,
            activity_options=(
                dict(activity_options) if activity_options is not None else None
            ),
        )
    elif activity_options is not None:
        if isinstance(model, TemporalModel):
            model = TemporalModel(
                model=model.model, activity_options=dict(activity_options)
            )
        else:
            raise ValueError(
                "activity_options requires model= to be a model-name string "
                "or a TemporalModel; got "
                f"{type(model).__name__ if model is not None else 'no model'}."
            )
    if args:
        args = (model, *args[1:])
    else:
        kwargs["model"] = model
    return deepagents_mod.create_deep_agent(*args, **kwargs)
