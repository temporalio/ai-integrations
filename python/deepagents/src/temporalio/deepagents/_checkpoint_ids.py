"""Replay-stable LangGraph checkpoint ids inside Temporal workflows.

LangGraph mints every checkpoint id with ``uuid6`` — wall-clock time plus
``random`` (``langgraph/checkpoint/base/id.py``) — and derives everything
address-shaped from it: task ids hash the checkpoint id, a task's
``checkpoint_ns`` embeds its task id, and ``Interrupt.id`` (the key a
human-in-the-loop approval is surfaced under) is a hash of that namespace.
The Deep Agents loop runs inside the workflow, so those ids are minted by
workflow code and re-derived on every replay, worker restart, and workflow
eviction: the *same* history then yields a *different* interrupt id, and an
id that left the workflow (a Query result shown to a reviewer, an Update
payload, a log line) no longer matches what the re-executing workflow
computes.

Replay can supply neither wall-clock time nor entropy, so while a worker is
running, :func:`install_checkpoint_id_patch` swaps the ``uuid6`` bindings
LangGraph's checkpoint modules hold for :func:`_workflow_uuid6`, a drop-in
built only from replay-safe inputs. Only workflow runs that carry the
plugin's :class:`CheckpointIdInterceptor` get the drop-in: outside workflows,
in workflows on another Worker of the same process that runs LangGraph
without the plugin, and in Query handlers and Update validators (which
Temporal does not re-run on replay, so they have nothing to reproduce), the
original ``uuid6`` still answers, so that code is untouched and leaves no
per-run state behind.

The time field carries the workflow's own time, with a per-run floor.
``workflow.time_ns()`` is the current workflow task's *recorded* timestamp:
during replay (and during the re-execution of an evicted run, which replays
each original task with its original timestamp before reaching new events)
it returns exactly the value the original execution saw, so an id minted at
a given point in the code is the same in every execution of that history.
Multiple mints can fall between two events — same timestamp — so each id is
floored past the run's previous one: distinct and string-increasing by
creation, like uuid6 itself. Ids also sort in real uuid6 time range, next to
any unpatched ids they might be compared with.
"""

from __future__ import annotations

import contextvars
import hashlib
import importlib.util
import threading
from typing import Any

from temporalio import workflow
from temporalio.worker import (
    ExecuteWorkflowInput,
    HandleQueryInput,
    HandleSignalInput,
    HandleUpdateInput,
    Interceptor,
    WorkflowInboundInterceptor,
    WorkflowInterceptorClassInput,
)

_original_uuid6: Any = None
_langgraph_uuid: Any = None
_patched_modules: list[Any] = []
# Active install() nestings: two workers with this plugin can be alive in one
# process, and the first to shut down must not unpatch the survivor.
_patch_depth = 0
# Workers can start and stop on separate event-loop threads. The check of
# ``_original_uuid6``, the LangGraph imports, the rebinding and the depth
# count are one critical section for install and uninstall alike, so two
# installers never both see "unpatched" and the second never saves the
# first one's wrapper as the original.
_patch_lock = threading.Lock()

# Modules whose own ``uuid6`` binding is swapped. Every ``from ... import
# uuid6`` freezes a binding in the importing module, so each holder — the
# defining module included, for importers this version has not seen — is
# rebound individually.
_PATCHED_MODULE_NAMES = (
    "langgraph.checkpoint.base.id",
    "langgraph.checkpoint.base",
    "langgraph.pregel._checkpoint",
)

# 100-ns ticks between uuid6's epoch (1582-10-15) and the Unix epoch — the
# constant langgraph's own uuid6 adds to Unix 100-ns ticks.
_UUID6_EPOCH_OFFSET_TICKS = 122192928000000000


class _InstanceState:
    """Highest tick minted so far by one workflow *instance*.

    Owned by that instance's :class:`CheckpointIdInterceptor` inbound
    interceptor, so it lives exactly as long as the instance and is freed
    with it whether or not the run method was ever reached (a start payload
    that fails to decode never gets there). Per instance, not per run,
    because one process can hold two live instances of one run (a
    workflow-task timeout moves a run to another Worker while the first
    still caches its instance, or a Replayer re-executes it), and one
    instance must not reset or advance the other's floor.
    """

    __slots__ = ("last_tick",)

    def __init__(self) -> None:
        self.last_tick = 0


# The state the code running now mints against. The inbound interceptor sets
# it at the top of each recorded handler — the run method, Signal handlers,
# Update handlers — inside the task the SDK runs that handler in, and every
# task the handler starts (LangGraph's included) inherits it, so the state
# is found without asking ``workflow.info()`` for an identity that an
# outbound interceptor may replace. ``None`` (the default, and what Query
# handlers and Update validators run with) means "not a recorded execution
# path of a plugin-enabled instance": LangGraph's own ``uuid6`` answers.
_current_state: contextvars.ContextVar[_InstanceState | None] = contextvars.ContextVar(
    "temporalio_deepagents_checkpoint_ids", default=None
)


def _uuid6_int(tick: int, node: int, clock_seq: int) -> int:
    """Pack a 60-bit tick, 48-bit node and 14-bit clock_seq into uuid6's layout.

    Field packing copied from ``langgraph.checkpoint.base.id.uuid6`` so the
    string form is indistinguishable from an unpatched id: the tick is split
    around the version nibble exactly as uuid6 splits a timestamp, which
    keeps string order identical to tick order.
    """
    time_high_and_time_mid = (tick >> 12) & 0xFFFFFFFFFFFF
    time_low_and_version = tick & 0x0FFF
    value = time_high_and_time_mid << 80
    value |= time_low_and_version << 64
    value |= (clock_seq & 0x3FFF) << 48
    value |= node & 0xFFFFFFFFFFFF
    return value


def _run_id_bits(run_id: str) -> tuple[int, int]:
    """Derive (node, clock_seq) bits from a run id treated as opaque text.

    Run ids are not assumed to be UUIDs or any other format; hashing makes
    the mapping total and uniform.
    """
    digest = hashlib.sha256(run_id.encode()).digest()
    return (
        int.from_bytes(digest[:6], "big"),
        int.from_bytes(digest[6:8], "big") & 0x3FFF,
    )


def _workflow_uuid6(
    state: _InstanceState, node: int | None = None, clock_seq: int | None = None
) -> Any:
    """Drop-in for LangGraph's ``uuid6`` whose value depends only on the code path.

    The id keeps uuid6's exact bit layout and version, so checkpoint ids keep
    the properties LangGraph relies on: unique within a run, and
    string-ordered by creation (``InMemorySaver`` resolves a thread's latest
    checkpoint with ``max()``). Only the inputs change:

    - the 60-bit time field carries ``workflow.time_ns()`` (uuid6's 100ns
      ticks) instead of wall-clock time, floored past the run's previous
      tick so bursts between two events stay distinct. That timestamp comes
      from the recorded history, so re-execution — replay, worker restart,
      eviction — re-derives the same value at the same point in the code
      (see the module docstring);
    - the 48-bit node field is fixed per run, derived from the run id
      recorded on the run's ``WorkflowExecutionStarted`` event
      (``Info.original_execution_run_id``), so different executions of the
      same workflow — a retry, a cron run, a continue-as-new run each record
      a new started event — mint different ids without drawing from the
      workflow RNG (which would couple checkpoint ids to unrelated user
      code's randomness). A reset copies that event into the new run, so
      the copied history re-derives the ids the original run minted (and
      any approval already recorded against them still matches) instead of
      hashing the new run id. LangGraph normally passes ``clock_seq=step``,
      so two runs alive at the same moment differ in those 48 hash bits —
      distinct with overwhelming probability, not by construction.

    Args:
        state: The minting instance's tick floor.
        node: Passed through by LangGraph as ``None`` (uuid6's random-node
            default); hashed from the run id when absent.
        clock_seq: LangGraph always passes the superstep here; hashed from
            the run id only to mirror the original signature.
    """
    # Preserved across reset; equal to the run id otherwise.
    run_id = workflow.info().original_execution_run_id
    # uuid6 ticks of the workflow's current time. ``workflow.time_ns()`` is
    # the current activation's recorded timestamp — original on first pass,
    # the same recorded value on every re-execution — so ids land in the
    # same time range as unpatched uuid6 ids and order sensibly next to
    # them. The floor advances past the run's previous tick, so mints that
    # fall between two events (same timestamp) stay distinct and increasing.
    tick = max(
        workflow.time_ns() // 100 + _UUID6_EPOCH_OFFSET_TICKS,
        state.last_tick + 1,
    )
    state.last_tick = tick
    run_node, run_clock_seq = _run_id_bits(run_id)
    value = _uuid6_int(
        tick,
        node if node is not None else run_node,
        clock_seq if clock_seq is not None else run_clock_seq,
    )
    return _langgraph_uuid(int=value, version=6)


def install_checkpoint_id_patch() -> None:
    """Point LangGraph's checkpoint-id mint at the replay-safe drop-in.

    Every module in ``_PATCHED_MODULE_NAMES`` is imported and verified to
    still bind the original function *before* anything is rebound, and a
    failure leaves no half-applied patch. With LangGraph installed, a moved
    module or renamed binding raises ``AttributeError`` at worker startup
    instead of degrading into silently nondeterministic ids; plain
    ``ImportError`` is raised only when LangGraph itself is absent, which the
    plugin treats as "nothing to patch". Idempotent, and safe under
    overlapping install/uninstall pairs; pair with
    :func:`uninstall_checkpoint_id_patch`.
    """
    with _patch_lock:
        _install_locked()


def _install_locked() -> None:
    global _original_uuid6, _langgraph_uuid, _patched_modules, _patch_depth
    if _original_uuid6 is not None:
        _patch_depth += 1
        return
    if importlib.util.find_spec("langgraph") is None:
        raise ImportError("langgraph is not installed")
    try:
        id_module = importlib.import_module("langgraph.checkpoint.base.id")
        original = id_module.uuid6
        modules = [id_module]
        for name in _PATCHED_MODULE_NAMES[1:]:
            module = importlib.import_module(name)
            if getattr(module, "uuid6", None) is not original:
                raise AttributeError(
                    f"{name} does not bind langgraph.checkpoint.base.id.uuid6"
                )
            modules.append(module)
    except AttributeError:
        raise
    except ImportError as exc:
        # LangGraph is installed (checked above), so this import can only
        # fail because the layout changed — surface it, don't swallow it.
        raise AttributeError(
            f"the LangGraph checkpoint-id modules moved ({exc})"
        ) from exc
    _langgraph_uuid = id_module.UUID
    _original_uuid6 = original
    _patch_depth = 1

    def patched_uuid6(node: int | None = None, clock_seq: int | None = None) -> Any:
        # Inert outside workflows: the same worker process may run plain
        # client- or activity-side LangGraph code, which keeps real uuid6.
        # Inert too in a workflow run without CheckpointIdInterceptor (another
        # Worker in this process running LangGraph without the plugin), and
        # in Query handlers and Update validators: no state is set there.
        # Delegates to the local ``original``, not the module global, so a
        # reference imported while patched (``from ... import uuid6``) keeps
        # working after the last worker uninstalls the patch.
        state = _current_state.get()
        if state is not None and workflow.in_workflow():
            return _workflow_uuid6(state, node, clock_seq)
        return original(node, clock_seq)

    for module in modules:
        setattr(module, "uuid6", patched_uuid6)
    _patched_modules = modules


def uninstall_checkpoint_id_patch() -> None:
    """Restore the original ``uuid6`` bindings.

    Restore only when the last overlapping install has uninstalled.
    """
    with _patch_lock:
        _uninstall_locked()


def _uninstall_locked() -> None:
    global _original_uuid6, _langgraph_uuid, _patched_modules, _patch_depth
    if _original_uuid6 is None:
        return
    _patch_depth -= 1
    if _patch_depth > 0:
        return
    for module in _patched_modules:
        setattr(module, "uuid6", _original_uuid6)
    _original_uuid6 = None
    _langgraph_uuid = None
    _patched_modules = []
    _patch_depth = 0


class CheckpointIdInterceptor(Interceptor):
    """Enable replay-stable checkpoint ids for a run, for its instance's lifetime."""

    def workflow_interceptor_class(
        self, input: WorkflowInterceptorClassInput
    ) -> type[WorkflowInboundInterceptor]:
        """Return an inbound interceptor that owns the instance's tick floor."""

        class Inbound(WorkflowInboundInterceptor):
            # One inbound interceptor is built per workflow instance, so its
            # state is the instance's own. A newly constructed instance
            # re-executes its run from the start (first task after an
            # eviction, every replay), so the tick floor starts at 0 with it
            # to re-derive the same ids.
            def __init__(self, next: WorkflowInboundInterceptor) -> None:
                super().__init__(next)
                self._state = _InstanceState()

            async def execute_workflow(self, input: ExecuteWorkflowInput) -> Any:
                _current_state.set(self._state)
                return await self.next.execute_workflow(input)

            async def handle_signal(self, input: HandleSignalInput) -> None:
                _current_state.set(self._state)
                return await self.next.handle_signal(input)

            async def handle_update_handler(self, input: HandleUpdateInput) -> Any:
                _current_state.set(self._state)
                return await self.next.handle_update_handler(input)

            # Neither runs on replay: a mint here must not advance the floor
            # the recorded handlers re-derive their ids from, so both run
            # with no state and get LangGraph's own uuid6.
            async def handle_query(self, input: HandleQueryInput) -> Any:
                _current_state.set(None)
                return await self.next.handle_query(input)

            def handle_update_validator(self, input: HandleUpdateInput) -> None:
                # Synchronous, and the Update handler runs later in the same
                # task: restore what was there.
                token = _current_state.set(None)
                try:
                    self.next.handle_update_validator(input)
                finally:
                    _current_state.reset(token)

        return Inbound
