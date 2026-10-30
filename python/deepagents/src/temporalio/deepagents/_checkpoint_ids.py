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
built only from replay-safe inputs. Outside workflows the original ``uuid6``
still answers, so client-side LangGraph use in the same process is
untouched.

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

import hashlib
import importlib.util
from typing import Any

from temporalio import workflow
from temporalio.worker import (
    ExecuteWorkflowInput,
    Interceptor,
    WorkflowInboundInterceptor,
    WorkflowInterceptorClassInput,
    WorkflowOutboundInterceptor,
)

_original_uuid6: Any = None
_langgraph_uuid: Any = None
_patched_modules: list[Any] = []
# Active install() nestings: two workers with this plugin can be alive in one
# process, and the first to shut down must not unpatch the survivor.
_patch_depth = 0

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

# Highest tick minted so far, per workflow run. Keyed by run id because one
# worker interleaves many runs on a single event loop and a shared value
# would make a run's ids depend on that interleaving; plain dict operations,
# with no await between them, cannot interleave. Entries are dropped when
# the run finishes and reset when its workflow instance is (re)created — see
# :class:`CheckpointIdInterceptor` for why resetting is replay-safe.
_last_tick: dict[str, int] = {}


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


def _workflow_uuid6(node: int | None = None, clock_seq: int | None = None) -> Any:
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
    - the 48-bit node field is fixed per run, derived from the run id, so
      different executions of the same workflow mint different ids without
      drawing from the workflow RNG (which would couple checkpoint ids to
      unrelated user code's randomness). LangGraph normally passes
      ``clock_seq=step``, so two runs alive at the same moment differ in
      those 48 hash bits — distinct with overwhelming probability, not by
      construction.

    Args:
        node: Passed through by LangGraph as ``None`` (uuid6's random-node
            default); hashed from the run id when absent.
        clock_seq: LangGraph always passes the superstep here; hashed from
            the run id only to mirror the original signature.
    """
    run_id = workflow.info().run_id
    # uuid6 ticks of the workflow's current time. ``workflow.time_ns()`` is
    # the current activation's recorded timestamp — original on first pass,
    # the same recorded value on every re-execution — so ids land in the
    # same time range as unpatched uuid6 ids and order sensibly next to
    # them. The floor advances past the run's previous tick, so mints that
    # fall between two events (same timestamp) stay distinct and increasing.
    tick = max(
        workflow.time_ns() // 100 + _UUID6_EPOCH_OFFSET_TICKS,
        _last_tick.get(run_id, 0) + 1,
    )
    _last_tick[run_id] = tick
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
        if workflow.in_workflow():
            return _workflow_uuid6(node, clock_seq)
        return _original_uuid6(node, clock_seq)

    for module in modules:
        setattr(module, "uuid6", patched_uuid6)
    _patched_modules = modules


def uninstall_checkpoint_id_patch() -> None:
    """Restore the original ``uuid6`` bindings and drop all tick floors.

    Restore only when the last overlapping install has uninstalled.
    """
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
    _last_tick.clear()


class CheckpointIdInterceptor(Interceptor):
    """Scope the per-run tick floor to the workflow instance's lifetime."""

    def workflow_interceptor_class(
        self, input: WorkflowInterceptorClassInput
    ) -> type[WorkflowInboundInterceptor]:
        """Return an inbound interceptor that resets the tick floor per instance."""

        class Inbound(WorkflowInboundInterceptor):
            # Set in init(); defaulted so the declaration is a complete one.
            _run_id: str = ""

            def init(self, outbound: WorkflowOutboundInterceptor) -> None:
                # Captured once here: the cleanup in execute_workflow must
                # not depend on workflow context still being available while
                # the instance is being torn down.
                self._run_id = outbound.info().run_id
                # A newly constructed instance re-executes its run from the
                # start (first task after an eviction, every replay), so the
                # tick floor must restart with it to re-derive the same ids.
                _last_tick.pop(self._run_id, None)
                super().init(outbound)

            async def execute_workflow(self, input: ExecuteWorkflowInput) -> Any:
                try:
                    return await self.next.execute_workflow(input)
                finally:
                    # This instance's run is over; drop its floor so a
                    # long-lived worker does not accumulate an entry per run.
                    # (Replay of a still-running execution in this same
                    # process would restart the live instance's numbering —
                    # an unsupported combination, noted for completeness.)
                    _last_tick.pop(self._run_id, None)

        return Inbound
