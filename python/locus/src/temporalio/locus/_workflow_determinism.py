"""Give Locus's loop the workflow's clock, ids and randomness.

Locus's agent loop reads the wall clock (time budgets, ``TimeLimit``, event and
state timestamps), makes ids (run ids, tool-call ids) and jitters its retry
delays. Those modules pass through the workflow sandbox, so on replay they
would read new values and a run could take a different branch than History
records.

Where Locus has ``locus.core.sources``, the seam its loop reads these from,
the plugin installs sources that use the workflow's. Older Locus releases have
no seam, so there the plugin replaces the module-local ``time``, ``datetime``,
``uuid4`` and ``random`` names in the Locus modules the loop runs.

Either way every read checks where it runs: inside a workflow it uses the
workflow's deterministic source; in activities and in plain Locus code in the
same process it uses the real one.
"""

from __future__ import annotations

import datetime as _datetime
import importlib
import random as _random
import threading
import time as _time
import uuid as _uuid
from typing import Any

from temporalio import workflow

# The Locus modules the agent loop and the multi-agent shapes run, and that read
# a clock, make an id or draw a random number.
_MODULES = (
    "locus.core.state",
    "locus.core.events",
    "locus.core.interrupt",
    "locus.core.messages",
    "locus.core.termination",
    "locus.agent.agent",
    "locus.agent.runtime_loop",
    "locus.agent.model_retry",
    "locus.tools.executor",
    "locus.multiagent.functional",
    "locus.multiagent.graph",
    "locus.multiagent.handoff",
    "locus.multiagent.orchestrator",
    "locus.multiagent.specialist",
    "locus.multiagent.swarm",
)


def _in_workflow() -> bool:
    return workflow.in_workflow() and not workflow.unsafe.is_sandbox_unrestricted()


class _Time:
    """``time``, with workflow time for the clock reads."""

    def time(self) -> float:
        return workflow.time() if _in_workflow() else _time.time()

    # Locus only subtracts these, to measure a duration or a budget.
    def monotonic(self) -> float:
        return workflow.time() if _in_workflow() else _time.monotonic()

    def perf_counter(self) -> float:
        return workflow.time() if _in_workflow() else _time.perf_counter()

    def __getattr__(self, name: str) -> Any:
        return getattr(_time, name)


class _Random:
    """``random``, with the workflow's seeded generator for draws."""

    def uniform(self, a: float, b: float) -> float:
        return (workflow.random() if _in_workflow() else _random).uniform(a, b)

    def random(self) -> float:
        return (workflow.random() if _in_workflow() else _random).random()

    def __getattr__(self, name: str) -> Any:
        return getattr(_random, name)


class _DatetimeMeta(type):
    # The stand-in replaces the name ``datetime`` in Locus modules, so checks
    # against it must accept the plain datetimes it returns.
    def __instancecheck__(cls, instance: object) -> bool:
        return isinstance(instance, _datetime.datetime)

    def __subclasscheck__(cls, subclass: type) -> bool:
        return issubclass(subclass, _datetime.datetime)


class _Datetime(_datetime.datetime, metaclass=_DatetimeMeta):
    """``datetime``, with workflow time for ``now``."""

    @classmethod
    def now(cls, tz: _datetime.tzinfo | None = None) -> _datetime.datetime:  # type: ignore[override]
        if not _in_workflow():
            return _datetime.datetime.now(tz)
        now = workflow.now()
        return now.astimezone(tz) if tz is not None else now.replace(tzinfo=None)

    @classmethod
    def utcnow(cls) -> _datetime.datetime:  # type: ignore[override]
        if not _in_workflow():
            return _datetime.datetime.now(_datetime.UTC).replace(tzinfo=None)
        return workflow.now().replace(tzinfo=None)


def _uuid4() -> _uuid.UUID:
    return workflow.uuid4() if _in_workflow() else _uuid.uuid4()


_clock = _Time()
_rng = _Random()
_REPLACEMENTS: dict[str, tuple[Any, Any]] = {
    "time": (_time, _clock),
    "random": (_random, _rng),
    "datetime": (_datetime.datetime, _Datetime),
    "uuid4": (_uuid.uuid4, _uuid4),
}
_lock = threading.Lock()


class _WorkflowSources:
    """Locus runtime sources: the workflow's inside a workflow, else the previous ones."""

    def __init__(self, outside: Any) -> None:
        self._outside = outside

    def now(self) -> _datetime.datetime:
        return workflow.now() if _in_workflow() else self._outside.now()

    def time(self) -> float:
        return workflow.time() if _in_workflow() else self._outside.time()

    def monotonic(self) -> float:
        return workflow.time() if _in_workflow() else self._outside.monotonic()

    def uuid4(self) -> _uuid.UUID:
        return workflow.uuid4() if _in_workflow() else self._outside.uuid4()

    def uniform(self, a: float, b: float) -> float:
        if _in_workflow():
            return workflow.random().uniform(a, b)
        return self._outside.uniform(a, b)


def _install_sources() -> bool:
    """Install workflow sources through Locus's seam; ``False`` if it has none."""
    try:
        sources = importlib.import_module("locus.core.sources")
    except ImportError:
        return False
    current = sources.get_sources()
    if not isinstance(current, _WorkflowSources):
        sources.set_sources(_WorkflowSources(current))
    return True


def install_workflow_determinism() -> None:
    """Give Locus's loop the workflow's clock, ids and randomness. Idempotent."""
    with _lock:
        if _install_sources():
            return
        for module_name in _MODULES:
            module = importlib.import_module(module_name)
            for name, (original, replacement) in _REPLACEMENTS.items():
                if getattr(module, name, None) is original:
                    setattr(module, name, replacement)
        retry = importlib.import_module("locus.agent.model_retry")
        # model_retry binds these at import time, as test seams.
        if getattr(retry, "_uniform", None) is _random.uniform:
            retry._uniform = _rng.uniform  # type: ignore[attr-defined]
        if getattr(retry, "_monotonic", None) is _time.monotonic:
            retry._monotonic = _clock.monotonic  # type: ignore[attr-defined]
