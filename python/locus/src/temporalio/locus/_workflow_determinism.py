"""Give Locus's loop the workflow's clock, ids and randomness.

Locus's agent loop reads the wall clock (time budgets, ``TimeLimit``, event and
state timestamps), makes ids (run ids, tool-call ids) and jitters its retry
delays. Those modules pass through the workflow sandbox, so on replay they
would read new values and a run could take a different branch than History
records.

Locus reads all of them through ``locus.core.sources``. The plugin installs
sources there that check where each read runs: inside a workflow it uses the
workflow's deterministic source; in activities and in plain Locus code in the
same process it uses the sources that were installed before.
"""

from __future__ import annotations

import datetime
import threading
import uuid

from locus.core import sources
from locus.core.sources import RuntimeSources
from temporalio import workflow

_lock = threading.Lock()


def _in_workflow() -> bool:
    return workflow.in_workflow() and not workflow.unsafe.is_sandbox_unrestricted()


class _WorkflowSources:
    """Locus runtime sources: the workflow's inside a workflow, else the previous ones."""

    def __init__(self, outside: RuntimeSources) -> None:
        self._outside = outside

    def now(self) -> datetime.datetime:
        return workflow.now() if _in_workflow() else self._outside.now()

    def time(self) -> float:
        return workflow.time() if _in_workflow() else self._outside.time()

    # Locus only subtracts these, to measure a duration or a budget.
    def monotonic(self) -> float:
        return workflow.time() if _in_workflow() else self._outside.monotonic()

    def uuid4(self) -> uuid.UUID:
        return workflow.uuid4() if _in_workflow() else self._outside.uuid4()

    def uniform(self, a: float, b: float) -> float:
        if _in_workflow():
            return workflow.random().uniform(a, b)
        return self._outside.uniform(a, b)


def install_workflow_determinism() -> None:
    """Give Locus's loop the workflow's clock, ids and randomness. Idempotent."""
    with _lock:
        current = sources.get_sources()
        if not isinstance(current, _WorkflowSources):
            sources.set_sources(_WorkflowSources(current))
