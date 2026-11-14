"""Use Workflow time in Strands' duration measurements.

Strands' timing modules import ``time`` directly, and those modules pass through
the Workflow sandbox. Adapt their module-local clock references rather than
changing the process-wide ``time`` module or replacing Strands' metrics objects.
Every read checks its execution context, so Activities and native Strands calls
in the same worker keep their normal clocks.
"""

import importlib
import threading
import time
from typing import Any

from temporalio import workflow


class _WorkflowTime:
    def time(self) -> float:
        return workflow.time() if workflow.in_workflow() else time.time()

    def monotonic(self) -> float:
        # Strands uses this only to subtract timestamps (for tool-hook durations).
        return workflow.time() if workflow.in_workflow() else time.monotonic()

    def __getattr__(self, name: str) -> Any:
        return getattr(time, name)


_clock = _WorkflowTime()
_install_lock = threading.Lock()


def install_workflow_time() -> None:
    """Adapt Strands' timing dependencies for both Workers and Replayers."""
    with _install_lock:
        names = (
            "strands.telemetry.metrics",
            "strands.event_loop.streaming",
            "strands.tools.executors._executor",
        )
        modules = [importlib.import_module(name) for name in names]
        for module in modules:
            clock = getattr(module, "time", None)
            if clock is not time and clock is not _clock:
                raise RuntimeError(
                    f"Unsupported Strands clock in {module.__name__}; "
                    "the Strands plugin requires a module-local time dependency "
                    "to make duration metrics replay-safe."
                )
        for module in modules:
            setattr(module, "time", _clock)
