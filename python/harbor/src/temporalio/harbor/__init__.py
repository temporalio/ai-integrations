"""Run harbor evaluation jobs durably, one Activity per trial.

Register :class:`HarborPlugin` on the Client. In a workflow, resolve a harbor
``JobConfig`` with :func:`plan_job`, run each trial with :func:`execute_trial`,
and aggregate the outcomes into harbor's ``JobResult`` with
:func:`aggregate_job`. Subclass :class:`TrialHooks` to take part in how the
worker runs each trial.

This package is experimental and may change in future versions.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from temporalio.harbor._hooks import TrialContext, TrialHooks
    from temporalio.harbor._plugin import HarborPlugin
    from temporalio.harbor._types import TemporalJobPlan, TrialOutcome, TrialPlan
    from temporalio.harbor._workflow import aggregate_job, execute_trial, plan_job

__all__ = [
    "HarborPlugin",
    "TemporalJobPlan",
    "TrialContext",
    "TrialHooks",
    "TrialOutcome",
    "TrialPlan",
    "aggregate_job",
    "execute_trial",
    "plan_job",
]

_MODULES = {
    "HarborPlugin": "_plugin",
    "TemporalJobPlan": "_types",
    "TrialContext": "_hooks",
    "TrialHooks": "_hooks",
    "TrialOutcome": "_types",
    "TrialPlan": "_types",
    "aggregate_job": "_workflow",
    "execute_trial": "_workflow",
    "plan_job": "_workflow",
}


def __getattr__(name: str) -> Any:
    """Load each public symbol on first use, keeping worker-only code out of workflows."""
    module = _MODULES.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(f"temporalio.harbor.{module}"), name)
