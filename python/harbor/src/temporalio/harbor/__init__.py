"""Run harbor evaluation jobs durably, one Activity per trial.

Register :class:`HarborPlugin` on the Client. In a workflow, resolve a harbor
``JobConfig`` with :func:`plan_job`, run each trial with :func:`execute_trial`,
and compute the job's statistics with :func:`aggregate_job`; the numbers match
what ``harbor run`` reports for the same job.

This package is experimental and may change in future versions.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from temporalio.harbor._plugin import HarborPlugin
    from temporalio.harbor._retry import retry_policy_from_harbor
    from temporalio.harbor._types import JobPlan
    from temporalio.harbor._workflow import aggregate_job, execute_trial, plan_job

__all__ = [
    "HarborPlugin",
    "JobPlan",
    "aggregate_job",
    "execute_trial",
    "plan_job",
    "retry_policy_from_harbor",
]

_MODULES = {
    "HarborPlugin": "_plugin",
    "JobPlan": "_types",
    "aggregate_job": "_workflow",
    "execute_trial": "_workflow",
    "plan_job": "_workflow",
    "retry_policy_from_harbor": "_retry",
}


def __getattr__(name: str) -> Any:
    """Load each public symbol on first use, keeping worker-only code out of workflows."""
    module = _MODULES.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(f"temporalio.harbor.{module}"), name)
