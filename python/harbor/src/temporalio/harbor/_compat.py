"""The one place this package reaches past harbor's public API.

Before a job runs any trial, harbor's ``Job`` resolves its datasets into tasks
and works out which metrics each dataset is scored with. Those resolvers are
the only faithful account of how registry, package and local datasets behave:
a package dataset, for one, downloads a ``metric.py`` that becomes a
``uv-script`` metric. They are static methods, so they are called here without
constructing a ``Job``, whose constructor creates job directories, locks and
log handlers.

Once a trial has run and failed, harbor's ``TrialQueue`` decides whether to
run it again and how long to wait first. Calling its methods, rather than
restating them, keeps the plugin's default retries identical to harbor's in
whichever harbor release is installed. A ``TrialQueue`` holds only a semaphore
and its configuration, so one is built per decision.

All of these are private to harbor. Importing this module checks their shape
and fails with the installed harbor version rather than drifting silently when
a harbor release moves them.
"""

from __future__ import annotations

import inspect
from importlib.metadata import version
from typing import Any

from harbor.job import Job
from harbor.metrics.base import BaseMetric
from harbor.models.job.config import JobConfig, RetryConfig
from harbor.models.trial.config import TaskConfig
from harbor.trial.queue import TrialQueue


def _missing(owner: type, name: str, params: tuple[str, ...]) -> ImportError:
    return ImportError(
        f"temporalio-harbor needs harbor's {owner.__name__}.{name}"
        f"({', '.join(params)}), which harbor {version('harbor')} does not "
        "provide in that form"
    )


def _require(owner: type, name: str, params: tuple[str, ...]) -> None:
    member = inspect.getattr_static(owner, name, None)
    if isinstance(member, staticmethod):
        member = member.__func__
    elif params[:1] != ("self",):
        member = None  # a static method is required, and this is not one
    if inspect.isfunction(member):
        if tuple(inspect.signature(member).parameters) == params:
            return
    raise _missing(owner, name, params)


_require(Job, "_resolve_task_configs", ("config",))
_require(Job, "_resolve_metrics", ("config", "task_configs"))
_require(TrialQueue, "_should_retry_exception", ("self", "exception_type"))
_require(TrialQueue, "_calculate_backoff_delay_sec", ("self", "attempt"))
# The constructor gains optional parameters over time; only these two matter.
if not {"n_concurrent", "retry_config"} <= set(
    inspect.signature(TrialQueue.__init__).parameters
):
    raise _missing(TrialQueue, "__init__", ("n_concurrent", "retry_config"))


async def resolve_task_configs(config: JobConfig) -> list[TaskConfig]:
    """Every task the job runs: its explicit tasks, then each dataset's."""
    return await Job._resolve_task_configs(config)


async def resolve_metrics(
    config: JobConfig, task_configs: list[TaskConfig]
) -> dict[str, list[BaseMetric[Any]]]:
    """The metrics harbor scores each dataset with, keyed by dataset name."""
    return await Job._resolve_metrics(config, task_configs)


def should_retry_exception(config: RetryConfig, exception_type: str) -> bool:
    """Whether harbor retries a trial that recorded ``exception_type``."""
    queue = TrialQueue(n_concurrent=1, retry_config=config)
    return queue._should_retry_exception(exception_type)


def backoff_delay_sec(config: RetryConfig, attempt: int) -> float:
    """How long harbor waits after the ``attempt``-th failure, counting from 0."""
    queue = TrialQueue(n_concurrent=1, retry_config=config)
    return queue._calculate_backoff_delay_sec(attempt)
