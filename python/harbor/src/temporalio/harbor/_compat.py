"""The one place this package reaches past harbor's public API.

Before a job runs any trial, harbor's ``Job`` resolves its datasets into tasks
and works out which metrics each dataset is scored with. Those resolvers are
the only faithful account of how registry, package and local datasets behave:
a package dataset, for one, downloads a ``metric.py`` that becomes a
``uv-script`` metric. They are static methods, so they are called here without
constructing a ``Job``, whose constructor creates job directories, locks and
log handlers.

Both are private to harbor. Importing this module checks their shape and fails
with the installed harbor version rather than drifting silently when a harbor
release moves them.
"""

from __future__ import annotations

import inspect
from importlib.metadata import version
from typing import Any

from harbor.job import Job
from harbor.metrics.base import BaseMetric
from harbor.models.job.config import JobConfig
from harbor.models.trial.config import TaskConfig


def _require_static(name: str, params: tuple[str, ...]) -> None:
    member = inspect.getattr_static(Job, name, None)
    if isinstance(member, staticmethod):
        if tuple(inspect.signature(member.__func__).parameters) == params:
            return
    raise ImportError(
        f"temporalio-harbor needs harbor's Job.{name}({', '.join(params)}), "
        f"which harbor {version('harbor')} does not provide in that form"
    )


_require_static("_resolve_task_configs", ("config",))
_require_static("_resolve_metrics", ("config", "task_configs"))


async def resolve_task_configs(config: JobConfig) -> list[TaskConfig]:
    """Every task the job runs: its explicit tasks, then each dataset's."""
    return await Job._resolve_task_configs(config)


async def resolve_metrics(
    config: JobConfig, task_configs: list[TaskConfig]
) -> dict[str, list[BaseMetric[Any]]]:
    """The metrics harbor scores each dataset with, keyed by dataset name."""
    return await Job._resolve_metrics(config, task_configs)
