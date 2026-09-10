"""Test helpers for temporalio-mcp."""

# template-override: MCP owns its test helpers.

import uuid
from collections.abc import Callable, Sequence

from temporalio.client import Client
from temporalio.worker import Worker, WorkflowRunner
from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner


def new_worker(
    client: Client,
    *workflows: type,
    activities: Sequence[Callable] = [],
    task_queue: str | None = None,
    workflow_runner: WorkflowRunner = SandboxedWorkflowRunner(),
    max_cached_workflows: int = 1000,
    workflow_failure_exception_types: Sequence[type[BaseException]] = [],
    **kwargs,  # type:ignore[reportMissingParameterType]
) -> Worker:
    return Worker(
        client,
        task_queue=task_queue or str(uuid.uuid4()),
        workflows=workflows,
        activities=activities,
        workflow_runner=workflow_runner,
        max_cached_workflows=max_cached_workflows,
        workflow_failure_exception_types=workflow_failure_exception_types,
        **kwargs,
    )
