"""Classify terminal agent errors that escape workflow code."""

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from strands.types.exceptions import (
    MaxTokensReachedException,
    StructuredOutputException,
)

from temporalio import workflow
from temporalio.exceptions import ApplicationError
from temporalio.worker import (
    ExecuteWorkflowInput,
    HandleSignalInput,
    HandleUpdateInput,
    Interceptor,
    WorkflowInboundInterceptor,
    WorkflowInterceptorClassInput,
)


@contextmanager
def _classify_agent_failure() -> Generator[None, None, None]:
    try:
        yield
    except (StructuredOutputException, MaxTokensReachedException) as err:
        # Keep existing failure policies and histories that already recorded a failure.
        if not workflow.is_failure_exception(err) and workflow.patched(
            "temporalio-strands-agents-terminal-workflow-errors"
        ):
            raise ApplicationError(
                str(err), type=type(err).__name__, non_retryable=True
            ) from err
        raise


class _AgentFailureInboundInterceptor(WorkflowInboundInterceptor):
    async def execute_workflow(self, input: ExecuteWorkflowInput) -> Any:
        with _classify_agent_failure():
            return await super().execute_workflow(input)

    async def handle_signal(self, input: HandleSignalInput) -> None:
        with _classify_agent_failure():
            await super().handle_signal(input)

    async def handle_update_handler(self, input: HandleUpdateInput) -> Any:
        with _classify_agent_failure():
            return await super().handle_update_handler(input)


class AgentFailureInterceptor(Interceptor):
    """Install terminal agent error handling on the worker."""

    def workflow_interceptor_class(
        self, input: WorkflowInterceptorClassInput
    ) -> type[WorkflowInboundInterceptor]:
        """Return the interceptor that handles errors escaping workflow code."""
        return _AgentFailureInboundInterceptor
