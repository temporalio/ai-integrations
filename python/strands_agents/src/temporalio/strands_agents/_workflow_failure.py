"""Fail the Workflow when Strands' agent loop gives up on a model answer.

Strands raises ``StructuredOutputException`` and ``MaxTokensReachedException`` in
Workflow code from model answers that are already in History, so a retried
Workflow Task would raise them again. Workflow code still gets them unchanged and
can catch them. Only an exception that escapes the Workflow, or one of its signal or
update handlers, is classified here.
"""

from typing import Any, NoReturn

from strands.types.exceptions import (
    MaxTokensReachedException,
    StructuredOutputException,
)

import temporalio.worker
from temporalio import workflow
from temporalio.exceptions import ApplicationError

AGENT_LOOP_ERRORS_FAIL_WORKFLOW_PATCH = (
    "temporalio.strands_agents.agent-loop-errors-fail-workflow"
)
"""Patch ID that gates failing the Workflow on an escaped agent-loop exception.

Runs that started an invocation before the patch keep failing the Workflow Task, as
they did before.
"""

_AGENT_LOOP_EXCEPTIONS = (StructuredOutputException, MaxTokensReachedException)


def agent_loop_errors_fail_workflow() -> bool:
    """Return whether an escaped agent-loop exception fails this Workflow run.

    ``TemporalAgent`` calls this before it schedules the first model Activity of an
    invocation, so a run that was already retrying a Workflow Task replays the call
    without the patch marker and keeps the old behavior. The SDK memoizes the answer
    for the rest of the run.
    """
    return workflow.patched(AGENT_LOOP_ERRORS_FAIL_WORKFLOW_PATCH)


def _raise_workflow_failure(e: BaseException) -> NoReturn:
    if not agent_loop_errors_fail_workflow():
        raise e
    # A retryable ApplicationError fails the Workflow and leaves another run to the
    # Workflow retry policy, which asks the model again.
    failure = ApplicationError(str(e), type=type(e).__name__)
    raise failure.with_traceback(e.__traceback__) from e.__cause__


class AgentLoopErrorInterceptor(temporalio.worker.Interceptor):
    """Worker interceptor that fails the Workflow on an escaped agent-loop exception."""

    def workflow_interceptor_class(
        self, input: temporalio.worker.WorkflowInterceptorClassInput
    ) -> type[temporalio.worker.WorkflowInboundInterceptor]:
        """Return the inbound interceptor that classifies escaped exceptions."""
        return _AgentLoopErrorWorkflowInboundInterceptor


class _AgentLoopErrorWorkflowInboundInterceptor(
    temporalio.worker.WorkflowInboundInterceptor
):
    async def execute_workflow(
        self, input: temporalio.worker.ExecuteWorkflowInput
    ) -> Any:
        try:
            return await self.next.execute_workflow(input)
        except _AGENT_LOOP_EXCEPTIONS as e:
            _raise_workflow_failure(e)

    async def handle_signal(self, input: temporalio.worker.HandleSignalInput) -> None:
        try:
            return await self.next.handle_signal(input)
        except _AGENT_LOOP_EXCEPTIONS as e:
            _raise_workflow_failure(e)

    async def handle_update_handler(
        self, input: temporalio.worker.HandleUpdateInput
    ) -> Any:
        try:
            return await self.next.handle_update_handler(input)
        except _AGENT_LOOP_EXCEPTIONS as e:
            _raise_workflow_failure(e)
