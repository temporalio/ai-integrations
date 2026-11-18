"""Fail the Workflow when a ``ModelBehaviorError`` escapes it.

``Runner.run`` raises the Agents SDK's own exceptions to its caller unchanged, so
Workflow code can catch them as it would outside Temporal. Only an exception that
escapes the Workflow, or one of its signal or update handlers, is classified here.
"""

from typing import Any, NoReturn

from agents.exceptions import ModelBehaviorError

import temporalio.worker
from temporalio import workflow
from temporalio.exceptions import ApplicationError

MODEL_BEHAVIOR_ERROR_FAILS_WORKFLOW_PATCH = (
    "temporalio.openai_agents.model-behavior-error-fails-workflow"
)
"""Patch ID that gates failing the Workflow on an escaped ``ModelBehaviorError``.

Runs started before the patch keep failing the Workflow Task, as they did before.
"""


def model_behavior_error_fails_workflow() -> bool:
    """Return whether an escaped ``ModelBehaviorError`` fails this Workflow run.

    ``Runner.run`` calls this before it schedules any model Activity, so a run that
    was already retrying a Workflow Task replays the call without the patch marker
    and keeps the old behavior. The SDK memoizes the answer for the rest of the run.
    """
    return workflow.patched(MODEL_BEHAVIOR_ERROR_FAILS_WORKFLOW_PATCH)


def fails_workflow_if_uncaught(e: BaseException) -> bool:
    """Return whether ``e`` fails the Workflow if Workflow code doesn't catch it.

    ``Runner.run`` uses it to wait for the run's tool operations to finish cancelling
    first, as it does for other Workflow failures.
    """
    return isinstance(e, ModelBehaviorError) and model_behavior_error_fails_workflow()


def _raise_workflow_failure(e: ModelBehaviorError) -> NoReturn:
    if not model_behavior_error_fails_workflow():
        raise e
    # The model's answer is in the History, so retrying the Workflow Task would get
    # the same answer and fail the same way. A retryable ApplicationError fails the
    # Workflow and leaves another run to the Workflow retry policy.
    failure = ApplicationError(str(e), type=type(e).__name__)
    raise failure.with_traceback(e.__traceback__) from e.__cause__


class ModelBehaviorErrorInterceptor(temporalio.worker.Interceptor):
    """Worker interceptor that fails the Workflow on an escaped ``ModelBehaviorError``."""

    def workflow_interceptor_class(
        self, input: temporalio.worker.WorkflowInterceptorClassInput
    ) -> type[temporalio.worker.WorkflowInboundInterceptor]:
        """Return the inbound interceptor that classifies escaped exceptions."""
        return _ModelBehaviorErrorWorkflowInboundInterceptor


class _ModelBehaviorErrorWorkflowInboundInterceptor(
    temporalio.worker.WorkflowInboundInterceptor
):
    async def execute_workflow(
        self, input: temporalio.worker.ExecuteWorkflowInput
    ) -> Any:
        try:
            return await self.next.execute_workflow(input)
        except ModelBehaviorError as e:
            _raise_workflow_failure(e)

    async def handle_signal(self, input: temporalio.worker.HandleSignalInput) -> None:
        try:
            return await self.next.handle_signal(input)
        except ModelBehaviorError as e:
            _raise_workflow_failure(e)

    async def handle_update_handler(
        self, input: temporalio.worker.HandleUpdateInput
    ) -> Any:
        try:
            return await self.next.handle_update_handler(input)
        except ModelBehaviorError as e:
            _raise_workflow_failure(e)
