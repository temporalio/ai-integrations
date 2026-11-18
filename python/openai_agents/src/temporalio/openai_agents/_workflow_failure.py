"""Fail the Workflow when a ``ModelBehaviorError`` escapes it.

``Runner.run`` raises the Agents SDK's own exceptions to its caller unchanged, so
Workflow code can catch them as it would outside Temporal. Only an exception that
escapes the Workflow, or one of its signal or update handlers, is classified here.
"""

import asyncio
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
    """Return whether ``e`` fails the Workflow if Workflow code doesn't catch it."""
    return isinstance(e, ModelBehaviorError) and model_behavior_error_fails_workflow()


_STOPPED_RUNS_ATTRIBUTE = "_temporal_openai_agents_stopped_runs"


def defer_tool_operations_wait(e: BaseException, run: Any) -> None:
    """Record that ``run`` stopped with ``e`` and may still have tools cancelling.

    ``Runner.run`` raises ``e`` without waiting, so Workflow code can catch it as it would
    outside Temporal. If ``e`` escapes the Workflow, the interceptor waits for the
    recorded runs' tool operations before it fails the Workflow, as ``Runner.run`` does
    for other Workflow failures. An agent used as a tool records its run first, then
    each enclosing run records its own.
    """
    runs: list[Any] | None = e.__dict__.get(_STOPPED_RUNS_ATTRIBUTE)
    if runs is None:
        runs = []
        setattr(e, _STOPPED_RUNS_ATTRIBUTE, runs)
    runs.append(run)


async def _raise_workflow_failure(e: ModelBehaviorError) -> NoReturn:
    if not model_behavior_error_fails_workflow():
        raise e
    # Imported here: the runner imports this module.
    from temporalio.openai_agents._openai_runner import _wait_for_tool_operations

    cancel_requested = False
    for run in e.__dict__.get(_STOPPED_RUNS_ATTRIBUTE, ()):
        # Finish every recorded run's wait, even after a cancel request arrived during an
        # earlier one, so no WAIT_COMPLETED tool is still cancelling when the Workflow closes.
        if await _wait_for_tool_operations(run):
            cancel_requested = True
    if cancel_requested:
        raise asyncio.CancelledError() from e
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
    def __init__(self, next: temporalio.worker.WorkflowInboundInterceptor) -> None:
        super().__init__(next)
        # Signal handlers whose escaped ModelBehaviorError waits for tool cleanup before it
        # fails the Workflow.
        self._failing_signals = 0

    async def execute_workflow(
        self, input: temporalio.worker.ExecuteWorkflowInput
    ) -> Any:
        try:
            result = await self.next.execute_workflow(input)
        except ModelBehaviorError as e:
            await _raise_workflow_failure(e)
        # A Signal failure that is still waiting for tool cleanup must fail the Workflow,
        # so the run method doesn't complete it first.
        await workflow.wait_condition(lambda: self._failing_signals == 0)
        return result

    async def handle_signal(self, input: temporalio.worker.HandleSignalInput) -> None:
        try:
            return await self.next.handle_signal(input)
        except ModelBehaviorError as e:
            self._failing_signals += 1
            try:
                await _raise_workflow_failure(e)
            finally:
                self._failing_signals -= 1

    async def handle_update_handler(
        self, input: temporalio.worker.HandleUpdateInput
    ) -> Any:
        try:
            return await self.next.handle_update_handler(input)
        except ModelBehaviorError as e:
            await _raise_workflow_failure(e)
