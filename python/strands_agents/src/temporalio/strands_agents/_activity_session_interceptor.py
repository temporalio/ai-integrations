"""Worker interceptor that places same-queue activities in the workflow session.

Runs inside the workflow sandbox, so it imports only ``temporalio`` and the
standard library.
"""

from __future__ import annotations

from typing import Any

import temporalio.common
import temporalio.worker
import temporalio.workflow


class _ActivitySessionInterceptor(temporalio.worker.Interceptor):  # type:ignore[reportUnusedClass]
    """Fill ``session_options`` on activities the workflow starts on its own queue."""

    def __init__(self, options: temporalio.common.ActivitySessionOptions) -> None:
        self.options = options

    def workflow_interceptor_class(
        self, input: temporalio.worker.WorkflowInterceptorClassInput
    ) -> type[temporalio.worker.WorkflowInboundInterceptor]:
        options = self.options

        class _Inbound(temporalio.worker.WorkflowInboundInterceptor):
            def init(
                self, outbound: temporalio.worker.WorkflowOutboundInterceptor
            ) -> None:
                super().init(_ActivitySessionOutboundInterceptor(outbound, options))

        return _Inbound


class _ActivitySessionOutboundInterceptor(
    temporalio.worker.WorkflowOutboundInterceptor
):
    def __init__(
        self,
        next: temporalio.worker.WorkflowOutboundInterceptor,
        options: temporalio.common.ActivitySessionOptions,
    ) -> None:
        super().__init__(next)
        self._options = options

    def start_activity(
        self, input: temporalio.worker.StartActivityInput
    ) -> temporalio.workflow.ActivityHandle[Any]:
        # A user's ``workflow.activity_session`` scope has already filled the
        # field; activities on other task queues cannot join the session.
        if input.session_options is None and (
            input.task_queue is None
            or input.task_queue == temporalio.workflow.info().task_queue
        ):
            input.session_options = self._options
        return self.next.start_activity(input)
