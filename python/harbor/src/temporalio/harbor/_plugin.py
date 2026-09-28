"""Worker plugin that runs harbor trials as Activities."""

from __future__ import annotations

import dataclasses
from datetime import timedelta

from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.converter import DataConverter
from temporalio.harbor._activity import HarborActivities
from temporalio.plugin import SimplePlugin
from temporalio.worker import WorkflowRunner
from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner


class HarborPlugin(SimplePlugin):
    """Run harbor evaluation jobs durably, one Activity per trial.

    This class is experimental and may change in future versions.

    Registers the Activities that :func:`temporalio.harbor.plan_job`,
    :func:`temporalio.harbor.execute_trial` and
    :func:`temporalio.harbor.aggregate_job` schedule. Harbor's configuration
    and results are pydantic models, so the plugin installs Temporal's pydantic
    data converter unless a custom converter is already configured.

    Register it on the Client, so both workflows and the starter use the same
    converter, and it is applied to every Worker built from that Client.
    """

    def __init__(
        self, *, heartbeat_interval: timedelta = timedelta(seconds=30)
    ) -> None:
        """Create the plugin.

        Args:
            heartbeat_interval: How often a running trial heartbeats. Must be
                well under the ``heartbeat_timeout`` given to
                :func:`temporalio.harbor.execute_trial`.
        """
        activities = HarborActivities(heartbeat_interval=heartbeat_interval)

        def data_converter(converter: DataConverter | None) -> DataConverter:
            if converter is None or converter == DataConverter.default:
                return pydantic_data_converter
            return converter

        def workflow_runner(runner: WorkflowRunner | None) -> WorkflowRunner:
            if runner is None:
                raise ValueError("No WorkflowRunner provided to the harbor plugin")
            if isinstance(runner, SandboxedWorkflowRunner):
                # Workflow code builds and reads harbor's pydantic models; they
                # must be the same classes the data converter produces.
                return dataclasses.replace(
                    runner,
                    restrictions=runner.restrictions.with_passthrough_modules("harbor"),
                )
            return runner

        super().__init__(
            name="HarborPlugin",
            data_converter=data_converter,
            activities=[
                activities.resolve_job,
                activities.run_trial,
                activities.compute_metrics,
            ],
            workflow_runner=workflow_runner,
        )
