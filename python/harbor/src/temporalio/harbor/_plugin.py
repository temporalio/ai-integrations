"""Worker plugin that runs harbor trials as Activities."""

from __future__ import annotations

import dataclasses

from temporalio.contrib.pydantic import (
    PydanticPayloadConverter,
    pydantic_data_converter,
)
from temporalio.converter import DataConverter, DefaultPayloadConverter
from temporalio.harbor._activity import HarborActivities
from temporalio.harbor._hooks import TrialHooks
from temporalio.plugin import SimplePlugin
from temporalio.worker import WorkflowRunner
from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner


class HarborPlugin(SimplePlugin):
    """Run harbor evaluation jobs durably, one Activity per trial.

    This class is experimental and may change in future versions.

    Registers the Activities that :func:`temporalio.harbor.plan_job`,
    :func:`temporalio.harbor.execute_trial` and
    :func:`temporalio.harbor.aggregate_job` schedule, and passes harbor
    through the workflow sandbox. Harbor's configuration and results are
    pydantic models, so the plugin makes Temporal's pydantic payload converter
    the data converter's payload converter, keeping any payload codec and
    failure converter. A custom payload converter is left in place.

    Register it on the Client, so both workflows and the starter use the same
    converter, and it is applied to every Worker built from that Client.
    """

    def __init__(self, *, trial_hooks: TrialHooks | None = None) -> None:
        """Create the plugin.

        Args:
            trial_hooks: How this worker takes part in running each trial.
                Defaults to :class:`temporalio.harbor.TrialHooks`, which runs
                each trial as harbor's own ``Trial`` runs it.
        """
        activities = HarborActivities(
            hooks=trial_hooks if trial_hooks is not None else TrialHooks()
        )

        def data_converter(converter: DataConverter | None) -> DataConverter:
            if converter is None:
                return pydantic_data_converter
            if converter.payload_converter_class is DefaultPayloadConverter:
                return dataclasses.replace(
                    converter, payload_converter_class=PydanticPayloadConverter
                )
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
                activities.plan_job,
                activities.run_trial,
                activities.aggregate_job,
            ],
            workflow_runner=workflow_runner,
        )
