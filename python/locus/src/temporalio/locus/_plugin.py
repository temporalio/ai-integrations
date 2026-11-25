"""The Temporal plugin for the Locus agent SDK."""

from __future__ import annotations

import dataclasses
import importlib.metadata
from collections.abc import Callable

from locus.models.base import ModelProtocol
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.converter import DataConverter, DefaultPayloadConverter
from temporalio.plugin import SimplePlugin
from temporalio.worker import WorkflowRunner
from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner

from ._model_activity import ModelActivities
from ._sync_tools import run_sync_tools_in_workflows
from ._workflow_determinism import install_workflow_determinism


class LocusPlugin(SimplePlugin):
    """Temporal plugin for the Locus agent SDK.

    ``models`` maps a name to a factory that returns a Locus model. A factory
    is called on the worker on first use, outside the workflow sandbox, and the
    model is kept for the worker's lifetime. Use the same name in
    ``TemporalModel(name, ...)`` in workflow code::

        plugin = LocusPlugin(models={"default": lambda: get_model("oci:openai.gpt-5.5")})
        client = await Client.connect("localhost:7233", plugins=[plugin])

    Credentials stay in the factory, on the worker; they never enter Event
    History.
    """

    def __init__(self, *, models: dict[str, Callable[[], ModelProtocol]]) -> None:
        """Register the model activities for ``models``."""
        if not models:
            raise ValueError("LocusPlugin needs at least one model factory")
        super().__init__(
            "oracle.LocusPlugin",
            data_converter=_data_converter,
            activities=ModelActivities(dict(models)).activities(),
            workflow_runner=_workflow_runner,
        )

    def version(self) -> str:
        """Return this package's version, which worker heartbeats report."""
        try:
            return importlib.metadata.version("temporalio-locus")
        except importlib.metadata.PackageNotFoundError:
            return ""


def _workflow_runner(runner: WorkflowRunner | None) -> WorkflowRunner:
    if not runner:
        raise ValueError("No WorkflowRunner provided to the Locus plugin.")
    install_workflow_determinism()
    run_sync_tools_in_workflows()
    if isinstance(runner, SandboxedWorkflowRunner):
        return dataclasses.replace(
            runner,
            restrictions=runner.restrictions.with_passthrough_modules(
                "locus",
                # ``pydantic`` is in the SDK's default passthrough; extend it to its
                # compiled core and ``Annotated`` helper.
                "pydantic_core",
                "annotated_types",
            ),
        )
    return runner


def _data_converter(converter: DataConverter | None) -> DataConverter:
    if converter is None:
        return pydantic_data_converter
    if converter.payload_converter_class is DefaultPayloadConverter:
        return dataclasses.replace(
            converter,
            payload_converter_class=pydantic_data_converter.payload_converter_class,
        )
    return converter
