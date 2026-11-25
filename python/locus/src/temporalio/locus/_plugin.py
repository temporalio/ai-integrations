"""The Temporal plugin for the Locus agent SDK."""

from __future__ import annotations

import dataclasses
import importlib.metadata
from collections.abc import Callable, Sequence
from datetime import timedelta
from typing import TYPE_CHECKING, Any, cast

from locus.models.base import ModelProtocol
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.converter import DataConverter, DefaultPayloadConverter
from temporalio.plugin import SimplePlugin
from temporalio.worker import WorkflowRunner
from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner

from ._model_activity import ModelActivities
from ._sync_tools import run_sync_tools_in_workflows
from ._workflow_determinism import install_workflow_determinism

if TYPE_CHECKING:
    from mcp import Client as MCPClient


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

    ``mcp_clients`` maps a server name to a factory that takes no arguments
    and returns an ``mcp.Client`` (MCP Python SDK v2). It registers the
    Activities of a ``temporalio.mcp.MCPPlugin`` for each server, and needs
    the ``mcp`` extra. Use the same name in ``TemporalMCPClient(name, ...)``
    in workflow code. The worker calls a factory each time it opens a
    connection, so return a new ``mcp.Client`` and transport each time. Do
    not also register the same server name through ``MCPPlugin`` or another
    plugin on the same worker.

    ``mcp_connection_idle_timeout`` is how long the worker keeps an idle MCP
    connection open for reuse; ``None`` keeps it, zero disables reuse.
    """

    def __init__(
        self,
        *,
        models: dict[str, Callable[[], ModelProtocol]],
        mcp_clients: dict[str, Callable[[], MCPClient]] | None = None,
        mcp_connection_idle_timeout: timedelta | None = timedelta(minutes=5),
    ) -> None:
        """Register the model activities for ``models`` and the MCP activities."""
        if not models:
            raise ValueError("LocusPlugin needs at least one model factory")
        activities: list[Callable[..., Any]] = list(
            ModelActivities(dict(models)).activities()
        )
        mcp_plugin = None
        if mcp_clients:
            try:
                from temporalio.mcp import MCPPlugin
            except ImportError as err:
                raise RuntimeError(
                    "mcp_clients requires the 'mcp' extra and MCP Python SDK v2; "
                    "install temporalio-locus[mcp]"
                ) from err
            mcp_plugin = MCPPlugin(
                {
                    server: _no_argument_factory(server, factory)
                    for server, factory in mcp_clients.items()
                },
                connection_idle_timeout=mcp_connection_idle_timeout,
            )
            # MCPPlugin passes its activities as a list.
            activities.extend(cast(Sequence[Callable[..., Any]], mcp_plugin.activities))
        super().__init__(
            "oracle.LocusPlugin",
            data_converter=_data_converter,
            activities=activities,
            workflow_runner=_workflow_runner,
            run_context=mcp_plugin.run_context if mcp_plugin else None,
        )

    def version(self) -> str:
        """Return this package's version, which worker heartbeats report."""
        try:
            return importlib.metadata.version("temporalio-locus")
        except importlib.metadata.PackageNotFoundError:
            return ""


def _no_argument_factory(
    server: str, factory: Callable[[], MCPClient]
) -> Callable[[], MCPClient]:
    # MCPPlugin passes a factory argument to any factory with a positional
    # parameter, even a defaulted one such as ``lambda params=params: ...``,
    # so the wrapper takes none.
    if not callable(factory):
        raise TypeError(f"MCP server factory {server!r} is not callable")  # type: ignore[reportUnreachable]

    def create() -> MCPClient:
        return factory()

    return create


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
                # Imported by ``TemporalMCPClient`` when the ``mcp`` extra is
                # installed; naming a module that is absent is harmless.
                "mcp",
                "mcp_types",
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
