from __future__ import annotations

from collections.abc import Callable, Collection, Sequence
from dataclasses import replace
from datetime import timedelta
from typing import TYPE_CHECKING, cast

from botocore.config import Config as BotocoreConfig
from strands.models import BedrockModel, Model
from strands.models.bedrock import DEFAULT_READ_TIMEOUT
from strands.types.exceptions import ContextWindowOverflowException

from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.converter import DataConverter, DefaultPayloadConverter
from temporalio.exceptions import ApplicationError
from temporalio.plugin import SimplePlugin
from temporalio.worker import WorkflowRunner
from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner

from ._failure_converter import StrandsFailureConverter
from ._model_activity import ModelActivity
from ._sandbox_activity import SandboxActivities, SandboxFactory
from ._worker_env_ref import AllowAllWorkerEnvVars

if TYPE_CHECKING:
    from mcp import Client


def _default_bedrock_model() -> Model:
    # Temporal owns retries so Botocore must not retry within an activity attempt.
    return BedrockModel(
        boto_client_config=BotocoreConfig(
            read_timeout=DEFAULT_READ_TIMEOUT,
            retries={"max_attempts": 0},
        )
    )


class StrandsPlugin(SimplePlugin):
    """Temporal Worker plugin for the Strands Agents SDK.

    When ``models`` is supplied, registers a single pair of model invocation
    activities; each call carries the chosen ``model_name`` in its input and
    the worker resolves it against the factories. Factories are called lazily
    on first use, then cached for the worker's lifetime. Use the same name in
    ``TemporalAgent(model=...)`` inside the workflow.

    When ``mcp_clients`` is supplied, registers the activities of a
    ``temporalio.mcp.MCPPlugin`` (``temporalio.mcp.{server}.list-tools``,
    ``temporalio.mcp.{server}.call-tool`` and the other MCP operations) for each
    entry. Each factory takes no arguments, returns an ``mcp.Client`` (MCP
    Python SDK v2) and needs the ``mcp`` extra. Workflow-side
    ``TemporalMCPClient(server="...")`` discovers tools by running
    ``list-tools``; whether it lists once per workflow or once per agent turn is
    controlled by its ``cache_tools`` option. Don't register the same server
    name through ``MCPPlugin`` or ``OpenAIAgentsPlugin`` on the same worker:
    the activity names clash.

    ``mcp_connection_idle_timeout`` controls how long a worker-process MCP
    connection is kept open between activities before it is disconnected; the
    timer resets on every reuse. Defaults to 5 minutes. ``None`` keeps
    connections until the worker shuts down, and zero closes them as soon as
    they are idle. Only servers that negotiate the modern MCP protocol share a
    connection; a server on an older protocol, such as one built with MCP
    Python SDK 1.x, gets a new connection for every activity.

    When ``sandboxes`` is supplied, registers one stable set of activities that
    dispatches each operation by sandbox name. Each factory receives the
    requesting Workflow run's context and may return a sandbox directly or
    awaitably. The factory is called for every Activity so it can find or recover
    the backing environment. Use the same name in workflow-side
    ``TemporalSandbox(name)`` instances.

    ``resolvable_worker_env_vars`` controls which worker environment variables
    sandbox command ``env`` references may resolve immediately before execution.
    """

    def __init__(
        self,
        *,
        models: dict[str, Callable[[], Model]] | None = None,
        mcp_clients: dict[str, Callable[[], Client]] | None = None,
        sandboxes: dict[str, SandboxFactory] | None = None,
        mcp_connection_idle_timeout: timedelta | None = timedelta(minutes=5),
        resolvable_worker_env_vars: Collection[str] | AllowAllWorkerEnvVars = (),
    ) -> None:
        """Build the plugin from optional model, MCP, and sandbox factories.

        If ``models`` is omitted, registers a single ``BedrockModel`` factory
        under the name ``"bedrock"`` with Botocore retries disabled.

        A sandbox factory may return an async context manager when its adapter
        needs cleanup after the Activity finishes.
        """
        default_name: str | None = None
        if models is None:
            models = {"bedrock": _default_bedrock_model}
            default_name = "bedrock"
        activities: list[Callable] = []
        if models:
            ma = ModelActivity(models, default_name=default_name)
            activities.extend(
                [
                    ma.invoke_model,
                    ma.invoke_model_streaming,
                    ma.invoke_stateful_model,
                    ma.invoke_stateful_model_streaming,
                    ma.get_model_config,
                ]
            )

        sandbox_activities = (
            SandboxActivities(
                sandboxes,
                resolvable_worker_env_vars,
            )
            if sandboxes
            else None
        )
        if sandbox_activities is not None:
            activities.extend(sandbox_activities.activities())

        mcp_plugin = None
        if mcp_clients:
            try:
                from temporalio.mcp import MCPPlugin
            except ImportError as err:
                raise RuntimeError(
                    "mcp_clients requires the 'mcp' extra and MCP Python SDK v2; "
                    "install temporalio-strands-agents[mcp]"
                ) from err

            mcp_plugin = MCPPlugin(
                {
                    server: _client_factory(server, factory)
                    for server, factory in mcp_clients.items()
                },
                connection_idle_timeout=mcp_connection_idle_timeout,
            )
            # MCPPlugin passes its activities as a list.
            activities.extend(cast(Sequence[Callable], mcp_plugin.activities))

        super().__init__(
            "aws.StrandsPlugin",
            workflow_runner=_workflow_runner,
            data_converter=_data_converter,
            activities=activities or None,
            # Strands re-raises this when its conversation manager cannot reduce the
            # context any further; fail the workflow instead of retrying the task forever.
            workflow_failure_exception_types=[ContextWindowOverflowException],
            run_context=mcp_plugin.run_context if mcp_plugin else None,
        )


def _client_factory(server: str, factory: Callable[[], Client]) -> Callable[[], Client]:
    # MCPPlugin passes a factory argument (None) to any factory with a positional
    # parameter, even a defaulted one such as ``lambda params=params: ...``, so the
    # wrapper takes none.
    import mcp

    def create() -> Client:
        client = factory()
        if not isinstance(client, mcp.Client):
            # Retryable, so a run waits for the worker to be fixed instead of failing.
            kind = type(client)
            raise ApplicationError(
                f"MCP server factory {server!r} returned "
                f"{kind.__module__}.{kind.__qualname__}, not mcp.Client; "
                "mcp_clients factories must return an mcp.Client "
                "(see the upgrade notes in the plugin README)"
            )
        return client

    return create


def _workflow_runner(runner: WorkflowRunner | None) -> WorkflowRunner:
    if not runner:
        raise ValueError("No WorkflowRunner provided to the Strands plugin.")
    if isinstance(runner, SandboxedWorkflowRunner):
        return replace(
            runner,
            restrictions=runner.restrictions.with_passthrough_modules(
                "strands",
                "strands_tools",
                "mcp",
                "mcp_types",
                # ``pydantic`` is already in the SDK default passthrough; extend it
                # to its compiled validation core and ``Annotated`` helper.
                "pydantic_core",
                "annotated_types",
            ),
        )
    return runner


def _data_converter(converter: DataConverter | None) -> DataConverter:
    if converter is None:
        converter = pydantic_data_converter
    elif converter.payload_converter_class is not DefaultPayloadConverter:
        return converter
    return replace(
        converter,
        payload_converter_class=pydantic_data_converter.payload_converter_class,
        failure_converter_class=StrandsFailureConverter,
    )
