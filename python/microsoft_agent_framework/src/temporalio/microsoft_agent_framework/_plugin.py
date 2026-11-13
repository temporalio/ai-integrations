"""The plugin users add to ``plugins=[...]``."""

from __future__ import annotations

import importlib.metadata
import inspect
import logging
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from datetime import timedelta
from types import FrameType
from typing import TYPE_CHECKING, Any, cast

import opentelemetry.metrics
import opentelemetry.trace
from agent_framework.exceptions import AgentFrameworkException

from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.converter import DataConverter, DefaultPayloadConverter
from temporalio.exceptions import ApplicationError
from temporalio.microsoft_agent_framework._activities import (
    MCP_CLIENT_FACTORY_ERROR_TYPE,
    ChatClientActivities,
    ChatClientFactory,
)
from temporalio.microsoft_agent_framework._serde import HostedMCPHeaders
from temporalio.microsoft_agent_framework._workflow_runtime import (
    install_workflow_runtime,
)
from temporalio.microsoft_agent_framework.workflow import (
    MicrosoftAgentFrameworkWorkflowError,
)
from temporalio.plugin import SimplePlugin
from temporalio.worker import ReplayerConfig, WorkerConfig, WorkflowRunner
from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner

if TYPE_CHECKING:
    from mcp import Client

logger = logging.getLogger(__name__)

# What agent_framework core imports at module load, so workflow files that
# import it share one copy with the host instead of re-importing it per run.
_PASSTHROUGH_MODULES = (
    # The framework itself: the agent, the loop, sessions and middleware.
    "agent_framework",
    # pydantic's compiled validation core; pydantic is already an SDK default.
    "pydantic_core",
    # pydantic's Annotated constraint helpers.
    "annotated_types",
    # The framework's msgspec serialization of sessions and messages.
    "msgspec",
    # pydantic's type introspection helpers.
    "typing_inspection",
    # The framework's telemetry layers import the OpenTelemetry API eagerly.
    "opentelemetry",
    # The framework's settings loader imports python-dotenv eagerly.
    "dotenv",
    # The MCP SDK types that TemporalMCPToolProvider and temporalio-mcp read.
    "mcp",
    "mcp_types",
)


class MicrosoftAgentFrameworkPlugin(SimplePlugin):
    """Temporal plugin for Microsoft Agent Framework agents.

    Add it to a ``Client`` or a ``Worker`` (not both). ``chat_clients`` maps a
    model name to a worker-side factory that builds the provider's chat
    client, for example ``{"gpt-4o": lambda: OpenAIChatClient(model="gpt-4o",
    async_client=AsyncOpenAI(max_retries=0))}``; a workflow selects it with
    ``TemporalChatClient("gpt-4o")``. Factories run on first use and are
    cached for the worker's lifetime, so credentials stay on the worker. The
    plugin registers the ``microsoft_agent_framework.invoke_model`` and
    ``microsoft_agent_framework.invoke_model_streaming`` Activities.

    ``mcp_clients`` maps a server name to a factory returning an ``mcp.Client``
    (MCP Python SDK v2; needs the ``mcp`` extra) and registers the
    ``temporalio.mcp.{server}.*`` Activities of a ``temporalio.mcp.MCPPlugin``
    for each entry; a workflow uses ``TemporalMCPToolProvider("server")``. Do
    not register the same server name through ``MCPPlugin`` on the same worker.
    ``mcp_connection_idle_timeout`` is how long an idle worker-side MCP
    connection stays open between Activities (default 5 minutes; ``None``
    keeps it until shutdown, zero closes it after each Activity).

    ``hosted_mcp_headers`` keeps the headers of hosted MCP tools (the
    provider's ``get_mcp_tool(...)`` mappings) on the worker: a mapping from
    the server's label or URL to its headers, or to a callable returning them,
    which the model Activity adds to the tool before the provider call. A
    hosted MCP tool that carries ``headers`` or ``authorization`` in workflow
    code fails the run with ``MicrosoftAgentFrameworkWorkflowError``, because
    the tool mapping is recorded in workflow history.

    The plugin also sets the Pydantic data converter unless the client has its
    own payload converter, passes the framework's modules through the workflow
    sandbox, installs workflow-aware clocks and id generators and the inline
    run of synchronous tools for the framework's agent code, and registers ``MicrosoftAgentFrameworkWorkflowError`` and
    the framework's ``AgentFrameworkException`` as workflow failure types.
    """

    def __init__(
        self,
        *,
        chat_clients: Mapping[str, ChatClientFactory] | None = None,
        mcp_clients: Mapping[str, Callable[[], Client]] | None = None,
        mcp_connection_idle_timeout: timedelta | None = timedelta(minutes=5),
        hosted_mcp_headers: Mapping[str, HostedMCPHeaders] | None = None,
    ) -> None:
        """Build the plugin from the chat client and MCP client factories.

        Raises:
            RuntimeError: If ``mcp_clients`` is given without the ``mcp`` extra.
        """
        model_activities = ChatClientActivities(
            chat_clients or {}, hosted_mcp_headers=hosted_mcp_headers
        )
        activities: list[Callable[..., Any]] = [
            model_activities.invoke_model,
            model_activities.invoke_model_streaming,
        ]
        mcp_plugin = None
        if mcp_clients:
            try:
                from temporalio.mcp import MCPPlugin
            except ImportError as err:
                raise RuntimeError(
                    "mcp_clients requires the 'mcp' extra and MCP Python SDK v2; "
                    "install temporalio-microsoft-agent-framework[mcp]"
                ) from err
            mcp_plugin = MCPPlugin(
                {
                    server: _mcp_client_factory(server, factory)
                    for server, factory in mcp_clients.items()
                },
                connection_idle_timeout=mcp_connection_idle_timeout,
            )
            activities.extend(cast(Sequence[Callable[..., Any]], mcp_plugin.activities))
        super().__init__(
            "microsoft.AgentFrameworkPlugin",
            activities=activities,
            data_converter=_data_converter,
            workflow_runner=_workflow_runner,
            # Every framework exception raised in workflow code is deterministic
            # (model and MCP errors arrive as Activity failures), so retrying the
            # workflow task cannot help; fail the workflow instead.
            workflow_failure_exception_types=[
                MicrosoftAgentFrameworkWorkflowError,
                AgentFrameworkException,
            ],
            run_context=mcp_plugin.run_context if mcp_plugin else None,
        )

    def version(self) -> str:
        """Return this package's version, which worker heartbeats report."""
        try:
            return importlib.metadata.version("temporalio-microsoft-agent-framework")
        except importlib.metadata.PackageNotFoundError:
            return ""

    def configure_worker(self, config: WorkerConfig) -> WorkerConfig:
        """See the base class; also warns when the global OpenTelemetry providers are not replay-safe."""
        _warn_if_global_otel_providers_not_replay_safe()
        return super().configure_worker(config)

    def configure_replayer(self, config: ReplayerConfig) -> ReplayerConfig:
        """See the base class; also warns when the global OpenTelemetry providers are not replay-safe."""
        _warn_if_global_otel_providers_not_replay_safe()
        return super().configure_replayer(config)


def _mcp_client_factory(
    server: str, factory: Callable[[], Client]
) -> Callable[[], Client]:
    # MCPPlugin passes a factory argument to any factory with a positional
    # parameter, even a defaulted one, so the wrapper takes none.
    import mcp

    def create() -> Client:
        # Same policy as the chat client factories: a factory that raises may
        # recover (a secret store, a token endpoint, one worker of many), so
        # the attempt fails retryably and the retry policy bounds the wait; a
        # factory that returns the wrong thing is wrong on every attempt.
        try:
            client = factory()
        except Exception as err:
            logger.error(
                "MCP client factory for %r raised %s: %s",
                server,
                type(err).__name__,
                err,
            )
            raise ApplicationError(
                f"The MCP client factory for {server!r} raised "
                f"{type(err).__name__}: {err}",
                type=MCP_CLIENT_FACTORY_ERROR_TYPE,
                non_retryable=False,
            ) from err
        if not isinstance(client, mcp.Client):
            kind = type(client)
            raise ApplicationError(
                f"The MCP client factory for {server!r} returned "
                f"{kind.__module__}.{kind.__qualname__}, not mcp.Client; "
                "mcp_clients factories must return an mcp.Client",
                type=MCP_CLIENT_FACTORY_ERROR_TYPE,
                non_retryable=True,
            )
        return client

    return create


def _workflow_runner(runner: WorkflowRunner | None) -> WorkflowRunner:
    if runner is None:
        raise ValueError("No WorkflowRunner provided to MicrosoftAgentFrameworkPlugin.")
    # Workers and Replayers both pass through here, so both get the
    # workflow-aware clocks and ids, and both fail early on an unsupported
    # framework version.
    install_workflow_runtime()
    if isinstance(runner, SandboxedWorkflowRunner):
        return replace(
            runner,
            restrictions=runner.restrictions.with_passthrough_modules(
                *_PASSTHROUGH_MODULES
            ),
        )
    return runner


def _data_converter(converter: DataConverter | None) -> DataConverter:
    if converter is None:
        return pydantic_data_converter
    if converter.payload_converter_class is DefaultPayloadConverter:
        # Keep the caller's codec and failure converter; only the payload
        # converter changes, so Pydantic models cross the boundary.
        return replace(
            converter,
            payload_converter_class=pydantic_data_converter.payload_converter_class,
        )
    return converter


def _stacklevel_outside_temporalio() -> int:
    # Attribute the warning to the nearest frame outside temporalio, such as
    # the user's Worker(...) or Replayer(...) call.
    level = 1
    own_frame: FrameType | None = inspect.currentframe()
    frame = own_frame.f_back if own_frame is not None else None
    while frame is not None:
        module = frame.f_globals.get("__name__", "")
        if module != "temporalio" and not module.startswith("temporalio."):
            return level
        frame = frame.f_back
        level += 1
    return 1


def _warn_if_global_otel_providers_not_replay_safe() -> None:
    # Agent Framework records tool metrics and agent spans through the
    # process-global OpenTelemetry providers from workflow code, so a provider
    # that is not replay-safe re-emits them on every replay. Only an
    # OpenTelemetry SDK provider used directly as the global is a sure case;
    # unset, no-op and unknown providers stay silent.
    try:
        from opentelemetry.sdk.metrics import MeterProvider as SdkMeterProvider
        from opentelemetry.sdk.trace import TracerProvider as SdkTracerProvider
    except ImportError:
        return
    stacklevel = _stacklevel_outside_temporalio()
    if isinstance(opentelemetry.metrics.get_meter_provider(), SdkMeterProvider):
        warnings.warn(
            "The global OpenTelemetry MeterProvider is not replay-safe: Agent Framework "
            "records tool metrics from workflow code, so every workflow replay will "
            "re-record them. Wrap your provider in "
            "temporalio.contrib.opentelemetry.ReplaySafeMeterProvider and make it the "
            "first and only global provider set: "
            "opentelemetry.metrics.set_meter_provider(ReplaySafeMeterProvider(provider))",
            UserWarning,
            stacklevel=stacklevel,
        )
    if isinstance(opentelemetry.trace.get_tracer_provider(), SdkTracerProvider):
        warnings.warn(
            "The global OpenTelemetry TracerProvider is not replay-safe: Agent Framework "
            "creates agent spans from workflow code, so every workflow replay will "
            "re-emit them. Install a replay-safe provider: "
            "opentelemetry.trace.set_tracer_provider("
            "temporalio.contrib.opentelemetry.create_tracer_provider())",
            UserWarning,
            stacklevel=stacklevel,
        )
