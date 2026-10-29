import asyncio
from datetime import timedelta
from typing import Any

from strands import Agent
from strands.agent._concurrency import _ConcurrencyController, _InflightInvocation
from strands.agent.conversation_manager import NullConversationManager
from strands.hooks import BeforeModelCallEvent
from strands.types.agent import AgentInput
from strands.types.content import Messages
from strands.types.tools import AgentTool

from temporalio import workflow
from temporalio.common import Priority, RetryPolicy
from temporalio.workflow import ActivityCancellationType, VersioningIntent

from ._temporal_mcp_client import MCPToolNameConflictError, TemporalMCPClient
from ._temporal_model import TemporalModel
from ._temporal_sandbox import TemporalSandbox

_SNAPSHOT_DISABLED = (
    "TemporalAgent disables take_snapshot()/load_snapshot(). Temporal "
    "workflows already persist agent state durably via the event history at "
    "a finer granularity than Strands snapshots. Remove the snapshot call "
    "and rely on Temporal's durable execution instead."
)


def _duplicate_tool_name(
    name: str,
    other: str,
    provider: TemporalMCPClient,
    holder: TemporalMCPClient | None,
) -> MCPToolNameConflictError:
    # Strands' words when it refuses a name, then where the two tools come from.
    if other == name:
        refused = f"Tool name '{name}' already exists."
    else:
        refused = f"Tool name '{name}' already exists as '{other}'."
    if holder is None:
        source = (
            f"MCP server '{provider.server}' lists it, and another tool of the "
            "agent has it. Set a prefix on the server's MCPClient, or drop the "
            "tool with tool_filters."
        )
    elif holder is provider:
        both = "it twice" if other == name else "both names"
        source = f"MCP server '{provider.server}' lists {both}."
    elif holder.server == provider.server:
        source = (
            "The agent has two TemporalMCPClient handles for MCP server "
            f"'{provider.server}'. Pass only one."
        )
    else:
        source = (
            f"MCP servers '{holder.server}' and '{provider.server}' both list it. "
            "Set a different prefix on one server's MCPClient, or drop the tool "
            "with tool_filters."
        )
    return MCPToolNameConflictError(f"{refused} {source}")


def _check_mcp_tools(
    listings: dict[TemporalMCPClient, list[AgentTool]],
    held: dict[str, str],
    dynamic: set[str],
) -> None:
    # The rule ToolRegistry.register_tool() applies, for the turn's tools
    # against each other and the agent's other tools, before the registry
    # changes: a name may not be taken, nor differ from a taken one only by '-'
    # or '_'.
    seen: dict[str, tuple[str, TemporalMCPClient]] = {}
    for provider, tools in listings.items():
        for tool in tools:
            name = tool.tool_name
            key = name.replace("-", "_")
            if key in seen:
                other, holder = seen[key]
                raise _duplicate_tool_name(name, other, provider, holder)
            if key in held or name in dynamic:
                raise _duplicate_tool_name(name, held.get(key, name), provider, None)
            seen[key] = (name, provider)


class _WorkflowInflightInvocation(_InflightInvocation):
    def register_waiter(self) -> asyncio.Future:
        # Strands uses asyncio.wrap_future to notify duplicates across threads.
        # On Python <3.13 that calls loop methods the Workflow loop does not
        # implement. Workflow invocations share one loop, so wait on its state.
        return asyncio.create_task(workflow.wait_condition(lambda: self.settled))


class _WorkflowConcurrencyController(_ConcurrencyController):
    def _check_idempotency(
        self, idempotency_token: Any
    ) -> tuple[_InflightInvocation | None, Any]:
        waiting_on, registered_token = super()._check_idempotency(idempotency_token)
        if registered_token is not None:
            self._inflight = _WorkflowInflightInvocation()
        return waiting_on, registered_token


class TemporalAgent(Agent):
    """A Strands ``Agent`` that routes model calls through a Temporal activity.

    ``model`` is the name of a factory registered in
    ``StrandsPlugin(models={...})``. The activity options apply to every model
    invocation this agent makes. ``sandbox`` is a workflow-side
    ``TemporalSandbox`` whose name selects a worker-side factory. All other
    keyword arguments are forwarded to Strands' ``Agent`` (``tools``, ``hooks``, ``system_prompt``,
    ``structured_output_model``, ``messages``, etc.).

    Strands' ``retry_strategy`` is disabled; configure retries via
    ``retry_policy`` here and on the activity options accepted by
    ``activity_as_tool``, ``activity_as_hook``, and ``TemporalMCPClient``.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        task_queue: str | None = None,
        schedule_to_close_timeout: timedelta | None = None,
        schedule_to_start_timeout: timedelta | None = None,
        start_to_close_timeout: timedelta | None = None,
        heartbeat_timeout: timedelta | None = None,
        retry_policy: RetryPolicy | None = None,
        cancellation_type: ActivityCancellationType = ActivityCancellationType.TRY_CANCEL,
        versioning_intent: VersioningIntent | None = None,
        summary: str | None = None,
        priority: Priority = Priority.default,
        streaming_topic: str | None = None,
        streaming_batch_interval: timedelta = timedelta(milliseconds=100),
        sandbox: TemporalSandbox | None = None,
        model_state: dict[str, Any] | None = None,
        **agent_kwargs: Any,
    ) -> None:
        """Build a TemporalAgent from a registered model name and activity options."""
        if agent_kwargs.get("retry_strategy") is not None:
            raise ValueError(
                "TemporalAgent disables Strands retries; configure retries via "
                "retry_policy on TemporalAgent and on the activity options "
                "passed to workflow.activity_as_tool, workflow.activity_as_hook, "
                "or TemporalMCPClient. Remove retry_strategy from "
                "TemporalAgent(...) or pass retry_strategy=None."
            )
        agent_kwargs["retry_strategy"] = None
        self._manager_given = agent_kwargs.get(
            "conversation_manager"
        ) is not None or agent_kwargs.get("context_manager") not in (None, False)

        temporal_model = TemporalModel(
            model_name=model,
            task_queue=task_queue,
            schedule_to_close_timeout=schedule_to_close_timeout,
            schedule_to_start_timeout=schedule_to_start_timeout,
            start_to_close_timeout=start_to_close_timeout,
            heartbeat_timeout=heartbeat_timeout,
            retry_policy=retry_policy,
            cancellation_type=cancellation_type,
            versioning_intent=versioning_intent,
            summary=summary,
            priority=priority,
            streaming_topic=streaming_topic,
            streaming_batch_interval=streaming_batch_interval,
        )
        super().__init__(model=temporal_model, sandbox=sandbox, **agent_kwargs)
        self._concurrency = _WorkflowConcurrencyController(self._concurrency.mode)
        self._temporal_model = temporal_model
        # A model state carried from an earlier run, such as a stateful model's
        # last response id across Continue-As-New.
        if model_state:
            self._model_state = dict(model_state)

        # Strands invokes ToolProvider.load_tools() once at construction on a
        # separate run_async thread that has no workflow runtime, so a
        # TemporalMCPClient cannot list its tools there. Instead refresh from a
        # BeforeModelCallEvent hook, which runs on the workflow loop just before
        # the registry is read each turn. cache_tools=True lists once (guarded
        # by _fetched); cache_tools=False re-lists every turn.
        self._mcp_providers = [
            provider
            for provider in self.tool_registry._tool_providers
            if isinstance(provider, TemporalMCPClient)
        ]
        # Names this agent's MCP tools hold: those load_tools() registered at
        # construction, then each turn's. A name stays the servers' while they
        # list it: a tool put under it meanwhile is replaced at the next listing,
        # and removed with the name.
        self._mcp_names = {
            tool.tool_name
            for provider in self._mcp_providers
            for tool in provider._tools
        }
        if self._mcp_providers:
            self.hooks.add_callback(BeforeModelCallEvent, self._refresh_mcp_tools)

    async def _refresh_mcp_tools(self, event: BeforeModelCallEvent) -> None:
        # List the providers that are due in order, as one hook per provider
        # did, and check the listings so far after each one, so a refused turn
        # lists no further. Then apply the turn's tools together, so a name can
        # move from one server to another in one turn.
        reg = event.agent.tool_registry
        # The agent's other tools. Strands compares a new name with the
        # registry's names also with '-' and '_' swapped, and with dynamic tools'
        # names (such as structured output's) exactly.
        held = {
            name.replace("-", "_"): name
            for name in reg.registry
            if name not in self._mcp_names
        }
        dynamic = set(reg.dynamic_tools) - self._mcp_names
        listings: dict[TemporalMCPClient, list[AgentTool]] = {}
        for provider in self._mcp_providers:
            if not (provider._cache_tools and provider._fetched):
                await provider._refresh()
            # Read now: another agent sharing the provider may list it again
            # while this one waits for the next listing.
            listings[provider] = provider._tools
            _check_mcp_tools(listings, held, dynamic)
        new = {tool.tool_name for tools in listings.values() for tool in tools}
        # Tools the servers dropped or renamed since the last turn. There is no
        # public unregister, so remove them from the registry directly.
        for name in self._mcp_names - new:
            reg.registry.pop(name, None)
            reg.dynamic_tools.pop(name, None)
        # replace() swaps an existing tool in place (no hot-reload guard);
        # register_tool() adds a newly-discovered one.
        for tools in listings.values():
            for tool in tools:
                if tool.tool_name in reg.registry:
                    reg.replace(tool)
                else:
                    reg.register_tool(tool)
        self._mcp_names = new

    async def _convert_prompt_to_messages(self, prompt: AgentInput) -> Messages:
        """Load the worker-side model's config, then convert the prompt.

        Strands reads the model's context window limit for proactive
        compression and its model id for tracing in workflow code, before the
        first model call. ``stream_async`` awaits this method first after it
        takes the invocation lock and registers the idempotency token, and
        before it starts the agent span. Loading here keeps Strands' rejection,
        deduplication and cancellation of overlapping invocations unchanged.
        """
        await self._temporal_model._load_config()
        if self._temporal_model.stateful:
            self._use_server_side_state()
        return await super()._convert_prompt_to_messages(prompt)

    def _use_server_side_state(self) -> None:
        # What Strands' Agent does at construction for a stateful model, which
        # TemporalAgent can only learn once the worker model's config is read.
        if self._manager_given:
            raise ValueError(
                "context_manager and conversation_manager cannot be used with a "
                "stateful model. The model manages conversation state server-side."
            )
        if not isinstance(self.conversation_manager, NullConversationManager):
            self.conversation_manager = NullConversationManager()

    @property
    def model_state(self) -> dict[str, Any]:
        """A copy of the model state, such as a stateful model's last response id.

        Carry it with the messages across Continue-As-New and pass it back as
        ``TemporalAgent(model_state=...)``.
        """
        return dict(self._model_state)

    def take_snapshot(self, *_args: Any, **_kwargs: Any) -> Any:
        """Disabled; Temporal's event history is the source of truth."""
        raise NotImplementedError(_SNAPSHOT_DISABLED)

    def load_snapshot(self, *_args: Any, **_kwargs: Any) -> Any:
        """Disabled; Temporal's event history is the source of truth."""
        raise NotImplementedError(_SNAPSHOT_DISABLED)
