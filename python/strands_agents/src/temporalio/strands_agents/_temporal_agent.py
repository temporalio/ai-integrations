import asyncio
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from strands import Agent
from strands.agent._concurrency import _ConcurrencyController, _InflightInvocation
from strands.agent.conversation_manager import NullConversationManager
from strands.hooks import BeforeModelCallEvent, HookCallback
from strands.types.agent import AgentInput
from strands.types.content import Messages
from strands.types.tools import AgentTool

from temporalio import workflow
from temporalio.common import Priority, RetryPolicy
from temporalio.workflow import ActivityCancellationType, VersioningIntent

from ._temporal_mcp_client import TemporalMCPClient
from ._temporal_model import TemporalModel
from ._temporal_sandbox import TemporalSandbox


@dataclass
class _MCPListing:
    """A provider's listing for this turn.

    The listing is kept here: another agent sharing the provider may list it
    again before this agent checks it.
    """

    provider: TemporalMCPClient
    new: dict[str, AgentTool]


_SNAPSHOT_DISABLED = (
    "TemporalAgent disables take_snapshot()/load_snapshot(). Temporal "
    "workflows already persist agent state durably via the event history at "
    "a finer granularity than Strands snapshots. Remove the snapshot call "
    "and rely on Temporal's durable execution instead."
)


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
        # The tools each provider holds in this agent's registry: what
        # load_tools() returned at construction, then each accepted listing.
        # Kept per agent, because agents that share a provider each refresh it.
        self._mcp_owned: dict[TemporalMCPClient, dict[str, AgentTool]] = {}
        # Providers whose last listing this agent refused or left unchecked.
        # This agent lists them again on its next turn, even with
        # cache_tools=True. The provider itself is not reset, so another agent
        # sharing it keeps the listing it accepted.
        self._mcp_unaccepted: set[TemporalMCPClient] = set()
        # Providers listed early in this turn by an earlier provider's hook.
        self._mcp_listed_early: set[TemporalMCPClient] = set()
        for provider in self._mcp_providers:
            self.hooks.add_callback(
                BeforeModelCallEvent, self._make_mcp_refresh_hook(provider)
            )

    def _make_mcp_refresh_hook(
        self, provider: TemporalMCPClient
    ) -> HookCallback[BeforeModelCallEvent]:
        registry = self.tool_registry.registry
        self._mcp_owned.setdefault(
            provider,
            {
                tool.tool_name: tool
                for tool in provider._tools
                if registry.get(tool.tool_name) is tool
            },
        )

        async def hook(event: BeforeModelCallEvent) -> None:
            if provider in self._mcp_listed_early:
                self._mcp_listed_early.discard(provider)
                return
            listing = await self._list_mcp_tools(provider)
            if listing is None:
                return
            listings = [listing]
            # A name held by a provider that lists again later in this turn
            # may be stale. List those providers now, in the same order, and
            # check the listings together.
            later = [
                other
                for other in self._mcp_providers[
                    self._mcp_providers.index(provider) + 1 :
                ]
                if not self._cached(other)
            ]
            if self._held_by(event, listing, later):
                try:
                    for other in later:
                        listings.append(await self._fetch_mcp_tools(other))
                        self._mcp_listed_early.add(other)
                except BaseException:
                    # Nothing was checked; the next turn lists them again.
                    self._mcp_unaccepted.update(u.provider for u in listings)
                    self._mcp_listed_early.clear()
                    raise
            try:
                self._reconcile_mcp_tools(event, listings)
            except BaseException:
                # The rest of this turn's hooks do not run.
                self._mcp_listed_early.clear()
                raise

        return hook

    def _cached(self, provider: TemporalMCPClient) -> bool:
        # cache_tools=True and listed, and this agent has not refused it.
        return (
            provider._cache_tools
            and provider._fetched
            and provider not in self._mcp_unaccepted
        )

    async def _list_mcp_tools(self, provider: TemporalMCPClient) -> _MCPListing | None:
        if self._cached(provider):
            return None
        return await self._fetch_mcp_tools(provider)

    @staticmethod
    async def _fetch_mcp_tools(provider: TemporalMCPClient) -> _MCPListing:
        await provider._refresh()
        return _MCPListing(provider, {tool.tool_name: tool for tool in provider._tools})

    def _held_by(
        self,
        event: BeforeModelCallEvent,
        listing: _MCPListing,
        providers: list[TemporalMCPClient],
    ) -> bool:
        # Whether a listed name, or one that differs only by '-' or '_',
        # is held by one of these providers' tools in this agent.
        reg = event.agent.tool_registry.registry
        listed = {name.replace("-", "_") for name in listing.new}
        return any(
            reg.get(name) is tool and name.replace("-", "_") in listed
            for provider in providers
            for name, tool in self._mcp_owned[provider].items()
        )

    def _reconcile_mcp_tools(
        self, event: BeforeModelCallEvent, listings: list[_MCPListing]
    ) -> None:
        reg = event.agent.tool_registry
        news = [(listing, listing.new) for listing in listings]

        def mine(provider: TemporalMCPClient, name: str) -> bool:
            # The registry entry is still the tool this provider registered
            # here; another tool may hold the same name.
            owned = self._mcp_owned[provider]
            return name in owned and reg.registry.get(name) is owned[name]

        def refuse() -> None:
            # This agent lists them again next turn instead of treating them
            # as accepted (cache_tools=True would skip them).
            self._mcp_unaccepted.update(listing.provider for listing in listings)

        # Tools the servers dropped or renamed since their last listing. There
        # is no public unregister, so remove them from the registry directly,
        # but only while the entry is still that provider's.
        drop: set[str] = set()
        for listing, new in news:
            provider = listing.provider
            gone = {
                name
                for name in self._mcp_owned[provider]
                if name not in new and mine(provider, name)
            }
            drop |= gone
        # A name another tool holds, or that two listings share, is refused,
        # as Strands refuses it natively. Check before changing the registry.
        taken: list[str] = []
        seen: set[str] = set()
        for listing, new in news:
            for name in new:
                if name in seen or (
                    name in reg.registry
                    and name not in drop
                    and not mine(listing.provider, name)
                ):
                    taken.append(name)
                seen.add(name)
        if taken:
            refuse()
            raise ValueError(
                f"Tool name '{taken[0]}' already exists. Cannot register tools "
                "with exact same name. Set a different prefix on the "
                "MCPClient or drop the tool with tool_filters."
            )
        registry, dynamic_tools = dict(reg.registry), dict(reg.dynamic_tools)
        try:
            for name in drop:
                reg.registry.pop(name, None)
                reg.dynamic_tools.pop(name, None)
            # replace() swaps an existing entry in place (no hot-reload
            # guard); register_tool() adds a newly-discovered one.
            for _listing, new in news:
                for name, tool in new.items():
                    if name in reg.registry:
                        reg.replace(tool)
                    else:
                        reg.register_tool(tool)
        except ValueError:
            # Strands refused a name, e.g. one that differs from another
            # tool's only by '-' or '_'. Undo the whole listing.
            reg.registry.clear()
            reg.registry.update(registry)
            reg.dynamic_tools.clear()
            reg.dynamic_tools.update(dynamic_tools)
            refuse()
            raise
        self._accept_mcp_tools(reg.registry, listings)

    def _accept_mcp_tools(
        self, registry: dict[str, AgentTool], listings: list[_MCPListing]
    ) -> None:
        for listing in listings:
            provider = listing.provider
            self._mcp_owned[provider] = {
                name: tool
                for name, tool in listing.new.items()
                if registry.get(name) is tool
            }
            self._mcp_unaccepted.discard(provider)

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
