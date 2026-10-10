"""Resuming a serialized ``RunState`` inside a workflow.

The Agents SDK resumes a ``RunState`` with the agent objects stored in the state, which are
the caller's own agents rather than the plugin's converted copies. The plugin routes model
resolution through ``RunConfig.model_provider`` so those agents still call the model as
Temporal activities.
"""

import asyncio
import json
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import timedelta
from typing import Any

import pytest
from agents import (
    Agent,
    AgentOutputSchemaBase,
    Handoff,
    Model,
    ModelResponse,
    ModelSettings,
    ModelTracing,
    RunConfig,
    RunContextWrapper,
    RunHooks,
    Runner,
    RunResult,
    RunResultStreaming,
    RunState,
    Tool,
    TResponseInputItem,
    Usage,
    function_tool,
    handoff,
)
from agents.items import TResponseStreamEvent
from agents.run_config import CallModelData, ModelInputData
from agents.tool_context import ToolContext
from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
    ResponseUsage,
)
from openai.types.responses.response_usage import (
    InputTokensDetails,
    OutputTokensDetails,
)

from temporalio import workflow
from temporalio.client import Client
from temporalio.contrib.workflow_streams import WorkflowStream
from temporalio.openai_agents import ModelActivityParameters
from temporalio.openai_agents._model_parameters import ModelSummaryProvider
from temporalio.openai_agents._temporal_model_provider import (
    _CaptureAgentFilter,  # pyright: ignore[reportPrivateUsage]
    _TemporalModelProvider,  # pyright: ignore[reportPrivateUsage]
    install_model_provider,  # pyright: ignore[reportPrivateUsage]
)
from temporalio.openai_agents._temporal_model_stub import (
    _TemporalModelStub,  # pyright: ignore[reportPrivateUsage]
)
from temporalio.openai_agents.testing import AgentEnvironment, ResponseBuilders
from tests.helpers import new_worker


@function_tool(needs_approval=True)
async def transfer_funds(amount: int) -> str:
    """Move money between accounts."""
    return f"transferred {amount}"


class ScriptedModel(Model):
    """Model whose reply is chosen by ``reply(system_instructions, input)``.

    Records every call so tests can inspect what reached the model activity.
    """

    __test__ = False

    def __init__(
        self,
        reply: Callable[[str | None, str | list[TResponseInputItem]], ModelResponse],
    ) -> None:
        self.reply = reply
        self.seen_instructions: list[str | None] = []

    def _next(
        self, system_instructions: str | None, input: str | list[TResponseInputItem]
    ) -> ModelResponse:
        self.seen_instructions.append(system_instructions)
        return self.reply(system_instructions, input)

    async def get_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
        **kwargs: Any,
    ) -> ModelResponse:
        return self._next(system_instructions, input)

    async def stream_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
        **kwargs: Any,
    ) -> AsyncIterator[TResponseStreamEvent]:
        response = self._next(system_instructions, input)
        yield ResponseCompletedEvent(
            response=Response(
                id="resp_test",
                created_at=0,
                model="test",
                object="response",
                output=list(response.output),
                parallel_tool_calls=True,
                tool_choice="auto",
                tools=[],
                usage=ResponseUsage(
                    input_tokens=1,
                    output_tokens=1,
                    total_tokens=2,
                    input_tokens_details=InputTokensDetails.model_validate(
                        {"cached_tokens": 0, "cache_write_tokens": 0}
                    ),
                    output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
                ),
            ),
            sequence_number=0,
            type="response.completed",
        )


def _has_tool_output(
    input: str | list[TResponseInputItem], call_id: str | None = None
) -> bool:
    return isinstance(input, list) and any(
        isinstance(item, dict)
        and item.get("type") == "function_call_output"
        and (call_id is None or item.get("call_id") == call_id)
        for item in input
    )


def _banker_reply(
    _instructions: str | None, input: str | list[TResponseInputItem]
) -> ModelResponse:
    """Banker calls ``transfer_funds`` until it sees the tool output."""
    if _has_tool_output(input, "call-transfer"):
        return ResponseBuilders.output_message("Transfer complete.")
    return ResponseBuilders.tool_call(
        '{"amount": 5}', "transfer_funds", call_id="call-transfer"
    )


def _triage_reply(
    instructions: str | None, input: str | list[TResponseInputItem]
) -> ModelResponse:
    """Triage hands off to the Banker; the Banker then behaves as in ``_banker_reply``."""
    if instructions == "Hand off to the Banker.":
        return ResponseBuilders.tool_call(
            "{}", "transfer_to_banker", call_id="call-handoff"
        )
    return _banker_reply(instructions, input)


def _make_banker() -> Agent[None]:
    return Agent[None](
        name="Banker",
        instructions="Move money when asked.",
        tools=[transfer_funds],
    )


def _make_triage(use_handoff_object: bool) -> Agent[None]:
    banker = _make_banker()
    return Agent[None](
        name="Triage",
        instructions="Hand off to the Banker.",
        handoffs=[handoff(banker) if use_handoff_object else banker],
    )


class YieldingHooks(RunHooks[None]):
    """Hooks that suspend between the input filter and the model call."""

    async def on_llm_start(
        self,
        context: RunContextWrapper[None],
        agent: Agent[None],
        system_prompt: str | None,
        input_items: list[TResponseInputItem],
    ) -> None:
        await asyncio.sleep(0)


async def _run(
    agent: Agent[None],
    input: str | RunState[None],
    *,
    streamed: bool,
    run_config: RunConfig | None = None,
    hooks: RunHooks[None] | None = None,
) -> RunResult | RunResultStreaming:
    if not streamed:
        return await Runner.run(agent, input, run_config=run_config, hooks=hooks)
    result = Runner.run_streamed(agent, input, run_config=run_config, hooks=hooks)
    async for _ in result.stream_events():
        pass
    return result


async def _approve_and_resume(
    agent: Agent[None],
    state: RunState[None],
    *,
    streamed: bool,
    run_config: RunConfig | None = None,
    hooks: RunHooks[None] | None = None,
) -> list[str]:
    for interruption in state.get_interruptions():
        state.approve(interruption)
    resumed = await _run(
        agent, state, streamed=streamed, run_config=run_config, hooks=hooks
    )
    return [
        str(resumed.final_output),
        *[str(item.output) for item in resumed.new_items if hasattr(item, "output")],
    ]


@workflow.defn
class ResumeRunStateWorkflow:
    """Interrupt for approval, restore the state from its string, approve and resume."""

    @workflow.init
    def __init__(self, streamed: bool, use_handoff: str) -> None:
        self.stream = WorkflowStream()

    @workflow.run
    async def run(self, streamed: bool, use_handoff: str) -> list[str]:
        agent = (
            _make_banker()
            if use_handoff == "none"
            else _make_triage(use_handoff == "object")
        )
        result = await _run(agent, "Move 5 dollars.", streamed=streamed)
        assert len(result.interruptions) == 1
        assert result.last_agent.name == "Banker"
        state = await RunState.from_string(agent, result.to_state().to_string())
        return await _approve_and_resume(agent, state, streamed=streamed)


@workflow.defn
class ResumeAcrossContinueAsNewWorkflow:
    """Carry the serialized state through ``continue_as_new`` and resume it there."""

    @workflow.run
    async def run(self, state_string: str | None) -> list[str]:
        agent = _make_banker()
        if state_string is None:
            result = await Runner.run(agent, "Move 5 dollars.")
            workflow.continue_as_new(result.to_state().to_string())
        state = await RunState.from_string(agent, state_string)
        return [
            f"continued={workflow.info().continued_run_id is not None}",
            *await _approve_and_resume(agent, state, streamed=False),
        ]


@workflow.defn
class ResumeWithInputFilterWorkflow:
    """Resume a state with the user's own ``call_model_input_filter``."""

    def __init__(self) -> None:
        self.filter_agents: list[str] = []

    @workflow.run
    async def run(self, kind: str, resume: bool) -> list[str]:
        def tag(data: CallModelData[Any]) -> ModelInputData:
            self.filter_agents.append(data.agent.name)
            return ModelInputData(
                input=data.model_data.input,
                instructions=f"{data.model_data.instructions} [filtered]",
            )

        async def async_tag(data: CallModelData[Any]) -> ModelInputData:
            await asyncio.sleep(0)
            return tag(data)

        config = RunConfig(
            call_model_input_filter=async_tag if kind == "async" else tag
        )
        agent = _make_banker()
        result = await Runner.run(agent, "Move 5 dollars.", run_config=config)
        if not resume:
            return [*self.filter_agents]
        state = await RunState.from_string(agent, result.to_state().to_string())
        await _approve_and_resume(agent, state, streamed=False, run_config=config)
        return [*self.filter_agents]


@workflow.defn
class ConcurrentNestedResumeWorkflow:
    """Resume two saved states concurrently from tools that share the parent's run config."""

    @workflow.run
    async def run(self) -> str:
        agents = [
            Agent[None](
                name=name,
                instructions=f"You are {name}.",
                tools=[transfer_funds],
            )
            for name in ("Alpha", "Beta")
        ]
        results = await asyncio.gather(
            *[Runner.run(agent, "Move 5 dollars.") for agent in agents]
        )
        states = [
            await RunState.from_string(agent, result.to_state().to_string())
            for agent, result in zip(agents, results)
        ]

        def resume_tool(agent: Agent[None], state: RunState[None]) -> Tool:
            @function_tool(name_override=f"resume_{agent.name.lower()}")
            async def resume(ctx: ToolContext[None]) -> str:
                """Approve and resume the saved run."""
                for interruption in state.get_interruptions():
                    state.approve(interruption)
                result = await Runner.run(
                    agent, state, run_config=ctx.run_config, hooks=YieldingHooks()
                )
                return str(result.final_output)

            return resume

        orchestrator = Agent[None](
            name="Orchestrator",
            instructions="You are Orchestrator.",
            tools=[resume_tool(a, s) for a, s in zip(agents, states)],
        )
        result = await Runner.run(orchestrator, "Resume both.")
        return str(result.final_output)


@workflow.defn
class ConcurrentAgentToolsWorkflow:
    """Run two agents-as-tools concurrently within one run (fresh, not resumed)."""

    @workflow.run
    async def run(self) -> str:
        helpers = [
            Agent[None](name=name, instructions=f"You are {name}.")
            for name in ("Alpha", "Beta")
        ]
        orchestrator = Agent[None](
            name="Orchestrator",
            instructions="You are Orchestrator.",
            tools=[
                helper.as_tool(
                    tool_name=f"ask_{helper.name.lower()}",
                    tool_description=f"Ask {helper.name}",
                )
                for helper in helpers
            ],
        )
        result = await Runner.run(orchestrator, "Ask both.")
        return str(result.final_output)


def _orchestrator_reply(tool_names: list[str]) -> Callable[..., ModelResponse]:
    """Orchestrator calls all ``tool_names`` in one response, then finishes."""

    def reply(
        instructions: str | None, input: str | list[TResponseInputItem]
    ) -> ModelResponse:
        if instructions != "You are Orchestrator.":
            return _banker_reply(instructions, input)
        if _has_tool_output(input):
            return ResponseBuilders.output_message("done")
        return ModelResponse(
            output=[
                item
                for name in tool_names
                for item in ResponseBuilders.tool_call(
                    '{"input": "hi"}' if name.startswith("ask") else "{}",
                    name,
                    call_id=f"call-{name}",
                ).output
            ],
            usage=Usage(),
            response_id=None,
        )

    return reply


class AgentAndInstructions(ModelSummaryProvider):
    def provide(
        self,
        agent: Agent[Any] | None,
        instructions: str | None,
        input: str | list[TResponseInputItem],
    ) -> str:
        return f"{agent.name if agent else None}|{instructions}"


ALL_WORKFLOWS = [
    ResumeRunStateWorkflow,
    ResumeAcrossContinueAsNewWorkflow,
    ResumeWithInputFilterWorkflow,
    ConcurrentNestedResumeWorkflow,
    ConcurrentAgentToolsWorkflow,
]


async def _execute(
    client: Client,
    model: Model,
    workflow_run: Any,
    *args: object,
    summary_override: ModelSummaryProvider | None = None,
) -> tuple[Any, list[str]]:
    """Run a workflow and return its result and the activity summaries from its history."""
    async with AgentEnvironment(
        model=model,
        model_params=ModelActivityParameters(
            start_to_close_timeout=timedelta(seconds=30),
            streaming_topic="events",
            summary_override=summary_override,
        ),
    ) as env:
        client = env.applied_on_client(client)
        async with new_worker(client, *ALL_WORKFLOWS) as worker:
            handle = await client.start_workflow(
                workflow_run,
                args=args,
                id=f"resume-run-state-{uuid.uuid4()}",
                task_queue=worker.task_queue,
                execution_timeout=timedelta(seconds=30),
            )
            result = await handle.result(follow_runs=True)
            summaries = [
                json.loads(e.user_metadata.summary.data)
                async for e in handle.fetch_history_events()
                if e.HasField("activity_task_scheduled_event_attributes")
            ]
            return result, summaries


async def test_resume_run_state_from_string(client: Client):
    result, _ = await _execute(
        client,
        ScriptedModel(_banker_reply),
        ResumeRunStateWorkflow.run,
        False,
        "none",
    )
    assert result == ["Transfer complete.", "transferred 5"]


async def test_resume_run_state_from_string_streamed(client: Client):
    result, _ = await _execute(
        client,
        ScriptedModel(_banker_reply),
        ResumeRunStateWorkflow.run,
        True,
        "none",
    )
    assert result == ["Transfer complete.", "transferred 5"]


@pytest.mark.parametrize("streamed", [False, True])
@pytest.mark.parametrize("use_handoff", ["agent", "object"])
async def test_resume_run_state_after_handoff(
    client: Client, streamed: bool, use_handoff: str
):
    result, _ = await _execute(
        client,
        ScriptedModel(_triage_reply),
        ResumeRunStateWorkflow.run,
        streamed,
        use_handoff,
    )
    assert result == ["Transfer complete.", "transferred 5"]


async def test_resume_run_state_across_continue_as_new(client: Client):
    result, _ = await _execute(
        client,
        ScriptedModel(_banker_reply),
        ResumeAcrossContinueAsNewWorkflow.run,
        None,
    )
    assert result == ["continued=True", "Transfer complete.", "transferred 5"]


@pytest.mark.parametrize("kind", ["sync", "async"])
@pytest.mark.parametrize("resume", [False, True])
async def test_user_input_filter_is_forwarded(client: Client, kind: str, resume: bool):
    model = ScriptedModel(_banker_reply)
    filter_agents, _ = await _execute(
        client, model, ResumeWithInputFilterWorkflow.run, kind, resume
    )
    expected_agents = ["Banker", "Banker"] if resume else ["Banker"]
    assert filter_agents == expected_agents
    assert model.seen_instructions == ["Move money when asked. [filtered]"] * len(
        expected_agents
    )


async def test_activity_summary_names_each_concurrently_resumed_agent(client: Client):
    result, summaries = await _execute(
        client,
        ScriptedModel(_orchestrator_reply(["resume_alpha", "resume_beta"])),
        ConcurrentNestedResumeWorkflow.run,
        summary_override=AgentAndInstructions(),
    )
    assert result == "done"
    assert sorted(summaries) == sorted(
        [
            "Orchestrator|You are Orchestrator.",
            "Orchestrator|You are Orchestrator.",
            *[f"{name}|You are {name}." for name in ("Alpha", "Beta")] * 2,
        ]
    )


async def test_activity_summary_names_each_concurrent_agent_tool(client: Client):
    result, summaries = await _execute(
        client,
        ScriptedModel(_orchestrator_reply(["ask_alpha", "ask_beta"])),
        ConcurrentAgentToolsWorkflow.run,
        summary_override=AgentAndInstructions(),
    )
    assert result == "done"
    assert sorted(summaries) == sorted(
        [
            "Orchestrator|You are Orchestrator.",
            "Alpha|You are Alpha.",
            "Beta|You are Beta.",
            "Orchestrator|You are Orchestrator.",
        ]
    )


async def test_default_summary_is_agent_name_after_resume(client: Client):
    _, summaries = await _execute(
        client, ScriptedModel(_banker_reply), ResumeRunStateWorkflow.run, False, "none"
    )
    assert summaries == ["Banker", "Banker"]


def _call_data(agent: Agent[Any]) -> CallModelData[Any]:
    return CallModelData(
        model_data=ModelInputData(input=[], instructions="base"),
        agent=agent,
        context=None,
    )


async def test_capture_filter_returns_user_result_unchanged():
    agent = _make_banker()
    returned = ModelInputData(input=[], instructions="from user")
    seen: list[CallModelData[Any]] = []

    def sync_filter(data: CallModelData[Any]) -> ModelInputData:
        seen.append(data)
        return returned

    async def async_filter(_data: CallModelData[Any]) -> ModelInputData:
        return returned

    provider = _TemporalModelProvider(ModelActivityParameters())
    data = _call_data(agent)

    sync_config = provider.install(RunConfig(call_model_input_filter=sync_filter))
    assert sync_config.call_model_input_filter is not None
    assert sync_config.call_model_input_filter(data) is returned
    assert seen == [data]

    async_config = provider.install(RunConfig(call_model_input_filter=async_filter))
    assert async_config.call_model_input_filter is not None
    awaitable = async_config.call_model_input_filter(data)
    assert isinstance(awaitable, Awaitable)
    assert await awaitable is returned

    no_filter = provider.install(RunConfig())
    assert no_filter.call_model_input_filter is not None
    assert no_filter.call_model_input_filter(data) is data.model_data


async def test_capture_filter_resolves_agent_lazily_and_is_installed_once():
    agent = _make_banker()
    params = ModelActivityParameters()
    config = install_model_provider(params, RunConfig())
    provider = config.model_provider
    assert isinstance(provider, _TemporalModelProvider)
    stub = provider.get_model("gpt-4o")
    assert isinstance(stub, _TemporalModelStub)
    assert stub._current_agent is not None  # pyright: ignore[reportPrivateUsage]
    assert stub._current_agent() is None  # pyright: ignore[reportPrivateUsage]

    assert config.call_model_input_filter is not None
    config.call_model_input_filter(_call_data(agent))
    assert stub._current_agent() is agent  # pyright: ignore[reportPrivateUsage]

    # A nested run reuses the parent's provider and filter instead of wrapping them again.
    nested = install_model_provider(params, config)
    assert nested.model_provider is provider
    assert isinstance(nested.call_model_input_filter, _CaptureAgentFilter)
    assert nested.call_model_input_filter.wrapped is None
