"""Integration tests for OpenAI Agents streaming support.

Streaming is opt-in via ``Runner.run_streamed``. Events flow back to the
workflow through ``RunResultStreaming.stream_events()`` (in batch after
each model activity completes) and to external consumers in real time
via the configured stream topic.
"""

import asyncio
import logging
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import pytest
from agents import (
    Agent,
    AgentHookContext,
    AgentHooks,
    AgentOutputSchemaBase,
    GuardrailFunctionOutput,
    Handoff,
    InputGuardrailTripwireTriggered,
    Model,
    ModelResponse,
    ModelSettings,
    ModelTracing,
    RunConfig,
    RunContextWrapper,
    Runner,
    Tool,
    ToolCallOutputItem,
    TResponseInputItem,
    Usage,
    function_tool,
    input_guardrail,
)
from agents.items import TResponseStreamEvent
from agents.models import get_default_model_settings
from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
    ResponseErrorEvent,
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseTextConfig,
    ResponseTextDeltaEvent,
    ResponseUsage,
)
from openai.types.responses.response_usage import (
    InputTokensDetails,
    OutputTokensDetails,
)
from openai.types.shared import Reasoning
from openai.types.shared.response_format_text import ResponseFormatText

from temporalio import activity, workflow
from temporalio.client import Client, WorkflowFailureError, WorkflowHandle
from temporalio.contrib.workflow_streams import WorkflowStream, WorkflowStreamClient
from temporalio.openai_agents import ModelActivityParameters
from temporalio.openai_agents.testing import AgentEnvironment
from temporalio.workflow import ActivityCancellationType
from tests.helpers import assert_eventually, new_worker

logger = logging.getLogger(__name__)


class StreamingTestModel(Model):
    """Test model that yields text deltas followed by a ResponseCompletedEvent."""

    __test__ = False

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
        return ModelResponse(
            output=[
                ResponseOutputMessage(
                    id="msg_test",
                    content=[
                        ResponseOutputText(
                            text="Hello world!",
                            annotations=[],
                            type="output_text",
                            logprobs=[],
                        )
                    ],
                    role="assistant",
                    status="completed",
                    type="message",
                )
            ],
            usage=Usage(),
            response_id=None,
        )

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
        # Yield text deltas
        yield ResponseTextDeltaEvent(
            content_index=0,
            delta="Hello ",
            item_id="item1",
            output_index=0,
            sequence_number=0,
            type="response.output_text.delta",
            logprobs=[],
        )
        yield ResponseTextDeltaEvent(
            content_index=0,
            delta="world!",
            item_id="item1",
            output_index=0,
            sequence_number=1,
            type="response.output_text.delta",
            logprobs=[],
        )

        # Yield the final completed event
        response = Response(
            id="resp_test",
            created_at=0,
            error=None,
            incomplete_details=None,
            instructions=None,
            metadata={},
            model="test",
            object="response",
            output=[
                ResponseOutputMessage(
                    id="msg_test",
                    content=[
                        ResponseOutputText(
                            text="Hello world!",
                            annotations=[],
                            type="output_text",
                            logprobs=[],
                        )
                    ],
                    role="assistant",
                    status="completed",
                    type="message",
                )
            ],
            parallel_tool_calls=True,
            temperature=1.0,
            tool_choice="auto",
            tools=[],
            top_p=1.0,
            status="completed",
            text=ResponseTextConfig(format=ResponseFormatText(type="text")),
            truncation="disabled",
            usage=ResponseUsage(
                input_tokens=10,
                output_tokens=5,
                total_tokens=15,
                input_tokens_details=InputTokensDetails.model_validate(
                    {"cached_tokens": 0, "cache_write_tokens": 0}
                ),
                output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
            ),
        )
        yield ResponseCompletedEvent(
            response=response, sequence_number=2, type="response.completed"
        )


@workflow.defn
class StreamingOpenAIWorkflow:
    """Test workflow that opts into streaming via ``Runner.run_streamed``.

    Workflow code consumes events from ``stream_events()`` and exposes
    the seen event types via a query so the test can verify both the
    in-workflow iteration and the external stream subscriber observe the
    same events.
    """

    @workflow.init
    def __init__(self, prompt: str) -> None:
        self.stream = WorkflowStream()
        self.workflow_event_types: list[str] = []

    @workflow.run
    async def run(self, prompt: str) -> str:
        agent = Agent[None](
            name="Assistant",
            instructions="You are a test agent.",
        )
        result = Runner.run_streamed(starting_agent=agent, input=prompt)
        async for event in result.stream_events():
            raw = getattr(event, "data", None)
            event_type = getattr(raw, "type", None)
            if event_type is not None:
                self.workflow_event_types.append(event_type)
        return result.final_output

    @workflow.query
    def get_workflow_event_types(self) -> list[str]:
        return self.workflow_event_types


@workflow.defn
class StreamingRequiresTopicWorkflow:
    """Workflow that opts into ``Runner.run_streamed`` while the model
    plugin was configured without a ``streaming_topic``.

    The stub raises before scheduling the streaming activity; this
    propagates out of ``Runner.run_streamed`` and fails the workflow.
    """

    @workflow.run
    async def run(self, prompt: str) -> str:
        agent = Agent[None](
            name="Assistant",
            instructions="You are a test agent.",
        )
        result = Runner.run_streamed(starting_agent=agent, input=prompt)
        async for _ in result.stream_events():
            pass
        return result.final_output


@pytest.mark.asyncio
async def test_streaming_publishes_raw_events(client: Client):
    """Both the workflow consumer (via stream_events) and the stream
    topic see the same native OpenAI events, in order, with no
    normalization."""
    async with AgentEnvironment(
        model=StreamingTestModel(),
        model_params=ModelActivityParameters(
            start_to_close_timeout=timedelta(seconds=30),
            streaming_topic="events",
        ),
    ) as env:
        client = env.applied_on_client(client)
        workflow_id = f"openai-streaming-test-{uuid.uuid4()}"

        async with new_worker(
            client, StreamingOpenAIWorkflow, max_cached_workflows=0
        ) as worker:
            handle = await client.start_workflow(
                StreamingOpenAIWorkflow.run,
                "Hello",
                id=workflow_id,
                task_queue=worker.task_queue,
                execution_timeout=timedelta(seconds=30),
            )

            stream = WorkflowStreamClient.create(client, workflow_id)
            published: list[TResponseStreamEvent] = []

            async def collect_events() -> None:
                async for item in stream.subscribe(
                    ["events"],
                    from_offset=0,
                    # TResponseStreamEvent is a discriminated union
                    # (Annotated[..., Discriminator]); Pydantic decodes
                    # it via TypeAdapter at runtime, but the type
                    # checkers see ``Annotated`` rather than ``type``.
                    result_type=TResponseStreamEvent,  # type: ignore[arg-type,call-overload]
                    poll_cooldown=timedelta(milliseconds=50),
                ):
                    published.append(item.data)
                    if item.data.type == "response.completed":
                        break

            collect_task = asyncio.create_task(collect_events())
            result = await handle.result()
            await asyncio.wait_for(collect_task, timeout=10.0)

            workflow_event_types = await handle.query(
                StreamingOpenAIWorkflow.get_workflow_event_types
            )

    assert result == "Hello world!"

    published_types = [e.type for e in published]
    assert published_types == [
        "response.output_text.delta",
        "response.output_text.delta",
        "response.completed",
    ], f"Unexpected published event sequence: {published_types}"

    deltas = [e.delta for e in published if e.type == "response.output_text.delta"]
    assert deltas == ["Hello ", "world!"]

    # Workflow-side iteration sees the same model events in the same order.
    assert workflow_event_types == published_types


@pytest.mark.asyncio
async def test_streaming_requires_topic(client: Client):
    """``Runner.run_streamed`` fails fast when the plugin has no topic
    configured. The error is raised in ``stream_response`` before any
    streaming activity is scheduled."""
    async with AgentEnvironment(
        model=StreamingTestModel(),
        model_params=ModelActivityParameters(
            start_to_close_timeout=timedelta(seconds=30),
            streaming_topic=None,
        ),
    ) as env:
        client = env.applied_on_client(client)
        async with new_worker(
            client, StreamingRequiresTopicWorkflow, max_cached_workflows=0
        ) as worker:
            with pytest.raises(WorkflowFailureError) as exc_info:
                await client.execute_workflow(
                    StreamingRequiresTopicWorkflow.run,
                    "Hi",
                    id=f"openai-streaming-requires-topic-{uuid.uuid4()}",
                    task_queue=worker.task_queue,
                    execution_timeout=timedelta(seconds=30),
                )

    assert "streaming_topic" in str(exc_info.value.cause)


@pytest.mark.asyncio
async def test_streaming_rejects_local_activity(client: Client):
    """``Runner.run_streamed`` fails fast when the plugin is configured
    with ``use_local_activity=True``. Local activities support neither
    heartbeats nor the workflow-stream signal channel."""
    async with AgentEnvironment(
        model=StreamingTestModel(),
        model_params=ModelActivityParameters(
            start_to_close_timeout=timedelta(seconds=30),
            streaming_topic="events",
            use_local_activity=True,
        ),
    ) as env:
        client = env.applied_on_client(client)
        async with new_worker(
            client, StreamingRequiresTopicWorkflow, max_cached_workflows=0
        ) as worker:
            with pytest.raises(WorkflowFailureError) as exc_info:
                await client.execute_workflow(
                    StreamingRequiresTopicWorkflow.run,
                    "Hi",
                    id=f"openai-streaming-rejects-local-{uuid.uuid4()}",
                    task_queue=worker.task_queue,
                    execution_timeout=timedelta(seconds=30),
                )

    assert "use_local_activity" in str(exc_info.value.cause)


class RecordingStreamingModel(StreamingTestModel):
    """Streaming test model that exposes a model name and records each call's settings."""

    def __init__(self, model: str) -> None:
        self.model = model
        self.settings: list[ModelSettings] = []

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
        self.settings.append(model_settings)
        async for event in super().stream_response(
            system_instructions,
            input,
            model_settings,
            tools,
            output_schema,
            handoffs,
            tracing,
            **kwargs,
        ):
            yield event


@workflow.defn
class StreamingRunOverrideWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        # Names no model; the run sets a reasoning effort explicitly.
        agent = Agent[None](name="Assistant", instructions="You are a test agent.")
        result = Runner.run_streamed(
            starting_agent=agent,
            input=prompt,
            run_config=RunConfig(
                model_settings=ModelSettings(reasoning=Reasoning(effort="low"))
            ),
        )
        async for _ in result.stream_events():
            pass
        return result.final_output


async def test_streamed_run_level_model_settings_survive_the_alignment(
    client: Client, monkeypatch: pytest.MonkeyPatch
):
    # Same case as the non-streamed test: implicit gpt-5 defaults merged
    # with a run-level effort of "low" equal the defaults, the provider
    # resolves gpt-5.4, and the run-level effort must survive.
    monkeypatch.setenv("OPENAI_DEFAULT_MODEL", "gpt-5")
    model = RecordingStreamingModel("gpt-5.4")
    async with AgentEnvironment(
        model=model,
        model_params=ModelActivityParameters(
            start_to_close_timeout=timedelta(seconds=30),
            streaming_topic="events",
        ),
    ) as env:
        client = env.applied_on_client(client)
        async with new_worker(
            client, StreamingRunOverrideWorkflow, max_cached_workflows=0
        ) as worker:
            await client.execute_workflow(
                StreamingRunOverrideWorkflow.run,
                "Hello",
                id=f"openai-streaming-settings-{uuid.uuid4()}",
                task_queue=worker.task_queue,
                execution_timeout=timedelta(seconds=30),
            )
    assert model.settings == [
        get_default_model_settings("gpt-5.4").resolve(
            ModelSettings(reasoning=Reasoning(effort="low"))
        )
    ]
    assert model.settings[0].reasoning is not None
    assert model.settings[0].reasoning.effort == "low"


class SyncToolStreamingModel(StreamingTestModel):
    """Streams a call to the sync ``add`` tool, then the base model's reply."""

    def __init__(self) -> None:
        self.calls = 0

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
        self.calls += 1
        async for event in super().stream_response(
            system_instructions,
            input,
            model_settings,
            tools,
            output_schema,
            handoffs,
            tracing,
            **kwargs,
        ):
            if self.calls == 1 and isinstance(event, ResponseCompletedEvent):
                tool_call = ResponseFunctionToolCall(
                    arguments='{"a":2,"b":3}',
                    call_id="call-add",
                    name="add",
                    type="function_call",
                    id="item-add",
                    status="completed",
                )
                response = event.response.model_copy(update={"output": [tool_call]})
                event = event.model_copy(update={"response": response})
            yield event


def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@workflow.defn
class StreamingSyncToolWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> list[Any]:
        agent = Agent[None](
            name="Assistant",
            instructions="You are a test agent.",
            tools=[function_tool(add)],
        )
        result = Runner.run_streamed(starting_agent=agent, input=prompt)
        async for _ in result.stream_events():
            pass
        return [
            item.output
            for item in result.new_items
            if isinstance(item, ToolCallOutputItem)
        ]


async def test_streamed_run_runs_sync_function_tools(client: Client):
    async with AgentEnvironment(
        model=SyncToolStreamingModel(),
        model_params=ModelActivityParameters(
            start_to_close_timeout=timedelta(seconds=30),
            streaming_topic="events",
        ),
    ) as env:
        client = env.applied_on_client(client)
        async with new_worker(
            client, StreamingSyncToolWorkflow, max_cached_workflows=0
        ) as worker:
            outputs = await client.execute_workflow(
                StreamingSyncToolWorkflow.run,
                "Add 2 and 3",
                id=f"openai-streaming-sync-tool-{uuid.uuid4()}",
                task_queue=worker.task_queue,
                execution_timeout=timedelta(seconds=30),
            )
    # The sync tool ran in the streamed run, so the model got its result.
    assert outputs == [5]


class HoldingStreamingModel(StreamingTestModel):
    """Streaming test model that sends its first delta, then, on the activity's first
    attempt, waits up to ``hold`` seconds (10 by default) before the rest.

    Records the attempts and whether its stream was cancelled during the wait.
    """

    def __init__(self, hold: float = 10) -> None:
        self.hold = hold
        self.cancelled = False
        self.attempts: list[int] = []

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
        events = super().stream_response(
            system_instructions,
            input,
            model_settings,
            tools,
            output_schema,
            handoffs,
            tracing,
            **kwargs,
        )
        attempt = activity.info().attempt
        self.attempts.append(attempt)
        yield await anext(events)
        if attempt == 1:
            try:
                await asyncio.sleep(self.hold)
            except asyncio.CancelledError:
                self.cancelled = True
                raise
        async for event in events:
            yield event


@input_guardrail
async def refuse_after_a_second(
    _context: RunContextWrapper[None],
    _agent: Agent,
    _input: str | list[TResponseInputItem],
) -> GuardrailFunctionOutput:
    """Runs in parallel with the model call (the default) and trips after one second."""
    await workflow.sleep(1)
    return GuardrailFunctionOutput(output_info=None, tripwire_triggered=True)


@input_guardrail
async def clean_up_slowly(
    _context: RunContextWrapper[None],
    _agent: Agent,
    _input: str | list[TResponseInputItem],
) -> GuardrailFunctionOutput:
    """A parallel guardrail that takes 12 s to clean up when the framework cancels it."""
    try:
        await workflow.sleep(60)
    except asyncio.CancelledError:
        await workflow.sleep(12)
        raise
    return GuardrailFunctionOutput(output_info=None, tripwire_triggered=False)


class DelayModelCall(AgentHooks[None]):
    """Delays the model call by two seconds, so the guardrail trips before it."""

    async def on_start(self, context: AgentHookContext[None], agent: Agent) -> None:
        await workflow.sleep(2)


@dataclass
class GuardrailRun:
    trip: bool = True
    sibling: bool = False
    delay_model: bool = False
    # Stay open until the subscriber has read the activity's last event.
    hold_open: bool = True


@workflow.defn
class StreamingInputGuardrailWorkflow:
    @workflow.init
    def __init__(self, run: GuardrailRun) -> None:
        self.stream = WorkflowStream()
        self.subscriber_done = False

    @workflow.signal
    def finish(self) -> None:
        self.subscriber_done = True

    @workflow.run
    async def run(self, run: GuardrailRun) -> str:
        guardrails = [refuse_after_a_second] if run.trip else []
        if run.sibling:
            guardrails.append(clean_up_slowly)
        agent = Agent[None](
            name="Assistant",
            instructions="You are a test agent.",
            input_guardrails=guardrails,
            hooks=DelayModelCall() if run.delay_model else None,
        )
        result = Runner.run_streamed(starting_agent=agent, input="Hello")
        try:
            async for _ in result.stream_events():
                pass
        except InputGuardrailTripwireTriggered:
            output = "refused"
        else:
            output = str(result.final_output)
        if run.hold_open:
            await workflow.wait_condition(lambda: self.subscriber_done)
        return output


async def collect_until_the_end(
    client: Client,
    workflow_id: str,
    published: list[TResponseStreamEvent] | None = None,
) -> list[TResponseStreamEvent]:
    """Read the topic into ``published`` until an ``error`` or ``response.completed`` event."""
    stream = WorkflowStreamClient.create(client, workflow_id)
    if published is None:
        published = []
    async for item in stream.subscribe(
        ["events"],
        from_offset=0,
        result_type=TResponseStreamEvent,  # type: ignore[arg-type,call-overload]
        poll_cooldown=timedelta(milliseconds=50),
    ):
        published.append(item.data)
        if item.data.type in ("error", "response.completed"):
            break
    return published


async def streaming_activity_count(handle: WorkflowHandle[Any, Any]) -> int:
    count = 0
    async for event in handle.fetch_history_events():
        if (
            event.HasField("activity_task_scheduled_event_attributes")
            and event.activity_task_scheduled_event_attributes.activity_type.name
            == "invoke_model_activity_streaming"
        ):
            count += 1
    return count


@pytest.mark.asyncio
@pytest.mark.parametrize("sibling", [False, True], ids=["alone", "sibling-cleaning-up"])
async def test_streaming_input_guardrail_trip_stops_the_topic(
    client: Client, sibling: bool
):
    """A parallel input guardrail that trips while the model is streaming cancels the
    streaming activity: topic subscribers get an error event, not the rest of the answer
    and ``response.completed``. The cancel does not wait for the other parallel
    guardrails to finish cleaning up."""
    model = HoldingStreamingModel()
    async with AgentEnvironment(
        model=model,
        model_params=ModelActivityParameters(
            start_to_close_timeout=timedelta(seconds=30),
            heartbeat_timeout=timedelta(seconds=3),
            streaming_topic="events",
        ),
    ) as env:
        client = env.applied_on_client(client)
        workflow_id = f"openai-streaming-guardrail-{uuid.uuid4()}"

        async with new_worker(
            client, StreamingInputGuardrailWorkflow, max_cached_workflows=0
        ) as worker:
            handle = await client.start_workflow(
                StreamingInputGuardrailWorkflow.run,
                GuardrailRun(sibling=sibling),
                id=workflow_id,
                task_queue=worker.task_queue,
                execution_timeout=timedelta(seconds=50),
            )
            published = await asyncio.wait_for(
                collect_until_the_end(client, workflow_id), timeout=20.0
            )
            await handle.signal(StreamingInputGuardrailWorkflow.finish)
            result = await handle.result()

    assert result == "refused"
    published_types = [e.type for e in published]
    assert "response.completed" not in published_types, published_types
    last = published[-1]
    assert isinstance(last, ResponseErrorEvent)
    assert last.code == "cancelled"
    assert model.cancelled
    assert model.attempts == [1]


@pytest.mark.asyncio
@pytest.mark.parametrize("sibling", [False, True], ids=["alone", "sibling-cleaning-up"])
async def test_streaming_input_guardrail_known_trip_skips_the_model(
    client: Client, sibling: bool
):
    """When a parallel input guardrail has already tripped as the model is called, the
    streaming activity is not scheduled, even while other guardrails are still cleaning
    up."""
    model = HoldingStreamingModel()
    async with AgentEnvironment(
        model=model,
        model_params=ModelActivityParameters(
            start_to_close_timeout=timedelta(seconds=30),
            heartbeat_timeout=timedelta(seconds=3),
            streaming_topic="events",
        ),
    ) as env:
        client = env.applied_on_client(client)
        async with new_worker(
            client, StreamingInputGuardrailWorkflow, max_cached_workflows=0
        ) as worker:
            handle = await client.start_workflow(
                StreamingInputGuardrailWorkflow.run,
                GuardrailRun(sibling=sibling, delay_model=True, hold_open=False),
                id=f"openai-streaming-known-trip-{uuid.uuid4()}",
                task_queue=worker.task_queue,
                execution_timeout=timedelta(seconds=50),
            )
            result = await handle.result()
            scheduled = await streaming_activity_count(handle)

    assert result == "refused"
    assert scheduled == 0
    assert model.attempts == []


def delay_stream_publishing(monkeypatch: pytest.MonkeyPatch, seconds: float) -> None:
    """Hold every stream publish signal until ``seconds`` after the first one is sent."""
    signal = WorkflowHandle.signal
    deadline: float | None = None

    async def delayed_signal(
        self: WorkflowHandle[Any, Any], name: Any, *args: Any, **kwargs: Any
    ) -> None:
        nonlocal deadline
        if name == "__temporal_workflow_stream_publish":
            loop = asyncio.get_running_loop()
            if deadline is None:
                deadline = loop.time() + seconds
            await asyncio.sleep(max(0.0, deadline - loop.time()))
        await signal(self, name, *args, **kwargs)

    monkeypatch.setattr(WorkflowHandle, "signal", delayed_signal)


def clean_up_first_publish_slowly(
    monkeypatch: pytest.MonkeyPatch, seconds: float
) -> None:
    """Hold the first stream publish signal until it is cancelled, then take ``seconds``
    to clean up before it raises."""
    signal = WorkflowHandle.signal
    first = True

    async def slow_cleanup_signal(
        self: WorkflowHandle[Any, Any], name: Any, *args: Any, **kwargs: Any
    ) -> None:
        nonlocal first
        if name == "__temporal_workflow_stream_publish" and first:
            first = False
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                await asyncio.sleep(seconds)
                raise
        await signal(self, name, *args, **kwargs)

    monkeypatch.setattr(WorkflowHandle, "signal", slow_cleanup_signal)


@pytest.mark.asyncio
@pytest.mark.parametrize("max_cached_workflows", [1000, 0], ids=["cached", "uncached"])
@pytest.mark.parametrize(
    "cancelled_while",
    ["streaming", "publishing", "stopping-the-flusher"],
    ids=["streaming", "publishing", "stopping-the-flusher"],
)
async def test_streaming_input_guardrail_trip_waits_for_the_error_event(
    client: Client,
    monkeypatch: pytest.MonkeyPatch,
    max_cached_workflows: int,
    cancelled_while: str,
):
    """With ``WAIT_CANCELLATION_COMPLETED``, a workflow that returns right after the trip
    still has the error event at the end of its stream, and the Worker shuts down after
    the cancel, with or without a Workflow cache. That holds whether the cancel reaches
    the activity while the model streams or after the model finished, while the activity
    still publishes its events or stops the stream's background flusher."""
    model: StreamingTestModel
    if cancelled_while == "streaming":
        model = HoldingStreamingModel()
    elif cancelled_while == "publishing":
        # The model finishes at once, but the topic gets nothing for 8 s, so the cancel
        # arrives while the activity is still publishing.
        model = StreamingTestModel()
        delay_stream_publishing(monkeypatch, 8)
    else:
        # The background flusher's first publish hangs. The model finishes after
        # 0.3 s, and stopping the flusher takes 8 s, so the cancel arrives while the
        # stream's context manager waits for the flusher.
        model = HoldingStreamingModel(hold=0.3)
        clean_up_first_publish_slowly(monkeypatch, 8)
    async with AgentEnvironment(
        model=model,
        model_params=ModelActivityParameters(
            start_to_close_timeout=timedelta(seconds=30),
            heartbeat_timeout=timedelta(seconds=3),
            streaming_topic="events",
            cancellation_type=ActivityCancellationType.WAIT_CANCELLATION_COMPLETED,
        ),
    ) as env:
        client = env.applied_on_client(client)
        workflow_id = f"openai-streaming-guardrail-wait-{uuid.uuid4()}"
        async with new_worker(
            client,
            StreamingInputGuardrailWorkflow,
            max_cached_workflows=max_cached_workflows,
        ) as worker:
            result = await client.execute_workflow(
                StreamingInputGuardrailWorkflow.run,
                GuardrailRun(hold_open=False),
                id=workflow_id,
                task_queue=worker.task_queue,
                execution_timeout=timedelta(seconds=30),
            )
            stream = WorkflowStreamClient.create(client, workflow_id)
            offset = await stream.get_offset()
            last: TResponseStreamEvent | None = None
            async for item in stream.subscribe(
                ["events"],
                from_offset=offset - 1,
                result_type=TResponseStreamEvent,  # type: ignore[arg-type,call-overload]
            ):
                last = item.data
                break

    assert result == "refused"
    assert isinstance(last, ResponseErrorEvent)
    assert last.code == "cancelled"
    if cancelled_while == "streaming":
        # The model held after its first delta and was cancelled, so the stream holds
        # that delta and the error event.
        assert isinstance(model, HoldingStreamingModel)
        assert model.cancelled
        assert offset == 2
    else:
        # Two deltas, response.completed, and the error event.
        if isinstance(model, HoldingStreamingModel):
            assert not model.cancelled
        assert offset == 4


@pytest.mark.asyncio
async def test_streaming_worker_shutdown_publishes_no_error_event(client: Client):
    """A Worker shutdown cancels the streaming activity, but its retry can still answer,
    so subscribers get no error event before the retry's events."""
    model = HoldingStreamingModel()
    async with AgentEnvironment(
        model=model,
        model_params=ModelActivityParameters(
            start_to_close_timeout=timedelta(seconds=30),
            heartbeat_timeout=timedelta(seconds=3),
            streaming_topic="events",
        ),
    ) as env:
        client = env.applied_on_client(client)
        workflow_id = f"openai-streaming-shutdown-{uuid.uuid4()}"
        task_queue = str(uuid.uuid4())

        # Only one Worker per task queue can be registered in a process at a time.
        async with new_worker(
            client,
            StreamingInputGuardrailWorkflow,
            task_queue=task_queue,
            max_cached_workflows=0,
        ):
            handle = await client.start_workflow(
                StreamingInputGuardrailWorkflow.run,
                GuardrailRun(trip=False),
                id=workflow_id,
                task_queue=task_queue,
                execution_timeout=timedelta(seconds=50),
            )
            published: list[TResponseStreamEvent] = []
            collect_task = asyncio.create_task(
                collect_until_the_end(client, workflow_id, published)
            )

            async def first_delta_read() -> None:
                assert published

            # Shut the Worker down once the subscriber has the first attempt's delta.
            await assert_eventually(first_delta_read, timeout=timedelta(seconds=20))
        async with new_worker(
            client,
            StreamingInputGuardrailWorkflow,
            task_queue=task_queue,
            max_cached_workflows=0,
        ):
            await asyncio.wait_for(collect_task, timeout=30.0)
            await handle.signal(StreamingInputGuardrailWorkflow.finish)
            result = await handle.result()

    assert result == "Hello world!"
    assert model.attempts == [1, 2]
    assert [e.type for e in published] == [
        "response.output_text.delta",
        "response.output_text.delta",
        "response.output_text.delta",
        "response.completed",
    ]
