"""Integration tests for OpenAI Agents streaming support.

Streaming is opt-in via ``Runner.run_streamed``. Events flow back to the
workflow through ``RunResultStreaming.stream_events()`` (in batch after
each model activity completes) and to external consumers in real time
via the configured stream topic.
"""

import asyncio
import base64
import json
import logging
import uuid
import weakref
from collections.abc import AsyncIterator
from datetime import timedelta
from typing import Any

import openai._models
import pydantic
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
    Runner,
    Tool,
    TResponseInputItem,
    Usage,
)
from agents.exceptions import ModelBehaviorError
from agents.items import TResponseStreamEvent
from agents.models import get_default_model_settings
from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
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

from temporalio import workflow
from temporalio.api.common.v1 import Payload
from temporalio.api.enums.v1 import EventType
from temporalio.client import Client, WorkflowFailureError
from temporalio.contrib.workflow_streams import WorkflowStream, WorkflowStreamClient
from temporalio.converter import (
    ActivitySerializationContext,
    DataConverter,
    EncodingPayloadConverter,
    SerializationContext,
    WithSerializationContext,
)
from temporalio.openai_agents import (
    ModelActivityParameters,
    OpenAIAgentsPlugin,
    OpenAIPayloadConverter,
)
from temporalio.openai_agents.testing import AgentEnvironment
from temporalio.worker import Replayer
from tests.helpers import new_worker

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


class DriftedStreamingTestModel(StreamingTestModel):
    """Streams a final event that fails the openai SDK's strict validation.

    Its usage has no ``cache_write_tokens``, which the SDK requires, as when the
    API stops sending a field.
    """

    __test__ = False

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
            if isinstance(event, ResponseCompletedEvent) and event.response.usage:
                event.response.usage.input_tokens_details = (
                    InputTokensDetails.model_construct(cached_tokens=0)
                )
            yield event


def empty_openai_discriminator_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start from the openai discriminator cache of a fresh process.

    The openai SDK fills this cache when it parses an event that fails strict
    validation, so a Worker process that did not run the model Activity has it
    empty.
    """
    monkeypatch.setattr(
        openai._models, "DISCRIMINATOR_CACHE", weakref.WeakKeyDictionary()
    )


class Base64TextJSONConverter(EncodingPayloadConverter, WithSerializationContext):
    """JSON with ``text`` and ``delta`` strings base64-encoded on the wire.

    It records the context it decoded each streamed result in.
    """

    contexts: list[SerializationContext | None] = []

    def __init__(self, context: SerializationContext | None = None) -> None:
        self._json = OpenAIPayloadConverter().converters[b"json/plain"]
        self._context = context

    @property
    def encoding(self) -> str:
        return "json/x-base64-text"

    def with_context(self, context: SerializationContext) -> "Base64TextJSONConverter":
        return Base64TextJSONConverter(context)

    @staticmethod
    def _map_text(value: Any, convert: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: convert(item)
                if key in ("text", "delta") and isinstance(item, str)
                else Base64TextJSONConverter._map_text(item, convert)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [Base64TextJSONConverter._map_text(item, convert) for item in value]
        return value

    def to_payload(self, value: Any) -> Payload | None:
        payload = self._json.to_payload(value)
        if payload is None:
            return None
        data = self._map_text(
            json.loads(payload.data), lambda s: base64.b64encode(s.encode()).decode()
        )
        return Payload(
            metadata={"encoding": self.encoding.encode()},
            data=json.dumps(data).encode(),
        )

    def from_payload(self, payload: Payload, type_hint: type | None = None) -> Any:
        if b"response.completed" in payload.data:
            Base64TextJSONConverter.contexts.append(self._context)
        data = self._map_text(
            json.loads(payload.data), lambda s: base64.b64decode(s).decode()
        )
        return self._json.from_payload(
            Payload(
                metadata={"encoding": b"json/plain"}, data=json.dumps(data).encode()
            ),
            type_hint,
        )


class Base64TextPayloadConverter(OpenAIPayloadConverter):
    """An accepted OpenAIPayloadConverter with its own JSON wire form."""

    def __init__(self) -> None:
        super().__init__()
        self._set_converters(
            *(
                Base64TextJSONConverter() if c.encoding == "json/plain" else c
                for c in self.converters.values()
            )
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
async def test_streaming_drifted_event_in_fresh_process(
    client: Client, monkeypatch: pytest.MonkeyPatch
):
    """A final event that fails strict validation still decodes as itself.

    The Worker process starts with an empty openai discriminator cache, as one
    that did not run the model Activity, and the history then replays in
    another such process.
    """
    with pytest.raises(pydantic.ValidationError):
        InputTokensDetails.model_validate({"cached_tokens": 0})
    empty_openai_discriminator_cache(monkeypatch)
    async with AgentEnvironment(
        model=DriftedStreamingTestModel(),
        model_params=ModelActivityParameters(
            start_to_close_timeout=timedelta(seconds=30),
            streaming_topic="events",
        ),
    ) as env:
        client = env.applied_on_client(client)
        async with new_worker(
            client,
            StreamingOpenAIWorkflow,
            max_cached_workflows=0,
            workflow_failure_exception_types=[ModelBehaviorError],
        ) as worker:
            handle = await client.start_workflow(
                StreamingOpenAIWorkflow.run,
                "Hello",
                id=f"openai-streaming-drifted-{uuid.uuid4()}",
                task_queue=worker.task_queue,
                execution_timeout=timedelta(seconds=30),
            )
            assert await handle.result() == "Hello world!"
            history = await handle.fetch_history()

    empty_openai_discriminator_cache(monkeypatch)
    await Replayer(
        workflows=[StreamingOpenAIWorkflow],
        plugins=[
            OpenAIAgentsPlugin(
                model_params=ModelActivityParameters(streaming_topic="events")
            )
        ],
    ).replay_workflow(history)


@pytest.mark.asyncio
async def test_streaming_uses_configured_payload_converter(client: Client):
    """Streamed results decode with an accepted OpenAIPayloadConverter subclass.

    Its wire form differs from plain JSON, and it gets the Activity's
    serialization context, as before the fix.
    """
    Base64TextJSONConverter.contexts.clear()
    config = client.config()
    config["data_converter"] = DataConverter(
        payload_converter_class=Base64TextPayloadConverter
    )
    client = Client(**config)
    async with AgentEnvironment(
        model=StreamingTestModel(),
        model_params=ModelActivityParameters(
            start_to_close_timeout=timedelta(seconds=30),
            streaming_topic="events",
        ),
    ) as env:
        client = env.applied_on_client(client)
        async with new_worker(
            client, StreamingOpenAIWorkflow, max_cached_workflows=0
        ) as worker:
            handle = await client.start_workflow(
                StreamingOpenAIWorkflow.run,
                "Hello",
                id=f"openai-streaming-converter-{uuid.uuid4()}",
                task_queue=worker.task_queue,
                execution_timeout=timedelta(seconds=30),
            )
            assert await handle.result() == "Hello world!"
            history = await handle.fetch_history()

    (completed,) = (
        e
        for e in history.events
        if e.event_type == EventType.EVENT_TYPE_ACTIVITY_TASK_COMPLETED
    )
    (result,) = completed.activity_task_completed_event_attributes.result.payloads
    assert result.metadata["encoding"] == b"json/x-base64-text"
    assert b"Hello world!" not in result.data
    assert any(
        isinstance(c, ActivitySerializationContext)
        and c.activity_type == "invoke_model_activity_streaming"
        for c in Base64TextJSONConverter.contexts
    ), Base64TextJSONConverter.contexts


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
