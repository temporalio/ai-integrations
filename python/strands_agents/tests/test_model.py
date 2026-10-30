import asyncio
from collections.abc import AsyncIterable
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any
from uuid import uuid4

import pytest
from strands.agent.conversation_manager import (
    SlidingWindowConversationManager,
    SummarizingConversationManager,
)
from strands.hooks import BeforeModelCallEvent
from strands.tools.executors import ConcurrentToolExecutor, SequentialToolExecutor
from strands.types.content import Messages
from strands.types.streaming import StreamEvent

from temporalio import workflow
from temporalio.client import Client, WorkflowFailureError
from temporalio.common import RetryPolicy
from temporalio.contrib.workflow_streams import WorkflowStream
from temporalio.exceptions import CancelledError
from temporalio.strands_agents import StrandsPlugin, TemporalAgent
from temporalio.strands_agents._model_activity import ModelActivity
from temporalio.worker import Replayer, Worker
from tests.common import (
    blocking_lookup,
    blocking_mcp_call_tool,
    blocking_tool,
    blocking_tool_started,
    get_activities,
)
from tests.mock_model import MockModel


@workflow.defn
class ModelWorkflow:
    def __init__(self) -> None:
        self.agent = TemporalAgent(
            model="mock",
            start_to_close_timeout=timedelta(seconds=15),
        )

    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await self.agent.invoke_async(prompt)
        return str(result)


async def test_model(client: Client):
    task_queue = f"test_model-{uuid4()}"
    plugin = StrandsPlugin(models={"mock": lambda: MockModel(["Done!"])})

    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[ModelWorkflow],
        plugins=[plugin],
        max_cached_workflows=0,
    ):
        handle = await client.start_workflow(
            ModelWorkflow.run,
            "Hello",
            id=f"test_model_{uuid4()}",
            task_queue=task_queue,
        )
        assert await handle.result() == "Done!\n"

    history = await handle.fetch_history()
    assert get_activities(history) == ["strands_get_model_config", "invoke_model"]

    await Replayer(
        workflows=[ModelWorkflow],
        plugins=[plugin],
    ).replay_workflow(history)


class ConfiguredMockModel(MockModel):
    def get_config(self) -> dict[str, Any]:
        return {"model_id": "mock-1", "context_window_limit": 1000, "api_key": "k"}


@workflow.defn
class ModelConfigWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> dict[str, Any]:
        agent = TemporalAgent(
            model="configured", start_to_close_timeout=timedelta(seconds=15)
        )
        # The config loads inside Strands' invocation guard, so the guard
        # rejects an overlapping first invocation, as it does natively.
        first, second = await asyncio.gather(
            agent.invoke_async(prompt),
            agent.invoke_async(prompt),
            return_exceptions=True,
        )
        completed = await asyncio.gather(
            agent.invoke_async(prompt, idempotency_token="completed"),
            agent.invoke_async(prompt, idempotency_token="completed"),
        )

        # A duplicate of a first invocation that is cancelled while the config
        # loads is aborted, not run; the next invocation reuses the read.
        other = TemporalAgent(
            model="configured", start_to_close_timeout=timedelta(seconds=15)
        )
        primary = asyncio.create_task(
            other.invoke_async("cancelled", idempotency_token="t")
        )
        duplicate = asyncio.create_task(
            other.invoke_async("duplicate", idempotency_token="t")
        )
        await asyncio.sleep(0)
        primary.cancel()
        cancelled = await asyncio.gather(primary, duplicate, return_exceptions=True)
        await other.invoke_async(prompt)
        return {
            "invocations": [str(first), type(second).__name__],
            "deduplicated": [str(result) for result in completed],
            "cancelled": [type(r).__name__ for r in cancelled],
            "prompts": [
                block["text"]
                for message in other.messages
                if message["role"] == "user"
                for block in message["content"]
                if "text" in block
            ],
            "context_window_limit": agent.model.context_window_limit,
            "config": agent.model.get_config(),
        }


async def test_model_config_reaches_workflow(client: Client):
    """Workflow code sees the worker model's id and context window limit."""
    task_queue = f"test_model_config_reaches_workflow-{uuid4()}"
    # One spare response, so a duplicate that wrongly runs fails the asserts.
    plugin = StrandsPlugin(
        models={
            "configured": lambda: ConfiguredMockModel(
                ["Done!", "Again!", "Later!", "Spare!"]
            )
        }
    )

    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[ModelConfigWorkflow],
        plugins=[plugin],
        max_cached_workflows=0,
    ):
        handle = await client.start_workflow(
            ModelConfigWorkflow.run,
            "Hello",
            id=f"test_model_config_reaches_workflow_{uuid4()}",
            task_queue=task_queue,
        )
        assert await handle.result() == {
            "invocations": ["Done!\n", "ConcurrencyException"],
            "deduplicated": ["Again!\n", "Again!\n"],
            "cancelled": ["CancelledError", "IdempotencyAbortedError"],
            "prompts": ["Hello"],
            "context_window_limit": 1000,
            "config": {"model_id": "mock-1", "context_window_limit": 1000},
        }

    history = await handle.fetch_history()
    assert get_activities(history) == [
        "strands_get_model_config",
        "invoke_model",
        "invoke_model",
        "strands_get_model_config",
        "invoke_model",
    ]

    await Replayer(
        workflows=[ModelConfigWorkflow],
        plugins=[plugin],
    ).replay_workflow(history)


class StatefulMockModel(MockModel):
    """Keeps the conversation "server-side", as OpenAIResponsesModel(stateful=True) does."""

    calls: list[dict[str, Any]] = []

    @property
    def stateful(self) -> bool:
        return True

    async def stream(self, *args: Any, **kwargs: Any) -> AsyncIterable[StreamEvent]:
        model_state = kwargs.get("model_state")
        StatefulMockModel.calls.append(
            {
                "previous_response_id": (model_state or {}).get("response_id"),
                "messages": len(args[0]),
            }
        )
        if model_state is not None:
            model_state["response_id"] = f"resp_{len(StatefulMockModel.calls)}"
        async for event in super().stream(*args, **kwargs):
            yield event


@workflow.defn
class StatefulModelWorkflow:
    def __init__(self) -> None:
        self.stream = WorkflowStream()

    @workflow.run
    async def run(self, mode: str) -> list[int] | str:
        kwargs: dict[str, Any] = {}
        if mode == "manager":
            kwargs["conversation_manager"] = SlidingWindowConversationManager()
        if mode == "streaming":
            kwargs["streaming_topic"] = "events"
        if mode == "models-queue":
            kwargs["task_queue"] = workflow.info().task_queue + "-models"
        agent = TemporalAgent(
            model="stateful",
            start_to_close_timeout=timedelta(seconds=15),
            retry_policy=RetryPolicy(
                initial_interval=timedelta(milliseconds=100), backoff_coefficient=1
            ),
            **kwargs,
        )
        messages_after_turn = []
        try:
            for turn in ("turn one", "turn two"):
                await agent.invoke_async(turn)
                messages_after_turn.append(len(agent.messages))
        except ValueError as err:
            return f"refused: {err}"
        return messages_after_turn


async def test_stateful_model_keeps_state_server_side(client: Client):
    """A stateful worker model gets its response id back and the agent clears its messages."""
    task_queue = f"test_stateful_model_keeps_state_server_side-{uuid4()}"
    plugin = StrandsPlugin(
        models={"stateful": lambda: StatefulMockModel(["one", "two", "three", "four"])}
    )

    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[StatefulModelWorkflow],
        plugins=[plugin],
        max_cached_workflows=0,
    ):
        for mode, activity_type in (
            ("plain", "strands_invoke_stateful_model"),
            ("streaming", "strands_invoke_stateful_model_streaming"),
        ):
            StatefulMockModel.calls = []
            handle = await client.start_workflow(
                StatefulModelWorkflow.run,
                mode,
                id=f"test_stateful_model_{mode}_{uuid4()}",
                task_queue=task_queue,
            )
            assert await handle.result() == [0, 0]
            assert StatefulMockModel.calls == [
                {"previous_response_id": None, "messages": 1},
                {"previous_response_id": "resp_1", "messages": 1},
            ]
            history = await handle.fetch_history()
            assert get_activities(history) == [
                "strands_get_model_config",
                activity_type,
                activity_type,
            ]
            await Replayer(
                workflows=[StatefulModelWorkflow],
                plugins=[plugin],
            ).replay_workflow(history)

        refused = await client.execute_workflow(
            StatefulModelWorkflow.run,
            "manager",
            id=f"test_stateful_model_manager_{uuid4()}",
            task_queue=task_queue,
        )
        assert refused == (
            "refused: context_manager and conversation_manager cannot be used with "
            "a stateful model. The model manages conversation state server-side."
        )


async def test_stateful_model_waits_for_a_stateful_worker(client: Client):
    """A worker without the stateful activities cannot take a stateful call and drop its state."""
    task_queue = f"test_stateful_model_waits_for_a_stateful_worker-{uuid4()}"
    StatefulMockModel.calls = []
    plugin = StrandsPlugin(
        models={"stateful": lambda: StatefulMockModel(["one", "two", "three"])}
    )
    # During a rolling upgrade, the config can come from an upgraded worker
    # while a worker without stateful support still polls the queue: it has
    # the config and the stateless model activities only.
    legacy = ModelActivity({"stateful": lambda: StatefulMockModel(["one", "two"])})

    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[StatefulModelWorkflow],
        plugins=[plugin],
        max_cached_workflows=0,
    ):
        handle = await client.start_workflow(
            StatefulModelWorkflow.run,
            "models-queue",
            id=f"test_stateful_model_waits_{uuid4()}",
            task_queue=task_queue,
        )
        async with Worker(
            client,
            task_queue=task_queue + "-models",
            activities=[
                legacy.get_model_config,
                legacy.invoke_model,
                legacy.invoke_model_streaming,
            ],
        ):
            activity_type, failure = "", ""
            for _ in range(100):
                pending = (await handle.describe()).raw_description.pending_activities
                if pending and pending[0].attempt > 2:
                    activity_type = pending[0].activity_type.name
                    failure = pending[0].last_failure.message
                    break
                await asyncio.sleep(0.1)
            assert activity_type == "strands_invoke_stateful_model"
            assert "is not registered on this worker" in failure
            assert StatefulMockModel.calls == []

        async with Worker(client, task_queue=task_queue + "-models", plugins=[plugin]):
            assert await handle.result() == [0, 0]
        assert StatefulMockModel.calls == [
            {"previous_response_id": None, "messages": 1},
            {"previous_response_id": "resp_1", "messages": 1},
        ]


@dataclass
class StatefulChatInput:
    turns: list[str]
    messages: Messages = field(default_factory=list)
    model_state: dict[str, Any] = field(default_factory=dict)


@workflow.defn
class StatefulContinueAsNewWorkflow:
    @workflow.run
    async def run(self, input: StatefulChatInput) -> int:
        agent = TemporalAgent(
            model="stateful",
            start_to_close_timeout=timedelta(seconds=15),
            messages=list(input.messages),
            model_state=input.model_state,
        )
        await agent.invoke_async(input.turns[0])
        if input.turns[1:]:
            workflow.continue_as_new(
                StatefulChatInput(
                    turns=input.turns[1:],
                    messages=agent.messages,
                    model_state=agent.model_state,
                )
            )
        return len(agent.messages)


async def test_stateful_model_state_crosses_continue_as_new(client: Client):
    """The README's Continue-As-New handoff carries the response id to the next run."""
    task_queue = f"test_stateful_model_state_crosses_continue_as_new-{uuid4()}"
    StatefulMockModel.calls = []
    plugin = StrandsPlugin(
        models={"stateful": lambda: StatefulMockModel(["one", "two"])}
    )

    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[StatefulContinueAsNewWorkflow],
        plugins=[plugin],
        max_cached_workflows=0,
    ):
        assert (
            await client.execute_workflow(
                StatefulContinueAsNewWorkflow.run,
                StatefulChatInput(turns=["turn one", "turn two"]),
                id=f"test_stateful_model_can_{uuid4()}",
                task_queue=task_queue,
            )
            == 0
        )
    assert StatefulMockModel.calls == [
        {"previous_response_id": None, "messages": 1},
        {"previous_response_id": "resp_1", "messages": 1},
    ]


class UsageMockModel(ConfiguredMockModel):
    """Records the user turns of each call; the first reply reports 800 input tokens."""

    def __init__(self, responses: list[str | dict[str, Any]]) -> None:
        super().__init__(responses)
        self.user_turns: list[list[str]] = []

    async def stream(
        self, messages: Any, *args: Any, **kwargs: Any
    ) -> AsyncIterable[StreamEvent]:  # type: ignore[override]
        self.user_turns.append(
            [m["content"][0]["text"] for m in messages if m["role"] == "user"]
        )
        async for event in super().stream(messages, *args, **kwargs):
            yield event
        if len(self.user_turns) == 1:
            # 80% of the model's 1,000-token window, over the default compression
            # threshold of 70%, but only 0.4% of Strands' 200,000-token fallback.
            yield {
                "metadata": {
                    "usage": {
                        "inputTokens": 800,
                        "outputTokens": 1,
                        "totalTokens": 801,
                    },
                    "metrics": {"latencyMs": 0},
                }
            }


def _redact_turn_1_in(messages: Messages) -> None:
    for message in messages:
        for block in message["content"]:
            if block.get("text") == "turn 1":
                block["text"] = "[redacted]"


def _redact_turn_1(event: BeforeModelCallEvent) -> None:
    """Rewrites the first prompt once a later turn follows it, as a redaction hook would."""
    _redact_turn_1_in(event.agent.messages[:-1])


@workflow.defn
class CompressionWorkflow:
    def __init__(self) -> None:
        self.agent: TemporalAgent | None = None

    @workflow.update
    def redact(self) -> None:
        """Rewrites the first prompt from outside the agent's run."""
        assert self.agent is not None
        _redact_turn_1_in(self.agent.messages)

    @workflow.run
    async def run(
        self,
        prompts: list[str],
        summarize: bool = False,
        redact: bool = False,
        summarizer_tool: str | None = None,
        executor: str = "concurrent",
    ) -> None:
        # A summarization agent with a blocking "activity" or "mcp" tool.
        summarizer = (
            TemporalAgent(
                model="summarizer",
                start_to_close_timeout=timedelta(seconds=15),
                tools=[blocking_tool(summarizer_tool)],
                tool_executor=SequentialToolExecutor()
                if executor == "sequential"
                else ConcurrentToolExecutor(),
            )
            if summarizer_tool
            else None
        )
        agent = self.agent = TemporalAgent(
            model="usage",
            start_to_close_timeout=timedelta(seconds=15),
            conversation_manager=SummarizingConversationManager(
                summary_ratio=0.5,
                preserve_recent_messages=1,
                proactive_compression=True,
                summarization_agent=summarizer,
            )
            if summarize
            else SlidingWindowConversationManager(proactive_compression=True),
        )
        if redact:
            # Ahead of the manager's proactive compression callback.
            agent.hooks.add_callback(BeforeModelCallEvent, _redact_turn_1, order=-50)
        for prompt in prompts:
            await agent.invoke_async(prompt)


@pytest.mark.parametrize(
    ("summarize", "redact", "responses", "user_turns"),
    [
        (False, False, ["first", "second"], [["turn 1"], ["turn 2"]]),
        # The summary of the first turn replaces it, from a model call of its own.
        (
            True,
            False,
            ["first", "summary", "second"],
            [
                ["turn 1"],
                ["turn 1", "Please summarize this conversation."],
                ["summary", "turn 2"],
            ],
        ),
        # A callback ahead of the manager changes the history before it is
        # summarized, as it does natively.
        (
            True,
            True,
            ["first", "summary", "second"],
            [
                ["turn 1"],
                ["[redacted]", "Please summarize this conversation."],
                ["summary", "turn 2"],
            ],
        ),
    ],
)
async def test_proactive_compression_uses_model_context_window(
    client: Client,
    summarize: bool,
    redact: bool,
    responses: list[str],
    user_turns: list[list[str]],
):
    """Compression shrinks the first turn before the second model call, as it does natively."""
    task_queue = f"test_proactive_compression-{uuid4()}"
    model = UsageMockModel(list(responses))
    plugin = StrandsPlugin(models={"usage": lambda: model})

    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[CompressionWorkflow],
        plugins=[plugin],
        max_cached_workflows=0,
    ):
        handle = await client.start_workflow(
            CompressionWorkflow.run,
            args=[["turn 1", "turn 2"], summarize, redact],
            id=f"test_proactive_compression_{uuid4()}",
            task_queue=task_queue,
        )
        await handle.result()

    assert model.user_turns == user_turns
    await Replayer(workflows=[CompressionWorkflow], plugins=[plugin]).replay_workflow(
        await handle.fetch_history()
    )


class BlockingSummaryModel(UsageMockModel):
    """Holds the summary call until ``release`` is set."""

    def __init__(self, responses: list[str | dict[str, Any]]) -> None:
        super().__init__(responses)
        self.summarizing = asyncio.Event()
        self.release = asyncio.Event()

    async def stream(
        self, messages: Any, *args: Any, **kwargs: Any
    ) -> AsyncIterable[StreamEvent]:  # type: ignore[override]
        if len(self.user_turns) == 1:
            self.summarizing.set()
            await self.release.wait()
        async for event in super().stream(messages, *args, **kwargs):
            yield event


@pytest.mark.parametrize(
    ("summarizer_tool", "executor"),
    [
        (None, "concurrent"),
        ("activity", "concurrent"),
        ("activity", "sequential"),
        ("mcp", "concurrent"),
        ("mcp", "sequential"),
    ],
)
async def test_cancel_during_summary_cancels_the_run(
    client: Client, summarizer_tool: str | None, executor: str
):
    """A cancel while the summary runs ends the run instead of failing the summary.

    The summary is blocked in the agent's model or in a summarization agent's tool.
    """
    task_queue = f"test_cancel_during_summary-{uuid4()}"
    if summarizer_tool:
        model = UsageMockModel(["first", "second"])
        summarizer_model = MockModel(
            [{"name": "lookup", "input": {"query": "q"}}, "summary"]
        )
        blocked = blocking_tool_started
        blocked.clear()
        release = asyncio.Event()
    else:
        model = BlockingSummaryModel(["first", "summary", "second"])
        summarizer_model = MockModel([])
        blocked, release = model.summarizing, model.release
    plugin = StrandsPlugin(
        models={"usage": lambda: model, "summarizer": lambda: summarizer_model}
    )

    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[CompressionWorkflow],
        activities=[blocking_lookup, blocking_mcp_call_tool],
        plugins=[plugin],
        max_cached_workflows=0,
    ):
        handle = await client.start_workflow(
            CompressionWorkflow.run,
            args=[["turn 1", "turn 2"], True, False, summarizer_tool, executor],
            id=f"test_cancel_during_summary_{uuid4()}",
            task_queue=task_queue,
            execution_timeout=timedelta(seconds=30),
        )
        await asyncio.wait_for(blocked.wait(), timeout=10)
        await handle.cancel()
        try:
            with pytest.raises(WorkflowFailureError) as exc_info:
                await asyncio.wait_for(handle.result(), timeout=10)
        finally:
            release.set()

    assert isinstance(exc_info.value.cause, CancelledError)
    # No model call for the second turn followed the cancelled summary.
    if summarizer_tool:
        assert model.user_turns == [["turn 1"]]
    else:
        assert model.user_turns == [
            ["turn 1"],
            ["turn 1", "Please summarize this conversation."],
        ]
    await Replayer(workflows=[CompressionWorkflow], plugins=[plugin]).replay_workflow(
        await handle.fetch_history()
    )


async def test_history_changed_during_summary_is_summarized(client: Client):
    """An update that changes the history while the summary runs gets a new summary."""
    task_queue = f"test_history_changed_during_summary-{uuid4()}"
    model = BlockingSummaryModel(["first", "stale summary", "summary", "second"])
    plugin = StrandsPlugin(models={"usage": lambda: model})

    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[CompressionWorkflow],
        plugins=[plugin],
        max_cached_workflows=0,
    ):
        handle = await client.start_workflow(
            CompressionWorkflow.run,
            args=[["turn 1", "turn 2"], True],
            id=f"test_history_changed_during_summary_{uuid4()}",
            task_queue=task_queue,
        )
        await asyncio.wait_for(model.summarizing.wait(), timeout=10)
        try:
            await handle.execute_update(CompressionWorkflow.redact)
        finally:
            model.release.set()
        await handle.result()

    # The stale summary is dropped; the manager gets the redacted turn's summary.
    assert model.user_turns == [
        ["turn 1"],
        ["turn 1", "Please summarize this conversation."],
        ["[redacted]", "Please summarize this conversation."],
        ["summary", "turn 2"],
    ]
    await Replayer(workflows=[CompressionWorkflow], plugins=[plugin]).replay_workflow(
        await handle.fetch_history()
    )
