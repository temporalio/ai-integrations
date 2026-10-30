import asyncio
import importlib
from collections.abc import AsyncIterable
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any
from uuid import uuid4

import pytest
from strands import Agent, tool
from strands.agent.conversation_manager import SlidingWindowConversationManager
from strands.multiagent import Swarm
from strands.types.content import Messages
from strands.types.streaming import StreamEvent

from temporalio import workflow
from temporalio.client import Client, WorkflowFailureError
from temporalio.common import RetryPolicy
from temporalio.contrib.workflow_streams import WorkflowStream
from temporalio.exceptions import ApplicationError
from temporalio.strands_agents import StrandsPlugin, TemporalAgent
from temporalio.strands_agents._model_activity import ModelActivity
from temporalio.strands_agents._sync_calls import _MESSAGE
from temporalio.worker import Replayer, Worker
from temporalio.worker.workflow_sandbox import RestrictedWorkflowAccessError
from temporalio.workflow import NondeterminismError
from tests.common import get_activities
from tests.mock_model import MockModel

with workflow.unsafe.imports_passed_through():
    from tests import prebound_run_async


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


@tool
def echo_tool(text: str) -> str:
    """Return the text."""
    return text


@workflow.defn
class SyncCallWorkflow:
    @workflow.run
    async def run(self, call: str) -> str:
        # Plain Strands objects with their own model; building them must work.
        agent = Agent(model=MockModel([]), tools=[echo_tool], callback_handler=None)
        swarm = Swarm([agent])
        if call == "agent":
            agent("Hello")
        elif call == "tool":
            agent.tool.echo_tool(text="Hello")
        elif call == "swarm":
            swarm("Hello")
        elif call == "prebound":
            # run_async bound by a module imported before the guard was installed.
            prebound_run_async.run_async(lambda: agent.invoke_async("Hello"))
        elif call == "unrestricted":
            # The SDK's escape hatch: the call runs on Strands' thread, unrecorded.
            with workflow.unsafe.sandbox_unrestricted():
                answer = Agent(model=MockModel(["answer"]), callback_handler=None)
                return str(answer("Hello")).strip()
        return "built"


@pytest.mark.parametrize("call", ["agent", "tool", "swarm", "prebound"])
async def test_sync_call_is_refused(client: Client, call: str):
    """The sync form would run on a thread outside the sandbox, unrecorded."""
    task_queue = f"test_sync_call_is_refused-{uuid4()}"
    plugin = StrandsPlugin(models={"mock": lambda: MockModel([])})

    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[SyncCallWorkflow],
        plugins=[plugin],
        workflow_failure_exception_types=[NondeterminismError],
    ):
        handle = await client.start_workflow(
            SyncCallWorkflow.run,
            call,
            id=f"test_sync_call_is_refused_{uuid4()}",
            task_queue=task_queue,
        )
        with pytest.raises(WorkflowFailureError) as err:
            await handle.result()
    cause = err.value.cause
    assert isinstance(cause, ApplicationError)
    assert cause.type == RestrictedWorkflowAccessError.__name__
    assert cause.message == _MESSAGE

    if call == "agent":  # One replay is enough: every case hits the same guard.
        await Replayer(
            workflows=[SyncCallWorkflow],
            plugins=[plugin],
            workflow_failure_exception_types=[NondeterminismError],
        ).replay_workflow(await handle.fetch_history())


async def test_sync_call_workflow_builds_without_calls(client: Client):
    """Building agents and swarms in Workflow code still works, and so does a sync
    call the Workflow makes inside ``workflow.unsafe.sandbox_unrestricted()``."""
    task_queue = f"test_sync_call_workflow_builds_without_calls-{uuid4()}"
    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[SyncCallWorkflow],
        plugins=[StrandsPlugin(models={"mock": lambda: MockModel([])})],
    ):
        result = await client.execute_workflow(
            SyncCallWorkflow.run,
            "none",
            id=f"test_sync_call_workflow_builds_without_calls_{uuid4()}",
            task_queue=task_queue,
        )
        unrestricted = await client.execute_workflow(
            SyncCallWorkflow.run,
            "unrestricted",
            id=f"test_sync_call_workflow_builds_without_calls_{uuid4()}",
            task_queue=task_queue,
        )
    assert result == "built"
    assert unrestricted == "answer"


async def test_sync_call_in_sandbox_import_is_refused(client: Client):
    """The sandbox imports Workflow modules before the Workflow runtime is attached."""
    from tests import sync_import_workflow

    # Building the Worker installs the guard before it validates the Workflow.
    with pytest.raises(RuntimeError, match="Failed validating workflow") as err:
        Worker(
            client,
            task_queue=f"test_sync_call_in_sandbox_import_is_refused-{uuid4()}",
            workflows=[sync_import_workflow.SyncImportWorkflow],
            plugins=[StrandsPlugin(models={"mock": lambda: MockModel([])})],
        )
    assert isinstance(err.value.__cause__, RestrictedWorkflowAccessError)
    assert str(err.value.__cause__) == _MESSAGE

    # Outside Workflow code the guard lets the sync call run.
    importlib.reload(sync_import_workflow)
    assert sync_import_workflow.import_result == "import answer"


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


@workflow.defn
class CompressionWorkflow:
    @workflow.run
    async def run(self, prompts: list[str]) -> None:
        agent = TemporalAgent(
            model="usage",
            start_to_close_timeout=timedelta(seconds=15),
            conversation_manager=SlidingWindowConversationManager(
                proactive_compression=True
            ),
        )
        for prompt in prompts:
            await agent.invoke_async(prompt)


async def test_proactive_compression_uses_model_context_window(client: Client):
    """Compression drops the first turn before the second model call, as it does natively."""
    task_queue = f"test_proactive_compression-{uuid4()}"
    model = UsageMockModel(["first", "second"])

    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[CompressionWorkflow],
        plugins=[StrandsPlugin(models={"usage": lambda: model})],
    ):
        await client.execute_workflow(
            CompressionWorkflow.run,
            ["turn 1", "turn 2"],
            id=f"test_proactive_compression_{uuid4()}",
            task_queue=task_queue,
        )

    assert model.user_turns == [["turn 1"], ["turn 2"]]
