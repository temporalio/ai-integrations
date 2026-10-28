import json
from collections.abc import AsyncIterable
from dataclasses import dataclass
from datetime import timedelta
from typing import Any
from uuid import uuid4

import pytest
from strands import Agent
from strands.models import Model
from strands.types.exceptions import (
    ContextWindowOverflowException,
    MaxTokensReachedException,
)
from strands.types.streaming import StreamEvent

from temporalio import workflow
from temporalio.api.enums.v1 import EventType
from temporalio.client import Client, WorkflowFailureError, WorkflowHistory
from temporalio.common import RetryPolicy
from temporalio.contrib.workflow_streams import WorkflowStream
from temporalio.exceptions import ActivityError, ApplicationError
from temporalio.strands_agents import StrandsPlugin, TemporalAgent
from temporalio.strands_agents._temporal_model import TemporalModel
from temporalio.worker import Replayer, Worker
from tests.common import get_activities
from tests.mock_model import MockModel

# The model pretends its context window only fits this many messages.
CONTEXT_WINDOW = 3

HISTORY: list[dict[str, Any]] = [
    {"role": "user", "content": [{"text": "q1"}]},
    {"role": "assistant", "content": [{"text": "a1"}]},
    {"role": "user", "content": [{"text": "q2"}]},
    {"role": "assistant", "content": [{"text": "a2"}]},
]


def _history() -> list[dict[str, Any]]:
    return [dict(message) for message in HISTORY]


def _texts(messages: Any) -> list[str]:
    return [message["content"][0]["text"] for message in messages]


def _patch_ids(history: WorkflowHistory) -> list[str]:
    ids: list[str] = []
    for event in history.events:
        if event.event_type != EventType.EVENT_TYPE_MARKER_RECORDED:
            continue
        marker = event.marker_recorded_event_attributes
        if marker.marker_name == "core_patch":
            payload = marker.details["patch-data"].payloads[0]
            ids.append(json.loads(payload.data)["id"])
    return ids


class SmallContextModel(MockModel):
    """Raises ``ContextWindowOverflowException`` when given too many messages."""

    def __init__(self, *, window: int = CONTEXT_WINDOW) -> None:
        super().__init__([])
        self._window = window
        self.seen: list[list[str]] = []

    async def stream(
        self, messages: Any, *_args: Any, **_kwargs: Any
    ) -> AsyncIterable[StreamEvent]:  # type: ignore[override]
        self.seen.append(_texts(messages))
        if len(messages) > self._window:
            raise ContextWindowOverflowException("prompt is too long")
        self._responses = ["ok"]
        async for event in super().stream(messages):
            yield event


class RaisingModel(Model):
    """Raises the configured exception on every call."""

    def __init__(self, error: Exception) -> None:
        self._error: Exception | None = error

    def update_config(self, **_model_config: Any) -> None:
        return None

    def get_config(self) -> dict[str, Any]:
        return {}

    def structured_output(self, *_args: Any, **_kwargs: Any):
        raise NotImplementedError

    async def stream(self, *_args: Any, **_kwargs: Any) -> AsyncIterable[StreamEvent]:
        if self._error is not None:
            raise self._error
        yield {"messageStart": {"role": "assistant"}}


def _client_with(client: Client, plugin: StrandsPlugin) -> Client:
    # The failure converter that tags ContextWindowOverflowException comes from the
    # plugin's data converter, which the activity worker reads from the client config.
    config = client.config()
    config["plugins"] = [*config["plugins"], plugin]
    return Client(**config)


@workflow.defn
class OverflowAgentWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        agent = TemporalAgent(
            model="small",
            messages=_history(),  # type: ignore[arg-type]
            start_to_close_timeout=timedelta(seconds=15),
        )
        return str(await agent.invoke_async(prompt))


async def test_overflow_is_recovered_by_conversation_manager(client: Client):
    # A plain Agent reduces the context and retries when the model overflows.
    control_model = SmallContextModel()
    control = await Agent(
        model=control_model,
        messages=_history(),  # type: ignore[arg-type]
        callback_handler=None,
    ).invoke_async("q3")
    assert str(control) == "ok\n"
    assert len(control_model.seen) == 2
    assert control_model.seen[0] == ["q1", "a1", "q2", "a2", "q3"]
    assert len(control_model.seen[1]) <= CONTEXT_WINDOW

    # TemporalAgent must behave the same: the overflow fails the first activity
    # attempt, Strands reduces the conversation, and the second call succeeds.
    model = SmallContextModel()
    task_queue = f"test_overflow_recovered-{uuid4()}"
    plugin = StrandsPlugin(models={"small": lambda: model})
    client = _client_with(client, plugin)

    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[OverflowAgentWorkflow],
        max_cached_workflows=0,
    ):
        handle = await client.start_workflow(
            OverflowAgentWorkflow.run,
            "q3",
            id=f"test_overflow_recovered_{uuid4()}",
            task_queue=task_queue,
        )
        assert await handle.result() == "ok\n"

    assert model.seen == control_model.seen
    history = await handle.fetch_history()
    assert get_activities(history) == ["invoke_model", "invoke_model"]
    assert [
        event
        for event in history.events
        if event.event_type == EventType.EVENT_TYPE_ACTIVITY_TASK_FAILED
    ]
    # New executions record the patch marker that gates the exception translation, so
    # that histories recorded without it keep the ActivityError behavior on replay.
    assert _patch_ids(history) == ["strands-agents.context-window-overflow-recovery"]

    await Replayer(
        workflows=[OverflowAgentWorkflow],
        plugins=[plugin],
    ).replay_workflow(history)


async def test_unrecoverable_overflow_fails_the_workflow(client: Client):
    # When the conversation manager cannot reduce the context any further, Strands
    # re-raises. That must fail the workflow, not fail the workflow task forever.
    model = SmallContextModel(window=0)
    task_queue = f"test_overflow_unrecoverable-{uuid4()}"
    client = _client_with(client, StrandsPlugin(models={"small": lambda: model}))

    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[OverflowAgentWorkflow],
        max_cached_workflows=0,
    ):
        with pytest.raises(WorkflowFailureError) as exc_info:
            await client.execute_workflow(
                OverflowAgentWorkflow.run,
                "q3",
                id=f"test_overflow_unrecoverable_{uuid4()}",
                task_queue=task_queue,
                execution_timeout=timedelta(seconds=20),
            )

    cause = exc_info.value.cause
    assert isinstance(cause, ApplicationError)
    assert cause.type == "ContextWindowOverflowException"
    assert cause.non_retryable


@dataclass
class StreamResult:
    outcome: str
    error_type: str | None = None
    message: str | None = None
    chained_from: str | None = None


@workflow.defn
class ModelStreamWorkflow:
    def __init__(self) -> None:
        self.stream = WorkflowStream()

    @workflow.run
    async def run(self, streaming: bool) -> StreamResult:
        model = TemporalModel(
            "raising",
            start_to_close_timeout=timedelta(seconds=15),
            retry_policy=RetryPolicy(maximum_attempts=1),
            streaming_topic="events" if streaming else None,
        )
        try:
            async for _ in model.stream(
                [{"role": "user", "content": [{"text": "hi"}]}]
            ):
                pass
        except ContextWindowOverflowException as error:
            return StreamResult(
                "overflow",
                type(error).__name__,
                str(error),
                type(error.__cause__).__name__,
            )
        except ActivityError as error:
            cause = error.__cause__
            return StreamResult(
                "activity_error",
                cause.type if isinstance(cause, ApplicationError) else None,
                cause.message if isinstance(cause, ApplicationError) else None,
            )
        return StreamResult("returned")


async def _stream_failure(
    client: Client, error: Exception, *, streaming: bool
) -> StreamResult:
    task_queue = f"test_model_stream_failure-{uuid4()}"
    client = _client_with(
        client, StrandsPlugin(models={"raising": lambda: RaisingModel(error)})
    )
    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[ModelStreamWorkflow],
        max_cached_workflows=0,
    ):
        return await client.execute_workflow(
            ModelStreamWorkflow.run,
            streaming,
            id=f"test_model_stream_failure_{uuid4()}",
            task_queue=task_queue,
        )


@pytest.mark.parametrize("streaming", [False, True])
async def test_stream_raises_context_window_overflow(client: Client, streaming: bool):
    result = await _stream_failure(
        client,
        ContextWindowOverflowException("prompt is too long"),
        streaming=streaming,
    )
    assert result == StreamResult(
        "overflow",
        "ContextWindowOverflowException",
        "prompt is too long",
        "ActivityError",
    )


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize(
    ("error", "expected_type"),
    [
        (RuntimeError("provider is down"), "RuntimeError"),
        (MaxTokensReachedException("max tokens"), "MaxTokensReachedException"),
    ],
)
async def test_stream_leaves_other_failures_unchanged(
    client: Client, error: Exception, expected_type: str, streaming: bool
):
    result = await _stream_failure(client, error, streaming=streaming)
    assert result.outcome == "activity_error"
    assert result.error_type == expected_type
    assert result.message is not None
