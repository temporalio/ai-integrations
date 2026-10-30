import copy
from collections.abc import AsyncIterable, Callable
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, cast
from uuid import uuid4

import pytest
from strands import Agent
from strands.agent.conversation_manager import SummarizingConversationManager
from strands.models import Model
from strands.types.exceptions import (
    ContextWindowOverflowException,
    MaxTokensReachedException,
)
from strands.types.streaming import StreamEvent

from temporalio import workflow
from temporalio.api.enums.v1 import EventType
from temporalio.client import Client, WorkflowFailureError
from temporalio.common import RetryPolicy
from temporalio.contrib.workflow_streams import WorkflowStream
from temporalio.exceptions import ActivityError, ApplicationError
from temporalio.strands_agents import StrandsPlugin, TemporalAgent
from temporalio.strands_agents._summary import _summarize_with_agent
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
    return [
        next((block["text"] for block in message["content"] if "text" in block), "")
        for message in messages
    ]


def _contents(messages: Any) -> list[dict[str, Any]]:
    # Without the ids and metadata that differ between runs.
    return [
        {"role": message["role"], "content": message["content"]} for message in messages
    ]


REASONING: StreamEvent = {
    "contentBlockDelta": {"delta": {"reasoningContent": {"text": "thinking"}}}
}
TEXT: StreamEvent = {"contentBlockDelta": {"delta": {"text": "ok"}}}
CITATION: StreamEvent = cast(
    Any,
    {
        "contentBlockDelta": {
            "delta": {"citation": {"title": "doc", "location": {"web": {"url": "u"}}}}
        }
    },
)
STOP: StreamEvent = {"contentBlockStop": {}}


def _reply(*blocks: list[StreamEvent]) -> list[StreamEvent]:
    """The events of a reply made of the given content blocks."""
    events: list[StreamEvent] = [{"messageStart": {"role": "assistant"}}]
    for block in blocks:
        events.extend([*block, STOP])
    events.append({"messageStop": {"stopReason": "end_turn"}})
    return events


class SmallContextModel(MockModel):
    """Raises ``ContextWindowOverflowException`` when given too many messages."""

    def __init__(
        self, *, window: int = CONTEXT_WINDOW, reply: list[StreamEvent] | None = None
    ) -> None:
        super().__init__([])
        self._window = window
        self._reply = reply
        self.seen: list[list[str]] = []
        self.requests: list[Any] = []

    async def stream(
        self, messages: Any, *_args: Any, **_kwargs: Any
    ) -> AsyncIterable[StreamEvent]:  # type: ignore[override]
        self.seen.append(_texts(messages))
        self.requests.append(_contents(copy.deepcopy(messages)))
        if len(messages) > self._window:
            raise ContextWindowOverflowException("prompt is too long")
        if self._reply is not None:
            for event in self._reply:
                yield event
            return
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


def _manager(kind: str, summarizer: Callable[[], Agent]) -> dict[str, Any]:
    """Agent arguments for the default manager or a summarizing one."""
    if kind == "sliding_window":
        return {}
    return {
        "conversation_manager": SummarizingConversationManager(
            summary_ratio=0.5,
            preserve_recent_messages=1,
            summarization_agent=summarizer()
            if kind.startswith("summarization_agent")
            else None,
        )
    }


def _redact_on_force_stop(agent: Agent, manager: str) -> Agent:
    """Rewrites the first prompt when the overflow stops the run, before Strands reduces."""
    if manager.endswith("redacted_on_force_stop"):

        def callback_handler(**kwargs: Any) -> None:
            if kwargs.get("force_stop"):
                agent.messages[0] = {
                    "role": "user",
                    "content": [{"text": "[redacted]"}],
                }

        agent.callback_handler = callback_handler
    return agent


def _client_with(client: Client, plugin: StrandsPlugin) -> Client:
    # The failure converter that tags ContextWindowOverflowException comes from the
    # plugin's data converter, which the activity worker reads from the client config.
    config = client.config()
    config["plugins"] = [*config["plugins"], plugin]
    return Client(**config)


@workflow.defn
class OverflowAgentWorkflow:
    @workflow.run
    async def run(self, prompt: str, manager: str = "sliding_window") -> str:
        def summarizer() -> Agent:
            return TemporalAgent(
                model="reasoning"
                if manager == "summarization_agent_reasoning"
                else "small",
                start_to_close_timeout=timedelta(seconds=15),
            )

        agent = TemporalAgent(
            model="small",
            messages=_history(),  # type: ignore[arg-type]
            start_to_close_timeout=timedelta(seconds=15),
            **_manager(manager, summarizer),
        )
        return str(await _redact_on_force_stop(agent, manager).invoke_async(prompt))


def _summarizer_model(manager: str) -> SmallContextModel | None:
    """A separate model for the summarization agent, which reasons before replying."""
    if manager != "summarization_agent_reasoning":
        return None
    return SmallContextModel(reply=_reply([REASONING], [TEXT]))


@pytest.mark.parametrize(
    "manager",
    [
        "sliding_window",
        "summarizing",
        "summarization_agent",
        "summarization_agent_reasoning",
        "summarizing_redacted_on_force_stop",
    ],
)
async def test_overflow_is_recovered_by_conversation_manager(
    client: Client, manager: str
):
    # A plain Agent reduces the context and retries when the model overflows. A
    # summarizing manager first asks the model, or its summarization agent, for a
    # summary, which it turns into a user message as the Strands version does. A
    # callback handler that changes the history on the overflow's force stop event
    # changes what is summarized.
    control_model = SmallContextModel()
    control_summarizer = _summarizer_model(manager) or control_model
    control_agent = Agent(
        model=control_model,
        messages=_history(),  # type: ignore[arg-type]
        callback_handler=None,
        **_manager(
            manager, lambda: Agent(model=control_summarizer, callback_handler=None)
        ),
    )
    control = await _redact_on_force_stop(control_agent, manager).invoke_async("q3")
    assert str(control) == "ok\n"
    assert len(control_model.seen) >= 2
    assert control_model.seen[0] == ["q1", "a1", "q2", "a2", "q3"]
    if manager.endswith("redacted_on_force_stop"):
        assert control_model.seen[1][0] == "[redacted]"
    assert len(control_model.seen[-1]) <= CONTEXT_WINDOW

    # TemporalAgent must behave the same: the overflow fails the first activity
    # attempt, Strands reduces the conversation, and the next calls succeed.
    model = SmallContextModel()
    summarizer = _summarizer_model(manager) or model
    task_queue = f"test_overflow_recovered-{uuid4()}"
    plugin = StrandsPlugin(
        models={"small": lambda: model, "reasoning": lambda: summarizer}
    )
    client = _client_with(client, plugin)

    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[OverflowAgentWorkflow],
        max_cached_workflows=0,
    ):
        handle = await client.start_workflow(
            OverflowAgentWorkflow.run,
            args=["q3", manager],
            id=f"test_overflow_recovered_{uuid4()}",
            task_queue=task_queue,
            execution_timeout=timedelta(seconds=20),
        )
        assert await handle.result() == "ok\n"

    assert model.requests == control_model.requests
    assert summarizer.requests == control_summarizer.requests
    history = await handle.fetch_history()
    model_calls = len(model.seen) + (
        len(summarizer.seen) if summarizer is not model else 0
    )
    assert get_activities(history).count("invoke_model") == model_calls
    assert [
        event
        for event in history.events
        if event.event_type == EventType.EVENT_TYPE_ACTIVITY_TASK_FAILED
    ]
    await Replayer(
        workflows=[OverflowAgentWorkflow],
        plugins=[plugin],
    ).replay_workflow(history)


@pytest.mark.parametrize(
    "reply",
    [
        _reply([TEXT]),
        _reply([REASONING], [TEXT]),
        _reply([TEXT, CITATION]),
        _reply([REASONING]),
    ],
    ids=["text", "reasoning", "cited", "no_text"],
)
async def test_summarization_agent_reply_becomes_the_native_summary(
    reply: list[StreamEvent],
):
    # The awaited summarization agent's reply becomes the summary Strands'
    # synchronous call makes of it, or fails as that call does.
    def summarizer() -> Agent:
        return Agent(model=SmallContextModel(reply=reply), callback_handler=None)

    manager = SummarizingConversationManager(summarization_agent=summarizer())
    try:
        expected: Any = _contents(
            [manager._generate_summary_with_agent(_history()[:2])]  # type: ignore[arg-type]
        )
    except Exception as error:
        expected = repr(error)
    try:
        actual: Any = _contents(
            [await _summarize_with_agent(summarizer(), _history()[:2])]  # type: ignore[arg-type]
        )
    except Exception as error:
        actual = repr(error)
    assert actual == expected


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
