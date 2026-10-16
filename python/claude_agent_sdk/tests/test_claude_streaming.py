"""Caller-facing partials on a Workflow Stream topic.

The durable result stays ``ClaudeTurnResult``. Live events are advisory: each attempt numbers
its own sequence from 0, and an attempt that dies can leave events on the topic after the
workflow has moved on.
"""

from __future__ import annotations

import asyncio
import inspect
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from claude_agent_sdk import InMemorySessionStore
from pydantic_core import to_json

from temporalio.claude_agent_sdk import (
    ClaudeAgentPlugin,
    ClaudePartialEvent,
    FileLeaseStore,
    run_claude_turn,
)
from temporalio.claude_agent_sdk._activity import (
    PartialSink,
    _partial_stream,
    _PartialPublisher,
)
from temporalio.claude_agent_sdk._models import (
    CLAUDE_TURN_CONFIG_ERROR_TYPE,
    RUN_CLAUDE_TURN_ACTIVITY,
    ClaudeTurnInput,
    partial_event,
)
from temporalio.claude_agent_sdk.testing import result as cli_result
from temporalio.claude_agent_sdk.testing import text_round
from temporalio.client import Client, WorkflowFailureError
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.contrib.workflow_streams import WorkflowStreamClient
from temporalio.contrib.workflow_streams._stream import _PUBLISH_SIGNAL
from temporalio.exceptions import ApplicationError
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from .conftest import scripted_cli as _cli
from .helpers import fixtures as fx
from .helpers import histories
from .helpers.workflows import (
    ClaudeChat,
    ClaudeStream,
    ClaudeStreamDurable,
    ClaudeStreamSink,
    ClaudeStreamUnhosted,
)

pytestmark = fx.POSIX_ONLY

ANSWER = "Hello world"
CHUNKS = ["Hello ", "world"]


def _message(event: dict[str, Any], parent: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(event=event, parent_tool_use_id=parent)


def _delta(kind: str, **fields: Any) -> dict[str, Any]:
    return {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": kind, **fields},
    }


def test_partial_event_keeps_text_and_thinking_apart() -> None:
    text = partial_event(
        _delta("text_delta", text="Hello "), attempt_id="p.1", sequence=0
    )
    thinking = partial_event(
        _delta("thinking_delta", thinking="User wants "),
        attempt_id="p.1",
        sequence=1,
    )
    signature = partial_event(
        _delta("signature_delta", signature="sig"), attempt_id="p.1", sequence=2
    )
    tool = partial_event(
        _delta("input_json_delta", partial_json='{"city":'),
        attempt_id="p.1",
        sequence=3,
    )
    start = partial_event(
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": "not a delta"},
        },
        attempt_id="p.1",
        sequence=4,
    )
    sub = partial_event(
        _delta("text_delta", text="SUBAGENT"),
        attempt_id="p.1",
        sequence=5,
        parent_tool_use_id="toolu_task",
    )

    assert (text.kind, text.text, text.thinking) == (
        "content_block_delta",
        "Hello ",
        None,
    )
    assert (thinking.text, thinking.thinking) == (None, "User wants ")
    assert signature.text is None and signature.thinking is None
    assert tool.text is None and tool.thinking is None
    assert start.kind == "content_block_start" and start.text is None
    assert sub.text == "SUBAGENT" and sub.parent_tool_use_id == "toolu_task"
    assert [text.sequence, thinking.sequence, signature.sequence] == [0, 1, 2]


def test_each_attempt_numbers_its_own_sequence_and_abandoned_events_remain() -> None:
    """A second attempt does not renumber or drop what the first already published."""
    log: list[ClaudePartialEvent] = []
    first = _PartialPublisher("prompt.1", log.append)
    second = _PartialPublisher("prompt.2", log.append)
    first.on_message(_message(_delta("text_delta", text="Hel")))
    first.on_message(_message(_delta("text_delta", text="lo")))
    first.on_message(SimpleNamespace(event="not-a-dict"))
    second.on_message(_message(_delta("text_delta", text="Hello")))

    assert [(event.attempt_id, event.sequence, event.text) for event in log] == [
        ("prompt.1", 0, "Hel"),
        ("prompt.1", 1, "lo"),
        ("prompt.2", 0, "Hello"),
    ]


def test_unset_streaming_fields_stay_out_of_the_activity_payload() -> None:
    raw = to_json(ClaudeTurnInput(prompt="hi", report_progress=True))
    assert b"streaming_topic" not in raw
    assert b"streaming_batch_interval_ms" not in raw
    assert b"streaming_context" not in raw


def test_a_topic_is_part_of_the_activity_payload() -> None:
    turn = ClaudeTurnInput(
        prompt="hi",
        streaming_topic="claude",
        streaming_batch_interval_ms=20,
        streaming_context={"turn": "t-1", "n": [1, 2]},
    )
    data = turn.model_dump(mode="json")
    assert data["streaming_topic"] == "claude"
    assert data["streaming_batch_interval_ms"] == 20
    assert data["streaming_context"] == {"turn": "t-1", "n": [1, 2]}
    assert ClaudeTurnInput.model_validate(data) == turn


def test_partial_event_is_the_public_stream_type() -> None:
    import temporalio.claude_agent_sdk as pkg

    assert pkg.ClaudePartialEvent is ClaudePartialEvent
    assert pkg.PartialSink is PartialSink
    assert (
        inspect.signature(run_claude_turn).parameters["streaming_context"].default
        is None
    )
    assert "streaming_topic" in inspect.signature(run_claude_turn).parameters
    assert (
        inspect.signature(run_claude_turn).parameters["streaming_topic"].default is None
    )
    event = ClaudePartialEvent(
        attempt_id="u.1", sequence=0, kind="content_block_delta", text="Hi"
    )
    assert ClaudePartialEvent.model_validate_json(event.model_dump_json()) == event


async def test_no_topic_does_not_open_a_stream_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def opened(**_kwargs: Any) -> Any:
        raise AssertionError("stream client opened")

    monkeypatch.setattr(
        "temporalio.claude_agent_sdk._activity.WorkflowStreamClient.from_within_activity",
        opened,
    )
    async with _partial_stream(ClaudeTurnInput(prompt="p")) as publisher:
        assert publisher is None


async def test_a_topic_publishes_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    published: list[ClaudePartialEvent] = []

    class _Client:
        def topic(self, name: str, *, type: type[Any] | None = None) -> _Client:
            assert name == "claude"
            assert type is ClaudePartialEvent
            return self

        def publish(self, value: ClaudePartialEvent) -> None:
            published.append(value)

        async def __aenter__(self) -> _Client:
            return self

        async def __aexit__(self, *_exc: object) -> None:
            return None

    seen: dict[str, Any] = {}

    def open_client(**kwargs: Any) -> _Client:
        seen.update(kwargs)
        return _Client()

    monkeypatch.setattr(
        "temporalio.claude_agent_sdk._activity.WorkflowStreamClient.from_within_activity",
        open_client,
    )
    turn = ClaudeTurnInput(
        prompt="p",
        prompt_uuid="prompt",
        attempt=2,
        streaming_topic="claude",
        streaming_batch_interval_ms=20,
    )
    async with _partial_stream(turn) as publisher:
        assert publisher is not None
        publisher.on_message(_message(_delta("text_delta", text="Hello ")))
        publisher.on_message(_message(_delta("text_delta", text="world")))

    assert seen["batch_interval"] == timedelta(milliseconds=20)
    assert [(event.attempt_id, event.sequence, event.text) for event in published] == [
        ("prompt.2", 0, "Hello "),
        ("prompt.2", 1, "world"),
    ]


async def test_a_sink_without_a_topic_gets_batches_with_the_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def opened(**_kwargs: Any) -> Any:
        raise AssertionError("stream client opened")

    monkeypatch.setattr(
        "temporalio.claude_agent_sdk._activity.WorkflowStreamClient.from_within_activity",
        opened,
    )
    batches: list[tuple[list[str | None], Any]] = []
    turn = ClaudeTurnInput(
        prompt="p",
        prompt_uuid="u",
        streaming_context={"id": 7},
        streaming_batch_interval_ms=10_000,
    )

    async def sink(events: list[ClaudePartialEvent], context: Any) -> None:
        batches.append(([event.text for event in events], context))

    async with _partial_stream(turn, sink) as publisher:
        assert publisher is not None
        publisher.on_message(_message(_delta("text_delta", text="Hi")))
        publisher.on_message(_message(_delta("thinking_delta", thinking="hm")))
        await asyncio.sleep(0.05)
        assert batches == []

    # One batch: the long interval coalesced both, and closing the stream delivered them.
    assert batches == [(["Hi", None], {"id": 7})]


async def test_a_slow_sink_gets_later_partials_in_the_next_batch_in_order() -> None:
    batches: list[list[int]] = []
    release = asyncio.Event()

    async def sink(events: list[ClaudePartialEvent], _context: Any) -> None:
        batches.append([event.sequence for event in events])
        await release.wait()

    turn = ClaudeTurnInput(prompt="p", streaming_batch_interval_ms=0)
    async with _partial_stream(turn, sink) as publisher:
        assert publisher is not None
        publisher.on_message(_message(_delta("text_delta", text="a")))
        await asyncio.sleep(0.05)
        publisher.on_message(_message(_delta("text_delta", text="b")))
        publisher.on_message(_message(_delta("text_delta", text="c")))
        release.set()

    assert batches == [[0], [1, 2]]


async def test_a_failing_sink_is_logged_once_and_the_partials_go_on(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    published: list[ClaudePartialEvent] = []

    class _Client:
        def topic(self, _name: str, *, type: type[Any] | None = None) -> _Client:
            assert type is ClaudePartialEvent
            return self

        def publish(self, value: ClaudePartialEvent) -> None:
            published.append(value)

        async def __aenter__(self) -> _Client:
            return self

        async def __aexit__(self, *_exc: object) -> None:
            return None

    monkeypatch.setattr(
        "temporalio.claude_agent_sdk._activity.WorkflowStreamClient.from_within_activity",
        lambda **_kwargs: _Client(),
    )
    calls: list[list[str | None]] = []

    def sink(events: list[ClaudePartialEvent], _context: Any) -> None:
        calls.append([event.text for event in events])
        raise RuntimeError("sink down")

    turn = ClaudeTurnInput(
        prompt="p",
        prompt_uuid="u",
        streaming_topic="claude",
        streaming_batch_interval_ms=0,
    )
    with caplog.at_level("WARNING"):
        async with _partial_stream(turn, sink) as publisher:
            assert publisher is not None
            publisher.on_message(_message(_delta("text_delta", text="a")))
            await asyncio.sleep(0.05)
            publisher.on_message(_message(_delta("text_delta", text="b")))

    assert [event.text for event in published] == ["a", "b"]
    assert calls == [["a"], ["b"]]
    warnings = [r for r in caplog.records if "partial_sink failed" in r.getMessage()]
    assert len(warnings) == 1 and "u.1" in warnings[0].getMessage()


def _turn(text: str, chunks: list[str]) -> list[dict[str, Any]]:
    return [
        *fx.START,
        *text_round("msg_stream", chunks, input_tokens=4, output_tokens=2),
        cli_result(text=text),
    ]


@asynccontextmanager
async def _worker(
    env: WorkflowEnvironment,
    tmp_path: Path,
    workflows: list[type],
    scenario: dict[str, Any],
    **plugin: Any,
) -> AsyncIterator[tuple[Client, str]]:
    config: dict[str, Any] = {
        **env.client.config(),
        "data_converter": pydantic_data_converter,
    }
    client = Client(**config)
    cli, _record = _cli(tmp_path / "cli", **scenario)
    task_queue = f"claude-{uuid.uuid4().hex[:8]}"
    async with Worker(
        client,
        task_queue=task_queue,
        workflows=workflows,
        plugins=[
            ClaudeAgentPlugin(
                cli_path=cli, workspace_root=tmp_path / "workspaces", **plugin
            )
        ],
    ):
        yield client, task_queue


async def test_a_live_stream_and_the_final_result(
    env: WorkflowEnvironment, tmp_path: Path
) -> None:
    async with _worker(
        env, tmp_path, [ClaudeStream], {"turn": _turn(ANSWER, CHUNKS)}
    ) as (client, task_queue):
        handle = await client.start_workflow(
            ClaudeStream.run, "hi", id=task_queue, task_queue=task_queue
        )
        stream = WorkflowStreamClient.create(client, handle.id)
        published: list[ClaudePartialEvent] = []

        async def collect() -> None:
            async for item in stream.subscribe(
                ["claude"],
                from_offset=0,
                result_type=ClaudePartialEvent,
                poll_cooldown=timedelta(milliseconds=20),
            ):
                published.append(item.data)
                if item.data.kind == "message_stop":
                    return

        collect_task = asyncio.create_task(collect())
        await asyncio.wait_for(collect_task, timeout=20)
        await handle.signal(ClaudeStream.finish)
        assert await handle.result() == ANSWER

    texts = [event.text for event in published if event.text is not None]
    assert texts == CHUNKS
    assert [event.sequence for event in published] == list(range(len(published)))
    assert {event.attempt_id for event in published} == {published[0].attempt_id}
    assert published[0].attempt_id.endswith(".1")
    assert published[0].kind == "message_start"


async def test_a_partial_sink_gets_the_turns_partials_with_its_context(
    env: WorkflowEnvironment, tmp_path: Path
) -> None:
    got: list[tuple[ClaudePartialEvent, Any]] = []

    async def sink(events: list[ClaudePartialEvent], context: Any) -> None:
        got.extend((event, context) for event in events)

    async with _worker(
        env,
        tmp_path,
        [ClaudeStreamSink],
        {"turn": _turn(ANSWER, CHUNKS)},
        partial_sink=sink,
    ) as (client, task_queue):
        handle = await client.start_workflow(
            ClaudeStreamSink.run,
            {"channel": "c-1"},
            id=task_queue,
            task_queue=task_queue,
        )
        assert await handle.result() == ANSWER
        history = histories.decoded((await handle.fetch_history()).to_json())

    assert [event.text for event, _ in got if event.text is not None] == CHUNKS
    assert all(context == {"channel": "c-1"} for _, context in got)
    assert [event.sequence for event, _ in got] == list(range(len(got)))
    assert _PUBLISH_SIGNAL not in history


async def test_streaming_without_a_hosted_stream_fails_before_the_activity(
    env: WorkflowEnvironment, tmp_path: Path
) -> None:
    async with _worker(
        env, tmp_path, [ClaudeStreamUnhosted], {"turn": _turn(ANSWER, CHUNKS)}
    ) as (client, task_queue):
        with pytest.raises(WorkflowFailureError) as exc_info:
            await client.execute_workflow(
                ClaudeStreamUnhosted.run,
                "hi",
                id=task_queue,
                task_queue=task_queue,
            )
        handle = client.get_workflow_handle(task_queue)
        history = histories.decoded((await handle.fetch_history()).to_json())

    cause = exc_info.value.cause
    assert isinstance(cause, ApplicationError)
    assert cause.type == CLAUDE_TURN_CONFIG_ERROR_TYPE
    assert "WorkflowStream" in cause.message
    assert RUN_CLAUDE_TURN_ACTIVITY not in history


async def test_no_topic_adds_no_stream_signals(
    env: WorkflowEnvironment, tmp_path: Path
) -> None:
    async with _worker(
        env, tmp_path, [ClaudeChat], {"turn": _turn(ANSWER, [ANSWER])}
    ) as (client, task_queue):
        handle = await client.start_workflow(
            ClaudeChat.run, id=task_queue, task_queue=task_queue
        )
        assert await handle.execute_update(ClaudeChat.chat, "hello") == ANSWER
        history = histories.decoded((await handle.fetch_history()).to_json())
        await handle.terminate()

    assert _PUBLISH_SIGNAL not in history
    assert "streaming_topic" not in history
    assert "streaming_context" not in history


@pytest.mark.timeout(90)
async def test_a_lost_attempt_leaves_its_partials_beside_the_resume(
    env: WorkflowEnvironment, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CLI dies mid-answer. Its partials stay on the topic; the resumed attempt starts
    its own sequence and the workflow result is the completed text.
    """
    prompt = "Stream please now"
    full = "One two three four"
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude-config"))
    async with _worker(
        env,
        tmp_path,
        [ClaudeStreamDurable],
        {
            "plans": [[prompt, [{"text": full}]]],
            "crash_at": "mid_answer",
            "crash_mode": "die",
            "crash_prompt": prompt,
            "trigger_settle": 0.4,
            "trigger_dir": str(tmp_path / "triggers"),
            "mirror_lag_rounds": 0,
            "log": str(tmp_path / "cli.jsonl"),
        },
        session_store=InMemorySessionStore(),
        lease_store=FileLeaseStore(tmp_path / "leases"),
        disconnect_timeout=timedelta(seconds=1),
    ) as (client, task_queue):
        handle = await client.start_workflow(
            ClaudeStreamDurable.run, prompt, id=task_queue, task_queue=task_queue
        )
        stream = WorkflowStreamClient.create(client, handle.id)
        published: list[ClaudePartialEvent] = []

        async def collect() -> None:
            async for item in stream.subscribe(
                ["claude"],
                from_offset=0,
                result_type=ClaudePartialEvent,
                poll_cooldown=timedelta(milliseconds=20),
            ):
                published.append(item.data)
                if item.data.kind == "message_stop":
                    return

        collect_task = asyncio.create_task(collect())
        await asyncio.wait_for(collect_task, timeout=40)
        await handle.signal(ClaudeStreamDurable.finish)
        assert await handle.result() == full

    by_attempt: dict[str, list[ClaudePartialEvent]] = {}
    for event in published:
        by_attempt.setdefault(event.attempt_id, []).append(event)
    assert len(by_attempt) >= 2
    for events in by_attempt.values():
        assert [event.sequence for event in events] == list(range(len(events)))
    attempt_ids = list(dict.fromkeys(event.attempt_id for event in published))
    first = [event.text for event in by_attempt[attempt_ids[0]] if event.text]
    second = [event.text for event in by_attempt[attempt_ids[-1]] if event.text]
    assert "".join(first) and "".join(first) != full
    assert full.startswith("".join(first))
    assert "".join(second) == full
