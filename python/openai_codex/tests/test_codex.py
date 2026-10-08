"""End-to-end tests against the real ``codex app-server`` binary and a fake Responses API.

No credentials needed: Codex talks to a local scripted server through a custom model provider.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import timedelta
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from pydantic import BaseModel, ValidationError

from temporalio import activity
from temporalio.api.enums.v1 import EventType
from temporalio.client import Client, WorkflowFailureError
from temporalio.openai_codex import CodexObserver, CodexPlugin, CodexTokenUsage
from temporalio.openai_codex._app_server import AppServer
from temporalio.openai_codex.testing import FakeResponsesServer
from temporalio.openai_codex.workflow import codex_tool, function_schema
from tests._workflows import (
    CodexWorkflow,
    ObservedCodexWorkflow,
    lookup,
    record,
)
from tests.helpers import new_worker

requires_codex_binary = pytest.mark.skipif(
    find_spec("codex_cli_bin") is None,
    reason="needs the Codex binary (the `bundled-codex` extra)",
)

PROMPT = 'Please look up the ticket.\nCALL:lookup|{"q":"42"}'


class RecordingObserver:
    """A CodexObserver that keeps what it was told."""

    def __init__(self, events: list[tuple[str, Any]], context: dict[str, Any]) -> None:
        self.events = events
        self.context = context

    def model_interaction_started(self, model: str | None) -> None:  # pyright: ignore[reportUnusedParameter]
        self.events.append(("started", self.context))

    def reply_delta(self, text: str) -> None:
        self.events.append(("delta", text))

    def model_interaction_ended(
        self,
        model: str | None,  # pyright: ignore[reportUnusedParameter]
        usage: CodexTokenUsage | None,
    ) -> None:
        self.events.append(("ended", usage))


@pytest_asyncio.fixture  # type: ignore[reportUntypedFunctionDecorator]
async def fake() -> AsyncIterator[FakeResponsesServer]:
    server = FakeResponsesServer()
    await server.start()
    yield server
    await server.stop()


@pytest.fixture(autouse=True)
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A ledger for Activity side effects, and no way to reach the real API."""
    ledger = tmp_path / "ledger.txt"
    ledger.touch()
    monkeypatch.setenv("CODEX_TEST_LEDGER", str(ledger))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("CODEX_API_KEY", raising=False)
    return ledger


@asynccontextmanager
async def codex_worker(
    client: Client,
    fake: FakeResponsesServer,
    tmp_path: Path,
    events: list[tuple[str, Any]] | None = None,
) -> AsyncIterator[str]:
    @asynccontextmanager
    async def observer_factory(context: dict[str, Any]) -> AsyncIterator[CodexObserver]:
        yield RecordingObserver(events if events is not None else [], context)

    plugin = CodexPlugin(
        config_overrides=fake.config_overrides,
        home_root=str(tmp_path),
        observer_factory=observer_factory,
    )
    worker = new_worker(
        client,
        CodexWorkflow,
        ObservedCodexWorkflow,
        activities=[record],
        plugins=[plugin],
    )
    async with worker:
        yield worker.task_queue


def offered_tools(fake: FakeResponsesServer) -> set[str]:
    return {
        tool.get("name") or tool.get("type")
        for body in fake.requests
        for tool in body.get("tools", [])
    }


def test_codex_tool_derives_the_spec_from_the_function() -> None:
    tool = codex_tool(lookup)
    assert tool.spec.name == "lookup"
    assert tool.spec.description == "Look up a ticket by id."
    assert tool.spec.input_schema["type"] == "object"
    assert tool.spec.input_schema["properties"]["q"]["type"] == "string"
    assert tool.spec.input_schema["required"] == ["q"]


class Address(BaseModel):
    """A postal address."""

    city: str


def ship(order_id: str, to: Address, express: bool = False) -> str:
    """Ship an order."""
    return f"{order_id}:{to.city}:{express}"


def test_function_schema_parses_arguments_into_the_functions_own_types() -> None:
    schema = function_schema(ship)
    assert schema.spec.name == "ship"
    assert schema.spec.input_schema["required"] == ["order_id", "to"]
    kwargs = schema.parse({"order_id": "o1", "to": {"city": "Paris"}})
    assert list(kwargs) == ["order_id", "to", "express"]
    assert isinstance(kwargs["to"], Address)
    assert ship(**kwargs) == "o1:Paris:False"
    with pytest.raises(ValidationError):
        schema.parse({"order_id": "o1"})


@requires_codex_binary
async def test_host_tool_result_reaches_the_model(
    client: Client, fake: FakeResponsesServer, tmp_path: Path
) -> None:
    async with codex_worker(client, fake, tmp_path) as task_queue:
        handle = await client.start_workflow(
            CodexWorkflow.run,
            PROMPT,
            id=f"codex-{uuid.uuid4()}",
            task_queue=task_queue,
        )
        assert "ticket-42: resolved" in await handle.result()
        state = await handle.query(CodexWorkflow.state)
        assert state["thread_id"]
        assert state["rollout_lines"] > 0
        assert list(state["tool_results"].values()) == ["ticket-42: resolved"]

    # Codex's built-in tools are off in EVERY request, including segments that resume the thread
    # (the thread config does not persist across a resume).
    assert len(fake.requests) >= 2
    offered = offered_tools(fake)
    assert {"lookup", "record", "explode"} <= offered
    assert not offered & {
        "exec_command",
        "write_stdin",
        "apply_patch",
        "view_image",
        "create_goal",
    }


@requires_codex_binary
async def test_activity_tool_runs_once_with_a_stable_activity_id(
    client: Client, fake: FakeResponsesServer, tmp_path: Path, isolated_env: Path
) -> None:
    async with codex_worker(client, fake, tmp_path) as task_queue:
        handle = await client.start_workflow(
            CodexWorkflow.run,
            'Record it.\nCALL:record|{"note":"hello"}',
            id=f"codex-{uuid.uuid4()}",
            task_queue=task_queue,
        )
        assert "recorded:hello" in await handle.result()
        history = await handle.fetch_history()

    (entry,) = isolated_env.read_text().split()
    note, activity_id = entry.split("|")
    assert note == "hello"
    assert activity_id.startswith("tool-call_")
    scheduled = [
        e.activity_task_scheduled_event_attributes.activity_id
        for e in history.events
        if e.event_type == EventType.EVENT_TYPE_ACTIVITY_TASK_SCHEDULED
    ]
    assert activity_id in scheduled


@requires_codex_binary
async def test_a_failing_tool_is_a_result_the_model_sees(
    client: Client, fake: FakeResponsesServer, tmp_path: Path
) -> None:
    async with codex_worker(client, fake, tmp_path) as task_queue:
        handle = await client.start_workflow(
            CodexWorkflow.run,
            'Try it.\nCALL:explode|{"reason":"boom"}',
            id=f"codex-{uuid.uuid4()}",
            task_queue=task_queue,
        )
        result = await handle.result()
    assert "failed" in result
    assert "boom" in result


@requires_codex_binary
async def test_observer_sees_the_turn(
    client: Client, fake: FakeResponsesServer, tmp_path: Path
) -> None:
    events: list[tuple[str, Any]] = []
    async with codex_worker(client, fake, tmp_path, events) as task_queue:
        handle = await client.start_workflow(
            ObservedCodexWorkflow.run,
            PROMPT,
            id=f"codex-{uuid.uuid4()}",
            task_queue=task_queue,
        )
        await handle.result()

    kinds = [kind for kind, _ in events]
    # One segment before the tool call and one after it: two spans.
    assert kinds.count("started") == kinds.count("ended") == 2
    assert all(
        payload == {"turn": "t1"} for kind, payload in events if kind == "started"
    )
    assert "ticket-42: resolved" in "".join(p for kind, p in events if kind == "delta")
    final_usage = [p for kind, p in events if kind == "ended"][-1]
    assert final_usage is not None and final_usage.total_tokens


@requires_codex_binary
async def test_crash_after_a_tool_ran_does_not_run_it_again(
    client: Client,
    fake: FakeResponsesServer,
    tmp_path: Path,
    isolated_env: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_call = AppServer.call

    async def crashing_call(self: AppServer, method: str, params: Any = None) -> Any:
        result = await original_call(self, method, params)
        # The segment that resumes after the tool: SIGKILL the app-server right after the turn
        # starts, on the first attempt only, as a dying Worker would.
        if (
            method == "turn/start"
            and params["input"][0]["text"] == "Continue."
            and activity.info().attempt == 1
        ):
            self.kill()
        return result

    monkeypatch.setattr(AppServer, "call", crashing_call)

    async with codex_worker(client, fake, tmp_path) as task_queue:
        handle = await client.start_workflow(
            CodexWorkflow.run,
            'Record it.\nCALL:record|{"note":"once"}',
            id=f"codex-{uuid.uuid4()}",
            task_queue=task_queue,
        )
        assert "recorded:once" in await handle.result()
        history = await handle.fetch_history()

    # The side effect happened exactly once, though the segment after it ran twice.
    assert [line.split("|")[0] for line in isolated_env.read_text().split()] == ["once"]
    scheduled: dict[int, str] = {}
    retried: list[str] = []
    for event in history.events:
        if event.event_type == EventType.EVENT_TYPE_ACTIVITY_TASK_SCHEDULED:
            attrs = event.activity_task_scheduled_event_attributes
            scheduled[event.event_id] = attrs.activity_type.name
        if event.event_type == EventType.EVENT_TYPE_ACTIVITY_TASK_STARTED:
            started = event.activity_task_started_event_attributes
            if started.attempt > 1:
                retried.append(scheduled[started.scheduled_event_id])
    assert retried == ["openai_codex.run_segment"]

    # The model saw the real tool output exactly once, never Codex's synthetic "aborted".
    outputs = [
        item["output"]
        for item in fake.requests[-1]["input"]
        if item.get("type") == "function_call_output"
    ]
    assert outputs == ["recorded:once"]


@requires_codex_binary
async def test_a_failed_model_call_fails_the_turn_instead_of_replying_empty(
    client: Client, fake: FakeResponsesServer, tmp_path: Path
) -> None:
    fake.failure_status = 400  # not retried by Codex: the turn ends as failed
    async with codex_worker(client, fake, tmp_path) as task_queue:
        handle = await client.start_workflow(
            CodexWorkflow.run,
            PROMPT,
            id=f"codex-{uuid.uuid4()}",
            task_queue=task_queue,
            execution_timeout=timedelta(seconds=60),
        )
        with pytest.raises(WorkflowFailureError):
            await handle.result()
