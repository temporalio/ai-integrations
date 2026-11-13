"""Replay safety: workflow-aware ids and clocks, Activity-side occurrence ids, seam guards."""

from __future__ import annotations

import importlib
import time
import uuid
from typing import Any

import pytest
from agent_framework import Agent, FunctionTool, tool

from temporalio import workflow
from temporalio.client import Client
from temporalio.microsoft_agent_framework import (
    MicrosoftAgentFrameworkPlugin,
    TemporalChatClient,
    _workflow_runtime,
)
from temporalio.microsoft_agent_framework.testing import (
    fake_chat_client,
    function_call_response,
    text_response,
)
from temporalio.worker import Replayer
from tests.helpers.microsoft_agent_framework import (
    INVOKE_MODEL,
    SlowFakeChatClient,
    activity_results,
    count_scheduled_activities,
    new_worker,
    workflow_id,
)


@workflow.defn
class IdsWorkflow:
    @workflow.run
    async def run(self) -> dict[str, str]:
        agent = Agent(client=TemporalChatClient("model"), name="ids")
        session = agent.create_session()
        await agent.run(f"{agent.id} {session.session_id}", session=session)
        return {"agent_id": agent.id, "session_id": session.session_id}


async def test_ids_stable_across_replay(client: Client) -> None:
    factory = fake_chat_client(["noted"])
    plugin = MicrosoftAgentFrameworkPlugin(chat_clients={"model": factory})
    async with new_worker(client, IdsWorkflow, plugin=plugin) as worker:
        handle = await client.start_workflow(
            IdsWorkflow.run, id=workflow_id("ids"), task_queue=worker.task_queue
        )
        result = await handle.result()
    # The prompt was built in the first workflow task; the result comes from a
    # later task that replayed the workflow from scratch (max_cached_workflows=0).
    prompt = factory().calls[0]["messages"][0].text
    assert prompt == f"{result['agent_id']} {result['session_id']}"
    assert uuid.UUID(result["agent_id"]).version == 4
    assert uuid.UUID(result["session_id"]).version == 4
    assert result["agent_id"] != result["session_id"]


@workflow.defn
class BudgetWorkflow:
    @workflow.run
    async def run(self) -> dict[str, Any]:
        log: list[str] = []

        @tool
        async def get_weather(city: str) -> str:
            """Get the weather for a city."""
            log.append(city)
            return f"rainy in {city}"

        client = TemporalChatClient(
            "model", function_invocation_configuration={"max_duration_seconds": 0.5}
        )
        agent = Agent(client=client, tools=[get_weather])
        response = await agent.run("weather in Paris, then Oslo")
        return {"text": response.text, "log": log}


async def test_duration_budget_uses_workflow_clock(client: Client) -> None:
    slow = SlowFakeChatClient(
        [
            function_call_response("get_weather", {"city": "Paris"}, call_id="call-1"),
            text_response("Out of time."),
        ],
        delay=0.7,
    )
    plugin = MicrosoftAgentFrameworkPlugin(chat_clients={"model": lambda: slow})
    async with new_worker(client, BudgetWorkflow, plugin=plugin) as worker:
        handle = await client.start_workflow(
            BudgetWorkflow.run, id=workflow_id("budget"), task_queue=worker.task_queue
        )
        result = await handle.result()
        history = await handle.fetch_history()
    assert result == {"text": "Out of time.", "log": ["Paris"]}
    # The 0.7 second model call used up the 0.5 second budget on the workflow
    # clock, so the loop asked for a final answer without tools.
    first, second = slow.calls
    assert first["options"].get("tool_choice") != "none"
    assert second["options"]["tool_choice"] == "none"
    assert await count_scheduled_activities(handle) == {INVOKE_MODEL: 2}
    # Replaying later, with no real time elapsing, makes the same decision.
    await Replayer(
        workflows=[BudgetWorkflow], plugins=[MicrosoftAgentFrameworkPlugin()]
    ).replay_workflow(history)


@workflow.defn
class OccurrenceIdWorkflow:
    @workflow.run
    async def run(self) -> dict[str, Any]:
        @tool
        async def get_weather(city: str) -> str:
            """Get the weather for a city."""
            return f"rainy in {city}"

        agent = Agent(client=TemporalChatClient("model"), tools=[get_weather])
        response = await agent.run("weather in Paris?")
        calls = [
            c
            for m in response.messages
            for c in m.contents
            if c.type == "function_call"
        ]
        return {"ids": [c.id for c in calls], "text": response.text}


async def test_occurrence_ids_from_activity(client: Client) -> None:
    factory = fake_chat_client(
        [
            function_call_response(
                "get_weather", {"city": "Paris"}, call_id="call-1", id=None
            ),
            text_response("Rainy."),
        ]
    )
    plugin = MicrosoftAgentFrameworkPlugin(chat_clients={"model": factory})
    async with new_worker(client, OccurrenceIdWorkflow, plugin=plugin) as worker:
        handle = await client.start_workflow(
            OccurrenceIdWorkflow.run,
            id=workflow_id("occurrence"),
            task_queue=worker.task_queue,
        )
        result = await handle.result()
    (occurrence_id,) = result["ids"]
    assert occurrence_id.startswith("af-call-")
    # The id is already in the recorded Activity result, so a replay reads it
    # from history instead of minting a new one in the workflow.
    first_result, _ = await activity_results(handle)
    recorded = [
        c["id"]
        for m in first_result["messages"]
        for c in m["contents"]
        if c["type"] == "function_call"
    ]
    assert recorded == [occurrence_id]
    # The provider's second call saw the same id in the transcript.
    second_call = factory().calls[1]
    assert [
        c.id
        for m in second_call["messages"]
        for c in m.contents
        if c.type == "function_call"
    ] == [occurrence_id]


def test_shims_installed_and_inert_outside_workflows() -> None:
    _workflow_runtime.install_workflow_runtime()
    for module_name, attr, _expected, shim in _workflow_runtime._SEAMS:
        assert getattr(importlib.import_module(module_name), attr) is shim, (
            module_name,
            attr,
        )
    tools = importlib.import_module("agent_framework._tools")
    sessions = importlib.import_module("agent_framework._sessions")
    assert uuid.UUID(tools.uuid4().hex).version == 4
    assert uuid.UUID(sessions.uuid.uuid4().hex).version == 4
    assert sessions.uuid.UUID is uuid.UUID
    before = time.perf_counter()
    assert tools.perf_counter() >= before
    assert abs(tools.time() - time.time()) < 5
    # Installing again is a no-op.
    _workflow_runtime.install_workflow_runtime()


@pytest.mark.parametrize(
    ("module_name", "attr"),
    [(module, attr) for module, attr, _, _ in _workflow_runtime._SEAMS],
)
def test_seam_guard(
    monkeypatch: pytest.MonkeyPatch, module_name: str, attr: str
) -> None:
    module = importlib.import_module(module_name)

    def foreign() -> None:
        return None

    monkeypatch.setattr(module, attr, foreign)
    with pytest.raises(RuntimeError) as excinfo:
        _workflow_runtime.install_workflow_runtime()
    message = str(excinfo.value)
    assert f"{module_name}.{attr}" in message
    assert _workflow_runtime.installed_version() in message


def test_sync_tool_seam_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    async def without_flag(_self: Any, _call_kwargs: Any) -> Any:
        return None

    monkeypatch.setattr(FunctionTool, "_invoke_function", without_flag)
    with pytest.raises(RuntimeError, match="_invoke_sync_on_event_loop"):
        _workflow_runtime.check_sync_tool_seam()
