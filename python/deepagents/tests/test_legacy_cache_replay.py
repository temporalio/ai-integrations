"""The retired legacy cache still replays pre-change histories.

The checked-in history was recorded before this change: two model-issued tool
calls with identical name+args, deduped by the legacy result cache to ONE
scheduled ``deepagents.invoke_tool`` activity. Its replay exercises the
unpatched branch of ``deepagents.retire-result-cache`` end to end — the only
executable coverage that branch has, and the tripwire that keeps it from
being "simplified" away while pre-change workflows are still in flight.

Fixture provenance: recorded on sdk-python main (pre-#1806) by driving a
scripted two-identical-tool-call agent through ``run_deep_agent`` on a local
dev server. ``legacy_cache_carried_messages_across_can.json`` was recorded the
same way on ai-integrations ``5e91c26`` with ``_legacy_result_cache`` forced
on (no patch marker), for a wrapped tool whose arguments hold a message.
Delete these tests and the fixtures together when the patch id goes through
``workflow.deprecate_patch``.
"""

from __future__ import annotations

import sys
from collections.abc import Awaitable, Callable
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 11), reason="deepagents requires Python >= 3.11"
)
pytest.importorskip("deepagents")
pytest.importorskip("langchain_core")

from temporalio import workflow
from temporalio.client import WorkflowHistory
from temporalio.deepagents import (
    DeepAgentsPlugin,
    run_deep_agent,
    tool_as_activity,
)
from temporalio.deepagents.testing import mock_model_provider
from temporalio.worker import Replayer

with workflow.unsafe.imports_passed_through():
    from langchain_core.messages import AIMessage
    from langchain_core.tools import tool

# Bind deepagents symbols off the module importorskip returns: a static
# `from deepagents import ...` cannot resolve on Python 3.10 (deepagents
# needs >= 3.11), and with the package absent the type checkers mis-resolve
# the name against this same-named test directory.
create_deep_agent = pytest.importorskip("deepagents").create_deep_agent

_HISTORY = Path(__file__).parent / "histories" / "legacy_dedup_repeated_tool_calls.json"


@tool("pairing_weather")
def legacy_replay_pairing_weather(city: str) -> str:
    """Return the weather for a city."""
    return f"weather:{city}"


# Distinct Python symbols from test_tools.py's same-named definitions; the
# DEFN name must match the workflow type recorded in the fixture history.
@workflow.defn(name="RepeatedToolCallWorkflow")
class LegacyReplayToolCallWorkflow:
    @workflow.run
    async def run(self) -> str:
        t = tool_as_activity(
            legacy_replay_pairing_weather, start_to_close_timeout=timedelta(seconds=10)
        )
        agent = create_deep_agent(model="fake:model", tools=[t])
        result = await run_deep_agent(
            agent,
            {"messages": [{"role": "user", "content": "Check Paris twice."}]},
            continue_as_new_after=10_000,
        )
        return str(result["messages"][-1].content)


RESPONSES: list[Any] = [
    AIMessage(
        content="",
        tool_calls=[
            {"name": "pairing_weather", "args": {"city": "Paris"}, "id": "first-call"}
        ],
    ),
    AIMessage(
        content="",
        tool_calls=[
            {"name": "pairing_weather", "args": {"city": "Paris"}, "id": "second-call"}
        ],
    ),
    AIMessage(content="done"),
]


@pytest.mark.asyncio
async def test_legacy_dedup_history_replays() -> None:
    plugin = DeepAgentsPlugin(model_provider=mock_model_provider(RESPONSES))
    replayer = Replayer(workflows=[LegacyReplayToolCallWorkflow], plugins=[plugin])
    history = WorkflowHistory.from_json("legacy-replay", _HISTORY.read_text())
    await replayer.replay_workflow(history)


with workflow.unsafe.imports_passed_through():
    from langchain_core.messages import HumanMessage
    from langchain_core.tools import tool as _lc_tool

from temporalio.deepagents import _activity
from temporalio.deepagents._tools import register_tool
from temporalio.deepagents.workflow import call_tool

_HISTORIES = Path(__file__).parent / "histories"


@_lc_tool("legacy_echo")
def legacy_replay_echo(x: int) -> str:
    """Echo a number."""
    return f"echo:{x}"


@_lc_tool("legacy_last_message")
def legacy_replay_last_message(messages: list) -> str:
    """Count the messages."""
    return f"messages:{len(messages)}"


register_tool(legacy_replay_echo)


class _SameCallAgent:
    def __init__(self, call: Callable[[], Awaitable[Any]]) -> None:
        self._call = call

    async def ainvoke(self, input: Any) -> dict:
        messages = list(input.get("messages", [])) if isinstance(input, dict) else []
        await self._call()
        done = len(messages) >= 1
        return {
            "messages": [*messages, "turn"],
            "todos": [{"content": "w", "status": "completed" if done else "pending"}],
        }


@workflow.defn(name="LegacyCanDedupWorkflow")
class LegacyReplayCanDedupWorkflow:
    @workflow.run
    async def run(self, input: dict, state_snapshot: dict | None = None) -> dict:
        from temporalio.deepagents import run_deep_agent

        return await run_deep_agent(
            _SameCallAgent(
                lambda: call_tool(
                    _activity.ToolActivityInput(
                        tool_name="legacy_echo", tool_call_id="tc-1", args={"x": 1}
                    ),
                    summary="tool:legacy_echo",
                    start_to_close_timeout=timedelta(seconds=10),
                )
            ),
            input,
            continue_as_new_after=1,
            state_snapshot=state_snapshot,
        )


@workflow.defn(name="LegacyCanDedupMessagesWorkflow")
class LegacyReplayCanDedupMessagesWorkflow:
    @workflow.run
    async def run(self, input: dict, state_snapshot: dict | None = None) -> dict:
        from temporalio.deepagents import run_deep_agent

        # A wrapped tool whose arguments hold a message: the legacy key must
        # hash them as recorded.
        wrapped = tool_as_activity(
            legacy_replay_last_message, start_to_close_timeout=timedelta(seconds=10)
        )
        args = {"messages": [HumanMessage(content="Weather in Paris?")]}
        return await run_deep_agent(
            _SameCallAgent(lambda: wrapped.ainvoke(args)),
            input,
            continue_as_new_after=1,
            state_snapshot=state_snapshot,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("workflow_cls", "fixture"),
    [
        (LegacyReplayCanDedupWorkflow, "legacy_cache_carried_across_can.json"),
        (
            LegacyReplayCanDedupMessagesWorkflow,
            "legacy_cache_carried_messages_across_can.json",
        ),
    ],
)
async def test_legacy_carried_cache_history_replays(
    workflow_cls: type, fixture: str
) -> None:
    """The legacy rehydrate-and-hit path replays: this continued-run history
    was recorded pre-change with the SAME tool call re-issued after the
    boundary and served from the CARRIED cache — it contains zero
    invoke_tool activities, so replay only succeeds if the unpatched branch
    seeds the cache from the inbound snapshot and serves the hit."""
    import json

    text = (_HISTORIES / fixture).read_text()
    events = json.loads(text)["events"]
    scheduled = [
        e
        for e in events
        if e.get("activityTaskScheduledEventAttributes", {})
        .get("activityType", {})
        .get("name")
        == "deepagents.invoke_tool"
    ]
    assert scheduled == [], "fixture must contain a fully cache-served run"

    plugin = DeepAgentsPlugin()
    replayer = Replayer(workflows=[workflow_cls], plugins=[plugin])
    history = WorkflowHistory.from_json("legacy-can-replay", text)
    await replayer.replay_workflow(history)
