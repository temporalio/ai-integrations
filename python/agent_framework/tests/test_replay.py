"""Recorded histories replay through a ``Replayer`` configured with the plugin.

The histories under ``tests/histories`` were recorded by ``tests/record_histories.py``
from the same workflows the suite runs. Re-record them when one of those
workflows changes on purpose.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from temporalio.agent_framework import AgentFrameworkPlugin
from temporalio.agent_framework.testing import (
    fake_chat_client,
    function_call_response,
    text_response,
)
from temporalio.client import Client, WorkflowHistory
from temporalio.worker import Replayer
from tests.helpers.agent_framework import new_worker, workflow_id
from tests.test_agent import RunInput, ToolLoopWorkflow
from tests.test_approval import ApprovalWorkflow
from tests.test_mcp import MCPWorkflow
from tests.test_streaming import StreamingWorkflow

HISTORIES = Path(__file__).with_name("histories")
WORKFLOWS = [ToolLoopWorkflow, ApprovalWorkflow, StreamingWorkflow, MCPWorkflow]


@pytest.mark.parametrize(
    "file_name", ["tool_loop.json", "approval.json", "streaming.json", "mcp.json"]
)
async def test_replay_recorded_histories(file_name: str) -> None:
    history = (HISTORIES / file_name).read_text(encoding="utf-8")
    await Replayer(
        workflows=WORKFLOWS, plugins=[AgentFrameworkPlugin()]
    ).replay_workflow(WorkflowHistory.from_json("recorded", history))


async def test_replay_fresh_history(client: Client) -> None:
    plugin = AgentFrameworkPlugin(
        chat_clients={
            "model": fake_chat_client(
                [
                    function_call_response(
                        "get_weather", {"city": "Paris"}, call_id="call-1"
                    ),
                    text_response("Rainy in Paris."),
                ]
            )
        }
    )
    async with new_worker(client, ToolLoopWorkflow, plugin=plugin) as worker:
        handle = await client.start_workflow(
            ToolLoopWorkflow.run,
            RunInput(),
            id=workflow_id("replay-fresh"),
            task_queue=worker.task_queue,
        )
        await handle.result()
        history = await handle.fetch_history()
    # A fresh Replayer, with no factories, replays the recorded run.
    await Replayer(
        workflows=[ToolLoopWorkflow], plugins=[AgentFrameworkPlugin()]
    ).replay_workflow(history)
