"""Recorded histories replay through a ``Replayer`` configured with the plugin.

The histories under ``tests/histories`` were recorded by ``tests/record_histories.py``
from the same workflows the suite runs. Re-record them when one of those
workflows changes on purpose.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from temporalio.agent_framework import AgentFrameworkPlugin
from temporalio.client import Client, WorkflowHistory
from temporalio.worker import Replayer
from tests.record_histories import RECORDERS
from tests.test_agent import ToolLoopWorkflow
from tests.test_approval import ApprovalWorkflow
from tests.test_mcp import MCPWorkflow
from tests.test_streaming import StreamingWorkflow

HISTORIES = Path(__file__).with_name("histories")
WORKFLOWS = [ToolLoopWorkflow, ApprovalWorkflow, StreamingWorkflow, MCPWorkflow]


@pytest.mark.parametrize(
    "file_name",
    [
        "tool_loop.json",
        "approval.json",
        "streaming.json",
        # Recorded when the streaming Activity returned one entry per provider
        # update, including the provider layer's duplicate of each call and
        # one entry per text chunk. The workflow code must keep replaying it.
        "streaming_unmerged_record.json",
        "mcp.json",
    ],
)
async def test_replay_recorded_histories(file_name: str) -> None:
    history = (HISTORIES / file_name).read_text(encoding="utf-8")
    await Replayer(
        workflows=WORKFLOWS, plugins=[AgentFrameworkPlugin()]
    ).replay_workflow(WorkflowHistory.from_json("recorded", history))


@pytest.mark.parametrize("file_name", sorted(RECORDERS))
async def test_replay_fresh_history(client: Client, file_name: str) -> None:
    # Record each scenario now and replay it with a fresh Replayer that has no
    # factories, so a change in the Activity result shape is exercised against
    # a history of the current code, not only the committed files.
    history = await RECORDERS[file_name](client)
    await Replayer(
        workflows=WORKFLOWS, plugins=[AgentFrameworkPlugin()]
    ).replay_workflow(history)
