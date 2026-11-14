"""Recorded histories replay through a ``Replayer`` configured with the plugin.

The histories under ``tests/histories`` were recorded by ``tests/record_histories.py``
from the same workflows the suite runs. A recording made before a release
stays as it is; a change that needs a new one adds a file next to it
(``tests/histories/README.md``), and every file in the directory replays here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from temporalio.client import Client, WorkflowHistory
from temporalio.microsoft_agent_framework import MicrosoftAgentFrameworkPlugin
from temporalio.worker import Replayer
from tests.record_histories import RECORDERS
from tests.test_agent import ToolLoopWorkflow
from tests.test_approval import ApprovalWorkflow
from tests.test_mcp import MCPWorkflow
from tests.test_streaming import StreamingWorkflow

HISTORIES = Path(__file__).with_name("histories")
WORKFLOWS = [ToolLoopWorkflow, ApprovalWorkflow, StreamingWorkflow, MCPWorkflow]


# Every file in the directory, so a recording added for an older shape (see
# tests/histories/README.md) keeps replaying without being listed here.
@pytest.mark.parametrize(
    "file_name", sorted(path.name for path in HISTORIES.glob("*.json"))
)
async def test_replay_recorded_histories(file_name: str) -> None:
    history = (HISTORIES / file_name).read_text(encoding="utf-8")
    await Replayer(
        workflows=WORKFLOWS, plugins=[MicrosoftAgentFrameworkPlugin()]
    ).replay_workflow(WorkflowHistory.from_json("recorded", history))


@pytest.mark.parametrize("file_name", sorted(RECORDERS))
async def test_replay_fresh_history(client: Client, file_name: str) -> None:
    # Record each scenario now and replay it with a fresh Replayer that has no
    # factories, so a change in the Activity result shape is exercised against
    # a history of the current code, not only the committed files.
    history = await RECORDERS[file_name](client)
    await Replayer(
        workflows=WORKFLOWS, plugins=[MicrosoftAgentFrameworkPlugin()]
    ).replay_workflow(history)
