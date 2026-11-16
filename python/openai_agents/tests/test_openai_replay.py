import warnings
from pathlib import Path

import pytest

from temporalio.client import WorkflowHistory
from temporalio.openai_agents import OpenAIAgentsPlugin
from temporalio.worker import Replayer
from tests.helpers.stateful_replay import (
    StatefulMcpCancelWaitWorkflow,
    StatefulMcpCancelWorkflow,
    StatefulMcpEmptyBlockWorkflow,
    StatefulMcpGatherWaitWorkflow,
    StatefulMcpGatherWorkflow,
    StatefulMcpRaisingBlockWorkflow,
)
from tests.test_openai import (
    AgentsAsToolsWorkflow,
    CallerAgentWorkflow,
    CustomerServiceWorkflow,
    HelloWorldAgent,
    InputGuardrailWorkflow,
    OutputGuardrailWorkflow,
    ResearchWorkflow,
    RunModelProviderWorkflow,
    ToolsWorkflow,
)

# Replay is promised only within a major (see AGENTS.md), so only the histories
# in the current major's folder replay.
REPLAY_MAJOR = 2
HISTORIES = Path(__file__).with_name("histories") / f"{REPLAY_MAJOR}.x"


@pytest.mark.parametrize(
    "file_name",
    [
        # Recorded on this version, with the patch marker: the run gets its own agent back.
        "caller-agent-marker-workflow-history.json",
        # Recorded on this version, with the patch marker.
        "stateful-mcp-cancel-wait-marker-workflow-history.json",
        "stateful-mcp-gather-wait-marker-workflow-history.json",
        # Recorded on main at e3fef4e: blocks left before their session activity
        # starts fail with a bare CancelledError. Without the
        # openai-agents-stateful-mcp-unstarted-session patch, the empty block
        # fails replay.
        "stateful-mcp-empty-block-workflow-history.json",
        "stateful-mcp-raising-block-workflow-history.json",
        # Recorded on this version, with the patch marker.
        "stateful-mcp-empty-block-marker-workflow-history.json",
        "stateful-mcp-raising-block-marker-workflow-history.json",
        "tools-workflow-callable-class-history.json",
    ],
)
async def test_replay(file_name: str) -> None:
    with (HISTORIES / file_name).open("r") as f:
        history_json = f.read()

    with warnings.catch_warnings(record=True) as recorder:
        warnings.filterwarnings(
            "always", message=r"Module .* was imported after initial workflow load"
        )
        await Replayer(
            workflows=[
                ResearchWorkflow,
                ToolsWorkflow,
                CustomerServiceWorkflow,
                AgentsAsToolsWorkflow,
                CallerAgentWorkflow,
                HelloWorldAgent,
                InputGuardrailWorkflow,
                OutputGuardrailWorkflow,
                RunModelProviderWorkflow,
                StatefulMcpGatherWorkflow,
                StatefulMcpGatherWaitWorkflow,
                StatefulMcpCancelWorkflow,
                StatefulMcpCancelWaitWorkflow,
                StatefulMcpEmptyBlockWorkflow,
                StatefulMcpRaisingBlockWorkflow,
            ],
            plugins=[OpenAIAgentsPlugin()],
        ).replay_workflow(WorkflowHistory.from_json("fake", history_json))

    # Sandbox imports during an activation count toward the deadlock timeout
    assert not [
        str(w.message)
        for w in recorder
        if "was imported after initial workflow load" in str(w.message)
    ]
