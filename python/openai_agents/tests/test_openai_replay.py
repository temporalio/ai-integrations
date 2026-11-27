import warnings
from pathlib import Path

import pytest

from temporalio.client import WorkflowHistory
from temporalio.openai_agents import OpenAIAgentsPlugin
from temporalio.worker import Replayer
from tests.helpers.stateful_replay import (
    StatefulMcpCancellingBlockWorkflow,
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
    OutputTypeWorkflow,
    RefusalFallbackWorkflow,
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
        # The run gets its own agent back, so it starts a timer.
        "caller-agent-workflow-history.json",
        # The final output didn't match output_type, so the run failed the Workflow
        # after recording the patch marker.
        "invalid-output-type-workflow-history.json",
        # Recorded before the patch: the run recovers from a UserError whose cause is
        # a ModelBehaviorError, so Runner.run must keep raising the UserError.
        "refusal-fallback-workflow-history.json",
        # Recorded with the workflow worker stopped, so that one activation
        # resolves a stateful MCP call together with another activity (gather)
        # or with a workflow cancel (cancel).
        "stateful-mcp-cancel-wait-workflow-history.json",
        "stateful-mcp-gather-wait-workflow-history.json",
        # Blocks left before their session activity starts, recorded without
        # the openai-agents-stateful-mcp-unstarted-session patch marker: each
        # fails with a bare CancelledError. Without the patch, the empty block
        # fails replay.
        "stateful-mcp-cancelling-block-workflow-history.json",
        "stateful-mcp-empty-block-workflow-history.json",
        "stateful-mcp-raising-block-workflow-history.json",
        # The same blocks recorded with the marker.
        "stateful-mcp-cancelling-block-marker-workflow-history.json",
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
                OutputTypeWorkflow,
                RefusalFallbackWorkflow,
                RunModelProviderWorkflow,
                StatefulMcpGatherWorkflow,
                StatefulMcpGatherWaitWorkflow,
                StatefulMcpCancelWorkflow,
                StatefulMcpCancelWaitWorkflow,
                StatefulMcpEmptyBlockWorkflow,
                StatefulMcpRaisingBlockWorkflow,
                StatefulMcpCancellingBlockWorkflow,
            ],
            plugins=[OpenAIAgentsPlugin()],
        ).replay_workflow(WorkflowHistory.from_json("fake", history_json))

    # Sandbox imports during an activation count toward the deadlock timeout
    assert not [
        str(w.message)
        for w in recorder
        if "was imported after initial workflow load" in str(w.message)
    ]
