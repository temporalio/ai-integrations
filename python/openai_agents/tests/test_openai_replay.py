import warnings
from pathlib import Path

import pytest

from temporalio.client import WorkflowHistory
from temporalio.openai_agents import OpenAIAgentsPlugin
from temporalio.worker import Replayer
from tests.helpers.stateful_replay import (
    StatefulMcpCancelWaitWorkflow,
    StatefulMcpCancelWorkflow,
    StatefulMcpGatherWaitWorkflow,
    StatefulMcpGatherWorkflow,
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
from tests.test_openai_model_parameters import ExplicitModelConfigWorkflow

# Replay is promised only within a major (see AGENTS.md), so only the histories
# in the current major's folder replay.
REPLAY_MAJOR = 2
HISTORIES = Path(__file__).with_name("histories") / f"{REPLAY_MAJOR}.x"


@pytest.mark.parametrize(
    "file_name",
    [
        "explicit-model-config-workflow-history.json",
        # The run gets its own agent back, so it starts a timer.
        "caller-agent-workflow-history.json",
        # Recorded with the workflow worker stopped, so that one activation
        # resolves a stateful MCP call together with another activity (gather)
        # or with a workflow cancel (cancel).
        "stateful-mcp-cancel-wait-workflow-history.json",
        "stateful-mcp-gather-wait-workflow-history.json",
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
                ExplicitModelConfigWorkflow,
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
            ],
            plugins=[OpenAIAgentsPlugin()],
        ).replay_workflow(WorkflowHistory.from_json("fake", history_json))

    # Sandbox imports during an activation count toward the deadlock timeout
    assert not [
        str(w.message)
        for w in recorder
        if "was imported after initial workflow load" in str(w.message)
    ]
