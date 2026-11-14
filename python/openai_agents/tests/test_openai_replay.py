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


@pytest.mark.parametrize(
    "file_name",
    [
        "agents-as-tools-workflow-history.json",
        # Recorded on main at 9b6b4c4, which runs copies of the agents: no timer. Without
        # the openai_agents.run-caller-agents patch, it fails replay.
        "caller-agent-workflow-history.json",
        # Recorded on this version, with the patch marker: the run gets its own agent back.
        "caller-agent-marker-workflow-history.json",
        "customer-service-workflow-history.json",
        "hello-workflow-history.json",
        "input-guardrail-workflow-history.json",
        "output-guardrail-workflow-history.json",
        "research-workflow-history.json",
        "run-model-provider-workflow-history.json",
        # Recorded on main at ea04762 with the workflow worker stopped, so
        # that one activation resolves a stateful MCP call together with
        # another activity (gather) or with a workflow cancel (cancel).
        "stateful-mcp-cancel-workflow-history.json",
        "stateful-mcp-gather-workflow-history.json",
        "stateful-mcp-gather-wait-workflow-history.json",
        # Recorded on this version, with the patch marker.
        "stateful-mcp-cancel-wait-marker-workflow-history.json",
        "stateful-mcp-gather-wait-marker-workflow-history.json",
        "tools-workflow-callable-class-history.json",
        "tools-workflow-history.json",
    ],
)
async def test_replay(file_name: str) -> None:
    with (Path(__file__).with_name("histories") / file_name).open("r") as f:
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
            ],
            plugins=[OpenAIAgentsPlugin()],
        ).replay_workflow(WorkflowHistory.from_json("fake", history_json))

    # Sandbox imports during an activation count toward the deadlock timeout
    assert not [
        str(w.message)
        for w in recorder
        if "was imported after initial workflow load" in str(w.message)
    ]
