import warnings
from datetime import timedelta
from pathlib import Path

import pytest
from agents import Agent, Runner

from temporalio import workflow
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
    CustomerServiceWorkflow,
    HelloWorldAgent,
    InputGuardrailWorkflow,
    OutputGuardrailWorkflow,
    ResearchWorkflow,
    RunModelProviderWorkflow,
    ToolsWorkflow,
)


@workflow.defn(name="CallerAgentWorkflow")
class LegacyCallerAgentWorkflow:
    """Keep an unmodified upstream Agent to exercise the pre-API replay path."""

    @workflow.run
    async def run(self, prompt: str) -> str:
        agent = Agent[None](
            name="Assistant", instructions="You only respond in haikus."
        )
        result = await Runner.run(starting_agent=agent, input=prompt)
        if result.last_agent is agent:
            await workflow.sleep(timedelta(milliseconds=100))
        return result.final_output


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
                LegacyCallerAgentWorkflow,
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
