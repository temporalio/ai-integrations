from pathlib import Path

import pytest
from google.adk.models import LLMRegistry

from temporalio.client import WorkflowHistory
from temporalio.google_adk import GoogleAdkPlugin
from temporalio.worker import Replayer
from tests.test_adk_graph_workflows import (
    SequentialGraphWorkflow,
)
from tests.test_adk_hitl import HumanInputGraphWorkflow
from tests.test_google_adk_agents import (
    MultiAgentWorkflow,
    ResearchModel,
    WeatherAgent,
    WeatherModel,
)

# Replay is promised only within a major (see AGENTS.md), so only the histories
# in the current major's folder replay.
REPLAY_MAJOR = 0
HISTORIES = Path(__file__).with_name("histories") / f"{REPLAY_MAJOR}.x"


@pytest.mark.parametrize(
    "file_name",
    [
        "multi_agent.json",
        "single_agent.json",
        "graph_workflow.json",
        "hitl_workflow.json",
    ],
)
async def test_replay(file_name: str) -> None:
    with (HISTORIES / file_name).open("r") as f:
        history_json = f.read()

        LLMRegistry.register(ResearchModel)
        LLMRegistry.register(WeatherModel)
        await Replayer(
            workflows=[
                MultiAgentWorkflow,
                WeatherAgent,
                SequentialGraphWorkflow,
                HumanInputGraphWorkflow,
            ],
            plugins=[GoogleAdkPlugin()],
        ).replay_workflow(WorkflowHistory.from_json("fake", history_json))
