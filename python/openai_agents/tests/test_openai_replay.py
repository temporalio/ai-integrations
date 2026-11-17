import warnings
from pathlib import Path

import pytest

from temporalio.client import WorkflowHistory
from temporalio.openai_agents import OpenAIAgentsPlugin
from temporalio.worker import Replayer
from tests.test_openai import CallerAgentWorkflow, ToolsWorkflow
from tests.test_openai_model_parameters import ExplicitModelConfigWorkflow

# The explicit API starts a new major; earlier histories remain archived.
REPLAY_MAJOR = 2
HISTORIES = Path(__file__).with_name("histories") / f"{REPLAY_MAJOR}.x"


@pytest.mark.parametrize(
    "file_name",
    [
        "explicit-model-config-workflow-history.json",
        "caller-agent-workflow-history.json",
        "tools-workflow-callable-class-history.json",
    ],
)
async def test_replay(file_name: str) -> None:
    history = WorkflowHistory.from_json("explicit", (HISTORIES / file_name).read_text())
    with warnings.catch_warnings(record=True) as recorder:
        warnings.filterwarnings(
            "always", message=r"Module .* was imported after initial workflow load"
        )
        await Replayer(
            workflows=[ExplicitModelConfigWorkflow, CallerAgentWorkflow, ToolsWorkflow],
            plugins=[OpenAIAgentsPlugin()],
        ).replay_workflow(history)
    assert not [
        str(w.message)
        for w in recorder
        if "was imported after initial workflow load" in str(w.message)
    ]
