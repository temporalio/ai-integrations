import warnings
from pathlib import Path

import pytest

from temporalio.client import WorkflowHistory
from temporalio.openai_agents import ModelActivityParameters, OpenAIAgentsPlugin
from temporalio.worker import Replayer
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
from tests.test_openai_streaming import StreamingInputGuardrailWorkflow


@pytest.mark.parametrize(
    "file_name",
    [
        "agents-as-tools-workflow-history.json",
        "customer-service-workflow-history.json",
        "hello-workflow-history.json",
        "input-guardrail-workflow-history.json",
        "output-guardrail-workflow-history.json",
        "research-workflow-history.json",
        "run-model-provider-workflow-history.json",
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
                HelloWorldAgent,
                InputGuardrailWorkflow,
                OutputGuardrailWorkflow,
                RunModelProviderWorkflow,
            ],
            plugins=[OpenAIAgentsPlugin()],
        ).replay_workflow(WorkflowHistory.from_json("fake", history_json))

    # Sandbox imports during an activation count toward the deadlock timeout
    assert not [
        str(w.message)
        for w in recorder
        if "was imported after initial workflow load" in str(w.message)
    ]


async def test_replay_streaming_input_guardrail_trip() -> None:
    # Recorded before the streaming model activity was cancelled on a tripwire: the parallel
    # input guardrail tripped mid-stream and the activity ran to completion. Streamed runs
    # import modules during activations, so this case does not check sandbox imports.
    history_json = (
        Path(__file__).with_name("histories")
        / "streaming-input-guardrail-workflow-history.json"
    ).read_text()
    await Replayer(
        workflows=[StreamingInputGuardrailWorkflow],
        plugins=[
            OpenAIAgentsPlugin(
                model_params=ModelActivityParameters(streaming_topic="events")
            )
        ],
    ).replay_workflow(WorkflowHistory.from_json("fake", history_json))
