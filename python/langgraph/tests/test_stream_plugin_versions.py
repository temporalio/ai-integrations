"""Incompatible tracing plugins are refused before Worker/Replayer execution."""

import sys
from importlib.machinery import ModuleSpec
from types import ModuleType

import pytest

from temporalio import workflow
from temporalio.client import Client, WorkflowHistory
from temporalio.langgraph import LangGraphPlugin
from temporalio.langgraph._event_loop import validate_stream_id_plugins
from temporalio.plugin import SimplePlugin
from temporalio.worker import Replayer, UnsandboxedWorkflowRunner, Worker


@workflow.defn
class NoHistoryWorkflow:
    @workflow.run
    async def run(self) -> None:
        return None


@pytest.mark.parametrize(
    "configuration", ["legacy-first", "stream-first", "legacy-client", "stream-client"]
)
async def test_reject_legacy_langsmith_before_worker_execution(
    client: Client, configuration: str
) -> None:
    stream = LangGraphPlugin()
    legacy = SimplePlugin("langchain.LangSmithPlugin", version="1.0.0")
    plugins = [legacy, stream] if configuration == "legacy-first" else [stream, legacy]
    if configuration == "legacy-client":
        client = Client(client.service_client, plugins=[legacy])
        plugins = [stream]
    elif configuration == "stream-client":
        client = Client(client.service_client, plugins=[stream])
        plugins = [legacy]
    with pytest.raises(ValueError, match="LangSmithPlugin.*upgrade"):
        Worker(
            client,
            task_queue="unused",
            workflows=[NoHistoryWorkflow],
            workflow_runner=UnsandboxedWorkflowRunner(),
            plugins=plugins,
        )


@pytest.mark.parametrize("legacy_first", [True, False])
async def test_reject_legacy_langsmith_before_replay(legacy_first: bool) -> None:
    stream = LangGraphPlugin()
    legacy = SimplePlugin("langchain.LangSmithPlugin", version="1.0.0")
    replayer = Replayer(
        workflows=[NoHistoryWorkflow],
        workflow_runner=UnsandboxedWorkflowRunner(),
        plugins=[legacy, stream] if legacy_first else [stream, legacy],
    )
    with pytest.raises(ValueError, match="LangSmithPlugin.*upgrade"):
        await replayer.replay_workflow(WorkflowHistory("unused", []))


@pytest.mark.parametrize("compatible", [False, True])
def test_installed_tracing_capability_even_without_configured_plugin(
    monkeypatch: pytest.MonkeyPatch, compatible: bool
) -> None:
    class TracingPlugin:
        pass

    if compatible:
        setattr(TracingPlugin, "_temporal_langchain_stream_ids", True)
    module = ModuleType("temporalio.langsmith")
    module.__spec__ = ModuleSpec("temporalio.langsmith", loader=None)
    setattr(module, "LangSmithPlugin", TracingPlugin)
    monkeypatch.setitem(sys.modules, "temporalio.langsmith", module)
    if compatible:
        validate_stream_id_plugins([])
    else:
        with pytest.raises(ValueError, match="LangSmithPlugin.*upgrade"):
            validate_stream_id_plugins([])
