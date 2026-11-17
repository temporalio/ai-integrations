"""Call configuration reaches model and tool Activities through public APIs."""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator
from datetime import timedelta
from typing import Any

import pytest

from temporalio import workflow
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Replayer, Worker

with workflow.unsafe.imports_passed_through():
    from langchain_core.callbacks import AsyncCallbackManager
    from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
    from langchain_core.runnables import ConfigurableField, RunnableConfig
    from langchain_core.tools import tool

    from temporalio.contrib.workflow_streams import WorkflowStream
    from temporalio.deepagents import DeepAgentsPlugin, TemporalModel, tool_as_activity
    from temporalio.deepagents.testing import FakeModel


def visible(config: Any) -> dict:
    config = config or {}
    result = {
        key: config.get(key)
        for key in ("configurable", "tags", "metadata", "run_name", "recursion_limit")
    }
    result["run_id"] = (
        str(config["run_id"]) if config.get("run_id") is not None else None
    )
    return result


class ConfigInspectModel(FakeModel):
    async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> AIMessage:
        return AIMessage(content=json.dumps(visible(config)))

    async def astream(
        self, input: Any, config: Any = None, **kwargs: Any
    ) -> AsyncIterator[AIMessageChunk]:
        yield AIMessageChunk(content=json.dumps(visible(config)))


def configurable_provider(_: str = "") -> Any:
    return FakeModel(["default model response"]).configurable_fields(
        responses=ConfigurableField(id="answers")
    )


@workflow.defn
class ConfigurableModelWorkflow:
    @workflow.run
    async def run(self) -> dict:
        config: RunnableConfig = {
            "configurable": {"answers": ["per-run configured response"]}
        }
        native = await configurable_provider().ainvoke(
            [HumanMessage(content="hello")], config=config
        )
        durable = await TemporalModel(model="fake:configurable").ainvoke(
            [HumanMessage(content="hello")], config=config
        )
        return {"native": native.content, "durable": durable.content}


@tool
async def config_tool(query: str, config: RunnableConfig) -> str:
    """Return the caller's request configuration."""
    return json.dumps({"query": query, **visible(config)})


@workflow.defn
class ConfiguredCallWorkflow:
    @workflow.init
    def __init__(self, which: str) -> None:
        self.stream = WorkflowStream()

    @workflow.run
    async def run(self, which: str) -> dict:
        config: RunnableConfig = {
            "configurable": {"tenant": "acme"},
            "tags": ["manual-review"],
            "metadata": {"request": "q1"},
            "run_id": uuid.UUID(int=42),
            "run_name": "configured-call",
            "recursion_limit": 71,
        }
        if which in {"model", "portable", "stream", "stream-topic"}:
            native = await ConfigInspectModel(responses=["ignored"]).ainvoke(
                [HumanMessage(content="hello")], config=config
            )
            if which == "portable":
                config["callbacks"] = AsyncCallbackManager([])
                config["configurable"]["__pregel_store"] = object()
                config["configurable"]["not_portable"] = object()
                config["metadata"]["not_portable"] = object()
            model = TemporalModel(model="fake:config")
            if which.startswith("stream"):
                chunks = [
                    chunk
                    async for chunk in model.astream(
                        [HumanMessage(content="hello")], config=config
                    )
                ]
                durable = AIMessage(
                    content="".join(str(chunk.content) for chunk in chunks)
                )
            else:
                durable = await model.ainvoke(
                    [HumanMessage(content="hello")], config=config
                )
        else:
            call = {
                "name": config_tool.name,
                "args": {"query": "manual"},
                "id": "model-call",
                "type": "tool_call",
            }
            native = await config_tool.ainvoke(call, config=config)
            wrapped = tool_as_activity(
                config_tool, start_to_close_timeout=timedelta(seconds=10)
            )
            durable = await wrapped.ainvoke(call, config=config)
        return {
            "native": json.loads(str(native.content)),
            "durable": json.loads(str(durable.content)),
        }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "which", ["model", "tool", "portable", "stream", "stream-topic"]
)
async def test_call_config_matches_native(env: WorkflowEnvironment, which: str) -> None:
    task_queue = f"ga-config-{uuid.uuid4()}"
    plugin = DeepAgentsPlugin(
        model_provider=lambda _: ConfigInspectModel(responses=["ignored"]),
        streaming_topic="configured-model" if which == "stream-topic" else None,
    )
    async with Worker(
        env.client,
        task_queue=task_queue,
        workflows=[ConfiguredCallWorkflow],
        plugins=[plugin],
    ):
        handle = await env.client.start_workflow(
            ConfiguredCallWorkflow.run,
            which,
            id=str(uuid.uuid4()),
            task_queue=task_queue,
            execution_timeout=timedelta(seconds=20),
        )
        out = await handle.result()
    await Replayer(
        workflows=[ConfiguredCallWorkflow], plugins=[plugin]
    ).replay_workflow(await handle.fetch_history())
    assert out["durable"]["configurable"] == out["native"]["configurable"], out
    assert out["durable"]["tags"] == out["native"]["tags"], out
    assert out["durable"]["metadata"]["request"] == "q1"
    assert "not_portable" not in out["durable"]["metadata"]
    assert out["durable"]["run_name"] == out["native"]["run_name"]
    assert out["durable"]["run_id"] == out["native"]["run_id"]
    assert out["durable"]["recursion_limit"] == out["native"]["recursion_limit"]


@pytest.mark.asyncio
async def test_configurable_model_uses_per_run_choice(
    env: WorkflowEnvironment,
) -> None:
    task_queue = f"ga-configurable-{uuid.uuid4()}"
    plugin = DeepAgentsPlugin(model_provider=configurable_provider)
    async with Worker(
        env.client,
        task_queue=task_queue,
        workflows=[ConfigurableModelWorkflow],
        plugins=[plugin],
    ):
        out = await env.client.execute_workflow(
            ConfigurableModelWorkflow.run, id=str(uuid.uuid4()), task_queue=task_queue
        )
    assert out["durable"] == out["native"], out


@workflow.defn
class ConcurrentConfigWorkflow:
    @workflow.run
    async def run(self) -> list[str]:
        async def ask(tenant: str) -> str:
            response = await TemporalModel(model="fake:config").ainvoke(
                [HumanMessage(content=tenant)],
                config={"configurable": {"tenant": tenant}},
            )
            return json.loads(str(response.content))["configurable"]["tenant"]

        return list(await asyncio.gather(ask("first"), ask("second")))


@workflow.defn
class StreamConfigScopeWorkflow:
    @workflow.run
    async def run(self) -> dict:
        model = TemporalModel(model="fake:config")
        stream = model.astream(
            [HumanMessage(content="first")],
            config={"configurable": {"tenant": "first"}},
        )
        first = await anext(stream)
        second = await model.ainvoke([HumanMessage(content="second")])
        if close := getattr(stream, "aclose", None):
            await close()
        return {
            "first": json.loads(str(first.content)),
            "second": json.loads(str(second.content)),
        }


@pytest.mark.parametrize(
    "workflow_type", [ConcurrentConfigWorkflow, StreamConfigScopeWorkflow]
)
async def test_call_configuration_is_scoped_to_one_call(
    env: WorkflowEnvironment, workflow_type: Any
) -> None:
    task_queue = f"config-scope-{uuid.uuid4()}"
    plugin = DeepAgentsPlugin(
        model_provider=lambda _: ConfigInspectModel(responses=["ignored"])
    )
    async with Worker(
        env.client, task_queue=task_queue, workflows=[workflow_type], plugins=[plugin]
    ):
        handle = await env.client.start_workflow(
            workflow_type.run, id=str(uuid.uuid4()), task_queue=task_queue
        )
        out = await handle.result()
    await Replayer(workflows=[workflow_type], plugins=[plugin]).replay_workflow(
        await handle.fetch_history()
    )
    if workflow_type is ConcurrentConfigWorkflow:
        assert out == ["first", "second"]
    else:
        assert out["first"]["configurable"] == {"tenant": "first"}
        assert not out["second"]["configurable"]
