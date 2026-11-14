"""Plugin composition: registration, data converter, sandbox passthrough, warnings."""

from __future__ import annotations

import dataclasses
import importlib.metadata
from typing import Any

import pytest
from agent_framework import Agent

from temporalio import workflow
from temporalio.api.common.v1 import Payload
from temporalio.client import Client
from temporalio.contrib.pydantic import (
    PydanticPayloadConverter,
    pydantic_data_converter,
)
from temporalio.converter import (
    CompositePayloadConverter,
    DataConverter,
    DefaultPayloadConverter,
    PayloadCodec,
)
from temporalio.microsoft_agent_framework import (
    MicrosoftAgentFrameworkPlugin,
    TemporalChatClient,
)
from temporalio.microsoft_agent_framework._plugin import (
    _data_converter,
    _workflow_runner,
)
from temporalio.microsoft_agent_framework.testing import fake_chat_client
from temporalio.worker import Worker
from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner
from tests.helpers.microsoft_agent_framework import new_worker, workflow_id


@workflow.defn
class HelloWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        return (await Agent(client=TemporalChatClient("model")).run(prompt)).text


async def test_client_and_worker_registration(client: Client) -> None:
    plugin = MicrosoftAgentFrameworkPlugin(
        chat_clients={"model": fake_chat_client(["Hello."])}
    )
    config = client.config()
    config["plugins"] = [plugin]
    plugged = Client(**config)
    task_queue = workflow_id("agent-framework-client-plugin")
    # On the Client only: the Worker built from it picks the plugin up.
    async with Worker(
        plugged,
        task_queue=task_queue,
        workflows=[HelloWorkflow],
        max_cached_workflows=0,
    ):
        assert (
            await plugged.execute_workflow(
                HelloWorkflow.run,
                "hi",
                id=workflow_id("client-plugin"),
                task_queue=task_queue,
            )
            == "Hello."
        )
    # On the Worker only works the same way (every other test does this).
    async with new_worker(client, HelloWorkflow, plugin=plugin) as worker:
        assert (
            await client.execute_workflow(
                HelloWorkflow.run,
                "hi",
                id=workflow_id("worker-plugin"),
                task_queue=worker.task_queue,
            )
            == "Hello."
        )
    # On both, the SDK registers the Activities twice and refuses the Worker.
    with pytest.warns(UserWarning, match="both client and worker"):
        with pytest.raises(ValueError, match="More than one activity"):
            Worker(
                plugged,
                task_queue=task_queue,
                workflows=[HelloWorkflow],
                plugins=[plugin],
            )


class _NoopCodec(PayloadCodec):
    async def encode(self, payloads: Any) -> list[Payload]:
        return list(payloads)

    async def decode(self, payloads: Any) -> list[Payload]:
        return list(payloads)


class _CustomPayloadConverter(CompositePayloadConverter):
    def __init__(self) -> None:
        super().__init__(*DefaultPayloadConverter.default_encoding_payload_converters)


def test_data_converter_composition() -> None:
    assert _data_converter(None) is pydantic_data_converter

    with_codec = dataclasses.replace(DataConverter.default, payload_codec=_NoopCodec())
    upgraded = _data_converter(with_codec)
    assert upgraded.payload_converter_class is PydanticPayloadConverter
    assert isinstance(upgraded.payload_codec, _NoopCodec)
    assert upgraded.failure_converter_class is with_codec.failure_converter_class

    assert _data_converter(pydantic_data_converter) is pydantic_data_converter

    custom = DataConverter(payload_converter_class=_CustomPayloadConverter)
    assert _data_converter(custom) is custom


def test_passthrough_modules() -> None:
    runner = _workflow_runner(SandboxedWorkflowRunner())
    assert isinstance(runner, SandboxedWorkflowRunner)
    passthrough = set(runner.restrictions.passthrough_modules)
    assert {
        "agent_framework",
        "pydantic_core",
        "annotated_types",
        "msgspec",
        "typing_inspection",
        "opentelemetry",
        "dotenv",
        "mcp",
        "mcp_types",
    } <= passthrough
    with pytest.raises(ValueError, match="No WorkflowRunner"):
        _workflow_runner(None)


def test_version_and_name() -> None:
    plugin = MicrosoftAgentFrameworkPlugin()
    assert plugin.name() == "microsoft.AgentFrameworkPlugin"
    # The release workflow injects the tag version before the tests run, so
    # the assertion follows the installed distribution.
    assert plugin.version() == importlib.metadata.version(
        "temporalio-microsoft-agent-framework"
    )


def test_warns_on_replay_unsafe_otel_providers(monkeypatch: pytest.MonkeyPatch) -> None:
    trace_sdk = pytest.importorskip("opentelemetry.sdk.trace")
    import opentelemetry.trace

    monkeypatch.setattr(
        opentelemetry.trace, "get_tracer_provider", lambda: trace_sdk.TracerProvider()
    )
    plugin = MicrosoftAgentFrameworkPlugin()
    with pytest.warns(UserWarning, match="TracerProvider is not replay-safe"):
        plugin.configure_worker({"workflow_runner": SandboxedWorkflowRunner()})  # type: ignore[typeddict-item]
    with pytest.warns(UserWarning, match="TracerProvider is not replay-safe"):
        plugin.configure_replayer({"workflow_runner": SandboxedWorkflowRunner()})  # type: ignore[typeddict-item]
