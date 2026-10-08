"""FunctionTool metadata must survive the trip to the model activity.

The model activity only sees a data-conversion friendly copy of each tool, so any
FunctionTool field that is not copied is silently missing from the Responses request.
These tests run the SDK's real Responses tool converter on the original tools and on the
tools rebuilt inside the activity, and require identical payloads.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from datetime import timedelta
from typing import Any, cast

import pytest
from agents import (
    Agent,
    AgentOutputSchemaBase,
    FunctionTool,
    Handoff,
    Model,
    ModelResponse,
    ModelSettings,
    ModelTracing,
    Runner,
    Tool,
    ToolSearchTool,
    TResponseInputItem,
    function_tool,
    tool_namespace,
)
from agents.computer import Button, Computer
from agents.items import TResponseStreamEvent
from agents.models.openai_responses import Converter
from agents.tool import (
    ComputerTool,
    LocalShellTool,
    ProgrammaticToolCallingTool,
)

from temporalio import workflow
from temporalio.api.common.v1 import Payload
from temporalio.client import Client, WorkflowFailureError
from temporalio.common import RetryPolicy
from temporalio.converter import DataConverter
from temporalio.exceptions import ApplicationError
from temporalio.openai_agents import AgentsWorkflowError, ModelActivityParameters
from temporalio.openai_agents._invoke_model_activity import (
    ActivityModelInput,
    FunctionToolInput,
    _build_tool,
    _build_tools_and_handoffs,
)
from temporalio.openai_agents._temporal_model_stub import _TemporalModelStub
from temporalio.openai_agents._temporal_openai_agents import OpenAIPayloadConverter
from temporalio.openai_agents._temporal_worker_env_ref import _WorkerEnvRefResolver
from temporalio.openai_agents.testing import AgentEnvironment, ResponseBuilders
from tests.helpers import new_worker


def _plain_tool() -> FunctionTool:
    @function_tool
    def plain(x: int) -> str:
        """A plain tool."""
        return str(x)

    return plain


def _deferred_tool() -> FunctionTool:
    @function_tool(defer_loading=True)
    def deferred(x: int) -> str:
        """A deferred tool."""
        return str(x)

    return deferred


def _namespaced_tools(*, with_deferred: bool = True) -> list[FunctionTool]:
    return tool_namespace(
        name="crm",
        description="Customer relationship tools",
        tools=[_plain_tool(), *([_deferred_tool()] if with_deferred else [])],
    )


def _programmatic_tool() -> FunctionTool:
    @function_tool(allowed_callers=["direct", "programmatic"])
    def programmatic(x: int) -> str:
        """A tool that generated code may call."""
        return str(x)

    return programmatic


def _output_schema_tool() -> FunctionTool:
    @function_tool(
        output_json_schema={
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        }
    )
    def with_output(x: int) -> str:
        """A tool with a declared output schema."""
        return str(x)

    return with_output


# Each case maps a name to a function returning the tools of one agent.
TOOL_CASES: dict[str, Any] = {
    "plain": lambda: [_plain_tool()],
    "defer_loading": lambda: [_deferred_tool(), ToolSearchTool()],
    "namespace": lambda: _namespaced_tools(with_deferred=False),
    "namespace_with_tool_search": lambda: [*_namespaced_tools(), ToolSearchTool()],
    "allowed_callers": lambda: [_programmatic_tool()],
    "output_json_schema": lambda: [_output_schema_tool()],
    "everything": lambda: [
        *_namespaced_tools(),
        _deferred_tool(),
        _programmatic_tool(),
        _output_schema_tool(),
        ToolSearchTool(),
    ],
}


def _responses_tools(tools: list[Tool]) -> list[dict[str, Any]]:
    """The Responses request tool payload, as plain dicts."""
    return cast(list[dict[str, Any]], Converter.convert_tools(tools, []).tools)


def _stub() -> _TemporalModelStub:
    return _TemporalModelStub(
        model_name="gpt-5",
        model_params=ModelActivityParameters(),
        agent=None,
    )


def _build_input(tools: list[Tool]) -> ActivityModelInput:
    activity_input, _summary = _stub()._build_activity_input(
        system_instructions=None,
        input="hi",
        model_settings=ModelSettings(),
        tools=tools,
        output_schema=None,
        handoffs=[],
        tracing=ModelTracing.DISABLED,
        previous_response_id=None,
        conversation_id=None,
        prompt=None,
    )
    return activity_input


async def _through_data_converter(
    activity_input: ActivityModelInput,
) -> ActivityModelInput:
    converter = DataConverter(payload_converter_class=OpenAIPayloadConverter)
    payloads = await converter.encode([activity_input])
    (decoded,) = await converter.decode(payloads, [ActivityModelInput])
    return decoded


@pytest.mark.parametrize("case", sorted(TOOL_CASES))
async def test_function_tool_responses_payload_survives_activity_input(case: str):
    tools: list[Tool] = TOOL_CASES[case]()
    decoded = await _through_data_converter(_build_input(tools))
    rebuilt, _handoffs = _build_tools_and_handoffs(decoded, _WorkerEnvRefResolver(()))

    assert _responses_tools(rebuilt) == _responses_tools(tools)
    assert (
        Converter.convert_tools(rebuilt, []).includes
        == Converter.convert_tools(tools, []).includes
    )


async def test_function_tool_metadata_is_present_in_responses_payload():
    # Guards the test above against comparing two payloads that are both missing the metadata.
    tools: list[Tool] = TOOL_CASES["everything"]()
    decoded = await _through_data_converter(_build_input(tools))
    rebuilt, _handoffs = _build_tools_and_handoffs(decoded, _WorkerEnvRefResolver(()))
    payload = _responses_tools(rebuilt)

    namespace = next(x for x in payload if x.get("type") == "namespace")
    assert namespace["name"] == "crm"
    assert namespace["description"] == "Customer relationship tools"
    inner = {x["name"]: x for x in namespace["tools"]}
    assert "defer_loading" not in inner["plain"]
    assert inner["deferred"]["defer_loading"] is True

    top_level = {x["name"]: x for x in payload if x.get("type") == "function"}
    assert top_level["deferred"]["defer_loading"] is True
    assert top_level["programmatic"]["allowed_callers"] == ["direct", "programmatic"]
    assert top_level["with_output"]["output_schema"]["properties"] == {
        "answer": {"type": "string"}
    }


def test_function_tool_metadata_round_trips_through_activity_input():
    tool = _namespaced_tools()[1]
    (tool_input,) = _build_input([tool]).get("tools") or []
    assert isinstance(tool_input, FunctionToolInput)
    assert tool_input.defer_loading is True
    assert tool_input.namespace == "crm"
    assert tool_input.namespace_description == "Customer relationship tools"

    rebuilt = _build_tool(tool_input, _WorkerEnvRefResolver(()))
    assert isinstance(rebuilt, FunctionTool)
    assert rebuilt.defer_loading is True
    assert rebuilt.qualified_name == tool.qualified_name == "crm.deferred"


async def test_function_tool_input_without_new_fields_still_decodes():
    # The shape recorded in workflow histories before the metadata fields existed.
    old_input: dict[str, Any] = {
        "input": "hi",
        "model_settings": {},
        "tracing": 0,
        "tools": [
            {
                "name": "old_tool",
                "description": "Recorded before metadata was carried",
                "params_json_schema": {
                    "type": "object",
                    "properties": {"x": {"type": "integer"}},
                    "required": ["x"],
                    "additionalProperties": False,
                },
                "strict_json_schema": True,
            }
        ],
    }
    payload = Payload(
        metadata={"encoding": b"json/plain"}, data=json.dumps(old_input).encode()
    )
    converter = DataConverter(payload_converter_class=OpenAIPayloadConverter)
    (decoded,) = await converter.decode([payload], [ActivityModelInput])

    (tool_input,) = decoded["tools"]
    assert isinstance(tool_input, FunctionToolInput)
    assert tool_input.defer_loading is False
    assert tool_input.namespace is None
    assert tool_input.namespace_description is None
    assert tool_input.allowed_callers is None
    assert tool_input.output_json_schema is None

    (rebuilt,), _ = _build_tools_and_handoffs(decoded, _WorkerEnvRefResolver(()))
    (converted,) = _responses_tools([rebuilt])
    assert converted == {
        "name": "old_tool",
        "description": "Recorded before metadata was carried",
        "parameters": old_input["tools"][0]["params_json_schema"],
        "strict": True,
        "type": "function",
    }


class _ResponsesConvertingModel(Model):
    """Runs the SDK's real Responses tool conversion, as the OpenAI model does in the activity.

    The converted tool payload is returned as the message text so the workflow can hand it back
    to the test. Conversion raises exactly where a real request would be rejected.
    """

    async def get_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
        **kwargs: Any,
    ) -> ModelResponse:
        converted = Converter.convert_tools(tools, handoffs)
        return ResponseBuilders.output_message(json.dumps(converted.tools))

    def stream_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
        **kwargs: Any,
    ) -> AsyncIterator[TResponseStreamEvent]:
        raise NotImplementedError()


@workflow.defn
class ToolSearchWorkflow:
    @workflow.run
    async def run(self, case: str) -> str:
        agent = Agent[None](
            name="tool-search-agent", instructions="x", tools=TOOL_CASES[case]()
        )
        result = await Runner.run(starting_agent=agent, input="hi")
        return result.final_output


@pytest.mark.parametrize("case", ["defer_loading", "namespace_with_tool_search"])
async def test_deferred_function_tool_with_tool_search_workflow(
    client: Client, case: str
):
    async with AgentEnvironment(
        model=_ResponsesConvertingModel(),
        model_params=ModelActivityParameters(
            start_to_close_timeout=timedelta(seconds=10),
            retry_policy=RetryPolicy(maximum_attempts=1),
        ),
    ) as env:
        client = env.applied_on_client(client)
        async with new_worker(client, ToolSearchWorkflow) as worker:
            result = await client.execute_workflow(
                ToolSearchWorkflow.run,
                case,
                id=f"tool-search-{uuid.uuid4()}",
                task_queue=worker.task_queue,
                execution_timeout=timedelta(seconds=30),
            )
    assert json.loads(result) == _responses_tools(TOOL_CASES[case]())


class _FakeComputer(Computer):
    def screenshot(self) -> str:
        return ""

    def click(self, x: int, y: int, button: Button) -> None: ...

    def double_click(self, x: int, y: int) -> None: ...

    def scroll(self, x: int, y: int, scroll_x: int, scroll_y: int) -> None: ...

    def type(self, text: str) -> None: ...

    def wait(self) -> None: ...

    def move(self, x: int, y: int) -> None: ...

    def keypress(self, keys: list[str]) -> None: ...

    def drag(self, path: list[tuple[int, int]]) -> None: ...


async def _local_shell_executor(_request: Any) -> str:
    return ""


UNSUPPORTED_TOOLS: dict[str, Any] = {
    "local_shell": lambda: LocalShellTool(executor=_local_shell_executor),
    "computer": lambda: ComputerTool(computer=_FakeComputer()),
    "programmatic_tool_calling": lambda: ProgrammaticToolCallingTool(),
}


@pytest.mark.parametrize("name", sorted(UNSUPPORTED_TOOLS))
def test_unsupported_tool_is_rejected_when_building_activity_input(name: str):
    with pytest.raises(AgentsWorkflowError, match="not supported"):
        _build_input([UNSUPPORTED_TOOLS[name]()])


@workflow.defn
class UnsupportedToolWorkflow:
    @workflow.run
    async def run(self, name: str) -> str:
        agent = Agent[None](
            name="unsupported-tool-agent",
            instructions="x",
            tools=[UNSUPPORTED_TOOLS[name]()],
        )
        result = await Runner.run(starting_agent=agent, input="hi")
        return result.final_output


@pytest.mark.parametrize("name", sorted(UNSUPPORTED_TOOLS))
async def test_unsupported_tool_fails_workflow_instead_of_hanging(
    client: Client, name: str
):
    async with AgentEnvironment(model=_ResponsesConvertingModel()) as env:
        client = env.applied_on_client(client)
        async with new_worker(client, UnsupportedToolWorkflow) as worker:
            handle = await client.start_workflow(
                UnsupportedToolWorkflow.run,
                name,
                id=f"unsupported-tool-{uuid.uuid4()}",
                task_queue=worker.task_queue,
                # A workflow-task failure would retry until this timeout instead.
                execution_timeout=timedelta(seconds=60),
            )
            with pytest.raises(WorkflowFailureError) as err:
                await handle.result()

    cause = err.value.cause
    assert isinstance(cause, ApplicationError)
    assert cause.type == "AgentsWorkflowError"
    assert "not supported by the Temporal OpenAI Agents plugin" in cause.message
