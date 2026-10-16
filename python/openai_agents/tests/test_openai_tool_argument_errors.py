"""Malformed model arguments to activity and Nexus tools reach the model as tool errors.

Like a stock ``function_tool``, ``activity_as_tool`` and ``nexus_operation_as_tool`` return the
argument error text to the model so that it can retry. The workflow must neither fail nor
retry its workflow task forever, and the activity or operation must not run for bad arguments.
With ``failure_error_function=None`` the workflow fails promptly instead.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import nexusrpc
import pytest
from agents import (
    Agent,
    ModelBehaviorError,
    ModelResponse,
    RunContextWrapper,
    Runner,
    Tool,
    _debug,
    default_tool_error_function,
    function_tool,
)
from agents.function_schema import function_schema
from agents.items import ToolCallOutputItem

import temporalio.openai_agents as openai_agents
from temporalio import activity, workflow
from temporalio.api.enums.v1 import EventType
from temporalio.client import Client, WorkflowFailureError
from temporalio.exceptions import ApplicationError
from temporalio.openai_agents import AgentsWorkflowError, ModelActivityParameters
from temporalio.openai_agents.testing import (
    AgentEnvironment,
    ResponseBuilders,
    TestModel,
)
from temporalio.openai_agents.workflow import (
    _parse_tool_arguments,  # type: ignore[reportPrivateUsage]
    _tool_argument_error,  # type: ignore[reportPrivateUsage]
)
from temporalio.testing import WorkflowEnvironment
from tests.helpers import new_worker
from tests.helpers.nexus import make_nexus_endpoint_name

# The worker runs in this process, so activities and operations can record their calls here.
ACTIVITY_CALLS: list[str] = []
OPERATION_CALLS: list[str] = []


@dataclass
class Location:
    city: str
    country: str = "Japan"


@activity.defn
async def forecast(city: str, days: int) -> str:
    """Forecast the weather for a city."""
    ACTIVITY_CALLS.append(f"forecast:{city}:{days}")
    return f"Sunny in {city} for {days} days"


@activity.defn
async def forecast_at(where: Location) -> str:
    """Forecast the weather at a location."""
    ACTIVITY_CALLS.append(f"forecast_at:{where.city}")
    return f"Sunny in {where.city}, {where.country}"


@activity.defn
async def forecast_everywhere() -> str:
    """Forecast the weather everywhere."""
    ACTIVITY_CALLS.append("forecast_everywhere")
    return "Sunny everywhere"


@activity.defn
async def forecast_failure(city: str) -> str:
    """Fail to forecast the weather."""
    ACTIVITY_CALLS.append(f"forecast_failure:{city}")
    raise ApplicationError("No forecast", non_retryable=True)


def custom_error(_ctx: RunContextWrapper[Any], error: Exception) -> str:
    return f"CUSTOM: {type(error).__name__}"


@workflow.defn
class ArgumentErrorsWorkflow:
    @workflow.run
    async def run(self, mode: str) -> list[str]:
        timeout = timedelta(seconds=10)
        tools: list[Tool]
        if mode == "custom":
            tools = [
                openai_agents.workflow.activity_as_tool(
                    forecast,
                    start_to_close_timeout=timeout,
                    failure_error_function=custom_error,
                )
            ]
        else:
            tools = [
                openai_agents.workflow.activity_as_tool(
                    fn, start_to_close_timeout=timeout
                )
                for fn in (
                    forecast,
                    forecast_at,
                    forecast_everywhere,
                    forecast_failure,
                )
            ]
        result = await Runner.run(
            Agent[None](name="Weather", instructions="Be helpful.", tools=tools),
            "What is the weather?",
        )
        outputs = [
            str(item.output)
            for item in result.new_items
            if isinstance(item, ToolCallOutputItem)
        ]
        return outputs + [str(result.final_output)]


@nexusrpc.service
class ForecastService:
    forecast_operation: nexusrpc.Operation[Location, str]


@nexusrpc.handler.service_handler(service=ForecastService)
class ForecastServiceHandler:
    @nexusrpc.handler.sync_operation
    async def forecast_operation(
        self,
        ctx: nexusrpc.handler.StartOperationContext,  # type: ignore[reportUnusedParameter]
        input: Location,
    ) -> str:
        OPERATION_CALLS.append(input.city)
        return f"Sunny in {input.city}, {input.country}"


@workflow.defn
class NexusArgumentErrorsWorkflow:
    @workflow.run
    async def run(self) -> list[str]:
        tool = openai_agents.workflow.nexus_operation_as_tool(
            ForecastService.forecast_operation,
            service=ForecastService,
            endpoint=make_nexus_endpoint_name(workflow.info().task_queue),
            schedule_to_close_timeout=timedelta(seconds=10),
        )
        result = await Runner.run(
            Agent[None](name="Weather", instructions="Be helpful.", tools=[tool]),
            "What is the weather?",
        )
        outputs = [
            str(item.output)
            for item in result.new_items
            if isinstance(item, ToolCallOutputItem)
        ]
        return outputs + [str(result.final_output)]


def _scripted_model(arguments: str, tool_name: str) -> TestModel:
    return TestModel.returning_responses(
        [
            ResponseBuilders.tool_call(arguments, tool_name),
            ResponseBuilders.output_message("recovered"),
        ]
    )


async def _run_workflow(
    client: Client,
    arguments: str,
    tool_name: str,
    wf: Any,
    *args: Any,
    env: WorkflowEnvironment | None = None,
    **worker_kwargs: Any,
) -> list[str]:
    async with AgentEnvironment(
        model=_scripted_model(arguments, tool_name),
        model_params=ModelActivityParameters(
            start_to_close_timeout=timedelta(seconds=30),
        ),
    ) as agent_env:
        client = agent_env.applied_on_client(client)
        async with new_worker(client, wf, **worker_kwargs) as worker:
            if env is not None:
                await env.create_nexus_endpoint(
                    make_nexus_endpoint_name(worker.task_queue), worker.task_queue
                )
            handle = await client.start_workflow(
                wf.run,
                *args,
                id=f"tool-argument-errors-{uuid.uuid4()}",
                task_queue=worker.task_queue,
                execution_timeout=timedelta(seconds=20),
            )
            return await handle.result()


ACTIVITIES: list[Callable[..., Any]] = [
    forecast,
    forecast_at,
    forecast_everywhere,
    forecast_failure,
]

DEFAULT_RUN_ERROR = "An error occurred while running the tool."


@pytest.mark.parametrize(
    "tool_name,arguments,expected",
    [
        pytest.param(
            "forecast",
            '{"city": "Tokyo"}',
            [DEFAULT_RUN_ERROR, "Invalid JSON input for tool forecast"],
            id="missing_argument",
        ),
        pytest.param(
            "forecast",
            '{"city": "Tokyo", "days": "many"}',
            [DEFAULT_RUN_ERROR, "Invalid JSON input for tool forecast"],
            id="wrong_type",
        ),
        pytest.param(
            "forecast",
            '{"city": "Tokyo"',
            [DEFAULT_RUN_ERROR, "Invalid JSON input for tool forecast"],
            id="not_json",
        ),
        pytest.param(
            "forecast",
            "[1, 2]",
            [DEFAULT_RUN_ERROR, "expected a JSON object"],
            id="json_list",
        ),
        pytest.param(
            "forecast",
            "null",
            [DEFAULT_RUN_ERROR, "expected a JSON object"],
            id="json_null",
        ),
        pytest.param(
            "forecast",
            "",
            [DEFAULT_RUN_ERROR, "Invalid JSON input for tool forecast"],
            id="empty_string",
        ),
        pytest.param(
            "forecast_at",
            '{"where": {"country": "Japan"}}',
            [DEFAULT_RUN_ERROR, "Invalid JSON input for tool forecast_at"],
            id="nested_missing_field",
        ),
        pytest.param(
            "forecast_at",
            '{"where": "Tokyo"}',
            [DEFAULT_RUN_ERROR, "Invalid JSON input for tool forecast_at"],
            id="nested_wrong_type",
        ),
    ],
)
async def test_activity_tool_argument_errors_reach_model(
    client: Client, tool_name: str, arguments: str, expected: list[str]
):
    ACTIVITY_CALLS.clear()
    result = await _run_workflow(
        client,
        arguments,
        tool_name,
        ArgumentErrorsWorkflow,
        "default",
        activities=ACTIVITIES,
    )
    *tool_outputs, final = result
    assert final == "recovered"
    assert len(tool_outputs) == 1
    for text in expected:
        assert text in tool_outputs[0]
    assert ACTIVITY_CALLS == []


@pytest.mark.parametrize(
    "tool_name,arguments,expected_output,expected_call",
    [
        pytest.param(
            "forecast",
            '{"city": "Tokyo", "days": 3}',
            "Sunny in Tokyo for 3 days",
            "forecast:Tokyo:3",
            id="valid",
        ),
        pytest.param(
            "forecast",
            '{"city": "Tokyo", "days": "3", "extra": true}',
            "Sunny in Tokyo for 3 days",
            "forecast:Tokyo:3",
            id="coerced_and_unknown_argument",
        ),
        pytest.param(
            "forecast_at",
            '{"where": {"city": "Kyoto"}}',
            "Sunny in Kyoto, Japan",
            "forecast_at:Kyoto",
            id="nested",
        ),
        pytest.param(
            "forecast_everywhere",
            "{}",
            "Sunny everywhere",
            "forecast_everywhere",
            id="no_parameters",
        ),
        pytest.param(
            "forecast_everywhere",
            "",
            "Sunny everywhere",
            "forecast_everywhere",
            id="no_parameters_empty_string",
        ),
    ],
)
async def test_activity_tool_valid_arguments_run_activity(
    client: Client,
    tool_name: str,
    arguments: str,
    expected_output: str,
    expected_call: str,
):
    ACTIVITY_CALLS.clear()
    result = await _run_workflow(
        client,
        arguments,
        tool_name,
        ArgumentErrorsWorkflow,
        "default",
        activities=ACTIVITIES,
    )
    assert result == [expected_output, "recovered"]
    assert ACTIVITY_CALLS == [expected_call]


async def test_activity_tool_activity_failure_still_fails_workflow(client: Client):
    ACTIVITY_CALLS.clear()
    with pytest.raises(WorkflowFailureError) as e:
        await _run_workflow(
            client,
            '{"city": "Tokyo"}',
            "forecast_failure",
            ArgumentErrorsWorkflow,
            "default",
            activities=ACTIVITIES,
        )
    cause = e.value.cause
    assert isinstance(cause, ApplicationError)
    assert cause.type == AgentsWorkflowError.__name__
    assert "Workflow failure exception in Agents Framework" in cause.message
    assert ACTIVITY_CALLS == ["forecast_failure:Tokyo"]


async def test_activity_tool_custom_failure_error_function(client: Client):
    ACTIVITY_CALLS.clear()
    result = await _run_workflow(
        client,
        '{"city": "Tokyo"}',
        "forecast",
        ArgumentErrorsWorkflow,
        "custom",
        activities=ACTIVITIES,
    )
    assert result == ["CUSTOM: ModelBehaviorError", "recovered"]
    assert ACTIVITY_CALLS == []


@pytest.mark.requires_local_server
@pytest.mark.parametrize(
    "arguments,expected",
    [
        pytest.param(
            '{"input": {"country": "Japan"}}',
            [DEFAULT_RUN_ERROR, "Invalid JSON input for tool forecast_operation"],
            id="missing_field",
        ),
        pytest.param(
            '{"input": {"city": 3}}',
            [DEFAULT_RUN_ERROR, "Invalid JSON input for tool forecast_operation"],
            id="wrong_type",
        ),
        pytest.param(
            '{"input": ',
            [DEFAULT_RUN_ERROR, "Invalid JSON input for tool forecast_operation"],
            id="not_json",
        ),
        pytest.param(
            "[1]",
            [
                DEFAULT_RUN_ERROR,
                "Invalid JSON input for tool forecast_operation: expected a JSON object",
            ],
            id="json_list",
        ),
        pytest.param(
            "",
            [DEFAULT_RUN_ERROR, "Invalid JSON input for tool forecast_operation"],
            id="empty_string",
        ),
    ],
)
async def test_nexus_tool_argument_errors_reach_model(
    client: Client, env: WorkflowEnvironment, arguments: str, expected: list[str]
):
    if env.supports_time_skipping:
        pytest.skip("Nexus tests don't work with time-skipping server")
    OPERATION_CALLS.clear()
    result = await _run_workflow(
        client,
        arguments,
        "forecast_operation",
        NexusArgumentErrorsWorkflow,
        env=env,
        nexus_service_handlers=[ForecastServiceHandler()],
    )
    *tool_outputs, final = result
    assert final == "recovered"
    assert len(tool_outputs) == 1
    for text in expected:
        assert text in tool_outputs[0]
    assert OPERATION_CALLS == []


@pytest.mark.requires_local_server
async def test_nexus_tool_valid_arguments_run_operation(
    client: Client, env: WorkflowEnvironment
):
    if env.supports_time_skipping:
        pytest.skip("Nexus tests don't work with time-skipping server")
    OPERATION_CALLS.clear()
    result = await _run_workflow(
        client,
        '{"input": {"city": "Tokyo"}}',
        "forecast_operation",
        NexusArgumentErrorsWorkflow,
        env=env,
        nexus_service_handlers=[ForecastServiceHandler()],
    )
    assert result == ["Sunny in Tokyo, Japan", "recovered"]
    assert OPERATION_CALLS == ["Tokyo"]


BAD_ARGUMENTS = [
    '{"city": "Tokyo"}',
    '{"city": "Tokyo", "days": "many"}',
    "{",
    "[]",
    "3",
    "",
]


def test_parse_tool_arguments():
    schema = function_schema(forecast)
    parsed = _parse_tool_arguments(schema, '{"city": "Tokyo", "days": "3"}')
    assert schema.to_call_args(parsed)[0] == ["Tokyo", 3]
    for bad in [*BAD_ARGUMENTS, None]:
        with pytest.raises(ModelBehaviorError) as e:
            _parse_tool_arguments(schema, bad)
        assert "Invalid JSON input for tool forecast" in str(e.value)


@pytest.mark.parametrize("dont_log_tool_data", [True, False])
@pytest.mark.parametrize("arguments", BAD_ARGUMENTS)
async def test_argument_error_text_matches_function_tool(
    monkeypatch: pytest.MonkeyPatch, arguments: str, dont_log_tool_data: bool
):
    """The model sees the same text as it would from a stock function_tool."""
    monkeypatch.setattr(_debug, "DONT_LOG_TOOL_DATA", dont_log_tool_data)

    @function_tool
    async def forecast(city: str, days: int) -> str:
        return f"{city}:{days}"

    async def tool_output(tool: Tool) -> str:
        result = await Runner.run(
            Agent[None](
                name="Weather",
                model=_scripted_model(arguments, "forecast"),
                tools=[tool],
            ),
            "What is the weather?",
        )
        [output] = [
            item.output
            for item in result.new_items
            if isinstance(item, ToolCallOutputItem)
        ]
        return str(output)

    temporal_tool = openai_agents.workflow.activity_as_tool(
        globals()["forecast"], start_to_close_timeout=timedelta(seconds=10)
    )
    expected = await tool_output(forecast)
    assert expected.startswith("An error occurred while ")
    assert await tool_output(temporal_tool) == expected


async def test_tool_argument_error_without_formatter_raises_application_error():
    schema = function_schema(forecast)
    with pytest.raises(ModelBehaviorError) as e:
        _parse_tool_arguments(schema, "{")
    with pytest.raises(ApplicationError) as raised:
        await _tool_argument_error(None, RunContextWrapper(None), e.value)
    assert raised.value.type == "ModelBehaviorError"
    assert raised.value.non_retryable
    assert raised.value.message == str(e.value)
    assert raised.value.__cause__ is e.value


@dataclass
class RunInfo:
    token: str


@workflow.defn
class ToolArgumentModesWorkflow:
    @workflow.run
    async def run(self, kind: str, mode: str) -> list[str]:
        run_info = RunInfo(token=workflow.info().workflow_id)

        async def async_error(ctx: RunContextWrapper[Any], error: Exception) -> str:
            await workflow.sleep(timedelta(milliseconds=10))
            return (
                f"ASYNC {type(error).__name__} token={ctx.context.token} "
                f"wrapper={isinstance(ctx, RunContextWrapper)} "
                f"same_context={ctx.context is run_info}"
            )

        failure_error_function: Any = {
            "default": default_tool_error_function,
            "none": None,
            "async": async_error,
        }[mode]
        tool: Tool
        if kind == "activity":
            tool = openai_agents.workflow.activity_as_tool(
                forecast,
                start_to_close_timeout=timedelta(seconds=10),
                failure_error_function=failure_error_function,
            )
        else:
            tool = openai_agents.workflow.nexus_operation_as_tool(
                ForecastService.forecast_operation,
                service=ForecastService,
                endpoint=make_nexus_endpoint_name(workflow.info().task_queue),
                schedule_to_close_timeout=timedelta(seconds=10),
                failure_error_function=failure_error_function,
            )
        result = await Runner.run(
            Agent[RunInfo](name="Weather", instructions="Be helpful.", tools=[tool]),
            "What is the weather?",
            context=run_info,
        )
        outputs = [
            str(item.output)
            for item in result.new_items
            if isinstance(item, ToolCallOutputItem)
        ]
        return outputs + [str(result.final_output)]


class RecordingModel(TestModel):
    """Scripted model that records the input of every turn."""

    def __init__(self, responses: list[ModelResponse]) -> None:
        responses_iter = iter(responses)
        super().__init__(lambda: next(responses_iter))
        self.inputs: list[str] = []

    async def get_response(  # type: ignore[override]
        self, system_instructions: Any, input: Any, *args: Any, **kwargs: Any
    ) -> ModelResponse:
        self.inputs.append(json.dumps(input, default=str))
        return await super().get_response(system_instructions, input, *args, **kwargs)


@dataclass
class ToolCase:
    tool_name: str
    missing_argument: str
    malformed_json: str
    valid: str
    expected_output: str
    expected_call: str


TOOL_CASES = {
    "activity": ToolCase(
        tool_name="forecast",
        missing_argument='{"city": "Tokyo"}',
        malformed_json='{"city": "Tokyo", "days": 3',
        valid='{"city": "Tokyo", "days": 3}',
        expected_output="Sunny in Tokyo for 3 days",
        expected_call="forecast:Tokyo:3",
    ),
    "nexus": ToolCase(
        tool_name="forecast_operation",
        missing_argument='{"input": {"country": "Japan"}}',
        malformed_json='{"input": {"city": "Tokyo"',
        valid='{"input": {"city": "Tokyo"}}',
        expected_output="Sunny in Tokyo, Japan",
        expected_call="Tokyo",
    ),
}

TOOL_KINDS = [
    pytest.param("activity", id="activity"),
    pytest.param("nexus", id="nexus", marks=pytest.mark.requires_local_server),
]


@dataclass
class ModeRun:
    result: list[str] | WorkflowFailureError
    event_types: list[EventType.ValueType]
    model: RecordingModel
    workflow_id: str
    calls: list[str]


async def _run_mode(
    client: Client,
    env: WorkflowEnvironment,
    kind: str,
    mode: str,
    responses: list[ModelResponse],
) -> ModeRun:
    if kind == "nexus" and env.supports_time_skipping:
        pytest.skip("Nexus tests don't work with time-skipping server")
    calls = ACTIVITY_CALLS if kind == "activity" else OPERATION_CALLS
    calls.clear()
    model = RecordingModel(responses)
    async with AgentEnvironment(
        model=model,
        model_params=ModelActivityParameters(
            start_to_close_timeout=timedelta(seconds=30),
        ),
    ) as agent_env:
        client = agent_env.applied_on_client(client)
        async with new_worker(
            client,
            ToolArgumentModesWorkflow,
            activities=ACTIVITIES,
            nexus_service_handlers=[ForecastServiceHandler()],
        ) as worker:
            if kind == "nexus":
                await env.create_nexus_endpoint(
                    make_nexus_endpoint_name(worker.task_queue), worker.task_queue
                )
            handle = await client.start_workflow(
                ToolArgumentModesWorkflow.run,
                args=[kind, mode],
                id=f"tool-argument-modes-{uuid.uuid4()}",
                task_queue=worker.task_queue,
                execution_timeout=timedelta(seconds=15),
            )
            result: list[str] | WorkflowFailureError
            try:
                result = await handle.result()
            except WorkflowFailureError as e:
                result = e
            history = await handle.fetch_history()
            return ModeRun(
                result=result,
                event_types=[event.event_type for event in history.events],
                model=model,
                workflow_id=handle.id,
                calls=list(calls),
            )


@pytest.mark.parametrize("kind", TOOL_KINDS)
async def test_corrected_tool_call_runs_after_invalid_one(
    client: Client, env: WorkflowEnvironment, kind: str
):
    case = TOOL_CASES[kind]
    run = await _run_mode(
        client,
        env,
        kind,
        "default",
        [
            ResponseBuilders.tool_call(
                case.missing_argument, case.tool_name, call_id="call-1"
            ),
            ResponseBuilders.tool_call(case.valid, case.tool_name, call_id="call-2"),
            ResponseBuilders.output_message("done"),
        ],
    )
    assert isinstance(run.result, list)
    error_output, tool_output, final = run.result
    assert DEFAULT_RUN_ERROR in error_output
    assert tool_output == case.expected_output
    assert final == "done"
    assert run.calls == [case.expected_call]
    assert len(run.model.inputs) == 3
    assert DEFAULT_RUN_ERROR in run.model.inputs[1]
    assert case.expected_output in run.model.inputs[2]
    assert EventType.EVENT_TYPE_WORKFLOW_TASK_FAILED not in run.event_types
    assert EventType.EVENT_TYPE_WORKFLOW_EXECUTION_COMPLETED in run.event_types


@pytest.mark.parametrize("kind", TOOL_KINDS)
@pytest.mark.parametrize("bad_arguments", ["missing_argument", "malformed_json"])
async def test_no_failure_error_function_fails_workflow(
    client: Client, env: WorkflowEnvironment, kind: str, bad_arguments: str
):
    case = TOOL_CASES[kind]
    run = await _run_mode(
        client,
        env,
        kind,
        "none",
        [
            ResponseBuilders.tool_call(
                getattr(case, bad_arguments), case.tool_name, call_id="call-1"
            ),
            ResponseBuilders.output_message("unreachable"),
        ],
    )
    assert isinstance(run.result, WorkflowFailureError)
    agents_error = run.result.cause
    assert isinstance(agents_error, ApplicationError)
    assert agents_error.type == AgentsWorkflowError.__name__
    assert f"Invalid JSON input for tool {case.tool_name}" in agents_error.message
    argument_error = agents_error.cause
    assert isinstance(argument_error, ApplicationError)
    assert argument_error.type == "ModelBehaviorError"
    assert argument_error.non_retryable
    assert argument_error.message == f"Invalid JSON input for tool {case.tool_name}"
    assert run.calls == []
    assert len(run.model.inputs) == 1
    assert EventType.EVENT_TYPE_WORKFLOW_TASK_FAILED not in run.event_types
    assert EventType.EVENT_TYPE_WORKFLOW_EXECUTION_FAILED in run.event_types
    assert EventType.EVENT_TYPE_WORKFLOW_EXECUTION_TIMED_OUT not in run.event_types


@pytest.mark.parametrize("kind", TOOL_KINDS)
async def test_async_failure_error_function_output_reaches_model(
    client: Client, env: WorkflowEnvironment, kind: str
):
    case = TOOL_CASES[kind]
    run = await _run_mode(
        client,
        env,
        kind,
        "async",
        [
            ResponseBuilders.tool_call(
                case.missing_argument, case.tool_name, call_id="call-1"
            ),
            ResponseBuilders.output_message("recovered"),
        ],
    )
    assert isinstance(run.result, list)
    expected = (
        f"ASYNC ModelBehaviorError token={run.workflow_id} "
        "wrapper=True same_context=True"
    )
    assert run.result == [expected, "recovered"]
    assert len(run.model.inputs) == 2
    assert expected in run.model.inputs[1]
    assert run.calls == []
    assert EventType.EVENT_TYPE_WORKFLOW_TASK_FAILED not in run.event_types
