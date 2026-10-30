"""The tool seam: existing activities and wrapped tools route to activities.

These need LangChain (a tool is a ``BaseTool``) and guard on its import. Each
runs a real workflow that invokes the wrapped tool and asserts it scheduled the
expected activity.
"""

from __future__ import annotations

import sys
import uuid
from collections.abc import Sequence
from datetime import timedelta
from types import SimpleNamespace
from typing import Annotated, Any

import pytest

from temporalio.testing import WorkflowEnvironment

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 11), reason="deepagents requires Python >= 3.11"
)

pytest.importorskip("langchain_core")

from temporalio import activity, workflow  # noqa: E402
from temporalio.worker import Replayer, Worker  # noqa: E402
from temporalio.workflow import ActivityConfig  # noqa: E402
from tests.helpers.deepagents import (  # noqa: E402
    count_scheduled_activities,
    scheduled_activity_attrs,
)

with workflow.unsafe.imports_passed_through():
    from temporalio.deepagents import (  # noqa: E402
        DeepAgentsPlugin,
        activity_as_tool,
        run_deep_agent,
        tool_as_activity,
    )
    from temporalio.deepagents._tools import warn_unwrapped_tools  # noqa: E402
    from temporalio.deepagents.testing import mock_model_provider  # noqa: E402

INVOKE_TOOL = "deepagents.invoke_tool"
# Dedicated queue for tool activities, so tests can prove task-queue routing.
TOOL_TASK_QUEUE = "da-tool-queue"

# Bind deepagents symbols off the module importorskip returns: a static
# `from deepagents import ...` cannot resolve on Python 3.10 (deepagents
# needs >= 3.11), and with the package absent the type checkers mis-resolve
# the name against this same-named test directory.
create_deep_agent = pytest.importorskip("deepagents").create_deep_agent

from langchain_core.messages import HumanMessage, ToolMessage  # noqa: E402
from langchain_core.tools import tool  # noqa: E402
from langgraph.prebuilt import InjectedState  # noqa: E402
from pydantic import BaseModel, ConfigDict  # noqa: E402


def pairing_weather(city: str) -> str:
    """Return the weather for a city."""
    return f"weather:{city}"


@tool
def last_user_message(state: Annotated[dict, InjectedState]) -> str:
    """Return the last user message from the conversation."""
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            return f"user said: {message.content}"
    return "No user messages found"


@tool
def message_summary(state: Annotated[dict, InjectedState]) -> str:
    """List the conversation's message types and contents."""
    return ",".join(
        f"{type(m).__name__}:{getattr(m, 'content', '?')}" for m in state["messages"]
    )


@tool
def saved_artifact(state: Annotated[dict, InjectedState]) -> str:
    """Return the first tool result's artifact."""
    for message in state["messages"]:
        if isinstance(message, ToolMessage):
            artifact = message.artifact
            return f"{type(artifact).__name__}:{artifact['page_content']}"
    return "No tool results found"


@tool
def echo_record(record: dict) -> str:
    """Echo a record's tenant and marker fields."""
    return f"{record['tenant']}:{record['__deepagents_tool_arg__']}"


class Record(BaseModel):
    model_config = ConfigDict(extra="allow")
    tenant: str


@tool
def echo_model(record: Record) -> str:
    """Echo a typed record's tenant and marker fields."""
    return f"{record.tenant}:{(record.model_extra or {})['__deepagents_tool_arg__']}"


@activity.defn
async def echo_activity(text: str) -> str:
    return f"echo:{text}"


@activity.defn
async def forecast_activity(city: str, days: int, units: str = "C") -> str:
    return f"{city}:{days}:{units}"


@activity.defn
async def quota_activity() -> str:
    return "quota:42"


@workflow.defn
class ActivityAsToolWorkflow:
    @workflow.run
    async def run(self, text: str) -> str:
        tool = activity_as_tool(
            echo_activity, start_to_close_timeout=timedelta(seconds=10)
        )
        return await tool.ainvoke({"text": text})


@workflow.defn
class ActivityParametersAsToolWorkflow:
    @workflow.run
    async def run(self) -> list[str]:
        forecast = activity_as_tool(
            forecast_activity, start_to_close_timeout=timedelta(seconds=10)
        )
        quota = activity_as_tool(
            quota_activity, start_to_close_timeout=timedelta(seconds=10)
        )
        return [
            await forecast.ainvoke({"city": "Paris", "days": 3}),
            await forecast.ainvoke({"city": "Oslo", "days": 1, "units": "F"}),
            await quota.ainvoke({}),
        ]


@workflow.defn
class ToolAsActivityWorkflow:
    @workflow.run
    async def run(self, city: str) -> str:
        def get_weather(city: str) -> str:
            """Look up the weather for a city."""
            return f"sunny in {city}"

        tool = tool_as_activity(
            get_weather, start_to_close_timeout=timedelta(seconds=10)
        )
        # The wrapper returns plain CONTENT (the tool node stamps the model's
        # tool_call_id onto it) — not a pre-built ToolMessage.
        return str(await tool.ainvoke({"city": city}))


@pytest.mark.asyncio
async def test_activity_as_tool(env: WorkflowEnvironment) -> None:
    plugin = DeepAgentsPlugin()
    async with Worker(
        env.client,
        task_queue="da-act-tool",
        workflows=[ActivityAsToolWorkflow],
        activities=[echo_activity],
        plugins=[plugin],
    ):
        handle = await env.client.start_workflow(
            ActivityAsToolWorkflow.run,
            "hi",
            id=f"da-act-tool-{uuid.uuid4()}",
            task_queue="da-act-tool",
        )
        out = await handle.result()

    assert out == "echo:hi"
    counts = await count_scheduled_activities(handle)
    assert counts["echo_activity"] == 1, counts
    # Without activity_options the scheduled attributes keep exactly the shape
    # this helper always produced — the workflow's own task queue, a zero
    # heartbeat timeout, the required start-to-close (retry_policy is forwarded
    # verbatim, None included, so it records the SDK default as before) — so
    # call sites that predate the argument record (and replay, below)
    # unchanged.
    attrs = await scheduled_activity_attrs(handle, "echo_activity")
    assert attrs is not None
    assert attrs.task_queue.name == "da-act-tool"
    assert attrs.heartbeat_timeout.ToTimedelta() == timedelta(0)
    assert attrs.start_to_close_timeout.ToTimedelta() == timedelta(seconds=10)
    await Replayer(
        workflows=[ActivityAsToolWorkflow], plugins=[DeepAgentsPlugin()]
    ).replay_workflow(await handle.fetch_history())


def test_activity_as_tool_accepts_activity_options() -> None:
    """The helper takes activity_options for the tool seam's other options.
    Constructed outside a workflow so that, on code without the argument, the
    failure is the TypeError itself rather than a failing workflow task; the
    end-to-end scheduling is covered below.
    """
    tool = activity_as_tool(
        echo_activity,
        start_to_close_timeout=timedelta(seconds=10),
        activity_options={"task_queue": TOOL_TASK_QUEUE},
    )
    assert tool.name == "echo_activity"


@pytest.mark.parametrize(
    "option",
    ["start_to_close_timeout", "retry_policy", "summary", "activity_id", "arg", "args"],
)
def test_activity_as_tool_rejects_owned_activity_options(option: str) -> None:
    """Options the helper owns are rejected at construction: silently dropping
    a caller's start_to_close_timeout / retry_policy / summary hides the
    mistake in history, arg/args would collide with the arguments built from
    the model's call, and one static activity_id cannot serve every invocation
    of the tool.
    """
    with pytest.raises(ValueError, match=option):
        activity_as_tool(
            echo_activity,
            start_to_close_timeout=timedelta(seconds=10),
            activity_options={option: "reserved"},
        )


@workflow.defn
class ActivityAsToolOptionsWorkflow:
    @workflow.run
    async def run(self, text: str) -> list[str]:
        options = ActivityConfig(
            heartbeat_timeout=timedelta(seconds=3),
            task_queue=TOOL_TASK_QUEUE,
        )
        tool = activity_as_tool(
            echo_activity,
            start_to_close_timeout=timedelta(seconds=10),
            activity_options=options,
        )
        # A multi-parameter activity with a defaulted parameter: the options
        # compose with the per-parameter argument mapping.
        forecast = activity_as_tool(
            forecast_activity,
            start_to_close_timeout=timedelta(seconds=10),
            activity_options=options,
        )
        return [
            await tool.ainvoke({"text": text}),
            await forecast.ainvoke({"city": "Paris", "days": 3}),
        ]


@pytest.mark.asyncio
async def test_activity_as_tool_activity_options(env: WorkflowEnvironment) -> None:
    """activity_options carries the full ActivityConfig, not just the timeout:
    the tool activity lands on the routed task queue with the heartbeat timeout
    recorded, alongside the explicit start_to_close_timeout. Without it a
    heartbeating activity tool cannot be routed off the workflow's task queue,
    and a lost worker is only noticed at start-to-close. Only the workflow's
    own worker runs the plugin here — the routed activity worker is
    plugin-free, so the scheduling must work without any plugin-side
    cooperation.
    """
    plugin = DeepAgentsPlugin()
    async with (
        Worker(
            env.client,
            task_queue="da-act-tool-opts",
            workflows=[ActivityAsToolOptionsWorkflow],
            plugins=[plugin],
        ),
        Worker(
            env.client,
            task_queue=TOOL_TASK_QUEUE,
            activities=[echo_activity, forecast_activity],
        ),
    ):
        handle = await env.client.start_workflow(
            ActivityAsToolOptionsWorkflow.run,
            "hi",
            id=f"da-act-tool-opts-{uuid.uuid4()}",
            task_queue="da-act-tool-opts",
        )
        out = await handle.result()

    assert out == ["echo:hi", "Paris:3:C"]
    for name in ("echo_activity", "forecast_activity"):
        attrs = await scheduled_activity_attrs(handle, name)
        assert attrs is not None, name
        assert attrs.task_queue.name == TOOL_TASK_QUEUE
        assert attrs.heartbeat_timeout.ToTimedelta() == timedelta(seconds=3)
        # Explicit argument and activity_options compose.
        assert attrs.start_to_close_timeout.ToTimedelta() == timedelta(seconds=10)
    await Replayer(
        workflows=[ActivityAsToolOptionsWorkflow], plugins=[DeepAgentsPlugin()]
    ).replay_workflow(await handle.fetch_history())


@pytest.mark.asyncio
async def test_activity_as_tool_passes_each_parameter(
    env: WorkflowEnvironment,
) -> None:
    async with Worker(
        env.client,
        task_queue="da-act-tool-params",
        workflows=[ActivityParametersAsToolWorkflow],
        activities=[forecast_activity, quota_activity],
        plugins=[DeepAgentsPlugin()],
    ):
        out = await env.client.execute_workflow(
            ActivityParametersAsToolWorkflow.run,
            id=f"da-act-tool-params-{uuid.uuid4()}",
            task_queue="da-act-tool-params",
            execution_timeout=timedelta(seconds=30),
        )
    assert out == ["Paris:3:C", "Oslo:1:F", "quota:42"]


@pytest.mark.asyncio
async def test_tool_as_activity(env: WorkflowEnvironment) -> None:
    plugin = DeepAgentsPlugin()
    async with Worker(
        env.client,
        task_queue="da-tool-act",
        workflows=[ToolAsActivityWorkflow],
        plugins=[plugin],
    ):
        handle = await env.client.start_workflow(
            ToolAsActivityWorkflow.run,
            "Paris",
            id=f"da-tool-act-{uuid.uuid4()}",
            task_queue="da-tool-act",
        )
        out = await handle.result()

    assert "sunny in Paris" in out
    counts = await count_scheduled_activities(handle)
    assert counts[INVOKE_TOOL] == 1, counts


def test_builtin_tool_in_workflow(recwarn: pytest.WarningsRecorder) -> None:
    # Built-in tool names never warn (they are pure, in-workflow); an unwrapped
    # user tool does warn so the Workflow-vs-Activity choice is conscious.
    warn_unwrapped_tools([SimpleNamespace(name="write_todos")])
    assert len(recwarn) == 0

    warn_unwrapped_tools([SimpleNamespace(name="scrape_website")])
    assert any("scrape_website" in str(w.message) for w in recwarn)


# Turn-2 requests observed by the fake model, captured activity-side. Module
# state is shared with the in-process worker, same as the tool registries.
_captured_requests: list[list] = []


@workflow.defn
class ToolCallIdPairingWorkflow:
    @workflow.run
    async def run(self, city: str) -> str:
        weather_tool = tool_as_activity(
            pairing_weather, start_to_close_timeout=timedelta(seconds=10)
        )
        agent = create_deep_agent(
            model="anthropic:claude-sonnet-4-5",
            tools=[weather_tool],
            system_prompt="Use the weather tool.",
        )
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": f"Weather in {city}?"}]}
        )
        return str(result["messages"][-1].content)


@workflow.defn
class RepeatedToolCallWorkflow:
    @workflow.run
    async def run(self, input: dict, state_snapshot: dict | None = None) -> str:
        tool = tool_as_activity(
            pairing_weather, start_to_close_timeout=timedelta(seconds=10)
        )
        agent = create_deep_agent(model="fake:model", tools=[tool])
        result = await run_deep_agent(
            agent,
            input,
            continue_as_new_after=10_000,
            state_snapshot=state_snapshot,
        )
        return str(result["messages"][-1].content)


@workflow.defn
class InjectedStateToolWorkflow:
    @workflow.run
    async def run(self, question: str) -> list[str]:
        with workflow.unsafe.imports_passed_through():
            from langchain_core.documents import Document
            from langchain_core.messages import AIMessage, ChatMessage, FunctionMessage

        wrapped = (
            last_user_message,
            message_summary,
            saved_artifact,
            echo_record,
            echo_model,
        )
        tools = [
            tool_as_activity(t, start_to_close_timeout=timedelta(seconds=10))
            for t in wrapped
        ]
        agent = create_deep_agent(model="fake:model", tools=tools)
        save_call = {"name": "save", "args": {}, "id": "save-call"}
        messages = [
            ChatMessage(role="critic", content="be brief"),
            FunctionMessage(name="lookup", content="sunny"),
            AIMessage(content="", tool_calls=[save_call]),
            ToolMessage(
                content="saved",
                tool_call_id="save-call",
                artifact=Document(page_content="notes", metadata={"page": 1}),
            ),
            HumanMessage(content=question),
        ]
        result = await agent.ainvoke({"messages": messages})
        return [
            str(m.content)
            for m in result["messages"]
            if m.type == "tool" and m.tool_call_id != "save-call"
        ]


@pytest.mark.asyncio
async def test_wrapped_tool_reads_messages_from_injected_state(
    env: WorkflowEnvironment,
) -> None:
    from langchain_core.messages import AIMessage

    # Arguments that hold the marker key an earlier version of the encoding
    # used, as a dict or as a model, reach the tool unchanged.
    record = {"tenant": "keep", "__deepagents_tool_arg__": "ordinary value"}
    calls = [
        {"name": "last_user_message", "args": {}, "id": "state-call"},
        {"name": "message_summary", "args": {}, "id": "summary-call"},
        {"name": "saved_artifact", "args": {}, "id": "artifact-call"},
        {"name": "echo_record", "args": {"record": record}, "id": "record-call"},
        {"name": "echo_model", "args": {"record": record}, "id": "model-call"},
    ]
    responses = [AIMessage(content="", tool_calls=calls), AIMessage(content="done")]
    plugin = DeepAgentsPlugin(model_provider=mock_model_provider(responses))
    async with Worker(
        env.client,
        task_queue="da-injected-state",
        workflows=[InjectedStateToolWorkflow],
        plugins=[plugin],
    ):
        out = await env.client.execute_workflow(
            InjectedStateToolWorkflow.run,
            "Weather in Paris?",
            id=f"da-injected-state-{uuid.uuid4()}",
            task_queue="da-injected-state",
        )

    # The wrapped tools get LangChain messages, including the legacy chat and
    # function messages, as they do without Temporal. A message's other fields,
    # such as the tool result's Document artifact, keep their JSON form.
    assert out == [
        "user said: Weather in Paris?",
        "ChatMessage:be brief,FunctionMessage:sunny,AIMessage:,ToolMessage:saved,"
        "HumanMessage:Weather in Paris?,AIMessage:",
        "dict:notes",
        "keep:ordinary value",
        "keep:ordinary value",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("recorded", [False, True])
async def test_invoke_tool_rebuilds_only_recorded_messages(recorded: bool) -> None:
    from langchain_core import messages as lc_messages

    from temporalio.deepagents import _serde
    from temporalio.deepagents._activity import DeepAgentActivities, ToolActivityInput
    from temporalio.deepagents._tools import register_tool
    from temporalio.testing import ActivityEnvironment

    converter = _serde.data_converter.payload_converter

    def send(input: ToolActivityInput) -> ToolActivityInput:
        payloads = converter.to_payloads([input])
        return converter.from_payloads(payloads, [ToolActivityInput])[0]

    register_tool(message_summary)
    # Input from older workflow code records no messages and passes through.
    args: dict[str, Any] = {"state": {"messages": [HumanMessage(content="hi")]}}
    input = ToolActivityInput(
        tool_name="message_summary", tool_call_id="c-1", args=args
    )
    if recorded:
        input.arg_messages = _serde.find_tool_arg_messages(args)
    output = await ActivityEnvironment().run(
        DeepAgentActivities().invoke_tool, send(input)
    )
    expected = "HumanMessage:hi" if recorded else "dict:?"
    assert _serde.load_object(output.message).content == expected

    if recorded:
        # Every recorded message class, chunks included, comes back.
        chunks = [
            lc_messages.AIMessageChunk(content="a"),
            lc_messages.ChatMessageChunk(content="c", role="critic"),
            lc_messages.FunctionMessageChunk(content="f", name="fn"),
            lc_messages.HumanMessageChunk(content="h"),
            lc_messages.SystemMessageChunk(content="s"),
            lc_messages.ToolMessageChunk(content="t", tool_call_id="1"),
        ]
        input = ToolActivityInput(
            tool_name="message_summary",
            tool_call_id="c-2",
            args={"chunks": chunks},
            arg_messages=_serde.find_tool_arg_messages({"chunks": chunks}),
        )
        sent = send(input)
        rebuilt = _serde.load_tool_arg_messages(sent.args, sent.arg_messages)
        assert rebuilt == {"chunks": chunks}


@pytest.mark.asyncio
async def test_wrapped_tool_result_pairs_with_model_tool_call_id(
    env: WorkflowEnvironment,
) -> None:
    """The tool_result the model sees on turn 2 must carry the model's OWN
    tool_call_id. Regression: ``tool_as_activity`` returned the activity-built
    ``ToolMessage`` whose workflow-generated id a real provider (Anthropic)
    rejects as an unpaired ``tool_result`` — offline fakes never validate the
    pairing, so only a live-model run surfaced it. The wrapper now returns
    plain content and the tool node stamps the correct id.
    """
    from langchain_core.messages import AIMessage, ToolMessage

    from temporalio.deepagents.testing import FakeModel

    _captured_requests.clear()

    class RecordingModel(FakeModel):
        def __init__(self, responses: Sequence[Any]) -> None:
            super().__init__(responses)

        async def _agenerate(
            self,
            messages: list[Any],
            stop: list[str] | None = None,
            run_manager: Any = None,
            **kwargs: Any,
        ) -> Any:
            _captured_requests.append(list(messages))
            return await super()._agenerate(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )

    tool_turn = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "pairing_weather",
                "args": {"city": "Paris"},
                "id": "toolu_scripted_pairing_id",
            }
        ],
    )
    final = AIMessage(content="It is sunny in Paris.")
    responses = [tool_turn, final]
    cursor = {"i": 0}

    def provider(_model_name: str) -> RecordingModel:
        reply = responses[cursor["i"] % len(responses)]
        cursor["i"] += 1
        return RecordingModel([reply])

    plugin = DeepAgentsPlugin(model_provider=provider)
    async with Worker(
        env.client,
        task_queue="da-toolcall-pairing",
        workflows=[ToolCallIdPairingWorkflow],
        plugins=[plugin],
        max_cached_workflows=0,
    ):
        handle = await env.client.start_workflow(
            ToolCallIdPairingWorkflow.run,
            "Paris",
            id=f"da-toolcall-pairing-{uuid.uuid4()}",
            task_queue="da-toolcall-pairing",
        )
        out = await handle.result()

    assert "sunny" in out.lower()
    # Turn 2's request must contain the tool result under the MODEL's id.
    assert len(_captured_requests) >= 2, len(_captured_requests)
    tool_messages = [m for m in _captured_requests[1] if isinstance(m, ToolMessage)]
    assert tool_messages, _captured_requests[1]
    assert tool_messages[0].tool_call_id == "toolu_scripted_pairing_id", tool_messages[
        0
    ].tool_call_id
    assert "weather:Paris" in str(tool_messages[0].content)


@pytest.mark.asyncio
async def test_repeated_tool_calls_with_same_args_both_run(
    env: WorkflowEnvironment,
) -> None:
    from langchain_core.messages import AIMessage

    responses = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "pairing_weather",
                    "args": {"city": "Paris"},
                    "id": "first-call",
                }
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "pairing_weather",
                    "args": {"city": "Paris"},
                    "id": "second-call",
                }
            ],
        ),
        AIMessage(content="done"),
    ]
    plugin = DeepAgentsPlugin(model_provider=mock_model_provider(responses))
    async with Worker(
        env.client,
        task_queue="da-repeated-tool-calls",
        workflows=[RepeatedToolCallWorkflow],
        plugins=[plugin],
    ):
        handle = await env.client.start_workflow(
            RepeatedToolCallWorkflow.run,
            {"messages": [{"role": "user", "content": "Check Paris twice."}]},
            id=f"da-repeated-tool-calls-{uuid.uuid4()}",
            task_queue="da-repeated-tool-calls",
        )
        assert await handle.result() == "done"

    counts = await count_scheduled_activities(handle)
    assert counts[INVOKE_TOOL] == 2, counts
