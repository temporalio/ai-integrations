"""The tool seam: existing activities and wrapped tools route to activities.

These need LangChain (a tool is a ``BaseTool``) and guard on its import. Each
runs a real workflow that invokes the wrapped tool and asserts it scheduled the
expected activity.
"""

from __future__ import annotations

import asyncio
import functools
import sys
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import timedelta
from types import SimpleNamespace
from typing import Annotated, Any, Self

import pytest

from temporalio.testing import WorkflowEnvironment

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 11), reason="deepagents requires Python >= 3.11"
)

pytest.importorskip("langchain_core")

from temporalio import activity, workflow  # noqa: E402
from temporalio.client import WorkflowFailureError  # noqa: E402
from temporalio.common import RetryPolicy  # noqa: E402
from temporalio.exceptions import (  # noqa: E402
    ActivityError,
    ApplicationError,
    CancelledError,
)
from temporalio.worker import Replayer, Worker  # noqa: E402
from tests.helpers.deepagents import count_scheduled_activities  # noqa: E402

with workflow.unsafe.imports_passed_through():
    from langchain_core.messages import AIMessage, ToolMessage  # noqa: E402

    from temporalio.deepagents import (  # noqa: E402
        DeepAgentsPlugin,
        activity_as_tool,
        run_deep_agent,
        tool_as_activity,
    )
    from temporalio.deepagents._tools import warn_unwrapped_tools  # noqa: E402
    from temporalio.deepagents.testing import (  # noqa: E402
        FakeModel,
        mock_model_provider,
    )

INVOKE_TOOL = "deepagents.invoke_tool"

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
async def echo_twice(text: str, /, times: int = 2, sep: str = ",") -> str:
    # `text` is positional-only; the model still passes it by name.
    return sep.join([text] * times)


class EchoActivities:
    @activity.defn
    async def echo_method(self, text: Annotated[str, "The text to echo."]) -> str:
        return f"method:{text}"

    @activity.defn
    async def echo_self_typed(self: Self, text: str) -> str:
        return f"method:{text}"


@activity.defn
class EchoClass:
    """An echo activity whose prefix is set on the worker."""

    # A class-level annotation for the constructor's parameter: without it, the
    # tool built from the constructor failed to build at all.
    prefix: str

    def __init__(self, prefix: str) -> None:
        self.prefix = prefix

    async def __call__(self, text: Annotated[str, "The text to echo."]) -> str:
        """Echo text with the worker's prefix."""
        return f"{self.prefix}:{text}"


@activity.defn
async def forecast_activity(city: str, days: int, units: str = "C") -> str:
    return f"{city}:{days}:{units}"


@activity.defn
async def quota_activity() -> str:
    return "quota:42"


@workflow.defn
class ActivityAsToolWorkflow:
    @workflow.run
    async def run(self, text: str) -> list[str]:
        calls: list[tuple[Callable[..., Any], dict[str, Any]]] = [
            (echo_activity, {"text": text}),
            # `times` is left out and `sep` is passed: LangChain fills in the
            # default before it calls the tool.
            (echo_twice, {"text": text, "sep": "+"}),
            (EchoActivities.echo_method, {"text": text}),
            (EchoClass, {"text": text}),
        ]
        results = []
        for activity_ref, args in calls:
            tool = activity_as_tool(
                activity_ref,
                start_to_close_timeout=timedelta(seconds=10),
                retry_policy=RetryPolicy(maximum_attempts=1),
            )
            results.append(await tool.ainvoke(args))
        return results


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
        activities=[
            echo_activity,
            echo_twice,
            EchoActivities().echo_method,
            EchoClass("worker"),
        ],
        plugins=[plugin],
    ):
        handle = await env.client.start_workflow(
            ActivityAsToolWorkflow.run,
            "hi",
            id=f"da-act-tool-{uuid.uuid4()}",
            task_queue="da-act-tool",
        )
        out = await handle.result()

    # Each activity gets the model's arguments as its own parameters; the
    # callable class's registered instance gets them in __call__.
    assert out == ["echo:hi", "hi+hi", "method:hi", "worker:hi"]
    counts = await count_scheduled_activities(handle)
    assert counts == {
        "echo_activity": 1,
        "echo_twice": 1,
        "echo_method": 1,
        "EchoClass": 1,
    }, counts


class EchoBase:
    async def __call__(self, text: str) -> str:
        """Echo text from the base class."""
        return text


@activity.defn
class DocumentedEchoClass(EchoBase):
    """Echo text from the class docstring."""

    async def __call__(self, text: str) -> str:
        return text


@activity.defn
class InheritedEchoClass(EchoBase):
    async def __call__(self, text: str) -> str:
        return text


@activity.defn
@dataclass
class EchoDataclass(EchoBase):
    # No docstring, so dataclasses writes "EchoDataclass(prefix: 'str')".
    prefix: str


@activity.defn
class StaticEchoClass:
    @staticmethod
    async def __call__(text: str) -> str:
        return text


@activity.defn
class ClassMethodEchoClass:
    @classmethod
    async def __call__(cls, text: str) -> str:
        return text


async def _echo_with(_receiver: object, text: str) -> str:
    return text


@activity.defn
class PartialMethodEchoClass:
    __call__ = functools.partialmethod(_echo_with)


@activity.defn
class PartialKeywordEchoClass:
    # Binding `text` by keyword makes it keyword-only.
    __call__ = functools.partialmethod(_echo_with, text="hi")


@pytest.mark.parametrize(
    "activity_ref, args, description",
    [
        # No `self` for a method passed as Class.method, even annotated (a string
        # here, under `from __future__ import annotations`).
        (
            EchoActivities.echo_method,
            {"text": "The text to echo."},
            "Temporal activity echo_method.",
        ),
        (
            EchoActivities.echo_self_typed,
            {"text": None},
            "Temporal activity echo_self_typed.",
        ),
        # A class offers __call__'s parameters and docstring, not its constructor's
        # or its class docstring.
        (
            EchoClass,
            {"text": "The text to echo."},
            "Echo text with the worker's prefix.",
        ),
        # Without a __call__ docstring, the class's own docstring, else that of
        # an inherited __call__.
        (DocumentedEchoClass, {"text": None}, "Echo text from the class docstring."),
        (InheritedEchoClass, {"text": None}, "Echo text from the base class."),
        # Not the docstring dataclasses writes, which names the constructor's
        # parameters.
        (EchoDataclass, {"text": None}, "Echo text from the base class."),
        # The receiver of each __call__ form is dropped, as an instance binds it.
        (StaticEchoClass, {"text": None}, "Temporal activity StaticEchoClass."),
        (
            ClassMethodEchoClass,
            {"text": None},
            "Temporal activity ClassMethodEchoClass.",
        ),
        (
            PartialMethodEchoClass,
            {"text": None},
            "Temporal activity PartialMethodEchoClass.",
        ),
    ],
)
def test_activity_as_tool_schema(
    activity_ref: Any, args: dict[str, str | None], description: str
) -> None:
    # `args` maps each parameter the model is offered to its description.
    tool = activity_as_tool(activity_ref, start_to_close_timeout=timedelta(seconds=10))
    assert {name: arg.get("description") for name, arg in tool.args.items()} == args
    assert tool.description == description


@activity.defn
class NoCallActivity:
    pass


@activity.defn
class KeywordOnlyActivity:
    async def __call__(self, *, text: str) -> str:
        return text


@activity.defn
async def varargs_activity(text: str, *more: str) -> str:
    return text + "".join(more)


@activity.defn
async def kwargs_activity(text: str, **extra: str) -> str:
    return text + "".join(extra.values())


class BadAnnotationActivity(EchoClass):
    # Not decorated, so only activity_as_tool reads this annotation.
    async def __call__(self, text: Missing) -> str:  # type: ignore
        return text


@pytest.mark.parametrize(
    "activity_ref, message",
    [
        (NoCallActivity, "to have a __call__"),
        (KeywordOnlyActivity, "keyword-only argument 'text'"),
        (PartialKeywordEchoClass, "partialmethod binds an argument by keyword"),
        (varargs_activity, r"\*more"),
        (kwargs_activity, r"\*\*extra"),
        (BadAnnotationActivity, "name 'Missing' is not defined"),
        (EchoClass("workflow"), "not an instance of it"),
    ],
)
def test_activity_as_tool_rejects(activity_ref: Any, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        activity_as_tool(activity_ref, start_to_close_timeout=timedelta(seconds=10))


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


# Requests each model turn saw, captured activity-side. Module state is shared
# with the in-process worker, same as the tool registries.
_captured_requests: list[list] = []


class _RecordingModel(FakeModel):
    """A fake model that records every request, so a test can assert what the
    model was actually shown on a later turn."""

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


def _recording_provider(script: Sequence[Any]) -> Callable[[str], _RecordingModel]:
    """A ``model_provider`` serving ``script`` one scripted turn per model call."""
    cursor = {"i": 0}

    def provider(_model_name: str) -> _RecordingModel:
        reply = script[cursor["i"] % len(script)]
        cursor["i"] += 1
        return _RecordingModel([reply])

    return provider


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
    _captured_requests.clear()

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
    plugin = DeepAgentsPlugin(model_provider=_recording_provider([tool_turn, final]))
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


# ---------------------------------------------------------------------------
# Terminal tool failures returned to the model through tool-call middleware
# ---------------------------------------------------------------------------
#
# Neither helper converts failures itself. An application that wants a failed
# tool handed to the model adds an ``AgentMiddleware.awrap_tool_call`` hook to
# ``create_deep_agent(middleware=[...])``: by the time the ActivityError reaches
# the hook, Temporal's retry policy has already finished, so the hook only
# decides what the model sees. These tests pin that the hook works for both
# seams, keeps the structured ``status="error"``, replays, can be selective,
# and leaves cancellation alone.

with workflow.unsafe.imports_passed_through():
    from langchain.agents.middleware import AgentMiddleware  # noqa: E402

from temporalio.exceptions import is_cancelled_exception  # noqa: E402


class ReturnToolFailuresToModel(AgentMiddleware):
    """Hand terminal failures of the named tools to the model as error results.

    The same shape as the README example: await the tool, re-raise
    cancellation and failures of tools not listed, and answer the model's own
    tool call with a ``ToolMessage(status="error")`` otherwise.
    """

    def __init__(self, tool_names: Sequence[str]) -> None:
        super().__init__()
        self.tool_names = frozenset(tool_names)

    async def awrap_tool_call(self, request: Any, handler: Any) -> Any:
        try:
            return await handler(request)
        except ActivityError as err:
            name = request.tool_call["name"]
            if is_cancelled_exception(err) or name not in self.tool_names:
                raise
            return ToolMessage(
                content=f"{name} is unavailable ({err.cause}); try another source.",
                name=name,
                tool_call_id=request.tool_call["id"],
                status="error",
            )


@activity.defn
async def rejected_search(query: str) -> str:
    """Search a source that rejects the query for good (non-retryable)."""
    raise ApplicationError(
        f"source rejected {query!r}", type="SourceRejected", non_retryable=True
    )


@activity.defn
async def down_search(query: str) -> str:
    """Search a source that is down (retryable, never recovers)."""
    raise RuntimeError(f"search backend down for {query!r}")


# Set once the stalled_search body has started, so the cancellation test
# cancels a provably running activity instead of racing a fixed sleep.
_stalled_started = asyncio.Event()


@activity.defn
async def stalled_search(query: str) -> str:
    """Search a source that never answers (outlives this test's cancel)."""
    _stalled_started.set()
    await asyncio.sleep(120)
    return f"never answered {query!r}"


# Bounded, fast retries: an exhausted policy costs the test a fraction of a
# second instead of Temporal's default backoff.
_FAST_RETRY = RetryPolicy(
    initial_interval=timedelta(milliseconds=20),
    maximum_interval=timedelta(milliseconds=20),
    maximum_attempts=2,
)


async def _agent_final_message(tool: Any, middleware: Sequence[Any] = ()) -> str:
    """Run a one-tool agent on the fixed prompt, return its final message."""
    agent = create_deep_agent(
        model="fake:model", tools=[tool], middleware=list(middleware)
    )
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "search for arXiv:2601.00001"}]}
    )
    return str(result["messages"][-1].content)


def _failing_tool(seam: str, mode: str) -> Any:
    """The failing search tool on ``seam`` (``activity`` / ``tool``), failing
    ``non_retryable`` or after ``exhausted`` retries."""
    body = rejected_search if mode == "non_retryable" else down_search
    retry = None if mode == "non_retryable" else _FAST_RETRY
    if seam == "activity":
        return activity_as_tool(
            body,
            start_to_close_timeout=timedelta(seconds=20),
            retry_policy=retry,
        )
    return tool_as_activity(
        body,
        start_to_close_timeout=timedelta(seconds=20),
        activity_options={"retry_policy": retry},
    )


@workflow.defn
class FailingToolMiddlewareWorkflow:
    @workflow.run
    async def run(self, seam: str, mode: str, recovered: list[str]) -> str:
        """One-tool agent whose tool always fails; the middleware recovers the
        tool names in ``recovered`` and lets every other failure propagate."""
        return await _agent_final_message(
            _failing_tool(seam, mode), [ReturnToolFailuresToModel(recovered)]
        )


@workflow.defn
class FailingToolDefaultWorkflow:
    """Same failing tool with no middleware at all."""

    @workflow.run
    async def run(self, seam: str) -> str:
        return await _agent_final_message(_failing_tool(seam, "non_retryable"))


@workflow.defn
class StalledToolMiddlewareWorkflow:
    @workflow.run
    async def run(self) -> str:
        tool = activity_as_tool(
            stalled_search, start_to_close_timeout=timedelta(seconds=120)
        )
        return await _agent_final_message(
            tool, [ReturnToolFailuresToModel(["stalled_search"])]
        )


def _search_tool_call(tool_name: str) -> AIMessage:
    """A scripted model turn that calls the failing search tool."""
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": tool_name,
                "args": {"query": "arXiv:2601.00001"},
                "id": "call_search",
                "type": "tool_call",
            }
        ],
    )


# Turn the model never reaches when the tool failure fails the run instead.
_UNREACHED = AIMessage(content="unreached")


def _error_chain(exc: BaseException) -> list[BaseException]:
    chain: list[BaseException] = []
    err: BaseException | None = exc
    while err is not None:
        chain.append(err)
        err = err.__cause__
    return chain


@pytest.mark.asyncio
@pytest.mark.parametrize("seam", ["activity", "tool"])
@pytest.mark.parametrize("mode", ["non_retryable", "exhausted"])
async def test_middleware_hands_failed_tool_to_the_model(
    env: WorkflowEnvironment, seam: str, mode: str
) -> None:
    """A tool activity that failed for good reaches the model as an error result.

    LangGraph's ToolNode re-raises every error but an argument error, so without
    the hook the ActivityError fails the whole run (see the next test). With an
    ``awrap_tool_call`` middleware the run completes, the model's next turn
    carries a ``status="error"`` ToolMessage paired with its own tool call id,
    and the history replays.
    """
    _captured_requests.clear()
    tool_name = (
        rejected_search.__name__ if mode == "non_retryable" else down_search.__name__
    )
    plugin = DeepAgentsPlugin(
        model_provider=_recording_provider(
            [
                _search_tool_call(tool_name),
                AIMessage(content="I could not search, so I am reporting the failure."),
            ]
        )
    )
    queue = f"da-tool-failure-{seam}-{mode}"
    async with Worker(
        env.client,
        task_queue=queue,
        workflows=[FailingToolMiddlewareWorkflow],
        activities=[rejected_search, down_search],
        plugins=[plugin],
    ):
        handle = await env.client.start_workflow(
            FailingToolMiddlewareWorkflow.run,
            args=[seam, mode, [tool_name]],
            id=f"da-tool-failure-{uuid.uuid4()}",
            task_queue=queue,
        )
        out = await handle.result()

    assert "could not search" in out, out
    tool_messages = [m for m in _captured_requests[1] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1, _captured_requests[1]
    message = tool_messages[0]
    assert message.status == "error", message
    assert message.tool_call_id == "call_search", message
    text = str(message.content)
    assert text.startswith(f"{tool_name} is unavailable ("), text
    if mode == "non_retryable":
        assert "source rejected" in text, text
    else:
        assert "search backend down" in text, text
    # The middleware rebuilds its message from the recorded ActivityError, so
    # replaying the same history must reproduce it without nondeterminism.
    history = await handle.fetch_history()
    await Replayer(
        workflows=[FailingToolMiddlewareWorkflow], plugins=[DeepAgentsPlugin()]
    ).replay_workflow(history)


@pytest.mark.asyncio
@pytest.mark.parametrize("seam", ["activity", "tool"])
async def test_failed_tool_without_middleware_fails_the_run(
    env: WorkflowEnvironment, seam: str
) -> None:
    """With no hook, the ActivityError propagates and the run fails."""
    plugin = DeepAgentsPlugin(
        model_provider=mock_model_provider(
            [_search_tool_call("rejected_search"), _UNREACHED]
        )
    )
    queue = f"da-tool-failure-default-{seam}"
    async with Worker(
        env.client,
        task_queue=queue,
        workflows=[FailingToolDefaultWorkflow],
        activities=[rejected_search, down_search],
        plugins=[plugin],
    ):
        handle = await env.client.start_workflow(
            FailingToolDefaultWorkflow.run,
            args=[seam],
            id=f"da-tool-failure-default-{uuid.uuid4()}",
            task_queue=queue,
        )
        with pytest.raises(WorkflowFailureError) as excinfo:
            await handle.result()

    assert any(
        isinstance(e, ApplicationError) and e.type == "SourceRejected"
        for e in _error_chain(excinfo.value)
    ), [f"{type(e).__name__}: {e}" for e in _error_chain(excinfo.value)]


@pytest.mark.asyncio
async def test_middleware_propagates_failures_of_tools_it_does_not_recover(
    env: WorkflowEnvironment,
) -> None:
    """The hook chooses per tool: a failure of a tool it does not list still
    fails the run, exactly as with no middleware."""
    plugin = DeepAgentsPlugin(
        model_provider=mock_model_provider(
            [_search_tool_call("rejected_search"), _UNREACHED]
        )
    )
    async with Worker(
        env.client,
        task_queue="da-tool-failure-unlisted",
        workflows=[FailingToolMiddlewareWorkflow],
        activities=[rejected_search, down_search],
        plugins=[plugin],
    ):
        handle = await env.client.start_workflow(
            FailingToolMiddlewareWorkflow.run,
            args=["activity", "non_retryable", ["some_other_tool"]],
            id=f"da-tool-failure-unlisted-{uuid.uuid4()}",
            task_queue="da-tool-failure-unlisted",
        )
        with pytest.raises(WorkflowFailureError) as excinfo:
            await handle.result()

    assert any(
        isinstance(e, ApplicationError) and e.type == "SourceRejected"
        for e in _error_chain(excinfo.value)
    ), [f"{type(e).__name__}: {e}" for e in _error_chain(excinfo.value)]


@pytest.mark.asyncio
async def test_middleware_does_not_swallow_cancellation(
    env: WorkflowEnvironment,
) -> None:
    """Cancelling the run mid-tool must cancel the workflow, not feed the
    cancellation to the model as a tool result."""
    _stalled_started.clear()
    plugin = DeepAgentsPlugin(
        model_provider=mock_model_provider(
            [_search_tool_call("stalled_search"), _UNREACHED]
        )
    )
    async with Worker(
        env.client,
        task_queue="da-tool-failure-cancel",
        workflows=[StalledToolMiddlewareWorkflow],
        activities=[stalled_search],
        plugins=[plugin],
    ):
        handle = await env.client.start_workflow(
            StalledToolMiddlewareWorkflow.run,
            id=f"da-tool-failure-cancel-{uuid.uuid4()}",
            task_queue="da-tool-failure-cancel",
        )
        # Cancel only once the activity body is provably running: the pending
        # dispatch must then re-raise through the hook.
        await asyncio.wait_for(_stalled_started.wait(), timeout=30)
        await handle.cancel()
        with pytest.raises(WorkflowFailureError) as excinfo:
            await handle.result()

    assert isinstance(excinfo.value.cause, CancelledError), excinfo.value.cause
