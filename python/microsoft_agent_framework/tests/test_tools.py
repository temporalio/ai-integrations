"""Tools: in-workflow functions, ``activity_as_tool``, declaration-only, hosted, middleware."""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import pytest
from agent_framework import (
    Agent,
    ChatContext,
    ChatMiddleware,
    ChatResponse,
    Content,
    FunctionInvocationContext,
    FunctionMiddleware,
    FunctionTool,
    Message,
    tool,
)

from temporalio import activity, workflow
from temporalio.client import Client, WorkflowFailureError
from temporalio.exceptions import ApplicationError
from temporalio.microsoft_agent_framework import (
    MicrosoftAgentFrameworkPlugin,
    TemporalChatClient,
)
from temporalio.microsoft_agent_framework.testing import (
    fake_chat_client,
    function_call_response,
    text_response,
)
from temporalio.microsoft_agent_framework.workflow import activity_as_tool
from tests.helpers.microsoft_agent_framework import (
    INVOKE_MODEL,
    contents_summary,
    count_scheduled_activities,
    new_worker,
    workflow_id,
)

# Imported through the passthrough so both workflows below share one tool object.
with workflow.unsafe.imports_passed_through():
    from tests.shared_tools import shared_sync_thread, shared_sync_weather


@activity.defn
async def fetch_user(user_id: str) -> dict[str, Any]:
    """Fetch a user record."""
    return {"id": user_id, "name": "Ada"}


@activity.defn
async def failing_lookup(key: str) -> str:
    """Look a key up, and fail."""
    raise ApplicationError(f"no record for {key}", non_retryable=True)


class _FunctionRecorder(FunctionMiddleware):
    def __init__(self, log: list[str]) -> None:
        self.log = log

    async def process(
        self,
        context: FunctionInvocationContext,
        call_next: Callable[[], Awaitable[None]],
    ) -> None:
        self.log.append(f"before:{context.function.name}")
        try:
            await call_next()
        except Exception as err:
            self.log.append(f"error:{type(err).__name__}")
            raise
        self.log.append(f"after:{context.function.name}")


class _ChatRecorder(ChatMiddleware):
    def __init__(self, log: list[str]) -> None:
        self.log = log

    async def process(
        self, context: ChatContext, call_next: Callable[[], Awaitable[None]]
    ) -> None:
        self.log.append(f"chat:{len(context.messages)}")
        await call_next()


class WeirdTool:
    """Neither a function nor a mapping: the plugin cannot send it to the model."""

    name = "weird"


@dataclass
class ToolsInput:
    scenario: str
    prompt: str = "do the thing"


@workflow.defn
class ToolsWorkflow:
    @workflow.run
    async def run(self, input: ToolsInput) -> dict[str, Any]:
        log: list[str] = []

        @tool
        async def async_weather(city: str) -> str:
            """Async weather."""
            log.append(f"async:{city}")
            return f"rainy in {city}"

        @tool
        def sync_weather(city: str) -> str:
            """Sync weather."""
            log.append(f"sync:{city}")
            return f"sunny in {city}"

        declared = FunctionTool(
            name="render_chart",
            description="Render a chart on the client.",
            func=None,
            input_model={
                "type": "object",
                "properties": {"title": {"type": "string"}},
                "required": ["title"],
            },
        )
        tools: list[Any] = []
        middleware: list[Any] = []
        configuration: Any = None
        if input.scenario == "activity":
            tools = [
                activity_as_tool(
                    fetch_user, start_to_close_timeout=timedelta(seconds=10)
                )
            ]
        elif input.scenario == "activity_approval":
            tools = [
                activity_as_tool(
                    fetch_user,
                    approval_mode="always_require",
                    start_to_close_timeout=timedelta(seconds=10),
                )
            ]
        elif input.scenario == "pure":
            tools = [async_weather]
        elif input.scenario == "sync":
            tools = [sync_weather]
        elif input.scenario == "declared":
            tools = [declared]
        elif input.scenario == "hosted":
            tools = [{"type": "web_search"}]
        elif input.scenario == "middleware":
            tools = [async_weather]
            middleware = [_FunctionRecorder(log), _ChatRecorder(log)]
        elif input.scenario == "weird":
            tools = [WeirdTool()]
        elif input.scenario == "failing_activity":
            tools = [
                activity_as_tool(
                    failing_lookup, start_to_close_timeout=timedelta(seconds=10)
                )
            ]
            middleware = [_FunctionRecorder(log)]
            configuration = {"include_detailed_errors": True}

        agent = Agent(
            client=TemporalChatClient(
                "model", function_invocation_configuration=configuration
            ),
            tools=tools,
            middleware=middleware,
        )
        if input.scenario == "activity_approval":
            session = agent.create_session()
            response = await agent.run(input.prompt, session=session)
            requests = response.user_input_requests
            log.append(f"requests:{len(requests)}")
            response = await agent.run(
                [c.to_function_approval_response(True) for c in requests],
                session=session,
            )
        else:
            response = await agent.run(input.prompt)
        return {
            "text": response.text,
            "contents": contents_summary(response),
            "log": log,
        }


async def _run(
    client: Client, plugin: MicrosoftAgentFrameworkPlugin, input: ToolsInput
) -> tuple[Any, Any]:
    async with new_worker(
        client, ToolsWorkflow, plugin=plugin, activities=[fetch_user, failing_lookup]
    ) as worker:
        handle = await client.start_workflow(
            ToolsWorkflow.run,
            input,
            id=workflow_id(f"tools-{input.scenario}"),
            task_queue=worker.task_queue,
        )
        result = await handle.result()
    return result, await count_scheduled_activities(handle)


async def test_activity_as_tool_dispatches_activity(client: Client) -> None:
    factory = fake_chat_client(
        [
            function_call_response("fetch_user", {"user_id": "u1"}, call_id="call-1"),
            text_response("Ada it is."),
        ]
    )
    plugin = MicrosoftAgentFrameworkPlugin(chat_clients={"model": factory})
    result, counts = await _run(client, plugin, ToolsInput("activity"))
    assert result["text"] == "Ada it is."
    assert result["contents"] == [
        "call:fetch_user",
        'result:{"id": "u1", "name": "Ada"}',
        "text:Ada it is.",
    ]
    assert counts == {INVOKE_MODEL: 2, "fetch_user": 1}
    (declared,) = factory().calls[0]["options"]["tools"]
    assert declared.name == "fetch_user"
    assert declared.description == "Fetch a user record."
    assert declared.parameters()["required"] == ["user_id"]


async def test_activity_tool_requires_approval(client: Client) -> None:
    plugin = MicrosoftAgentFrameworkPlugin(
        chat_clients={
            "model": fake_chat_client(
                [
                    function_call_response(
                        "fetch_user", {"user_id": "u1"}, call_id="call-1"
                    ),
                    text_response("Approved and fetched."),
                ]
            )
        }
    )
    result, counts = await _run(client, plugin, ToolsInput("activity_approval"))
    assert result["log"] == ["requests:1"]
    assert result["text"] == "Approved and fetched."
    assert counts == {INVOKE_MODEL: 2, "fetch_user": 1}


async def test_pure_tool_stays_in_workflow(client: Client) -> None:
    plugin = MicrosoftAgentFrameworkPlugin(
        chat_clients={
            "model": fake_chat_client(
                [
                    function_call_response(
                        "async_weather", {"city": "Rome"}, call_id="call-1"
                    ),
                    text_response("Rainy in Rome."),
                ]
            )
        }
    )
    result, counts = await _run(client, plugin, ToolsInput("pure"))
    assert result["log"] == ["async:Rome"]
    assert result["contents"][1] == "result:rainy in Rome"
    assert counts == {INVOKE_MODEL: 2}


async def test_sync_tool_runs_inline(client: Client) -> None:
    plugin = MicrosoftAgentFrameworkPlugin(
        chat_clients={
            "model": fake_chat_client(
                [
                    function_call_response(
                        "sync_weather", {"city": "Rome"}, call_id="call-1"
                    ),
                    text_response("Sunny in Rome."),
                ]
            )
        }
    )
    result, counts = await _run(client, plugin, ToolsInput("sync"))
    # Without the plugin's middleware the loop would hand the model
    # "Error: Function failed." and the tool would never run.
    assert result["log"] == ["sync:Rome"]
    assert result["contents"][1] == "result:sunny in Rome"
    assert counts == {INVOKE_MODEL: 2}


async def test_declaration_only_tool(client: Client) -> None:
    factory = fake_chat_client(
        [
            function_call_response(
                "render_chart", {"title": "Sales"}, call_id="call-1"
            ),
            text_response("never reached"),
        ]
    )
    plugin = MicrosoftAgentFrameworkPlugin(chat_clients={"model": factory})
    result, counts = await _run(client, plugin, ToolsInput("declared"))
    # The loop returns the call to the caller instead of executing anything.
    assert result["contents"] == ["call:render_chart"]
    assert result["text"] == ""
    assert counts == {INVOKE_MODEL: 1}
    (declared,) = factory().calls[0]["options"]["tools"]
    assert declared.parameters()["properties"] == {"title": {"type": "string"}}


async def test_hosted_tool_passthrough(client: Client) -> None:
    hosted = ChatResponse(
        messages=[
            Message(
                role="assistant",
                contents=[
                    Content.from_function_call(
                        "ws-1", "web_search", arguments={"query": "news"}, hosted=True
                    ),
                    Content.from_function_result("ws-1", result="hosted search result"),
                    Content.from_text("Here is the news."),
                ],
            )
        ]
    )
    factory = fake_chat_client([hosted])
    plugin = MicrosoftAgentFrameworkPlugin(chat_clients={"model": factory})
    result, counts = await _run(client, plugin, ToolsInput("hosted"))
    assert result["text"] == "Here is the news."
    assert "result:hosted search result" in result["contents"]
    assert counts == {INVOKE_MODEL: 1}
    assert factory().calls[0]["options"]["tools"] == [{"type": "web_search"}]


async def test_middleware_runs_in_workflow(client: Client) -> None:
    plugin = MicrosoftAgentFrameworkPlugin(
        chat_clients={
            "model": fake_chat_client(
                [
                    function_call_response(
                        "async_weather", {"city": "Rome"}, call_id="call-1"
                    ),
                    text_response("Rainy in Rome."),
                ]
            )
        }
    )
    result, counts = await _run(client, plugin, ToolsInput("middleware"))
    assert result["log"] == [
        "chat:1",
        "before:async_weather",
        "async:Rome",
        "after:async_weather",
        "chat:3",
    ]
    assert counts == {INVOKE_MODEL: 2}


async def test_unsupported_tool_kind(client: Client) -> None:
    plugin = MicrosoftAgentFrameworkPlugin(
        chat_clients={"model": fake_chat_client(["unused"])}
    )
    with pytest.raises(WorkflowFailureError) as excinfo:
        await _run(client, plugin, ToolsInput("weird"))
    cause = excinfo.value.cause
    assert isinstance(cause, ApplicationError)
    assert cause.type == "MicrosoftAgentFrameworkWorkflowError"
    assert cause.non_retryable
    assert "WeirdTool" in cause.message


async def test_activity_tool_failure_becomes_tool_result(client: Client) -> None:
    plugin = MicrosoftAgentFrameworkPlugin(
        chat_clients={
            "model": fake_chat_client(
                [
                    function_call_response(
                        "failing_lookup", {"key": "k1"}, call_id="call-1"
                    ),
                    text_response("Could not look it up."),
                ]
            )
        }
    )
    result, counts = await _run(client, plugin, ToolsInput("failing_activity"))
    assert result["text"] == "Could not look it up."
    assert result["log"] == ["before:failing_lookup", "error:ApplicationError"]
    (result_text,) = [c for c in result["contents"] if c.startswith("result:")]
    assert result_text.startswith("result:Error: Function failed.")
    assert "no record for k1" in result_text
    assert counts == {INVOKE_MODEL: 2, "failing_lookup": 1}


def test_activity_as_tool_rejects_non_activities() -> None:
    def plain(x: int) -> int:
        return x

    with pytest.raises(ValueError, match="@activity.defn"):
        activity_as_tool(plain)
    with pytest.raises(ValueError, match="function or a method"):
        activity_as_tool(WeirdTool)  # type: ignore[arg-type]

    @activity.defn
    async def variadic(*_args: int) -> int:
        return 0

    with pytest.raises(ValueError, match=r"\*args"):
        activity_as_tool(variadic)


def test_activity_as_tool_method_drops_self() -> None:
    class Activities:
        @activity.defn
        async def lookup(self, city: str, verbose: bool = False) -> str:
            """Look a city up."""
            return f"{city} {verbose}"

    as_tool = activity_as_tool(Activities.lookup, name="city_lookup")
    spec = as_tool.to_json_schema_spec()["function"]
    assert spec["name"] == "city_lookup"
    assert spec["description"] == "Look a city up."
    assert list(spec["parameters"]["properties"]) == ["city", "verbose"]
    assert spec["parameters"]["required"] == ["city"]


@workflow.defn
class SyncFlagWorkflow:
    @workflow.run
    async def run(self) -> dict[str, Any]:
        @tool
        def sync_weather(city: str) -> str:
            """Sync weather."""
            return f"sunny in {city}"

        agent = Agent(client=TemporalChatClient("model"), tools=[sync_weather])
        response = await agent.run("weather in Rome?")
        return {
            "text": response.text,
            "inline_after_run": sync_weather._invoke_sync_on_event_loop,
        }


async def test_sync_tool_switch_is_restored(client: Client) -> None:
    plugin = MicrosoftAgentFrameworkPlugin(
        chat_clients={
            "model": fake_chat_client(
                [
                    function_call_response(
                        "sync_weather", {"city": "Rome"}, call_id="call-1"
                    ),
                    text_response("Sunny."),
                ]
            )
        }
    )
    async with new_worker(client, SyncFlagWorkflow, plugin=plugin) as worker:
        result = await client.execute_workflow(
            SyncFlagWorkflow.run,
            id=workflow_id("sync-flag"),
            task_queue=worker.task_queue,
        )
    # The plugin leaves the framework's own inline switch on the tool alone.
    assert result == {"text": "Sunny.", "inline_after_run": False}


class _PauseBeforeTool(FunctionMiddleware):
    """Keeps the run inside the tool call for a while before the tool body runs."""

    async def process(
        self,
        context: FunctionInvocationContext,
        call_next: Callable[[], Awaitable[None]],
    ) -> None:
        await workflow.sleep(0.5)
        await call_next()


@workflow.defn
class SharedToolWorkflow:
    @workflow.run
    async def run(self, city: str) -> dict[str, Any]:
        agent = Agent(
            client=TemporalChatClient("model"),
            tools=[shared_sync_weather],
            middleware=[_PauseBeforeTool()],
        )
        response = await agent.run(f"weather in {city}?")
        return {"text": response.text, "contents": contents_summary(response)}


async def test_shared_sync_tool_across_workflows(client: Client) -> None:
    def plugin_for(city: str) -> MicrosoftAgentFrameworkPlugin:
        return MicrosoftAgentFrameworkPlugin(
            chat_clients={
                "model": fake_chat_client(
                    [
                        function_call_response(
                            "shared_sync_weather", {"city": city}, call_id="call-1"
                        ),
                        text_response(f"Sunny in {city}."),
                    ]
                )
            }
        )

    # Two workers, two workflow task threads, one tool object between them.
    async with (
        new_worker(client, SharedToolWorkflow, plugin=plugin_for("Rome")) as rome,
        new_worker(client, SharedToolWorkflow, plugin=plugin_for("Oslo")) as oslo,
    ):
        handles = [
            await client.start_workflow(
                SharedToolWorkflow.run,
                "Rome",
                id=workflow_id("shared-tool-rome"),
                task_queue=rome.task_queue,
            ),
            await client.start_workflow(
                SharedToolWorkflow.run,
                "Oslo",
                id=workflow_id("shared-tool-oslo"),
                task_queue=oslo.task_queue,
            ),
        ]
        results = [await handle.result() for handle in handles]
    assert results[0]["contents"][1] == "result:sunny in Rome"
    assert results[1]["contents"][1] == "result:sunny in Oslo"
    assert [r["text"] for r in results] == ["Sunny in Rome.", "Sunny in Oslo."]
    # The shared object's own switch was never touched.
    assert shared_sync_weather._invoke_sync_on_event_loop is False


class _HoldInTool(FunctionMiddleware):
    """Keeps the run inside a tool call until the workflow is released."""

    def __init__(self, wf: SharedWindowWorkflow) -> None:
        self._wf = wf

    async def process(
        self,
        context: FunctionInvocationContext,
        call_next: Callable[[], Awaitable[None]],
    ) -> None:
        self._wf.in_tool_call = True
        await workflow.wait_condition(lambda: self._wf.released)
        await call_next()


@workflow.defn
class SharedWindowWorkflow:
    def __init__(self) -> None:
        self.in_tool_call = False
        self.released = False

    @workflow.run
    async def run(self) -> dict[str, Any]:
        agent = Agent(
            client=TemporalChatClient("model"),
            tools=[shared_sync_thread],
            middleware=[_HoldInTool(self)],
        )
        response = await agent.run("which thread?")
        return {"text": response.text, "contents": contents_summary(response)}

    @workflow.query
    def holding(self) -> bool:
        return self.in_tool_call

    @workflow.signal
    def release(self) -> None:
        self.released = True


async def test_sync_tool_outside_workflow(client: Client) -> None:
    plugin = MicrosoftAgentFrameworkPlugin(
        chat_clients={
            "model": fake_chat_client(
                [
                    function_call_response(
                        "shared_sync_thread", {"label": "inside"}, call_id="call-1"
                    ),
                    text_response("Done."),
                ]
            )
        }
    )
    # A caching worker keeps the workflow suspended inside the tool call, as a
    # production worker would, instead of evicting it between tasks.
    async with new_worker(
        client, SharedWindowWorkflow, plugin=plugin, max_cached_workflows=100
    ) as worker:
        handle = await client.start_workflow(
            SharedWindowWorkflow.run,
            id=workflow_id("shared-window"),
            task_queue=worker.task_queue,
        )
        deadline = time.monotonic() + 20
        while not await handle.query(SharedWindowWorkflow.holding):
            assert time.monotonic() < deadline, "the workflow never reached the tool"
            await asyncio.sleep(0.05)
        # The same tool object, called from outside any workflow in the same
        # process while the workflow sits inside its own call of it: Agent
        # Framework still hands it to a thread here.
        (outside,) = await shared_sync_thread.invoke(arguments={"label": "outside"})
        await handle.signal(SharedWindowWorkflow.release)
        result = await handle.result()
    label, _, thread = (outside.text or "").partition(":")
    assert label == "outside"
    assert int(thread) != threading.get_ident()
    # Inside the workflow the same tool ran inline, on the workflow task.
    assert result["contents"][1].startswith("result:inside:")
    assert result["text"] == "Done."
