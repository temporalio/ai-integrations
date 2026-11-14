"""Sessions: continue-as-new carry, service-side conversations, file providers."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from agent_framework import (
    Agent,
    AgentSession,
    ChatResponse,
    Content,
    FileHistoryProvider,
    FileSessionStore,
    FunctionInvocationContext,
    FunctionMiddleware,
    Message,
    SummarizationStrategy,
    TruncationStrategy,
    tool,
)

from temporalio import workflow
from temporalio.api.enums.v1 import PendingActivityState
from temporalio.client import (
    Client,
    WorkflowExecutionStatus,
    WorkflowFailureError,
    WorkflowHandle,
)
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
from tests.helpers.microsoft_agent_framework import (
    INVOKE_MODEL,
    SlowFakeChatClient,
    count_scheduled_activities,
    new_worker,
    role_of,
    scheduled_activity_events,
    summary_of,
    workflow_id,
)


@dataclass
class ChatInput:
    session: dict[str, Any] | None = None
    continue_after_turns: int | None = 1
    pending: list[str] = field(default_factory=list)
    done: bool = False


@workflow.defn
class ChatWorkflow:
    def __init__(self) -> None:
        self._pending: list[str] = []
        self._done = False

    @workflow.signal
    def user_says(self, prompt: str) -> None:
        self._pending.append(prompt)

    @workflow.signal
    def end_chat(self) -> None:
        self._done = True

    @workflow.run
    async def run(self, input: ChatInput) -> dict[str, Any]:
        agent = Agent(client=TemporalChatClient("model"), instructions="Chat.")
        session = (
            AgentSession.from_dict(input.session)
            if input.session
            else agent.create_session()
        )
        # Prompts and an end_chat that arrived while the previous run was busy.
        self._pending = list(input.pending) + self._pending
        self._done = self._done or input.done
        replies: list[str] = []
        while True:
            await workflow.wait_condition(lambda: bool(self._pending) or self._done)
            if not self._pending:
                return {
                    "replies": replies,
                    "session_id": session.session_id,
                    "messages": len(session.state["in_memory"]["messages"]),
                }
            response = await agent.run(self._pending.pop(0), session=session)
            replies.append(response.text)
            if (
                input.continue_after_turns
                and len(replies) >= input.continue_after_turns
            ):
                # The continued run carries the session, the prompts not yet
                # answered and a pending end_chat, and keeps going until the end.
                workflow.continue_as_new(
                    ChatInput(
                        session=session.to_dict(),
                        continue_after_turns=None,
                        pending=self._pending,
                        done=self._done,
                    )
                )


@workflow.defn
class StoredConversationWorkflow:
    @workflow.run
    async def run(self, prompts: list[str]) -> dict[str, Any]:
        agent = Agent(
            client=TemporalChatClient("model"), default_options={"store": True}
        )
        session = agent.create_session()
        texts = [(await agent.run(prompt, session=session)).text for prompt in prompts]
        return {"texts": texts, "service_session_id": session.service_session_id}


@workflow.defn
class DefaultStoreWorkflow:
    @workflow.run
    async def run(self, prompts: list[str]) -> dict[str, Any]:
        agent = Agent(client=TemporalChatClient("model"))
        session = agent.create_session()
        texts = [(await agent.run(prompt, session=session)).text for prompt in prompts]
        return {"texts": texts, "service_session_id": session.service_session_id}


@workflow.defn
class FileHistoryWorkflow:
    @workflow.run
    async def run(self, storage_path: str) -> str:
        agent = Agent(
            client=TemporalChatClient("model"),
            context_providers=[FileHistoryProvider(storage_path)],
        )
        session = agent.create_session()
        return (await agent.run("hello", session=session)).text


@workflow.defn
class FileSessionStoreWorkflow:
    @workflow.run
    async def run(self, storage_path: str) -> str:
        agent = Agent(client=TemporalChatClient("model"))
        store = FileSessionStore(storage_path)
        session = await store.get("chat-1") or agent.create_session()
        response = await agent.run("hello", session=session)
        await store.set("chat-1", session)
        return response.text


async def _wait_for_status(
    handle: WorkflowHandle, status: WorkflowExecutionStatus
) -> None:
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if (await handle.describe()).status == status:
            return
        await asyncio.sleep(0.1)
    raise AssertionError(f"workflow did not reach {status}")


async def test_continue_as_new_carries_session(client: Client) -> None:
    factory = fake_chat_client(["First reply.", "Second reply."])
    plugin = MicrosoftAgentFrameworkPlugin(chat_clients={"model": factory})
    async with new_worker(client, ChatWorkflow, plugin=plugin) as worker:
        handle = await client.start_workflow(
            ChatWorkflow.run,
            ChatInput(),
            id=workflow_id("chat"),
            task_queue=worker.task_queue,
        )
        await handle.signal(ChatWorkflow.user_says, "first question")
        first_run = client.get_workflow_handle(handle.id, run_id=handle.result_run_id)
        await _wait_for_status(first_run, WorkflowExecutionStatus.CONTINUED_AS_NEW)
        await handle.signal(ChatWorkflow.user_says, "second question")
        await handle.signal(ChatWorkflow.end_chat)
        result = await handle.result()

    assert result["replies"] == ["Second reply."]
    assert result["messages"] == 4
    first_call, second_call = factory().calls
    # The second run rebuilt the session from the carried dict: its first
    # model call already holds the first turn.
    assert [role_of(m) for m in first_call["messages"]] == ["user"]
    assert [role_of(m) for m in second_call["messages"]] == [
        "user",
        "assistant",
        "user",
    ]
    assert second_call["messages"][1].text == "First reply."
    assert (await count_scheduled_activities(first_run)).total() == 1


async def _wait_for_started_activity(handle: WorkflowHandle) -> None:
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        description = await handle.describe()
        if any(
            pending.state == PendingActivityState.PENDING_ACTIVITY_STATE_STARTED
            for pending in description.raw_description.pending_activities
        ):
            return
        await asyncio.sleep(0.05)
    raise AssertionError("the first turn's model Activity never started")


async def test_continue_as_new_carries_pending_prompts(client: Client) -> None:
    slow = SlowFakeChatClient(["First reply.", "Second reply."], delay=1.0)
    plugin = MicrosoftAgentFrameworkPlugin(chat_clients={"model": lambda: slow})
    async with new_worker(client, ChatWorkflow, plugin=plugin) as worker:
        handle = await client.start_workflow(
            ChatWorkflow.run,
            ChatInput(),
            id=workflow_id("chat-pending"),
            task_queue=worker.task_queue,
        )
        await handle.signal(ChatWorkflow.user_says, "first question")
        # The second prompt and the end of the chat arrive while the first
        # turn's model call (1 second) is running, not before it starts.
        await _wait_for_started_activity(handle)
        await handle.signal(ChatWorkflow.user_says, "second question")
        await handle.signal(ChatWorkflow.end_chat)
        result = await handle.result()
    # The run continued as new after the first turn and carried the prompt
    # and the end_chat with it; without that the new run would wait forever.
    first_run = client.get_workflow_handle(handle.id, run_id=handle.result_run_id)
    status = (await first_run.describe()).status
    assert status == WorkflowExecutionStatus.CONTINUED_AS_NEW
    assert result["replies"] == ["Second reply."]
    assert result["messages"] == 4


async def test_store_true_uses_service_conversation(client: Client) -> None:
    def stored(text: str) -> ChatResponse:
        return ChatResponse(
            messages=[Message(role="assistant", contents=[Content.from_text(text)])],
            conversation_id="conv-1",
        )

    factory = fake_chat_client([stored("one"), stored("two")])
    plugin = MicrosoftAgentFrameworkPlugin(chat_clients={"model": factory})
    async with new_worker(client, StoredConversationWorkflow, plugin=plugin) as worker:
        result = await client.execute_workflow(
            StoredConversationWorkflow.run,
            ["first", "second"],
            id=workflow_id("stored"),
            task_queue=worker.task_queue,
        )
    assert result == {"texts": ["one", "two"], "service_session_id": "conv-1"}
    first_call, second_call = factory().calls
    # The provider receives the explicit store=True, so this cannot pass on a
    # client that always sends False.
    assert first_call["options"]["store"] is True
    assert second_call["options"]["store"] is True
    assert first_call["options"].get("conversation_id") is None
    assert second_call["options"]["conversation_id"] == "conv-1"
    # The service keeps the history, so the second call carries only the new turn.
    assert [m.text for m in second_call["messages"]] == ["second"]


async def test_store_defaults_to_false(client: Client) -> None:
    factory = fake_chat_client(["one", "two"])
    plugin = MicrosoftAgentFrameworkPlugin(chat_clients={"model": factory})
    async with new_worker(client, DefaultStoreWorkflow, plugin=plugin) as worker:
        result = await client.execute_workflow(
            DefaultStoreWorkflow.run,
            ["first", "second"],
            id=workflow_id("default-store"),
            task_queue=worker.task_queue,
        )
    assert result == {"texts": ["one", "two"], "service_session_id": None}
    first_call, second_call = factory().calls
    # With store unset the provider is told not to keep the conversation, so
    # the session's history is the one transcript and the second call resends it.
    assert first_call["options"]["store"] is False
    assert second_call["options"]["store"] is False
    assert [m.text for m in second_call["messages"]] == ["first", "one", "second"]


@pytest.mark.parametrize(
    ("workflow_type", "provider"),
    [
        (FileHistoryWorkflow, "FileHistoryProvider"),
        (FileSessionStoreWorkflow, "FileSessionStore"),
    ],
    ids=["history-provider", "session-store"],
)
async def test_file_backed_sessions_fail_the_workflow(
    client: Client, tmp_path: Path, workflow_type: type, provider: str
) -> None:
    plugin = MicrosoftAgentFrameworkPlugin(
        chat_clients={"model": fake_chat_client(["unused"])}
    )
    async with new_worker(client, workflow_type, plugin=plugin) as worker:
        handle = await client.start_workflow(
            workflow_type.run,  # type: ignore[attr-defined]
            str(tmp_path / "sessions"),
            id=workflow_id("file-backed"),
            task_queue=worker.task_queue,
        )
        with pytest.raises(WorkflowFailureError) as excinfo:
            await handle.result()
    # The provider reads its file through asyncio.to_thread; the plugin's
    # stand-in turns that into the plugin's own failure, so the workflow fails
    # (not the workflow task) before any model call, and the message names the
    # call. A future release that made this work would double-persist.
    cause = excinfo.value.cause
    assert isinstance(cause, ApplicationError)
    assert cause.type == "MicrosoftAgentFrameworkWorkflowError"
    assert provider in cause.message and "to_dict" in cause.message
    assert await count_scheduled_activities(handle) == {}


class _HistoryProbe(FunctionMiddleware):
    """Records how many messages the session had persisted when a tool ran."""

    def __init__(self, log: list[str]) -> None:
        self.log = log

    async def process(
        self,
        context: FunctionInvocationContext,
        call_next: Callable[[], Awaitable[None]],
    ) -> None:
        session = context.session
        state = session.state.get("in_memory", {}) if session is not None else {}
        self.log.append(f"persisted_before_tool:{len(state.get('messages', []))}")
        await call_next()


@workflow.defn
class PerCallPersistenceWorkflow:
    @workflow.run
    async def run(self, persist_per_call: bool) -> dict[str, Any]:
        log: list[str] = []

        @tool
        async def get_weather(city: str) -> str:
            """Get the weather for a city."""
            return f"rainy in {city}"

        agent = Agent(
            client=TemporalChatClient("model"),
            tools=[get_weather],
            middleware=[_HistoryProbe(log)],
            require_per_service_call_history_persistence=persist_per_call,
        )
        session = agent.create_session()
        response = await agent.run("weather in Oslo?", session=session)
        return {
            "text": response.text,
            "log": log,
            "messages": len(session.state["in_memory"]["messages"]),
        }


@workflow.defn
class CompactionWorkflow:
    @workflow.run
    async def run(self, turns: int) -> dict[str, Any]:
        agent = Agent(
            client=TemporalChatClient("model"),
            instructions="Chat.",
            compaction_strategy=TruncationStrategy(max_n=3, compact_to=1),
        )
        session = agent.create_session()
        texts = [
            (await agent.run(f"turn {i}", session=session)).text for i in range(turns)
        ]
        return {"texts": texts, "messages": len(session.state["in_memory"]["messages"])}


@workflow.defn
class SummarizationWorkflow:
    @workflow.run
    async def run(self, turns: int) -> dict[str, Any]:
        strategy = SummarizationStrategy(
            client=TemporalChatClient("summarizer", summary="summarizer"),
            target_count=2,
            threshold=0,
        )
        agent = Agent(
            client=TemporalChatClient("model"),
            instructions="Chat.",
            compaction_strategy=strategy,
        )
        session = agent.create_session()
        texts = [
            (await agent.run(f"turn {i}", session=session)).text for i in range(turns)
        ]
        return {"texts": texts, "messages": len(session.state["in_memory"]["messages"])}


@pytest.mark.parametrize(
    ("persist_per_call", "persisted_before_tool"), [(True, 2), (False, 0)]
)
async def test_require_per_service_call_history_persistence(
    client: Client, persist_per_call: bool, persisted_before_tool: int
) -> None:
    factory = fake_chat_client(
        [
            function_call_response("get_weather", {"city": "Oslo"}, call_id="call-1"),
            text_response("Rainy."),
        ]
    )
    plugin = MicrosoftAgentFrameworkPlugin(chat_clients={"model": factory})
    async with new_worker(client, PerCallPersistenceWorkflow, plugin=plugin) as worker:
        handle = await client.start_workflow(
            PerCallPersistenceWorkflow.run,
            persist_per_call,
            id=workflow_id("per-call"),
            task_queue=worker.task_queue,
        )
        result = await handle.result()
    assert result["text"] == "Rainy."
    assert result["messages"] == 4
    # With per-service-call persistence the first model call's messages are in
    # the session before the tool runs; otherwise they land after the run.
    assert result["log"] == [f"persisted_before_tool:{persisted_before_tool}"]
    assert await count_scheduled_activities(handle) == {INVOKE_MODEL: 2}


async def test_truncation_compaction_runs_in_workflow(client: Client) -> None:
    factory = fake_chat_client(["r1", "r2", "r3"])
    plugin = MicrosoftAgentFrameworkPlugin(chat_clients={"model": factory})
    async with new_worker(client, CompactionWorkflow, plugin=plugin) as worker:
        result = await client.execute_workflow(
            CompactionWorkflow.run,
            3,
            id=workflow_id("compaction"),
            task_queue=worker.task_queue,
        )
    assert result == {"texts": ["r1", "r2", "r3"], "messages": 6}
    # The session keeps the whole transcript; the strategy trims what the
    # model sees once it passes max_n messages.
    seen = [len(call["messages"]) for call in factory().calls]
    assert seen == [1, 3, 1]


async def test_summarization_strategy_calls_the_summarizer(client: Client) -> None:
    main = fake_chat_client(["r1", "r2", "r3", "r4"])
    summarizer = fake_chat_client(["Summary of the earlier turns."])
    plugin = MicrosoftAgentFrameworkPlugin(
        chat_clients={"model": main, "summarizer": summarizer}
    )
    async with new_worker(client, SummarizationWorkflow, plugin=plugin) as worker:
        handle = await client.start_workflow(
            SummarizationWorkflow.run,
            4,
            id=workflow_id("summarization"),
            task_queue=worker.task_queue,
        )
        result = await handle.result()
    assert result["texts"] == ["r1", "r2", "r3", "r4"]
    # The summary call ran as its own Activity on the summarizer client.
    summaries = [summary_of(e) for e in await scheduled_activity_events(handle)]
    assert "summarizer" in summaries
    assert summarizer().calls
    assert any(
        "Summary of the earlier turns." in m.text
        for call in main().calls
        for m in call["messages"]
    )
