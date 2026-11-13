"""Sessions: continue-as-new carry, service-side conversations, file providers."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_framework import (
    Agent,
    AgentSession,
    ChatResponse,
    Content,
    FileHistoryProvider,
    Message,
)

from temporalio import workflow
from temporalio.agent_framework import AgentFrameworkPlugin, TemporalChatClient
from temporalio.agent_framework.testing import fake_chat_client
from temporalio.client import Client, WorkflowExecutionStatus, WorkflowHandle
from tests.helpers.agent_framework import (
    count_scheduled_activities,
    new_worker,
    role_of,
    workflow_id,
)


@dataclass
class ChatInput:
    session: dict[str, Any] | None = None
    continue_after_turns: int | None = 1


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
                # The continued run carries the session and keeps going until end_chat.
                workflow.continue_as_new(
                    ChatInput(session=session.to_dict(), continue_after_turns=None)
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
class FileHistoryWorkflow:
    @workflow.run
    async def run(self, storage_path: str) -> str:
        agent = Agent(
            client=TemporalChatClient("model"),
            context_providers=[FileHistoryProvider(storage_path)],
        )
        session = agent.create_session()
        try:
            await agent.run("hello", session=session)
        except Exception as err:
            return type(err).__name__
        return "ran"


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
    plugin = AgentFrameworkPlugin(chat_clients={"model": factory})
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


async def test_store_true_uses_service_conversation(client: Client) -> None:
    def stored(text: str) -> ChatResponse:
        return ChatResponse(
            messages=[Message(role="assistant", contents=[Content.from_text(text)])],
            conversation_id="conv-1",
        )

    factory = fake_chat_client([stored("one"), stored("two")])
    plugin = AgentFrameworkPlugin(chat_clients={"model": factory})
    async with new_worker(client, StoredConversationWorkflow, plugin=plugin) as worker:
        result = await client.execute_workflow(
            StoredConversationWorkflow.run,
            ["first", "second"],
            id=workflow_id("stored"),
            task_queue=worker.task_queue,
        )
    assert result == {"texts": ["one", "two"], "service_session_id": "conv-1"}
    first_call, second_call = factory().calls
    assert first_call["options"].get("conversation_id") is None
    assert second_call["options"]["conversation_id"] == "conv-1"
    # The service keeps the history, so the second call carries only the new turn.
    assert [m.text for m in second_call["messages"]] == ["second"]


async def test_file_history_provider_fails_clearly(
    client: Client, tmp_path: Path
) -> None:
    plugin = AgentFrameworkPlugin(chat_clients={"model": fake_chat_client(["unused"])})
    async with new_worker(client, FileHistoryWorkflow, plugin=plugin) as worker:
        handle = await client.start_workflow(
            FileHistoryWorkflow.run,
            str(tmp_path / "history"),
            id=workflow_id("file-history"),
            task_queue=worker.task_queue,
        )
        result = await handle.result()
    # The provider reads its file through asyncio.to_thread, which the
    # workflow event loop does not implement, so the run fails before any
    # model call. A future release that made this work would double-persist.
    assert result == "NotImplementedError"
    assert await count_scheduled_activities(handle) == {}
