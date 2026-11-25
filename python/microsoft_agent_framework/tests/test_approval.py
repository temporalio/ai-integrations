"""Function approval with Signals, Updates and Queries."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

import pytest
from agent_framework import Agent, Content, tool

from temporalio import workflow
from temporalio.client import Client, WorkflowHandle, WorkflowUpdateFailedError
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
    contents_summary,
    count_scheduled_activities,
    new_worker,
    workflow_id,
)


@dataclass
class Decision:
    request_id: str
    approved: bool


@workflow.defn
class ApprovalWorkflow:
    def __init__(self) -> None:
        self._decisions: dict[str, bool] = {}
        self._pending: list[dict[str, Any]] = []

    @workflow.signal
    def approve(self, decision: Decision) -> None:
        self._decisions[decision.request_id] = decision.approved

    @workflow.update
    async def decide(self, decision: Decision) -> str:
        if decision.request_id not in {request["id"] for request in self._pending}:
            raise ApplicationError(
                f"unknown or stale approval request {decision.request_id!r}",
                non_retryable=True,
            )
        self._decisions[decision.request_id] = decision.approved
        return "recorded"

    @decide.validator
    def validate_decide(self, decision: Decision) -> None:
        # Shape only: the handler checks the id against workflow state, which a
        # validator cannot rely on under the time-skipping test server.
        if not decision.request_id:
            raise ValueError("request_id is required")

    @workflow.query
    def pending_requests(self) -> list[dict[str, Any]]:
        return self._pending

    @workflow.run
    async def run(self, prompt: str) -> dict[str, Any]:
        log: list[str] = []

        @tool(approval_mode="always_require")
        async def delete_thing(name: str) -> str:
            """Delete a thing. Needs approval."""
            log.append(name)
            return f"deleted {name}"

        agent = Agent(client=TemporalChatClient("model"), tools=[delete_thing])
        session = agent.create_session()
        response = await agent.run(prompt, session=session)
        requests = response.user_input_requests
        ids = [request.id or "" for request in requests]
        self._pending = [request.to_dict() for request in requests]
        await workflow.wait_condition(lambda: all(id in self._decisions for id in ids))
        replies = [
            request.to_function_approval_response(self._decisions[id])
            for request, id in zip(requests, ids, strict=True)
        ]
        self._pending = []
        final = await agent.run(replies, session=session)
        return {
            "requests": [
                (
                    request.type,
                    request.id,
                    request.function_call.name if request.function_call else None,
                )
                for request in requests
            ],
            "text": final.text,
            "contents": contents_summary(final),
            "log": log,
            "session_state": sorted(session.state),
        }


def _script(*, with_id: bool = True) -> list[Any]:
    return [
        function_call_response(
            "delete_thing",
            {"name": "x"},
            call_id="call-1",
            id="fc-1" if with_id else None,
        ),
        text_response("Deleted x."),
    ]


async def _pending(handle: WorkflowHandle) -> list[dict[str, Any]]:
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        pending = await handle.query(ApprovalWorkflow.pending_requests)
        if pending:
            return pending
        await asyncio.sleep(0.1)
    raise AssertionError("the workflow never surfaced an approval request")


async def test_approval_via_signal(client: Client) -> None:
    plugin = MicrosoftAgentFrameworkPlugin(
        chat_clients={"model": fake_chat_client(_script())}
    )
    async with new_worker(client, ApprovalWorkflow, plugin=plugin) as worker:
        handle = await client.start_workflow(
            ApprovalWorkflow.run,
            "delete x",
            id=workflow_id("approval-signal"),
            task_queue=worker.task_queue,
        )
        (pending,) = await _pending(handle)
        assert pending["type"] == "function_approval_request"
        assert pending["function_call"]["name"] == "delete_thing"
        await handle.signal(ApprovalWorkflow.approve, Decision(pending["id"], True))
        result = await handle.result()
    assert result["requests"] == [["function_approval_request", "fc-1", "delete_thing"]]
    assert result["log"] == ["x"]
    assert result["text"] == "Deleted x."
    assert "tool_approval" in result["session_state"]
    assert await count_scheduled_activities(handle) == {INVOKE_MODEL: 2}


async def test_approval_via_update(client: Client) -> None:
    plugin = MicrosoftAgentFrameworkPlugin(
        chat_clients={"model": fake_chat_client(_script())}
    )
    async with new_worker(client, ApprovalWorkflow, plugin=plugin) as worker:
        handle = await client.start_workflow(
            ApprovalWorkflow.run,
            "delete x",
            id=workflow_id("approval-update"),
            task_queue=worker.task_queue,
        )
        (pending,) = await _pending(handle)
        with pytest.raises(WorkflowUpdateFailedError) as failed:
            await handle.execute_update(
                ApprovalWorkflow.decide, Decision("bogus", True)
            )
        assert isinstance(failed.value.cause, ApplicationError)
        outcome = await handle.execute_update(
            ApprovalWorkflow.decide, Decision(pending["id"], True)
        )
        result = await handle.result()
    assert outcome == "recorded"
    assert result["log"] == ["x"]


async def test_pending_requests_query(client: Client) -> None:
    plugin = MicrosoftAgentFrameworkPlugin(
        chat_clients={"model": fake_chat_client(_script())}
    )
    async with new_worker(client, ApprovalWorkflow, plugin=plugin) as worker:
        handle = await client.start_workflow(
            ApprovalWorkflow.run,
            "delete x",
            id=workflow_id("approval-query"),
            task_queue=worker.task_queue,
        )
        (pending,) = await _pending(handle)
        # The query returns the request's dict form, which rebuilds as content.
        request = Content.from_dict(pending)
        assert request.type == "function_approval_request"
        assert request.function_call is not None
        assert request.function_call.arguments == {"name": "x"}
        await handle.signal(ApprovalWorkflow.approve, Decision(request.id or "", True))
        await handle.result()
        assert await handle.query(ApprovalWorkflow.pending_requests) == []


async def test_rejected_approval(client: Client) -> None:
    plugin = MicrosoftAgentFrameworkPlugin(
        chat_clients={"model": fake_chat_client(_script())}
    )
    async with new_worker(client, ApprovalWorkflow, plugin=plugin) as worker:
        handle = await client.start_workflow(
            ApprovalWorkflow.run,
            "delete x",
            id=workflow_id("approval-reject"),
            task_queue=worker.task_queue,
        )
        (pending,) = await _pending(handle)
        await handle.signal(ApprovalWorkflow.approve, Decision(pending["id"], False))
        result = await handle.result()
    assert result["log"] == []
    (rejection,) = [c for c in result["contents"] if c.startswith("result:")]
    assert "rejected" in rejection
    assert await count_scheduled_activities(handle) == {INVOKE_MODEL: 2}


async def test_approval_survives_worker_restart(client: Client) -> None:
    # The provider omits the function call id, so the id that keys the
    # approval is minted on the Activity side and recorded in history.
    plugin = MicrosoftAgentFrameworkPlugin(
        chat_clients={"model": fake_chat_client(_script(with_id=False))}
    )
    task_queue = f"agent-framework-restart-{workflow_id('q')}"
    async with new_worker(
        client, ApprovalWorkflow, plugin=plugin, task_queue=task_queue
    ):
        handle = await client.start_workflow(
            ApprovalWorkflow.run,
            "delete x",
            id=workflow_id("approval-restart"),
            task_queue=task_queue,
        )
        (pending,) = await _pending(handle)
    assert pending["id"].startswith("af-call-")

    async with new_worker(
        client, ApprovalWorkflow, plugin=plugin, task_queue=task_queue
    ):
        (still_pending,) = await _pending(handle)
        assert still_pending["id"] == pending["id"]
        await handle.signal(ApprovalWorkflow.approve, Decision(pending["id"], True))
        result = await handle.result()
    assert result["log"] == ["x"]
    assert result["requests"][0][1] == pending["id"]
    assert await count_scheduled_activities(handle) == {INVOKE_MODEL: 2}
