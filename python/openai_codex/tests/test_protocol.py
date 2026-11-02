"""Real Workflow callback, cancellation, and result-loss regressions."""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

import pytest

from temporalio import activity
from temporalio.client import Client, WorkflowFailureError
from temporalio.openai_codex import ConnectActivityInput, OpenAICodexPlugin
from temporalio.openai_codex._types import Message
from temporalio.worker import (
    ActivityInboundInterceptor,
    ExecuteActivityInput,
    Interceptor,
    Replayer,
    Worker,
)
from tests.helpers.protocol import ProtocolConnector, ProtocolProcess
from tests.helpers.workflows import (
    ClosingApprovalWorkflow,
    ConnectingWorkflow,
    ConnectionLossWorkflow,
    InitializingWorkflow,
    JoiningTurnWorkflow,
    PendingWorkflow,
    ProtocolWorkflow,
)


class InitializingProcess(ProtocolProcess):
    def __init__(self) -> None:
        super().__init__()
        self.initializing = asyncio.Event()

    async def write(self, message: Message) -> None:
        if message.get("method") == "initialize":
            self.writes.append(message)
            self.initializing.set()
            return
        await super().write(message)


class JoiningProcess(ProtocolProcess):
    def __init__(self) -> None:
        super().__init__()
        self.started = False
        self.join_request: Message | None = None

    async def write(self, message: Message) -> None:
        method = message.get("method")
        turn = {"id": "turn", "items": [], "status": "inProgress"}
        if method == "turn/start":
            self.writes.append(message)
            if self.started:
                self.join_request = message
                self.output.put_nowait(
                    [
                        {
                            "method": "item/agentMessage/delta",
                            "params": {
                                "threadId": "thread",
                                "turnId": "turn",
                                "itemId": "item",
                                "delta": "during join",
                            },
                        }
                    ]
                )
            else:
                self.started = True
                self.output.put_nowait(
                    [{"id": message["id"], "result": {"turn": turn}}]
                )
        elif method == "release_join":
            self.writes.append(message)
            assert self.join_request is not None
            self.output.put_nowait(
                [
                    {"id": self.join_request["id"], "result": {"turn": turn}},
                    {
                        "method": "turn/completed",
                        "params": {
                            "threadId": "thread",
                            "turn": {**turn, "status": "completed"},
                        },
                    },
                ]
            )
        else:
            await super().write(message)


class LoseResults(Interceptor):
    def intercept_activity(
        self, next: ActivityInboundInterceptor
    ) -> ActivityInboundInterceptor:
        class Inbound(ActivityInboundInterceptor):
            async def execute_activity(self, input: ExecuteActivityInput) -> Any:
                result = await self.next.execute_activity(input)
                if activity.info().attempt == 1 and activity.info().activity_type in {
                    "openai-codex-read",
                    "openai-codex-write",
                }:
                    raise RuntimeError("Injected lost Activity completion")
                return result

        return Inbound(next)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("lost_results", [False, True])
async def test_approval_callbacks(
    client: Client, asynchronous: bool, lost_results: bool
) -> None:
    connector = ProtocolConnector()
    queue = f"protocol-{uuid.uuid4()}"
    async with Worker(
        client,
        task_queue=queue,
        workflows=[ProtocolWorkflow],
        plugins=[OpenAICodexPlugin(connector=connector)],
        interceptors=[LoseResults()] if lost_results else [],
    ):
        handle = await client.start_workflow(
            ProtocolWorkflow.run, asynchronous, id=queue, task_queue=queue
        )
        assert await handle.result() == ["item/commandExecution/requestApproval"]
    process = connector.processes[0]
    assert process.closed
    assert len(process.writes) == (4 if asynchronous else 3)
    assert len({str(msg["id"]) for msg in process.writes}) == len(process.writes)
    await Replayer(
        workflows=[ProtocolWorkflow], plugins=[OpenAICodexPlugin()]
    ).replay_workflow(await handle.fetch_history())


async def test_workflow_cancellation_closes_process(client: Client) -> None:
    connector = ProtocolConnector()
    queue = f"cancel-{uuid.uuid4()}"
    async with Worker(
        client,
        task_queue=queue,
        workflows=[PendingWorkflow],
        plugins=[OpenAICodexPlugin(connector=connector)],
    ):
        handle = await client.start_workflow(
            PendingWorkflow.run, id=queue, task_queue=queue
        )
        import asyncio

        while not connector.processes:
            await asyncio.sleep(0.01)
        await handle.cancel()
        with pytest.raises(WorkflowFailureError):
            await handle.result()
        assert connector.processes[0].closed


async def test_cancellation_with_unavailable_connector_queue(client: Client) -> None:
    connector = ProtocolConnector()
    queue = f"connect-unavailable-{uuid.uuid4()}"
    async with Worker(
        client,
        task_queue=queue,
        workflows=[ConnectingWorkflow],
        plugins=[OpenAICodexPlugin(connector=connector)],
    ):
        handle = await client.start_workflow(
            ConnectingWorkflow.run,
            args=[f"unavailable-{uuid.uuid4()}", False],
            id=queue,
            task_queue=queue,
        )
        while not any(
            event.HasField("activity_task_scheduled_event_attributes")
            for event in (await handle.fetch_history()).events
        ):
            await asyncio.sleep(0.01)
        await handle.cancel()
        with pytest.raises(WorkflowFailureError):
            await asyncio.wait_for(handle.result(), 5)
        assert not connector.processes
    await Replayer(
        workflows=[ConnectingWorkflow], plugins=[OpenAICodexPlugin()]
    ).replay_workflow(await handle.fetch_history())


async def test_cancellation_stops_connect_retries(client: Client) -> None:
    entered = asyncio.Event()
    attempts = 0

    class UnavailableConnector:
        async def connect(self, request: ConnectActivityInput) -> ProtocolProcess:
            nonlocal attempts
            del request
            attempts += 1
            entered.set()
            raise OSError("connector unavailable")

    queue = f"connect-retry-{uuid.uuid4()}"
    async with Worker(
        client,
        task_queue=queue,
        workflows=[ConnectingWorkflow],
        plugins=[OpenAICodexPlugin(connector=UnavailableConnector())],
    ):
        handle = await client.start_workflow(
            ConnectingWorkflow.run, args=[None, False], id=queue, task_queue=queue
        )
        await entered.wait()
        await handle.cancel()
        with pytest.raises(WorkflowFailureError):
            await asyncio.wait_for(handle.result(), 5)
        assert attempts == 1
    await Replayer(
        workflows=[ConnectingWorkflow], plugins=[OpenAICodexPlugin()]
    ).replay_workflow(await handle.fetch_history())


async def test_cancellation_closes_launch_with_lost_result(client: Client) -> None:
    committed = asyncio.Event()

    class LoseConnectResult(Interceptor):
        def intercept_activity(
            self, next: ActivityInboundInterceptor
        ) -> ActivityInboundInterceptor:
            class Inbound(ActivityInboundInterceptor):
                async def execute_activity(self, input: ExecuteActivityInput) -> Any:
                    result = await self.next.execute_activity(input)
                    if activity.info().activity_type == "openai-codex-connect":
                        committed.set()
                        raise RuntimeError("Injected lost launch result")
                    return result

            return Inbound(next)

    connector = ProtocolConnector()
    queue = f"connect-lost-result-{uuid.uuid4()}"
    async with Worker(
        client,
        task_queue=queue,
        workflows=[ConnectingWorkflow],
        plugins=[OpenAICodexPlugin(connector=connector)],
        interceptors=[LoseConnectResult()],
    ):
        handle = await client.start_workflow(
            ConnectingWorkflow.run, args=[None, False], id=queue, task_queue=queue
        )
        await committed.wait()
        await handle.cancel()
        with pytest.raises(WorkflowFailureError):
            await asyncio.wait_for(handle.result(), 5)
        assert connector.connects == 1
        assert connector.processes[0].closed
    await Replayer(
        workflows=[ConnectingWorkflow], plugins=[OpenAICodexPlugin()]
    ).replay_workflow(await handle.fetch_history())


@pytest.mark.parametrize("abandon", [False, True])
async def test_cancelled_launch_closes_late_process(
    client: Client, abandon: bool
) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()
    closed = asyncio.Event()

    class Process(ProtocolProcess):
        async def close(self) -> None:
            await super().close()
            closed.set()

    process = Process()

    class DelayedConnector:
        async def connect(self, request: ConnectActivityInput) -> ProtocolProcess:
            assert request.handle is not None
            entered.set()
            await release.wait()
            return process

    queue = f"connect-late-{uuid.uuid4()}"
    async with Worker(
        client,
        task_queue=queue,
        workflows=[ConnectingWorkflow],
        plugins=[OpenAICodexPlugin(connector=DelayedConnector())],
    ):
        handle = await client.start_workflow(
            ConnectingWorkflow.run, args=[None, abandon], id=queue, task_queue=queue
        )
        try:
            await entered.wait()
            await handle.cancel()
            with pytest.raises(WorkflowFailureError):
                await asyncio.wait_for(handle.result(), 5)
        finally:
            release.set()
        await asyncio.wait_for(closed.wait(), 5)
    await Replayer(
        workflows=[ConnectingWorkflow], plugins=[OpenAICodexPlugin()]
    ).replay_workflow(await handle.fetch_history())


async def test_approval_callback_can_close_client(client: Client) -> None:
    connector = ProtocolConnector()
    queue = f"approval-close-{uuid.uuid4()}"
    async with Worker(
        client,
        task_queue=queue,
        workflows=[ClosingApprovalWorkflow],
        plugins=[OpenAICodexPlugin(connector=connector)],
    ):
        handle = await client.start_workflow(
            ClosingApprovalWorkflow.run, id=queue, task_queue=queue
        )
        assert await asyncio.wait_for(handle.result(), 5)
        assert connector.processes[0].closed
    await Replayer(
        workflows=[ClosingApprovalWorkflow], plugins=[OpenAICodexPlugin()]
    ).replay_workflow(await handle.fetch_history())


@pytest.mark.parametrize("lazy", [False, True])
async def test_initialization_cancellation_closes_process(
    client: Client, lazy: bool
) -> None:
    process = InitializingProcess()
    connector = ProtocolConnector(lambda: process)
    queue = f"init-cancel-{uuid.uuid4()}"
    async with Worker(
        client,
        task_queue=queue,
        workflows=[InitializingWorkflow],
        plugins=[OpenAICodexPlugin(connector=connector)],
    ):
        handle = await client.start_workflow(
            InitializingWorkflow.run, lazy, id=queue, task_queue=queue
        )
        await process.initializing.wait()
        await handle.cancel()
        with pytest.raises(WorkflowFailureError):
            await handle.result()
        assert process.closed
    await Replayer(
        workflows=[InitializingWorkflow], plugins=[OpenAICodexPlugin()]
    ).replay_workflow(await handle.fetch_history())


@pytest.mark.parametrize("close_first", [False, True])
async def test_joining_handle_retains_consumed_notifications(
    client: Client, close_first: bool
) -> None:
    connector = ProtocolConnector(JoiningProcess)
    queue = f"join-{uuid.uuid4()}"
    async with Worker(
        client,
        task_queue=queue,
        workflows=[JoiningTurnWorkflow],
        plugins=[OpenAICodexPlugin(connector=connector)],
    ):
        handle = await client.start_workflow(
            JoiningTurnWorkflow.run, close_first, id=queue, task_queue=queue
        )
        assert await handle.result() == ["item/agentMessage/delta", "turn/completed"]
    await Replayer(
        workflows=[JoiningTurnWorkflow], plugins=[OpenAICodexPlugin()]
    ).replay_workflow(await handle.fetch_history())


async def test_worker_loss_does_not_restart_connection(client: Client) -> None:
    connector = ProtocolConnector()
    queue = f"loss-{uuid.uuid4()}"
    async with Worker(
        client,
        task_queue=queue,
        workflows=[ConnectionLossWorkflow],
        plugins=[OpenAICodexPlugin(connector=connector)],
    ):
        handle = await client.start_workflow(
            ConnectionLossWorkflow.run, id=queue, task_queue=queue
        )
        import asyncio

        while not connector.processes:
            await asyncio.sleep(0.01)
    async with Worker(
        client,
        task_queue=queue,
        workflows=[ConnectionLossWorkflow],
        plugins=[OpenAICodexPlugin(connector=connector)],
    ):
        await handle.signal(ConnectionLossWorkflow.resume)
        assert "Lost the connection to Codex" in await handle.result()
    assert connector.connects == 1


@pytest.mark.parametrize("same_instance", [False, True])
async def test_repeated_plugin_registration(
    client: Client, same_instance: bool
) -> None:
    connector = ProtocolConnector()
    plugin = OpenAICodexPlugin(connector=connector)
    queue = f"duplicate-{uuid.uuid4()}"
    async with Worker(
        client,
        task_queue=queue,
        workflows=[ProtocolWorkflow],
        plugins=[
            plugin,
            plugin if same_instance else OpenAICodexPlugin(connector=connector),
        ],
    ):
        assert await client.execute_workflow(
            ProtocolWorkflow.run, False, id=queue, task_queue=queue
        ) == ["item/commandExecution/requestApproval"]
    assert connector.connects == 1
    assert connector.processes[0].closed


@pytest.mark.parametrize("high_level", [False, True])
@pytest.mark.parametrize("device", [False, True])
async def test_login_completion_before_response(
    client: Client, high_level: bool, device: bool
) -> None:
    from tests.helpers.workflows import LoginWorkflow

    connector = ProtocolConnector()
    queue = f"login-{uuid.uuid4()}"
    async with Worker(
        client,
        task_queue=queue,
        workflows=[LoginWorkflow],
        plugins=[OpenAICodexPlugin(connector=connector)],
    ):
        handle = await client.start_workflow(
            LoginWorkflow.run, args=[high_level, device], id=queue, task_queue=queue
        )
        assert await handle.result() is True
    await Replayer(
        workflows=[LoginWorkflow], plugins=[OpenAICodexPlugin()]
    ).replay_workflow(await handle.fetch_history())


async def test_broken_pipe_still_closes_owned_connection(client: Client) -> None:
    from tests.helpers.workflows import DisconnectWorkflow

    connector = ProtocolConnector()
    queue = f"disconnect-{uuid.uuid4()}"
    async with Worker(
        client,
        task_queue=queue,
        workflows=[DisconnectWorkflow],
        plugins=[OpenAICodexPlugin(connector=connector)],
    ):
        assert (
            await client.execute_workflow(
                DisconnectWorkflow.run, id=queue, task_queue=queue
            )
            == "broken pipe"
        )
        assert connector.processes[0].closed
