"""Real Workflow callback, cancellation, and result-loss regressions."""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

import pytest
from openai_codex import CodexConfig, CodexError

from temporalio import activity
from temporalio.client import Client, WorkflowFailureError
from temporalio.contrib.workflow_streams import WorkflowStreamClient
from temporalio.openai_codex import ConnectActivityInput, OpenAICodexPlugin
from temporalio.openai_codex import _plugin as plugin_module
from temporalio.openai_codex._types import Message
from temporalio.runtime import Runtime, TelemetryConfig
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
    StreamingWorkflow,
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


@pytest.mark.parametrize("lost_results", [False, True])
async def test_external_stream_preserves_turn_stream_and_replay(
    client: Client, lost_results: bool
) -> None:
    deltas: list[Message] = [
        {
            "method": method,
            "params": {
                "threadId": "thread",
                "turnId": "turn",
                "itemId": "item",
                "delta": text,
                **({"contentIndex": 0} if "reasoning" in method else {}),
            },
        }
        for method, text in [
            ("item/agentMessage/delta", "hello"),
            ("item/reasoning/textDelta", "thinking"),
            ("item/commandExecution/outputDelta", "output"),
        ]
    ]

    class StreamingProcess(ProtocolProcess):
        async def write(self, message: Message) -> None:
            if message.get("method") != "turn/start":
                await super().write(message)
                return
            self.writes.append(message)
            self.output.put_nowait(
                [
                    {
                        "id": message["id"],
                        "result": {
                            "turn": {"id": "turn", "items": [], "status": "inProgress"}
                        },
                    },
                    *deltas,
                    {
                        "method": "turn/completed",
                        "params": {
                            "threadId": "thread",
                            "turn": {"id": "turn", "items": [], "status": "completed"},
                        },
                    },
                ]
            )

    class StreamingConnector(ProtocolConnector):
        async def connect(self, request: ConnectActivityInput) -> ProtocolProcess:
            assert type(request.config) is CodexConfig
            assert request.config == CodexConfig(cwd="/workspace")
            await super().connect(request)
            return self.processes[-1]

    connector = StreamingConnector(StreamingProcess)
    queue = f"streaming-{uuid.uuid4()}"
    async with Worker(
        client,
        task_queue=queue,
        workflows=[StreamingWorkflow],
        plugins=[OpenAICodexPlugin(connector=connector)],
        interceptors=[LoseResults()] if lost_results else [],
    ):
        handle = await client.start_workflow(
            StreamingWorkflow.run, id=queue, task_queue=queue
        )
        stream = WorkflowStreamClient.create(client, handle.id)
        items: list[Message] = []
        async for item in stream.subscribe(["codex"], result_type=dict):
            items.append(item.data)
            if len(items) == len(deltas):
                break
        await handle.signal(StreamingWorkflow.finish)
        assert await handle.result() == [
            *(delta["method"] for delta in deltas),
            "turn/completed",
        ]
        # Read the stream again after completion, including any duplicate publications.
        assert [
            item.data async for item in stream.subscribe(["codex"], result_type=dict)
        ] == deltas
        assert items == deltas
    assert connector.processes[0].closed
    await Replayer(
        workflows=[StreamingWorkflow], plugins=[OpenAICodexPlugin()]
    ).replay_workflow(await handle.fetch_history())


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
    assert len(process.writes) == (6 if asynchronous else 5)
    request_ids = [str(msg["id"]) for msg in process.writes if "id" in msg]
    assert len(set(request_ids)) == len(request_ids)
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
        while not await handle.query(PendingWorkflow.is_connected):
            await asyncio.sleep(0.01)
        await handle.cancel()
        with pytest.raises(WorkflowFailureError):
            await handle.result()
        assert connector.processes[0].closed


async def test_close_activity_failure_preserves_success_and_replay(
    client: Client,
) -> None:
    class FailingCloseProcess(ProtocolProcess):
        attempts = 0

        async def close(self) -> None:
            self.attempts += 1
            if self.attempts == 1:
                raise CodexError("cleanup unavailable")
            await super().close()

    process = FailingCloseProcess()
    connector = ProtocolConnector(lambda: process)
    queue = f"close-failure-{uuid.uuid4()}"
    async with Worker(
        client,
        task_queue=queue,
        workflows=[InitializingWorkflow],
        plugins=[OpenAICodexPlugin(connector=connector)],
    ):
        handle = await client.start_workflow(
            InitializingWorkflow.run, False, id=queue, task_queue=queue
        )
        await handle.result()
        assert process.attempts == 1 and not process.closed
    assert process.closed
    history = await handle.fetch_history()
    assert not any(
        event.HasField("workflow_task_failed_event_attributes")
        for event in history.events
    )
    await Replayer(
        workflows=[InitializingWorkflow], plugins=[OpenAICodexPlugin()]
    ).replay_workflow(history)


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


async def test_worker_shutdown_closes_launch_with_lost_result(client: Client) -> None:
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
        assert not connector.processes[0].closed
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
            del request
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
    assert closed.is_set()
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
@pytest.mark.parametrize("on_client", [False, True])
async def test_repeated_plugin_registration(
    client: Client, same_instance: bool, on_client: bool
) -> None:
    connector = ProtocolConnector()
    plugin = OpenAICodexPlugin(connector=connector)
    queue = f"duplicate-{uuid.uuid4()}"
    other = plugin if same_instance else OpenAICodexPlugin(connector=connector)
    if on_client:
        client = Client(client.service_client, plugins=[plugin])
        with pytest.warns(UserWarning, match="same plugin type"):
            worker = Worker(
                client,
                task_queue=queue,
                workflows=[ProtocolWorkflow],
                plugins=[other],
            )
    else:
        worker = Worker(
            client,
            task_queue=queue,
            workflows=[ProtocolWorkflow],
            plugins=[plugin, other],
        )
    async with worker:
        assert await client.execute_workflow(
            ProtocolWorkflow.run, False, id=queue, task_queue=queue
        ) == ["item/commandExecution/requestApproval"]
    assert connector.connects == 1
    assert connector.processes[0].closed


async def test_shared_plugin_supports_workers_on_the_same_queue(
    client: Client, monkeypatch: pytest.MonkeyPatch
) -> None:
    connector = ProtocolConnector()
    plugin = OpenAICodexPlugin(connector=connector)
    queue = f"shared-workers-{uuid.uuid4()}"
    connector_workers: list[Worker] = []
    connectors_started = asyncio.Event()

    def create_connector_worker(*args: Any, **kwargs: Any) -> Worker:
        worker = Worker(*args, **kwargs)
        connector_workers.append(worker)
        if len(connector_workers) == 2:
            connectors_started.set()
        return worker

    monkeypatch.setattr(plugin_module, "Worker", create_connector_worker)
    first = Worker(
        client, task_queue=queue, workflows=[ProtocolWorkflow], plugins=[plugin]
    )
    # Core disallows overlapping pollers in one Runtime; each Worker needs its own.
    second_client = await Client.connect(
        client.service_client.config.target_host,
        namespace=client.namespace,
        runtime=Runtime(telemetry=TelemetryConfig()),
    )
    second = Worker(
        second_client, task_queue=queue, workflows=[ProtocolWorkflow], plugins=[plugin]
    )
    async with first, second:
        await asyncio.wait_for(connectors_started.wait(), 5)
        assert len(connector_workers) == 2
        assert len({worker.task_queue for worker in connector_workers}) == 2
        results = await asyncio.gather(
            *(
                client.execute_workflow(
                    ProtocolWorkflow.run, False, id=f"{queue}-{index}", task_queue=queue
                )
                for index in range(2)
            )
        )
        assert results == [["item/commandExecution/requestApproval"]] * 2
    assert connector.connects == 2
    assert all(process.closed for process in connector.processes)


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
