# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Integration tests for the stateful (pooled) MCP toolset support in
``temporalio.google_adk._mcp``.

These drive the real ``connect`` -> dedicated-worker -> ``get_tools`` path
through a Temporal workflow. The first tests use an in-memory fake ``McpToolset``
(no subprocess, no real MCP server). They prove the properties the reviewer
asked for: one toolset is built per workflow run, ``factory_argument`` is
consumed exactly once per run, distinct runs never share a toolset, and the
toolset is torn down when the workflow completes. Most other tests run a real
MCP server over stdio.
"""

import asyncio
import dataclasses
import sys
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from types import FrameType, SimpleNamespace
from typing import Any, cast

import pytest
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from google.adk.tools.tool_context import ToolContext
from mcp import StdioServerParameters

from temporalio import workflow
from temporalio.api.common.v1 import Payload, WorkflowExecution
from temporalio.api.enums.v1 import EventType, PendingActivityState
from temporalio.api.workflow.v1 import PendingActivityInfo
from temporalio.api.workflowservice.v1 import (
    ResetActivityRequest,
    ResetWorkflowExecutionRequest,
)
from temporalio.client import (
    Client,
    DescribeWorkflowInput,
    Interceptor,
    OutboundInterceptor,
    WorkflowFailureError,
    WorkflowHandle,
    WorkflowHistory,
)
from temporalio.common import RetryPolicy
from temporalio.converter import (
    DefaultFailureConverterWithEncodedAttributes,
    PayloadCodec,
)
from temporalio.exceptions import ActivityError, ApplicationError, CancelledError
from temporalio.service import RPCError, RPCStatusCode
from temporalio.worker import Replayer, Worker
from temporalio.workflow import ActivityCancellationType, ActivityConfig
from tests.helpers.stateful_replay import (
    StatefulMcpCancelWaitWorkflow,
    StatefulMcpCancelWorkflow,
    StatefulMcpGatherWorkflow,
)

with workflow.unsafe.imports_passed_through():
    from google.adk.events import EventActions

    from temporalio.google_adk import (
        GoogleAdkPlugin,
        TemporalStatefulMcpToolSet,
        TemporalStatefulMcpToolSetProvider,
        _mcp,
    )

# Populated inside the activity worker process (same process as the test) each
# time the toolset factory runs, so tests can inspect what was built.
CREATED: list["_FakeToolset"] = []


class _FakeTool:
    def __init__(self, name: str) -> None:
        self.name = name
        self.description = "a fake tool"
        self.is_long_running = False
        self.custom_metadata: dict[str, Any] | None = None

    def _get_declaration(self) -> None:
        return None

    async def run_async(
        self,
        *,
        args: dict[str, Any],
        tool_context: Any,  # pyright: ignore[reportUnusedParameter]
    ) -> Any:
        return {"echo": args}


class _FakeToolset:
    """In-memory stand-in for ``google.adk.tools.mcp_tool.McpToolset``."""

    def __init__(self, factory_argument: Any = None) -> None:
        self.factory_argument = factory_argument
        self.get_tools_calls = 0
        self.closed = False

    async def get_tools(self) -> list[_FakeTool]:
        self.get_tools_calls += 1
        return [_FakeTool("echo")]

    async def close(self) -> None:
        self.closed = True


def _factory(arg: Any) -> McpToolset:
    toolset = _FakeToolset(arg)
    CREATED.append(toolset)
    return cast(McpToolset, cast(object, toolset))


@workflow.defn
class StatefulMcpWorkflow:
    @workflow.run
    async def run(self, factory_argument: Any | None) -> list[str]:
        async with TemporalStatefulMcpToolSet(
            "stateful_set",
            factory_argument=factory_argument,
        ) as toolset:
            # Two calls against the same persistent toolset -- exercises reuse.
            tools_1 = await toolset.get_tools()
            tools_2 = await toolset.get_tools()
            return [t.name for t in tools_1] + [t.name for t in tools_2]


def _make_client(
    client: Client, provider: TemporalStatefulMcpToolSetProvider
) -> Client:
    new_config = client.config()
    new_config["plugins"] = [GoogleAdkPlugin(toolset_providers=[provider])]
    return Client(**new_config)


async def test_stateful_one_toolset_per_run_and_teardown(client: Client):
    CREATED.clear()
    provider = TemporalStatefulMcpToolSetProvider("stateful_set", _factory)
    client = _make_client(client, provider)

    async with Worker(
        client,
        task_queue="adk-stateful-mcp",
        workflows=[StatefulMcpWorkflow],
        max_cached_workflows=0,
    ):
        result = await client.execute_workflow(
            StatefulMcpWorkflow.run,
            None,
            id=f"stateful-mcp-{uuid.uuid4()}",
            task_queue="adk-stateful-mcp",
            execution_timeout=timedelta(seconds=60),
        )

    assert result == ["echo", "echo"]
    # Exactly one toolset built for the whole run...
    assert len(CREATED) == 1
    # ...reused across both get_tools calls (state maintained, not per-call)...
    assert CREATED[0].get_tools_calls == 2
    # ...and closed when the workflow completed (no leak).
    assert CREATED[0].closed
    # The dedicated-worker toolset registry is empty after teardown.
    assert provider._toolsets == {}


async def test_stateful_factory_argument_consumed_once(client: Client):
    CREATED.clear()
    provider = TemporalStatefulMcpToolSetProvider("stateful_set", _factory)
    client = _make_client(client, provider)

    async with Worker(
        client,
        task_queue="adk-stateful-mcp-arg",
        workflows=[StatefulMcpWorkflow],
        max_cached_workflows=0,
    ):
        await client.execute_workflow(
            StatefulMcpWorkflow.run,
            {"tenant": "acme"},
            id=f"stateful-mcp-{uuid.uuid4()}",
            task_queue="adk-stateful-mcp-arg",
            execution_timeout=timedelta(seconds=60),
        )

    assert len(CREATED) == 1
    assert CREATED[0].factory_argument == {"tenant": "acme"}


async def test_stateful_no_cross_run_sharing(client: Client):
    CREATED.clear()
    provider = TemporalStatefulMcpToolSetProvider("stateful_set", _factory)
    client = _make_client(client, provider)

    async with Worker(
        client,
        task_queue="adk-stateful-mcp-iso",
        workflows=[StatefulMcpWorkflow],
        max_cached_workflows=0,
    ):
        await client.execute_workflow(
            StatefulMcpWorkflow.run,
            "tenant-a",
            id=f"stateful-mcp-a-{uuid.uuid4()}",
            task_queue="adk-stateful-mcp-iso",
            execution_timeout=timedelta(seconds=60),
        )
        await client.execute_workflow(
            StatefulMcpWorkflow.run,
            "tenant-b",
            id=f"stateful-mcp-b-{uuid.uuid4()}",
            task_queue="adk-stateful-mcp-iso",
            execution_timeout=timedelta(seconds=60),
        )

    # Two distinct runs -> two distinct toolsets, each with its own argument.
    assert len(CREATED) == 2
    assert {t.factory_argument for t in CREATED} == {"tenant-a", "tenant-b"}
    assert all(t.closed for t in CREATED)
    assert provider._toolsets == {}


_MCP_SERVER = """
try:
    from mcp.server.mcpserver import MCPServer
except ImportError:  # MCP 1.x
    from mcp.server.fastmcp import FastMCP as MCPServer

server = MCPServer("echo")


@server.tool()
def echo(text: str) -> str:
    return text


server.run()
"""


def _stdio_toolset(server: str, *, timeout: float = 30) -> McpToolset:
    return McpToolset(
        connection_params=StdioConnectionParams(
            server_params=StdioServerParameters(
                command=sys.executable, args=["-c", server]
            ),
            timeout=timeout,
        )
    )


@dataclass
class _ReplacedInput:
    name: str = "stateful_server"
    heartbeat_timeout: float | None = None
    start_to_close_timeout: float = 20
    schedule_to_start_timeout: bool = True
    session_task_queue: str | None = None
    wait_cancellation: bool = False
    abandon: bool = False
    wait_for_session_end: bool = False


@dataclass
class _RaisedError:
    type: str
    application_type: str | None
    non_retryable: bool
    message: str


@workflow.defn
class StatefulMcpWorkerReplacedWorkflow:
    def __init__(self) -> None:
        self._listed: list[str] | None = None
        self._resume = False
        self._raised: list[_RaisedError] = []

    @workflow.run
    async def run(self, input: _ReplacedInput) -> list[str]:
        config = ActivityConfig(
            start_to_close_timeout=timedelta(seconds=input.start_to_close_timeout),
            retry_policy=RetryPolicy(maximum_attempts=1),
        )
        if input.schedule_to_start_timeout:
            config["schedule_to_start_timeout"] = timedelta(seconds=5)
        if input.wait_cancellation:
            config["cancellation_type"] = (
                ActivityCancellationType.WAIT_CANCELLATION_COMPLETED
            )
        if input.abandon:
            config["cancellation_type"] = ActivityCancellationType.ABANDON
        server_session_config = ActivityConfig(
            start_to_close_timeout=timedelta(hours=1)
        )
        if input.heartbeat_timeout is not None:
            server_session_config["heartbeat_timeout"] = timedelta(
                seconds=input.heartbeat_timeout
            )
        if input.session_task_queue:
            server_session_config["task_queue"] = input.session_task_queue
        async with TemporalStatefulMcpToolSet(
            input.name, config=config, server_session_config=server_session_config
        ) as toolset:
            try:
                self._listed = [t.name for t in await toolset.get_tools()]
            except ActivityError:
                self._listed = []
            await workflow.wait_condition(lambda: self._resume)
            if input.wait_for_session_end:
                # A failure in server history can precede workflow delivery.
                # Wait until the workflow has observed the session ending.
                session = toolset._connect_handle
                assert session is not None
                await workflow.wait_condition(session.done)
            try:
                return [t.name for t in await toolset.get_tools()]
            except ApplicationError as err:
                # What a caller sees inside the block, before cleanup.
                cause: BaseException | None = err
                while cause is not None:
                    self._raised.append(
                        _RaisedError(
                            type(cause).__name__,
                            getattr(cause, "type", None),
                            getattr(cause, "non_retryable", False),
                            str(cause),
                        )
                    )
                    cause = cause.__cause__
                raise

    @workflow.query
    def raised(self) -> list[_RaisedError]:
        return self._raised

    @workflow.update
    async def wait_listed(self) -> list[str]:
        await workflow.wait_condition(lambda: self._listed is not None)
        return self._listed or []

    @workflow.signal
    def resume(self) -> None:
        self._resume = True


async def _session_activity(
    handle: WorkflowHandle[Any, Any],
) -> PendingActivityInfo | None:
    description = await handle.describe()
    for pending in description.raw_description.pending_activities:
        if pending.activity_type.name.endswith("-server-session"):
            return pending
    return None


async def _has_event(handle: WorkflowHandle[Any, Any], event_type: int) -> bool:
    async for event in handle.fetch_history_events():
        if event.event_type == event_type:
            return True
    return False


def _assert_refused(err: WorkflowFailureError) -> None:
    # What leaves the block is the type the docstrings name.
    assert isinstance(err.cause, ApplicationError)
    assert err.cause.type == "DedicatedWorkerFailure"
    # Refused by the retried session activity, not only by the next
    # operation timing out.
    cause: BaseException | None = err.cause
    while cause is not None and not (
        isinstance(cause, ApplicationError) and cause.non_retryable
    ):
        cause = cause.__cause__
    assert isinstance(cause, ApplicationError)
    assert cause.type == "DedicatedWorkerFailure"


def _encoding_in(name: str) -> bool:
    frame: FrameType | None = sys._getframe()
    while frame and frame.f_code.co_name != name:
        frame = frame.f_back
    return frame is not None


class _DelayMayBeOpenCodec(PayloadCodec):
    """Delays the worker's "may be open" heartbeats, and longer its activity failures.

    A "may be open" heartbeat sent before the attempt clears it then finishes
    encoding after the clearing one, but before the attempt's failure.
    """

    def __init__(self) -> None:
        self.heartbeat_delays = 0
        self.failure_delays = 0

    async def encode(self, payloads: Sequence[Payload]) -> list[Payload]:
        if _encoding_in("_heartbeat_async"):
            if any(_mcp._SESSION_MAY_BE_OPEN.encode() in p.data for p in payloads):
                self.heartbeat_delays += 1
                await asyncio.sleep(1.5)
        elif _encoding_in("_handle_start_activity_task") and _encoding_in(
            "encode_failure"
        ):
            self.failure_delays += 1
            await asyncio.sleep(3)
        return list(payloads)

    async def decode(self, payloads: Sequence[Payload]) -> list[Payload]:
        return list(payloads)


@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "case",
    [
        "session_opened",
        "heartbeat_details_replaced",
        "first_connection_fails",
        "first_connection_fails_delayed_heartbeats",
        "first_connection_fails_close_wait_ends_delayed_heartbeats",
        "heartbeat_timeout_session_opened",
        "heartbeat_timeout_first_connection_fails",
    ],
)
async def test_stateful_session_after_worker_replaced(
    client: Client, case: str, monkeypatch: pytest.MonkeyPatch
):
    first_connection_fails = "first_connection_fails" in case
    heartbeat_timeout = case.startswith("heartbeat_timeout")
    delayed_heartbeats = case.endswith("delayed_heartbeats")
    if heartbeat_timeout or delayed_heartbeats:
        monkeypatch.setattr(_mcp, "_HEARTBEAT_INTERVAL", 1.0)
    if "close_wait_ends" in case:
        # The attempt stops waiting for a heartbeat still being encoded before
        # it finishes, and the failure is encoded after it would.
        monkeypatch.setattr(_mcp, "_CLOSE_TIMEOUT", 0.5)
    # Only the test watchdog should expire during graceful replacement.
    # Service timeouts must not turn a slow runner into a lost session.
    activity_timeout = timedelta(hours=1).total_seconds()
    monkeypatch.setattr(_mcp, "_STORE_TIMEOUT", activity_timeout)
    codec = _DelayMayBeOpenCodec()
    if delayed_heartbeats:
        # Each heartbeat takes longer to encode than the interval, so one is
        # always being encoded when the worker shuts down.
        config = client.config()
        config["data_converter"] = dataclasses.replace(
            client.data_converter,
            payload_codec=codec,
            failure_converter_class=DefaultFailureConverterWithEncodedAttributes,
        )
        client = Client(**config)
    factory_calls = 0

    def factory(_: Any) -> McpToolset:
        nonlocal factory_calls
        factory_calls += 1
        server = (
            "pass" if first_connection_fails and factory_calls == 1 else _MCP_SERVER
        )
        return _stdio_toolset(server, timeout=activity_timeout)

    provider = TemporalStatefulMcpToolSetProvider("stateful_server", factory)
    client = _make_client(client, provider)
    task_queue = f"adk-stateful-mcp-replaced-{uuid.uuid4()}"

    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[StatefulMcpWorkerReplacedWorkflow],
        max_cached_workflows=0,
    ):
        handle = await client.start_workflow(
            StatefulMcpWorkerReplacedWorkflow.run,
            _ReplacedInput(
                heartbeat_timeout=activity_timeout if heartbeat_timeout else None,
                start_to_close_timeout=activity_timeout,
                schedule_to_start_timeout=False,
                wait_for_session_end=not first_connection_fails,
            ),
            id=f"stateful-mcp-replaced-{uuid.uuid4()}",
            task_queue=task_queue,
        )
        listed = await handle.execute_update(
            StatefulMcpWorkerReplacedWorkflow.wait_listed
        )
        assert listed == ([] if first_connection_fails else ["echo"])
        # Listing tools runs only after the session's first heartbeat is stored.
        if case == "heartbeat_details_replaced":
            # Replace the session activity's heartbeat details from outside the
            # worker. Its failure on shutdown carries the marker again.
            session = await _session_activity(handle)
            assert session is not None
            await client.get_async_activity_handle(
                workflow_id=handle.id,
                run_id=handle.result_run_id,
                activity_id=session.activity_id,
            ).heartbeat()

    # The first worker shut down gracefully, as in a deployment, and Temporal
    # retries the session activity on the next one.
    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[StatefulMcpWorkerReplacedWorkflow],
        max_cached_workflows=0,
    ):
        # The workflow waits for session failure before its next operation.
        await handle.signal(StatefulMcpWorkerReplacedWorkflow.resume)
        if first_connection_fails:
            # No session was opened, so the retry builds the toolset again,
            # also with a heartbeat timeout: a clean shutdown clears its
            # "may be open" marker.
            assert await handle.result() == ["echo"]
            assert factory_calls == 2
            if delayed_heartbeats:
                # The SDK functions the codec looks for still exist.
                assert codec.heartbeat_delays and codec.failure_delays
            return
        with pytest.raises(WorkflowFailureError) as err:
            await handle.result()

    # A new toolset would open a new MCP session without the old one's state.
    assert factory_calls == 1
    _assert_refused(err.value)


@pytest.mark.timeout(90)
@pytest.mark.parametrize(
    "case",
    [
        "operation_waiting",
        "operation_waiting_abandon",
        "operation_waiting_wait_cancellation",
        "operation_after_refusal",
        "operation_after_refusal_wait_cancellation",
    ],
)
async def test_stateful_operations_fail_after_session_refused(
    client: Client, case: str
):
    # Without a schedule-to-start timeout, nothing but the session's end
    # stops an operation waiting on the session's task queue.
    factory_calls = 0

    def factory(_: Any) -> McpToolset:
        nonlocal factory_calls
        factory_calls += 1
        return _stdio_toolset(_MCP_SERVER)

    provider = TemporalStatefulMcpToolSetProvider("stateful_server", factory)
    client = _make_client(client, provider)
    task_queue = f"adk-stateful-mcp-refused-{uuid.uuid4()}"
    host_task_queue = f"{task_queue}-host"

    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[StatefulMcpWorkerReplacedWorkflow],
        max_cached_workflows=0,
    ):
        async with Worker(client, task_queue=host_task_queue):
            handle = await client.start_workflow(
                StatefulMcpWorkerReplacedWorkflow.run,
                _ReplacedInput(
                    schedule_to_start_timeout=False,
                    session_task_queue=host_task_queue,
                    wait_cancellation=case.endswith("wait_cancellation"),
                    abandon=case.endswith("abandon"),
                    wait_for_session_end=case.startswith("operation_after_refusal"),
                ),
                id=f"stateful-mcp-refused-{uuid.uuid4()}",
                task_queue=task_queue,
                execution_timeout=timedelta(seconds=60),
            )
            assert await handle.execute_update(
                StatefulMcpWorkerReplacedWorkflow.wait_listed
            ) == ["echo"]
        # The session's host shut down gracefully after it opened a session.

        if case.startswith("operation_waiting"):
            await handle.signal(StatefulMcpWorkerReplacedWorkflow.resume)
            async with asyncio.timeout(20):
                while not any(
                    pending.activity_type.name.endswith("-list-tools")
                    for pending in (
                        await handle.describe()
                    ).raw_description.pending_activities
                ):
                    await asyncio.sleep(0.1)

        # The retried session activity is refused on the next host.
        async with Worker(client, task_queue=host_task_queue):
            if case.startswith("operation_after_refusal"):
                await handle.signal(StatefulMcpWorkerReplacedWorkflow.resume)
            async with asyncio.timeout(30):
                with pytest.raises(WorkflowFailureError) as err:
                    await handle.result()
        raised = await handle.query(StatefulMcpWorkerReplacedWorkflow.raised)

    assert factory_calls == 1
    _assert_refused(err.value)
    # Inside the block, the operation's error has the session's failure as its
    # cause in every cancellation mode, not the operation's cancellation.
    assert raised[0].type == "ApplicationError", raised
    assert raised[0].application_type == "DedicatedWorkerFailure", raised
    assert [e.type for e in raised[1:3]] == ["ActivityError", "ApplicationError"], (
        raised
    )
    assert raised[2].application_type == "DedicatedWorkerFailure", raised
    assert raised[2].non_retryable, raised
    assert "An earlier attempt" in raised[2].message, raised
    # The waiting operation was cancelled unless its cancellation type waits
    # for the cancel to be confirmed; none was scheduled after the session
    # ended.
    scheduled = [
        event.activity_task_scheduled_event_attributes.activity_type.name
        async for event in handle.fetch_history_events()
        if event.event_type == EventType.EVENT_TYPE_ACTIVITY_TASK_SCHEDULED
    ]
    assert len([n for n in scheduled if n.endswith("-list-tools")]) == (
        2 if case.startswith("operation_waiting") else 1
    )
    assert await _has_event(
        handle, EventType.EVENT_TYPE_ACTIVITY_TASK_CANCEL_REQUESTED
    ) == (case == "operation_waiting")


@pytest.mark.timeout(90)
async def test_stateful_session_heartbeats_within_heartbeat_timeout(client: Client):
    # A 5-second heartbeat timeout, well under the 30-second heartbeat
    # interval, with nothing patched: a healthy block stays open past it.
    factory_calls = 0

    def factory(_: Any) -> McpToolset:
        nonlocal factory_calls
        factory_calls += 1
        return _stdio_toolset(_MCP_SERVER)

    provider = TemporalStatefulMcpToolSetProvider("stateful_server", factory)
    client = _make_client(client, provider)
    task_queue = f"adk-stateful-mcp-heartbeat-{uuid.uuid4()}"
    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[StatefulMcpWorkerReplacedWorkflow],
        max_cached_workflows=0,
    ):
        handle = await client.start_workflow(
            StatefulMcpWorkerReplacedWorkflow.run,
            _ReplacedInput(heartbeat_timeout=5),
            id=f"stateful-mcp-heartbeat-{uuid.uuid4()}",
            task_queue=task_queue,
            execution_timeout=timedelta(seconds=60),
        )
        assert await handle.execute_update(
            StatefulMcpWorkerReplacedWorkflow.wait_listed
        ) == ["echo"]
        await asyncio.sleep(15)
        await handle.signal(StatefulMcpWorkerReplacedWorkflow.resume)
        assert await handle.result() == ["echo"]
    assert factory_calls == 1


class _DenyDescribe(Interceptor):
    def __init__(self, status: RPCStatusCode) -> None:
        super().__init__()
        self.status = status

    def intercept_client(self, next: OutboundInterceptor) -> OutboundInterceptor:
        return _DenyDescribeOutbound(next, self.status)


class _DenyDescribeOutbound(OutboundInterceptor):
    def __init__(self, next: OutboundInterceptor, status: RPCStatusCode) -> None:
        super().__init__(next)
        self.status = status

    async def describe_workflow(self, input: DescribeWorkflowInput) -> Any:
        raise RPCError("Request refused.", self.status, b"")


class _StallDescribeOutbound(OutboundInterceptor):
    async def describe_workflow(self, input: DescribeWorkflowInput) -> Any:
        await asyncio.Event().wait()


class _StallDescribe(Interceptor):
    def intercept_client(self, next: OutboundInterceptor) -> OutboundInterceptor:
        return _StallDescribeOutbound(next)


class _OnceDescribe(Interceptor):
    """Changes only the first describe: raises UNAVAILABLE, or reports another attempt."""

    def __init__(self, case: str) -> None:
        super().__init__()
        self.case = case
        self.calls = 0

    def intercept_client(self, next: OutboundInterceptor) -> OutboundInterceptor:
        return _OnceDescribeOutbound(next, self)


class _OnceDescribeOutbound(OutboundInterceptor):
    def __init__(self, next: OutboundInterceptor, owner: _OnceDescribe) -> None:
        super().__init__(next)
        self.owner = owner

    async def describe_workflow(self, input: DescribeWorkflowInput) -> Any:
        self.owner.calls += 1
        first = self.owner.calls == 1
        if first and self.owner.case == "unavailable_once":
            raise RPCError("Unavailable.", RPCStatusCode.UNAVAILABLE, b"")
        description = await super().describe_workflow(input)
        if first and self.owner.case == "other_attempt_once":
            for pending in description.raw_description.pending_activities:
                pending.attempt += 1
        return description


class _StallMayBeOpenEncodeCodec(PayloadCodec):
    """Never finishes encoding "may be open" heartbeats after the first ``passed``."""

    def __init__(self, passed: int = 0) -> None:
        self.passed = passed

    async def encode(self, payloads: Sequence[Payload]) -> list[Payload]:
        if _encoding_in("_heartbeat_async") and any(
            _mcp._SESSION_MAY_BE_OPEN.encode() in p.data for p in payloads
        ):
            if self.passed:
                self.passed -= 1
            else:
                await asyncio.Event().wait()
        return list(payloads)

    async def decode(self, payloads: Sequence[Payload]) -> list[Payload]:
        return list(payloads)


class _StallReadBackCodec(PayloadCodec):
    """Never finishes decoding the details the session activity reads back."""

    async def encode(self, payloads: Sequence[Payload]) -> list[Payload]:
        return list(payloads)

    async def decode(self, payloads: Sequence[Payload]) -> list[Payload]:
        if _encoding_in("wait_until_stored"):
            await asyncio.Event().wait()
        return list(payloads)


@pytest.mark.timeout(90)
@pytest.mark.parametrize(
    "case",
    [
        "refused",
        "not_found",
        "describe_stalls",
        "decode_stalls",
        "heartbeat_encode_stalls",
        "unavailable_once",
        "other_attempt_once",
    ],
)
async def test_stateful_session_without_describe_permission(
    client: Client, case: str, monkeypatch: pytest.MonkeyPatch
):
    # The worker's client may not describe the workflow: the session activity
    # fails at once with an error that says so, and the run fails with it. A
    # describe or decoding that never ends fails the attempt at the deadline.
    # A transient describe error is retried within the attempt, and a describe
    # that shows another attempt fails this one, which is retried.
    monkeypatch.setattr(_mcp, "_STORE_TIMEOUT", 2.0)
    monkeypatch.setattr(_mcp, "_CLOSE_TIMEOUT", 1.0)
    factory_calls = 0

    def factory(_: Any) -> McpToolset:
        nonlocal factory_calls
        factory_calls += 1
        return _stdio_toolset(_MCP_SERVER)

    provider = TemporalStatefulMcpToolSetProvider("stateful_server", factory)
    config = client.config()
    if case.endswith("code_stalls"):
        # A heartbeat that never finishes encoding would be the attempt's last
        # heartbeat task without the clear that close() sends after it; the
        # attempt must still complete.
        codec = (
            _StallReadBackCodec()
            if case == "decode_stalls"
            else _StallMayBeOpenEncodeCodec()
        )
        config["data_converter"] = dataclasses.replace(
            client.data_converter, payload_codec=codec
        )
    else:
        interceptor = (
            _DenyDescribe(RPCStatusCode.PERMISSION_DENIED)
            if case == "refused"
            else _DenyDescribe(RPCStatusCode.NOT_FOUND)
            if case == "not_found"
            else _StallDescribe()
            if case.endswith("stalls")
            else _OnceDescribe(case)
        )
        config["interceptors"] = [*config["interceptors"], interceptor]
    worker_client = _make_client(Client(**config), provider)
    task_queue = f"adk-stateful-mcp-describe-{uuid.uuid4()}"
    async with Worker(
        worker_client,
        task_queue=task_queue,
        workflows=[StatefulMcpWorkerReplacedWorkflow],
        max_cached_workflows=0,
    ):
        handle = await client.start_workflow(
            StatefulMcpWorkerReplacedWorkflow.run,
            _ReplacedInput(),
            id=f"stateful-mcp-describe-{uuid.uuid4()}",
            task_queue=task_queue,
            execution_timeout=timedelta(seconds=60),
        )
        if case.endswith("_once"):
            assert await handle.execute_update(
                StatefulMcpWorkerReplacedWorkflow.wait_listed
            ) == ["echo"]
            session = await _session_activity(handle)
            assert session
            assert session.attempt == (2 if case == "other_attempt_once" else 1)
            await handle.signal(StatefulMcpWorkerReplacedWorkflow.resume)
            assert await handle.result() == ["echo"]
            assert factory_calls == 1
            return
        if case not in ("refused", "not_found"):
            # The attempt fails at the 2-second deadline and is retried.
            async with asyncio.timeout(20):
                while True:
                    session = await _session_activity(handle)
                    if session and session.last_failure.message:
                        break
                    await asyncio.sleep(0.2)
            await handle.terminate()
            assert factory_calls == 0
            assert session.last_failure.message == (
                "The server did not show the stateful MCP session activity's "
                "first heartbeat within 2 seconds."
            )
            return
        async with asyncio.timeout(20):
            with pytest.raises(WorkflowFailureError) as err:
                await handle.result()

    assert factory_calls == 0
    _assert_refused(err.value)
    messages: list[str] = []
    cause: BaseException | None = err.value
    while cause is not None:
        messages.append(str(cause))
        cause = cause.__cause__
    if case == "refused":
        assert any("permission to describe" in m for m in messages), messages
    else:
        # Only a permission error gets the permission advice.
        assert any("could not describe this workflow run" in m for m in messages)
        assert not any("permission" in m for m in messages), messages


class _HoldFirstDescribe(Interceptor):
    """Holds the first describe until ``release`` is set."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0
        self.release = asyncio.Event()

    def intercept_client(self, next: OutboundInterceptor) -> OutboundInterceptor:
        return _HoldFirstDescribeOutbound(next, self)


class _HoldFirstDescribeOutbound(OutboundInterceptor):
    def __init__(self, next: OutboundInterceptor, owner: _HoldFirstDescribe) -> None:
        super().__init__(next)
        self.owner = owner

    async def describe_workflow(self, input: DescribeWorkflowInput) -> Any:
        self.owner.calls += 1
        if self.owner.calls == 1:
            await self.owner.release.wait()
        return await super().describe_workflow(input)


@workflow.defn
class StatefulMcpTwoBlocksWorkflow:
    @workflow.run
    async def run(self) -> list[str]:
        config = ActivityConfig(
            start_to_close_timeout=timedelta(seconds=20),
            schedule_to_start_timeout=timedelta(seconds=10),
            retry_policy=RetryPolicy(maximum_attempts=1),
        )
        async with TemporalStatefulMcpToolSet("stateful_server", config=config):
            await workflow.sleep(0.5)
        async with TemporalStatefulMcpToolSet(
            "stateful_server", config=config
        ) as toolset:
            # List once the first session has finished its read-back.
            await workflow.sleep(5)
            return [t.name for t in await toolset.get_tools()]


@pytest.mark.timeout(90)
async def test_stateful_sessions_with_one_name_in_one_worker(client: Client):
    # The first block exits while its session is still reading back its first
    # heartbeat. The second block's session registers meanwhile. The first
    # session's read-back then ends, and it must not register over the second.
    factory_calls = 0

    def factory(_: Any) -> McpToolset:
        nonlocal factory_calls
        factory_calls += 1
        return _stdio_toolset(_MCP_SERVER)

    provider = TemporalStatefulMcpToolSetProvider("stateful_server", factory)
    hold = _HoldFirstDescribe()
    config = client.config()
    config["interceptors"] = [*config["interceptors"], hold]
    worker_client = _make_client(Client(**config), provider)
    task_queue = f"adk-stateful-mcp-one-name-{uuid.uuid4()}"
    async with Worker(
        worker_client,
        task_queue=task_queue,
        workflows=[StatefulMcpTwoBlocksWorkflow],
        max_cached_workflows=0,
    ):
        handle = await client.start_workflow(
            StatefulMcpTwoBlocksWorkflow.run,
            id=f"stateful-mcp-one-name-{uuid.uuid4()}",
            task_queue=task_queue,
            execution_timeout=timedelta(seconds=60),
        )
        # Only the second session can register while the first is held.
        async with asyncio.timeout(20):
            while not provider._toolsets:
                await asyncio.sleep(0.05)
        hold.release.set()
        result = await handle.result()
    assert result == ["echo"]
    assert factory_calls == 1


@workflow.defn
class StatefulMcpBlockEndsWorkflow:
    @workflow.run
    async def run(self, case: str) -> list[str]:
        config = ActivityConfig(
            start_to_close_timeout=timedelta(seconds=20),
            retry_policy=RetryPolicy(maximum_attempts=1),
        )
        if case in ("factory_fails", "invalid_factory_result"):
            # A session that ends on its own, without a refused retry.
            async with TemporalStatefulMcpToolSet(
                "stateful_server",
                config=config,
                server_session_config=ActivityConfig(
                    start_to_close_timeout=timedelta(minutes=1),
                    retry_policy=(
                        None
                        if case == "invalid_factory_result"
                        else RetryPolicy(maximum_attempts=1)
                    ),
                ),
            ) as toolset:
                return [t.name for t in await toolset.get_tools()]
        if case == "compensation_raise_first":
            # After a caught workflow cancel, a block whose body fails before
            # its first await leaves with the body's error.
            try:
                await workflow.wait_condition(lambda: False)
            except asyncio.CancelledError:
                async with TemporalStatefulMcpToolSet("stateful_server", config=config):
                    raise ApplicationError("compensation failed", non_retryable=True)
        # A tool kept from an earlier block, called first in a later block.
        async with TemporalStatefulMcpToolSet(
            "stateful_server", config=config
        ) as toolset:
            tools = await toolset.get_tools()
        async with TemporalStatefulMcpToolSet("stateful_server", config=config):
            await tools[0].run_async(args={"text": "x"}, tool_context=_tool_context())
        return []


@pytest.mark.timeout(90)
@pytest.mark.parametrize(
    "case",
    [
        "factory_fails",
        "invalid_factory_result",
        "tool_from_earlier_block",
        "compensation_raise_first",
    ],
)
async def test_stateful_block_exit_error(client: Client, case: str):
    factory_calls = 0

    def factory(_: Any) -> McpToolset:
        nonlocal factory_calls
        factory_calls += 1
        if case == "factory_fails":
            raise RuntimeError("factory failed")
        if case == "invalid_factory_result":
            return cast(McpToolset, cast(object, None))
        return _stdio_toolset(_MCP_SERVER)

    provider = TemporalStatefulMcpToolSetProvider("stateful_server", factory)
    client = _make_client(client, provider)
    task_queue = f"adk-stateful-mcp-block-ends-{uuid.uuid4()}"
    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[StatefulMcpBlockEndsWorkflow],
        max_cached_workflows=0,
    ):
        handle = await client.start_workflow(
            StatefulMcpBlockEndsWorkflow.run,
            case,
            id=f"stateful-mcp-block-ends-{uuid.uuid4()}",
            task_queue=task_queue,
            execution_timeout=timedelta(seconds=60),
        )
        if case == "compensation_raise_first":
            async with asyncio.timeout(20):
                while not await _has_event(
                    handle, EventType.EVENT_TYPE_WORKFLOW_TASK_COMPLETED
                ):
                    await asyncio.sleep(0.1)
            await handle.cancel()
        with pytest.raises(WorkflowFailureError) as err:
            await handle.result()
    if case == "compensation_raise_first":
        assert isinstance(err.value.cause, ApplicationError)
        assert err.value.cause.message == "compensation failed"
        return
    # Whatever ended the session, the error is the one the docstrings name.
    assert isinstance(err.value.cause, ApplicationError)
    assert err.value.cause.type == "DedicatedWorkerFailure"
    messages: list[str] = []
    causes: list[BaseException] = []
    cause: BaseException | None = err.value.cause
    while cause is not None:
        messages.append(str(cause))
        causes.append(cause)
        cause = cause.__cause__
    expected = {
        "factory_fails": "factory failed",
        "invalid_factory_result": "MCP toolset factory 'stateful_server-stateful'",
    }.get(case, "List the toolset's tools again inside an open block.")
    if case == "invalid_factory_result":
        failure = next(
            c
            for c in causes
            if isinstance(c, ApplicationError)
            and c.type == "MCPToolsetFactoryTypeError"
        )
        assert failure.non_retryable
        assert factory_calls == 1
    assert any(expected in message for message in messages), messages


@workflow.defn
class StatefulMcpTimerAtExitWorkflow:
    def __init__(self) -> None:
        self._listed = False

    @workflow.run
    async def run(self, host_task_queue: str, case: str) -> str:
        async with TemporalStatefulMcpToolSet(
            "stateful_server",
            config=ActivityConfig(
                start_to_close_timeout=timedelta(seconds=20),
                retry_policy=RetryPolicy(maximum_attempts=1),
            ),
            server_session_config=ActivityConfig(
                task_queue=host_task_queue,
                start_to_close_timeout=timedelta(hours=1),
                retry_policy=RetryPolicy(maximum_attempts=1),
            ),
        ) as toolset:
            await toolset.get_tools()
            self._listed = True
            if case == "timer_first":
                await workflow.sleep(2)
            else:
                await workflow.wait_condition(lambda: False)
        return "block exited normally"

    @workflow.update
    async def wait_listed(self) -> None:
        await workflow.wait_condition(lambda: self._listed)


@pytest.mark.timeout(90)
@pytest.mark.parametrize("case", ["timer_first", "cancel_first", "cancel_healthy"])
async def test_stateful_session_ends_with_the_blocks_last_task(
    client: Client, case: str
):
    # The session ends in the block's last workflow task, after the block's
    # last timer fires or after a workflow cancel. The SDK then drops the
    # session's failure when the block cancels it. After the timer, the block
    # must still raise DedicatedWorkerFailure; a workflow cancel already
    # leaving the block must stay a cancel.
    provider = TemporalStatefulMcpToolSetProvider(
        "stateful_server", lambda _: _stdio_toolset(_MCP_SERVER)
    )
    client = _make_client(client, provider)
    task_queue = f"adk-stateful-mcp-timer-exit-{uuid.uuid4()}"
    host_task_queue = f"{task_queue}-host"
    async with Worker(client, task_queue=host_task_queue):
        async with Worker(
            client,
            task_queue=task_queue,
            workflows=[StatefulMcpTimerAtExitWorkflow],
            max_cached_workflows=0,
        ):
            handle = await client.start_workflow(
                StatefulMcpTimerAtExitWorkflow.run,
                args=[host_task_queue, case],
                id=f"stateful-mcp-timer-exit-{uuid.uuid4()}",
                task_queue=task_queue,
                execution_timeout=timedelta(seconds=60),
            )
            await handle.execute_update(StatefulMcpTimerAtExitWorkflow.wait_listed)
            if case == "cancel_healthy":
                # Control: the session is still running when the cancel leaves.
                await handle.cancel()
                with pytest.raises(WorkflowFailureError) as err:
                    await handle.result()
                assert isinstance(err.value.cause, CancelledError)
                return
        # No workflow worker: the timer fires, or the cancel is requested,
        # while nothing processes it.
        if case == "cancel_first":
            await handle.cancel()
        event = (
            EventType.EVENT_TYPE_TIMER_FIRED
            if case == "timer_first"
            else EventType.EVENT_TYPE_WORKFLOW_EXECUTION_CANCEL_REQUESTED
        )
        async with asyncio.timeout(20):
            while not await _has_event(handle, event):
                await asyncio.sleep(0.1)
    # The session's host shut down gracefully, and its one attempt failed.
    async with asyncio.timeout(20):
        while not await _has_event(handle, EventType.EVENT_TYPE_ACTIVITY_TASK_FAILED):
            await asyncio.sleep(0.1)
    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[StatefulMcpTimerAtExitWorkflow],
        max_cached_workflows=0,
    ):
        async with asyncio.timeout(20):
            with pytest.raises(WorkflowFailureError) as err:
                await handle.result()
    if case == "cancel_first":
        assert isinstance(err.value.cause, CancelledError)
        return
    assert isinstance(err.value.cause, ApplicationError)
    assert err.value.cause.type == "DedicatedWorkerFailure"
    # The SDK dropped the session's result, so the error has no cause.
    assert err.value.cause.cause is None


@pytest.mark.timeout(90)
async def test_stateful_session_worker_shutdown_while_closing(
    client: Client, monkeypatch: pytest.MonkeyPatch
):
    # The first "may be open" heartbeat reaches the server, the later ones hang
    # in encoding, and the read-back hangs, so the attempt fails at its deadline
    # and then waits in close() for those heartbeats. A worker shutdown during
    # that wait must still let the attempt end, with the clear as its last
    # heartbeat details.
    monkeypatch.setattr(_mcp, "_STORE_TIMEOUT", 2.0)
    monkeypatch.setattr(_mcp, "_CLOSE_TIMEOUT", 60.0)
    monkeypatch.setattr(_mcp, "_HEARTBEAT_INTERVAL", 0.5)
    closing = asyncio.Event()
    close = _mcp._SessionHeartbeats.close

    async def close_and_flag(self: _mcp._SessionHeartbeats) -> None:
        # close() does not await before its wait, so once this is set, the
        # attempt waits there.
        closing.set()
        await close(self)

    monkeypatch.setattr(_mcp._SessionHeartbeats, "close", close_and_flag)
    provider = TemporalStatefulMcpToolSetProvider(
        "stateful_server", lambda _: _stdio_toolset(_MCP_SERVER)
    )
    config = client.config()
    config["data_converter"] = dataclasses.replace(
        client.data_converter, payload_codec=_StallMayBeOpenEncodeCodec(passed=1)
    )
    config["interceptors"] = [*config["interceptors"], _StallDescribe()]
    worker_client = _make_client(Client(**config), provider)
    task_queue = f"adk-stateful-mcp-shutdown-closing-{uuid.uuid4()}"
    async with Worker(
        worker_client,
        task_queue=task_queue,
        workflows=[StatefulMcpWorkerReplacedWorkflow],
        max_cached_workflows=0,
    ) as worker:
        handle = await client.start_workflow(
            StatefulMcpWorkerReplacedWorkflow.run,
            _ReplacedInput(),
            id=f"stateful-mcp-shutdown-closing-{uuid.uuid4()}",
            task_queue=task_queue,
            execution_timeout=timedelta(seconds=60),
        )
        async with asyncio.timeout(20):
            await closing.wait()
        async with asyncio.timeout(20):
            await worker.shutdown()
    # The attempt's failure carried the clear over the first heartbeat, so its
    # retry may build the toolset.
    async with asyncio.timeout(20):
        while True:
            session = await _session_activity(handle)
            if session and session.attempt == 2:
                break
            await asyncio.sleep(0.2)
    assert not session.heartbeat_details.payloads
    await handle.terminate()


# Without a heartbeat timeout the worker sends a heartbeat at most every 30
# seconds, and the reset arrives in its response.
@pytest.mark.timeout(90)
async def test_stateful_session_stops_on_activity_reset(
    client: Client, monkeypatch: pytest.MonkeyPatch
):
    CREATED.clear()
    monkeypatch.setattr(_mcp, "_HEARTBEAT_INTERVAL", 1.0)
    provider = TemporalStatefulMcpToolSetProvider("stateful_set", _factory)
    client = _make_client(client, provider)
    task_queue = f"adk-stateful-mcp-reset-{uuid.uuid4()}"

    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[StatefulMcpWorkerReplacedWorkflow],
        max_cached_workflows=0,
    ):
        handle = await client.start_workflow(
            StatefulMcpWorkerReplacedWorkflow.run,
            _ReplacedInput(name="stateful_set"),
            id=f"stateful-mcp-reset-{uuid.uuid4()}",
            task_queue=task_queue,
            execution_timeout=timedelta(seconds=50),
        )
        assert await handle.execute_update(
            StatefulMcpWorkerReplacedWorkflow.wait_listed
        ) == ["echo"]
        session = await _session_activity(handle)
        assert session
        await client.workflow_service.reset_activity(
            ResetActivityRequest(
                namespace=client.namespace,
                execution=WorkflowExecution(workflow_id=handle.id),
                id=session.activity_id,
            )
        )
        # The next heartbeat reports the reset, and the original session
        # stops instead of serving the reset attempt.
        async with asyncio.timeout(45):
            while not CREATED[0].closed:
                await asyncio.sleep(0.1)
        # Let the reset attempt be refused before the next get_tools.
        async with asyncio.timeout(30):
            while not await _has_event(
                handle, EventType.EVENT_TYPE_ACTIVITY_TASK_FAILED
            ):
                await asyncio.sleep(0.2)
        await handle.signal(StatefulMcpWorkerReplacedWorkflow.resume)
        with pytest.raises(WorkflowFailureError) as err:
            await handle.result()

    assert len(CREATED) == 1
    assert CREATED[0].get_tools_calls == 1
    _assert_refused(err.value)


@pytest.mark.timeout(90)
async def test_stateful_session_after_workflow_reset(
    client: Client, monkeypatch: pytest.MonkeyPatch
):
    CREATED.clear()
    monkeypatch.setattr(_mcp, "_HEARTBEAT_INTERVAL", 1.0)
    provider = TemporalStatefulMcpToolSetProvider("stateful_set", _factory)
    client = _make_client(client, provider)
    task_queue = f"adk-stateful-mcp-workflow-reset-{uuid.uuid4()}"

    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[StatefulMcpWorkerReplacedWorkflow],
        max_cached_workflows=0,
    ):
        handle = await client.start_workflow(
            StatefulMcpWorkerReplacedWorkflow.run,
            _ReplacedInput(name="stateful_set"),
            id=f"stateful-mcp-workflow-reset-{uuid.uuid4()}",
            task_queue=task_queue,
            execution_timeout=timedelta(seconds=50),
        )
        assert await handle.execute_update(
            StatefulMcpWorkerReplacedWorkflow.wait_listed
        ) == ["echo"]
        last_task = [
            event.event_id
            async for event in handle.fetch_history_events()
            if event.event_type == EventType.EVENT_TYPE_WORKFLOW_TASK_COMPLETED
        ][-1]
        await client.workflow_service.reset_workflow_execution(
            ResetWorkflowExecutionRequest(
                namespace=client.namespace,
                workflow_execution=WorkflowExecution(
                    workflow_id=handle.id, run_id=handle.result_run_id or ""
                ),
                reason="test",
                workflow_task_finish_event_id=last_task,
                request_id=str(uuid.uuid4()),
            )
        )
        # The reset run has a new session activity, which builds its own
        # toolset, and the old run's session stops.
        async with asyncio.timeout(45):
            while len(CREATED) < 2 or not CREATED[0].closed:
                await asyncio.sleep(0.1)
        reset_handle = client.get_workflow_handle(handle.id)
        await reset_handle.signal(StatefulMcpWorkerReplacedWorkflow.resume)
        assert await reset_handle.result() == ["echo"]

    assert len(CREATED) == 2
    assert CREATED[1].get_tools_calls == 1
    assert all(t.closed for t in CREATED)


_SLOW_START = "import time; time.sleep(2.5)\n"


class _DelayWorkerMarkerCodec(PayloadCodec):
    """Delays the worker's own heartbeats that carry the session marker."""

    def __init__(self) -> None:
        self.delays = 0

    async def encode(self, payloads: Sequence[Payload]) -> list[Payload]:
        if any(_mcp._SESSION_OPENED.encode() in p.data for p in payloads):
            if _encoding_in("_heartbeat_async"):
                self.delays += 1
                await asyncio.sleep(3.5)
        return list(payloads)

    async def decode(self, payloads: Sequence[Payload]) -> list[Payload]:
        return list(payloads)


async def test_stateful_session_marker_kept_by_worker_heartbeats(
    client: Client, monkeypatch: pytest.MonkeyPatch
):
    # The worker sends the marker in its own heartbeats, after earlier ones it
    # buffered. Once stored, the marker must stay, even when encoding the
    # heartbeats that carry it is slow.
    monkeypatch.setattr(_mcp, "_HEARTBEAT_INTERVAL", 1.0)
    factory_calls = 0

    def factory(_: Any) -> McpToolset:
        nonlocal factory_calls
        factory_calls += 1
        # Start late, so that Core holds back an earlier worker heartbeat
        # (it sends at most one per 4 s here) when the session opens.
        return _stdio_toolset(_SLOW_START + _MCP_SERVER)

    provider = TemporalStatefulMcpToolSetProvider("stateful_server", factory)
    codec = _DelayWorkerMarkerCodec()
    config = client.config()
    config["data_converter"] = dataclasses.replace(
        client.data_converter, payload_codec=codec
    )
    client = _make_client(Client(**config), provider)
    task_queue = f"adk-stateful-mcp-marker-{uuid.uuid4()}"
    observed: list[bool] = []

    async def observe(handle: WorkflowHandle[Any, Any]) -> None:
        while True:
            session = await _session_activity(handle)
            if session:
                observed.append(
                    any(
                        _mcp._SESSION_OPENED.encode() in p.data
                        for p in session.heartbeat_details.payloads
                    )
                )
            await asyncio.sleep(0.05)

    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[StatefulMcpWorkerReplacedWorkflow],
        max_cached_workflows=0,
    ):
        handle = await client.start_workflow(
            StatefulMcpWorkerReplacedWorkflow.run,
            _ReplacedInput(heartbeat_timeout=5),
            id=f"stateful-mcp-marker-{uuid.uuid4()}",
            task_queue=task_queue,
            execution_timeout=timedelta(seconds=50),
        )
        observer = asyncio.create_task(observe(handle))
        try:
            assert await handle.execute_update(
                StatefulMcpWorkerReplacedWorkflow.wait_listed
            ) == ["echo"]
            # Wait for the worker's delayed marker heartbeat, then check that
            # the marker stays.
            async with asyncio.timeout(15):
                while True not in observed:
                    await asyncio.sleep(0.1)
            await asyncio.sleep(3)
        finally:
            observer.cancel()

    assert codec.delays
    assert True in observed
    # Once stored, the marker never left the heartbeat details.
    assert False not in observed[observed.index(True) :]

    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[StatefulMcpWorkerReplacedWorkflow],
        max_cached_workflows=0,
    ):
        await handle.signal(StatefulMcpWorkerReplacedWorkflow.resume)
        with pytest.raises(WorkflowFailureError) as err:
            await handle.result()

    assert factory_calls == 1
    _assert_refused(err.value)


_STATE_MCP_SERVER = """
import asyncio

try:
    from mcp.server.mcpserver import MCPServer
except ImportError:  # MCP 1.x
    from mcp.server.fastmcp import FastMCP as MCPServer

server = MCPServer("state")
value = "EMPTY"


@server.tool()
def put(text: str) -> str:
    global value
    value = text
    return "stored"


@server.tool()
def get() -> str:
    return value


@server.tool()
async def wait() -> str:
    await asyncio.sleep(3600)
    return "done"


server.run()
"""

# A worker serving only the session activity, in a process the test kills.
_SESSION_HOST = """
import asyncio
import sys

from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters

from temporalio.client import Client
from temporalio.converter import DataConverter
from temporalio.google_adk import (
    GoogleAdkPlugin,
    TemporalStatefulMcpToolSetProvider,
    _mcp,
)
from temporalio.worker import Worker

target, namespace, task_queue, calls_file, server, delay = sys.argv[1:]
_mcp._HEARTBEAT_INTERVAL = 1.0
sys.path.insert(0, sys.argv[0].rsplit("/", 1)[0])


def factory(_):
    with open(calls_file, "a") as f:
        f.write("x")
    return McpToolset(
        connection_params=StdioConnectionParams(
            server_params=StdioServerParameters(
                command=sys.executable, args=["-c", server]
            ),
            timeout=30,
        )
    )


async def main():
    codec = None
    if delay == "1":
        from codec import DelayWorkerMarkerCodec

        codec = DelayWorkerMarkerCodec()
    elif delay == "2":
        from codec import DelayHeartbeatsCodec

        codec = DelayHeartbeatsCodec()
    client = await Client.connect(
        target,
        namespace=namespace,
        data_converter=DataConverter(payload_codec=codec),
        plugins=[
            GoogleAdkPlugin(
                toolset_providers=[
                    TemporalStatefulMcpToolSetProvider("stateful_state", factory)
                ]
            )
        ],
    )
    async with Worker(client, task_queue=task_queue):
        print("ready", flush=True)
        await asyncio.Event().wait()


asyncio.run(main())
"""

_SESSION_HOST_CODEC = """
import asyncio
import sys

from temporalio.converter import PayloadCodec


class DelayWorkerMarkerCodec(PayloadCodec):
    async def encode(self, payloads):
        if any(b"mcp-session-opened" in p.data for p in payloads):
            frame = sys._getframe()
            while frame and frame.f_code.co_name != "_heartbeat_async":
                frame = frame.f_back
            if frame:
                await asyncio.sleep(3.5)
        return list(payloads)

    async def decode(self, payloads):
        return list(payloads)


class DelayHeartbeatsCodec(PayloadCodec):
    async def encode(self, payloads):
        frame = sys._getframe()
        while frame and frame.f_code.co_name != "_heartbeat_async":
            frame = frame.f_back
        if frame:
            await asyncio.sleep(3)
        return list(payloads)

    async def decode(self, payloads):
        return list(payloads)
"""


def _tool_context() -> ToolContext:
    return cast(
        ToolContext,
        cast(
            object,
            SimpleNamespace(
                tool_confirmation=None,
                function_call_id=None,
                _event_actions=EventActions(),
            ),
        ),
    )


@workflow.defn
class StatefulMcpHostKilledWorkflow:
    def __init__(self) -> None:
        self._stored = False
        self._resume = False

    @workflow.run
    async def run(
        self, host_task_queue: str, heartbeat_timeout: bool, call_waiting: bool
    ) -> str:
        session_config = ActivityConfig(
            task_queue=host_task_queue, start_to_close_timeout=timedelta(seconds=10)
        )
        if heartbeat_timeout:
            session_config["start_to_close_timeout"] = timedelta(hours=1)
            session_config["heartbeat_timeout"] = timedelta(seconds=5)
        config = ActivityConfig(
            start_to_close_timeout=timedelta(seconds=20),
            schedule_to_start_timeout=timedelta(seconds=5),
            retry_policy=RetryPolicy(maximum_attempts=1),
        )
        if call_waiting:
            # Its cancellation is never acknowledged once the host is gone.
            config["start_to_close_timeout"] = timedelta(minutes=2)
            config["cancellation_type"] = (
                ActivityCancellationType.WAIT_CANCELLATION_COMPLETED
            )
        async with TemporalStatefulMcpToolSet(
            "stateful_state",
            config=config,
            server_session_config=session_config,
        ) as toolset:
            if not heartbeat_timeout:
                # Let a periodic heartbeat start the worker's 30-second
                # throttle, so the marker heartbeat is held back.
                await workflow.sleep(2)
            tools = {t.name: t for t in await toolset.get_tools()}
            await tools["put"].run_async(
                args={"text": "kept"}, tool_context=_tool_context()
            )
            self._stored = True
            if call_waiting:
                return str(
                    await tools["wait"].run_async(args={}, tool_context=_tool_context())
                )
            await workflow.wait_condition(lambda: self._resume)
            tools = {t.name: t for t in await toolset.get_tools()}
            return str(
                await tools["get"].run_async(args={}, tool_context=_tool_context())
            )

    @workflow.update
    async def wait_stored(self) -> None:
        await workflow.wait_condition(lambda: self._stored)

    @workflow.signal
    def resume(self) -> None:
        self._resume = True


@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    "case",
    [
        "heartbeat_timeout",
        "heartbeat_timeout_delayed_marker",
        "heartbeat_timeout_slow_heartbeats",
        "heartbeat_timeout_call_waiting",
        "no_heartbeat_timeout",
    ],
)
async def test_stateful_session_after_host_killed(
    client: Client, tmp_path: Path, case: str
):
    delay_worker_heartbeats = case == "heartbeat_timeout_delayed_marker"
    # Every heartbeat takes 3 seconds to encode, so the session would open and
    # be used before its first heartbeat reaches the server if the session
    # activity did not wait for it.
    slow_heartbeats = case == "heartbeat_timeout_slow_heartbeats"
    call_waiting = case == "heartbeat_timeout_call_waiting"
    (tmp_path / "host.py").write_text(_SESSION_HOST)
    (tmp_path / "codec.py").write_text(_SESSION_HOST_CODEC)
    calls_file = tmp_path / "calls"
    calls_file.touch()
    replacement_calls = 0

    def factory(_: Any) -> McpToolset:
        nonlocal replacement_calls
        replacement_calls += 1
        return _stdio_toolset(_STATE_MCP_SERVER)

    provider = TemporalStatefulMcpToolSetProvider("stateful_state", factory)
    client = _make_client(client, provider)
    task_queue = f"adk-stateful-mcp-killed-{uuid.uuid4()}"
    host_task_queue = f"{task_queue}-host"

    host = await asyncio.create_subprocess_exec(
        sys.executable,
        str(tmp_path / "host.py"),
        client.service_client.config.target_host,
        client.namespace,
        host_task_queue,
        str(calls_file),
        (_SLOW_START if delay_worker_heartbeats else "") + _STATE_MCP_SERVER,
        "1" if delay_worker_heartbeats else "2" if slow_heartbeats else "0",
        stdout=asyncio.subprocess.PIPE,
    )
    try:
        assert host.stdout
        # The host imports ADK first, which is slow on a loaded machine.
        async with asyncio.timeout(60):
            assert (await host.stdout.readline()).strip() == b"ready"
        async with Worker(
            client,
            task_queue=task_queue,
            workflows=[StatefulMcpHostKilledWorkflow],
            max_cached_workflows=0,
        ):
            handle = await client.start_workflow(
                StatefulMcpHostKilledWorkflow.run,
                args=[host_task_queue, case != "no_heartbeat_timeout", call_waiting],
                id=f"stateful-mcp-killed-{uuid.uuid4()}",
                task_queue=task_queue,
                execution_timeout=timedelta(seconds=50),
            )
            await handle.execute_update(StatefulMcpHostKilledWorkflow.wait_stored)
            if delay_worker_heartbeats:
                # While the host's marker heartbeats are delayed, kill it
                # before the marker reaches the server.
                loop = asyncio.get_running_loop()
                deadline = loop.time() + 8
                while loop.time() < deadline:
                    session = await _session_activity(handle)
                    if session and not any(
                        _mcp._SESSION_OPENED.encode() in p.data
                        for p in session.heartbeat_details.payloads
                    ):
                        break
                    await asyncio.sleep(0.05)
            if call_waiting:
                # Kill the host while it runs the wait tool call.
                async with asyncio.timeout(20):
                    while not any(
                        pending.activity_type.name.endswith("-call-tool")
                        and pending.state
                        == PendingActivityState.PENDING_ACTIVITY_STATE_STARTED
                        for pending in (
                            await handle.describe()
                        ).raw_description.pending_activities
                    ):
                        await asyncio.sleep(0.1)
            host.kill()
            await host.wait()

            # After the heartbeat or start-to-close timeout, Temporal retries
            # the session activity on another host.
            async with Worker(client, task_queue=host_task_queue):
                # Let the retried session activity fail before the next
                # get_tools, which would otherwise time out first.
                async with asyncio.timeout(30):
                    while not await _has_event(
                        handle, EventType.EVENT_TYPE_ACTIVITY_TASK_FAILED
                    ):
                        await asyncio.sleep(0.2)
                if not call_waiting:
                    # A waiting call needs no signal: the run may already
                    # have ended.
                    await handle.signal(StatefulMcpHostKilledWorkflow.resume)
                # A waiting call fails with the session, not after its
                # start-to-close timeout.
                async with asyncio.timeout(10):
                    with pytest.raises(WorkflowFailureError) as err:
                        await handle.result()
    finally:
        if host.returncode is None:
            host.kill()
            await host.wait()

    # get never ran against a new, empty MCP session.
    assert calls_file.read_text() == "x"
    assert replacement_calls == 0
    _assert_refused(err.value)


@workflow.defn
class StatefulMcpReopenedWorkflow:
    def __init__(self) -> None:
        self._listed = False
        self._signals = 0

    @workflow.run
    async def run(
        self,
        host_task_queue: str,
        wait_cancellation: bool,
        session_activity_ids: list[str] | None = None,
    ) -> str:
        config = ActivityConfig(
            start_to_close_timeout=timedelta(seconds=20),
            retry_policy=RetryPolicy(maximum_attempts=1),
        )
        if wait_cancellation:
            config["cancellation_type"] = (
                ActivityCancellationType.WAIT_CANCELLATION_COMPLETED
            )
        session_config = ActivityConfig(
            task_queue=host_task_queue, start_to_close_timeout=timedelta(hours=1)
        )
        session_ids = iter(session_activity_ids or [])

        def session_config_for_block() -> ActivityConfig:
            # Caller-chosen session activity IDs do not decide which block an
            # operation belongs to.
            activity_id = next(session_ids, None)
            if activity_id is None:
                return session_config
            return ActivityConfig(
                task_queue=host_task_queue,
                start_to_close_timeout=timedelta(hours=1),
                activity_id=activity_id,
            )

        try:
            async with TemporalStatefulMcpToolSet(
                "stateful_state",
                config=config,
                server_session_config=session_config_for_block(),
            ) as toolset:
                tools = {t.name: t for t in await toolset.get_tools()}
                self._listed = True
                await workflow.wait_condition(lambda: self._signals >= 1)
                await tools["put"].run_async(
                    args={"text": "OLD-PAYLOAD"}, tool_context=_tool_context()
                )
        except ApplicationError as err:
            # The caller opens a new block, as the README suggests.
            if err.type != "DedicatedWorkerFailure":
                raise
        async with TemporalStatefulMcpToolSet(
            "stateful_state",
            config=config,
            server_session_config=session_config_for_block(),
        ) as toolset:
            tools = {t.name: t for t in await toolset.get_tools()}
            await workflow.wait_condition(lambda: self._signals >= 2)
            return str(
                await tools["get"].run_async(args={}, tool_context=_tool_context())
            )

    @workflow.update
    async def wait_listed(self) -> None:
        await workflow.wait_condition(lambda: self._listed)

    @workflow.signal
    def resume(self) -> None:
        self._signals += 1


@pytest.mark.timeout(90)
@pytest.mark.parametrize(
    "cancellation,session_activity_ids",
    [
        ("wait_cancellation_completed", None),
        ("try_cancel", None),
        ("wait_cancellation_completed", ["session-a", "session-b"]),
        ("wait_cancellation_completed", ["1000", "1001"]),
    ],
    ids=[
        "wait_cancellation_completed",
        "try_cancel",
        "named_session_ids",
        "numeric_session_ids",
    ],
)
async def test_stateful_operation_of_ended_session_not_run_by_next_block(
    client: Client, cancellation: str, session_activity_ids: list[str] | None
):
    factory_calls = 0

    def factory(_: Any) -> McpToolset:
        nonlocal factory_calls
        factory_calls += 1
        return _stdio_toolset(_STATE_MCP_SERVER)

    provider = TemporalStatefulMcpToolSetProvider("stateful_state", factory)
    client = _make_client(client, provider)
    task_queue = f"adk-stateful-mcp-reopened-{uuid.uuid4()}"
    host_task_queue = f"{task_queue}-host"

    async def pending_call_tool(handle: WorkflowHandle[Any, Any]) -> bool:
        return any(
            pending.activity_type.name.endswith("-call-tool")
            for pending in (await handle.describe()).raw_description.pending_activities
        )

    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[StatefulMcpReopenedWorkflow],
        max_cached_workflows=0,
    ):
        async with Worker(client, task_queue=host_task_queue):
            handle = await client.start_workflow(
                StatefulMcpReopenedWorkflow.run,
                args=[
                    host_task_queue,
                    cancellation == "wait_cancellation_completed",
                    session_activity_ids,
                ],
                id=f"stateful-mcp-reopened-{uuid.uuid4()}",
                task_queue=task_queue,
                execution_timeout=timedelta(seconds=60),
            )
            await handle.execute_update(StatefulMcpReopenedWorkflow.wait_listed)
        # The session's host shut down gracefully after it opened a session, so
        # the put waits on the session's task queue.
        await handle.signal(StatefulMcpReopenedWorkflow.resume)
        async with asyncio.timeout(20):
            while not await pending_call_tool(handle):
                await asyncio.sleep(0.1)

        # The retried session is refused, and the workflow opens a second block
        # with the same name. With WAIT_CANCELLATION_COMPLETED the put stays on
        # the run's task queue, and the second block's session refuses it.
        async with Worker(client, task_queue=host_task_queue):
            async with asyncio.timeout(30):
                while not await _has_event(
                    handle, EventType.EVENT_TYPE_ACTIVITY_TASK_FAILED
                ):
                    await asyncio.sleep(0.2)
            await handle.signal(StatefulMcpReopenedWorkflow.resume)
            async with asyncio.timeout(30):
                result = await handle.result()

    # The first block's put never ran against the second block's MCP session.
    assert "'text': 'EMPTY'" in result
    assert factory_calls == 2
    call_tools: set[int] = set()
    operation_queues: set[str] = set()
    completed = 0
    async for event in handle.fetch_history_events():
        if event.event_type == EventType.EVENT_TYPE_ACTIVITY_TASK_SCHEDULED:
            scheduled = event.activity_task_scheduled_event_attributes
            if scheduled.activity_type.name.endswith("-call-tool"):
                call_tools.add(event.event_id)
            if not scheduled.activity_type.name.endswith("-server-session"):
                operation_queues.add(scheduled.task_queue.name)
        elif event.event_type == EventType.EVENT_TYPE_ACTIVITY_TASK_COMPLETED:
            completed += (
                event.activity_task_completed_event_attributes.scheduled_event_id
                in call_tools
            )
    assert len(call_tools) == 2
    assert completed == 1
    # Both blocks' operations use the queue that sessions of every version
    # poll, so a session recorded by an earlier version keeps serving them.
    assert operation_queues == {f"stateful_state-stateful@{handle.result_run_id}"}


@pytest.mark.parametrize(
    "file_name",
    [
        "stateful_mcp_gather.json",
        "stateful_mcp_cancel.json",
        "stateful_mcp_cancel_wait.json",
    ],
)
async def test_stateful_mcp_histories_from_main_replay(file_name: str) -> None:
    # Recorded on main at 88745dc with the workflow worker stopped, so that one
    # activation resolves a stateful tool call together with another activity
    # (gather) or with a workflow cancel (cancel; cancel_wait with
    # WAIT_CANCELLATION_COMPLETED).
    history_json = (Path(__file__).with_name("histories") / file_name).read_text()
    await Replayer(
        workflows=[
            StatefulMcpGatherWorkflow,
            StatefulMcpCancelWorkflow,
            StatefulMcpCancelWaitWorkflow,
        ],
        plugins=[GoogleAdkPlugin()],
    ).replay_workflow(WorkflowHistory.from_json("replay", history_json))
