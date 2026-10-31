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
through a Temporal workflow, mostly against an in-memory fake ``McpToolset``
(no subprocess, no real MCP server) so they run in CI. They prove the properties
the reviewer asked for: one toolset is built per workflow run, ``factory_argument``
is consumed exactly once per run, distinct runs never share a toolset, and the
toolset is torn down when the workflow completes. The worker replacement and
host loss tests run a real MCP server over stdio.
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
from temporalio.api.enums.v1 import EventType
from temporalio.api.workflow.v1 import PendingActivityInfo
from temporalio.api.workflowservice.v1 import (
    ResetActivityRequest,
    ResetWorkflowExecutionRequest,
)
from temporalio.client import Client, WorkflowFailureError, WorkflowHandle
from temporalio.common import RetryPolicy
from temporalio.converter import PayloadCodec
from temporalio.exceptions import ActivityError, ApplicationError
from temporalio.worker import Worker
from temporalio.workflow import ActivityConfig

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
from mcp.server.fastmcp import FastMCP

server = FastMCP("echo")


@server.tool()
def echo(text: str) -> str:
    return text


server.run()
"""


def _stdio_toolset(server: str) -> McpToolset:
    return McpToolset(
        connection_params=StdioConnectionParams(
            server_params=StdioServerParameters(
                command=sys.executable, args=["-c", server]
            ),
            timeout=30,
        )
    )


@dataclass
class _ReplacedInput:
    name: str = "stateful_server"
    schedule_to_start: bool = True
    heartbeat_timeout: bool = False


@workflow.defn
class StatefulMcpWorkerReplacedWorkflow:
    def __init__(self) -> None:
        self._listed: list[str] | None = None
        self._resume = False

    @workflow.run
    async def run(self, input: _ReplacedInput) -> list[str]:
        config = ActivityConfig(
            start_to_close_timeout=timedelta(seconds=20),
            retry_policy=RetryPolicy(maximum_attempts=1),
        )
        if input.schedule_to_start:
            config["schedule_to_start_timeout"] = timedelta(seconds=5)
        server_session_config = ActivityConfig(
            start_to_close_timeout=timedelta(hours=1)
        )
        if input.heartbeat_timeout:
            server_session_config["heartbeat_timeout"] = timedelta(seconds=5)
        async with TemporalStatefulMcpToolSet(
            input.name, config=config, server_session_config=server_session_config
        ) as toolset:
            try:
                self._listed = [t.name for t in await toolset.get_tools()]
            except ActivityError:
                self._listed = []
            await workflow.wait_condition(lambda: self._resume)
            return [t.name for t in await toolset.get_tools()]

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
    cause: BaseException | None = err.cause
    while isinstance(cause, ActivityError):
        cause = cause.cause
    # Refused by the retried session activity, not only by the next
    # operation timing out.
    assert isinstance(cause, ApplicationError)
    assert cause.type == "DedicatedWorkerFailure"
    assert cause.non_retryable


@pytest.mark.parametrize(
    "case",
    [
        "session_opened",
        "session_opened_no_schedule_to_start",
        "first_connection_fails",
        "heartbeat_timeout_first_connection_fails",
    ],
)
async def test_stateful_session_after_worker_replaced(
    client: Client, case: str, monkeypatch: pytest.MonkeyPatch
):
    first_connection_fails = case.endswith("first_connection_fails")
    heartbeat_timeout = case.startswith("heartbeat_timeout")
    if heartbeat_timeout:
        monkeypatch.setattr(_mcp, "_HEARTBEAT_INTERVAL", 1.0)
    factory_calls = 0

    def factory(_: Any) -> McpToolset:
        nonlocal factory_calls
        factory_calls += 1
        server = (
            "pass" if first_connection_fails and factory_calls == 1 else _MCP_SERVER
        )
        return _stdio_toolset(server)

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
                schedule_to_start=case != "session_opened_no_schedule_to_start",
                heartbeat_timeout=heartbeat_timeout,
            ),
            id=f"stateful-mcp-replaced-{uuid.uuid4()}",
            task_queue=task_queue,
            execution_timeout=timedelta(seconds=50),
        )
        listed = await handle.execute_update(
            StatefulMcpWorkerReplacedWorkflow.wait_listed
        )
        assert listed == ([] if first_connection_fails else ["echo"])
        if heartbeat_timeout:
            async with asyncio.timeout(20):
                while True:
                    session = await _session_activity(handle)
                    if session and session.heartbeat_details.payloads:
                        break
                    await asyncio.sleep(0.1)

    # The first worker shut down gracefully, as in a deployment, and Temporal
    # retries the session activity on the next one.
    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[StatefulMcpWorkerReplacedWorkflow],
        max_cached_workflows=0,
    ):
        if case == "session_opened":
            # Let the retried session activity fail before the next get_tools.
            async with asyncio.timeout(30):
                while not await _has_event(
                    handle, EventType.EVENT_TYPE_ACTIVITY_TASK_FAILED
                ):
                    await asyncio.sleep(0.2)
        await handle.signal(StatefulMcpWorkerReplacedWorkflow.resume)
        if case == "first_connection_fails":
            # No session was opened, so the retry builds the toolset again.
            assert await handle.result() == ["echo"]
            assert factory_calls == 2
            return
        with pytest.raises(WorkflowFailureError) as err:
            await handle.result()

    # A new toolset would open a new MCP session without the old one's state.
    # With a heartbeat timeout, every heartbeat carries the marker, so the
    # retry is refused even though no session was opened.
    assert factory_calls == 1
    _assert_refused(err.value)
    # get_tools failed when the session ended, without a timeout, also when no
    # schedule-to-start timeout was set.
    assert not await _has_event(handle, EventType.EVENT_TYPE_ACTIVITY_TASK_TIMED_OUT)


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
        async with asyncio.timeout(20):
            while not CREATED[0].closed:
                await asyncio.sleep(0.1)
        await handle.signal(StatefulMcpWorkerReplacedWorkflow.resume)
        with pytest.raises(WorkflowFailureError) as err:
            await handle.result()

    assert len(CREATED) == 1
    assert CREATED[0].get_tools_calls == 1
    _assert_refused(err.value)


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
        async with asyncio.timeout(20):
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

    async def encode(self, payloads: Sequence[Payload]) -> list[Payload]:
        if any(_mcp._SESSION_OPENED.encode() in p.data for p in payloads):
            frame: FrameType | None = sys._getframe()
            while frame and frame.f_code.co_name != "_heartbeat_async":
                frame = frame.f_back
            if frame:
                await asyncio.sleep(3.5)
        return list(payloads)

    async def decode(self, payloads: Sequence[Payload]) -> list[Payload]:
        return list(payloads)


async def test_stateful_session_marker_kept_by_worker_heartbeats(
    client: Client, monkeypatch: pytest.MonkeyPatch
):
    # With a heartbeat timeout the worker heartbeats too, and Core buffers
    # them. A worker heartbeat recorded before the marker must not erase it
    # once stored, even when the worker's own marker heartbeat is slow.
    monkeypatch.setattr(_mcp, "_HEARTBEAT_INTERVAL", 1.0)
    factory_calls = 0

    def factory(_: Any) -> McpToolset:
        nonlocal factory_calls
        factory_calls += 1
        # Start late, so that Core holds back an earlier worker heartbeat
        # (it sends at most one per 4 s here) when the session opens.
        return _stdio_toolset(_SLOW_START + _MCP_SERVER)

    provider = TemporalStatefulMcpToolSetProvider("stateful_server", factory)
    config = client.config()
    config["data_converter"] = dataclasses.replace(
        client.data_converter, payload_codec=_DelayWorkerMarkerCodec()
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
            _ReplacedInput(heartbeat_timeout=True),
            id=f"stateful-mcp-marker-{uuid.uuid4()}",
            task_queue=task_queue,
            execution_timeout=timedelta(seconds=50),
        )
        observer = asyncio.create_task(observe(handle))
        try:
            assert await handle.execute_update(
                StatefulMcpWorkerReplacedWorkflow.wait_listed
            ) == ["echo"]
            # Watch the worker's delayed heartbeats arrive.
            await asyncio.sleep(6)
        finally:
            observer.cancel()

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
from mcp.server.fastmcp import FastMCP

server = FastMCP("state")
value = "EMPTY"


@server.tool()
def put(text: str) -> str:
    global value
    value = text
    return "stored"


@server.tool()
def get() -> str:
    return value


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
    async def run(self, host_task_queue: str) -> str:
        async with TemporalStatefulMcpToolSet(
            "stateful_state",
            config=ActivityConfig(
                start_to_close_timeout=timedelta(seconds=20),
                retry_policy=RetryPolicy(maximum_attempts=1),
            ),
            server_session_config=ActivityConfig(
                task_queue=host_task_queue,
                start_to_close_timeout=timedelta(hours=1),
                heartbeat_timeout=timedelta(seconds=5),
            ),
        ) as toolset:
            tools = {t.name: t for t in await toolset.get_tools()}
            await tools["put"].run_async(
                args={"text": "kept"}, tool_context=_tool_context()
            )
            self._stored = True
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


@pytest.mark.parametrize("delay_worker_heartbeats", [False, True])
async def test_stateful_session_after_host_killed(
    client: Client, tmp_path: Path, delay_worker_heartbeats: bool
):
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
        "1" if delay_worker_heartbeats else "0",
        stdout=asyncio.subprocess.PIPE,
    )
    try:
        assert host.stdout
        async with asyncio.timeout(30):
            assert await host.stdout.readline() == b"ready\n"
        async with Worker(
            client,
            task_queue=task_queue,
            workflows=[StatefulMcpHostKilledWorkflow],
            max_cached_workflows=0,
        ):
            handle = await client.start_workflow(
                StatefulMcpHostKilledWorkflow.run,
                host_task_queue,
                id=f"stateful-mcp-killed-{uuid.uuid4()}",
                task_queue=task_queue,
                execution_timeout=timedelta(seconds=50),
            )
            await handle.execute_update(StatefulMcpHostKilledWorkflow.wait_stored)
            if delay_worker_heartbeats:
                # While the host's own marker heartbeats are delayed, kill it
                # as soon as the marker leaves the heartbeat details.
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
            host.kill()
            await host.wait()

            # After the heartbeat timeout, Temporal retries the session
            # activity on another host.
            async with Worker(client, task_queue=host_task_queue):
                await handle.signal(StatefulMcpHostKilledWorkflow.resume)
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
