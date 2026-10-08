"""End-to-end tests against the real ``codex app-server`` binary and a fake Responses API.

No credentials needed: Codex talks to a local scripted server through a custom model provider.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import timedelta
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from pydantic import BaseModel, ValidationError

from temporalio import activity
from temporalio.api.enums.v1 import EventType
from temporalio.client import Client, WorkflowFailureError
from temporalio.exceptions import ActivityError, ApplicationError
from temporalio.openai_codex import (
    CodexActivities,
    CodexObserver,
    CodexPlugin,
    CodexTokenUsage,
)
from temporalio.openai_codex._app_server import AppServer
from temporalio.openai_codex.testing import FakeResponsesServer
from temporalio.openai_codex.workflow import (
    CodexSession,
    _check_sandbox,
    codex_tool,
    function_schema,
)
from temporalio.worker import Worker
from tests._workflows import (
    ChatWorkflow,
    CodexWorkflow,
    NativeCodexWorkflow,
    NativeInput,
    ObservedCodexWorkflow,
    lookup,
    record,
)
from tests.helpers import new_worker

requires_codex_binary = pytest.mark.skipif(
    find_spec("codex_cli_bin") is None,
    reason="needs the Codex binary (the `bundled-codex` extra)",
)

PROMPT = 'Please look up the ticket.\nCALL:lookup|{"q":"42"}'


class RecordingObserver:
    """A CodexObserver that keeps what it was told."""

    def __init__(self, events: list[tuple[str, Any]], context: dict[str, Any]) -> None:
        self.events = events
        self.context = context

    def model_interaction_started(self, model: str | None) -> None:  # pyright: ignore[reportUnusedParameter]
        self.events.append(("started", self.context))

    def reply_delta(self, text: str) -> None:
        self.events.append(("delta", text))

    def item_started(self, item: dict[str, Any]) -> None:
        self.events.append(("item_started", item["type"]))

    def item_completed(self, item: dict[str, Any]) -> None:
        self.events.append(("item_completed", (item["type"], item.get("status"))))

    def model_interaction_ended(
        self,
        model: str | None,  # pyright: ignore[reportUnusedParameter]
        usage: CodexTokenUsage | None,
    ) -> None:
        self.events.append(("ended", usage))


@pytest_asyncio.fixture  # type: ignore[reportUntypedFunctionDecorator]
async def fake() -> AsyncIterator[FakeResponsesServer]:
    server = FakeResponsesServer()
    await server.start()
    yield server
    await server.stop()


@pytest.fixture(autouse=True)
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A ledger for Activity side effects, and no way to reach the real API."""
    ledger = tmp_path / "ledger.txt"
    ledger.touch()
    monkeypatch.setenv("CODEX_TEST_LEDGER", str(ledger))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("CODEX_API_KEY", raising=False)
    return ledger


@asynccontextmanager
async def codex_worker(
    client: Client,
    fake: FakeResponsesServer,
    tmp_path: Path,
    events: list[tuple[str, Any]] | None = None,
) -> AsyncIterator[str]:
    @asynccontextmanager
    async def observer_factory(context: dict[str, Any]) -> AsyncIterator[CodexObserver]:
        yield RecordingObserver(events if events is not None else [], context)

    plugin = CodexPlugin(
        config_overrides=fake.config_overrides,
        home_root=str(tmp_path),
        observer_factory=observer_factory,
    )
    worker = new_worker(
        client,
        CodexWorkflow,
        ObservedCodexWorkflow,
        NativeCodexWorkflow,
        ChatWorkflow,
        activities=[record],
        plugins=[plugin],
    )
    async with worker:
        yield worker.task_queue


def offered_tools(fake: FakeResponsesServer) -> set[str]:
    return {
        tool.get("name") or tool.get("type")
        for body in fake.requests
        for tool in body.get("tools", [])
    }


def test_codex_tool_derives_the_spec_from_the_function() -> None:
    tool = codex_tool(lookup)
    assert tool.spec.name == "lookup"
    assert tool.spec.description == "Look up a ticket by id."
    assert tool.spec.input_schema["type"] == "object"
    assert tool.spec.input_schema["properties"]["q"]["type"] == "string"
    assert tool.spec.input_schema["required"] == ["q"]


class Address(BaseModel):
    """A postal address."""

    city: str


def ship(order_id: str, to: Address, express: bool = False) -> str:
    """Ship an order."""
    return f"{order_id}:{to.city}:{express}"


def test_function_schema_parses_arguments_into_the_functions_own_types() -> None:
    schema = function_schema(ship)
    assert schema.spec.name == "ship"
    assert schema.spec.input_schema["required"] == ["order_id", "to"]
    kwargs = schema.parse({"order_id": "o1", "to": {"city": "Paris"}})
    assert list(kwargs) == ["order_id", "to", "express"]
    assert isinstance(kwargs["to"], Address)
    assert ship(**kwargs) == "o1:Paris:False"
    with pytest.raises(ValidationError):
        schema.parse({"order_id": "o1"})


@requires_codex_binary
async def test_host_tool_result_reaches_the_model(
    client: Client, fake: FakeResponsesServer, tmp_path: Path
) -> None:
    async with codex_worker(client, fake, tmp_path) as task_queue:
        handle = await client.start_workflow(
            CodexWorkflow.run,
            PROMPT,
            id=f"codex-{uuid.uuid4()}",
            task_queue=task_queue,
        )
        assert "ticket-42: resolved" in await handle.result()
        state = await handle.query(CodexWorkflow.state)
        assert state["thread_id"]
        assert state["rollout_lines"] > 0
        assert list(state["tool_results"].values()) == ["ticket-42: resolved"]

    # Codex's built-in tools are off in EVERY request, including segments that resume the thread
    # (the thread config does not persist across a resume).
    assert len(fake.requests) >= 2
    offered = offered_tools(fake)
    assert {"lookup", "record", "explode"} <= offered
    assert not offered & {
        "exec_command",
        "write_stdin",
        "apply_patch",
        "view_image",
        "create_goal",
    }


@requires_codex_binary
async def test_activity_tool_runs_once_with_a_stable_activity_id(
    client: Client, fake: FakeResponsesServer, tmp_path: Path, isolated_env: Path
) -> None:
    async with codex_worker(client, fake, tmp_path) as task_queue:
        handle = await client.start_workflow(
            CodexWorkflow.run,
            'Record it.\nCALL:record|{"note":"hello"}',
            id=f"codex-{uuid.uuid4()}",
            task_queue=task_queue,
        )
        assert "recorded:hello" in await handle.result()
        history = await handle.fetch_history()

    (entry,) = isolated_env.read_text().split()
    note, activity_id = entry.split("|")
    assert note == "hello"
    assert activity_id.startswith("tool-call_")
    scheduled = [
        e.activity_task_scheduled_event_attributes.activity_id
        for e in history.events
        if e.event_type == EventType.EVENT_TYPE_ACTIVITY_TASK_SCHEDULED
    ]
    assert activity_id in scheduled


@requires_codex_binary
async def test_a_failing_tool_is_a_result_the_model_sees(
    client: Client, fake: FakeResponsesServer, tmp_path: Path
) -> None:
    async with codex_worker(client, fake, tmp_path) as task_queue:
        handle = await client.start_workflow(
            CodexWorkflow.run,
            'Try it.\nCALL:explode|{"reason":"boom"}',
            id=f"codex-{uuid.uuid4()}",
            task_queue=task_queue,
        )
        result = await handle.result()
    assert "failed" in result
    assert "boom" in result


@requires_codex_binary
async def test_observer_sees_the_turn(
    client: Client, fake: FakeResponsesServer, tmp_path: Path
) -> None:
    events: list[tuple[str, Any]] = []
    async with codex_worker(client, fake, tmp_path, events) as task_queue:
        handle = await client.start_workflow(
            ObservedCodexWorkflow.run,
            PROMPT,
            id=f"codex-{uuid.uuid4()}",
            task_queue=task_queue,
        )
        await handle.result()

    kinds = [kind for kind, _ in events]
    # One segment before the tool call and one after it: two spans.
    assert kinds.count("started") == kinds.count("ended") == 2
    assert all(
        payload == {"turn": "t1"} for kind, payload in events if kind == "started"
    )
    assert "ticket-42: resolved" in "".join(p for kind, p in events if kind == "delta")
    final_usage = [p for kind, p in events if kind == "ended"][-1]
    assert final_usage is not None and final_usage.total_tokens


@requires_codex_binary
async def test_crash_after_a_tool_ran_does_not_run_it_again(
    client: Client,
    fake: FakeResponsesServer,
    tmp_path: Path,
    isolated_env: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_call = AppServer.call

    async def crashing_call(self: AppServer, method: str, params: Any = None) -> Any:
        result = await original_call(self, method, params)
        # The segment that resumes after the tool: SIGKILL the app-server right after the turn
        # starts, on the first attempt only, as a dying Worker would.
        if (
            method == "turn/start"
            and params["input"][0]["text"] == "Continue."
            and activity.info().attempt == 1
        ):
            self.kill()
        return result

    monkeypatch.setattr(AppServer, "call", crashing_call)

    async with codex_worker(client, fake, tmp_path) as task_queue:
        handle = await client.start_workflow(
            CodexWorkflow.run,
            'Record it.\nCALL:record|{"note":"once"}',
            id=f"codex-{uuid.uuid4()}",
            task_queue=task_queue,
        )
        assert "recorded:once" in await handle.result()
        history = await handle.fetch_history()

    # The side effect happened exactly once, though the segment after it ran twice.
    assert [line.split("|")[0] for line in isolated_env.read_text().split()] == ["once"]
    scheduled: dict[int, str] = {}
    retried: list[str] = []
    for event in history.events:
        if event.event_type == EventType.EVENT_TYPE_ACTIVITY_TASK_SCHEDULED:
            attrs = event.activity_task_scheduled_event_attributes
            scheduled[event.event_id] = attrs.activity_type.name
        if event.event_type == EventType.EVENT_TYPE_ACTIVITY_TASK_STARTED:
            started = event.activity_task_started_event_attributes
            if started.attempt > 1:
                retried.append(scheduled[started.scheduled_event_id])
    assert retried == ["openai_codex.run_segment"]

    # The model saw the real tool output exactly once, never Codex's synthetic "aborted".
    outputs = [
        item["output"]
        for item in fake.requests[-1]["input"]
        if item.get("type") == "function_call_output"
    ]
    assert outputs == ["recorded:once"]


@requires_codex_binary
async def test_a_failed_model_call_fails_the_turn_instead_of_replying_empty(
    client: Client, fake: FakeResponsesServer, tmp_path: Path
) -> None:
    fake.failure_status = 400  # not retried by Codex: the turn ends as failed
    async with codex_worker(client, fake, tmp_path) as task_queue:
        handle = await client.start_workflow(
            CodexWorkflow.run,
            PROMPT,
            id=f"codex-{uuid.uuid4()}",
            task_queue=task_queue,
            execution_timeout=timedelta(seconds=60),
        )
        with pytest.raises(WorkflowFailureError):
            await handle.result()


# ---------------------------------------------------------------------------
# Native mode: Codex runs its own tools; the Workflow decides every approval.
# ---------------------------------------------------------------------------

ECHO = 'Write it.\nCALL:exec_command|{"cmd":"echo hello > out.txt && cat out.txt"}'
PATCH = "*** Begin Patch\n*** Add File: hello.txt\n+hi from patch\n*** End Patch\n"
PATCH_PROMPT = "Patch it.\nCALL:apply_patch|" + json.dumps(PATCH)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    path = tmp_path / "workspace"
    path.mkdir()
    return path


async def run_native(
    client: Client,
    fake: FakeResponsesServer,
    tmp_path: Path,
    workspace: Path,
    prompt: str,
    mode: str = "approve",
    model: str | None = None,
    events: list[tuple[str, Any]] | None = None,
    approval_policy: str = "untrusted",
) -> tuple[str, list[Any]]:
    async with codex_worker(client, fake, tmp_path, events) as task_queue:
        handle = await client.start_workflow(
            NativeCodexWorkflow.run,
            NativeInput(prompt, str(workspace), mode, model, approval_policy),
            id=f"codex-{uuid.uuid4()}",
            task_queue=task_queue,
        )
        result = await handle.result()
        return result, await handle.query(NativeCodexWorkflow.asked_requests)


def test_native_mode_needs_a_workspace() -> None:
    with pytest.raises(ValueError, match="cwd"):
        CodexSession()


@requires_codex_binary
async def test_an_approved_command_runs_natively(
    client: Client, fake: FakeResponsesServer, tmp_path: Path, workspace: Path
) -> None:
    result, asked = await run_native(client, fake, tmp_path, workspace, ECHO)

    assert (workspace / "out.txt").read_text() == "hello\n"
    assert "hello" in result
    (request,) = asked
    assert request.kind == "command"
    assert request.command == "echo hello > out.txt && cat out.txt"
    assert Path(request.cwd or "").resolve() == workspace.resolve()
    # Native mode keeps Codex's own tools.
    assert "exec_command" in offered_tools(fake)


@requires_codex_binary
async def test_a_relative_workspace_is_resolved_on_the_worker(
    client: Client,
    fake: FakeResponsesServer,
    tmp_path: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    async with codex_worker(client, fake, tmp_path) as task_queue:
        handle = await client.start_workflow(
            NativeCodexWorkflow.run,
            NativeInput(ECHO, workspace.name),  # relative to the Worker's directory
            id=f"codex-{uuid.uuid4()}",
            task_queue=task_queue,
        )
        assert "hello" in await handle.result()
    assert (workspace / "out.txt").read_text() == "hello\n"


@requires_codex_binary
async def test_a_missing_workspace_fails_the_workflow_clearly(
    client: Client, fake: FakeResponsesServer, tmp_path: Path
) -> None:
    async with codex_worker(client, fake, tmp_path) as task_queue:
        handle = await client.start_workflow(
            NativeCodexWorkflow.run,
            NativeInput(ECHO, str(tmp_path / "no-such-folder")),
            id=f"codex-{uuid.uuid4()}",
            task_queue=task_queue,
            execution_timeout=timedelta(seconds=60),
        )
        with pytest.raises(WorkflowFailureError) as excinfo:
            await handle.result()
    activity_error = excinfo.value.cause
    assert isinstance(activity_error, ActivityError)
    assert isinstance(activity_error.cause, ApplicationError)
    assert "does not exist" in activity_error.cause.message


@requires_codex_binary
async def test_a_declined_command_does_not_run(
    client: Client, fake: FakeResponsesServer, tmp_path: Path, workspace: Path
) -> None:
    result, asked = await run_native(
        client, fake, tmp_path, workspace, ECHO, mode="decline"
    )

    assert not (workspace / "out.txt").exists()
    assert "rejected" in result
    assert len(asked) == 1


@requires_codex_binary
async def test_without_an_approval_handler_everything_is_declined(
    client: Client, fake: FakeResponsesServer, tmp_path: Path, workspace: Path
) -> None:
    result, _ = await run_native(client, fake, tmp_path, workspace, ECHO, mode="none")

    assert not (workspace / "out.txt").exists()
    assert "rejected" in result


@requires_codex_binary
async def test_untrusted_policy_asks_even_for_a_listing(
    client: Client, fake: FakeResponsesServer, tmp_path: Path, workspace: Path
) -> None:
    _, asked = await run_native(
        client, fake, tmp_path, workspace, 'List.\nCALL:exec_command|{"cmd":"ls"}'
    )

    assert [r.command for r in asked] == ["ls"]


@requires_codex_binary
async def test_on_request_policy_lets_sandboxed_commands_run_without_asking(
    client: Client, fake: FakeResponsesServer, tmp_path: Path, workspace: Path
) -> None:
    result, asked = await run_native(
        client,
        fake,
        tmp_path,
        workspace,
        ECHO,
        approval_policy="on-request",
    )

    # Inside the workspace sandbox, Codex does not need permission: the handler is never asked.
    assert asked == []
    assert (workspace / "out.txt").read_text() == "hello\n"
    assert "hello" in result


@requires_codex_binary
async def test_the_workflow_can_wait_on_a_human_for_as_long_as_it_takes(
    client: Client, fake: FakeResponsesServer, tmp_path: Path, workspace: Path
) -> None:
    async with codex_worker(client, fake, tmp_path) as task_queue:
        handle = await client.start_workflow(
            NativeCodexWorkflow.run,
            NativeInput(ECHO, str(workspace), "signal"),
            id=f"codex-{uuid.uuid4()}",
            task_queue=task_queue,
        )
        # The approval is pending in the Workflow; the command has not run.
        pending = []
        for _ in range(100):
            pending = await handle.query(NativeCodexWorkflow.pending)
            if pending:
                break
            await asyncio.sleep(0.2)
        assert pending, "the approval request never reached the Workflow"
        await asyncio.sleep(1.5)  # longer than a heartbeat: the segment keeps waiting
        assert not (workspace / "out.txt").exists()
        status = (await handle.describe()).status
        assert status is not None and status.name == "RUNNING"

        await handle.signal(NativeCodexWorkflow.decide, args=[pending[0].item_id, True])
        assert "hello" in await handle.result()
    assert (workspace / "out.txt").read_text() == "hello\n"


@requires_codex_binary
async def test_an_approved_patch_is_applied(
    client: Client, fake: FakeResponsesServer, tmp_path: Path, workspace: Path
) -> None:
    result, asked = await run_native(
        client, fake, tmp_path, workspace, PATCH_PROMPT, model="gpt-5.5"
    )

    assert (workspace / "hello.txt").read_text() == "hi from patch\n"
    (request,) = asked
    assert request.kind == "file_change"
    assert [c["path"].endswith("hello.txt") for c in request.changes] == [True]
    assert request.changes[0]["diff"] == "hi from patch\n"
    assert "Success" in result


@requires_codex_binary
async def test_a_declined_patch_is_not_applied(
    client: Client, fake: FakeResponsesServer, tmp_path: Path, workspace: Path
) -> None:
    result, _ = await run_native(
        client, fake, tmp_path, workspace, PATCH_PROMPT, mode="decline", model="gpt-5.5"
    )

    assert not (workspace / "hello.txt").exists()
    assert "rejected" in result


@requires_codex_binary
async def test_the_observer_sees_codexs_own_tool_activity(
    client: Client, fake: FakeResponsesServer, tmp_path: Path, workspace: Path
) -> None:
    events: list[tuple[str, Any]] = []
    await run_native(client, fake, tmp_path, workspace, ECHO, events=events)

    assert ("item_started", "commandExecution") in events
    assert ("item_completed", ("commandExecution", "completed")) in events


@requires_codex_binary
async def test_a_crash_after_approval_does_not_ask_again(
    client: Client,
    fake: FakeResponsesServer,
    tmp_path: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_init = AppServer.__init__

    def init(self: AppServer, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        inner = self._on_notification

        def on_notification(note: dict[str, Any]) -> None:
            if inner is not None:
                inner(note)
            item = (note.get("params") or {}).get("item") or {}
            if (
                note["method"] == "item/completed"
                and item.get("type") == "commandExecution"
                and activity.info().attempt == 1
            ):
                self.kill()  # the command ran, then the Worker "dies" before the turn ends

        self._on_notification = on_notification

    monkeypatch.setattr(AppServer, "__init__", init)

    result, asked = await run_native(
        client,
        fake,
        tmp_path,
        workspace,
        'Log it.\nCALL:exec_command|{"cmd":"echo run >> log.txt"}',
    )

    assert "Process exited with code 0" in result
    # Codex, resuming from the last committed rollout, asked to run the same command again. The
    # Workflow had already approved it this turn, so the handler was asked only once...
    assert len(asked) == 1
    # ...but the command itself ran twice: native tool effects are not exactly-once.
    assert (workspace / "log.txt").read_text() == "run\nrun\n"


@requires_codex_binary
async def test_a_dead_attempts_approval_is_dropped_when_the_retry_asks_again(
    client: Client,
    fake: FakeResponsesServer,
    tmp_path: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    servers: list[AppServer] = []
    original_init = AppServer.__init__

    def init(self: AppServer, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        servers.append(self)

    original_ask = CodexActivities._ask_workflow  # pyright: ignore[reportPrivateUsage]

    async def ask(self: CodexActivities, request: Any) -> Any:
        if activity.info().attempt == 1:
            # The question reaches the Workflow and waits there...
            asking = asyncio.create_task(original_ask(self, request))
            await asyncio.sleep(1)
            # ...then the Worker dies: nobody will ever hear the answer.
            asking.cancel()
            servers[-1].kill()
            await asyncio.sleep(60)
        return await original_ask(self, request)

    monkeypatch.setattr(AppServer, "__init__", init)
    monkeypatch.setattr(CodexActivities, "_ask_workflow", ask)

    async with codex_worker(client, fake, tmp_path) as task_queue:
        handle = await client.start_workflow(
            NativeCodexWorkflow.run,
            NativeInput(ECHO, str(workspace), "signal"),
            id=f"codex-{uuid.uuid4()}",
            task_queue=task_queue,
        )
        # The retry asks again. Only that question may be left waiting: the dead attempt's is gone.
        pending: list[Any] = []
        for _ in range(150):
            pending = await handle.query(NativeCodexWorkflow.pending)
            if any(r.attempt == 2 for r in pending):
                break
            await asyncio.sleep(0.2)
        assert [r.attempt for r in pending] == [2], pending
        assert (await handle.query(NativeCodexWorkflow.asked_requests))[0].attempt == 1

        await handle.signal(NativeCodexWorkflow.decide, args=[pending[0].item_id, True])
        assert "hello" in await handle.result()
    assert (workspace / "out.txt").read_text() == "hello\n"


# ---------------------------------------------------------------------------
# Safe defaults.
# ---------------------------------------------------------------------------


@requires_codex_binary
async def test_commands_do_not_see_the_workers_secrets(
    client: Client,
    fake: FakeResponsesServer,
    tmp_path: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "s3cr3t-aws")
    monkeypatch.setenv("DATABASE_URL", "postgres://user:s3cr3t-db@host/db")
    monkeypatch.setenv("MY_API_TOKEN", "s3cr3t-token")
    result, _ = await run_native(
        client, fake, tmp_path, workspace, 'Show env.\nCALL:exec_command|{"cmd":"env"}'
    )

    assert "PATH=" in result  # the command did run and print its environment
    assert "s3cr3t" not in result


def test_full_access_sandbox_must_be_asked_for_explicitly() -> None:
    with pytest.raises(ValueError, match="allow_full_access"):
        _check_sandbox("danger-full-access", False)
    _check_sandbox("danger-full-access", True)
    _check_sandbox("workspace-write", False)
    with pytest.raises(ValueError, match="sandbox"):
        _check_sandbox("everything", False)


# ---------------------------------------------------------------------------
# Cancellation and shutdown.
# ---------------------------------------------------------------------------


async def wait_for_pending(handle: Any) -> Any:
    for _ in range(100):
        pending = await handle.query(NativeCodexWorkflow.pending)
        if pending:
            return pending
        await asyncio.sleep(0.2)
    raise AssertionError("the approval request never reached the Workflow")


@requires_codex_binary
async def test_cancelling_while_a_command_runs_stops_the_command(
    client: Client, fake: FakeResponsesServer, tmp_path: Path, workspace: Path
) -> None:
    marker = f"sleep-{uuid.uuid4().hex}"
    command = {"cmd": f"sleep 60; echo {marker} > after.txt", "yield_time_ms": 40000}
    prompt = f"Wait.\nCALL:exec_command|{json.dumps(command)}"
    async with codex_worker(client, fake, tmp_path) as task_queue:
        handle = await client.start_workflow(
            NativeCodexWorkflow.run,
            NativeInput(prompt, str(workspace), "approve", heartbeat_seconds=3),
            id=f"codex-{uuid.uuid4()}",
            task_queue=task_queue,
        )
        for _ in range(100):  # until the command is really running
            if await _process_running(f"sleep 60; echo {marker}"):
                break
            await asyncio.sleep(0.2)
        else:
            raise AssertionError("the command never started")

        await handle.cancel()
        try:
            r = await asyncio.wait_for(handle.result(), 30)
        except WorkflowFailureError:
            pass
        else:
            events = [e.event_type for e in (await handle.fetch_history()).events]
            raise AssertionError(
                f"workflow returned {r!r}; {[EventType.Name(e) for e in events]}"
            )
        status = (await handle.describe()).status
        assert status is not None and status.name == "CANCELED"
        # The Workflow waited for the segment to stop Codex: nothing is left running.
        assert not await _process_running(f"sleep 60; echo {marker}")
    assert not (workspace / "after.txt").exists()


async def _process_running(needle: str) -> bool:
    proc = await asyncio.create_subprocess_exec(
        "pgrep", "-f", needle, stdout=asyncio.subprocess.PIPE
    )
    out, _ = await proc.communicate()
    return any(pid != str(os.getpid()) for pid in out.decode().split()) and bool(
        out.strip()
    )


@requires_codex_binary
async def test_cancelling_while_an_approval_is_pending_ends_cleanly(
    client: Client, fake: FakeResponsesServer, tmp_path: Path, workspace: Path
) -> None:
    async with codex_worker(client, fake, tmp_path) as task_queue:
        handle = await client.start_workflow(
            NativeCodexWorkflow.run,
            NativeInput(ECHO, str(workspace), "signal"),
            id=f"codex-{uuid.uuid4()}",
            task_queue=task_queue,
        )
        await wait_for_pending(handle)
        await handle.cancel()
        with pytest.raises(WorkflowFailureError):
            await asyncio.wait_for(handle.result(), 30)
        assert await handle.query(NativeCodexWorkflow.pending) == []
    assert not (workspace / "out.txt").exists()


@requires_codex_binary
async def test_another_worker_continues_after_a_worker_shuts_down_mid_approval(
    client: Client, fake: FakeResponsesServer, tmp_path: Path, workspace: Path
) -> None:
    task_queue = str(uuid.uuid4())

    def worker() -> Worker:
        return Worker(
            client,
            task_queue=task_queue,
            workflows=[NativeCodexWorkflow],
            plugins=[
                CodexPlugin(
                    config_overrides=fake.config_overrides, home_root=str(tmp_path)
                )
            ],
            graceful_shutdown_timeout=timedelta(0),
        )

    first = worker()
    async with first:
        handle = await client.start_workflow(
            NativeCodexWorkflow.run,
            NativeInput(ECHO, str(workspace), "signal"),
            id=f"codex-{uuid.uuid4()}",
            task_queue=task_queue,
        )
        await wait_for_pending(handle)
    # The first Worker is gone, mid-approval. A second one takes the retried segment.
    async with worker():
        pending = await wait_for_pending(handle)
        await handle.signal(NativeCodexWorkflow.decide, args=[pending[0].item_id, True])
        assert "hello" in await asyncio.wait_for(handle.result(), 60)
    assert (workspace / "out.txt").read_text() == "hello\n"


@requires_codex_binary
async def test_a_command_still_running_when_the_segment_ends_is_stopped(
    client: Client, fake: FakeResponsesServer, tmp_path: Path, workspace: Path
) -> None:
    marker = f"sleep-{uuid.uuid4().hex}"
    command = {"cmd": f"sleep 60; echo {marker} > after.txt", "yield_time_ms": 1000}
    result, _ = await run_native(
        client,
        fake,
        tmp_path,
        workspace,
        f"Start it.\nCALL:exec_command|{json.dumps(command)}",
    )

    assert (
        "Process running" in result
    )  # the turn ended while the command was still going
    await asyncio.sleep(0.5)
    assert not await _process_running(f"sleep 60; echo {marker}")


# ---------------------------------------------------------------------------
# Configuration drift on resume.
# ---------------------------------------------------------------------------


def plugin_for(fake: FakeResponsesServer | None, tmp_path: Path) -> CodexPlugin:
    return CodexPlugin(
        config_overrides=fake.config_overrides if fake else [], home_root=str(tmp_path)
    )


async def answered(handle: Any, count: int) -> list[str]:
    for _ in range(100):
        replies = await handle.query(ChatWorkflow.answered)
        if len(replies) >= count:
            return replies
        await asyncio.sleep(0.2)
    raise AssertionError("the turn never finished")


@requires_codex_binary
async def test_a_resumed_thread_uses_the_workers_current_model_provider(
    client: Client, tmp_path: Path, workspace: Path
) -> None:
    before, after = FakeResponsesServer("before"), FakeResponsesServer("after")
    await before.start()
    await after.start()
    try:
        task_queue = str(uuid.uuid4())
        async with Worker(
            client,
            task_queue=task_queue,
            workflows=[ChatWorkflow],
            plugins=[plugin_for(before, tmp_path)],
        ):
            handle = await client.start_workflow(
                ChatWorkflow.run,
                str(workspace),
                id=f"codex-{uuid.uuid4()}",
                task_queue=task_queue,
            )
            await handle.signal(ChatWorkflow.say, "first")
            await answered(handle, 1)
        sent_before = len(before.requests)

        # The Worker is reconfigured to another provider; the thread was recorded under the first.
        async with Worker(
            client,
            task_queue=task_queue,
            workflows=[ChatWorkflow],
            plugins=[plugin_for(after, tmp_path)],
        ):
            await handle.signal(ChatWorkflow.say, "second")
            await answered(handle, 2)
            await handle.signal(ChatWorkflow.say, "")
            await handle.result()
        assert len(before.requests) == sent_before
        assert after.requests and after.requests[-1]["model"] == "after-model"
    finally:
        await before.stop()
        await after.stop()


@requires_codex_binary
async def test_resuming_under_a_worker_that_lacks_the_provider_fails_clearly(
    client: Client, tmp_path: Path, workspace: Path
) -> None:
    before = FakeResponsesServer("before")
    await before.start()
    try:
        task_queue = str(uuid.uuid4())
        async with Worker(
            client,
            task_queue=task_queue,
            workflows=[ChatWorkflow],
            plugins=[plugin_for(before, tmp_path)],
        ):
            handle = await client.start_workflow(
                ChatWorkflow.run,
                str(workspace),
                id=f"codex-{uuid.uuid4()}",
                task_queue=task_queue,
            )
            await handle.signal(ChatWorkflow.say, "first")
            await answered(handle, 1)
        async with Worker(
            client,
            task_queue=task_queue,
            workflows=[ChatWorkflow],
            plugins=[plugin_for(None, tmp_path)],
        ):
            await handle.signal(ChatWorkflow.say, "second")
            with pytest.raises(WorkflowFailureError) as raised:
                await asyncio.wait_for(handle.result(), 30)
        cause = raised.value.cause
        assert isinstance(cause, ActivityError)
        assert isinstance(cause.cause, ApplicationError)
        assert cause.cause.type == "CodexConfigDrift"
        assert "config_overrides" in str(cause.cause)
    finally:
        await before.stop()


# ---------------------------------------------------------------------------
# Token usage: counted per turn from the thread's running total.
# ---------------------------------------------------------------------------

PER_CALL = 18  # what the fake Responses server reports for every model call


@requires_codex_binary
async def test_a_host_tool_turn_counts_every_segments_model_call(
    client: Client, fake: FakeResponsesServer, tmp_path: Path
) -> None:
    async with codex_worker(client, fake, tmp_path) as task_queue:
        handle = await client.start_workflow(
            CodexWorkflow.run, PROMPT, id=f"codex-{uuid.uuid4()}", task_queue=task_queue
        )
        await handle.result()
        usage = (await handle.query(CodexWorkflow.state))["usage"]

    # Two segments (before and after the tool call), one model call each.
    assert len(fake.requests) == 2
    assert usage["total_tokens"] == 2 * PER_CALL


@requires_codex_binary
async def test_usage_is_per_turn_across_a_resumed_thread(
    client: Client, fake: FakeResponsesServer, tmp_path: Path, workspace: Path
) -> None:
    script = "\n".join('CALL:exec_command|{"cmd":"echo %d"}' % i for i in range(2))
    async with codex_worker(client, fake, tmp_path) as task_queue:
        handle = await client.start_workflow(
            ChatWorkflow.run,
            str(workspace),
            id=f"codex-{uuid.uuid4()}",
            task_queue=task_queue,
        )
        await handle.signal(ChatWorkflow.say, f"go\n{script}")
        await answered(handle, 1)
        calls_first = len(fake.requests)
        await handle.signal(ChatWorkflow.say, "plain follow-up")
        await answered(handle, 2)
        calls_second = len(fake.requests) - calls_first
        await handle.signal(ChatWorkflow.say, "")
        await handle.result()
        first, second = await handle.query(ChatWorkflow.turn_usages)
    assert first is not None and second is not None

    # Every turn is billed for exactly its own model calls, even though each segment runs in a
    # new process that starts from the thread's restored running total.
    assert calls_first == 3 and first.total_tokens == calls_first * PER_CALL
    assert second.total_tokens == calls_second * PER_CALL
    assert first.input_tokens and second.input_tokens
