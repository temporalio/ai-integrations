"""Real Modal sandbox turns. Skipped unless credentials are set; not run in CI.

Requires ``MODAL_TOKEN_ID``, ``MODAL_TOKEN_SECRET``, and either ``ANTHROPIC_API_KEY``
or ``CLAUDE_MODAL_SECRET`` (a Modal Secret that provides ``ANTHROPIC_API_KEY``).
Optional: ``CLAUDE_MODAL_IMAGE_ID`` (a prebuilt image with the pinned CLI),
``CLAUDE_MODAL_CLI_PATH`` (the CLI inside that image), ``CLAUDE_MODAL_RUNTIME``
(``gvisor`` or ``vm``). Snapshots expire after an hour; Modal cannot delete one. Every
sandbox a test starts is terminated in teardown.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import subprocess
import sys
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any

import modal
import pytest
import pytest_asyncio

from temporalio.claude_agent_sdk import (
    ClaudeAgentPlugin,
    ClaudePartialEvent,
    FileLeaseStore,
)
from temporalio.claude_agent_sdk._activity import ClaudeTurnActivity
from temporalio.claude_agent_sdk._models import ClaudeTurnInput, ClaudeTurnOptions
from temporalio.claude_agent_sdk.sandbox.modal import (
    SESSION_TAG,
    ModalSandbox,
    ModalSandboxRuntime,
)
from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.contrib.workflow_streams import WorkflowStreamClient
from temporalio.testing import ActivityEnvironment, WorkflowEnvironment
from temporalio.worker import Worker

from .helpers import kill_worker as kw
from .helpers.workflows import ClaudeChat, ClaudeSandboxFailover, ClaudeStream

pytestmark = pytest.mark.skipif(
    not (
        os.environ.get("MODAL_TOKEN_ID")
        and os.environ.get("MODAL_TOKEN_SECRET")
        and (
            os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("CLAUDE_MODAL_SECRET")
        )
    ),
    reason=(
        "real Modal sandbox: set MODAL_TOKEN_ID, MODAL_TOKEN_SECRET, and "
        "ANTHROPIC_API_KEY or CLAUDE_MODAL_SECRET"
    ),
)

_PLUGIN_ROOT = Path(__file__).resolve().parents[1]
_MODEL = "claude-haiku-5-5"
_DEFAULT_CLI = (
    "/usr/local/lib/python3.12/site-packages/claude_agent_sdk/_bundled/claude"
)


def _image() -> object:
    if image_id := os.environ.get("CLAUDE_MODAL_IMAGE_ID"):
        return modal.Image.from_id(image_id)
    return modal.Image.debian_slim(python_version="3.12").pip_install(
        "claude-agent-sdk==0.2.165"
    )


@dataclass(frozen=True)
class _TrackedRuntime(ModalSandboxRuntime):
    """Remembers every sandbox it starts, so teardown can end them."""

    started: list[ModalSandbox] = field(default_factory=list)

    async def start(
        self,
        *,
        session_id: str | None,
        snapshot: str | None,
        generation: int | None = None,
    ) -> ModalSandbox:
        sandbox = await super().start(
            session_id=session_id, snapshot=snapshot, generation=generation
        )
        self.started.append(sandbox)
        return sandbox


def _runtime() -> _TrackedRuntime:
    secret = (
        modal.Secret.from_name(os.environ["CLAUDE_MODAL_SECRET"])
        if os.environ.get("CLAUDE_MODAL_SECRET")
        else modal.Secret.from_dict(
            {"ANTHROPIC_API_KEY": os.environ["ANTHROPIC_API_KEY"]}
        )
    )
    runtime = os.environ.get("CLAUDE_MODAL_RUNTIME")
    return _TrackedRuntime(
        image=_image(),
        cli_path=os.environ.get("CLAUDE_MODAL_CLI_PATH", _DEFAULT_CLI),
        secrets=(secret,),
        # Workflows that pass no model still run the cheap one.
        env={"ANTHROPIC_MODEL": _MODEL},
        runtime=runtime if runtime in ("gvisor", "vm") else None,  # type: ignore[arg-type]
        timeout=timedelta(minutes=10),
        snapshot_ttl=timedelta(hours=1),
    )


@pytest_asyncio.fixture  # type: ignore[reportUntypedFunctionDecorator]
async def runtime() -> AsyncIterator[_TrackedRuntime]:
    """A Modal runtime whose sandboxes are all terminated when the test ends."""
    tracked = _runtime()
    yield tracked
    for sandbox in tracked.started:
        with contextlib.suppress(Exception):  # already gone
            await sandbox.terminate()


async def _live(app_name: str, session_id: str) -> list[Any]:
    """Running sandboxes tagged with ``session_id``. Finished ones are not listed."""
    app = await modal.App.lookup.aio(app_name, create_if_missing=True)
    return [
        sandbox
        async for sandbox in modal.Sandbox.list.aio(
            app_id=app.app_id, tags={SESSION_TAG: session_id}
        )
    ]


@pytest.mark.timeout(240)
async def test_a_real_modal_turn_snapshots_and_the_next_turn_resumes(
    runtime: _TrackedRuntime,
) -> None:
    """One model turn in Modal, then a second sandbox that still knows the first turn."""
    activity = ClaudeTurnActivity(sandbox=runtime)
    session_id = str(uuid.uuid4())
    options = ClaudeTurnOptions(model=_MODEL)

    first = await ActivityEnvironment().run(
        activity.run_turn,
        ClaudeTurnInput(
            prompt="The code word is bluebird. Reply with exactly: noted",
            session_id=session_id,
            options=options,
        ),
    )
    assert first.text
    assert "noted" in first.text.lower()
    assert first.sandbox_snapshot

    second = await ActivityEnvironment().run(
        activity.run_turn,
        ClaudeTurnInput(
            prompt="What code word did I give you? Reply with that word only.",
            session_id=session_id,
            resume=True,
            sandbox_snapshot=first.sandbox_snapshot,
            options=options,
        ),
    )
    assert second.text
    assert "bluebird" in second.text.lower()
    assert second.sandbox_snapshot
    assert second.sandbox_snapshot != first.sandbox_snapshot


@pytest.mark.timeout(180)
async def test_a_real_modal_turn_returns_structured_output(
    runtime: _TrackedRuntime,
) -> None:
    """A structured reply from a Modal sandbox. Separate from transcript resume."""
    activity = ClaudeTurnActivity(sandbox=runtime)
    result = await ActivityEnvironment().run(
        activity.run_turn,
        ClaudeTurnInput(
            prompt="Return the word bluebird.",
            session_id=str(uuid.uuid4()),
            options=ClaudeTurnOptions(
                model=_MODEL,
                output_schema={
                    "type": "object",
                    "properties": {"word": {"type": "string"}},
                    "required": ["word"],
                    "additionalProperties": False,
                },
            ),
        ),
    )
    assert result.structured_output == {"word": "bluebird"}
    assert result.sandbox_snapshot


@pytest.mark.timeout(240)
async def test_a_real_modal_turn_streams_partials_and_the_result_is_authoritative(
    env: WorkflowEnvironment, tmp_path: Path, runtime: _TrackedRuntime
) -> None:
    """Partials reach a hosted Workflow Stream; the workflow result is the turn's own text."""
    client = kw.client(env)
    task_queue = f"claude-modal-{uuid.uuid4().hex[:8]}"
    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[ClaudeStream],
        plugins=[ClaudeAgentPlugin(sandbox=runtime, workspace_root=tmp_path / "host")],
    ):
        handle = await client.start_workflow(
            ClaudeStream.run,
            "Reply with exactly: the bluebird sings at dawn",
            id=task_queue,
            task_queue=task_queue,
        )
        stream = WorkflowStreamClient.create(client, handle.id)
        published: list[ClaudePartialEvent] = []

        async def collect() -> None:
            async for item in stream.subscribe(
                ["claude"],
                from_offset=0,
                result_type=ClaudePartialEvent,
                poll_cooldown=timedelta(milliseconds=50),
            ):
                published.append(item.data)
                if item.data.kind == "message_stop":
                    return

        await asyncio.wait_for(collect(), timeout=180)
        await handle.signal(ClaudeStream.finish)
        text = await asyncio.wait_for(handle.result(), timeout=30)

    assert published
    assert {event.attempt_id for event in published} == {published[0].attempt_id}
    assert published[0].attempt_id.endswith(".1")
    assert [event.sequence for event in published] == list(range(len(published)))
    deltas = [
        event.text
        for event in published
        if event.kind == "content_block_delta"
        and event.text
        and event.parent_tool_use_id is None
    ]
    assert deltas, [event.kind for event in published]
    streamed = "".join(deltas).strip()
    assert "bluebird" in streamed.lower()
    assert streamed in text
    assert runtime.started


@pytest.mark.timeout(240)
async def test_a_real_modal_turn_calls_a_workflow_tool_once(
    env: WorkflowEnvironment, tmp_path: Path, runtime: _TrackedRuntime
) -> None:
    """The CLI in Modal calls a workflow tool; the recorded result reaches the answer."""
    client = kw.client(env)
    task_queue = f"claude-modal-{uuid.uuid4().hex[:8]}"
    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[ClaudeChat],
        plugins=[ClaudeAgentPlugin(sandbox=runtime, workspace_root=tmp_path / "host")],
    ):
        handle = await client.start_workflow(
            ClaudeChat.run, id=task_queue, task_queue=task_queue
        )
        try:
            reply = await asyncio.wait_for(
                handle.execute_update(
                    ClaudeChat.chat_with_tool,
                    "Call get_weather for Paris and reply with only the result.",
                ),
                timeout=200,
            )
            calls = await handle.query(ClaudeChat.weather_calls)
        finally:
            await handle.terminate()

    assert "sunny" in reply.lower(), reply
    assert calls == 1
    assert runtime.started


def _failover_plugin(workspace: str, leases: str) -> ClaudeAgentPlugin:
    return ClaudeAgentPlugin(
        sandbox=_runtime(),
        workspace_root=workspace,
        lease_store=FileLeaseStore(leases),
        disconnect_timeout=timedelta(seconds=1),
    )


async def _serve(spec: dict[str, str]) -> None:
    """Worker A of the worker-loss test, run in its own process so the test can SIGKILL it."""
    client = await Client.connect(
        spec["target"],
        namespace=spec["namespace"],
        data_converter=pydantic_data_converter,
    )
    async with Worker(
        client,
        task_queue=spec["task_queue"],
        workflows=[ClaudeSandboxFailover],
        max_cached_workflows=0,
        plugins=[_failover_plugin(spec["workspace"], spec["leases"])],
    ):
        Path(spec["ready"]).write_text("up")
        await asyncio.Event().wait()


@pytest.mark.timeout(480)
async def test_a_lost_worker_fails_over_to_the_committed_snapshot(
    env: WorkflowEnvironment, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Worker A is SIGKILLed mid tool call; B resumes the snapshot and ends A's stray sandbox."""
    caplog.set_level(
        logging.WARNING, logger="temporalio.claude_agent_sdk.sandbox.modal"
    )
    app_name = _runtime().app_name
    session_id = str(uuid.uuid4())
    leases = str(tmp_path / "leases")
    client = kw.client(env)
    task_queue = f"claude-modal-{uuid.uuid4().hex[:8]}"
    ready = tmp_path / "worker-a-ready"
    spec = {
        "target": env.client.service_client.config.target_host,
        "namespace": env.client.namespace,
        "task_queue": task_queue,
        "workspace": str(tmp_path / "host-a"),
        "leases": leases,
        "ready": str(ready),
    }
    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import asyncio, json, sys\n"
            f"from {__name__} import {_serve.__name__} as serve\n"
            "asyncio.run(serve(json.loads(sys.argv[1])))",
            json.dumps(spec),
        ],
        cwd=_PLUGIN_ROOT,
        env={**os.environ, "PYTHONPATH": str(_PLUGIN_ROOT)},
        start_new_session=True,
    )
    try:

        async def worker_a_ready() -> bool:
            assert proc.poll() is None, "worker A exited before it was ready"
            return ready.exists()

        await kw.until(worker_a_ready, timeout=90, interval=0.5)
        handle = await client.start_workflow(
            ClaudeSandboxFailover.run,
            session_id,
            id=task_queue,
            task_queue=task_queue,
        )

        async def tool_called() -> bool:
            return await handle.query(ClaudeSandboxFailover.calls) >= 1

        await kw.until(tool_called, timeout=240, interval=0.5)
        first_snapshot, _ = await handle.query(ClaudeSandboxFailover.snapshots)
        assert first_snapshot
        (stray,) = await _live(app_name, session_id)

        kw.killpg(proc.pid)
        proc.wait(timeout=10)
        assert await stray.poll.aio() is None, "A's sandbox ended with its worker"

        async with Worker(
            client,
            task_queue=task_queue,
            workflows=[ClaudeSandboxFailover],
            max_cached_workflows=0,
            plugins=[_failover_plugin(str(tmp_path / "host-b"), leases)],
        ):
            await handle.signal(ClaudeSandboxFailover.release)
            text = await asyncio.wait_for(handle.result(), timeout=300)
            calls = await handle.query(ClaudeSandboxFailover.calls)
            snapshots = await handle.query(ClaudeSandboxFailover.snapshots)
        history = await handle.fetch_history()

        assert await stray.poll.aio() is not None, "A's sandbox is still running"
        assert stray.object_id in caplog.text
        assert await _live(app_name, session_id) == []
    finally:
        kw.killpg(proc.pid)
        if proc.poll() is None:
            proc.wait(timeout=10)
        for sandbox in await _live(app_name, session_id):
            with contextlib.suppress(Exception):
                await sandbox.terminate.aio(wait=True)

    assert "bluebird" in text.lower()
    assert "sunny" in text.lower()
    assert calls == 1
    assert snapshots[0] == first_snapshot
    assert snapshots[1] and snapshots[1] != first_snapshot
    completed = [
        event
        for event in history.events
        if event.HasField("activity_task_completed_event_attributes")
    ]
    timed_out = [
        event
        for event in history.events
        if event.HasField("activity_task_timed_out_event_attributes")
    ]
    assert len(completed) == 2, "one per turn: the first, then the resumed second"
    assert len(timed_out) == 1, "A's attempt, lost with its worker"
