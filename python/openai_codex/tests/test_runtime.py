"""Exercise the published SDK and its matching bundled runtime, then replay."""

from __future__ import annotations

import asyncio
import sys
import uuid
from pathlib import Path

import pytest
from openai_codex import AsyncCodex, CodexConfig
from openai_codex.async_client import AsyncCodexClient

from temporalio.client import Client, WorkflowFailureError
from temporalio.exceptions import ApplicationError
from temporalio.openai_codex import OpenAICodexPlugin
from temporalio.worker import Replayer, Worker
from tests.helpers.model import LocalModel
from tests.helpers.workflows import (
    ChatWorkflow,
    FailedTurnWorkflow,
    StdoutFailureWorkflow,
)


async def test_provider_failure_fails_workflow(client: Client, tmp_path: Path) -> None:
    model = LocalModel(tmp_path, error_status=401)
    queue = f"provider-failure-{uuid.uuid4()}"
    try:
        async with Worker(
            client,
            task_queue=queue,
            workflows=[FailedTurnWorkflow],
            plugins=[OpenAICodexPlugin()],
        ):
            handle = await client.start_workflow(
                FailedTurnWorkflow.run, model.config(), id=queue, task_queue=queue
            )
            with pytest.raises(WorkflowFailureError) as raised:
                await asyncio.wait_for(handle.result(), 10)
            error = raised.value.cause
            assert isinstance(error, ApplicationError)
            assert error.type == "CodexError"
            assert "401" in error.message
            assert model.requests
        history = await handle.fetch_history()
        assert not any(
            event.HasField("workflow_task_failed_event_attributes")
            for event in history.events
        )
        await Replayer(
            workflows=[FailedTurnWorkflow], plugins=[OpenAICodexPlugin()]
        ).replay_workflow(history)
    finally:
        model.close()


async def test_terminal_stdout_error_fails_workflow(client: Client) -> None:
    queue = f"stdout-failure-{uuid.uuid4()}"
    async with Worker(
        client,
        task_queue=queue,
        workflows=[StdoutFailureWorkflow],
        plugins=[OpenAICodexPlugin()],
    ):
        handle = await client.start_workflow(
            StdoutFailureWorkflow.run,
            CodexConfig(
                launch_args_override=(
                    sys.executable,
                    "-c",
                    "print('x' * (16 * 1024 * 1024 + 1), flush=True)",
                )
            ),
            id=queue,
            task_queue=queue,
        )
        with pytest.raises(WorkflowFailureError) as raised:
            await asyncio.wait_for(handle.result(), 5)
        error = raised.value.cause
        assert isinstance(error, ApplicationError)
        assert error.type == "TransportClosedError"
        assert "limit" in error.message
    history = await handle.fetch_history()
    assert not any(
        event.HasField("workflow_task_failed_event_attributes")
        for event in history.events
    )
    read_failures = [
        event.activity_task_failed_event_attributes
        for event in history.events
        if event.HasField("activity_task_failed_event_attributes")
    ]
    assert len(read_failures) == 1
    assert read_failures[0].failure.application_failure_info.non_retryable
    await Replayer(
        workflows=[StdoutFailureWorkflow], plugins=[OpenAICodexPlugin()]
    ).replay_workflow(history)


@pytest.mark.parametrize("high_level", [False, True])
async def test_bundled_runtime(
    client: Client, tmp_path: Path, high_level: bool
) -> None:
    model = LocalModel(tmp_path)
    plugin = OpenAICodexPlugin()
    queue = f"codex-{uuid.uuid4()}"
    try:
        async with Worker(
            client, task_queue=queue, workflows=[ChatWorkflow], plugins=[plugin]
        ):
            handle = await client.start_workflow(
                ChatWorkflow.run,
                args=[model.config(), high_level],
                id=queue,
                task_queue=queue,
            )
            result = await handle.result()
            assert result == ["Hello from local Codex."] * (2 if high_level else 1)
        assert model.requests
        await Replayer(
            workflows=[ChatWorkflow], plugins=[OpenAICodexPlugin()]
        ).replay_workflow(await handle.fetch_history())
    finally:
        model.close()


async def test_upstream_outside_workflow(tmp_path: Path) -> None:
    OpenAICodexPlugin()
    OpenAICodexPlugin()
    model = LocalModel(tmp_path)
    try:
        async with AsyncCodex(model.config()) as codex:
            assert type(codex._client) is AsyncCodexClient
            thread = await codex.thread_start()
            assert (
                await thread.run("Say hello.")
            ).final_response == "Hello from local Codex."
    finally:
        model.close()


@pytest.mark.parametrize("high_level", [False, True])
async def test_steer_and_interrupt(
    client: Client, tmp_path: Path, high_level: bool
) -> None:
    import asyncio

    from tests.helpers.workflows import ControlWorkflow

    model = LocalModel(tmp_path)
    model.gate.clear()
    queue = f"control-{uuid.uuid4()}"
    try:
        async with Worker(
            client,
            task_queue=queue,
            workflows=[ControlWorkflow],
            plugins=[OpenAICodexPlugin()],
        ):
            handle = await client.start_workflow(
                ControlWorkflow.run,
                args=[model.config(), high_level],
                id=queue,
                task_queue=queue,
            )
            while not model.requests:
                await asyncio.sleep(0.01)
            await handle.signal(ControlWorkflow.resume)
            assert await handle.result() == "interrupted"
        await Replayer(
            workflows=[ControlWorkflow], plugins=[OpenAICodexPlugin()]
        ).replay_workflow(await handle.fetch_history())
    finally:
        model.close()
