"""Large tool results through Temporal's External Storage.

Without it, a result over 2 MB cannot be recorded at all, and smaller large results
still fill the Workflow's history (each result is stored twice: as the tool's result
and in the next segment's input). With it, payloads over the threshold (256 KiB by
default) are kept in the store and the history holds small references.
"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

import pytest

from temporalio.claude_agent_sdk import (
    ClaudeAgentPlugin,
    ClaudeAgentSdkRunner,
    FileSessionStore,
    SegmentRunner,
)
from temporalio.claude_agent_sdk.testing import ScriptedClaude
from temporalio.client import Client
from temporalio.converter import DataConverter, ExternalStorage
from temporalio.worker import Worker
from tests.helpers.fake_messages_api import engine_env, start_with_policy
from tests.helpers.storage import FolderStorageDriver
from tests.helpers.workers import FAIL_FAST
from tests.refund import shop
from tests.storage.workflows import FetchWorkflow, fetch_document, fetch_policy

pytestmark = pytest.mark.timeout(180)


async def fetch(
    address: str,
    tmp: Path,
    kb: int,
    converter: DataConverter,
    runner: SegmentRunner | None = None,
) -> tuple[str, int, int]:
    """Run one fetch; return the answer, the history size, and the largest event size."""
    client = await Client.connect(address, data_converter=converter)
    queue = f"fetch-{uuid.uuid4().hex[:8]}"
    plugin = ClaudeAgentPlugin(runner or ScriptedClaude(fetch_policy, tmp / "fake"))
    async with Worker(
        client,
        task_queue=queue,
        workflows=[FetchWorkflow],
        activities=[fetch_document],
        plugins=[plugin],
        **FAIL_FAST,
    ):
        handle = await client.start_workflow(
            FetchWorkflow.run, f"fetch {kb} KB", id=queue, task_queue=queue
        )
        answer = await asyncio.wait_for(handle.result(), 150)
        info = (await handle.describe()).raw_description.workflow_execution_info
        largest = max(e.ByteSize() for e in (await handle.fetch_history()).events)
    return answer, info.history_size_bytes, largest


@pytest.mark.usefixtures("shop_dir")
async def test_a_result_over_2_mb_reaches_claude_through_external_storage(
    address: str, tmp_path: Path
) -> None:
    driver = FolderStorageDriver(tmp_path / "blobs")
    converter = DataConverter(external_storage=ExternalStorage(drivers=[driver]))
    answer, history_bytes, largest = await fetch(address, tmp_path, 3000, converter)
    assert answer == "got 3072000 characters"  # Claude saw the whole document
    assert history_bytes < 20_000 and largest < 4_000  # the history holds references
    assert driver.stored_bytes() > 3_072_000
    assert len(shop.executions("fetch_document")) == 1


@pytest.mark.usefixtures("shop_dir")
async def test_without_external_storage_a_result_over_2_mb_cannot_be_recorded(
    address: str, tmp_path: Path
) -> None:
    answer, _, _ = await fetch(address, tmp_path, 3000, DataConverter.default)
    assert answer.startswith("fetch failed")  # Claude never saw the document
    assert "exceeded the error limit" in answer


@pytest.mark.timeout(240)
@pytest.mark.usefixtures("shop_dir")
async def test_real_engine_reads_a_large_result_from_external_storage(
    address: str, tmp_path: Path
) -> None:
    api = start_with_policy(fetch_policy)
    (tmp_path / "work").mkdir()
    runner = ClaudeAgentSdkRunner(
        session_store=FileSessionStore(tmp_path / "sessions"),
        cwd=str(tmp_path / "work"),
        env=engine_env(api, str(tmp_path / "cfg")),
    )
    driver = FolderStorageDriver(tmp_path / "blobs")
    converter = DataConverter(external_storage=ExternalStorage(drivers=[driver]))
    try:
        # Over 1 MiB: the engine echoes the result as one line, over the SDK's
        # default message limit, which the runner raises.
        answer, history_bytes, largest = await fetch(
            address, tmp_path, 2000, converter, runner
        )
    finally:
        api.stop()
    assert answer == "got 2048000 characters"  # through the engine, intact
    assert history_bytes < 20_000 and largest < 4_000
    assert api.errors == [] and runner.stub_calls == 0
