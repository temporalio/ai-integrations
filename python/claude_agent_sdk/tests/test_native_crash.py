"""Hard worker loss during a native batch, using the actual published engine."""

from __future__ import annotations

import asyncio
import os
import shlex
import sys
import uuid
from pathlib import Path
from typing import Any

import pytest

from temporalio.client import Client
from tests.helpers.fake_messages_api import FakeMessagesAPI, engine_env, history_of
from tests.helpers.processes import alive, descendants
from tests.native_batches.workflows import NativeWorkflow
from tests.test_crash import kill, start_worker, wait_until


@pytest.mark.skipif(
    os.name == "nt",
    reason="This test uses POSIX Bash; Windows job cleanup has separate coverage",
)
@pytest.mark.timeout(120)
async def test_native_worker_loss_preserves_completed_calls(
    client: Client, address: str, tmp_path: Path
) -> None:
    """A killed controller is fenced; retry skips the committed native operation."""
    effects, started = tmp_path / "effects.txt", tmp_path / "started.txt"

    def command(script: str) -> str:
        return shlex.join([sys.executable, "-c", script])

    calls = [
        {
            "type": "tool_use",
            "id": "finished_original",
            "name": "Bash",
            "input": {
                "command": command(
                    f"from pathlib import Path; p=Path({str(effects)!r}); p.open('a').write('committed\\n'); print('finished')"
                )
            },
        },
        {
            "type": "tool_use",
            "id": "interrupted_original",
            "name": "Bash",
            "input": {
                "command": command(
                    f"import os,time; from pathlib import Path; p=Path({str(started)!r}); retry=p.exists(); p.write_text(str(os.getpid())); time.sleep(0 if retry else 100); print('recovered')"
                )
            },
        },
    ]

    def policy(body: dict[str, Any]) -> list[dict[str, Any]]:
        _, _, history = history_of(body)
        return calls if not history else [{"type": "text", "text": "recovered"}]

    api = FakeMessagesAPI(policy).start()
    queue = "native-crash-" + uuid.uuid4().hex
    env = {**engine_env(api, str(tmp_path / "cfg")), "ENGINE_CWD": str(tmp_path)}
    workers = []
    handle = None
    try:
        workers.append(
            await start_worker(
                address,
                queue,
                env,
                tmp_path / "w1.log",
                module="tests.native_batches.worker",
            )
        )
        handle = await client.start_workflow(
            NativeWorkflow.run,
            args=["run native operations", ["Bash"]],
            id=queue,
            task_queue=queue,
        )
        await wait_until(started.exists)
        old_tool = int(started.read_text())
        old_tree = descendants(workers[0].pid)
        assert old_tool in old_tree
        kill(workers[0])
        await wait_until(lambda: not any(alive(pid) for pid in old_tree), timeout=20)
        workers.append(
            await start_worker(
                address,
                queue,
                env,
                tmp_path / "w2.log",
                module="tests.native_batches.worker",
            )
        )
        assert await asyncio.wait_for(handle.result(), 45) == "recovered"
        assert effects.read_text().splitlines() == ["committed"]
        scheduled = [
            e.activity_task_scheduled_event_attributes.activity_id
            async for e in handle.fetch_history_events()
            if e.HasField("activity_task_scheduled_event_attributes")
        ]
        assert scheduled.count("tool-finished_original") == 1
        assert scheduled.count("tool-interrupted_original") == 1
        assert any(t.startswith("native-batch-") for t in scheduled)
        assert api.errors == []
    finally:
        for worker in workers:
            if worker.poll() is None:
                kill(worker)
        if handle is not None:
            try:
                await handle.terminate()
            except Exception:
                pass
        api.stop()
