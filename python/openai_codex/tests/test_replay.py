"""Replay safety: recorded Workflow histories must keep replaying against the current code.

A change to ``CodexSession`` (or to a Workflow built on it) that alters what an in-flight Workflow
does, in a way that is not guarded with ``workflow.patched``, makes these fail with a determinism
error. Regenerate the histories deliberately, after deciding the change is safe to ship::

    CODEX_RECORD_HISTORIES=1 make test PYTEST_ARGS="-k record_histories"
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from pathlib import Path

import pytest

from temporalio.client import Client, WorkflowHistory
from temporalio.openai_codex.testing import FakeResponsesServer
from temporalio.worker import Replayer
from tests._workflows import (
    ChatWorkflow,
    CodexWorkflow,
    NativeCodexWorkflow,
    NativeInput,
    ObservedCodexWorkflow,
)
from tests.test_codex import (  # noqa: F401  (fixtures)
    ECHO,
    PROMPT,
    codex_worker,
    fake,  # pyright: ignore[reportUnusedImport]
    isolated_env,  # pyright: ignore[reportUnusedImport]
    requires_codex_binary,
    workspace,  # pyright: ignore[reportUnusedImport]
)

HISTORIES = Path(__file__).parent / "histories"
WORKFLOWS = [CodexWorkflow, ObservedCodexWorkflow, NativeCodexWorkflow, ChatWorkflow]


@pytest.mark.skipif(
    os.environ.get("CODEX_RECORD_HISTORIES") != "1",
    reason="only run to (re)record the golden histories",
)
@requires_codex_binary
async def test_record_histories(
    client: Client,
    fake: FakeResponsesServer,
    tmp_path: Path,
    workspace: Path,  # noqa: F811
) -> None:
    HISTORIES.mkdir(exist_ok=True)
    async with codex_worker(client, fake, tmp_path) as task_queue:

        async def record(name: str, run, arg, *signals: tuple) -> None:  # type: ignore[no-untyped-def]
            handle = await client.start_workflow(
                run, arg, id=f"golden-{name}", task_queue=task_queue
            )
            for signal, value in signals:
                await handle.signal(signal, value)
            await asyncio.wait_for(handle.result(), 60)
            history = await handle.fetch_history()
            (HISTORIES / f"{name}.json").write_text(history.to_json() + "\n")

        await record("host_tool", CodexWorkflow.run, PROMPT)
        await record(
            "native_approved",
            NativeCodexWorkflow.run,
            NativeInput(ECHO, str(workspace), "approve"),
        )
        await record(
            "native_declined",
            NativeCodexWorkflow.run,
            NativeInput(ECHO, str(workspace), "decline"),
        )
        await record(
            "two_turns",
            ChatWorkflow.run,
            str(workspace),
            (ChatWorkflow.say, "first"),
            (ChatWorkflow.say, "second"),
            (ChatWorkflow.say, ""),
        )


def history_files() -> list[Path]:
    return sorted(HISTORIES.glob("*.json"))


def test_golden_histories_exist() -> None:
    assert {p.stem for p in history_files()} >= {
        "host_tool",
        "native_approved",
        "native_declined",
        "two_turns",
    }


@pytest.mark.parametrize("path", history_files(), ids=lambda p: p.stem)
async def test_recorded_histories_still_replay(path: Path) -> None:
    replayer = Replayer(workflows=WORKFLOWS)
    await replayer.replay_workflow(
        WorkflowHistory.from_json(
            f"replay-{uuid.uuid4()}", json.loads(path.read_text())
        )
    )
