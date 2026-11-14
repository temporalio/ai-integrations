"""Record the histories that ``tests/test_replay.py`` replays.

Run from ``python/microsoft_agent_framework`` with the same environment ``make test`` uses::

    UV_NO_EDITABLE=1 uv run --locked python -m tests.record_histories

Each recorder runs one of the suite's workflows against a local dev server
with scripted chat clients and an in-process MCP server, then writes
``handle.fetch_history()`` as JSON under ``tests/histories``. A file that
exists is not replaced: a recording made before a release stays as it is
(``tests/histories/README.md``). ``--suffix NAME`` writes ``<scenario>-NAME.json``
next to the old files for a new shape; ``--overwrite "<reason>"`` replaces
existing files and prints the reason next to each.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import uuid
from collections.abc import Awaitable, Callable
from datetime import timedelta
from pathlib import Path

from mcp import Client as MCPClient

from temporalio.client import Client, WorkflowHistory
from temporalio.microsoft_agent_framework import MicrosoftAgentFrameworkPlugin
from temporalio.microsoft_agent_framework.testing import (
    fake_chat_client,
    function_call_response,
    text_response,
)
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker
from tests import DEV_SERVER_DOWNLOAD_VERSION
from tests.echo_mcp_server import server
from tests.test_agent import RunInput, ToolLoopWorkflow
from tests.test_approval import ApprovalWorkflow, Decision
from tests.test_mcp import MCPInput, MCPWorkflow
from tests.test_streaming import StreamingWorkflow, StreamInput

HISTORIES = Path(__file__).with_name("histories")


async def _record(
    client: Client,
    plugin: MicrosoftAgentFrameworkPlugin,
    workflow: type,
    start: Callable[[str], Awaitable[WorkflowHistory]],
) -> WorkflowHistory:
    task_queue = f"record-{uuid.uuid4()}"
    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[workflow],
        plugins=[plugin],
        max_cached_workflows=0,
    ):
        return await start(task_queue)


async def record_tool_loop(client: Client) -> WorkflowHistory:
    plugin = MicrosoftAgentFrameworkPlugin(
        chat_clients={
            "model": fake_chat_client(
                [
                    function_call_response(
                        "get_weather", {"city": "Paris"}, call_id="call-1"
                    ),
                    function_call_response(
                        "get_weather", {"city": "Oslo"}, call_id="call-2"
                    ),
                    text_response("Rainy in both."),
                ]
            )
        }
    )

    async def start(task_queue: str) -> WorkflowHistory:
        handle = await client.start_workflow(
            ToolLoopWorkflow.run,
            RunInput(prompt="Paris and Oslo?"),
            id=f"record-tool-loop-{uuid.uuid4()}",
            task_queue=task_queue,
        )
        await handle.result()
        return await handle.fetch_history()

    return await _record(client, plugin, ToolLoopWorkflow, start)


async def record_approval(client: Client) -> WorkflowHistory:
    plugin = MicrosoftAgentFrameworkPlugin(
        chat_clients={
            "model": fake_chat_client(
                [
                    function_call_response(
                        "delete_thing", {"name": "x"}, call_id="call-1"
                    ),
                    text_response("Deleted x."),
                ]
            )
        }
    )

    async def start(task_queue: str) -> WorkflowHistory:
        handle = await client.start_workflow(
            ApprovalWorkflow.run,
            "delete x",
            id=f"record-approval-{uuid.uuid4()}",
            task_queue=task_queue,
        )
        pending: list[dict] = []
        while not pending:
            await asyncio.sleep(0.1)
            pending = await handle.query(ApprovalWorkflow.pending_requests)
        await handle.execute_update(
            ApprovalWorkflow.decide, Decision(pending[0]["id"], True)
        )
        await handle.result()
        return await handle.fetch_history()

    return await _record(client, plugin, ApprovalWorkflow, start)


async def record_streaming(client: Client) -> WorkflowHistory:
    plugin = MicrosoftAgentFrameworkPlugin(
        chat_clients={
            "model": fake_chat_client(
                [
                    function_call_response(
                        "get_weather", {"city": "Oslo"}, call_id="call-1"
                    ),
                    text_response("Streamed answer: rainy in Oslo, bring a coat."),
                ]
            )
        }
    )

    async def start(task_queue: str) -> WorkflowHistory:
        handle = await client.start_workflow(
            StreamingWorkflow.run,
            StreamInput(topic="events"),
            id=f"record-streaming-{uuid.uuid4()}",
            task_queue=task_queue,
        )
        await handle.result()
        return await handle.fetch_history()

    return await _record(client, plugin, StreamingWorkflow, start)


async def record_mcp(client: Client) -> WorkflowHistory:
    plugin = MicrosoftAgentFrameworkPlugin(
        chat_clients={
            "model": fake_chat_client(
                [
                    function_call_response(
                        "echo", {"value": "ab", "times": 2}, call_id="call-1"
                    ),
                    text_response("Echoed."),
                ]
            )
        },
        mcp_clients={"echo": lambda: MCPClient(server)},
    )

    async def start(task_queue: str) -> WorkflowHistory:
        handle = await client.start_workflow(
            MCPWorkflow.run,
            MCPInput(),
            id=f"record-mcp-{uuid.uuid4()}",
            task_queue=task_queue,
        )
        await handle.result()
        return await handle.fetch_history()

    return await _record(client, plugin, MCPWorkflow, start)


RECORDERS = {
    "tool_loop.json": record_tool_loop,
    "approval.json": record_approval,
    "streaming.json": record_streaming,
    "mcp.json": record_mcp,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse ``--suffix`` and ``--overwrite REASON``."""
    parser = argparse.ArgumentParser(
        description="Record the histories that tests/test_replay.py replays."
    )
    parser.add_argument(
        "--suffix",
        default="",
        help="write <scenario>-SUFFIX.json next to the existing files, for a new shape",
    )
    parser.add_argument(
        "--overwrite",
        metavar="REASON",
        default=None,
        help="replace files that exist and print REASON next to each; without it "
        "an existing file is refused",
    )
    return parser.parse_args(argv)


def target_path(file_name: str, suffix: str) -> Path:
    """Return where a scenario's recording goes, with the suffix before the extension."""
    stem, _, extension = file_name.rpartition(".")
    return HISTORIES / (f"{stem}-{suffix}.{extension}" if suffix else file_name)


async def main(argv: list[str] | None = None) -> int:
    """Record the scenarios; return 1 when an existing file was refused."""
    args = parse_args(argv)
    # Constructing the in-process MCP server installs a rich logging handler,
    # whose lazy imports inside the workflow sandbox trip the deadlock detector.
    logging.basicConfig(level=logging.WARNING, force=True)
    HISTORIES.mkdir(exist_ok=True)
    refused: list[Path] = []
    targets: dict[Path, Callable[[Client], Awaitable[WorkflowHistory]]] = {}
    for file_name, record in RECORDERS.items():
        path = target_path(file_name, args.suffix)
        if path.exists() and args.overwrite is None:
            print(
                f"refusing to overwrite {path}: a recording made before a release "
                "stays as it is (tests/histories/README.md); pass --suffix for a new "
                'file next to it, or --overwrite "<reason>" to replace it'
            )
            refused.append(path)
            continue
        targets[path] = record
    if not targets:
        return 1 if refused else 0
    env = await WorkflowEnvironment.start_local(
        dev_server_download_version=DEV_SERVER_DOWNLOAD_VERSION
    )
    try:
        for path, record in targets.items():
            history = await asyncio.wait_for(
                record(env.client), timeout=timedelta(minutes=2).total_seconds()
            )
            existed = path.exists()
            path.write_text(
                json.dumps(json.loads(history.to_json()), indent=2), encoding="utf-8"
            )
            print(f"overwrote {path}: {args.overwrite}" if existed else f"wrote {path}")
    finally:
        await env.shutdown()
    return 1 if refused else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
