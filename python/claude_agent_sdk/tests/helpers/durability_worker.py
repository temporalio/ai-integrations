"""One worker process for the cross-worker durability tests. Run as ``python -m``."""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any, cast

from temporalio.claude_agent_sdk import (
    ClaudeAgentPlugin,
    FileLeaseStore,
    FileSessionStore,
    _native,
)
from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.worker import Worker
from tests.coding_assistant.manager import CodingAssistant
from tests.coding_assistant.tracker import TicketTracker
from tests.helpers import workflows


async def main() -> None:
    spec = json.loads(sys.argv[1])
    client = await Client.connect(
        spec["target"],
        namespace=spec["namespace"],
        data_converter=pydantic_data_converter,
    )
    name = spec.get("workflow", "ClaudeDurable")
    # "tickets" runs the example coding assistant, with its tracker and native tools.
    tracker = TicketTracker(Path(spec["tickets"])) if "tickets" in spec else None
    if tracker is not None:
        # What conftest's ``verified`` does in-process: the fake CLI never starts a sandbox.
        _native.sandbox_backend_available = lambda: True
        _native.seccomp_requirement = lambda: None
    async with Worker(
        client,
        task_queue=spec["task_queue"],
        workflows=[CodingAssistant if tracker else getattr(workflows, name)],
        activities=[tracker.fetch_ticket, tracker.add_comment] if tracker else [],
        max_cached_workflows=0,
        plugins=[
            ClaudeAgentPlugin(
                # None runs the CLI bundled in the SDK wheel.
                cli_path=spec.get("cli"),
                workspace_root=spec["workspace"],
                session_store=cast(Any, FileSessionStore(spec["store"])),
                lease_store=FileLeaseStore(spec["leases"]),
                disconnect_timeout=timedelta(seconds=1),
                enable_native_tools=tracker is not None,
            )
        ],
    ):
        Path(spec["ready"]).write_text("up")
        await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())
