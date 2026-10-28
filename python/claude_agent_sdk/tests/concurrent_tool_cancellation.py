"""A single-scenario workflow exercising the real concurrent tool cleanup path."""

from __future__ import annotations

import asyncio
from datetime import timedelta

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from temporalio.claude_agent_sdk._models import ClaudeToolRequest
    from temporalio.claude_agent_sdk._session import ClaudeSession, _ToolBridge


@workflow.defn
class ConcurrentToolCancellation:
    """Each cancelled tool schedules a distinct cleanup Activity command."""

    @workflow.run
    async def run(self) -> None:
        started = 0

        async def tool(index: int) -> None:
            nonlocal started
            started += 1
            try:
                await workflow.wait_condition(lambda: False)
            finally:
                await workflow.execute_activity(
                    f"cleanup_{index}",
                    index,
                    start_to_close_timeout=timedelta(seconds=5),
                )

        bridge = _ToolBridge(
            "token", {"tool": tool}, 8, ClaudeSession("session"), "message"
        )
        for index in range(6):
            bridge.submit(
                ClaudeToolRequest(
                    token="token",
                    tool_use_id=f"tool-{index}",
                    name="tool",
                    arguments={"index": index},
                )
            )
        await workflow.wait_condition(lambda: started == 6)
        tasks = tuple(bridge._tasks)
        bridge.close()
        await asyncio.gather(*tasks, return_exceptions=True)
