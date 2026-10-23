"""Starts an AgentWorkflow conversation and sends it each line you type: ``python -m samples.run``.

Start the Worker first with ``python -m samples.worker``.
"""

import asyncio
import json
import uuid

from samples.worker import TASK_QUEUE
from samples.workflow import AgentWorkflow, ToolApproval
from temporalio.client import Client, WorkflowHandle


async def approve_tools(handle: WorkflowHandle[AgentWorkflow, None]) -> None:
    """Ask whether to run each tool call that waits for approval."""
    while True:
        for request in await handle.query(AgentWorkflow.pending_approvals):
            question = (
                f"Run {request.tool_name} {json.dumps(request.tool_input)}? [y/N] "
            )
            answer = await asyncio.to_thread(input, question)
            approval = ToolApproval(request.tool_use_id, answer.strip().lower() == "y")
            await handle.execute_update(AgentWorkflow.approve_tool, approval)
        await asyncio.sleep(0.5)


async def main() -> None:
    client = await Client.connect("localhost:7233")
    handle = await client.start_workflow(
        AgentWorkflow.run, id=f"agent-{uuid.uuid4().hex[:8]}", task_queue=TASK_QUEUE
    )
    approving = asyncio.create_task(approve_tools(handle))
    while message := await asyncio.to_thread(input, "You: "):
        answer = await handle.execute_update(AgentWorkflow.send_message, message)
        print(f"Claude: {answer}")
    approving.cancel()
    await handle.execute_update(AgentWorkflow.end_conversation)
    await handle.result()


if __name__ == "__main__":
    asyncio.run(main())
