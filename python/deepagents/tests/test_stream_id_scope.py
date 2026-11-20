"""Streaming IDs stay local to configured workers and writable handlers."""

import asyncio
from uuid import uuid4

import pytest

from temporalio import workflow
from temporalio.client import Client
from temporalio.deepagents import DeepAgentsPlugin
from temporalio.deepagents._event_loop import install_workflow_run_ids
from temporalio.worker import Replayer, Worker, WorkflowInterceptorClassInput
from temporalio.worker.workflow_sandbox import (
    SandboxedWorkflowRunner,
    SandboxRestrictions,
)

with workflow.unsafe.imports_passed_through():
    from langchain_core.callbacks import manager
    from langchain_core.runnables import RunnableLambda


def echo(value: str) -> str:
    return value


@workflow.defn
class RunIdScopeWorkflow:
    def __init__(self) -> None:
        self.done = False

    @workflow.query
    def read(self) -> str:
        return RunnableLambda(echo).invoke("query")

    @workflow.update
    async def finish(self) -> None:
        self.done = True

    @finish.validator
    def validate_finish(self) -> None:
        assert RunnableLambda(echo).invoke("validator") == "validator"

    @workflow.run
    async def run(self) -> int | None:
        version = getattr(manager, "uuid7")().version
        await workflow.wait_condition(lambda: self.done)
        return version


@pytest.mark.parametrize("configured", [True, False])
async def test_read_only_and_unconfigured_workers(
    client: Client, configured: bool
) -> None:
    # A separate configured worker can install the shim in this process first.
    install_workflow_run_ids(WorkflowInterceptorClassInput(unsafe_extern_functions={}))
    plugin = DeepAgentsPlugin()
    queue = f"id-scope-{uuid4()}"
    runner = SandboxedWorkflowRunner(
        restrictions=SandboxRestrictions.default.with_passthrough_modules(
            "langchain_core", "langsmith"
        )
    )
    async with Worker(
        client,
        task_queue=queue,
        workflows=[RunIdScopeWorkflow],
        plugins=[plugin] if configured else [],
        workflow_runner=runner,
    ):
        handle = await client.start_workflow(
            RunIdScopeWorkflow.run,
            id=str(uuid4()),
            task_queue=queue,
        )
        try:
            assert (
                await asyncio.wait_for(handle.query(RunIdScopeWorkflow.read), 5)
                == "query"
            )
            await asyncio.wait_for(handle.execute_update(RunIdScopeWorkflow.finish), 5)
            assert await asyncio.wait_for(handle.result(), 5) == (
                4 if configured else 7
            )
            history = await handle.fetch_history()
        except BaseException:
            await handle.terminate()
            raise
    if not configured:
        assert "langchain-replay-stable-run-ids" not in history.to_json()
    await Replayer(
        workflows=[RunIdScopeWorkflow],
        plugins=[plugin] if configured else [],
        workflow_runner=runner,
    ).replay_workflow(history)
