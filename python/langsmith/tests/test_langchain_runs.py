"""LangChain runs in workflows keep their IDs and times across replay."""

from __future__ import annotations

import uuid
from collections import Counter
from datetime import timedelta

from langchain_core.runnables import RunnableLambda

from temporalio import activity, workflow
from temporalio.client import Client
from temporalio.langsmith import LangSmithPlugin
from temporalio.testing import WorkflowEnvironment
from tests.helpers import new_worker
from tests.test_integration import _make_client_and_collector, _wait_for_tracing_io


@activity.defn
async def shout(text: str) -> str:
    return text.upper()


@workflow.defn
class LangChainWorkflow:
    @workflow.run
    async def run(self) -> str:
        async def prepare(text: str) -> str:
            return text

        async def call_activity(text: str) -> str:
            return await workflow.execute_activity(
                shout, text, start_to_close_timeout=timedelta(seconds=10)
            )

        chain = RunnableLambda(prepare, name="prepare") | RunnableLambda(
            call_activity, name="call_activity"
        )
        return await chain.ainvoke("hello")


async def test_open_langchain_runs_end_under_their_posted_ids(
    client: Client,
    env: WorkflowEnvironment,  # type: ignore[reportUnusedParameter]
) -> None:
    """A LangChain run open across a replay is ended under the ID it was posted with."""
    temporal_client, _, mock = _make_client_and_collector(
        client, add_temporal_runs=False
    )
    task_queue = f"langchain-runs-{uuid.uuid4()}"
    async with new_worker(
        temporal_client,
        LangChainWorkflow,
        activities=[shout],
        task_queue=task_queue,
        max_cached_workflows=0,
    ):
        result = await temporal_client.execute_workflow(
            LangChainWorkflow.run,
            id=f"langchain-runs-{uuid.uuid4()}",
            task_queue=task_queue,
        )
    assert result == "HELLO"

    plugins = temporal_client.config()["plugins"]
    assert plugins
    plugin = plugins[0]
    assert isinstance(plugin, LangSmithPlugin)
    _wait_for_tracing_io(plugin)

    posts = [call.kwargs for call in mock.create_run.call_args_list]
    assert Counter(post["name"] for post in posts) == {
        "RunnableSequence": 1,
        "prepare": 1,
        "call_activity": 1,
    }
    posted = {str(post["id"]) for post in posts}
    patched = {str(call.kwargs["run_id"]) for call in mock.update_run.call_args_list}
    assert patched == posted
