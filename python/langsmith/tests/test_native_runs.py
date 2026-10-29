"""Native LangSmith roots retain IDs, timestamps, and routing across replay."""

from __future__ import annotations

import uuid
from datetime import timedelta

from langsmith import get_current_run_tree, trace, traceable
from langsmith.run_trees import RunTree

from temporalio import workflow
from temporalio.client import Client
from temporalio.langsmith import LangSmithPlugin
from temporalio.langsmith._interceptor import LangSmithInterceptor
from temporalio.testing import WorkflowEnvironment
from tests.helpers import new_worker
from tests.test_integration import _make_client_and_collector


def _snapshot(run: RunTree) -> dict[str, str]:
    return {
        "id": str(run.id),
        "parent_id": str(run.parent_run_id or ""),
        "trace_id": str(run.trace_id),
        "dotted_order": run.dotted_order,
        "start_time": run.start_time.isoformat(),
    }


@traceable(name="native-child")
def _native_child() -> dict[str, str]:
    run = get_current_run_tree()
    assert run is not None
    return _snapshot(run)


@traceable(name="native-root")
async def _native_root() -> list[dict[str, str]]:
    run = get_current_run_tree()
    assert run is not None
    await workflow.sleep(timedelta(milliseconds=1))
    return [_snapshot(run), _native_child()]


@workflow.defn
class NativeRunTreeWorkflow:
    def __init__(self) -> None:
        self.snapshots: list[dict[str, str]] = []

    @workflow.run
    async def run(self) -> list[dict[str, str]]:
        self.snapshots = await _native_root()
        with trace("trace-root") as root:
            await workflow.sleep(timedelta(milliseconds=1))
            with trace("trace-child") as child:
                self.snapshots.extend([_snapshot(root), _snapshot(child)])
        async with trace("async-trace-root") as root:
            await workflow.sleep(timedelta(milliseconds=1))
            async with trace("async-trace-child") as child:
                self.snapshots.extend([_snapshot(root), _snapshot(child)])
        root = RunTree(name="direct-root")
        child = root.create_child(name="direct-child")
        root.post()
        child.post()
        child.end(outputs={"direct": "child"})
        child.patch()
        root.end(outputs={"direct": "root"})
        root.patch()
        self.snapshots.extend([_snapshot(root), _snapshot(child)])
        return self.snapshots

    @workflow.query
    def snapshot(self) -> list[dict[str, str]]:
        return self.snapshots


async def test_native_runs_are_stable_after_worker_replay(
    client: Client,
    env: WorkflowEnvironment,  # type: ignore[reportUnusedParameter]
) -> None:
    """A fresh worker rebuilds every native run identically and sends no duplicate I/O."""
    temporal_client, _, mock = _make_client_and_collector(
        client, add_temporal_runs=False
    )
    snapshots: list[dict[str, str]]
    task_queue = f"native-runs-{uuid.uuid4()}"
    async with new_worker(
        temporal_client, NativeRunTreeWorkflow, task_queue=task_queue
    ):
        handle = await temporal_client.start_workflow(
            NativeRunTreeWorkflow.run,
            id=f"native-runs-{uuid.uuid4()}",
            task_queue=task_queue,
        )
        snapshots = await handle.result()

    plugins = temporal_client.config()["plugins"]
    assert plugins
    plugin = plugins[0]
    assert isinstance(plugin, LangSmithPlugin)
    assert plugin.interceptors
    interceptor = plugin.interceptors[0]
    assert isinstance(interceptor, LangSmithInterceptor)
    interceptor._executor.submit(lambda: None).result()
    counts = (mock.create_run.call_count, mock.update_run.call_count)
    assert counts == (8, 8)
    assert len({run["id"] for run in snapshots}) == 8
    for root, child in zip(snapshots[::2], snapshots[1::2]):
        assert root["parent_id"] == ""
        assert root["trace_id"] == root["id"]
        assert child["parent_id"] == root["id"]
        assert child["trace_id"] == root["id"]
        assert child["dotted_order"].startswith(root["dotted_order"] + ".")

    async with new_worker(
        temporal_client, NativeRunTreeWorkflow, task_queue=task_queue
    ):
        assert await handle.query(NativeRunTreeWorkflow.snapshot) == snapshots

    interceptor._executor.submit(lambda: None).result()
    assert (mock.create_run.call_count, mock.update_run.call_count) == counts
