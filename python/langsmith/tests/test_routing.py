"""Agent destinations and real LangSmith replica serialization across Temporal."""

from __future__ import annotations

import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock

import langsmith
import pytest
from langsmith import Client as LangSmithClient
from langsmith import traceable, tracing_context
from langsmith.run_trees import RunTree, WriteReplica
from langsmith.utils import LangSmithUserError

from temporalio import workflow
from temporalio.client import Client
from temporalio.langsmith import LangSmithInterceptor, LangSmithPlugin
from temporalio.langsmith._interceptor import (
    HEADER_KEY,
    _extract_context,
    _inject_context,
    _install_root_factory_override,
    _RootReplaySafeRunTreeFactory,
)
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Replayer
from tests.helpers import new_worker
from tests.helpers.langsmith import InMemoryRunCollector, make_mock_ls_client
from tests.helpers.nexus import make_nexus_endpoint_name
from tests.test_integration import (
    DirectTraceableNexusService,
    NexusDirectTraceableWorkflow,
    TraceableActivityWorkflow,
    traceable_activity,
)

# Older supported LangSmith versions can still use project-based tracing.
pytest.importorskip("langsmith.address")

ADDRESS = "lrn:agents/support/environments/staging"
OTHER_ADDRESS = "lrn:agents/support/environments/production"
REPLICAS: list[WriteReplica] = [
    {"project_name": "primary", "primary": True},
    {"project_name": "default"},
]


@traceable(name="routing-step")
async def _routing_step() -> str:
    await workflow.execute_activity(
        traceable_activity, start_to_close_timeout=timedelta(seconds=30)
    )
    await workflow.execute_child_workflow(
        TraceableActivityWorkflow.run, id=f"child-{workflow.info().workflow_id}"
    )
    return await workflow.execute_child_workflow(
        NexusDirectTraceableWorkflow.run, id=f"nexus-{workflow.info().workflow_id}"
    )


@traceable(name="addressed-step", address=ADDRESS)
async def _addressed_step() -> str:
    return await _routing_step()


@workflow.defn
class RoutingWorkflow:
    @workflow.run
    async def run(self, source: str) -> str:
        if source == "decorator":
            return await _addressed_step()
        if source == "context":
            with tracing_context(address=ADDRESS):
                return await _routing_step()
        if source.startswith("replicas"):
            replicas: list[WriteReplica] = (
                REPLICAS
                if source == "replicas"
                else [REPLICAS[0], {"address": OTHER_ADDRESS}]
            )
            with tracing_context(replicas=replicas):
                return await _routing_step()
        return await _routing_step()


@pytest.mark.parametrize("constructor", [LangSmithPlugin, LangSmithInterceptor])
def test_conflicting_destinations(constructor: Any) -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        constructor(client=MagicMock(), project_name="project", address=ADDRESS)


def test_address_validation() -> None:
    with pytest.raises(LangSmithUserError, match="address"):
        LangSmithPlugin(client=MagicMock(), address="not-an-address")
    plugin = LangSmithPlugin(
        client=MagicMock(),
        address=ADDRESS.replace("staging", "STAGING"),
    )
    assert plugin.interceptors
    assert isinstance(plugin.interceptors[0], LangSmithInterceptor)
    assert plugin.interceptors[0]._address == ADDRESS


def test_address_requires_supported_langsmith(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(langsmith, "address", None)
    with pytest.raises(ValueError, match="langsmith>=0.14.4"):
        LangSmithPlugin(client=MagicMock(), address=ADDRESS)
    LangSmithPlugin(client=MagicMock(), project_name="legacy-project")


@pytest.mark.parametrize("source", ["decorator", "context", "extra", "env"])
def test_native_root_destination(source: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Sync traceables retain native destination precedence through the factory."""
    _install_root_factory_override()
    client = make_mock_ls_client(InMemoryRunCollector())
    if source == "env":
        monkeypatch.setenv("LANGSMITH_AGENT_ID", "support")
        monkeypatch.setenv("LANGSMITH_AGENT_ENVIRONMENT", "staging")
    with ThreadPoolExecutor(max_workers=1) as executor:
        factory = _RootReplaySafeRunTreeFactory(ls_client=client, executor=executor)

        @traceable(address=ADDRESS if source == "decorator" else None)
        def traced() -> str:
            return "done"

        with tracing_context(
            parent=factory,
            client=client,
            address=ADDRESS if source == "context" else None,
        ):
            assert (
                traced(  # type: ignore[call-arg]  # LangSmith adds this argument.
                    langsmith_extra={"address": ADDRESS} if source == "extra" else {}
                )
                == "done"
            )
    assert client.create_run.call_count == 1
    assert client.create_run.call_args.kwargs["address"] == ADDRESS
    assert client.create_run.call_args.kwargs.get("session_name") is None
    assert client.create_run.call_args.kwargs.get("parent_run_id") is None


def test_factory_explicit_address_overrides_default_project() -> None:
    """An explicit root address must not be combined with the factory's default."""
    with ThreadPoolExecutor(max_workers=1) as executor:
        factory = _RootReplaySafeRunTreeFactory(
            ls_client=MagicMock(), executor=executor
        )
        child = factory.create_child(name="root", run_type="chain", address=ADDRESS)
    assert child.address == ADDRESS
    assert child.session_name is None


def test_native_project_context_overrides_agent_factory() -> None:
    """An explicit project context retains LangSmith's native precedence."""
    _install_root_factory_override()
    client = make_mock_ls_client(InMemoryRunCollector())
    with ThreadPoolExecutor(max_workers=1) as executor:
        factory = _RootReplaySafeRunTreeFactory(
            ls_client=client, executor=executor, destination={"address": ADDRESS}
        )

        @traceable(address=OTHER_ADDRESS)
        def traced() -> str:
            return "done"

        with tracing_context(parent=factory, client=client, project_name="evaluation"):
            assert traced() == "done"
    assert client.create_run.call_args.kwargs["session_name"] == "evaluation"
    assert client.create_run.call_args.kwargs.get("address") is None


def _wait_for_io(plugin: LangSmithPlugin) -> None:
    assert plugin.interceptors
    interceptor = plugin.interceptors[0]
    assert isinstance(interceptor, LangSmithInterceptor)
    interceptor._executor.submit(lambda: None).result()


async def _execute(
    client: Client,
    env: WorkflowEnvironment,
    plugin: LangSmithPlugin,
    source: str,
    caller_context: dict[str, Any] | None = None,
) -> tuple[Any, list[tuple[str, RunTree]]]:
    """Run with clean worker context across activity, child, and Nexus boundaries."""
    task_queue = f"routing-{uuid.uuid4()}"
    config = client.config()
    config["plugins"] = [plugin]
    temporal_client = Client(**config)
    originals: list[tuple[str, RunTree]] = []
    assert plugin.interceptors
    interceptor = plugin.interceptors[0]
    assert isinstance(interceptor, LangSmithInterceptor)

    @traceable(name="caller")
    async def caller(run_tree: RunTree) -> Any:
        originals.append(("caller", run_tree))
        return await temporal_client.start_workflow(
            RoutingWorkflow.run,
            source,
            id=f"routing-{uuid.uuid4()}",
            task_queue=task_queue,
        )

    async with new_worker(
        temporal_client,
        RoutingWorkflow,
        TraceableActivityWorkflow,
        NexusDirectTraceableWorkflow,
        activities=[traceable_activity],
        nexus_service_handlers=[DirectTraceableNexusService()],
        task_queue=task_queue,
        max_cached_workflows=0,
        workflow_failure_exception_types=[Exception],
    ) as worker:
        await env.create_nexus_endpoint(
            make_nexus_endpoint_name(worker.task_queue), worker.task_queue
        )
        if caller_context is not None:
            with tracing_context(client=interceptor._client, **caller_context):
                handle = await caller()
        else:
            handle = await temporal_client.start_workflow(
                RoutingWorkflow.run,
                source,
                id=f"routing-{uuid.uuid4()}",
                task_queue=task_queue,
            )
        assert await handle.result() == "response to: nexus-input"
    _wait_for_io(plugin)
    return handle, originals


async def _assert_replay_is_silent(
    handle: Any, plugin: LangSmithPlugin, mock: MagicMock, client: Client
) -> None:
    before = (mock.create_run.call_count, mock.update_run.call_count)
    histories = [await handle.fetch_history()]
    for prefix in ("child-", "nexus-"):
        histories.append(
            await client.get_workflow_handle(prefix + handle.id).fetch_history()
        )
    replayer = Replayer(
        workflows=[
            RoutingWorkflow,
            TraceableActivityWorkflow,
            NexusDirectTraceableWorkflow,
        ],
        plugins=[plugin],
    )
    for history in histories:
        await replayer.replay_workflow(history)
    _wait_for_io(plugin)
    assert (mock.create_run.call_count, mock.update_run.call_count) == before


@pytest.mark.parametrize(
    "source,add_temporal_runs",
    [
        ("plugin", False),
        ("plugin", True),
        ("decorator", False),
        ("context", False),
        ("headers", False),
        ("headers", True),
    ],
)
async def test_agent_destination_across_temporal(
    client: Client, env: WorkflowEnvironment, source: str, add_temporal_runs: bool
) -> None:
    """Destination and parent links survive activities, child workflows, and Nexus."""
    mock = make_mock_ls_client(InMemoryRunCollector())
    plugin = LangSmithPlugin(
        client=mock,
        address=ADDRESS if source == "plugin" else None,
        project_name="worker-fallback" if source == "headers" else None,
        add_temporal_runs=add_temporal_runs,
    )
    handle, _ = await _execute(
        client,
        env,
        plugin,
        source,
        {"address": ADDRESS, "enabled": True} if source == "headers" else None,
    )
    creates = [call.kwargs for call in mock.create_run.call_args_list]
    assert {run["name"] for run in creates} >= {
        "routing-step",
        "traceable_activity",
        "nexus_direct_traceable",
    }
    assert all(run.get("address") == ADDRESS for run in creates)
    assert all(run.get("session_name") is None for run in creates)
    ids = {str(run["id"]) for run in creates}
    assert len(ids) == len(creates)
    assert all(
        not run.get("parent_run_id") or str(run["parent_run_id"]) in ids
        for run in creates
    )
    updates = [call.kwargs for call in mock.update_run.call_args_list]
    assert {str(run["run_id"]) for run in updates} == ids
    assert all(
        run.get("address") == ADDRESS and run.get("end_time") is not None
        for run in updates
    )
    await _assert_replay_is_silent(handle, plugin, mock, client)


@pytest.mark.parametrize(
    "source,add_temporal_runs",
    [("headers", False), ("headers", True), ("replicas", False)],
)
@pytest.mark.parametrize("secondary_address", [None, OTHER_ADDRESS])
async def test_replica_ids_and_updates_across_temporal(
    client: Client,
    env: WorkflowEnvironment,
    source: str,
    add_temporal_runs: bool,
    secondary_address: str | None,
) -> None:
    """Only the primary keeps original IDs; each complete tree gets matching updates."""
    mock = make_mock_ls_client(InMemoryRunCollector())
    plugin = LangSmithPlugin(client=mock, add_temporal_runs=add_temporal_runs)
    replicas: list[WriteReplica] = (
        REPLICAS
        if secondary_address is None
        else [REPLICAS[0], {"address": secondary_address}]
    )
    handle, originals = await _execute(
        client,
        env,
        plugin,
        source + "-agent" if source == "replicas" and secondary_address else source,
        {"replicas": replicas, "enabled": True} if source == "headers" else None,
    )
    creates = [call.kwargs for call in mock.create_run.call_args_list]
    updates = [call.kwargs for call in mock.update_run.call_args_list]
    secondary = secondary_address or "default"
    destinations = {run.get("address") or run["session_name"] for run in creates}
    assert destinations == {"primary", secondary}
    by_destination = {
        destination: {
            str(run["id"]): run
            for run in creates
            if (run.get("address") or run["session_name"]) == destination
        }
        for destination in destinations
    }
    assert len(by_destination["primary"]) == len(by_destination[secondary])
    assert by_destination["primary"].keys().isdisjoint(by_destination[secondary])
    assert len(creates) == sum(len(runs) for runs in by_destination.values())
    if originals:
        original_id = str(originals[0][1].id)
        assert original_id in by_destination["primary"]
        assert original_id not in by_destination[secondary]
    for destination, runs in by_destination.items():
        for run_id, run in runs.items():
            parent_id = run.get("parent_run_id")
            if parent_id:
                parent = runs[str(parent_id)]
                assert run["trace_id"] == parent["trace_id"]
                assert run["dotted_order"].startswith(parent["dotted_order"] + ".")
            else:
                assert str(run["trace_id"]) == run_id
            matching = [
                update
                for update in updates
                if (update.get("address") or update["session_name"]) == destination
                and str(update["run_id"]) == run_id
            ]
            assert len(matching) == 1
            assert matching[0]["parent_run_id"] == run.get("parent_run_id")
            assert matching[0]["trace_id"] == run["trace_id"]
            assert matching[0]["end_time"] is not None
            assert matching[0]["outputs"] is not None
    await _assert_replay_is_silent(handle, plugin, mock, client)


def test_replica_headers_exclude_credentials_and_keep_parent_time() -> None:
    """A restored parent keeps stable ordering and only safe replica routing."""
    client = make_mock_ls_client(InMemoryRunCollector())
    root = RunTree(
        name="root",
        ls_client=client,
        start_time=datetime(2025, 1, 1, tzinfo=timezone.utc),
        replicas=[
            {
                "project_name": "primary",
                "primary": True,
                "api_key": "must-not-be-recorded",
                "api_url": "https://private-ingestion.invalid",
                "auth": {"api_key": "also-private"},
                "client": MagicMock(spec=LangSmithClient),
                "updates": {
                    "metadata": {"team": "support"},
                    "outputs": {"secret": "private"},
                },
            },
            {"address": OTHER_ADDRESS},
        ],
    )
    headers = _inject_context({}, root)
    assert b"must-not-be-recorded" not in headers[HEADER_KEY].data
    assert b"also-private" not in headers[HEADER_KEY].data
    assert b"private-ingestion" not in headers[HEADER_KEY].data
    with ThreadPoolExecutor(max_workers=1) as executor:
        restored = _extract_context(headers, executor, client)
    assert restored is not None
    assert restored.start_time == root.start_time
    assert restored.replicas == [
        {
            "project_name": "primary",
            "primary": True,
            "updates": {"metadata": {"team": "support"}},
        },
        {"address": OTHER_ADDRESS},
    ]
