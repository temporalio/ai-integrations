"""Tests that LangSmith's env destination settings never break workflows or split a trace."""

from __future__ import annotations

import uuid
from datetime import timedelta
from typing import Any
from unittest.mock import MagicMock

import pytest
from langsmith import traceable
from langsmith.run_trees import RunTree
from langsmith.utils import LangSmithUserError

from temporalio import common, workflow
from temporalio.client import Client
from temporalio.testing import WorkflowEnvironment
from tests.helpers import new_worker
from tests.helpers.nexus import make_nexus_endpoint_name
from tests.test_integration import (
    DirectTraceableNexusService,
    _make_client_and_collector,
    _outer_chain,
    _sync_outer_chain,
    nested_traceable_activity,
)

_BAD_AGENT_ENVS = {
    "half-set": {"LANGSMITH_AGENT_ID": "support"},
    "invalid": {
        "LANGSMITH_AGENT_ID": "Not_A_Label",
        "LANGSMITH_AGENT_ENVIRONMENT": "staging",
    },
    "beside-project": {
        "LANGSMITH_PROJECT": "env-project",
        "LANGSMITH_AGENT_ID": "support",
        "LANGSMITH_AGENT_ENVIRONMENT": "staging",
    },
}
_VALID_AGENT_ENV = {
    "LANGSMITH_AGENT_ID": "support",
    "LANGSMITH_AGENT_ENVIRONMENT": "staging",
}


@workflow.defn
class EnvDestinationWorkflow:
    @workflow.run
    async def run(self) -> str:
        await _outer_chain("async")
        _sync_outer_chain("sync")
        await workflow.execute_activity(
            nested_traceable_activity,
            start_to_close_timeout=timedelta(minutes=1),
            retry_policy=common.RetryPolicy(maximum_attempts=1),
        )
        nexus_client = workflow.create_nexus_client(
            endpoint=make_nexus_endpoint_name(workflow.info().task_queue),
            service=DirectTraceableNexusService,
        )
        return await nexus_client.execute_operation(
            operation=DirectTraceableNexusService.direct_traceable_op,
            input="nexus-input",
        )


@traceable(name="own_project_call", project_name="own-project")
async def _own_project_call() -> str:
    return "own"


@workflow.defn
class OwnProjectTraceableWorkflow:
    @workflow.run
    async def run(self) -> str:
        first = await _own_project_call()
        await workflow.sleep(timedelta(milliseconds=10))
        return first + await _own_project_call()


def _set_destination_env(monkeypatch: pytest.MonkeyPatch, env: dict[str, str]) -> None:
    for name in (
        "LANGSMITH_PROJECT",
        "LANGSMITH_AGENT_ID",
        "LANGSMITH_AGENT_ENVIRONMENT",
    ):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)


async def _run_workflow(
    client: Client, env: WorkflowEnvironment, **plugin_kwargs: Any
) -> MagicMock:
    temporal_client, _, mock_ls_client = _make_client_and_collector(
        client, **plugin_kwargs
    )
    async with new_worker(
        temporal_client,
        EnvDestinationWorkflow,
        activities=[nested_traceable_activity],
        nexus_service_handlers=[DirectTraceableNexusService()],
        max_cached_workflows=0,
    ) as worker:
        await env.create_nexus_endpoint(
            make_nexus_endpoint_name(worker.task_queue), worker.task_queue
        )
        result = await temporal_client.execute_workflow(
            EnvDestinationWorkflow.run,
            id=f"env-destination-{uuid.uuid4()}",
            task_queue=worker.task_queue,
        )
    assert result == "response to: nexus-input"
    return mock_ls_client


def _destinations_by_trace(
    mock_ls_client: MagicMock,
) -> dict[str, list[tuple[Any, Any]]]:
    by_trace: dict[str, list[tuple[Any, Any]]] = {}
    for call in mock_ls_client.create_run.call_args_list:
        by_trace.setdefault(str(call.kwargs["trace_id"]), []).append(
            (call.kwargs.get("session_name"), call.kwargs.get("address"))
        )
    return by_trace


@pytest.mark.requires_local_server
@pytest.mark.parametrize("add_temporal_runs", [False, True])
@pytest.mark.parametrize("agent_env", _BAD_AGENT_ENVS.values(), ids=_BAD_AGENT_ENVS)
async def test_bad_agent_env_does_not_break_workflows(
    client: Client,
    env: WorkflowEnvironment,
    monkeypatch: pytest.MonkeyPatch,
    agent_env: dict[str, str],
    add_temporal_runs: bool,
) -> None:
    if env.supports_time_skipping:
        pytest.skip("Time-skipping server doesn't persist headers.")
    _set_destination_env(monkeypatch, agent_env)

    mock_ls_client = await _run_workflow(
        client, env, add_temporal_runs=add_temporal_runs
    )

    for trace_id, destinations in _destinations_by_trace(mock_ls_client).items():
        assert all(d == destinations[0] for d in destinations), (
            f"Trace {trace_id} spans destinations: {destinations}"
        )


@pytest.mark.parametrize("add_temporal_runs", [False, True])
@pytest.mark.parametrize("agent_env", _BAD_AGENT_ENVS.values(), ids=_BAD_AGENT_ENVS)
async def test_bad_agent_env_leaves_workflow_code_untraced(
    client: Client,
    env: WorkflowEnvironment,  # type:ignore[reportUnusedParameter]
    monkeypatch: pytest.MonkeyPatch,
    agent_env: dict[str, str],
    add_temporal_runs: bool,
) -> None:
    _set_destination_env(monkeypatch, agent_env)
    try:
        RunTree(name="env-probe")
    except LangSmithUserError:
        pass
    else:
        pytest.skip("This LangSmith version can address runs from this env.")
    temporal_client, _, mock_ls_client = _make_client_and_collector(
        client, add_temporal_runs=add_temporal_runs
    )

    async with new_worker(
        temporal_client, OwnProjectTraceableWorkflow, max_cached_workflows=0
    ) as worker:
        result = await temporal_client.execute_workflow(
            OwnProjectTraceableWorkflow.run,
            id=f"env-untraced-{uuid.uuid4()}",
            task_queue=worker.task_queue,
        )

    assert result == "ownown"
    assert mock_ls_client.create_run.call_args_list == []


@pytest.mark.requires_local_server
@pytest.mark.parametrize(
    "agent_env",
    [*_BAD_AGENT_ENVS.values(), _VALID_AGENT_ENV],
    ids=[*_BAD_AGENT_ENVS, "valid"],
)
async def test_plugin_project_outranks_env_destination(
    client: Client,
    env: WorkflowEnvironment,
    monkeypatch: pytest.MonkeyPatch,
    agent_env: dict[str, str],
) -> None:
    if env.supports_time_skipping:
        pytest.skip("Time-skipping server doesn't persist headers.")
    _set_destination_env(monkeypatch, agent_env)

    mock_ls_client = await _run_workflow(
        client,
        env,
        add_temporal_runs=False,
        project_name="plugin-project",
    )

    by_trace = _destinations_by_trace(mock_ls_client)
    assert by_trace
    for trace_id, destinations in by_trace.items():
        assert all(d == ("plugin-project", None) for d in destinations), (
            f"Trace {trace_id} left the plugin's project: {destinations}"
        )
