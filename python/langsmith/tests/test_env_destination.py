"""Tests for where runs go under LangSmith's env destination settings.

LangSmith's env defaults (``LANGSMITH_PROJECT``, ``LANGSMITH_AGENT_ID`` and
``LANGSMITH_AGENT_ENVIRONMENT``) name where a run that names no destination goes.
An env LangSmith can't use never breaks a workflow or splits a trace, and runs
below a parent run still join it.
"""

from __future__ import annotations

import uuid
from datetime import timedelta
from typing import Any
from unittest.mock import MagicMock

import pytest
from langsmith import Client as LangSmithClient
from langsmith import traceable, tracing_context
from langsmith.run_trees import RunTree
from langsmith.utils import LangSmithUserError

from temporalio import common, workflow
from temporalio.client import Client
from temporalio.langsmith._interceptor import (
    LangSmithInterceptor,
    _env_destination,
    _RootReplaySafeRunTreeFactory,
)
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


def _env_address() -> Any:
    """The address the current env names, in the installed LangSmith's own form."""
    address = getattr(RunTree(name="env-probe"), "address", None)
    if address is None:
        pytest.skip("This LangSmith version has no agent addresses.")
    return address


async def _run_workflow(
    client: Client,
    env: WorkflowEnvironment,
    caller_project: str | None = None,
    **plugin_kwargs: Any,
) -> MagicMock:
    """Run EnvDestinationWorkflow, under a caller run in ``caller_project`` if set."""
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

        async def execute() -> str:
            return await temporal_client.execute_workflow(
                EnvDestinationWorkflow.run,
                id=f"env-destination-{uuid.uuid4()}",
                task_queue=worker.task_queue,
            )

        if caller_project is None:
            result = await execute()
        else:
            with tracing_context(client=mock_ls_client, enabled=True):
                result = await traceable(name="caller", project_name=caller_project)(
                    execute
                )()
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


_USER_RUNS = {
    "outer_chain",
    "inner_llm_call",
    "sync_outer_chain",
    "sync_inner_llm_call",
    "nested_traceable_activity",
    "nexus_direct_traceable",
}


@pytest.mark.requires_local_server
@pytest.mark.parametrize("add_temporal_runs", [False, True])
@pytest.mark.parametrize(
    "destination_env",
    [{"LANGSMITH_PROJECT": "env-project"}, _VALID_AGENT_ENV],
    ids=["project", "agent"],
)
async def test_env_destination_receives_every_run(
    client: Client,
    env: WorkflowEnvironment,
    monkeypatch: pytest.MonkeyPatch,
    destination_env: dict[str, str],
    add_temporal_runs: bool,
) -> None:
    if env.supports_time_skipping:
        pytest.skip("Time-skipping server doesn't persist headers.")
    _set_destination_env(monkeypatch, destination_env)
    expected = (
        ("env-project", None)
        if "LANGSMITH_PROJECT" in destination_env
        else (None, _env_address())
    )

    mock_ls_client = await _run_workflow(
        client, env, add_temporal_runs=add_temporal_runs
    )

    names = {call.kwargs["name"] for call in mock_ls_client.create_run.call_args_list}
    assert _USER_RUNS <= names
    for trace_id, destinations in _destinations_by_trace(mock_ls_client).items():
        assert all(d == expected for d in destinations), (
            f"Trace {trace_id} left {expected}: {destinations}"
        )


@pytest.mark.requires_local_server
@pytest.mark.parametrize("add_temporal_runs", [False, True])
@pytest.mark.parametrize("agent_env", _BAD_AGENT_ENVS.values(), ids=_BAD_AGENT_ENVS)
async def test_bad_agent_env_runs_under_a_parent_join_it(
    client: Client,
    env: WorkflowEnvironment,
    monkeypatch: pytest.MonkeyPatch,
    agent_env: dict[str, str],
    add_temporal_runs: bool,
) -> None:
    if env.supports_time_skipping:
        pytest.skip("Time-skipping server doesn't persist headers.")
    _set_destination_env(monkeypatch, agent_env)
    if _env_destination(LangSmithClient(api_key="test")) is not None:
        pytest.skip("This LangSmith version can address runs from this env.")

    mock_ls_client = await _run_workflow(
        client,
        env,
        caller_project="caller-project",
        add_temporal_runs=add_temporal_runs,
    )

    # The plugin has no destination of its own, so it creates no runs; user
    # runs join the caller's trace and project.
    names = {call.kwargs["name"] for call in mock_ls_client.create_run.call_args_list}
    assert names == {"caller", *_USER_RUNS}
    by_trace = _destinations_by_trace(mock_ls_client)
    assert len(by_trace) == 1
    assert all(d == ("caller-project", None) for d in next(iter(by_trace.values())))


@pytest.mark.parametrize(
    ("destination_env", "expected"),
    [
        ({}, {"project_name": "default"}),
        ({"LANGSMITH_PROJECT": "env-project"}, {"project_name": "env-project"}),
    ],
    ids=["unset", "project"],
)
def test_env_destination_names_a_project(
    monkeypatch: pytest.MonkeyPatch,
    destination_env: dict[str, str],
    expected: dict[str, Any],
) -> None:
    _set_destination_env(monkeypatch, destination_env)
    assert _env_destination(LangSmithClient(api_key="test")) == expected


def test_env_destination_names_an_agent_address(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_destination_env(monkeypatch, _VALID_AGENT_ENV)
    assert _env_destination(LangSmithClient(api_key="test")) == {
        "address": _env_address()
    }


@pytest.mark.parametrize("agent_env", _BAD_AGENT_ENVS.values(), ids=_BAD_AGENT_ENVS)
def test_env_destination_is_none_for_a_bad_env(
    monkeypatch: pytest.MonkeyPatch, agent_env: dict[str, str]
) -> None:
    _set_destination_env(monkeypatch, agent_env)
    try:
        RunTree(name="env-probe")
    except LangSmithUserError:
        pass
    else:
        pytest.skip("This LangSmith version can address runs from this env.")
    assert _env_destination(LangSmithClient(api_key="test")) is None


@pytest.mark.parametrize("agent_env", _BAD_AGENT_ENVS.values(), ids=_BAD_AGENT_ENVS)
def test_empty_project_name_is_unset(
    monkeypatch: pytest.MonkeyPatch, agent_env: dict[str, str]
) -> None:
    _set_destination_env(monkeypatch, agent_env)
    ls_client = MagicMock()
    interceptor = LangSmithInterceptor(
        client=ls_client, project_name="", add_temporal_runs=True
    )

    assert interceptor._destination == _env_destination(ls_client)
    with interceptor.maybe_run("StartWorkflow:EmptyProjectName"):
        pass


def test_root_factory_children_inherit_an_address(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_destination_env(monkeypatch, _VALID_AGENT_ENV)
    address = _env_address()
    _set_destination_env(monkeypatch, {"LANGSMITH_PROJECT": "env-project"})
    factory = _RootReplaySafeRunTreeFactory(
        ls_client=MagicMock(),
        executor=MagicMock(),
        destination={"address": address},
    )

    child = factory.create_child(name="traceable-fn", run_type="chain")

    assert getattr(child, "address", None) == address
    assert child.session_name is None
