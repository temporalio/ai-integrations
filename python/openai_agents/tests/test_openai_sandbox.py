"""Tests for sandbox validation in TemporalOpenAIRunner."""

import asyncio
import dataclasses
import io
import json
import socket
import uuid
from collections.abc import Callable, Collection
from datetime import timedelta
from pathlib import Path
from typing import Any, Literal

import pytest
from agents import (
    Agent,
    FunctionTool,
    ModelResponse,
    RunConfig,
    Runner,
    RunState,
    Tool,
)
from agents.sandbox import Capability, Manifest, SandboxAgent, SandboxRunConfig
from agents.sandbox.entries import DockerVolumeMountStrategy, S3Mount
from agents.sandbox.errors import (
    ExecTransportError,
    SandboxError,
    WorkspaceArchiveReadError,
)
from agents.sandbox.manifest import Environment
from agents.sandbox.session.base_sandbox_session import BaseSandboxSession
from agents.sandbox.session.pty_types import PtyExecUpdate
from agents.sandbox.session.sandbox_client import (
    BaseSandboxClient,
    BaseSandboxClientOptions,
)
from agents.sandbox.session.sandbox_session import SandboxSession
from agents.sandbox.session.sandbox_session_state import SandboxSessionState
from agents.sandbox.snapshot import NoopSnapshot
from agents.sandbox.types import ExecResult
from agents.sandbox.workspace_paths import SandboxPathGrant
from pydantic import BaseModel, TypeAdapter
from pydantic_core import to_json

from temporalio import workflow
from temporalio.client import Client, WorkflowFailureError, WorkflowHandle
from temporalio.exceptions import ActivityError, ApplicationError
from temporalio.openai_agents import (
    AgentsWorkflowError,
    ModelActivityParameters,
    OnHostSandboxClientProvider,
    OpenAIAgentsPlugin,
    OpenAIPayloadConverter,
    SandboxClientProvider,
    TemporalWorkerEnvValue,
)
from temporalio.openai_agents._openai_runner import _has_sandbox_agent
from temporalio.openai_agents._temporal_worker_env_ref import (
    AllowAllWorkerEnvVars,
)
from temporalio.openai_agents.sandbox._temporal_activity_models import (
    CreateSessionArgs,
    ExecArgs,
    HydrateWorkspaceArgs,
    OnHostSandboxSessionState,
    PersistWorkspaceArgs,
    PersistWorkspaceResult,
    PtyExecStartArgs,
    PtyExecUpdateResult,
    PtyWriteStdinArgs,
    ReadArgs,
    ReadResult,
    ResumeSessionArgs,
    RunningArgs,
    RunningResult,
    SessionResult,
    StartArgs,
    StopArgs,
    WriteArgs,
    _HasState,
)
from temporalio.openai_agents.sandbox._temporal_activity_models import (
    ExecResult as ExecResultModel,
)
from temporalio.openai_agents.sandbox._temporal_sandbox_client import (
    TemporalSandboxClient,
)
from temporalio.openai_agents.sandbox._temporal_sandbox_session import (
    TemporalSandboxSession,
    _session_activity_config,
)
from temporalio.openai_agents.sandbox._temporal_worker_env_value import (
    _resolvable_worker_env_vars,
)
from temporalio.openai_agents.testing import (
    AgentEnvironment,
    ResponseBuilders,
    TestModel,
    TestModelProvider,
)
from temporalio.openai_agents.workflow import temporal_sandbox_client
from temporalio.testing import ActivityEnvironment
from temporalio.worker import Worker, WorkerTuner
from temporalio.workflow import ActivityConfig
from tests.helpers import new_worker

# ── _has_sandbox_agent unit tests ──


def test_has_sandbox_agent_regular_agent():
    assert _has_sandbox_agent(Agent[None](name="regular")) is False


def test_has_sandbox_agent_sandbox_starting():
    assert _has_sandbox_agent(SandboxAgent[None](name="sandbox")) is True


def test_has_sandbox_agent_sandbox_direct_handoff():
    sandbox = SandboxAgent[None](name="sandbox")
    regular = Agent[None](name="regular", handoffs=[sandbox])
    assert _has_sandbox_agent(regular) is True


def test_has_sandbox_agent_sandbox_deep_handoff():
    sandbox = SandboxAgent[None](name="sandbox")
    middle = Agent[None](name="middle", handoffs=[sandbox])
    top = Agent[None](name="top", handoffs=[middle])
    assert _has_sandbox_agent(top) is True


def test_has_sandbox_agent_no_sandbox_in_chain():
    c = Agent[None](name="c")
    b = Agent[None](name="b", handoffs=[c])
    a = Agent[None](name="a", handoffs=[b])
    assert _has_sandbox_agent(a) is False


def test_has_sandbox_agent_circular_no_sandbox():
    a: Agent[Any] = Agent[None](name="a")
    b: Agent[Any] = Agent[None](name="b", handoffs=[a])
    a.handoffs = [b]
    assert _has_sandbox_agent(a) is False


def test_has_sandbox_agent_circular_with_sandbox():
    sandbox = SandboxAgent[None](name="sandbox")
    a: Agent[Any] = Agent[None](name="a", handoffs=[sandbox])
    b: Agent[Any] = Agent[None](name="b", handoffs=[a])
    a.handoffs = [b, sandbox]
    assert _has_sandbox_agent(b) is True


# ── temporal_sandbox_client helper tests ──


def test_temporal_sandbox_client_returns_temporal_client():
    client = temporal_sandbox_client("my-backend")
    assert isinstance(client, TemporalSandboxClient)
    assert client._name == "my-backend"
    assert client.backend_id == "my-backend"


def test_temporal_sandbox_client_with_config():
    config = ActivityConfig(start_to_close_timeout=timedelta(minutes=10))
    client = temporal_sandbox_client("my-backend", config=config)
    assert isinstance(client, TemporalSandboxClient)
    assert client._config == config


# ── Workflow validation tests ──


def _mock_model():
    return TestModel.returning_responses([ResponseBuilders.output_message("test")])


@workflow.defn
class SandboxValidationWorkflow:
    """Single workflow that validates all sandbox configuration error cases."""

    @workflow.run
    async def run(self) -> str:
        # Case 1: SandboxAgent without run_config.sandbox
        try:
            agent = SandboxAgent[None](name="sandbox")
            await Runner.run(starting_agent=agent, input="hello")
            return "FAIL: no-config should have raised"
        except ValueError as e:
            assert "run_config.sandbox is not configured" in str(e)

        # Case 2: SandboxAgent reachable via handoff without run_config.sandbox
        try:
            sandbox = SandboxAgent[None](name="sandbox_target")
            router = Agent[None](name="router", handoffs=[sandbox])
            await Runner.run(starting_agent=router, input="hello")
            return "FAIL: handoff-no-config should have raised"
        except ValueError as e:
            assert "run_config.sandbox is not configured" in str(e)

        # Case 3: SandboxRunConfig with client=None
        try:
            agent = SandboxAgent[None](name="sandbox")
            await Runner.run(
                starting_agent=agent,
                input="hello",
                run_config=RunConfig(
                    sandbox=SandboxRunConfig(client=None),  # type: ignore[arg-type]
                ),
            )
            return "FAIL: null-client should have raised"
        except ValueError as e:
            assert "run_config.sandbox.client must be set" in str(e)

        # Case 4: Non-TemporalSandboxClient in run_config.sandbox.client
        try:
            agent = SandboxAgent[None](name="sandbox")
            await Runner.run(
                starting_agent=agent,
                input="hello",
                run_config=RunConfig(
                    sandbox=SandboxRunConfig(client=object()),  # type: ignore[arg-type]
                ),
            )
            return "FAIL: wrong-client should have raised"
        except ValueError as e:
            assert "temporal_sandbox_client(name)" in str(e)

        return "OK"


async def test_sandbox_validation_errors(client: Client):
    """All sandbox configuration errors should be caught immediately in the workflow."""
    async with AgentEnvironment(model=_mock_model()) as env:
        client = env.applied_on_client(client)
        async with new_worker(
            client,
            SandboxValidationWorkflow,
            workflow_failure_exception_types=[ValueError, AssertionError],
        ) as worker:
            result = await client.execute_workflow(
                SandboxValidationWorkflow.run,
                id=f"sandbox-validation-{uuid.uuid4()}",
                task_queue=worker.task_queue,
                execution_timeout=timedelta(seconds=10),
            )
            assert result == "OK"


# ── Mock sandbox infrastructure for delegation tests ──


class TestSessionState(SandboxSessionState):
    """Concrete ``SandboxSessionState`` subclass for tests that don't need a real backend."""

    __test__ = False
    type: Literal["test"] = "test"  # type: ignore


class _MockSandboxSession(BaseSandboxSession):
    """Minimal mock session that tracks calls and returns canned results."""

    def __init__(self, manifest: Manifest | None = None) -> None:
        self.state = TestSessionState(
            manifest=manifest or Manifest(),
            snapshot=NoopSnapshot(id=str(uuid.uuid4())),
        )
        self.exec_calls: list[tuple] = []
        self.read_calls: list[Path] = []
        self.write_calls: list[tuple[Path, bytes]] = []
        self.running_calls: int = 0
        self.start_calls: int = 0
        self.stop_calls: int = 0
        self.shutdown_calls: int = 0
        self.persist_workspace_calls: int = 0
        self.hydrate_workspace_calls: int = 0

    async def start(self) -> None:
        self.start_calls += 1

    async def stop(self) -> None:
        self.stop_calls += 1

    async def shutdown(self) -> None:
        self.shutdown_calls += 1

    async def running(self) -> bool:
        self.running_calls += 1
        return True

    async def _exec_internal(
        self,
        *command: str | Path,
        timeout: float | None = None,
    ) -> ExecResult:
        self.exec_calls.append((command, timeout))
        return ExecResult(stdout=b"ok\n", stderr=b"", exit_code=0)

    async def read(self, path: Path, *, user: Any = None) -> io.IOBase:  # type: ignore[reportUnusedParameter]
        self.read_calls.append(path)
        return io.BytesIO(b"file-content")

    async def write(self, path: Path, data: io.IOBase, *, user: Any = None) -> None:  # type: ignore[reportUnusedParameter]
        self.write_calls.append((path, data.read()))

    async def persist_workspace(self) -> io.IOBase:
        self.persist_workspace_calls += 1
        return io.BytesIO(b"workspace-archive")

    async def hydrate_workspace(self, data: io.IOBase) -> None:
        self.hydrate_workspace_calls += 1

    def supports_pty(self) -> bool:
        return False


class _MockSandboxClient(BaseSandboxClient[BaseSandboxClientOptions | None]):
    """Mock client that tracks create/resume/delete calls and delegates to a mock session."""

    backend_id = "mock"
    supports_default_options = True

    def __init__(self, session: _MockSandboxSession | None = None) -> None:
        self.inner_session = session or _MockSandboxSession()
        self.session = self._wrap_session(self.inner_session)
        self.create_calls: int = 0
        self.resume_calls: int = 0
        self.delete_calls: int = 0
        self.resolved_envs: dict[str, str] | None = None

    async def create(
        self,
        *,
        snapshot: Any = None,
        manifest: Manifest | None = None,
        options: BaseSandboxClientOptions | None = None,
    ) -> SandboxSession:
        self.create_calls += 1
        if manifest is not None:
            self.inner_session.state.manifest = manifest
            self.resolved_envs = await manifest.environment.resolve()
        return self.session

    async def resume(self, state: SandboxSessionState) -> SandboxSession:
        self.resume_calls += 1
        self.inner_session.state = state
        return self.session

    async def delete(self, session: SandboxSession) -> SandboxSession:
        self.delete_calls += 1
        return session

    def deserialize_session_state(self, payload: dict[str, Any]) -> SandboxSessionState:
        return SandboxSessionState.model_validate(payload)


async def test_start_and_stop_propagate_state_across_activity_serialization(
    monkeypatch: pytest.MonkeyPatch,
):
    class ChangingSession(_MockSandboxSession):
        async def start(self) -> None:
            await super().start()
            self.state.workspace_root_ready = True

        async def stop(self) -> None:
            await super().stop()
            self.state.workspace_root_ready = False

    inner = ChangingSession()
    provider = SandboxClientProvider("mock", _MockSandboxClient(inner))
    activities = _activity_map(provider)
    state = _make_state()
    session = TemporalSandboxSession(
        "mock", ActivityConfig(start_to_close_timeout=timedelta(seconds=30)), state
    )

    async def execute_activity(
        name: str, *, arg: Any, result_type: Any, **_kwargs: Any
    ):
        assert result_type is SessionResult
        args_type = StartArgs if name.endswith("_start") else StopArgs
        # Activities run on another process; mutations to the argument cannot
        # reach the workflow unless they are returned in the activity result.
        remote_args = args_type.model_validate(arg.model_dump(mode="json"))
        remote_result = await activities[name](remote_args)
        return SessionResult.model_validate(remote_result.model_dump(mode="json"))

    monkeypatch.setattr(workflow, "execute_activity", execute_activity)
    await session.start()
    assert session.state.workspace_root_ready is True
    await session.stop()
    assert session.state.workspace_root_ready is False
    assert inner.start_calls == inner.stop_calls == 1


# ── SandboxClientProvider unit tests (delegation) ──


@pytest.fixture
def mock_client() -> _MockSandboxClient:
    return _MockSandboxClient()


@pytest.fixture
def sandbox_activities(mock_client: _MockSandboxClient) -> SandboxClientProvider:
    return SandboxClientProvider("mock", mock_client)


def _make_state(manifest: Manifest | None = None) -> TestSessionState:
    return TestSessionState(
        manifest=manifest or Manifest(),
        snapshot=NoopSnapshot(id=str(uuid.uuid4())),
    )


def _activity_map(
    sandbox_activities: SandboxClientProvider,
    resolvable_worker_env_vars: Collection[str] = (),
) -> dict[str, Any]:
    """Build a short-name → callable dict from all() for easy test dispatch."""
    return {
        act.__temporal_activity_definition.name: act  # type: ignore[attr-defined, union-attr]
        for act in sandbox_activities._get_activities(resolvable_worker_env_vars)
    }


async def test_activities_create_session_delegates(
    sandbox_activities: SandboxClientProvider,
    mock_client: _MockSandboxClient,
):
    """create_session activity should delegate to the real client's create()."""
    acts = _activity_map(sandbox_activities)
    args = CreateSessionArgs(
        snapshot_spec=None,
        manifest=Manifest(),
        client_options=None,
    )
    result = await acts["mock-sandbox_client_create"](args)
    assert mock_client.create_calls == 1
    assert result.state is not None
    assert isinstance(result.supports_pty, bool)


async def test_create_session_activity_resolves_worker_env_value_but_returns_it_unresolved(
    sandbox_activities: SandboxClientProvider,
    mock_client: _MockSandboxClient,
    monkeypatch: pytest.MonkeyPatch,
):
    secret = "sk-activity-boundary-secret"
    monkeypatch.setenv("WORKER_ACTIVITY_SECRET", secret)

    def payload_bytes(value: BaseModel) -> bytes:
        payload = OpenAIPayloadConverter().to_payload(value)
        assert payload is not None
        return payload.data

    args = CreateSessionArgs(
        snapshot_spec=None,
        manifest=Manifest(
            environment=Environment(
                value={"API_KEY": TemporalWorkerEnvValue(name="WORKER_ACTIVITY_SECRET")}
            )
        ),
        client_options=None,
    )
    assert secret.encode() not in payload_bytes(args)

    acts = _activity_map(sandbox_activities, ["WORKER_ACTIVITY_SECRET"])
    result = await acts["mock-sandbox_client_create"](args)

    assert mock_client.resolved_envs == {"API_KEY": secret}
    returned = payload_bytes(result)
    assert secret.encode() not in returned
    assert b"temporal.worker_env_value" in returned


async def test_create_session_activity_refuses_an_unlisted_worker_env_value(
    sandbox_activities: SandboxClientProvider,
    monkeypatch: pytest.MonkeyPatch,
):
    secret = "sk-unlisted-secret"
    monkeypatch.setenv("WORKER_ACTIVITY_SECRET", secret)

    args = CreateSessionArgs(
        snapshot_spec=None,
        manifest=Manifest(
            environment=Environment(
                value={"API_KEY": TemporalWorkerEnvValue(name="WORKER_ACTIVITY_SECRET")}
            )
        ),
        client_options=None,
    )

    acts = _activity_map(sandbox_activities, ["SOMETHING_ELSE"])
    with pytest.raises(ApplicationError) as exc_info:
        await acts["mock-sandbox_client_create"](args)

    assert exc_info.value.type == "TemporalWorkerEnvValueUnresolved"
    assert exc_info.value.non_retryable
    assert "WORKER_ACTIVITY_SECRET" in str(exc_info.value)
    assert "resolvable_worker_env_vars" in str(exc_info.value)
    assert secret not in str(exc_info.value)


async def test_the_resolvable_names_are_snapshotted_when_the_activities_are_built(
    sandbox_activities: SandboxClientProvider,
    mock_client: _MockSandboxClient,
    monkeypatch: pytest.MonkeyPatch,
):
    secret = "sk-snapshot-secret"
    monkeypatch.setenv("WORKER_ACTIVITY_SECRET", secret)

    args = CreateSessionArgs(
        snapshot_spec=None,
        manifest=Manifest(
            environment=Environment(
                value={"API_KEY": TemporalWorkerEnvValue(name="WORKER_ACTIVITY_SECRET")}
            )
        ),
        client_options=None,
    )

    resolvable = ["WORKER_ACTIVITY_SECRET"]
    acts = _activity_map(sandbox_activities, resolvable)
    resolvable.clear()

    await acts["mock-sandbox_client_create"](args)
    assert mock_client.resolved_envs == {"API_KEY": secret}


class _ScopeRecordingSession(_MockSandboxSession):
    def __init__(self, scopes: list[frozenset[str] | AllowAllWorkerEnvVars]) -> None:
        super().__init__()
        self._scopes = scopes

    def supports_pty(self) -> bool:
        return True

    async def shutdown(self) -> None:
        self._scopes.append(_resolvable_worker_env_vars.get(frozenset()))
        await super().shutdown()

    async def pty_exec_start(
        self, *command: str | Path, **kwargs: Any
    ) -> PtyExecUpdate:
        return PtyExecUpdate(
            process_id=1, output=b"", exit_code=None, original_token_count=None
        )

    async def pty_write_stdin(self, **kwargs: Any) -> PtyExecUpdate:
        return PtyExecUpdate(
            process_id=1, output=b"", exit_code=None, original_token_count=None
        )


class _ScopeRecordingClient(_MockSandboxClient):
    def __init__(self) -> None:
        self.scopes: list[frozenset[str] | AllowAllWorkerEnvVars] = []
        super().__init__(_ScopeRecordingSession(self.scopes))

    async def create(self, **kwargs: Any) -> SandboxSession:
        self.scopes.append(_resolvable_worker_env_vars.get(frozenset()))
        return await super().create(**kwargs)

    async def resume(self, state: SandboxSessionState) -> SandboxSession:
        self.scopes.append(_resolvable_worker_env_vars.get(frozenset()))
        return await super().resume(state)


async def test_every_activity_runs_its_body_inside_the_resolvable_names_scope():
    recording_client = _ScopeRecordingClient()
    provider = SandboxClientProvider("mock", recording_client)
    acts = _activity_map(provider, ["A_RESOLVABLE_NAME"])
    state = _make_state()
    args_by_activity: dict[str, Any] = {
        "mock-sandbox_client_create": CreateSessionArgs(
            snapshot_spec=None, manifest=Manifest(), client_options=None
        ),
        "mock-sandbox_client_resume": ResumeSessionArgs(state=state),
        "mock-sandbox_client_delete": StopArgs(state=state),
        "mock-sandbox_session_exec": ExecArgs(state=state, command=["ls"], shell=True),
        "mock-sandbox_session_read": ReadArgs(state=state, path="/tmp/f"),
        "mock-sandbox_session_write": WriteArgs(state=state, path="/tmp/f", data=b"d"),
        "mock-sandbox_session_running": RunningArgs(state=state),
        "mock-sandbox_session_persist_workspace": PersistWorkspaceArgs(state=state),
        "mock-sandbox_session_hydrate_workspace": HydrateWorkspaceArgs(
            state=state, data=b"d"
        ),
        "mock-sandbox_session_pty_exec_start": PtyExecStartArgs(
            state=state, command=["ls"]
        ),
        "mock-sandbox_session_pty_write_stdin": PtyWriteStdinArgs(
            state=state, session_id=1, chars="x"
        ),
        "mock-sandbox_session_start": StartArgs(state=state),
        "mock-sandbox_session_stop": StopArgs(state=state),
        "mock-sandbox_session_shutdown": StopArgs(state=state),
    }
    assert set(args_by_activity) == set(acts)

    for name, args in args_by_activity.items():
        # Shutdown is the one activity that does not resume on a cache miss, so
        # it alone needs a cached session to reach the client.
        cached = name == "mock-sandbox_session_shutdown"
        provider._sessions = (
            {str(state.session_id): recording_client.session} if cached else {}
        )
        recording_client.scopes.clear()
        await acts[name](args)
        assert recording_client.scopes == [frozenset({"A_RESOLVABLE_NAME"})], name


def _plugin_with_one_shot_names(
    *sandbox_clients: SandboxClientProvider,
) -> OpenAIAgentsPlugin:
    return OpenAIAgentsPlugin(
        model_params=ModelActivityParameters(
            start_to_close_timeout=timedelta(seconds=30),
        ),
        model_provider=TestModelProvider(
            TestModel.returning_responses([ResponseBuilders.output_message("done")])
        ),
        sandbox_clients=list(sandbox_clients),
        resolvable_worker_env_vars=(name for name in ["A_RESOLVABLE_NAME"]),  # type: ignore[arg-type]
    )


def _plugin_activity_map(plugin: OpenAIAgentsPlugin) -> dict[str, Any]:
    build_activities = plugin.activities
    assert callable(build_activities)
    return {
        act.__temporal_activity_definition.name: act  # type: ignore[attr-defined, union-attr]
        for act in build_activities([])
    }


def _hosted_resolvable_names(
    acts: dict[str, Any],
) -> frozenset[str] | AllowAllWorkerEnvVars:
    return acts["invoke_model_activity"].__self__._env_refs._allowed


async def test_a_one_shot_names_iterable_reaches_the_hosted_and_the_sandbox_activities():
    recording_client = _ScopeRecordingClient()
    acts = _plugin_activity_map(
        _plugin_with_one_shot_names(SandboxClientProvider("mock", recording_client))
    )

    assert _hosted_resolvable_names(acts) == frozenset({"A_RESOLVABLE_NAME"})
    await acts["mock-sandbox_client_create"](
        CreateSessionArgs(snapshot_spec=None, manifest=Manifest(), client_options=None)
    )
    assert recording_client.scopes == [frozenset({"A_RESOLVABLE_NAME"})]


async def test_every_sandbox_provider_on_a_plugin_gets_the_names():
    first = _ScopeRecordingClient()
    second = _ScopeRecordingClient()
    acts = _plugin_activity_map(
        _plugin_with_one_shot_names(
            SandboxClientProvider("first", first),
            SandboxClientProvider("second", second),
        )
    )

    args = CreateSessionArgs(
        snapshot_spec=None, manifest=Manifest(), client_options=None
    )
    await acts["first-sandbox_client_create"](args)
    await acts["second-sandbox_client_create"](args)
    assert first.scopes == [frozenset({"A_RESOLVABLE_NAME"})]
    assert second.scopes == [frozenset({"A_RESOLVABLE_NAME"})]


async def test_a_second_worker_built_from_one_plugin_gets_the_names():
    recording_client = _ScopeRecordingClient()
    plugin = _plugin_with_one_shot_names(
        SandboxClientProvider("mock", recording_client)
    )
    _plugin_activity_map(plugin)
    acts = _plugin_activity_map(plugin)

    assert _hosted_resolvable_names(acts) == frozenset({"A_RESOLVABLE_NAME"})
    await acts["mock-sandbox_client_create"](
        CreateSessionArgs(snapshot_spec=None, manifest=Manifest(), client_options=None)
    )
    assert recording_client.scopes == [frozenset({"A_RESOLVABLE_NAME"})]


async def test_activities_resume_session_delegates(
    sandbox_activities: SandboxClientProvider,
    mock_client: _MockSandboxClient,
):
    """resume_session activity should delegate to the real client's resume()."""
    acts = _activity_map(sandbox_activities)
    state = _make_state()
    args = ResumeSessionArgs(state=state)
    result = await acts["mock-sandbox_client_resume"](args)
    assert mock_client.resume_calls == 1
    assert result.state is not None


async def test_activities_exec_delegates(
    sandbox_activities: SandboxClientProvider,
    mock_client: _MockSandboxClient,
):
    """exec activity should delegate to the real session's exec()."""
    acts = _activity_map(sandbox_activities)
    # First create a session so the activities cache is populated
    await acts["mock-sandbox_client_create"](
        CreateSessionArgs(snapshot_spec=None, manifest=Manifest(), client_options=None)
    )
    state = mock_client.inner_session.state

    args = ExecArgs(state=state, command=["echo", "hello"], timeout=10.0, shell=True)
    result = await acts["mock-sandbox_session_exec"](args)
    assert result.stdout == b"ok\n"
    assert result.stderr == b""
    assert result.exit_code == 0
    assert len(mock_client.inner_session.exec_calls) == 1


async def test_activities_read_delegates(
    sandbox_activities: SandboxClientProvider,
    mock_client: _MockSandboxClient,
):
    """read activity should delegate to the real session's read()."""
    acts = _activity_map(sandbox_activities)
    await acts["mock-sandbox_client_create"](
        CreateSessionArgs(snapshot_spec=None, manifest=Manifest(), client_options=None)
    )
    state = mock_client.inner_session.state

    args = ReadArgs(state=state, path="/tmp/test.txt")
    result = await acts["mock-sandbox_session_read"](args)
    assert result.data == b"file-content"
    assert len(mock_client.inner_session.read_calls) == 1


async def test_activities_write_delegates(
    sandbox_activities: SandboxClientProvider,
    mock_client: _MockSandboxClient,
):
    """write activity should delegate to the real session's write()."""
    acts = _activity_map(sandbox_activities)
    await acts["mock-sandbox_client_create"](
        CreateSessionArgs(snapshot_spec=None, manifest=Manifest(), client_options=None)
    )
    state = mock_client.inner_session.state

    args = WriteArgs(state=state, path="/tmp/out.txt", data=b"written-data")
    await acts["mock-sandbox_session_write"](args)
    assert len(mock_client.inner_session.write_calls) == 1
    assert mock_client.inner_session.write_calls[0][1] == b"written-data"


async def test_activities_running_delegates(
    sandbox_activities: SandboxClientProvider,
    mock_client: _MockSandboxClient,
):
    """running activity should delegate to the real session's running()."""
    acts = _activity_map(sandbox_activities)
    await acts["mock-sandbox_client_create"](
        CreateSessionArgs(snapshot_spec=None, manifest=Manifest(), client_options=None)
    )
    state = mock_client.inner_session.state

    args = RunningArgs(state=state)
    result = await acts["mock-sandbox_session_running"](args)
    assert result.is_running is True
    assert mock_client.inner_session.running_calls == 1


async def test_activities_client_delete_delegates(
    sandbox_activities: SandboxClientProvider,
    mock_client: _MockSandboxClient,
):
    """client_delete activity should delegate to the real client's delete()."""
    acts = _activity_map(sandbox_activities)
    await acts["mock-sandbox_client_create"](
        CreateSessionArgs(snapshot_spec=None, manifest=Manifest(), client_options=None)
    )
    state = mock_client.inner_session.state

    args = StopArgs(state=state)
    await acts["mock-sandbox_client_delete"](args)

    assert mock_client.delete_calls == 1


async def test_activities_session_shutdown_clears_cache(
    sandbox_activities: SandboxClientProvider,
    mock_client: _MockSandboxClient,
):
    """session_shutdown activity should call session.shutdown() and evict from cache."""
    acts = _activity_map(sandbox_activities)
    await acts["mock-sandbox_client_create"](
        CreateSessionArgs(snapshot_spec=None, manifest=Manifest(), client_options=None)
    )
    state = mock_client.inner_session.state
    session_key = str(state.session_id)

    # Session should be cached
    assert session_key in sandbox_activities._sessions

    args = StopArgs(state=state)
    await acts["mock-sandbox_session_shutdown"](args)

    assert mock_client.inner_session.shutdown_calls == 1
    # Session should be evicted from cache
    assert session_key not in sandbox_activities._sessions


async def test_activities_session_shutdown_noop_for_unknown_session(
    sandbox_activities: SandboxClientProvider,
):
    """session_shutdown should be a no-op if the session isn't in the cache."""
    acts = _activity_map(sandbox_activities)
    state = _make_state()
    args = StopArgs(state=state)
    # Should not raise
    await acts["mock-sandbox_session_shutdown"](args)


async def test_activities_session_caching(
    sandbox_activities: SandboxClientProvider,
    mock_client: _MockSandboxClient,
):
    """Multiple operations on the same session should reuse the cached session."""
    acts = _activity_map(sandbox_activities)
    await acts["mock-sandbox_client_create"](
        CreateSessionArgs(snapshot_spec=None, manifest=Manifest(), client_options=None)
    )
    state = mock_client.inner_session.state

    # Multiple exec calls should not trigger additional resume calls
    await acts["mock-sandbox_session_exec"](
        ExecArgs(state=state, command=["cmd1"], shell=True)
    )
    await acts["mock-sandbox_session_exec"](
        ExecArgs(state=state, command=["cmd2"], shell=True)
    )
    assert mock_client.resume_calls == 0
    assert len(mock_client.inner_session.exec_calls) == 2


async def test_activities_all_returns_all_activity_methods(
    sandbox_activities: SandboxClientProvider,
):
    """all() should return all 14 activity callables with prefixed names."""
    activities = sandbox_activities._get_activities(())
    assert len(activities) == 14
    # Verify they are all activity-decorated callables with prefixed names
    activity_names = set()
    for act in activities:
        assert hasattr(act, "__temporal_activity_definition")
        activity_names.add(act.__temporal_activity_definition.name)  # type: ignore[union-attr]
    expected = {
        "mock-sandbox_client_create",
        "mock-sandbox_client_resume",
        "mock-sandbox_client_delete",
        "mock-sandbox_session_exec",
        "mock-sandbox_session_read",
        "mock-sandbox_session_write",
        "mock-sandbox_session_running",
        "mock-sandbox_session_persist_workspace",
        "mock-sandbox_session_hydrate_workspace",
        "mock-sandbox_session_pty_exec_start",
        "mock-sandbox_session_pty_write_stdin",
        "mock-sandbox_session_start",
        "mock-sandbox_session_stop",
        "mock-sandbox_session_shutdown",
    }
    assert activity_names == expected


async def test_multiple_providers_register_distinct_activities():
    """Multiple SandboxClientProviders should produce distinct prefixed activity sets."""
    client1 = _MockSandboxClient()
    client2 = _MockSandboxClient()
    provider1 = SandboxClientProvider("daytona", client1)
    provider2 = SandboxClientProvider("local", client2)

    activities1 = provider1._get_activities(())
    activities2 = provider2._get_activities(())

    names1 = {a.__temporal_activity_definition.name for a in activities1}  # type: ignore
    names2 = {a.__temporal_activity_definition.name for a in activities2}  # type: ignore

    # No overlap
    assert names1.isdisjoint(names2)
    # Both have 14 activities
    assert len(names1) == 14
    assert len(names2) == 14
    # Verify prefixes
    assert all(
        n.startswith("daytona-sandbox_client_")
        or n.startswith("daytona-sandbox_session_")
        for n in names1
    )
    assert all(
        n.startswith("local-sandbox_client_") or n.startswith("local-sandbox_session_")
        for n in names2
    )


# ── SandboxError retryable mapping tests ──


class _ExecRaisingSession(_MockSandboxSession):
    """Mock session whose exec() raises a chosen SandboxError."""

    def __init__(self, error: SandboxError) -> None:
        super().__init__()
        self._error = error

    async def _exec_internal(
        self,
        *command: str | Path,  # type: ignore[reportUnusedParameter]
        timeout: float | None = None,  # type: ignore[reportUnusedParameter]
    ) -> ExecResult:
        raise self._error


async def _exec_with_error(error: SandboxError) -> None:
    provider = SandboxClientProvider(
        "mock", _MockSandboxClient(_ExecRaisingSession(error))
    )
    acts = _activity_map(provider)
    state = (
        await acts["mock-sandbox_client_create"](
            CreateSessionArgs(
                snapshot_spec=None, manifest=Manifest(), client_options=None
            )
        )
    ).state
    await acts["mock-sandbox_session_exec"](
        ExecArgs(state=state, command=["boom"], shell=True)
    )


async def test_exec_terminal_error_becomes_non_retryable_application_error():
    """retryable is False should map to a non-retryable ApplicationError."""
    with pytest.raises(ApplicationError) as exc_info:
        await _exec_with_error(ExecTransportError(command=["boom"], retryable=False))
    assert exc_info.value.non_retryable is True
    assert exc_info.value.type == "exec_transport_error"


async def test_exec_transient_error_propagates_unchanged():
    """retryable is True should let the original SandboxError propagate."""
    with pytest.raises(ExecTransportError):
        await _exec_with_error(ExecTransportError(command=["boom"], retryable=True))


async def test_exec_unclassified_error_propagates_unchanged():
    """retryable is None should let the original SandboxError propagate (not converted)."""
    with pytest.raises(ExecTransportError):
        await _exec_with_error(ExecTransportError(command=["boom"], retryable=None))


class _ShutdownRaisingSession(_MockSandboxSession):
    """Mock session whose shutdown() raises a chosen SandboxError."""

    def __init__(self, error: SandboxError) -> None:
        super().__init__()
        self._error = error

    async def shutdown(self) -> None:
        raise self._error


async def _create_shutdown_raising(
    error: SandboxError,
) -> tuple[dict[str, Any], SandboxClientProvider, StopArgs, str]:
    provider = SandboxClientProvider(
        "mock", _MockSandboxClient(_ShutdownRaisingSession(error))
    )
    acts = _activity_map(provider)
    state = (
        await acts["mock-sandbox_client_create"](
            CreateSessionArgs(
                snapshot_spec=None, manifest=Manifest(), client_options=None
            )
        )
    ).state
    key = str(state.session_id)
    assert key in provider._sessions
    return acts, provider, StopArgs(state=state), key


async def test_shutdown_terminal_error_evicts_session_and_raises():
    """A terminal shutdown error maps to a non-retryable ApplicationError and
    evicts the dead session from the cache."""
    acts, provider, args, key = await _create_shutdown_raising(
        ExecTransportError(command=["shutdown"], retryable=False)
    )

    with pytest.raises(ApplicationError) as exc_info:
        await acts["mock-sandbox_session_shutdown"](args)
    assert exc_info.value.non_retryable is True
    assert key not in provider._sessions


async def test_shutdown_retryable_error_keeps_session_cached():
    """A retryable shutdown error propagates unchanged and leaves the session
    cached so the activity's retry can still shut it down."""
    acts, provider, args, key = await _create_shutdown_raising(
        ExecTransportError(command=["shutdown"], retryable=True)
    )

    with pytest.raises(ExecTransportError):
        await acts["mock-sandbox_session_shutdown"](args)
    assert key in provider._sessions


class _RunningRaisingSession(_MockSandboxSession):
    """Mock session whose running() raises a chosen SandboxError."""

    def __init__(self, error: SandboxError) -> None:
        super().__init__()
        self._error = error

    async def running(self) -> bool:
        raise self._error


async def test_running_terminal_error_becomes_non_retryable_application_error():
    """A terminal SandboxError from a non-exec activity also maps to a
    non-retryable ApplicationError, with type set to its error_code."""
    error = WorkspaceArchiveReadError(path=Path("/workspace"), retryable=False)
    provider = SandboxClientProvider(
        "mock", _MockSandboxClient(_RunningRaisingSession(error))
    )
    acts = _activity_map(provider)
    state = (
        await acts["mock-sandbox_client_create"](
            CreateSessionArgs(
                snapshot_spec=None, manifest=Manifest(), client_options=None
            )
        )
    ).state

    with pytest.raises(ApplicationError) as exc_info:
        await acts["mock-sandbox_session_running"](RunningArgs(state=state))
    assert exc_info.value.non_retryable is True
    assert exc_info.value.type == "workspace_archive_read_error"


# ── End-to-end test: Runner + SandboxAgent through Temporal activities ──


class _TestSandboxCapability(Capability):
    """Minimal capability exposing exec, read, and write via FunctionTools."""

    def __init__(self) -> None:
        super().__init__(type="test_sandbox")
        self._session: BaseSandboxSession | None = None

    def bind(self, session: BaseSandboxSession) -> None:
        self._session = session

    def tools(self) -> list[Tool]:
        session = self._session

        async def _run_cmd(ctx: Any, args: str) -> str:  # type: ignore[reportUnusedParameter]
            import json

            cmd = json.loads(args)["cmd"]
            result = await session.exec(cmd, shell=True)  # type: ignore[union-attr]
            return result.stdout.decode()

        async def _read_file(ctx: Any, args: str) -> str:  # type: ignore[reportUnusedParameter]
            import json

            path = json.loads(args)["path"]
            handle = await session.read(Path(path))  # type: ignore[union-attr]
            return handle.read().decode()

        async def _write_file(ctx: Any, args: str) -> str:  # type: ignore[reportUnusedParameter]
            import json

            parsed = json.loads(args)
            await session.write(  # type: ignore[union-attr]
                Path(parsed["path"]), io.BytesIO(parsed["data"].encode())
            )
            return "ok"

        return [
            FunctionTool(
                name="run_command",
                description="Run a shell command",
                params_json_schema={
                    "type": "object",
                    "properties": {"cmd": {"type": "string"}},
                    "required": ["cmd"],
                },
                on_invoke_tool=_run_cmd,
            ),
            FunctionTool(
                name="read_file",
                description="Read a file",
                params_json_schema={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
                on_invoke_tool=_read_file,
            ),
            FunctionTool(
                name="write_file",
                description="Write a file",
                params_json_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "data": {"type": "string"},
                    },
                    "required": ["path", "data"],
                },
                on_invoke_tool=_write_file,
            ),
        ]


class _TestSandboxClientOptions(BaseSandboxClientOptions):
    type: str = "test"  # type: ignore[reportIncompatibleVariableOverride]


@workflow.defn
class SandboxE2EWorkflow:
    @workflow.run
    async def run(self) -> str:
        agent = SandboxAgent[None](
            name="sandbox-e2e", capabilities=[_TestSandboxCapability()]
        )
        result = await Runner.run(
            starting_agent=agent,
            input="run a command",
            run_config=RunConfig(
                sandbox=SandboxRunConfig(
                    client=temporal_sandbox_client("mock"),
                    options=_TestSandboxClientOptions(),
                ),
            ),
        )
        return result.final_output


_HOST_PATH = "/host/private-dir"
_HOST_PATH_GRANT_MANIFEST = Manifest(
    extra_path_grants=(
        SandboxPathGrant(path="/workspace/shared", host_path=_HOST_PATH),
    )
)
# A clean grant first, so a check that only inspects index 0 fails this.
_TRAILING_GRANT_MANIFEST = Manifest(
    extra_path_grants=(
        SandboxPathGrant(path="/workspace/clean"),
        SandboxPathGrant(path="/workspace/shared", host_path=_HOST_PATH),
    )
)
_TWO_BOUND_GRANTS_MANIFEST = Manifest(
    extra_path_grants=(
        SandboxPathGrant(path="/workspace/shared", host_path=_HOST_PATH),
        SandboxPathGrant(path="/workspace/other", host_path="/host/second-dir"),
    )
)


class _GrantInjectingCapability(Capability):
    def __init__(self) -> None:
        super().__init__(type="grant_injecting")

    def process_manifest(self, manifest: Manifest) -> Manifest:
        return manifest.model_copy(
            update={
                "extra_path_grants": (
                    *manifest.extra_path_grants,
                    SandboxPathGrant(path="/workspace/injected", host_path=_HOST_PATH),
                )
            }
        )


@workflow.defn
class HostPathGrantWorkflow:
    @workflow.run
    async def run(self, route: str) -> str:
        agent = SandboxAgent[None](name="sandbox-grant")
        client = temporal_sandbox_client("mock")
        options = _TestSandboxClientOptions()
        expected = "/workspace/shared"

        if route == "run_config_manifest":
            sandbox = SandboxRunConfig(
                client=client, options=options, manifest=_HOST_PATH_GRANT_MANIFEST
            )
        elif route == "default_manifest":
            agent = SandboxAgent[None](
                name="sandbox-grant", default_manifest=_HOST_PATH_GRANT_MANIFEST
            )
            sandbox = SandboxRunConfig(client=client, options=options)
        elif route == "session_state":
            sandbox = SandboxRunConfig(
                client=client,
                options=options,
                session_state=TestSessionState(
                    manifest=_HOST_PATH_GRANT_MANIFEST,
                    snapshot=NoopSnapshot(id=str(workflow.uuid4())),
                ),
            )
        elif route == "capability":
            # The manifest must be present but empty: upstream skips capabilities
            # when there is no manifest at all.
            agent = SandboxAgent[None](
                name="sandbox-grant", capabilities=[_GrantInjectingCapability()]
            )
            sandbox = SandboxRunConfig(
                client=client, options=options, manifest=Manifest()
            )
            expected = "/workspace/injected"
        elif route == "trailing_grant":
            sandbox = SandboxRunConfig(
                client=client, options=options, manifest=_TRAILING_GRANT_MANIFEST
            )
        elif route == "two_bound_grants":
            sandbox = SandboxRunConfig(
                client=client, options=options, manifest=_TWO_BOUND_GRANTS_MANIFEST
            )
            expected = "/workspace/shared, /workspace/other"
        else:
            raise AssertionError(f"unknown route {route}")

        try:
            await Runner.run(
                starting_agent=agent,
                input="hello",
                run_config=RunConfig(sandbox=sandbox),
            )
        except AgentsWorkflowError as e:
            assert expected in str(e), str(e)
            # The guard must not name the host path: this text reaches history.
            assert _HOST_PATH not in str(e), str(e)
            return "REJECTED"
        return "NOT REJECTED"


@pytest.mark.parametrize(
    "route",
    [
        "run_config_manifest",
        "default_manifest",
        "session_state",
        "capability",
        "trailing_grant",
        "two_bound_grants",
    ],
)
async def test_host_path_grants_are_rejected_per_manifest_source(
    client: Client, route: str
):
    mock_sandbox_client = _MockSandboxClient(_MockSandboxSession())
    plugin = OpenAIAgentsPlugin(
        model_params=ModelActivityParameters(
            start_to_close_timeout=timedelta(seconds=30),
        ),
        model_provider=TestModelProvider(
            TestModel.returning_responses([ResponseBuilders.output_message("done")])
        ),
        sandbox_clients=[SandboxClientProvider("mock", mock_sandbox_client)],
    )
    new_config = client.config()
    new_config["plugins"] = [plugin]
    test_client = Client(**new_config)

    async with new_worker(
        test_client,
        HostPathGrantWorkflow,
        workflow_failure_exception_types=[Exception],
    ) as worker:
        result = await test_client.execute_workflow(
            HostPathGrantWorkflow.run,
            route,
            id=f"host-path-grant-{route}-{uuid.uuid4()}",
            task_queue=worker.task_queue,
            execution_timeout=timedelta(seconds=15),
        )

    assert result == "REJECTED"
    assert mock_sandbox_client.create_calls == 0
    assert mock_sandbox_client.resume_calls == 0


@workflow.defn
class UncaughtHostPathGrantWorkflow:
    @workflow.run
    async def run(self) -> str:
        await Runner.run(
            starting_agent=SandboxAgent[None](name="sandbox-grant"),
            input="hello",
            run_config=RunConfig(
                sandbox=SandboxRunConfig(
                    client=temporal_sandbox_client("mock"),
                    options=_TestSandboxClientOptions(),
                    manifest=_HOST_PATH_GRANT_MANIFEST,
                ),
            ),
        )
        return "NOT REJECTED"


async def test_host_path_grant_fails_the_workflow_on_a_production_like_worker(
    client: Client,
):
    """Given no test-only ``workflow_failure_exception_types``, so only the plugin's own applies."""
    mock_sandbox_client = _MockSandboxClient(_MockSandboxSession())
    plugin = OpenAIAgentsPlugin(
        model_params=ModelActivityParameters(
            start_to_close_timeout=timedelta(seconds=30),
        ),
        model_provider=TestModelProvider(
            TestModel.returning_responses([ResponseBuilders.output_message("done")])
        ),
        sandbox_clients=[SandboxClientProvider("mock", mock_sandbox_client)],
    )
    new_config = client.config()
    new_config["plugins"] = [plugin]
    test_client = Client(**new_config)

    async with new_worker(test_client, UncaughtHostPathGrantWorkflow) as worker:
        with pytest.raises(WorkflowFailureError) as exc_info:
            await test_client.execute_workflow(
                UncaughtHostPathGrantWorkflow.run,
                id=f"host-path-uncaught-{uuid.uuid4()}",
                task_queue=worker.task_queue,
                execution_timeout=timedelta(seconds=15),
            )

    cause = exc_info.value.cause
    assert isinstance(cause, ApplicationError), cause
    assert cause.type == "AgentsWorkflowError", cause.type
    assert "/workspace/shared" in str(cause)
    assert _HOST_PATH not in str(cause)
    assert mock_sandbox_client.create_calls == 0


_PLUGIN_ENV_NAME = "WORKER_ENV_VALUE_THROUGH_THE_PLUGIN"


@workflow.defn
class SandboxWorkerEnvValueWorkflow:
    @workflow.run
    async def run(self) -> str:
        await Runner.run(
            starting_agent=SandboxAgent[None](name="sandbox-env"),
            input="hello",
            run_config=RunConfig(
                sandbox=SandboxRunConfig(
                    client=temporal_sandbox_client("mock"),
                    options=_TestSandboxClientOptions(),
                    manifest=Manifest(
                        environment=Environment(
                            value={
                                "API_KEY": TemporalWorkerEnvValue(name=_PLUGIN_ENV_NAME)
                            }
                        )
                    ),
                ),
            ),
        )
        return "RAN"


@pytest.mark.parametrize(
    ("resolvable", "resolves"),
    [([_PLUGIN_ENV_NAME], True), (["SOMETHING_ELSE"], False)],
)
async def test_a_sandbox_activity_resolves_only_the_variables_its_plugin_names(
    client: Client,
    monkeypatch: pytest.MonkeyPatch,
    resolvable: list[str],
    resolves: bool,
):
    secret = "sk-through-the-plugin"
    monkeypatch.setenv(_PLUGIN_ENV_NAME, secret)
    mock_sandbox_client = _MockSandboxClient(_MockSandboxSession())
    plugin = OpenAIAgentsPlugin(
        model_params=ModelActivityParameters(
            start_to_close_timeout=timedelta(seconds=30),
        ),
        model_provider=TestModelProvider(
            TestModel.returning_responses([ResponseBuilders.output_message("done")])
        ),
        sandbox_clients=[SandboxClientProvider("mock", mock_sandbox_client)],
        resolvable_worker_env_vars=resolvable,
    )
    new_config = client.config()
    new_config["plugins"] = [plugin]
    test_client = Client(**new_config)

    async def execute(worker: Any) -> str:
        return await test_client.execute_workflow(
            SandboxWorkerEnvValueWorkflow.run,
            id=f"sandbox-env-value-{uuid.uuid4()}",
            task_queue=worker.task_queue,
            execution_timeout=timedelta(seconds=15),
        )

    async with new_worker(test_client, SandboxWorkerEnvValueWorkflow) as worker:
        if resolves:
            assert await execute(worker) == "RAN"
            assert mock_sandbox_client.resolved_envs == {"API_KEY": secret}
        else:
            with pytest.raises(WorkflowFailureError) as exc_info:
                await execute(worker)
            activity_error = exc_info.value.cause
            assert isinstance(activity_error, ActivityError), activity_error
            cause = activity_error.cause
            assert isinstance(cause, ApplicationError), cause
            assert cause.type == "TemporalWorkerEnvValueUnresolved"
            assert _PLUGIN_ENV_NAME in str(cause)
            assert secret not in str(cause)


@workflow.defn
class ResolveOnWorkflowThreadWorkflow:
    @workflow.run
    async def run(self) -> str:
        try:
            await TemporalWorkerEnvValue(name="WORKER_THREAD_PROBE_NAME").resolve()
        except ApplicationError as e:
            return e.message
        return "NO RAISE"


async def test_worker_env_value_resolve_raises_inside_a_real_workflow(client: Client):
    """``in_workflow()`` is genuinely True here, unlike the monkeypatched unit test."""
    plugin = OpenAIAgentsPlugin(
        model_params=ModelActivityParameters(
            start_to_close_timeout=timedelta(seconds=30),
        ),
        model_provider=TestModelProvider(
            TestModel.returning_responses([ResponseBuilders.output_message("done")])
        ),
    )
    new_config = client.config()
    new_config["plugins"] = [plugin]
    test_client = Client(**new_config)

    async with new_worker(
        test_client,
        ResolveOnWorkflowThreadWorkflow,
        workflow_failure_exception_types=[Exception],
    ) as worker:
        result = await test_client.execute_workflow(
            ResolveOnWorkflowThreadWorkflow.run,
            id=f"resolve-in-workflow-{uuid.uuid4()}",
            task_queue=worker.task_queue,
            execution_timeout=timedelta(seconds=15),
        )

    assert "must run in an activity" in result


@workflow.defn
class LiveSandboxSessionWorkflow:
    @workflow.run
    async def run(self) -> str:
        try:
            await Runner.run(
                starting_agent=SandboxAgent[None](name="sandbox-live"),
                input="hello",
                run_config=RunConfig(
                    sandbox=SandboxRunConfig(
                        # Rejected on presence, so the value is never used.
                        session=object(),  # type: ignore[arg-type]
                    ),
                ),
            )
        except AgentsWorkflowError as e:
            assert "run_config.sandbox.session" in str(e), str(e)
            return "REJECTED"
        return "NOT REJECTED"


async def test_live_sandbox_session_is_rejected(client: Client):
    mock_sandbox_client = _MockSandboxClient(_MockSandboxSession())
    plugin = OpenAIAgentsPlugin(
        model_params=ModelActivityParameters(
            start_to_close_timeout=timedelta(seconds=30),
        ),
        model_provider=TestModelProvider(
            TestModel.returning_responses([ResponseBuilders.output_message("done")])
        ),
        sandbox_clients=[SandboxClientProvider("mock", mock_sandbox_client)],
    )
    new_config = client.config()
    new_config["plugins"] = [plugin]
    test_client = Client(**new_config)

    async with new_worker(
        test_client,
        LiveSandboxSessionWorkflow,
        workflow_failure_exception_types=[Exception],
    ) as worker:
        result = await test_client.execute_workflow(
            LiveSandboxSessionWorkflow.run,
            id=f"live-session-{uuid.uuid4()}",
            task_queue=worker.task_queue,
            execution_timeout=timedelta(seconds=15),
        )

    assert result == "REJECTED"
    assert mock_sandbox_client.create_calls == 0


async def test_sandbox_e2e_runner(client: Client):
    """End-to-end: Runner.run() with SandboxAgent exercises the full sandbox
    lifecycle (create, start, stop, shutdown, delete) through Temporal activities."""
    mock_session = _MockSandboxSession()
    mock_sandbox_client = _MockSandboxClient(mock_session)

    mock_model = TestModel.returning_responses(
        [
            ResponseBuilders.tool_call(
                '{"cmd": "echo hello"}',
                "run_command",
                call_id="call-run-command",
                item_id="item-run-command",
            ),
            ResponseBuilders.tool_call(
                '{"path": "/tmp/test.txt"}',
                "read_file",
                call_id="call-read-file",
                item_id="item-read-file",
            ),
            ResponseBuilders.tool_call(
                '{"path": "/tmp/out.txt", "data": "hello"}',
                "write_file",
                call_id="call-write-file",
                item_id="item-write-file",
            ),
            ResponseBuilders.output_message("Done."),
        ]
    )

    plugin = OpenAIAgentsPlugin(
        model_params=ModelActivityParameters(
            start_to_close_timeout=timedelta(seconds=30),
        ),
        model_provider=TestModelProvider(mock_model),
        sandbox_clients=[SandboxClientProvider("mock", mock_sandbox_client)],
    )

    new_config = client.config()
    new_config["plugins"] = [plugin]
    test_client = Client(**new_config)

    async with new_worker(
        test_client,
        SandboxE2EWorkflow,
        workflow_failure_exception_types=[Exception],
    ) as worker:
        result = await test_client.execute_workflow(
            SandboxE2EWorkflow.run,
            id=f"sandbox-e2e-{uuid.uuid4()}",
            task_queue=worker.task_queue,
            execution_timeout=timedelta(seconds=10),
        )

    assert result == "Done."
    # Full sandbox lifecycle exercised through Temporal activities
    assert mock_sandbox_client.create_calls == 1, "client.create() not called"
    assert mock_session.start_calls == 1, "session.start() not called"
    assert len(mock_session.exec_calls) >= 1, "session.exec() not called"
    assert len(mock_session.read_calls) >= 1, "session.read() not called"
    assert len(mock_session.write_calls) >= 1, "session.write() not called"
    assert mock_session.stop_calls >= 1, "session.stop() not called"
    assert mock_session.shutdown_calls >= 1, "session.shutdown() not called"
    assert mock_sandbox_client.delete_calls == 1, "client.delete() not called"


# ── OnHostSandboxClientProvider ──


class _DockerLikeClient(_MockSandboxClient):
    backend_id = "docker"


class _UnixLocalLikeClient(_MockSandboxClient):
    backend_id = "unix_local"


@pytest.mark.parametrize("client_type", [_DockerLikeClient, _UnixLocalLikeClient])
def test_sandbox_client_provider_refuses_docker_and_unix_local(
    client_type: type[_MockSandboxClient],
):
    with pytest.raises(ValueError, match="Use OnHostSandboxClientProvider"):
        SandboxClientProvider("local", client_type())
    # The on-host provider takes them.
    OnHostSandboxClientProvider("local", client_type())


def test_an_on_host_providers_task_queue_is_the_main_queue_slash_name_at_worker_name():
    named = OnHostSandboxClientProvider("docker", _MockSandboxClient(), worker_name="a")
    assert named._task_queue_for("agents") == "agents/docker@a"
    unnamed = OnHostSandboxClientProvider("docker", _MockSandboxClient())
    assert unnamed._task_queue_for("agents") == f"agents/docker@{socket.gethostname()}"


class _BackendSessionState(SandboxSessionState):
    """A backend's state with a field of its own.

    Workflow sandboxes re-import this module and register their own copy of the
    class by its ``type``, so a parsed state can be another copy: check ``type``,
    not ``isinstance``.
    """

    type: Literal["test_backend"] = "test_backend"  # type: ignore
    container_id: str = "backend-container"


def _lost_state() -> OnHostSandboxSessionState:
    return OnHostSandboxSessionState(
        manifest=Manifest(),
        snapshot=NoopSnapshot(id=str(uuid.uuid4())),
        type="test",
        task_queue="main/mock@gone",
    )


def _on_host_activities(
    client: _MockSandboxClient,
) -> tuple[OnHostSandboxClientProvider, dict[str, Any]]:
    """The provider's Activities, each run where it runs on a Worker on ``main``.

    Create runs on the main queue and every other Activity on the extra Worker's.
    """
    provider = OnHostSandboxClientProvider("mock", client, worker_name="box")

    def on_its_queue(name: str, act: Callable[..., Any]) -> Callable[..., Any]:
        env = ActivityEnvironment()
        env.info = dataclasses.replace(
            env.info,
            task_queue="main"
            if name == "mock-sandbox_client_create"
            else "main/mock@box",
        )
        return lambda *args: env.run(act, *args)

    return provider, {
        name: on_its_queue(name, act)
        for act in provider._get_activities(())
        for name in [act.__temporal_activity_definition.name]  # type: ignore[attr-defined, union-attr]
    }


_CREATE_ARGS = CreateSessionArgs(
    snapshot_spec=None, manifest=Manifest(), client_options=None
)


def _backend_session() -> _MockSandboxSession:
    session = _MockSandboxSession()
    session.state = _BackendSessionState(
        manifest=Manifest(), snapshot=NoopSnapshot(id=str(uuid.uuid4()))
    )
    return session


async def test_an_on_host_provider_returns_its_own_state_with_its_task_queue():
    session = _backend_session()
    _, acts = _on_host_activities(_MockSandboxClient(session))

    created = await acts["mock-sandbox_client_create"](_CREATE_ARGS)
    results = [
        created,
        await acts["mock-sandbox_session_start"](StartArgs(state=created.state)),
        await acts["mock-sandbox_session_stop"](StopArgs(state=created.state)),
        await acts["mock-sandbox_client_resume"](
            ResumeSessionArgs(state=created.state)
        ),
    ]

    for result in results:
        # As the Workflow receives it.
        state = SessionResult.model_validate(result.model_dump(mode="json")).state
        assert isinstance(state, OnHostSandboxSessionState)
        assert state.task_queue == "main/mock@box"
        assert state.session_id == session.state.session_id
        assert state.type == "test_backend"
        assert state.model_dump()["container_id"] == "backend-container"


async def test_an_on_host_provider_resumes_a_session_it_holds_without_the_backend():
    client = _MockSandboxClient()
    _, acts = _on_host_activities(client)
    created = await acts["mock-sandbox_client_create"](_CREATE_ARGS)

    resumed = await acts["mock-sandbox_client_resume"](
        ResumeSessionArgs(state=created.state)
    )

    assert resumed.state == created.state
    assert client.resume_calls == 0


async def test_an_on_host_provider_rebuilds_a_session_it_does_not_hold_with_the_backends_resume_once():
    session = _backend_session()
    client = _MockSandboxClient(session)
    _, acts = _on_host_activities(client)
    created = await acts["mock-sandbox_client_create"](_CREATE_ARGS)
    # The state as the Workflow keeps it, sent to a process that doesn't hold it,
    # such as this Worker after a restart.
    saved = ResumeSessionArgs.model_validate_json(
        ResumeSessionArgs(state=created.state).model_dump_json()
    )
    _, restarted = _on_host_activities(client)

    resumed = await restarted["mock-sandbox_client_resume"](saved)
    await restarted["mock-sandbox_session_exec"](
        ExecArgs(state=resumed.state, command=["true"])
    )
    await restarted["mock-sandbox_client_resume"](
        ResumeSessionArgs(state=resumed.state)
    )

    assert client.resume_calls == 1
    # The backend got its own state back.
    rebuilt = session.state.model_dump()
    assert rebuilt["type"] == "test_backend"
    assert rebuilt["container_id"] == "backend-container"
    assert rebuilt["session_id"] == created.state.session_id
    assert "task_queue" not in rebuilt
    assert isinstance(resumed.state, OnHostSandboxSessionState)
    assert resumed.state.task_queue == "main/mock@box"


class _DockerLikeSessionState(SandboxSessionState):
    """A backend state whose ``type`` allows a Docker-volume mount."""

    type: Literal["docker"] = "docker"  # type: ignore


class _MountCredentialCheckingClient(_MockSandboxClient):
    """Refuses to resume a state without its mount credentials, as OpenAI's backends do."""

    async def resume(self, state: SandboxSessionState) -> SandboxSession:
        state.assert_path_grants_rebound()
        return await super().resume(state)


def _credentialed_manifest() -> Manifest:
    return Manifest(
        entries={
            "data": S3Mount(
                bucket="bucket",
                access_key_id="key-id",
                secret_access_key="secret",
                mount_strategy=DockerVolumeMountStrategy(driver="rclone"),
            )
        }
    )


async def test_an_on_host_session_with_mount_credentials_comes_back_after_a_pause_across_a_restart(
    monkeypatch: pytest.MonkeyPatch,
):
    session = _MockSandboxSession()
    session.state = _DockerLikeSessionState(
        manifest=Manifest(), snapshot=NoopSnapshot(id=str(uuid.uuid4()))
    )
    client = _MountCredentialCheckingClient(session)
    _, acts = _on_host_activities(client)
    created = await acts["mock-sandbox_client_create"](
        CreateSessionArgs(manifest=_credentialed_manifest())
    )
    # As the Workflow receives it.
    state = SessionResult.model_validate_json(created.model_dump_json()).state
    await acts["mock-sandbox_session_exec"](ExecArgs(state=state, command=["true"]))

    # How OpenAI's Runner saves the state at a pause, restores it, and rebinds
    # its mount credentials from the run's manifest.
    workflow_client = TemporalSandboxClient("mock", backend_id="docker")
    restored = workflow_client.deserialize_session_state(
        json.loads(json.dumps(workflow_client.serialize_session_state(state)))
    )
    rebound = restored.rebind_persisted_mount_authority(
        _credentialed_manifest(), provider_backend_id=workflow_client.backend_id
    )
    # The resume reaches the Worker after a restart, so it doesn't hold the session.
    _, restarted = _on_host_activities(client)
    results: list[SessionResult] = []

    async def execute_activity(
        name: str,
        *,
        arg: ResumeSessionArgs,
        result_type: Any = None,  # type: ignore[reportUnusedParameter]
        **kwargs: Any,  # type: ignore[reportUnusedParameter]
    ) -> SessionResult:
        sent = ResumeSessionArgs.model_validate_json(arg.model_dump_json())
        result = await restarted[name](sent)
        results.append(SessionResult.model_validate_json(result.model_dump_json()))
        return results[-1]

    monkeypatch.setattr(workflow, "execute_activity", execute_activity)
    await workflow_client.resume(rebound)

    assert client.resume_calls == 1
    mount = session.state.manifest.entries["data"]
    assert isinstance(mount, S3Mount)
    assert mount.secret_access_key == "secret"
    await restarted["mock-sandbox_session_exec"](
        ExecArgs(state=results[0].state, command=["true"])
    )


async def test_a_hosted_session_with_mount_credentials_runs_on_a_worker_that_did_not_create_it(
    monkeypatch: pytest.MonkeyPatch,
):
    session = _MockSandboxSession()
    session.state = _DockerLikeSessionState(
        manifest=Manifest(), snapshot=NoopSnapshot(id=str(uuid.uuid4()))
    )
    client = _MountCredentialCheckingClient(session)
    creator = _activity_map(SandboxClientProvider("mock", client))
    other = _activity_map(SandboxClientProvider("mock", client))

    async def execute_activity(
        name: str,
        *,
        arg: BaseModel,
        result_type: Any,
        **kwargs: Any,  # type: ignore[reportUnusedParameter]
    ) -> Any:
        # Create runs on one Worker and the operation on another.
        acts = creator if name == "mock-sandbox_client_create" else other
        result = await acts[name](type(arg).model_validate_json(arg.model_dump_json()))
        return result_type.model_validate_json(result.model_dump_json())

    monkeypatch.setattr(workflow, "execute_activity", execute_activity)
    workflow_session = await TemporalSandboxClient("mock").create(
        manifest=_credentialed_manifest(), options=_TestSandboxClientOptions()
    )
    await workflow_session.exec("true")

    assert client.resume_calls == 1
    mount = session.state.manifest.entries["data"]
    assert isinstance(mount, S3Mount)
    assert mount.secret_access_key == "secret"


async def test_an_on_host_provider_does_not_resume_a_state_it_did_not_return():
    client = _MockSandboxClient()
    _, acts = _on_host_activities(client)

    with pytest.raises(ApplicationError) as err:
        await acts["mock-sandbox_client_resume"](ResumeSessionArgs(state=_make_state()))

    assert err.value.type == "SandboxLost"
    assert client.resume_calls == 0


_SESSION_OPERATIONS: list[tuple[str, Callable[[SandboxSessionState], _HasState]]] = [
    ("sandbox_session_exec", lambda s: ExecArgs(state=s, command=["true"])),
    ("sandbox_session_read", lambda s: ReadArgs(state=s, path="/f")),
    ("sandbox_session_write", lambda s: WriteArgs(state=s, path="/f", data=b"")),
    ("sandbox_session_running", lambda s: RunningArgs(state=s)),
    ("sandbox_session_persist_workspace", lambda s: PersistWorkspaceArgs(state=s)),
    (
        "sandbox_session_hydrate_workspace",
        lambda s: HydrateWorkspaceArgs(state=s, data=b""),
    ),
    (
        "sandbox_session_pty_exec_start",
        lambda s: PtyExecStartArgs(state=s, command=["true"]),
    ),
    (
        "sandbox_session_pty_write_stdin",
        lambda s: PtyWriteStdinArgs(state=s, session_id=1, chars=""),
    ),
    ("sandbox_session_start", lambda s: StartArgs(state=s)),
    ("sandbox_session_stop", lambda s: StopArgs(state=s)),
    ("sandbox_session_shutdown", lambda s: StopArgs(state=s)),
]


@pytest.mark.parametrize(
    "operation,args", _SESSION_OPERATIONS, ids=[op for op, _ in _SESSION_OPERATIONS]
)
async def test_an_operation_other_than_resume_on_a_session_an_on_host_provider_does_not_hold_fails_with_sandbox_lost(
    operation: str, args: Callable[[SandboxSessionState], _HasState]
):
    client = _MockSandboxClient()
    _, acts = _on_host_activities(client)

    with pytest.raises(ApplicationError) as err:
        await acts[f"mock-{operation}"](args(_lost_state()))

    assert err.value.type == "SandboxLost"
    assert err.value.non_retryable
    assert "'main/mock@gone'" in str(err.value)
    assert client.resume_calls == 0


async def test_deleting_a_session_an_on_host_provider_does_not_hold_does_nothing():
    client = _MockSandboxClient()
    _, acts = _on_host_activities(client)

    await acts["mock-sandbox_client_delete"](StopArgs(state=_lost_state()))

    assert client.delete_calls == 0
    assert client.resume_calls == 0


async def test_an_on_host_provider_keeps_a_shut_down_session_until_it_deletes_it():
    client = _MockSandboxClient()
    provider, acts = _on_host_activities(client)
    state = (await acts["mock-sandbox_client_create"](_CREATE_ARGS)).state

    await acts["mock-sandbox_session_shutdown"](StopArgs(state=state))
    assert client.inner_session.shutdown_calls == 1
    await acts["mock-sandbox_client_delete"](StopArgs(state=state))

    assert client.delete_calls == 1
    assert not provider._sessions
    with pytest.raises(ApplicationError, match="is lost"):
        await acts["mock-sandbox_session_exec"](ExecArgs(state=state, command=["x"]))


class _DeleteFailsOnceClient(_MockSandboxClient):
    def __init__(self, error: Exception) -> None:
        super().__init__()
        self.error = error

    async def delete(self, session: SandboxSession) -> SandboxSession:
        await super().delete(session)
        if self.delete_calls == 1:
            raise self.error
        return session


@pytest.mark.parametrize("non_retryable", [False, True], ids=["retryable", "terminal"])
async def test_an_on_host_provider_keeps_a_session_whose_delete_failed_only_for_a_retry(
    non_retryable: bool,
):
    client = _DeleteFailsOnceClient(
        ApplicationError("delete failed", non_retryable=non_retryable)
    )
    _, acts = _on_host_activities(client)
    state = (await acts["mock-sandbox_client_create"](_CREATE_ARGS)).state

    with pytest.raises(ApplicationError, match="delete failed"):
        await acts["mock-sandbox_client_delete"](StopArgs(state=state))
    # Temporal retries only the retryable error, but run it again either way.
    await acts["mock-sandbox_client_delete"](StopArgs(state=state))

    assert client.delete_calls == (1 if non_retryable else 2)


_ON_HOST_STATE = OnHostSandboxSessionState(
    manifest=Manifest(),
    snapshot=NoopSnapshot(id="on-host"),
    type="test",
    task_queue="main/mock@box",
)


def test_a_session_state_without_a_task_queue_keeps_the_activity_config():
    def config() -> ActivityConfig:
        return ActivityConfig(
            start_to_close_timeout=timedelta(seconds=30), task_queue="configured"
        )

    assert _session_activity_config(config(), _make_state()) == config()


@pytest.mark.parametrize("on_host", [True, False], ids=["on-host", "no-queue"])
async def test_the_workflow_sends_a_sessions_operations_to_the_task_queue_in_its_state(
    monkeypatch: pytest.MonkeyPatch, on_host: bool
):
    state = _ON_HOST_STATE if on_host else _make_state()
    results: dict[Any, Any] = {
        ExecResultModel: ExecResultModel(stdout=b"", stderr=b"", exit_code=0),
        ReadResult: ReadResult(data=b""),
        RunningResult: RunningResult(is_running=True),
        PersistWorkspaceResult: PersistWorkspaceResult(data=b""),
        PtyExecUpdateResult: PtyExecUpdateResult(
            process_id=None, output=b"", exit_code=0, original_token_count=None
        ),
        SessionResult: SessionResult(state=state, supports_pty=True),
        None: None,
    }
    calls: list[tuple[str, dict[str, Any]]] = []

    async def execute_activity(
        name: str,
        *,
        arg: Any,  # type: ignore[reportUnusedParameter]
        result_type: Any = None,
        **kwargs: Any,
    ) -> Any:
        calls.append((name, kwargs))
        return results[result_type]

    monkeypatch.setattr(workflow, "execute_activity", execute_activity)
    config = ActivityConfig(start_to_close_timeout=timedelta(seconds=30))
    client = TemporalSandboxClient("mock", config)
    session = TemporalSandboxSession("mock", config, state)

    await client.create(options=_TestSandboxClientOptions())
    await client.resume(state)
    await session.exec("true")
    await session.read(Path("/f"))
    await session.write(Path("/f"), io.BytesIO(b""))
    await session.running()
    await session.persist_workspace()
    await session.hydrate_workspace(io.BytesIO(b""))
    await session.pty_exec_start("true")
    await session.pty_write_stdin(session_id=1, chars="")
    await session.start()
    await session.stop()
    await session.shutdown()
    await client.delete(session)

    create, *operations = calls
    assert create == (
        "mock-sandbox_client_create",
        {"start_to_close_timeout": timedelta(seconds=30)},
    )
    assert len(operations) == 13
    for name, kwargs in operations:
        assert kwargs["start_to_close_timeout"] == timedelta(seconds=30), name
        assert kwargs.get("task_queue") == ("main/mock@box" if on_host else None), name


async def test_an_on_host_state_keeps_the_backends_fields_and_its_task_queue_through_a_saved_run_state():
    _, acts = _on_host_activities(_MockSandboxClient(_backend_session()))
    created = await acts["mock-sandbox_client_create"](_CREATE_ARGS)
    # As the Workflow receives it.
    state = SessionResult.model_validate_json(created.model_dump_json()).state
    client = TemporalSandboxClient("mock")

    # How OpenAI saves a session state in a RunState and reads it back.
    payload = json.loads(json.dumps(client.serialize_session_state(state)))
    restored = client.deserialize_session_state(payload)
    # And as an Activity argument or a continue-as-new input carries it.
    carried = ResumeSessionArgs.model_validate_json(
        ResumeSessionArgs(state=restored).model_dump_json()
    ).state

    for kept in (restored, carried):
        assert isinstance(kept, OnHostSandboxSessionState)
        assert kept.task_queue == "main/mock@box"
        assert kept.type == "test_backend"
        assert kept.model_dump()["container_id"] == "backend-container"


async def test_an_on_host_state_with_a_docker_volume_mount_passes_openais_mount_check(
    monkeypatch: pytest.MonkeyPatch,
):
    mount = S3Mount(
        bucket="bucket", mount_strategy=DockerVolumeMountStrategy(driver="rclone")
    )
    # As the Workflow receives it.
    state = SessionResult.model_validate_json(
        SessionResult(
            state=OnHostSandboxSessionState(
                type="docker",
                manifest=Manifest(entries={"data": mount}),
                snapshot=NoopSnapshot(id="on-host"),
                task_queue="main/mock@box",
            ),
            supports_pty=False,
        ).model_dump_json()
    ).state
    results: dict[Any, Any] = {
        SessionResult: SessionResult(state=state, supports_pty=False),
        PersistWorkspaceResult: PersistWorkspaceResult(data=b""),
    }

    async def execute_activity(
        name: str,  # type: ignore[reportUnusedParameter]
        *,
        arg: Any,  # type: ignore[reportUnusedParameter]
        result_type: Any = None,
        **kwargs: Any,  # type: ignore[reportUnusedParameter]
    ) -> Any:
        return results[result_type]

    monkeypatch.setattr(workflow, "execute_activity", execute_activity)
    session = await TemporalSandboxClient("mock").resume(state)

    # OpenAI checks the mounts against the session's backend before it persists
    # the workspace.
    await session.persist_workspace()


class _LifecycleSession(_MockSandboxSession):
    def __init__(
        self,
        client: "_LifecycleSandboxClient",
        number: int,
        state: SandboxSessionState | None = None,
    ) -> None:
        super().__init__()
        self.state = state or _BackendSessionState(
            manifest=Manifest(),
            snapshot=NoopSnapshot(id=str(uuid.uuid4())),
            container_id=f"container-{number}",
        )
        self._lifecycle = client
        self.number = number

    async def _exec_internal(
        self,
        *command: str | Path,
        timeout: float | None = None,
    ) -> ExecResult:
        commands = self._lifecycle.workspaces[self.number]
        commands.append(str(command[-1]))
        self._lifecycle.events.append(f"exec#{self.number}")
        output = f"#{self.number} ran {','.join(commands)}"
        return ExecResult(stdout=output.encode(), stderr=b"", exit_code=0)


class _LifecycleSandboxClient(_MockSandboxClient):
    """A machine's sandboxes: numbered sessions, and workspaces that outlive them.

    Records each lifecycle event. Like a backend's saved workspace, a session's
    commands survive its delete, and ``resume()`` finds them by the container id
    in its state.
    """

    def __init__(self) -> None:
        super().__init__()
        self.events: list[str] = []
        self.live: list[int] = []
        self.workspaces: dict[int, list[str]] = {}

    async def create(
        self,
        *,
        snapshot: Any = None,
        manifest: Manifest | None = None,
        options: BaseSandboxClientOptions | None = None,
    ) -> SandboxSession:
        number = len(self.workspaces) + 1
        self.workspaces[number] = []
        self.events.append(f"create#{number}")
        self.live.append(number)
        return self._wrap_session(_LifecycleSession(self, number))

    async def resume(self, state: SandboxSessionState) -> SandboxSession:
        saved = state.model_dump()
        assert saved["type"] == "test_backend"
        number = int(saved["container_id"].removeprefix("container-"))
        self.events.append(f"resume#{number}")
        self.live.append(number)
        return self._wrap_session(_LifecycleSession(self, number, state))

    async def delete(self, session: SandboxSession) -> SandboxSession:
        inner = session._inner
        assert isinstance(inner, _LifecycleSession)
        self.events.append(f"delete#{inner.number}")
        self.live.remove(inner.number)
        return session


@workflow.defn
class OnHostOperationsWorkflow:
    """Creates a session, waits for ``proceed``, runs each command, and deletes it.

    Returns the commands' outputs, or the failure type of the first that failed.
    """

    def __init__(self) -> None:
        self._created = False
        self._proceed = False

    @workflow.query
    def session_created(self) -> bool:
        return self._created

    @workflow.signal
    def proceed(self) -> None:
        self._proceed = True

    @workflow.run
    async def run(self, commands: list[str]) -> str:
        client = temporal_sandbox_client("lifecycle")
        session = await client.create(options=_TestSandboxClientOptions())
        self._created = True
        await workflow.wait_condition(lambda: self._proceed)
        outputs: list[str] = []
        try:
            for cmd in commands:
                outputs.append((await session.exec(cmd)).stdout.decode())
        except ActivityError as err:
            if not isinstance(err.cause, ApplicationError):
                raise
            outputs.append(f"failed: {err.cause.type}")
        finally:
            await client.delete(session)
        return "\n".join(outputs)


class _ApprovalCapability(Capability):
    """Runs commands through ``run_now``, and through ``run_approved`` after an approval."""

    def __init__(self) -> None:
        super().__init__(type="approval_test_sandbox")
        self._session: BaseSandboxSession | None = None

    def bind(self, session: BaseSandboxSession) -> None:
        self._session = session

    def tools(self) -> list[Tool]:
        session = self._session

        async def run(ctx: Any, args: str) -> str:  # type: ignore[reportUnusedParameter]
            result = await session.exec(json.loads(args)["cmd"])  # type: ignore[union-attr]
            return result.stdout.decode()

        params = {
            "type": "object",
            "properties": {"cmd": {"type": "string"}},
            "required": ["cmd"],
        }
        return [
            FunctionTool(
                name="run_now",
                description="Run a command",
                params_json_schema=params,
                on_invoke_tool=run,
            ),
            FunctionTool(
                name="run_approved",
                description="Run a command once approved",
                params_json_schema=params,
                on_invoke_tool=run,
                needs_approval=True,
            ),
        ]


@workflow.defn
class OnHostApprovalWorkflow:
    """Runs a sandbox agent until it pauses for approval, approves, and resumes.

    ``how`` is where it resumes: ``"in_place"``, after a ``continue_as_new``
    (``"continue_as_new"``), or once the ``resume`` signal arrives
    (``"on_signal"``). Returns the resumed run's command outputs.
    """

    def __init__(self) -> None:
        self._paused = False
        self._resume = False

    @workflow.query
    def paused(self) -> bool:
        return self._paused

    @workflow.signal
    def resume(self) -> None:
        self._resume = True

    @workflow.run
    async def run(self, how: str, saved: str | None = None) -> str:
        agent = SandboxAgent[None](
            name="on-host-approval", capabilities=[_ApprovalCapability()]
        )
        run_config = RunConfig(
            sandbox=SandboxRunConfig(
                client=temporal_sandbox_client("lifecycle"),
                options=_TestSandboxClientOptions(),
            )
        )
        if saved is None:
            result = await Runner.run(agent, "run commands", run_config=run_config)
            state = result.to_state()
            for interruption in result.interruptions:
                state.approve(interruption)
            if how == "continue_as_new":
                workflow.continue_as_new(args=[how, state.to_string()])
            self._paused = True
            if how == "on_signal":
                await workflow.wait_condition(lambda: self._resume)
        else:
            state = await RunState.from_string(agent, saved)
        resumed = await Runner.run(agent, state, run_config=run_config)
        return "\n".join(
            str(item.output)
            for item in resumed.new_items
            if item.type == "tool_call_output_item"
        )


_APPROVAL_RESPONSES = [
    ResponseBuilders.tool_call(
        '{"cmd": "first"}', "run_now", call_id="call-first", item_id="item-first"
    ),
    ResponseBuilders.tool_call(
        '{"cmd": "second"}',
        "run_approved",
        call_id="call-second",
        item_id="item-second",
    ),
    ResponseBuilders.output_message("Done."),
]


def _on_host_client(
    client: Client,
    backend: _LifecycleSandboxClient,
    worker_name: str,
    responses: list[ModelResponse] | None = None,
) -> Client:
    plugin = OpenAIAgentsPlugin(
        model_params=ModelActivityParameters(
            start_to_close_timeout=timedelta(seconds=30)
        ),
        model_provider=TestModelProvider(
            TestModel.returning_responses(responses or [])
        ),
        sandbox_clients=[
            OnHostSandboxClientProvider("lifecycle", backend, worker_name=worker_name)
        ],
    )
    config = client.config()
    config["plugins"] = [plugin]
    return Client(**config)


async def _until_created(handle: WorkflowHandle[Any, str]) -> None:
    while not await handle.query(OnHostOperationsWorkflow.session_created):
        await asyncio.sleep(0.1)


async def _scheduled_queues(handle: WorkflowHandle[Any, Any]) -> list[tuple[str, str]]:
    return [
        (
            event.activity_task_scheduled_event_attributes.activity_type.name,
            event.activity_task_scheduled_event_attributes.task_queue.name,
        )
        for event in (await handle.fetch_history()).events
        if event.HasField("activity_task_scheduled_event_attributes")
    ]


async def test_every_operation_on_an_on_host_sandbox_runs_on_the_worker_that_created_it(
    client: Client,
):
    first = _LifecycleSandboxClient()
    second = _LifecycleSandboxClient()
    commands = ["a", "b", "c", "d", "e", "f"]
    task_queue = str(uuid.uuid4())

    async with new_worker(
        _on_host_client(client, first, "A"),
        OnHostOperationsWorkflow,
        task_queue=task_queue,
        max_cached_workflows=0,
    ) as worker_a:
        # Only A runs, so A creates the sandbox.
        handle = await worker_a.client.start_workflow(
            OnHostOperationsWorkflow.run,
            commands,
            id=f"on-host-operations-{uuid.uuid4()}",
            task_queue=task_queue,
            execution_timeout=timedelta(seconds=30),
        )
        await _until_created(handle)
        # B, as another process would, has its own connection. It also polls the
        # shared queue, so it runs some of the Workflow Tasks and could take any
        # Activity sent there.
        other_connection = await Client.connect(
            client.service_client.config.target_host, namespace=client.namespace
        )
        async with new_worker(
            _on_host_client(other_connection, second, "B"),
            OnHostOperationsWorkflow,
            task_queue=task_queue,
            max_cached_workflows=0,
        ):
            await handle.signal(OnHostOperationsWorkflow.proceed)
            output = await handle.result()
        queues = await _scheduled_queues(handle)

    assert output == "\n".join(
        f"#1 ran {','.join(commands[: i + 1])}" for i in range(len(commands))
    )
    assert queues[0] == ("lifecycle-sandbox_client_create", task_queue)
    assert {queue for _, queue in queues[1:]} == {f"{task_queue}/lifecycle@A"}
    assert first.events == ["create#1", *["exec#1"] * len(commands), "delete#1"]
    assert second.events == []


async def test_a_restart_in_the_middle_of_a_step_loses_an_on_host_sandbox_and_deleting_it_does_nothing(
    client: Client,
):
    before = _LifecycleSandboxClient()
    after = _LifecycleSandboxClient()
    task_queue = str(uuid.uuid4())

    async with new_worker(
        _on_host_client(client, before, "restarted"),
        OnHostOperationsWorkflow,
        task_queue=task_queue,
    ) as worker:
        handle = await worker.client.start_workflow(
            OnHostOperationsWorkflow.run,
            ["a"],
            id=f"on-host-restart-{uuid.uuid4()}",
            task_queue=task_queue,
            execution_timeout=timedelta(seconds=30),
        )
        await _until_created(handle)

    # The new process has the same name, so it polls the same queue, but holds no
    # sandbox.
    async with new_worker(
        _on_host_client(client, after, "restarted"),
        OnHostOperationsWorkflow,
        task_queue=task_queue,
    ):
        await handle.signal(OnHostOperationsWorkflow.proceed)
        output = await handle.result()

    assert output == "failed: SandboxLost"
    assert after.events == []
    # The plugin doesn't clean up the sandbox the old process left behind.
    assert before.live == [1]
    failures = [
        event.activity_task_failed_event_attributes.failure
        for event in (await handle.fetch_history()).events
        if event.HasField("activity_task_failed_event_attributes")
    ]
    # SandboxLost is not retried, and the delete succeeds.
    assert len(failures) == 1
    assert failures[0].application_failure_info.non_retryable


async def test_two_workers_built_from_one_client_each_serve_their_own_on_host_sandboxes(
    client: Client,
):
    backend = _LifecycleSandboxClient()
    shared = _on_host_client(client, backend, "shared")
    task_queues = [f"{uuid.uuid4()}-first", f"{uuid.uuid4()}-second"]

    async def run(task_queue: str) -> tuple[str, list[tuple[str, str]]]:
        handle = await shared.start_workflow(
            OnHostOperationsWorkflow.run,
            ["a"],
            id=f"on-host-shared-{uuid.uuid4()}",
            task_queue=task_queue,
            execution_timeout=timedelta(seconds=30),
        )
        await _until_created(handle)
        await handle.signal(OnHostOperationsWorkflow.proceed)
        return await handle.result(), await _scheduled_queues(handle)

    async with (
        new_worker(shared, OnHostOperationsWorkflow, task_queue=task_queues[0]),
        new_worker(shared, OnHostOperationsWorkflow, task_queue=task_queues[1]),
    ):
        results = await asyncio.gather(*(run(q) for q in task_queues))

    assert sorted(output for output, _ in results) == ["#1 ran a", "#2 ran a"]
    for (_, queues), task_queue in zip(results, task_queues):
        assert queues[0] == ("lifecycle-sandbox_client_create", task_queue)
        assert {queue for _, queue in queues[1:]} == {f"{task_queue}/lifecycle@shared"}


async def _extra_worker_of(monkeypatch: pytest.MonkeyPatch, main: Worker) -> Worker:
    """Run ``main`` briefly and return the extra Worker it started."""
    created: list[Worker] = []

    class _Recorded(Worker):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            created.append(self)

    monkeypatch.setattr(
        "temporalio.openai_agents._temporal_openai_agents.Worker", _Recorded
    )
    async with main:
        while not created:
            await asyncio.sleep(0.01)
    [extra] = created
    return extra


async def test_an_on_host_providers_extra_worker_takes_its_main_workers_activity_setup(
    client: Client, monkeypatch: pytest.MonkeyPatch
):
    plugin_client = _on_host_client(client, _LifecycleSandboxClient(), "box")
    task_queue = str(uuid.uuid4())
    limited = await _extra_worker_of(
        monkeypatch,
        Worker(
            plugin_client,
            task_queue=task_queue,
            workflows=[OnHostOperationsWorkflow],
            max_concurrent_activities=3,
            max_activities_per_second=7,
            max_task_queue_activities_per_second=11,
            graceful_shutdown_timeout=timedelta(seconds=2),
        ),
    )
    config = limited.config()
    assert limited.task_queue == f"{task_queue}/lifecycle@box"
    assert config.get("max_concurrent_activities") == 3
    assert config.get("tuner") is None
    assert config.get("max_activities_per_second") == 7
    assert config.get("max_task_queue_activities_per_second") == 11
    assert config.get("graceful_shutdown_timeout") == timedelta(seconds=2)

    tuner = WorkerTuner.create_resource_based(
        target_memory_usage=0.8, target_cpu_usage=0.9
    )
    tuned = await _extra_worker_of(
        monkeypatch,
        Worker(
            plugin_client,
            task_queue=str(uuid.uuid4()),
            workflows=[OnHostOperationsWorkflow],
            tuner=tuner,
        ),
    )
    assert tuned.config().get("tuner") is tuner
    assert tuned.config().get("max_concurrent_activities") is None


@pytest.mark.parametrize("how", ["in_place", "continue_as_new"])
async def test_an_on_host_sandbox_comes_back_after_an_approval_pause(
    client: Client, how: str
):
    backend = _LifecycleSandboxClient()
    task_queue = str(uuid.uuid4())

    async with new_worker(
        _on_host_client(client, backend, "paused", _APPROVAL_RESPONSES),
        OnHostApprovalWorkflow,
        task_queue=task_queue,
    ) as worker:
        handle = await worker.client.start_workflow(
            OnHostApprovalWorkflow.run,
            args=[how, None],
            id=f"on-host-approval-{uuid.uuid4()}",
            task_queue=task_queue,
            execution_timeout=timedelta(seconds=30),
        )
        output = await handle.result()
        # The resumed run, after a continue-as-new.
        queues = await _scheduled_queues(worker.client.get_workflow_handle(handle.id))

    # The resumed run's items include the paused run's, and the second command
    # ran in the same workspace as the first.
    assert output == "#1 ran first\n#1 ran first,second"
    # OpenAI's SDK deletes the sandbox at the pause and resumes it after.
    assert backend.events == [
        "create#1",
        "exec#1",
        "delete#1",
        "resume#1",
        "exec#1",
        "delete#1",
    ]
    assert (
        "lifecycle-sandbox_client_resume",
        f"{task_queue}/lifecycle@paused",
    ) in queues
    for name, queue in queues:
        if name.startswith("lifecycle-") and name != "lifecycle-sandbox_client_create":
            assert queue == f"{task_queue}/lifecycle@paused", name


async def test_an_on_host_sandbox_paused_across_a_worker_restart_comes_back_through_the_backends_resume(
    client: Client,
):
    # The same machine, so the backend finds the sandbox's saved workspace.
    machine = _LifecycleSandboxClient()
    task_queue = str(uuid.uuid4())

    async with new_worker(
        _on_host_client(client, machine, "restarted", _APPROVAL_RESPONSES[:2]),
        OnHostApprovalWorkflow,
        task_queue=task_queue,
    ) as worker:
        handle = await worker.client.start_workflow(
            OnHostApprovalWorkflow.run,
            args=["on_signal", None],
            id=f"on-host-approval-restart-{uuid.uuid4()}",
            task_queue=task_queue,
            execution_timeout=timedelta(seconds=30),
        )
        while not await handle.query(OnHostApprovalWorkflow.paused):
            await asyncio.sleep(0.1)

    # A new process with the same name, which holds no sandbox.
    async with new_worker(
        _on_host_client(client, machine, "restarted", _APPROVAL_RESPONSES[2:]),
        OnHostApprovalWorkflow,
        task_queue=task_queue,
    ):
        await handle.signal(OnHostApprovalWorkflow.resume)
        output = await handle.result()

    # The resumed run's items include the paused run's, and the second command
    # ran in the same workspace as the first.
    assert output == "#1 ran first\n#1 ran first,second"
    assert machine.events == [
        "create#1",
        "exec#1",
        "delete#1",
        "resume#1",
        "exec#1",
        "delete#1",
    ]
    assert (
        "lifecycle-sandbox_client_resume",
        f"{task_queue}/lifecycle@restarted",
    ) in await _scheduled_queues(handle)


# ── JsonSafeBytes lossless serialization tests ──

# Payloads that exercise edge cases for bytes → JSON → bytes roundtrip.
_BYTE_PAYLOADS = [
    pytest.param(b"", id="empty"),
    pytest.param(b"hello world", id="ascii"),
    pytest.param(b"\xc3\xa9\xc3\xa0", id="valid-utf8"),  # éà
    pytest.param(bytes(range(256)), id="all-byte-values"),
    pytest.param(b"\xff\xfe\x80\x90\x00\x01", id="non-utf8-binary"),
    pytest.param(b"ok\nWarning: \xff\xfe binary \x80\x90\x00\x01", id="mixed"),
    pytest.param(b"\x00\x00\x00", id="null-bytes"),
]


def _roundtrip(model_cls: Any, **kwargs: Any) -> Any:
    """Serialize a model to JSON via pydantic_core and deserialize back."""
    json_bytes = to_json(model_cls(**kwargs))
    return TypeAdapter(model_cls).validate_json(json_bytes)


@pytest.mark.parametrize("payload", _BYTE_PAYLOADS)
def test_exec_result_bytes_roundtrip(payload: bytes):
    """ExecResult.stdout/stderr must survive a JSON roundtrip unchanged."""
    restored = _roundtrip(ExecResultModel, stdout=payload, stderr=payload, exit_code=1)
    assert restored.stdout == payload
    assert restored.stderr == payload
    assert restored.exit_code == 1


@pytest.mark.parametrize("payload", _BYTE_PAYLOADS)
def test_pty_exec_update_result_bytes_roundtrip(payload: bytes):
    """PtyExecUpdateResult.output must survive a JSON roundtrip unchanged."""
    restored = _roundtrip(
        PtyExecUpdateResult,
        process_id=1,
        output=payload,
        exit_code=0,
        original_token_count=None,
    )
    assert restored.output == payload


@pytest.mark.parametrize("payload", _BYTE_PAYLOADS)
def test_read_result_bytes_roundtrip(payload: bytes):
    """ReadResult.data must survive a JSON roundtrip unchanged."""
    restored = _roundtrip(ReadResult, data=payload)
    assert restored.data == payload


@pytest.mark.parametrize("payload", _BYTE_PAYLOADS)
def test_persist_workspace_result_bytes_roundtrip(payload: bytes):
    """PersistWorkspaceResult.data must survive a JSON roundtrip unchanged."""
    restored = _roundtrip(PersistWorkspaceResult, data=payload)
    assert restored.data == payload


@pytest.mark.parametrize("payload", _BYTE_PAYLOADS)
def test_write_args_bytes_roundtrip(payload: bytes):
    """WriteArgs.data must survive a JSON roundtrip unchanged (workflow → activity)."""
    restored = _roundtrip(WriteArgs, state=_make_state(), path="/tmp/f", data=payload)
    assert restored.data == payload


@pytest.mark.parametrize("payload", _BYTE_PAYLOADS)
def test_hydrate_workspace_args_bytes_roundtrip(payload: bytes):
    """HydrateWorkspaceArgs.data must survive a JSON roundtrip unchanged."""
    restored = _roundtrip(HydrateWorkspaceArgs, state=_make_state(), data=payload)
    assert restored.data == payload
