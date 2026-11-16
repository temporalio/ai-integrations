"""Unit tests for ``StrandsPlugin(session_options=...)``."""

from __future__ import annotations

from datetime import timedelta
from types import SimpleNamespace
from typing import Any, cast

import pytest

import temporalio.workflow
from temporalio import workflow
from temporalio.common import ActivitySessionOptions, Priority, UseWorkflowSession
from temporalio.plugin import SimplePlugin
from temporalio.strands_agents import StrandsPlugin
from temporalio.strands_agents._activity_session_interceptor import (
    _ActivitySessionInterceptor,
    _ActivitySessionOutboundInterceptor,
)
from temporalio.worker import (
    Replayer,
    ReplayerConfig,
    StartActivityInput,
    UnsandboxedWorkflowRunner,
    WorkerConfig,
    WorkerSessionConfig,
)
from tests.mock_model import MockModel

SESSION = ActivitySessionOptions(placement=UseWorkflowSession())


@workflow.defn
class _NoopWorkflow:
    @workflow.run
    async def run(self) -> None:
        return None


class _RecordingOutbound:
    def __init__(self) -> None:
        self.activities: list[Any] = []
        self.local_activities: list[Any] = []

    def start_activity(self, input: Any) -> Any:
        self.activities.append(input)
        return "handle"

    def start_local_activity(self, input: Any) -> Any:
        self.local_activities.append(input)
        return "local-handle"


class _FakeClient:
    def config(self, **_kwargs: Any) -> dict[str, Any]:
        return {}


def _plugin(**kwargs: Any) -> StrandsPlugin:
    return StrandsPlugin(models={"m": lambda: MockModel([])}, **kwargs)


def _input(task_queue: str | None, session_options: Any = None) -> StartActivityInput:
    return StartActivityInput(
        activity="act",
        args=[],
        activity_id=None,
        task_queue=task_queue,
        schedule_to_close_timeout=None,
        schedule_to_start_timeout=None,
        start_to_close_timeout=timedelta(seconds=1),
        heartbeat_timeout=None,
        retry_policy=None,
        cancellation_type=workflow.ActivityCancellationType.TRY_CANCEL,
        headers={},
        disable_eager_execution=False,
        versioning_intent=None,
        summary=None,
        priority=Priority.default,
        arg_types=None,
        ret_type=None,
        session_options=session_options,
    )


@pytest.fixture
def outbound(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[_ActivitySessionOutboundInterceptor, _RecordingOutbound]:
    monkeypatch.setattr(
        temporalio.workflow, "info", lambda: SimpleNamespace(task_queue="wf-tq")
    )
    recorder = _RecordingOutbound()
    return _ActivitySessionOutboundInterceptor(recorder, SESSION), recorder  # type: ignore[arg-type]


def _worker_config(**kwargs: Any) -> WorkerConfig:
    config: dict[str, Any] = {
        "client": _FakeClient(),
        "task_queue": "wf-tq",
        "workflow_runner": UnsandboxedWorkflowRunner(),
        **kwargs,
    }
    return cast(WorkerConfig, cast(object, config))


def _replayer_config(**kwargs: Any) -> ReplayerConfig:
    config: dict[str, Any] = {
        "workflows": [_NoopWorkflow],
        "workflow_runner": UnsandboxedWorkflowRunner(),
        **kwargs,
    }
    return cast(ReplayerConfig, cast(object, config))


def _session_interceptors(config: WorkerConfig | ReplayerConfig) -> list[Any]:
    return [
        i
        for i in config.get("interceptors") or []
        if isinstance(i, _ActivitySessionInterceptor)
    ]


def test_fills_when_task_queue_and_session_unset(outbound: Any) -> None:
    interceptor, recorder = outbound
    assert interceptor.start_activity(_input(None)) == "handle"
    assert recorder.activities[0].session_options is SESSION


def test_fills_when_task_queue_matches_workflow(outbound: Any) -> None:
    interceptor, recorder = outbound
    interceptor.start_activity(_input("wf-tq"))
    assert recorder.activities[0].session_options is SESSION


def test_leaves_other_task_queue_unset(outbound: Any) -> None:
    interceptor, recorder = outbound
    interceptor.start_activity(_input("other-tq"))
    assert recorder.activities[0].session_options is None


def test_keeps_preset_session_options(outbound: Any) -> None:
    interceptor, recorder = outbound
    preset = ActivitySessionOptions(placement=UseWorkflowSession())
    interceptor.start_activity(_input(None, preset))
    assert recorder.activities[0].session_options is preset


def test_local_activity_untouched(outbound: Any) -> None:
    interceptor, recorder = outbound
    local_input = object()
    assert interceptor.start_local_activity(local_input) == "local-handle"
    assert recorder.local_activities == [local_input]
    assert recorder.activities == []


def test_workflow_interceptor_class_wraps_outbound() -> None:
    interceptor = _ActivitySessionInterceptor(SESSION)
    inbound_cls = interceptor.workflow_interceptor_class(None)  # type: ignore[arg-type]
    assert inbound_cls is not None
    captured: list[Any] = []

    class _Next:
        def init(self, outbound: Any) -> None:
            captured.append(outbound)

    inbound_cls(_Next()).init(_RecordingOutbound())  # type: ignore[arg-type]
    assert isinstance(captured[0], _ActivitySessionOutboundInterceptor)


def test_default_plugin_has_no_session_interceptor() -> None:
    plugin = _plugin()
    assert plugin.interceptors is None
    worker_config = plugin.configure_worker(_worker_config(workflows=[_NoopWorkflow]))
    assert _session_interceptors(worker_config) == []
    replayer_config = plugin.configure_replayer(_replayer_config())
    assert _session_interceptors(replayer_config) == []
    Replayer(workflows=[_NoopWorkflow], plugins=[plugin])


@pytest.mark.parametrize(
    "extra",
    [{"workflows": [_NoopWorkflow]}, {"activities": [lambda: None]}],
    ids=["workflows", "activity-only"],
)
def test_configure_worker_requires_session_config(extra: dict[str, Any]) -> None:
    plugin = _plugin(session_options=SESSION)
    with pytest.raises(ValueError, match=r"StrandsPlugin.*Worker\(\.\.\.\)"):
        plugin.configure_worker(_worker_config(**extra))


def test_replayer_requires_session_config() -> None:
    plugin = _plugin(session_options=SESSION)
    with pytest.raises(ValueError, match=r"WorkerSessionConfig.*Replayer\(\.\.\.\)"):
        plugin.configure_replayer(_replayer_config())
    with pytest.raises(ValueError, match="plugin listed before StrandsPlugin"):
        Replayer(workflows=[_NoopWorkflow], plugins=[plugin])


def test_session_config_present_succeeds() -> None:
    plugin = _plugin(session_options=SESSION)
    worker_config = plugin.configure_worker(
        _worker_config(workflows=[_NoopWorkflow], session_config=WorkerSessionConfig())
    )
    assert len(_session_interceptors(worker_config)) == 1
    assert _session_interceptors(worker_config)[0].options is SESSION
    replayer_config = plugin.configure_replayer(
        _replayer_config(session_config=WorkerSessionConfig())
    )
    assert len(_session_interceptors(replayer_config)) == 1
    Replayer(
        workflows=[_NoopWorkflow],
        plugins=[plugin],
        session_config=WorkerSessionConfig(),
    )


class _SessionConfigPlugin(SimplePlugin):
    def __init__(self) -> None:
        super().__init__("test.SessionConfigPlugin")

    def configure_worker(self, config: WorkerConfig) -> WorkerConfig:
        config = super().configure_worker(config)
        config["session_config"] = WorkerSessionConfig()
        return config

    def configure_replayer(self, config: ReplayerConfig) -> ReplayerConfig:
        config = super().configure_replayer(config)
        config["session_config"] = WorkerSessionConfig()
        return config


def test_preceding_plugin_session_config_satisfies_check() -> None:
    plugin = _plugin(session_options=SESSION)
    earlier = _SessionConfigPlugin()
    worker_config = plugin.configure_worker(
        earlier.configure_worker(_worker_config(workflows=[_NoopWorkflow]))
    )
    assert len(_session_interceptors(worker_config)) == 1
    Replayer(workflows=[_NoopWorkflow], plugins=[earlier, plugin])
    with pytest.raises(ValueError):
        Replayer(workflows=[_NoopWorkflow], plugins=[plugin, earlier])
