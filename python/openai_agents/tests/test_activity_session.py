"""Unit tests for OpenAIAgentsPlugin activity session support."""

from __future__ import annotations

from datetime import timedelta
from types import SimpleNamespace
from typing import Any, cast

import pytest
from agents import Model, ModelProvider

import temporalio.workflow
from temporalio import activity, workflow
from temporalio.common import ActivitySessionOptions, Priority, UseWorkflowSession
from temporalio.openai_agents import ModelActivityParameters, OpenAIAgentsPlugin
from temporalio.openai_agents._activity_session_interceptor import (
    _ActivitySessionInterceptor,
    _ActivitySessionOutboundInterceptor,
)
from temporalio.openai_agents._trace_interceptor import (
    OpenAIAgentsContextPropagationInterceptor,
)
from temporalio.plugin import SimplePlugin
from temporalio.worker import (
    Replayer,
    ReplayerConfig,
    StartActivityInput,
    StartLocalActivityInput,
    WorkerConfig,
    WorkerSessionConfig,
    WorkflowInboundInterceptor,
    WorkflowInterceptorClassInput,
    WorkflowOutboundInterceptor,
)
from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner

OPTIONS = ActivitySessionOptions(placement=UseWorkflowSession())


@workflow.defn
class _TrivialWorkflow:
    @workflow.run
    async def run(self) -> None:
        pass


@activity.defn
async def _trivial_activity() -> None:
    pass


class _RecordingOutbound:
    def __init__(self) -> None:
        self.activity_inputs: list[StartActivityInput] = []
        self.local_activity_inputs: list[StartLocalActivityInput] = []

    def start_activity(self, input: StartActivityInput) -> Any:
        self.activity_inputs.append(input)
        return "activity-handle"

    def start_local_activity(self, input: StartLocalActivityInput) -> Any:
        self.local_activity_inputs.append(input)
        return "local-activity-handle"


class _NoModelProvider(ModelProvider):
    def get_model(self, model_name: str | None) -> Model:
        raise NotImplementedError


def _plugin(**kwargs: Any) -> OpenAIAgentsPlugin:
    return OpenAIAgentsPlugin(
        model_params=ModelActivityParameters(
            start_to_close_timeout=timedelta(seconds=10)
        ),
        model_provider=_NoModelProvider(),
        **kwargs,
    )


class _FakeClient:
    def config(self, **_kwargs: Any) -> dict[str, Any]:
        return {}


@pytest.fixture
def workflow_info(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        temporalio.workflow, "info", lambda: SimpleNamespace(task_queue="wf-tq")
    )


def _activity_input(
    task_queue: str | None,
    session_options: ActivitySessionOptions | None = None,
) -> StartActivityInput:
    return StartActivityInput(
        activity="act",
        args=[],
        activity_id=None,
        task_queue=task_queue,
        schedule_to_close_timeout=None,
        schedule_to_start_timeout=None,
        start_to_close_timeout=timedelta(seconds=10),
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


def _outbound() -> tuple[_ActivitySessionOutboundInterceptor, _RecordingOutbound]:
    recorder = _RecordingOutbound()
    return (
        _ActivitySessionOutboundInterceptor(
            cast(WorkflowOutboundInterceptor, cast(object, recorder)), OPTIONS
        ),
        recorder,
    )


def _session_interceptors(interceptors: Any) -> list[Any]:
    return [i for i in interceptors or [] if isinstance(i, _ActivitySessionInterceptor)]


def _worker_config(**kwargs: Any) -> WorkerConfig:
    return cast(
        WorkerConfig,
        cast(
            object,
            {
                "client": _FakeClient(),
                "workflow_runner": SandboxedWorkflowRunner(),
                **kwargs,
            },
        ),
    )


def _replayer_config(**kwargs: Any) -> ReplayerConfig:
    return cast(
        ReplayerConfig,
        cast(
            object,
            {
                "workflows": [_TrivialWorkflow],
                "workflow_runner": SandboxedWorkflowRunner(),
                **kwargs,
            },
        ),
    )


@pytest.mark.usefixtures("workflow_info")
def test_fills_when_task_queue_unset() -> None:
    outbound, recorder = _outbound()
    assert outbound.start_activity(_activity_input(None)) == "activity-handle"
    assert recorder.activity_inputs[0].session_options is OPTIONS


@pytest.mark.usefixtures("workflow_info")
def test_fills_when_task_queue_matches_workflow() -> None:
    outbound, recorder = _outbound()
    outbound.start_activity(_activity_input("wf-tq"))
    assert recorder.activity_inputs[0].session_options is OPTIONS


@pytest.mark.usefixtures("workflow_info")
def test_does_not_fill_other_task_queue() -> None:
    outbound, recorder = _outbound()
    outbound.start_activity(_activity_input("other-tq"))
    assert recorder.activity_inputs[0].session_options is None


@pytest.mark.usefixtures("workflow_info")
def test_keeps_existing_session_options() -> None:
    outbound, recorder = _outbound()
    existing = ActivitySessionOptions(placement=UseWorkflowSession())
    outbound.start_activity(_activity_input(None, existing))
    assert recorder.activity_inputs[0].session_options is existing


@pytest.mark.usefixtures("workflow_info")
def test_local_activity_passes_through() -> None:
    outbound, recorder = _outbound()
    local_input = StartLocalActivityInput(
        activity="act",
        args=[],
        activity_id=None,
        schedule_to_close_timeout=None,
        schedule_to_start_timeout=None,
        start_to_close_timeout=timedelta(seconds=10),
        retry_policy=None,
        local_retry_threshold=None,
        cancellation_type=workflow.ActivityCancellationType.TRY_CANCEL,
        headers={},
        summary=None,
    )
    assert outbound.start_local_activity(local_input) == "local-activity-handle"
    assert recorder.local_activity_inputs == [local_input]
    assert "session_options" not in vars(local_input)


def test_interceptor_wraps_outbound() -> None:
    inbound_class = _ActivitySessionInterceptor(OPTIONS).workflow_interceptor_class(
        cast(WorkflowInterceptorClassInput, cast(object, None))
    )
    assert inbound_class is not None
    recorder = _RecordingOutbound()
    captured: list[WorkflowOutboundInterceptor] = []
    inbound = inbound_class(
        cast(
            WorkflowInboundInterceptor,
            cast(object, SimpleNamespace(init=captured.append)),
        )
    )
    inbound.init(cast(WorkflowOutboundInterceptor, cast(object, recorder)))
    assert isinstance(captured[0], _ActivitySessionOutboundInterceptor)
    assert captured[0].next is recorder


def test_default_plugin_is_unchanged() -> None:
    plugin = _plugin()
    assert plugin.interceptors is not None
    interceptors = list(plugin.interceptors)
    assert len(interceptors) == 1
    assert isinstance(interceptors[0], OpenAIAgentsContextPropagationInterceptor)

    worker_config = plugin.configure_worker(
        _worker_config(workflows=[_TrivialWorkflow])
    )
    assert not _session_interceptors(worker_config.get("interceptors"))
    replayer_config = plugin.configure_replayer(_replayer_config())
    assert not _session_interceptors(replayer_config.get("interceptors"))


@pytest.mark.parametrize(
    "extra",
    [{"workflows": [_TrivialWorkflow]}, {"activities": [_trivial_activity]}],
    ids=["workflows", "activity-only"],
)
def test_configure_worker_requires_session_config(extra: dict[str, Any]) -> None:
    plugin = _plugin(session_options=OPTIONS)
    with pytest.raises(ValueError, match="OpenAIAgentsPlugin") as err:
        plugin.configure_worker(_worker_config(**extra))
    message = str(err.value)
    assert "session_config=temporalio.worker.WorkerSessionConfig(...)" in message
    assert "Worker(...)/Replayer(...)" in message
    assert "plugin listed before OpenAIAgentsPlugin" in message


def test_configure_worker_requires_session_config_without_activities() -> None:
    plugin = _plugin(session_options=OPTIONS, register_activities=False)
    with pytest.raises(ValueError, match="requires session_config"):
        plugin.configure_worker(_worker_config(workflows=[_TrivialWorkflow]))


def test_configure_replayer_requires_session_config() -> None:
    plugin = _plugin(session_options=OPTIONS)
    with pytest.raises(ValueError, match="requires session_config"):
        plugin.configure_replayer(_replayer_config())


def test_replayer_construction_requires_session_config() -> None:
    plugin = _plugin(session_options=OPTIONS)
    with pytest.raises(ValueError, match="requires session_config"):
        Replayer(workflows=[_TrivialWorkflow], plugins=[plugin])


def test_session_config_satisfies_check() -> None:
    plugin = _plugin(session_options=OPTIONS)
    session_config = WorkerSessionConfig()

    worker_config = plugin.configure_worker(
        _worker_config(workflows=[_TrivialWorkflow], session_config=session_config)
    )
    assert len(_session_interceptors(worker_config.get("interceptors"))) == 1

    replayer_config = plugin.configure_replayer(
        _replayer_config(session_config=session_config)
    )
    assert len(_session_interceptors(replayer_config.get("interceptors"))) == 1

    replayer = Replayer(
        workflows=[_TrivialWorkflow],
        plugins=[plugin],
        session_config=session_config,
    )
    assert (
        len(
            _session_interceptors(
                replayer.config(active_config=True).get("interceptors")
            )
        )
        == 1
    )


class _SessionConfigPlugin(SimplePlugin):
    def __init__(self) -> None:
        super().__init__(name="SessionConfigPlugin")

    def configure_worker(self, config: WorkerConfig) -> WorkerConfig:
        config = super().configure_worker(config)
        config["session_config"] = WorkerSessionConfig()
        return config

    def configure_replayer(self, config: ReplayerConfig) -> ReplayerConfig:
        config = super().configure_replayer(config)
        config["session_config"] = WorkerSessionConfig()
        return config


def test_earlier_plugin_can_supply_session_config() -> None:
    setter = _SessionConfigPlugin()
    plugin = _plugin(session_options=OPTIONS)

    worker_config = plugin.configure_worker(
        setter.configure_worker(_worker_config(workflows=[_TrivialWorkflow]))
    )
    assert worker_config.get("session_config") is not None

    Replayer(workflows=[_TrivialWorkflow], plugins=[setter, plugin])

    with pytest.raises(ValueError, match="requires session_config"):
        Replayer(workflows=[_TrivialWorkflow], plugins=[plugin, setter])
