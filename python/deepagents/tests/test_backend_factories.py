"""Backend configuration reaches an activity worker in a separate process."""

from __future__ import annotations

import asyncio
import inspect
import sys
import uuid
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable

import pytest

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.converter import DataConverter
from temporalio.deepagents import DeepAgentsPlugin, TemporalBackend
from temporalio.deepagents._activity import BackendOpInput, DeepAgentActivities
from temporalio.deepagents._tools import lookup_backend, registered_backends
from temporalio.exceptions import ApplicationError
from temporalio.testing import ActivityEnvironment, WorkflowEnvironment
from temporalio.worker import Replayer, Worker
from temporalio.workflow import ActivityConfig
from tests.helpers.deepagents import count_scheduled_activities

_backends = pytest.importorskip("deepagents.backends")
FilesystemBackend = _backends.FilesystemBackend
LocalShellBackend = _backends.LocalShellBackend


@workflow.defn
class FactoryFilesystemWorkflow:
    @workflow.run
    async def run(self, config: dict[str, Any]) -> str:
        backend = TemporalBackend(
            config["factory"],
            factory_argument=(
                None
                if config["factory"] == "worker-filesystem"
                else {"root_dir": config["root_dir"], "virtual_mode": True}
            ),
            backend_type=FilesystemBackend,
            activity_options=ActivityConfig(
                start_to_close_timeout=timedelta(seconds=10),
                retry_policy=RetryPolicy(
                    initial_interval=timedelta(milliseconds=1), maximum_attempts=2
                ),
            ),
        )
        written = await backend.awrite("/notes.txt", "remote hello")
        assert written.error is None
        result = await backend.aread("/notes.txt")
        listing = await backend.als("/")
        assert listing.error is None
        assert any(item["path"] == "/notes.txt" for item in listing.entries)
        return str(result)


@workflow.defn
class FactoryShellWorkflow:
    @workflow.run
    async def run(self, root_dir: str) -> str:
        backend = TemporalBackend(
            "shell",
            factory_argument=root_dir,
            backend_type=LocalShellBackend,
            metadata={"id": "remote-sandbox"},
            activity_options=ActivityConfig(
                start_to_close_timeout=timedelta(seconds=10)
            ),
        )
        assert backend.id == "remote-sandbox"
        result = await backend.aexecute("echo remote-shell", timeout=5)
        return result.output


@pytest.mark.parametrize(
    "factory_name",
    [
        "filesystem",
        "worker-filesystem",
        "async-filesystem",
        "retry-filesystem",
        "shell",
    ],
)
async def test_backend_factory_on_another_process(
    env: WorkflowEnvironment, tmp_path: Path, factory_name: str
) -> None:
    """No workflow-side registry or backend construction is available to activities.

    Cache eviction and replay also happen without any factory configured on the
    workflow worker, including replay after the activity process has exited.
    """
    task_queue = f"da-factory-{uuid.uuid4()}"
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "tests.helpers.backend_worker",
        env.client.service_client.config.target_host,
        task_queue,
        str(tmp_path),
        stdout=asyncio.subprocess.PIPE,
    )
    try:
        assert process.stdout is not None
        assert (
            await asyncio.wait_for(process.stdout.readline(), timeout=30) == b"READY\n"
        )
        workflow_class = (
            FactoryShellWorkflow
            if factory_name == "shell"
            else FactoryFilesystemWorkflow
        )
        argument: Any = (
            str(tmp_path)
            if factory_name == "shell"
            else {"root_dir": str(tmp_path), "factory": factory_name}
        )
        async with Worker(
            env.client,
            task_queue=task_queue,
            workflows=[workflow_class],
            plugins=[DeepAgentsPlugin()],
            no_remote_activities=True,
            max_cached_workflows=0,
            workflow_failure_exception_types=[Exception],
        ):
            handle = await env.client.start_workflow(
                workflow_class.run,
                argument,
                id=f"da-factory-{uuid.uuid4()}",
                task_queue=task_queue,
                execution_timeout=timedelta(seconds=30),
            )
            result = await handle.result()
        if factory_name == "shell":
            assert "remote-shell" in result
        else:
            assert "remote hello" in result
            assert (tmp_path / "notes.txt").read_text() == "remote hello"
        counts = await count_scheduled_activities(handle)
        assert counts["deepagents.backend_op"] == (1 if factory_name == "shell" else 3)
    finally:
        if process.returncode is None:
            process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=10)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()

    await Replayer(
        workflows=[workflow_class], plugins=[DeepAgentsPlugin()]
    ).replay_workflow(await handle.fetch_history())


def test_factory_wrapper_capabilities_without_construction() -> None:
    protocol = pytest.importorskip("deepagents.backends.protocol")
    supports_execution = pytest.importorskip(
        "deepagents.middleware.filesystem"
    ).supports_execution
    before = dict(registered_backends())
    filesystem = TemporalBackend("filesystem", backend_type=FilesystemBackend)
    shell = TemporalBackend(
        "shell", backend_type=LocalShellBackend, metadata={"id": "sandbox-123"}
    )
    assert registered_backends() == before
    assert lookup_backend(shell._ref) is None
    assert shell.id == "sandbox-123"
    assert supports_execution(filesystem) is False
    assert not hasattr(filesystem, "aexecute")
    assert supports_execution(shell) is True
    assert protocol.execute_accepts_timeout(type(shell)) is True
    assert inspect.signature(getattr(type(shell), "execute")) == inspect.signature(
        LocalShellBackend.execute
    )
    assert protocol._supports_delete(filesystem) is True

    class NeverConstruct:
        def __init__(self) -> None:
            raise AssertionError("Backend construction ran on the workflow side")

    assert (
        protocol._supports_delete(
            TemporalBackend("custom", backend_type=NeverConstruct)
        )
        is False
    )


@pytest.mark.parametrize("async_factory", [False, True])
async def test_factory_is_called_for_each_operation_and_is_worker_local(
    async_factory: bool,
) -> None:
    seen: list[dict[str, Any]] = []

    class Backend:
        def read(self, path: str) -> str:
            return f"{path}:{len(seen)}"

    def build(config: dict[str, Any]) -> Backend:
        seen.append(config)
        return Backend()

    async def async_build(config: dict[str, Any]) -> Backend:
        return build(config)

    activities = DeepAgentActivities(
        backends={"remote": async_build if async_factory else build}
    )
    input = BackendOpInput(
        backend_ref="unused-ref",
        op="read",
        args=["notes.txt"],
        backend_factory="remote",
        factory_argument={"sandbox_id": "sandbox-123"},
    )
    env = ActivityEnvironment()
    assert (await env.run(activities.backend_op, input)).result == "notes.txt:1"
    assert (await env.run(activities.backend_op, input)).result == "notes.txt:2"
    assert seen == [{"sandbox_id": "sandbox-123"}] * 2
    # Factories on one worker must not be visible to another plugin instance.
    with pytest.raises(ApplicationError) as exc:
        await env.run(DeepAgentActivities().backend_op, input)
    assert exc.value.type == "DeepAgentsUnknownBackendFactory"
    assert exc.value.non_retryable is True


@pytest.mark.parametrize("async_factory", [False, True])
async def test_parameterless_backend_factory(async_factory: bool) -> None:
    class Backend:
        def read(self) -> str:
            return "configured-on-worker"

    async def async_build() -> Backend:
        return Backend()

    factory: Callable[[], Any]
    if async_factory:
        factory = async_build
    else:
        factory = lambda: Backend()
    plugin = DeepAgentsPlugin(backends={"remote": factory})
    input = BackendOpInput(backend_ref="unused", op="read", backend_factory="remote")
    env = ActivityEnvironment()
    assert (
        await env.run(plugin._activities.backend_op, input)
    ).result == "configured-on-worker"
    input.factory_argument = "unexpected"
    with pytest.raises(ApplicationError) as exc:
        await env.run(plugin._activities.backend_op, input)
    assert exc.value.type == "DeepAgentsInvalidBackendFactoryArgument"
    assert exc.value.non_retryable is True


async def test_one_parameter_backend_factory_receives_none_by_default() -> None:
    received: list[Any] = []

    class Backend:
        def read(self) -> str:
            return "result"

    def factory(argument: Any = "factory-default") -> Backend:
        received.append(argument)
        return Backend()

    activities = DeepAgentActivities(backends={"remote": factory})
    await ActivityEnvironment().run(
        activities.backend_op,
        BackendOpInput(backend_ref="unused", op="read", backend_factory="remote"),
    )
    assert received == [None]


def test_invalid_backend_factory_is_rejected_at_registration() -> None:
    def two_parameters(first: Any, second: Any) -> Any:
        return first, second

    def keyword_only(*, argument: Any) -> Any:
        return argument

    for factory in (two_parameters, keyword_only):
        with pytest.raises(TypeError, match="one positional factory_argument"):
            DeepAgentsPlugin(backends={"remote": factory})


def test_named_backend_requires_capabilities_and_instance_rejects_factory_options() -> (
    None
):
    with pytest.raises(ValueError, match="backend_type is required"):
        TemporalBackend("remote")
    with pytest.raises(ValueError, match="require a named backend factory"):
        TemporalBackend(object(), factory_argument="sandbox-123")


async def test_legacy_backend_input_decodes_with_factory_defaults() -> None:
    converter = DataConverter.default
    payloads = await converter.encode(
        [{"backend_ref": "legacy", "op": "read", "args": ["notes.txt"], "kwargs": {}}]
    )
    (input,) = await converter.decode(payloads, [BackendOpInput])
    assert input == BackendOpInput(backend_ref="legacy", op="read", args=["notes.txt"])


async def test_factory_backend_unknown_operation() -> None:
    activities = DeepAgentActivities(backends={"remote": lambda _: object()})
    with pytest.raises(ApplicationError) as exc:
        await ActivityEnvironment().run(
            activities.backend_op,
            BackendOpInput(backend_ref="unused", op="read", backend_factory="remote"),
        )
    assert exc.value.type == "DeepAgentsUnknownBackendOp"
    assert exc.value.non_retryable is True
