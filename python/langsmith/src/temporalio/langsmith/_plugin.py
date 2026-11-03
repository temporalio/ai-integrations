"""LangSmith plugin for Temporal SDK."""

from __future__ import annotations

import copy
import dataclasses
import functools
import types
from collections.abc import AsyncGenerator, Sequence
from contextlib import asynccontextmanager
from typing import Any

import langsmith

# langsmith conditionally imports langchain_core when it is installed.
# Pre-import the lazily-loaded submodule so it is in sys.modules before the
# workflow sandbox starts; otherwise the sandbox's __getattr__-triggered
# import hits restrictions on concurrent.futures.ThreadPoolExecutor.
try:
    import langchain_core.runnables.config  # noqa: F401  # pyright: ignore[reportUnusedImport]
except ImportError:
    pass

import temporalio.converter
import temporalio.workflow
from temporalio.bridge.proto.workflow_activation import WorkflowActivation
from temporalio.bridge.proto.workflow_completion import WorkflowActivationCompletion
from temporalio.langsmith._interceptor import LangSmithInterceptor, _EndingEvictions
from temporalio.plugin import SimplePlugin
from temporalio.worker import (
    WorkerConfig,
    WorkflowInstance,
    WorkflowInstanceDetails,
    WorkflowRunner,
)
from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner


class _EndingEvictionInstance(WorkflowInstance):
    """A workflow instance that marks the eviction after its workflow ended.

    Its activations go through ``_EndingEvictions``.
    """

    def __init__(self, instance: WorkflowInstance) -> None:
        self._instance = instance
        self._evictions = _EndingEvictions(instance.activate)

    def activate(self, act: WorkflowActivation) -> WorkflowActivationCompletion:
        return self._evictions.activate(act)

    def get_serialization_context(
        self, command_info: Any
    ) -> temporalio.converter.SerializationContext | None:
        return self._instance.get_serialization_context(command_info)

    def get_external_store_context(self, command_info: Any) -> Any:
        return self._instance.get_external_store_context(command_info)

    def get_info(self) -> temporalio.workflow.Info:
        return self._instance.get_info()

    def get_thread_id(self) -> int | None:
        return self._instance.get_thread_id()


class _MarkingRunner:
    """A base of the runner classes made by ``_marking_class``."""

    __slots__ = ()


@functools.cache
def _marking_class(cls: type[WorkflowRunner]) -> type[WorkflowRunner]:
    """Make a subclass of a runner class whose instances mark ending evictions."""

    def create_instance(
        self: WorkflowRunner, det: WorkflowInstanceDetails
    ) -> WorkflowInstance:
        return _EndingEvictionInstance(cls.create_instance(self, det))

    def body(ns: dict[str, Any]) -> None:
        ns.update(
            __slots__=(),
            __module__=cls.__module__,
            __qualname__=cls.__qualname__,
            create_instance=create_instance,
        )

    return types.new_class(cls.__name__, (cls, _MarkingRunner), exec_body=body)


class _Runner(WorkflowRunner):
    """A workflow runner whose instances mark ending evictions."""

    def __init__(self, runner: WorkflowRunner) -> None:
        self._runner = runner

    def prepare_workflow(self, defn: temporalio.workflow._Definition) -> None:
        self._runner.prepare_workflow(defn)

    def create_instance(self, det: WorkflowInstanceDetails) -> WorkflowInstance:
        return _EndingEvictionInstance(self._runner.create_instance(det))

    def set_worker_level_failure_exception_types(
        self, types: Sequence[type[BaseException]]
    ) -> None:
        self._runner.set_worker_level_failure_exception_types(types)


def _mark_ending_evictions(runner: WorkflowRunner) -> WorkflowRunner:
    """Copy a runner so its instances mark the eviction after their workflow ends.

    The copy's class is a subclass of the runner's own, so the copy keeps the
    runner's behavior and type: a ``SandboxedWorkflowRunner``, or a subclass
    of it, stays one, and other plugins can still change its restrictions.
    A runner whose class can't be subclassed, or that can't be copied, is
    wrapped instead.
    """
    if isinstance(runner, (_MarkingRunner, _Runner)):
        return runner
    try:
        marked = copy.copy(runner)
        object.__setattr__(marked, "__class__", _marking_class(type(runner)))
    except (TypeError, copy.Error):
        return _Runner(runner)
    return marked


class LangSmithPlugin(SimplePlugin):
    """LangSmith tracing plugin for Temporal SDK.

    .. warning::
        This class is experimental and may change in future versions.
        Use with caution in production environments.

    Provides automatic LangSmith run creation for workflows, activities,
    and other Temporal operations with context propagation.
    """

    def __init__(
        self,
        *,
        client: langsmith.Client | None = None,
        project_name: str | None = None,
        add_temporal_runs: bool = False,
        default_metadata: dict[str, Any] | None = None,
        default_tags: list[str] | None = None,
    ) -> None:
        """Initialize the LangSmith plugin.

        Args:
            client: A langsmith.Client instance. If None, one will be created
                automatically (using LANGSMITH_API_KEY env var).
            project_name: Fallback LangSmith project name for traces with no
                parent. Temporal runs inherit their parent's destination when
                present. When unset, root runs use LangSmith's environment defaults.
            add_temporal_runs: Whether to create LangSmith runs for Temporal
                operations. Defaults to False.
            default_metadata: Default metadata to attach to all runs.
            default_tags: Default tags to attach to all runs.
        """
        interceptor = LangSmithInterceptor(
            client=client,
            project_name=project_name,
            add_temporal_runs=add_temporal_runs,
            default_metadata=default_metadata,
            default_tags=default_tags,
        )
        interceptors = [interceptor]

        def workflow_runner(runner: WorkflowRunner | None) -> WorkflowRunner:
            if not runner:
                raise ValueError("No WorkflowRunner provided to the LangSmith plugin.")
            if isinstance(runner, SandboxedWorkflowRunner):
                runner = dataclasses.replace(
                    runner,
                    restrictions=runner.restrictions.with_passthrough_modules(
                        "langsmith",
                        "langchain_core",
                        "opentelemetry",
                    ),
                )
            return _mark_ending_evictions(runner)

        @asynccontextmanager
        async def run_context() -> AsyncGenerator[None, None]:
            try:
                yield
            finally:
                interceptor._client.flush()

        super().__init__(
            "langchain.LangSmithPlugin",
            interceptors=interceptors,
            workflow_runner=workflow_runner,
            run_context=run_context,
        )

    def configure_worker(self, config: WorkerConfig) -> WorkerConfig:
        """Configure the worker, and the runner of workflows not sandboxed."""
        config = super().configure_worker(config)
        runner = config.get("unsandboxed_workflow_runner")
        if runner is not None:
            config["unsandboxed_workflow_runner"] = _mark_ending_evictions(runner)
        return config
