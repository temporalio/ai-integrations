"""Integration tests for LangSmith plugin with real Temporal worker."""

from __future__ import annotations

import asyncio
import dataclasses
import gc
import random
import threading
import time
import uuid
from collections import Counter
from collections.abc import AsyncIterator, Sequence
from datetime import datetime, timedelta
from typing import Any
from unittest.mock import MagicMock, patch

import langsmith
import langsmith._internal._uuid as ls_uuid
import langsmith.env
import nexusrpc.handler
import pytest
from langchain_core.callbacks import AsyncCallbackHandler, BaseCallbackHandler
from langchain_core.env import get_runtime_environment
from langchain_core.runnables import RunnableGenerator, RunnableLambda
from langchain_core.tracers.langchain import LangChainTracer
from langsmith import traceable, tracing_context
from langsmith.run_helpers import get_tracing_context
from langsmith.run_trees import WriteReplica

from temporalio import activity, common, nexus, workflow
from temporalio.api.common.v1 import Payload
from temporalio.api.enums.v1 import EventType, WorkflowTaskFailedCause
from temporalio.client import (
    Client,
    WorkflowExecutionStatus,
    WorkflowFailureError,
    WorkflowHandle,
    WorkflowUpdateStage,
)
from temporalio.converter import DataConverter, PayloadCodec
from temporalio.exceptions import ApplicationError
from temporalio.langsmith import LangSmithInterceptor, LangSmithPlugin
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import (
    HandleUpdateInput,
    Interceptor,
    Replayer,
    UnsandboxedWorkflowRunner,
    Worker,
    WorkflowInboundInterceptor,
    WorkflowInterceptorClassInput,
    WorkflowRunner,
)
from temporalio.worker.workflow_sandbox import (
    SandboxedWorkflowRunner,
    SandboxRestrictions,
)
from tests.helpers import new_worker
from tests.helpers.langsmith import (
    InMemoryRunCollector,
    build_trace_trees,
    find_trace_trees,
    make_mock_ls_client,
)
from tests.helpers.nexus import make_nexus_endpoint_name
from tests.helpers.trace import assert_trace_hierarchy

# ---------------------------------------------------------------------------
# Shared @traceable functions and activities
# ---------------------------------------------------------------------------


@traceable(name="inner_llm_call")
async def _inner_llm_call(prompt: str) -> str:
    """Simulates an LLM call decorated with @traceable."""
    return f"response to: {prompt}"


@traceable(name="outer_chain")
async def _outer_chain(prompt: str) -> str:
    """A @traceable that calls another @traceable."""
    return await _inner_llm_call(prompt)


@traceable
@activity.defn
async def traceable_activity() -> str:
    """Activity that calls a @traceable function."""
    result = await _inner_llm_call("hello")
    return result


@traceable
@activity.defn
async def nested_traceable_activity() -> str:
    """Activity with two levels of @traceable nesting."""
    result = await _outer_chain("hello")
    return result


# ---------------------------------------------------------------------------
# Shared workflows
# ---------------------------------------------------------------------------


@workflow.defn
class TraceableActivityWorkflow:
    @workflow.run
    async def run(self, _input: str = "") -> str:
        return await workflow.execute_activity(
            traceable_activity,
            start_to_close_timeout=timedelta(minutes=1),
            retry_policy=common.RetryPolicy(maximum_attempts=1),
        )


class _RunIdRecorder(AsyncCallbackHandler):
    """Records the run ID LangChain gives each named run's callbacks."""

    def __init__(self) -> None:
        self.run_ids: dict[str, str] = {}

    async def on_chain_start(
        self,
        serialized: dict[str, Any] | None,
        inputs: Any,
        *,
        run_id: uuid.UUID,
        **kwargs: Any,
    ) -> None:
        self.run_ids[str(kwargs.get("name"))] = str(run_id)


@workflow.defn
class LangChainRunnableWorkflow:
    """Runs a LangChain runnable in workflow code; its last step calls an activity.

    A nonempty input is the runnable's ``run_id``.  Returns the run IDs the
    runnable's callbacks got, by run name.
    """

    @workflow.run
    async def run(self, run_id: str = "") -> dict[str, str]:
        # Async steps: a sync step would run in an executor, which workflows lack.
        async def prepare(prompt: str) -> str:
            return prompt

        async def call_activity(_prompt: str) -> str:
            return await workflow.execute_activity(
                traceable_activity,
                start_to_close_timeout=timedelta(minutes=1),
                retry_policy=common.RetryPolicy(maximum_attempts=1),
            )

        chain = RunnableLambda(prepare, name="prepare") | RunnableLambda(
            call_activity, name="call_activity"
        )
        recorder = _RunIdRecorder()
        config: Any = {"callbacks": [recorder]}
        if run_id:
            config["run_id"] = uuid.UUID(run_id)
        await chain.ainvoke("hello", config=config)
        return recorder.run_ids


@traceable(name="call_traceable_activity")
async def _call_traceable_activity() -> str:
    """A @traceable helper around an activity."""
    return await workflow.execute_activity(
        traceable_activity,
        start_to_close_timeout=timedelta(minutes=1),
        retry_policy=common.RetryPolicy(maximum_attempts=1),
    )


@traceable(name="stream_traceable_activity")
async def _stream_traceable_activity() -> str:
    """Another @traceable helper around an activity."""
    return await workflow.execute_activity(
        traceable_activity,
        start_to_close_timeout=timedelta(minutes=1),
        retry_policy=common.RetryPolicy(maximum_attempts=1),
    )


@workflow.defn
class TraceableHelperWorkflow:
    """Calls a @traceable helper around an activity."""

    @workflow.run
    async def run(self) -> str:
        return await _call_traceable_activity()


# The clients of the tracers workflows make of their own; tests set them.
_own_tracer_clients: list[Any] = []


@workflow.defn
class LangChainHelperWorkflow:
    """Runs a LangChain runnable whose steps call @traceable helpers.

    The ``stream`` step is a generator, whose run LangChain posts only when
    it ends.  ``variant`` is one of:

    - ``"own_tracer"``: the runnable gets a ``LangChainTracer`` built with
      the first client in ``_own_tracer_clients``.
    - ``"tags"``: it runs in a ``tracing_context`` with only tags.
    - ``"replica"``: it runs in a ``tracing_context`` whose write replica
      has the second client in ``_own_tracer_clients``.
    - ``"ambient_replica"``: the same, with the context's client.

    Run it unsandboxed.
    """

    @workflow.run
    async def run(self, variant: str = "") -> str:
        async def prepare(prompt: str) -> str:
            return prompt

        async def call_helper(_prompt: str) -> str:
            return await _call_traceable_activity()

        async def stream(chunks: AsyncIterator[str]) -> AsyncIterator[str]:
            async for chunk in chunks:
                yield chunk + await _stream_traceable_activity()

        async def finish(text: str) -> str:
            return text

        chain = (
            RunnableLambda(prepare, name="prepare")
            | RunnableLambda(call_helper, name="call_helper")
            | RunnableGenerator(stream, name="stream")
            | RunnableLambda(finish, name="finish")
        )
        config: Any = {}
        context: dict[str, Any] = {}
        if variant == "own_tracer":
            config["callbacks"] = [LangChainTracer(client=_own_tracer_clients[0])]
        elif variant == "tags":
            context["tags"] = ["feature"]
        elif variant in ("replica", "ambient_replica"):
            replica_client = (
                _own_tracer_clients[1]
                if variant == "replica"
                else get_tracing_context()["client"]
            )
            context["replicas"] = [{"project_name": "copy", "client": replica_client}]
        if not context:
            return await chain.ainvoke("hello", config=config)
        with tracing_context(**context):
            return await chain.ainvoke("hello", config=config)


# A tracer workflows share; tests set it.
_shared_tracers: list[Any] = []


@workflow.defn
class LangChainSignalWorkflow:
    """Runs a LangChain runnable whose step waits for a signal.

    With ``shared_tracer``, the runnable gets the tracer in
    ``_shared_tracers``, so run it unsandboxed.  The ``step`` update runs
    ``prepare | step``, with ``tag`` after each name, and its step can wait
    for ``release_step``.  With ``fail_once``, its first call raises after
    the chain.
    """

    def __init__(self) -> None:
        self._released = False
        self._step_released = False
        self._failed = False

    @workflow.signal
    def release(self) -> None:
        self._released = True

    @workflow.signal
    def release_step(self) -> None:
        self._step_released = True

    @workflow.update
    async def step(
        self, wait: bool = False, tag: str = "", fail_once: bool = False
    ) -> str:
        async def prepare(text: str) -> str:
            return text

        async def step(text: str) -> str:
            if wait:
                await workflow.wait_condition(lambda: self._step_released)
            return text

        chain = RunnableLambda(prepare, name=f"prepare{tag}") | RunnableLambda(
            step, name=f"step{tag}"
        )
        result = await chain.ainvoke("update")
        if fail_once and not self._failed:
            self._failed = True
            raise ApplicationError("first call")
        return result

    @workflow.run
    async def run(self, shared_tracer: bool = False) -> str:
        async def wait(text: str) -> str:
            await workflow.wait_condition(lambda: self._released)
            return text

        config: Any = {"callbacks": _shared_tracers[:1]} if shared_tracer else {}
        return await RunnableLambda(wait, name="wait").ainvoke("secret", config=config)


class _CallUpdateAgainInterceptor(Interceptor):
    """Calls an update's handler again when it raises."""

    def workflow_interceptor_class(
        self, input: WorkflowInterceptorClassInput
    ) -> type[WorkflowInboundInterceptor]:
        return _CallUpdateAgainInbound


class _CallUpdateAgainInbound(WorkflowInboundInterceptor):
    async def handle_update_handler(self, input: HandleUpdateInput) -> Any:
        try:
            return await super().handle_update_handler(input)
        except ApplicationError:
            return await super().handle_update_handler(input)


@workflow.defn
class LangChainRaceWorkflow:
    """Races two LangChain runnables and returns when the first ends.

    The step of ``race_a`` waits for ``release``; ``race_b``'s never ends.
    ``fail`` fails the workflow from its signal handler, and ``crash`` fails
    the workflow task.  ``release_slowly`` and ``fail_slowly`` block their
    activation for half a second, so that a signal sent meanwhile makes the
    server reject its completion.  After ``release_unserializable``, the workflow
    returns a result that can't be serialized, which fails the workflow
    task.  After ``release_uncodable`` it returns ``_UNCODABLE``, and
    ``fail_uncodable`` fails it with that as its details, which
    ``_FailingCodec`` can fail to encode.  With ``deferred``, the runnables
    take their input as a stream.
    """

    def __init__(self) -> None:
        self._released = False
        self._unserializable = False
        self._uncodable = False

    @workflow.signal
    def release(self) -> None:
        self._released = True

    @workflow.signal
    def release_unserializable(self) -> None:
        self._released = True
        self._unserializable = True

    @workflow.signal
    def release_uncodable(self) -> None:
        self._released = True
        self._uncodable = True

    @workflow.signal
    def fail(self) -> None:
        raise ApplicationError("failed by signal")

    @workflow.signal
    def fail_uncodable(self) -> None:
        raise ApplicationError("failed by signal", _UNCODABLE)

    @workflow.signal
    def crash(self) -> None:
        raise RuntimeError("workflow task failed by signal")

    @workflow.signal
    def release_slowly(self) -> None:
        self._released = True
        _block()

    @workflow.signal
    def fail_slowly(self) -> None:
        _block()
        raise ApplicationError("failed by signal")

    @workflow.signal
    def other(self) -> None:
        pass

    @workflow.run
    async def run(self, deferred: bool = False) -> str:
        async def wait(name: str) -> str:
            await workflow.wait_condition(lambda: self._released and name == "a")
            return name

        async def wait_each(names: AsyncIterator[str]) -> AsyncIterator[str]:
            async for name in names:
                yield await wait(name)

        tasks = [
            asyncio.create_task(
                (
                    RunnableGenerator(wait_each, name=f"race_{name}")
                    if deferred
                    else RunnableLambda(wait, name=f"race_{name}")
                ).ainvoke(name)
            )
            for name in "ab"
        ]
        done, _ = await workflow.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        result: Any = object() if self._unserializable else done.pop().result()
        return _UNCODABLE if self._uncodable else result


def _block() -> None:
    """Block a live activation for half a second, well under the deadlock timeout."""
    if not workflow.unsafe.is_replaying_history_events():
        with workflow.unsafe.sandbox_unrestricted():
            time.sleep(0.5)


_UNCODABLE = "uncodable"


class _FailingCodec(PayloadCodec):
    """Fails to encode a payload of ``_UNCODABLE`` while ``broken``."""

    def __init__(self) -> None:
        self.broken = True

    async def encode(self, payloads: Sequence[Payload]) -> list[Payload]:
        if self.broken and any(p.data == f'"{_UNCODABLE}"'.encode() for p in payloads):
            raise RuntimeError("codec failed")
        return list(payloads)

    async def decode(self, payloads: Sequence[Payload]) -> list[Payload]:
        return list(payloads)


class _SubclassedRunner(SandboxedWorkflowRunner):
    """A subclass of the sandboxed runner, changing nothing."""


@workflow.defn
class LangChainTwoClientsWorkflow:
    """Runs two LangChain runnables at once, each traced to its own client.

    The step of the runnable for "a" or "b" waits for that name's signal.
    With ``shared``, "ab" or "ba", it runs one runnable instead, traced to
    both clients in that order, whose step waits for that signal.  The
    clients are the two in ``_own_tracer_clients``, so run it unsandboxed.
    """

    def __init__(self) -> None:
        self._released: set[str] = set()

    @workflow.signal
    def release(self, name: str) -> None:
        self._released.add(name)

    @workflow.run
    async def run(self, shared: str = "") -> list[str]:
        async def wait(name: str) -> str:
            await workflow.wait_condition(lambda: name in self._released)
            return name

        if shared:
            clients = dict(zip("ab", _own_tracer_clients))
            tracers: list[BaseCallbackHandler] = [
                LangChainTracer(client=clients[name]) for name in shared
            ]
            return [
                await RunnableLambda(wait, name=f"wait_{shared}").ainvoke(
                    shared, config={"callbacks": tracers}
                )
            ]
        return list(
            await asyncio.gather(
                *(
                    RunnableLambda(wait, name=f"wait_{name}").ainvoke(
                        name, config={"callbacks": [LangChainTracer(client=client)]}
                    )
                    for name, client in zip("ab", _own_tracer_clients)
                )
            )
        )


@activity.defn
async def primary_activity() -> str:
    return "primary"


@activity.defn
async def fallback_activity() -> str:
    return "fallback"


@workflow.defn
class LangChainTracingErrorWorkflow:
    """Schedules a fallback activity when LangChain's tracer raises."""

    @workflow.run
    async def run(self) -> str:
        async def step(text: str) -> str:
            return text

        tracer = LangChainTracer(client=get_tracing_context()["client"])
        tracer.raise_error = True
        try:
            await RunnableLambda(step, name="step").ainvoke(
                "secret", config={"callbacks": [tracer]}
            )
            activity_fn = primary_activity
        except ValueError:
            activity_fn = fallback_activity
        return await workflow.execute_activity(
            activity_fn, start_to_close_timeout=timedelta(minutes=1)
        )


@nexusrpc.handler.service_handler
class NexusService:
    @nexus.workflow_run_operation
    async def run_operation(
        self, ctx: nexus.WorkflowRunOperationContext, input: str
    ) -> nexus.WorkflowHandle[str]:
        return await ctx.start_workflow(
            TraceableActivityWorkflow.run,
            input,
            id=f"nexus-wf-{ctx.request_id}",
        )


# ---------------------------------------------------------------------------
# Simple/basic workflows and activities
# ---------------------------------------------------------------------------


@traceable
@activity.defn
async def simple_activity() -> str:
    return "activity-done"


@workflow.defn
class SimpleWorkflow:
    @workflow.run
    async def run(self) -> str:
        result = await workflow.execute_activity(
            simple_activity,
            start_to_close_timeout=timedelta(minutes=1),
            retry_policy=common.RetryPolicy(maximum_attempts=1),
        )
        return result


# ---------------------------------------------------------------------------
# Signal/query/update workflows
# ---------------------------------------------------------------------------


@traceable(name="step_with_activity")
async def _step_with_activity() -> str:
    """A @traceable step that wraps an activity call."""
    return await workflow.execute_activity(
        nested_traceable_activity,
        start_to_close_timeout=timedelta(minutes=1),
        retry_policy=common.RetryPolicy(maximum_attempts=1),
    )


@traceable(name="step_with_child_workflow")
async def _step_with_child_workflow() -> str:
    """A @traceable step that wraps a child workflow call."""
    return await workflow.execute_child_workflow(
        TraceableActivityWorkflow.run,
        id=f"step-child-{workflow.info().workflow_id}",
    )


@traceable(name="step_with_nexus")
async def _step_with_nexus() -> str:
    """A @traceable step that wraps a nexus operation."""
    nexus_client = workflow.create_nexus_client(
        endpoint=make_nexus_endpoint_name(workflow.info().task_queue),
        service=NexusService,
    )
    nexus_handle = await nexus_client.start_operation(
        operation=NexusService.run_operation,
        input="test-input",
    )
    return await nexus_handle


@workflow.defn
class ComprehensiveWorkflow:
    def __init__(self) -> None:
        self._signal_received = False
        self._complete = False

    @workflow.run
    async def run(self) -> str:
        await workflow.execute_activity(
            nested_traceable_activity,
            start_to_close_timeout=timedelta(minutes=1),
            retry_policy=common.RetryPolicy(maximum_attempts=1),
        )
        await _step_with_activity()
        await workflow.execute_local_activity(
            nested_traceable_activity,
            start_to_close_timeout=timedelta(minutes=1),
            retry_policy=common.RetryPolicy(maximum_attempts=1),
        )
        await _outer_chain("from-workflow")
        await workflow.execute_child_workflow(
            TraceableActivityWorkflow.run,
            id=f"child-{workflow.info().workflow_id}",
        )
        await _step_with_child_workflow()
        nexus_client = workflow.create_nexus_client(
            endpoint=make_nexus_endpoint_name(workflow.info().task_queue),
            service=NexusService,
        )
        nexus_handle = await nexus_client.start_operation(
            operation=NexusService.run_operation,
            input="test-input",
        )
        await nexus_handle
        await _step_with_nexus()

        await workflow.wait_condition(lambda: self._signal_received)
        await workflow.execute_activity(
            nested_traceable_activity,
            start_to_close_timeout=timedelta(minutes=1),
            retry_policy=common.RetryPolicy(maximum_attempts=1),
        )
        await workflow.wait_condition(lambda: self._complete)
        return "comprehensive-done"

    @workflow.signal
    def my_signal(self, _value: str) -> None:
        self._signal_received = True

    @workflow.query
    def my_query(self) -> bool:
        return self._signal_received

    @workflow.update
    def my_update(self, value: str) -> str:
        self._complete = True
        return f"updated-{value}"

    @my_update.validator
    def validate_my_update(self, value: str) -> None:
        if not value:
            raise ValueError("empty")

    @workflow.update
    def my_unvalidated_update(self, value: str) -> str:
        return f"unvalidated-{value}"


# ---------------------------------------------------------------------------
# Error workflows and activities
# ---------------------------------------------------------------------------


@traceable
@activity.defn
async def failing_activity() -> str:
    raise ApplicationError("activity-failed", non_retryable=True)


@traceable
@activity.defn
async def benign_failing_activity() -> str:
    from temporalio.exceptions import ApplicationErrorCategory

    raise ApplicationError(
        "benign-fail",
        non_retryable=True,
        category=ApplicationErrorCategory.BENIGN,
    )


@workflow.defn
class FailingWorkflow:
    @workflow.run
    async def run(self) -> str:
        raise ApplicationError("workflow-failed", non_retryable=True)


@workflow.defn
class ActivityFailureWorkflow:
    @workflow.run
    async def run(self) -> str:
        return await workflow.execute_activity(
            failing_activity,
            start_to_close_timeout=timedelta(minutes=1),
            retry_policy=common.RetryPolicy(maximum_attempts=1),
        )


@workflow.defn
class ContinueAsNewWorkflow:
    @workflow.run
    async def run(self, iteration: int) -> int:
        if iteration == 0:
            workflow.continue_as_new(iteration + 1)
        return iteration


@workflow.defn
class WaitForeverWorkflow:
    @workflow.run
    async def run(self) -> None:
        await workflow.wait_condition(lambda: False)


@workflow.defn
class BenignErrorWorkflow:
    @workflow.run
    async def run(self) -> str:
        return await workflow.execute_activity(
            benign_failing_activity,
            start_to_close_timeout=timedelta(minutes=1),
            retry_policy=common.RetryPolicy(maximum_attempts=1),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_plugin_and_collector(
    **kwargs: Any,
) -> tuple[LangSmithPlugin, InMemoryRunCollector, MagicMock]:
    """Create a LangSmithPlugin wired to an InMemoryRunCollector via mock client."""
    collector = InMemoryRunCollector()
    mock_ls_client = make_mock_ls_client(collector)
    plugin = LangSmithPlugin(client=mock_ls_client, **kwargs)
    return plugin, collector, mock_ls_client


def _wait_for_tracing_io(plugin: LangSmithPlugin) -> None:
    """Wait for run updates still queued on the plugin's single-worker executor."""
    assert plugin.interceptors
    interceptor: Any = plugin.interceptors[0]
    interceptor._executor.submit(lambda: None).result()


def _make_client_and_collector(
    client: Client, **kwargs: Any
) -> tuple[Client, InMemoryRunCollector, MagicMock]:
    """Create a Temporal Client with LangSmith plugin and an InMemoryRunCollector."""
    plugin, collector, mock_ls_client = _make_plugin_and_collector(**kwargs)
    config = client.config()
    config["plugins"] = [plugin]
    return Client(**config), collector, mock_ls_client


def _replica_ids_are_pure() -> bool:
    """Whether LangSmith derives a replica's run IDs from the run's alone.

    Earlier releases add the clock to an ID derived from one that is not a
    UUIDv7, so a run's post and patch to a replica name different runs, with
    or without a replay.
    """
    clock = MagicMock()
    clock.time_ns.side_effect = [1_000_000_000, 2_000_000_000]
    run_id = uuid.uuid4()
    with patch.object(ls_uuid, "time", clock, create=True):
        return ls_uuid.uuid7_deterministic(
            run_id, "copy"
        ) == ls_uuid.uuid7_deterministic(run_id, "copy")


def _make_temporal_client(
    client: Client, mock_ls_client: MagicMock, **kwargs: Any
) -> Client:
    """Create a Temporal Client with a fresh LangSmith plugin."""
    plugin = LangSmithPlugin(client=mock_ls_client, **kwargs)
    config = client.config()
    config["plugins"] = [plugin]
    return Client(**config)


# ---------------------------------------------------------------------------
# TestBasicTracing
# ---------------------------------------------------------------------------


class TestBasicTracing:
    async def test_workflow_activity_trace_hierarchy(
        self,
        client: Client,
        env: WorkflowEnvironment,  # type:ignore[reportUnusedParameter]
    ) -> None:
        """StartWorkflow → RunWorkflow → StartActivity → RunActivity hierarchy."""
        temporal_client, collector, _ = _make_client_and_collector(
            client, add_temporal_runs=True
        )

        async with new_worker(
            temporal_client,
            SimpleWorkflow,
            activities=[simple_activity],
            max_cached_workflows=0,
        ) as worker:
            result = await temporal_client.start_workflow(
                SimpleWorkflow.run,
                id=f"basic-trace-{uuid.uuid4()}",
                task_queue=worker.task_queue,
            )
            assert await result.result() == "activity-done"

        expected = [
            "StartWorkflow:SimpleWorkflow",
            "RunWorkflow:SimpleWorkflow",
            "  StartActivity:simple_activity",
            "  RunActivity:simple_activity",
            "    simple_activity",
        ]
        assert_trace_hierarchy(build_trace_trees(collector), expected)

        # Verify run_type: RunActivity is "tool", others are "chain"
        for run in collector.runs:
            if run.name == "RunActivity:simple_activity":
                assert run.run_type == "tool", (
                    f"Expected RunActivity run_type='tool', got '{run.run_type}'"
                )
            else:
                assert run.run_type == "chain", (
                    f"Expected {run.name} run_type='chain', got '{run.run_type}'"
                )

        # Verify successful runs have outputs == {"status": "ok"}
        for run in collector.runs:
            if ":" in run.name:  # Interceptor runs use "Type:Name" format
                assert run.outputs == {"status": "ok"}, (
                    f"Expected {run.name} outputs={{'status': 'ok'}}, got {run.outputs}"
                )


# ---------------------------------------------------------------------------
# TestReplay
# ---------------------------------------------------------------------------


class TestReplay:
    @pytest.mark.parametrize(
        ("workflow_cls", "arg", "expected"),
        [
            # Workflow→activity→@traceable flow
            (
                TraceableActivityWorkflow,
                "",
                [
                    "StartWorkflow:TraceableActivityWorkflow",
                    "RunWorkflow:TraceableActivityWorkflow",
                    "  StartActivity:traceable_activity",
                    "  RunActivity:traceable_activity",
                    "    traceable_activity",
                    "      inner_llm_call",
                ],
            ),
            # LangChain's tracer writes these runs, not @traceable
            (
                LangChainRunnableWorkflow,
                "",
                [
                    "StartWorkflow:LangChainRunnableWorkflow",
                    "RunWorkflow:LangChainRunnableWorkflow",
                    "  RunnableSequence",
                    "    prepare",
                    "    call_activity",
                    "      StartActivity:traceable_activity",
                    "      RunActivity:traceable_activity",
                    "        traceable_activity",
                    "          inner_llm_call",
                ],
            ),
            # The same, with a run_id the caller supplied
            (
                LangChainRunnableWorkflow,
                "12345678-1234-4234-8234-123456789abc",
                [
                    "StartWorkflow:LangChainRunnableWorkflow",
                    "RunWorkflow:LangChainRunnableWorkflow",
                    "  RunnableSequence",
                    "    prepare",
                    "    call_activity",
                    "      StartActivity:traceable_activity",
                    "      RunActivity:traceable_activity",
                    "        traceable_activity",
                    "          inner_llm_call",
                ],
            ),
        ],
    )
    async def test_no_duplicate_traces_on_replay(
        self,
        client: Client,
        env: WorkflowEnvironment,  # type:ignore[reportUnusedParameter]
        workflow_cls: Any,
        arg: str,
        expected: list[str],
    ) -> None:
        """With max_cached_workflows=0 (forcing replay), no duplicate runs appear."""
        # LangChain's tracer reads the runtime environment, which the workflow
        # sandbox blocks until it is cached outside it.
        get_runtime_environment()
        temporal_client, collector, _ = _make_client_and_collector(
            client, add_temporal_runs=True
        )

        async with new_worker(
            temporal_client,
            workflow_cls,
            activities=[traceable_activity],
            max_cached_workflows=0,
        ) as worker:
            handle = await temporal_client.start_workflow(
                workflow_cls.run,
                arg,
                id=f"replay-test-{uuid.uuid4()}",
                task_queue=worker.task_queue,
            )
            result = await handle.result()

        assert_trace_hierarchy(build_trace_trees(collector), expected)
        if arg:
            # The run keeps the ID the caller gave it.
            [run] = [r for r in collector.runs if r.name == "RunnableSequence"]
            assert run.id == arg
        if workflow_cls is LangChainRunnableWorkflow:
            # Callbacks got the IDs LangSmith stored.
            assert set(result) == {"RunnableSequence", "prepare", "call_activity"}
            for name, run_id in result.items():
                [run] = [r for r in collector.runs if r.name == name]
                assert run.id == run_id

    @pytest.mark.parametrize(
        "variant",
        [
            "",
            "own_tracer",
            "unhashable_own_tracer",
            "tags",
            "replica",
            "ambient_replica",
        ],
    )
    @pytest.mark.parametrize("add_temporal_runs", [False, True])
    async def test_each_run_written_once_on_replay(
        self,
        client: Client,
        env: WorkflowEnvironment,  # type:ignore[reportUnusedParameter]
        add_temporal_runs: bool,
        variant: str,
    ) -> None:
        """With a cache of 0, a client at default settings writes each run once.

        So does a write replica with a client of its own, and a client that
        cannot be hashed, through a tracer of the workflow's own.
        """
        if variant == "replica" and "client" not in WriteReplica.__annotations__:
            pytest.skip("This LangSmith has no write replicas with a client")
        if variant.endswith("replica") and not _replica_ids_are_pure():
            pytest.skip("This LangSmith derives replica run IDs from the clock")
        get_runtime_environment()
        # The client's first post reads git metadata in a subprocess; read it
        # here so it does not stall the workflow past the deadlock timeout.
        langsmith.env.get_langchain_env_var_metadata()
        # (client, "post" or "patch", run)
        writes: list[tuple[str, str, dict[str, Any]]] = []

        class _RecordingClient(langsmith.Client):
            label = "plugin"

            def _create_run(self, *args: Any, **kwargs: Any) -> None:
                writes.append((self.label, "post", args[0]))

            def _update_run(self, *args: Any, **kwargs: Any) -> None:
                writes.append((self.label, "patch", args[0]))

        class _UnhashableClient(_RecordingClient):
            __hash__ = None  # type: ignore[assignment]

        client_cls = (
            _UnhashableClient if variant.startswith("unhashable") else _RecordingClient
        )
        ls_client = client_cls(api_url="http://localhost:1", api_key="test")
        replica_client = _RecordingClient(api_url="http://localhost:1", api_key="test")
        replica_client.label = "replica"
        _own_tracer_clients[:] = [ls_client, replica_client]
        plugin = LangSmithPlugin(client=ls_client, add_temporal_runs=add_temporal_runs)
        config = client.config()
        config["plugins"] = [plugin]
        temporal_client = Client(**config)

        async with new_worker(
            temporal_client,
            LangChainHelperWorkflow,
            activities=[traceable_activity],
            workflow_runner=UnsandboxedWorkflowRunner(),
            max_cached_workflows=0,
        ) as worker:
            await temporal_client.execute_workflow(
                LangChainHelperWorkflow.run,
                variant.removeprefix("unhashable_"),
                id=f"replay-test-{uuid.uuid4()}",
                task_queue=worker.task_queue,
            )
        _wait_for_tracing_io(plugin)

        # With a dedicated replica, LangChain's runs go to its client, and
        # the activity's to the plugin's.
        labels = {label for label, _, _ in writes}
        assert labels == ({"plugin", "replica"} if variant == "replica" else {"plugin"})
        for label in sorted(labels):
            self._check_written_once(
                [(op, run) for c, op, run in writes if c == label],
                langchain=label == ("replica" if variant == "replica" else "plugin"),
            )

    @staticmethod
    def _check_written_once(
        writes: list[tuple[str, dict[str, Any]]], *, langchain: bool
    ) -> None:
        """Check one client's writes of ``LangChainHelperWorkflow``'s runs.

        With ``langchain``, the client got LangChain's runs.
        """
        posts = [run for op, run in writes if op == "post"]
        patches = [run for op, run in writes if op == "patch"]
        assert len({str(run["id"]) for run in posts}) == len(posts)
        assert [run["error"] for run in posts + patches if run.get("error")] == []
        if not langchain:
            assert sorted(str(run["id"]) for run in patches) == sorted(
                str(run["id"]) for run in posts
            )
            return
        assert {
            "RunnableSequence",
            "prepare",
            "call_helper",
            "call_traceable_activity",
            "stream",
            "stream_traceable_activity",
            "finish",
        } <= {run["name"] for run in posts}
        # Each run is posted once and patched once, with no error.  The
        # activity's runs are posted once for each of the two calls.
        twice = {
            "StartActivity:traceable_activity",
            "RunActivity:traceable_activity",
            "traceable_activity",
            "inner_llm_call",
        }
        counts = Counter(run["name"] for run in posts)
        assert {
            name: count
            for name, count in counts.items()
            if count != (2 if name in twice else 1)
        } == {}
        # langchain-core releases that defer a generator's inputs post its
        # run when it ends, and do not patch it.
        [stream] = [run for run in posts if run["name"] == "stream"]
        assert sorted(str(run["id"]) for run in patches) == sorted(
            str(run["id"])
            for run in posts
            if not (run is stream and stream.get("end_time"))
        )
        assert [run["error"] for run in posts + patches if run.get("error")] == []

        # Runs start in the order they were made, no earlier than their parents.
        def as_datetime(value: datetime | str) -> datetime:
            if isinstance(value, datetime):
                return value
            return datetime.fromisoformat(value.replace("Z", "+00:00"))

        starts = {str(run["id"]): as_datetime(run["start_time"]) for run in posts}
        # No run ends before it starts.
        for run in posts + patches:
            if run.get("end_time"):
                end = as_datetime(run["end_time"])
                assert end >= starts[str(run["id"])], run.get("name", run["id"])
        ids = {run["name"]: str(run["id"]) for run in posts}
        steps = ["prepare", "call_helper", "stream", "finish"]
        assert all(starts[ids[a]] < starts[ids[b]] for a, b in zip(steps, steps[1:]))
        # A run's post and patch, and its children, agree on its dotted order.
        orders = {str(run["id"]): run["dotted_order"] for run in posts}
        for run in patches:
            if run.get("dotted_order"):
                assert run["dotted_order"] == orders[str(run["id"])], run
        for run in posts:
            # The run's own segment carries its start time.
            assert run["dotted_order"].split(".")[-1] == as_datetime(
                run["start_time"]
            ).strftime("%Y%m%dT%H%M%S%fZ") + str(run["id"]), run["name"]
            parent = str(run.get("parent_run_id"))
            if parent in starts:
                assert starts[parent] <= starts[str(run["id"])], run["name"]
                assert run["dotted_order"].startswith(orders[parent] + "."), run

    @pytest.mark.parametrize(
        ("failing_step", "expected"), [("redaction", "fallback"), ("write", "primary")]
    )
    async def test_tracing_error_raised_again_on_replay(
        self,
        client: Client,
        env: WorkflowEnvironment,  # type:ignore[reportUnusedParameter]
        failing_step: str,
        expected: str,
    ) -> None:
        """A redaction error workflow code catches is raised on replay too.

        An error from the write itself never reaches workflow code, since a
        replay does not write.
        """
        get_runtime_environment()

        def hide_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
            if failing_step == "redaction" and "secret" in str(inputs):
                raise ValueError("redaction failed")
            return inputs

        class _NoWriteClient(langsmith.Client):
            def _create_run(self, *args: Any, **kwargs: Any) -> None:
                if failing_step == "write":
                    raise ValueError("write failed")

            def _update_run(self, *args: Any, **kwargs: Any) -> None:
                pass

        ls_client = _NoWriteClient(
            api_url="http://localhost:1",
            api_key="test",
            auto_batch_tracing=False,
            hide_inputs=hide_inputs,
        )
        config = client.config()
        config["plugins"] = [LangSmithPlugin(client=ls_client)]
        temporal_client = Client(**config)

        async with new_worker(
            temporal_client,
            LangChainTracingErrorWorkflow,
            activities=[primary_activity, fallback_activity],
            max_cached_workflows=0,
        ) as worker:
            result = await temporal_client.execute_workflow(
                LangChainTracingErrorWorkflow.run,
                id=f"replay-test-{uuid.uuid4()}",
                task_queue=worker.task_queue,
                run_timeout=timedelta(seconds=20),
            )
        assert result == expected

    @pytest.mark.parametrize("first", ["a", "b", "ab", "ba"])
    async def test_held_writes_keep_their_client(
        self,
        client: Client,
        env: WorkflowEnvironment,  # type:ignore[reportUnusedParameter]
        first: str,
    ) -> None:
        """Each client writes only its own runs, each once per worker.

        With "a" or "b", two runs, one per client, and that run is released
        first.  With "ab" or "ba", one run traced to both clients in that
        order.
        """
        get_runtime_environment()
        langsmith.env.get_langchain_env_var_metadata()
        # (client, "post" or "patch", run), as written and as processed
        writes: list[tuple[str, str, dict[str, Any]]] = []
        processed: list[tuple[str, dict[str, Any]]] = []

        class _RecordingClient(langsmith.Client):
            label = ""

            def _create_run(self, *args: Any, **kwargs: Any) -> None:
                writes.append((self.label, "post", args[0]))

            def _update_run(self, *args: Any, **kwargs: Any) -> None:
                writes.append((self.label, "patch", args[0]))

        def ls_client(label: str) -> Any:
            def process(runs: Any) -> Any:
                processed.extend((label, run) for run in runs)
                return runs

            result = _RecordingClient(
                api_url="http://localhost:1",
                api_key="test",
                auto_batch_tracing=False,
                process_buffered_run_ops=process,
                run_ops_buffer_size=1,
            )
            result.label = label
            return result

        def temporal_client(label: str) -> Client:
            config = client.config()
            config["plugins"] = [LangSmithPlugin(client=ls_client(label))]
            return Client(**config)

        async def wait_for_tasks(handle: WorkflowHandle[Any, Any], count: int) -> None:
            for _ in range(100):
                history = await handle.fetch_history()
                if (
                    sum(
                        e.event_type == EventType.EVENT_TYPE_WORKFLOW_TASK_COMPLETED
                        for e in history.events
                    )
                    >= count
                ):
                    return
                await asyncio.sleep(0.1)
            raise AssertionError("workflow task not completed")

        shared = first if len(first) == 2 else ""
        second = shared or ("b" if first == "a" else "a")
        task_queue = str(uuid.uuid4())
        workflow_id = f"replay-test-{uuid.uuid4()}"
        _own_tracer_clients[:] = [ls_client("a1"), ls_client("b1")]
        # A cache of 0: each workflow task replays, so the tracers and the
        # client wrappers of the last are gone.
        async with new_worker(
            temporal_client("plugin1"),
            LangChainTwoClientsWorkflow,
            task_queue=task_queue,
            workflow_runner=UnsandboxedWorkflowRunner(),
            max_cached_workflows=0,
        ):
            handle = await client.start_workflow(
                LangChainTwoClientsWorkflow.run,
                shared,
                id=workflow_id,
                task_queue=task_queue,
            )
            await wait_for_tasks(handle, 1)
            gc.collect()
            if not shared:
                await handle.signal(LangChainTwoClientsWorkflow.release, first)
                await wait_for_tasks(handle, 2)

        # A new worker, with new clients, replays the run started above.
        _own_tracer_clients[:] = [ls_client("a2"), ls_client("b2")]
        async with new_worker(
            temporal_client("plugin2"),
            LangChainTwoClientsWorkflow,
            task_queue=task_queue,
            workflow_runner=UnsandboxedWorkflowRunner(),
        ):
            await handle.signal(LangChainTwoClientsWorkflow.release, second)
            assert await handle.result() == ([shared] if shared else ["a", "b"])

        for _ in range(100):
            if len(writes) >= (6 if shared else 5):
                break
            await asyncio.sleep(0.05)
        await asyncio.sleep(0.2)
        names = {str(run["id"]): run["name"] for _, op, run in writes if op == "post"}

        def by_client(
            records: list[tuple[str, str, str]],
        ) -> dict[str, list[tuple[str, str]]]:
            result: dict[str, list[tuple[str, str]]] = {}
            for label, op, name in records:
                result.setdefault(label, []).append((op, name))
            return {label: sorted(ops) for label, ops in result.items()}

        if shared:
            # Each client of the first worker posted the open run, and each
            # client of the second posted it again and patched it.
            run = f"wait_{shared}"
            expected = {
                **{f"{c}1": [("post", run)] for c in "ab"},
                **{f"{c}2": [("patch", run), ("post", run)] for c in "ab"},
            }
        else:
            expected = {
                f"{first}1": [("patch", f"wait_{first}"), ("post", f"wait_{first}")],
                f"{second}1": [("post", f"wait_{second}")],
                f"{second}2": [
                    ("patch", f"wait_{second}"),
                    ("post", f"wait_{second}"),
                ],
            }
        assert (
            by_client([(c, op, names[str(run["id"])]) for c, op, run in writes])
            == expected
        )
        # Each client's process_buffered_run_ops got only its own writes.
        assert by_client(
            [(c, "run", names[str(run["id"])]) for c, run in processed]
        ) == {
            label: [("run", name) for _, name in ops] for label, ops in expected.items()
        }

    async def test_buffered_processing_after_worker_restart(
        self,
        client: Client,
        env: WorkflowEnvironment,  # type:ignore[reportUnusedParameter]
    ) -> None:
        """A run a new worker writes after replay goes through process_buffered_run_ops."""
        get_runtime_environment()
        writes: list[tuple[str, dict[str, Any]]] = []

        class _RecordingClient(langsmith.Client):
            def _create_run(self, *args: Any, **kwargs: Any) -> None:
                writes.append(("post", args[0]))

            def _update_run(self, *args: Any, **kwargs: Any) -> None:
                writes.append(("patch", args[0]))

        def redact(runs: Any) -> list[dict[str, Any]]:
            return [
                {
                    k: "[redacted]" if k in ("inputs", "outputs") else v
                    for k, v in run.items()
                }
                for run in runs
            ]

        def temporal_client() -> Client:
            # A LangSmith client of its own, as a worker in a new process has.
            ls_client = _RecordingClient(
                api_url="http://localhost:1",
                api_key="test",
                auto_batch_tracing=False,
                process_buffered_run_ops=redact,
                run_ops_buffer_size=1,
            )
            config = client.config()
            config["plugins"] = [LangSmithPlugin(client=ls_client)]
            return Client(**config)

        task_queue = str(uuid.uuid4())
        workflow_id = f"replay-test-{uuid.uuid4()}"
        async with new_worker(
            temporal_client(),
            LangChainSignalWorkflow,
            task_queue=task_queue,
            workflow_runner=UnsandboxedWorkflowRunner(),
            max_cached_workflows=0,
        ):
            handle = await temporal_client().start_workflow(
                LangChainSignalWorkflow.run, id=workflow_id, task_queue=task_queue
            )
            for _ in range(100):
                history = await handle.fetch_history()
                if any(
                    e.event_type == EventType.EVENT_TYPE_WORKFLOW_TASK_COMPLETED
                    for e in history.events
                ):
                    break
                await asyncio.sleep(0.1)

        # A new worker, with a new plugin, replays the run started above.
        async with new_worker(
            temporal_client(),
            LangChainSignalWorkflow,
            task_queue=task_queue,
            workflow_runner=UnsandboxedWorkflowRunner(),
        ):
            await handle.signal(LangChainSignalWorkflow.release)
            assert await handle.result() == "secret"

        for _ in range(100):
            if sum(op == "patch" for op, _ in writes) >= 1:
                break
            await asyncio.sleep(0.05)
        posts = [run for op, run in writes if op == "post"]
        # The first worker's post, and the second's after replay.
        assert [run["name"] for run in posts] == ["wait", "wait"]
        assert len({run["id"] for run in posts}) == 1
        assert writes
        assert "secret" not in str(writes)

    async def test_tracer_used_again_without_plugin(
        self,
        client: Client,
        env: WorkflowEnvironment,  # type:ignore[reportUnusedParameter]
    ) -> None:
        """A tracer the plugin traced through writes as before on a worker without it.

        Its client stays the caller's, so outside the plugin, where runs get
        no stable IDs, its writes are not held back on replay.
        """
        get_runtime_environment()
        langsmith.env.get_langchain_env_var_metadata()
        writes: list[tuple[str, dict[str, Any]]] = []

        class _RecordingClient(langsmith.Client):
            def _create_run(self, *args: Any, **kwargs: Any) -> None:
                writes.append(("post", args[0]))

            def _update_run(self, *args: Any, **kwargs: Any) -> None:
                writes.append(("patch", args[0]))

        ls_client = _RecordingClient(
            api_url="http://localhost:1", api_key="test", auto_batch_tracing=False
        )
        tracer = LangChainTracer(client=ls_client)
        _shared_tracers[:] = [tracer]
        plugin_client, _, _ = _make_client_and_collector(client)

        async def run_once(temporal_client: Client) -> None:
            # A cache of 0: each workflow task after the first replays.
            async with new_worker(
                temporal_client,
                LangChainSignalWorkflow,
                workflow_runner=UnsandboxedWorkflowRunner(),
                max_cached_workflows=0,
            ) as worker:
                handle = await temporal_client.start_workflow(
                    LangChainSignalWorkflow.run,
                    True,
                    id=f"replay-test-{uuid.uuid4()}",
                    task_queue=worker.task_queue,
                )
                for _ in range(100):
                    history = await handle.fetch_history()
                    if any(
                        e.event_type == EventType.EVENT_TYPE_WORKFLOW_TASK_COMPLETED
                        for e in history.events
                    ):
                        break
                    await asyncio.sleep(0.1)
                await handle.signal(LangChainSignalWorkflow.release)
                assert await handle.result() == "secret"

        await run_once(plugin_client)
        assert tracer.client is ls_client
        # With the plugin, the run is posted once and patched once.
        assert sorted((op, str(run["id"])) for op, run in writes) == [
            ("patch", str(writes[0][1]["id"])),
            ("post", str(writes[0][1]["id"])),
        ]
        del writes[:]

        # Without it, each copy of the run is posted and patched, as LangChain
        # does without the plugin.
        await run_once(client)
        assert tracer.client is ls_client
        posts = [str(run["id"]) for op, run in writes if op == "post"]
        patches = [str(run["id"]) for op, run in writes if op == "patch"]
        assert posts
        assert sorted(set(patches)) == sorted(posts)

    @pytest.mark.parametrize("max_cached_workflows", [0, 1000])
    @pytest.mark.parametrize("deferred", [False, True])
    @pytest.mark.parametrize("fail", [False, True])
    @pytest.mark.parametrize(
        "case",
        [
            "",
            "subclassed",
            "interceptor",
            "replayed",
            "task",
            "codec",
            "rejected",
            "interceptor-task",
            "interceptor-codec",
            "interceptor-rejected",
            "interceptor-sandboxed-codec",
        ],
    )
    async def test_open_run_ended_when_workflow_ends(
        self,
        client: Client,
        env: WorkflowEnvironment,  # type:ignore[reportUnusedParameter]
        max_cached_workflows: int,
        deferred: bool,
        fail: bool,
        case: str,
    ) -> None:
        """A run still open when its workflow ends is ended, with an error.

        The eviction after the workflow ends cancels the run's coroutine.
        The workflow returns, or a signal handler fails it.  A run with
        deferred inputs is posted only when it ends.  With ``task_fails``,
        the workflow task in which the workflow would end fails instead, and
        the eviction after it doesn't end the runs: the workflow goes on.
        It fails in the workflow (``task``), or when the worker encodes its
        completion (``codec``), which is then fixed so the workflow ends.
        With ``subclassed``, the workers run a subclass of the sandboxed
        runner.  With ``interceptor``, ``LangSmithInterceptor`` is registered
        on the client without the plugin, and the workflow is not sandboxed.
        With ``replayed``, the workflow's history is replayed, with the
        plugin, while the eviction after it ended is between end writes.
        With ``rejected``, the server rejects the completion that would end
        the workflow, which then ends in a later task.  The ``interceptor-``
        cases fail or reject the task with the interceptor registered alone,
        on a sandboxed runner with ``interceptor-sandboxed-codec``.
        """
        interceptor = case.startswith("interceptor")
        variant = case.rpartition("-")[2] if interceptor else case
        variant = "" if variant == "interceptor" else variant
        task_fails = variant if variant in ("task", "codec") else ""
        rejected = variant == "rejected"
        get_runtime_environment()
        langsmith.env.get_langchain_env_var_metadata()
        writes: list[tuple[str, dict[str, Any]]] = []
        # With ``replayed``, the first end write with an error, which only the
        # eviction after the workflow ends makes, waits for ``resumed``.
        paused, resumed = threading.Event(), threading.Event()

        class _RecordingClient(langsmith.Client):
            def _create_run(self, *args: Any, **kwargs: Any) -> None:
                self._pause(args[0])
                writes.append(("post", args[0]))

            def _update_run(self, *args: Any, **kwargs: Any) -> None:
                self._pause(args[0])
                writes.append(("patch", args[0]))

            def _pause(self, run: dict[str, Any]) -> None:
                if case == "replayed" and run.get("error") and not paused.is_set():
                    paused.set()
                    assert resumed.wait(30)

        ls_client = _RecordingClient(
            api_url="http://localhost:1",
            api_key="test",
            auto_batch_tracing=False,
        )
        config = client.config()
        plugin = LangSmithPlugin(client=ls_client)
        if interceptor:
            config["interceptors"] = [LangSmithInterceptor(client=ls_client)]
        else:
            config["plugins"] = [plugin]
        codec = _FailingCodec()
        config["data_converter"] = dataclasses.replace(
            DataConverter.default, payload_codec=codec
        )
        temporal_client = Client(**config)
        runner: WorkflowRunner = SandboxedWorkflowRunner()
        if case == "subclassed":
            runner = _SubclassedRunner()
        elif "sandboxed" in case:
            runner = SandboxedWorkflowRunner(
                restrictions=SandboxRestrictions.default.with_passthrough_modules(
                    "langsmith", "langchain_core", "opentelemetry"
                )
            )
        elif interceptor:
            runner = UnsandboxedWorkflowRunner()

        def end_writes(name: str) -> list[dict[str, Any]]:
            names = {str(run["id"]): run["name"] for op, run in writes if op == "post"}
            return [
                run
                for _, run in writes
                if names.get(str(run["id"])) == name and run.get("end_time")
            ]

        async def assert_task_failed_and_runs_open(
            handle: WorkflowHandle[Any, Any],
        ) -> None:
            await wait_for(handle, EventType.EVENT_TYPE_WORKFLOW_TASK_FAILED)
            # Let the retried workflow task fail too.
            await asyncio.sleep(1)
            status = (await handle.describe()).status
            assert status == WorkflowExecutionStatus.RUNNING
            # The runs open when the workflow task failed are not ended, and
            # langchain-core releases that defer a RunnableGenerator's inputs
            # don't post them.  Only race_a's step can finish, in the failed
            # task, as in any task that fails.
            for name in ("race_a", "race_b") if fail else ("race_b",):
                assert not end_writes(name), name
                assert deferred or any(
                    op == "post" and run["name"] == name for op, run in writes
                ), name

        async def tasks_started(handle: WorkflowHandle[Any, Any]) -> int:
            history = await handle.fetch_history()
            return sum(
                e.event_type == EventType.EVENT_TYPE_WORKFLOW_TASK_STARTED
                for e in history.events
            )

        async def wait_for(
            handle: WorkflowHandle[Any, Any], event: EventType.ValueType
        ) -> None:
            for _ in range(100):
                history = await handle.fetch_history()
                if any(e.event_type == event for e in history.events):
                    return
                await asyncio.sleep(0.1)
            raise AssertionError(f"no {event} in history")

        task_queue = str(uuid.uuid4())
        # A worker that stops with the workflow cached drops its instance with
        # no eviction, so the second worker below takes it from a cache of 0.
        restarts = task_fails == "task" and fail
        async with new_worker(
            temporal_client,
            LangChainRaceWorkflow,
            task_queue=task_queue,
            workflow_runner=runner,
            max_cached_workflows=0 if restarts else max_cached_workflows,
        ):
            handle = await temporal_client.start_workflow(
                LangChainRaceWorkflow.run,
                deferred,
                id=f"replay-test-{uuid.uuid4()}",
                task_queue=task_queue,
            )
            await wait_for(handle, EventType.EVENT_TYPE_WORKFLOW_TASK_COMPLETED)
            if rejected:
                started = await tasks_started(handle)
                await handle.signal(
                    LangChainRaceWorkflow.fail_slowly
                    if fail
                    else LangChainRaceWorkflow.release_slowly
                )
                # Sent once the signal's task has started, while it blocks.
                for _ in range(100):
                    if await tasks_started(handle) > started:
                        break
                    await asyncio.sleep(0.02)
                await handle.signal(LangChainRaceWorkflow.other)
            if not task_fails:
                if fail:
                    if not rejected:
                        await handle.signal(LangChainRaceWorkflow.fail)
                    with pytest.raises(WorkflowFailureError):
                        await handle.result()
                else:
                    if not rejected:
                        await handle.signal(LangChainRaceWorkflow.release)
                    assert await handle.result() == "a"
                if rejected:
                    history = await handle.fetch_history()
                    assert any(
                        e.workflow_task_failed_event_attributes.cause
                        == WorkflowTaskFailedCause.WORKFLOW_TASK_FAILED_CAUSE_UNHANDLED_COMMAND
                        for e in history.events
                    )
                if case == "replayed":
                    # Released only when the replay is done, or the eviction
                    # makes no end write and the wait times out.
                    try:
                        assert await asyncio.to_thread(paused.wait, 30)
                        await Replayer(
                            workflows=[LangChainRaceWorkflow],
                            plugins=[plugin],
                            data_converter=config["data_converter"],
                        ).replay_workflow(await handle.fetch_history())
                    finally:
                        resumed.set()
            elif task_fails == "codec":
                await handle.signal(
                    LangChainRaceWorkflow.fail_uncodable
                    if fail
                    else LangChainRaceWorkflow.release_uncodable
                )
                await assert_task_failed_and_runs_open(handle)
                # The same workflow goes on, and ends, once it can be encoded.
                codec.broken = False
                if fail:
                    with pytest.raises(WorkflowFailureError):
                        await handle.result()
                else:
                    assert await handle.result() == _UNCODABLE
            elif not fail:
                await handle.signal(LangChainRaceWorkflow.release_unserializable)
                await assert_task_failed_and_runs_open(handle)
        if task_fails == "task":
            if restarts:
                # Sent while no worker polls, both signals reach the workflow in
                # one activation: the first fails the workflow, the second
                # fails the workflow task.
                await handle.signal(LangChainRaceWorkflow.fail)
                await handle.signal(LangChainRaceWorkflow.crash)
                async with new_worker(
                    temporal_client,
                    LangChainRaceWorkflow,
                    task_queue=task_queue,
                    workflow_runner=runner,
                    max_cached_workflows=max_cached_workflows,
                ):
                    await assert_task_failed_and_runs_open(handle)
            await handle.terminate()
            return

        names = {str(run["id"]): run["name"] for op, run in writes if op == "post"}
        assert sorted(set(names.values())) == ["race_a", "race_b"]
        for name in ("race_a", "race_b"):
            # A released race_a's step finishes again in each workflow task
            # that fails, as any run does; each run open when the workflow
            # ends is ended once.
            repeats = (
                (task_fails == "codec" or rejected) and not fail and name == "race_a"
            )
            ops = [op for op, run in writes if names[str(run["id"])] == name]
            ops = sorted(set(ops) if repeats else ops)
            # langchain-core releases that defer a RunnableGenerator's inputs
            # post its run only when it ends; earlier ones post it when it
            # starts, like any other run.
            assert ops == ["patch", "post"] or (deferred and ops == ["post"]), name
            ended = end_writes(name)
            assert len(ended) == 1 or (repeats and ended), name
            # Only race_a's step can finish, and only if released.
            for run in ended:
                assert bool(run.get("error")) == (fail or name == "race_b"), name

    @pytest.mark.parametrize("handler", ["run", "update"])
    @pytest.mark.parametrize("first", [False, True])
    async def test_run_id_kept_when_add_temporal_runs_changes(
        self,
        client: Client,
        env: WorkflowEnvironment,  # type:ignore[reportUnusedParameter]
        first: bool,
        handler: str,
    ) -> None:
        """A run open across Workers with different ``add_temporal_runs`` keeps its ID.

        The run is the run method's ``wait``, or an update's ``step``, whose
        root factory, made only with ``add_temporal_runs`` off, draws from a
        generator apart from its LangChain runs'.
        """
        get_runtime_environment()
        langsmith.env.get_langchain_env_var_metadata()
        writes: list[tuple[str, dict[str, Any]]] = []

        class _RecordingClient(langsmith.Client):
            def _create_run(self, *args: Any, **kwargs: Any) -> None:
                writes.append(("post", args[0]))

            def _update_run(self, *args: Any, **kwargs: Any) -> None:
                writes.append(("patch", args[0]))

        def temporal_client(add_temporal_runs: bool) -> Client:
            config = client.config()
            config["plugins"] = [
                LangSmithPlugin(
                    client=_RecordingClient(
                        api_url="http://localhost:1",
                        api_key="test",
                        auto_batch_tracing=False,
                    ),
                    add_temporal_runs=add_temporal_runs,
                )
            ]
            return Client(**config)

        task_queue = str(uuid.uuid4())
        update: Any = None
        # A cache of 0, so the second Worker replays the open run.
        async with new_worker(
            temporal_client(first),
            LangChainSignalWorkflow,
            task_queue=task_queue,
            max_cached_workflows=0,
        ):
            handle = await client.start_workflow(
                LangChainSignalWorkflow.run,
                id=f"replay-test-{uuid.uuid4()}",
                task_queue=task_queue,
            )
            for _ in range(100):
                history = await handle.fetch_history()
                if any(
                    e.event_type == EventType.EVENT_TYPE_WORKFLOW_TASK_COMPLETED
                    for e in history.events
                ):
                    break
                await asyncio.sleep(0.1)
            if handler == "update":
                update = await handle.start_update(
                    LangChainSignalWorkflow.step,
                    args=[True],
                    wait_for_stage=WorkflowUpdateStage.ACCEPTED,
                )
        async with new_worker(
            temporal_client(not first),
            LangChainSignalWorkflow,
            task_queue=task_queue,
            max_cached_workflows=0,
        ):
            if update is not None:
                await handle.signal(LangChainSignalWorkflow.release_step)
                assert await update.result() == "update"
            await handle.signal(LangChainSignalWorkflow.release)
            assert await handle.result() == "secret"

        # Each Worker's client posted the run, under one ID, and the second
        # ended it.
        name = "wait" if handler == "run" else "step"
        [run_id] = {
            str(run["id"]) for op, run in writes if op == "post" and run["name"] == name
        }
        assert sorted(op for op, run in writes if str(run["id"]) == run_id) == [
            "patch",
            "post",
            "post",
        ]

    @pytest.mark.parametrize("case", ["", "open", "two", "called_again"])
    async def test_update_runs_keep_their_own_ids(
        self,
        client: Client,
        env: WorkflowEnvironment,  # type:ignore[reportUnusedParameter]
        case: str,
    ) -> None:
        """An update's runs don't take the IDs of the run method's runs.

        On a worker without the workflow cached, an update right after a first
        workflow task with no commands runs its handler, then the run method
        again, live in one task.  A replay runs the run method first.  With
        ``open``, the update's run stays open into a later task, whose replay
        must make it again under its own ID for its end to reach it.  With
        ``two``, two updates sent while no worker polls run in one task, and
        each keeps IDs of its own.  With ``called_again``, an interceptor
        outside the plugin's calls the update's handler again after it
        raises, in the same task, and the second call gets new IDs.
        """
        get_runtime_environment()
        langsmith.env.get_langchain_env_var_metadata()
        writes: list[tuple[str, dict[str, Any]]] = []

        class _RecordingClient(langsmith.Client):
            def _create_run(self, *args: Any, **kwargs: Any) -> None:
                writes.append(("post", args[0]))

            def _update_run(self, *args: Any, **kwargs: Any) -> None:
                writes.append(("patch", args[0]))

        plugin = LangSmithPlugin(
            client=_RecordingClient(
                api_url="http://localhost:1",
                api_key="test",
                auto_batch_tracing=False,
            )
        )
        config = client.config()
        extra: dict[str, Any] = {}
        if case == "called_again":
            # The Worker's interceptors are outside its plugins' ones.
            extra = {
                "plugins": [plugin],
                "interceptors": [_CallUpdateAgainInterceptor()],
                "workflow_runner": UnsandboxedWorkflowRunner(),
            }
        else:
            config["plugins"] = [plugin]
        temporal_client = Client(**config)
        task_queue = str(uuid.uuid4())

        def worker() -> Worker:
            return new_worker(
                temporal_client,
                LangChainSignalWorkflow,
                task_queue=task_queue,
                max_cached_workflows=0,
                **extra,
            )

        async with worker():
            handle = await temporal_client.start_workflow(
                LangChainSignalWorkflow.run,
                id=f"replay-test-{uuid.uuid4()}",
                task_queue=task_queue,
            )
            for _ in range(100):
                history = await handle.fetch_history()
                if any(
                    e.event_type == EventType.EVENT_TYPE_WORKFLOW_TASK_COMPLETED
                    for e in history.events
                ):
                    break
                await asyncio.sleep(0.1)
            if case != "two":
                update = await handle.start_update(
                    LangChainSignalWorkflow.step,
                    args=[case == "open", "", case == "called_again"],
                    wait_for_stage=WorkflowUpdateStage.ACCEPTED
                    if case == "open"
                    else WorkflowUpdateStage.COMPLETED,
                )
                if case == "open":
                    await handle.signal(LangChainSignalWorkflow.release_step)
                assert await update.result() == "update"
                await handle.signal(LangChainSignalWorkflow.release)
                assert await handle.result() == "secret"
        if case == "two":
            updates = [
                asyncio.create_task(
                    handle.execute_update(
                        LangChainSignalWorkflow.step, args=[False, tag]
                    )
                )
                for tag in ("_a", "_b")
            ]
            # Let the server admit both before a worker polls.
            await asyncio.sleep(1)
            async with worker():
                assert await asyncio.gather(*updates) == ["update", "update"]
                await handle.signal(LangChainSignalWorkflow.release)
                assert await handle.result() == "secret"
            # Both were accepted in one workflow task.
            history = await handle.fetch_history()
            kinds = [
                e.event_type
                for e in history.events
                if e.event_type
                in (
                    EventType.EVENT_TYPE_WORKFLOW_EXECUTION_UPDATE_ACCEPTED,
                    EventType.EVENT_TYPE_WORKFLOW_TASK_COMPLETED,
                )
            ]
            accepted = EventType.EVENT_TYPE_WORKFLOW_EXECUTION_UPDATE_ACCEPTED
            first = kinds.index(accepted)
            assert kinds[first : first + 2] == [accepted, accepted], kinds

        posts = [run for op, run in writes if op == "post"]
        names: dict[str, set[str]] = {}
        for run in posts:
            names.setdefault(str(run["id"]), set()).add(run["name"])
        assert all(len(each) == 1 for each in names.values()), names
        # No two posts of the updates' runs share an ID, as a second call of
        # one update, with the same names, would.
        update_ids = [str(run["id"]) for run in posts if run["name"] != "wait"]
        assert len(set(update_ids)) == len(update_ids), update_ids
        # The trace is the root factory's run, built for each handler call.
        steps = ["step_a", "step_b"] if case == "two" else ["step"]
        traces = {
            name: {str(run["trace_id"]) for run in posts if run["name"] == name}
            for name in ["wait", *steps]
        }
        for step in steps:
            assert traces[step] and not traces[step] & traces["wait"], traces
        # Each end goes to a run posted with the same name, trace and parent,
        # and the update's run gets its end.  Start times, and so dotted
        # orders, still come from one chain for the workflow.
        posted = {str(run["id"]): run for run in posts}
        patches = [run for op, run in writes if op == "patch"]
        for run in patches:
            post = posted.get(str(run["id"]))
            assert post is not None, run
            for key in ("name", "trace_id", "parent_run_id"):
                if run.get(key) is not None:
                    assert str(run[key]) == str(post.get(key)), (key, run, post)
        step_ids = {key for key, run in posted.items() if run["name"] in steps}
        ended = {str(run["id"]) for run in patches if run.get("end_time")}
        assert step_ids and step_ids <= ended, (step_ids, writes)

    async def test_update_generators_released(
        self,
        client: Client,
        env: WorkflowEnvironment,  # type:ignore[reportUnusedParameter]
    ) -> None:
        """An update's run ID generators go when its handler is done.

        They don't stay with the workflow while it is cached.
        """
        get_runtime_environment()
        langsmith.env.get_langchain_env_var_metadata()

        class _RecordingClient(langsmith.Client):
            def _create_run(self, *args: Any, **kwargs: Any) -> None:
                pass

            def _update_run(self, *args: Any, **kwargs: Any) -> None:
                pass

        def live_generators() -> int:
            gc.collect()
            return sum(type(o) is random.Random for o in gc.get_objects())

        config = client.config()
        config["plugins"] = [
            LangSmithPlugin(
                client=_RecordingClient(
                    api_url="http://localhost:1",
                    api_key="test",
                    auto_batch_tracing=False,
                )
            )
        ]
        temporal_client = Client(**config)
        async with new_worker(temporal_client, LangChainSignalWorkflow) as worker:
            handle = await temporal_client.start_workflow(
                LangChainSignalWorkflow.run,
                id=f"replay-test-{uuid.uuid4()}",
                task_queue=worker.task_queue,
            )
            assert (
                await handle.execute_update(LangChainSignalWorkflow.step, args=[False])
                == "update"
            )
            before = live_generators()
            for _ in range(50):
                assert (
                    await handle.execute_update(
                        LangChainSignalWorkflow.step, args=[False]
                    )
                    == "update"
                )
            # Two per update, 100 in all, if they stayed.
            assert live_generators() - before < 10
            await handle.signal(LangChainSignalWorkflow.release)
            assert await handle.result() == "secret"

    async def test_run_under_header_parent_keeps_start_time(
        self,
        client: Client,
        env: WorkflowEnvironment,  # type:ignore[reportUnusedParameter]
    ) -> None:
        """A run under a parent from headers is patched as it was posted.

        With a cache of 0 the run's patch comes after a replay, which reads
        the headers again.
        """
        get_runtime_environment()
        langsmith.env.get_langchain_env_var_metadata()
        writes: list[tuple[str, dict[str, Any]]] = []

        class _RecordingClient(langsmith.Client):
            def _create_run(self, *args: Any, **kwargs: Any) -> None:
                writes.append(("post", args[0]))

            def _update_run(self, *args: Any, **kwargs: Any) -> None:
                writes.append(("patch", args[0]))

        ls_client = _RecordingClient(
            api_url="http://localhost:1", api_key="test", auto_batch_tracing=False
        )
        config = client.config()
        config["plugins"] = [LangSmithPlugin(client=ls_client)]
        temporal_client = Client(**config)
        async with new_worker(
            temporal_client,
            TraceableHelperWorkflow,
            activities=[traceable_activity],
            max_cached_workflows=0,
        ) as worker:
            with langsmith.trace("client", client=ls_client):
                handle = await temporal_client.start_workflow(
                    TraceableHelperWorkflow.run,
                    id=f"replay-test-{uuid.uuid4()}",
                    task_queue=worker.task_queue,
                )
            await handle.result()

        def writes_of(op: str) -> list[dict[str, Any]]:
            return [run for o, run in writes if o == op and str(run["id"]) in ids]

        for _ in range(100):
            ids = {
                str(run["id"])
                for op, run in writes
                if op == "post" and run["name"] == "call_traceable_activity"
            }
            if ids and writes_of("patch"):
                break
            await asyncio.sleep(0.05)
        [posted] = writes_of("post")
        [patched] = writes_of("patch")

        def start(run: dict[str, Any]) -> datetime:
            value = run["start_time"]
            return (
                value if isinstance(value, datetime) else datetime.fromisoformat(value)
            )

        assert start(patched) == start(posted)
        assert patched["dotted_order"] == posted["dotted_order"]


# ---------------------------------------------------------------------------
# TestErrorTracing
# ---------------------------------------------------------------------------


class TestErrorTracing:
    async def test_activity_failure_marked(
        self,
        client: Client,
        env: WorkflowEnvironment,  # type:ignore[reportUnusedParameter]
    ) -> None:
        """A failing activity run is marked with an error."""
        temporal_client, collector, _ = _make_client_and_collector(
            client, add_temporal_runs=True
        )

        async with new_worker(
            temporal_client,
            ActivityFailureWorkflow,
            activities=[failing_activity],
            workflow_failure_exception_types=[ApplicationError],
            max_cached_workflows=0,
        ) as worker:
            handle = await temporal_client.start_workflow(
                ActivityFailureWorkflow.run,
                id=f"act-fail-{uuid.uuid4()}",
                task_queue=worker.task_queue,
            )
            with pytest.raises(WorkflowFailureError):
                await handle.result()

        expected = [
            "StartWorkflow:ActivityFailureWorkflow",
            "RunWorkflow:ActivityFailureWorkflow",
            "  StartActivity:failing_activity",
            "  RunActivity:failing_activity",
            "    failing_activity",
        ]
        assert_trace_hierarchy(build_trace_trees(collector), expected)
        # Verify the RunActivity run has an error
        activity_runs = [
            r for r in collector.runs if r.name == "RunActivity:failing_activity"
        ]
        assert len(activity_runs) == 1
        assert activity_runs[0].error == "ApplicationError: activity-failed"

    async def test_workflow_failure_marked(
        self,
        client: Client,
        env: WorkflowEnvironment,  # type:ignore[reportUnusedParameter]
    ) -> None:
        """A failing workflow run is marked with an error."""
        temporal_client, collector, _ = _make_client_and_collector(
            client, add_temporal_runs=True
        )

        async with new_worker(
            temporal_client,
            FailingWorkflow,
            workflow_failure_exception_types=[ApplicationError],
            max_cached_workflows=0,
        ) as worker:
            handle = await temporal_client.start_workflow(
                FailingWorkflow.run,
                id=f"wf-fail-{uuid.uuid4()}",
                task_queue=worker.task_queue,
            )
            with pytest.raises(WorkflowFailureError):
                await handle.result()

        expected = [
            "StartWorkflow:FailingWorkflow",
            "RunWorkflow:FailingWorkflow",
        ]
        assert_trace_hierarchy(build_trace_trees(collector), expected)
        # Verify the RunWorkflow run has an error
        wf_runs = [r for r in collector.runs if r.name == "RunWorkflow:FailingWorkflow"]
        assert len(wf_runs) == 1
        assert wf_runs[0].error == "ApplicationError: workflow-failed"

    async def test_benign_error_not_marked(
        self,
        client: Client,
        env: WorkflowEnvironment,  # type:ignore[reportUnusedParameter]
    ) -> None:
        """A benign ApplicationError does NOT mark the run as errored."""
        temporal_client, collector, _ = _make_client_and_collector(
            client, add_temporal_runs=True
        )

        async with new_worker(
            temporal_client,
            BenignErrorWorkflow,
            activities=[benign_failing_activity],
            workflow_failure_exception_types=[ApplicationError],
            max_cached_workflows=0,
        ) as worker:
            handle = await temporal_client.start_workflow(
                BenignErrorWorkflow.run,
                id=f"benign-{uuid.uuid4()}",
                task_queue=worker.task_queue,
            )
            with pytest.raises(WorkflowFailureError):
                await handle.result()

        expected = [
            "StartWorkflow:BenignErrorWorkflow",
            "RunWorkflow:BenignErrorWorkflow",
            "  StartActivity:benign_failing_activity",
            "  RunActivity:benign_failing_activity",
            "    benign_failing_activity",
        ]
        assert_trace_hierarchy(build_trace_trees(collector), expected)
        # The RunActivity run for benign error should NOT have error set
        activity_runs = [
            r for r in collector.runs if r.name == "RunActivity:benign_failing_activity"
        ]
        assert len(activity_runs) == 1
        assert activity_runs[0].error is None
        assert activity_runs[0].end_time is not None

    async def test_continue_as_new_ends_run(
        self,
        client: Client,
        env: WorkflowEnvironment,  # type:ignore[reportUnusedParameter]
    ) -> None:
        """The run that continues as new is ended without an error."""
        plugin, collector, _ = _make_plugin_and_collector(add_temporal_runs=True)
        config = client.config()
        config["plugins"] = [plugin]
        temporal_client = Client(**config)

        async with new_worker(
            temporal_client, ContinueAsNewWorkflow, max_cached_workflows=0
        ) as worker:
            result = await temporal_client.execute_workflow(
                ContinueAsNewWorkflow.run,
                0,
                id=f"continue-as-new-{uuid.uuid4()}",
                task_queue=worker.task_queue,
            )
        assert result == 1

        _wait_for_tracing_io(plugin)
        wf_runs = [
            r for r in collector.runs if r.name == "RunWorkflow:ContinueAsNewWorkflow"
        ]
        assert len(wf_runs) == 2
        assert all(r.end_time is not None for r in collector.runs)
        assert all(r.error is None for r in wf_runs)
        assert [r.outputs for r in wf_runs] == [
            {"status": "continued_as_new"},
            {"status": "ok"},
        ]

    async def test_cancelled_workflow_ends_run(
        self,
        client: Client,
        env: WorkflowEnvironment,  # type:ignore[reportUnusedParameter]
    ) -> None:
        """A cancelled workflow's run is ended as cancelled, not left open."""
        plugin, collector, _ = _make_plugin_and_collector(add_temporal_runs=True)
        config = client.config()
        config["plugins"] = [plugin]
        temporal_client = Client(**config)

        async with new_worker(
            temporal_client, WaitForeverWorkflow, max_cached_workflows=0
        ) as worker:
            handle = await temporal_client.start_workflow(
                WaitForeverWorkflow.run,
                id=f"cancelled-{uuid.uuid4()}",
                task_queue=worker.task_queue,
            )
            await handle.describe()
            await handle.cancel()
            with pytest.raises(WorkflowFailureError):
                await handle.result()

        _wait_for_tracing_io(plugin)
        wf_runs = [
            r for r in collector.runs if r.name == "RunWorkflow:WaitForeverWorkflow"
        ]
        assert len(wf_runs) == 1
        assert wf_runs[0].end_time is not None
        assert wf_runs[0].error is None
        assert wf_runs[0].outputs == {"status": "cancelled"}


# ---------------------------------------------------------------------------
# TestComprehensiveTracing
# ---------------------------------------------------------------------------


class TestComprehensiveTracing:
    @pytest.mark.requires_local_server
    async def test_comprehensive_with_temporal_runs(
        self, client: Client, env: WorkflowEnvironment
    ) -> None:
        """Full trace hierarchy, with a fresh worker answering the queries.

        The first worker runs the workflow to completion. The second worker has
        nothing cached, so it must replay the whole history — every span the
        first worker already emitted — to answer each query, and must emit only
        the query handler run. user_pipeline only wraps start_workflow, so
        query/signal/update traces are naturally separate root traces.
        """
        if env.supports_time_skipping:
            pytest.skip("Time-skipping server doesn't persist headers.")

        task_queue = f"comprehensive-{uuid.uuid4()}"
        workflow_id = f"comprehensive-{uuid.uuid4()}"
        collector = InMemoryRunCollector()
        mock_ls = make_mock_ls_client(collector)
        temporal_client_1 = _make_temporal_client(
            client, mock_ls, add_temporal_runs=True
        )

        @traceable(name="user_pipeline")
        async def user_pipeline() -> WorkflowHandle[Any, Any]:
            return await temporal_client_1.start_workflow(
                ComprehensiveWorkflow.run,
                id=workflow_id,
                task_queue=task_queue,
            )

        with tracing_context(client=mock_ls, enabled=True):
            # Start workflow — no worker yet, just a server RPC
            handle = await user_pipeline()

            # Phase 1: the worker runs the workflow to completion. Updates and
            # the signal are handled at the next workflow task wherever the run
            # is, so nothing here needs to know how far it has progressed.
            async with new_worker(
                temporal_client_1,
                ComprehensiveWorkflow,
                TraceableActivityWorkflow,
                activities=[nested_traceable_activity, traceable_activity],
                nexus_service_handlers=[NexusService()],
                task_queue=task_queue,
                max_cached_workflows=0,
            ) as worker:
                await env.create_nexus_endpoint(
                    make_nexus_endpoint_name(worker.task_queue),
                    worker.task_queue,
                )
                await handle.execute_update(
                    ComprehensiveWorkflow.my_unvalidated_update, "test"
                )
                await handle.execute_update(ComprehensiveWorkflow.my_update, "finish")
                await handle.signal(ComprehensiveWorkflow.my_signal, "hello")
                result = await handle.result()

            # Phase 2: a fresh worker answers queries on the completed workflow.
            # With nothing cached, each query replays the full history — every
            # span the first worker already emitted — and must add only the
            # query handler run.
            temporal_client_2 = _make_temporal_client(
                client, mock_ls, add_temporal_runs=True
            )
            async with new_worker(
                temporal_client_2,
                ComprehensiveWorkflow,
                TraceableActivityWorkflow,
                activities=[nested_traceable_activity, traceable_activity],
                nexus_service_handlers=[NexusService()],
                task_queue=task_queue,
                max_cached_workflows=0,
            ):
                handle_2 = temporal_client_2.get_workflow_handle(workflow_id)
                assert await handle_2.query(ComprehensiveWorkflow.my_query)
                # Raw-client query — root-level trace
                assert await client.get_workflow_handle(workflow_id).query(
                    ComprehensiveWorkflow.my_query
                )

        assert result == "comprehensive-done"

        trace_trees = build_trace_trees(collector)

        # user_pipeline trace: StartWorkflow + full workflow execution tree
        workflow_trace_trees = find_trace_trees(trace_trees, "user_pipeline")
        assert len(workflow_trace_trees) == 1
        expected_workflow = [
            "user_pipeline",
            "  StartWorkflow:ComprehensiveWorkflow",
            "  RunWorkflow:ComprehensiveWorkflow",
            "    StartActivity:nested_traceable_activity",
            "    RunActivity:nested_traceable_activity",
            "      nested_traceable_activity",
            "        outer_chain",
            "          inner_llm_call",
            # step-wrapped activity
            "    step_with_activity",
            "      StartActivity:nested_traceable_activity",
            "      RunActivity:nested_traceable_activity",
            "        nested_traceable_activity",
            "          outer_chain",
            "            inner_llm_call",
            "    StartActivity:nested_traceable_activity",
            "    RunActivity:nested_traceable_activity",
            "      nested_traceable_activity",
            "        outer_chain",
            "          inner_llm_call",
            "    outer_chain",
            "      inner_llm_call",
            "    StartChildWorkflow:TraceableActivityWorkflow",
            "    RunWorkflow:TraceableActivityWorkflow",
            "      StartActivity:traceable_activity",
            "      RunActivity:traceable_activity",
            "        traceable_activity",
            "          inner_llm_call",
            # step-wrapped child workflow
            "    step_with_child_workflow",
            "      StartChildWorkflow:TraceableActivityWorkflow",
            "      RunWorkflow:TraceableActivityWorkflow",
            "        StartActivity:traceable_activity",
            "        RunActivity:traceable_activity",
            "          traceable_activity",
            "            inner_llm_call",
            "    StartNexusOperation:NexusService/run_operation",
            "    RunStartNexusOperationHandler:NexusService/run_operation",
            "      StartWorkflow:TraceableActivityWorkflow",
            "      RunWorkflow:TraceableActivityWorkflow",
            "        StartActivity:traceable_activity",
            "        RunActivity:traceable_activity",
            "          traceable_activity",
            "            inner_llm_call",
            # step-wrapped nexus operation
            "    step_with_nexus",
            "      StartNexusOperation:NexusService/run_operation",
            "      RunStartNexusOperationHandler:NexusService/run_operation",
            "        StartWorkflow:TraceableActivityWorkflow",
            "        RunWorkflow:TraceableActivityWorkflow",
            "          StartActivity:traceable_activity",
            "          RunActivity:traceable_activity",
            "            traceable_activity",
            "              inner_llm_call",
            # post-signal
            "    StartActivity:nested_traceable_activity",
            "    RunActivity:nested_traceable_activity",
            "      nested_traceable_activity",
            "        outer_chain",
            "          inner_llm_call",
        ]
        assert_trace_hierarchy(workflow_trace_trees, expected_workflow)

        # Raw-client query — no parent context, appears as root
        raw_query_trace_trees = [
            trace for trace in trace_trees if trace.name.startswith("HandleQuery:")
        ]
        assert len(raw_query_trace_trees) == 1

        # Each remaining operation is its own root trace
        query_trace_trees = find_trace_trees(trace_trees, "QueryWorkflow:my_query")
        assert len(query_trace_trees) == 1
        assert_trace_hierarchy(
            query_trace_trees,
            [
                "QueryWorkflow:my_query",
                "  HandleQuery:my_query",
            ],
        )

        signal_trace_trees = find_trace_trees(trace_trees, "SignalWorkflow:my_signal")
        assert len(signal_trace_trees) == 1
        assert_trace_hierarchy(
            signal_trace_trees,
            [
                "SignalWorkflow:my_signal",
                "  HandleSignal:my_signal",
            ],
        )

        update_trace_trees = find_trace_trees(
            trace_trees, "StartWorkflowUpdate:my_update"
        )
        assert len(update_trace_trees) == 1
        assert_trace_hierarchy(
            update_trace_trees,
            [
                "StartWorkflowUpdate:my_update",
                "  ValidateUpdate:my_update",
                "  HandleUpdate:my_update",
            ],
        )

        # Update without a validator — no ValidateUpdate trace
        unvalidated_trace_trees = find_trace_trees(
            trace_trees, "StartWorkflowUpdate:my_unvalidated_update"
        )
        assert len(unvalidated_trace_trees) == 1
        assert_trace_hierarchy(
            unvalidated_trace_trees,
            [
                "StartWorkflowUpdate:my_unvalidated_update",
                "  HandleUpdate:my_unvalidated_update",
            ],
        )

    @pytest.mark.requires_local_server
    async def test_comprehensive_without_temporal_runs(
        self, client: Client, env: WorkflowEnvironment
    ) -> None:
        """Same workflow with add_temporal_runs=False and a fresh query worker.

        Only @traceable runs appear. Context propagation via headers still works.
        user_pipeline only wraps start_workflow, so the query trace is a separate root.
        """
        if env.supports_time_skipping:
            pytest.skip("Time-skipping server doesn't persist headers.")

        task_queue = f"comprehensive-no-runs-{uuid.uuid4()}"
        workflow_id = f"comprehensive-no-runs-{uuid.uuid4()}"
        collector = InMemoryRunCollector()
        mock_ls = make_mock_ls_client(collector)
        temporal_client_1 = _make_temporal_client(
            client, mock_ls, add_temporal_runs=False
        )

        @traceable(name="user_pipeline")
        async def user_pipeline() -> WorkflowHandle[Any, Any]:
            return await temporal_client_1.start_workflow(
                ComprehensiveWorkflow.run,
                id=workflow_id,
                task_queue=task_queue,
            )

        with tracing_context(client=mock_ls, enabled=True):
            handle = await user_pipeline()

            # Phase 1: the worker runs the workflow to completion. Updates and
            # the signal are handled at the next workflow task wherever the run
            # is, so nothing here needs to know how far it has progressed.
            async with new_worker(
                temporal_client_1,
                ComprehensiveWorkflow,
                TraceableActivityWorkflow,
                activities=[nested_traceable_activity, traceable_activity],
                nexus_service_handlers=[NexusService()],
                task_queue=task_queue,
                max_cached_workflows=0,
            ) as worker:
                await env.create_nexus_endpoint(
                    make_nexus_endpoint_name(worker.task_queue),
                    worker.task_queue,
                )
                await handle.execute_update(
                    ComprehensiveWorkflow.my_unvalidated_update, "test"
                )
                await handle.execute_update(ComprehensiveWorkflow.my_update, "finish")
                await handle.signal(ComprehensiveWorkflow.my_signal, "hello")
                result = await handle.result()

            # Phase 2: a fresh worker answers queries on the completed workflow.
            # With nothing cached, each query replays the full history — every
            # span the first worker already emitted — and must add only the
            # query handler run.
            temporal_client_2 = _make_temporal_client(
                client, mock_ls, add_temporal_runs=False
            )
            async with new_worker(
                temporal_client_2,
                ComprehensiveWorkflow,
                TraceableActivityWorkflow,
                activities=[nested_traceable_activity, traceable_activity],
                nexus_service_handlers=[NexusService()],
                task_queue=task_queue,
                max_cached_workflows=0,
            ):
                handle_2 = temporal_client_2.get_workflow_handle(workflow_id)
                # Raw-client query — no interceptor, produces nothing
                assert await client.get_workflow_handle(workflow_id).query(
                    ComprehensiveWorkflow.my_query
                )
                assert await handle_2.query(ComprehensiveWorkflow.my_query)

        assert result == "comprehensive-done"

        trace_trees = build_trace_trees(collector)

        # Main workflow trace (only @traceable runs, nested under user_pipeline)
        workflow_trace_trees = find_trace_trees(trace_trees, "user_pipeline")
        assert len(workflow_trace_trees) == 1
        expected_workflow = [
            "user_pipeline",
            "  nested_traceable_activity",
            "    outer_chain",
            "      inner_llm_call",
            "  step_with_activity",
            "    nested_traceable_activity",
            "      outer_chain",
            "        inner_llm_call",
            "  nested_traceable_activity",
            "    outer_chain",
            "      inner_llm_call",
            "  outer_chain",
            "    inner_llm_call",
            "  traceable_activity",
            "    inner_llm_call",
            "  step_with_child_workflow",
            "    traceable_activity",
            "      inner_llm_call",
            "  traceable_activity",
            "    inner_llm_call",
            "  step_with_nexus",
            "    traceable_activity",
            "      inner_llm_call",
            # post-signal
            "  nested_traceable_activity",
            "    outer_chain",
            "      inner_llm_call",
        ]
        assert_trace_hierarchy(workflow_trace_trees, expected_workflow)

        # Neither the traced nor the raw query leaves a Temporal run.
        assert not [
            trace
            for trace in trace_trees
            if trace.name.startswith(("QueryWorkflow:", "HandleQuery:"))
        ]


# ---------------------------------------------------------------------------
# TestBackgroundIOIntegration — _RootReplaySafeRunTreeFactory + sync @traceable
# ---------------------------------------------------------------------------


@traceable(name="sync_inner_llm_call")
def _sync_inner_llm_call(prompt: str) -> str:
    """Sync @traceable simulating an LLM call."""
    return f"sync-response to: {prompt}"


@traceable(name="sync_outer_chain")
def _sync_outer_chain(prompt: str) -> str:
    """Sync @traceable that calls another sync @traceable."""
    return _sync_inner_llm_call(prompt)


@traceable(name="async_calls_sync")
async def _async_calls_sync(prompt: str) -> str:
    """Async @traceable that calls a sync @traceable — the interesting mixed case."""
    return _sync_inner_llm_call(prompt)


@workflow.defn
class FactoryTraceableWorkflow:
    """Workflow exercising _RootReplaySafeRunTreeFactory with async, sync, and mixed @traceable.

    Covers three code paths through create_child:
    - async→async nesting
    - sync→sync nesting (sync @traceable entry to factory)
    - async→sync nesting (cross-boundary case)
    """

    @workflow.run
    async def run(self) -> str:
        r1 = await _outer_chain("async")
        r2 = _sync_outer_chain("sync")
        r3 = await _async_calls_sync("mixed")
        # Activity with nested @traceable
        await workflow.execute_activity(
            nested_traceable_activity,
            start_to_close_timeout=timedelta(minutes=1),
            retry_policy=common.RetryPolicy(maximum_attempts=1),
        )
        return f"{r1}|{r2}|{r3}"


class TestBackgroundIOIntegration:
    """Integration tests for workflows using add_temporal_runs=False without external context.

    Exercises the _RootReplaySafeRunTreeFactory path with sync, async, and mixed @traceable
    nesting. Verifies root-run creation, correct nesting hierarchy, and replay safety.
    """

    async def test_factory_traceable_no_external_context(
        self,
        client: Client,
        env: WorkflowEnvironment,  # type:ignore[reportUnusedParameter]
    ) -> None:
        """Exercises _RootReplaySafeRunTreeFactory: add_temporal_runs=False, no external context.

        Uses a workflow with async→async, sync→sync, and async→sync @traceable
        nesting, plus an activity with nested @traceable. Verifies:
        - Each top-level @traceable becomes a root run (factory creates root children)
        - Nested @traceable calls nest correctly under their parent
        - Activity @traceable also produces correct hierarchy
        - No phantom factory run appears in collected runs
        - No duplicate run IDs after replay (max_cached_workflows=0)
        """
        temporal_client, collector, _ = _make_client_and_collector(
            client, add_temporal_runs=False
        )

        async with new_worker(
            temporal_client,
            FactoryTraceableWorkflow,
            activities=[nested_traceable_activity],
            max_cached_workflows=0,
        ) as worker:
            handle = await temporal_client.start_workflow(
                FactoryTraceableWorkflow.run,
                id=f"factory-{uuid.uuid4()}",
                task_queue=worker.task_queue,
            )
            result = await handle.result()

        assert (
            result
            == "response to: async|sync-response to: sync|sync-response to: mixed"
        )

        expected = [
            "outer_chain",
            "  inner_llm_call",
            "sync_outer_chain",
            "  sync_inner_llm_call",
            "async_calls_sync",
            "  sync_inner_llm_call",
            "nested_traceable_activity",
            "  outer_chain",
            "    inner_llm_call",
        ]
        assert_trace_hierarchy(build_trace_trees(collector), expected)

        # Verify no duplicate run IDs (replay safety with max_cached_workflows=0)
        run_ids = [r.id for r in collector.runs]
        assert len(run_ids) == len(set(run_ids)), (
            f"Duplicate run IDs found (replay issue): {run_ids}"
        )

    async def test_factory_passes_project_name_to_children(
        self,
        client: Client,
        env: WorkflowEnvironment,  # type:ignore[reportUnusedParameter]
    ) -> None:
        """Factory children inherit project_name (session_name) from plugin config."""
        temporal_client, _collector, mock_ls_client = _make_client_and_collector(
            client, add_temporal_runs=False, project_name="my-ls-project"
        )

        async with new_worker(
            temporal_client,
            FactoryTraceableWorkflow,
            activities=[nested_traceable_activity],
            max_cached_workflows=0,
        ) as worker:
            handle = await temporal_client.start_workflow(
                FactoryTraceableWorkflow.run,
                id=f"factory-proj-{uuid.uuid4()}",
                task_queue=worker.task_queue,
            )
            await handle.result()

        # Verify create_run calls include session_name from project_name
        for call in mock_ls_client.create_run.call_args_list:
            session = call.kwargs.get("session_name")
            assert session == "my-ls-project", (
                f"Expected session_name='my-ls-project', got {session!r} "
                f"in create_run call: {call.kwargs.get('name')}"
            )

    async def test_mixed_sync_async_traceable_with_temporal_runs(
        self,
        client: Client,
        env: WorkflowEnvironment,  # type:ignore[reportUnusedParameter]
    ) -> None:
        """Exercises _ReplaySafeRunTree.create_child with mixed sync/async @traceable.

        With add_temporal_runs=True, the interceptor creates a real
        _ReplaySafeRunTree as parent. This test verifies create_child
        propagation works at every level regardless of sync/async, with
        correct parent-child hierarchy and no duplicate run IDs after replay.
        """
        temporal_client, collector, _ = _make_client_and_collector(
            client, add_temporal_runs=True
        )

        async with new_worker(
            temporal_client,
            FactoryTraceableWorkflow,
            activities=[nested_traceable_activity],
            max_cached_workflows=0,
        ) as worker:
            handle = await temporal_client.start_workflow(
                FactoryTraceableWorkflow.run,
                id=f"mixed-temporal-{uuid.uuid4()}",
                task_queue=worker.task_queue,
            )
            result = await handle.result()

        assert (
            result
            == "response to: async|sync-response to: sync|sync-response to: mixed"
        )

        # With add_temporal_runs=True, Temporal operations get their own runs.
        # @traceable calls nest under the RunWorkflow run.
        expected = [
            "StartWorkflow:FactoryTraceableWorkflow",
            "RunWorkflow:FactoryTraceableWorkflow",
            "  outer_chain",
            "    inner_llm_call",
            "  sync_outer_chain",
            "    sync_inner_llm_call",
            "  async_calls_sync",
            "    sync_inner_llm_call",
            "  StartActivity:nested_traceable_activity",
            "  RunActivity:nested_traceable_activity",
            "    nested_traceable_activity",
            "      outer_chain",
            "        inner_llm_call",
        ]
        assert_trace_hierarchy(build_trace_trees(collector), expected)

        # Verify no duplicate run IDs (replay safety with max_cached_workflows=0)
        run_ids = [r.id for r in collector.runs]
        assert len(run_ids) == len(set(run_ids)), (
            f"Duplicate run IDs found (replay issue): {run_ids}"
        )


# --- Nexus service with direct @traceable call in handler ---


@traceable(name="nexus_direct_traceable")
async def _nexus_direct_traceable(input: str) -> str:
    """A @traceable function called directly from a nexus handler."""
    return await _inner_llm_call(input)


@nexusrpc.handler.service_handler
class DirectTraceableNexusService:
    """Nexus service that calls @traceable directly (not via activity)."""

    @nexusrpc.handler.sync_operation
    async def direct_traceable_op(
        self,
        ctx: nexusrpc.handler.StartOperationContext,  # type:ignore[reportUnusedParameter]
        input: str,
    ) -> str:
        return await _nexus_direct_traceable(input)


@workflow.defn
class NexusDirectTraceableWorkflow:
    """Workflow that calls a nexus operation whose handler uses @traceable directly."""

    @workflow.run
    async def run(self) -> str:
        nexus_client = workflow.create_nexus_client(
            endpoint=make_nexus_endpoint_name(workflow.info().task_queue),
            service=DirectTraceableNexusService,
        )
        return await nexus_client.execute_operation(
            operation=DirectTraceableNexusService.direct_traceable_op,
            input="nexus-input",
        )


# ---------------------------------------------------------------------------
# TestNexusInboundTracing
# ---------------------------------------------------------------------------


class TestNexusInboundTracing:
    """Verifies nexus handlers receive tracing_context for @traceable collection."""

    @pytest.mark.requires_local_server
    async def test_nexus_direct_traceable_without_temporal_runs(
        self,
        client: Client,
        env: WorkflowEnvironment,
    ) -> None:
        """@traceable in nexus handler works with add_temporal_runs=False.

        The worker must be started OUTSIDE tracing_context so that nexus handler
        tasks inherit a clean contextvars state. Only the client call gets
        tracing_context — the interceptor's tracing_context setup (or lack
        thereof) is the only thing that should provide context to the handler.
        """
        if env.supports_time_skipping:
            pytest.skip("Time-skipping server doesn't persist headers.")

        task_queue = f"nexus-direct-{uuid.uuid4()}"
        collector = InMemoryRunCollector()
        mock_ls = make_mock_ls_client(collector)
        temporal_client = _make_temporal_client(
            client, mock_ls, add_temporal_runs=False
        )

        # Worker starts OUTSIDE tracing_context — nexus handler tasks get clean context
        async with new_worker(
            temporal_client,
            NexusDirectTraceableWorkflow,
            nexus_service_handlers=[DirectTraceableNexusService()],
            task_queue=task_queue,
            max_cached_workflows=0,
        ) as worker:
            await env.create_nexus_endpoint(
                make_nexus_endpoint_name(worker.task_queue),
                worker.task_queue,
            )
            # Only the client call gets tracing context, not the worker
            with tracing_context(client=mock_ls, enabled=True):
                handle = await temporal_client.start_workflow(
                    NexusDirectTraceableWorkflow.run,
                    id=f"nexus-direct-{uuid.uuid4()}",
                    task_queue=worker.task_queue,
                )
                result = await handle.result()

        assert result == "response to: nexus-input"

        # @traceable runs from inside the nexus handler should be collected
        # via the interceptor's tracing_context setup.
        expected = [
            "nexus_direct_traceable",
            "  inner_llm_call",
        ]
        assert_trace_hierarchy(build_trace_trees(collector), expected)


# ---------------------------------------------------------------------------
# TestBuiltinQueryFiltering
# ---------------------------------------------------------------------------


@workflow.defn
class QueryFilteringWorkflow:
    """Workflow with a user query and a signal to complete."""

    def __init__(self) -> None:
        self._complete = False

    @workflow.run
    async def run(self) -> str:
        await workflow.wait_condition(lambda: self._complete)
        return "done"

    @workflow.signal
    def complete(self) -> None:
        self._complete = True

    @workflow.query
    def my_query(self) -> str:
        return "query-result"


class TestBuiltinQueryFiltering:
    """Verifies __temporal_ prefixed queries are not traced."""

    async def test_temporal_prefixed_query_not_traced(
        self,
        client: Client,
    ) -> None:
        """__temporal_workflow_metadata query should not produce a trace,
        but user-defined queries should still be traced.

        Uses add_temporal_runs=False on the query client to suppress
        client-side QueryWorkflow traces, isolating the test to
        worker-side HandleQuery traces only.
        """

        task_queue = f"query-filter-{uuid.uuid4()}"
        collector = InMemoryRunCollector()
        mock_ls = make_mock_ls_client(collector)

        # Worker client: add_temporal_runs=True so HandleQuery traces are created
        worker_client = _make_temporal_client(client, mock_ls, add_temporal_runs=True)
        # Query client: add_temporal_runs=False to suppress client-side traces
        query_client = _make_temporal_client(client, mock_ls, add_temporal_runs=False)

        async with new_worker(
            worker_client,
            QueryFilteringWorkflow,
            task_queue=task_queue,
            max_cached_workflows=0,
        ) as worker:
            handle = await query_client.start_workflow(
                QueryFilteringWorkflow.run,
                id=f"query-filter-{uuid.uuid4()}",
                task_queue=worker.task_queue,
            )

            await handle.signal(QueryFilteringWorkflow.complete)
            assert await handle.result() == "done"

            # Queries on the completed workflow: a built-in one, which must NOT
            # be traced, then a user query, which must.
            await handle.query("__temporal_workflow_metadata")
            await handle.query(QueryFilteringWorkflow.my_query)

        # The built-in query leaves no run; everything else the worker did is here.
        # Roots are compared by name: the RunWorkflow run reaches the collector
        # asynchronously, so its order relative to the handler runs varies.
        assert_trace_hierarchy(
            sorted(build_trace_trees(collector), key=lambda trace: trace.name),
            [
                "HandleQuery:my_query",
                "HandleSignal:complete",
                "RunWorkflow:QueryFilteringWorkflow",
            ],
        )
