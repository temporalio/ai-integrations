"""Tests for LangSmithPlugin construction, configuration, and end-to-end usage."""

from __future__ import annotations

import dataclasses
import importlib.metadata
import threading
import uuid
from collections.abc import Callable
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest
from langsmith import traceable, tracing_context

from temporalio.bridge.proto.workflow_activation import (
    RemoveFromCache,
    WorkflowActivation,
    WorkflowActivationJob,
)
from temporalio.bridge.proto.workflow_commands import (
    CancelWorkflowExecution,
    CompleteWorkflowExecution,
    ContinueAsNewWorkflowExecution,
    FailWorkflowExecution,
    WorkflowCommand,
)
from temporalio.bridge.proto.workflow_completion import (
    Failure,
    Success,
    WorkflowActivationCompletion,
)
from temporalio import workflow
from temporalio.client import Client, WorkflowHandle
from temporalio.langsmith import LangSmithInterceptor, LangSmithPlugin
from temporalio.langsmith._interceptor import _activation
from temporalio.langsmith._plugin import (
    _EndingEvictionInstance,
    _mark_ending_evictions,
    _Runner,
)
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import (
    UnsandboxedWorkflowRunner,
    WorkerConfig,
    WorkflowInstance,
    WorkflowRunner,
)
from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner
from tests.helpers import new_worker
from tests.helpers.langsmith import (
    build_trace_trees,
    find_trace_trees,
)
from tests.helpers.nexus import make_nexus_endpoint_name
from tests.helpers.trace import assert_trace_hierarchy
from tests.test_integration import (
    ComprehensiveWorkflow,
    NexusService,
    TraceableActivityWorkflow,
    _make_client_and_collector,
    nested_traceable_activity,
    traceable_activity,
)


class TestPluginConstruction:
    """Tests for LangSmithPlugin construction."""

    def test_construction_stores_all_config(self) -> None:
        """All constructor kwargs are stored on the interceptor."""
        mock_client = MagicMock()
        plugin = LangSmithPlugin(
            client=mock_client,
            project_name="my-project",
            add_temporal_runs=False,
            default_metadata={"env": "prod"},
            default_tags=["v1"],
        )
        assert plugin.interceptors is not None
        assert len(plugin.interceptors) > 0
        interceptor = plugin.interceptors[0]
        assert isinstance(interceptor, LangSmithInterceptor)
        assert interceptor._client is mock_client
        assert interceptor._project_name == "my-project"
        assert interceptor._add_temporal_runs is False
        assert interceptor._default_metadata == {"env": "prod"}
        assert interceptor._default_tags == ["v1"]

    def test_reports_package_version(self) -> None:
        plugin = LangSmithPlugin(client=MagicMock())
        assert plugin.version() == importlib.metadata.version("temporalio-langsmith")


@workflow.defn(sandboxed=False)
class _UnsandboxedMarkWorkflow:
    """Returns whether its first activation was marked.

    A runner of the plugin's marks every activation.  Without one,
    ``_watch_activations`` marks them from the second on.
    """

    @workflow.run
    async def run(self) -> bool:
        return _activation.marked


class TestEndingEvictions:
    """The eviction after a workflow ends is marked, and no other."""

    def test_marked_after_workflow_ends(self) -> None:
        """Only an eviction after an accepted, not replayed, ending task is marked."""
        marked: list[bool] = []
        completions: list[WorkflowActivationCompletion] = []

        def activate(act: WorkflowActivation) -> WorkflowActivationCompletion:
            assert _activation.marked
            marked.append(_activation.ending_run_id == act.run_id)
            return completions.pop(0)

        # Completions are made per case: the worker can change one after
        # ``activate`` returns it.
        def ends() -> WorkflowActivationCompletion:
            return ends_with(
                WorkflowCommand(complete_workflow_execution=CompleteWorkflowExecution())
            )()

        def ends_with(
            command: WorkflowCommand,
        ) -> Callable[[], WorkflowActivationCompletion]:
            return lambda: WorkflowActivationCompletion(
                successful=Success(commands=[command])
            )

        def waits() -> WorkflowActivationCompletion:
            return WorkflowActivationCompletion(successful=Success())

        def fails() -> WorkflowActivationCompletion:
            return WorkflowActivationCompletion(failed=Failure())

        # What the worker does with a completion once ``activate`` has
        # returned it: it sends it, or fails it there if encoding its payloads
        # fails.  After a deadlock it sends another completion instead.
        def sent(completion: WorkflowActivationCompletion) -> None:
            completion.run_id = "run"

        def encoding_fails(completion: WorkflowActivationCompletion) -> None:
            sent(completion)
            completion.failed.failure.message = "Failed encoding completion"

        def not_sent(_: WorkflowActivationCompletion) -> None:
            pass

        def task(replaying: bool = False) -> WorkflowActivation:
            return WorkflowActivation(run_id="run", is_replaying=replaying)

        def eviction(reason: Any) -> WorkflowActivation:
            job = WorkflowActivationJob(
                remove_from_cache=RemoveFromCache(reason=reason)
            )
            return WorkflowActivation(run_id="run", is_replaying=True, jobs=[job])

        reasons = RemoveFromCache.EvictionReason
        fail = WorkflowCommand(fail_workflow_execution=FailWorkflowExecution())
        cancel = WorkflowCommand(cancel_workflow_execution=CancelWorkflowExecution())
        continue_as_new = WorkflowCommand(
            continue_as_new_workflow_execution=ContinueAsNewWorkflowExecution()
        )
        cases: list[
            tuple[
                list[
                    tuple[
                        WorkflowActivation,
                        Callable[[], WorkflowActivationCompletion],
                        Callable[[WorkflowActivationCompletion], None],
                    ]
                ],
                Any,
                bool,
            ]
        ] = [
            # The workflow ends; Core evicts it, from a full cache or not.
            ([(task(), ends, sent)], reasons.WORKFLOW_EXECUTION_ENDING, True),
            ([(task(), ends, sent)], reasons.CACHE_FULL, True),
            # It fails, is cancelled or continues as new.
            ([(task(), ends_with(fail), sent)], reasons.CACHE_FULL, True),
            ([(task(), ends_with(cancel), sent)], reasons.CACHE_FULL, True),
            ([(task(), ends_with(continue_as_new), sent)], reasons.CACHE_FULL, True),
            # The task that would end it fails, as when its result can't be
            # serialized, or after a signal handler failed the workflow.  With
            # no room in the cache, Core gives the eviction after it as full.
            ([(task(), fails, sent)], reasons.LANG_FAIL, False),
            ([(task(), fails, sent)], reasons.CACHE_FULL, False),
            (
                [(task(), ends, sent), (task(), fails, sent)],
                reasons.CACHE_FULL,
                False,
            ),
            # Encoding the completion fails, as when a payload codec raises,
            # after the instance returned it: the ending one, or a later one.
            ([(task(), ends, encoding_fails)], reasons.CACHE_FULL, False),
            ([(task(), ends, encoding_fails)], reasons.LANG_FAIL, False),
            (
                [(task(), ends, sent), (task(), waits, encoding_fails)],
                reasons.CACHE_FULL,
                False,
            ),
            (
                [(task(), ends, sent), (task(), waits, sent)],
                reasons.CACHE_FULL,
                True,
            ),
            # The activation that would end it deadlocks, and returns later.
            ([(task(), ends, not_sent)], reasons.CACHE_FULL, False),
            # The server rejects the completion.
            ([(task(), ends, sent)], reasons.UNHANDLED_COMMAND, False),
            # A replayed end, as in the replayer, and no end.
            ([(task(replaying=True), ends, sent)], reasons.CACHE_FULL, False),
            ([(task(), waits, sent)], reasons.CACHE_FULL, False),
        ]
        for tasks, reason, expected in cases:
            inner = MagicMock(spec=WorkflowInstance)
            inner.activate.side_effect = activate
            instance = _EndingEvictionInstance(inner)
            completions[:] = [make() for _, make, _ in tasks] + [waits()]
            del marked[:]
            for act, _, change in tasks:
                change(instance.activate(act))
            instance.activate(eviction(reason))
            assert marked == [False] * len(tasks) + [expected], (tasks, reason)
            assert not _activation.marked and _activation.ending_run_id is None

    def test_mark_is_the_activations_own(self) -> None:
        """Another activation of the same workflow doesn't change the mark.

        While the eviction after the workflow ended is between end writes,
        another activation of the same run, as in a replay with the plugin,
        starts and returns in another thread.
        """
        evicting, replayed = threading.Event(), threading.Event()
        seen: list[str | None] = []

        def ends() -> WorkflowActivationCompletion:
            return WorkflowActivationCompletion(
                successful=Success(
                    commands=[
                        WorkflowCommand(
                            complete_workflow_execution=CompleteWorkflowExecution()
                        )
                    ]
                )
            )

        def activate_live(act: WorkflowActivation) -> WorkflowActivationCompletion:
            if not act.jobs:
                return ends()
            seen.append(_activation.ending_run_id)
            evicting.set()
            assert replayed.wait(10)
            seen.append(_activation.ending_run_id)
            return WorkflowActivationCompletion(successful=Success())

        live_inner = MagicMock(spec=WorkflowInstance)
        live_inner.activate.side_effect = activate_live
        live = _EndingEvictionInstance(live_inner)
        replay_inner = MagicMock(spec=WorkflowInstance)
        replay_inner.activate.side_effect = lambda _: ends()
        replay = _EndingEvictionInstance(replay_inner)

        live.activate(WorkflowActivation(run_id="run")).run_id = "run"
        job = WorkflowActivationJob(
            remove_from_cache=RemoveFromCache(
                reason=RemoveFromCache.EvictionReason.CACHE_FULL
            )
        )
        eviction = threading.Thread(
            target=live.activate,
            args=(WorkflowActivation(run_id="run", is_replaying=True, jobs=[job]),),
        )
        eviction.start()
        try:
            assert evicting.wait(10)
            replay.activate(WorkflowActivation(run_id="run", is_replaying=True))
        finally:
            replayed.set()
            eviction.join(10)
        assert seen == ["run", "run"]
        assert not _activation.marked and _activation.ending_run_id is None

    @pytest.mark.parametrize("subclassed", [False, True])
    def test_runners_marked(self, subclassed: bool) -> None:
        """Both of a worker's runners mark ending evictions.

        Each keeps its class's behavior and stays an instance of it, so a
        plugin after this one can still change a ``SandboxedWorkflowRunner``'s
        restrictions.  With ``subclassed``, the runners are subclasses that
        make their own instances.
        """
        made: list[WorkflowInstance] = []

        class _Sandboxed(SandboxedWorkflowRunner):
            def create_instance(self, det: Any) -> WorkflowInstance:
                made.append(MagicMock(spec=WorkflowInstance))
                return made[-1]

        class _Unsandboxed(UnsandboxedWorkflowRunner):
            def create_instance(self, det: Any) -> WorkflowInstance:
                made.append(MagicMock(spec=WorkflowInstance))
                return made[-1]

        client = MagicMock()
        client.config.return_value = {}
        config = LangSmithPlugin(client=MagicMock()).configure_worker(
            WorkerConfig(
                client=cast(Client, client),
                workflow_runner=_Sandboxed()
                if subclassed
                else SandboxedWorkflowRunner(),
                unsandboxed_workflow_runner=(
                    _Unsandboxed() if subclassed else UnsandboxedWorkflowRunner()
                ),
            )
        )
        sandboxed = config.get("workflow_runner")
        assert isinstance(
            sandboxed, _Sandboxed if subclassed else SandboxedWorkflowRunner
        )
        assert "langsmith" in sandboxed.restrictions.passthrough_modules
        sandboxed = dataclasses.replace(
            sandboxed,
            restrictions=sandboxed.restrictions.with_passthrough_modules("other"),
        )
        unsandboxed = config.get("unsandboxed_workflow_runner")
        assert isinstance(
            unsandboxed, _Unsandboxed if subclassed else UnsandboxedWorkflowRunner
        )
        for runner, base in (
            (sandboxed, SandboxedWorkflowRunner),
            (unsandboxed, UnsandboxedWorkflowRunner),
        ):
            if subclassed:
                instance = runner.create_instance(MagicMock())
                inner = made[-1]
            else:
                with patch.object(base, "create_instance") as create_instance:
                    instance = runner.create_instance(MagicMock())
                inner = create_instance.return_value
            assert isinstance(instance, _EndingEvictionInstance), runner
            assert instance._instance is inner, runner
            # Marking again changes nothing.
            assert _mark_ending_evictions(runner) is runner

    @pytest.mark.parametrize("delegating", [False, True])
    def test_runner_that_cannot_be_copied_marked(self, delegating: bool) -> None:
        """A runner that can't be copied with a subclass is wrapped instead.

        Its class can't be subclassed, or, with ``delegating``, it delegates
        through ``__getattr__``, on which ``copy.copy`` recurses.
        """

        class _Final(UnsandboxedWorkflowRunner):
            def __init_subclass__(cls, **kwargs: Any) -> None:
                raise TypeError("final")

        class _Delegating(WorkflowRunner):
            def __init__(self) -> None:
                self._inner = UnsandboxedWorkflowRunner()

            def __getattr__(self, name: str) -> Any:
                return getattr(self._inner, name)

            def prepare_workflow(self, defn: Any) -> None:
                self._inner.prepare_workflow(defn)

            def create_instance(self, det: Any) -> WorkflowInstance:
                return self._inner.create_instance(det)

        runner = _mark_ending_evictions(_Delegating() if delegating else _Final())
        with patch.object(
            UnsandboxedWorkflowRunner, "create_instance"
        ) as create_instance:
            instance = runner.create_instance(MagicMock())
        assert isinstance(instance, _EndingEvictionInstance)
        assert instance._instance is create_instance.return_value

    def test_wrappers_forward(self) -> None:
        """The wrapping runner and instance pass every call to what they wrap."""
        inner_runner = MagicMock(spec=WorkflowRunner)
        runner = _Runner(inner_runner)
        defn, det, types = MagicMock(), MagicMock(), [ValueError]
        runner.prepare_workflow(defn)
        inner_runner.prepare_workflow.assert_called_once_with(defn)
        runner.set_worker_level_failure_exception_types(types)
        inner_runner.set_worker_level_failure_exception_types.assert_called_once_with(
            types
        )
        instance = runner.create_instance(det)
        inner_runner.create_instance.assert_called_once_with(det)
        assert isinstance(instance, _EndingEvictionInstance)
        assert instance._instance is inner_runner.create_instance.return_value

        inner = MagicMock(spec=WorkflowInstance)
        instance = _EndingEvictionInstance(inner)
        command_info = MagicMock()
        assert (
            instance.get_serialization_context(command_info)
            is inner.get_serialization_context.return_value
        )
        inner.get_serialization_context.assert_called_once_with(command_info)
        assert (
            instance.get_external_store_context(command_info)
            is inner.get_external_store_context.return_value
        )
        inner.get_external_store_context.assert_called_once_with(command_info)
        assert instance.get_info() is inner.get_info.return_value
        assert instance.get_thread_id() is inner.get_thread_id.return_value

    async def test_unsandboxed_workflow_marked(self, client: Client) -> None:
        """A workflow not sandboxed runs on the plugin's runner from the start."""
        config = client.config()
        config["plugins"] = [LangSmithPlugin(client=MagicMock())]
        temporal_client = Client(**config)
        async with new_worker(temporal_client, _UnsandboxedMarkWorkflow) as worker:
            assert await temporal_client.execute_workflow(
                _UnsandboxedMarkWorkflow.run,
                id=f"marked-{uuid.uuid4()}",
                task_queue=worker.task_queue,
            )


class TestPluginIntegration:
    """End-to-end test using LangSmithPlugin as a Temporal client plugin."""

    @pytest.mark.requires_local_server
    async def test_comprehensive_plugin_trace_hierarchy(
        self, client: Client, env: WorkflowEnvironment
    ) -> None:
        """Plugin wired to a real Temporal worker produces the full trace hierarchy.

        user_pipeline only wraps start_workflow, so query/signal/update
        traces are naturally separate root traces.
        """
        if env.supports_time_skipping:
            pytest.skip("Time-skipping server doesn't persist headers.")

        temporal_client, collector, mock_ls_client = _make_client_and_collector(
            client, add_temporal_runs=True
        )

        task_queue = f"plugin-comprehensive-{uuid.uuid4()}"
        workflow_id = f"plugin-comprehensive-{uuid.uuid4()}"

        @traceable(name="user_pipeline")
        async def user_pipeline() -> WorkflowHandle[Any, Any]:
            return await temporal_client.start_workflow(
                ComprehensiveWorkflow.run,
                id=workflow_id,
                task_queue=task_queue,
            )

        with tracing_context(client=mock_ls_client, enabled=True):
            handle = await user_pipeline()

            async with new_worker(
                temporal_client,
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
                # Updates and the signal are handled at the next workflow task
                # wherever the run is; the queries then hit the completed
                # workflow, which has no workflow task in flight by definition.
                await handle.execute_update(
                    ComprehensiveWorkflow.my_unvalidated_update, "test"
                )
                await handle.execute_update(ComprehensiveWorkflow.my_update, "finish")
                await handle.signal(ComprehensiveWorkflow.my_signal, "hello")
                result = await handle.result()
                assert await handle.query(ComprehensiveWorkflow.my_query)

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
