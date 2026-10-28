"""Histories recorded before the overflow translation existed still replay.

Before ``TemporalModel.stream`` rebuilt ``ContextWindowOverflowException``, a model
overflow reached the Workflow as an ``ActivityError``. A Workflow could catch it and
schedule a fallback Activity. The checked-in history is that execution: an
``invoke_model`` Activity that fails with a ``ContextWindowOverflowException``, then an
``old_overflow_fallback`` Activity. It carries no patch marker, so replaying it must
keep the ``ActivityError`` behavior; otherwise Strands would catch the rebuilt
exception, reduce the conversation, and schedule a second ``invoke_model``.

A second history covers the plugin's ``workflow_failure_exception_types`` entry for
``ContextWindowOverflowException``. That setting only changes what happens when the
exception escapes Workflow code, which an execution recorded before this change could
only do by failing its Workflow Task forever. Replaying such a history still works, so
that setting needs no patch.

Fixture provenance: both histories were recorded on a local dev server with the code
from upstream ``main`` at 5e91c26, before this change, using the workflows below and
``handle.fetch_history()`` (only the worker identity and task queue name were
normalized). The second one was fetched while the Workflow Task was failing. Delete
this test and the fixtures together once ``strands-agents.context-window-overflow-recovery``
goes through ``workflow.deprecate_patch``.
"""

from datetime import timedelta
from pathlib import Path

from strands.types.exceptions import ContextWindowOverflowException

from temporalio import workflow
from temporalio.client import WorkflowHistory
from temporalio.exceptions import ActivityError
from temporalio.strands_agents import StrandsPlugin, TemporalAgent
from temporalio.worker import Replayer
from tests.mock_model import MockModel

_HISTORY = (
    Path(__file__).parent / "histories" / "legacy_overflow_activity_error_fallback.json"
)


# The name must match the workflow type recorded in the fixture history.
@workflow.defn(name="LegacyOverflowFallbackWorkflow")
class LegacyOverflowFallbackWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        agent = TemporalAgent(
            model="small",
            messages=[
                {"role": "user", "content": [{"text": "q1"}]},
                {"role": "assistant", "content": [{"text": "a1"}]},
                {"role": "user", "content": [{"text": "q2"}]},
                {"role": "assistant", "content": [{"text": "a2"}]},
            ],
            start_to_close_timeout=timedelta(seconds=15),
        )
        try:
            return str(await agent.invoke_async(prompt))
        except ActivityError:
            return await workflow.execute_activity(
                "old_overflow_fallback",
                prompt,
                start_to_close_timeout=timedelta(seconds=15),
            )


async def test_replays_history_recorded_before_overflow_translation():
    history = WorkflowHistory.from_json(
        "legacy-overflow-fallback", _HISTORY.read_text()
    )
    await Replayer(
        workflows=[LegacyOverflowFallbackWorkflow],
        plugins=[StrandsPlugin(models={"small": lambda: MockModel([])})],
    ).replay_workflow(history)


_ESCAPING_HISTORY = (
    Path(__file__).parent
    / "histories"
    / "legacy_overflow_escaping_workflow_task_failure.json"
)


@workflow.defn(name="EscapingOverflowWorkflow")
class EscapingOverflowWorkflow:
    @workflow.run
    async def run(self) -> str:
        await workflow.execute_activity(
            "some_activity", start_to_close_timeout=timedelta(seconds=10)
        )
        raise ContextWindowOverflowException("raised in workflow code")


async def test_replays_history_of_overflow_escaping_workflow_code():
    # Recorded on 5e91c26, where the escaped exception failed the Workflow Task in a
    # loop. Registering it as a Workflow failure type does not change the commands.
    history = WorkflowHistory.from_json(
        "escaping-overflow", _ESCAPING_HISTORY.read_text()
    )
    await Replayer(
        workflows=[EscapingOverflowWorkflow],
        plugins=[StrandsPlugin(models={})],
    ).replay_workflow(history)
