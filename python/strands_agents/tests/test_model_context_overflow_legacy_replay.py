"""A history recorded before the overflow translation existed still replays.

It covers the plugin's ``workflow_failure_exception_types`` entry for
``ContextWindowOverflowException``. That setting only changes what happens when the
exception escapes Workflow code, which an execution recorded before this change could
only do by failing its Workflow Task forever. Replaying such a history still works, so
that setting needs no patch.

Fixture provenance: the history was recorded on a local dev server with the code from
upstream ``main`` at 5e91c26, before this change, using the workflow below and
``handle.fetch_history()`` (only the worker identity and task queue name were
normalized). It was fetched while the Workflow Task was failing.

A history of an agent turn from that version no longer replays: ``TemporalAgent`` now
reads the model's config with an activity before its first model call.
"""

from datetime import timedelta
from pathlib import Path

from strands.types.exceptions import ContextWindowOverflowException

from temporalio import workflow
from temporalio.client import WorkflowHistory
from temporalio.strands_agents import StrandsPlugin
from temporalio.worker import Replayer

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
