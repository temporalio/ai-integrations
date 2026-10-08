"""OpenAI Codex integration for Temporal: durable Codex agent loops with host-run tools.

Installed from the ``temporalio-openai-codex`` distribution as ``temporalio.openai_codex``.

Codex runs its agent loop in an external ``codex app-server`` process, so this integration makes the
*process* durable instead of wrapping model calls. Each turn is one or more segment Activities; the
conversation (Codex's append-only rollout) lives in the Workflow, and every tool the model calls is a
host tool you provide (Codex's built-in tools are turned off). A Worker dying mid-turn just retries
the segment from the last committed rollout, and a tool that already ran is not run again.

Worker side::

    plugin = CodexPlugin(env={"OPENAI_API_KEY": os.environ["OPENAI_API_KEY"]})
    client = await Client.connect("localhost:7233", plugins=[plugin])
    worker = Worker(client, task_queue="codex", workflows=[MyWorkflow], activities=[my_activity])

Workflow side (see :mod:`temporalio.openai_codex.workflow`)::

    @workflow.defn
    class MyWorkflow:
        def __init__(self) -> None:
            self.codex = CodexSession(tools=[codex_tool(lookup), activity_as_tool(record)])

        @workflow.run
        async def run(self, prompt: str) -> str:
            return (await self.codex.run(prompt)).text

.. warning::
    Pre-release. It relies on Codex app-server APIs that are themselves experimental
    (``dynamicTools``, ``thread/inject_items``) and on the rollout file format, so the Codex version is
    pinned by the ``bundled-codex`` extra.
"""

from ._activity import CodexActivities, CodexObserver, ObserverFactory
from ._models import (
    CODEX_RUN_SEGMENT_ACTIVITY,
    CodexPendingCall,
    CodexSegmentInput,
    CodexSegmentResult,
    CodexTokenUsage,
    CodexToolSpec,
    CodexTurnResult,
)
from ._plugin import CodexPlugin

__all__ = [
    "CODEX_RUN_SEGMENT_ACTIVITY",
    "CodexActivities",
    "CodexObserver",
    "CodexPendingCall",
    "CodexPlugin",
    "CodexSegmentInput",
    "CodexSegmentResult",
    "CodexTokenUsage",
    "CodexToolSpec",
    "CodexTurnResult",
    "ObserverFactory",
]
