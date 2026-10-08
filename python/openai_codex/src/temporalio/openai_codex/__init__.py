"""OpenAI Codex integration for Temporal: durable Codex agent loops with Workflow-owned approvals.

Installed from the ``temporalio-openai-codex`` distribution as ``temporalio.openai_codex``.

Codex runs its agent loop in an external ``codex app-server`` process, so this integration makes the
*process* durable instead of wrapping model calls. Codex runs its own tools (shell, ``apply_patch``, ...)
inside its sandbox; every action that needs approval is put to an ``approval_handler`` that runs in your
Workflow, so it can wait on a human or a policy for as long as it takes. The conversation (Codex's
append-only rollout) lives in the Workflow, so a Worker dying mid-turn just retries from the last
committed rollout. Your own tools can run beside Codex's, or instead of them (``native_tools=False``),
as Activities with exactly-once effects.

Worker side::

    plugin = CodexPlugin(env={"OPENAI_API_KEY": os.environ["OPENAI_API_KEY"]})
    client = await Client.connect("localhost:7233", plugins=[plugin])
    worker = Worker(client, task_queue="codex", workflows=[MyWorkflow])

Workflow side (see :mod:`temporalio.openai_codex.workflow`)::

    @workflow.defn
    class MyWorkflow:
        @workflow.init
        def __init__(self, workspace: str) -> None:
            self.codex = CodexSession(cwd=workspace, approval_handler=self.approve)

        async def approve(self, request: CodexApprovalRequest) -> CodexApprovalDecision:
            ...  # wait for a human, apply a policy, ...

        @workflow.run
        async def run(self, workspace: str) -> str:
            return (await self.codex.run("Fix the failing tests.")).text

.. warning::
    Pre-release. It relies on Codex app-server APIs that are themselves experimental
    (``dynamicTools``, ``thread/inject_items``) and on the rollout file format, so the Codex version is
    pinned by the ``bundled-codex`` extra.
"""

from ._activity import CodexActivities, CodexObserver, ObserverFactory
from ._models import (
    CODEX_APPROVAL_UPDATE,
    CODEX_RUN_SEGMENT_ACTIVITY,
    CodexApprovalDecision,
    CodexApprovalRequest,
    CodexPendingCall,
    CodexSegmentInput,
    CodexSegmentResult,
    CodexTokenUsage,
    CodexToolSpec,
    CodexTurnResult,
)
from ._plugin import CodexPlugin

__all__ = [
    "CODEX_APPROVAL_UPDATE",
    "CODEX_RUN_SEGMENT_ACTIVITY",
    "CodexActivities",
    "CodexApprovalDecision",
    "CodexApprovalRequest",
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
