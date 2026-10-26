"""Golden histories: recorded Workflows must replay the same way.

``python -m tests.record_histories`` recorded them with the scripted runner (see its
``SCENARIOS``): refunds approved, rejected, for an unknown order, and cancelled during
the tool or while waiting for approval; calls in one message; Continue-As-New in a
long task, with live output, and in chats over Signals and Updates; approval by
Signal; a Claude Code tool with approval, and one whose steps fail before and after
its call could start; Edits and a Write whose steps return Claude Code's own record;
and tasks that fail. A change to the Workflow code that would
break Workflows already running fails here.

The ``first-version-*`` histories were recorded with the plugin as first published
(commit 766c647: one call at a time, the conversation in a session store). They
replay with this code, so Workflows started with that version keep their decisions
after an upgrade (on Workers that keep the session store they started with). New
decisions since then are behind ``workflow.patched``.
"""

from __future__ import annotations

import warnings
from typing import Any

import pytest

from temporalio.claude_agent_sdk import ClaudeAgentPlugin, DurableClaudeAgent
from temporalio.claude_agent_sdk.testing import ScriptedClaude
from temporalio.client import WorkflowHistory
from temporalio.common import RetryPolicy
from temporalio.worker import Replayer
from tests.endless.policy import count_policy
from tests.record_histories import HISTORIES, SCENARIOS, WORKFLOWS

FILES = sorted(p.name for p in HISTORIES.glob("*.json"))
FIRST_VERSION = "first-version-"


def test_every_scenario_has_a_golden_history() -> None:
    names = {name.split("-run-")[0].removesuffix(".json") for name in FILES}
    first = {n for n in names if n.startswith(FIRST_VERSION)}
    assert names - first == set(SCENARIOS)
    # Every scenario the first version could run (it had no calls in parallel, no
    # Claude Code tools as Activities, and no conversation in the Workflow).
    first_version = {n.removeprefix(FIRST_VERSION) for n in first}
    assert first_version <= set(SCENARIOS)
    assert set(SCENARIOS) - first_version == {
        "refund-approval-store",
        "parallel-approval",
        "bash-approval",
        "bash-retried",
        "file-steps",
    }


@pytest.mark.parametrize("file_name", FILES)
async def test_replay(file_name: str) -> None:
    history = WorkflowHistory.from_json(
        file_name, (HISTORIES / file_name).read_text(encoding="utf-8")
    )
    with warnings.catch_warnings(record=True) as recorder:
        warnings.filterwarnings(
            "always", message=r"Module .* was imported after initial workflow load"
        )
        await Replayer(
            workflows=WORKFLOWS,
            plugins=[ClaudeAgentPlugin(ScriptedClaude(count_policy))],
        ).replay_workflow(history)
    # Sandbox imports during an activation count toward the deadlock timeout.
    assert not [
        str(w.message)
        for w in recorder
        if "was imported after initial workflow load" in str(w.message)
    ]


@pytest.mark.parametrize(
    "change",
    [
        {"tool_activity_retry_policy": RetryPolicy(maximum_attempts=1)},
        {
            "tool_activity_retry_policy": RetryPolicy(
                maximum_attempts=5, non_retryable_error_types=["ToolCallNotRun"]
            )
        },
        {"tool_activity_retry_policy": None},
        {"repeatable_tools": ["Bash"]},
    ],
    ids=["one-attempt", "not-run-is-final", "no-policy", "repeatable"],
)
@pytest.mark.parametrize("file_name", ["bash-retried.json", "bash-approval.json"])
async def test_tool_step_settings_can_change_under_running_workflows(
    monkeypatch: pytest.MonkeyPatch, file_name: str, change: dict[str, Any]
) -> None:
    """Whether a tool step is tried again is decided by the step and recorded, so a
    Workflow that already tried a step again replays after its retry settings or
    ``repeatable_tools`` change (as with Temporal's own retries before)."""
    real = DurableClaudeAgent.__init__

    def changed(self: DurableClaudeAgent, *args: Any, **kwargs: Any) -> None:
        real(self, *args, **{**kwargs, **change})

    monkeypatch.setattr(DurableClaudeAgent, "__init__", changed)
    history = WorkflowHistory.from_json(
        file_name, (HISTORIES / file_name).read_text(encoding="utf-8")
    )
    await Replayer(
        workflows=WORKFLOWS,
        plugins=[ClaudeAgentPlugin(ScriptedClaude(count_policy))],
    ).replay_workflow(history)
