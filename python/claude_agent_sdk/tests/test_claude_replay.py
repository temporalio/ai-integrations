"""Replays the committed histories (histories/*.json) against today's workflow-side code, so a
change that would break a running Claude workflow's replay fails here. Re-record them with
``CLAUDE_RECORD_HISTORIES=1 make test PYTEST_ARGS=tests/test_claude_workflow.py`` after an
intended change.
"""

from __future__ import annotations

import pytest

from temporalio.claude_agent_sdk import ClaudeAgentPlugin
from temporalio.claude_agent_sdk._models import RUN_CLAUDE_TURN_ACTIVITY
from temporalio.client import WorkflowHistory
from temporalio.worker import Replayer

from .coding_assistant.manager import CodingAssistant
from .concurrent_tool_cancellation import ConcurrentToolCancellation
from .helpers.histories import FORBIDDEN, HISTORIES, decoded
from .helpers.workflows import ClaudeChat, ClaudeNative, ClaudeStructured

CASES: dict[str, type] = {
    "coding_assistant": CodingAssistant,
    "concurrent_tool_cancellation": ConcurrentToolCancellation,
    "native_approve": ClaudeNative,
    "session_chat": ClaudeChat,
    "structured_output": ClaudeStructured,
    "tool_turn": ClaudeChat,
}


def test_every_scenario_has_a_committed_history():
    assert sorted(p.stem for p in HISTORIES.glob("*.json")) == sorted(CASES)


@pytest.mark.parametrize("name", sorted(CASES))
async def test_a_committed_history_replays(name: str):
    text = (HISTORIES / f"{name}.json").read_text()
    assert not [f for f in FORBIDDEN if f in decoded(text)], (
        "machine-specific or secret values"
    )
    history = WorkflowHistory.from_json(name, text)
    expected_activity = (
        "cleanup_0"
        if name == "concurrent_tool_cancellation"
        else RUN_CLAUDE_TURN_ACTIVITY
    )
    assert expected_activity in decoded(text)

    result = await Replayer(
        workflows=[CASES[name]],
        plugins=[ClaudeAgentPlugin(register_activities=False)],
    ).replay_workflow(history, raise_on_replay_failure=False)

    assert result.replay_failure is None, result.replay_failure
