"""ScriptedClaude, the scripted stand-in for Claude that tests use, keeps its contract.

Like the real runner, it continues from the checkpoint it is given, refuses a
conversation kept where it does not keep it, and gives a retried segment a new tool
call id. These tests drive it directly, without a Workflow.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from temporalio.claude_agent_sdk import (
    ConversationRef,
    DeferredCall,
    SegmentInput,
    SegmentOutput,
    ToolOutcome,
    ToolSpec,
    ToolStepInput,
)
from temporalio.claude_agent_sdk.testing import (
    Final,
    HistoryItem,
    ScriptedClaude,
    ToolCall,
)
from tests.endless.policy import count_policy

COUNT = [ToolSpec("count", "Count one step.", {"type": "object"})]


def first(prompt: str = "count to 1") -> SegmentInput:
    return SegmentInput(
        session_id="s1234567", prompt=prompt, tools=COUNT, transcript=[]
    )


def then(out: SegmentOutput, entries: list[Any], injected: Any) -> SegmentInput:
    """The next segment of a conversation the caller holds (as a Workflow would)."""
    return SegmentInput(
        session_id=out.session_id,
        prompt=None,
        tools=COUNT,
        checkpoint=out.checkpoint,
        transcript=entries,
        injected=cast("dict[str, ToolOutcome]", injected),
    )


async def test_a_result_can_come_as_the_converters_dict() -> None:
    claude = ScriptedClaude(count_policy)
    out = await claude.run(first(), 1)
    assert out.deferred is not None
    entries = list(out.transcript_add)
    result = {out.deferred.id: {"content": {"n": 1}, "is_error": False}}
    done = await claude.run(then(out, entries, result), 1)
    assert done.result == "counted to 1" and done.transcript_keep == 1


async def test_a_retried_segment_decides_again_with_a_new_call_id() -> None:
    claude = ScriptedClaude(count_policy)
    one = await claude.run(first(), 1)
    two = await claude.run(first(), 2)
    assert one.deferred is not None and two.deferred is not None
    assert two.deferred.id == f"{one.deferred.id}_2"


def both(prompt: str, history: list[HistoryItem]) -> list[ToolCall] | Final:
    del prompt
    if history:
        return Final(", ".join(str(h.content) for h in history))
    return [ToolCall("count", {"n": 1}), ToolCall("Bash", {"command": "ls"})]


async def test_calls_in_one_message_get_ids_in_order_and_their_kind() -> None:
    claude = ScriptedClaude(both)
    inp = first()
    inp.tool_activities = ["Bash"]
    out = await claude.run(inp, 1)
    assert out.deferred is not None
    calls = [out.deferred, *out.siblings]
    assert [c.id for c in calls] == ["toolu_s1234567_01", "toolu_s1234567_01a"]
    assert [c.kind for c in calls] == ["durable", "engine"]
    results = {c.id: ToolOutcome(content=c.name) for c in calls}
    done = await claude.run(then(out, list(out.transcript_add), results), 1)
    assert done.result == "count, Bash"


async def test_a_paused_call_must_get_its_result() -> None:
    claude = ScriptedClaude(count_policy)
    out = await claude.run(first(), 1)
    assert out.deferred is not None
    missing = await claude.run(then(out, list(out.transcript_add), {}), 1)
    assert missing.is_error
    assert missing.error == f"The paused call {out.deferred.id} got no result"


async def test_a_held_conversation_must_end_at_the_checkpoint() -> None:
    claude = ScriptedClaude(count_policy)
    out = await claude.run(first(), 1)
    entries = list(out.transcript_add)
    no_checkpoint = then(out, entries, {})
    no_checkpoint.checkpoint = None
    refused = await claude.run(no_checkpoint, 1)
    assert refused.error == "The Workflow holds a conversation but no checkpoint"
    elsewhere = then(out, entries, {})
    elsewhere.checkpoint = "cp_other"
    refused = await claude.run(elsewhere, 1)
    assert refused.error == (
        f"The conversation the Workflow holds ends at {out.checkpoint}, not at "
        "checkpoint cp_other"
    )
    with pytest.raises(RuntimeError, match="The Workflow holds no conversation"):
        await claude.run(then(out, [], {}), 1)


async def test_a_folder_keeps_sessions_but_never_a_held_conversation(
    tmp_path: Path,
) -> None:
    claude = ScriptedClaude(count_policy, tmp_path)
    out = await claude.run(first(), 1)
    assert out.transcript_keep is None  # kept in the folder, like a session store
    assert out.checkpoint is not None and out.deferred is not None
    held = then(out, [{"uuid": "x"}], {})
    with pytest.raises(RuntimeError, match="kept in its Workflow"):
        await claude.run(held, 1)
    unknown = then(out, [], {out.deferred.id: ToolOutcome(content={"n": 1})})
    unknown.transcript = None
    unknown.checkpoint = "cp_missing"
    with pytest.raises(RuntimeError, match="Unknown checkpoint cp_missing"):
        await claude.run(unknown, 1)


async def test_a_folder_from_an_older_version_still_continues(tmp_path: Path) -> None:
    """It kept one paused call as an object, not a list."""
    claude = ScriptedClaude(count_policy, tmp_path)
    call = {"id": "toolu_old", "name": "count", "input": {"n": 1}, "kind": "durable"}
    state = {"prompt": "count to 1", "history": [], "pending": call, "turn": 1}
    (tmp_path / "s1234567").mkdir()
    (tmp_path / "s1234567" / "cp_old.json").write_text(json.dumps(state))
    inp = SegmentInput(
        session_id="s1234567",
        prompt=None,
        tools=COUNT,
        checkpoint="cp_old",
        injected={"toolu_old": ToolOutcome(content={"n": 1})},
        conversation=ConversationRef(query="q", agent="0", entries=0),
    )
    done = await claude.run(inp, 1)
    assert done.result == "counted to 1"


async def test_tool_steps_run_the_stand_in_for_the_tool() -> None:
    async def later(args: dict[str, Any]) -> str:
        return f"ran {args['command']}"

    claude = ScriptedClaude(
        count_policy,
        engine_tools={
            "Bash": later,
            "Write": lambda args: ToolOutcome(content="refused", is_error=True),
        },
    )

    def step(name: str) -> ToolStepInput:
        call = DeferredCall(id="t", name=name, input={"command": "ls"}, kind="engine")
        return ToolStepInput(session_id="s", checkpoint="c", call=call)

    assert await claude.run_tool_step(step("Bash"), 1) == ToolOutcome(content="ran ls")
    assert await claude.run_tool_step(step("Write"), 1) == ToolOutcome(
        content="refused", is_error=True
    )
    assert await claude.run_tool_step(step("Grep"), 1) == ToolOutcome(
        content={"ran": "Grep"}
    )
