"""Whole native batches executed by the published Claude Code engine."""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Any

import pytest
from claude_agent_sdk import HookMatcher

from temporalio.claude_agent_sdk import (
    ClaudeAgentPlugin,
    ClaudeAgentSdkRunner,
    SegmentInput,
    ToolSpec,
    ToolStepInput,
)
from temporalio.client import (
    Client,
    WorkflowHandle,
    WorkflowUpdateFailedError,
    WorkflowUpdateRPCTimeoutOrCancelledError,
)
from temporalio.worker import Worker
from tests.helpers.fake_messages_api import FakeMessagesAPI, engine_env, history_of
from tests.native_batches.activities import record
from tests.native_batches.workflows import ConcurrentNativeWorkflow, NativeWorkflow

TOOLS = [ToolSpec("unused", "unused durable tool", {"type": "object"})]


@pytest.mark.timeout(120)
async def test_native_read_edit_read_keeps_file_state(tmp_path: Path) -> None:
    """An Edit result survives the next segment without rereading or revalidation."""
    note = tmp_path / "note.txt"
    note.write_text("before\n")
    rounds = [
        [
            {
                "type": "tool_use",
                "id": "native_read",
                "name": "Read",
                "input": {"file_path": str(note)},
            }
        ],
        [
            {
                "type": "tool_use",
                "id": "native_edit",
                "name": "Edit",
                "input": {
                    "file_path": str(note),
                    "old_string": "before",
                    "new_string": "after",
                },
            }
        ],
        [
            {
                "type": "tool_use",
                "id": "native_after",
                "name": "Read",
                "input": {"file_path": str(note)},
            }
        ],
    ]

    def policy(body: dict[str, Any]) -> list[dict[str, Any]]:
        _, _, history = history_of(body)
        if len(history) < len(rounds):
            return rounds[len(history)]
        assert all(not h.is_error for h in history), history
        return [{"type": "text", "text": "done"}]

    api = FakeMessagesAPI(policy).start()
    runner = ClaudeAgentSdkRunner(
        cwd=str(tmp_path), env=engine_env(api, str(tmp_path / "cfg"))
    )
    transcript: list[dict[str, Any]] = []
    inp = SegmentInput(
        str(uuid.uuid4()),
        "edit the note",
        TOOLS,
        builtin_tools=["Read", "Edit"],
        tool_activities=["*"],
        execution_protocol=2,
    )
    try:
        for _ in range(4):
            segment = await runner.run(inp, 1)
            assert not segment.is_error, segment.error
            assert segment.transcript_keep is not None
            transcript[segment.transcript_keep :] = segment.transcript_add
            if segment.deferred is None:
                assert segment.result == "done"
                break
            calls = [segment.deferred, *segment.siblings]
            outcomes = await asyncio.gather(
                *(
                    runner.run_tool_step(
                        ToolStepInput(
                            segment.session_id,
                            segment.checkpoint or "",
                            call,
                            tools=TOOLS,
                            builtin_tools=inp.builtin_tools,
                            transcript=transcript,
                            execution_protocol=2,
                            batch=calls,
                        ),
                        1,
                    )
                    for call in calls
                )
            )
            assert all(not o.is_error for o in outcomes), outcomes
            assert all(o.entries for o in outcomes)
            inp = SegmentInput(
                segment.session_id,
                None,
                TOOLS,
                builtin_tools=inp.builtin_tools,
                checkpoint=segment.checkpoint,
                injected=dict(zip((c.id for c in calls), outcomes)),
                transcript=transcript,
                tool_activities=["*"],
                execution_protocol=2,
            )
        else:
            pytest.fail("native agent did not finish")
    finally:
        api.stop()
    assert note.read_text() == "after\n"
    assert api.errors == []


@pytest.mark.timeout(90)
async def test_native_question_uses_a_validated_workflow_update(
    client: Client, tmp_path: Path
) -> None:
    """A native question gets its recorded answer through the engine's permission API."""
    question = "Which color?"

    def policy(body: dict[str, Any]) -> list[dict[str, Any]]:
        _, _, history = history_of(body)
        if not history:
            return [
                {
                    "type": "tool_use",
                    "id": "native_question",
                    "name": "AskUserQuestion",
                    "input": {
                        "questions": [
                            {
                                "question": question,
                                "header": "Color",
                                "options": [
                                    {"label": "Red", "description": "Red"},
                                    {"label": "Blue", "description": "Blue"},
                                ],
                                "multiSelect": False,
                            }
                        ]
                    },
                }
            ]
        return [{"type": "text", "text": str(history[-1].content)}]

    api = FakeMessagesAPI(policy).start()
    runner = ClaudeAgentSdkRunner(
        cwd=str(tmp_path), env=engine_env(api, str(tmp_path / "cfg"))
    )
    queue = "native-question-" + uuid.uuid4().hex
    try:
        async with Worker(
            client,
            task_queue=queue,
            workflows=[NativeWorkflow],
            activities=[record],
            plugins=[ClaudeAgentPlugin(runner, heartbeat_every=0.5)],
        ):
            handle = await client.start_workflow(
                NativeWorkflow.run,
                args=["ask a question", ["AskUserQuestion"]],
                id=queue,
                task_queue=queue,
            )

            async def wait_for_question() -> None:
                while not await handle.query(NativeWorkflow.interactions):
                    await asyncio.sleep(0.05)

            await asyncio.wait_for(wait_for_question(), 20)
            with pytest.raises(WorkflowUpdateFailedError) as rejected:
                await handle.execute_update(
                    NativeWorkflow.respond, args=["native_question", {}]
                )
            assert "Answers must map" in str(rejected.value.cause)
            assert await handle.execute_update(
                NativeWorkflow.respond, args=["native_question", {question: "Red"}]
            )
            result = await handle.result()
            assert "Red" in result, result
            assert not await handle.query(NativeWorkflow.interactions)
    finally:
        api.stop()
    assert api.errors == []


@pytest.mark.timeout(60)
async def test_native_reads_use_engine_parallel_scheduler(tmp_path: Path) -> None:
    """Both Read hooks must be active at once, proving native parallel dispatch."""
    note = tmp_path / "note.txt"
    note.write_text("parallel reads\n")
    executing = False
    started: set[str] = set()
    both = asyncio.Event()

    async def observe(data: Any, tid: str | None, context: Any) -> Any:
        del context
        if executing and data["tool_name"] == "Read":
            assert tid is not None
            started.add(tid)
            if len(started) == 2:
                both.set()
            await asyncio.wait_for(both.wait(), 5)
        return {}

    api = FakeMessagesAPI(
        lambda _: [
            {
                "type": "tool_use",
                "id": tid,
                "name": "Read",
                "input": {"file_path": str(note)},
            }
            for tid in ("read_one", "read_two")
        ]
    ).start()
    runner = ClaudeAgentSdkRunner(
        cwd=str(tmp_path),
        env=engine_env(api, str(tmp_path / "cfg")),
        extra_options={"hooks": {"PreToolUse": [HookMatcher(hooks=[observe])]}},
    )
    try:
        segment = await runner.run(
            SegmentInput(
                str(uuid.uuid4()),
                "read twice",
                TOOLS,
                builtin_tools=["Read"],
                tool_activities=["*"],
                execution_protocol=2,
            ),
            1,
        )
        assert not segment.is_error and segment.deferred is not None
        calls = [segment.deferred, *segment.siblings]
        assert [c.id for c in calls] == ["read_one", "read_two"]
        executing = True
        outcomes = await asyncio.gather(
            *(
                runner.run_tool_step(
                    ToolStepInput(
                        segment.session_id,
                        segment.checkpoint or "",
                        call,
                        tools=TOOLS,
                        builtin_tools=["Read"],
                        transcript=segment.transcript_add,
                        execution_protocol=2,
                        batch=calls,
                    ),
                    1,
                )
                for call in calls
            )
        )
        assert all(not o.is_error for o in outcomes), outcomes
        assert started == {"read_one", "read_two"}
    finally:
        api.stop()


@pytest.mark.timeout(60)
async def test_native_writes_are_ordered_barriers(tmp_path: Path) -> None:
    """The native scheduler retains Write/Read/Write/Read order."""
    note = tmp_path / "new.txt"
    blocks = [
        {
            "type": "tool_use",
            "id": "write_one",
            "name": "Write",
            "input": {"file_path": str(note), "content": "one\n"},
        },
        {
            "type": "tool_use",
            "id": "read_one",
            "name": "Read",
            "input": {"file_path": str(note)},
        },
        {
            "type": "tool_use",
            "id": "write_two",
            "name": "Write",
            "input": {"file_path": str(note), "content": "two\n"},
        },
        {
            "type": "tool_use",
            "id": "read_two",
            "name": "Read",
            "input": {"file_path": str(note)},
        },
    ]
    api = FakeMessagesAPI(lambda _: blocks).start()
    runner = ClaudeAgentSdkRunner(
        cwd=str(tmp_path), env=engine_env(api, str(tmp_path / "cfg"))
    )
    try:
        segment = await runner.run(
            SegmentInput(
                str(uuid.uuid4()),
                "ordered writes",
                TOOLS,
                builtin_tools=["Read", "Write"],
                tool_activities=["*"],
                execution_protocol=2,
            ),
            1,
        )
        assert not segment.is_error and segment.deferred is not None
        assert not note.exists()
        calls = [segment.deferred, *segment.siblings]
        outcomes = await asyncio.gather(
            *(
                runner.run_tool_step(
                    ToolStepInput(
                        segment.session_id,
                        segment.checkpoint or "",
                        call,
                        tools=TOOLS,
                        builtin_tools=["Read", "Write"],
                        transcript=segment.transcript_add,
                        execution_protocol=2,
                        batch=calls,
                    ),
                    1,
                )
                for call in calls
            )
        )
        assert all(not o.is_error for o in outcomes), outcomes
        assert "one" in str(outcomes[1].blocks or outcomes[1].content)
        assert "two" in str(outcomes[3].blocks or outcomes[3].content)
        assert note.read_text() == "two\n"
    finally:
        api.stop()


@pytest.mark.timeout(120)
async def test_native_subagent_tools_are_activities_with_one_parent_slot(
    client: Client, tmp_path: Path
) -> None:
    """A parent Agent cannot starve its child's native or durable Activity."""
    effects = tmp_path / "effects.txt"
    subtask = "NATIVE CHILD: run the two tools"

    def policy(body: dict[str, Any]) -> list[dict[str, Any]]:
        uses, texts, history = history_of(body)
        if any(subtask in text for text in texts):
            if not history:
                return [
                    {
                        "type": "tool_use",
                        "id": "child_bash",
                        "name": "Bash",
                        "input": {"command": f"echo native >> {effects}"},
                    }
                ]
            if len(history) == 1:
                return [
                    {
                        "type": "tool_use",
                        "id": "child_record",
                        "name": "mcp__durable__record",
                        "input": {"path": str(effects), "text": "durable"},
                    }
                ]
            assert all(not h.is_error for h in history), history
            return [{"type": "text", "text": "child done"}]
        if not uses:
            return [
                {
                    "type": "tool_use",
                    "id": "parent_agent",
                    "name": "Agent",
                    "input": {
                        "description": "run a child",
                        "prompt": subtask,
                        "subagent_type": "general-purpose",
                    },
                }
            ]
        assert all(not h.is_error for h in history), history
        return [{"type": "text", "text": "parent done"}]

    api = FakeMessagesAPI(policy).start()
    runner = ClaudeAgentSdkRunner(
        cwd=str(tmp_path), env=engine_env(api, str(tmp_path / "cfg"))
    )
    queue = "native-child-" + uuid.uuid4().hex
    try:
        async with Worker(
            client,
            task_queue=queue,
            workflows=[NativeWorkflow],
            activities=[record],
            plugins=[ClaudeAgentPlugin(runner, heartbeat_every=0.5)],
            max_concurrent_activities=1,
        ):
            result = await client.execute_workflow(
                NativeWorkflow.run,
                args=["delegate", ["Agent", "Bash"]],
                id=queue,
                task_queue=queue,
            )
            assert result == "parent done"
            kinds = []
            async for event in client.get_workflow_handle(queue).fetch_history_events():
                if event.HasField("activity_task_scheduled_event_attributes"):
                    a = event.activity_task_scheduled_event_attributes
                    kinds.append((a.activity_type.name, a.activity_id))
            assert ("run_claude_tool_step", "tool-parent_agent") in kinds
            assert ("run_claude_tool_step", "tool-child_bash") in kinds
            assert ("record", "tool-child_record") in kinds
    finally:
        api.stop()
    assert effects.read_text().splitlines() == ["native", "durable"]
    assert api.errors == []


@pytest.mark.timeout(90)
async def test_native_child_recovers_original_pending_id(
    client: Client, tmp_path: Path
) -> None:
    """A lost controller restores the accepted child batch and skips its completed call."""
    effects = tmp_path / "recovery.txt"
    subtask = "RECOVER CHILD: run the accepted batch"
    parked = asyncio.Event()
    recovered = False

    async def observe(data: Any, tid: str | None, context: Any) -> Any:
        del context
        if tid == "child_pending" and not recovered:
            assert data["tool_name"] == "Bash"
            parked.set()
            await asyncio.Event().wait()
        return {}

    def policy(body: dict[str, Any]) -> list[dict[str, Any]]:
        uses, texts, history = history_of(body)
        if any(subtask in text for text in texts):
            if not history:
                return [
                    {
                        "type": "tool_use",
                        "id": tid,
                        "name": "Bash",
                        "input": {"command": f"echo {value} >> {effects}"},
                    }
                    for tid, value in (("child_done", "one"), ("child_pending", "two"))
                ]
            return [{"type": "text", "text": "child done"}]
        if not uses:
            return [
                {
                    "type": "tool_use",
                    "id": "parent_agent",
                    "name": "Agent",
                    "input": {
                        "description": "recover a child",
                        "prompt": subtask,
                        "subagent_type": "general-purpose",
                    },
                }
            ]
        return [{"type": "text", "text": "recovered"}]

    api = FakeMessagesAPI(policy).start()
    runner = ClaudeAgentSdkRunner(
        cwd=str(tmp_path),
        env=engine_env(api, str(tmp_path / "cfg")),
        extra_options={"hooks": {"PreToolUse": [HookMatcher(hooks=[observe])]}},
    )
    queue = "native-recovery-" + uuid.uuid4().hex
    try:
        async with Worker(
            client,
            task_queue=queue,
            workflows=[NativeWorkflow],
            activities=[record],
            plugins=[ClaudeAgentPlugin(runner, heartbeat_every=0.5)],
            max_concurrent_activities=1,
        ):
            handle = await client.start_workflow(
                NativeWorkflow.run,
                args=["delegate", ["Agent", "Bash"]],
                id=queue,
                task_queue=queue,
            )
            await asyncio.wait_for(parked.wait(), 20)
            assert effects.read_text().splitlines() == ["one"]
            batch = next(iter(runner._native_batches.values()))
            old = batch.task
            assert old is not None
            recovered = True
            old.cancel()
            await asyncio.gather(old, return_exceptions=True)
            assert await asyncio.wait_for(handle.result(), 30) == "recovered"
            calls = await handle.query(NativeWorkflow.calls)
            assert {c["id"] for c in calls} == {
                "parent_agent",
                "child_done",
                "child_pending",
            }
            assert all(c["status"] == "done" for c in calls), calls
    finally:
        api.stop()
    assert effects.read_text().splitlines() == ["one", "two"]
    assert api.errors == []


@pytest.mark.timeout(90)
async def test_mixed_batch_with_one_application_activity_slot(
    client: Client, tmp_path: Path
) -> None:
    """A custom tool before native calls cannot starve the native controller."""
    note = tmp_path / "mixed.txt"
    calls: list[dict[str, Any]] = [
        {
            "type": "tool_use",
            "id": "custom_first",
            "name": "mcp__durable__record",
            "input": {"path": str(note), "text": "before"},
        },
        {
            "type": "tool_use",
            "id": "native_read",
            "name": "Read",
            "input": {"file_path": str(note)},
        },
        {
            "type": "tool_use",
            "id": "native_edit",
            "name": "Edit",
            "input": {
                "file_path": str(note),
                "old_string": "before",
                "new_string": "after",
            },
        },
        {
            "type": "tool_use",
            "id": "custom_last",
            "name": "mcp__durable__record",
            "input": {"path": str(note), "text": "last"},
        },
    ]

    def policy(body: dict[str, Any]) -> list[dict[str, Any]]:
        _, _, history = history_of(body)
        if not history:
            return calls
        assert {h.id for h in history} == {c["id"] for c in calls}
        assert all(not h.is_error for h in history), history
        return [{"type": "text", "text": "mixed done"}]

    api = FakeMessagesAPI(policy).start()
    runner = ClaudeAgentSdkRunner(
        cwd=str(tmp_path), env=engine_env(api, str(tmp_path / "cfg"))
    )
    queue = "native-mixed-" + uuid.uuid4().hex
    try:
        async with Worker(
            client,
            task_queue=queue,
            workflows=[NativeWorkflow],
            activities=[record],
            plugins=[ClaudeAgentPlugin(runner, heartbeat_every=0.5)],
            max_concurrent_activities=1,
        ):
            result = await client.execute_workflow(
                NativeWorkflow.run,
                args=["run a mixed batch", ["Read", "Edit"]],
                id=queue,
                task_queue=queue,
            )
            assert result == "mixed done"
            assert note.read_text().splitlines() == ["after", "last"]
    finally:
        api.stop()
    assert api.errors == []


@pytest.mark.timeout(120)
async def test_large_native_batch_releases_activity_capacity(
    client: Client, tmp_path: Path
) -> None:
    """A batch larger than the native worker's capacity still makes progress."""
    note = tmp_path / "many.txt"
    note.write_text("many reads\n")
    calls: list[dict[str, Any]] = [
        {
            "type": "tool_use",
            "id": f"read_{i}",
            "name": "Read",
            "input": {"file_path": str(note)},
        }
        for i in range(110)
    ]

    def policy(body: dict[str, Any]) -> list[dict[str, Any]]:
        _, _, history = history_of(body)
        if not history:
            return calls
        assert len(history) == len(calls)
        assert all(not h.is_error for h in history), history
        return [{"type": "text", "text": "all read"}]

    api = FakeMessagesAPI(policy).start()
    runner = ClaudeAgentSdkRunner(
        cwd=str(tmp_path), env=engine_env(api, str(tmp_path / "cfg"))
    )
    queue = "native-many-" + uuid.uuid4().hex
    try:
        async with Worker(
            client,
            task_queue=queue,
            workflows=[NativeWorkflow],
            activities=[record],
            plugins=[ClaudeAgentPlugin(runner, heartbeat_every=0.5)],
            max_concurrent_activities=1,
        ):
            assert (
                await client.execute_workflow(
                    NativeWorkflow.run,
                    args=["read many times", ["Read"]],
                    id=queue,
                    task_queue=queue,
                )
                == "all read"
            )
            scheduled = [
                e.activity_task_scheduled_event_attributes.activity_id
                async for e in client.get_workflow_handle(queue).fetch_history_events()
                if e.HasField("activity_task_scheduled_event_attributes")
            ]
            assert {"tool-" + c["id"] for c in calls}.issubset(scheduled)
            assert not runner._native_batches
    finally:
        api.stop()
    assert api.errors == []


@pytest.mark.timeout(90)
async def test_native_tasks_survive_transcript_branches(
    client: Client, tmp_path: Path
) -> None:
    """TaskCreate/Get/Update share their native state across model and tool processes."""

    def policy(body: dict[str, Any]) -> list[dict[str, Any]]:
        _, _, history = history_of(body)
        names = ["TaskCreate", "TaskGet", "TaskUpdate", "TaskGet"]
        inputs = [
            {
                "subject": "durable native task",
                "description": "keep this task",
                "activeForm": "Keeping task",
            },
            {"taskId": "1"},
            {"taskId": "1", "status": "completed"},
            {"taskId": "1"},
        ]
        if len(history) < len(names):
            i = len(history)
            return [
                {
                    "type": "tool_use",
                    "id": f"task_{i}",
                    "name": names[i],
                    "input": inputs[i],
                }
            ]
        assert all(not h.is_error for h in history), history
        assert "durable native task" in str(history[1].content)
        assert "completed" in str(history[-1].content)
        return [{"type": "text", "text": "tasks kept"}]

    api = FakeMessagesAPI(policy).start()
    runner = ClaudeAgentSdkRunner(
        cwd=str(tmp_path), env=engine_env(api, str(tmp_path / "cfg"))
    )
    queue = "native-tasks-" + uuid.uuid4().hex
    try:
        async with Worker(
            client,
            task_queue=queue,
            workflows=[NativeWorkflow],
            activities=[record],
            plugins=[ClaudeAgentPlugin(runner, heartbeat_every=0.5)],
        ):
            assert (
                await client.execute_workflow(
                    NativeWorkflow.run,
                    args=[
                        "track a native task",
                        ["TaskCreate", "TaskGet", "TaskUpdate"],
                    ],
                    id=queue,
                    task_queue=queue,
                )
                == "tasks kept"
            )
    finally:
        api.stop()
    assert api.errors == []


@pytest.mark.timeout(90)
async def test_native_admission_retries_a_lost_update_reply(
    client: Client, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A lost admission reply reuses the accepted Update without repeating effects."""
    note = tmp_path / "reply.txt"
    note.write_text("accepted once\n")
    attempts: list[str] = []
    completed: list[str | None] = []
    original = WorkflowHandle.execute_update

    async def lose_reply(self: Any, update: Any, *args: Any, **kwargs: Any) -> Any:
        value = await original(self, update, *args, **kwargs)
        id = kwargs.get("id", "")
        if id.startswith("claude-ready-"):
            attempts.append(id)
            if len(attempts) == 1:
                raise WorkflowUpdateRPCTimeoutOrCancelledError()
        return value

    async def observe(data: Any, tid: str | None, context: Any) -> Any:
        del context
        if data["tool_name"] == "Read":
            completed.append(tid)
        return {}

    def policy(body: dict[str, Any]) -> list[dict[str, Any]]:
        _, _, history = history_of(body)
        if not history:
            return [
                {
                    "type": "tool_use",
                    "id": "retry_read",
                    "name": "Read",
                    "input": {"file_path": str(note)},
                }
            ]
        assert len(history) == 1 and not history[0].is_error, history
        return [{"type": "text", "text": "reply recovered"}]

    monkeypatch.setattr(WorkflowHandle, "execute_update", lose_reply)
    api = FakeMessagesAPI(policy).start()
    runner = ClaudeAgentSdkRunner(
        cwd=str(tmp_path),
        env=engine_env(api, str(tmp_path / "cfg")),
        extra_options={"hooks": {"PostToolUse": [HookMatcher(hooks=[observe])]}},
    )
    queue = "native-reply-" + uuid.uuid4().hex
    try:
        async with Worker(
            client,
            task_queue=queue,
            workflows=[NativeWorkflow],
            activities=[record],
            plugins=[ClaudeAgentPlugin(runner, heartbeat_every=0.5)],
            max_concurrent_activities=1,
        ):
            assert (
                await client.execute_workflow(
                    NativeWorkflow.run,
                    args=["read despite a lost reply", ["Read"]],
                    id=queue,
                    task_queue=queue,
                )
                == "reply recovered"
            )
            accepted = [
                e.workflow_execution_update_accepted_event_attributes.accepted_request.meta.update_id
                async for e in client.get_workflow_handle(queue).fetch_history_events()
                if e.HasField("workflow_execution_update_accepted_event_attributes")
            ]
            assert accepted.count(attempts[0]) == 1
    finally:
        api.stop()
    assert len(attempts) == 2 and len(set(attempts)) == 1
    assert completed == ["retry_read"]
    assert api.errors == []


@pytest.mark.timeout(90)
async def test_concurrent_agents_have_independent_native_controllers(
    client: Client, tmp_path: Path
) -> None:
    """Controller Activity IDs remain unique when two agents share a Workflow."""
    note = tmp_path / "agents.txt"
    note.write_text("independent agents\n")

    def policy(body: dict[str, Any]) -> list[dict[str, Any]]:
        _, texts, history = history_of(body)
        name = next(
            n for n in ("one", "two") if any(f"native agent {n}" in t for t in texts)
        )
        if not history:
            return [
                {
                    "type": "tool_use",
                    "id": f"read_{name}",
                    "name": "Read",
                    "input": {"file_path": str(note)},
                }
            ]
        assert len(history) == 1 and history[0].id == f"read_{name}", history
        assert not history[0].is_error
        return [{"type": "text", "text": name}]

    api = FakeMessagesAPI(policy).start()
    runner = ClaudeAgentSdkRunner(
        cwd=str(tmp_path), env=engine_env(api, str(tmp_path / "cfg"))
    )
    queue = "native-agents-" + uuid.uuid4().hex
    try:
        async with Worker(
            client,
            task_queue=queue,
            workflows=[ConcurrentNativeWorkflow],
            activities=[record],
            plugins=[ClaudeAgentPlugin(runner, heartbeat_every=0.5)],
        ):
            assert await client.execute_workflow(
                ConcurrentNativeWorkflow.run, id=queue, task_queue=queue
            ) == ["one", "two"]
            controllers = [
                e.activity_task_scheduled_event_attributes.activity_id
                async for e in client.get_workflow_handle(queue).fetch_history_events()
                if e.HasField("activity_task_scheduled_event_attributes")
                and e.activity_task_scheduled_event_attributes.activity_id.startswith(
                    "native-batch-"
                )
            ]
            assert len(controllers) == 2 and len(set(controllers)) == 2
    finally:
        api.stop()
    assert api.errors == []
