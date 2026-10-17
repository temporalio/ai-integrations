"""Workspace failover: an agent's folder follows it to any Worker.

With a workspace store, each agent works in its own folder, ``<cwd>/<id>``. After
each segment and tool step the runner saves what changed; the Workflow takes the
snapshot from the step's result; each segment and tool step first makes its
Worker's folder match the agent's current snapshot.

The Workflow tests use a scripted runner that reports made-up snapshots. The real
engine tests drive the runner's segments and tool steps as the Workflow does
(``Agent``), on runners that stand for Workers on other machines: each has its own
Claude Code folders, and a "new machine" gets the same ``cwd`` path after the old
folder is removed.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import stat
import sys
import uuid
from pathlib import Path
from typing import Any

import pytest

from temporalio.api.common.v1 import WorkflowExecution
from temporalio.api.workflowservice.v1 import ResetWorkflowExecutionRequest
from temporalio.claude_agent_sdk import (
    TOOL_CALL_INTERRUPTED,
    TOOL_CALL_NOT_RUN,
    WORKSPACE_NOT_FOUND,
    WORKSPACE_TOO_LARGE,
    ClaudeAgentPlugin,
    ClaudeAgentSdkRunner,
    DeferredCall,
    FileWorkspaceStore,
    SegmentInput,
    SegmentOutput,
    ToolOutcome,
    ToolSpec,
    ToolStepInput,
    Workspace,
)
from temporalio.claude_agent_sdk._workspace import Workspaces
from temporalio.claude_agent_sdk.testing import (
    Final,
    HistoryItem,
    ScriptedClaude,
    ToolCall,
)
from temporalio.client import Client
from temporalio.exceptions import ApplicationError
from temporalio.worker import Worker
from tests.helpers.fake_messages_api import FakeMessagesAPI, engine_env, history_of
from tests.helpers.workers import FAIL_FAST
from tests.test_workspace import describe
from tests.workspace.workflows import Done, FolderWorkflow, Tasks

pytestmark = pytest.mark.timeout(240)
POSIX = sys.platform != "win32"
TOOLS = [ToolSpec("count", "Count one step.", {"type": "object"})]
STALE = "modified since read"
FAKE_AUTH = {"ANTHROPIC_API_KEY": "sk-ant-fake-not-real"}  # no engine starts
INTERRUPTED = (
    "This call was interrupted (its step timed out), so it may have run, in full or "
    "in part. Check its effects before you run it again."
)


# ---- the Workflow, with a scripted runner ----


def two_commands(prompt: str, history: list[HistoryItem]) -> ToolCall | Final:
    del prompt
    if len(history) < 2:
        return ToolCall("Bash", {"command": f"echo {len(history) + 1}"})
    return Final("done")


class Snapshots(ScriptedClaude):
    """Reports a new made-up snapshot after each segment and tool step, and records
    the folder each one was given."""

    def __init__(
        self,
        *,
        fail_step: int = 0,
        none_at: int = 0,
        too_large_step: int = 0,
        missing_step: int = 0,
    ) -> None:
        super().__init__(two_commands, engine_tools={"Bash": lambda args: "ran"})
        self.seen: list[tuple[str, str, Workspace | None]] = []
        self.injected: list[dict[str, Any]] = []
        """The results each segment delivered to Claude."""
        self.prompts: list[str | None] = []
        """The prompt each segment started with (``None`` when it resumed)."""
        self.made = 0
        self.fail_step = fail_step  # the n-th step fails after its call may have run
        self.none_at = none_at  # the n-th snapshot is missing (a Worker with no store)
        self.too_large_step = too_large_step  # the n-th step's folder cannot be kept
        self.missing_step = missing_step  # the n-th step's folder cannot be restored
        self.fail_first_segment = False

    def snapshot(self) -> str | None:
        self.made += 1
        if self.made == self.none_at:
            return None
        return f"{self.made:064x}"

    async def run(self, inp: SegmentInput, attempt: int) -> SegmentOutput:
        self.seen.append(("segment", inp.session_id, inp.workspace))
        self.injected.append(dict(inp.injected))
        self.prompts.append(inp.prompt)
        if self.fail_first_segment:
            self.fail_first_segment = False
            return SegmentOutput(session_id=inp.session_id, is_error=True, error="boom")
        out = await super().run(inp, attempt)
        out.workspace = self.snapshot()
        return out

    async def run_tool_step(self, step: ToolStepInput, attempt: int) -> ToolOutcome:
        self.seen.append(("step", step.session_id, step.workspace))
        steps = sum(1 for kind, _, _ in self.seen if kind == "step")
        if steps == self.fail_step:
            raise ApplicationError("the Worker died", type=TOOL_CALL_INTERRUPTED)
        if steps == self.too_large_step:
            raise ApplicationError(
                "the folder was not saved", type=TOOL_CALL_INTERRUPTED
            ) from ApplicationError(
                "more than workspace_max_bytes", type=WORKSPACE_TOO_LARGE
            )
        if steps == self.missing_step:
            raise ApplicationError(
                "the folder was not restored",
                type=TOOL_CALL_NOT_RUN,
                non_retryable=True,
            ) from ApplicationError(
                "an object is not in the workspace store",
                type=WORKSPACE_NOT_FOUND,
                non_retryable=True,
            )
        outcome = await super().run_tool_step(step, attempt)
        outcome.workspace = self.snapshot()
        return outcome


async def run_tasks(client: Client, runner: Snapshots, tasks: Tasks) -> Done:
    queue = f"folder-{uuid.uuid4().hex[:8]}"
    async with Worker(
        client,
        task_queue=queue,
        workflows=[FolderWorkflow],
        plugins=[ClaudeAgentPlugin(runner, heartbeat_every=1.0)],
        **FAIL_FAST,
    ):
        handle = await client.start_workflow(
            FolderWorkflow.run, args=[tasks, None], id=queue, task_queue=queue
        )
        return await asyncio.wait_for(handle.result(), 60)


def snapshots_given(runner: Snapshots) -> list[tuple[str, str | None]]:
    return [(kind, None if ws is None else ws.snapshot) for kind, _, ws in runner.seen]


async def test_each_step_gets_the_snapshot_of_the_last_completed_one(
    client: Client,
) -> None:
    runner = Snapshots()
    state = (await run_tasks(client, runner, Tasks(["go"]))).state
    made = [f"{n:064x}" for n in range(1, 6)]
    assert snapshots_given(runner) == [
        ("segment", None),
        ("step", made[0]),
        ("segment", made[1]),
        ("step", made[2]),
        ("segment", made[3]),
    ]
    assert state.workspace is not None
    assert state.workspace.snapshot == made[4]
    # The folder is named once: the agent's first session id.
    assert state.workspace.id == state.session_id
    assert {ws.id for _, _, ws in runner.seen if ws is not None} == {state.session_id}


async def test_a_step_that_failed_keeps_the_snapshot_before_it(client: Client) -> None:
    """A step whose call may have run saved nothing the Workflow takes: the next
    segment gets the snapshot from before the step, so its changes are rolled back."""
    runner = Snapshots(fail_step=1)
    await run_tasks(client, runner, Tasks(["go"]))
    given = snapshots_given(runner)
    assert given[:3] == [
        ("segment", None),
        ("step", f"{1:064x}"),
        ("segment", f"{1:064x}"),  # not the failed step's
    ]


async def test_a_worker_without_a_store_after_one_with_a_store_stops_the_task(
    client: Client,
) -> None:
    """A step whose Worker has no store stops the task, once its call's result is
    pending: the call ran, and Claude gets its real result with the next task."""
    runner = Snapshots(none_at=2)  # only the first step's Worker has no store
    done = await run_tasks(client, runner, Tasks(["go", "next"]))
    assert len(done.failed) == 1
    assert "workspace_store" in done.failed[0]
    delivered = [r for r in runner.injected if r]
    assert len(delivered[0]) == 1
    (outcome,) = delivered[0].values()
    content = (
        outcome.content if isinstance(outcome, ToolOutcome) else outcome["content"]
    )
    assert content == "ran"  # not "did not run"


async def test_the_folder_moves_to_the_next_run_and_survives_a_failed_task(
    client: Client,
) -> None:
    """Continue-As-New carries the folder and its snapshot. A first task that fails
    before anything committed starts the next one in a new session, in the same
    folder."""
    runner = Snapshots()
    runner.fail_first_segment = True
    done = await run_tasks(
        client, runner, Tasks(["first", "second", "third"], hand_over=True)
    )
    state = done.state
    assert len(done.failed) == 1 and "boom" in done.failed[0]
    first_session = runner.seen[0][1]
    assert state.workspace is not None and state.workspace.id == first_session
    assert state.session_id != first_session  # a new session after the failed task
    assert {ws.id for _, _, ws in runner.seen if ws is not None} == {first_session}
    assert state.runs == 3
    # The third task's first segment (a new run) gets the second task's last snapshot.
    segments = [i for i, (kind, _, _) in enumerate(runner.seen) if kind == "segment"]
    third = runner.seen[segments[4]]
    assert third[2] is not None and third[2].snapshot == f"{5:064x}"


@pytest.mark.parametrize("side", ["save", "restore"])
async def test_a_tool_step_that_cannot_keep_the_folder_stops_the_task(
    client: Client, side: str
) -> None:
    """An install that took the folder past ``workspace_max_bytes`` (the step fails as
    one whose call may have run, and the next step rolls its changes back), or a
    snapshot whose object is gone (the call did not run): the task stops with the
    reason, instead of Claude running the call again and again."""
    if side == "save":
        runner = Snapshots(too_large_step=1)
        reason, told = "workspace_max_bytes", "may have run"
    else:
        runner = Snapshots(missing_step=1)
        reason, told = "workspace store", "did not run"
    done = await run_tasks(client, runner, Tasks(["go", "next"]))
    assert len(done.failed) == 1
    assert reason in done.failed[0]
    # The first task stopped right after its step: the next segment is the second
    # task's, and it delivers the step's result.
    assert [kind for kind, _, _ in runner.seen][:3] == ["segment", "step", "segment"]
    assert runner.prompts[:2] == ["go", "next"]
    delivered = [r for r in runner.injected if r]
    (outcome,) = delivered[0].values()
    content = (
        outcome.content if isinstance(outcome, ToolOutcome) else outcome["content"]
    )
    assert told in str(content)


async def test_a_reset_takes_the_folder_back_with_the_workflow(client: Client) -> None:
    """Reset to the Workflow task after the first tool step: the new run's next
    segment gets that step's snapshot, so its folder is the one of that point."""
    runner = Snapshots()
    queue = f"folder-reset-{uuid.uuid4().hex[:8]}"
    async with Worker(
        client,
        task_queue=queue,
        workflows=[FolderWorkflow],
        plugins=[ClaudeAgentPlugin(runner, heartbeat_every=1.0)],
        **FAIL_FAST,
    ):
        handle = await client.start_workflow(
            FolderWorkflow.run, args=[Tasks(["go"]), None], id=queue, task_queue=queue
        )
        await asyncio.wait_for(handle.result(), 60)
        kinds: dict[int, str] = {}
        stepped = reset_to = None
        async for event in handle.fetch_history_events():
            if event.HasField("activity_task_scheduled_event_attributes"):
                attrs = event.activity_task_scheduled_event_attributes
                kinds[event.event_id] = attrs.activity_type.name
            done = event.activity_task_completed_event_attributes
            if (
                event.HasField("activity_task_completed_event_attributes")
                and stepped is None
                and kinds.get(done.scheduled_event_id) == "run_claude_tool_step"
            ):
                stepped = event.event_id
            if (
                event.HasField("workflow_task_completed_event_attributes")
                and stepped is not None
                and reset_to is None
            ):
                reset_to = event.event_id
        assert reset_to is not None
        before = len(runner.seen)
        reset = await client.workflow_service.reset_workflow_execution(
            ResetWorkflowExecutionRequest(
                namespace=client.namespace,
                workflow_execution=WorkflowExecution(
                    workflow_id=queue, run_id=handle.result_run_id or ""
                ),
                reason="test",
                workflow_task_finish_event_id=reset_to,
                request_id=str(uuid.uuid4()),
            )
        )
        again = client.get_workflow_handle(queue, run_id=reset.run_id)
        await asyncio.wait_for(again.result(), 60)
    kind, _, workspace = runner.seen[before]
    assert kind == "segment" and workspace is not None
    assert workspace.snapshot == f"{2:064x}"  # what the first step saved


# ---- the real engine ----


class Agent:
    """Drives one agent's segments and tool steps as the Workflow does: any runner can
    run any step, and each completed step's snapshot becomes the agent's."""

    def __init__(self, workspace_id: str, builtin: list[str], activities: list[str]):
        self.session_id = str(uuid.uuid4())
        self.checkpoint: str | None = None
        self.conversation: list[dict[str, Any]] = []
        self.workspace = Workspace(workspace_id)
        self.builtin = builtin
        self.activities = activities

    def _ws(self) -> Workspace:
        return Workspace(self.workspace.id, self.workspace.snapshot)

    async def segment(
        self,
        runner: ClaudeAgentSdkRunner,
        prompt: str | None = None,
        injected: dict[str, ToolOutcome] | None = None,
    ) -> SegmentOutput:
        out = await runner.run(
            SegmentInput(
                session_id=self.session_id,
                prompt=prompt,
                tools=TOOLS,
                builtin_tools=self.builtin,
                tool_activities=self.activities,
                checkpoint=self.checkpoint,
                injected=injected or {},
                transcript=self.conversation,
                workspace=self._ws(),
            ),
            1,
        )
        assert not out.is_error, out.error
        assert out.workspace is not None
        self.session_id = out.session_id
        self.checkpoint = out.checkpoint
        keep = out.transcript_keep
        assert keep is not None
        self.conversation = [*self.conversation[:keep], *out.transcript_add]
        self.workspace.snapshot = out.workspace
        return out

    async def step(
        self, runner: ClaudeAgentSdkRunner, out: SegmentOutput, *, lost: bool = False
    ) -> ToolOutcome:
        """Run the paused call's step. ``lost``: its Worker died before the result
        reached the Workflow: Claude is told the call may have run."""
        assert out.deferred is not None and out.deferred.kind == "engine", out
        outcome = await runner.run_tool_step(
            ToolStepInput(
                session_id=out.session_id,
                checkpoint=out.checkpoint or "",
                call=out.deferred,
                tools=TOOLS,
                builtin_tools=self.builtin,
                transcript=self.conversation,
                workspace=self._ws(),
            ),
            1,
        )
        assert outcome.workspace is not None
        if lost:
            return ToolOutcome(content=INTERRUPTED, is_error=True)
        self.workspace.snapshot = outcome.workspace
        outcome.workspace = None
        return outcome

    async def round(
        self, runner: ClaudeAgentSdkRunner, out: SegmentOutput, *, lost: bool = False
    ) -> SegmentOutput:
        """The paused call's step, then the next segment, on ``runner``."""
        assert out.deferred is not None
        outcome = await self.step(runner, out, lost=lost)
        return await self.segment(runner, injected={out.deferred.id: outcome})


class Script:
    """A scripted model: each agent (told apart by its prompt) makes the calls of
    ``turns[agent][n]`` once it has made ``n`` messages of calls, then answers
    ``FINAL`` with the results it saw."""

    def __init__(self, turns: dict[str, list[list[tuple[str, dict[str, Any]]]]]):
        self.turns = turns
        self.api = FakeMessagesAPI(self.decide).start()

    def decide(self, body: dict[str, Any]) -> list[dict[str, Any]]:
        _, texts, history = history_of(body)
        agent = next(t for t in texts if t.strip() in self.turns).strip()
        made = sum(
            1
            for m in body.get("messages", [])
            if m.get("role") == "assistant"
            and isinstance(m.get("content"), list)
            and any(b.get("type") == "tool_use" for b in m["content"])
        )
        if made < len(self.turns[agent]):
            return [self.api.call(n, a) for n, a in self.turns[agent][made]]
        seen = " | ".join(
            f"{h.name}:{'error ' if h.is_error else ''}{str(h.content)[:80]}"
            for h in history
        )
        return [{"type": "text", "text": f"FINAL {seen}"}]


def worker(
    tmp_path: Path, name: str, api: FakeMessagesAPI, **options: Any
) -> ClaudeAgentSdkRunner:
    """A runner for a Worker on its own machine: its own Claude Code folders, the
    shared ``cwd`` path and the shared store."""
    root = tmp_path / "agents"
    root.mkdir(exist_ok=True)
    return ClaudeAgentSdkRunner(
        cwd=str(root),
        env=engine_env(api, str(tmp_path / f"cfg-{name}")),
        workspace_store=FileWorkspaceStore(tmp_path / "store"),
        **options,
    )


def new_machine(tmp_path: Path, workspace_id: str) -> None:
    """The next Worker's disk does not have the agent's folder."""
    folder = tmp_path / "agents" / workspace_id
    for path in folder.rglob("*"):
        if not path.is_symlink():
            os.chmod(path, 0o700)
    shutil.rmtree(folder)


async def test_real_engine_the_folder_follows_the_agent_to_a_new_machine(
    tmp_path: Path,
) -> None:
    """Claude writes, runs and reads on Worker A; A's machine is gone; on Worker B,
    with no copy of the folder, Claude edits the file it read on A (no "modified
    since read": the time came back to the nanosecond), and runs the script it
    made executable on A."""
    folder = tmp_path / "agents" / "agent-1"
    notes = str(folder / "notes.txt")
    script = Script(
        {
            "Work.": [
                [("Write", {"file_path": notes, "content": "start\nEND\n"})],
                [
                    (
                        "Bash",
                        {
                            "command": "printf '#!/bin/sh\\necho ran\\n' > run.sh && "
                            "chmod 755 run.sh && mkdir -p sub/empty && "
                            "printf data > sub/d.bin",
                            "description": "make",
                        },
                    )
                ],
                [("Read", {"file_path": notes})],
                [
                    (
                        "Edit",
                        {
                            "file_path": notes,
                            "old_string": "END",
                            "new_string": "tok1\nEND",
                        },
                    )
                ],
                [("Bash", {"command": "sh ./run.sh > out.txt", "description": "run"})],
            ]
        }
    )
    a = worker(tmp_path, "a", script.api)
    b = worker(tmp_path, "b", script.api)
    agent = Agent(
        "agent-1", ["Read", "Write", "Edit", "Bash"], ["Write", "Edit", "Bash"]
    )
    try:
        out = await agent.segment(a, "Work.")
        out = await agent.round(a, out)  # Write
        out = await agent.round(a, out)  # Bash, then Read in the segment, then Edit
        assert out.deferred is not None and out.deferred.name == "Edit"
        made_on_a = describe(folder)
        new_machine(tmp_path, "agent-1")
        out = await agent.round(b, out)  # Edit, on B
        out = await agent.round(b, out)  # Bash
    finally:
        script.api.stop()
    assert out.result is not None and out.result.startswith("FINAL"), out.result
    assert "error" not in out.result, out.result
    assert STALE not in json.dumps(script.api.requests)
    assert script.api.errors == []
    assert (folder / "notes.txt").read_text(encoding="utf-8") == "start\ntok1\nEND\n"
    assert (folder / "out.txt").read_text(encoding="utf-8").strip() == "ran"
    assert (folder / "sub" / "empty").is_dir()
    assert (folder / "sub" / "d.bin").read_bytes() == b"data"
    for name in ("run.sh", "sub/d.bin", "sub/empty"):
        assert describe(folder)[name] == made_on_a[name]  # mode and time too
    if POSIX:
        assert stat.S_IMODE((folder / "run.sh").stat().st_mode) == 0o755
    # The last snapshot, restored on a third machine, is this folder.
    third = Workspaces(
        str(tmp_path / "third"),
        FileWorkspaceStore(tmp_path / "store"),
        max_bytes=None,
        exclude=(),
    )
    async with third.use(agent._ws()) as use:  # type: ignore[reportPrivateUsage]
        assert use.changed
    assert describe(tmp_path / "third" / "agent-1") == describe(folder)


@pytest.mark.parametrize("where", ["same", "new"])
async def test_real_engine_what_a_lost_step_changed_is_rolled_back(
    tmp_path: Path, where: str
) -> None:
    """A step whose result never reached the Workflow (its Worker died after the
    command ran): Claude is told it may have run, and the folder is the one the
    Workflow committed, on the same machine and on a new one. Claude checks, finds
    no trace, and runs it again: the effect is in the folder once."""
    folder = tmp_path / "agents" / "agent-2"
    script = Script(
        {
            "Log.": [
                [
                    (
                        "Bash",
                        {"command": "echo tok1 >> effects.log", "description": "log"},
                    )
                ],
                [
                    (
                        "Bash",
                        {
                            "command": "cat effects.log 2>/dev/null || echo none; "
                            "echo tok1 >> effects.log",
                            "description": "check, then log",
                        },
                    )
                ],
            ]
        }
    )
    a = worker(tmp_path, "a", script.api)
    agent = Agent("agent-2", ["Bash"], ["Bash"])
    try:
        out = await agent.segment(a, "Log.")
        outcome = await agent.step(a, out, lost=True)
        assert (folder / "effects.log").read_text(
            encoding="utf-8"
        ) == "tok1\n"  # it ran
        if where == "new":
            new_machine(tmp_path, "agent-2")
            runner = worker(tmp_path, "b", script.api)
        else:
            runner = a
        assert out.deferred is not None
        out = await agent.segment(runner, injected={out.deferred.id: outcome})
        assert not (folder / "effects.log").exists()  # rolled back
        out = await agent.round(runner, out)
    finally:
        script.api.stop()
    assert out.result is not None
    assert "may have run" in out.result and "none" in out.result, out.result
    assert (folder / "effects.log").read_text(encoding="utf-8") == "tok1\n"


async def test_real_engine_two_agents_on_one_worker_keep_their_own_folders(
    tmp_path: Path,
) -> None:
    """Two agents on one Worker work in folders of their own; restoring one never
    touches the other, even its changes no step committed yet."""
    script = Script(
        {
            "X.": [[("Bash", {"command": "echo x > x.txt", "description": "x"})]],
            "Y.": [[("Bash", {"command": "echo y > y.txt", "description": "y"})]],
        }
    )
    a = worker(tmp_path, "a", script.api)
    x = Agent("agent-x", ["Bash"], ["Bash"])
    y = Agent("agent-y", ["Bash"], ["Bash"])
    agents = tmp_path / "agents"
    try:
        out_x = await x.segment(a, "X.")
        out_y = await y.segment(a, "Y.")
        out_x = await x.round(a, out_x)
        (agents / "agent-y" / "uncommitted.txt").write_text("u", encoding="utf-8")
        await y.step(a, out_y)  # Y's step: Y's folder comes back first
        assert not (agents / "agent-y" / "uncommitted.txt").exists()
        (agents / "agent-y" / "uncommitted.txt").write_text("u", encoding="utf-8")
        await x.segment(a, "X.")  # a new task of X: Y's folder stays as it is
    finally:
        script.api.stop()
    assert out_x.result is not None and out_x.result.startswith("FINAL")
    assert sorted(p.name for p in (agents / "agent-x").iterdir()) == ["x.txt"]
    assert sorted(p.name for p in (agents / "agent-y").iterdir()) == [
        "uncommitted.txt",
        "y.txt",
    ]


class Counting(ClaudeAgentSdkRunner):
    """Counts the steps whose call ran in its warm engine."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.warm_steps = 0

    async def _step_in_warm(self, step: Any, warm: Any, allowed: list[bool]) -> Any:
        self.warm_steps += 1
        return await super()._step_in_warm(step, warm, allowed)  # type: ignore[reportPrivateUsage]


async def test_real_engine_warm_engines_stay_warm_while_the_folder_is_as_they_left_it(
    tmp_path: Path,
) -> None:
    """With warm engines, a call waits in its segment's engine and runs there when
    its step comes, as long as the folder needed no change. A change between the
    segment and its step (a process that wrote there) is rolled back, and the call
    runs in a new engine instead."""
    rounds = [
        [("Bash", {"command": f"echo {n} >> log.txt", "description": "log"})]
        for n in range(1, 5)
    ]
    script = Script({"Count.": rounds})
    (tmp_path / "agents").mkdir()
    runner = Counting(
        cwd=str(tmp_path / "agents"),
        env=engine_env(script.api, str(tmp_path / "cfg")),
        workspace_store=FileWorkspaceStore(tmp_path / "store"),
        warm_engines=2,
        warm_seconds=60.0,
    )
    agent = Agent("agent-w", ["Bash"], ["Bash"])
    folder = tmp_path / "agents" / "agent-w"
    try:
        out = await agent.segment(runner, "Count.")
        out = await agent.round(runner, out)
        out = await agent.round(runner, out)
        warm_before = runner.warm_steps
        (folder / "stray.txt").write_text("written between steps", encoding="utf-8")
        out = await agent.round(runner, out)  # its call does not run warm
        assert runner.warm_steps == warm_before
        assert not (folder / "stray.txt").exists()
        out = await agent.round(runner, out)
    finally:
        runner._end_all_warm()  # type: ignore[reportPrivateUsage]
        script.api.stop()
    assert out.result is not None and out.result.startswith("FINAL"), out.result
    assert warm_before >= 1  # calls ran warm while the folder was as they left it
    assert (folder / "log.txt").read_text(encoding="utf-8").split() == [
        "1",
        "2",
        "3",
        "4",
    ]


class OutOfOrderStore(FileWorkspaceStore):
    """A folder store whose saves fail (a store that is down)."""

    async def save(self, key: str, path: str) -> None:
        raise ConnectionError("the store is down")


async def test_real_engine_a_store_that_fails_a_tool_step_says_whether_its_call_ran(
    tmp_path: Path,
) -> None:
    """A step that cannot restore the folder fails before its call runs, not
    retryable when the snapshot is missing; a step that cannot save it after its call
    ran fails as one whose call may have run (and the next step rolls back what it
    changed)."""
    script = Script(
        {
            "Log.": [
                [("Bash", {"command": "echo ran >> effects.log", "description": "log"})]
            ]
        }
    )
    folder = tmp_path / "agents" / "agent-s"
    a = worker(tmp_path, "a", script.api)
    agent = Agent("agent-s", ["Bash"], ["Bash"])
    try:
        out = await agent.segment(a, "Log.")
        assert out.deferred is not None
        # Missing objects, on a new machine: the call never runs.
        shutil.move(tmp_path / "store", tmp_path / "store-kept")
        new_machine(tmp_path, "agent-s")
        b = worker(tmp_path, "b", script.api)
        with pytest.raises(ApplicationError) as caught:
            await agent.step(b, out)
        assert caught.value.type == TOOL_CALL_NOT_RUN and caught.value.non_retryable
        cause = caught.value.__cause__
        assert isinstance(cause, ApplicationError) and cause.type == WORKSPACE_NOT_FOUND
        assert not (folder / "effects.log").exists()
        # The store is back but cannot save: the call ran, the step says it may have.
        shutil.move(tmp_path / "store-kept", tmp_path / "store")
        c = ClaudeAgentSdkRunner(
            cwd=str(tmp_path / "agents"),
            env=engine_env(script.api, str(tmp_path / "cfg-c")),
            workspace_store=OutOfOrderStore(tmp_path / "store"),
        )
        with pytest.raises(ApplicationError) as caught:
            await agent.step(c, out)
        assert caught.value.type == TOOL_CALL_INTERRUPTED
        assert (folder / "effects.log").read_text(encoding="utf-8") == "ran\n"
        # The folder cannot be kept (too large): the same, with the reason behind it.
        d = worker(tmp_path, "d", script.api, workspace_max_bytes=1)
        with pytest.raises(ApplicationError) as caught:
            await agent.step(d, out)
        assert caught.value.type == TOOL_CALL_INTERRUPTED
        cause = caught.value.__cause__
        assert isinstance(cause, ApplicationError) and cause.type == WORKSPACE_TOO_LARGE
    finally:
        script.api.stop()


def bash_step(workspace: Workspace) -> ToolStepInput:
    return ToolStepInput(
        session_id=str(uuid.uuid4()),
        checkpoint="cp_1",
        call=DeferredCall("toolu_1", "Bash", {"command": "echo ran"}, "engine"),
        tools=TOOLS,
        builtin_tools=["Bash"],
        transcript=[],
        workspace=workspace,
    )


async def test_a_call_that_ran_without_its_hook_is_never_run_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A call can run although its hook gave no answer (an allow rule in a settings
    file can let it): once the step has the call's result, a save that fails says
    the call may have run, so the Workflow never runs it again."""
    seed = Workspaces(
        str(tmp_path / "seed"),
        FileWorkspaceStore(tmp_path / "store"),
        max_bytes=None,
        exclude=(),
    )
    async with seed.use(Workspace("agent-h")) as use:
        snapshot = await use.save()
    root = tmp_path / "agents"
    root.mkdir()
    runner = ClaudeAgentSdkRunner(
        cwd=str(root),
        env=FAKE_AUTH,
        workspace_store=OutOfOrderStore(tmp_path / "store"),
    )

    async def ran_without_its_hook(
        step: ToolStepInput, allowed: list[bool]
    ) -> ToolOutcome:
        del step, allowed  # no hook said "allow"
        (root / "agent-h" / "effects.log").write_text("ran\n", encoding="utf-8")
        return ToolOutcome(content="ran")

    monkeypatch.setattr(runner, "_tool_step", ran_without_its_hook)
    with pytest.raises(ApplicationError) as caught:
        await runner.run_tool_step(bash_step(Workspace("agent-h", snapshot)), 1)
    assert caught.value.type == TOOL_CALL_INTERRUPTED


async def test_a_worker_without_a_store_runs_no_step_of_an_agent_with_one(
    tmp_path: Path,
) -> None:
    """The step would run in ``cwd``, where the agent's files are not: it fails
    before it starts, and is tried again (on a Worker with the store, once there is
    one), as when a session store is missing."""
    root = tmp_path / "agents"
    root.mkdir()
    runner = ClaudeAgentSdkRunner(cwd=str(root), env=FAKE_AUTH)
    workspace = Workspace("agent-n", "0" * 64)
    with pytest.raises(RuntimeError, match="workspace_store"):
        await runner.run(
            SegmentInput(
                session_id=str(uuid.uuid4()),
                prompt=None,
                tools=TOOLS,
                builtin_tools=["Bash"],
                tool_activities=["Bash"],
                checkpoint="cp_1",
                transcript=[],
                workspace=workspace,
            ),
            1,
        )
    with pytest.raises(ApplicationError) as caught:
        await runner.run_tool_step(bash_step(workspace), 1)
    assert caught.value.type == TOOL_CALL_NOT_RUN and not caught.value.non_retryable
    assert "workspace_store" in str(caught.value)


async def test_real_engine_an_agent_that_started_without_a_store_keeps_its_folder(
    tmp_path: Path,
) -> None:
    """An agent whose first segment ran on a Worker with no store (its Workflow has a
    checkpoint and no snapshot) keeps working in the runner's cwd on a Worker with a
    store: its files stay where they are, and no snapshot is made."""
    script = Script(
        {"Old.": [[("Bash", {"command": "echo old > old.txt", "description": "old"})]]}
    )
    root = tmp_path / "agents"
    root.mkdir()
    plain = ClaudeAgentSdkRunner(
        cwd=str(root), env=engine_env(script.api, str(tmp_path / "cfg"))
    )
    stored = worker(tmp_path, "s", script.api)
    session = str(uuid.uuid4())
    try:
        first = await plain.run(
            SegmentInput(
                session_id=session,
                prompt="Old.",
                tools=TOOLS,
                builtin_tools=["Bash"],
                tool_activities=["Bash"],
                transcript=[],
                workspace=Workspace(session),
            ),
            1,
        )
        assert first.workspace is None and first.deferred is not None
        outcome = await stored.run_tool_step(
            ToolStepInput(
                session_id=first.session_id,
                checkpoint=first.checkpoint or "",
                call=first.deferred,
                tools=TOOLS,
                builtin_tools=["Bash"],
                transcript=list(first.transcript_add),
                workspace=Workspace(session),  # no snapshot: started without a store
            ),
            1,
        )
    finally:
        script.api.stop()
    assert outcome.workspace is None and not outcome.is_error
    assert (root / "old.txt").read_text(encoding="utf-8").strip() == "old"
    assert not (root / session).exists()


class Broken(ClaudeAgentSdkRunner):
    """A segment that writes a file, then reports an error that retrying cannot fix."""

    async def _run(self, inp: SegmentInput, attempt: int) -> SegmentOutput:
        folder = self._folder()  # type: ignore[reportPrivateUsage]
        assert folder is not None
        Path(folder, "half.txt").write_text("half", encoding="utf-8")
        return SegmentOutput(session_id=inp.session_id, is_error=True, error="broken")


async def test_a_segment_that_reports_an_error_saves_nothing(tmp_path: Path) -> None:
    api = FakeMessagesAPI(lambda body: []).start()
    (tmp_path / "agents").mkdir()
    runner = Broken(
        cwd=str(tmp_path / "agents"),
        env=engine_env(api, str(tmp_path / "cfg")),
        workspace_store=FileWorkspaceStore(tmp_path / "store"),
    )
    try:
        out = await runner.run(
            SegmentInput(
                session_id=str(uuid.uuid4()),
                prompt="go",
                tools=TOOLS,
                transcript=[],
                workspace=Workspace("agent-e"),
            ),
            1,
        )
    finally:
        api.stop()
    assert out.is_error and out.workspace is None
    assert not (tmp_path / "store").exists()  # nothing saved
    assert (
        tmp_path / "agents" / "agent-e" / "half.txt"
    ).exists()  # until the next step
