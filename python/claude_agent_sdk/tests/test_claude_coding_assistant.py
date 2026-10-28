"""The example coding assistant (``tests/coding_assistant``) end to end against the fake CLI.

The first scenario runs every feature together: three messages, each handed from planner to
coder to reviewer, with workflow tools, an unasked read, an approved edit, a denied command,
streaming, continue-as-new, and a fourth message the session's spending cap refuses. Its first
run's history is replayed by ``test_claude_replay.py``.

The second SIGKILLs the worker process in the middle of the coder's turn and lets another
worker finish the message from the shared SessionStore and LeaseStore.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import uuid
from datetime import timedelta
from pathlib import Path
from typing import Any, cast

import pytest

from temporalio.api.enums.v1 import EventType
from temporalio.claude_agent_sdk import (
    ClaudeAgentPlugin,
    ClaudePartialEvent,
    FileLeaseStore,
    FileSessionStore,
)
from temporalio.claude_agent_sdk.testing import FakeClaudeCLI
from temporalio.client import (
    Client,
    WorkflowContinuedAsNewError,
    WorkflowUpdateFailedError,
)
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.contrib.workflow_streams import WorkflowStreamClient
from temporalio.exceptions import ApplicationError
from temporalio.runtime import MetricBuffer, Runtime, TelemetryConfig
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from .coding_assistant.manager import (
    SESSION_CAP_REACHED,
    TOPIC,
    AssistantInput,
    CodingAssistant,
)
from .coding_assistant.reviewer import Review
from .coding_assistant.tracker import TicketTracker
from .helpers import fixtures as fx
from .helpers import histories
from .helpers import kill_worker as kw
from .test_claude_tracing import _install as install_tracer

pytestmark = fx.POSIX_ONLY

_PLUGIN_ROOT = Path(__file__).resolve().parents[1]
TICKET = {"id": "ENG-42", "title": "add() subtracts", "file": "app.py"}
FIXED = "def add(a, b):\n    return a + b\n"
EDIT = {"file_path": "app.py", "old_string": "a - b", "new_string": "a + b"}
FIX_PLAN = {"files": ["app.py"], "steps": ["Make add() add."]}
NO_FILES = {"files": [], "steps": ["Delete the build directory."]}
REVIEWS = [
    {"approved": True, "files": ["app.py"], "summary": "add() now adds."},
    {"approved": False, "files": [], "summary": "The delete was refused."},
    {"approved": True, "files": [], "summary": "Nothing needed tidying."},
]


def _costing(
    lines: list[dict[str, Any]], total_cost_usd: float
) -> list[dict[str, Any]]:
    """``lines`` ending in a result that reports the session's running total, as Claude Code
    does for a session it resumed from its own transcript."""
    return [*lines[:-1], {**lines[-1], "total_cost_usd": total_cost_usd}]


def _structured(output: dict[str, Any], *tools: tuple[str, str, dict[str, Any]]):
    """A typed reply, after one round calling ``tools`` when there are any."""
    turn = fx.structured_turn(output)
    rounds = fx.tool_round("msg_tools", list(tools)) if tools else []
    return [*turn[: len(fx.START)], *rounds, *turn[len(fx.START) :]]


# Each role's side of the conversation, by prompt. "Continue." is the coder's run that resumes
# a deferred native call after the workflow decided it, approved or denied.
TURNS = [
    [
        "Plan: Fix ENG-42",
        _structured(
            FIX_PLAN, ("toolu_ticket", "lookup_ticket", {"ticket_id": "ENG-42"})
        ),
    ],
    [
        "Implement: Fix ENG-42",
        [
            fx.SYSTEM_INIT,
            *fx.native_round("msg_read", "toolu_read", "Read", {"file_path": "app.py"}),
            *fx.native_round(
                "msg_edit", "toolu_edit", "Edit", EDIT, writes={"app.py": FIXED}
            ),
        ],
    ],
    [
        "Continue.",
        [
            *fx.tool_round(
                "msg_comment",
                [
                    (
                        "toolu_comment_{{RUN}}",
                        "comment_on_ticket",
                        {"ticket_id": "ENG-42", "comment": "Done."},
                    )
                ],
            ),
            *_costing(fx.answer("Done."), 0.02),
        ],
    ],
    ["Review: Fix ENG-42", fx.structured_turn(REVIEWS[0])],
    ["Plan: Delete", fx.structured_turn(NO_FILES)],
    [
        "Implement: Delete",
        [
            fx.SYSTEM_INIT,
            *fx.native_round("msg_rm", "toolu_rm", "Bash", {"command": "rm -rf build"}),
        ],
    ],
    ["Review: Delete", fx.structured_turn(REVIEWS[1])],
    ["Plan: Tidy", fx.structured_turn({"files": [], "steps": []})],
    ["Implement: Tidy", [fx.SYSTEM_INIT, *_costing(fx.answer("Nothing to do."), 0.04)]],
    ["Review: Tidy", fx.structured_turn(REVIEWS[2])],
]


async def test_the_coding_assistant_hands_off_and_runs_every_feature_together(
    env: WorkflowEnvironment,
    tmp_path: Path,
    verified: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert verified is None
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude-config"))
    tracer = install_tracer(monkeypatch)
    tickets = _tickets(tmp_path)
    log, effects = tmp_path / "log.jsonl", tmp_path / "effects.jsonl"
    cli = FakeClaudeCLI(
        tmp_path / "cli",
        turns=TURNS,
        state=str(tmp_path / "state.json"),
        effects=str(effects),
        log=str(log),
    )
    # Core worker metrics share the buffer, and a full buffer drops new updates.
    metrics = MetricBuffer(100_000)
    client = await Client.connect(
        env.client.service_client.config.target_host,
        namespace=env.client.namespace,
        data_converter=pydantic_data_converter,
        runtime=Runtime(telemetry=TelemetryConfig(metrics=metrics)),
    )
    workflow_id = f"coding-assistant-{uuid.uuid4().hex[:8]}"
    tracker = TicketTracker(tickets)
    async with Worker(
        client,
        task_queue=workflow_id,
        workflows=[CodingAssistant],
        activities=[tracker.fetch_ticket, tracker.add_comment],
        plugins=[
            ClaudeAgentPlugin(
                cli_path=cli.cli_path,
                workspace_root=tmp_path / "workspaces",
                enable_native_tools=True,
            )
        ],
    ):
        started = await client.start_workflow(
            CodingAssistant.run,
            AssistantInput(max_budget_usd=0.02, max_cost_usd=0.03),
            id=workflow_id,
            task_queue=workflow_id,
        )
        first = client.get_workflow_handle(workflow_id, run_id=started.result_run_id)
        # Messages go to whichever run currently holds the conversation.
        assistant = client.get_workflow_handle(workflow_id)

        # 1. The planner reads the ticket through the tracker Activity. The coder reads unasked,
        #    its edit is approved because the plan names the file, it comments on the ticket,
        #    and its reply streams as it is written. The reviewer's verdict is the result.
        partials = asyncio.create_task(_streamed_text(client, workflow_id, "Done."))
        reviews = [await assistant.execute_update(CodingAssistant.ask, "Fix ENG-42.")]
        assert await asyncio.wait_for(partials, timeout=20) == "Done."

        # 2. A command off the allowlist is denied, with the reason.
        reviews.append(
            await assistant.execute_update(
                CodingAssistant.ask, "Delete the build directory."
            )
        )
        assert await first.query(CodingAssistant.decisions) == [
            "Edit approved",
            "Bash denied",
        ]

        # Two messages per run: the conversation continues as new with the coder's session.
        with pytest.raises(WorkflowContinuedAsNewError):
            await first.result(follow_runs=False)
        await histories.record("coding_assistant", first)

        # 3. The coder's budget is lowered to what the session's cap leaves.
        reviews.append(await assistant.execute_update(CodingAssistant.ask, "Tidy up."))
        assert await assistant.query(CodingAssistant.cost_usd) == pytest.approx(0.04)

        # 4. The session has spent its cap, so the next message is refused before any role runs.
        with pytest.raises(WorkflowUpdateFailedError) as refused:
            await assistant.execute_update(CodingAssistant.ask, "Anything else?")
        await assistant.terminate()

    assert reviews == [Review.model_validate(r) for r in REVIEWS]
    cause = refused.value.cause
    assert isinstance(cause, ApplicationError)
    assert cause.type == SESSION_CAP_REACHED

    runs = _log(log)
    assert [run["prompt"].partition("\n")[0] for run in runs] == [
        "Plan: Fix ENG-42.",
        "Implement: Fix ENG-42.",
        "Continue.",
        "Review: Fix ENG-42.",
        "Plan: Delete the build directory.",
        "Implement: Delete the build directory.",
        "Continue.",
        "Review: Delete the build directory.",
        "Plan: Tidy up.",
        "Implement: Tidy up.",
        "Review: Tidy up.",
    ]
    # The handoffs: the plan goes to the coder, the coder's reply to the reviewer.
    assert '"app.py"' in runs[1]["prompt"]
    assert "Coder's report:\nDone." in runs[3]["prompt"]
    assert json.loads(runs[0]["mcp"][0]["result"]["content"][0]["text"]) == TICKET
    assert _budget(runs[0]) == 0.02  # each role's own budget
    assert _budget(runs[9]) == pytest.approx(
        0.03 - 0.02
    )  # what the cap leaves the coder
    denial = [hook for run in runs for hook in run["hooks"] if hook["id"] == "toolu_rm"]
    assert denial[-1]["reasons"][0].startswith("Only these commands are allowed")
    ran = [json.loads(line)["id"] for line in effects.read_text().splitlines()]
    assert ran == ["toolu_read", "toolu_edit"]
    (fixed,) = (tmp_path / "workspaces").glob("*/app.py")
    assert fixed.read_text() == FIXED
    assert _comments(tickets) == ["Done.", "Done."]

    spans = [span.name for span in tracer.spans]
    assert spans.count("invoke_agent Claude Code") == len(runs)
    assert {"execute_tool lookup_ticket", "execute_tool comment_on_ticket"} <= set(
        spans
    )
    outcomes = [
        update.attributes["outcome"]
        for update in metrics.retrieve_updates()
        if update.metric.name == "claude_turns"
    ]
    assert outcomes == ["success"] * 9  # three roles, three messages


# The crash model's plans, by prompt: each tool step is one model round. The coder's edit
# defers for approval; the run resuming it edits, comments, then dies mid-answer.
PLANS = [
    [
        "Plan: Fix ENG-42",
        [
            {"tool": "lookup_ticket", "input": {"ticket_id": "ENG-42"}},
            {"text": "Planned.", "structured_output": FIX_PLAN},
        ],
    ],
    [
        "Implement: Fix ENG-42",
        [
            {"native": "Edit", "input": EDIT, "writes": {"app.py": FIXED}},
            {
                "tool": "comment_on_ticket",
                "input": {"ticket_id": "ENG-42", "comment": "add() now adds."},
            },
            {"text": "Fixed add() and commented on ENG-42."},
        ],
    ],
    ["Review: Fix ENG-42", [{"text": "Reviewed.", "structured_output": REVIEWS[0]}]],
]


@pytest.mark.timeout(150)
async def test_the_coding_assistant_finishes_a_message_after_its_worker_is_killed(
    env: WorkflowEnvironment,
    tmp_path: Path,
    verified: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Worker A runs the planner, and the coder's approved edit and ticket comment, then is
    SIGKILLed while the coder writes its answer. The transcript store has the edit's result but
    not the comment's (it trails the CLI by two rounds). Worker B, with an empty config dir,
    resumes the coder from the store: Claude reissues the comment and the workflow answers it
    from its record, does not repeat the edit, and the reviewer runs on B. The client's one
    update returns the one review.
    """
    assert verified is None
    disk_a, disk_b = tmp_path / "disk-a", tmp_path / "disk-b"
    workspace = (tmp_path / "workspace").resolve()
    store, leases, triggers = (
        tmp_path / "store",
        tmp_path / "leases",
        tmp_path / "triggers",
    )
    for path in (disk_a, disk_b, workspace):
        path.mkdir()
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(disk_b))
    tickets = _tickets(tmp_path)
    log, effects = tmp_path / "log.jsonl", tmp_path / "effects.jsonl"
    cli = FakeClaudeCLI(
        tmp_path / "cli",
        plans=PLANS,
        crash_at="mid_answer",
        crash_prompt="Continue.",
        trigger_dir=str(triggers),
        mirror_lag_rounds=2,
        state=str(tmp_path / "state.json"),
        effects=str(effects),
        log=str(log),
    )
    client = kw.client(env)
    task_queue = f"coding-assistant-{uuid.uuid4().hex[:8]}"
    ready = tmp_path / "worker-a-ready"
    spec = {
        "target": env.client.service_client.config.target_host,
        "namespace": env.client.namespace,
        "task_queue": task_queue,
        "cli": cli.cli_path,
        "workspace": str(workspace),
        "store": str(store),
        "leases": str(leases),
        "tickets": str(tickets),
        "ready": str(ready),
    }
    proc = subprocess.Popen(
        [sys.executable, "-m", "tests.helpers.durability_worker", json.dumps(spec)],
        cwd=_PLUGIN_ROOT,
        env={
            **os.environ,
            "CLAUDE_CONFIG_DIR": str(disk_a),
            "CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK": "1",
            "PYTHONPATH": str(_PLUGIN_ROOT),
        },
        start_new_session=True,
    )
    cli_pid = 0
    ask: asyncio.Task[Review] | None = None
    try:
        await kw.until(ready.exists)
        handle = await client.start_workflow(
            CodingAssistant.run,
            AssistantInput(),
            id=task_queue,
            task_queue=task_queue,
        )
        ask = asyncio.create_task(
            handle.execute_update(CodingAssistant.ask, "Fix ENG-42.")
        )
        marker = triggers / "mid_answer"
        await kw.until(marker.exists, timeout=60)
        cli_pid = int(json.loads(marker.read_text())["pid"])
        kw.killpg(proc.pid)
        kw.killpg(cli_pid)
        proc.wait(timeout=5)
        await kw.until(lambda: not kw.alive(cli_pid))
        assert _comments(tickets) == ["add() now adds."], "A posted the comment"
        shutil.rmtree(disk_a)

        tracker = TicketTracker(tickets)
        async with Worker(
            client,
            task_queue=task_queue,
            workflows=[CodingAssistant],
            activities=[tracker.fetch_ticket, tracker.add_comment],
            max_cached_workflows=0,
            plugins=[
                ClaudeAgentPlugin(
                    cli_path=cli.cli_path,
                    workspace_root=workspace,
                    session_store=cast(Any, FileSessionStore(store)),
                    lease_store=FileLeaseStore(leases),
                    disconnect_timeout=timedelta(seconds=1),
                    enable_native_tools=True,
                )
            ],
        ):
            review = await asyncio.wait_for(ask, timeout=100)
            decisions = await handle.query(CodingAssistant.decisions)
        history = await handle.fetch_history()
        await handle.terminate()
    finally:
        if ask is not None and not ask.done():
            ask.cancel()
        kw.killpg(proc.pid)
        if cli_pid:
            kw.killpg(cli_pid)
        if proc.poll() is None:
            proc.wait(timeout=5)

    assert review == Review.model_validate(REVIEWS[0])
    assert decisions == ["Edit approved"], "approved once, on A"
    # The comment ran once, on A: one Activity in the history, one comment on the ticket.
    assert _comments(tickets) == ["add() now adds."]
    scheduled = [
        e.activity_task_scheduled_event_attributes.activity_type.name
        for e in history.events
        if e.event_type == EventType.EVENT_TYPE_ACTIVITY_TASK_SCHEDULED
    ]
    assert scheduled.count("add_comment") == 1
    asks = {
        e.event_id
        for e in history.events
        if e.event_type == EventType.EVENT_TYPE_WORKFLOW_EXECUTION_UPDATE_ACCEPTED
        and e.workflow_execution_update_accepted_event_attributes.accepted_request.input.name
        == "ask"
    }
    completed = [
        e
        for e in history.events
        if e.event_type == EventType.EVENT_TYPE_WORKFLOW_EXECUTION_UPDATE_COMPLETED
        and e.workflow_execution_update_completed_event_attributes.accepted_event_id
        in asks
    ]
    assert len(asks) == len(completed) == 1, "one authoritative result"
    # The edit ran once, by A's CLI, and B did not run it again.
    (edit,) = [json.loads(line) for line in effects.read_text().splitlines()]
    assert (edit["name"], edit["pid"]) == ("Edit", cli_pid)
    (fixed,) = workspace.glob("*/app.py")
    assert fixed.read_text() == FIXED

    # A's killed run never logged. B resumed the coder from the store (A's config dir is gone),
    # and Claude reissued the comment, which the workflow answered from its record.
    runs = _log(log)
    assert [run["prompt"].partition("\n")[0] for run in runs[:2]] == [
        "Plan: Fix ENG-42.",
        "Implement: Fix ENG-42.",
    ]
    resumed, reviewed = runs[2:]
    assert not disk_a.exists()
    assert resumed["prompt"].startswith("Continue")
    assert any(arg.startswith("--resume=") for arg in resumed["argv"])
    assert [step.get("tool") or "text" for step in resumed["steps"]] == [
        "comment_on_ticket",
        "text",
    ]
    assert resumed["mcp"][0]["result"]["content"][0]["text"] == "posted"
    assert reviewed["prompt"].startswith("Review: Fix ENG-42.")


def _tickets(tmp_path: Path) -> Path:
    tickets = tmp_path / "tickets"
    tickets.mkdir()
    (tickets / "ENG-42.json").write_text(json.dumps(TICKET))
    return tickets


def _comments(tickets: Path) -> list[str]:
    path = tickets / "ENG-42.comments.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def _log(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


async def _streamed_text(client: Client, workflow_id: str, until: str) -> str:
    """The text partials on the assistant's topic, up to and including ``until``."""
    seen = ""
    async for item in WorkflowStreamClient.create(client, workflow_id).subscribe(
        [TOPIC],
        from_offset=0,
        result_type=ClaudePartialEvent,
        poll_cooldown=timedelta(milliseconds=20),
    ):
        seen += item.data.text or ""
        if until in seen:
            return until
    return seen


def _budget(run: dict[str, Any]) -> float:
    argv: list[str] = run["argv"]
    return float(argv[argv.index("--max-budget-usd") + 1])
