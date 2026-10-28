"""The bundled Claude Code on this host, with a scripted Anthropic API."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import sys
import threading
import uuid
from collections.abc import AsyncIterator, Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import claude_agent_sdk as sdk
import pytest

from temporalio.claude_agent_sdk import ClaudeAgentSDKPlugin
from temporalio.client import Client, WorkflowHandle
from temporalio.worker import Interceptor, Replayer
from tests.helpers.example import TASK_QUEUE, AgentWorkflow, ToolApproval, create_worker
from tests.helpers.workflows import (
    Chat,
    ChatWorkflow,
    bash,
    start_workflow,
)

CLAUDE = Path(sdk.__file__).parent / "_bundled" / "claude"


class LocalCli:
    """The CLIs the Activities run on this host.

    ``cli_path`` runs the bundled Claude Code, or what :meth:`set_cli` makes it run.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.cli_path = root / "bin" / "claude"
        self.cli_path.parent.mkdir()
        self.set_cli(f'exec "{CLAUDE}" "$@"')

    def set_cli(self, script: str) -> None:
        """Make the CLI run ``script``, recording each pid, but answer the version check as Claude Code."""
        self.cli_path.write_text(
            "#!/bin/sh\n"
            f'[ "$1" = -v ] && exec "{CLAUDE}" -v\n'
            f'echo $$ >> "{self.root / "cli.pids"}"\n'
            f"{script}\n"
        )
        self.cli_path.chmod(0o755)

    def cli_pids(self) -> list[int]:
        """Each CLI started so far."""
        pids = self.root / "cli.pids"
        return [int(pid) for pid in pids.read_text().split()] if pids.exists() else []

    def running(self) -> list[int]:
        """Each CLI started so far that is still running."""
        return [pid for pid in self.cli_pids() if _running(pid)]

    def kill_processes(self) -> None:
        """Kill each CLI started so far."""
        # Windows has no SIGKILL. No test runs there, but the type checkers do.
        kill_signal = signal.SIGTERM if sys.platform == "win32" else signal.SIGKILL
        for pid in self.running():
            os.kill(pid, kill_signal)


def _running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


# Answers for "write it", which writes notes.txt in the workspace, and "read it".
ANSWERS = {
    "write it": bash("toolu_1", "echo hi > notes.txt"),
    "tool_result:toolu_1": "Wrote it.",
    "read it": bash("toolu_2", "cat notes.txt"),
    "tool_result:toolu_2": "Read it.",
}
# What the Claude Agent SDK gets for "write it", but for the approval.
WRITE_IT_MESSAGES: list[Any] = [
    ["AssistantMessage", ["toolu_1", "Bash", {"command": "echo hi > notes.txt"}]],
    ["UserMessage", ["toolu_1", "(Bash completed with no output)", False]],
    ["AssistantMessage", ["Wrote it."]],
    ["Result", "Wrote it."],
]


@contextlib.asynccontextmanager
async def example_chat(
    client: Client,
) -> AsyncIterator[WorkflowHandle[AgentWorkflow, None]]:
    """Start the example chat on its Worker, and end the conversation afterwards.

    Ends the Workflow's latest run, such as one after a reset, and replays its
    history, which fails if it is not deterministic.
    """
    async with create_worker(client):
        handle = await client.start_workflow(
            AgentWorkflow.run, id=f"example-{uuid.uuid4()}", task_queue=TASK_QUEUE
        )
        yield handle
        latest = client.get_workflow_handle(handle.id)
        await latest.execute_update(AgentWorkflow.end_conversation)
        await latest.result()
    plugin = ClaudeAgentSDKPlugin()
    replayer = Replayer(workflows=[AgentWorkflow], plugins=[plugin])
    await replayer.replay_workflow(await latest.fetch_history())


async def say(handle: WorkflowHandle[Any, Any], message: str) -> str:
    """Send the example chat ``message``, approving each tool call, and return its answer."""
    reply = asyncio.ensure_future(
        handle.execute_update(AgentWorkflow.send_message, message)
    )
    while not reply.done():
        for request in await handle.query(AgentWorkflow.pending_approvals):
            approval = ToolApproval(request.tool_use_id, True)
            await handle.execute_update(AgentWorkflow.approve_tool, approval)
        await asyncio.wait([reply], timeout=0.05)
    return await reply


class Harness(LocalCli):
    """Chats with Claude Code on this host, with a scripted API in place of Anthropic's.

    The API answers each request with the first key of ``answers`` that its last
    user message contains; a tool result reads as ``tool_result:<tool_use_id>``.
    An answer is a text, a tool use, or a function of the request that returns
    one. Each request's key goes in ``asked``.

    While a chat runs, ``pending`` has the tool calls waiting on a person, and
    ``hook_inputs`` each hook callback's input so far.

    Claude Code runs on the test's Worker, with this process's environment and
    working directory, so until the test ends they point it at the scripted
    API, a config dir under ``root``, and ``workspace``.
    """

    def __init__(
        self, client: Client, root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        super().__init__(root)
        self.client = client
        self.answers: dict[str, Any] = {}
        self.asked: list[str] = []
        self.pending: list[str] = []
        self.hook_inputs: list[dict[str, Any]] = []
        self.interceptors: list[Interceptor] = []
        self.api = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        threading.Thread(target=self.api.serve_forever, daemon=True).start()
        self.workspace = root / "workspace"
        self.workspace.mkdir()
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.setenv(
            "ANTHROPIC_BASE_URL", f"http://127.0.0.1:{self.api.server_address[1]}"
        )
        monkeypatch.setenv("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1")
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(root / "config"))
        monkeypatch.chdir(self.workspace)

    async def run(
        self,
        answers: dict[str, Any],
        prompts: list[str],
        *,
        approve: bool = True,
        decide_after: float = 0,
        interrupt_when: Callable[[], object] | None = None,
        **options: Any,
    ) -> list[Any]:
        """Chat to the end, and replay its history.

        Decides every tool call ``approve``, ``decide_after`` seconds after it is
        asked. Interrupts the chat the first time ``interrupt_when`` holds.
        ``options`` are the rest of :class:`Chat`.
        """
        self.answers = answers
        asked_at: dict[str, float] = {}  # When each tool call was first seen waiting.
        decided: set[str] = set()
        loop = asyncio.get_running_loop()
        chat = Chat(prompts, str(self.cli_path), **options)
        async with self.start(ChatWorkflow, chat) as handle:
            result = asyncio.ensure_future(handle.result())
            while not result.done():
                self.pending = await handle.query(ChatWorkflow.pending_approvals)
                self.hook_inputs = await handle.query(ChatWorkflow.hooks_run)
                for tool_use_id in self.pending:
                    asked_at.setdefault(tool_use_id, loop.time())
                due = [
                    tool_use_id
                    for tool_use_id, at in asked_at.items()
                    if tool_use_id not in decided and at <= loop.time() - decide_after
                ]
                for tool_use_id in due:
                    decided.add(tool_use_id)
                    decision = [tool_use_id, approve]
                    await handle.execute_update(ChatWorkflow.decide, args=decision)
                if interrupt_when is not None and interrupt_when():
                    interrupt_when = None
                    await handle.execute_update(ChatWorkflow.interrupt)
                await asyncio.wait([result], timeout=0.05)
            messages = await result
            self.hook_inputs = await handle.query(ChatWorkflow.hooks_run)
        return messages

    @contextlib.asynccontextmanager
    async def start(
        self, workflow_type: type, *args: Any
    ) -> AsyncIterator[WorkflowHandle[Any, Any]]:
        """Start ``workflow_type``: see :func:`start_workflow`."""
        async with start_workflow(
            self.client,
            workflow_type,
            *args,
            interceptors=self.interceptors,
        ) as handle:
            yield handle

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        harness = self

        class FakeAnthropic(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                key = next((k for k in harness.answers if k in prompt_of(body)), None)
                if key is not None:
                    harness.asked.append(key)
                answer = harness.answers.get(key or "", "")
                if callable(answer):
                    answer = answer(body)
                model = body.get("model", "claude")
                self.reply(stream(answer, model))

            def reply(self, data: bytes) -> None:
                # A CLI that was interrupted is no longer reading.
                with contextlib.suppress(OSError):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)

            def log_message(self, format: str, *args: Any) -> None:
                pass

        return FakeAnthropic


def prompt_of(body: dict[str, Any]) -> str:
    """The request's last user message."""
    content = next(
        m["content"] for m in reversed(body["messages"]) if m["role"] == "user"
    )
    if isinstance(content, str):
        return content
    return " ".join(
        b.get("text") or f"tool_result:{b.get('tool_use_id')}" for b in content
    )


def stream(answer: Any, model: str = "claude") -> bytes:
    """The Messages API's event stream for a text answer, or a tool use."""
    tool = isinstance(answer, dict)
    message = {
        "id": f"msg_{uuid.uuid4().hex}",
        "type": "message",
        "model": model,
        "role": "assistant",
        "content": [],
        "usage": {"input_tokens": 1},
    }
    if tool:
        start = {**answer, "input": {}}
        delta = {
            "type": "input_json_delta",
            "partial_json": json.dumps(answer["input"]),
        }
    else:
        start = {"type": "text", "text": ""}
        delta = {"type": "text_delta", "text": answer}
    stop = {"stop_reason": "tool_use" if tool else "end_turn"}
    events: list[dict[str, Any]] = [
        {"type": "message_start", "message": message},
        {"type": "content_block_start", "index": 0, "content_block": start},
        {"type": "content_block_delta", "index": 0, "delta": delta},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": stop, "usage": {"output_tokens": 1}},
        {"type": "message_stop"},
    ]
    return "".join(
        f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events
    ).encode()
