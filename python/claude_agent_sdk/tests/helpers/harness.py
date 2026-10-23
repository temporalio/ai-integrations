"""The bundled Claude Code in a local sandbox, with a scripted Anthropic API."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import os
import shutil
import signal
import tempfile
import threading
import uuid
from asyncio.subprocess import DEVNULL, PIPE
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import claude_agent_sdk as sdk

from samples.worker import TASK_QUEUE, create_worker
from samples.workflow import AgentWorkflow, ToolApproval
from temporalio.claude_agent_sdk import (
    ClaudeAgentSDKPlugin,
    ExecResult,
    Sandbox,
    SandboxError,
    SandboxFactory,
)
from temporalio.claude_agent_sdk._home import SandboxHome
from temporalio.client import Client, WorkflowHandle
from temporalio.worker import Interceptor, Replayer
from tests.helpers.workflows import (
    Chat,
    ChatWorkflow,
    bash,
    start_workflow,
    until,
)

CLAUDE = Path(sdk.__file__).parent / "_bundled" / "claude"


class LocalSandbox(Sandbox):
    """The sandbox is a directory on this host, and its session's storage another, which outlives it.

    Each harness runs one session.
    """

    def __init__(self, root: Path, env: dict[str, str]) -> None:
        self.root, self.env = root, env
        self.path = root / "sandbox"
        self.storage = root / "storage"
        self.workspace = str(self.storage / "workspace")
        self.process_groups: list[int] = []
        (root / "bin").mkdir()
        self.set_cli(f'exec "{CLAUDE}" "$@"')

    @property
    def cli_path(self) -> str:
        return str(self.root / "bin" / "claude")

    @property
    def home(self) -> str:
        return str(self.storage / "home")

    @property
    def files(self) -> SandboxHome:
        """Where the integration keeps its files in ``home``."""
        return SandboxHome(self.home)

    @property
    def config(self) -> Path:
        """Claude Code's config dir."""
        return Path(self.files.config)

    def set_cli(self, script: str) -> None:
        """Make the CLI run ``script``."""
        pid = self.root / "cli.pid"
        record = f'echo $$ > "{pid}"; echo $$ >> "{pid}s"'
        Path(self.cli_path).write_text(f"#!/bin/sh\n{record}\n{script}\n")
        Path(self.cli_path).chmod(0o755)

    async def start(self, *, env: Mapping[str, str], **kwargs: Any) -> None:
        await self.terminate()
        self.path.mkdir()
        Path(self.workspace).mkdir(parents=True, exist_ok=True)
        (self.path / "env.json").write_text(json.dumps({**env, **self.env}))

    async def exec(self, argv: Sequence[str], **kwargs: Any) -> ExecResult:
        run = await self._run(argv, stdin=DEVNULL, stderr=PIPE)
        stdout, stderr = await run.communicate()
        return ExecResult(run.returncode or 0, stdout, stderr)

    async def stream(self, argv: Sequence[str], **kwargs: Any) -> AsyncIterator[bytes]:
        run = await self._run(argv, stdin=DEVNULL, stderr=DEVNULL)
        assert run.stdout is not None
        try:
            while chunk := await run.stdout.read(1 << 16):
                yield chunk
            await run.wait()
        finally:
            # As DockerSandbox does: stop the command, but not what it started.
            if run.returncode is None:
                run.terminate()
                await run.wait()
        if not self.path.is_dir():
            raise SandboxError("Sandbox is gone")

    async def _run(
        self, argv: Sequence[str], **pipes: Any
    ) -> asyncio.subprocess.Process:
        if not self.path.is_dir():
            raise SandboxError("Sandbox is gone")
        path = f"{self.root / 'bin'}{os.pathsep}{os.environ['PATH']}"
        sandbox_env = json.loads((self.path / "env.json").read_text())
        # Each command leads its own process group, so terminate can kill what it
        # left running.
        run = await asyncio.create_subprocess_exec(
            *argv,
            stdout=PIPE,
            env={**os.environ, "PATH": path, **sandbox_env},
            cwd=self.workspace,
            start_new_session=True,
            **pipes,
        )
        self.process_groups.append(run.pid)
        return run

    async def write_files(
        self, files: Mapping[str, tuple[bytes, int]], **kwargs: Any
    ) -> None:
        if not self.path.is_dir():
            raise SandboxError("Sandbox is gone")
        for path, (content, mode) in files.items():
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            # Each file appears whole: written aside, then moved into place.
            descriptor, aside = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
            with os.fdopen(descriptor, "wb") as file:
                file.write(content)
            os.chmod(aside, mode)
            os.replace(aside, target)

    async def terminate(self, **kwargs: Any) -> None:
        self.kill_processes()
        shutil.rmtree(self.path, ignore_errors=True)

    def kill_processes(self) -> None:
        """Kill everything any sandbox command left running."""
        for group in self.process_groups:
            with contextlib.suppress(OSError):
                os.killpg(group, signal.SIGKILL)
        self.process_groups.clear()


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
    client: Client, factory: SandboxFactory
) -> AsyncIterator[WorkflowHandle[AgentWorkflow, None]]:
    """Start the example chat on its Worker, and end the conversation afterwards.

    Ends the Workflow's latest run, such as one after a reset, and replays its
    history, which fails if it is not deterministic.
    """
    async with create_worker(client, factory):
        handle = await client.start_workflow(
            AgentWorkflow.run, id=f"example-{uuid.uuid4()}", task_queue=TASK_QUEUE
        )
        yield handle
        latest = client.get_workflow_handle(handle.id)
        await latest.execute_update(AgentWorkflow.end_conversation)
        await latest.result()
    plugin = ClaudeAgentSDKPlugin({"docker": factory})
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


class Harness:
    """Chats through a local sandbox, with a scripted API in place of Anthropic's.

    The API answers each request with the first key of ``answers`` that its last
    user message contains; a tool result reads as ``tool_result:<tool_use_id>``.
    An answer is a text, a tool use, a function of the request that returns one,
    or a :class:`Hang`. Each request goes in ``requests``, and its key in
    ``asked``.

    While a chat runs, ``pending`` has the tool calls waiting on a person, and
    ``hook_inputs`` each hook callback's input so far.

    ``factory`` is what the plugin registers as the sandbox ``local``: by
    default one that returns ``sandbox``.
    """

    def __init__(self, client: Client, root: Path) -> None:
        self.client, self.root = client, root
        self.answers: dict[str, Any] = {}
        self.asked: list[str] = []
        self.requests: list[dict[str, Any]] = []
        self.pending: list[str] = []
        self.hook_inputs: list[dict[str, Any]] = []
        self.release = threading.Event()
        self.interceptors: list[Interceptor] = []
        # The Workflow started last.
        self.handle: WorkflowHandle[Any, Any] | None = None
        self.api = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        threading.Thread(target=self.api.serve_forever, daemon=True).start()
        self.sandbox = LocalSandbox(root, self.api_env())
        self.factory: SandboxFactory = lambda _: self.sandbox

    def api_env(self, host: str = "127.0.0.1") -> dict[str, str]:
        """The environment that points Claude Code at the scripted API, on ``host``."""
        return {
            "ANTHROPIC_API_KEY": "test-key",
            "ANTHROPIC_BASE_URL": f"http://{host}:{self.api.server_address[1]}",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        }

    async def run(
        self,
        answers: dict[str, Any],
        prompts: list[str],
        *,
        approve: bool = True,
        decide_after: float = 0,
        interrupt_when: Callable[[], object] | None = None,
        release_when: Callable[[], object] | None = None,
        **options: Any,
    ) -> list[Any]:
        """Chat to the end, and replay its history.

        Decides every tool call ``approve``, ``decide_after`` seconds after it is
        asked. Interrupts the chat the first time ``interrupt_when`` holds, and
        sets ``release`` once ``release_when`` holds. ``options`` are the rest
        of :class:`Chat`.
        """
        self.answers = answers
        asked_at: dict[str, float] = {}  # When each tool call was first seen waiting.
        decided: set[str] = set()
        loop = asyncio.get_running_loop()
        async with self.start(ChatWorkflow, Chat(prompts, **options)) as handle:
            self.handle = handle
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
                if release_when is not None and release_when():
                    self.release.set()
                await asyncio.wait([result], timeout=0.05)
            messages = await result
            self.hook_inputs = await handle.query(ChatWorkflow.hooks_run)
        return messages

    async def execute(
        self, workflow_type: type, *args: Any, answers: dict[str, Any]
    ) -> Any:
        """Run ``workflow_type`` to its result, with ``answers``, and replay its history."""
        self.answers = answers
        async with self.start(workflow_type, *args) as handle:
            return await handle.result()

    @contextlib.asynccontextmanager
    async def start(
        self, workflow_type: type, *args: Any
    ) -> AsyncIterator[WorkflowHandle[Any, Any]]:
        """Start ``workflow_type`` with the local sandbox: see :func:`start_workflow`."""
        async with start_workflow(
            self.client,
            self.factory,
            workflow_type,
            *args,
            interceptors=self.interceptors,
        ) as handle:
            self.handle = handle
            yield handle

    @property
    def run_id(self) -> str:
        """The first run of the Workflow started last, or else the run ActivityEnvironment gives Activities."""
        if self.handle is None:
            return "test-run"
        return self.handle.result_run_id or ""

    def cli_pids(self) -> list[int]:
        """Each CLI started so far."""
        pids = self.root / "cli.pids"
        return [int(pid) for pid in pids.read_text().split()] if pids.exists() else []

    def kill_cli(self) -> None:
        """Kill the CLI started last."""
        os.kill(int((self.root / "cli.pid").read_text()), signal.SIGKILL)

    async def kill_cli_and_wait(self) -> None:
        """Kill the CLI started last, and wait until its exit is recorded."""
        # Launching returns before the CLI's shell has started.
        await until((self.root / "cli.pid").exists)
        self.kill_cli()
        await until(Path(self.sandbox.files.cli.exited).exists)

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        harness = self

        class FakeAnthropic(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                harness.requests.append(body)
                key = next((k for k in harness.answers if k in prompt_of(body)), None)
                if key is not None:
                    harness.asked.append(key)
                answer = harness.answers.get(key or "", "")
                if callable(answer):
                    answer = answer(body)
                if isinstance(answer, Hang):
                    harness.release.wait(120)
                    answer = answer.answer
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


@dataclasses.dataclass
class Hang:
    """An answer that does not come until the harness is released."""

    answer: Any = "Too late."


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
