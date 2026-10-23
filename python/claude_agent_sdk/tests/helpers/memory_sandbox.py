"""A sandbox in this process, with a scripted Claude Code: no processes, files on disk, or model.

The Activities run shell scripts in the sandbox. :class:`MemorySandbox` runs
each one's in-memory equivalent on a dict of files, so a change to a script
fails here until its equivalent is changed too. The integration tests run the
scripts themselves, with the bundled Claude Code.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import json
import uuid
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any

from temporalio.claude_agent_sdk import (
    ExecResult,
    Sandbox,
    SandboxError,
    SandboxFactory,
)
from temporalio.claude_agent_sdk._payloads import END_OF_INPUT
from temporalio.claude_agent_sdk._shell import script
from temporalio.client import Client, WorkflowHandle
from tests.helpers.workflows import Chat, ChatWorkflow, start_workflow

# The exit status sh reports for a CLI that was killed.
_KILLED = 137
# How often the CLI and the follower look for more input.
_POLL = 0.01


class MemorySandbox(Sandbox):
    """Runs the Activities' commands on files in a dict, with a scripted Claude Code as the CLI.

    Claude Code answers each prompt with the first key of ``answers`` it
    contains; a tool result reads as ``tool_result:<tool_use_id>``. A text
    answer ends the turn. A tool use asks the Claude Agent SDK whether to run
    it, unless it is allowed, and goes on with the answer to its result. Each key asked goes
    in ``asked``, and each tool call run in ``ran``.

    Launched with ``--include-partial-messages``, it emits two ``stream_event``
    lines before each assistant message, as Claude Code does.

    ``crash_before`` is a key: the CLI is killed the first time it is asked that
    key, before it answers. ``fail`` is the stderr of a CLI that exits at once
    without output.
    """

    def __init__(self) -> None:
        self.answers: dict[str, Any] = {}
        self.asked: list[str] = []
        self.ran: list[list[Any]] = []
        self.crash_before: str | None = None
        self.fail: str | None = None
        self.files: dict[str, bytearray] = {}
        # The flags of each CLI.
        self.cli_args: list[list[str]] = []
        # The running sandbox's boot ID.
        self._boot: str | None = None
        # The mode of each file write_files wrote.
        self.modes: dict[str, int] = {}
        self._clis: dict[int, _Cli] = {}
        self._ids = itertools.count(1)

    @property
    def cli_path(self) -> str:
        return "claude"

    @property
    def home(self) -> str:
        return "/home/agent/.claude-agent"

    async def start(self, *, env: Mapping[str, str], **kwargs: Any) -> None:
        # As LocalSandbox does: a new sandbox ends the last.
        await self.terminate()
        self._boot = f"boot-{next(self._ids)}"

    async def exec(self, argv: Sequence[str], **kwargs: Any) -> ExecResult:
        self._check()
        await asyncio.sleep(0)
        if argv[0] == "cat":
            return ExecResult(0, bytes(self.files.get(argv[1], b"")), b"")
        text, args = argv[2], list(argv[4:])
        if text == script("launch"):
            return self._launch(args)
        if text == script("read_stdout"):
            return ExecResult(0, self._read(args[0], int(args[1])), b"")
        raise AssertionError(f"MemorySandbox has no equivalent of {list(argv)}")

    async def write_files(
        self, files: Mapping[str, tuple[bytes, int]], **kwargs: Any
    ) -> None:
        self._check()
        for path, (content, mode) in files.items():
            self.files[path] = bytearray(content)
            self.modes[path] = mode

    async def stream(self, argv: Sequence[str], **kwargs: Any) -> AsyncIterator[bytes]:
        """Follow the CLI's stdout, as follow_stdout.sh does."""
        self._check()
        assert argv[2] == script("follow_stdout"), argv
        stdout, start, exited, pid, boot = argv[4:]
        at = int(start) - 1
        while True:
            data = self._read(stdout, at + 1)
            if data:
                at += len(data)
                yield data
            self._check()
            if not self._alive(exited, int(pid), boot):
                return
            await asyncio.sleep(_POLL)

    async def terminate(self, **kwargs: Any) -> None:
        self._boot = None
        self.stop()

    def crash(self) -> None:
        """Kill the running CLI."""
        [cli] = [cli for cli in self._clis.values() if cli.running]
        cli.exit(_KILLED)

    def stop(self) -> None:
        """Stop every CLI."""
        for cli in self._clis.values():
            cli.stop()

    def _check(self) -> None:
        if self._boot is None:
            raise SandboxError("Sandbox is gone")

    def _read(self, path: str, start: int) -> bytes:
        """The file from byte ``start``, counted from 1 as tail does."""
        return bytes(self.files.get(path, b""))[start - 1 :]

    def _alive(self, exited: str, pid: int, boot: str) -> bool:
        cli = self._clis.get(pid)
        return (
            exited not in self.files
            and boot == self._boot
            and cli is not None
            and cli.running
        )

    def _launch(self, args: list[str]) -> ExecResult:
        """Start the CLI once, as launch.sh does."""
        stdin, stdout, log, pid, _new_pid, exited, _run, _cli = args[:8]
        flags = args[8:]
        if pid not in self.files:
            self.files[stdout] = bytearray()
            number = next(self._ids)
            boot = self._boot
            self.files[pid] = bytearray(f"{number} {boot}\n".encode())
            self.cli_args.append(flags)
            cli = _Cli(self, flags, stdin, stdout, log, exited)
            self._clis[number] = cli
            cli.start()
        return ExecResult(0, bytes(self.files[pid]), b"")


class _Cli:
    """A scripted Claude Code, which reads stream-json from the files in its stdin directory and writes it to its stdout file."""

    def __init__(
        self,
        sandbox: MemorySandbox,
        flags: list[str],
        stdin: str,
        stdout: str,
        log: str,
        exited: str,
    ) -> None:
        self.sandbox = sandbox
        self.stdin, self.stdout, self.log, self.exited = stdin, stdout, log, exited
        self.session_id = str(uuid.uuid4())
        self.allowed = (flag(flags, "--allowedTools") or "").split(",")
        self.partial = "--include-partial-messages" in flags
        self.running = True
        self._task: asyncio.Task[None] | None = None
        self._turns: asyncio.Queue[str] = asyncio.Queue()
        # The Claude Agent SDK's answers this CLI waits on, by request ID.
        self._waiting: dict[str, asyncio.Future[dict[str, Any]]] = {}

    def start(self) -> None:
        fail = self.sandbox.fail
        if fail is not None:
            self.sandbox.files[self.log] = bytearray(fail.encode())
            self.exit(2)
            return
        self._task = asyncio.create_task(self._run())

    def exit(self, status: int) -> None:
        """End, with ``status`` in the exit file, as run_cli.sh writes it."""
        self.stop()
        self.sandbox.files[self.exited] = bytearray(f"{status}\n".encode())

    def stop(self) -> None:
        """End, without writing the exit file, as when the sandbox is lost."""
        self.running = False
        # Not the CLI's own task: the crash it asks for ends it.
        with contextlib.suppress(RuntimeError):  # No event loop, at teardown.
            if self._task is not asyncio.current_task():
                self._task_cancel()

    def _task_cancel(self) -> None:
        if self._task is not None:
            self._task.cancel()

    async def _run(self) -> None:
        turns = asyncio.create_task(self._take_turns())
        try:
            await self._read_stdin()
            await self._turns.join()
        finally:
            turns.cancel()
        self.exit(0)

    async def _read_stdin(self) -> None:
        """Handle each line of the stdin files in order, 1.jsonl, 2.jsonl, ..., up to END_OF_INPUT, as run_cli.sh feeds them."""
        for number in itertools.count(1):
            path = f"{self.stdin}/{number}.jsonl"
            while path not in self.sandbox.files:
                await asyncio.sleep(_POLL)
            for line in bytes(self.sandbox.files[path]).decode().splitlines():
                if line == END_OF_INPUT:
                    return
                self._on_line(json.loads(line))

    def _on_line(self, message: dict[str, Any]) -> None:
        kind = message["type"]
        if kind == "control_request":
            response = {"subtype": "success", "request_id": message["request_id"]}
            self._emit(
                {"type": "control_response", "response": {**response, "response": {}}}
            )
        elif kind == "control_response":
            response = message["response"]
            waiting = self._waiting.pop(response["request_id"], None)
            if waiting is not None:
                waiting.set_result(response)
        elif kind == "user":
            content = message["message"]["content"]
            self._turns.put_nowait(
                content if isinstance(content, str) else _text(content)
            )

    async def _take_turns(self) -> None:
        while True:
            turn = await self._turns.get()
            await self._take(turn)
            self._turns.task_done()

    async def _take(self, prompt: str) -> None:
        """Answer the prompt, running each tool the model calls, up to a text."""
        self._emit({"type": "system", "subtype": "init", "session_id": self.session_id})
        said = prompt
        while not isinstance(answer := await self._answer(said), str):
            self._assistant(answer)
            said = f"tool_result:{answer['id']}"
            self._emit(_user(await self._use(answer)))
        self._assistant({"type": "text", "text": answer})
        self._result(answer)
        idle = {"type": "system", "subtype": "session_state_changed", "state": "idle"}
        self._emit(idle)

    async def _answer(self, said: str) -> Any:
        sandbox = self.sandbox
        key = next((k for k in sandbox.answers if k in said), None)
        if key is not None:
            first = key not in sandbox.asked
            sandbox.asked.append(key)
            if first and key == sandbox.crash_before:
                sandbox.crash()
                await asyncio.Event().wait()
        return sandbox.answers.get(key or "", "")

    async def _use(self, tool_use: dict[str, Any]) -> dict[str, Any]:
        """Run the tool call, once the Claude Agent SDK allows it unless it is allowed, and return its result."""
        name, tool_input, tool_use_id = (
            tool_use["name"],
            tool_use["input"],
            tool_use["id"],
        )
        if name not in self.allowed:
            request_id = str(uuid.uuid4())
            answer = asyncio.get_running_loop().create_future()
            self._waiting[request_id] = answer
            request = {
                "subtype": "can_use_tool",
                "tool_name": name,
                "input": tool_input,
                "tool_use_id": tool_use_id,
                "permission_suggestions": None,
            }
            self._emit(
                {
                    "type": "control_request",
                    "request_id": request_id,
                    "request": request,
                }
            )
            response = await answer
            decision = response.get("response") or {}
            if decision.get("behavior") != "allow":
                message = decision.get("message") or response.get("error", "")
                return _tool_result(tool_use_id, message, is_error=True)
            tool_input = decision.get("updatedInput") or tool_input
        self.sandbox.ran.append([name, tool_input])
        return _tool_result(tool_use_id, f"Ran {name}", is_error=False)

    def _assistant(self, block: dict[str, Any]) -> None:
        if self.partial:
            for index in (0, 1):
                event = {"type": "content_block_delta", "index": index}
                self._emit(
                    {
                        "type": "stream_event",
                        "uuid": str(uuid.uuid4()),
                        "session_id": self.session_id,
                        "event": event,
                    }
                )
        message = {
            "id": f"msg_{uuid.uuid4().hex}",
            "model": "claude",
            "role": "assistant",
            "content": [block],
        }
        self._emit(
            {"type": "assistant", "message": message, "parent_tool_use_id": None}
        )

    def _result(self, text: str) -> None:
        self._emit(
            {
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "duration_ms": 1,
                "duration_api_ms": 1,
                "num_turns": 1,
                "session_id": self.session_id,
                "result": text,
            }
        )

    def _emit(self, message: dict[str, Any]) -> None:
        if self.running:
            self.sandbox.files[self.stdout] += f"{json.dumps(message)}\n".encode()


def flag(flags: Sequence[str], name: str) -> str | None:
    """The value of the CLI flag ``name``, given as ``name=value`` or ``name value``."""
    for index, given in enumerate(flags):
        if given.startswith(f"{name}="):
            return given.partition("=")[2]
        if given == name:
            return flags[index + 1]
    return None


def _text(content: list[dict[str, Any]]) -> str:
    return " ".join(block.get("text", "") for block in content)


def _tool_result(tool_use_id: str, content: str, *, is_error: bool) -> dict[str, Any]:
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": content,
        "is_error": is_error,
    }


def _user(block: dict[str, Any]) -> dict[str, Any]:
    message = {"role": "user", "content": [block]}
    return {"type": "user", "message": message, "parent_tool_use_id": None}


class MemoryHarness:
    """Chats through a :class:`MemorySandbox`.

    ``factory`` is what the plugin registers as the sandbox ``local``: by
    default one that returns ``sandbox``.
    """

    def __init__(self, client: Client) -> None:
        self.client = client
        self.sandbox = MemorySandbox()
        self.factory: SandboxFactory = lambda _: self.sandbox
        # The Workflow started last.
        self.handle: WorkflowHandle[Any, Any] | None = None

    @contextlib.asynccontextmanager
    async def start(
        self, workflow_type: type, *args: Any
    ) -> AsyncIterator[WorkflowHandle[Any, Any]]:
        """Start ``workflow_type``: see :func:`start_workflow`."""
        async with start_workflow(
            self.client, self.factory, workflow_type, *args
        ) as handle:
            self.handle = handle
            yield handle

    async def execute(
        self, workflow_type: type, *args: Any, answers: dict[str, Any]
    ) -> Any:
        """Run ``workflow_type`` to its result, with ``answers``."""
        self.sandbox.answers = answers
        async with self.start(workflow_type, *args) as handle:
            return await handle.result()

    async def run(
        self,
        answers: dict[str, Any],
        prompts: list[str],
        *,
        approve: bool = True,
        **options: Any,
    ) -> list[Any]:
        """Chat to the end, deciding each tool call ``approve``."""
        self.sandbox.answers = answers
        decided: set[str] = set()
        async with self.start(ChatWorkflow, Chat(prompts, **options)) as handle:
            result = asyncio.ensure_future(handle.result())
            while not result.done():
                for tool_use_id in await handle.query(ChatWorkflow.pending_approvals):
                    if tool_use_id in decided:
                        continue
                    decided.add(tool_use_id)
                    decision = [tool_use_id, approve]
                    await handle.execute_update(ChatWorkflow.decide, args=decision)
                await asyncio.wait([result], timeout=0.05)
            return await result

    @property
    def run_id(self) -> str:
        """The first run of the Workflow started last."""
        assert self.handle is not None
        return self.handle.result_run_id or ""
