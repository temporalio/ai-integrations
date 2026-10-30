"""The Workflows the tests chat through, and how they describe what the Claude Agent SDK gets."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import uuid
from collections.abc import AsyncIterator, Callable, Sequence
from typing import Any

import claude_agent_sdk as sdk
from claude_agent_sdk.types import HookEvent

from temporalio import workflow
from temporalio.claude_agent_sdk import (
    ClaudeAgentSDKPlugin,
    TemporalTransport,
)
from temporalio.client import Client, WorkflowHandle
from temporalio.contrib.workflow_streams import WorkflowStream
from temporalio.worker import Interceptor, Replayer, Worker


@dataclasses.dataclass
class Hook:
    """A Claude Agent SDK hook a ChatWorkflow registers: one matcher with one callback.

    The callback records its input, waits ``wait`` seconds, then returns ``output``,
    or raises with ``fail``.
    """

    event: HookEvent
    output: dict[str, Any] = dataclasses.field(default_factory=dict[str, Any])
    matcher: str | None = None
    wait: float = 0
    fail: bool = False
    timeout: float | None = None


@dataclasses.dataclass
class Chat:
    """The prompts a ChatWorkflow sends, and the options it sends them with."""

    prompts: list[str]
    cli_path: str | None = None
    """The CLI to run on the Worker, or ``None`` for the bundled one."""
    allowed_tools: list[str] = dataclasses.field(default_factory=list[str])
    deny_interrupt: bool = False
    """Deny with ``interrupt=True``."""
    tools: list[str] | None = None
    hooks: list[Hook] = dataclasses.field(default_factory=list[Hook])
    sdk_tools: bool = False
    """Serve ``mcp__app__echo`` from a Claude Agent SDK MCP server in the Workflow."""
    follow_ups: int = 0
    """How many turns more to read after the prompts', which Claude Code starts on its own."""


@workflow.defn
class ChatWorkflow:
    """Sends each prompt, asks a person about each tool call, and describes it all."""

    @workflow.init
    def __init__(self, chat: Chat) -> None:
        self.messages: list[Any] = []
        self.pending: list[str] = []
        self.decisions: dict[str, bool] = {}
        self.chat = chat
        self.client: sdk.ClaudeSDKClient | None = None
        self.hook_inputs: list[dict[str, Any]] = []

    @workflow.run
    async def run(self, chat: Chat) -> list[Any]:
        hooks: dict[HookEvent, list[sdk.HookMatcher]] = {}
        for hook in chat.hooks:
            matcher = sdk.HookMatcher(hook.matcher, [self.callback(hook)], hook.timeout)
            hooks.setdefault(hook.event, []).append(matcher)
        options = sdk.ClaudeAgentOptions(
            cli_path=chat.cli_path,
            can_use_tool=self.approve,
            permission_mode="default",
            allowed_tools=chat.allowed_tools,
            tools=chat.tools,
            hooks=hooks,
            mcp_servers={"app": self.app_server()} if chat.sdk_tools else {},
        )
        transport = TemporalTransport(options)
        async with sdk.ClaudeSDKClient(options=options, transport=transport) as client:
            self.client = client
            for prompt in chat.prompts:
                await client.query(prompt)
                await self.receive(client)
            for _ in range(chat.follow_ups):
                await self.receive(client)
        return self.messages

    async def receive(self, client: sdk.ClaudeSDKClient) -> None:
        async for message in client.receive_response():
            if not isinstance(message, sdk.SystemMessage):
                self.messages.append(describe(message))

    async def approve(
        self, _tool: str, _input: dict[str, Any], context: sdk.ToolPermissionContext
    ) -> sdk.PermissionResult:
        tool_use_id = context.tool_use_id or ""
        self.messages.append(["Ask", tool_use_id])
        self.pending.append(tool_use_id)
        try:
            await workflow.wait_condition(lambda: tool_use_id in self.decisions)
        except asyncio.CancelledError:
            self.messages.append(["Cancelled", tool_use_id])
            raise
        finally:
            self.pending.remove(tool_use_id)
        if not self.decisions[tool_use_id]:
            return sdk.PermissionResultDeny(
                message="A person denied it", interrupt=self.chat.deny_interrupt
            )
        return sdk.PermissionResultAllow()

    def app_server(self) -> sdk.McpSdkServerConfig:
        """A Claude Agent SDK MCP server whose ``echo`` tool records each call."""

        @sdk.tool("echo", "Echo the text back", {"text": str})
        async def echo(args: dict[str, Any]) -> dict[str, Any]:
            self.messages.append(["Echo", args["text"]])
            return {"content": [{"type": "text", "text": f"echo: {args['text']}"}]}

        return sdk.create_sdk_mcp_server("app", tools=[echo])

    def callback(self, hook: Hook) -> sdk.HookCallback:
        async def callback(
            hook_input: Any, tool_use_id: str | None, _context: sdk.HookContext
        ) -> Any:
            self.hook_inputs.append(hook_input)
            self.messages.append(["Hook", hook_input["hook_event_name"], tool_use_id])
            if hook.wait:
                await workflow.sleep(hook.wait)
            if hook.fail:
                raise RuntimeError("The hook failed")
            return hook.output

        return callback

    @workflow.query
    def hooks_run(self) -> list[dict[str, Any]]:
        """Each hook callback's input, in the order they ran."""
        return self.hook_inputs

    @workflow.update
    async def interrupt(self) -> None:
        assert self.client is not None
        await self.client.interrupt()

    @workflow.update
    def decide(self, tool_use_id: str, allow: bool) -> None:
        self.decisions[tool_use_id] = allow

    @workflow.query
    def pending_approvals(self) -> list[str]:
        return self.pending


@workflow.defn
class OneShotWorkflow:
    """Sends the prompt with the Claude Agent SDK's ``query`` function, which ends the input after it."""

    @workflow.run
    async def run(
        self, prompt: str, allowed_tools: list[str] | None = None
    ) -> list[Any]:
        options = sdk.ClaudeAgentOptions(
            permission_mode="default", allowed_tools=allowed_tools or []
        )
        messages: list[Any] = []
        transport = TemporalTransport(options)
        async for message in sdk.query(
            prompt=prompt, options=options, transport=transport
        ):
            if not isinstance(message, sdk.SystemMessage):
                messages.append(describe(message))
        return messages


@workflow.defn
class StreamingWorkflow:
    """Sends the prompt with ``include_partial_messages`` and publishes the stream events to the ``events`` topic. Returns the messages once told to ``finish``."""

    @workflow.init
    def __init__(self, prompt: str) -> None:
        del prompt
        self.events = WorkflowStream().topic("events")
        self.finished = False

    @workflow.run
    async def run(self, prompt: str) -> list[Any]:
        options = sdk.ClaudeAgentOptions(include_partial_messages=True)
        transport = TemporalTransport(options, stream=self.events)
        messages: list[Any] = []
        async with sdk.ClaudeSDKClient(options=options, transport=transport) as client:
            await client.query(prompt)
            async for message in client.receive_response():
                if not isinstance(message, sdk.SystemMessage):
                    messages.append(describe(message))
        # Subscribers read the stream only while the Workflow runs.
        await workflow.wait_condition(lambda: self.finished)
        return messages

    @workflow.signal
    def finish(self) -> None:
        self.finished = True


# The text Claude Code sends itself to continue an interrupted turn.
CONTINUE = "Continue from where you left off."


@workflow.defn
class ResumingWorkflow:
    """Sends the prompt, asks a person about each tool call, and redoes the turn from the prompt if Claude Code goes away, as the samples do."""

    def __init__(self) -> None:
        self.pending: list[str] = []
        self.decisions: dict[str, bool] = {}
        self.launches = 0

    @workflow.run
    async def run(self, prompt: str, cli_path: str) -> str:
        session_id = None  # Claude Code's, once it started
        prompt_id = None  # The prompt's, once Claude Code saved it
        while True:
            try:
                self.launches += 1
                options = sdk.ClaudeAgentOptions(
                    cli_path=cli_path,
                    can_use_tool=self.approve,
                    permission_mode="default",
                    # Required to receive the prompt's UUID in the response stream.
                    extra_args={"replay-user-messages": None},
                    resume=session_id if prompt_id else None,
                    resume_session_at=prompt_id,
                )
                transport = TemporalTransport(options)
                async with sdk.ClaudeSDKClient(options, transport=transport) as client:
                    await client.query(CONTINUE if prompt_id else prompt)
                    async for message in client.receive_response():
                        if (
                            isinstance(message, sdk.SystemMessage)
                            and message.subtype == "init"
                        ):
                            session_id = message.data["session_id"]
                        elif (
                            isinstance(message, sdk.UserMessage)
                            and message.uuid
                            and not prompt_id
                        ):
                            prompt_id = message.uuid
                        elif isinstance(message, sdk.ResultMessage):
                            return message.result or ""
                raise RuntimeError("Claude Code ended the turn without a result")
            except sdk.CLIConnectionError:
                # Redo the turn from the prompt if Claude Code saved it, or start again.
                pass

    async def approve(
        self, tool_name: str, tool_input: dict[str, Any], context: Any
    ) -> sdk.PermissionResult:
        del tool_name, tool_input
        tool_use_id = context.tool_use_id or ""
        self.pending.append(tool_use_id)
        try:
            await workflow.wait_condition(lambda: tool_use_id in self.decisions)
        finally:
            self.pending.remove(tool_use_id)
        return sdk.PermissionResultAllow()

    @workflow.query
    def pending_approvals(self) -> list[str]:
        return self.pending

    @workflow.query
    def launch_count(self) -> int:
        return self.launches

    @workflow.update
    def decide(self, tool_use_id: str) -> None:
        self.decisions[tool_use_id] = True


@workflow.defn
class ForkingWorkflow:
    """Asks in a new session, then in a fork of it, then in the original again."""

    @workflow.run
    async def run(self, first: str, in_fork: str, in_original: str) -> list[str]:
        """The session IDs the Workflow chose, then the one each turn ran in."""
        original = str(workflow.uuid4())
        fork = str(workflow.uuid4())
        turns = [
            (first, sdk.ClaudeAgentOptions(session_id=original)),
            (
                in_fork,
                sdk.ClaudeAgentOptions(
                    resume=original, fork_session=True, session_id=fork
                ),
            ),
            (in_original, sdk.ClaudeAgentOptions(resume=original)),
        ]
        ran_in: list[str] = []
        for prompt, options in turns:
            transport = TemporalTransport(options)
            async with sdk.ClaudeSDKClient(options, transport=transport) as client:
                await client.query(prompt)
                async for message in client.receive_response():
                    if isinstance(message, sdk.ResultMessage):
                        ran_in.append(message.session_id)
        return [original, fork, *ran_in]


@workflow.defn
class CliRunWorkflow:
    """Runs the CLI at ``cli_path`` to its end, and returns the lines ``options.stderr`` got.

    With ``settings``, a JSON string, the sandbox is enabled too.
    """

    @workflow.run
    async def run(self, cli_path: str, settings: str | None = None) -> list[str]:
        lines: list[str] = []
        options = sdk.ClaudeAgentOptions(
            cli_path=cli_path,
            stderr=lines.append,
            settings=settings,
            sandbox={"enabled": True} if settings else None,
        )
        transport = TemporalTransport(options)
        await transport.connect()
        await transport.end_input()
        async for _ in transport.read_messages():
            pass
        await transport.close()
        return lines


def describe(message: sdk.Message) -> list[Any]:
    """The message's type and the fields of each content block."""
    if isinstance(message, sdk.ResultMessage):
        if message.is_error:
            return ["Result", message.result, message.subtype]
        return ["Result", message.result]
    blocks = getattr(message, "content", [])
    return [type(message).__name__, *[list(dataclasses.astuple(b)) for b in blocks]]


def tool_use(id: str, name: str, **tool_input: Any) -> dict[str, Any]:
    """An answer that calls the tool ``name`` with ``tool_input``."""
    return {"type": "tool_use", "id": id, "name": name, "input": tool_input}


def bash(id: str, command: str) -> dict[str, Any]:
    """An answer that asks to run ``command``."""
    return tool_use(id, "Bash", command=command)


async def until(condition: Callable[[], object], timeout: float = 30) -> None:
    """Wait until ``condition`` holds, and fail if it does not within ``timeout`` seconds."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not condition():
        if loop.time() > deadline:
            raise TimeoutError(f"Waited {timeout} seconds for {condition}")
        await asyncio.sleep(0.05)


@contextlib.asynccontextmanager
async def start_workflow(
    client: Client,
    workflow_type: type,
    *args: Any,
    interceptors: Sequence[Interceptor] = (),
) -> AsyncIterator[WorkflowHandle[Any, Any]]:
    """Start ``workflow_type`` on a Worker of its own, with the plugin.

    Once the block ends, replays the Workflow's history, which fails if it
    is not deterministic.
    """
    task_queue = str(uuid.uuid4())
    plugin = ClaudeAgentSDKPlugin()
    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[workflow_type],
        plugins=[plugin],
        interceptors=interceptors,
    ):
        handle = await client.start_workflow(
            workflow_type.__name__, args=args, id=task_queue, task_queue=task_queue
        )
        yield handle
    replayer = Replayer(workflows=[workflow_type], plugins=[plugin])
    await replayer.replay_workflow(await handle.fetch_history())
