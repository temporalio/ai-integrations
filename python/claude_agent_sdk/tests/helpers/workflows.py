"""The Workflows the tests chat through, and how they describe what the Claude Agent SDK gets."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import hashlib
import json
import uuid
from collections.abc import AsyncIterator, Callable, Sequence
from typing import Any, Literal

import claude_agent_sdk as sdk
from claude_agent_sdk.types import HookEvent

from temporalio import workflow
from temporalio.claude_agent_sdk import (
    ClaudeAgentSDKPlugin,
    SandboxFactory,
    TemporalTransport,
)
from temporalio.client import Client, WorkflowHandle
from temporalio.contrib.workflow_streams import WorkflowStream
from temporalio.worker import Interceptor, Replayer, Worker

# The stream topic a ChatWorkflow's partial messages go to.
PARTIAL = "partial"


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
    allowed_tools: list[str] = dataclasses.field(default_factory=list[str])
    deny_interrupt: bool = False
    """Deny with ``interrupt=True``."""
    tools: list[str] | None = None
    control: (
        Literal["set_model", "set_permission_mode", "mcp_status", "mcp_toggle"] | None
    ) = None
    """A control request to send once connected."""
    hooks: list[Hook] = dataclasses.field(default_factory=list[Hook])
    sdk_tools: bool = False
    """Serve ``mcp__app__echo`` from a Claude Agent SDK MCP server in the Workflow."""
    partial: bool = False
    """Ask for partial messages, which go to the Workflow's stream topic :data:`PARTIAL`."""
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
        self.stream = WorkflowStream()
        self.client: sdk.ClaudeSDKClient | None = None
        self.transport: TemporalTransport | None = None
        self.hook_inputs: list[dict[str, Any]] = []
        # How many prompts the say update sent.
        self.said = 0

    @workflow.run
    async def run(self, chat: Chat) -> list[Any]:
        hooks: dict[HookEvent, list[sdk.HookMatcher]] = {}
        for hook in chat.hooks:
            matcher = sdk.HookMatcher(hook.matcher, [self.callback(hook)], hook.timeout)
            hooks.setdefault(hook.event, []).append(matcher)
        options = sdk.ClaudeAgentOptions(
            can_use_tool=self.approve,
            permission_mode="default",
            allowed_tools=chat.allowed_tools,
            tools=chat.tools,
            hooks=hooks,
            mcp_servers={"app": self.app_server()} if chat.sdk_tools else {},
            include_partial_messages=chat.partial,
        )
        transport = TemporalTransport(
            options,
            sandbox="local",
            stream=self.stream.topic(PARTIAL) if chat.partial else None,
        )
        self.transport = transport
        async with sdk.ClaudeSDKClient(options=options, transport=transport) as client:
            self.client = client
            await self.send_control(client)
            for prompt in chat.prompts:
                await client.query(prompt)
                await self.receive(client)
            received = 0
            while received < self.said + chat.follow_ups:
                await self.receive(client)
                received += 1
        return self.messages

    async def send_control(self, client: sdk.ClaudeSDKClient) -> None:
        """Send the chat's control request, and record the error an MCP one fails with."""
        control = self.chat.control
        if control == "set_model":
            await client.set_model("haiku")
        elif control == "set_permission_mode":
            await client.set_permission_mode("acceptEdits")
        elif control is not None:
            try:
                if control == "mcp_status":
                    await client.get_mcp_status()
                else:
                    await client.toggle_mcp_server("none", False)
            except Exception as error:
                self.messages.append(["Error", str(error)])

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
    async def say(self, prompt: str) -> None:
        """Send a prompt, whose turn the Workflow reads after the chat's."""
        assert self.client is not None
        await self.client.query(prompt)
        self.said += 1

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

    @workflow.query
    def received(self) -> list[Any]:
        """What the Workflow described so far."""
        return self.messages


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
        transport = TemporalTransport(options, sandbox="local")
        messages: list[Any] = []
        async for message in sdk.query(
            prompt=prompt, options=options, transport=transport
        ):
            if not isinstance(message, sdk.SystemMessage):
                messages.append(describe(message))
        return messages


@workflow.defn
class EndInputWorkflow:
    """Sends the prompts and ends the input, then reads to the end, and closes when told."""

    def __init__(self) -> None:
        self.closing = False

    @workflow.run
    async def run(self, prompts: list[str]) -> list[str]:
        transport = TemporalTransport(sdk.ClaudeAgentOptions(), sandbox="local")
        await transport.connect()
        for prompt in prompts:
            message = {"type": "user", "message": {"role": "user", "content": prompt}}
            await transport.write(json.dumps(message) + "\n")
        await transport.end_input()
        kinds = [
            message["type"]
            async for message in transport.read_messages()
            if message["type"] in ("assistant", "user", "result")
        ]
        await workflow.wait_condition(lambda: self.closing)
        await transport.close()
        return kinds

    @workflow.signal
    def close(self) -> None:
        self.closing = True


@workflow.defn
class ReadToEndWorkflow:
    """Sends a prompt, then returns the kind of each message the transport has, up to its end."""

    @workflow.run
    async def run(self, prompt: str) -> list[str]:
        transport = TemporalTransport(sdk.ClaudeAgentOptions(), sandbox="local")
        await transport.connect()
        message = {"type": "user", "message": {"role": "user", "content": prompt}}
        await transport.write(json.dumps(message) + "\n")
        return [message["type"] async for message in transport.read_messages()]


@workflow.defn
class CloseFirstWorkflow:
    """Sends a prompt, closes the transport once its turn's result came, then reads what came."""

    @workflow.run
    async def run(self, prompt: str) -> list[str]:
        transport = TemporalTransport(sdk.ClaudeAgentOptions(), sandbox="local")
        await transport.connect()
        message = {"type": "user", "message": {"role": "user", "content": prompt}}
        await transport.write(json.dumps(message) + "\n")
        await workflow.wait_condition(
            lambda: any(m["type"] == "result" for m in transport._inbox)
        )
        await transport.close()
        return [
            message["type"]
            async for message in transport.read_messages()
            if message["type"] in ("assistant", "result")
        ]


def describe(message: sdk.Message) -> list[Any]:
    """The message's type and the fields of each content block."""
    if isinstance(message, sdk.ResultMessage):
        if message.is_error:
            return ["Result", message.result, message.subtype]
        return ["Result", short(message.result)]
    blocks = getattr(message, "content", [])
    return [
        type(message).__name__,
        *[[short(field) for field in dataclasses.astuple(b)] for b in blocks],
    ]


def short(value: Any) -> Any:
    """A text too big for a Workflow's result as its length and hash, or the value."""
    if isinstance(value, str) and len(value) > 100_000:
        return f"{len(value)} characters, sha256 {hashlib.sha256(value.encode()).hexdigest()}"
    return value


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
    factory: SandboxFactory,
    workflow_type: type,
    *args: Any,
    interceptors: Sequence[Interceptor] = (),
) -> AsyncIterator[WorkflowHandle[Any, Any]]:
    """Start ``workflow_type`` on a Worker of its own, with ``factory`` as the sandbox ``local``.

    Once the block ends, replays the Workflow's history, which fails if it
    is not deterministic.
    """
    task_queue = str(uuid.uuid4())
    plugin = ClaudeAgentSDKPlugin({"local": factory})
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
