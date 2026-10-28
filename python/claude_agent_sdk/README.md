# Claude Agent SDK Temporal Integration

> Release stage: [Pre-release](https://docs.temporal.io/develop/python/integrations/claude-agent-sdk).

Run Claude Code agents as Temporal Workflows. As in the Claude Agent SDK, a transport connects
your code to Claude Code, and Temporal Activities carry its input and output.

## Install

```bash
uv add temporalio-claude-agent-sdk
```

## Benefits of Temporal to the Claude Agent SDK

Temporal provides durable orchestration for Claude agents:

- **Durable approvals and hooks.** An agent can pause for a person's approval or your own
  checks and pick up where it left off, however long the wait.
- **Durable tools.** Custom tools you define run as Workflow code and can execute Activities, with
  Temporal's retries and timeouts. A crash doesn't lose their work.
- **Operational visibility with Temporal UI.** See every message, tool call, approval, and
  retry in the UI included with Temporal.

## What's Included

- **`ClaudeAgentSDKPlugin`**: Add it to your Worker to use the Claude Agent SDK in your Workflows.
- **`TemporalTransport`**: The transport for the Claude Agent SDK in Workflows. It launches Claude
  Code and carries its input and output through the plugin's Activities.
- **`SubprocessConnector`**: The plugin's default connector, which runs Claude Code in a
  subprocess of the Worker. See [Subprocess Connector](#subprocess-connector).
- **`Connector`** and **`ClaudeCodeProcess`**: Implement these to run Claude Code somewhere else, such as
  in a container. See [Custom Connector](#custom-connector).

## How It Works

Claude Code runs whole, as in the Claude Agent SDK, so it keeps all its built-in tools. Your
Workflow drives it through the Claude Agent SDK's usual `ClaudeSDKClient`, and Temporal retries each
step that fails.

- **Activities move the messages.** The transport sends your messages to Claude Code through
  Activities, and always has one Activity waiting for what Claude Code writes next.
- **The connector decides where Claude Code runs.** By default, it runs on the Worker. Your own
  connector can run it anywhere it can reach.

## Usage

### Basic Setup

#### Workflow Side

Use the Claude Agent SDK in the Workflow as you would without Temporal, and pass it a Temporal
`transport`. Pass the transport the same `options` as the client.

```python
from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient, ResultMessage
from temporalio import workflow
from temporalio.claude_agent_sdk import TemporalTransport


@workflow.defn
class ReviewWorkflow:
    @workflow.run
    async def run(self) -> str:
        options = ClaudeAgentOptions()
        transport = TemporalTransport(options)
        async with ClaudeSDKClient(options=options, transport=transport) as client:
            await client.query("Review utils.py for bugs that would cause crashes.")
            async for message in client.receive_response():
                if isinstance(message, ResultMessage):
                    return message.result or ""
        return ""
```

#### Worker Side

Add `ClaudeAgentSDKPlugin` and the Workflow to the Worker.

```python
from temporalio.claude_agent_sdk import ClaudeAgentSDKPlugin
from temporalio.worker import Worker


worker = Worker(
    client,
    task_queue="agents",
    workflows=[ReviewWorkflow],
    plugins=[ClaudeAgentSDKPlugin()],
)
```

## Transports

Always pass the Claude Agent SDK a `transport` in a Workflow. Without one, the Claude
Agent SDK would start Claude Code from Workflow code.

Pass the transport the same `options` you pass the client. The transport uses them to start and
run Claude Code, and the client uses them for the conversation.

### Subprocess Connector

By default, the plugin uses `SubprocessConnector`, which runs Claude Code with the Claude Agent
SDK's own `SubprocessCLITransport` in the Worker process:

- As in the Claude Agent SDK, it finds Claude Code, checks its version, and runs it as the Worker's
  user, with the Worker's environment, in the Worker's files. Paths in `ClaudeAgentOptions`, such
  as `cwd` and `cli_path`, are on the Worker.
- Claude Code stops when the Worker process does. The transport then raises `CLIConnectionError`.
  To continue the conversation, start a new client with `options.resume` set to the session ID.
- The plugin routes every Activity after the launch to the Worker process that runs Claude Code.

`TemporalTransport` takes options for each of its Activities, anything `workflow.execute_activity`
takes, such as a task queue, timeouts, or a retry policy:

```python
from temporalio.claude_agent_sdk import TemporalTransport

transport = TemporalTransport(
    options, activity_options={"connect": {"task_queue": "claude"}}
)
async with ClaudeSDKClient(options=options, transport=transport) as client:
    ...
```

### Custom Connector

To run Claude Code somewhere other than the Worker, such as in a container or on a remote host,
give the plugin your own `Connector`. The transport and its Activities stay the same; you don't
write any Workflow code or Activities.

A connector has one method, `connect`. It gets a `ConnectActivityInput`, starts Claude Code with its
`args` and `env`, and returns a `ClaudeCodeProcess`. The plugin keeps that process in the Worker
process that started it and runs every Activity after the launch there, as with a pipe.

A `ClaudeCodeProcess` has `write`, `end_input`, `read`, and `close`. `read()` waits for Claude Code's next
output, returns it with any more that is already there, and returns an empty list once Claude Code
has exited with status 0. Raise the
Claude Agent SDK's errors, such as `ProcessError` or `CLINotFoundError`, only for failures no retry
can fix, and `CLIConnectionError` only once Claude Code is surely gone. Raise any other error, such
as a dropped network call, to have it retried. Activities can be retried and can overlap, so make
`write`, `end_input` and `close` safe to call more than once. The plugin calls `read` one at a
time, and keeps what it returns until the Workflow confirms it got it, so a lost Activity result
loses no output.

Make Claude Code stop when the Worker that started it stops, such as when its connection drops.
Otherwise it keeps running, spending tokens and running tools, and can write to the session after
the user resumes it.

`close` stops Claude Code and everything it started. Whether it also deletes the host is up to you:
a connector that creates its own sandbox per launch can delete it there, while one that runs Claude
Code on compute something else owns, such as an orchestrator's, stops only the process and leaves
deleting the compute to its owner.

If the launch is cancelled while `connect` runs, the plugin waits for it and closes the process it
returns, since the Workflow never gets a handle to close it. A launch can also be retried on another
Worker, such as when the first Worker went away after starting Claude Code. To find what the first
attempt created, name it after the launch, from `activity.info()`'s `workflow_run_id` and
`activity_id`, and reuse it if it exists.

```python
from temporalio.claude_agent_sdk import ClaudeCodeProcess, ConnectActivityInput


class RemoteConnector:
    async def connect(self, request: ConnectActivityInput) -> ClaudeCodeProcess:
        box = await hosts.create()
        process = await box.start([request.cli_path or "claude", *request.args], env=request.env, cwd=request.cwd)
        return RemoteConnection(box, process)


# On the Worker
ClaudeAgentSDKPlugin(connector=RemoteConnector())
```

Paths in `ClaudeAgentOptions`, such as `cwd` and plugin directories, are on Claude Code's host,
as they are for the Claude Agent SDK. Bake the files into its image or mount them, or copy them
there in `connect`; getting them onto the host is the connector's job, not the transport's. That
includes plugin directories with their scripts and references. Pass `settings` as a JSON string,
not a file path.

### When Claude Code Stops

As in the Claude Agent SDK, the transport raises when Claude Code stops: `ProcessError` if it
exited with an error, and `CLIConnectionError` if it's gone, such as when the Worker connected to
it went away. The transport never starts another Claude Code by itself.

To continue the conversation, choose the session ID in the Workflow, and start a new client with
`options.resume` set to it. With `CLAUDE_CODE_RESUME_INTERRUPTED_TURN=1`, the new Claude Code
re-runs the turn that was cut off, so the new client only reads its reply:

```python
import dataclasses

from claude_agent_sdk import CLIConnectionError

session_id = str(workflow.uuid4())
options = ClaudeAgentOptions(
    session_id=session_id, env={"CLAUDE_CODE_RESUME_INTERRUPTED_TURN": "1"}
)
while True:
    try:
        async with ClaudeSDKClient(options=options, transport=TemporalTransport(options)) as client:
            if options.resume is None:
                await client.query(prompt)
            async for message in client.receive_response():
                ...
        break
    except CLIConnectionError:
        options = dataclasses.replace(options, session_id=None, resume=session_id)
```

`workflow.uuid4()` is the same on replay, and the session has its ID before Claude Code writes
anything, so the Workflow can resume it even mid-turn. The new Claude Code reads the session from
its config directory, so it must run where that directory is, such as on a volume every Worker
mounts.

When a Worker shuts down, the plugin stops each Claude Code that Worker is connected to, as SIGTERM does.
Claude Code then leaves a turn in progress, including a pending tool approval, for the resumed
session to continue, so a person is asked again. Process managers stop a Worker with SIGTERM, but
Python exits on SIGTERM without shutting the Worker down, so turn it into a shutdown:

```python
stop = asyncio.Event()
asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, stop.set)
async with Worker(client, task_queue="my-task-queue", workflows=[MyWorkflow], plugins=[ClaudeAgentSDKPlugin()]):
    await stop.wait()
```

### Long Conversations

Claude Code's output reaches the Workflow as Activity results, so every turn adds to Workflow
history. A long conversation eventually fills it. When `workflow.info().is_continue_as_new_suggested()`
is true, finish the current turn, disconnect, and continue as new with the session ID. The next run
resumes the session, as above:

```python
await workflow.wait_condition(
    lambda: workflow.info().is_continue_as_new_suggested()
    and workflow.all_handlers_finished()
)
await client.disconnect()
workflow.continue_as_new(session_id)

# In the next run
options = ClaudeAgentOptions(resume=session_id)
```

Wait for `workflow.all_handlers_finished()` first, so no message in flight loses its reply.

Prompts and Claude Code's output, such as tool results and images, are Temporal payloads, so each
must fit within the server's payload size limit. For larger ones, configure
[external storage](https://github.com/temporalio/sdk-python#external-storage) on the client's data
converter. Reading a large image or PDF also needs a higher `max_buffer_size` in
`ClaudeAgentOptions`: as without Temporal, the Claude Agent SDK rejects any message over 1 MB by
default.

### Secrets

`ClaudeAgentOptions.env` reaches Claude Code as Activity input, so it is recorded in Workflow
history. Don't put secrets in it.

Set secrets such as `ANTHROPIC_API_KEY` where Claude Code runs instead. By default, Claude Code
inherits the Worker's environment, which never passes through an Activity.

## Callbacks

Approvals, custom tools, and hooks run in your Workflow, so they can use everything a Workflow
can: wait days for a person's input through a Signal or Update, execute Activities with retries
and timeouts, or set a timer. Like all Workflow code, they must be deterministic.

### Tool Approval

Set `can_use_tool` on `ClaudeAgentOptions`, and Claude Code asks it before each tool call, except
tools you pre-approve with `allowed_tools`. In this example, a reviewer lists pending calls with a
Query and decides each with an Update:

```python
from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKClient,
    PermissionResult,
    PermissionResultAllow,
    PermissionResultDeny,
    ToolPermissionContext,
)
from temporalio import workflow
from temporalio.claude_agent_sdk import TemporalTransport


@workflow.defn
class GovernedAgentWorkflow:
    def __init__(self) -> None:
        self.pending: dict[str, str] = {}
        self.decisions: dict[str, bool] = {}

    @workflow.run
    async def run(self, prompt: str) -> None:
        options = ClaudeAgentOptions(
            tools=["Bash", "Read", "Glob", "Grep"],
            allowed_tools=["Read", "Glob", "Grep"],
            can_use_tool=self.approve,
        )
        transport = TemporalTransport(options)
        async with ClaudeSDKClient(options=options, transport=transport) as client:
            await client.query(prompt)
            async for _ in client.receive_response():
                pass

    async def approve(
        self, tool_name: str, tool_input: dict[str, object], context: ToolPermissionContext
    ) -> PermissionResult:
        tool_use_id = context.tool_use_id or ""
        self.pending[tool_use_id] = tool_name
        await workflow.wait_condition(lambda: tool_use_id in self.decisions)
        del self.pending[tool_use_id]
        if self.decisions[tool_use_id]:
            return PermissionResultAllow()
        return PermissionResultDeny(message=f"{tool_name} was denied")

    @workflow.query
    def pending_tools(self) -> dict[str, str]:
        return self.pending

    @workflow.update
    def decide_tool(self, tool_use_id: str, approved: bool) -> None:
        self.decisions[tool_use_id] = approved
```

### Custom Tools

Pass tools you define with `@tool` and `create_sdk_mcp_server` in `mcp_servers`, and the model
calls them like any other tool.

```python
from datetime import timedelta

from claude_agent_sdk import ClaudeAgentOptions, create_sdk_mcp_server, tool
from temporalio import workflow


@tool("lookup_order", "Look up an order by ID", {"order_id": str})
async def lookup_order(args):
    order = await workflow.execute_activity(
        fetch_order, args["order_id"], start_to_close_timeout=timedelta(seconds=30)
    )
    return {"content": [{"type": "text", "text": order}]}


options = ClaudeAgentOptions(
    mcp_servers={"orders": create_sdk_mcp_server("orders", tools=[lookup_order])},
    allowed_tools=["mcp__orders__lookup_order"],
)
```

### Hooks

Set hook callbacks in `ClaudeAgentOptions.hooks`, for any hook event, and Claude Code calls them
and applies their output.

## Progress

To show callers what the agent is doing while a turn runs, publish its messages to a
[Workflow Stream](https://docs.temporal.io/develop/python/workflows/workflow-streams)
(experimental). Callers subscribe and get each update as it's published:

```python
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
)
from temporalio import workflow
from temporalio.claude_agent_sdk import TemporalTransport
from temporalio.contrib.workflow_streams import WorkflowStream


@workflow.defn
class ProgressWorkflow:
    @workflow.init
    def __init__(self, prompt: str) -> None:
        self.progress = WorkflowStream().topic("progress", type=str)

    @workflow.run
    async def run(self, prompt: str) -> str:
        options = ClaudeAgentOptions()
        transport = TemporalTransport(options)
        async with ClaudeSDKClient(options=options, transport=transport) as client:
            await client.query(prompt)
            async for message in client.receive_response():
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            self.progress.publish(block.text)
                        elif isinstance(block, ToolUseBlock):
                            self.progress.publish(f"Running {block.name}")
                elif isinstance(message, ResultMessage):
                    return message.result or ""
        return ""
```

A caller subscribes with:

```python
from temporalio.contrib.workflow_streams import WorkflowStreamClient

stream = WorkflowStreamClient.create(client, workflow_id)
async for item in stream.subscribe(["progress"], result_type=str):
    print(item.data)
```

Claude Code's messages reach the Workflow as soon as they're written, in the results of the transport's read Activities.

### Token-by-Token Output

To stream Claude's output as it's generated, set `include_partial_messages`, and pass the
transport a `stream` topic of the Workflow's stream. Each `StreamEvent` goes
to that topic for callers outside the Workflow: the Workflow itself doesn't get them, but still
gets every complete message.

```python
from temporalio.claude_agent_sdk import TemporalTransport


@workflow.defn
class StreamingWorkflow:
    @workflow.init
    def __init__(self, prompt: str) -> None:
        self.events = WorkflowStream().topic("events")

    @workflow.run
    async def run(self, prompt: str) -> str:
        options = ClaudeAgentOptions(include_partial_messages=True)
        transport = TemporalTransport(
            options, stream=self.events
        )
        async with ClaudeSDKClient(options=options, transport=transport) as client:
            await client.query(prompt)
            async for message in client.receive_response():
                if isinstance(message, ResultMessage):
                    return message.result or ""
        return ""
```

Callers subscribe to `events` as above; each item's data is a `stream_event` message. `include_partial_messages` and `stream` go together:
passing one without the other raises `ValueError`.

If the Workflow resumes a turn that was cut off (see [When Claude Code Stops](#when-claude-code-stops)),
the new Claude Code generates that reply again, so callers see the partial first attempt followed by
the full retry.

## Observability

To trace a Workflow and the Claude Code it runs as one trace, register
[`OpenTelemetryPlugin`](https://github.com/temporalio/sdk-python/tree/main/temporalio/contrib/opentelemetry)
on the client, and turn on Claude Code's
[tracing](https://code.claude.com/docs/en/agent-sdk/observability) where it runs:

```python
import opentelemetry.trace
from temporalio.contrib.opentelemetry import OpenTelemetryPlugin, create_tracer_provider

opentelemetry.trace.set_tracer_provider(create_tracer_provider())
client = await Client.connect("localhost:7233", plugins=[OpenTelemetryPlugin()])
```

```shell
export CLAUDE_CODE_ENABLE_TELEMETRY=1
export CLAUDE_CODE_ENHANCED_TELEMETRY_BETA=1
export OTEL_TRACES_EXPORTER=otlp
export OTEL_EXPORTER_OTLP_PROTOCOL=grpc
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
```

As in the Claude Agent SDK, `SubprocessConnector`
[passes Claude Code the active trace context](https://code.claude.com/docs/en/agent-sdk/observability#link-traces-to-your-application)
in `TRACEPARENT` and `TRACESTATE` when it starts it. Claude Code reads them then, so the spans of
every turn, such as `claude_code.interaction`, `claude_code.llm_request` and `claude_code.tool`,
are children of the connect Activity's span. A custom connector can do the same with
`opentelemetry.propagate.inject`.

Set these variables in the Worker's environment rather than in `options.env`, which is recorded in
Workflow history, especially `OTEL_EXPORTER_OTLP_HEADERS` if it holds an API key.

## Hardening

Consider these additional protections when running Claude Code unattended.

### Restrict access outside the workspace

Configure Claude Code's permission rules and Bash sandbox to restrict access outside the workspace.
Claude's [permissions](https://code.claude.com/docs/en/permissions) control its built-in file tools,
while [sandboxing](https://code.claude.com/docs/en/sandboxing) restricts Bash commands and the
processes they start.

### Run Claude Code outside the Worker

Claude Code can do anything the Worker's user can on its host. For stronger isolation, run Claude
Code in a separate container, VM, or remote environment. Use a
[custom connector](#custom-connector) that reaches that environment instead of launching Claude
Code on the Worker.
