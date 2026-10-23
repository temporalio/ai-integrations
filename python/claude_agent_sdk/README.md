# Claude Agent SDK Temporal Integration

> Release stage: [Pre-release](https://docs.temporal.io/develop/python/integrations/claude-agent-sdk).

Run Claude Code agents as Temporal Workflows. Claude Code runs in a sandbox, and if the Worker,
Claude Code, or the sandbox fails, it picks up the conversation where it left off.

## Install

```bash
uv add temporalio-claude-agent-sdk
```

## Benefits of Temporal to the Claude Agent SDK

Temporal provides durable orchestration for Claude agents:

- **Bring your own sandbox.** Run Claude Code in any sandbox service by implementing one
  `Sandbox` interface, and register as many as you need by name. 
- **Durable approvals and hooks.** An agent can pause for a person's approval or your own
  checks and pick up where it left off, however long the wait.
- **Durable tools.** Custom tools you define get Temporal's retries and timeouts, and a crash
  doesn't lose their work. Claude Code's built-in tools run in the sandbox.
- **Operational visibility with Temporal UI.** See every message, tool call, approval, and
  retry in the UI included with Temporal.

## What's Included

- **`TemporalTransport`**: Connects your Workflow to Claude Code running in a sandbox. It starts
  the sandbox and the CLI, and carries messages between them and your Workflow.
- **`ClaudeAgentSDKPlugin`**: Add it to your Worker with a `SandboxFactory` for each of your
  [sandboxes](#sandboxes), each under a name. A factory builds a `Sandbox` from a `SandboxContext`.
- **`Sandbox`** and **`SandboxError`**: What you implement to run Claude Code in any sandbox
  service.

## How It Works

Claude Code runs whole, as you'd run it yourself, inside a sandbox, so it keeps all its built-in
tools and works in the sandbox's files. Your Workflow drives it through the Claude Agent SDK's
usual `ClaudeSDKClient`, and Temporal retries each step that fails.

- **Activities move the messages.** Your messages go to Claude Code through Activities. A
  long-running Activity follows everything Claude Code writes and sends it to your Workflow as
  Signals.
- **The sandbox stays isolated.** The sandbox runs only Claude Code and never connects to
  Temporal.

## Usage

### Basic Setup

#### Workflow Side

Use the Claude Agent SDK's own `ClaudeSDKClient` in the Workflow, with a `TemporalTransport`.
Pass the transport your sandbox's registered name and the same `options` as the client.

```python
from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient, ResultMessage
from temporalio import workflow
from temporalio.claude_agent_sdk import TemporalTransport


@workflow.defn
class ReviewWorkflow:
    @workflow.run
    async def run(self) -> str:
        options = ClaudeAgentOptions()
        transport = TemporalTransport(options, sandbox="my-sandbox")
        async with ClaudeSDKClient(options=options, transport=transport) as client:
            await client.query("Review utils.py for bugs that would cause crashes.")
            async for message in client.receive_response():
                if isinstance(message, ResultMessage):
                    return message.result or ""
        return ""
```

#### Worker Side

Register a factory that builds [your sandbox](#sandboxes) with `ClaudeAgentSDKPlugin`, under the
name passed to `TemporalTransport`. Add the plugin and Workflow to the Worker.

```python
from temporalio.claude_agent_sdk import ClaudeAgentSDKPlugin
from temporalio.worker import Worker


worker = Worker(
    client,
    task_queue="agents",
    workflows=[ReviewWorkflow],
    plugins=[ClaudeAgentSDKPlugin({"my-sandbox": lambda context: MySandbox()})],
)
```

### Secrets

Give secrets to your sandbox on the Worker, so they stay out of Workflow history, or point Claude
Code at a credential proxy with `ANTHROPIC_BASE_URL`.

Don't put secrets in `ClaudeAgentOptions.env`: it is recorded in Workflow history.

## Sandboxes

Claude Code runs in a sandbox you choose. Connect any sandbox service by implementing `Sandbox`.

> [!WARNING]
> Any Workflow run can connect to a session, including one another run is still using, such as
> after a reset. The session's files can then store duplicate work from both runs.

### Your Own Sandbox

Implement five methods, and the transport calls them as it needs:

- **`start`** starts the session's sandbox.
- **`exec`** and **`stream`** run commands in the sandbox. Raise `SandboxError` when you can't
  connect to the sandbox.
- **`write_files`** copies files from the Worker into the sandbox, such as Claude Code's input
  and the plugins in `options.plugins`. Each file has its content and mode. Copy them however
  your sandbox service works best, as long as each file appears whole. For example, write it
  under a temporary name, then rename it.
- **`terminate`** ends the sandbox.

The sandbox needs `sh`, `cat`, `mkdir`, `mv`, `rm`, `sleep`, `tail`, `wc`, and
Claude Code.

Register a factory function with `ClaudeAgentSDKPlugin` under a name. The factory takes a
`SandboxContext` and returns a `Sandbox`. The context holds the sandbox name and the Workflow's
namespace, ID, run ID, and first run ID, and you decide how to key the sandbox from them. The
first run ID stays the same through a reset and continue-as-new, so it's a good key for
resuming. The factory runs on every Activity attempt, so if a sandbox is lost, the retry gets a
new one. You can also use the context to pick sandbox sizes per Workflow:

```python
from temporalio.claude_agent_sdk import ClaudeAgentSDKPlugin, Sandbox, SandboxContext


def sandbox_for(context: SandboxContext) -> Sandbox:
    if context.workflow_id.startswith("big-"):
        return MySandbox(context.first_execution_run_id, cpus=8)
    return MySandbox(context.first_execution_run_id)

plugin = ClaudeAgentSDKPlugin({"my-sandbox": sandbox_for})
```

## Callbacks

Approvals, custom tools, and hooks run in your Workflow, so they can use everything a Workflow
can: wait days for a person's input through a Signal or Update, execute Activities with retries
and timeouts, or set a timer. Because they run in the Workflow, callbacks must
be deterministic.

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
        transport = TemporalTransport(options, sandbox="my-sandbox")
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
        transport = TemporalTransport(options, sandbox="my-sandbox")
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

Each message Claude Code writes reaches the Workflow as one Signal, as soon as it's written.

### Token-by-Token Output

To stream Claude's output as it's generated, set `include_partial_messages` and pass the
transport a topic of the Workflow's stream. Each `StreamEvent` goes to that topic for callers
outside the Workflow: the Workflow itself doesn't get them, but still gets every complete message.

```python
@workflow.defn
class StreamingWorkflow:
    @workflow.init
    def __init__(self, prompt: str) -> None:
        self.events = WorkflowStream().topic("events")

    @workflow.run
    async def run(self, prompt: str) -> str:
        options = ClaudeAgentOptions(include_partial_messages=True)
        transport = TemporalTransport(options, sandbox="my-sandbox", stream=self.events)
        async with ClaudeSDKClient(options=options, transport=transport) as client:
            await client.query(prompt)
            async for message in client.receive_response():
                if isinstance(message, ResultMessage):
                    return message.result or ""
        return ""
```

Callers subscribe to `events` as above; each item's data is a `stream_event` message. If a Worker
fails, a caller may see a few events again. `include_partial_messages` and `stream` go together:
passing one without the other raises `ValueError`.

## Hardening

By default, the transport stops Claude Code's file tools (Edit, Write) from changing the session
transcripts that resuming a session reads. Bash isn't covered. To protect the transcripts from Bash
too, turn on Claude Code's Bash sandbox. It's off by default because it changes how Bash works:

- Bash can write only to the workspace and the temp directory.
- Connecting to a host not in `sandbox.network.allowedDomains` asks `can_use_tool`.
- Sandboxed Bash commands are approved without asking, unless you set
  `autoAllowBashIfSandboxed` to `false`.
- Rules in `disallowed_tools` stop applying to Bash; put your rules in `settings` instead.

```python
import json

from claude_agent_sdk import ClaudeAgentOptions

options = ClaudeAgentOptions(
    settings=json.dumps(
        {
            "sandbox": {
                "enabled": True,
                # No retrying a command outside the sandbox with dangerouslyDisableSandbox.
                "allowUnsandboxedCommands": False,
                # Fail instead of running Bash unsandboxed when the sandbox can't start.
                "failIfUnavailable": True,
            }
        }
    ),
)
```

Pass `settings` as a JSON string; the transport can't combine `ClaudeAgentOptions.sandbox` with a
settings file.
