# Temporal OpenAI Codex integration

**Pre-release.** Run the [Python Codex SDK](https://github.com/openai/codex/tree/main/sdk/python)
in Temporal Workflows. Install with
`uv add temporalio-openai-codex`. Import the integration as
`temporalio.openai_codex` and the upstream SDK as `openai_codex`.

Each SDK keeps its exact `openai-codex-cli-bin` dependency. The
default connector uses that bundled runtime and its companion PATH entries,
including when another Codex executable is installed on the Worker.

## Use AsyncCodex

Register the plugin on the Worker (or its Client):

```python
from temporalio.openai_codex import OpenAICodexPlugin
from temporalio.worker import Worker

worker = Worker(
    client,
    task_queue="codex",
    workflows=[CodingWorkflow],
    plugins=[OpenAICodexPlugin()],
)
```

Use the upstream async API in the Workflow:

```python
from openai_codex import AsyncCodex, CodexConfig, Sandbox
from temporalio import workflow


@workflow.defn
class CodingWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        async with AsyncCodex(CodexConfig(cwd="/workspace")) as codex:
            thread = await codex.thread_start(sandbox=Sandbox.read_only)
            result = await thread.run(prompt)
            return result.final_response or ""
```

The plugin configures passthrough imports for `openai_codex` and
`temporalio.openai_codex`, so Workflow code can import both directly.

The plugin adapts construction and asynchronous turn streams inside Workflows,
including replay. Threads, turn handles, login handles, notifications, and
`TurnResult` are the upstream types. Ordinary SDK calls outside Workflows keep
using upstream implementations. Synchronous `Codex` is for Activity or application
code, not Workflow code.

In Workflows, collected failed turns raise `CodexError` with the upstream error
message, so an uncaught provider failure fails the Workflow. Streaming still yields
the upstream turn-completion notification, including its failure status and error.

## Approvals and Activity options

Pass `TemporalCodexConfig`, a subclass of the upstream `CodexConfig`, to
`AsyncCodex` for Workflow-specific options:

```python
from datetime import timedelta
from openai_codex import AsyncCodex
from temporalio.openai_codex import CodexActivityOptions, TemporalCodexConfig
from temporalio.workflow import ActivityConfig

async def approve(method, params):
    # Runs in Workflow code. It can wait for a Signal or Update, or call an Activity.
    return {"decision": "accept"}

async with AsyncCodex(TemporalCodexConfig(
    cwd="/workspace",
    approval_handler=approve,
    activity_options=CodexActivityOptions(
        connect=ActivityConfig(start_to_close_timeout=timedelta(minutes=2)),
    ),
)) as codex:
    thread = await codex.thread_start()
    result = await thread.run("Describe the project.")
```

Synchronous and asynchronous approval handlers receive `(method, params)` as in
upstream `CodexClient`. Keep callbacks deterministic; put I/O in Activities. Without
a handler, command and file-change approval requests receive `{"decision": "accept"}`
and other server requests receive `{}`, matching the SDK default. Set Codex's
approval and sandbox policies deliberately for your application. These additional
options apply inside Workflows; ordinary SDK calls use upstream behavior.

Only the base `CodexConfig` fields are passed to the connect Activity. Approval
callbacks, Activity options, and stream topic handles stay in Workflow code.

## Token-by-Token Output

Use the optional `stream` topic to expose generated output to callers outside the
Workflow. [Workflow Streams](https://docs.temporal.io/develop/python/workflows/workflow-streams)
is currently in Public Preview. Create the stream in `@workflow.init`, then pass
its topic in `TemporalCodexConfig`:

```python
from openai_codex import AsyncCodex
from temporalio import workflow
from temporalio.contrib.workflow_streams import WorkflowStream
from temporalio.openai_codex import TemporalCodexConfig


@workflow.defn
class StreamingWorkflow:
    @workflow.init
    def __init__(self, prompt: str) -> None:
        self.events = WorkflowStream().topic("codex")

    @workflow.run
    async def run(self, prompt: str) -> str:
        async with AsyncCodex(TemporalCodexConfig(stream=self.events)) as codex:
            thread = await codex.thread_start()
            result = await thread.run(prompt)
            return result.final_response or ""
```

An external caller subscribes using the Workflow ID:

```python
from temporalio.contrib.workflow_streams import WorkflowStreamClient

stream = WorkflowStreamClient.create(client, workflow_id)
async for item in stream.subscribe(["codex"], result_type=dict):
    notification = item.data
    print(notification["method"], notification["params"])
```

Each item is a JSON-RPC delta notification with `method` and `params`, including
thread, turn, and item IDs when provided by Codex. Notifications whose method ends
in `delta` (case insensitive) are published once by the connection's reader,
including agent text, reasoning, and command output. They remain available through
upstream `turn.stream()` as well. Activity retries and multiple turn subscribers
do not duplicate these publications. Explicitly repeating a turn can generate
new output; consumers should use the IDs to associate it with the correct turn.
Streaming adds Workflow memory use; apply the Workflow Streams retention guidance
for long conversations.

## Connectors and launch configuration

`OpenAICodexPlugin(connector=...)` accepts a `Connector` implementing
`async connect(ConnectActivityInput) -> CodexProcess`. The input contains the upstream
`CodexConfig`. `CodexProcess` provides `write(message)`, `read() -> list[dict]`, and
`close()`. A read returns an ordered batch or an empty list at EOF. Close must be
idempotent, tolerate overlap, support retries after cancellation, and unblock
pending reads and writes. A transient
write failure must guarantee that the message was not committed; report ambiguous
or permanent failures with `CodexError` so they are not blindly retried.

The default `SubprocessConnector` honors `codex_bin`, `launch_args_override`,
`config_overrides`, `cwd`, and `env`. It merges `env` into the Worker's environment.
Launch paths and runtime discovery are resolved on the Activity Worker. Custom
connectors can reach a remote process; the returned connection object still lives
on the Worker that created it. The connector owns cleanup of remote resources
that outlive the Worker. No Nexus service or tool-registration API is needed.

`CodexActivityOptions` overrides `connect`, `write`, `read`, and `close`, each using
the Temporal SDK's `workflow.ActivityConfig`. Only
`connect.task_queue` selects the queue where Codex starts. The other Activities use
the private queue polled by the Worker holding the connection. Activities default
to five-minute start-to-close timeouts and ten-second heartbeat timeouts. Operations
on an established connection also have a thirty-second schedule-to-start timeout
for detecting a lost Worker.

## Recovery and operational limits

Completed writes stay cached for the life of the connection, so Activity retries
and lost completion results do not resend prompts, approvals, or notifications.
Read batches stay buffered until a later read acknowledges them. Heartbeats let
Temporal cancel blocked operations; context exit closes the process, and orderly
Worker shutdown closes its connections.
If a connect result is lost, its idle connection remains with the owning Worker
until shutdown. Cleanup failures do not replace a Workflow's result or original error.

A process connection is not durable storage. Losing its Worker produces
`TransportClosedError`; the plugin does not automatically start a replacement
session. Keep persisted thread IDs in Workflow state. Handle the error by creating
a new client and explicitly calling `thread_resume` with the saved ID, using a
Worker with access to the same Codex home and workspace. Ephemeral threads and
in-flight process state cannot be recovered this way. Retrying a Workflow or
resetting its history can create new sessions and repeat application side effects.

Configure credentials on Workers, such as an authenticated `CODEX_HOME` or an
appropriate model provider. Avoid passing secrets through Workflow inputs,
`CodexConfig.env`, or login API-key calls: Activity inputs/results and notification
content are recorded in Temporal history. Use an appropriate payload codec if
sensitive content must pass through history. Tests require no real credentials.

Each protocol write and output batch adds Activity events to Workflow history.
Long streams and long-lived sessions grow history; bound conversations and close
connections before continuing as new. Worker memory also holds unacknowledged
output and write-deduplication records until close. Reads batch near Temporal's
payload warning threshold; a single oversized message is not split and can exceed
the server's payload limit. Use upstream options to limit output and an external
storage/payload strategy where needed. The subprocess reader accepts lines up to
16 MiB; the Temporal payload limit is usually smaller. Exceeding that line limit
or another terminal stdout read failure raises `TransportClosedError` without
retrying the failed reader.
