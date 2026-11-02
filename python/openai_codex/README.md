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

## Explicit protocol client

`TemporalCodexClient` mirrors `openai_codex.async_client.AsyncCodexClient` operations
and adds `notify`, approval callbacks, and Activity overrides. Context entry starts
the process; **call `initialize()` explicitly** before other protocol operations.
Requests accept upstream parameter models or wire-format dictionaries and return
upstream Pydantic models.

```python
from datetime import timedelta
from temporalio.openai_codex import CodexActivityOptions, TemporalCodexClient
from temporalio.workflow import ActivityConfig

async def approve(method, params):
    # This runs in Workflow code. It can wait for a Workflow Signal or Update,
    # or schedule an Activity, while the message reader continues running.
    return {"decision": "accept"}

async with TemporalCodexClient(
    approval_handler=approve,
    activity_options=CodexActivityOptions(
        connect=ActivityConfig(start_to_close_timeout=timedelta(minutes=2)),
    ),
) as codex:
    await codex.initialize()
    thread = await codex.thread_start({"sandbox": "read-only"})
    turn = await codex.turn_start(thread.thread.id, "Describe the project.")
    completed = await codex.wait_for_turn_completed(turn.turn.id)
```

Synchronous and asynchronous approval handlers receive `(method, params)` as in
upstream `CodexClient`. Keep callbacks deterministic; put I/O in Activities. Without
a handler, command and file-change approval requests receive `{"decision": "accept"}`
and other server requests receive `{}`, matching the SDK default. Set Codex's
approval and sandbox policies deliberately for your application.

The client also supports account login/read/logout and login-completion routing,
thread lifecycle methods, typed `request`, independent turn subscriptions, steering,
interruption, streaming deltas, and persisted goal operations. Goal operations
reserve a thread route and receive its continuation turns; cancel a goal operation
to pause it and interrupt the active physical turn.

## Connectors and launch configuration

`OpenAICodexPlugin(connector=...)` accepts a `Connector` implementing
`async connect(ConnectActivityInput) -> CodexProcess`. The input contains the upstream
`CodexConfig` and a reserved connection `handle`, which a remote connector can use
as an idempotency key. `CodexProcess` provides `write(message)`, `read() -> list[dict]`, and
`close()`. A read returns an ordered batch or an empty list at EOF. Close must be
idempotent, tolerate overlap, and unblock pending reads and writes. A transient
write failure must guarantee that the message was not committed; report ambiguous
or permanent failures with `CodexError` so they are not blindly retried.

The default `SubprocessConnector` honors `codex_bin`, `launch_args_override`,
`config_overrides`, `cwd`, and `env`. It merges `env` into the Worker's environment.
Launch paths and runtime discovery are resolved on the Activity Worker. Custom
connectors can reach a remote process; the returned connection object still lives
on the Worker that created it. No Nexus service or tool-registration API is needed.

`CodexActivityOptions` overrides `connect`, `write`, `read`, and `close`, each using
the Temporal SDK's `workflow.ActivityConfig`. Only
`connect.task_queue` selects the queue that reserves the owning Worker. Launch
then runs on that Worker's private queue, so cancellation can close a launch even
if its result is lost. Both steps use the connect options. The other Activities use the private
queue polled by the Worker holding the connection. They default to five-minute
start-to-close timeouts, ten-second heartbeat timeouts, and a thirty-second
schedule-to-start timeout for detecting a lost Worker.

## Recovery and operational limits

Completed writes stay cached for the life of the connection, so Activity retries
and lost completion results do not resend prompts, approvals, or notifications.
Read batches stay buffered until a later read acknowledges them. Heartbeats let
Temporal cancel blocked operations; context exit closes the process, and orderly
Worker shutdown closes its connections.

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
output and write-deduplication records until close, and closed connection IDs until
Worker shutdown to reject delayed launches. Reads batch near Temporal's
payload warning threshold; a single oversized message is not split and can exceed
the server's payload limit. Use upstream options to limit output and an external
storage/payload strategy where needed. The subprocess reader accepts lines up to
16 MiB; the Temporal payload limit is usually smaller. Exceeding that line limit
or another terminal stdout read failure raises `TransportClosedError` without
retrying the failed reader.
