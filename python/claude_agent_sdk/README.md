# Claude Agent SDK Temporal Integration

> Release stage: [Pre-release](https://docs.temporal.io/develop/python/integrations/claude-agent-sdk).

Run Claude Code agents from Temporal Workflows with the
[Claude Agent SDK](https://code.claude.com/docs/en/agent-sdk/overview). `TemporalTransport` carries
Claude Code's input and output through Activities, so approvals, hooks, and custom tools run in the
Workflow and survive Worker restarts.

## Install

```bash
uv add temporalio-claude-agent-sdk
```

## Usage

Use the Claude Agent SDK in the Workflow as usual, and pass it a `TemporalTransport` with the same
options as the client:

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

Add `ClaudeAgentSDKPlugin` to the Worker:

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

## Documentation

The [Temporal documentation](https://docs.temporal.io/develop/python/integrations/claude-agent-sdk)
covers:

- [Custom tools](https://docs.temporal.io/develop/python/integrations/claude-agent-sdk#provide-custom-tools)
- [Hooks and human approval](https://docs.temporal.io/develop/python/integrations/claude-agent-sdk#add-guardrails-to-agent-behavior)
- [Streaming output](https://docs.temporal.io/develop/python/integrations/claude-agent-sdk#stream-output-from-a-workflow)
- [Running Claude Code outside the Worker](https://docs.temporal.io/develop/python/integrations/claude-agent-sdk#configure-a-custom-connector)
- [Recovering from a lost Worker](https://docs.temporal.io/develop/python/integrations/claude-agent-sdk#manage-the-worker-connection-to-claude-code)
- [Configuration and errors](https://docs.temporal.io/develop/python/integrations/claude-agent-sdk#reference-and-troubleshooting)

The [samples](https://github.com/temporalio/samples-python/tree/main/claude_agent_sdk_plugin) show
each of these end to end.

## Checkpoint a session between Workers

`SessionCheckpoint` is opt-in. Use it in an Activity on the Worker that owns
the session after Claude Code has shut down. Give `checkpoint` the dedicated
`CLAUDE_CONFIG_DIR`, the session ID, its working directory, and a storage
callback. The callback receives a mapping of relative paths to native files and
the previous storage key, if any. It returns an immutable key after it has
saved the files. Persist the returned versioned reference and the native reply
ID together.

On a replacement Worker, await `restore` in an Activity before connecting
through `TemporalTransport`. Supply the saved reference, session and reply IDs,
the matching working directory, a fresh staging directory, and a callback that
materializes and verifies the stored files. Activate staging at the intended
`CLAUDE_CONFIG_DIR`, then reconnect with `resume` and `resume_session_at`.
The restore uses a checkpoint-scoped `SessionStore` to load the main and
subagent transcripts. The native JSONL remains the only persisted transcript.

Main and subagent transcripts and subagent metadata are required when present.
A worker-side `include_optional(relative_path, kind, size)` predicate can omit
`tool-result` files. The policy has no access to authentication, settings,
skills, diagnostic logs, or locks. Callbacks and file contents stay on the
Worker. Keep each in-flight restore in a separate staging directory and
activate only the confirmed attempt.
