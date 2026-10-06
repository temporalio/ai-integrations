# Durable Claude Agent SDK agents on Temporal

> ⚠️ **Experimental.** The API may change.

This prototype branch pins the [main-agent recovery SDK fork](https://github.com/brianstrauch/claude-agent-sdk-python/tree/feature/main-agent-recovery)
at commit [`3ac4b25`](https://github.com/brianstrauch/claude-agent-sdk-python/commit/3ac4b25733d1302c89d0dadd13cab268e64d1213).
See [Main-agent recovery](#main-agent-recovery-with-the-sdk-fork-2026-09-30)
for setup and the recovery/suspension experiments.

## High-priority prototype fixes (2026-10-06)

The live native-tool prototype in [`tests/hybrid/native.py`](tests/hybrid/native.py)
now runs **Read, Edit, Write, Bash and configured MCP tools** through the same
long-lived CLI. Each original call still has its own Temporal Activity and stored
result. Workspace tools execute in order, and their permission is withheld until
the Workflow has accepted the original ID and scheduled its Activity. Ten native
Bash rounds used **one CLI process** and measured **181.9 ms median** between model
requests on a small local workspace (Python 3.14.4, Temporal 1.33.0, pinned SDK fork,
CLI 2.1.273, deterministic local Messages API). This measurement includes Activity
dispatch, result storage and workspace snapshots; provider latency and larger
workspaces will change it.

The execution ledger refuses a second permission for an ID that may already have
executed. Tool Activity retries rejoin the original executor or return its committed
native result. Worker replacement returns committed results under their original
IDs without running the effect again. A Worker lost after an external effect but
before outcome publication **fails closed and requires reconciliation**. This is
an explicit ambiguous outcome, not a successful result: arbitrary Bash commands
and non-idempotent MCP services cannot safely be retried without a service receipt
or equivalent idempotency support. The tests use an external append-only file and
a deliberately non-idempotent MCP charge service, and verify one effect after both
Activity retry and Worker loss.

Native Edit and Write retain Claude's own execution, schemas and results. After a
committed overwrite, the SDK fork completes the original pending transcript before
starting the replacement CLI. The engine therefore sees the original successful
result and does not revalidate the already-modified file. Both Edit and an
overwriting Write after a Read are covered across Worker loss, including Workflow
history replay. This is the prototype's workaround for
[claude-code#99041](https://github.com/anthropics/claude-code/issues/99041).

Workspace snapshots now preserve a complete managed tree of regular files and
directories, including binary contents, empty directories, file deletions,
permissions and modification times. The original result and tree are committed
together. Replacement Workers restore the tree before starting their CLI. A local
OS lock prevents checkout while the previous executor is alive, and database
ownership checks fence stale result/snapshot writers. Links, special files and
snapshots over 64 MiB fail publication. All Workers still need the shared
transactional store and the same logical working-directory path. This is not a
filesystem sandbox or a rollback mechanism for external effects.

The packaged segment runner and live native prototype now require a supervisor.
On Linux the supervisor becomes a subreaper before startup, so orphaned tool
processes are adopted by it. On macOS it retains process lifetime IDs and original
parent IDs throughout execution, including after a child detaches or its parent
exits. The engine waits at an exec gate until its own identity has been recorded.
POSIX cleanup freezes and stops owned processes before releasing the workspace
lock; it requires `/bin/ps`, plus Linux `prctl` or macOS `proc_pidinfo` support.
The macOS identity interface is private and fails startup if unavailable. It can
recover direct detached children after immediate engine exit; a fast double fork
whose intermediate parent disappears before discovery still requires a stronger
execution boundary on macOS. On Windows it assigns a suspended engine to a
kill-on-close job before letting it run. The supervisor retains its
lock until cleanup finishes. A missing launcher, unavailable lock, or failed job
setup prevents engine startup; there is no warning-and-continue fallback. Crash
probes cover Worker loss, normal engine exit, engine SIGKILL, and immediate exit
after spawning a detached child. They verify that writes stop before a replacement
can obtain the lock, while unrelated processes continue running. A real in-flight
Bash command also stops before reaching its delayed external write. Actual
Windows job behavior is implemented but was not exercised on this macOS host.

These changes extend the test prototype; the exported durable-agent API still
uses segments. Subagent recovery and the older checkpoint/replay experiments
retain their documented limits below.

```bash
cd python/claude_agent_sdk
make sync
make lint
make test PYTEST_ARGS='tests/test_process.py tests/hybrid/test_workspace_snapshot.py tests/hybrid/test_native_effects.py -n 3 -q'
# Print the native Bash round measurement:
make test PYTEST_ARGS='tests/hybrid/test_native_effects.py -k native_rounds -n 0 -s -q'
```

Initial validation on this host: the full regression run passed **251 tests**, with
the opt-in benchmark skipped. The recovery/cleanup suite passed **42 tests**,
and two additional real-engine permission-callback regressions passed for the
live client and segment query. Lint, wheel/sdist checks and isolated artifact
smoke installs passed. Repository conventions passed in a disposable clean
worktree; the working checkout retains ignored virtualenv directories from other
branches, which the conventions discovery otherwise treats as missing plugins.

The P1 supervision follow-up passed **263 tests** in the full regression run,
with the opt-in benchmark and Linux-only double-fork case skipped on macOS.
A final process-only check passed **13 tests**. Linux container probes covered Worker loss,
normal engine exit, engine SIGKILL, immediate exit and double-fork cleanup. The
process tests also verify startup fencing, PID reuse protection, lock reuse and
continued execution of unrelated processes.

Temporal integration for Anthropic's [Claude Agent SDK](https://code.claude.com/docs/en/agent-sdk/overview), published as [`temporalio-claude-agent-sdk`](https://pypi.org/project/temporalio-claude-agent-sdk/) and imported as `temporalio.claude_agent_sdk`.

- **Every durable tool call Claude makes is its own Temporal Activity.** Finished calls never run again after a crash, retries follow your retry policy, and the Activity ID (`tool-<tool_use_id>`) doubles as an idempotency key for the systems a tool touches.
- **Tools can wait for a human.** Mark a tool `needs_approval=True` and the agent waits, for minutes or weeks, until someone approves or rejects the call.
- **Crashes resume cleanly, on any Worker.** Each model step ends at a checkpoint that Temporal records. A step that runs again (after a crash, a timeout, or on another machine) starts from that checkpoint, so nothing a failed attempt added to the conversation reaches Claude.

## Install

```bash
uv add temporalio-claude-agent-sdk
```

It depends on `claude-agent-sdk>=0.2.153`, which bundles Claude Code 2.1.273 or newer (see [Requirements](#requirements-and-limits)).

## Quick start

A Workflow with an agent inside. Tools are ordinary Activities that take one dict argument:

```python
from datetime import timedelta
from typing import Any

from temporalio import activity, workflow
from temporalio.claude_agent_sdk import DurableClaudeAgent, activity_as_tool


@activity.defn
async def look_up_order(args: dict[str, Any]) -> dict[str, Any]:
    """Look up an order by id."""
    ...


@activity.defn
async def issue_refund(args: dict[str, Any]) -> dict[str, Any]:
    """Refund an order. Moves real money."""
    key = activity.info().activity_id  # stable across retries: use it as the idempotency key
    ...


@workflow.defn
class RefundAgent:
    def __init__(self) -> None:
        self.agent = DurableClaudeAgent(
            system_prompt="You handle refund requests for an online store.",
            tools=[
                activity_as_tool(look_up_order),
                activity_as_tool(issue_refund, needs_approval=True),
            ],
            approvers=["manager@shop.example"],
        )

    @workflow.run
    async def run(self, request: str) -> str:
        return await self.agent.run(request)

    @workflow.update
    def review(self, tool_use_id: str, approved: bool, approver: str) -> None:
        self.agent.decide(tool_use_id, approved, approver)

    @review.validator
    def check_review(self, tool_use_id: str, approved: bool, approver: str) -> None:
        self.agent.validate_decision(tool_use_id, approver)  # refused Updates never reach history

    @workflow.query
    def pending_approvals(self) -> list[dict[str, Any]]:
        return self.agent.pending_approvals()
```

Tool names use 1 to 50 letters, digits, `_` or `-` (Claude Code renames anything else, and the call could never run). Claude's arguments are not checked against `input_schema`: validate them in the Activity, like any untrusted input. `activity_as_tool` also takes `schedule_to_close_timeout`, `heartbeat_timeout` and `task_queue`.

The Worker runs Claude through the plugin:

```python
from temporalio.claude_agent_sdk import ClaudeAgentPlugin, ClaudeAgentSdkRunner, FileSessionStore

runner = ClaudeAgentSdkRunner(
    session_store=FileSessionStore("/shared/claude-sessions"),
    cwd="/srv/agent",
)
worker = Worker(
    client,
    task_queue="agents",
    workflows=[RefundAgent],
    activities=[look_up_order, issue_refund],
    plugins=[ClaudeAgentPlugin(runner)],
)
```

The conversation lives in a session store that every Worker must reach. `FileSessionStore` is for tests and one machine; for production, implement `SessionStore` on your database or object storage (the Claude Agent SDK repository has example stores for S3, Redis and Postgres). The store keys sessions by the engine's working directory, so give every Worker the same `cwd`. The runner refuses a path for which Claude Code and the SDK would compute different keys (for example with decomposed Unicode or emoji).

Pass the plugin to the Worker, or to the Client the Worker is built from, not both. Each segment starts a Claude Code process: 0.7 to 0.9 seconds and up to about 270 MB of memory each (measured on Linux with an instant local model), so cap parallel segments with the Worker's `max_concurrent_activities`.

Other [`ClaudeAgentOptions`](https://code.claude.com/docs/en/agent-sdk/python) go in the runner's `extra_options` (for example `permission_mode`, `agents`, `hooks`, `setting_sources`, `thinking`). `env`, `mcp_servers` and `allowed_tools` are merged with the plugin's own, and `system_prompt` (a string, or a preset such as Claude Code's own prompt) is the default for agents that set none. Options the agent or the plugin sets (`model`, `tools`, `max_turns`, `cwd`, `settings`, session and resume options, and the same engine flags in `extra_args`) are refused. External MCP servers run inside the segment, like built-in tools, and hooks from settings you load must not decide on the plugin's `mcp__durable__` tools.

`approvers` checks the name the caller passes. It is not authentication: control who may send Updates with Temporal's own access control.

## Long-running agents

A Workflow's history holds at most 51,200 events or 50 MB (Temporal's default limits), and every durable tool call adds about 12 events. So one run fits about 4,000 tool calls, and fewer when results are large. Continue-As-New starts a fresh history. The conversation stays in the session store, so the agent's state is small.

Turn on `auto_continue_as_new` and give the Workflow a `state` argument:

```python
from temporalio.claude_agent_sdk import AgentState, DurableClaudeAgent


@workflow.defn
class ResearchAgent:
    @workflow.init
    def __init__(self, prompt: str, state: AgentState | None = None) -> None:
        self.agent = DurableClaudeAgent(
            tools=[...], state=state, max_segments=None, auto_continue_as_new=True
        )

    @workflow.run
    async def run(self, prompt: str, state: AgentState | None = None) -> str:
        return await self.agent.run(prompt)  # continues as new inside when needed
```

- When the server suggests it (`is_continue_as_new_suggested()`, which counts history length, history size and Updates), the agent continues as new at the next safe point between two tool calls. It lets running Update and Signal handlers finish and continues as new with `[prompt, state]`. The new run calls `agent.run(prompt)` again; the agent sees the unfinished task in its state and continues it instead of sending the prompt again.
- It is off by default, because the new run's arguments must match your run method. For other arguments, pass `continue_as_new_args`, a function that builds them from the `AgentState`. `continue_as_new_after_events` uses a fixed history length instead of the server's suggestion.
- Call `agent.run()` from the Workflow's run method, not from a separate asyncio task, and let no handler wait for the task to finish: continuing as new waits for every handler to finish. With `auto_continue_as_new`, a call from a handler fails at once: an Update handler fails its Update, a Signal handler fails the Workflow.
- **Chats.** Calling `agent.run(message)` again continues the same Claude session, so Claude remembers earlier turns. Turns without tool calls never reach a point between tool calls, so between messages, when no handler is waiting for an answer, call `await agent.continue_as_new()` from the run method if `agent.should_continue_as_new()` is true; `continue_as_new_args` carries your own state, such as an inbox. At the start of a new run, if `agent.busy` is true, `await agent.run()` finishes the task that was interrupted.
- `max_segments` caps a task across all its runs (default 50). The task stops before it runs a tool whose result could no longer reach Claude. Use `None` for tasks that may take thousands of steps.
- **Start with every argument.** Temporal applies a Workflow's argument types only when the caller passes as many arguments as `run` declares. If `run` has other typed arguments besides `state`, start the Workflow with `state=None` included, or those arguments arrive as plain dicts.

## Failures and cancellation

- A task fails with an `ApplicationError` at `max_segments`, when a step reports an error that running it again cannot fix (Claude Code's `max_turns` or `max_budget_usd`, a broken pause, or a Messages API request that would be refused again: invalid (400), an unknown model (404), too large (413), or a prompt too long for the model), or if the engine hands back a tool call that already ran. Other engine and API errors, such as a low credit balance, a rejected API key, rate limits or overload, fail the step's Activity, and Temporal retries it with `segment_retry_policy` (by default without limit, so the step continues once the cause is fixed; each retry copies the session). When the Activity fails for good, the task fails with an `ActivityError`. Catch `temporalio.exceptions.FailureError` for both.
- After a failed task, the agent can take the next one: it continues from the last checkpoint, and a tool call Claude was still waiting for gets an error result, delivered with the next prompt.
- Cancelling the Workflow cancels the running step or tool call, and the Workflow ends as cancelled. By default Temporal does not wait for a cancelled Activity. `activity_as_tool(..., cancellation_type=workflow.ActivityCancellationType.WAIT_CANCELLATION_COMPLETED)` waits until the call finishes or acknowledges the cancellation through a heartbeat, so the history records what really happened; `segment_cancellation_type` does the same for a running step.
- A running step hears of a cancel (or of its own timeout) with its next heartbeat: at most 0.8 × `segment_heartbeat_timeout` later (30 seconds by default, so up to 24 seconds). From then on the engine cannot start a built-in tool while the SDK stops it, which takes up to about 10 seconds (a durable call only pauses the run). If the Worker process dies, its engine runs until its current turn ends, built-in tools included, unless the Worker's child processes stop with it (as in a container or a systemd service).

## Large tool results

Every tool result is stored in the Workflow's history twice: as the tool Activity's result, and in the next step's input. A single payload over 2 MB cannot be recorded at all (the SDK stops it with `[TMPRL1103] Attempted to upload payloads with size that exceeded the error limit`), and results of a few hundred KB fill the 50 MB history quickly.

The plugin works with Temporal's [External Storage](https://docs.temporal.io/external-storage) unchanged. Payloads over a threshold (256 KiB by default) go to your store, and the history keeps small references. Configure it on the Client, as for any Temporal application; Workers built from that Client use it too:

```python
from temporalio.converter import DataConverter, ExternalStorage

client = await Client.connect(
    "localhost:7233",
    data_converter=DataConverter(external_storage=ExternalStorage(drivers=[my_s3_driver])),
)
```

Use the same External Storage on every Client that starts or queries these Workflows. Measured on a local dev server at Temporal's default limits, with 40 tool results of 1 MB: without it, the run was terminated after 25 results ("Workflow history size exceeds limit"); with it, the task finished in one run with 84 KB of history.

## Live output

Turn on `live_output=True` and the agent publishes its events through Temporal's [Workflow Streams](https://docs.temporal.io/develop/python/workflows/workflow-streams). A UI reads them with `follow_agent`, which follows Continue-As-New by itself:

```python
from temporalio.claude_agent_sdk import follow_agent

async for event in follow_agent(client, workflow_id):
    print(event["type"], event.get("text") or event.get("name") or event.get("result"))
    if event["type"] in ("done", "error", "cancelled"):
        break
```

- Event types: `prompt`, `text`, `tool_call`, `approval_needed`, `tool_result`, `retry`, `continued_as_new`, `done`, `error` and `cancelled`. Every event has the time it happened (`at`) and its `offset`.
- Claude's text arrives one assistant message at a time, not token by token. `text` and `retry` events carry the `segment` and `attempt` that produced them: after a `retry` event, the earlier text of that segment is superseded.
- After a disconnect, continue from the last offset plus one. Continue-As-New carries only the newest events (`live_output_keep`, 1,000, and `live_output_keep_bytes`, 256 KiB); asked for an older offset, the stream starts at the oldest event it still has, so compare offsets to see a gap.
- A `text`, `result` or `error` longer than 32,768 characters, or a tool `input` longer than that as JSON, is cut, and the event gets `truncated: true`. The full answer is the Workflow's result.
- After its final event, a task waits `live_output_linger` (500 ms) so subscribers receive it before the Workflow closes.
- Measured on a local dev server, three runs of 203 events: 107 to 110 ms median and 161 to 162 ms at the 95th percentile, from event to subscriber. Workflow Streams is built for UIs and progress, not real-time voice.
- Continue-As-New carries the stream in the new run's input, next to the agent's state. The stream shrinks to fit, so a large tool result waiting to be delivered still fits under Temporal's 2 MB payload limit. The rest of the input is measured before any codec or External Storage, which can only overstate it, so the stream may keep fewer events than would fit.
- **Every subscriber poll is an Update**, and a Workflow accepts at most 10 Updates in flight and 2,000 per run (Temporal's defaults). Idle subscribers hold Updates in flight, so with many direct subscribers an Update approval can be refused (tested: with 12, it failed with `RESOURCE_EXHAUSTED`). Approve by Signal instead (`decide()` ignores invalid decisions there), or fan events out to viewers through one subscriber in your backend. For long runs with live output, turn on `auto_continue_as_new`: the server counts the polls toward its suggestion.
- Create the agent while the Workflow is being initialized (in its `__init__`), where Workflow Streams registers its handlers; later, the constructor raises.

## How it works

The agent loop runs in *segments*. A segment is one Activity that runs the Claude Code engine from a prompt (or a tool result) until Claude either calls a durable tool or finishes.

1. Durable tools are declared to Claude as SDK MCP tools.
2. A `PreToolUse` command hook answers `defer` whenever Claude calls one ([documented in the hooks guide](https://code.claude.com/docs/en/hooks)). The engine stops with `stop_reason: "tool_deferred"`, and the segment returns the call.
3. The Workflow runs the call as its own Activity, after an approval if the tool needs one.
4. The next segment resumes the session with the result as a normal `tool_result` message.

**Checkpoints.** After a segment, the runner reads the session back from the store and returns its last transcript entry, where the engine would resume; the Workflow records that checkpoint with the segment's result. Reading it back also proves the turn reached the store. A segment that runs again continues in a copy of the session that ends at the checkpoint (`fork_session_via_store`), and Claude decides again from there, with new tool call ids. So does a segment whose session went on past its checkpoint, for example after a Workflow reset: the check rides on the load the SDK does anyway. After Claude sent several durable calls at once, the checkpoint is the paused call's deferral marker (the denied calls' results come after it, and the engine would not resume the paused call past them), so the next segment also continues in a copy. Tested on the real engine: a Worker killed while Claude is answering, an attempt that hangs past its timeout, and results lost after a step finished.

**Fail closed.** If a Claude Code version ever runs a durable tool itself, or ignores a pause, the segment fails with a non-retryable error that names the engine version. Nothing runs outside Temporal. If the engine ever hands back a tool call id that already ran (the agent remembers every call of the current run and the last 256 before it), the Workflow stops instead of running it twice.

## Requirements and limits

- **Claude Code 2.1.273 or newer.** Tested: when Claude calls two tools in one message, Claude Code 2.1.259 replaces the paused call's result with `[Tool result missing due to internal error]`, so Claude asks for the same tool again. The runner checks `claude -v` before starting the engine and refuses older engines.
- **One durable tool call at a time.** The engine keeps one paused call per run. The runner asks Claude for one call per message; if Claude sends several, the first durable call pauses and every call after it in the same message, durable or built-in, is told to call again after its result.
- **Claude Code's built-in tools** (Bash, Edit, and so on) are off unless you pass `builtin_tools`. When enabled, they run inside the segment Activity, not as their own Activities, on the Worker's disk in the runner's `cwd` (shared by every agent on that Worker). They can run again when a segment runs again, possibly on another Worker with a different disk, and files a failed attempt wrote are not rolled back. Give them work that is safe to repeat, or make the work a durable tool.
- **A session store every Worker can reach**, and the same `cwd` on every Worker. Every segment reads the whole session twice (the SDK to resume it, the runner to record the checkpoint), so reading grows with the session's length. A segment that runs again reads it once more and writes a copy; the earlier copy stays in the store, so remove old sessions with your store's own retention.
- **Subagents run in the foreground, without durable tools.** The runner turns off Claude Code's background tasks (a background subagent kept the engine working after it paused). A subagent (Claude Code's `Agent` tool) can use built-in tools; a durable tool call from a subagent cannot pause the run, so the step fails closed.
- **Long conversations.** Continue-As-New keeps the Workflow's history small, but the conversation itself grows until Claude Code compacts it. A request the model refuses as too long fails the task.
- **Logins that survive resumes.** Use `ANTHROPIC_API_KEY`, Amazon Bedrock, Google Vertex AI, Microsoft Foundry, or `CLAUDE_CODE_OAUTH_TOKEN` from `claude setup-token`. A Claude app login cannot refresh itself when a session is resumed from a session store.

## Security

- Credentials stay on Workers: the engine inherits the Worker's environment (API keys, cloud credentials, proxies). They reach Workflow history only if a built-in tool shows them to Claude (for example Bash running `env`), so give Workers that enable built-in tools only the secrets they need.
- The conversation does: prompts, Claude's text, tool arguments and results are in the Workflow's history and in the session store. Encrypt payloads with a codec and protect the store like any other data store.
- Claude chooses tool arguments. Treat them as untrusted input in your Activities, and require approval (`needs_approval=True`) for tools that move money or delete data.

## Testing your agents

`temporalio.claude_agent_sdk.testing.ScriptedClaude` is a segment runner that plays Claude with a Python policy. It needs no engine and no API key. It keeps sessions in a folder, so tests can kill a Worker and continue on a new one, and its checkpoints behave like the real runner's: a segment that runs again decides again, with a new tool call id.

## Documentation

- Repository conventions: [`AGENTS.md`](https://github.com/temporalio/ai-integrations/blob/main/AGENTS.md)

## Develop

```bash
make sync   # install (non-editable) into .venv
make lint
make test   # the real Claude Code engine against a local fake Messages API; no credentials
```

## Bounded native call replay experiment (2026-09-30)

`NativeReplayWorkflow` addresses the three blockers in the earlier checkpointed
runner for **main-agent Read/Edit**: whole native batches, errors produced before
`PreToolUse`, and execution without model continuation. Subagents remain excluded.
This is a test-only adapter around the published SDK and real Claude Code CLI;
it does not modify the compiled engine or add a production public API.

The model Activity runs one ordinary CLI turn with the original native tool
schemas. Its preparation hook denies execution. Eager mirroring writes the real
assistant calls to an isolated attempt store. The shared SQLite service then
atomically publishes the whole original assistant batch and immutable receipts
containing each original ID, arguments, transcript UUID and source digest. It
discards this preparation's denial and validation artifacts. Temporal records
all accepted receipts before scheduling `tool-<original ID>` Activities. Once a
batch is accepted, recovery never asks a provider to regenerate its calls.

Each tool Activity materializes an isolated execution context from the durable
conversation. It removes the unresolved calls from that private copy and keeps
completed calls and their real native results, including the Read metadata Edit
needs. A local Messages response cache returns the **exact recorded assistant
tool block**, with its original ID and arguments. The regular CLI turn executes
the native tool and stops at `max_turns=1`. The cache invokes no provider model.
The Activity verifies one cached response, the original call identity, the
bounded-turn result and the actual native outcome before publication. Validation
errors are genuine native results produced inside this independently retried
Activity, even when the engine skips its permission hook.

One transaction publishes the real executor's result carrier (including native
`toolUseResult` metadata), the exact `tool_result`, and the next workspace snapshot.
Only the carrier's parent UUID is attached to the canonical conversation head;
the result content and metadata stay unchanged. The original canonical batch
stays intact. No engine deferral markers or tool results are invented, and calls
are never matched by arguments or regenerated IDs. This **does** shape a private
execution transcript and deliver native results back into a conversation; it
is not a native prepare/execute/commit RPC implemented by the engine.

Read-only batches execute concurrently against one immutable snapshot. Batches
containing Edit execute in original order, preserving prior Read context and
committed mutations. The managed workspace is still one regular `note.txt`,
preserving bytes, mode and modification time. Attempt fencing rejects stale
publication; retries return committed outcomes or restore the last committed
snapshot before native execution. A supervising host stops orphan CLIs before
replacement work uses their workspace.

| Capability | Result | Reproduction in `tests/hybrid/test_native_replay.py` |
|---|---|---|
| Three native Reads from one response, concurrent execution, original IDs | **supported** | `test_native_replay_boundaries[parallel]`; all three permission callbacks overlap before any execution finishes |
| Read and Edit in one original response, with native Read context | **supported** | Same test's `[mixed]`; two real model requests, no rediscovery of an omitted call |
| Two native Edits in one response | **supported** | Same test's `[edits]`; original order, preserved Read/Edit metadata and three committed workspace versions |
| Native validation error as an independent durable Activity result | **supported** | Same test's `[validation]`; a missing Edit match keeps the actual error under its original ID |
| Execution without model continuation or HTTP-error guard | **supported** | Every boundary test verifies one cached response per executor attempt and the bounded CLI result |
| Worker loss before execution, after local Edit, or after publication | **supported** | `test_native_replay_worker_loss`; original accepted IDs and outcomes survive deletion of local workspace, profiles and attempt transcripts |
| Worker loss during a partially completed batch | **supported** | Same test's `partial-batch` case; one cached outcome, two pending original IDs, no repeat of the completed Read |
| Worker loss after the first Edit in a mutable batch | **supported** | Same test's `partial-edits` case; restore the first mutation and complete the second original ID without rerunning committed calls |
| Worker loss after publishing a native validation error | **supported** | Same test's invalid Edit `after-commit-before-completion` case; unchanged file and cached original error |
| Lost batch-preparation acknowledgment | **supported** | Same test's Read `after-checkpoint-before-completion` case; retry returns the accepted receipt without another model request |
| Failed atomic outcome publication | **supported** | Boundary test's `[publication]`; failed transaction rolls back result/transcript/snapshot, retry executes original Edit against restored input |
| Preparation hook failure | **supported** | `test_preparation_hook_failure_denies_native_edit`; explicit denial, no accepted Edit or file mutation, committed Read preserved |

The single Read/Edit run starts **five CLI processes**, makes **three actual
primary model requests** and **two local cached-response requests**, with zero
refused continuation requests. The three-Read batch starts five CLIs, makes two
actual model requests and three cached requests. Cache transport requests are
reported separately because they do not execute a model. A failed publication or
loss before publication may repeat native execution and its cache request.

This closes the three experimental blockers within the tested scope. It does
not establish full production durability for every Claude Code tool or preserve
the original hybrid's process reuse. General filesystem state, permissions,
other built-ins, arbitrary Bash effects and production shared storage need their
own protocols. Activity side effects require idempotency or reconciliation;
the Edit rollback tests establish one published outcome, not exactly-once
external execution. Production idle thresholds and public suspension settings
remain deferred.

The direct deferred-resume engine behavior is still reproducible:
`test_startup_stop_hook_does_not_bound_deferred_resume` in
`tests/hybrid/test_native_boundaries.py` shows that startup PostToolUse stop hooks
do not prevent continuation, and PostToolBatch does not run on that path.
`test_cached_native_call_turn_is_bounded` demonstrates the ordinary-turn boundary
used by the new adapter. The engine also documents that
[native deferral only supports one call](https://code.claude.com/docs/en/hooks#defer-a-tool-call-for-later).
The adapter avoids deferred auto-resume entirely.

```bash
make sync
make test PYTEST_ARGS='tests/hybrid/test_native_replay.py tests/hybrid/test_native_boundaries.py -n 3 -q'
```

All **30 native capability and boundary probes** pass with the locked published
SDK `0.2.153` / CLI `2.1.273` / Temporal `1.33.0`, the newest allowed SDK
`0.2.154` / CLI `2.1.274` / Temporal `1.34.0`, and that newest SDK with CLI
`2.1.285`. This includes the 14 bounded replay cases, three engine boundary
probes, and 13 earlier checkpointed executor cases. The new adapter requires no
experimental SDK option.

The complete locked regression run passes **177 tests**, with **49 skips** for
experimental-SDK recovery cases and the opt-in benchmark. The subsequently added
preparation-failure probe and six bounded-turn scenarios also pass separately.
With the sibling
recovery SDK `0.2.162` / CLI `2.1.273`, the full regression run passed **223 tests**
with one benchmark skip, and the two subsequently added mutable-batch probes
also pass. Strict Temporal handler and coroutine cleanup warnings remain enabled.
Locked and newest lint, wheel/sdist checks, isolated artifact installation smoke
tests, repository conventions, and all **101 tooling tests** pass. The primary
environment was restored to the committed lockfile. No dependency manifest,
public API, CI workflow or sibling SDK changes are part of this extension.

## Earlier checkpointed native tool Activity experiment (2026-09-30)

The earlier test-only runner, `NativeExecutionWorkflow`, gives **Read** and **Edit**
independent native executors. It recovers calls that the live-hook experiment
below must block, provided the CLI created a single-call checkpoint before
Temporal accepted the tool intent. It uses the real native schemas, executor
and output; there is no MCP substitute or model request to regenerate an
accepted call. Subagents remain excluded.

The model Activity defers its one native call through the existing engine
mechanism. Its transcript lives in a disposable attempt store until the shared
SQLite service commits the actual engine checkpoint and its immutable recovery
receipt in one transaction. The receipt records the original ID, arguments,
transcript UUID and input workspace version. The Workflow records that receipt
in history before scheduling `native_execution` with Activity ID
`tool-<original ID>`. A lost Activity acknowledgment returns the stored receipt.
A crash before publication discards the attempt; the model Activity may retry
with a different, unaccepted ID, but no tool executor has run.

The tool Activity restores the committed workspace and materializes the frozen
engine checkpoint into a fresh CLI profile using the SDK's resume helpers.
That CLI executes the original deferred native call. An isolated transcript
mirror captures its actual result, including Read context required by Edit.
One transaction publishes the engine's entries through that result, its exact
`tool_result` block and the next workspace snapshot. An Activity retry returns
this cached outcome if publication succeeded. Otherwise it discards local
writes and resumes the same checkpoint against the original input snapshot.
The workspace preserves the managed file's **bytes, mode and modification
time**; restoring modification time avoids a false Edit file-change warning.

This is a workaround for a missing execute-tool boundary. The resumed CLI also
tries to continue to the model. A separate local Messages endpoint refuses that
request with HTTP 400, before any model executes. The prototype verifies that
the native result was committed and excludes the later API-error entries from
the canonical conversation. `max_turns`, a budget limit and a post-tool stop
hook did not establish a reliable tool-only exit in the exploratory probes.
Deferred auto-execution can also precede SDK hook initialization, so the runner
validates the stored native call and engine deferral before spawning; it does
not depend on a resume hook for recovery identity or failure injection.

| Checkpointed capability | Result | Reproduction in `tests/hybrid/test_native_executor.py` |
|---|---|---|
| Separate native Read/Edit Activities with original IDs and Read context | **supported** | `test_checkpointed_native_execution`; real built-ins, two tool Activities, history replay |
| Worker loss before native execution | **supported** | `test_checkpointed_native_executor_survives_worker_loss[Read-before-execution]` and `[Edit-before-execution]` |
| Worker loss after an Edit writes locally, before publication | **supported** | Same test's `[Edit-after-write]`; restore original snapshot, execute original ID again, publish once |
| Worker loss after result/workspace publication but before Activity completion | **supported** | Same test's Read/Edit `after-commit-before-completion` cases; cached original outcomes, no repeated native execution |
| Checkpoint publication and a lost model Activity acknowledgment | **supported** | `test_native_checkpoint_receipt_survives_worker_loss` and `test_unpublished_native_checkpoint_is_discarded_after_worker_loss` |
| Failed atomic result/workspace/transcript publication | **supported** | `test_failed_native_publication_retries_from_checkpoint`; injected failure rolls back all three, retry restores original input |
| Native batches | **blocked** | `test_native_checkpoint_rejects_batches_before_execution`; reject the response before accepting intents or changing the workspace |
| Native validation errors before deferral | **supported** | `test_native_validation_before_hook_is_durable` and `test_native_validation_outcome_survives_worker_loss`; publish the real error, unchanged snapshot and receipt atomically, and cache the terminal answer |
| Native execution that exits without attempting model continuation | **unsupported** | Every successful executor probe records the refused continuation; the test endpoint remains necessary |
| Recover an ordinary, uncheckpointed pending native call | **blocked** | The earlier live-hook `test_ordinary_resume_interrupts_pending_native_calls` still produces interruption results; this runner changes the acceptance protocol rather than repairing such sessions |

Each Worker-loss probe stops the Worker and its recorded CLI, deletes the local
workspace and configuration, and resumes on a replacement Worker. Tool-loss
probes also delete every disposable attempt transcript. Replacement CLIs receive
fresh profiles materialized from the shared service and accepted checkpoints.
SQLite ownership and workspace version checks fence publication; the
supervising host must still stop orphan CLIs and
enforce exclusive workspace access. SQLite here models a shared transactional
service; it is not a shipped production backend or a general filesystem store.

The no-failure Read/Edit run starts **five CLI processes**: three model Activities
and two native executor Activities. It makes three primary model requests, plus
two to four locally refused continuation attempts in the observed runs. The
earlier hybrid live-hook path uses one process for that run. This experiment
gains independently retryable native executors at the cost of process reuse;
it is not a production hybrid replacement. An upstream protocol would need
tool-only execution, reliable
initialization ordering, checkpointed validation outcomes and native batch
restoration to retain both durability and process reuse.

Rollback covers the managed local file only. The Edit-after-write probe executes
twice against restored input and publishes once; it does **not** establish
exactly-once external execution. Arbitrary Bash effects still need idempotency
or reconciliation, and external session/workspace storage remains required.
These results do not justify a full native durability claim or a production
refactor yet. No public API, dependency policy or CI workflow changes were made.

Run this runner with published dependencies or the local recovery SDK:

```bash
make sync
make test PYTEST_ARGS='tests/hybrid/test_native_executor.py -n 4 -q'
# Keep an already installed experimental SDK wheel when testing that lane:
UV_NO_SYNC=1 make test PYTEST_ARGS='tests/hybrid/test_native_executor.py -n 4 -q'
```

The original **11 capability probes passed** with the locked published SDK `0.2.153`,
CLI `2.1.273`, Temporal `1.33.0` and MCP `2.2.0`. They also pass with the newest
allowed published SDK `0.2.154`, its CLI `2.1.274` and Temporal `1.34.0`, and with
the sibling recovery SDK `0.2.162` using CLIs `2.1.273` and `2.1.285`. The new
runner needs no experimental SDK option. Lint passes in the locked, newest and
experimental environments. These passes include the explicit blocker probes.

Before the bounded replay extension, the full suite with the local recovery SDK passed **207 tests**, with one opt-in
benchmark skipped and strict Temporal handler/coroutine cleanup warnings
enabled. Wheel/sdist validation, isolated installation smoke tests, repository
conventions and all 101 tooling tests pass. The original live-hook and main-agent
MCP recovery probes remain part of that suite.

## Native Read/Edit and recoverable workspace experiment (2026-09-30)

The hybrid experiment now includes Claude Code's actual **Read** and **Edit**
tools. They retain their native schemas and implementations; no MCP replacement
tools are offered in these probes. A `PreToolUse` hook verifies the original
native ID and stored assistant call, requests a tool Activity through the
Workflow, then waits for that Activity's execution permit. The CLI executes the
tool. A post-tool hook stages the workspace snapshot, and eager transcript
mirroring captures the engine's original `tool_result` block. The shared test
store commits that result and snapshot in one SQLite transaction before the tool
Activity returns. Read/Edit execution is sequential because they share mutable
workspace state.

The Activity controls permission and records the outcome, but borrows its
executor from the live CLI Activity. On retry it can reuse a committed result;
it has no independent native executor for an unfinished call. Replacement first
joins the original accepted tool Update before registering a new CLI attempt,
so the outcome is acknowledged in Workflow history. This barrier also keeps
history replay valid when the Worker lost a tool Activity's completion.

The recoverable workspace is deliberately narrow: **one managed `note.txt`**,
with bytes, file mode, modification time and a snapshot version in the shared
SQLite test service. Tests SIGKILL the Worker, have the supervising harness stop
its recorded CLI,
delete the entire local workspace and Claude configuration directory, then
start another Worker with a fresh configuration. It materializes the committed
snapshot at the **same logical absolute path**, and loads the conversation from
session storage. This models a shared service plus disposable Worker disks;
it does not ship a production filesystem snapshotter or lease service. The
database fences workspace/result commits; a host supervisor must still enforce
exclusive session access and stop the old CLI before restoring the workspace.
The probes issue one native call per model response. Concurrent native batches
and subagents are outside this experiment.

| Native capability | Result | Reproduction in `tests/hybrid/test_native_workspace.py` |
|---|---|---|
| Real Read/Edit, one CLI, one Activity per original native ID | **supported** | `test_native_read_edit_are_individually_scheduled`; three primary model requests and two tool Activities |
| Original Read context survives Worker loss, allowing Edit without another Read | **supported** | `test_read_context_restores_before_native_edit`; earlier answer and transcript retained |
| Restore committed results/workspace before Activity completion or before transcript mirroring | **supported** | `test_native_worker_loss_reuses_committed_results`; Read/Edit, both phases; original IDs/content, one native execution per call, history replay |
| Reject a stale workspace/result writer after ownership changes | **supported** | `test_stale_native_workspace_writer_is_fenced` |
| Execute an original pending Read/Edit after its executor is lost | **blocked** | `test_uncommitted_native_execution_blocks_recovery[Read-before-execution]` and `[Edit-before-execution]`; ledger retained, replacement CLI startup refused |
| Finish an Edit after its local write but before committing its result/workspace | **blocked** | Same test's `[Edit-after-write]` case; uncommitted local bytes are discarded and the last committed snapshot is restored |
| Ordinary CLI resume re-dispatches an unfinished native call | **unsupported** | `test_ordinary_resume_interrupts_pending_native_calls`; with the opt-in recovery callback disabled, all three uncommitted phases produce an engine interruption result for the original ID, with no native execution hook |

Pending native suspension is refused and preserves the live process. These
hooks do not implement the MCP bridge's result-delivery parking protocol. The
blocked recovery cases do not ask the model to rediscover tools, synthesize
native output, or silently repeat the Edit. The checkpointed runner above
provides an alternative protocol for new single-call intents. Ordinary pending
calls accepted by this live-hook path still need a verified native recovery
protocol. Snapshot rollback here covers only the managed local file; it cannot
undo external effects from arbitrary Bash commands. Activity effects still
require idempotency. Subagents remain outside this design.

The ordinary-resume probe checks the engine's own interruption behavior, rather
than inferring a limitation from the prototype's refusal to fabricate a result.
The observed engine result is `[Request interrupted by user for tool use]` under
the original pending ID, including after an Edit already changed local bytes.
The callback-enabled blocker tests preserve the ledger and fail before launching
a replacement CLI when no original native result was committed. Neither path
claims recovery of an unfinished built-in operation.

Run the native probes with the local recovery SDK wheel installed:

```bash
UV_NO_SYNC=1 make test PYTEST_ARGS='tests/hybrid/test_native_workspace.py -n 4 -q'
```

All **13 native probes pass** with the local recovery SDK `0.2.162`, both
Claude Code CLI `2.1.273` and `2.1.285`, and with the newest allowed Temporal
`1.34.0` dependencies in an isolated worktree. Primary results use locked
Temporal `1.33.0` and MCP `2.2.0`. These passes include expected blocker
reproductions, not support for every recovery phase. With published SDK
`0.2.153`, the two baseline probes pass and the eleven recovery probes skip;
recovery still requires the sibling's experimental wheel. The earlier full
plugin suite passed **196 tests**, with one opt-in benchmark skipped and strict coroutine
cleanup warnings enabled. Lint, wheel/sdist validation, isolated installation
smoke tests and the 101 repository tooling tests pass. No plugin CI workflow or
public API was added.

## Main-agent recovery with the SDK fork (2026-09-30)

The current design covers **main agents only**. The
[SDK fork](https://github.com/brianstrauch/claude-agent-sdk-python/tree/feature/main-agent-recovery) implements an opt-in
`ClaudeAgentOptions.recover_pending_tool` callback. The hybrid test Activity uses
that callback on retry to reconnect original native tool IDs to the Workflow
ledger. Approval waits and already scheduled Activities remain in Temporal;
completed outcomes are reused. Missing request Updates are reconstructed from
stored native assistant calls, without an argument match or a replacement model
call.

Recovery completes pending tools **before the replacement CLI starts**. The SDK
selects the active conversation branch and all streamed assistant blocks sharing
its API message ID, retains delivered results, and obtains missing outcomes from
the callback. It conditionally appends ordinary `tool_result` entries and reads
the transcript back before materializing it for resume. No private deferral
markers are fabricated. This resolves the SDK/CLI handoff by presenting a
completed batch to the engine; the old Python MCP callbacks are not resurrected.

`SessionStore.append_if_unchanged` is a new optional upstream adapter operation,
required for this recovery mode. The test store implements it with a SQLite
transaction serialized with ordinary transcript writes. A changed head, failed
write, conflicting native ID, or failed readback prevents CLI startup. The host
still owns the session lease and attempt fencing. Independent read-only echo
callbacks opt into parallel recovery; the SDK default preserves call order.
External Activity effects continue to require idempotency.

The Workflow also records each completed user task's answer and transcript proof
through an Update. A retry partway through a multi-task burst skips acknowledged
tasks and retains their original answers. CLI checkpoints and Continue-As-New
remain internal test helpers; production public APIs and packaged registry
dependency bounds are unchanged. This checkout pins the fork through
`tool.uv.sources` and `uv.lock`.

| Main-agent capability | Result | Reproduction |
|---|---|---|
| Recover original calls before tool scheduling, after completion/before delivery, and in a partially delivered batch | **supported** | `test_cli_loss_recovers_original_main_calls`, resume and explicit stored fork |
| Replace a Worker before its request Update reaches Temporal | **supported** | `test_replacement_worker_recovers_main_calls[...-before-update]` |
| Keep approval decisions, reject the correct call, reuse scheduled Activities and completed outcomes | **supported** | Remaining Worker-loss cases; original IDs, zero duplicate schedules, two primary model requests |
| Preserve earlier task answers when a later task loses its Worker | **supported** | `test_worker_loss_during_second_task_preserves_first_answer`; Workflow history replay verified |
| Conditional persistence, lost write acknowledgment, callback cancellation and failed readback | **supported** | Sibling `tests/test_main_agent_recovery.py`, both asyncio and Trio |
| Suspend pending approvals, running Activities, completed/undelivered results and partial batches using a controlled process stop | **supported** | `tests/hybrid/test_pending_suspension.py`; original IDs and outcomes retained |
| Recover a stopped session on another Worker, repeat suspension, and roll over after outstanding outcomes settle | **supported** | Worker replacement, repeated-stop, lost-Activity-completion and Continue-As-New cases in the suspension tests |
| Graceful engine suspension through ordinary cancellation/EOF | **blocked** | No verified engine pause handshake; the original resume probes do not restore unresolved callbacks. The supported host stop below avoids this path |

Pending suspension is now a **test-only controlled hard stop**. Partial-message
events establish that the main model response ended with `tool_use`; incomplete
streams cannot be suspended. The Activity waits until every callback's request
Update has reached the Workflow. It then parks callback result delivery, checks
every native call and delivered outcome against storage, and calls the
subprocess's `kill()` before cancelling callbacks or closing stdin. After process
exit and teardown it requires an unchanged transcript readback. This ordering
prevents cancellation outcomes from reaching the old CLI and adds no fabricated
transcript marker or native pause command.

The Activity records a pending checkpoint through a Workflow Update and returns
a stopped receipt. Its heartbeat loop finishes too. Existing approval handlers
and tool Activities continue in Temporal; new requests from the stopped attempt
are rejected. `HybridWorkflow.resume_pending` records the intention to resume,
and the Workflow waits for all outstanding outcomes and handlers before starting
a replacement CLI Activity. The SDK recovery callback then persists original
results before CLI startup. Continue-As-New uses the same stopped checkpoint and
completed outcomes after handlers finish. If the Worker dies after checkpoint
acknowledgment but before Activity completion, the retried Activity returns that
receipt without creating another CLI.

Missing call storage, an unfinished stream, child transcripts, or failed storage
reads refuse suspension and preserve the live process. A changed post-stop
readback never produces a successful suspension receipt. The host still owns
exclusive session access and orphan cleanup after Worker SIGKILL. The experiment
does not introduce a production suspension API or idle policy.

Worker-loss recovery tests model a supervising host stopping the old CLI after
Worker SIGKILL. Python cannot guarantee that teardown after SIGKILL itself; the
existing unsupervised orphan probe remains an explicit limitation. The new
recovery code rejects subagent execution and does not attempt child restoration.
Historical subagent probes below remain baseline evidence, outside this design.

Validation uses a **non-editable local SDK 0.2.162 wheel**, built from the sibling
feature branch (the same source tree now shared as `3ac4b25`), with the plugin's
other committed locked dependencies.
After adding pending suspension, the full plugin suite passed **183 tests**, with
the optional benchmark skipped, using CLI **2.1.273**. All **22 pending-suspension
cases** passed with both CLI **2.1.273** and the sibling SDK's pinned CLI
**2.1.285**. The earlier 15 main-agent recovery cases also passed with 2.1.285.
A disposable newest-policy lane selected Temporal **1.34.0** and passed all
**43 suspension/recovery/Workflow tests** and lint (Python 3.14.4, CLI 2.1.273),
leaving the committed lock unchanged. The upstream SDK suite passed **1,633
tests** (6 optional skips), including 46 recovery cases across asyncio and Trio.
Lint, repository conventions, 101 tooling tests, wheel/sdist checks and isolated
smoke installs passed. The earlier cleanup warning came from the negative
two-live-agent test failing during Workflow construction, before Temporal awaited
its allocated run coroutine. That test Workflow now captures the constructor's
validation error and reports it from its run method; the warning suppression was
removed, and unawaited coroutines and unraisable exceptions now fail the suite.
The isolated reproduction and the full **183-test suite** complete pytest
teardown cleanly with warnings treated as errors. The published dependency-floor
lane (Temporal **1.33.0**, Claude SDK **0.2.153**) also passed lint and **146 tests**
with clean teardown; 38 cases skip without the local recovery feature or the
opt-in benchmark. Tests use the real CLI, the strict deterministic local Messages
API and the pinned Temporal dev server without provider credentials.

To reproduce from this branch, no sibling checkout is required:

```bash
cd python/claude_agent_sdk
make sync
.venv/bin/python tests/hybrid/install_cli.py
make lint
make test PYTEST_ARGS='tests/hybrid/test_main_recovery.py tests/hybrid/test_pending_suspension.py -n 4 -q'
make test PYTEST_ARGS='-n 4 -q'
```

On Windows, run the helper with `.venv/Scripts/python.exe` instead.

`make sync` installs SDK `0.2.162` from fork commit `3ac4b25`. Git installations
contain no bundled Claude Code binary, so the helper extracts CLI **2.1.273**
from the published SDK **0.2.153** wheel into this plugin's virtualenv `bin`
directory (`Scripts` on Windows). It preserves the installed fork SDK and the
machine's system Claude installation. The SDK finds that binary on the PATH
provided by `make test`. This is the same SDK source tree and CLI combination
used for the experimental recovery results above. Built wheel/sdist metadata
continues to use the published registry dependency; the fork pin applies to
development through uv. The pytest controller also runs the helper automatically
when a source SDK has no bundled CLI and the virtualenv CLI is missing, so the
shared CI test target needs no extra setup or plugin-specific job.

Sharing validation (2026-10-01): the pinned fork passed **1,633 SDK tests** with
six optional skips. The integration's full locked suite passed **226 tests**, with
only the opt-in benchmark skipped. Automatic CLI provisioning and partial-batch
recovery passed after deleting the virtualenv CLI. Lint, repository conventions,
wheel/sdist checks and isolated smoke installs passed.

For further SDK edits in a sibling checkout, use the existing wheel installer:

```bash
# From python/claude_agent_sdk; first install the pinned fork and test CLI.
make sync
.venv/bin/python tests/hybrid/install_cli.py
.venv/bin/python tests/hybrid/install_sdk.py \
  --sdk-dir /absolute/path/to/claude-agent-sdk-python --cli .venv/bin/claude
# On Windows pass --cli .venv/Scripts/claude.exe instead.
# UV_NO_SYNC keeps uv run from replacing the local wheel with the committed lock.
UV_NO_SYNC=1 make lint
UV_NO_SYNC=1 make test PYTEST_ARGS='tests/hybrid/test_main_recovery.py tests/hybrid/test_pending_suspension.py -n 4 -q'
UV_NO_SYNC=1 make test PYTEST_ARGS='-n 4 -q'
# Test another pinned binary without changing the installed SDK or dependency lock:
HYBRID_CLI_PATH=/absolute/path/to/claude UV_NO_SYNC=1 \
  make test PYTEST_ARGS='tests/hybrid/test_main_recovery.py tests/hybrid/test_pending_suspension.py -n 4 -q'
# Restore the pinned fork dependency afterwards:
make sync
```

The recovery tests run with the pinned fork. They skip when testing the
published baseline SDK, which lacks the opt-in API.
`tests/hybrid/test_recovery.py` and the original Worker-loss probes continue
to verify the ordinary resume limitation with recovery disabled. The SDK change
and the test integration are separate feature history; no plugin-specific CI,
public suspension configuration, idle threshold or release-version change is
introduced. A main-agent hybrid runner is feasible under this persistence and
ownership contract; a production refactor is separate work.

## Hybrid feasibility experiment (published SDK baseline, 2026-09-30)

The test-only prototype in [`tests/hybrid`](tests/hybrid) keeps one
`ClaudeSDKClient` and its bundled CLI alive for an agent burst. Its low-level MCP
server reads `claudecode/toolUseId` from request metadata, verifies the ID, tool
name and arguments against the observed assistant call and the stored transcript,
then awaits a Workflow Update. The Workflow records approvals, tool Activity
scheduling and outcomes under the native ID. The echo fixture retains its full
JSON Schema and truthful read-only/idempotency annotations; it does not add IDs
to model arguments or use an argument match to recover a regenerated call.

Each CLI Activity attempt registers before accepting callbacks. Older attempts
are fenced. Duplicate requests share an Activity/outcome, conflicting IDs fail,
and eager transcript mirroring must reach the transactional test store before a
tool can be scheduled. The store supports parent transcripts and child subkeys.
Completed checkpoints are acknowledged through Updates in Workflow history.
Continue-As-New waits for handlers and for the CLI Activity's teardown; it carries
the ledger, answers and checkpoint to the next run. These helpers add no installed
public API, idle threshold, suspension configuration or CI workflow.

**Published baseline assessment:** Live execution and
completed-task suspension work, but the published engine does not restore
unresolved MCP callbacks, including child calls. While an approval or batch is
pending, the prototype refuses suspension and leaves the CLI/callback alive.
If its CLI or Worker is lost with uncheckpointed calls, the Workflow preserves
its ledger and fails closed rather than silently reissuing the work.

| Capability | Result | Reproduction / scope |
|---|---|---|
| Reuse one process across tool rounds and messages | **supported** | `test_reuse_and_completed_task_suspension`; benchmark: 20 rounds per process |
| Three native eligible calls execute concurrently with their original IDs | **supported** | `test_native_batch_overlaps_without_rediscovery`, `test_temporal_batch_and_history_replay`; two model requests for a three-call batch, no rediscovery |
| Route approval/rejection to the correct call; keep a pending callback alive | **supported** | `test_approvals_duplicates_conflicts_and_stale_attempts`, `test_pending_approval_preserves_process_and_callback` |
| Defer a single call using PR #33's engine mechanism | **supported** | `test_existing_single_call_approval_deferral`; independent of suspending a live MCP callback |
| Stop after a completed task, restore context in a new CLI, Continue-As-New | **supported** | `test_reuse_and_completed_task_suspension`, `test_continue_as_new_after_stopped_cli_and_finished_handlers` |
| Recover an acknowledged checkpoint after the Worker loses the Activity reply | **supported** | `test_worker_loss_after_acknowledged_checkpoint_reuses_outcomes`; recorded answers/outcomes reused without another tool schedule or CLI start |
| Restore pending main-agent calls or a partially completed batch | **blocked** | `test_pending_recovery_probe`: CLI killed before scheduling, after completion/before delivery, or after one batch result; tests both resume/fork and prompt/result injection |
| Replace a Worker with uncheckpointed calls | **blocked** | `test_replacement_worker_preserves_ledger_and_fails_closed`: same three phases, a different Worker/config folder; ledger and completed outcomes survive, continuation is refused, no duplicate Activity schedules |
| Foreground/background subagent tools and overlapping parent/child requests | **supported** | `test_subagent_tools_are_temporal_activities`, `test_background_parent_child_requests_overlap` |
| Store/materialize completed child transcripts | **supported** | `test_subagent_live_and_stored`; `list_subkeys` and child `load` calls verified on a new client/config folder |
| Restore a pending child call after process loss | **blocked** | `test_child_process_loss_does_not_restore_pending_call`, foreground and background, both pending and completed-before-delivery; child transcript is loaded but its original callback and undelivered outcome are not restored |
| Delayed/failed writes, duplicate Updates, conflicting native IDs and stale attempts | **supported** | `test_no_tool_before_transcript_storage`, `test_inconsistent_native_id_is_rejected_before_execution`, `test_new_registration_fences_an_actual_older_attempt` and approval tests; no tool executes without the transcript proof |
| Cancellation tears down the CLI and callback tasks | **supported** | `test_cancellation_stops_cli_and_pending_callbacks`; heartbeats cover registration as well as active work |
| Guarantee orphan cleanup after Worker SIGKILL without supervision | **unsupported** | Worker-loss tests: Python teardown cannot run after SIGKILL; the test harness explicitly kills only the recorded child PIDs. A production implementation needs process supervision |
| Exactly-once external side effects | **unsupported** | `test_retry_after_side_effect_uses_stable_idempotency_key`: two Activity executions, one local effect because the test service deduplicates the stable key; external services must implement equivalent idempotency |

The main recovery blocker is observable in the model's next request. For an
unresolved native call, the engine supplies an error result containing
`[Request interrupted by user for tool use]` instead of restoring the callback.
A result already delivered and mirrored before a partial-batch crash survives;
a completed result held before MCP delivery does not. Sending normal
`tool_result` blocks under the original IDs on resume does not replace those
interrupted results. Forking through `fork_session_via_store` has the same
limitation. The probes do not invent deferral markers or edit transcripts to
claim a restore protocol. Child subkeys are discoverable and materialized, but
that alone does not restore pending child execution. The recovery tests assert
these observed blockers explicitly, rather than skipping or marking them xfail.

Primary measurements use the committed lock: Claude Agent SDK **0.2.153**,
bundled Claude Code **2.1.273**, MCP **2.2.0**, Temporal **1.33.0**, Python
**3.14.7**, macOS arm64. `make sync-latest` with the repository's two-week cooldown
selects the same Claude/MCP versions and Temporal **1.34.0**; capability tests
are repeated in a separate disposable worktree so the committed lock is retained.
No provider credentials are required: every engine test uses the existing local
strict Messages API and the pinned Temporal dev server.

The benchmark runs each mode three times, with 20 tool rounds each. Latency is
the interval between successive primary model requests (tool dispatch/result,
storage and next model turn); it excludes the initial process start. Total wall
time includes startup and teardown. Counts cover agent CLI starts and primary
model requests; they exclude version checks and incidental engine requests.
Both modes use real Temporal Workflows and Activities. The hybrid tool also
records a small SQLite idempotency fixture. There is no timing pass/fail threshold,
and these deterministic local-model numbers do not predict provider latency.
Raw samples are in [`tests/hybrid/benchmark-locked.json`](tests/hybrid/benchmark-locked.json).

| Mode | Trial | CLI starts | Model requests | Median round (ms) | p95 round (ms) | Wall time (s) |
|---|---:|---:|---:|---:|---:|---:|
| Segment | 1 | 21 | 21 | 359.3 | 376.1 | 7.68 |
| Hybrid | 1 | 1 | 21 | 151.4 | 169.2 | 3.41 |
| Segment | 2 | 21 | 21 | 368.6 | 383.5 | 7.76 |
| Hybrid | 2 | 1 | 21 | 149.4 | 169.2 | 3.42 |
| Segment | 3 | 21 | 21 | 367.5 | 385.6 | 7.76 |
| Hybrid | 3 | 1 | 21 | 152.5 | 171.1 | 3.51 |

Verification completed locally: `make sync`, `make lint`, the full locked suite
(144 passed, benchmark skipped), all capability tests in the newest allowed lane,
and the opt-in benchmark. Two additional completed-child recovery cases then
passed in both lanes (nine subagent cases total). Repository conventions,
101 tooling tests, wheel/sdist checks and clean-environment smoke installs passed.
This initial run emitted an unawaited-Workflow-coroutine RuntimeWarning during
pytest cleanup; the test cleanup fix described above resolves it. Validation
covers macOS/Python 3.14 on this machine; the other CI platforms/runtime versions
were not run here.

Run from `python/claude_agent_sdk`:

```bash
make sync
make lint
make test PYTEST_ARGS='tests/hybrid -n 4 -q'
# Print detailed blocker evidence (including native IDs and model-visible results):
make test PYTEST_ARGS='tests/hybrid/test_recovery.py tests/hybrid/test_subagents.py tests/hybrid/test_worker_loss.py -n 0 -s -q'
HYBRID_BENCHMARK=1 HYBRID_BENCHMARK_OUT=/tmp/hybrid-benchmark.json \
  make test PYTEST_ARGS='tests/hybrid/test_performance.py -n 0 -s -q'
# Run sync-latest in a disposable checkout, then repeat the capability command.
```

The published baseline did not establish pending-call recovery. The current
main-agent design and its suspension limits are described above.
