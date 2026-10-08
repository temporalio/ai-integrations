# Durable Claude Agent SDK agents on Temporal

> ⚠️ **Experimental.** The API may change.

Temporal integration for Anthropic's [Claude Agent SDK](https://code.claude.com/docs/en/agent-sdk/overview), published as [`temporalio-claude-agent-sdk`](https://pypi.org/project/temporalio-claude-agent-sdk/) and imported as `temporalio.claude_agent_sdk`.

- **Every durable tool call Claude makes is its own Temporal Activity.** Finished calls never run again after a crash, retries follow your retry policy, and the Activity ID (`tool-<tool_use_id>`) doubles as an idempotency key for the systems a tool touches. When Claude calls several durable tools in one message, they run at once.
- **Claude Code's commands too.** Bash commands and MCP tool calls run as their own Activities by default, including calls from native subagents, with the same approval policy. File tools and `WebFetch` still run inside the step.
- **Tools can wait for a human.** Mark a tool `needs_approval=True` and the agent waits, for minutes or weeks, until someone approves or rejects the call.
- **Crashes resume cleanly, on any Worker.** The conversation lives in the Workflow, like any other Workflow state: for the conversation, Workers share nothing but the Temporal server. Each model step ends at a checkpoint that Temporal records, and a step that runs again (after a crash, a timeout, or on another machine) starts from what the Workflow committed, so nothing a failed attempt added to the conversation reaches Claude.

## Install

```bash
uv add temporalio-claude-agent-sdk
```

It depends on `claude-agent-sdk>=0.2.153`, which bundles Claude Code 2.1.273 or newer (see [Requirements](#requirements-and-limits)).

**Windows.** claude-agent-sdk 0.2.157 and 0.2.160 to 0.2.163 publish no Windows wheel (from 0.2.160, the bundled Windows engine is over PyPI's 100 MiB file limit), and their source package has no engine, so the plugin skips those versions on Windows. If a newer release misses its Windows wheel too, pin `claude-agent-sdk` to a version that has one, or install Claude Code natively and pass `cli_path` to the runner.

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
from temporalio.claude_agent_sdk import ClaudeAgentPlugin, ClaudeAgentSdkRunner

worker = Worker(
    client,
    task_queue="agents",
    workflows=[RefundAgent],
    activities=[look_up_order, issue_refund],
    plugins=[ClaudeAgentPlugin(ClaudeAgentSdkRunner(cwd="/srv/agent"))],
)
```

The runner refuses a `cwd` for which Claude Code and the SDK would compute different session keys (for example with decomposed Unicode or emoji).

Pass the plugin to the Worker, or to the Client the Worker is built from, not both. Each segment starts a Claude Code process: 0.7 to 0.9 seconds and up to about 270 MB of memory each (measured on Linux with an instant local model), so cap parallel segments with the Worker's `max_concurrent_activities`.

Other [`ClaudeAgentOptions`](https://code.claude.com/docs/en/agent-sdk/python) go in the runner's `extra_options` (for example `permission_mode`, `agents`, `hooks`, `setting_sources`, `thinking`). `env`, `mcp_servers` and `allowed_tools` are merged with the plugin's own, and `system_prompt` (a string, or a preset such as Claude Code's own prompt) is the default for agents that set none. Options the agent or the plugin sets (`model`, `tools`, `max_turns`, `cwd`, `settings`, session and resume options, and the same engine flags in `extra_args`) are refused. The runner sets `permission_mode="default"` unless you pass one: since Claude Code 2.1.285, a run without one uses auto mode when telemetry is off or the provider is Bedrock, Vertex or Foundry, and auto mode asks the model whether each tool call may run. Tools of the MCP servers you pass here run in tool steps by default (`tool_activities` includes `mcp__*`), and each segment and tool step starts (or connects to) the servers again. The plugin decides on its `mcp__durable__` tools and on the tools in `tool_activities`: a `PreToolUse` hook you pass here that answers for one of them stops the task with an error that says so (tested), and hooks from settings files you load must not answer for them either. `enable_file_checkpointing` is refused: it cannot work with the session store the runner always gives the SDK.

`approvers` checks the name the caller passes. It is not authentication: control who may send Updates with Temporal's own access control.

## Claude Code's own tools

Claude Code's built-in tools are off unless you enable them with `builtin_tools`. The tools in `tool_activities` run as their own Activities, like durable tools: by default `("Bash", "mcp__*")`, that is Bash, and the tools of MCP servers you give the runner in `extra_options`. The others run inside the segment.

```python
self.agent = DurableClaudeAgent(
    builtin_tools=["Bash", "Read", "Grep"],
    tool_approvals=["Bash"],  # a human approves each command first
    tool_activity_retry_policy=RetryPolicy(maximum_attempts=1),
)
```

- Each call is an Activity `run_claude_tool_step` with ID `tool-<tool_use_id>`, so a step that runs again (after a crash or a timeout) never runs the command again. Tested: the model call after a command hangs past the step's timeout; as its own Activity the command ran once, inside the segment it ran twice.
- The Activity resumes Claude Code at the call and runs exactly that call, so Claude gets Claude Code's own result. Afterwards the engine asks the model to go on; a stand-in on the Worker (on 127.0.0.1) answers instead, so a tool call costs no model call, whatever provider the Worker uses (tested with Bedrock configured). The stand-in answers only its own runner's engines, which get a random key; another program on the machine gets 401 (tested). About 1.4 seconds per call (measured on Linux with Claude Code 2.1.273).
- `tool_approvals` makes calls wait for a decision, like `needs_approval` (`pending_approvals()` shows the command). A rejected command never runs. Each pattern must fall within `tool_activities`.
- A command that fails is a result for Claude, not a failed Activity. The Activity fails only when the step breaks before it has the command's result (a Worker dies while the command runs, a timeout). Temporal then retries it with `tool_activity_retry_policy`, by default without limit, so the command can run again; `maximum_attempts=1` gives "at most once". Once the step has the result, it keeps it, even if the engine fails afterwards (tested).
- Output over about 30 KB: Claude Code shows Claude a preview and saves the rest to a file in its temporary folder, which the step removes. Claude also gets the last 4 KB, which the step reads only from Claude Code's own folder for the session, so a command cannot make it read another file (tested).
- Several calls in one message: the first call that runs as an Activity (a durable tool, or a Claude Code tool in `tool_activities`) pauses the segment, and the durable calls after it run with it. A Claude Code call after it (a second Bash call, say) keeps its denial, and Claude calls it again. Read-only calls that Claude Code runs together, as one batch, have their hooks run at the same time: one of them pauses the segment, not always the first, and the others keep their denial (tested).
- Commands run on the disk of the Worker that runs the tool step, in the runner's `cwd`, and each command starts there: a `cd` does not carry over to the next command (each step is a new Claude Code run), so the plugin makes that the rule inside segments too, and tells Claude (tested). Give Workers the same files (a shared disk), or one Worker per agent.
- A command sees the Worker's own environment. The step points Claude Code at the stand-in model, then gives each Bash command the Worker's own values back, through `CLAUDE_ENV_FILE` (tested). `PowerShell` commands get the step's values instead: their `ANTHROPIC_BASE_URL` points at the stand-in. Bash commands see none of the plugin's own variables (`TCA_*`), in tool steps and in segments (tested).
- File tools (`Read`, `Edit`, `Write`, ...) stay in the segment: Claude Code checks a file tool call again when the next step delivers its result, so an `Edit` run in its own Activity happens, but Claude is told it failed, because the file changed since Claude read it (tested). So do tools that call the model themselves (`WebFetch`). Inside the segment, a tool can run again when the segment runs again: give such tools work that is safe to repeat.
- Native foreground and background subagents (`Agent`) can call durable tools and supported engine tools in `tool_activities`. The Workflow accepts each original child tool ID once, applies approvals, and runs its Activity. Duplicate requests share the recorded outcome; conflicting arguments and stale attempts are refused. Other tools run inside the subagent as usual.
- `PowerShell` uses the same path as Bash; its native delivery tests run on Windows. Enable the engine's PowerShell tool with `CLAUDE_CODE_USE_POWERSHELL_TOOL=1` in the runner's `env`. MCP tool names take patterns, such as `mcp__github__*`. `tool_activities=()` keeps engine tools inside the segment; custom durable tools still run as Activities.

Native child tools are enabled by default when a Workflow runs the agent. Enable the `Agent` built-in to let Claude delegate. Each live child keeps its segment Activity open while waiting for tool Activities. Set `segment_timeout` long enough for child approvals and tool retries, and keep heartbeats enabled. Give the Worker enough Activity slots, or route tools to another queue:

```python
agent = DurableClaudeAgent(
    builtin_tools=["Agent", "Bash"],
    tools=[activity_as_tool(look_up_order, task_queue="claude-tools")],
    tool_activity_task_queue="claude-tools",  # Bash, PowerShell, external MCP
)
```

Register `ClaudeAgentPlugin` on the tool Worker too. Its runner needs the same tool configuration and access to the files its commands use. Custom Activities may return `ToolOutcome` to preserve error flags and native text/image blocks. Child shell Activities replay the exact recorded call against a local model; the live child runs only a renderer of that recorded outcome. The segment verifies the result under the original child ID before committing. Parent pauses wait for child delivery to drain.

## Where the conversation lives

By default, in the Workflow. Each step reads the committed conversation with a Query on its own Workflow, a page at a time, so the conversation is not copied into every step's input. The step resumes Claude Code from it in memory and returns only what it added (about 5 to 9 KB per step with small tool results, measured with Claude Code 2.1.273); the Workflow splices that in. So:

- **No storage to run.** For the conversation, Workers share nothing but the Temporal server. Tested: each step of a conversation run by a new runner with its own working directory and config folder, and a Worker process killed after the refund, replaced by one with its own config folder that never saw the conversation.
- **Retries are clean by construction.** Every attempt starts from the conversation the Workflow committed; nothing an unfinished attempt added to it reaches Claude (files its tools wrote stay).
- **No copy of the conversation stays on the Worker's disk** after a step ends normally. Claude Code writes a new session to its config folder (`~/.claude/projects`, with tool outputs too large to show Claude in full); the runner removes the copy it created, never a session that was there before. Resumed steps run in a temporary folder the SDK removes. A Worker that is killed can leave a copy behind.
- **One payload per step.** Without External Storage, what a step adds must fit in one Temporal payload (2 MB by default), or the step fails with a message that says so, instead of retrying forever. The results of one message's calls must also fit in the next step's input together: if they do not, the largest are replaced by a note that tells Claude so (they stay in the history, as the tools' results; tested). Results close to 2 MB need [External Storage](#large-tool-results), which also carries steps and the conversation through Continue-As-New.
- **Reading costs a Query per page.** Pages hold up to 508 KiB, under the 512 KiB where Temporal starts to log a warning about a payload's size; with External Storage, just under its threshold (256 KiB by default), so they stay in the Query's response instead of going to the store at every step. An entry larger than that is a page of its own, the same payload at every step, so a content-addressed store (such as Temporal's S3 driver) keeps it once. Measured on a local dev server with 40 tool results of 100 KB (a 3.8 MB conversation), the last step read it in about 180 ms with 8 Queries, and in about 310 ms with 20 Queries with External Storage. On Temporal Cloud, every Query is an [Action](https://docs.temporal.io/cloud/actions); a higher External Storage threshold means fewer pages. A Workflow that is not in the Worker's cache (more open Workflows than the Worker's `max_cached_workflows`) replays its history before it answers the step's first Query.
- **Child conversations are kept too.** Native child transcripts are committed alongside the parent, read through the same paged Query, and carried in `AgentState.child_conversations` through Continue-As-New. Segments return only changed entries. With a session store, `child_subpaths` tells the next Worker which child transcripts to restore.

To keep conversations outside Temporal instead, give the runner a Claude Agent SDK `SessionStore` that every Worker can reach (`ClaudeAgentSdkRunner(session_store=...)`). `FileSessionStore` is for tests and one machine; for production, implement `SessionStore` on your database or object storage (the Claude Agent SDK repository has example stores for S3, Redis and Postgres). The store keys sessions by the engine's working directory, so give every Worker the same `cwd`. Every segment then reads the whole session twice (the SDK to resume it, the runner to record the checkpoint), so reading grows with the session's length; a segment that runs again reads it once more and writes a copy, and the earlier copy stays in the store, so remove old sessions with your store's own retention.

Every Worker of a task queue needs the same choice. A Worker set up the other way refuses the step (a model step or a tool step) with a message that says how to fix it, and Temporal retries the step until a matching Worker picks it up (tested).

## Long-running agents

A Workflow's history holds at most 51,200 events or 50 MB (Temporal's default limits), and every durable tool call adds about 12 events. So one run fits about 4,000 tool calls, and fewer when results are large. Continue-As-New starts a fresh history, and carries the agent's state, with its conversation, in the new run's input.

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

- When the server suggests it (`is_continue_as_new_suggested()`, which counts history length, history size and Updates), the agent continues as new before its next step: between two tool calls, or before a task's first step. It lets running Update and Signal handlers finish and continues as new with `[prompt, state]`. The new run calls `agent.run(prompt)` again; the agent sees the unfinished task in its state and continues it, so Claude gets the prompt once.
- It is off by default, because the new run's arguments must match your run method. For other arguments, pass `continue_as_new_args`, a function that builds them from the `AgentState`. `continue_as_new_after_events` uses a fixed history length instead of the server's suggestion.
- Call `agent.run()` from the Workflow's run method, not from a separate asyncio task, and let no handler wait for the task to finish: continuing as new waits for every handler to finish. With `auto_continue_as_new`, a call from a handler fails at once: an Update handler fails its Update, a Signal handler fails the Workflow.
- **Chats.** Calling `agent.run(message)` again continues the same Claude session, so Claude remembers earlier turns. With `auto_continue_as_new`, a message can be handed over before its first step (tested); `continue_as_new_args` carries your own state, such as an inbox. At the start of a new run, if `agent.busy` is true, `await agent.run()` finishes the task that was interrupted. To continue as new between messages instead, when no handler is waiting for an answer, call `await agent.continue_as_new()` from the run method if `agent.should_continue_as_new()` is true.
- `max_segments` caps a task across all its runs (default 50). The task stops before it runs a tool whose result could no longer reach Claude. Use `None` for tasks that may take thousands of steps.
- **Long conversations need External Storage.** Without it, the new run's input is one payload (2 MB by default), and a conversation outgrows it after a few hundred steps with small results. Then `should_continue_as_new()` stays false, with a warning in the Worker's log, and the agent keeps going in its run instead of failing to start the next one; `continue_as_new()` raises an `ApplicationError` that says so. With `auto_continue_as_new`, the agent checks before every step: when the next step could take the history past Temporal's limits (51,200 events or 50 MB, or `continue_as_new_after_events` if that is higher), it continues as new early if it can, and otherwise fails the task with an `ApplicationError` that says what to change, instead of the server terminating the Workflow (tested for both limits on a server with low limits; measured with 1 MB results: the task stopped at 46 MB). The guard assumes the default 50 MB size limit, also on a server that allows more. With External Storage, the conversation moves on past the 2 MB payload limit, up to your storage driver's own limit (tested with two 1.9 MB results). A runner with a session store keeps the state small.
- **Continue-As-New moves the whole conversation in one Workflow task.** The Workflow keeps each entry as JSON text, so this is quick: measured with External Storage on a local dev server, the handover's Workflow task took 0.2 s with 10 MB of conversation, 1.1 s with 40 MB and 3.0 s with 100 MB (with writing it to the store), and none failed. The Python Worker fails a Workflow task whose code runs for 2 seconds (deadlock detection), and such a task fails again on every retry, so keep conversations well under 100 MB, or use a session store.
- **Start with every argument.** Temporal applies a Workflow's argument types only when the caller passes as many arguments as `run` declares. If `run` has other typed arguments besides `state`, start the Workflow with `state=None` included, or those arguments arrive as plain dicts.

## Failures and cancellation

- A task fails with an `ApplicationError` at `max_segments`, when a step reports an error that running it again cannot fix (Claude Code's `max_turns` or `max_budget_usd`, a broken pause, or a Messages API request that would be refused again: invalid (400), an unknown model (404), too large (413), or a prompt too long for the model), or if the engine hands back a tool call that already ran. Other engine and API errors, such as a low credit balance, a rejected API key, rate limits or overload, fail the step's Activity, and Temporal retries it with `segment_retry_policy` (by default without limit, so the step continues once the cause is fixed; with a session store, each retry copies the session). When the Activity fails for good, the task fails with an `ActivityError`. Catch `temporalio.exceptions.FailureError` for both.
- After a failed task, the agent can take the next one: it continues from the last checkpoint, and a tool call Claude was still waiting for gets an error result, delivered with the next prompt.
- Cancelling the Workflow cancels the running step or tool call, and the Workflow ends as cancelled. By default Temporal does not wait for a cancelled Activity. `activity_as_tool(..., cancellation_type=workflow.ActivityCancellationType.WAIT_CANCELLATION_COMPLETED)` waits until the call finishes or acknowledges the cancellation through a heartbeat, so the history records what really happened; `segment_cancellation_type` does the same for a running step or tool step.
- A running step or tool step hears of a cancel (or of its own timeout) with its next heartbeat: at most 0.8 × `segment_heartbeat_timeout` later (30 seconds by default, so up to 24 seconds). From then on the engine cannot start a built-in tool while the SDK stops it, which takes up to about 10 seconds (a durable call only pauses the run).
- If the Worker process dies (a crash, an out-of-memory kill, SIGKILL), its engines end with it. On Linux the engine starts through a small launcher that asks the kernel for SIGTERM when the Worker's thread that started it ends (the event loop's, which lives as long as the Worker), then becomes the engine (about 20 ms per engine start); on SIGTERM Claude Code ends its commands. On macOS (and on Linux without `prctl`) the launcher stays as the engine's parent and sends it SIGTERM once the Worker is gone (it checks five times a second). On Windows every Claude Code process the SDK starts joins, within milliseconds, a job object that ends it, and the processes it starts, with the Worker process; the Worker process and its other child processes are not in it. Until the engine is gone, the hook denies every built-in call: the Worker's lock on the step's folder is free. Tested by killing the Worker during a model call that never answers: within seconds no process it started was left (before, Claude Code waited for the model, then went on with its turn). Where the launcher cannot run (a temporary folder mounted noexec) or files cannot be locked, steps still run, with a warning.

## Large tool results

Every tool result is stored in the Workflow's history three times: as the tool Activity's result, in the next step's input, and in that step's output, as part of the conversation (twice with a session store). A single payload over 2 MB cannot be recorded at all (the SDK stops it with `[TMPRL1103] Attempted to upload payloads with size that exceeded the error limit`), and results of a few hundred KB fill the 50 MB history quickly.

The plugin works with Temporal's [External Storage](https://docs.temporal.io/external-storage) unchanged. Payloads over a threshold (256 KiB by default) go to your store, and the history keeps small references, so Claude can read a conversation past the 2 MB limit, up to your storage driver's own limit (tested: two 2 MB results through the real engine). Configure it on the Client, as for any Temporal application; Workers built from that Client use it too:

```python
from temporalio.converter import DataConverter, ExternalStorage

client = await Client.connect(
    "localhost:7233",
    data_converter=DataConverter(external_storage=ExternalStorage(drivers=[my_s3_driver])),
)
```

Use the same External Storage on every Client that starts or queries these Workflows. Measured on a local dev server at Temporal's default limits, with 40 tool results of 1 MB and a store that names each payload by its SHA-256 (as Temporal's S3 driver does), so a page read again at every step is kept once:

| | Without External Storage | With External Storage |
|---|---|---|
| Conversation in the Workflow (default) | terminated after 17 results ("Workflow history size exceeds limit"); with `auto_continue_as_new`, the task failed in good order after 16 | one run, 83 KB of history, 126 MB in the store |
| Session store | terminated after 25; with `auto_continue_as_new`, 20 runs | one run, 89 KB of history, 43 MB in the store |

Temporal logs a warning (`TMPRL1103`) for each payload over 512 KiB, such as a 1 MB tool result, or the page that holds it.

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
- Measured on a local dev server, three runs of 203 events: 99 to 112 ms median and 164 to 169 ms at the 95th percentile, from event to subscriber. Workflow Streams is built for UIs and progress, not real-time voice.
- Continue-As-New carries the stream in the new run's input, next to the agent's state. Without External Storage, the stream shrinks to fit, measured as the new run's input carries it, so a large tool result waiting to be delivered still fits under Temporal's 2 MB payload limit (tested); the rest of the input is measured before any codec, which can only overstate it, so the stream may keep fewer events than would fit.
- **Every subscriber poll is an Update**, and a Workflow accepts at most 10 Updates in flight and 2,000 per run (Temporal's defaults). Idle subscribers hold Updates in flight, so with many direct subscribers an Update approval can be refused (tested: with 12, it failed with `RESOURCE_EXHAUSTED`). Approve by Signal instead (`decide()` ignores invalid decisions there), or fan events out to viewers through one subscriber in your backend. For long runs with live output, turn on `auto_continue_as_new`: the server counts the polls toward its suggestion.
- Create the agent while the Workflow is being initialized (in its `__init__`), where Workflow Streams registers its handlers; later, the constructor raises.

## How it works

The agent loop runs in *segments*. A segment is one Activity that runs the Claude Code engine from a prompt (or a tool result) until Claude calls a tool that runs as an Activity (a durable tool, or a Claude Code tool in `tool_activities`) or finishes.

1. Durable tools are declared to Claude as SDK MCP tools.
2. A `PreToolUse` command hook answers `defer` whenever Claude calls one ([documented in the hooks guide](https://code.claude.com/docs/en/hooks)). The engine stops with `stop_reason: "tool_deferred"`, and the segment returns the call.
3. The Workflow runs the call as its own Activity, after an approval if the tool needs one. A Claude Code tool in `tool_activities` pauses the segment the same way, and runs in a tool step (see [Claude Code's own tools](#claude-codes-own-tools)).
4. The next segment resumes the session with the result as a normal `tool_result` message.

**Several calls in one message.** The engine keeps one paused call per run, so the hook defers the first call of a message that runs as an Activity (a durable call, or a Claude Code tool in `tool_activities`) and denies the calls after it. The segment reports the denied durable calls, and the Workflow runs them with the paused one, at once, each its own Activity with its own approval. The next segment moves their denials before the point where the engine resumes and puts the real results in them, so Claude sees each call with its result, and the paused call's result arrives as usual. A Claude Code call after the paused one keeps its denial, and Claude calls it again. Tested on Claude Code 2.1.273 and 2.1.287, with the conversation in the Workflow and in a session store: three durable calls in one message, durable and built-in calls mixed, two such messages in a row, a step that runs again, and Continue-As-New while the results wait; and read-only calls that Claude Code runs together, as one batch, where any one of them can be the call that pauses.

**Checkpoints.** After a segment, the runner reads the session back and returns its last transcript entry, where the engine would resume; the Workflow records that checkpoint with the segment's result. After a message with several calls, the checkpoint is the paused call's deferral marker (the denied calls' results come after it, and the engine would not resume the paused call past them).

- With the conversation in the Workflow, the next segment resumes from the committed conversation up to the checkpoint, in memory, and returns what it added. A segment that runs again does the same, so Claude decides again from the checkpoint, with new tool call ids; after a Workflow reset, the conversation goes back with the Workflow's state.
- With a session store, reading the checkpoint back also proves the turn reached the store. A segment that runs again continues in a copy of the session that ends at the checkpoint (`fork_session_via_store`). So does a segment whose session went on past its checkpoint, for example after a Workflow reset (the check rides on the load the SDK does anyway), and the segment after a deferral marker.

Tested on the real engine, both ways: a Worker killed while Claude is answering, an attempt that hangs past its timeout, results lost after a step finished, and a step started from an older checkpoint, as after a Workflow reset. A real Workflow reset is tested with the scripted runner: the conversation goes back with the Workflow.

**Fail closed.** If a Claude Code version ever runs a durable tool itself, or ignores a pause, the segment fails with a non-retryable error that names the engine version. A durable tool never runs outside Temporal: inside the engine, it only returns an error. The hooks guide says Claude Code ignores `defer` when Claude makes several tool calls at once; the engines tested (2.1.273 to 2.1.287) pause at one call, also in a batch of read-only calls, and calls in one message depend on that. If an engine ignores a pause on a Claude Code tool in `tool_activities`, the call goes through Claude Code's own permission check, where it is not pre-approved: under the default permission mode, a command that changes something is refused (tested with a hook that gives no answer), and the segment fails because it did not pause. If the engine ever hands back a tool call id that already ran (the agent remembers every call of the current run, and the last 256 calls across runs), the Workflow stops instead of running it twice.

## Requirements and limits

- **Claude Code 2.1.273 or newer.** Checked by hand: when Claude calls two tools in one message, Claude Code 2.1.259 replaces the paused call's result with `[Tool result missing due to internal error]`, so Claude asks for the same tool again. The runner checks `claude -v` before starting the engine and refuses older engines.
- **Claude Code calls after a paused call wait.** In a message with several calls, a Claude Code call after the call that paused the segment is denied and called again by Claude in its next turn (durable calls all run; see [How it works](#how-it-works)). `ClaudeAgentSdkRunner(one_tool_at_a_time=True)` asks Claude for one call per message instead.
- **Claude Code's tools that stay in the segment** (file tools, `WebFetch`, subagents, and any tool not in `tool_activities`) run on the Worker's disk in the runner's `cwd` (shared by every agent on that Worker). They can run again when a segment runs again, possibly on another Worker with a different disk, and files a failed attempt wrote are not rolled back. See [Claude Code's own tools](#claude-codes-own-tools).
- **Each step reads the whole conversation** (with the Query, or from a session store), so reading grows with the conversation's length. Compaction does not shrink it: Claude Code keeps the entries from before a compaction, and resuming uses some of them (a preserved segment of recent messages; checked by hand with `/compact` on 2.1.273), so the plugin keeps them all.
- **Native child recovery fails closed after acceptance.** Before a child request is accepted, a segment may retry normally. After acceptance, losing the engine or the segment attempt ends the task: the plugin cannot safely restore a native pending callback. `AgentState.native_calls` keeps accepted requests and completed outcomes, with `delivered=False` when no segment verified delivery. The plugin never asks the model to rediscover those calls. Activity retries still have their configured at-least-once effect semantics. Child effects require a working launcher and Worker lock (and a verified job object on Windows); they are refused when that protection is unavailable.
- **Long conversations.** Continue-As-New keeps the Workflow's history small, but what Claude reads grows until Claude Code compacts it. A request the model refuses as too long fails the task.
- **Logins that survive resumes.** Use `ANTHROPIC_API_KEY`, Amazon Bedrock, Google Vertex AI, Microsoft Foundry, or `CLAUDE_CODE_OAUTH_TOKEN` from `claude setup-token`. Every step after the first resumes the session, and a Claude app login cannot refresh itself there.

## Security

- Credentials stay on Workers: the engine inherits the Worker's environment (API keys, cloud credentials, proxies). They reach Workflow history only if a built-in tool shows them to Claude (for example Bash running `env`), so give Workers that enable built-in tools only the secrets they need.
- The conversation does: prompts, Claude's text, tool arguments and results are in the Workflow's history (and in the session store, if you use one). Encrypt payloads with a codec, and protect a store like any other data store.
- Claude chooses tool arguments. Treat them as untrusted input in your Activities, and require approval (`needs_approval=True`) for tools that move money or delete data.

## Testing your agents

`temporalio.claude_agent_sdk.testing.ScriptedClaude` is a segment runner that plays Claude with a Python policy. It needs no engine and no API key. Like the real runner, it keeps the conversation in the Workflow, so tests can kill a Worker and continue on a new one (give it a folder to keep sessions there instead, like a session store), and its checkpoints behave like the real runner's: a segment that runs again decides again, with a new tool call id. A policy can return several calls (one message), and `engine_tools` plays Claude Code tools such as Bash in tool steps.

## Documentation

- Repository conventions: [`AGENTS.md`](https://github.com/temporalio/ai-integrations/blob/main/AGENTS.md)

## Develop

```bash
make sync   # install (non-editable) into .venv
make lint
make test   # the real Claude Code engine against a local fake Messages API; no credentials
```

`tests/test_replay.py` replays recorded Workflow histories of every scenario in `tests/record_histories.py`, and of each scenario the plugin as first published could run, recorded with that version, so a change that would break running Workflows fails a test. `python -m tests.record_histories NAME ...` records the named scenarios again (all of them, without names): after adding a scenario, record only that one, so the other files stay as they are.
