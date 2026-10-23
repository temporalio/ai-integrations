# Durable Claude Agent SDK agents on Temporal

> Release stage: **Pre-release**.

Temporal integration for Anthropic's [Claude Agent SDK](https://code.claude.com/docs/en/agent-sdk/overview), published as [`temporalio-claude-agent-sdk`](https://pypi.org/project/temporalio-claude-agent-sdk/) and imported as `temporalio.claude_agent_sdk`.

- **Every durable tool call Claude makes is its own Temporal Activity.** Finished calls never run again after a crash, retries follow your retry policy, and the Activity ID (`tool-<tool_use_id>`) doubles as an idempotency key for the systems a tool touches. When Claude calls several durable tools in one message, they run at once.
- **Claude Code's commands and edits too.** Bash commands and MCP tool calls run as their own Activities by default, like durable tools (a subagent cannot run them), and so can `Edit` and `Write`. A model step that runs again never runs them again, and neither does a tool step that broke after its call may have started: Claude is told so instead (unless you mark the tool as safe to repeat). `Read` and `WebFetch` still run inside the step.
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

Other [`ClaudeAgentOptions`](https://code.claude.com/docs/en/agent-sdk/python) go in the runner's `extra_options` (for example `permission_mode`, `agents`, `hooks`, `setting_sources`, `thinking`). `env`, `mcp_servers` and `allowed_tools` are merged with the plugin's own, and `system_prompt` (a string, or a preset such as Claude Code's own prompt) is the default for agents that set none. Options the agent or the plugin sets (`model`, `tools`, `max_turns`, `cwd`, `settings`, session and resume options, and the same engine flags in `extra_args`) are refused, and so are engine flags in `extra_args` that approve tool calls (`--permission-mode`, `--dangerously-skip-permissions`, `--allow-dangerously-skip-permissions`, `--permission-prompt-tool`): set `permission_mode` or `permission_prompt_tool_name` here instead, which a tool step can replace. The runner sets `permission_mode="default"` unless you pass one: since Claude Code 2.1.285, a run without one uses auto mode when telemetry is off or the provider is Bedrock, Vertex or Foundry, and auto mode asks the model whether each tool call may run. Tools of the MCP servers you pass here run in tool steps by default (`tool_activities` includes `mcp__*`), and each segment and tool step starts (or connects to) the servers again. The plugin decides on its `mcp__durable__` tools and on the tools in `tool_activities`: a `PreToolUse` hook you pass here that answers for one of them stops the task with an error that says so (tested), and hooks from settings files you load must not answer for them either. `enable_file_checkpointing` is refused: it cannot work with the session store the runner always gives the SDK.

`approvers` checks the name the caller passes. It is not authentication: control who may send Updates with Temporal's own access control.

## Claude Code's own tools

Claude Code's built-in tools are off unless you enable them with `builtin_tools`. The tools in `tool_activities` run as their own Activities, like durable tools: by default `("Bash", "mcp__*")`, that is Bash, and the tools of MCP servers you give the runner in `extra_options`. `Edit` and `Write` can run as Activities too; add them to `tool_activities`. The others run inside the segment.

```python
self.agent = DurableClaudeAgent(
    builtin_tools=["Bash", "Read", "Grep", "Edit", "Write"],
    tool_activities=["Bash", "Edit", "Write", "mcp__*"],
    tool_approvals=["Bash"],  # a human approves each command first
    repeatable_tools=["mcp__search__*"],  # safe to run again
)
```

- Each call is an Activity `run_claude_tool_step` with ID `tool-<tool_use_id>`, so a model step that runs again (after a crash or a timeout) never runs the command again. Tested: the model call after a command hangs past the step's timeout; as its own Activity the command ran once, inside the segment it ran twice.
- The Activity runs exactly that call in Claude Code, with Brian Strauch's bounded native call replay (from his hybrid prototype): Claude Code resumes a copy of the conversation that ends before the call, a stand-in model on the Worker (on 127.0.0.1) answers with the call itself, as Claude sent it, and the turn ends right after the call (`max_turns=1`). So Claude gets Claude Code's own result, and a tool call costs no model call, whatever provider the Worker uses (tested with Bedrock configured). The stand-in answers only the engine of a step that is running, which gets a random key of its own; another program on the machine gets 401 (tested). About a second per call on Linux, a little less for an `Edit` or a `Write` (median, measured with Claude Code 2.1.273 and 2.1.288).
- With `ClaudeAgentSdkRunner(warm_engines=N)` (experimental), the call waits in its segment's Claude Code instead, on the Worker, for its tool step. When the step comes to the same Worker, for that call and with the input Claude Code's hook saw, the call runs there, still inside its own Activity, and the turn then waits right after it for the next segment. A round with an `Edit`, a `Write` or Bash then takes about a tenth of the time: 0.14 to 0.24 s instead of 1.35 to 1.97 s (from one model request to the next; medians of 20 rounds in 6 trials, with the conversation in the Workflow, local fake model, Claude Code 2.1.273; `ROUND_TOOL` in `tests/bench`). Anything else (the step or the next segment on another Worker, a wait past `warm_seconds`, a Worker that stops or dies while the call waits) ends that Claude Code: its hook denies the waiting call, so it never runs there, its turn stops without asking Claude, and the step or the segment runs as above. A step that times out while its call runs there, or whose Worker stops or dies then, ends that Claude Code too, and Claude is told that the call may have run, as without warm engines (tested with a step that times out). Calls wait only in the permission modes `default` and `plan`, where Claude Code refuses a call whose hook gave no answer (not with `acceptEdits`, `bypassPermissions` or auto mode), and never when a message holds several calls. Tested on the real engine against the same tasks without warm engines, also with a model that takes its time after a call and with relative file paths: the same files, each command and edit once, the same number of model requests, the same total cost, and the same last request apart from ids, paths, Claude Code's own `Continue from where you left off.` line, its notes, and a job-control warning that bash on macOS prints with a command now and then, warm or not (Claude Code turns job control on in its commands when `~/.bashrc` exists: anthropics/claude-code#19057). A warm Claude Code keeps its notes between calls, as one Claude Code run does: for example, Claude can be told that a file changed on disk since it read it, that a file it edited is current in its context, or that the user has not heard from Claude for a while (after several model calls in one turn), depending on the Claude Code version. Upgrade every Worker before you turn this on: an older Worker cannot continue a session from a call that waited.
- For `Edit` and `Write`, the step also returns Claude Code's own record of the result, and the next segment puts that record where the call's result belongs in the conversation; Claude Code then continues the turn by itself. So Claude Code does not check the call again: an `Edit` after a `Read`, or a `Write` over a file Claude read, is reported as done, not as a file "modified since read" (anthropics/claude-code#99041; tested on every Claude Code release from 2.1.273 to 2.1.295, with the conversation in the Workflow and in a session store, also with a `Read` and an `Edit` in one message, and with a durable call after the `Write`). The record carries no copy of the file: Claude Code's metadata of the call (`toolUseResult`, which for an `Edit` holds the whole file as it was) is left out, as Claude Code does not need it to go on (tested: two `Edit`s of one file, and a `Write` over a file Claude read). When there is no such record (the step broke after the edit, or the call was rejected), the segment writes one of the same shape with what Claude gets instead (tested with a step that broke). Claude sees the result, then Claude Code's own line for a turn it continues by itself, `Continue from where you left off.` (the segment asks for this with [`CLAUDE_CODE_RESUME_INTERRUPTED_TURN`](https://code.claude.com/docs/en/env-vars)). A new task that comes right after an `Edit`'s or a `Write`'s result (the task stopped after its tool steps) goes as a message after the record, and Claude Code first closes the interrupted turn: Claude sees the result, then `No response requested.` in Claude's place, and then the new task (tested; Claude Code 2.1.280 and older also add `Continue from where you left off.` before that answer). Commands and MCP tools are not checked again, so their results reach Claude as a message, as before.
- If Claude Code says nothing for 2 minutes before it starts that turn, the segment fails and Temporal runs it again. After 3 attempts, the task stops with an error that names the Claude Code versions tested (tested).
- `tool_approvals` makes calls wait for a decision, like `needs_approval` (`pending_approvals()` shows the command). A rejected command never runs. Each pattern must fall within `tool_activities`.
- A command that fails is a result for Claude, not a failed Activity (tested). The step fails only when it breaks before it has the command's result, and then no retry runs the call again (unless the tool is in `repeatable_tools`, below). Just before the hook lets the call run, it creates a record in the step's folder; when the step ends, the runner creates the same record if the hook has not, so from then on the call can no longer start. Without that record, a failed step says the call did not run only if Claude Code never got the call, or the hook or Claude Code's permission check refused it: a rule from a settings file Claude Code trusts (a managed policy, say) could let Claude Code run a call the hook gave no answer for (tested). So a failed step knows on which side of the call's start it broke:
  - After (the engine went away, the step timed out, its Worker stopped responding or shut down): the call may have run. It gets no other attempt, and Claude gets `This call was interrupted (its step timed out), so it may have run, in full or in part. Check its effects before you run it again.` (with the cause in the parentheses).
  - Before (for example on a Worker set up the other way, or one where engine cleanup cannot be set up): the step is tried again, each attempt a new Activity with one try (`tool-<tool_use_id>-2`, ...), with the backoff, attempts and non-retryable error types of `tool_activity_retry_policy` (Temporal's defaults if None: unlimited attempts, from 1 second up to 100). A non-retryable type matches `ToolCallNotRun` or the type of the error behind it (for example `EngineCleanupUnavailable`). Each attempt adds about ten events to the history, where Temporal's own retries add none, so attempts also stop after about 5,000 events for one call (about 450 attempts), when the run's history is nearly full, and with `auto_continue_as_new`, when it is time to continue as new (Claude can then call again in the new run). If no attempt runs the call, Claude gets `This call did not run: its step failed before the call could start. You can call it again.`
  - Each failed step records in the history whether its call may be tried again, and the Workflow follows that record, so changing `tool_activity_retry_policy` or `repeatable_tools` never breaks the replay of a running Workflow (tested).
  - The step's own error stays in Temporal: it can name the Worker's folders and settings. In `tool_calls()`, the call's status is `interrupted` or `not run`.
  - A Worker that dies before the call started also reads as "may have run": only the step itself can tell.
  - Once the step has the result, it keeps it, even if the engine fails afterwards or the Worker starts to shut down (tested).
  - A runner of your own, or an `engine_tools` function of `ScriptedClaude` (see [Testing your agents](#testing-your-agents)), says on which side a step failed with an `ApplicationError` of type `ToolCallNotRun` or `ToolCallInterrupted` (the constants `TOOL_CALL_NOT_RUN` and `TOOL_CALL_INTERRUPTED`); any other failure reads as "may have run".
- `repeatable_tools` names the calls that are safe to run again (read-only or idempotent): their steps keep Temporal's retries, so the call can run again after a step that broke, and a failure reports the last attempt. Temporal sees the types `ToolCallNotRun` and `ToolCallInterrupted` there, so those are the names its `non_retryable_error_types` can use. Each pattern must fall within `tool_activities`. Claude Code sends each call to an MCP server its `tool_use_id` (`_meta["claudecode/toolUseId"]`; checked with an in-process server on every Claude Code release from 2.1.273 to 2.1.295), and a tool step runs the original call, so a server sees the same id in each attempt and can recognize a repeated call (tested, with a step that timed out and was retried).
- The permission mode does not decide on a call in `tool_activities`: the plugin's hook pauses it in the segment before Claude Code's permission check, and allows it in its tool step, so it runs whatever the mode, `plan` included (tested in `plan` mode with Bash and `Edit`). In the tool step, nothing else approves the call: not `allowed_tools`, the permission mode or `can_use_tool` from `extra_options` (tested with a hook that gives no answer, also with `bypassPermissions` in `extra_options`: Claude Code refuses the call, and the step fails as one whose call did not run, so it can be tried again). If deciding fails (for example, on input nested deeper than Python's `json` reads), the hook denies the call (tested). The hook lets the call run only with the input the segment reported, which is what an approval saw (tested; Claude Code makes a relative file path absolute in both, so they match). Deny and ask rules still apply: Claude Code checks them after hooks, and as no callback is asked in a tool step, an `ask` rule refuses the call there (tested with a deny rule for a command, an ask rule, and no rule; a deny rule for the whole tool takes the tool away from Claude, also tested). To decide on each call, use `tool_approvals`; to keep commands or edits out of a planning agent, leave Bash, `Edit` and `Write` out of `builtin_tools`.
- Output over about 30 KB: Claude Code shows Claude a preview and saves the rest to a file in its temporary folder, which the step removes. Claude also gets the last 4 KB, which the step reads only from Claude Code's own folder for the session, so a command cannot make it read another file (tested).
- Several calls in one message: the first call that runs as an Activity (a durable tool, or a Claude Code tool in `tool_activities`) pauses the segment, and the durable calls after it run with it. A Claude Code call after it (a second Bash call, say) keeps its denial, and Claude calls it again. Read-only calls that Claude Code runs together, as one batch, have their hooks run at the same time: one of them pauses the segment, not always the first, and the others keep their denial (tested).
- Commands run on the disk of the Worker that runs the tool step, in the runner's `cwd` (with a workspace store, in the agent's own folder), and each command starts there: a `cd` does not carry over to the next command (each step is a new Claude Code run), so the plugin makes that the rule inside segments too, and tells Claude (tested). Give Workers the same files (a shared disk), or one Worker per agent, or give the runner a workspace store, so each agent's files follow it to any Worker (see [Where the files live](#where-the-files-live)).
- A command sees the Worker's own environment. The step points Claude Code at the stand-in model, then gives each Bash command the Worker's own values back, through `CLAUDE_ENV_FILE` (tested). `PowerShell` commands get the step's values instead: their `ANTHROPIC_BASE_URL` points at the stand-in. So does an MCP server that Claude Code starts in a tool step, unless its own `env` sets `ANTHROPIC_BASE_URL` (tested): give a server that calls the Anthropic API itself its endpoint and key there. Bash commands see none of the plugin's own variables (`TCA_*`), in tool steps and in segments (tested); an MCP server in a tool step does see them (tested).
- `Edit` and `Write` in `tool_activities` run at most once, like commands: a model step that runs again never makes the edit again, and a step that broke after the edit may have started tells Claude to check the file. An `Edit` is not safe to run again (a second run fails, or changes the file twice), so leave it out of `repeatable_tools`. After the hook's approval, Claude Code still guards its sensitive files: an `Edit` of `.git/config` or `.claude/settings.json` is refused in its step. It does not keep the edit inside the working directory, just as it does not keep a command there: to decide on each edit, use `tool_approvals` (both tested). Other file tools (`Read`, `Grep`, ...) stay in the segment. So do tools that call the model themselves (`WebFetch`). Inside the segment, a tool can run again when the segment runs again: give such tools work that is safe to repeat.
- A subagent (`Agent`) runs inside the segment, and cannot call the tools that run as Activities, durable ones included: such a call is denied with a hint to leave it to the main agent (tested), so those calls always run as Activities, with their approvals. So with `Edit` and `Write` in `tool_activities`, only the main agent can call them.
- `PowerShell` uses the same path as Bash (not tested here); MCP tool names take patterns, such as `mcp__github__*`. `tool_activities=()` runs everything inside the segment.

## Where the conversation lives

By default, in the Workflow. Each step reads the committed conversation with a Query on its own Workflow, a page at a time, so the conversation is not copied into every step's input. The step resumes Claude Code from it in memory and returns only what it added (about 5 to 9 KB per step with small tool results, measured with Claude Code 2.1.273); the Workflow splices that in. So:

- **No storage to run.** For the conversation, Workers share nothing but the Temporal server. Tested: each step of a conversation run by a new runner with its own working directory and config folder, and a Worker process killed after the refund, replaced by one with its own config folder that never saw the conversation.
- **Retries are clean by construction.** Every attempt starts from the conversation the Workflow committed; nothing an unfinished attempt added to it reaches Claude (files its tools wrote stay, unless the runner has a [workspace store](#where-the-files-live)).
- **No copy of the conversation stays on the Worker's disk** after a step ends normally. Claude Code writes a new session to its config folder (`~/.claude/projects`, with tool outputs too large to show Claude in full); the runner removes the copy it created, never a session that was there before. Resumed steps run in a temporary folder the SDK removes. A Worker that is killed can leave a copy behind.
- **One payload per step.** Without External Storage, what a step adds must fit in one Temporal payload (2 MB by default), or the step fails with a message that says so, instead of retrying forever. The results of one message's calls must also fit in the next step's input together: if they do not, the largest are replaced by a note that tells Claude so (they stay in the history, as the tools' results; tested). Results close to 2 MB need [External Storage](#large-tool-results), which also carries steps and the conversation through Continue-As-New.
- **Reading costs a Query per page.** Pages hold up to 508 KiB, under the 512 KiB where Temporal starts to log a warning about a payload's size; with External Storage, just under its threshold (256 KiB by default), so they stay in the Query's response instead of going to the store at every step. An entry larger than that is a page of its own, the same payload at every step, so a content-addressed store (such as Temporal's S3 driver) keeps it once. Measured on a local dev server with 40 tool results of 100 KB (a 3.8 MB conversation), the last step read it in about 180 ms with 8 Queries, and in about 310 ms with 20 Queries with External Storage. On Temporal Cloud, every Query is an [Action](https://docs.temporal.io/cloud/actions); a higher External Storage threshold means fewer pages. A Workflow that is not in the Worker's cache (more open Workflows than the Worker's `max_cached_workflows`) replays its history before it answers the step's first Query.
- **A subagent's own steps are not kept**, only its report (the `Agent` call's result; tested). A session store keeps them too.

To keep conversations outside Temporal instead, give the runner a Claude Agent SDK `SessionStore` that every Worker can reach (`ClaudeAgentSdkRunner(session_store=...)`). `FileSessionStore` is for tests and one machine; for production, implement `SessionStore` on your database or object storage (the Claude Agent SDK repository has example stores for S3, Redis and Postgres). The store keys sessions by the engine's working directory, so give every Worker the same `cwd`. Every segment then reads the whole session twice (the SDK to resume it, the runner to record the checkpoint), so reading grows with the session's length; a segment that runs again reads it once more and writes a copy, and the earlier copy stays in the store, so remove old sessions with your store's own retention.

Every Worker of a task queue needs the same choice. A Worker set up the other way refuses the step (a model step or a tool step) with a message that says how to fix it, and Temporal retries the step until a matching Worker picks it up (tested).

## Where the files live

By default, on the Worker's disk: every agent on a Worker works in the runner's `cwd`, and a step that runs on another Worker finds that Worker's files. Give Workers the same files (a shared disk), or one Worker per agent. Or give the runner a workspace store, and each agent's files follow it to any Worker:

```python
from temporalio.claude_agent_sdk import ClaudeAgentSdkRunner, FileWorkspaceStore

runner = ClaudeAgentSdkRunner(
    cwd="/srv/agents",
    workspace_store=FileWorkspaceStore("/mnt/shared/workspaces"),
    workspace_exclude=["node_modules", ".venv"],
)
```

- **Each agent has a folder of its own**, `<cwd>/<id>`, the same path on every Worker, and Claude Code works there, commands included. The id is the agent's first session id (`agent.state().workspace.id`). The folder starts empty: let Claude fetch what it needs (a `git clone` in a tool step is saved like any other change).
- **After each segment and tool step, the runner saves what changed** in the folder to the store, before the Activity returns: each changed file under the SHA-256 of its bytes, and each directory as a small JSON tree, like git. The step's result names the new snapshot. Kept: files with their permission bits and their modification time to the nanosecond (so Claude Code never takes a restored file for one "modified since read"), empty directories, hidden files, and symbolic links (as links, never followed). A hard link is kept as a file. Left out: sockets, FIFOs, devices, other file systems mounted inside, and paths that match `workspace_exclude` (a pattern with no `/`, or only a trailing one, matches a name at any depth, such as `node_modules` or `*.log`; one with a `/` at its start or in its middle matches the path from the agent's folder, such as `build/*` or `/cache`). What is left out is never saved or restored, and stays as it is, unless a directory that holds it is not in the snapshot or cannot be read (it goes, with all it holds), or an entry of the snapshot takes its place. A snapshot saved before a pattern was added keeps what the pattern leaves out, and a restore leaves those paths alone too.
- **Before each segment and tool step, the folder is made to match the agent's current snapshot.** The objects it needs are read from the store and checked against their keys first, so a missing or a corrupt one stops the step before anything in the folder changes; then only what differs is removed, written or fixed. A folder as this Worker process left it needs no read from the store: each file's `stat` is compared with what it was, as git's index does, and only a file whose `stat` changed, or that changed in the 2 seconds before it was last read, is read again. A file that shares its data with another name (a hard link, as package caches make) gets a copy of its own before its time or mode is fixed, so the other name keeps them (on Windows, a read-only one becomes writable).
- **A snapshot counts only once the Workflow has the step's result.** An attempt that did not complete (its Worker died or was cut off, the step timed out) may have saved objects, but they never become current, and the next step rolls back what the attempt changed in the folder, on any Worker, the same one included. So a call that may have run leaves no change in the folder, except in paths that `workspace_exclude` leaves out; Claude is still told it may have run, for its other effects. Tested on the real engine: a file written, a script made executable and a file read on one Worker, then the file edited and the script run on another Worker whose disk does not have the folder (no "modified since read"); a command whose result was lost, on the same Worker and on a new one; two agents on one Worker; warm engines. And in the chaos test (`CHAOS_WORKSPACE=1`), where each Worker that replaces a killed one starts without the folder: the promises hold on the last snapshot, read back from the store.
- **Continue-As-New and resets.** The folder and its snapshot move to the next run in `AgentState`. After a Workflow reset to an earlier point, the next step gets the snapshot of that point (tested with the scripted runner), so the folder goes back too, as long as the store still has its objects.
- **Two steps of one agent never use its folder at once:** a lock in the Worker process, and a lock file in `<cwd>/.temporal-workspaces` for other Worker processes on the same machine (an attempt that still runs while the next one starts). The file system of `cwd` must support file locks.
- **Errors.** A missing object, or one that does not match its key, stops a segment with a non-retryable `ApplicationError` of type `WorkspaceNotFound` or `WorkspaceCorrupt`; files over `workspace_max_bytes` (1 GiB by default), with `WorkspaceTooLarge`; an agent's folder that cannot be used (a link, a file or a folder that cannot be read when it is saved, a file or a folder the restore may not change, a symbolic link that Windows cannot create), with `WorkspaceInvalid`. A tool step fails with its own error, with the workspace error as its cause: as `ToolCallNotRun` when it fails before its call runs (restoring the folder), as `ToolCallInterrupted` when it fails after (saving it; the next step rolls back what the call changed). Either way the task stops with the error once the calls of the message settled (tested): Claude would otherwise run the call again, say an install that keeps taking the folder past the limit. A failing store call is tried 3 times; then a segment fails and Temporal runs it again (it asks the model again), and a tool step fails as above, but its task goes on (a step whose call did not run is tried again; a call that may have run is not). A file the Worker cannot read is replaced by a restore.
- **Every Worker of a task queue needs the same store, `cwd`, `workspace_exclude` and kind of system.** A Worker without a store does not run a step of an agent whose folder has a snapshot: the step fails before it starts and is tried again, on a Worker with the store once there is one (as when a session store is missing). A Worker of an earlier version of the plugin knows nothing of the folder: with the conversation in the Workflow (the default), its step runs in its `cwd`, and the task stops once the results of the message are in (Claude gets them with the next task; what that step changed is not in the folder). An agent whose first segment ran without a store (on a Worker without one, or with an earlier version of the plugin) keeps working in `cwd`, without failover, on any Worker; its state names a folder (`workspace.id`) that it never uses. Set up the store on every Worker before new agents start. A pattern removed from `workspace_exclude` makes the next restore remove what it left out (the snapshot does not have it).
- **What it costs.** Each segment and tool step reads the `stat` of every file in the folder twice (before and after), hashes the files that changed, and saves them and the trees on their paths. Measured on a 2 vCPU Linux machine with a local `FileWorkspaceStore`, per step: about 5 ms for an empty folder, 40 ms with 2,000 files, 0.2 s with 10,000. A round with warm engines (a Bash call and the next segment) took about 0.19 to 0.21 s with 2,000 files instead of 0.15 to 0.16 s; with an empty folder, the store added nothing measurable. A folder restored on a new Worker is read from the store in full. A Worker keeps what it knows of the last 256 agents' folders; past that, a folder's files are hashed again.
- **Limits.** A process that writes into the folder between steps (a server started in the background) has those writes rolled back by the next step; what it writes during a step is saved with it, file by file, as read. Keep such output out of the folder, or exclude it. Only the agent's folder is saved: files elsewhere and effects outside the machine are not. Your own Activities (durable tools) do not run in the folder. On Windows, where a folder listing gives no change time, a change that keeps a file's size and puts its time back is not seen. A store call still running 30 seconds after its step was cancelled is cancelled too; give a store's calls their own timeouts.
- **The store.** `FileWorkspaceStore` keeps the objects in a folder: for tests, one machine, or a shared mount such as NFS or EFS. For object storage, implement `WorkspaceStore`: two async methods, `save(key, path)` (keep the file's bytes under the key, durably) and `load(key, path)` (write them to a new file; `KeyError` if missing). Everything in the folder goes to the store, a `.env` file with credentials included: protect the store as you protect the Workers' disks, or exclude such files. The plugin removes nothing from the store: remove an agent's objects (all under its `<id>/` prefix) once the agent is done, and its folders from the Workers' disks.

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
- If the Worker process dies (a crash, an out-of-memory kill, SIGKILL), its engines end with it. On Linux the engine starts through a small launcher that asks the kernel for SIGTERM when the Worker's thread that started it ends (the event loop's, which lives as long as the Worker), then becomes the engine (about 20 ms per engine start); on SIGTERM Claude Code ends its commands. On macOS (and on Linux without `prctl`) the launcher stays as the engine's parent and sends it SIGTERM once the Worker is gone (it checks five times a second). On Windows every Claude Code process the SDK starts joins, within milliseconds, a job object that ends it, and the processes it starts, with the Worker process; the Worker process and its other child processes are not in it. (The Worker's event loop checks every 5 ms while an engine starts, normally long before Claude Code starts a process of its own; on a loop blocked for longer, a process the engine starts before it joins, such as an MCP server, stays outside the job and can outlive a dead Worker.) Until the engine is gone, the hook denies every built-in call: the Worker's lock on the step's folder is free. Tested by killing the Worker during a model call that never answers: within seconds no process it started was left (before, Claude Code waited for the model, then went on with its turn). Where this cannot be set up (the launcher cannot run, for example in a temporary folder mounted noexec; the job object fails; or files cannot be locked, on some network or FUSE file systems), the step stops before Claude Code runs anything, with a retryable `ApplicationError` of type `EngineCleanupUnavailable` that says what to fix (on Windows the job object is created first; an engine can fail to join it only after it started: the step then ends that engine at once, and its hook denies every call). `ClaudeAgentSdkRunner(engine_cleanup="best_effort")` runs such steps anyway, with a warning, and then a dead Worker's engine finishes its current turn.

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
4. The next segment resumes the session with the result as a normal `tool_result` message. The result of an `Edit` or a `Write` in `tool_activities` goes in as Claude Code's own record of it instead (see [Claude Code's own tools](#claude-codes-own-tools)).

**Several calls in one message.** The engine keeps one paused call per run, so the hook defers the first call of a message that runs as an Activity (a durable call, or a Claude Code tool in `tool_activities`) and denies the calls after it. The segment reports the denied durable calls, and the Workflow runs them with the paused one, at once, each its own Activity with its own approval. The next segment moves their denials before the point where the engine resumes and puts the real results in them, so Claude sees each call with its result, and the paused call's result arrives as usual. A Claude Code call after the paused one keeps its denial, and Claude calls it again. Claude Code starts each call as soon as its block arrives, so when the model takes its time between the calls of a message (while it writes a long input, say), the segment pauses at the first call before the others arrive, and the session has them after the pause. The next segment first moves them back where they belong, so they are not lost. Tested on Claude Code 2.1.273 and 2.1.287, with the conversation in the Workflow and in a session store: three durable calls in one message, durable and built-in calls mixed, two such messages in a row, a step that runs again, and Continue-As-New while the results wait; and read-only calls that Claude Code runs together, as one batch, where any one of them can be the call that pauses. With a model that waits 1.5 s after each call, on Claude Code 2.1.273, 2.1.274, 2.1.288 and 2.1.295: three durable calls, durable and built-in calls mixed, Bash with a durable call, an Edit with a Bash, and a line of text after a call.

**Checkpoints.** After a segment, the runner reads the session back and returns its last transcript entry, where the engine would resume; the Workflow records that checkpoint with the segment's result. After a message with several calls, the checkpoint is the paused call's deferral marker (the denied calls' results come after it, and the engine would not resume the paused call past them).

- With the conversation in the Workflow, the next segment resumes from the committed conversation up to the checkpoint, in memory, and returns what it added. A segment that runs again does the same, so Claude decides again from the checkpoint, with new tool call ids; after a Workflow reset, the conversation goes back with the Workflow's state.
- With a session store, reading the checkpoint back also proves the turn reached the store. A segment that runs again continues in a copy of the session that ends at the checkpoint (`fork_session_via_store`). So does a segment whose session went on past its checkpoint, for example after a Workflow reset (the check rides on the load the SDK does anyway), the segment after a deferral marker, and the segment after an `Edit`'s or a `Write`'s tool step: its copy holds Claude Code's record in place of the pause's hook entries, as the conversation the Workflow holds does (tested: with the record added to the session itself, a later copy made Claude Code take a paused call for interrupted). To find where a record goes, a segment after an error result, or after an `Edit`'s or a `Write`'s step, reads the session from the store once more.

Tested on the real engine, both ways: a Worker killed while Claude is answering, an attempt that hangs past its timeout, results lost after a step finished, and a step started from an older checkpoint, as after a Workflow reset. A real Workflow reset is tested with the scripted runner: the conversation goes back with the Workflow.

**Fail closed.** If a Claude Code version ever runs a durable tool itself, or ignores a pause, the segment fails with a non-retryable error that names the engine version. A durable tool never runs outside Temporal: inside the engine, it only returns an error. The hooks guide says Claude Code ignores `defer` when Claude makes several tool calls at once; the engines tested (2.1.273 to 2.1.287) pause at one call, also in a batch of read-only calls, and calls in one message depend on that. If an engine ignores a pause on a Claude Code tool in `tool_activities`, the call goes through Claude Code's own permission check, where it is not pre-approved: under the default permission mode, a command that changes something is refused (tested with a hook that gives no answer), and the segment fails because it did not pause. A tool step pre-approves no tool at all: its call runs when the hook allows it. When the hook gives no answer, Claude Code refuses the call (tested the same way), unless a rule in a settings file it trusts allows it, and then the step says the call may have run (tested). If the engine ever hands back a tool call id that already ran (the agent remembers every call of the current run, and the last 256 calls across runs), the Workflow stops instead of running it twice.

## Requirements and limits

- **Claude Code 2.1.273 or newer.** Checked by hand: when Claude calls two tools in one message, Claude Code 2.1.259 replaces the paused call's result with `[Tool result missing due to internal error]`, so Claude asks for the same tool again. The runner checks `claude -v` before starting the engine and refuses older engines.
- **Claude Code calls after a paused call wait.** In a message with several calls, a Claude Code call after the call that paused the segment is denied and called again by Claude in its next turn (durable calls all run; see [How it works](#how-it-works)). `ClaudeAgentSdkRunner(one_tool_at_a_time=True)` asks Claude for one call per message instead.
- **Claude Code's tools that stay in the segment** (`Read`, `WebFetch`, subagents, and any other tool not in `tool_activities`) run on the Worker's disk in the runner's `cwd` (shared by every agent on that Worker; with a workspace store, the agent's own folder). They can run again when a segment runs again, possibly on another Worker with a different disk, and files a failed attempt wrote are not rolled back, unless the runner has a workspace store (see [Where the files live](#where-the-files-live)). See [Claude Code's own tools](#claude-codes-own-tools).
- **Each step reads the whole conversation** (with the Query, or from a session store), so reading grows with the conversation's length. Compaction does not shrink it: Claude Code keeps the entries from before a compaction, and resuming uses some of them (a preserved segment of recent messages; checked by hand with `/compact` on 2.1.273), so the plugin keeps them all.
- **Subagents run in the foreground, without the tools that run as Activities.** The runner turns off Claude Code's background tasks (a background subagent kept the engine working after it paused). A subagent (Claude Code's `Agent` tool) can use Claude Code's other tools, inside the segment; it cannot pause the run, so its call to a durable tool or to a tool in `tool_activities` is denied with a hint to leave it to the main agent, which can then call it (tested).
- **Upgrade every Worker before you add `Edit` or `Write` to `tool_activities`.** A Worker with an older version of this plugin refuses them, and drops Claude Code's record of their results: with such Workers on the task queue, Claude can be told that a file it edited "has been modified since read". [Worker Versioning](https://docs.temporal.io/worker-versioning) keeps a Workflow on Workers of one version.
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
