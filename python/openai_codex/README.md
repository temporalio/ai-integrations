# OpenAI Codex integration for Temporal: durable Codex agent loops with Workflow-owned approvals

Temporal integration for [OpenAI Codex](https://github.com/openai/codex), published as [`temporalio-openai-codex`](https://pypi.org/project/temporalio-openai-codex/) and imported as `temporalio.openai_codex`.

> **Pre-release.** It relies on Codex app-server APIs that are themselves experimental (`dynamicTools`,
> `thread/inject_items`) and on the rollout file format, so the Codex version is pinned.

Codex runs its agent loop in an external `codex app-server` process, so this integration makes the
*process* durable instead of wrapping model calls. Codex keeps doing what it is good at, running its own
shell, `apply_patch` and other tools inside its sandbox. Temporal adds three things:

- **Workflow-owned approvals.** Every command or file change Codex asks permission for is put to your
  `approval_handler`, which runs in the Workflow. It can wait on a Signal, a human, or a policy for as long
  as it takes, and the decision is recorded in history.
- **The conversation lives in the Workflow.** A turn runs in a segment Activity that rebuilds a fresh
  `CODEX_HOME` from the rollout (Codex's append-only conversation log) the Workflow holds, so any Worker can
  continue the thread.
- **Live visibility.** An observer sees Codex's own tool activity as it happens.

## Install

```bash
uv add 'temporalio-openai-codex[bundled-codex]'
```

The `bundled-codex` extra ships the exact-pinned Codex binary (about 120-160 MB per platform). Without it, set
`CODEX_BIN` or pass `CodexPlugin(codex_bin=...)`.

## Use

```python
from temporalio import workflow
from temporalio.openai_codex import CodexApprovalDecision, CodexApprovalRequest, CodexPlugin
from temporalio.openai_codex.workflow import CodexSession


@workflow.defn
class FixBugs:
    @workflow.init
    def __init__(self, workspace: str) -> None:
        self.decisions: dict[str, bool] = {}
        self.codex = CodexSession(cwd=workspace, approval_handler=self.approve)

    async def approve(self, request: CodexApprovalRequest) -> CodexApprovalDecision:
        # request.kind is "command" or "file_change"; request.command / request.changes say what.
        await workflow.wait_condition(lambda: request.item_id in self.decisions)
        return CodexApprovalDecision(self.decisions[request.item_id])

    @workflow.signal
    def decide(self, item_id: str, approved: bool) -> None:
        self.decisions[item_id] = approved

    @workflow.query
    def pending(self) -> list[CodexApprovalRequest]:
        return list(self.codex.pending_approvals.values())

    @workflow.run
    async def run(self, workspace: str) -> str:
        return (await self.codex.run("Make the failing tests pass.")).text
```

On the Worker, the plugin holds everything that must stay out of history (binary, model provider, credentials):

```python
plugin = CodexPlugin(env={"OPENAI_API_KEY": os.environ["OPENAI_API_KEY"]})
worker = Worker(client, task_queue="codex", workflows=[FixBugs], plugins=[plugin])
```

### Approvals

`CodexSession(approval_policy=...)` is Codex's own policy. `"untrusted"` (the default) asks before running
anything, `"on-request"` runs commands that stay inside the sandbox and asks only when the model requests more.
`sandbox` (`"read-only"`, `"workspace-write"`, `"danger-full-access"`) is Codex's sandbox. With no
`approval_handler`, everything that needs approval is declined.

If a segment crashes after you approved an action, Codex resumes from the last committed rollout and asks again.
The Workflow remembers what was approved earlier in the same turn and answers without calling your handler a
second time.

If a segment crashes while an approval is still **pending**, the retry asks again. The Workflow then cancels the
dead attempt's handler (it can never be answered) and records it as declined with the reason "superseded", so only
the retry's question is left waiting. Until the retry starts (up to the heartbeat timeout, 30 seconds by default)
the old question is still shown. Each request carries the `segment` and `attempt` that asked.

### Host tools

Pass `tools=[codex_tool(fn), activity_as_tool(activity_fn)]` to add your own tools next to Codex's. With
`native_tools=False`, Codex's own tools are turned off and only your tools run. Each call is then an Activity
keyed by the model's call id, so a side effect runs exactly once even if the Worker dies mid-turn.

### Streaming

`CodexSession.run(prompt, observer_context=...)` hands an opaque JSON context to the Worker's `observer_factory`
(an async context manager yielding a `CodexObserver`), which receives `model_interaction_started`,
`reply_delta`, `item_started` / `item_completed` (Codex's own commands and file changes) and
`model_interaction_ended`.

## Testing without credentials

`temporalio.openai_codex.testing.FakeResponsesServer` is a scripted fake of the Responses API. Point the plugin at
it with `CodexPlugin(config_overrides=fake.config_overrides)` and drive the real `codex` binary; each
`CALL:<tool>|<json>` line in the prompt is one tool call, for Codex's own tools (`exec_command`, `apply_patch`) or
yours.

## Security defaults

The defaults are the restrictive ones; loosening any of them is a deliberate act.

| Setting | Default | Loosen with |
|---|---|---|
| Approval policy | `untrusted`: Codex asks before running anything not known to be safe, even `ls` | `approval_policy="on-request"` |
| Sandbox | `workspace-write`: writes only inside `cwd` (and temp dirs), **network off** | `sandbox="read-only"` (tighter) or `"danger-full-access"` |
| Full access | refused unless `allow_full_access=True` is also passed | `CodexSession(allow_full_access=True)` |
| Environment of commands | Codex's `core` set only (`PATH`, `HOME`, `USER`, `SHELL`, `TMPDIR`, ...); the Worker's secrets (`*_KEY`, `*_TOKEN`, `DATABASE_URL`, ...) are **not** visible to commands the model runs | a `shell_environment_policy` entry in `config_overrides` (later overrides win) |
| Auth | the Codex process itself keeps the Worker's environment (it needs the provider key); a login file is copied into a private `0600` `CODEX_HOME` per segment and deleted after | |
| Tools | no web search, no `view_image` in host-tool mode; every approval goes through your handler, and without a handler everything is declined | |

Reads outside the workspace are still allowed by Codex's sandbox. Give the Worker only the files and secrets
it needs, and keep real credentials out of directories Codex can read.

## Cancellation and shutdown

- Cancelling the Workflow (or timing out the turn) cancels the segment Activity. The Activity asks Codex to
  `turn/interrupt` (waiting at most 3 seconds), then closes the app-server, so commands Codex started do not
  outlive the segment. The Workflow waits for this (`WAIT_CANCELLATION_COMPLETED`), and approvals still pending
  are dropped.
- A running segment learns it was cancelled on its next heartbeat, which the SDK throttles to about 80% of
  `heartbeat_timeout`. With the default 30 seconds, cancellation can take up to ~24 seconds; lower
  `heartbeat_timeout` if you need it faster.
- A Worker that shuts down mid-segment cancels the segment in the same way. The segment fails, and the retry
  runs on another Worker from the last committed rollout (the rules of "Limits" apply: an approved command
  that already ran may run again; approvals already given this turn are not asked twice). Set
  `graceful_shutdown_timeout` on the Worker to let a short segment finish first.

## Configuration changes between turns

A thread's rollout records the model provider and model it started with, but **the Worker's current
configuration is authoritative**: each resume passes the Worker's `model_provider` (from `config_overrides`)
and the session's `model` (or the Worker's `model`) to Codex. You can move a conversation to another provider
or model between turns, or to a Worker configured differently.

If a Worker has no `model_provider` that matches what the thread was recorded under, the segment fails with the
non-retryable `CodexConfigDrift` error, saying what to set, rather than retrying forever.

## Experimental Codex APIs

This integration is pinned to one Codex release (`openai-codex-cli-bin==0.156.0` in the `bundled-codex`
extra) because it depends on interfaces Codex marks experimental or does not document as stable. Upgrading
Codex is a deliberate step: rerun the tests, and the replay tests below, against the new binary.

- `dynamicTools` on `thread/start` and the `item/tool/call` server request (host tools).
- `thread/inject_items` (injecting the real result of a host tool call).
- `historyMode: "legacy"` on `thread/start` (self-contained rollouts that can be rebuilt on another Worker).
- The rollout JSONL file format and its location under `CODEX_HOME/sessions`, which the Worker rebuilds.
- Thread config (`features.*`, `web_search`) passed to `thread/start` and `thread/resume`; it does not persist
  across a resume, so it is passed every time.
- `approvalsReviewer: "user"` and the `item/commandExecution/requestApproval` and
  `item/fileChange/requestApproval` server requests (the file changes of a patch come from the earlier
  `item/started`).
- `modelProvider` and `model` on `thread/resume`.
- `turn/interrupt`, and pausing by killing the app-server process rather than interrupting (an interrupt makes
  Codex record its own "aborted" output).
- `shell_environment_policy` and `features.*` config keys.

## Changing the Workflow code

`CodexSession` runs inside your Workflow, so changing it (or your Workflow around it) while Workflows are in
flight is an ordinary Temporal versioning problem. `tests/histories/*.json` are recorded histories of the main
scenarios (host tool, approved and declined native commands, two turns on one thread);
`tests/test_replay.py` replays them against the current code and fails on any non-determinism. When you change
the Workflow-side code:

1. Guard behaviour changes with `workflow.patched("<change-id>")` so histories recorded before the change
   still replay, and keep the guard until no such Workflow can still be running.
2. Run `make test`; a determinism failure on a recorded history means the change needs a patch.
3. Record new histories only after deciding the change is safe:
   `CODEX_RECORD_HISTORIES=1 make test PYTEST_ARGS="-k record_histories"`.

## Limits

- **Native tool effects are not exactly-once.** A command that ran just before a crash runs again when Codex
  resumes and repeats it. Approvals are not repeated within a turn, but side effects are. Use host tools
  (`native_tools=False`) when a call must run once.
- **The workspace must exist where the segment runs.** `cwd` is a path on the Worker. Files Codex wrote do not
  move with a failover Worker; use a shared volume or the same path on every Worker.
- **A pending approval holds a segment open.** It heartbeats, and `segment_timeout` (an hour by default) must
  outlast your longest wait. If the Worker dies meanwhile, the next attempt asks again.
- **The sandbox is Codex's.** It restricts writes to the workspace but allows reads elsewhere; see "Security
  defaults".
- Host tools run one at a time, and each call restarts the app-server (about 0.1s).
- The rollout is sent in full on each segment's input; very long conversations need Continue-As-New and payload
  offload.

## Documentation

- Integration guide: https://docs.temporal.io/develop/python/integrations/openai-codex
- Repository conventions: [`AGENTS.md`](https://github.com/temporalio/ai-integrations/blob/main/AGENTS.md)

## Develop

```bash
make sync   # install (non-editable) into .venv
make lint
make test   # drives the real Codex binary against a local fake Responses API; no credentials
```
