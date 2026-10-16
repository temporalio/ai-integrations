"""The Claude turn Activity: the only place a Claude Code CLI subprocess runs.

One ``ClaudeSDKClient`` per attempt, with an independent heartbeat, a bounded interrupt and
drain on cancellation, and a bounded disconnect. The SDK's anyio-shielded close escalation is
skipped by the raw asyncio cancellation Temporal uses, so disconnect runs in its own task and
the CLI is killed on overrun. Locally the CLI runs through ``_launcher.py`` as leader of its own
process group, recorded in a per-attempt file named by the lease generation; a CLI with no
record fails the turn rather than run unkillable. Bash sessions and detached processes are
found by parent pid and by the launcher's ``PGID_FILE_ENV`` marker. A ``SandboxRuntime``
replaces all of this with its ``terminate`` and ``terminate_strays``.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import (
    AsyncExitStack,
    asynccontextmanager,
    suppress,
)
from dataclasses import dataclass, field, replace
from datetime import timedelta
from pathlib import Path, PurePath, PurePosixPath
from typing import TYPE_CHECKING, Any, NoReturn, Protocol, cast

from temporalio import activity
from temporalio.claude_agent_sdk import _launcher
from temporalio.claude_agent_sdk._bridge import (
    ToolBridgeClient,
    mcp_server,
)
from temporalio.claude_agent_sdk._cli import (
    CLIResolutionError,
    ResolvedCLI,
    resolve_cli,
)
from temporalio.claude_agent_sdk._lease import (
    LEASE_DIR_NAME,
    FileLeaseStore,
    Lease,
    LeaseStore,
    SessionLeased,
    SessionLeaseLost,
    fenced_session_store,
    session_lease,
)
from temporalio.claude_agent_sdk._models import (
    BRIDGE_MCP_SERVER,
    CLAUDE_CLI_UNTRACKED_ERROR_TYPE,
    CLAUDE_RESULT_ERROR_TYPE,
    CLAUDE_SANDBOX_LOST_ERROR_TYPE,
    CLAUDE_SESSION_LEASE_LOST_ERROR_TYPE,
    CLAUDE_SESSION_LEASED_ERROR_TYPE,
    CLAUDE_TURN_CONFIG_ERROR_TYPE,
    CLAUDE_TURN_REJECTED_ERROR_TYPE,
    CLAUDE_WORKSPACE_PERSIST_ERROR_TYPE,
    CLAUDE_WORKSPACE_RESTORE_ERROR_TYPE,
    CONTINUE_PROMPT,
    NATIVE_DEFAULT_MAX_TURNS,
    NEVER_RESUMED_ERROR_KINDS,
    READ_ONLY_NATIVE_TOOLS,
    REPEATABLE_EFFECTS,
    RESUMABLE_DETAIL,
    RUN_CLAUDE_TURN_ACTIVITY,
    TURN_PROGRESS_SIGNAL,
    ClaudeDeferredTool,
    ClaudePartialEvent,
    ClaudeTurnInput,
    ClaudeTurnProgress,
    ClaudeTurnResult,
    anthropic_usage,
    cli_tools,
    mcp_tool_name,
    native_tool_effect,
    partial_event,
    tool_call_key,
)
from temporalio.claude_agent_sdk._native import (
    DEFAULT_NATIVE_ENV_DENYLIST,
    ENV_NAME,
    NATIVE_TOOLS_VERIFIED,
    REMOTE_CREDENTIAL_ENV,
    NativeToolHooks,
    answer_call,
    content_blocks,
    ledger,
    merge_native_calls,
    native_calls,
    native_tools_refusal,
    prepare_state,
    record_native_finished,
    result_entry,
    sandbox_settings,
    settle_local_transcript,
)
from temporalio.claude_agent_sdk._native import (
    NATIVE_STATE_DIR_NAME as _NATIVE_STATE_DIR_NAME,
)
from temporalio.claude_agent_sdk._native_hook import call_key
from temporalio.claude_agent_sdk._sdk import load_sdk
from temporalio.claude_agent_sdk._tracing import (
    TurnTrace,
    cli_trace_env,
    turn_span,
)
from temporalio.claude_agent_sdk._workspace import (
    WorkspaceStore,
)
from temporalio.claude_agent_sdk.sandbox import (
    SandboxError,
    SandboxLost,
    SandboxRuntime,
    SandboxTurn,
    merge_remote_native_state,
    pull_state,
    push_state,
    sandbox_hook_command,
    sandbox_state_dir,
    settle_sandbox_transcript,
)
from temporalio.contrib.workflow_streams import WorkflowStreamClient
from temporalio.exceptions import ApplicationError

if TYPE_CHECKING:
    from claude_agent_sdk import (
        AssistantMessage,
        ClaudeAgentOptions,
        ClaudeSDKClient,
        MirrorErrorMessage,
        RateLimitEvent,
        ResultError,
        ResultMessage,
        SessionStore,
        StreamEvent,
        ToolResultBlock,
        UserMessage,
    )

logger = logging.getLogger(__name__)

_FALLBACK_HEARTBEAT_INTERVAL = timedelta(seconds=10)
CANCEL_LATENCY = timedelta(seconds=1)
"""How soon a running turn hears of a cancellation. Temporal delivers an Activity's cancellation
only in a heartbeat response, and the SDK throttles heartbeats to 80% of the heartbeat timeout
(24 s for 30 s), so the turn heartbeats this often and ``ClaudeAgentPlugin`` caps the worker's
``max_heartbeat_throttle_interval`` at the same value."""
_KILL_GRACE_SECONDS = 2.0
_SANDBOX_STATE_TIMEOUT_SECONDS = 20.0

# Claude Code's nonessential traffic (auto-updates, telemetry, error reporting, feedback, release
# notes, feature flags: https://code.claude.com/docs/en/env-vars) holds the CLI open for about a
# second after each turn's result. Off unless the worker's own environment sets the variable; an
# empty value there turns the traffic back on.
NONESSENTIAL_TRAFFIC_ENV = "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"

# Claude Code's own restart recovery (`CLAUDE_CODE_RESUME_*`, meant for its cloud workers)
# changes how `--resume` rebuilds a transcript and re-runs interrupted turns, and background
# tasks let a command outlive the run (and, on 2.1.294, leave an empty result ahead of the next
# prompt's: anthropics/claude-code#102162). The local CLI inherits the worker's environment, so
# inherited `CLAUDE_CODE_RESUME_*` values are blanked and these are always set. A managed
# CLAUDE.md loads despite `setting_sources=[]`, auto memory lives outside the workspace (so it is
# neither durable nor the session's), and claude.ai connectors are not the turn's MCP servers.
_RESUME_ENV_PREFIX = "CLAUDE_CODE_RESUME_"
_CLI_ENV = {
    "CLAUDE_CODE_RESUME_INTERRUPTED_TURN": "0",
    "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS": "1",
    "CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1",
    "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
    "ENABLE_CLAUDEAI_MCP_SERVERS": "false",
}

# One stream-json line holds a whole message; the eager SessionStore mirror echoes the prompt and
# each round's tool results. The SDK's 1 MiB default fails a turn on a larger prompt after it was
# sent. 32 MiB matches the Messages API request limit, which no round the CLI sent can exceed.
_MAX_MESSAGE_BYTES = 32 * 1024 * 1024

_WORKSPACE_DIR_NAME = "temporal-claude-agent-sdk"
_PGID_DIR_NAME = ".cli-process-group"
_LEDGER_WATCH_SECONDS = 0.2


class ClaudeTurnActivity:
    """Holds worker-side configuration for :data:`RUN_CLAUDE_TURN_ACTIVITY`.

    Args:
        cli_path: Claude Code executable; the CLI bundled in the SDK wheel when None. Resume
            and tool-call dedup require its ``--version`` to be verified for this SDK in
            ``NATIVE_TOOLS_VERIFIED``.
        interrupt_timeout: Bound on ``interrupt()`` after a cancellation.
        drain_timeout: Bound on reading the interrupted run's final result.
        disconnect_timeout: Bound on ``disconnect()`` before the CLI process is killed.
        session_store: ``claude_agent_sdk.SessionStore`` that mirrors transcripts eagerly so
            any worker can resume a session. Without one, a session lives on one worker's disk.
        workspace_root: Parent of each session's working directory; must be the same path on
            every worker, because the store keys transcripts by it.
        enable_native_tools: Let turns request Claude's built-in tools, only when the SDK and
            the turn's CLI are a pair in ``NATIVE_TOOLS_VERIFIED``. Native effects are
            at-least-once across host failure.
        lease_store: Where session leases live. The default is host-local; with a
            ``session_store`` or ``sandbox``, resuming a lost attempt needs a store every
            worker shares.
        workspace_store: Where session working directories are snapshotted, so files native
            tools write follow the session to another worker.
        native_env_denylist: ``fnmatch`` patterns of worker environment variables native Bash
            never sees; the CLI itself keeps them. ``()`` turns the scrubbing off.
        sandbox: Run each turn's CLI in a remote :class:`SandboxRuntime` whose snapshot is the
            session, so it cannot be combined with ``session_store`` or ``workspace_store``.
        partial_sink: Called in the Activity with each batch of every turn's
            :class:`ClaudePartialEvent` partials and that turn's ``streaming_context`` (None
            when the workflow set none), beside any ``streaming_topic``. Batches are in order,
            at most one per ``streaming_batch_interval``, and the last is delivered before the
            Activity returns. It may be async; it runs off the CLI stream's path, but a sink
            that never returns holds the turn open. Delivery is as advisory as the topic's: a
            retried or resumed attempt sends its partials again under a new ``attempt_id``. An
            exception from the sink is logged and the turn continues.
        require_seccomp_helper: On Linux, refuse native tools unless sandbox-runtime's seccomp
            helper is installed globally at ``SANDBOX_RUNTIME_PIN``. ``False`` is an explicit
            opt-out: Bash's sandbox then lacks that filter, and the worker logs a warning. No
            effect on macOS or with a ``sandbox``.
    """

    def __init__(
        self,
        *,
        cli_path: str | None = None,
        interrupt_timeout: timedelta = timedelta(seconds=5),
        drain_timeout: timedelta = timedelta(seconds=10),
        disconnect_timeout: timedelta = timedelta(seconds=15),
        session_store: SessionStore | None = None,
        workspace_root: str | Path | None = None,
        enable_native_tools: bool = False,
        lease_store: LeaseStore | None = None,
        workspace_store: WorkspaceStore | None = None,
        native_env_denylist: Sequence[str] = DEFAULT_NATIVE_ENV_DENYLIST,
        sandbox: SandboxRuntime | None = None,
        partial_sink: PartialSink | None = None,
        require_seccomp_helper: bool = True,
    ) -> None:
        """Store worker-side configuration. See the class docstring for parameters."""
        if sandbox is not None and (
            session_store is not None or workspace_store is not None
        ):
            raise ValueError(
                "a sandbox keeps the session in its snapshots, so it cannot be "
                "combined with session_store or workspace_store"
            )
        self._sandbox = sandbox
        self._partial_sink = partial_sink
        self._enable_native_tools = enable_native_tools
        self._require_seccomp_helper = require_seccomp_helper
        if (
            enable_native_tools
            and not require_seccomp_helper
            and sandbox is None
            and sys.platform.startswith("linux")
        ):
            logger.warning(
                "require_seccomp_helper=False: native Bash on this worker runs without "
                "sandbox-runtime's seccomp filter"
            )
        self._native_env_denylist = tuple(native_env_denylist)
        self._session_store = session_store
        self._workspace_store = workspace_store
        self._workspace_root = (
            Path(workspace_root)
            if workspace_root is not None
            else Path(tempfile.gettempdir()) / _WORKSPACE_DIR_NAME
        )
        self._lease_store = lease_store or FileLeaseStore(
            self._workspace_root / LEASE_DIR_NAME, shared=False
        )
        self._cli_path = cli_path
        self._resolved_cli: dict[bool, ResolvedCLI] = {}
        self._launchers: dict[str, str] = {}
        self._interrupt_timeout = interrupt_timeout.total_seconds()
        self._drain_timeout = drain_timeout.total_seconds()
        self._disconnect_timeout = disconnect_timeout.total_seconds()

    @activity.defn(name=RUN_CLAUDE_TURN_ACTIVITY)
    async def run_turn(self, input: ClaudeTurnInput) -> ClaudeTurnResult:
        """Run one Claude agent loop for ``input.prompt`` and return its result."""
        with turn_span(input) as trace:
            return await self._run_turn(input, trace)

    async def _run_turn(
        self, input: ClaudeTurnInput, trace: TurnTrace
    ) -> ClaudeTurnResult:
        if self._sandbox is None:
            _require_posix()
            _require_parent_death_watch()
        sdk = load_sdk()
        _validate_input(input)
        if (
            self._sandbox is not None
            and cli_tools(input)
            and not self._sandbox.supports_native_tools
        ):
            raise ApplicationError(
                "Claude turn rejected: this worker's sandbox runtime cannot run native Claude "
                'tools (Modal needs ModalSandboxRuntime(runtime="vm"): Claude Code\'s Bash '
                "sandbox does not start under gVisor)",
                type=CLAUDE_TURN_CONFIG_ERROR_TYPE,
                non_retryable=True,
            )
        cli = await self._cli(
            native=bool(cli_tools(input)) and self._enable_native_tools
        )
        if refusal := native_tools_refusal(
            input,
            enabled=self._enable_native_tools,
            sdk_version=sdk.__version__,
            cli=cli,
            in_sandbox=self._sandbox is not None,
            require_seccomp_helper=self._require_seccomp_helper,
        ):
            _reject(refusal)
        if (
            input.resumes_lost_attempt
            and not self._lease_store.shared
            and (self._session_store is not None or self._sandbox is not None)
        ):
            _reject(
                "it resumes a lost attempt, which may still be running on another worker, and "
                "this worker's lease store is not shared, so it cannot see that attempt's lease; "
                "configure ClaudeTurnActivity(lease_store=...) on storage every worker shares"
            )
        state_dir = self._native_state(input.session_id) if cli_tools(input) else None
        launcher = None if self._sandbox is not None else self._launcher(cli.path)
        group = _CliGroup()
        mirror_errors: list[str] = []
        stderr: deque[str] = deque(maxlen=_STDERR_TAIL_LINES)

        def on_stderr(line: str) -> None:
            logger.debug("Claude CLI stderr: %s", line)
            if line.strip():
                stderr.append(line[-_STDERR_TAIL_CHARS:])

        # Only a store copy of the transcript, mirrored eagerly, lets another attempt pick up
        # where this one stopped. A native call's ledger travels through the workflow, and a
        # native resume additionally needs a lease every worker sees. A remote snapshot is that
        # copy, and the lease generation is what makes one worker's sweep unable to kill
        # another's newer sandbox, so a sandbox failover also needs a shared lease store. A lost
        # worker's attempt may still be running, so its retry needs a shared lease store too.
        resumable = (
            self._sandbox is not None
            and input.session_id is not None
            and self._lease_store.shared
        ) or (
            self._session_store is not None
            and input.session_id is not None
            and (not cli_tools(input) or self._lease_store.shared)
        )
        cwd = self._workspace(input.session_id)
        progress = _Progress(
            input,
            state_dir,
            self._workspace_store,
            cwd,
            resumable,
            mirror_errors,
            shared=self._lease_store.shared,
            sandboxed=self._sandbox is not None,
        )
        # The lease comes first so the first heartbeat, the only one the server records at once,
        # already names it; acquiring heartbeats on its own while it waits.
        async with (
            _leased(self._lease_store, input, group, progress) as lease,
            _heartbeat(progress.details),
            _bridge(input, progress) as bridge,
            _sandbox_turn(self._sandbox, input, lease, resumable) as sandbox,
        ):
            group.remote = sandbox.kill if sandbox is not None else None
            remote_state = None
            pull: Callable[[], Awaitable[None]] | None = None
            if sandbox is not None and state_dir is not None:
                # The hook runs in the sandbox, so its files do too; this host's state directory
                # is the mirror the rest of the Activity reads.
                assert self._sandbox is not None
                assert input.session_id is not None
                remote_state = sandbox_state_dir(
                    self._sandbox.cwd(input.session_id), input.session_id
                )
                pull = _state_puller(sandbox, remote_state, state_dir)
            plan = await self._plan(input)
            input = input.model_copy(update={"resume": plan.resume})
            # Claude Code restores a session's cost from the transcript it resumes when that
            # transcript holds it: its own file, here or in a sandbox snapshot, does; a
            # SessionStore's copy does not.
            progress.cumulative_cost = plan.resume and self._session_store is None
            # The sandbox already refused an unverified CLI. A local version of None would
            # fail this check on every resume.
            if self._sandbox is None and (
                refusal := _unverified_cli_resume(
                    self._cli_path, cli, plan.resume, sdk.__version__
                )
            ):
                _reject(refusal)
            if bridge is not None:
                bridge.number(
                    input.tool_ordinals or {},
                    plan.prior,
                    input.tool_calls,
                    plan.answered,
                )
            await self._answer_dangling(input, plan, bridge, lease)
            if launcher is not None:
                group.path = self._pgid_file(input.session_id, lease)
                # Before the files and the state are rewritten: a leftover CLI still writes them.
                await asyncio.to_thread(_kill_leftover_groups, group.path, lease)
            await self._restore_workspace(input, cwd)
            hooks = None
            if state_dir is not None:
                deny_dirs: list[PurePath] = []
                if remote_state:
                    assert sandbox is not None
                    assert self._sandbox is not None and input.session_id is not None
                    session_dir = PurePosixPath(self._sandbox.cwd(input.session_id))
                    deny_dirs += [
                        PurePosixPath(remote_state).parent,
                        session_dir / ".git",
                    ]
                    # Every session's transcripts.
                    try:
                        config = await sandbox.config_dir()
                    except SandboxLost as exc:  # before the CLI started: nothing ran
                        raise _sandbox_lost(exc, input, 0, resumable) from exc
                    if config is not None:
                        deny_dirs.append(PurePosixPath(config))
                else:
                    deny_dirs.append(state_dir.parent)
                    if group.path is not None:
                        deny_dirs.append(group.path.parent)
                    if cwd is not None:
                        deny_dirs.append(cwd / ".git")
                    if config := os.environ.get("CLAUDE_CONFIG_DIR"):
                        deny_dirs.append(Path(config))
                hooks = NativeToolHooks(
                    state_dir,
                    progress.after_tool,
                    on_settled=progress.report,
                    deny_dirs=deny_dirs,
                    command=sandbox_hook_command(remote_state)
                    if remote_state
                    else None,
                    pull=pull,
                    tools=cli_tools(input),
                )
            # The ledger is the previous run's until prepare_state deletes it.
            prepared = False
            try:
                try:
                    input = await self._settle_transcript(input, sandbox, cwd, lease)
                    if state_dir is not None:
                        prepare_state(state_dir, input)
                        prepared = True
                        if sandbox is not None and remote_state is not None:
                            try:
                                # The snapshot's ledger is the record of effects the workflow
                                # may not have. Merge it before the host copy replaces it.
                                await merge_remote_native_state(
                                    sandbox.sandbox, remote_state, state_dir
                                )
                                await push_state(
                                    sandbox.sandbox, remote_state, state_dir
                                )
                            except Exception:
                                sandbox.ledger_unreadable = True
                                raise
                    options = self._options(
                        input,
                        bridge,
                        hooks,
                        group.path,
                        launcher or cli.path,
                        fenced_session_store(self._session_store, lease),
                    )
                    transport = None
                    if sandbox is None:
                        # The SDK pipes the CLI's stderr only to a callback; without one a
                        # crash reports just the exit code.
                        options = replace(options, stderr=on_stderr)
                    elif self._sandbox is not None:
                        # The CLI's directory in the sandbox: the same path on every sandbox.
                        in_sandbox = self._sandbox.cwd(
                            input.session_id or f"turn-{uuid.uuid4()}"
                        )
                        options = replace(options, cwd=in_sandbox)
                        transport = sandbox.new_transport(options)
                    client = sdk.ClaudeSDKClient(options, transport=transport)
                    lost_sandbox: SandboxLost | None = None
                    async with progress.watching():
                        try:
                            group.started = True
                            await client.connect()
                            if group.path is not None and group.pgid is None:
                                _untracked(
                                    "the launcher recorded no process group for the CLI"
                                )
                            if not _resumes_approved_call(input):
                                await client.query(
                                    _prompt(plan.prompt, plan.prompt_uuid)
                                )
                            result = await _consume(
                                client,
                                mirror_errors,
                                progress,
                                trace,
                                self._partial_sink,
                            )
                        except asyncio.CancelledError:
                            if bridge is not None:
                                # Ordinary cancellation unblocks calls before interrupting.
                                bridge.close("the turn was cancelled")
                            if bridge is not None and bridge.failure is not None:
                                _kill(group.pgid, _cli_process(client))
                                if group.remote is not None:
                                    group.remote()
                            else:
                                await self._interrupt_and_drain(client, progress)
                            raise
                        except SandboxLost as exc:
                            lost_sandbox = exc
                            raise
                        finally:
                            await self._disconnect(client, group)
                            if group.path is not None:
                                group.path.unlink(missing_ok=True)
                            if pull is not None:
                                try:
                                    await pull()  # the CLI's last ledger writes
                                except Exception as exc:
                                    assert sandbox is not None
                                    sandbox.ledger_unreadable = True
                                    _note_unreadable_native(state_dir, input)
                                    if lost_sandbox is None:
                                        raise SandboxError(
                                            "the sandbox native-tool ledger could not be "
                                            f"read, so this attempt is not committed: {exc}"
                                        ) from exc
                                    logger.warning(
                                        "the sandbox native-tool ledger could not be read "
                                        "after the sandbox was lost: %s",
                                        exc,
                                    )
                        if (
                            lost_sandbox is None
                            and sandbox is not None
                            and sandbox.transport is not None
                            and sandbox.transport.lost
                        ):
                            raise SandboxLost("its CLI could not be observed")
                except SandboxLost as exc:
                    raise _sandbox_lost(
                        exc,
                        input,
                        progress.num_turns,
                        resumable,
                        native_calls=progress.native_calls(),
                        cost_usd=progress.cost_usd,
                        cumulative_cost=progress.cumulative_cost,
                        ledger_unreadable=sandbox is not None
                        and sandbox.ledger_unreadable,
                    ) from exc
            except Exception as exc:
                # A lost sandbox already named the last committed snapshot. Do not replace
                # that with a snapshot of the attempt that was lost.
                if (
                    isinstance(exc, ApplicationError)
                    and exc.type == CLAUDE_SANDBOX_LOST_ERROR_TYPE
                ):
                    raise
                # The run may have had effects; its files are kept and its ledger reported.
                # A committed revision has to ride the failure, or the workflow resumes the old tree.
                # A failed persist must not replace the run's own failure, nor drop its spend.
                try:
                    revision = await self._persist_workspace(input, cwd, progress)
                except ApplicationError as persist:
                    logger.warning("%s", persist)
                    revision = progress.workspace_revision
                snapshot = None
                if sandbox is None or not sandbox.ledger_unreadable:
                    snapshot = await self._snapshot(
                        sandbox, input, progress, failing=True
                    )
                # Heartbeat fields ride the failure too. A CLI that exits without a result is not
                # an answer: the workflow resumes it when ``resumable`` is set, and charges what
                # this attempt had already reported.
                payload = {
                    **progress.details(lost=False),
                    "native_calls": progress.native_calls(),
                    "workspace_revision": revision,
                    "sandbox_snapshot": snapshot,
                    "session_store": self._session_store is not None,
                    # A failed run's re-requests still count against the session's cap.
                    "native_rerequests": _rerequests(state_dir if prepared else None),
                }
                if sandbox is not None:
                    # The snapshot holds this run's transcript, and a sandboxed attempt does not
                    # plan from it: a new session over it is refused as in use, and a resumed one
                    # would be sent the prompt again.
                    payload[RESUMABLE_DETAIL] = False
                    if sandbox.ledger_unreadable:
                        payload["sandbox_snapshot"] = None
                tail = _stderr_tail(stderr)
                if tail:
                    payload["stderr_tail"] = tail
                if (
                    isinstance(exc, ApplicationError)
                    and exc.details
                    and isinstance(exc.details[0], dict)
                ):
                    exc.details[0].update(payload)
                    raise
                if isinstance(exc, ApplicationError):
                    raise
                raise ApplicationError(
                    f"Claude run failed: {exc}"
                    + (f"\nCLI stderr: {tail}" if tail else ""),
                    payload,
                    type=CLAUDE_RESULT_ERROR_TYPE,
                    non_retryable=True,
                ) from exc
            try:
                revision = await self._persist_workspace(input, cwd, progress)
                snapshot = await self._snapshot(sandbox, input, progress, failing=False)
            except ApplicationError as exc:
                # The run completed, and the session pays for it.
                exc.details[0].update(
                    total_cost_usd=_reported_cost(result),
                    num_turns=result.num_turns,
                    cumulative_cost=progress.cumulative_cost,
                    native_rerequests=_rerequests(state_dir),
                    # Calls the missing snapshot does not hold stay started.
                    native_calls=progress.native_calls(),
                )
                raise
        return _to_turn_result(
            result,
            mirror_errors,
            _rerequests(state_dir),
            progress.native_calls(),
            revision,
            snapshot,
            progress.sandbox_violations,
            progress.cumulative_cost,
        )

    async def _settle_transcript(
        self,
        input: ClaudeTurnInput,
        sandbox: SandboxTurn | None,
        cwd: Path | None,
        lease: Lease | None,
    ) -> ClaudeTurnInput:
        """Settle answered deferrals in the transcript a resumed CLI reads, after answering a
        settling decision's call there; return the input without that decision.

        See ``_native.settle_answered_deferrals``. A SessionStore's copy is settled on load. A
        settled call has its result before the CLI starts, so the CLI has nothing deferred to
        re-present and the run is an ordinary turn on the message's prompt. The answer is
        written once: a retried attempt that finds it writes nothing more.
        """
        decision = input.native_decision
        answer = (
            (decision.call.tool_use_id, decision.reason or "")
            if decision is not None and decision.settle
            else None
        )
        session_id = input.session_id
        if not input.resume or session_id is None:
            return input
        if sandbox is not None and self._sandbox is not None:
            await settle_sandbox_transcript(
                sandbox.sandbox, self._sandbox.cwd(session_id), session_id, answer
            )
        elif self._session_store is None and cwd is not None:
            await asyncio.to_thread(
                settle_local_transcript, _local_transcript(cwd, session_id), answer
            )
        elif answer is not None and cwd is not None:
            key = {
                "project_key": load_sdk().project_key_for_directory(cwd),
                "session_id": session_id,
            }
            store = fenced_session_store(self._session_store, lease)
            entries = await store.load(key) or []  # type: ignore[arg-type]
            if entry := answer_call(entries, *answer):  # type: ignore[arg-type]
                await store.append(key, [entry])  # type: ignore[arg-type]
        return input.model_copy(update={"native_decision": None}) if answer else input

    async def _restore_workspace(
        self, input: ClaudeTurnInput, cwd: Path | None
    ) -> None:
        """Put the revision the workflow recorded into the session's working directory."""
        if self._workspace_store is None or input.session_id is None or cwd is None:
            return
        try:
            if input.workspace_revision is not None:
                restore = asyncio.create_task(
                    self._workspace_store.restore(
                        input.session_id, input.workspace_revision, cwd
                    )
                )
                try:
                    await asyncio.shield(restore)
                except asyncio.CancelledError:
                    # Cancelling a thread-backed restore does not stop its writes. Keep the
                    # enclosing lease renewing until it finishes, even on repeated cancellation.
                    while not restore.done():
                        try:
                            await asyncio.shield(restore)
                        except asyncio.CancelledError:
                            continue
                        except Exception:  # noqa: BLE001 - cancellation remains the outcome
                            break
                    if not restore.cancelled():
                        restore.exception()
                    raise
        except Exception as exc:
            raise ApplicationError(
                f"Claude turn rejected: workspace revision {input.workspace_revision} of session "
                f"{input.session_id} could not be restored: {exc!r}",
                {
                    "session_id": input.session_id,
                    "workspace_revision": input.workspace_revision,
                },
                type=CLAUDE_WORKSPACE_RESTORE_ERROR_TYPE,
                non_retryable=True,
            ) from exc

    async def _persist_workspace(
        self, input: ClaudeTurnInput, cwd: Path | None, progress: _Progress
    ) -> int | None:
        """Snapshot the session's working directory after a run; its new revision, or None
        without a store. A failure fails the turn; the workflow keeps the revision it had.
        """
        try:
            return await _persist(self._workspace_store, input.session_id, cwd)
        except Exception as exc:
            raise ApplicationError(
                f"Claude turn: the workspace of session {input.session_id} could not be "
                f"persisted after the run: {exc!r}. The session keeps revision "
                f"{input.workspace_revision}.",
                {
                    "session_id": input.session_id,
                    "native_calls": progress.native_calls(),
                    # The last revision that actually committed. None, or the revision this run
                    # started from, when the snapshot after the effect failed.
                    "workspace_revision": progress.workspace_revision,
                },
                type=CLAUDE_WORKSPACE_PERSIST_ERROR_TYPE,
                non_retryable=True,
            ) from exc

    async def _snapshot(
        self,
        sandbox: SandboxTurn | None,
        input: ClaudeTurnInput,
        progress: _Progress,
        *,
        failing: bool,
    ) -> str | None:
        """Snapshot the turn's sandbox after the run; its image id, or None without a sandbox.

        A snapshot that cannot be taken fails the turn (the session keeps the snapshot it had,
        and one resumed from it would forget this turn), unless the turn is already failing.
        A lost or stuck CLI has nothing new to commit. The run's non-repeatable calls are
        settled once the snapshot holds their effects.
        """
        if sandbox is None:
            return None
        try:
            snapshot = await sandbox.snapshot()
            progress.sandbox_snapshot = snapshot
            progress.settle_snapshotted()
            return snapshot
        except SandboxError as exc:
            if failing:
                logger.warning(
                    "Claude turn failed and its sandbox was not snapshotted: %s", exc
                )
                return None
            raise ApplicationError(
                f"Claude turn: the sandbox of session {input.session_id} could not be "
                f"snapshotted after the run: {exc}. The session keeps snapshot "
                f"{input.sandbox_snapshot}.",
                {
                    "session_id": input.session_id,
                    "sandbox_snapshot": input.sandbox_snapshot,
                },
                type=CLAUDE_WORKSPACE_PERSIST_ERROR_TYPE,
                non_retryable=True,
            ) from exc

    async def _plan(self, input: ClaudeTurnInput) -> _Plan:
        """Resume or start, and which prompt to send, from the transcript that actually exists.

        A store already holding this prompt's uuid resumes with ``CONTINUE_PROMPT``, since
        re-sending the prompt would restart the task. The harness calls stored after the prompt,
        or after the call a decision run resumes (it sends none), are this run's so far.
        """
        plan = _Plan(
            resume=input.resume, prompt=input.prompt, prompt_uuid=input.prompt_uuid
        )
        cwd = self._workspace(input.session_id)
        if input.session_id is None or cwd is None:
            return plan
        local = _local_transcript(cwd, input.session_id)
        if self._session_store is None:
            return replace(plan, resume=input.resume or local.exists())
        key = {
            "project_key": load_sdk().project_key_for_directory(cwd),
            "session_id": input.session_id,
        }
        loaded = await self._session_store.load(key) or []  # type: ignore[arg-type]
        # A retried mirror batch can land twice in a store that does not dedupe by uuid; the
        # CLI's loader keys entries by uuid, so they are counted once here too.
        entries = list({e.get("uuid") or i: e for i, e in enumerate(loaded)}.values())
        if not entries:
            if input.resume and not _pending_native_continuation(input):
                # The workflow already completed a run, so the shared transcript must exist.
                # Starting fresh or `--resume` of this worker's local file would drop that
                # conversation; the real CLI also exits "No conversation found" with no file.
                _reject(
                    f"session {input.session_id} has completed a run but the SessionStore "
                    "has no transcript"
                )
            if not input.resume:
                # A lost attempt on this worker may have written it before the store heard of the
                # session; the CLI refuses to start a session over it ("already in use").
                local.unlink(missing_ok=True)
            return plan
        at = next(
            (
                i
                for i in range(len(entries) - 1, -1, -1)
                if input.prompt_uuid and entries[i].get("uuid") == input.prompt_uuid
            ),
            None,
        )
        stored = _harness_calls(entries)  # type: ignore[arg-type]
        answered = frozenset(call_id for call_id, _, done in stored if done)
        if at is not None:
            return _Plan(
                resume=True,
                prompt=CONTINUE_PROMPT,
                prior=_harness_calls(entries[at + 1 :]),  # type: ignore[arg-type]
                dangling=_dangling_calls(entries[at + 1 :]),  # type: ignore[arg-type]
                answered=answered,
            )
        # A settled call belongs to an earlier message: this run is that message's successor.
        decision = input.native_decision
        decided = (
            decision.call.tool_use_id if decision and not decision.settle else None
        )
        after = next(
            (
                i
                for i in range(len(entries) - 1, -1, -1)
                if decided
                and any(
                    b.get("type") == "tool_use" and b.get("id") == decided
                    for b in content_blocks(entries[i])  # type: ignore[arg-type]
                )
            ),
            None,
        )
        # A run that resumes a decided native call has no prompt entry (the CLI writes its own
        # "Continue"), so its calls are the ones from the decided call's message on.
        return replace(
            plan,
            resume=True,
            prior=_harness_calls(entries[after + 1 :]) if after is not None else [],  # type: ignore[arg-type]
            dangling=_dangling_calls(entries[after:]) if after is not None else [],  # type: ignore[arg-type]
            answered=answered,
        )

    async def _answer_dangling(
        self,
        input: ClaudeTurnInput,
        plan: _Plan,
        bridge: ToolBridgeClient | None,
        lease: Lease | None,
    ) -> None:
        """Answer, in the stored transcript, each workflow-tool call a lost attempt left without
        a result, before the CLI starts.

        The resumed CLI would patch such a call as interrupted with an unknown outcome, and a
        model told that may never ask again, losing a pending approval. The workflow knows the
        call: asking it under the call's own id attaches to the record (a running call is not
        run again) or runs a call that never reached it once. The answers are appended as the
        CLI writes a result, under the lease, so the resumed CLI sees finished calls.
        """
        cwd = self._workspace(input.session_id)
        if (
            bridge is None
            or not plan.dangling
            or self._session_store is None
            or cwd is None
        ):
            return
        prefix = mcp_tool_name("")
        results = await asyncio.gather(
            *(
                bridge.call(
                    block["id"],
                    block["name"].removeprefix(prefix),
                    block.get("input") or {},
                )
                for _, block in plan.dangling
            )
        )
        if bridge.closed:
            return  # the CLI patches them, as without this step
        key = {
            "project_key": load_sdk().project_key_for_directory(cwd),
            "session_id": input.session_id,
        }
        answers = [
            result_entry(entry, result)
            for (entry, _), result in zip(plan.dangling, results, strict=True)
        ]
        await fenced_session_store(self._session_store, lease).append(key, answers)  # type: ignore[arg-type]

    def _workspace(self, session_id: str | None) -> Path | None:
        """The session's working directory, created on first use; none without a session."""
        if session_id is None or self._sandbox is not None:
            return None  # a sandboxed session's directory is in the sandbox
        path = self._workspace_root / session_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _native_state(self, session_id: str | None) -> Path:
        """The session's native-tool state directory: stable across attempts, and outside the
        workspace, so the sandboxed shell cannot rewrite the decisions that gate it.
        """
        assert session_id is not None  # native_tools_refusal requires one
        path = self._workspace_root / _NATIVE_STATE_DIR_NAME / session_id
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        return path

    def _pgid_file(self, session_id: str | None, lease: Lease | None) -> Path:
        """Where this attempt's CLI records its process group, the handle every kill uses.
        Outside the workspace, so the sandboxed shell cannot point a kill at another process.
        One file per attempt, ``<session_id>.<lease generation>``: an attempt that was taken over
        and resumes must never find its successor's group in its own record, and the next
        attempt sweeps only the generations before its own. Unique to a sessionless turn.
        """
        name = (
            f"{session_id}.{lease.generation}"
            if lease is not None
            else f"turn-{uuid.uuid4()}"
        )
        path = self._workspace_root / _PGID_DIR_NAME / name
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        return path

    async def _cli(self, *, native: bool) -> ResolvedCLI:
        """The CLI this worker's turns run, resolved (and its version probed) once per worker;
        ``native`` turns resolve without the ``PATH`` fallback. See ``_cli.py``.
        """
        if self._sandbox is not None:
            # The CLI is the sandbox image's; the sandbox checks its version when it starts.
            return ResolvedCLI(path=self._sandbox.cli_path, version=None)
        if native not in self._resolved_cli:
            # The version probe starts the CLI, which on a loaded host can outlast a short
            # heartbeat timeout before this attempt has sent its first heartbeat.
            async with _heartbeat(None):
                try:
                    self._resolved_cli[native] = await asyncio.to_thread(
                        resolve_cli, self._cli_path, native=native
                    )
                except CLIResolutionError as exc:
                    _reject(str(exc))
        return self._resolved_cli[native]

    def _launcher(self, cli: str) -> str | None:
        """The ``cli_path`` that runs exactly ``cli`` through ``_launcher.py``: a wrapper script,
        written once per worker. None off POSIX, where the SDK runs ``cli`` itself, in the
        worker's own process group. None when a sandbox runtime owns the CLI.
        """
        if self._sandbox is not None:
            return None
        _require_posix()
        if cli not in self._launchers:
            wrapper = Path(tempfile.mkdtemp(prefix="claude-cli-")) / "claude"
            command = shlex.join([sys.executable, "-I", "-S", _launcher.__file__, cli])
            wrapper.write_text(f'#!/bin/sh\nexec {command} "$@"\n')
            wrapper.chmod(0o700)
            self._launchers[cli] = str(wrapper)
        return self._launchers[cli]

    def _options(
        self,
        input: ClaudeTurnInput,
        bridge: ToolBridgeClient | None,
        hooks: NativeToolHooks | None,
        pgid_file: Path | None = None,
        cli_path: str | None = None,
        session_store: SessionStore | None = None,
    ) -> ClaudeAgentOptions:
        sdk = load_sdk()
        opts = input.options
        output_format = (
            {"type": "json_schema", "schema": _cli_schema(opts.output_schema)}
            if opts.output_schema is not None
            else None
        )
        # Harness tools are the only MCP tools, and only they are pre-approved. Without native
        # tools `dontAsk` denies everything else and `tools=[]` keeps every built-in off; with
        # them the hooks gate each built-in call (see `_native.py`).
        mcp_servers = (
            {BRIDGE_MCP_SERVER: mcp_server(input.tools, bridge)} if bridge else {}
        )
        native: dict[str, Any] = (
            {"tools": cli_tools(input), **hooks.options()}
            if hooks is not None
            else {"tools": [], "permission_mode": "dontAsk"}
        )
        # A sandboxed CLI does not inherit the worker's environment, so the flag has to be
        # set on the options even when this process already has it.
        inherited = NONESSENTIAL_TRAFFIC_ENV in os.environ and self._sandbox is None
        env = {} if inherited else {NONESSENTIAL_TRAFFIC_ENV: "1"}
        if self._sandbox is None:
            env.update({k: "" for k in os.environ if k.startswith(_RESUME_ENV_PREFIX)})
        env.update(_CLI_ENV)
        if pgid_file is not None:
            env[_launcher.PGID_FILE_ENV] = str(pgid_file)
        env.update(cli_trace_env())
        if hooks is not None:
            # The SDK hands the CLI the worker's whole environment, and native Bash output is
            # published to the workflow history and mirrored to the SessionStore.
            # A sandboxed CLI gets its credentials from the runtime, which this worker cannot
            # list, so common credential names and the denylist's literal names are denied.
            # A remote runtime omits the subprocess scrub (it lifts Bash confinement on
            # Claude Code 2.1.294 under Linux).
            secret_names = (*REMOTE_CREDENTIAL_ENV, *self._native_env_denylist)
            visible = (
                {**os.environ, **env}
                if self._sandbox is None
                else {**env, **{n: "" for n in secret_names if ENV_NAME.match(n)}}
            )
            native["sandbox"], scrub = sandbox_settings(
                visible, self._native_env_denylist, scrub=self._sandbox is None
            )
            env.update(scrub)
        return sdk.ClaudeAgentOptions(
            **native,
            allowed_tools=[mcp_tool_name(t.name) for t in input.tools]
            if bridge
            else [],
            mcp_servers=mcp_servers,
            strict_mcp_config=True,
            setting_sources=[],
            skills=[],
            plugins=[],
            include_partial_messages=True,
            # Otherwise an `@/path` in a prompt reads a worker file into the turn and a leading
            # `/` runs a CLI slash command instead of asking Claude.
            verbatim_prompts=True,
            max_buffer_size=_MAX_MESSAGE_BYTES,
            model=opts.model,
            system_prompt=opts.system_prompt,
            max_turns=(
                NATIVE_DEFAULT_MAX_TURNS
                if hooks is not None and opts.max_turns is None
                else opts.max_turns
            ),
            max_budget_usd=opts.max_budget_usd,
            thinking=opts.thinking,
            effort=opts.effort,
            output_format=output_format,
            session_id=None if input.resume else input.session_id,
            resume=input.session_id if input.resume else None,
            cwd=self._workspace(input.session_id),
            session_store=session_store or self._session_store,
            # Batched flushes on `result`, so a crashed attempt leaves nothing of its turn in
            # the store, not even the prompt; eager mirrors each model round as it completes.
            session_store_flush="eager"
            if self._session_store is not None
            else "batched",
            cli_path=cli_path,
            env=env,
        )

    async def _interrupt_and_drain(
        self, client: ClaudeSDKClient, progress: _Progress
    ) -> None:
        """Stop the run and read its final result so the transcript records the interruption,
        then report the spend it showed: a cancelled Activity returns no result to carry it.

        Best effort: cancellation can land before connect finished (interrupt then raises), and a
        wedged CLI must not hold the Activity open, so every step is bounded and failures are only
        logged — the bounded disconnect that follows still tears the process down.
        """
        try:
            async with asyncio.timeout(self._interrupt_timeout):
                await client.interrupt()
            async with asyncio.timeout(self._drain_timeout):
                async for message in client.receive_response():
                    if (cost := _reported_cost(message)) is not None:
                        progress.cost_usd = cost
        except Exception as exc:  # noqa: BLE001 - cleanup must reach disconnect
            logger.warning("Claude interrupt/drain after cancellation failed: %r", exc)
        if progress.cost_usd:
            await progress.report(spent=True)

    async def _disconnect(self, client: ClaudeSDKClient, group: _CliGroup) -> None:
        """Disconnect within the bound, else kill the CLI by this attempt's own process group
        (none when the CLI runs without the launcher).
        """
        task = asyncio.ensure_future(client.disconnect())
        try:
            done, _ = await asyncio.wait({task}, timeout=self._disconnect_timeout)
            if not done:
                logger.warning(
                    "Claude disconnect exceeded %.1fs; killing the CLI process group",
                    self._disconnect_timeout,
                )
                killed = _kill(group.pgid, _cli_process(client))
                if group.remote is not None:
                    group.remote()
                    killed = True
                await asyncio.wait({task}, timeout=_KILL_GRACE_SECONDS)
                if not killed:
                    _untracked(
                        "its disconnect overran and there is no handle to kill it by"
                    )
            # The CLI has exited; what it started (a Bash command, a hook) may not have.
            await _end_group(group.pgid)
        except asyncio.CancelledError:
            _kill(group.pgid, _cli_process(client))
            if group.remote is not None:
                group.remote()
            raise
        finally:
            if group.path is not None:
                _kill_detached(group.path)
        if (
            task.done()
            and not task.cancelled()
            and (exc := task.exception()) is not None
        ):
            if isinstance(exc, SandboxLost):
                raise exc
            logger.warning("Claude disconnect failed: %r", exc)


@dataclass(frozen=True)
class _Plan:
    """What :meth:`ClaudeTurnActivity._plan` decided: resume or start, the prompt and the uuid it
    is sent with, the run's harness calls already in the store (see :func:`_harness_calls`),
    those with no result at all (see :func:`_dangling_calls`), and the ids of every harness call
    the whole stored transcript answers (none when no stored transcript was read).
    """

    resume: bool
    prompt: str
    prompt_uuid: str | None = None
    prior: list[tuple[str, str, bool]] = field(default_factory=list)
    answered: frozenset[str] | None = None
    dangling: list[tuple[dict[str, Any], dict[str, Any]]] = field(default_factory=list)


async def _prompt(text: str, prompt_uuid: str | None) -> AsyncIterator[dict[str, Any]]:
    """The prompt as one stream-json user message; the CLI keeps its ``uuid`` on the transcript
    entry (and in the store copy), which is how a retry finds it.
    """
    message: dict[str, Any] = {
        "type": "user",
        "message": {"role": "user", "content": text},
        "parent_tool_use_id": None,
    }
    if prompt_uuid is not None:
        message["uuid"] = prompt_uuid
    yield message


_SCHEMA_MAPS = frozenset({"properties", "patternProperties", "$defs", "definitions"})
_SCHEMA_VALUES = frozenset({"const", "enum", "default", "examples"})


def _cli_schema(node: Any) -> Any:
    """``node`` as a schema both Claude Code 2.1.294 (strict draft-07 for ``--json-schema``) and
    the Messages API (draft 2020-12 for the tool it becomes) accept.

    The CLI rejects Pydantic's ``discriminator`` (an OpenAPI hint, redundant with the ``oneOf``
    it annotates) and ``prefixItems`` (a fixed-length tuple), and the API rejects draft-07's
    array form of ``items``. A tuple's items become ``anyOf`` its positions, its length kept by
    ``minItems``/``maxItems``. The workflow still validates the reply against the original.
    """
    if isinstance(node, list):
        return [_cli_schema(n) for n in cast(list[Any], node)]
    if not isinstance(node, dict):
        return node
    out: dict[str, Any] = {}
    for key, value in cast(dict[str, Any], node).items():
        if key == "discriminator":
            continue
        if key in _SCHEMA_MAPS and isinstance(value, dict):
            out[key] = {
                k: _cli_schema(v) for k, v in cast(dict[str, Any], value).items()
            }
        elif key in _SCHEMA_VALUES:
            out[key] = value
        elif key == "prefixItems":
            rest = node.get("items")
            positions = [
                *cast(list[Any], value),
                *([rest] if isinstance(rest, dict) else []),
            ]
            out["items"] = {"anyOf": _cli_schema(positions)}
        elif key != "items" or "prefixItems" not in node:
            out[key] = _cli_schema(value)
    return out


def _resumes_approved_call(input: ClaudeTurnInput) -> bool:
    """Whether this run exists only to run a call the workflow approved.

    The CLI then starts a turn of its own ("Continue from where you left off"), and a prompt
    sent as well is queued behind it: 2.1.294 answers it right after the next deferral, so that
    deferral is ignored and the model goes on (anthropics/claude-code#97196). A retry of a lost
    attempt, or a call the ledger already holds (the CLI has nothing to resume), still needs its
    prompt.
    """
    decision = input.native_decision
    return (
        decision is not None
        and decision.approved
        and not input.resumes_lost_attempt
        and all(c.tool_use_id != decision.call.tool_use_id for c in input.native_calls)
    )


def _local_transcript(cwd: Path, session_id: str) -> Path:
    """Where the CLI on this worker writes the session's transcript."""
    config = os.environ.get("CLAUDE_CONFIG_DIR") or str(Path.home() / ".claude")
    key = load_sdk().project_key_for_directory(cwd)
    return Path(config) / "projects" / key / f"{session_id}.jsonl"


# The text Claude Code 2.1.294 gives the error result it patches onto a tool call left without one
# when it resumes a session. Such a call's outcome is unknown, so it does not count as answered: a
# re-issued copy must reach the call it repeats. A CLI that reworded it would make re-issued calls
# run again (at-least-once, as before the record), not lose any.
_INTERRUPTED_RESULT = "[Tool call interrupted"


def _reported_cost(message: Any) -> float | None:
    """Spend a CLI message reports, if it reports any: ``total_cost_usd``, or the sum of
    ``model_usage[*].costUSD`` when that is missing or zero.

    Claude Code 2.1.294 reports spend only on the ``ResultMessage`` (no stream, system or
    assistant message carries a dollar figure), so an attempt lost before its result reports
    none: its heartbeats carry its rounds, not its spend.
    """
    cost = getattr(message, "total_cost_usd", None)
    if not cost:
        usage = getattr(message, "model_usage", None)
        if isinstance(usage, dict):
            cost = (
                sum(
                    float(m.get("costUSD") or 0)
                    for m in usage.values()
                    if isinstance(m, dict)
                )
                or cost
            )
    return float(cost) if isinstance(cost, int | float) else None


def _harness_calls(entries: list[dict[str, Any]]) -> list[tuple[str, str, bool]]:
    """Harness tool calls in ``entries`` in order, by id and :func:`tool_call_key`, each with
    whether it has a real result.
    """
    prefix = mcp_tool_name("")
    calls = {
        block["id"]: tool_call_key(
            block["name"].removeprefix(prefix), block.get("input") or {}
        )
        for entry in entries
        if entry.get("type") == "assistant"
        for block in content_blocks(entry)
        if block.get("type") == "tool_use"
        and str(block.get("name", "")).startswith(prefix)
    }
    answered = {
        block.get("tool_use_id")
        for entry in entries
        if entry.get("type") == "user"
        for block in content_blocks(entry)
        if block.get("type") == "tool_result"
        and not _result_text(block.get("content")).startswith(_INTERRUPTED_RESULT)
    }
    return [(call_id, key, call_id in answered) for call_id, key in calls.items()]


def _dangling_calls(
    entries: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Harness ``tool_use`` blocks in ``entries`` that no ``tool_result`` answers (an
    interrupted patch counts as one), each with the assistant entry that holds it.
    """
    prefix = mcp_tool_name("")
    answered = {
        block.get("tool_use_id")
        for entry in entries
        if entry.get("type") == "user"
        for block in content_blocks(entry)
        if block.get("type") == "tool_result"
    }
    return [
        (entry, block)
        for entry in entries
        if entry.get("type") == "assistant"
        for block in content_blocks(entry)
        if block.get("type") == "tool_use"
        and str(block.get("name", "")).startswith(prefix)
        and block.get("id") not in answered
    ]


def _result_text(content: Any) -> str:
    if isinstance(content, list):
        return "".join(str(c.get("text", "")) for c in content if isinstance(c, dict))
    return str(content or "")


_SANDBOX_VIOLATIONS = re.compile(
    r"<sandbox_violations>(.*?)</sandbox_violations>", re.DOTALL
)


def sandbox_violations(text: str) -> int:
    """Entries in the ``<sandbox_violations>`` blocks Claude Code appends to a sandboxed Bash
    result, one per non-blank line. An unclosed block counts nothing.

    A signal, not enforcement: the CLI does not report every denial (Linux filesystem denials
    are not reported, and the macOS log monitor drops entries).
    """
    return sum(
        1
        for block in _SANDBOX_VIOLATIONS.findall(text)
        for line in block.splitlines()
        if line.strip()
    )


def _rerequests(state_dir: Path | None) -> int:
    """Native calls the hook counted as re-requests in this run."""
    return (
        sum(e["rerequest"] for e in ledger(state_dir)) if state_dir is not None else 0
    )


class _StreamSdk(Protocol):
    """Published Claude SDK message types ``_consume`` narrows the CLI stream with.

    ``load_sdk`` returns a plain module, so these attributes would otherwise stay untyped and
    every read of a message field would need its own ignore.
    """

    AssistantMessage: type[AssistantMessage]
    MirrorErrorMessage: type[MirrorErrorMessage]
    RateLimitEvent: type[RateLimitEvent]
    ResultMessage: type[ResultMessage]
    ResultError: type[ResultError]
    StreamEvent: type[StreamEvent]
    ToolResultBlock: type[ToolResultBlock]
    UserMessage: type[UserMessage]


def _stream_sdk() -> _StreamSdk:
    """The loaded Claude SDK, typed as the message classes ``_consume`` reads."""
    # A module does not overlap the protocol, so the cast goes through object.
    return cast(_StreamSdk, cast(object, load_sdk()))


_DEFAULT_STREAM_BATCH = timedelta(milliseconds=100)

PartialSink = Callable[[list[ClaudePartialEvent], Any], Awaitable[None] | None]
"""Receives a batch of one turn attempt's partials and the turn's ``streaming_context``."""


class _PartialPublisher:
    """Numbers one attempt's partials and hands each to the stream topic.

    Sequence starts at 0 for this attempt only. A later attempt is a new publisher. Partials are
    not written onto heartbeats: those carry the ledger the workflow resumes from.
    """

    def __init__(
        self, attempt_id: str, publish: Callable[[ClaudePartialEvent], None]
    ) -> None:
        self._attempt_id = attempt_id
        self._publish = publish
        self._sequence = 0

    def on_message(self, message: Any) -> None:
        """Publish one CLI ``StreamEvent``. Anything without an event dict is ignored."""
        event = getattr(message, "event", None)
        if not isinstance(event, dict):
            return
        self._publish(
            partial_event(
                event,
                attempt_id=self._attempt_id,
                sequence=self._sequence,
                parent_tool_use_id=getattr(message, "parent_tool_use_id", None),
            )
        )
        self._sequence += 1


class _SinkBatcher:
    """Hands one attempt's partials to the worker's ``partial_sink`` in batches.

    The sink runs on its own task, so a slow sink delays its batches, not the CLI stream.
    Partials are advisory, so a failing sink loses partials, never the turn.
    """

    def __init__(
        self, sink: PartialSink, context: Any, interval: timedelta, attempt_id: str
    ) -> None:
        self._sink = sink
        self._context = context
        self._interval = interval.total_seconds()
        self._attempt_id = attempt_id
        self._pending: list[ClaudePartialEvent] = []
        self._ready = asyncio.Event()
        self._closed = False
        self._failed = False
        self._task = asyncio.create_task(self._run())

    def add(self, event: ClaudePartialEvent) -> None:
        self._pending.append(event)
        self._ready.set()

    async def aclose(self) -> None:
        """Deliver what is left, then stop."""
        self._closed = True
        self._ready.set()
        await self._task
        await self._flush()

    async def _run(self) -> None:
        while not self._closed:
            await self._ready.wait()
            if not self._closed:
                await asyncio.sleep(self._interval)
            self._ready.clear()
            await self._flush()

    async def _flush(self) -> None:
        if not self._pending:
            return
        batch, self._pending = self._pending, []
        try:
            sent = self._sink(batch, self._context)
            if inspect.isawaitable(sent):
                await sent
        except Exception:
            if not self._failed:
                logger.warning(
                    "Claude partial_sink failed on attempt %s; its later failures in this "
                    "attempt are not logged",
                    self._attempt_id,
                    exc_info=True,
                )
            self._failed = True


@asynccontextmanager
async def _partial_stream(
    turn: ClaudeTurnInput | None, sink: PartialSink | None = None
) -> AsyncIterator[_PartialPublisher | None]:
    """The attempt's publisher, or none when the turn has no streaming topic and no sink.

    No topic does not open a ``WorkflowStreamClient``, so the turn schedules no stream signals.
    """
    if turn is None or (not turn.streaming_topic and sink is None):
        yield None
        return
    interval = _DEFAULT_STREAM_BATCH
    if turn.streaming_batch_interval_ms is not None:
        interval = timedelta(milliseconds=turn.streaming_batch_interval_ms)
    targets: list[Callable[[ClaudePartialEvent], None]] = []
    async with AsyncExitStack() as stack:
        if turn.streaming_topic:
            client = WorkflowStreamClient.from_within_activity(batch_interval=interval)
            targets.append(
                client.topic(turn.streaming_topic, type=ClaudePartialEvent).publish
            )
            await stack.enter_async_context(client)
        if sink is not None:
            batcher = _SinkBatcher(
                sink, turn.streaming_context, interval, turn.attempt_id
            )
            stack.push_async_callback(batcher.aclose)
            targets.append(batcher.add)

        def publish(event: ClaudePartialEvent) -> None:
            for target in targets:
                target(event)

        yield _PartialPublisher(turn.attempt_id, publish)


async def _consume(
    client: ClaudeSDKClient,
    mirror_errors: list[str],
    progress: _Progress | None = None,
    trace: TurnTrace | None = None,
    sink: PartialSink | None = None,
) -> ResultMessage:
    """Read the CLI's messages and return the run's ``ResultMessage``.

    A ``ResultError`` after an ``is_error`` result is the same failure (the CLI then exits
    non-zero), so only the result becomes the Activity failure.
    """
    sdk = _stream_sdk()
    result: ResultMessage | None = None
    rounds: set[str | None] = set()
    resets_at: int | None = None
    error_kind: str | None = None
    turn = progress._input if progress is not None else None
    async with _partial_stream(turn, sink) as partials:
        try:
            async for message in client.receive_response():
                if trace is not None:
                    trace.on_message(message)
                if partials is not None and isinstance(message, sdk.StreamEvent):
                    partials.on_message(message)
                if progress is not None:
                    cost = _reported_cost(message)
                    if cost is not None:
                        progress.cost_usd = cost
                if isinstance(message, sdk.AssistantMessage) and message.error:
                    error_kind = message.error
                    if progress is not None:
                        progress.error_kind = error_kind
                if (
                    isinstance(message, sdk.AssistantMessage)
                    and message.parent_tool_use_id is None
                ):
                    rounds.add(message.message_id)
                    if progress is not None:
                        progress.num_turns = len(rounds)
                if isinstance(message, sdk.MirrorErrorMessage):
                    mirror_errors.append(message.error)
                    if progress is not None:
                        await progress.store_dropped_a_batch()
                elif isinstance(message, sdk.RateLimitEvent):
                    info = message.rate_limit_info
                    resets_at = info.resets_at if info.status == "rejected" else None
                elif (
                    isinstance(message, sdk.UserMessage)
                    and isinstance(message.content, list)
                    and progress is not None
                ):
                    progress.sandbox_violations += sum(
                        sandbox_violations(_result_text(block.content))
                        for block in message.content
                        if isinstance(block, sdk.ToolResultBlock)
                    )
                elif isinstance(message, sdk.ResultMessage):
                    result = message
        except sdk.ResultError as exc:
            if result is None:
                raise _result_error(exc, mirror_errors, resets_at, error_kind) from exc
    if result is None:
        raise ApplicationError(
            "Claude ended the run without a result.",
            {"mirror_errors": mirror_errors, "error_kind": error_kind},
            type=CLAUDE_RESULT_ERROR_TYPE,
            non_retryable=True,
        )
    # A resumed CLI's num_turns grows with the transcript (35, 42, 49 ... for runs with one model
    # call), and the message-wide max_turns sums per-run counts: report this run's rounds.
    result = replace(result, num_turns=len(rounds))
    if result.is_error:
        raise _result_error(result, mirror_errors, resets_at, error_kind)
    return result


@asynccontextmanager
async def _bridge(
    input: ClaudeTurnInput, progress: _Progress
) -> AsyncIterator[ToolBridgeClient | None]:
    """The turn's tool bridge back to the workflow that scheduled it, or none without tools."""
    if not (input.tools and input.bridge_token):
        yield None
        return
    async with ToolBridgeClient(
        activity.client().get_workflow_handle(
            activity.info().workflow_id  # type: ignore[arg-type]
        ),
        input.bridge_token,
    ) as bridge:
        body = asyncio.current_task()
        assert body is not None

        async def supervise() -> None:
            await bridge.wait_failure()
            body.cancel()

        supervisor = asyncio.create_task(supervise())
        try:
            yield bridge
        except asyncio.CancelledError as exc:
            if bridge.failure is None:
                raise
            body.uncancel()
            raise ApplicationError(
                f"Claude tool outcome unknown: {bridge.failure}. The CLI was stopped.",
                {"session_id": input.session_id, **progress.details(lost=False)},
                type=CLAUDE_RESULT_ERROR_TYPE,
                non_retryable=True,
            ) from exc
        finally:
            supervisor.cancel()
            await asyncio.gather(supervisor, return_exceptions=True)


_TOOL_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_TOOL_USE_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


def _pending_native_continuation(input: ClaudeTurnInput) -> bool:
    """Whether ``input`` is the decision run for a deferred native call the ledger does not record.

    Such a run may resume before the deferral's transcript reached the SessionStore.
    """
    if input.resumes_lost_attempt or input.resumes_api_error:
        return False
    decision = input.native_decision
    if decision is None:
        return False
    call = decision.call
    if any(entry.tool_use_id == call.tool_use_id for entry in input.native_calls):
        return False
    if decision.approved:
        return call.name in input.native_tools
    return any(
        denied.tool_use_id == call.tool_use_id and denied.name == call.name
        for denied in input.native_denied
    )


def _validate_input(input: ClaudeTurnInput) -> None:
    """Refuse a malformed session id or tool declaration before any CLI process starts."""
    problem = None
    names = [t.name for t in input.tools]
    if input.session_id is not None and not _is_uuid(input.session_id):
        problem = f"session_id {input.session_id!r} is not a canonical UUID"
    elif input.resume and input.session_id is None:
        problem = "resume without a session_id"
    elif input.tools and not input.bridge_token:
        problem = "tools need a bridge_token"
    elif input.bridge_token and not input.tools:
        problem = "a bridge_token without tools"
    elif len(set(names)) != len(names):
        problem = f"duplicate tool names in {names}"
    elif bad := [n for n in names if not _TOOL_NAME.match(n)]:
        problem = f"invalid MCP tool names {bad}"
    elif bad := sorted(set(input.native_auto_allow or ()) - READ_ONLY_NATIVE_TOOLS):
        problem = f"native_auto_allow names tools that are not read-only: {bad}"
    elif (decision := input.native_decision) and not input.resume:
        problem = "a native_decision outside a resumed turn"
    elif decision and decision.approved and not input.native_tools:
        # Only a denial may settle a call left deferred by an earlier message.
        problem = "an approved native_decision outside a native-tool turn"
    elif decision and not _TOOL_USE_ID.match(decision.call.tool_use_id):
        # It names the decision file the hook reads.
        problem = f"invalid tool_use_id {decision.call.tool_use_id!r}"
    if problem:
        _reject(problem)


def _reject(problem: str) -> NoReturn:
    raise ApplicationError(
        f"Claude turn rejected: {problem}.",
        type=CLAUDE_TURN_REJECTED_ERROR_TYPE,
        non_retryable=True,
    )


def _unverified_cli_resume(
    cli_path: str | None, cli: ResolvedCLI, resume: bool, sdk_version: str
) -> str | None:
    """Why a custom ``cli_path`` may not resume, or None when the turn may.

    An unknown or unparsed ``--version`` report fails closed.
    """
    if cli_path is None or not resume:
        return None
    if (sdk_version, cli.version) in NATIVE_TOOLS_VERIFIED:
        return None
    reported = cli.version or "(version unknown)"
    return (
        f"resume on cli_path {cli.path} is not verified for claude-agent-sdk {sdk_version} "
        f"with Claude Code CLI {reported}: resume and tool-call dedup follow the CLI protocol "
        f"measured only on (SDK, CLI) {sorted(NATIVE_TOOLS_VERIFIED)}"
    )


def _is_uuid(value: str) -> bool:
    try:
        return str(uuid.UUID(value)) == value
    except ValueError:
        return False


_STDERR_TAIL_LINES = 5
_STDERR_TAIL_CHARS = 600
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _stderr_tail(lines: Sequence[str]) -> str:
    """The local CLI's last stderr lines without terminal escapes.

    Capped because they are recorded in workflow history with the failure.
    """
    return _ANSI_ESCAPE.sub("", "\n".join(lines)).strip()[-_STDERR_TAIL_CHARS:]


def _result_error(
    source: Any,
    mirror_errors: list[str],
    resets_at: int | None = None,
    error_kind: str | None = None,
) -> ApplicationError:
    """One non-retryable failure from an error ``ResultMessage`` or a ``ResultError``.

    ``rate_limit_resets_at`` (Unix seconds) is the run's only retry-after hint.
    """
    errors = list(getattr(source, "errors", None) or [])
    detail = "; ".join(errors) or getattr(source, "result", None) or "no detail"
    reason = getattr(source, "terminal_reason", None) or getattr(
        source, "subtype", None
    )
    if error_kind:
        reason = f"{reason}, {error_kind}"
    return ApplicationError(
        f"Claude run failed ({reason}): {detail}",
        {
            "subtype": getattr(source, "subtype", None),
            "terminal_reason": getattr(source, "terminal_reason", None),
            "api_error_status": getattr(source, "api_error_status", None),
            "session_id": getattr(source, "session_id", None),
            "errors": errors,
            "mirror_errors": mirror_errors,
            "num_turns": getattr(source, "num_turns", None),
            "total_cost_usd": _reported_cost(source),
            "rate_limit_resets_at": resets_at,
            "error_kind": error_kind,
        },
        type=CLAUDE_RESULT_ERROR_TYPE,
        non_retryable=True,
    )


def _to_turn_result(
    result: ResultMessage,
    mirror_errors: list[str],
    native_rerequests: int,
    native_calls: list[dict[str, Any]] | None = None,
    workspace_revision: int | None = None,
    sandbox_snapshot: str | None = None,
    sandbox_violations: int = 0,
    cumulative_cost: bool = False,
) -> ClaudeTurnResult:
    deferred = result.deferred_tool_use
    return ClaudeTurnResult(
        native_calls=native_calls or [],  # type: ignore[arg-type]
        sandbox_violations=sandbox_violations or None,
        workspace_revision=workspace_revision,
        sandbox_snapshot=sandbox_snapshot,
        deferred=ClaudeDeferredTool(
            tool_use_id=deferred.id, name=deferred.name, input=deferred.input or {}
        )
        if deferred is not None
        else None,
        native_rerequests=native_rerequests,
        mirror_errors=mirror_errors,
        text=result.result,
        structured_output=result.structured_output,
        session_id=result.session_id,
        num_turns=result.num_turns,
        stop_reason=result.stop_reason,
        terminal_reason=result.terminal_reason,
        total_cost_usd=_reported_cost(result),
        cumulative_cost=cumulative_cost,
        duration_ms=result.duration_ms,
        duration_api_ms=result.duration_api_ms,
        usage=anthropic_usage(result.usage),
        model_usage=result.model_usage or None,  # type: ignore[arg-type]
        permission_denials=result.permission_denials or None,
    )


@asynccontextmanager
async def _leased(
    store: LeaseStore,
    input: ClaudeTurnInput,
    group: _CliGroup,
    progress: _Progress,
) -> AsyncIterator[Lease | None]:
    """Hold the session's lease for the turn, rejecting a second CLI on one transcript.

    The TTL is the heartbeat timeout. Every confirmed renewal is heartbeated, so the server's
    last record of a lost attempt names the beat its lease stopped at as nearly as it can; a
    retry finding the lease still there claims it at once. A lease lost mid-run kills the CLI
    and fails with an unknown outcome (non-retryable once the CLI may have run), carrying the
    progress details.
    """
    session_id = input.session_id
    details = progress.details
    ttl = activity.info().heartbeat_timeout or 2 * _FALLBACK_HEARTBEAT_INTERVAL

    def kill_cli() -> None:
        if (pgid := group.pgid) is not None:
            _kill(pgid, None)
        if group.remote is not None:
            group.remote()

    try:
        async with session_lease(
            store,
            session_id,
            ttl,
            on_lost=kill_cli,
            silent=input.lost_lease if input.resumes_lost_attempt else None,
            on_beat=lambda: activity.heartbeat(details()),
        ) as lease:
            progress.lease = lease
            yield lease
    except SessionLeased as exc:
        raise ApplicationError(
            f"Claude turn rejected: {exc}.", type=CLAUDE_SESSION_LEASED_ERROR_TYPE
        ) from exc
    except SessionLeaseLost as exc:
        raise ApplicationError(
            f"Claude turn outcome unknown: {exc}. Its CLI was killed; any tool it ran may "
            "have taken effect.",
            {"session_id": session_id, **details()},
            type=CLAUDE_SESSION_LEASE_LOST_ERROR_TYPE,
            non_retryable=group.started,
        ) from exc


class _Progress:
    """A running attempt's ledger, revision, rounds, spend and mirror errors, for heartbeats.

    Heartbeats are throttled, so with ``report_progress`` each change is also signaled
    (:data:`TURN_PROGRESS_SIGNAL`) before the CLI carries on.
    """

    def __init__(
        self,
        input: ClaudeTurnInput,
        state_dir: Path | None,
        store: WorkspaceStore | None,
        cwd: Path | None,
        resumable: bool,
        mirror_errors: list[str],
        *,
        shared: bool = True,
        sandboxed: bool = False,
    ) -> None:
        self.num_turns = 0
        self.cost_usd = 0.0
        self.cumulative_cost = False
        self.error_kind: str | None = None
        self.sandbox_violations = 0
        self.workspace_revision = input.workspace_revision
        self.sandbox_snapshot = input.sandbox_snapshot
        self.mirror_errors = mirror_errors
        self.lease: Lease | None = None
        self._input = input
        self._state_dir = state_dir
        self._store = store
        self._cwd = cwd
        self._resumable = resumable
        self._shared = shared
        self._reported: tuple[Any, ...] | None = None
        self._lock = asyncio.Lock()
        self._sandboxed = sandboxed
        self._unsnapshotted: dict[str, bool] = {}

    def native_calls(self) -> list[dict[str, Any]]:
        if self._state_dir is None:
            return []
        try:
            return native_calls(self._state_dir)
        except (OSError, ValueError):
            return []  # a corrupt ledger fails the hook closed; nothing to report

    def details(self, *, lost: bool = True) -> dict[str, Any]:
        """Heartbeat details; ``lost=False`` for a failure this attempt reports itself, whose CLI
        is gone, so its retry needs no shared lease store.
        """
        return {
            RESUMABLE_DETAIL: self._resumable
            and (self._shared or not lost)
            and not self.mirror_errors
            and self.error_kind not in NEVER_RESUMED_ERROR_KINDS,
            "num_turns": self.num_turns,
            "cost_usd": self.cost_usd,
            "cumulative_cost": self.cumulative_cost,
            "error_kind": self.error_kind,
            "workspace_revision": self.workspace_revision,
            "sandbox_snapshot": self.sandbox_snapshot,
            "native_calls": self.native_calls(),
            "mirror_errors": list(self.mirror_errors),
            "lease": [self.lease.generation, self.lease.beat] if self.lease else None,
        }

    async def store_dropped_a_batch(self) -> None:
        """A ``MirrorErrorMessage`` arrived: heartbeat and report it now, not at the next beat."""
        activity.heartbeat(self.details())
        await self.report()

    async def after_tool(self, name: str, call_id: str | None, failed: bool) -> bool:
        """Snapshot what a non-repeatable call may have written. False: the call stays started.

        A sandbox is snapshotted only after the run, so its call is settled then
        (:meth:`settle_snapshotted`): a sandbox lost before that has no record of the effect.
        """
        if native_tool_effect(name) in REPEATABLE_EFFECTS:
            return True
        if self._sandboxed:
            if call_id is not None:
                self._unsnapshotted[call_id] = failed
            return False
        if self._store is not None:
            try:
                revision = await _persist(
                    self._store, self._input.session_id, self._cwd
                )
            except Exception as exc:  # noqa: BLE001 - the end-of-run persist fails the turn
                logger.warning(
                    "Could not persist the workspace after %s: %r", name, exc
                )
                return False
            if revision is None:
                return False
            self.workspace_revision = revision
        return True

    def settle_snapshotted(self) -> None:
        """Mark the calls a committed sandbox snapshot now holds finished (or failed)."""
        if self._state_dir is not None:
            for call_id, failed in self._unsnapshotted.items():
                try:
                    record_native_finished(self._state_dir, call_id, failed=failed)
                except Exception as exc:  # noqa: BLE001 - left "started", a retry asks instead of re-running
                    logger.warning("Could not record %s as finished: %r", call_id, exc)
        self._unsnapshotted.clear()

    @asynccontextmanager
    async def watching(self) -> AsyncIterator[None]:
        """Report ledger changes the PostToolUse callback may miss (a call allowed before the SDK
        registered its callbacks) while the CLI runs.
        """
        if self._state_dir is None or not self._input.report_progress:
            yield
            return
        self._reported = (
            self.native_calls(),
            self.workspace_revision,
            list(self.mirror_errors),
        )

        async def watch() -> None:
            while True:
                await asyncio.sleep(_LEDGER_WATCH_SECONDS)
                await self.report()

        task = asyncio.create_task(watch())
        try:
            yield
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def report(self, *, spent: bool = False) -> None:
        """Signal what changed; ``spent`` (once, from a cancelled attempt) adds its cost."""
        if not self._input.report_progress or self._input.session_id is None:
            return
        async with self._lock:
            calls = self.native_calls()
            mirror_errors = list(self.mirror_errors)
            if (
                not spent
                and (calls, self.workspace_revision, mirror_errors) == self._reported
            ):
                return
            progress = ClaudeTurnProgress(
                session_id=self._input.session_id,
                attempt_id=self._input.attempt_id,
                native_calls=calls,  # type: ignore[arg-type]
                workspace_revision=self.workspace_revision,
                mirror_errors=mirror_errors,
                cost_usd=self.cost_usd if spent else None,
                cumulative_cost=spent and self.cumulative_cost,
            )
            try:
                handle = activity.client().get_workflow_handle(
                    activity.info().workflow_id  # type: ignore[arg-type]
                )
                await handle.signal(TURN_PROGRESS_SIGNAL, progress)
            except Exception as exc:  # noqa: BLE001 - the heartbeat and the result still carry it
                logger.warning("Could not report Claude turn progress: %r", exc)
                return
            self._reported = (calls, self.workspace_revision, mirror_errors)


async def _persist(
    store: WorkspaceStore | None, session_id: str | None, cwd: Path | None
) -> int | None:
    if store is None or session_id is None or cwd is None:
        return None
    return (await store.persist(session_id, cwd)).revision


@asynccontextmanager
async def _heartbeat(
    details: Callable[[], dict[str, Any]] | None,
) -> AsyncIterator[None]:
    """Heartbeat every ``CANCEL_LATENCY`` (at most half the heartbeat timeout) for the whole
    block, independent of token flow, so a long silent model round never looks like a dead
    worker and a cancellation is delivered fast. ``details()`` rides every heartbeat: a workflow
    reads it from the heartbeat timeout of an attempt whose worker died.

    With no ``details`` (setup before the attempt has any) the beats are bare and the first
    waits one interval, so a quick block sends none: the attempt's first heartbeat, the only one
    the server records at once, still names its lease, and a bare one that times out reads as
    an attempt that never started.
    """
    timeout = activity.info().heartbeat_timeout or 2 * _FALLBACK_HEARTBEAT_INTERVAL
    interval = min(timeout.total_seconds() / 2, CANCEL_LATENCY.total_seconds())

    async def beat() -> None:
        if details is None:
            await asyncio.sleep(interval)
        while True:
            activity.heartbeat(*(() if details is None else (details(),)))
            await asyncio.sleep(interval)

    task = asyncio.create_task(beat())
    try:
        yield
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


def _require_posix() -> None:
    """Refuse the turn off POSIX. There is no process-group handle, so the CLI would outlive the worker."""
    if os.name != "posix":
        _reject(
            "Claude turns run only on POSIX, where the CLI is tracked in its own process group "
            "and dies with the worker"
        )


def _require_parent_death_watch() -> None:
    """Refuse every turn when this host cannot reap the CLI after the worker is killed.

    A killed worker orphans the CLI with or without a SessionStore. The same-host
    sweep does not run after failover to another host, and a fresh turn has no
    later attempt to sweep.
    """
    if _launcher.parent_death_watch() is None:
        _reject(
            "this host has no parent-death watch (kqueue or pidfd), so a killed worker "
            "would leave the Claude CLI running"
        )


def _cli_process(client: ClaudeSDKClient) -> Any:
    # The SDK's private transport (0.2.165) has the only SDK-side handle on the CLI process. An
    # extra only: every kill goes by the launcher's recorded group first, and off POSIX (no
    # launcher, no native tools) this is all there is.
    return getattr(getattr(client, "_transport", None), "_process", None)


class _CliGroup:
    """This attempt's CLI process group: read from the attempt's own pid file once the launcher
    wrote it, then kept, so every kill of this attempt's CLI goes by the group it started.
    """

    def __init__(self) -> None:
        self.path: Path | None = None
        self.started = False  # True once the CLI may have run; a lease loss before that is retryable.
        self.remote: Callable[[], None] | None = None
        """Kills a CLI in a sandbox, which has no process group to signal."""
        self._pgid: int | None = None

    @property
    def pgid(self) -> int | None:
        if self._pgid is None:
            self._pgid = _recorded_group(self.path)
        return self._pgid


def _recorded_group(pgid_file: Path | None) -> int | None:
    """The process group the launcher recorded for this turn's CLI (its pid), if it has."""
    try:
        return int(pgid_file.read_text()) if pgid_file is not None else None
    except (FileNotFoundError, ValueError):
        return None


def _kill(pgid: int | None, process: Any) -> bool:
    """SIGKILL the CLI's group and its descendants' groups, and the SDK's process handle when it
    still has one. False, logged, when there is neither to kill by.
    """
    if pgid is not None and sys.platform != "win32":
        for group in _groups_under(pgid) | {pgid}:
            _signal_group(group, signal.SIGKILL)
    if process is not None and getattr(process, "returncode", 0) is None:
        with suppress(ProcessLookupError):
            process.kill()
    elif pgid is None and process is None:
        logger.error(
            "Cannot kill the Claude CLI: no recorded process group and no SDK handle"
        )
        return False
    return True


def _state_puller(
    sandbox: SandboxTurn, remote_state: str, state_dir: Path
) -> Callable[[], Awaitable[None]]:
    """``pull_state`` for one turn, bounded. A failure propagates: the caller must not commit
    a snapshot or a success over a ledger it could not read.
    """

    async def pull() -> None:
        async with asyncio.timeout(_SANDBOX_STATE_TIMEOUT_SECONDS):
            await pull_state(sandbox.sandbox, remote_state, state_dir)

    return pull


def _note_unreadable_native(state_dir: Path | None, input: ClaudeTurnInput) -> None:
    """Record the approved native call as started when its sandbox ledger could not be read.

    That call is the only native effect this attempt was allowed to run. Leaving it unrecorded
    would let a later attempt run it again. A status already settled is kept.
    """
    decision = input.native_decision
    if state_dir is None or decision is None or not decision.approved:
        return
    call = decision.call
    merge_native_calls(
        state_dir,
        [
            {
                "tool_use_id": call.tool_use_id,
                "tool_name": call.name,
                "key": call_key(call.name, call.input),
                "run": None,
                "status": "started",
            }
        ],
    )


def _sandbox_lost(
    exc: SandboxLost,
    input: ClaudeTurnInput,
    num_turns: int,
    resumable: bool,
    *,
    native_calls: list[dict[str, Any]] | None = None,
    ledger_unreadable: bool = False,
    cost_usd: float = 0.0,
    cumulative_cost: bool = False,
) -> ApplicationError:
    """The error for a turn whose sandbox was lost. The workflow resumes it from the last
    committed snapshot only when ``resumable`` (a shared lease store) is set and the native
    effect is safe to retry. The details name no new snapshot, and the spend the CLI had
    reported, which the session pays. An unreadable ledger is reported and does not by itself
    clear that flag.
    """
    details: dict[str, Any] = {
        "session_id": input.session_id,
        "num_turns": num_turns,
        "sandbox_snapshot": None,
        RESUMABLE_DETAIL: resumable,
        "cost_usd": cost_usd,
        "cumulative_cost": cumulative_cost,
    }
    if native_calls is not None:
        details["native_calls"] = native_calls
    if ledger_unreadable:
        details["ledger_unreadable"] = True
    return ApplicationError(
        f"Claude turn lost its sandbox: {exc}. The session resumes from its "
        f"last snapshot ({input.sandbox_snapshot}).",
        details,
        type=CLAUDE_SANDBOX_LOST_ERROR_TYPE,
    )


@asynccontextmanager
async def _sandbox_turn(
    runtime: SandboxRuntime | None,
    input: ClaudeTurnInput,
    lease: Lease | None,
    resumable: bool,
) -> AsyncIterator[SandboxTurn | None]:
    """The turn's sandbox (none without a runtime), terminated when the turn ends."""
    if runtime is None:
        yield None
        return

    async def owns() -> None:
        if lease is not None:
            await lease.check()

    try:
        turn = await SandboxTurn.start(
            runtime,
            session_id=input.session_id,
            snapshot=input.sandbox_snapshot,
            generation=None if lease is None else lease.generation,
            owns=owns if lease is not None else None,
        )
    except SandboxError as exc:
        _reject(str(exc))
    except SandboxLost as exc:  # dropped before the CLI could start: nothing ran
        raise _sandbox_lost(exc, input, 0, resumable) from exc
    try:
        yield turn
    finally:
        await turn.terminate()


def _untracked(problem: str) -> NoReturn:
    logger.error("Claude CLI not killable: %s", problem)
    raise ApplicationError(
        f"The Claude CLI could not be tracked: {problem}. It may still be running.",
        type=CLAUDE_CLI_UNTRACKED_ERROR_TYPE,
        non_retryable=True,
    )


def _groups_under(pid: int) -> set[int]:
    """The process groups of ``pid``'s descendants, never the worker's own. Claude Code starts
    each Bash command in a session (so a group) of its own, which a signal to the CLI's group
    does not reach; they are found while the CLI still runs, by parent pid.
    """
    if sys.platform == "win32":
        return set()
    try:
        table = _process_table()
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        logger.warning("Could not list the Claude CLI's descendants: %r", exc)
        return set()
    children: dict[int, list[tuple[int, int]]] = {}
    for child, parent, pgid in table:
        children.setdefault(parent, []).append((child, pgid))
    groups: set[int] = set()
    stack = [pid]
    while stack:
        for child, pgid in children.get(stack.pop(), []):
            groups.add(pgid)
            stack.append(child)
    return groups - {os.getpgrp()}


def _process_table() -> list[tuple[int, int, int]]:
    """``(pid, ppid, pgid)`` of every process: from /proc on Linux, from ``ps`` elsewhere."""
    if not Path("/proc/self/stat").exists():
        out = subprocess.run(
            ["ps", "-A", "-o", "pid=,ppid=,pgid="],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
        return [
            (int(a), int(b), int(c))
            for a, b, c in (line.split() for line in out.splitlines())
        ]
    table = []
    for stat in Path("/proc").glob("[0-9]*/stat"):
        try:
            # After the parenthesized command name: state, ppid, pgrp, ...
            fields = stat.read_text().rsplit(")", 1)[1].split()
        except OSError:
            continue  # exited meanwhile
        table.append((int(stat.parent.name), int(fields[1]), int(fields[2])))
    return table


async def _end_group(pgid: int | None) -> None:
    """SIGTERM what is left of the CLI's process group, then SIGKILL it after the kill grace.

    Safe after the CLI is reaped: a pid that still names a process group is never reused.
    """
    if sys.platform == "win32" or not _signal_group(pgid, signal.SIGTERM):
        return
    deadline = time.monotonic() + _KILL_GRACE_SECONDS
    while _signal_group(pgid, 0):
        if time.monotonic() >= deadline:
            _signal_group(pgid, signal.SIGKILL)
            return
        await asyncio.sleep(0.05)


def _signal_group(pgid: int | None, sig: int) -> bool:
    """Signal a process group; False when there is none (or no group to signal)."""
    if pgid is None or sys.platform == "win32":
        return False
    try:
        os.killpg(pgid, sig)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def _kill_leftover_groups(pgid_file: Path, lease: Lease | None) -> None:
    """Kill what the session's earlier generations left (see ``_kill_leftover_group``). Never a
    later generation's: an attempt that was taken over before its sweep would kill the CLI of
    the attempt that now holds the session.
    """
    if lease is None:
        return
    prefix = f"{lease.session_id}."
    for path in pgid_file.parent.iterdir():
        generation = path.name.removeprefix(prefix)
        if (
            path.name.startswith(prefix)
            and generation.isdigit()
            and int(generation) < lease.generation
        ):
            _kill_leftover_group(path)


def _kill_leftover_group(pgid_file: Path) -> None:
    """Kill the process group an earlier attempt's CLI on this session recorded, if that CLI is
    still running: a worker that died mid-turn left it, and it must not run beside the next one.

    Only a group whose leader is that CLI is killed: a process leading its own group, with the
    launcher's ``PGID_FILE_ENV`` naming this file in its environment. A process that reused the
    pid is left alone. So is a group whose CLI already exited (it cannot be told apart from an
    unrelated one); the CLI's own exit and the per-turn group kill keep that rare.
    """
    if sys.platform == "win32":
        return
    try:
        pgid = int(pgid_file.read_text())
    except (FileNotFoundError, ValueError):
        return
    marker = f"{_launcher.PGID_FILE_ENV}={pgid_file}"
    try:
        ours = os.getpgid(pgid) == pgid and marker in _command_and_environment(pgid)
    except (OSError, subprocess.SubprocessError):
        ours = False
    if ours:
        logger.warning(
            "Killing the leftover Claude CLI process group %d (%s)", pgid, pgid_file
        )
        for group in _groups_under(pgid) | {pgid}:
            _signal_group(group, signal.SIGKILL)
    _kill_detached(pgid_file)
    pgid_file.unlink(missing_ok=True)


def _kill_detached(pgid_file: Path) -> None:
    """SIGKILL every process that still carries this attempt's ``PGID_FILE_ENV`` in its
    environment. A command that detaches (``nohup x &``, ``setsid -f x``) leaves ``x`` adopted by
    init and outside the CLI's group, so neither the group kill nor the descendant walk finds it;
    it still inherits the CLI's environment (a command that clears it escapes this too).
    """
    if sys.platform == "win32":
        return
    marker = f"{_launcher.PGID_FILE_ENV}={pgid_file}"
    try:
        pids = _launcher.marked_pids(marker)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        logger.warning("Could not look for detached Claude CLI processes: %r", exc)
        return
    for pid in pids:
        logger.warning("Killing pid %d, detached from a finished Claude CLI turn", pid)
        with suppress(ProcessLookupError, PermissionError):
            os.kill(pid, signal.SIGKILL)


def _command_and_environment(pid: int) -> str:
    """A process's arguments and initial environment, as text: from /proc on Linux, from
    ``ps -E`` (the caller's own processes only) on macOS.
    """
    proc = Path(f"/proc/{pid}")
    if proc.is_dir():
        data = (proc / "cmdline").read_bytes() + b"\0" + (proc / "environ").read_bytes()
        return data.replace(b"\0", b"\n").decode(errors="replace")
    return subprocess.run(
        ["ps", "-E", "-ww", "-o", "command=", "-p", str(pid)],
        capture_output=True,
        text=True,
        timeout=5,
    ).stdout
