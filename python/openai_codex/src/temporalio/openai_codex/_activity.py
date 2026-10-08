"""The segment Activity: one ``codex app-server`` process per attempt, rebuilt from the rollout.

A *segment* is one Activity. It rebuilds a fresh ``CODEX_HOME`` from the rollout the Workflow holds,
starts ``codex app-server``, resumes the thread, injects the real output of the previous host-tool
call, and runs the turn. The turn ends when Codex answers, or (host-tool mode) when the model calls a
host tool: the segment then ends (the app-server is killed, see ``_run_segment``) and returns the
rollout lines it appended, and the Workflow runs the tool and starts the next segment. Because the
conversation lives in the Workflow, a retry on any Worker resumes from the last committed rollout.

In native mode Codex runs its own tools (shell, ``apply_patch``, ...) inside its sandbox. Whenever one
needs approval, the segment asks the Workflow through an Update and holds the request open until the
Workflow answers, heartbeating meanwhile.
"""

from __future__ import annotations

import asyncio
import contextlib
import glob
import os
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from typing import Any, Protocol

from temporalio import activity
from temporalio.client import Client
from temporalio.exceptions import ApplicationError

from ._app_server import AppServer, AppServerExited, default_codex_bin
from ._models import (
    CODEX_APPROVAL_UPDATE,
    CODEX_RUN_SEGMENT_ACTIVITY,
    CodexApprovalDecision,
    CodexApprovalRequest,
    CodexPendingCall,
    CodexSegmentInput,
    CodexSegmentResult,
    CodexTokenUsage,
)

# Per-thread Codex config that removes its built-in tools (host-tool mode), so every tool call is a
# host tool. (`apply_patch` has no feature flag; removing it needs a model-catalog override.)
THREAD_CONFIG: dict[str, Any] = {
    "features": {
        "shell_tool": False,
        "view_image": False,
        "goals": False,
        "multi_agent": False,
        "tool_suggest": False,
    },
    "web_search": "disabled",
}

# Safe defaults, applied before the Worker's own ``config_overrides`` so those can relax them.
# Commands the model runs get only Codex's "core" environment (PATH, HOME, ...), not the Worker's
# secrets: the Codex process itself keeps the full environment because it needs the provider key,
# but nothing the model runs should be able to read it.
SAFE_CONFIG_OVERRIDES: tuple[str, ...] = ('shell_environment_policy.inherit="core"',)

# How long a cancelled segment waits for Codex to stop its turn before the process is closed.
_INTERRUPT_TIMEOUT = 3.0

# Items worth surfacing to an observer: Codex's own tool activity (not messages or reasoning).
_OBSERVED_ITEM_TYPES = {
    "commandExecution",
    "fileChange",
    "mcpToolCall",
    "webSearch",
    "plan",
}


class CodexObserver(Protocol):
    """Receives live events from a segment (for example to stream them to a UI)."""

    def model_interaction_started(self, model: str | None) -> None:
        """A model interaction (one segment's turn) began."""

    def reply_delta(self, text: str) -> None:
        """An incremental chunk of the agent's reply text."""

    def item_started(self, item: dict[str, Any]) -> None:
        """Codex began one of its own tool actions (``item`` is Codex's raw item)."""

    def item_completed(self, item: dict[str, Any]) -> None:
        """Codex finished (or was denied) one of its own tool actions."""

    def model_interaction_ended(
        self, model: str | None, usage: CodexTokenUsage | None
    ) -> None:
        """The model interaction ended; ``usage`` is set when Codex reported it."""


ObserverFactory = Callable[[dict[str, Any]], AbstractAsyncContextManager[CodexObserver]]
"""Builds an observer, as an async context manager, from a segment's ``observer_context``."""


def _override_value(overrides: Sequence[str], key: str) -> str | None:
    """The string value a ``--config key="value"`` override sets (the last one wins)."""
    value: str | None = None
    for override in overrides:
        name, _, raw = override.partition("=")
        if name.strip() == key:
            value = raw.strip().strip('"')
    return value


def _token_usage(params: Mapping[str, Any]) -> CodexTokenUsage | None:
    last: Mapping[str, Any] | None = (params.get("tokenUsage") or {}).get(
        "last"
    ) or None
    if not last:
        return None
    return CodexTokenUsage(
        input_tokens=last.get("inputTokens"),
        output_tokens=last.get("outputTokens"),
        reasoning_tokens=last.get("reasoningOutputTokens"),
        total_tokens=last.get("totalTokens"),
    )


def _copy_auth_in(source: str, home: str) -> None:
    """Put a copy of the Codex login (``auth.json``) into a fresh ``CODEX_HOME``, readable only by us."""
    if not os.path.isfile(source):
        raise ApplicationError(
            f"the Codex auth file {source!r} does not exist on this Worker",
            type="CodexAuthMissing",
            non_retryable=True,
        )
    destination = os.path.join(home, "auth.json")
    shutil.copyfile(source, destination)
    os.chmod(destination, 0o600)


def _sync_auth_back(source: str, home: str) -> None:
    """Write the login back if Codex refreshed its tokens during the segment.

    Refresh tokens rotate: if a refresh happened in a throwaway home and the new tokens were thrown
    away with it, the original file would hold a refresh token that no longer works.
    """
    copy = os.path.join(home, "auth.json")
    if not os.path.isfile(copy) or not os.path.isfile(source):
        return
    with open(copy, "rb") as f:
        refreshed = f.read()
    with open(source, "rb") as f:
        if f.read() == refreshed:
            return
    temporary = f"{source}.{os.getpid()}.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "wb") as f:
        f.write(refreshed)
    os.replace(temporary, source)


def _rollout_files(home: str, thread_id: str) -> list[str]:
    return glob.glob(f"{home}/sessions/**/rollout-*-{thread_id}.jsonl", recursive=True)


class CodexActivities:
    """Worker-side configuration plus the segment Activity.

    Everything that must not enter Workflow history lives here, on the Worker: the Codex binary,
    model-provider config and environment (API keys).
    """

    def __init__(
        self,
        *,
        codex_bin: str | None = None,
        config_overrides: Sequence[str] = (),
        env: Mapping[str, str] | None = None,
        home_root: str | None = None,
        observer_factory: ObserverFactory | None = None,
        auth_file: str | None = None,
    ) -> None:
        """Configure the Worker side; see :class:`~temporalio.openai_codex.CodexPlugin`."""
        self._codex_bin = codex_bin
        self._config_overrides = (*SAFE_CONFIG_OVERRIDES, *config_overrides)
        self._env = dict(env or {})
        self._home_root = home_root
        self._observer_factory = observer_factory
        self._auth_file = os.path.expanduser(auth_file) if auth_file else None
        self._client: Client | None = None

    def bind_client(self, client: Client) -> None:
        """Give the Activity the Worker's client (it uses it to ask the Workflow for approvals)."""
        self._client = client

    @activity.defn(name=CODEX_RUN_SEGMENT_ACTIVITY)
    async def run_segment(self, inp: CodexSegmentInput) -> CodexSegmentResult:
        """Run one segment of a Codex turn."""
        home = tempfile.mkdtemp(prefix="codex-home-", dir=self._home_root)
        if self._auth_file is not None:
            try:
                _copy_auth_in(self._auth_file, home)
            except BaseException:
                shutil.rmtree(home, ignore_errors=True)
                raise
        # A long approval wait produces no Codex output, so beat on a timer for the whole segment.
        beat = asyncio.create_task(self._heartbeat())
        try:
            return await self._run_segment(inp, home)
        except AppServerExited as exc:
            raise ApplicationError(str(exc), type="CodexAppServerExited") from exc
        finally:
            beat.cancel()
            if self._auth_file is not None:
                with contextlib.suppress(OSError):
                    _sync_auth_back(self._auth_file, home)
            shutil.rmtree(home, ignore_errors=True)

    @staticmethod
    async def _interrupt(
        server: AppServer, thread_id: str, turn_id: str | None
    ) -> None:
        """Ask Codex to stop the turn, waiting a bounded time; the caller closes the process."""
        if turn_id is None or server.proc.returncode is not None:
            return

        async def stop() -> None:
            await server.call(
                "turn/interrupt", {"threadId": thread_id, "turnId": turn_id}
            )
            while (await server.notifications.get())["method"] not in (
                "turn/completed",
                "__eof__",
            ):
                pass

        with contextlib.suppress(Exception):
            await asyncio.wait_for(stop(), _INTERRUPT_TIMEOUT)

    @staticmethod
    async def _heartbeat() -> None:
        while True:
            activity.heartbeat("running")
            await asyncio.sleep(5)

    async def _ask_workflow(
        self, request: CodexApprovalRequest
    ) -> CodexApprovalDecision:
        """Ask the Workflow to decide an approval, waiting as long as it takes."""
        info = activity.info()
        workflow_id = info.workflow_id
        if self._client is None or workflow_id is None:
            return CodexApprovalDecision(
                False, "there is no Workflow to ask for approval"
            )
        handle = self._client.get_workflow_handle(
            workflow_id, run_id=info.workflow_run_id
        )
        return await handle.execute_update(
            CODEX_APPROVAL_UPDATE, request, result_type=CodexApprovalDecision
        )

    async def _run_segment(
        self, inp: CodexSegmentInput, home: str
    ) -> CodexSegmentResult:
        # 1. Rehydrate the committed rollout into the fresh CODEX_HOME (Codex resumes from just
        #    this file; no other state is needed).
        if inp.thread_id:
            assert inp.rollout_name is not None
            directory = os.path.join(home, "sessions", "2026", "01", "01")
            os.makedirs(directory)
            with open(os.path.join(directory, inp.rollout_name), "w") as f:
                f.write(inp.rollout)

        pending_calls: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        file_changes: dict[str, list[dict[str, Any]]] = {}

        def on_notification(note: dict[str, Any]) -> None:
            # A fileChange approval request carries only an item id; the changes arrive earlier, on
            # item/started, so remember them as they are read.
            if note["method"] == "item/started":
                item = (note.get("params") or {}).get("item") or {}
                if item.get("type") == "fileChange":
                    file_changes[item["id"]] = item.get("changes") or []

        async def on_request(method: str, params: Any) -> Any:
            if method == "item/tool/call":
                pending_calls.put_nowait(params)
                # Never answered: the Workflow runs the tool and the next segment injects the
                # result. Answering here would make Codex record an output we then duplicate.
                await asyncio.Event().wait()
            if method in (
                "item/commandExecution/requestApproval",
                "item/fileChange/requestApproval",
            ):
                is_command = method.startswith("item/commandExecution")
                request = CodexApprovalRequest(
                    kind="command" if is_command else "file_change",
                    item_id=params["itemId"],
                    command=(params.get("commandActions") or [{}])[0].get("command")
                    or params.get("command")
                    if is_command
                    else None,
                    cwd=params.get("cwd") if is_command else None,
                    reason=params.get("reason"),
                    changes=[]
                    if is_command
                    else file_changes.get(params["itemId"], []),
                    segment=activity.info().activity_id,
                    attempt=activity.info().attempt,
                )
                decision = await self._ask_workflow(request)
                if decision.interrupt:
                    return {"decision": "cancel"}
                return {"decision": "accept" if decision.approved else "decline"}
            raise RuntimeError(f"unsupported server request {method!r}")

        # Resolve the workspace on this Worker: the app-server is started inside it, so a relative
        # path would otherwise be resolved a second time, against itself.
        cwd = os.path.abspath(inp.cwd) if inp.cwd else home
        if inp.native_tools and not os.path.isdir(cwd):
            raise ApplicationError(
                f"the Codex workspace {cwd!r} does not exist on this Worker",
                type="CodexWorkspaceMissing",
                non_retryable=True,
            )
        server = AppServer(
            codex_bin=self._codex_bin or default_codex_bin(),
            home=home,
            config_overrides=self._config_overrides,
            env=self._env,
            cwd=cwd,
            on_request=on_request,
            on_notification=on_notification,
        )
        await server.start()
        try:
            policy: dict[str, Any] = (
                {"approvalPolicy": inp.approval_policy, "approvalsReviewer": "user"}
                if inp.native_tools
                else {"approvalPolicy": "never"}
            )
            # 2. Start or resume the thread.
            if inp.thread_id:
                thread_id = inp.thread_id
                # The thread config does NOT persist across a resume: without it Codex's built-in
                # tools (shell, apply_patch, goals, ...) come back in host-tool mode. Pass the
                # policy and sandbox every time.
                resume: dict[str, Any] = {
                    "threadId": thread_id,
                    "cwd": cwd,
                    "sandbox": inp.sandbox,
                    **policy,
                }
                if not inp.native_tools:
                    resume["config"] = THREAD_CONFIG
                # This Worker's configuration is authoritative, not what the thread started with:
                # a rollout names the provider and model it was recorded under, and the Worker
                # that resumes it may have been reconfigured since (or be a different Worker).
                provider = _override_value(self._config_overrides, "model_provider")
                if provider:
                    resume["modelProvider"] = provider
                model = inp.model or _override_value(self._config_overrides, "model")
                if model:
                    resume["model"] = model
                try:
                    await server.call("thread/resume", resume)
                except RuntimeError as exc:
                    if "Model provider" in str(exc) and "not found" in str(exc):
                        raise ApplicationError(
                            f"{exc}. The thread was recorded under a model provider this Worker "
                            "does not define; set `model_provider` (and `model_providers.<name>`) "
                            "in this Worker's `config_overrides`, which win on resume.",
                            type="CodexConfigDrift",
                            non_retryable=True,
                        ) from exc
                    raise
                # 3. Replace Codex's synthetic "aborted" output for the paused call with the real one.
                for call_id, output in inp.inject.items():
                    await server.call(
                        "thread/inject_items",
                        {
                            "threadId": thread_id,
                            "items": [
                                {
                                    "type": "function_call_output",
                                    "call_id": call_id,
                                    "output": output,
                                }
                            ],
                        },
                    )
            else:
                params: dict[str, Any] = {
                    "cwd": cwd,
                    "sandbox": inp.sandbox,
                    # Self-contained rollouts: required to rehydrate a thread on another Worker.
                    "historyMode": "legacy",
                    "dynamicTools": [
                        {
                            "type": "function",
                            "name": t.name,
                            "description": t.description,
                            "inputSchema": t.input_schema,
                        }
                        for t in inp.tools
                    ],
                    **policy,
                }
                if not inp.native_tools:
                    params["config"] = THREAD_CONFIG
                if inp.instructions:
                    params["developerInstructions"] = inp.instructions
                if inp.model:
                    params["model"] = inp.model
                started = await server.call("thread/start", params)
                thread_id = started["thread"]["id"]

            # 4. Run the turn, reporting to the observer when one is configured.
            observer_cm: AbstractAsyncContextManager[CodexObserver | None] = (
                self._observer_factory(inp.observer_context)
                if self._observer_factory is not None
                and inp.observer_context is not None
                else contextlib.nullcontext(None)
            )
            async with observer_cm as observer:
                if observer is not None:
                    observer.model_interaction_started(inp.model)
                started_turn = await server.call(
                    "turn/start",
                    {
                        "threadId": thread_id,
                        "input": [{"type": "text", "text": inp.prompt}],
                    },
                )
                turn_id = (started_turn.get("turn") or {}).get("id")
                final = ""
                usage: CodexTokenUsage | None = None

                async def pump() -> None:
                    nonlocal final, usage
                    while True:
                        note = await server.notifications.get()
                        method = note["method"]
                        activity.heartbeat(method)
                        if method == "__eof__":
                            raise AppServerExited(
                                "codex app-server exited mid-turn: "
                                + " | ".join(server.stderr[-3:])
                            )
                        params = note.get("params") or {}
                        if method == "item/agentMessage/delta" and observer is not None:
                            observer.reply_delta(params.get("delta", ""))
                        elif method == "thread/tokenUsage/updated":
                            usage = _token_usage(params) or usage
                        elif method in ("item/started", "item/completed"):
                            item = params.get("item") or {}
                            if method == "item/completed":
                                if item.get("type") == "agentMessage" and item.get(
                                    "text"
                                ):
                                    final = item["text"]
                            if (
                                observer is not None
                                and item.get("type") in _OBSERVED_ITEM_TYPES
                            ):
                                if method == "item/started":
                                    observer.item_started(item)
                                else:
                                    observer.item_completed(item)
                        elif method == "turn/completed":
                            turn_state = params.get("turn") or {}
                            if turn_state.get("status") == "failed":
                                # A failed turn (model error, bad request, ...) is a failed
                                # segment, not an empty reply: surface Codex's own message.
                                error = turn_state.get("error") or {}
                                raise ApplicationError(
                                    "codex turn failed: "
                                    f"{error.get('message') or error or 'unknown error'}",
                                    type="CodexTurnFailed",
                                )
                            return

                pump_task = asyncio.create_task(pump())
                call_task = asyncio.create_task(pending_calls.get())
                try:
                    done, _ = await asyncio.wait(
                        {pump_task, call_task}, return_when=asyncio.FIRST_COMPLETED
                    )
                except asyncio.CancelledError:
                    # The Activity was cancelled (the Workflow was cancelled or timed out, or this
                    # Worker is shutting down): stop the turn so Codex ends the commands it
                    # started, rather than leaving them behind when the process is closed.
                    pump_task.cancel()
                    call_task.cancel()
                    await self._interrupt(server, thread_id, turn_id)
                    raise
                call: dict[str, Any] | None = None
                if call_task in done:
                    call = call_task.result()
                    # Pause by killing the app-server, NOT turn/interrupt: an interrupt makes Codex
                    # record its own "aborted by user" output for the pending call, which would
                    # duplicate the real result injected next. A kill leaves a clean dangling
                    # function_call (no output) in the rollout.
                    pump_task.cancel()
                    server.kill()
                    await server.proc.wait()
                else:
                    call_task.cancel()
                    pump_task.result()  # re-raises if the app-server exited or the turn failed
                if observer is not None:
                    observer.model_interaction_ended(inp.model, usage)
        finally:
            await server.close()

        # 5. The process is gone (rollout flushed): ship only the lines this segment appended.
        files = _rollout_files(home, thread_id)
        if len(files) != 1:
            raise ApplicationError(
                f"expected one rollout for thread {thread_id}, found {files}"
            )
        with open(files[0]) as f:
            lines = f.read().splitlines()
        return CodexSegmentResult(
            thread_id=thread_id,
            rollout_name=os.path.basename(files[0]),
            tail="".join(line + "\n" for line in lines[inp.committed_lines :]),
            status="tool_call" if call else "done",
            final_response=final,
            call=(
                CodexPendingCall(
                    call_id=call["callId"],
                    tool=call["tool"],
                    arguments=call.get("arguments") or {},
                )
                if call
                else None
            ),
            usage=usage,
        )
