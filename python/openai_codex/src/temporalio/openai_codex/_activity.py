"""The segment Activity: one ``codex app-server`` process per attempt, rebuilt from the rollout.

A *segment* is one Activity. It rebuilds a fresh ``CODEX_HOME`` from the rollout the Workflow holds,
starts ``codex app-server``, resumes the thread, injects the real output of the previous tool call,
and runs the turn. When the model calls a host tool, the segment ends (the app-server is killed, see
``_run_segment``) and returns the rollout lines it appended; the Workflow runs the tool and starts
the next segment. Because the conversation lives in the Workflow, a retry on any Worker resumes
from the last committed rollout, and a tool that already ran is never run again.
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
from temporalio.exceptions import ApplicationError

from ._app_server import AppServer, AppServerExited, default_codex_bin
from ._models import (
    CODEX_RUN_SEGMENT_ACTIVITY,
    CodexPendingCall,
    CodexSegmentInput,
    CodexSegmentResult,
    CodexTokenUsage,
)

# Per-thread Codex config that removes its built-in tools, so every tool call is a host tool.
# (`apply_patch` has no feature flag; removing it needs a model-catalog override, not done yet.)
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


class CodexObserver(Protocol):
    """Receives live events from a segment (for example to stream them to a UI)."""

    def model_interaction_started(self, model: str | None) -> None:
        """A model interaction (one segment's turn) began."""

    def reply_delta(self, text: str) -> None:
        """An incremental chunk of the agent's reply text."""

    def model_interaction_ended(
        self, model: str | None, usage: CodexTokenUsage | None
    ) -> None:
        """The model interaction ended; ``usage`` is set when Codex reported it."""


ObserverFactory = Callable[[dict[str, Any]], AbstractAsyncContextManager[CodexObserver]]
"""Builds an observer, as an async context manager, from a segment's ``observer_context``."""


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
    ) -> None:
        """Configure the Worker side; see :class:`~temporalio.openai_codex.CodexPlugin`."""
        self._codex_bin = codex_bin
        self._config_overrides = tuple(config_overrides)
        self._env = dict(env or {})
        self._home_root = home_root
        self._observer_factory = observer_factory

    @activity.defn(name=CODEX_RUN_SEGMENT_ACTIVITY)
    async def run_segment(self, inp: CodexSegmentInput) -> CodexSegmentResult:
        """Run one segment of a Codex turn."""
        home = tempfile.mkdtemp(prefix="codex-home-", dir=self._home_root)
        try:
            return await self._run_segment(inp, home)
        except AppServerExited as exc:
            raise ApplicationError(str(exc), type="CodexAppServerExited") from exc
        finally:
            shutil.rmtree(home, ignore_errors=True)

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

        async def on_request(method: str, params: Any) -> Any:
            if method == "item/tool/call":
                pending_calls.put_nowait(params)
                # Never answered: the Workflow runs the tool and the next segment injects the
                # result. Answering here would make Codex record an output we then duplicate.
                await asyncio.Event().wait()
            raise RuntimeError(f"unsupported server request {method!r}")

        cwd = inp.cwd or home
        server = AppServer(
            codex_bin=self._codex_bin or default_codex_bin(),
            home=home,
            config_overrides=self._config_overrides,
            env=self._env,
            cwd=cwd,
            on_request=on_request,
        )
        await server.start()
        try:
            # 2. Start or resume the thread.
            if inp.thread_id:
                thread_id = inp.thread_id
                # The thread config does NOT persist across a resume: without it Codex's built-in
                # tools (shell, apply_patch, goals, ...) come back. Pass it every time.
                await server.call(
                    "thread/resume",
                    {
                        "threadId": thread_id,
                        "cwd": cwd,
                        "approvalPolicy": "never",
                        "sandbox": inp.sandbox,
                        "config": THREAD_CONFIG,
                    },
                )
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
                    "approvalPolicy": "never",
                    "sandbox": inp.sandbox,
                    # Self-contained rollouts: required to rehydrate a thread on another Worker.
                    "historyMode": "legacy",
                    "config": THREAD_CONFIG,
                    "dynamicTools": [
                        {
                            "type": "function",
                            "name": t.name,
                            "description": t.description,
                            "inputSchema": t.input_schema,
                        }
                        for t in inp.tools
                    ],
                }
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
                await server.call(
                    "turn/start",
                    {
                        "threadId": thread_id,
                        "input": [{"type": "text", "text": inp.prompt}],
                    },
                )
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
                        elif method == "item/completed":
                            item = params.get("item") or {}
                            if item.get("type") == "agentMessage" and item.get("text"):
                                final = item["text"]
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
                done, _ = await asyncio.wait(
                    {pump_task, call_task}, return_when=asyncio.FIRST_COMPLETED
                )
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
