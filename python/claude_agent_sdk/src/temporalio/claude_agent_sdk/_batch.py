"""Bounded native batches, using the published engine's own tool scheduler.

Preparation never runs a tool. Execution replays the accepted assistant blocks,
with their original IDs, and waits for each corresponding Activity before the
native tool is allowed to run. Native result carriers are preserved verbatim.
"""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import hashlib
import json
import os
import secrets
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, cast

from claude_agent_sdk import (
    AssistantMessage,
    HookMatcher,
    InMemorySessionStore,
    MirrorErrorMessage,
    PermissionResultAllow,
    ResultError,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
    create_sdk_mcp_server,
    project_key_for_directory,
    tool,
)
from temporalio import activity
from temporalio.client import (
    WorkflowUpdateRPCTimeoutOrCancelledError,
    WorkflowUpdateStage,
)
from temporalio.service import RPCError, RPCStatusCode

from ._conversation import external_storage_on, page_limit, read_conversation, too_large
from ._events import emit
from ._models import (
    DeferredCall,
    NativeRequest,
    SegmentInput,
    SegmentOutput,
    ToolOutcome,
    ToolStepInput,
)
from ._replay import ReplayModel


async def update(
    handle: Any, request: NativeRequest, id: str, *, accepted: bool = False
) -> Any:
    """Retry transient admission failures with the original Update identity."""
    while True:
        try:
            if accepted:
                return await handle.start_update(
                    "__claude_agent_native_tool",
                    request,
                    id=id,
                    wait_for_stage=WorkflowUpdateStage.ACCEPTED,
                )
            return await handle.execute_update(
                "__claude_agent_native_tool", request, id=id, result_type=ToolOutcome
            )
        except WorkflowUpdateRPCTimeoutOrCancelledError:
            await asyncio.sleep(0.1)
        except RPCError as error:
            if error.status not in (
                RPCStatusCode.DEADLINE_EXCEEDED,
                RPCStatusCode.CANCELLED,
                RPCStatusCode.UNAVAILABLE,
                RPCStatusCode.RESOURCE_EXHAUSTED,
            ):
                raise
            await asyncio.sleep(0.1)


def blocks(entry: Any) -> list[dict[str, Any]]:
    """Return the content blocks of an opaque native transcript entry."""
    content = entry.get("message", {}).get("content")
    return content if isinstance(content, list) else []


def result_ids(entry: dict[str, Any]) -> set[str]:
    """Return the original tool IDs answered by this native entry."""
    return {b["tool_use_id"] for b in blocks(entry) if b.get("type") == "tool_result"}


def notebook_read(response: Any) -> dict[str, Any] | None:
    """Retain notebook read state omitted by the engine's transcript resumer."""
    if not isinstance(response, dict) or response.get("type") != "notebook":
        return None
    path = Path(response["file"]["filePath"])
    try:
        before = path.stat()
        raw = path.read_text()
        after = path.stat()
        cells = json.loads(raw).get("cells", [])
    except (OSError, ValueError):
        return None
    native = response["file"]["cells"]
    if before.st_mtime_ns != after.st_mtime_ns or len(cells) != len(native):
        return None
    for actual, recorded in zip(cells, native):
        source = actual.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        if source != recorded.get("source"):
            return None
    return {"content": raw, "native": response}


def resume_seed(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Restore a notebook's native read cache privately, preserving its carrier.

    The published engine restores text reads from transcript metadata, but omits
    notebook reads. Its existing text-read resumer accepts the recorded raw JSON;
    canonical transcripts continue to contain the original notebook result.
    """
    entries = copy.deepcopy(entries)
    for entry in entries:
        state = entry.get("temporalNotebookRead")
        if state:
            content = state["content"]
            for block in blocks(entry):
                if block.get("type") == "tool_result":
                    state.setdefault("rendered", block.get("content"))
                    block["content"] = content
            entry["toolUseResult"] = {
                "type": "text",
                "file": {
                    "filePath": state["native"]["file"]["filePath"],
                    "content": content,
                    "numLines": len(content.splitlines()),
                    "startLine": 1,
                    "totalLines": len(content.splitlines()),
                },
            }
    return entries


def canonical_reads(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Undo private cache restoration before publishing a canonical transcript."""
    for entry in entries:
        if state := entry.get("temporalNotebookRead"):
            entry["toolUseResult"] = state["native"]
            if "rendered" in state:
                for block in blocks(entry):
                    if block.get("type") == "tool_result":
                        block["content"] = state["rendered"]
    return entries


def permission_mode(entries: list[dict[str, Any]], initial: str) -> str:
    """Carry accepted native plan transitions across engine processes."""
    names = {
        b["id"]: b.get("name")
        for e in entries
        for b in blocks(e)
        if b.get("type") == "tool_use"
    }
    mode = initial
    for entry in entries:
        for block in blocks(entry):
            if block.get("type") != "tool_result" or block.get("is_error"):
                continue
            name = names.get(block.get("tool_use_id"))
            if name == "EnterPlanMode":
                mode = "plan"
            elif name == "ExitPlanMode":
                mode = "default" if initial == "plan" else initial
    return mode


def splice(
    seed: list[dict[str, Any]], outcomes: dict[str, ToolOutcome]
) -> list[dict[str, Any]]:
    """Append completed carriers, retaining native file-tool metadata."""
    from ._runner import _result_content

    output = copy.deepcopy(seed)
    parent = next((e["uuid"] for e in reversed(output) if e.get("uuid")), None)
    assistant = next((e for e in reversed(output) if e.get("type") == "assistant"), {})
    for tid, outcome in outcomes.items():
        source = next(
            (
                e
                for e in reversed(seed)
                if any(
                    b.get("type") == "tool_use" and b.get("id") == tid
                    for b in blocks(e)
                )
            ),
            assistant,
        )
        entries: list[dict[str, Any]] = copy.deepcopy(outcome.entries)
        if not entries:
            entries = [
                {
                    **{
                        k: v
                        for k, v in assistant.items()
                        if k not in ("message", "type", "uuid", "requestId")
                    },
                    "type": "user",
                    "userType": "external",
                    "toolUseResult": _result_content(outcome),
                    "sourceToolAssistantUUID": source.get("uuid"),
                    "uuid": str(uuid.uuid5(uuid.NAMESPACE_URL, "claude-result:" + tid)),
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tid,
                                "content": _result_content(outcome),
                                "is_error": outcome.is_error,
                            }
                        ],
                    },
                }
            ]
        if set().union(*(result_ids(e) for e in entries)) != {tid}:
            raise ValueError(f"Native outcome does not answer exactly tool call {tid}")
        for entry in entries:
            entry["parentUuid"] = parent
            if "sourceToolAssistantUUID" in entry:
                entry["sourceToolAssistantUUID"] = source.get("uuid")
            output.append(entry)
            parent = entry.get("uuid", parent)
    return output


async def committed(runner: Any, inp: Any) -> list[dict[str, Any]]:
    """Read and truncate the accepted conversation, independent of local attempts."""
    from ._runner import _seed, _without_cost_state

    held = inp.transcript is not None or (
        inp.conversation is not None and inp.conversation.entries > 0
    )
    if runner._store is None:
        entries = await read_conversation(inp)
    elif held:
        raise RuntimeError(
            "This conversation is held by its Workflow, not a session store"
        )
    else:
        entries = (
            await runner._store.load(
                {
                    "project_key": project_key_for_directory(runner._cwd),
                    "session_id": inp.session_id,
                }
            )
            or []
        )
    if inp.checkpoint is not None:
        seed = _seed(entries, inp.checkpoint)
        if seed is None:
            raise RuntimeError(f"Missing accepted checkpoint {inp.checkpoint}")
        return _without_cost_state(seed)
    if entries:
        raise RuntimeError("A conversation without a checkpoint cannot be resumed")
    return []


def identity(session: str, checkpoint: str, calls: list[DeferredCall]) -> str:
    """Identify one immutable accepted batch in the caller's shared workspace."""
    value = json.dumps(
        [session, checkpoint, [dataclasses.asdict(c) for c in calls]], sort_keys=True
    )
    return hashlib.sha256(value.encode()).hexdigest()


def state_options(runner: Any, options: dict[str, Any], workspace: str) -> None:
    """Keep native task and plan files in the caller-managed shared workspace."""
    root = (
        Path(runner._cwd or os.getcwd())
        / ".temporal-claude"
        / "state"
        / hashlib.sha256(workspace.encode()).hexdigest()
    )
    options["env"]["TCA_NATIVE_STATE"] = str(root)
    options["env"]["CLAUDE_CODE_TASK_LIST_ID"] = workspace


async def native_state(handle: Any, agent: str) -> dict[str, Any]:
    """Read accepted calls and child conversations without an unbounded Query."""
    state: dict[str, Any] = {"calls": {}, "children": {}}
    start = 0
    while True:
        response = await handle.query(
            "__claude_agent_native_state", args=[agent, None, start, page_limit()]
        )
        state["calls"].update(response["calls"])
        start += len(response["calls"])
        if start >= response["total"]:
            break
        if not response["calls"]:
            raise RuntimeError("Native call Query made no progress")
    for child, total in response["children"].items():
        entries: list[dict[str, Any]] = []
        while len(entries) < total:
            response = await handle.query(
                "__claude_agent_native_state",
                args=[agent, child, len(entries), page_limit()],
            )
            if not response["entries"]:
                raise RuntimeError("Native child Query made no progress")
            entries.extend(response["entries"])
        state["children"][child] = entries
    return state


async def finalized(
    runner: Any, inp: SegmentInput, seed: list[dict[str, Any]]
) -> tuple[dict[str, ToolOutcome], list[dict[str, Any]], float]:
    """Prefer final native carriers after all Activities in a batch have completed.

    A PostToolUse response lets an Activity release its slot before later calls
    finish. Normal continuation waits for the controller's final transcript flush.
    If that controller was lost, its committed response is the recovery carrier.
    """
    if not inp.injected or inp.checkpoint is None:
        return inp.injected, [], 0.0
    calls = [
        DeferredCall(
            b["id"],
            b["name"].removeprefix("mcp__durable__"),
            dict(b["input"]),
            "durable" if b["name"].startswith("mcp__durable__") else "engine",
        )
        for e in seed
        for b in blocks(e)
        if b.get("type") == "tool_use" and b["id"] in inp.injected
    ]
    folder = (
        Path(runner._cwd or os.getcwd())
        / ".temporal-claude"
        / identity(inp.session_id, inp.checkpoint, calls)
    )
    lease = folder / "owner.lock"
    if not lease.exists():
        return inp.injected, [], 0.0
    from ._runner import _lock_file, _unlock_file

    fd = os.open(lease, os.O_RDWR)
    try:
        while True:
            try:
                _lock_file(fd)
            except OSError:
                await asyncio.sleep(0.05)
            else:
                _unlock_file(fd)
                break
        outcomes = {
            tid: ToolOutcome(**saved)
            if (
                saved := read(
                    folder / "outcomes" / hashlib.sha256(tid.encode()).hexdigest()
                )
            )
            is not None
            else outcome
            for tid, outcome in inp.injected.items()
        }
        return (
            outcomes,
            read(folder / "tail") or [],
            float((read(folder / "cost") or {}).get("usd", 0)),
        )
    finally:
        os.close(fd)


async def prepare(runner: Any, inp: SegmentInput, attempt: int) -> SegmentOutput:
    """Record a whole assistant message before any of its native calls execute."""
    from ._runner import (
        FINAL_RESULT_ERRORS,
        _as_messages,
        _command_env,
        _engine_messages,
        _final_api_error,
        _hold_worker_lock,
        _hook_folder,
        _hook_violation,
        _kept,
        _last_entry,
        _release_worker_lock,
        _without_cost_state,
    )

    seed = await committed(runner, inp)
    accepted = list(seed)
    outcomes, tail, helper_cost = await finalized(runner, inp, seed)
    seed = splice(seed, outcomes)
    for entry in tail:
        entry["parentUuid"] = seed[-1].get("uuid") if seed else None
        seed.append(entry)
    from ._runner import _attempt_session_id

    sid = _attempt_session_id(inp.session_id, attempt) if not seed else inp.session_id
    if runner._store is not None and seed:
        # Publish into a new immutable branch, so an Activity that retries after
        # publishing cannot append a second branch to the same canonical key.
        sid = str(uuid.uuid4())
        seed = [{**e, **({"sessionId": sid} if "sessionId" in e else {})} for e in seed]
    store: Any = InMemorySessionStore()
    key = {
        "project_key": project_key_for_directory(runner._cwd),
        "session_id": sid,
    }
    if seed:
        await store.append(key, resume_seed(seed))
    closure: str | None = None
    if inp.injected:
        # The published engine inserts a new user turn when resuming a transcript
        # that ends in tool results. Close that turn privately so the next query
        # continues the accepted task, with native file metadata already restored.
        assistant = next(e for e in reversed(seed) if e.get("type") == "assistant")
        message_fields = cast(dict[str, Any], assistant["message"])
        closure = str(uuid.uuid4())
        await store.append(
            key,
            [
                {
                    **assistant,
                    "uuid": closure,
                    "parentUuid": seed[-1].get("uuid"),
                    "message": {
                        **message_fields,
                        "id": "msg_private_resume",
                        "content": [
                            {
                                "type": "text",
                                "text": "<system-reminder>Tool results recorded.</system-reminder>",
                            }
                        ],
                        "stop_reason": "end_turn",
                    },
                }
            ],
        )
    if activity.in_activity() and inp.conversation is not None:
        info = activity.info()
        handle = activity.client().get_workflow_handle(
            info.workflow_id or "", run_id=info.workflow_run_id
        )
        state = await native_state(handle, inp.conversation.agent)
        for child, child_entries in state.get("children", {}).items():
            await store.append(
                {**key, "subpath": "subagents/agent-" + child},
                resume_seed(child_entries),
            )
    hook_dir = _hook_folder()
    violations: list[str] = []
    options = runner._engine_options(
        inp,
        {},
        sid,
        bool(seed),
        store,
        None,
        hook_dir,
        runner._durable_server(inp.tools, []),
        violations,
    )
    options["max_turns"] = 1
    options["permission_mode"] = permission_mode(seed, options["permission_mode"])

    async def deny(data: Any, tid: str | None, context: Any) -> Any:
        del data, tid, context
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": "The accepted batch must commit before its Activities run",
            }
        }

    hooks = options.get("hooks") or {}
    options["hooks"] = {
        **hooks,
        "PreToolUse": [HookMatcher(hooks=[deny]), *(hooks.get("PreToolUse") or [])],
    }
    options["cli_path"] = runner._cli_path
    state_options(runner, options, inp.workspace_id or inp.session_id)
    options["env"] = _command_env(
        options["env"],
        hook_dir,
        runner._cwd,
        {},
        {
            "TCA_PREPARE_BATCH": "1",
            "TCA_BATCH_LEASE": str(Path(hook_dir) / "engine.lock"),
        },
    )
    calls: list[DeferredCall] = []
    result: ResultMessage | None = None
    ours = not seed and not any(p.exists() for p in runner._local_copy(sid))
    lock = _hold_worker_lock(hook_dir)
    try:
        messages = await _as_messages(
            inp.prompt
            or "<system-reminder>Continue from the recorded tool results.</system-reminder>"
        )
        engine = _engine_messages(options, messages, bool(seed))
        try:
            async for message in engine:
                if isinstance(message, MirrorErrorMessage):
                    raise RuntimeError(message.error)
                if (
                    isinstance(message, AssistantMessage)
                    and message.parent_tool_use_id is None
                ):
                    for block in message.content:
                        if isinstance(block, ToolUseBlock):
                            calls.append(
                                DeferredCall(
                                    block.id,
                                    block.name.removeprefix("mcp__durable__"),
                                    dict(block.input),
                                    "durable"
                                    if block.name.startswith("mcp__durable__")
                                    else "engine",
                                )
                            )
                        elif isinstance(block, TextBlock) and block.text.strip():
                            emit({"type": "text", "text": block.text})
                elif isinstance(message, ResultMessage):
                    result = message
        except ResultError as error:
            if error.subtype != "error_max_turns" or result is None:
                if error.subtype in FINAL_RESULT_ERRORS or _final_api_error(error):
                    return SegmentOutput(
                        inp.session_id, is_error=True, error=str(error)
                    )
                raise
        finally:
            await engine.aclose()
    finally:
        _release_worker_lock(lock)
        shutil.rmtree(hook_dir, ignore_errors=True)
        if ours:
            runner._forget_local_copy(sid)
    if result is None:
        raise RuntimeError("Native preparation returned no result")
    if problem := _hook_violation(violations):
        return SegmentOutput(inp.session_id, is_error=True, error=problem)
    if result.is_error and not (calls and result.subtype == "error_max_turns"):
        return SegmentOutput(
            inp.session_id, is_error=True, error=str(result.errors or result.subtype)
        )
    if len({c.id for c in calls}) != len(calls):
        raise RuntimeError("Native preparation returned duplicate tool call IDs")
    entries = canonical_reads(_without_cost_state(await store.load(key) or []))
    if closure is not None:
        entries = [e for e in entries if e.get("uuid") != closure]
        for e in entries:
            if e.get("parentUuid") == closure:
                e["parentUuid"] = seed[-1].get("uuid")
    ids = {c.id for c in calls}
    stop = next((i for i, e in enumerate(entries) if result_ids(e) & ids), len(entries))
    entries = entries[:stop]
    checkpoint = _last_entry(entries)
    if checkpoint is None or (calls and result.subtype != "error_max_turns"):
        raise RuntimeError("Native preparation escaped its bounded turn")
    out = SegmentOutput(
        session_id=sid,
        checkpoint=checkpoint,
        deferred=calls[0] if calls else None,
        siblings=calls[1:],
        result=result.result if not calls else None,
        cost_usd=float(result.total_cost_usd or 0) + helper_cost,
        external_storage=external_storage_on(),
        execution_protocol=2,
    )
    if runner._store is None:
        keep, covered = _kept(accepted, entries)
        out.transcript_keep = keep
        out.transcript_add = entries[covered:]
        problem = too_large(out)
        if problem is not None:
            return SegmentOutput(inp.session_id, is_error=True, error=problem)
    else:
        # Preparation has only appended an accepted message. Private denials are
        # never published to a caller's store.
        await runner._store.append(key, entries)
    return out


def write(path: Path, value: Any) -> None:
    """Publish a mailbox record atomically, accessible only to this user."""
    temporary = path.with_name(path.name + "." + secrets.token_hex(8))
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def read(path: Path) -> Any:
    """Read an atomically published mailbox record, or None when absent."""
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None


class NativeBatch:
    """A shared-workspace native execution, independent of Activity slot ownership."""

    def __init__(
        self, runner: Any, step: ToolStepInput, entries: list[dict[str, Any]]
    ) -> None:
        """Create or join the batch's private shared-workspace mailbox."""
        self.runner = runner
        self.step = step
        self.entries = entries
        self.digest = identity(step.session_id, step.checkpoint, step.batch)
        self.folder = (
            Path(runner._cwd or os.getcwd()) / ".temporal-claude" / self.digest
        )
        self.folder.mkdir(parents=True, mode=0o700, exist_ok=True)
        for name in ("allowed", "outcomes", "cancelled", "children"):
            (self.folder / name).mkdir(mode=0o700, exist_ok=True)
        self.task: asyncio.Task[None] | None = None
        self.children: dict[str, list[dict[str, Any]]] = {}
        self.handle: Any = None
        self.queue: str | None = None
        if activity.in_activity():
            info = activity.info()
            self.queue = info.task_queue.rsplit(".__claude_native_", 1)[0]
            self.handle = activity.client().get_workflow_handle(
                info.workflow_id or "", run_id=info.workflow_run_id
            )

    async def restore_children(
        self, root_ids: set[str]
    ) -> tuple[dict[str, str], list[dict[str, Any]]]:
        """Reconstruct unfinished children from accepted calls and native carriers."""
        state: dict[str, Any] = {"calls": {}, "children": {}}
        if self.handle is not None and self.step.conversation is not None:
            state = await native_state(self.handle, self.step.conversation.agent)
        self.children = copy.deepcopy(state.get("children", {}))
        calls = {
            tid: value
            for tid, value in state.get("calls", {}).items()
            if value["request"]["controller"] == str(self.folder)
        }
        parents: dict[str, str] = {}
        for value in calls.values():
            request = value["request"]
            if request.get("parent"):
                parents[request["parent"]] = request["child"]
        reachable = set(root_ids)
        for _ in range(8):
            reachable.update(
                tid
                for tid, value in calls.items()
                if value["request"].get("parent") in reachable
            )
        parents = {tid: child for tid, child in parents.items() if tid in reachable}
        recovery: list[dict[str, Any]] = []
        for child in set(parents.values()):
            entries = copy.deepcopy(self.children.get(child, []))
            if not entries:
                entries = next(
                    (
                        copy.deepcopy(v["request"]["entries"])
                        for v in calls.values()
                        if v["request"]["child"] == child
                    ),
                    [],
                )
            answered = set().union(*(result_ids(e) for e in entries))
            additions: dict[str, ToolOutcome] = {}
            for tid, value in calls.items():
                if value["request"]["child"] != child or tid in answered:
                    continue
                outcome = value.get("outcome") or read(self.path("outcomes", tid))
                if outcome is not None:
                    additions[tid] = ToolOutcome(**outcome)
            entries = splice(entries, additions) if additions else entries
            answered = set().union(*(result_ids(e) for e in entries))
            unresolved = [
                b
                for e in entries
                for b in blocks(e)
                if b.get("type") == "tool_use" and b["id"] not in answered
            ]
            if unresolved:
                first = next(
                    i
                    for i, e in enumerate(entries)
                    if any(
                        b.get("id") in {u["id"] for u in unresolved} for b in blocks(e)
                    )
                )
                message_id = entries[first].get("message", {}).get("id")
                if message_id is not None:
                    first = next(
                        i
                        for i, e in enumerate(entries)
                        if e.get("type") == "assistant"
                        and e.get("message", {}).get("id") == message_id
                    )
                prefix = entries[:first]
                completed: list[dict[str, Any]] = []
                for accepted in entries[first:]:
                    if (
                        accepted.get("type") != "assistant"
                        or accepted.get("message", {}).get("id") != message_id
                    ):
                        continue
                    kept = [
                        b
                        for b in blocks(accepted)
                        if b.get("type") == "tool_use" and b["id"] in answered
                    ]
                    if kept:
                        prefix.append(
                            {
                                **accepted,
                                "message": {**accepted["message"], "content": kept},
                            }
                        )
                        completed.extend(kept)
                if completed:
                    completed_outcomes = {}
                    for b in completed:
                        carriers = [
                            e for e in entries[first + 1 :] if b["id"] in result_ids(e)
                        ]
                        completed_outcomes[b["id"]] = ToolOutcome(entries=carriers)
                    prefix = splice(prefix, completed_outcomes)
                prompt = next(
                    (
                        e.get("message", {}).get("content")
                        for e in entries
                        if e.get("type") == "user" and not result_ids(e)
                    ),
                    "",
                )
                if isinstance(prompt, list):
                    prompt = " ".join(b.get("text", "") for b in prompt)
                if not isinstance(prompt, str) or not prompt:
                    raise RuntimeError("An interrupted child has no native prompt")
                recovery.append(
                    {
                        "child": child,
                        "prompt": prompt,
                        "blocks": [
                            b
                            for e in entries[first:]
                            for b in blocks(e)
                            if e.get("type") == "assistant"
                            and e.get("message", {}).get("id") == message_id
                            and (b.get("type") != "tool_use" or b["id"] not in answered)
                        ],
                    }
                )
                entries = prefix
            self.children[child] = entries
        return parents, recovery

    def path(self, kind: str, tid: str) -> Path:
        """Map an untrusted native ID to a safe mailbox filename."""
        return self.folder / kind / hashlib.sha256(tid.encode()).hexdigest()

    async def result(
        self, call: DeferredCall, response: Any = None, attempt: int = 1
    ) -> ToolOutcome:
        """Authorize this Activity's call and await its own native result."""
        write(self.path("allowed", call.id), {"id": call.id, "response": response})
        self.path("cancelled", call.id).unlink(missing_ok=True)
        if attempt > 1 and (self.task is None or self.task.done()):
            (self.folder / "failure").unlink(missing_ok=True)
        try:
            while True:
                saved = read(self.path("outcomes", call.id))
                if saved is not None:
                    return ToolOutcome(**saved)
                failure = read(self.folder / "failure")
                if failure is not None and self.handle is None:
                    raise RuntimeError(failure["error"])
                if self.handle is None and (self.task is None or self.task.done()):
                    self.task = asyncio.create_task(self.execute())
                    self.task.add_done_callback(self.release)
                await asyncio.sleep(0.05)
        except asyncio.CancelledError:
            write(self.path("cancelled", call.id), {"id": call.id})
            write(self.folder / "abort", {"id": call.id})
            if self.task is not None:
                self.task.cancel()
            raise

    async def control(self, attempt: int) -> ToolOutcome:
        """Keep the controller recoverable without occupying a native tool slot."""
        if attempt > 1:
            (self.folder / "failure").unlink(missing_ok=True)
        try:
            while True:
                if failure := read(self.folder / "failure"):
                    raise RuntimeError(failure["error"])
                if all(
                    read(self.path("outcomes", c.id)) is not None
                    for c in self.step.batch
                ):
                    if self.task is not None:
                        await self.task
                    return ToolOutcome()
                if self.task is None or self.task.done():
                    self.task = asyncio.create_task(self.execute())
                    self.task.add_done_callback(self.release)
                await asyncio.sleep(0.05)
        finally:
            if self.task is not None and not self.task.done():
                write(self.folder / "abort", {"controller": True})
                self.task.cancel()
                await asyncio.gather(self.task, return_exceptions=True)

    def release(self, task: asyncio.Task[None]) -> None:
        """Release completed batch memory; durable receipts remain in the workspace."""
        if task.cancelled() or read(self.folder / "failure") is not None:
            return
        if all(read(self.path("outcomes", c.id)) is not None for c in self.step.batch):
            batches = self.runner._native_batches
            key = (self.step.session_id, self.step.checkpoint)
            if batches.get(key) is self:
                batches.pop(key)

    async def execute(self) -> None:
        """Execute the accepted message under an exclusive cross-worker lease."""
        from ._runner import (
            _as_messages,
            _command_env,
            _engine_messages,
            _hold_worker_lock,
            _hook_folder,
            _lock_file,
            _release_worker_lock,
            _unlock_file,
        )

        fd = os.open(self.folder / "owner.lock", os.O_RDWR | os.O_CREAT, 0o600)
        try:
            try:
                _lock_file(fd)
            except OSError:
                return
            calls = self.step.batch or [self.step.call]
            (self.folder / "abort").unlink(missing_ok=True)
            done = {
                c.id: ToolOutcome(**saved)
                for c in calls
                if (saved := read(self.path("outcomes", c.id))) is not None
            }
            done.update(self.step.batch_outcomes)
            pending = [c for c in calls if c.id not in done]
            expected = {c.id: c for c in pending}
            if not pending:
                return
            # A private execution context excludes the unresolved original message.
            # Completed calls keep their original result carriers and read state.
            ids = {c.id for c in calls}
            recovery_agents, recovery_blocks = await self.restore_children(ids)
            stop = next(
                i
                for i, e in enumerate(self.entries)
                if any(b.get("id") in ids for b in blocks(e))
            )
            message_id = self.entries[stop].get("message", {}).get("id")
            if message_id is not None:
                stop = next(
                    i
                    for i, e in enumerate(self.entries)
                    if e.get("type") == "assistant"
                    and e.get("message", {}).get("id") == message_id
                )
            seed = copy.deepcopy(self.entries[:stop])
            if done:
                for original in self.entries[stop:]:
                    kept = [
                        b
                        for b in blocks(original)
                        if b.get("type") == "tool_use" and b.get("id") in done
                    ]
                    if kept:
                        assistant = copy.deepcopy(original)
                        assistant["message"]["content"] = kept
                        seed.append(assistant)
                seed = splice(seed, done)
            sid = self.step.workspace_id or self.step.session_id
            key = {
                "project_key": project_key_for_directory(self.runner._cwd),
                "session_id": sid,
            }
            batch = self
            parents: dict[str, str] = {}
            depths: dict[str, int] = {c.id: 0 for c in calls}
            recovering_locks: dict[str, asyncio.Lock] = {}
            recovering_calls: dict[str, asyncio.Lock] = {}

            accepting = False
            announced: set[str] = set()

            async def announce(call: DeferredCall) -> None:
                if (
                    batch.handle is None
                    or call.kind != "engine"
                    or call.id in announced
                ):
                    return
                announced.add(call.id)
                await update(
                    batch.handle,
                    NativeRequest(
                        self.step.conversation.agent if self.step.conversation else "",
                        str(batch.folder),
                        call,
                    ),
                    id=f"claude-ready-{self.step.conversation.agent if self.step.conversation else ''}-{batch.digest}-{call.id}",
                )

            async def accept_child(child: str, tid: str) -> DeferredCall:
                child_key = {**key, "subpath": "subagents/agent-" + child}
                child_entries = await store.load(child_key) or []
                original: dict[str, Any] | None = None
                for _ in range(100):
                    original = next(
                        (
                            b
                            for e in child_entries
                            for b in blocks(e)
                            if b.get("type") == "tool_use" and b.get("id") == tid
                        ),
                        None,
                    )
                    if original is not None:
                        break
                    await asyncio.sleep(0.05)
                    child_entries = await store.load(child_key) or []
                if original is None:
                    raise RuntimeError(
                        "A child call did not reach its native transcript"
                    )
                for _ in range(100):
                    if tid in parents:
                        break
                    await asyncio.sleep(0.05)
                if tid not in parents:
                    raise RuntimeError("A child call has no original Agent parent")
                if batch.handle is None:
                    raise RuntimeError(
                        "Native subagents require a Workflow-owned runner"
                    )
                call = DeferredCall(
                    tid,
                    original["name"].removeprefix("mcp__durable__"),
                    dict(original["input"]),
                    "durable"
                    if original["name"].startswith("mcp__durable__")
                    else "engine",
                )
                expected[tid] = call
                depths[tid] = depths.get(parents[tid], 0) + 1
                request = NativeRequest(
                    self.step.conversation.agent if self.step.conversation else "",
                    str(batch.folder),
                    call,
                    child,
                    child_entries,
                    depths[tid],
                    parents[tid],
                )
                queue = self.queue
                ensure_worker = getattr(self.runner, "_native_workers", {}).get(queue)
                if ensure_worker is not None:
                    await ensure_worker(request.depth)
                await update(
                    batch.handle,
                    request,
                    id=f"claude-native-{request.agent}-{batch.digest}-{tid}",
                    accepted=True,
                )
                return call

            class Store(InMemorySessionStore):
                async def append(self, key: Any, entries: Any) -> None:
                    for entry in entries:
                        for completed_id in result_ids(entry):
                            early = read(batch.path("outcomes", completed_id))
                            if early and early.get("entries"):
                                state = early["entries"][0].get("temporalNotebookRead")
                                if state:
                                    entry["temporalNotebookRead"] = state
                    await super().append(key, entries)
                    if key.get("subpath"):
                        child = str(key["subpath"]).removeprefix("subagents/agent-")
                        loaded: Any = canonical_reads(
                            cast(
                                list[dict[str, Any]],
                                copy.deepcopy(await self.load(key) or []),
                            )
                        )
                        batch.children[child] = loaded
                        write(
                            batch.path("children", key["subpath"]), await self.load(key)
                        )
                    for entry in entries:
                        for block in blocks(entry):
                            tid = block.get("tool_use_id")
                            if (
                                block.get("type") == "tool_result"
                                and key.get("subpath")
                                and isinstance(tid, str)
                                and tid not in expected
                                and accepting
                            ):
                                # Native validation can reject a call before hooks run.
                                # Those calls get their own Activities as well.
                                await accept_child(
                                    str(key["subpath"]).removeprefix(
                                        "subagents/agent-"
                                    ),
                                    tid,
                                )
                            if block.get("type") != "tool_result" or (
                                tid not in ids and tid not in expected
                            ):
                                continue
                            if not key.get("subpath") and accepting and tid in expected:
                                await announce(expected[tid])
                            carrier = copy.deepcopy(entry)
                            carrier["message"]["content"] = [copy.deepcopy(block)]
                            early = read(batch.path("outcomes", tid))
                            if early and early.get("entries"):
                                state = early["entries"][0].get("temporalNotebookRead")
                                if state:
                                    carrier["temporalNotebookRead"] = state
                            content = block.get("content")
                            outcome = ToolOutcome(
                                content=content
                                if not isinstance(content, list)
                                else None,
                                blocks=content if isinstance(content, list) else None,
                                is_error=bool(block.get("is_error")),
                                entries=[carrier],
                                children=copy.deepcopy(batch.children)
                                if not key.get("subpath")
                                and expected.get(tid) is not None
                                and expected[tid].name == "Agent"
                                else {},
                            )
                            write(
                                batch.path("outcomes", tid), dataclasses.asdict(outcome)
                            )
                            if tid in recovering_calls:
                                recovering_calls.pop(tid).release()

            store: Any = Store()
            if seed:
                await store.append(key, resume_seed(seed))
            for child, child_entries in self.children.items():
                await store.append(
                    {**key, "subpath": "subagents/agent-" + child},
                    resume_seed(child_entries),
                )
            hook_dir = _hook_folder()
            replies: dict[str, list[ToolOutcome]] = {}

            def durable(spec: Any) -> Any:
                schema = dict(spec.input_schema)
                if schema.get("type") == "object":
                    schema.setdefault("properties", {})

                @tool(spec.name, spec.description, schema)
                async def answer(arguments: dict[str, Any]) -> dict[str, Any]:
                    del arguments
                    from ._runner import _result_content

                    queue = replies.get(spec.name, [])
                    if not queue:
                        raise RuntimeError(
                            "A durable tool ran before its Activity committed"
                        )
                    outcome = queue.pop(0)
                    content = _result_content(outcome)
                    return {
                        "content": content
                        if isinstance(content, list)
                        else [{"type": "text", "text": str(content)}],
                        "is_error": outcome.is_error,
                    }

                return answer

            owner_task = asyncio.current_task()

            async def watch_cancellation() -> None:
                while read(batch.folder / "abort") is None:
                    await asyncio.sleep(0.05)
                write(
                    batch.folder / "failure",
                    {"error": "A native Activity was cancelled"},
                )
                if owner_task is not None:
                    owner_task.cancel()

            async def decide_gate(data: Any, tid: str | None, context: Any) -> Any:
                del context
                call = expected.get(tid or "")
                if data.get("agent_id") and tid:
                    child = str(data["agent_id"])
                    call = await accept_child(child, tid)
                native_name = (
                    "mcp__durable__" + call.name
                    if call is not None and call.kind == "durable"
                    else call.name
                    if call is not None
                    else ""
                )
                if call is None or data["tool_name"] != native_name:
                    return {
                        "hookSpecificOutput": {
                            "hookEventName": "PreToolUse",
                            "permissionDecision": "deny",
                            "permissionDecisionReason": "Unaccepted native call",
                        }
                    }
                if not data.get("agent_id"):
                    await announce(call)
                if call.kind == "durable":
                    if batch.handle is None:
                        raise RuntimeError(
                            "Durable calls require a Workflow-owned runner"
                        )
                    request = NativeRequest(
                        self.step.conversation.agent if self.step.conversation else "",
                        str(batch.folder),
                        call,
                        str(data.get("agent_id") or ""),
                    )
                    outcome = await update(
                        batch.handle,
                        request,
                        id=f"claude-native-{request.agent}-{batch.digest}-{call.id}",
                    )
                    replies.setdefault(call.name, []).append(outcome)
                    return {
                        "hookSpecificOutput": {
                            "hookEventName": "PreToolUse",
                            "permissionDecision": "allow",
                        }
                    }
                last_query = 0.0
                while read(batch.path("allowed", call.id)) is None:
                    if batch.handle is not None and time.monotonic() - last_query >= 1:
                        last_query = time.monotonic()
                        outcome = await batch.handle.query(
                            "__claude_agent_native_outcome",
                            args=[
                                self.step.conversation.agent
                                if self.step.conversation
                                else "",
                                call.id,
                            ],
                            result_type=ToolOutcome | None,
                        )
                        if outcome is not None:
                            return {
                                "hookSpecificOutput": {
                                    "hookEventName": "PreToolUse",
                                    "permissionDecision": "deny",
                                    "permissionDecisionReason": str(outcome.content),
                                }
                            }
                    await asyncio.sleep(0.05)
                if read(batch.path("cancelled", call.id)) is not None:
                    return {
                        "hookSpecificOutput": {
                            "hookEventName": "PreToolUse",
                            "permissionDecision": "deny",
                            "permissionDecisionReason": "Activity cancelled",
                        }
                    }
                if call.name in ("AskUserQuestion", "ExitPlanMode"):
                    return {}
                if call.name == "Agent" and call.id in recovery_agents:
                    lock = recovering_locks.setdefault(
                        parents.get(call.id, "root"), asyncio.Lock()
                    )
                    await lock.acquire()
                    recovering_calls[call.id] = lock
                    child = recovery_agents[call.id]
                    model.active_children.append(child)
                    return {
                        "hookSpecificOutput": {
                            "hookEventName": "PreToolUse",
                            "permissionDecision": "allow",
                            "updatedInput": {**data["tool_input"], "resume": child},
                        }
                    }
                return {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "allow",
                    }
                }

            async def gate(data: Any, tid: str | None, context: Any) -> Any:
                try:
                    return await decide_gate(data, tid, context)
                except Exception as error:
                    # Claude Code treats a failing hook as advisory. Always send
                    # an explicit denial and stop this controller on bridge errors.
                    write(batch.folder / "failure", {"error": str(error)})
                    if owner_task is not None:
                        owner_task.cancel()
                    return {
                        "hookSpecificOutput": {
                            "hookEventName": "PreToolUse",
                            "permissionDecision": "deny",
                            "permissionDecisionReason": "Native Activity acceptance failed",
                        }
                    }

            async def permission(name: str, input: dict[str, Any], context: Any) -> Any:
                tid = context.tool_use_id
                if not tid or tid not in expected:
                    raise RuntimeError(
                        "Permission requested for an unaccepted native call"
                    )
                allowed = read(batch.path("allowed", tid)) or {}
                response = allowed.get("response")
                if name == "AskUserQuestion":
                    return PermissionResultAllow(
                        updated_input={**input, "answers": response}
                    )
                return PermissionResultAllow(updated_input=input)

            async def capture(data: Any, tid: str | None, context: Any) -> Any:
                del context
                call = expected.get(tid or "")
                if call is None or call.kind != "engine":
                    return {}
                # Child tool-result transcript rows can be held until every call
                # in the message finishes. Commit the native response at the
                # PostToolUse boundary so a later parked call cannot hold up this
                # Activity's completion. The eventual native carrier replaces
                # this recovery carrier before normal continuation.
                child = str(data.get("agent_id") or "")
                source_entries = self.children.get(child, []) if child else self.entries
                source = next(
                    (
                        e
                        for e in reversed(source_entries)
                        if any(b.get("id") == call.id for b in blocks(e))
                    ),
                    {},
                )
                response = data.get("tool_response", data.get("error", ""))
                is_error = data.get("hook_event_name") == "PostToolUseFailure"
                content = (
                    response
                    if isinstance(response, str)
                    else json.dumps(response, ensure_ascii=False)
                )
                carrier = {
                    **{
                        k: v
                        for k, v in source.items()
                        if k not in ("type", "message", "uuid", "requestId")
                    },
                    "type": "user",
                    "uuid": str(
                        uuid.uuid5(
                            uuid.NAMESPACE_URL, "claude-native-result:" + call.id
                        )
                    ),
                    "parentUuid": source.get("uuid"),
                    "sourceToolAssistantUUID": source.get("uuid"),
                    "toolUseResult": response,
                    "userType": "external",
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": call.id,
                                "content": content,
                                "is_error": is_error,
                            }
                        ],
                    },
                }
                if call.name == "Read" and not is_error:
                    state = notebook_read(response)
                    if state:
                        carrier["temporalNotebookRead"] = state
                if read(self.path("outcomes", call.id)) is None:
                    write(
                        self.path("outcomes", call.id),
                        dataclasses.asdict(
                            ToolOutcome(
                                content=content,
                                is_error=is_error,
                                entries=[carrier],
                                children=copy.deepcopy(self.children)
                                if call.name == "Agent"
                                else {},
                            )
                        ),
                    )
                return {}

            inp = SegmentInput(
                sid,
                "",
                self.step.tools,
                builtin_tools=self.step.builtin_tools,
                execution_protocol=2,
                tool_activities=["*"],
                model=self.step.model,
                system_prompt=self.step.system_prompt,
            )
            options = self.runner._engine_options(
                inp,
                {},
                sid,
                bool(seed),
                store,
                None,
                hook_dir,
                create_sdk_mcp_server(
                    "durable", tools=[durable(t) for t in self.step.tools]
                ),
                [],
            )
            options["max_turns"] = 1
            options["permission_mode"] = permission_mode(
                seed, options["permission_mode"]
            )
            state_options(
                self.runner, options, self.step.workspace_id or self.step.session_id
            )
            options["cli_path"] = self.runner._cli_path
            hooks = options.get("hooks") or {}
            options["hooks"] = {
                **hooks,
                "PreToolUse": [
                    HookMatcher(hooks=[gate], timeout=86400),
                    *(hooks.get("PreToolUse") or []),
                ],
                "PostToolUse": [
                    HookMatcher(hooks=[capture]),
                    *(hooks.get("PostToolUse") or []),
                ],
                "PostToolUseFailure": [
                    HookMatcher(hooks=[capture]),
                    *(hooks.get("PostToolUseFailure") or []),
                ],
            }
            options["can_use_tool"] = permission
            options["allowed_tools"] = []
            model = ReplayModel(
                [
                    b
                    for e in self.entries[stop:]
                    for b in blocks(e)
                    if e.get("type") == "assistant"
                    and e.get("message", {}).get("id") == message_id
                    and (b.get("type") != "tool_use" or b.get("id") in expected)
                ],
                {**os.environ, **options["env"]},
                Path(hook_dir),
            )
            model.recovery = recovery_blocks
            proxy_env = model.start()
            options["env"] = _command_env(
                options["env"],
                hook_dir,
                self.runner._cwd,
                proxy_env,
                {
                    "TCA_EXECUTE_BATCH": "1",
                    "TCA_BATCH_LEASE": str(batch.folder / "engine.lock"),
                },
            )
            lock = _hold_worker_lock(hook_dir)
            accepting = True
            cancellation = asyncio.create_task(watch_cancellation())
            try:
                engine = _engine_messages(
                    options,
                    await _as_messages(
                        "" if seed else "Execute the accepted tool calls."
                    ),
                    bool(seed),
                )
                try:
                    async for message in engine:
                        if isinstance(message, MirrorErrorMessage):
                            raise RuntimeError(message.error)
                        if (
                            isinstance(message, AssistantMessage)
                            and message.parent_tool_use_id
                        ):
                            for block in message.content:
                                if isinstance(block, ToolUseBlock):
                                    parents[block.id] = message.parent_tool_use_id
                        elif isinstance(message, ResultMessage):
                            write(
                                self.folder / "cost",
                                {"usd": float(message.total_cost_usd or 0)},
                            )
                except ResultError as error:
                    if error.subtype != "error_max_turns":
                        raise
                finally:
                    await engine.aclose()
                for call in pending:
                    if read(self.path("outcomes", call.id)) is None:
                        raise RuntimeError(f"Native batch lost the result of {call.id}")
                native = await store.load(key) or []
                boundary = next(
                    (
                        i
                        for i, e in enumerate(native)
                        if any(b.get("id") in expected for b in blocks(e))
                    ),
                    len(native),
                )
                sources = {}
                for e in native[boundary:]:
                    for b in blocks(e):
                        if b.get("type") == "tool_use" and b.get("id") in ids:
                            original = next(
                                (
                                    row
                                    for row in self.entries
                                    if any(u.get("id") == b["id"] for u in blocks(row))
                                ),
                                {},
                            )
                            sources[e.get("uuid")] = original.get("uuid")
                tail = []
                for e in native[boundary:]:
                    if e.get("type") not in (
                        "user",
                        "attachment",
                        "file-history-snapshot",
                    ) or result_ids(e):
                        continue
                    entry = copy.deepcopy(e)
                    source = entry.get("sourceToolAssistantUUID")
                    if source in sources:
                        entry["sourceToolAssistantUUID"] = sources[source]
                    tail.append(entry)
                write(self.folder / "tail", tail)
            finally:
                cancellation.cancel()
                await asyncio.gather(cancellation, return_exceptions=True)
                _release_worker_lock(lock)
                await model.close()
                shutil.rmtree(hook_dir, ignore_errors=True)
                if not seed:
                    self.runner._forget_local_copy(sid)
        except Exception as error:
            # Fail the owner task observably; the Activity supplies the retry policy.
            write(self.folder / "failure", {"error": str(error)})
        finally:
            _unlock_file(fd)
            os.close(fd)


async def execute(runner: Any, step: ToolStepInput, attempt: int) -> ToolOutcome:
    """Join the native controller and return only this Activity's result."""
    if step.controller is not None:
        folder = Path(step.controller)
        root = Path(runner._cwd or os.getcwd()) / ".temporal-claude"
        if (
            folder.parent != root
            or len(folder.name) != 64
            or not all(c in "0123456789abcdef" for c in folder.name)
        ):
            raise RuntimeError("Invalid native execution mailbox")
        digest = hashlib.sha256(step.call.id.encode()).hexdigest()
        write(
            folder / "allowed" / digest, {"id": step.call.id, "response": step.response}
        )
        (folder / "cancelled" / digest).unlink(missing_ok=True)
        try:
            while (saved := read(folder / "outcomes" / digest)) is None:
                await asyncio.sleep(0.05)
            return ToolOutcome(**saved)
        except asyncio.CancelledError:
            write(folder / "cancelled" / digest, {"id": step.call.id})
            write(folder / "abort", {"id": step.call.id})
            raise
    entries = await committed(runner, step)
    identity = (step.session_id, step.checkpoint)
    batches = runner._native_batches
    if identity not in batches:
        batches[identity] = NativeBatch(runner, step, entries)
    if step.control:
        return await batches[identity].control(attempt)
    return await batches[identity].result(step.call, step.response, attempt)
