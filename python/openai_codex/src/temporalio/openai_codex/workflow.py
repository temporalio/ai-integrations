"""Workflow-side API: host tools and the :class:`CodexSession` turn driver.

Every tool the model can call is a *host tool*: Codex's own built-in tools (shell, ``apply_patch``,
goals, ...) are turned off for the thread, and the tools you pass to :class:`CodexSession` are
offered to Codex as dynamic tools. A call runs in your Workflow (or as an Activity, with
:func:`activity_as_tool`), so it is recorded in history, retried by Temporal and, when it has side
effects, run exactly once even if the Worker dies mid-turn.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ApplicationError, CancelledError

with workflow.unsafe.imports_passed_through():
    from pydantic import BaseModel, create_model
    from pydantic_core import to_jsonable_python

from ._models import (
    CODEX_RUN_SEGMENT_ACTIVITY,
    CodexPendingCall,
    CodexSegmentInput,
    CodexSegmentResult,
    CodexTokenUsage,
    CodexToolSpec,
    CodexTurnResult,
)

__all__ = [
    "CodexPendingCall",
    "CodexSession",
    "CodexTool",
    "CodexToolSpec",
    "CodexTurnResult",
    "FunctionSchema",
    "activity_as_tool",
    "codex_tool",
    "function_schema",
]


@dataclass(frozen=True)
class CodexTool:
    """A host tool: how Codex sees it (``spec``) and what runs when the model calls it.

    ``handler`` receives the call (its ``call_id`` is stable and unique per model call, so it makes
    a good idempotency key) and returns the tool's output as text. Use :func:`codex_tool` or
    :func:`activity_as_tool` to build one from a function; construct one directly to integrate
    another execution layer.
    """

    spec: CodexToolSpec
    handler: Callable[[CodexPendingCall], Awaitable[str]]


def _stringify(result: Any) -> str:
    if isinstance(result, str):
        return result
    return json.dumps(to_jsonable_python(result))


@dataclass(frozen=True)
class FunctionSchema:
    """A function's tool spec plus a parser for the arguments the model sends for it."""

    spec: CodexToolSpec
    _args_model: type[BaseModel]
    _order: tuple[str, ...]

    def parse(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Validate the model's call arguments and return them as keyword arguments.

        Values are the function's own annotated types (for example Pydantic models for nested
        objects), in the function's parameter order.

        Raises:
            pydantic.ValidationError: If the arguments do not match the function's signature.
        """
        parsed = self._args_model.model_validate(arguments)
        return {name: getattr(parsed, name) for name in self._order}


def function_schema(
    fn: Callable[..., Any],
    *,
    name: str | None = None,
    description: str | None = None,
) -> FunctionSchema:
    """Derive a tool spec (JSON schema from the signature, description from the docstring) from ``fn``.

    This is what :func:`codex_tool` and :func:`activity_as_tool` use; call it directly to build a
    :class:`CodexTool` that executes the function some other way.

    Args:
        fn: The function (sync or async) whose signature describes the tool.
        name: Tool name shown to the model. Defaults to the function's name.
        description: Tool description. Defaults to the function's docstring.
    """
    tool_name = name or getattr(fn, "__name__", None) or "tool"
    try:
        signature = inspect.signature(fn, eval_str=True)
    except Exception:  # noqa: BLE001 - fall back to the unevaluated signature
        signature = inspect.signature(fn)
    fields: dict[str, Any] = {}
    order: list[str] = []
    for param in signature.parameters.values():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        annotation = (
            Any if param.annotation is inspect.Parameter.empty else param.annotation
        )
        default = ... if param.default is inspect.Parameter.empty else param.default
        fields[param.name] = (annotation, default)
        order.append(param.name)
    args_model = create_model(f"{tool_name}_args", **fields)
    schema = args_model.model_json_schema()
    schema.pop("title", None)
    spec = CodexToolSpec(
        name=tool_name,
        description=(description or inspect.getdoc(fn) or "").strip(),
        input_schema=schema,
    )
    return FunctionSchema(spec, args_model, tuple(order))


def codex_tool(
    fn: Callable[..., Awaitable[Any]],
    *,
    name: str | None = None,
    description: str | None = None,
) -> CodexTool:
    """Turn an ``async`` function into a host tool that runs in the Workflow.

    The model-facing JSON schema comes from the function's signature and the description from its
    docstring. Use this for deterministic tools; for anything with side effects use
    :func:`activity_as_tool`. The result is JSON-encoded unless it is already a string.

    Args:
        fn: The function the model's call arguments are passed to as keyword arguments.
        name: Tool name shown to the model. Defaults to the function's name.
        description: Tool description. Defaults to the function's docstring.
    """
    schema = function_schema(fn, name=name, description=description)

    async def handler(call: CodexPendingCall) -> str:
        return _stringify(await fn(**schema.parse(call.arguments)))

    return CodexTool(spec=schema.spec, handler=handler)


def activity_as_tool(
    activity_fn: Callable[..., Awaitable[Any]],
    *,
    name: str | None = None,
    description: str | None = None,
    start_to_close_timeout: timedelta = timedelta(minutes=1),
    retry_policy: RetryPolicy | None = None,
    task_queue: str | None = None,
) -> CodexTool:
    """Turn a Temporal Activity into a host tool, so each call is a durable, retried Activity.

    The Activity runs with ``activity_id=f"tool-{call_id}"``, which is stable for a given model call
    and therefore usable as an idempotency key for external side effects.

    Args:
        activity_fn: A function decorated with ``@activity.defn``.
        name: Tool name shown to the model. Defaults to the Activity function's name.
        description: Tool description. Defaults to the function's docstring.
        start_to_close_timeout: Activity start-to-close timeout.
        retry_policy: Activity retry policy.
        task_queue: Task queue to run the Activity on; defaults to the Workflow's.
    """
    schema = function_schema(activity_fn, name=name, description=description)

    async def handler(call: CodexPendingCall) -> str:
        result = await workflow.execute_activity(
            activity_fn,
            args=list(schema.parse(call.arguments).values()),
            activity_id=f"tool-{call.call_id}",
            start_to_close_timeout=start_to_close_timeout,
            retry_policy=retry_policy,
            task_queue=task_queue,
        )
        return _stringify(result)

    return CodexTool(spec=schema.spec, handler=handler)


class CodexSession:
    """One Codex thread, owned by a Workflow. Keep it on the Workflow instance.

    Holds the conversation (thread id plus the committed rollout) so it survives Worker loss, and
    drives each turn: segment Activity, then host tool, then the next segment, until Codex answers.
    Its attributes are plain data (:attr:`thread_id`, :attr:`rollout_name`, :attr:`rollout`,
    :attr:`tool_results`), so you can carry them across Continue-As-New yourself.
    """

    def __init__(
        self,
        *,
        tools: Sequence[CodexTool] = (),
        instructions: str | None = None,
        model: str | None = None,
        cwd: str | None = None,
        sandbox: str = "read-only",
        continue_prompt: str = "Continue.",
        max_segments: int = 50,
        segment_timeout: timedelta = timedelta(minutes=5),
        heartbeat_timeout: timedelta = timedelta(seconds=30),
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        """Create a session.

        Args:
            tools: Host tools the model may call.
            instructions: Developer instructions for the thread (applied when it starts).
            model: Model id to request for the thread; defaults to Codex's own default.
            cwd: Working directory for the app-server; defaults to a scratch directory.
            sandbox: Codex sandbox preset for the thread. Built-in tools are off, so this only
                matters if you re-enable some.
            continue_prompt: The user message that resumes the turn after each tool call.
            max_segments: Safety bound on tool round trips in one turn.
            segment_timeout: Start-to-close timeout of each segment Activity.
            heartbeat_timeout: Heartbeat timeout of each segment Activity.
            retry_policy: Retry policy of each segment Activity. Defaults to 3 attempts.
        """
        self._tools: dict[str, CodexTool] = {t.spec.name: t for t in tools}
        self._instructions = instructions
        self._model = model
        self._cwd = cwd
        self._sandbox = sandbox
        self._continue_prompt = continue_prompt
        self._max_segments = max_segments
        self._segment_timeout = segment_timeout
        self._heartbeat_timeout = heartbeat_timeout
        self._retry_policy = retry_policy or RetryPolicy(
            maximum_attempts=3, initial_interval=timedelta(seconds=1)
        )
        self.thread_id: str | None = None
        """The Codex thread id, once the first segment has started it."""
        self.rollout_name: str | None = None
        """File name of the thread's rollout."""
        self.rollout: str = ""
        """The committed rollout (Codex's append-only JSONL conversation log)."""
        self.tool_results: dict[str, str] = {}
        """Output of every host tool call so far, by call id."""

    async def run(
        self, prompt: str, *, observer_context: dict[str, Any] | None = None
    ) -> CodexTurnResult:
        """Run one turn: ``prompt`` in, Codex's final answer out.

        Args:
            prompt: The user message.
            observer_context: Opaque JSON handed to the Worker's observer factory (see
                :class:`~temporalio.openai_codex.CodexPlugin`) so live events can be routed.
        """
        text = prompt
        inject: dict[str, str] = {}
        usage: CodexTokenUsage | None = None
        for segments in range(1, self._max_segments + 1):
            seg: CodexSegmentResult = await workflow.execute_activity(
                CODEX_RUN_SEGMENT_ACTIVITY,
                CodexSegmentInput(
                    prompt=text,
                    thread_id=self.thread_id,
                    rollout_name=self.rollout_name,
                    rollout=self.rollout,
                    committed_lines=len(self.rollout.splitlines()),
                    inject=inject,
                    tools=[t.spec for t in self._tools.values()],
                    instructions=self._instructions,
                    model=self._model,
                    cwd=self._cwd,
                    sandbox=self._sandbox,
                    observer_context=observer_context,
                ),
                result_type=CodexSegmentResult,
                start_to_close_timeout=self._segment_timeout,
                heartbeat_timeout=self._heartbeat_timeout,
                retry_policy=self._retry_policy,
            )
            self.thread_id, self.rollout_name = seg.thread_id, seg.rollout_name
            self.rollout += seg.tail
            usage = seg.usage or usage
            if seg.status == "done" or seg.call is None:
                return CodexTurnResult(
                    text=seg.final_response, usage=usage, segments=segments
                )
            output = await self._run_host_tool(seg.call)
            self.tool_results[seg.call.call_id] = output
            text, inject = self._continue_prompt, {seg.call.call_id: output}
        raise ApplicationError(
            f"Codex turn exceeded max_segments={self._max_segments} tool round trips",
            non_retryable=True,
        )

    async def _run_host_tool(self, call: CodexPendingCall) -> str:
        tool = self._tools.get(call.tool)
        if tool is None:
            return f"Tool {call.tool!r} is not available."
        try:
            return await tool.handler(call)
        except (asyncio.CancelledError, CancelledError):
            raise
        except Exception as exc:  # noqa: BLE001 - a failed tool is a result the model sees
            return f"Tool {call.tool!r} failed: {exc}"
