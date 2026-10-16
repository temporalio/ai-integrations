"""OpenTelemetry spans for one Claude turn.

An ``invoke_agent`` span wraps the Activity, with a ``chat`` span per model
round and an ``execute_tool`` span per tool call. With no tracer, every hook
is a no-op. Prompts, replies, and tool payloads are omitted unless
``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`` is set. Errors record
only their type.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from temporalio.claude_agent_sdk._models import (
    ClaudeTurnInput,
    anthropic_usage,
    mcp_tool_name,
)
from temporalio.claude_agent_sdk._sdk import load_sdk
from temporalio.exceptions import ApplicationError

try:
    from opentelemetry import propagate, trace
except ImportError:  # tracing is optional
    trace = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

CAPTURE_CONTENT_ENV = "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"
AGENT_NAME = "Claude Code"
_PROVIDER = {"gen_ai.provider.name": "anthropic", "gen_ai.system": "anthropic"}


@contextmanager
def turn_span(input: ClaudeTurnInput) -> Iterator[TurnTrace]:
    """The ``invoke_agent`` span of one turn Activity, current for the whole turn."""
    if trace is None:
        yield TurnTrace(None, None, input)
        return
    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span(
        f"invoke_agent {AGENT_NAME}",
        kind=trace.SpanKind.INTERNAL,
        record_exception=False,
        set_status_on_exception=False,
    ) as span:
        turn = TurnTrace(tracer, span if span.is_recording() else None, input)
        try:
            yield turn
        except BaseException as exc:
            turn.failed(exc)
            raise
        finally:
            turn.close()


def cli_trace_env() -> dict[str, str]:
    """``TRACEPARENT``/``TRACESTATE`` of the current span, which Claude Code reads in Agent SDK
    sessions to parent its own spans and events (when its telemetry is enabled); empty without
    one. https://code.claude.com/docs/en/monitoring-usage#traces-beta
    """
    if trace is None or not trace.get_current_span().get_span_context().is_valid:
        return {}
    carrier: dict[str, str] = {}
    propagate.inject(carrier)  # type: ignore
    return {
        k.upper(): v for k, v in carrier.items() if k in ("traceparent", "tracestate")
    }


class TurnTrace:
    """Child spans and result attributes of one turn, fed every SDK message. Inert when the
    turn's span does not record.
    """

    def __init__(self, tracer: Any, span: Any, input: ClaudeTurnInput) -> None:
        """Open the turn span when ``span`` records."""
        self._tracer = tracer
        self._span = span
        self._input = input
        self._capture = span is not None and _truthy(
            os.environ.get(CAPTURE_CONTENT_ENV)
        )
        self._round: Any = None
        self._tools: dict[str, Any] = {}
        self._native_used: set[str] = set()
        self._model: str | None = None
        if span is None:
            return
        options = input.options
        _set(
            span,
            {
                **_PROVIDER,
                "gen_ai.operation.name": "invoke_agent",
                "gen_ai.agent.name": AGENT_NAME,
                "gen_ai.request.model": options.model,
                "gen_ai.conversation.id": input.session_id,
                "gen_ai.output.type": "json"
                if options.output_schema is not None
                else "text",
                "claude.resume": input.resume,
                "claude.resumes_lost_attempt": input.resumes_lost_attempt,
                "claude.resumes_api_error": input.resumes_api_error,
                "claude.attempt": input.attempt,
                "claude.tools": len(input.tools),
                "claude.native_tools.requested": list(input.native_tools) or None,
            },
        )
        if self._capture:
            _set(
                span,
                {
                    "gen_ai.input.messages": _messages("user", input.prompt),
                    "gen_ai.system_instructions": options.system_prompt
                    and json.dumps(
                        [{"type": "text", "content": options.system_prompt}]
                    ),
                },
            )

    def on_message(self, message: Any) -> None:
        """Update child spans from one SDK message. Errors are logged, not raised."""
        if (
            self._span is None
            or getattr(message, "parent_tool_use_id", None) is not None
        ):
            return  # Claude's internal subagents are not traced, as they are not streamed
        try:
            self._observe(message)
        except Exception:  # noqa: BLE001 - telemetry must never fail a turn
            logger.warning("Could not trace a Claude message", exc_info=True)

    def _observe(self, message: Any) -> None:
        sdk = load_sdk()
        if isinstance(message, sdk.StreamEvent):
            self._on_stream(message.event)
        elif isinstance(message, sdk.AssistantMessage):
            for block in message.content:
                if isinstance(block, sdk.ToolUseBlock):
                    self._tool_started(block)
        elif isinstance(message, sdk.UserMessage) and isinstance(message.content, list):
            for block in message.content:
                if isinstance(block, sdk.ToolResultBlock):
                    self._tool_finished(block)
        elif isinstance(message, sdk.ResultMessage):
            self._on_result(message)

    def failed(self, exc: BaseException) -> None:
        """Mark the turn span failed, recording only the error type."""
        if self._span is None:
            return
        error = (exc.type if isinstance(exc, ApplicationError) else None) or type(
            exc
        ).__name__
        self._span.set_attribute("error.type", error)
        self._span.set_status(trace.StatusCode.ERROR, error)  # type: ignore

    def close(self) -> None:
        """End what the run left open (a deferred or cancelled call, a cut-off round)."""
        if self._span is None:
            return
        for span in [self._round, *self._tools.values()]:
            if span is not None:
                span.end()
        self._round = None
        self._tools.clear()
        _set(
            self._span, {"claude.native_tools.used": sorted(self._native_used) or None}
        )

    def _child(self, name: str, kind: Any, attributes: dict[str, Any]) -> Any:
        span = self._tracer.start_span(
            name,
            context=trace.set_span_in_context(self._span),  # type: ignore
            kind=kind,
        )
        _set(span, {**_PROVIDER, **attributes})
        return span

    def _on_stream(self, event: dict[str, Any]) -> None:
        kind = event.get("type")
        if kind == "message_start":
            message = event.get("message") or {}
            self._model = message.get("model") or self._model
            if self._round is not None:
                self._round.end()
            request_model = self._input.options.model
            self._round = self._child(
                f"chat {request_model or self._model}",
                trace.SpanKind.CLIENT,  # type: ignore
                {
                    "gen_ai.operation.name": "chat",
                    "gen_ai.request.model": request_model,
                    "gen_ai.response.model": message.get("model"),
                    "gen_ai.response.id": message.get("id"),
                },
            )
        elif kind == "message_delta" and self._round is not None:
            stop = (event.get("delta") or {}).get("stop_reason")
            _set(
                self._round,
                {
                    **_usage_attributes(event.get("usage")),
                    "gen_ai.response.finish_reasons": [stop] if stop else None,
                },
            )
        elif kind == "message_stop" and self._round is not None:
            self._round.end()
            self._round = None

    def _tool_started(self, block: Any) -> None:
        prefix = mcp_tool_name("")
        bridged = block.name.startswith(prefix)
        name = block.name.removeprefix(prefix)
        if block.name in self._input.native_tools:
            self._native_used.add(block.name)
        self._tools[block.id] = self._child(
            f"execute_tool {name}",
            trace.SpanKind.INTERNAL,  # type: ignore
            {
                "gen_ai.operation.name": "execute_tool",
                "gen_ai.tool.name": name,
                "gen_ai.tool.call.id": block.id,
                "gen_ai.tool.type": "function" if bridged else "extension",
                "gen_ai.tool.call.arguments": json.dumps(block.input)
                if self._capture
                else None,
            },
        )

    def _tool_finished(self, block: Any) -> None:
        span = self._tools.pop(block.tool_use_id, None)
        if span is None:
            return
        if self._capture:
            content = block.content
            span.set_attribute(
                "gen_ai.tool.call.result",
                content if isinstance(content, str) else json.dumps(content),
            )
        if block.is_error:
            span.set_attribute("error.type", "tool_error")
            span.set_status(trace.StatusCode.ERROR, "tool_error")  # type: ignore
        span.end()

    def _on_result(self, result: Any) -> None:
        stop = result.stop_reason
        _set(
            self._span,
            {
                **_usage_attributes(result.usage),
                "gen_ai.response.model": self._model,
                "gen_ai.response.finish_reasons": [stop] if stop else None,
                "gen_ai.conversation.id": result.session_id,
                "claude.session_id": result.session_id,
                "claude.models_used": sorted(result.model_usage or {}) or None,
                "claude.total_cost_usd": result.total_cost_usd,
                "claude.num_turns": result.num_turns,
                "claude.duration_api_ms": result.duration_api_ms,
                "claude.terminal_reason": result.terminal_reason,
                "claude.result_subtype": result.subtype,
                "claude.api_error_status": result.api_error_status,
                "claude.deferred_tool": result.deferred_tool_use.name
                if result.deferred_tool_use is not None
                else None,
            },
        )
        if self._capture and result.result is not None:
            self._span.set_attribute(
                "gen_ai.output.messages", _messages("assistant", result.result)
            )


def _usage_attributes(usage: dict[str, Any] | None) -> dict[str, Any]:
    """Anthropic usage as GenAI usage attributes: input tokens include cache reads and writes,
    as the Anthropic conventions require, and thinking tokens are a slice of output tokens.
    """
    mapped = anthropic_usage(usage)
    if usage is None or mapped is None:
        return {}
    return {
        "gen_ai.usage.input_tokens": mapped.input_tokens,
        "gen_ai.usage.output_tokens": mapped.output_tokens,
        "gen_ai.usage.cache_read.input_tokens": usage.get("cache_read_input_tokens"),
        "gen_ai.usage.cache_write.input_tokens": usage.get(
            "cache_creation_input_tokens"
        ),
        "gen_ai.usage.reasoning.output_tokens": mapped.thought_tokens,
    }


def _messages(role: str, text: str) -> str:
    return json.dumps([{"role": role, "parts": [{"type": "text", "content": text}]}])


def _set(span: Any, attributes: dict[str, Any]) -> None:
    span.set_attributes({k: v for k, v in attributes.items() if v is not None})


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in ("1", "true", "yes", "on")
