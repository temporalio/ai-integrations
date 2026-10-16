"""Turn spans record GenAI attributes when a tracer is present, and do nothing without one."""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
from claude_agent_sdk import (
    AssistantMessage,
    ResultMessage,
    StreamEvent,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)
from opentelemetry.context import Context
from opentelemetry.trace import (
    Link,
    Span,
    SpanContext,
    SpanKind,
    Status,
    StatusCode,
    TraceFlags,
    Tracer,
    TraceState,
    format_span_id,
    format_trace_id,
    get_current_span,
    use_span,
)
from opentelemetry.util._decorator import _agnosticcontextmanager
from opentelemetry.util.types import Attributes

from temporalio.claude_agent_sdk._activity import (
    NONESSENTIAL_TRAFFIC_ENV,
    ClaudeTurnActivity,
)
from temporalio.claude_agent_sdk._models import (
    ClaudeToolDeclaration,
    ClaudeTurnInput,
    ClaudeTurnOptions,
)
from temporalio.claude_agent_sdk._tracing import cli_trace_env, turn_span

# trace-id from the W3C traceparent example.
_TRACE_ID = 0x4BF92F3577B34DA6A3CE929D0E0E4736
_TRACE_STATE = TraceState([("vendor", "one")])
_CAPTURE = "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"
_CONTENT_ATTRS = (
    "gen_ai.input.messages",
    "gen_ai.system_instructions",
    "gen_ai.output.messages",
    "gen_ai.tool.call.arguments",
    "gen_ai.tool.call.result",
)


class _Span(Span):
    """In-memory span. ``set_attributes`` always stores, so a non-recording span that is still
    written to fails the test.
    """

    def __init__(
        self,
        name: str,
        kind: SpanKind,
        context: SpanContext,
        parent_span_id: int | None,
        *,
        recording: bool,
    ) -> None:
        self.name = name
        self.kind = kind
        self._context = context
        self.parent_span_id = parent_span_id
        self.attributes: dict[str, Any] = {}
        self.ended = False
        self._recording = recording

    def end(self, end_time: int | None = None) -> None:
        self.ended = True

    def get_span_context(self) -> SpanContext:
        return self._context

    def set_attributes(self, attributes: Mapping[str, Any]) -> None:
        self.attributes.update(attributes)

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def add_event(
        self,
        name: str,
        attributes: Attributes = None,
        timestamp: int | None = None,
    ) -> None:
        return None

    def update_name(self, name: str) -> None:
        self.name = name

    def is_recording(self) -> bool:
        return self._recording

    def set_status(
        self, status: Status | StatusCode, description: str | None = None
    ) -> None:
        return None

    def record_exception(
        self,
        exception: BaseException,
        attributes: Attributes = None,
        timestamp: int | None = None,
        escaped: bool = False,
    ) -> None:
        return None


class _Tracer(Tracer):
    """Records every span ``turn_span`` opens. No exporter and no SDK."""

    def __init__(self, *, recording: bool) -> None:
        self.recording = recording
        self.spans: list[_Span] = []
        self._next_span_id = 1

    def start_span(
        self,
        name: str,
        context: Context | None = None,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Attributes = None,
        links: Sequence[Link] | None = None,
        start_time: int | None = None,
        record_exception: bool = True,
        set_status_on_exception: bool = True,
    ) -> _Span:
        del links, start_time, record_exception, set_status_on_exception
        parent = get_current_span(context).get_span_context()
        span_id = self._next_span_id
        self._next_span_id += 1
        span = _Span(
            name,
            kind,
            SpanContext(
                parent.trace_id if parent.is_valid else _TRACE_ID,
                span_id,
                False,
                TraceFlags(TraceFlags.SAMPLED),
                parent.trace_state if parent.is_valid else _TRACE_STATE,
            ),
            parent.span_id if parent.is_valid else None,
            recording=self.recording,
        )
        if attributes:
            span.set_attributes(attributes)
        self.spans.append(span)
        return span

    @_agnosticcontextmanager
    def start_as_current_span(
        self,
        name: str,
        context: Context | None = None,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Attributes = None,
        links: Sequence[Link] | None = None,
        start_time: int | None = None,
        record_exception: bool = True,
        set_status_on_exception: bool = True,
        end_on_exit: bool = True,
    ) -> Iterator[Span]:
        started: Span = self.start_span(
            name,
            context=context,
            kind=kind,
            attributes=attributes,
            links=links,
            start_time=start_time,
            record_exception=record_exception,
            set_status_on_exception=set_status_on_exception,
        )
        with use_span(
            started,
            end_on_exit=end_on_exit,
            record_exception=record_exception,
            set_status_on_exception=set_status_on_exception,
        ) as span:
            yield span


def _install(monkeypatch: pytest.MonkeyPatch, *, recording: bool = True) -> _Tracer:
    tracer = _Tracer(recording=recording)
    monkeypatch.setattr(
        "temporalio.claude_agent_sdk._tracing.trace.get_tracer",
        lambda *_a, **_k: tracer,
    )
    return tracer


def _turn() -> ClaudeTurnInput:
    return ClaudeTurnInput(
        prompt="Weather in Paris?",
        session_id="sess-1",
        resume=True,
        resumes_lost_attempt=True,
        attempt=2,
        options=ClaudeTurnOptions(model="claude-test"),
        tools=[ClaudeToolDeclaration(name="get_weather")],
    )


def _stream(event: dict[str, Any]) -> StreamEvent:
    return StreamEvent(uuid="u", session_id="sess-1", event=event)


def test_turn_span_is_a_noop_without_a_tracer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("temporalio.claude_agent_sdk._tracing.trace", None)
    assert cli_trace_env() == {}
    with turn_span(ClaudeTurnInput(prompt="hi")) as span:
        assert span._span is None
        span.on_message(None)
        span.close()


def test_turn_span_records_genai_spans_and_cli_trace_context(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(NONESSENTIAL_TRAFFIC_ENV, raising=False)
    tracer = _install(monkeypatch)
    turn = _turn()

    with turn_span(turn) as traced:
        traced.on_message(
            _stream(
                {
                    "type": "message_start",
                    "message": {"model": "claude-test", "id": "msg_1"},
                }
            )
        )
        traced.on_message(
            AssistantMessage(
                content=[
                    ToolUseBlock(
                        id="toolu_1",
                        name="mcp__temporal__get_weather",
                        input={"city": "Paris"},
                    )
                ],
                model="claude-test",
            )
        )
        traced.on_message(
            UserMessage(
                content=[ToolResultBlock(tool_use_id="toolu_1", content="sunny")]
            )
        )
        traced.on_message(_stream({"type": "message_stop"}))
        env = ClaudeTurnActivity(workspace_root=tmp_path)._options(turn, None, None).env
        invoke, chat, tool = tracer.spans
        context = invoke.get_span_context()
        # Options do not reveal Bedrock, Vertex, or Foundry, so the provider stays anthropic.
        assert invoke.attributes == {
            "gen_ai.provider.name": "anthropic",
            "gen_ai.system": "anthropic",
            "gen_ai.operation.name": "invoke_agent",
            "gen_ai.agent.name": "Claude Code",
            "gen_ai.request.model": "claude-test",
            "gen_ai.conversation.id": "sess-1",
            "gen_ai.output.type": "text",
            "claude.resume": True,
            "claude.resumes_lost_attempt": True,
            "claude.resumes_api_error": False,
            "claude.attempt": 2,
            "claude.tools": 1,
        }
        assert invoke.name == "invoke_agent Claude Code"
        assert invoke.kind is SpanKind.INTERNAL
        assert chat.name == "chat claude-test" and chat.kind is SpanKind.CLIENT
        assert chat.parent_span_id == context.span_id
        assert chat.attributes["gen_ai.operation.name"] == "chat"
        assert chat.attributes["gen_ai.response.id"] == "msg_1"
        assert chat.ended
        assert tool.name == "execute_tool get_weather"
        assert tool.kind is SpanKind.INTERNAL
        assert tool.parent_span_id == context.span_id
        assert tool.attributes == {
            "gen_ai.provider.name": "anthropic",
            "gen_ai.system": "anthropic",
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.tool.name": "get_weather",
            "gen_ai.tool.call.id": "toolu_1",
            "gen_ai.tool.type": "function",
        }
        assert tool.ended
        assert env is not None
        assert env["TRACEPARENT"] == (
            f"00-{format_trace_id(context.trace_id)}-"
            f"{format_span_id(context.span_id)}-{int(context.trace_flags):02x}"
        )
        assert env["TRACESTATE"] == "vendor=one"
        assert env[NONESSENTIAL_TRAFFIC_ENV] == "1"

    assert invoke.ended


def _drive_captured_turn(traced: Any) -> None:
    traced.on_message(
        _stream(
            {
                "type": "message_start",
                "message": {"model": "claude-test", "id": "msg_1"},
            }
        )
    )
    traced.on_message(
        AssistantMessage(
            content=[
                ToolUseBlock(
                    id="toolu_1",
                    name="mcp__temporal__get_weather",
                    input={"city": "Paris"},
                )
            ],
            model="claude-test",
        )
    )
    traced.on_message(
        UserMessage(content=[ToolResultBlock(tool_use_id="toolu_1", content="sunny")])
    )
    traced.on_message(_stream({"type": "message_stop"}))
    traced.on_message(
        ResultMessage(
            subtype="success",
            duration_ms=1,
            duration_api_ms=1,
            is_error=False,
            num_turns=1,
            session_id="sess-1",
            result="It is sunny.",
        )
    )


def test_capture_message_content_records_payloads_and_stays_off_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    turn = ClaudeTurnInput(
        prompt="Weather in Paris?",
        session_id="sess-1",
        options=ClaudeTurnOptions(model="claude-test", system_prompt="Be brief."),
        tools=[ClaudeToolDeclaration(name="get_weather")],
    )

    monkeypatch.delenv(_CAPTURE, raising=False)
    tracer = _install(monkeypatch)
    with turn_span(turn) as traced:
        _drive_captured_turn(traced)
    for span in tracer.spans:
        for key in _CONTENT_ATTRS:
            assert key not in span.attributes

    monkeypatch.setenv(_CAPTURE, "1")
    tracer = _install(monkeypatch)
    with turn_span(turn) as traced:
        _drive_captured_turn(traced)
    invoke, _chat, tool = tracer.spans
    assert invoke.attributes["gen_ai.input.messages"] == (
        '[{"role": "user", "parts": [{"type": "text", "content": "Weather in Paris?"}]}]'
    )
    assert invoke.attributes["gen_ai.system_instructions"] == (
        '[{"type": "text", "content": "Be brief."}]'
    )
    assert invoke.attributes["gen_ai.output.messages"] == (
        '[{"role": "assistant", "parts": [{"type": "text", "content": "It is sunny."}]}]'
    )
    assert tool.attributes["gen_ai.tool.call.arguments"] == '{"city": "Paris"}'
    assert tool.attributes["gen_ai.tool.call.result"] == "sunny"
    for key in _CONTENT_ATTRS:
        assert key not in _chat.attributes


def test_turn_span_ignores_a_non_recording_span(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracer = _install(monkeypatch, recording=False)
    with turn_span(_turn()) as traced:
        assert traced._span is None
        traced.on_message(_stream({"type": "message_start", "message": {"id": "m"}}))
        traced.failed(RuntimeError("down"))
        traced.close()
    assert len(tracer.spans) == 1
    assert tracer.spans[0].attributes == {}


def test_a_failing_span_write_does_not_fail_the_turn(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    tracer = _install(monkeypatch)
    with turn_span(_turn()) as traced:
        traced.on_message(
            _stream({"type": "message_start", "message": {"model": "claude-test"}})
        )

        def fail(attributes: Mapping[str, Any]) -> None:
            del attributes
            raise RuntimeError("exporter failed")

        tracer.spans[-1].set_attributes = fail  # type: ignore[method-assign]
        with caplog.at_level(
            logging.WARNING, logger="temporalio.claude_agent_sdk._tracing"
        ):
            traced.on_message(
                _stream(
                    {
                        "type": "message_delta",
                        "delta": {"stop_reason": "end_turn"},
                        "usage": {"output_tokens": 1},
                    }
                )
            )
    assert "Could not trace a Claude message" in caplog.text
