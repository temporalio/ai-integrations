"""OTEL-aware variant of OpenAI Agents trace interceptor."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from typing import Any

import opentelemetry.trace

import temporalio.converter

from ._trace_interceptor import (
    OpenAIAgentsContextPropagationInterceptor,
    _InputWithHeaders,
)

_propagated_trace_id: ContextVar[str | None] = ContextVar(
    "temporal_openai_agents_propagated_trace_id", default=None
)


class OTelOpenAIAgentsContextPropagationInterceptor(
    OpenAIAgentsContextPropagationInterceptor
):
    """OTEL-aware variant that enhances headers with OpenTelemetry span context."""

    def __init__(
        self,
        payload_converter: temporalio.converter.PayloadConverter = temporalio.converter.default().payload_converter,
        add_temporal_spans: bool = True,
    ) -> None:
        """Initialize OTEL-aware context propagation interceptor.

        Args:
            payload_converter: Converter for serializing trace context.
            add_temporal_spans: Whether to add Temporal-specific spans.
        """
        super().__init__(
            payload_converter=payload_converter,
            add_temporal_spans=add_temporal_spans,
        )

    def header_contents(self) -> dict[str, Any]:
        """Get header contents enhanced with OpenTelemetry span context.

        Returns:
            Dictionary containing trace context with OTEL span information.
        """
        otel_span = opentelemetry.trace.get_current_span()

        if otel_span and otel_span.get_span_context().is_valid:
            span_context = otel_span.get_span_context()
            return {
                **super().header_contents(),
                "otelSpanId": span_context.span_id,
                "otelTraceId": span_context.trace_id,
                "otelTraceFlags": int(span_context.trace_flags),
                "otelTraceState": span_context.trace_state.to_header(),
            }
        else:
            return super().header_contents()

    @contextmanager
    def context_from_header(
        self,
        input: _InputWithHeaders,
    ) -> Generator[None, None, None]:
        """Use propagated IDs as a remote parent without recording replicas."""
        span_info = self.get_header_contents(input)
        # Native OTel callers need not provide Agents trace/span IDs. Only
        # reconstruct Agents context when its trace ID is actually available.
        with (
            super().context_from_header(input=input)
            if span_info is not None and span_info.get("traceId")
            else nullcontext()
        ):
            if (
                span_info is not None
                and span_info.get("otelSpanId")
                and span_info.get("otelTraceId")
            ):
                span_context: opentelemetry.trace.SpanContext = (
                    opentelemetry.trace.SpanContext(
                        trace_id=span_info["otelTraceId"],
                        span_id=span_info["otelSpanId"],
                        is_remote=True,
                        trace_flags=opentelemetry.trace.TraceFlags(
                            span_info.get(
                                "otelTraceFlags", opentelemetry.trace.TraceFlags.SAMPLED
                            )
                        ),
                        trace_state=opentelemetry.trace.TraceState.from_header(
                            [span_info["otelTraceState"]]
                            if span_info.get("otelTraceState")
                            else []
                        ),
                    )
                )
                with opentelemetry.trace.use_span(
                    opentelemetry.trace.NonRecordingSpan(span_context),
                    end_on_exit=False,
                ):
                    token = _propagated_trace_id.set(span_info.get("traceId"))
                    try:
                        yield
                    finally:
                        _propagated_trace_id.reset(token)
            else:
                yield
