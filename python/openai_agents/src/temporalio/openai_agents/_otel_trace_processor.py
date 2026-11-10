"""Adapt OpenInference processors to Temporal's propagated trace context.

Imported only when OpenInference instrumentation is enabled. Callbacks belong to
each processor instance so draining processors retain their context cleanup even
after instrumentation is removed or another processor is installed.
"""

from __future__ import annotations

import threading
from contextvars import Token
from typing import Any, cast
from weakref import WeakKeyDictionary, WeakSet

from agents import Span, Trace
from openinference.instrumentation.openai_agents._processor import (
    OpenInferenceTracingProcessor,
)
from opentelemetry import trace as otel_trace
from opentelemetry.context import Context, attach, detach

from ._otel_trace_interceptor import _propagated_trace_id


class OTelTracingProcessorAdapter:
    """Bridge owned and propagated traces through one OpenInference processor."""

    def __init__(
        self,
        processor: OpenInferenceTracingProcessor,
        other_processors: list[OpenInferenceTracingProcessor],
    ) -> None:
        """Install callbacks while preserving OpenInference's span conversion."""
        for name in ("_root_spans", "_otel_spans"):
            if not isinstance(getattr(processor, name, None), dict):
                raise RuntimeError(
                    "OpenInferenceTracingProcessor internal API changed: expected "
                    f"{name} to be a dict. Update temporalio-openai-agents before "
                    "enabling OpenInference instrumentation."
                )
        self._processor = processor
        # OpenInference keeps draining processors registered until their last
        # trace/span ends. Weak references avoid retaining them after removal.
        self._other_processors = WeakSet(other_processors)
        self._lock = threading.RLock()
        self._active = True
        self._trace_tokens: WeakKeyDictionary[Trace, Token[Context]] = (
            WeakKeyDictionary()
        )
        self._on_trace_start = processor.on_trace_start
        self._on_trace_end = processor.on_trace_end
        self._on_span_start = processor.on_span_start
        self._on_span_end = processor.on_span_end
        setattr(processor, "on_trace_start", self.on_trace_start)
        setattr(processor, "on_trace_end", self.on_trace_end)
        setattr(processor, "on_span_start", self.on_span_start)
        setattr(processor, "on_span_end", self.on_span_end)

    @classmethod
    def install(
        cls,
        processor: OpenInferenceTracingProcessor,
        other_processors: list[OpenInferenceTracingProcessor],
    ) -> OTelTracingProcessorAdapter:
        """Reuse callbacks and in-flight state if uninstrumentation failed."""
        existing = getattr(processor.on_trace_start, "__self__", None)
        if isinstance(existing, cls):
            with existing._lock:
                existing._other_processors = WeakSet(other_processors)
                existing._active = True
            return existing
        return cls(processor, other_processors)

    def deactivate(self) -> None:
        """Stop accepting new traces while existing spans finish."""
        with self._lock:
            self._active = False

    def on_trace_start(self, trace: Trace) -> None:
        """Make a locally created trace root the current OpenTelemetry span."""
        with self._lock:
            if not self._active:
                return
            self._on_trace_start(trace)
            # A draining processor can decline to start a trace.
            root = self._processor._root_spans.get(trace.trace_id)
            if root is not None:
                self._trace_tokens[trace] = cast(
                    Token[Context],  # pyright: ignore[reportUnnecessaryCast]
                    attach(otel_trace.set_span_in_context(root)),
                )

    def on_trace_end(self, trace: Trace) -> None:
        """End the local root and restore its caller's OpenTelemetry context."""
        with self._lock:
            try:
                # The processor's registries also include starts delivered
                # before the adapter was installed on this instance.
                if trace.trace_id in self._processor._root_spans:
                    self._on_trace_end(trace)
            finally:
                token = self._trace_tokens.pop(trace, None)
                if token is not None:
                    detach(token)

    def on_span_start(self, span: Span[Any]) -> None:
        """Let the active processor record spans of explicitly propagated traces."""
        with self._lock:
            roots = self._processor._root_spans
            if not self._active or getattr(self._processor, "draining", False):
                if span.trace_id not in roots:
                    return
            elif (
                span.started_at
                and span.trace_id not in roots
                and _propagated_trace_id.get() == span.trace_id
            ):
                # Re-instrumentation leaves older processors draining. They
                # retain ownership of their local traces and in-flight spans.
                if any(
                    span.trace_id in processor._root_spans
                    or span.span_id in processor._otel_spans
                    for processor in self._other_processors
                ):
                    return
                parent = otel_trace.get_current_span().get_span_context()
                if parent.is_valid:
                    # OpenInference 2.6 requires a registered trace root. Expose
                    # the propagated parent only during this callback; it is
                    # never recorded or ended here. Serializing all callbacks
                    # prevents another callback from mistaking it for a local root.
                    remote_root = otel_trace.NonRecordingSpan(parent)
                    roots[span.trace_id] = remote_root
                    try:
                        self._on_span_start(span)
                    finally:
                        if roots.get(span.trace_id) is remote_root:
                            del roots[span.trace_id]
                    return
            self._on_span_start(span)

    def on_span_end(self, span: Span[Any]) -> None:
        """Finish owned spans, including those left in a draining processor."""
        with self._lock:
            if span.span_id not in self._processor._otel_spans:
                return
            self._on_span_end(span)
