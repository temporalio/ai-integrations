"""Provides support for integration with OpenAI Agents SDK tracing across workflows"""

import uuid
from types import TracebackType
from typing import Any, cast

from agents import SpanData, Trace, TracingProcessor
from agents.tracing import (
    TracingConfig,
    get_trace_provider,
)
from agents.tracing.config import TracingConfig
from agents.tracing.provider import (
    DefaultTraceProvider,
    SynchronousMultiTracingProcessor,
)
from agents.tracing.scope import Scope
from agents.tracing.spans import Span, SpanImpl, TSpanData
from agents.tracing.traces import TraceImpl

import temporalio.workflow
from temporalio import workflow
from temporalio.openai_agents._trace_interceptor import _refuse_tracing_key
from temporalio.workflow import ReadOnlyContextError


class ActivitySpanData(SpanData):
    """Captures fields from ActivityTaskScheduledEventAttributes for tracing."""

    def __init__(
        self,
        activity_id: str,
        activity_type: str,
        task_queue: str,
        schedule_to_close_timeout: float | None = None,
        schedule_to_start_timeout: float | None = None,
        start_to_close_timeout: float | None = None,
        heartbeat_timeout: float | None = None,
    ):
        """Initialize an ActivitySpanData instance."""
        self.activity_id = activity_id
        self.activity_type = activity_type
        self.task_queue = task_queue
        self.schedule_to_close_timeout = schedule_to_close_timeout
        self.schedule_to_start_timeout = schedule_to_start_timeout
        self.start_to_close_timeout = start_to_close_timeout
        self.heartbeat_timeout = heartbeat_timeout

    @property
    def type(self) -> str:
        """Return the type of this span data."""
        return "temporal-activity"

    def export(self) -> dict[str, Any]:
        """Export the span data as a dictionary."""
        return {
            "type": self.type,
            "activity_id": self.activity_id,
            "activity_type": self.activity_type,
            "task_queue": self.task_queue,
            "schedule_to_close_timeout": self.schedule_to_close_timeout,
            "schedule_to_start_timeout": self.schedule_to_start_timeout,
            "start_to_close_timeout": self.start_to_close_timeout,
            "heartbeat_timeout": self.heartbeat_timeout,
        }


def activity_span(
    activity_id: str,
    activity_type: str,
    task_queue: str,
    start_to_close_timeout: float,
) -> Span[ActivitySpanData]:
    """Create a trace span for a Temporal activity."""
    return get_trace_provider().create_span(
        span_data=ActivitySpanData(
            activity_id=activity_id,
            activity_type=activity_type,
            task_queue=task_queue,
            start_to_close_timeout=start_to_close_timeout,
        ),
    )


class _TemporalTracingProcessor(SynchronousMultiTracingProcessor):
    def __init__(
        self,
        impl: SynchronousMultiTracingProcessor,
        start_spans_in_replay: bool,
    ):
        super().__init__()
        self._impl = impl
        self._emit_spans_in_replay = start_spans_in_replay

    def add_tracing_processor(self, tracing_processor: TracingProcessor):
        self._impl.add_tracing_processor(tracing_processor)

    def set_processors(self, processors: list[TracingProcessor]):
        self._impl.set_processors(processors)

    def on_trace_start(self, trace: Trace) -> None:
        if not self._emit_spans_in_replay:
            if workflow.in_workflow() and workflow.unsafe.is_replaying_history_events():
                # In replay mode, don't report
                return

        self._impl.on_trace_start(trace)

    def on_trace_end(self, trace: Trace) -> None:
        if not self._emit_spans_in_replay:
            if workflow.in_workflow() and workflow.unsafe.is_replaying_history_events():
                # In replay mode, don't report
                return

        self._impl.on_trace_end(trace)

    def on_span_start(self, span: Span[Any]) -> None:
        if not self._emit_spans_in_replay:
            if workflow.in_workflow() and workflow.unsafe.is_replaying_history_events():
                # In replay mode, don't report
                return
        self._impl.on_span_start(span)

    def on_span_end(self, span: Span[Any]) -> None:
        if not self._emit_spans_in_replay:
            if workflow.in_workflow() and workflow.unsafe.is_replaying_history_events():
                # In replay mode, don't report
                return

        self._impl.on_span_end(span)

    def shutdown(self, timeout: float | None = None) -> None:
        self._impl.shutdown(timeout)

    def force_flush(self) -> None:
        self._impl.force_flush()


def _workflow_uuid() -> str:
    if (
        getattr(
            temporalio.workflow.instance(), "__temporal_openai_tracing_random", None
        )
        is None
    ):
        setattr(
            temporalio.workflow.instance(),
            "__temporal_openai_tracing_random",
            temporalio.workflow.new_random(),
        )
    random = getattr(temporalio.workflow.instance(), "__temporal_openai_tracing_random")
    return uuid.UUID(
        bytes=random.getrandbits(16 * 8).to_bytes(16, "big"), version=4
    ).hex[:24]


# Traces and spans created while tracing is disabled report here, so they keep their IDs
# but reach no processor.
_disabled_processor = SynchronousMultiTracingProcessor()


def _tracing_disabled() -> bool:
    # set_tracing_disabled() sets the flag on the provider installed at the time, which
    # can be the original provider or the one of any Worker that started since. Resolve
    # it from the installed provider down, so the latest call wins.
    provider = get_trace_provider()
    while isinstance(provider, TemporalTraceProvider):
        if provider._manual_disabled is not None:
            return provider._manual_disabled
        provider = provider._original_provider
    return bool(getattr(provider, "_disabled", False))


_TRACING_PATCH_PREFIX = "openai-agents-tracing-"


class _WorkflowTracingDecision:
    def __init__(self) -> None:
        # A run without markers reports, as it did before they existed.
        self.history_length = -1
        self.disabled = False


def _disabled_at_creation() -> bool:
    # A Workflow rebuilt from history recreates its traces and spans, possibly after the flag
    # changed. A Workflow reads the flag once per Workflow Task, and a patch marker named for
    # the Task records each change, so the rebuilt Workflow reads the marker instead and makes
    # the same decisions. Every new patch ID also enters the TemporalChangeVersion search
    # attribute, so only changes get one. The markers are recorded as deprecated, so builds
    # that do not check for them skip them on replay.
    disabled = _tracing_disabled()
    if not workflow.in_workflow() or workflow.unsafe.is_read_only():
        # Queries and update validators add nothing to history.
        return disabled
    instance = temporalio.workflow.instance()
    decision = getattr(instance, "__temporal_openai_tracing_decision", None)
    if decision is None:
        decision = _WorkflowTracingDecision()
        setattr(instance, "__temporal_openai_tracing_decision", decision)
    history_length = workflow.info().get_current_history_length()
    if decision.history_length != history_length:
        # Replay delivers a Task's markers before its code runs, so a change recorded later in
        # the Task would also apply to what came before it. The flag is read at the Task's
        # first creation instead.
        decision.history_length = history_length
        patch_id = (
            f"{_TRACING_PATCH_PREFIX}"
            f"{'enabled' if decision.disabled else 'disabled'}-{history_length}"
        )
        if workflow.unsafe.is_replaying():
            if workflow.patched(patch_id):
                decision.disabled = not decision.disabled
        elif disabled != decision.disabled:
            workflow.deprecate_patch(patch_id)
            decision.disabled = disabled
    return decision.disabled


def _reports_nothing(scope: Trace | Span[Any] | None) -> bool:
    return getattr(scope, "_processor", None) is _disabled_processor


def _report_nothing(scope: Trace | Span[Any]) -> None:
    if isinstance(scope, (TraceImpl, SpanImpl)):
        scope._processor = _disabled_processor


class TemporalTraceProvider(DefaultTraceProvider):
    """A trace provider that integrates with Temporal workflows."""

    def __init__(self, start_spans_in_replay: bool = False):
        """Initialize the TemporalTraceProvider."""
        super().__init__()
        self._original_provider = cast(DefaultTraceProvider, get_trace_provider())
        self._multi_processor = _TemporalTracingProcessor(
            self._original_provider._multi_processor,
            start_spans_in_replay,
        )

    def create_trace(
        self,
        name: str,
        trace_id: str | None = None,
        group_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        disabled: bool = False,
        tracing: TracingConfig | None = None,
    ) -> Trace:
        """Create a trace that reports nothing if tracing is disabled."""
        trace = super().create_trace(
            name,
            trace_id=trace_id,
            group_id=group_id,
            metadata=metadata,
            disabled=disabled,
            tracing=tracing,
        )
        # As in the Agents SDK, tracing being disabled when a trace is created decides
        # whether it reports its start and its end, so later changes keep them paired.
        if isinstance(trace, TraceImpl) and _disabled_at_creation():
            _report_nothing(trace)
        return trace

    def create_span(
        self,
        span_data: TSpanData,
        span_id: str | None = None,
        parent: Trace | Span[Any] | None = None,
        disabled: bool = False,
    ) -> Span[TSpanData]:
        """Create a span that reports nothing if tracing or its parent is disabled."""
        span = super().create_span(
            span_data, span_id=span_id, parent=parent, disabled=disabled
        )
        parents = (
            [parent]
            if parent is not None
            else [Scope.get_current_span(), Scope.get_current_trace()]
        )
        if isinstance(span, SpanImpl) and (
            any(_reports_nothing(p) for p in parents) or _disabled_at_creation()
        ):
            _report_nothing(span)
        return span

    def time_iso(self) -> str:
        """Return the current deterministic time in ISO 8601 format."""
        if workflow.in_workflow():
            return workflow.now().isoformat()
        return super().time_iso()

    def gen_trace_id(self) -> str:
        """Generate a new trace ID."""
        if workflow.in_workflow():
            try:
                """Generate a new trace ID."""
                return f"trace_{_workflow_uuid()}"
            except ReadOnlyContextError:
                return f"trace_{uuid.uuid4().hex}"
        return super().gen_trace_id()

    def gen_span_id(self) -> str:
        """Generate a span ID."""
        if workflow.in_workflow():
            try:
                """Generate a deterministic span ID."""
                return f"span_{_workflow_uuid()}"
            except ReadOnlyContextError:
                return f"span_{uuid.uuid4().hex[:24]}"
        return super().gen_span_id()

    def gen_group_id(self) -> str:
        """Generate a group ID."""
        if workflow.in_workflow():
            try:
                """Generate a deterministic group ID."""
                return f"group_{_workflow_uuid()}"
            except ReadOnlyContextError:
                return f"group_{uuid.uuid4().hex[:24]}"
        return super().gen_group_id()

    def create_trace(
        self,
        name: str,
        trace_id: str | None = None,
        group_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        disabled: bool = False,
        tracing: TracingConfig | None = None,
    ) -> Trace:
        """Create a new trace, refusing its own tracing key in workflow code."""
        if workflow.in_workflow() and tracing is not None:
            _refuse_tracing_key(tracing.get("api_key"))
        return super().create_trace(
            name,
            trace_id=trace_id,
            group_id=group_id,
            metadata=metadata,
            disabled=disabled,
            tracing=tracing,
        )

    def __enter__(self):
        """Enter the context of the Temporal trace provider."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException],
        exc_val: BaseException,
        exc_tb: TracebackType,
    ):
        """Exit the context of the Temporal trace provider."""
        self._multi_processor.shutdown()
