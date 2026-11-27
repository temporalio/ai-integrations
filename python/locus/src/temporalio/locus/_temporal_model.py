"""Workflow-side Locus model: every call is a Temporal activity."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import timedelta
from typing import Any

from pydantic_core import PydanticSerializationError, to_jsonable_python

from locus.core.events import ModelChunkEvent
from locus.core.messages import Message
from locus.models.base import ModelResponse
from temporalio import workflow
from temporalio.common import Priority, RetryPolicy
from temporalio.exceptions import ApplicationError
from temporalio.workflow import ActivityCancellationType, VersioningIntent

from ._model_activity import (
    COMPLETE_ACTIVITY,
    STREAM_ACTIVITY,
    ModelCallInput,
    ModelStreamResult,
)


class TemporalModel:
    """A Locus model whose ``complete`` and ``stream`` run as Temporal activities.

    ``model_name`` selects a factory in ``LocusPlugin(models={...})``; the real
    model, with its credentials, exists only on the worker. Building a
    ``TemporalModel`` does no I/O, so it is safe in workflow code.

    ``heartbeat_timeout`` defaults to 10 seconds; the activities heartbeat on a
    timer at half of it. Pass ``None`` to turn heartbeats off.

    ``stream`` is batched for the workflow: the activity collects the model's
    chunks and the workflow yields them once the call has finished.

    To show chunks to outside consumers while the model is still answering,
    set ``streaming_topic`` and host a
    :class:`temporalio.contrib.workflow_streams.WorkflowStream` on the
    workflow. Each chunk of a streamed call is then also published on that
    topic, batched on ``streaming_batch_interval``. Locus streams a call only
    in a streamed run, ``agent.run(prompt, stream_tokens=True)``.
    """

    def __init__(
        self,
        model_name: str,
        *,
        task_queue: str | None = None,
        schedule_to_close_timeout: timedelta | None = None,
        schedule_to_start_timeout: timedelta | None = None,
        start_to_close_timeout: timedelta | None = None,
        heartbeat_timeout: timedelta | None = timedelta(seconds=10),
        retry_policy: RetryPolicy | None = None,
        cancellation_type: ActivityCancellationType = ActivityCancellationType.TRY_CANCEL,
        versioning_intent: VersioningIntent | None = None,
        summary: str | None = None,
        priority: Priority = Priority.default,
        streaming_topic: str | None = None,
        streaming_batch_interval: timedelta = timedelta(milliseconds=100),
    ) -> None:
        """Set the model name, the activity options and the streaming settings."""
        if schedule_to_close_timeout is None and start_to_close_timeout is None:
            raise ValueError(
                "TemporalModel needs start_to_close_timeout or schedule_to_close_timeout"
            )
        self.model_name = model_name
        self._streaming_topic = streaming_topic
        self._streaming_batch_interval = streaming_batch_interval
        self._options: dict[str, Any] = {
            "task_queue": task_queue,
            "schedule_to_close_timeout": schedule_to_close_timeout,
            "schedule_to_start_timeout": schedule_to_start_timeout,
            "start_to_close_timeout": start_to_close_timeout,
            "heartbeat_timeout": heartbeat_timeout,
            "retry_policy": retry_policy,
            "cancellation_type": cancellation_type,
            "versioning_intent": versioning_intent,
            "summary": summary,
            "priority": priority,
        }

    def _input(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None,
        options: dict[str, Any],
    ) -> ModelCallInput:
        # The options cross into an Activity as JSON. One that cannot (a callback,
        # a client) would fail every Workflow Task, so fail the run instead.
        try:
            to_jsonable_python(options)
        except (PydanticSerializationError, TypeError, ValueError) as err:
            raise ApplicationError(
                f"A model call option cannot be sent to the model Activity: {err}. "
                "Set it on the model that the LocusPlugin factory builds instead.",
                type="LocusUnserializableModelOption",
                non_retryable=True,
            ) from err
        return ModelCallInput(
            model_name=self.model_name,
            messages=messages,
            tools=tools,
            options=options,
        )

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        """Run one completion as the ``locus_model_complete`` activity."""
        return await workflow.execute_activity(
            COMPLETE_ACTIVITY,
            self._input(messages, tools, kwargs),
            result_type=ModelResponse,
            **self._options,
        )

    async def stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ModelChunkEvent]:
        """Run one streamed completion as the ``locus_model_stream`` activity."""
        call = self._input(messages, tools, kwargs)
        call.streaming_topic = self._streaming_topic
        call.streaming_batch_interval_seconds = (
            self._streaming_batch_interval.total_seconds()
        )
        result = await workflow.execute_activity(
            STREAM_ACTIVITY,
            call,
            result_type=ModelStreamResult,
            **self._options,
        )
        for chunk in result.chunks:
            yield chunk
