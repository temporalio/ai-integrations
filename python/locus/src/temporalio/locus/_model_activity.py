"""Worker-side model activities: the only place a real Locus model runs."""

from __future__ import annotations

from collections.abc import Callable
from datetime import timedelta
from typing import Any

from pydantic import BaseModel, Field

from locus.agent.model_retry import RETRYABLE_REASONS
from locus.core.events import ModelChunkEvent
from locus.core.messages import Message
from locus.models.base import ModelProtocol, ModelResponse
from locus.models.failover import FailoverReason, classify
from temporalio import activity
from temporalio.contrib.workflow_streams import WorkflowStreamClient
from temporalio.exceptions import ApplicationError

from ._heartbeat import heartbeat_every

COMPLETE_ACTIVITY = "locus_model_complete"
STREAM_ACTIVITY = "locus_model_stream"


class ModelCallInput(BaseModel):
    """One model call: which registered model, and what the agent loop sent it."""

    model_name: str
    messages: list[Message]
    tools: list[dict[str, Any]] | None = None
    options: dict[str, Any] = Field(default_factory=dict)
    # Set only for a streamed call that also publishes its chunks to a
    # WorkflowStream topic for outside subscribers.
    streaming_topic: str | None = None
    streaming_batch_interval_seconds: float = 0.1


class ModelStreamResult(BaseModel):
    """The chunks of one streamed model call, in order."""

    chunks: list[ModelChunkEvent]


class ModelActivities:
    """Holds the registered model factories and exposes the model activities."""

    def __init__(self, factories: dict[str, Callable[[], ModelProtocol]]) -> None:
        """Store the factories; a model is built on first use and then kept."""
        self._factories = factories
        self._models: dict[str, ModelProtocol] = {}

    def _model(self, name: str) -> ModelProtocol:
        if name not in self._models:
            if name not in self._factories:
                raise ApplicationError(
                    f"Unknown model name {name!r}. Known: {sorted(self._factories)}",
                    type="LocusUnknownModel",
                    non_retryable=True,
                )
            self._models[name] = self._factories[name]()
        return self._models[name]

    @activity.defn(name=COMPLETE_ACTIVITY)
    async def complete(self, input: ModelCallInput) -> ModelResponse:
        """Run one completion on the named model."""
        model = self._model(input.model_name)
        async with heartbeat_every():
            try:
                return await model.complete(
                    input.messages, input.tools, **input.options
                )
            except Exception as err:
                failure = _as_failure(err)
                if failure is err:
                    # Not ``raise err from err``: an error that is its own cause
                    # cannot be converted to a Failure.
                    raise
                raise failure from err

    @activity.defn(name=STREAM_ACTIVITY)
    async def stream(self, input: ModelCallInput) -> ModelStreamResult:
        """Run one streamed completion on the named model and return its chunks.

        With ``streaming_topic`` set, each chunk is also published to that topic
        of the workflow's ``WorkflowStream`` as the model produces it. The
        workflow still gets every chunk in the result, so replay does not
        depend on what was published.
        """
        model = self._model(input.model_name)
        chunks: list[ModelChunkEvent] = []
        async with heartbeat_every():
            try:
                if input.streaming_topic is None:
                    async for chunk in model.stream(
                        input.messages, input.tools, **input.options
                    ):
                        chunks.append(chunk)
                else:
                    client = WorkflowStreamClient.from_within_activity(
                        batch_interval=timedelta(
                            seconds=input.streaming_batch_interval_seconds
                        ),
                    )
                    topic = client.topic(input.streaming_topic, type=ModelChunkEvent)
                    async with client:
                        async for chunk in model.stream(
                            input.messages, input.tools, **input.options
                        ):
                            chunks.append(chunk)
                            topic.publish(chunk)
            except Exception as err:
                failure = _as_failure(err)
                if failure is err:
                    # Not ``raise err from err``: an error that is its own cause
                    # cannot be converted to a Failure.
                    raise
                raise failure from err
        return ModelStreamResult(chunks=chunks)

    def activities(self) -> list[Callable[..., Any]]:
        """The activities to register on the worker."""
        return [self.complete, self.stream]


def _as_failure(err: Exception) -> Exception:
    """Make an error Locus classifies as final a non-retryable failure.

    Locus retries rate limits, overload, server errors and timeouts; a
    classified error outside that set (auth, billing, context overflow, a bad
    request) fails the same way on every attempt. An unclassified error is left
    to the activity's retry policy.
    """
    if isinstance(err, ApplicationError):
        return err
    reason = classify(err).reason
    if reason is FailoverReason.UNKNOWN or reason in RETRYABLE_REASONS:
        return err
    return ApplicationError(
        f"{type(err).__name__}: {err}", type=f"Locus.{reason.value}", non_retryable=True
    )
