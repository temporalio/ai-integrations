"""The model Activities: one provider round trip per Activity."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from datetime import timedelta
from functools import wraps
from typing import Any, TypeVar, cast

from agent_framework import SupportsChatGetResponse

from temporalio import activity
from temporalio.contrib.workflow_streams import WorkflowStreamClient
from temporalio.exceptions import ApplicationError
from temporalio.microsoft_agent_framework import _serde
from temporalio.service import RPCError, RPCStatusCode

INVOKE_MODEL = "microsoft_agent_framework.invoke_model"
INVOKE_MODEL_STREAMING = "microsoft_agent_framework.invoke_model_streaming"
STREAM_ERROR_TYPE = "microsoft_agent_framework.StreamError"
"""``ApplicationError.type`` of a streaming call whose updates did not all reach the topic."""
UNKNOWN_CHAT_CLIENT_TYPE = "MicrosoftAgentFrameworkUnknownChatClient"
"""``ApplicationError.type`` of a call for a model name the worker does not know."""
CHAT_CLIENT_FACTORY_ERROR_TYPE = "MicrosoftAgentFrameworkChatClientFactoryError"
"""``ApplicationError.type`` of a call whose chat client factory raised."""
"""``ApplicationError.type`` of a call for a model name the worker does not know."""

ChatClientFactory = Callable[[], SupportsChatGetResponse[Any]]


@dataclass
class InvokeModelInput:
    """One model round trip: the model name and the serialized request."""

    model: str
    messages: list[dict[str, Any]]
    options: dict[str, Any] = field(default_factory=dict)
    kwargs: dict[str, Any] = field(default_factory=dict)
    """The run's ``client_kwargs``, such as ``extra_headers``, for the provider client."""


@dataclass
class InvokeModelStreamingInput(InvokeModelInput):
    """A streamed round trip, optionally published to a workflow stream topic."""

    streaming_topic: str | None = None
    streaming_batch_interval_seconds: float = 0.1


F = TypeVar("F", bound=Callable[..., Any])


def auto_heartbeater(fn: F) -> F:
    """Heartbeat at half the Activity's heartbeat timeout while ``fn`` runs."""

    @wraps(fn)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        heartbeat_timeout = activity.info().heartbeat_timeout
        heartbeat_task: asyncio.Task[None] | None = None
        if heartbeat_timeout:
            heartbeat_task = asyncio.create_task(
                _heartbeat_every(heartbeat_timeout.total_seconds() / 2)
            )
        try:
            return await fn(*args, **kwargs)
        finally:
            if heartbeat_task is not None:
                heartbeat_task.cancel()
                # Let the cancellation land so no task outlives the Activity.
                await asyncio.wait([heartbeat_task])

    return cast(F, wrapper)


async def _heartbeat_every(delay: float) -> None:
    while True:
        await asyncio.sleep(delay)
        activity.heartbeat()


def _is_retryable_status(status: int) -> bool:
    return status in (408, 409, 429) or status >= 500


def _status_error(exc: BaseException) -> tuple[BaseException, int] | None:
    """Return the first exception in the cause chain that carries an HTTP status."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        status = getattr(current, "status_code", None)
        if (
            isinstance(status, int)
            and not isinstance(status, bool)
            and 400 <= status < 600
        ):
            return current, status
        current = current.__cause__
    return None


def _headers(exc: BaseException) -> dict[str, str]:
    headers: Any = getattr(exc, "headers", None)
    if headers is None:
        headers = getattr(getattr(exc, "response", None), "headers", None)
    if headers is None:
        return {}
    try:
        return {str(k).lower(): str(v) for k, v in dict(headers).items()}
    except (TypeError, ValueError):
        return {}


def _retry_after(headers: Mapping[str, str]) -> timedelta | None:
    try:
        if "retry-after-ms" in headers:
            return timedelta(milliseconds=int(headers["retry-after-ms"]))
        if "retry-after" in headers:
            return timedelta(seconds=int(headers["retry-after"]))
    except ValueError:
        return None
    return None


def classify_provider_error(exc: Exception) -> ApplicationError | None:
    """Map a provider error with an HTTP status onto Temporal's retry contract.

    Agent Framework wraps provider SDK errors in ``ChatClientException`` with
    the SDK error as ``__cause__``, so the chain is walked for a
    ``status_code``. 408, 409, 429 and 5xx stay retryable, with
    ``next_retry_delay`` from ``retry-after-ms`` or ``retry-after`` and
    ``x-should-retry`` honored; other 4xx are non-retryable. Returns ``None``
    for an error without a status, which stays retryable as it is.
    """
    found = _status_error(exc)
    if found is None:
        return None
    cause, status = found
    headers = _headers(cause)
    retryable = _is_retryable_status(status)
    should_retry = headers.get("x-should-retry")
    if should_retry == "false":
        retryable = False
    elif should_retry == "true":
        retryable = True
    return ApplicationError(
        str(exc),
        type=type(cause).__name__,
        non_retryable=not retryable,
        next_retry_delay=_retry_after(headers),
    )


@contextmanager
def _classify_provider_errors(model: str) -> Generator[None, None, None]:
    try:
        yield
    except Exception as err:
        translated = classify_provider_error(err)
        if translated is None:
            raise
        log = (
            activity.logger.error
            if translated.non_retryable
            else activity.logger.warning
        )
        log(
            "Model %r failed with %s (HTTP status error, %s)",
            model,
            translated.type,
            "not retried" if translated.non_retryable else "retried",
        )
        raise translated from err


def _stream_publish_error(err: Exception, model: str, topic: str) -> ApplicationError:
    """Fail a streaming call whose updates did not all reach the topic.

    A retry calls the model again, so the error is non-retryable when the
    retry would fail the same way: the server refused the batch as invalid
    (over the payload size limit, or past the run's Signal limit) or as over
    its gRPC message limit. Other failures, such as a server outage, stay
    retryable.
    """
    # The gRPC message size limit arrives as RESOURCE_EXHAUSTED with no status
    # of its own, so the message text tells it apart from rate limiting; the
    # Google ADK plugin matches the same text.
    non_retryable = isinstance(err, RPCError) and (
        err.status == RPCStatusCode.INVALID_ARGUMENT
        or (
            err.status == RPCStatusCode.RESOURCE_EXHAUSTED
            and "larger than max" in err.message
        )
    )
    return ApplicationError(
        f"model {model!r}: publish streamed updates to topic {topic!r}: {err}",
        type=STREAM_ERROR_TYPE,
        non_retryable=non_retryable,
    )


def _max_retries(client: Any) -> int:
    """Return the provider SDK's own retry count when the client exposes one.

    Agent Framework's OpenAI clients keep the SDK client on ``client``; the
    attribute is read by duck typing so no provider package is imported here.
    """
    for candidate in (client, getattr(client, "client", None)):
        retries = getattr(candidate, "max_retries", None)
        if isinstance(retries, int) and not isinstance(retries, bool):
            return retries
    return 0


class ChatClientActivities:
    """Holds the worker-side chat client factories and exposes the model Activities.

    Factories are called on first use and cached for the worker's lifetime, so
    credentials and connections never enter workflow history.
    """

    def __init__(self, factories: Mapping[str, ChatClientFactory]) -> None:
        """Store the factories keyed by the model names workflows use."""
        self._factories = dict(factories)
        self._clients: dict[str, SupportsChatGetResponse[Any]] = {}

    def _client(self, model: str) -> SupportsChatGetResponse[Any]:
        client = self._clients.get(model)
        if client is not None:
            return client
        factory = self._factories.get(model)
        if factory is None:
            known = sorted(self._factories) or "no names"
            raise ApplicationError(
                f"No chat client named {model!r} is registered on this worker; "
                f"MicrosoftAgentFrameworkPlugin(chat_clients=...) knows {known}.",
                type=UNKNOWN_CHAT_CLIENT_TYPE,
                non_retryable=True,
            )
        try:
            client = factory()
        except Exception as err:
            # A factory that cannot build its client (a missing API key, a bad
            # endpoint) fails the same way on every attempt; fail the run now
            # instead of retrying until someone fixes the worker.
            activity.logger.error(
                "Chat client factory for %r raised %s", model, type(err).__name__
            )
            raise ApplicationError(
                f"The chat client factory for {model!r} raised "
                f"{type(err).__name__}: {err}",
                type=CHAT_CLIENT_FACTORY_ERROR_TYPE,
                non_retryable=True,
            ) from err
        self._clients[model] = client
        retries = _max_retries(client)
        if retries > 0:
            activity.logger.warning(
                "Chat client %r retries %d times inside the provider SDK; disable "
                "those retries (for example AsyncOpenAI(max_retries=0)) so Temporal "
                "owns retries.",
                model,
                retries,
            )
        return client

    @activity.defn(name=INVOKE_MODEL)
    @auto_heartbeater
    async def invoke_model(self, input: InvokeModelInput) -> dict[str, Any]:
        """Run one model round trip and return the serialized ``ChatResponse``."""
        client = self._client(input.model)
        messages = _serde.load_messages(input.messages)
        options = _serde.load_options(input.options)
        with _classify_provider_errors(input.model):
            response = await client.get_response(
                messages, options=options, client_kwargs=input.kwargs
            )
        return _serde.dump_response(response)

    @activity.defn(name=INVOKE_MODEL_STREAMING)
    @auto_heartbeater
    async def invoke_model_streaming(
        self, input: InvokeModelStreamingInput
    ) -> list[dict[str, Any]]:
        """Run one streamed round trip and return the recorded updates.

        The record holds one entry per function call, usage or finish update
        and one entry per run of text (see ``_serde.StreamRecorder``). With
        ``streaming_topic`` set, every provider update is also published to
        that topic on the workflow's ``WorkflowStream`` as it arrives, batched
        at ``streaming_batch_interval_seconds``.
        """
        client = self._client(input.model)
        messages = _serde.load_messages(input.messages)
        options = _serde.load_options(input.options)
        recorder = _serde.StreamRecorder()

        async def consume(publish: Callable[[dict[str, Any]], None] | None) -> None:
            with _classify_provider_errors(input.model):
                async for update in client.get_response(
                    messages, stream=True, options=options, client_kwargs=input.kwargs
                ):
                    serialized = recorder.add(update)
                    if serialized is not None and publish is not None:
                        publish(serialized)

        if input.streaming_topic is None:
            await consume(None)
            return recorder.updates

        stream = WorkflowStreamClient.from_within_activity(
            batch_interval=timedelta(seconds=input.streaming_batch_interval_seconds)
        )
        topic = stream.topic(input.streaming_topic)
        # The stream's exit sends the last batch and raises the error of a
        # batch the background flush failed to send; either way subscribers
        # missed updates, so the call fails instead of returning as if they
        # had. A model error, or the Activity's cancellation, stays the error.
        await stream.__aenter__()
        try:
            await consume(topic.publish)
        except BaseException:
            with suppress(Exception):
                await stream.__aexit__(None, None, None)
            raise
        try:
            await stream.__aexit__(None, None, None)
        except Exception as err:
            raise _stream_publish_error(
                err, input.model, input.streaming_topic
            ) from err
        return recorder.updates
