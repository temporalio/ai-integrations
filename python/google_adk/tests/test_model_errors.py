"""Model activity failures carry Temporal's retry decision.

A Gemini request that can never succeed (400 for a bad argument, 401 or 403
for a bad key, 404 for an unknown model) must fail the model activity as
non-retryable, so the workflow fails instead of re-sending the request under
the default unlimited retry policy. Rate limits, timeouts and server errors
stay retryable. The TypeScript and Java Google ADK plugins and
``temporalio.google_genai`` classify these statuses the same way. A 409
conflict also stays retryable, as in the TypeScript and Java plugins.

A local HTTP server stands in for the Gemini API, so the real ADK ``Gemini``
model and ``google-genai`` client run end to end without credentials. A fake
model that raises errors directly covers SDKs that report the status as
``status_code``, and errors without an HTTP status.

A streamed call whose responses cannot all be published fails with
``googleadk.StreamError``, as in the Go and Java plugins. It is non-retryable
when a retry would call the model again and fail the same way.
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import AsyncGenerator, Iterator
from datetime import timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import ClassVar
from unittest.mock import AsyncMock, MagicMock

import pytest
from google.adk.models import BaseLlm, LLMRegistry
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai.errors import APIError
from google.genai.types import Content, Part

from temporalio.api.failure.v1 import Failure
from temporalio.converter import DataConverter
from temporalio.google_adk._model import (
    StreamingInvokeInput,
    TemporalLlmRequest,
    invoke_model,
    invoke_model_streaming,
)
from temporalio.service import RPCError, RPCStatusCode
from temporalio.testing import ActivityEnvironment

_STATUS_NAMES = {
    400: "INVALID_ARGUMENT",
    401: "UNAUTHENTICATED",
    403: "PERMISSION_DENIED",
    404: "NOT_FOUND",
    408: "DEADLINE_EXCEEDED",
    409: "ABORTED",
    429: "RESOURCE_EXHAUSTED",
    500: "INTERNAL",
    503: "UNAVAILABLE",
}


class _FakeGemini:
    """Answers every request with one HTTP status and records each request."""

    def __init__(self) -> None:
        self.status = 400
        self.requests = 0
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("content-length") or 0)
                if length:
                    self.rfile.read(length)
                fake.requests += 1
                body = json.dumps(
                    {
                        "error": {
                            "code": fake.status,
                            "message": "rejected by the fake Gemini API",
                            "status": _STATUS_NAMES[fake.status],
                        }
                    }
                ).encode()
                self.send_response(fake.status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: object) -> None:
                pass

        self._server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_port}"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture
def fake_gemini(monkeypatch: pytest.MonkeyPatch) -> Iterator[_FakeGemini]:
    fake = _FakeGemini()
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    monkeypatch.setenv("GOOGLE_GEMINI_BASE_URL", fake.url)
    yield fake
    fake.close()


class _StatusCodeError(Exception):
    """An SDK error that carries the HTTP status as ``status_code``."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"{status_code} from the fake provider")
        self.status_code = status_code


class _FailingModel(BaseLlm):
    """A non-Gemini model that raises ``error`` as soon as it is called."""

    error: ClassVar[Exception]

    @classmethod
    def supported_models(cls) -> list[str]:
        return ["failing-model"]

    def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        raise self.error


class _StreamingModel(BaseLlm):
    """Streams one response, then raises ``error``, waits, or ends."""

    error: ClassVar[Exception | None] = None
    wait: ClassVar[bool] = False
    streamed: ClassVar[asyncio.Event]

    @classmethod
    def supported_models(cls) -> list[str]:
        return ["streaming-model"]

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        yield LlmResponse(content=Content(role="model", parts=[Part(text="hello")]))
        self.streamed.set()
        if self.wait:
            await asyncio.Event().wait()
        if self.error is not None:
            raise self.error


def _request(model: str) -> TemporalLlmRequest:
    return TemporalLlmRequest(
        model=model,
        contents=[Content(role="user", parts=[Part(text="hi")])],
    )


async def _run(streaming: bool, model: str = "gemini-2.5-flash") -> BaseException:
    if streaming:
        env = ActivityEnvironment(client=MagicMock())
        call = env.run(
            invoke_model_streaming,
            StreamingInvokeInput(
                llm_request=_request(model),
                streaming_topic="events",
                streaming_batch_interval=timedelta(milliseconds=100),
            ),
        )
    else:
        call = ActivityEnvironment().run(invoke_model, _request(model))
    with pytest.raises(BaseException) as raised:
        await call
    return raised.value


async def _run_failing(error: Exception, streaming: bool) -> BaseException:
    _FailingModel.error = error
    LLMRegistry.register(_FailingModel)
    return await _run(streaming, model="failing-model")


async def _run_refused_stream(
    publish_error: Exception,
    model_error: Exception | None = None,
    cancel: bool = False,
) -> BaseException:
    """Stream one response while publishing it fails with ``publish_error``.

    The model then raises ``model_error``, or waits for the activity to be
    cancelled when ``cancel`` is set, or ends.
    """
    _StreamingModel.error = model_error
    _StreamingModel.wait = cancel
    _StreamingModel.streamed = asyncio.Event()
    LLMRegistry.register(_StreamingModel)
    client = MagicMock()
    client.data_converter = DataConverter.default
    signal = AsyncMock(side_effect=publish_error)
    client.get_workflow_handle.return_value.signal = signal
    env = ActivityEnvironment(client=client)
    call = asyncio.ensure_future(
        env.run(
            invoke_model_streaming,
            StreamingInvokeInput(
                llm_request=_request("streaming-model"),
                streaming_topic="events",
                # Only the stream's exit sends the batch.
                streaming_batch_interval=timedelta(hours=1),
            ),
        )
    )
    if cancel:
        await _StreamingModel.streamed.wait()
        env.cancel()
    with pytest.raises(BaseException) as raised:
        await call
    signal.assert_awaited_once()
    return raised.value


def _recorded(exc: BaseException) -> Failure:
    """The failure the worker records for an activity that raised ``exc``."""
    failure = Failure()
    converter = DataConverter.default
    converter.failure_converter.to_failure(exc, converter.payload_converter, failure)
    return failure


def _gemini_failure(exc: BaseException, status: int) -> Failure:
    """The recorded failure, checked to wrap Gemini's error for ``status``."""
    error = exc.__cause__
    assert isinstance(error, APIError), exc
    assert error.code == status, error
    failure = _recorded(exc)
    error_type = type(error).__name__
    assert failure.application_failure_info.type == error_type, failure
    assert failure.cause.application_failure_info.type == error_type, failure
    return failure


@pytest.mark.parametrize("streaming", [False, True], ids=["invoke", "streaming"])
@pytest.mark.parametrize("status", [400, 401, 403, 404])
async def test_permanent_status_is_non_retryable(
    fake_gemini: _FakeGemini, status: int, streaming: bool
) -> None:
    fake_gemini.status = status
    failure = _gemini_failure(await _run(streaming), status)
    assert fake_gemini.requests == 1
    assert failure.application_failure_info.non_retryable, failure


@pytest.mark.parametrize("streaming", [False, True], ids=["invoke", "streaming"])
@pytest.mark.parametrize("status", [408, 409, 429, 500, 503])
async def test_transient_status_stays_retryable(
    fake_gemini: _FakeGemini, status: int, streaming: bool
) -> None:
    fake_gemini.status = status
    failure = _gemini_failure(await _run(streaming), status)
    assert fake_gemini.requests == 1
    assert not failure.application_failure_info.non_retryable, failure


@pytest.mark.parametrize("streaming", [False, True], ids=["invoke", "streaming"])
@pytest.mark.parametrize(("status", "non_retryable"), [(404, True), (429, False)])
async def test_status_code_attribute_is_classified(
    status: int, non_retryable: bool, streaming: bool
) -> None:
    error = _StatusCodeError(status)
    exc = await _run_failing(error, streaming)
    assert exc.__cause__ is error
    failure = _recorded(exc)
    assert failure.application_failure_info.type == "_StatusCodeError", failure
    assert failure.application_failure_info.non_retryable == non_retryable, failure
    assert failure.cause.application_failure_info.type == "_StatusCodeError", failure


@pytest.mark.parametrize("streaming", [False, True], ids=["invoke", "streaming"])
async def test_unknown_model_is_non_retryable(streaming: bool) -> None:
    exc = await _run(streaming, model="no-such-model")
    failure = _recorded(exc)
    assert "no-such-model" in failure.message, failure
    assert failure.application_failure_info.non_retryable, failure


@pytest.mark.parametrize("streaming", [False, True], ids=["invoke", "streaming"])
async def test_error_without_status_stays_retryable(streaming: bool) -> None:
    error = ConnectionError("connection reset by the fake provider")
    exc = await _run_failing(error, streaming)
    assert exc is error
    failure = _recorded(exc)
    assert failure.application_failure_info.type == "ConnectionError", failure
    assert not failure.application_failure_info.non_retryable, failure


_BATCH_REFUSED = RPCError(
    "Blob data size exceeds limit.", RPCStatusCode.INVALID_ARGUMENT, b""
)


@pytest.mark.parametrize(
    ("publish_error", "non_retryable"),
    [
        (_BATCH_REFUSED, True),
        (
            RPCError(
                "grpc: received message larger than max (5242880 vs. 4194304)",
                RPCStatusCode.RESOURCE_EXHAUSTED,
                b"",
            ),
            True,
        ),
        (TimeoutError("Flush retry exceeded max_retry_duration (0:10:00)."), True),
        (
            RPCError(
                "namespace rate limit exceeded", RPCStatusCode.RESOURCE_EXHAUSTED, b""
            ),
            False,
        ),
        (RPCError("connection refused", RPCStatusCode.UNAVAILABLE, b""), False),
    ],
    ids=["refused", "too-large", "gave-up", "rate-limited", "unavailable"],
)
async def test_failed_stream_publish_fails_the_call(
    publish_error: Exception, non_retryable: bool
) -> None:
    exc = await _run_refused_stream(publish_error)
    assert exc.__cause__ is publish_error
    failure = _recorded(exc)
    assert failure.application_failure_info.type == "googleadk.StreamError", failure
    assert failure.application_failure_info.non_retryable == non_retryable, failure
    assert "'streaming-model'" in failure.message, failure
    assert "'events'" in failure.message, failure


async def test_model_error_wins_over_failed_stream_publish() -> None:
    error = _StatusCodeError(404)
    exc = await _run_refused_stream(_BATCH_REFUSED, model_error=error)
    assert exc.__cause__ is error
    failure = _recorded(exc)
    assert failure.application_failure_info.type == "_StatusCodeError", failure
    assert failure.application_failure_info.non_retryable, failure


async def test_cancellation_wins_over_failed_stream_publish() -> None:
    exc = await _run_refused_stream(_BATCH_REFUSED, cancel=True)
    assert isinstance(exc, asyncio.CancelledError), exc
