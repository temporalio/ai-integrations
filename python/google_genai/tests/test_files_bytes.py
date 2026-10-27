"""Binary file content crosses the Files and File Search store activities intact.

``files.upload`` and ``file_search_stores.upload_to_file_search_store`` accept a
binary stream, and ``files.download`` returns bytes. The tests here run the real
activities against a local HTTP server that speaks Gemini's resumable upload and
download protocol, then compare the bytes the server saw with the bytes the
workflow handed to the client:

* a workflow-level test per payload (a valid PNG, a non-UTF-8 byte string, every
  byte value, ASCII);
* activity-level tests, including inputs shaped as an older version recorded
  them;
* a replay of histories recorded on ``7eea646`` (upstream main before this
  change), when ``bytes`` went into the activity input model as JSON text.
"""

from __future__ import annotations

import base64
import io
import json
import threading
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from datetime import timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest
from google.auth.credentials import Credentials
from google.genai import Client as GeminiClient
from google.genai import types

from temporalio import workflow
from temporalio.client import Client, WorkflowHistory
from temporalio.google_genai import GoogleGenAIPlugin, TemporalAsyncClient
from temporalio.google_genai._gemini_activity import GeminiApiCaller
from temporalio.google_genai._models import (
    _GeminiUploadFileRequest,
    _GeminiUploadToFileSearchStoreRequest,
)
from temporalio.testing import ActivityEnvironment
from temporalio.worker import Replayer
from tests.helpers import new_worker

HISTORIES = Path(__file__).parent / "histories"

# A 1x1 PNG: the bytes of a real image, whose first byte (0x89) is not valid UTF-8.
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6"
    "kgAAAABJRU5ErkJggg=="
)
NON_UTF8 = b"\xff\xfe\x00\x80 not text \xc3\x28"
ALL_BYTE_VALUES = bytes(range(256)) * 4
ASCII = b"hello,world\n"

# What the server answers a download with: binary, and not valid UTF-8.
DOWNLOAD_BODY = PNG + NON_UTF8

PAYLOADS = [
    pytest.param(PNG, id="png"),
    pytest.param(NON_UTF8, id="non-utf8"),
    pytest.param(ALL_BYTE_VALUES, id="all-byte-values"),
    pytest.param(ASCII, id="ascii"),
]

STORE = "fileSearchStores/test-store"


class _FakeGeminiFilesServer:
    """Local server for Gemini's resumable uploads and file downloads.

    Keeps the body of every finalized upload, keyed by ``"files"`` or
    ``"file_search_store"``, and answers a file download with ``download_body``.
    """

    def __init__(self, download_body: bytes) -> None:
        self.uploads: dict[str, bytes] = {}
        self.start_requests: list[dict[str, str]] = []
        uploads = self.uploads
        start_requests = self.start_requests
        server = self

        class Handler(BaseHTTPRequestHandler):
            def _reply(
                self,
                body: bytes,
                *,
                content_type: str = "application/json",
                headers: dict[str, str] | None = None,
            ) -> None:
                self.send_response(200)
                self.send_header("content-type", content_type)
                for key, value in (headers or {}).items():
                    self.send_header(key, value)
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("content-length") or 0)
                body = self.rfile.read(length) if length else b""
                if self.path.startswith("/session/"):
                    kind = self.path.rsplit("/", 1)[1]
                    uploads[kind] = body
                    final = {
                        "files": {
                            "file": {
                                "name": "files/uploaded",
                                "sizeBytes": str(len(body)),
                            }
                        },
                        "file_search_store": {"name": "operations/store-upload"},
                    }[kind]
                    self._reply(
                        json.dumps(final).encode(),
                        headers={"x-goog-upload-status": "final"},
                    )
                    return
                kind = (
                    "file_search_store"
                    if "uploadToFileSearchStore" in self.path
                    else "files"
                )
                start_requests.append(
                    {
                        "kind": kind,
                        "size": self.headers.get(
                            "x-goog-upload-header-content-length", ""
                        ),
                        "mime_type": self.headers.get(
                            "x-goog-upload-header-content-type", ""
                        ),
                    }
                )
                self._reply(
                    b"{}",
                    headers={
                        "x-goog-upload-url": f"{server.url}/session/{kind}",
                        "x-goog-upload-status": "active",
                    },
                )

            def do_GET(self) -> None:  # noqa: N802
                self._reply(download_body, content_type="application/octet-stream")

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                pass

        self._server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_port}"

    def start(self) -> None:
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()


@contextmanager
def _gemini_files_client(
    download_body: bytes = DOWNLOAD_BODY,
) -> Generator[tuple[GeminiClient, _FakeGeminiFilesServer], None, None]:
    """Yield a real ``genai.Client`` pointed at a fake Gemini files server."""
    server = _FakeGeminiFilesServer(download_body)
    server.start()
    try:
        yield (
            GeminiClient(
                api_key="fake-test-key",
                http_options=types.HttpOptions(base_url=server.url),
            ),
            server,
        )
    finally:
        server.stop()


@workflow.defn
class BinaryFilesWorkflow:
    """Uploads ``data`` to Files and to a file search store, then downloads a file."""

    @workflow.run
    async def run(self, data: bytes) -> dict[str, str]:
        client = TemporalAsyncClient()
        uploaded = await client.files.upload(
            file=io.BytesIO(data),
            config=types.UploadFileConfig(mime_type="application/octet-stream"),
        )
        operation = await client.file_search_stores.upload_to_file_search_store(
            file_search_store_name=STORE,
            file=io.BytesIO(data),
            config=types.UploadToFileSearchStoreConfig(
                mime_type="application/octet-stream"
            ),
        )
        downloaded = await client.files.download(file="files/generated")
        assert isinstance(downloaded, bytes)
        return {
            "file": uploaded.name or "",
            "operation": operation.name or "",
            "downloaded": downloaded.hex(),
        }


@workflow.defn
class LegacyShapedUploadWorkflow:
    """Schedules the upload activity the way the previous version did.

    The content is in the request's ``file_bytes``, as JSON text.
    """

    @workflow.run
    async def run(self, data: bytes) -> str:
        uploaded = await workflow.execute_activity(
            "gemini_files_upload",
            _GeminiUploadFileRequest(
                file_bytes=data,
                config=types.UploadFileConfig(mime_type="text/plain"),
            ),
            result_type=types.File,
            start_to_close_timeout=timedelta(seconds=30),
        )
        return uploaded.name or ""


@pytest.mark.parametrize("data", PAYLOADS)
async def test_workflow_uploads_and_downloads_exact_bytes(client: Client, data: bytes):
    """The server receives the uploaded bytes unchanged; the download comes back as bytes."""
    with _gemini_files_client() as (gemini, server):
        config = client.config()
        config["plugins"] = [GoogleGenAIPlugin(gemini)]
        new_client = Client(**config)

        async with new_worker(new_client, BinaryFilesWorkflow) as worker:
            result = await new_client.execute_workflow(
                BinaryFilesWorkflow.run,
                data,
                id=f"gemini-binary-files-{uuid.uuid4()}",
                task_queue=worker.task_queue,
                # A workflow task that fails to encode its activity input is
                # retried until this expires, so fail instead of hanging.
                execution_timeout=timedelta(seconds=30),
            )

    assert server.uploads == {"files": data, "file_search_store": data}
    assert [(r["kind"], r["size"]) for r in server.start_requests] == [
        ("files", str(len(data))),
        ("file_search_store", str(len(data))),
    ]
    assert result == {
        "file": "files/uploaded",
        "operation": "operations/store-upload",
        "downloaded": DOWNLOAD_BODY.hex(),
    }


async def test_upload_content_is_base64_in_the_request_input(client: Client):
    """The activity input carries the content as base64 text, not as UTF-8 bytes."""
    with _gemini_files_client() as (gemini, _server):
        config = client.config()
        config["plugins"] = [GoogleGenAIPlugin(gemini)]
        new_client = Client(**config)

        async with new_worker(new_client, BinaryFilesWorkflow) as worker:
            handle = await new_client.start_workflow(
                BinaryFilesWorkflow.run,
                PNG,
                id=f"gemini-binary-files-wire-{uuid.uuid4()}",
                task_queue=worker.task_queue,
                execution_timeout=timedelta(seconds=30),
            )
            await handle.result()
            inputs = {
                e.activity_task_scheduled_event_attributes.activity_type.name: list(
                    e.activity_task_scheduled_event_attributes.input.payloads
                )
                async for e in handle.fetch_history_events()
                if e.HasField("activity_task_scheduled_event_attributes")
            }

    for activity_name in ("gemini_files_upload", "gemini_file_search_stores_upload"):
        (payload,) = inputs[activity_name]
        assert payload.metadata["encoding"] == b"json/plain"
        request = json.loads(payload.data)
        assert request["file_bytes"] is None
        assert base64.b64decode(request["file_base64"]) == PNG


class _StaticCredentials(Credentials):
    def refresh(self, request: Any) -> None:
        self.token = "fake-token"


async def _run_activity(gemini: GeminiClient, activity_name: str, req: Any) -> Any:
    caller = GeminiApiCaller(gemini, credentials=_StaticCredentials())
    fn = next(a for a in caller.activities() if a.__name__ == activity_name)
    return await ActivityEnvironment().run(fn, req)


def _upload_request(data: bytes) -> _GeminiUploadFileRequest:
    return _GeminiUploadFileRequest(
        file_base64=base64.b64encode(data).decode("ascii"),
        config=types.UploadFileConfig(mime_type="application/octet-stream"),
    )


def _store_request(data: bytes) -> _GeminiUploadToFileSearchStoreRequest:
    return _GeminiUploadToFileSearchStoreRequest(
        file_search_store_name=STORE,
        file_base64=base64.b64encode(data).decode("ascii"),
        config=types.UploadToFileSearchStoreConfig(
            mime_type="application/octet-stream"
        ),
    )


@pytest.mark.parametrize("data", PAYLOADS)
async def test_files_upload_activity_sends_exact_bytes(data: bytes):
    """``gemini_files_upload`` uploads the bytes it is given, unchanged."""
    with _gemini_files_client() as (gemini, server):
        result = await _run_activity(
            gemini, "gemini_files_upload", _upload_request(data)
        )

    assert server.uploads == {"files": data}
    assert result.name == "files/uploaded"


@pytest.mark.parametrize("data", PAYLOADS)
async def test_file_search_store_upload_activity_sends_exact_bytes(data: bytes):
    """``gemini_file_search_stores_upload`` uploads the bytes it is given, unchanged."""
    with _gemini_files_client() as (gemini, server):
        result = await _run_activity(
            gemini, "gemini_file_search_stores_upload", _store_request(data)
        )

    assert server.uploads == {"file_search_store": data}
    assert result.name == "operations/store-upload"


async def test_empty_upload_is_not_mistaken_for_a_path_upload():
    """An empty stream is still an in-memory upload, not a missing file."""
    with _gemini_files_client() as (gemini, server):
        await _run_activity(gemini, "gemini_files_upload", _upload_request(b""))

    assert server.uploads == {"files": b""}


async def test_upload_activities_read_text_bytes_from_the_request_model():
    """An activity scheduled before the upgrade still uploads its bytes.

    Earlier versions carried the bytes in ``file_bytes`` as JSON text. An
    activity that was scheduled that way, and is attempted by an upgraded worker,
    has no ``file_base64``.
    """
    with _gemini_files_client() as (gemini, server):
        await _run_activity(
            gemini,
            "gemini_files_upload",
            _GeminiUploadFileRequest(
                file_bytes=ASCII,
                config=types.UploadFileConfig(mime_type="text/plain"),
            ),
        )
        await _run_activity(
            gemini,
            "gemini_file_search_stores_upload",
            _GeminiUploadToFileSearchStoreRequest(
                file_search_store_name=STORE,
                file_bytes=ASCII,
                config=types.UploadToFileSearchStoreConfig(mime_type="text/plain"),
            ),
        )

    assert server.uploads == {"files": ASCII, "file_search_store": ASCII}


async def test_worker_runs_an_upload_scheduled_in_the_previous_shape(client: Client):
    """An upload activity input with ``file_bytes``, as recorded by 7eea646, still runs."""
    with _gemini_files_client() as (gemini, server):
        config = client.config()
        config["plugins"] = [GoogleGenAIPlugin(gemini)]
        new_client = Client(**config)

        async with new_worker(new_client, LegacyShapedUploadWorkflow) as worker:
            result = await new_client.execute_workflow(
                LegacyShapedUploadWorkflow.run,
                ASCII,
                id=f"gemini-legacy-upload-{uuid.uuid4()}",
                task_queue=worker.task_queue,
                execution_timeout=timedelta(seconds=30),
            )

    assert server.uploads == {"files": ASCII}
    assert result == "files/uploaded"


async def test_files_download_activity_returns_binary_bytes():
    """``gemini_files_download`` returns the body as bytes."""
    from temporalio.google_genai._models import _GeminiDownloadFileRequest

    with _gemini_files_client() as (gemini, _server):
        result = await _run_activity(
            gemini,
            "gemini_files_download",
            _GeminiDownloadFileRequest(file="files/generated"),
        )

    assert result == DOWNLOAD_BODY


# ---------------------------------------------------------------------------
# Replay of histories recorded before this change
# ---------------------------------------------------------------------------


def _history(name: str) -> WorkflowHistory:
    return WorkflowHistory.from_json(
        "legacy-replay", (HISTORIES / f"{name}.json").read_text()
    )


async def test_replay_history_recorded_with_text_bytes_in_the_input():
    """A history from before this change still replays.

    ``legacy_ascii_files.json`` was recorded on ``7eea646`` by running
    ``BinaryFilesWorkflow`` unchanged with ``ASCII`` bytes (the only kind that
    version could upload). It has the old activity inputs, with the bytes inside
    the request as a JSON string, and a binary download result. The replay runs
    the workflow code under test against them.
    """
    await Replayer(
        workflows=[BinaryFilesWorkflow],
        plugins=[GoogleGenAIPlugin(GeminiClient(api_key="fake-test-key"))],
    ).replay_workflow(_history("legacy_ascii_files"))
