"""Provider transport replay, including HTTPS, native auth, and AWS framing."""

from __future__ import annotations

import base64
import http.client
import json
import struct
import urllib.parse
import uuid
import zlib
from pathlib import Path
from typing import Any

from temporalio.claude_agent_sdk import ClaudeAgentPlugin, ClaudeAgentSdkRunner
from temporalio.claude_agent_sdk._replay import ReplayModel, bedrock_event, events
from temporalio.client import Client
from temporalio.worker import Worker
from tests.helpers.fake_messages_api import FakeMessagesAPI, engine_env, history_of
from tests.native_batches.activities import record
from tests.native_batches.workflows import NativeWorkflow


def test_signed_thinking_and_aws_stream_framing() -> None:
    """Replayed signatures and binary framing retain the provider's original data."""
    blocks = [
        {
            "type": "thinking",
            "thinking": "accepted thought",
            "signature": "original-signature",
        },
        {"type": "redacted_thinking", "data": "opaque-native-data"},
        {
            "type": "tool_use",
            "id": "original-id",
            "name": "Read",
            "input": {"file_path": "/example"},
        },
    ]
    encoded = events(
        {"content": blocks, "usage": {"input_tokens": 0, "output_tokens": 0}}
    )
    assert any(
        e.get("delta", {}).get("signature") == "original-signature" for e in encoded
    )
    assert any(
        e.get("content_block", {}).get("data") == "opaque-native-data" for e in encoded
    )
    for event in encoded:
        frame = bedrock_event(event)
        length, headers = struct.unpack(">II", frame[:8])
        assert length == len(frame)
        assert struct.unpack(">I", frame[8:12])[0] == zlib.crc32(frame[:8])
        assert struct.unpack(">I", frame[-4:])[0] == zlib.crc32(frame[:-4])
        assert (
            json.loads(base64.b64decode(json.loads(frame[12 + headers : -4])["bytes"]))
            == event
        )


async def test_stock_engine_uses_https_replay_without_another_primary_call(
    client: Client, tmp_path: Path
) -> None:
    """TLS replay keeps the real endpoint and trusts only the private authority."""
    note = tmp_path / "note.txt"
    note.write_text("TLS native file\n")
    primary: list[dict[str, Any]] = []

    def policy(body: dict[str, Any]) -> list[dict[str, Any]]:
        primary.append(body)
        _, _, history = history_of(body)
        if not history:
            return [
                {
                    "type": "tool_use",
                    "id": "tls-original-id",
                    "name": "Read",
                    "input": {"file_path": str(note)},
                }
            ]
        assert not history[-1].is_error
        return [{"type": "text", "text": "TLS native file"}]

    authority = ReplayModel([], {}, tmp_path)
    api = FakeMessagesAPI(policy).start()
    assert api._server is not None
    api._server.socket = authority._context("127.0.0.1").wrap_socket(
        api._server.socket, server_side=True
    )
    env = {
        **engine_env(api, str(tmp_path / "cfg")),
        "ANTHROPIC_BASE_URL": api.base_url.replace("http:", "https:"),
        "NODE_EXTRA_CA_CERTS": str(authority._ca_file),
    }
    runner = ClaudeAgentSdkRunner(cwd=str(tmp_path), env=env)
    queue = "native-tls-" + uuid.uuid4().hex
    try:
        async with Worker(
            client,
            task_queue=queue,
            workflows=[NativeWorkflow],
            activities=[record],
            plugins=[ClaudeAgentPlugin(runner, heartbeat_every=0.5)],
        ):
            assert "TLS native file" in await client.execute_workflow(
                NativeWorkflow.run,
                args=["read via HTTPS", ["Read"]],
                id=queue,
                task_queue=queue,
            )
    finally:
        api.stop()
    assert len(primary) == 2
    assert api.errors == []


async def test_proxy_requires_auth_and_preserves_helper_headers(tmp_path: Path) -> None:
    """Only the private engine can replay; helper authentication is left intact."""
    seen: list[dict[str, Any]] = []

    def helper(body: dict[str, Any]) -> list[dict[str, Any]]:
        seen.append(body)
        return [{"type": "text", "text": "helper forwarded"}]

    api = FakeMessagesAPI(helper).start()
    proxy = ReplayModel(
        [
            {
                "type": "tool_use",
                "id": "accepted-id",
                "name": "Read",
                "input": {"file_path": "/example"},
            }
        ],
        {},
        tmp_path,
    )
    address = urllib.parse.urlsplit(proxy.start()["HTTP_PROXY"])
    assert address.hostname is not None
    body = {
        "model": "original-provider-model",
        "messages": [{"role": "user", "content": "original helper prompt"}],
        "tools": [
            {"name": "mcp__durable__example", "input_schema": {"type": "object"}}
        ],
    }
    target = api.base_url + "/v1/messages"
    auth = "Basic " + base64.b64encode(("tca:" + proxy.key).encode()).decode()
    try:
        conn = http.client.HTTPConnection(address.hostname, address.port)
        conn.request(
            "POST", target, json.dumps(body), {"Content-Type": "application/json"}
        )
        assert conn.getresponse().status == 407
        conn.close()
        conn = http.client.HTTPConnection(address.hostname, address.port)
        headers = {
            "Proxy-Authorization": auth,
            "Content-Type": "application/json",
            "Authorization": "Bearer original-oauth",
            "x-api-key": "original-key",
        }
        conn.request("POST", target, json.dumps(body), headers)
        response = conn.getresponse()
        assert json.loads(response.read())["content"][0]["id"] == "accepted-id"
        conn.close()
        # The next model request is an actual helper request, with unchanged body.
        conn = http.client.HTTPConnection(address.hostname, address.port)
        conn.request("POST", target, json.dumps(body), headers)
        response = conn.getresponse()
        assert "helper forwarded" in response.read().decode()
        conn.close()
        assert seen == [body]
        forwarded = {k.lower(): v for k, v in api.request_headers[-1].items()}
        assert forwarded["authorization"] == "Bearer original-oauth"
        assert forwarded["x-api-key"] == "original-key"
        assert "proxy-authorization" not in forwarded
    finally:
        await proxy.close()
        api.stop()
