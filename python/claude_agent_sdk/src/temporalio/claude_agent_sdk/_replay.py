"""Private replay proxy preserving the engine's native provider transport."""

from __future__ import annotations

import asyncio
import base64
import datetime
import hmac
import http.client
import ipaddress
import json
import os
import secrets
import socket
import ssl
import struct
import threading
import urllib.error
import urllib.parse
import urllib.request
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID


def events(message: dict[str, Any]) -> list[dict[str, Any]]:
    """Encode native blocks, including signed thinking, as Messages events."""
    out: list[dict[str, Any]] = [
        {
            "type": "message_start",
            "message": {**message, "content": [], "stop_reason": None},
        }
    ]
    for index, block in enumerate(message["content"]):
        kind = block["type"]
        deltas: list[dict[str, Any]] = []
        start = dict(block)
        if kind == "tool_use":
            start["input"] = {}
            deltas.append(
                {"type": "input_json_delta", "partial_json": json.dumps(block["input"])}
            )
        elif kind == "text":
            start["text"] = ""
            deltas.append({"type": "text_delta", "text": block["text"]})
        elif kind == "thinking":
            start.update(thinking="", signature="")
            deltas.extend(
                [
                    {"type": "thinking_delta", "thinking": block["thinking"]},
                    {"type": "signature_delta", "signature": block["signature"]},
                ]
            )
        out.append(
            {"type": "content_block_start", "index": index, "content_block": start}
        )
        out.extend(
            {"type": "content_block_delta", "index": index, "delta": d} for d in deltas
        )
        out.append({"type": "content_block_stop", "index": index})
    out.extend(
        [
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use", "stop_sequence": None},
                "usage": {"output_tokens": 0},
            },
            {"type": "message_stop"},
        ]
    )
    return out


def bedrock_event(event: dict[str, Any]) -> bytes:
    """Wrap a Messages event in the native AWS binary EventStream framing."""
    headers = bytearray()
    for key, value in (
        (":event-type", "chunk"),
        (":content-type", "application/json"),
        (":message-type", "event"),
    ):
        name, data = key.encode(), value.encode()
        headers.extend(
            bytes([len(name)]) + name + b"\x07" + struct.pack(">H", len(data)) + data
        )
    payload = json.dumps(
        {"bytes": base64.b64encode(json.dumps(event).encode()).decode()}
    ).encode()
    prelude = struct.pack(">II", 16 + len(headers) + len(payload), len(headers))
    frame = prelude + struct.pack(">I", zlib.crc32(prelude)) + headers + payload
    return frame + struct.pack(">I", zlib.crc32(frame))


class ReplayModel:
    """Replay accepted messages while forwarding helpers with their original auth.

    Provider URLs and engine-generated authentication (including signed requests)
    are unchanged. Only this engine trusts the ephemeral local TLS authority.
    """

    def __init__(
        self, content: list[dict[str, Any]], env: dict[str, str], folder: Path
    ) -> None:
        """Cache an accepted message and retain the engine's network environment."""
        self.key = secrets.token_hex(32)
        self._content = content
        self._env = env
        self._folder = Path(folder)
        self._used = False
        self._lock = threading.RLock()
        self._server: ThreadingHTTPServer | None = None
        self._connections: set[Any] = set()
        self._contexts: dict[str, ssl.SSLContext] = {}
        self.recovery: list[dict[str, Any]] = []
        self.active_children: list[str] = []
        self._ca_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        subject = x509.Name(
            [
                x509.NameAttribute(
                    NameOID.COMMON_NAME,
                    "Temporal native replay " + secrets.token_hex(12),
                )
            ]
        )
        now = datetime.datetime.now(datetime.timezone.utc)
        self._ca = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(subject)
            .public_key(self._ca_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(minutes=5))
            .not_valid_after(now + datetime.timedelta(days=7))
            .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
            .sign(self._ca_key, hashes.SHA256())
        )
        self._ca_file = self._folder / "replay-ca.pem"
        pem = self._ca.public_bytes(serialization.Encoding.PEM)
        if env.get("NODE_EXTRA_CA_CERTS"):
            pem += Path(env["NODE_EXTRA_CA_CERTS"]).read_bytes()
        self._save(self._ca_file, pem)
        self._upstream_tls = ssl.create_default_context()
        if env.get("NODE_EXTRA_CA_CERTS"):
            self._upstream_tls.load_verify_locations(env["NODE_EXTRA_CA_CERTS"])
        if env.get("CLAUDE_CODE_CLIENT_CERT"):
            self._upstream_tls.load_cert_chain(
                env["CLAUDE_CODE_CLIENT_CERT"],
                env.get("CLAUDE_CODE_CLIENT_KEY"),
                env.get("CLAUDE_CODE_CLIENT_KEY_PASSPHRASE"),
            )
        self._opener = self._forwarder()

    @staticmethod
    def _save(path: Path, data: bytes) -> None:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)

    def _context(self, host: str) -> ssl.SSLContext:
        with self._lock:
            if host in self._contexts:
                return self._contexts[host]
            key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
            try:
                name: Any = x509.IPAddress(ipaddress.ip_address(host))
            except ValueError:
                name = x509.DNSName(host)
            now = datetime.datetime.now(datetime.timezone.utc)
            cert = (
                x509.CertificateBuilder()
                .subject_name(
                    x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, host[:64])])
                )
                .issuer_name(self._ca.subject)
                .public_key(key.public_key())
                .serial_number(x509.random_serial_number())
                .not_valid_before(now - datetime.timedelta(minutes=5))
                .not_valid_after(now + datetime.timedelta(days=7))
                .add_extension(x509.SubjectAlternativeName([name]), critical=False)
                .sign(self._ca_key, hashes.SHA256())
            )
            suffix = secrets.token_hex(8)
            cert_file, key_file = (
                self._folder / (suffix + ".pem"),
                self._folder / (suffix + ".key"),
            )
            self._save(cert_file, cert.public_bytes(serialization.Encoding.PEM))
            self._save(
                key_file,
                key.private_bytes(
                    serialization.Encoding.PEM,
                    serialization.PrivateFormat.PKCS8,
                    serialization.NoEncryption(),
                ),
            )
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(cert_file, key_file)
            self._contexts[host] = context
            return context

    def _forwarder(self) -> urllib.request.OpenerDirector:
        model = self

        class HTTP(urllib.request.HTTPHandler):
            def http_open(self, req: Any) -> Any:
                def connection(*args: Any, **kwargs: Any) -> Any:
                    conn = http.client.HTTPConnection(*args, **kwargs)
                    with model._lock:
                        model._connections.add(conn)
                    return conn

                return self.do_open(connection, req)

        class HTTPS(urllib.request.HTTPSHandler):
            def https_open(self, req: Any) -> Any:
                def connection(*args: Any, **kwargs: Any) -> Any:
                    conn = http.client.HTTPSConnection(*args, **kwargs)
                    with model._lock:
                        model._connections.add(conn)
                    return conn

                return self.do_open(connection, req, context=model._upstream_tls)

        class Redirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(
                self, req: Any, fp: Any, code: Any, msg: Any, headers: Any, newurl: Any
            ) -> None:
                return None

        proxies = {
            scheme: self._env[scheme + "_proxy"]
            if self._env.get(scheme + "_proxy")
            else self._env[scheme.upper() + "_PROXY"]
            for scheme in ("http", "https")
            if self._env.get(scheme + "_proxy")
            or self._env.get(scheme.upper() + "_PROXY")
        }
        return urllib.request.build_opener(
            urllib.request.ProxyHandler(proxies), HTTP(), HTTPS(), Redirect()
        )

    def _cached(self, body: dict[str, Any], path: str) -> list[dict[str, Any]] | None:
        if not body.get("tools") or "count_tokens" in path:
            return None
        with self._lock:
            if not self._used:
                self._used = True
                return self._content
            messages = json.dumps(body.get("messages", []), ensure_ascii=False)
            for child in reversed(self.active_children):
                cached = next((r for r in self.recovery if r["child"] == child), None)
                if cached is not None and cached["prompt"] in messages:
                    self.recovery.remove(cached)
                    return cached["blocks"]
        return None

    def start(self) -> dict[str, str]:
        """Start the private proxy and return engine-only environment overrides."""
        model = self
        authorization = (
            "Basic " + base64.b64encode(("tca:" + self.key).encode()).decode()
        )

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            authority = ""

            def log_message(self, format: str, *args: Any) -> None:
                del format, args

            def finish(self) -> None:
                try:
                    super().finish()
                finally:
                    if self.authority:
                        # CONNECT replaces the original socket. Explicitly close
                        # its TLS replacement so streaming clients receive EOF.
                        self.connection.close()
                        with model._lock:
                            model._connections.discard(self.connection)

            def _authorized(self) -> bool:
                return bool(self.authority) or hmac.compare_digest(
                    self.headers.get("Proxy-Authorization", ""), authorization
                )

            def do_CONNECT(self) -> None:
                if not self._authorized():
                    self.send_error(407)
                    return
                host = urllib.parse.urlsplit("https://" + self.path).hostname
                if host is None:
                    self.send_error(400)
                    return
                self.send_response(200, "Connection established")
                self.end_headers()
                self.wfile.flush()
                self.connection = model._context(host).wrap_socket(
                    self.connection, server_side=True
                )
                with model._lock:
                    model._connections.add(self.connection)
                self.rfile = self.connection.makefile("rb")
                self.wfile = self.connection.makefile("wb")
                self.authority = self.path
                self.close_connection = False

            def _request(self) -> None:
                if not self._authorized():
                    self.send_error(407)
                    return
                target = (
                    "https://" + self.authority + self.path
                    if self.authority
                    else self.path
                )
                if not target.startswith(("http://", "https://")):
                    self.send_error(400)
                    return
                if "chunked" in self.headers.get("Transfer-Encoding", "").lower():
                    chunks = []
                    while size := int(self.rfile.readline().split(b";", 1)[0], 16):
                        chunks.append(self.rfile.read(size))
                        self.rfile.read(2)
                    while self.rfile.readline().strip():
                        pass
                    raw = b"".join(chunks)
                else:
                    size = int(self.headers.get("Content-Length", "0"))
                    if not 0 <= size <= 1024**3:
                        self.send_error(413)
                        return
                    raw = self.rfile.read(size)
                content = None
                body: dict[str, Any] = {}
                if self.command == "POST":
                    try:
                        body = json.loads(
                            zlib.decompress(raw, 31)
                            if self.headers.get("Content-Encoding") == "gzip"
                            else raw
                        )
                        if isinstance(body, dict):
                            content = model._cached(body, self.path)
                    except (ValueError, zlib.error):
                        pass
                if content is not None:
                    message = {
                        "id": "msg_native_replay",
                        "type": "message",
                        "role": "assistant",
                        "model": body.get("model", "native-replay"),
                        "content": content,
                        "stop_reason": "tool_use",
                        "stop_sequence": None,
                        "usage": {"input_tokens": 0, "output_tokens": 0},
                    }
                    aws = "/invoke-with-response-stream" in self.path
                    stream = bool(body.get("stream")) or aws
                    self.send_response(200)
                    self.send_header(
                        "content-type",
                        "application/vnd.amazon.eventstream"
                        if aws
                        else "text/event-stream"
                        if stream
                        else "application/json",
                    )
                    self.send_header("connection", "close")
                    self.end_headers()
                    if stream:
                        for event in events(message):
                            self.wfile.write(
                                bedrock_event(event)
                                if aws
                                else f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode()
                            )
                    else:
                        self.wfile.write(json.dumps(message).encode())
                    self.wfile.flush()
                    self.close_connection = True
                    return
                headers = {
                    k: v
                    for k, v in self.headers.items()
                    if k.lower()
                    not in (
                        "proxy-authorization",
                        "proxy-connection",
                        "connection",
                        "transfer-encoding",
                        "content-length",
                    )
                }
                request = urllib.request.Request(
                    target, raw if raw else None, headers, method=self.command
                )
                response: Any
                try:
                    response = model._opener.open(request, timeout=600)
                except urllib.error.HTTPError as error:
                    response = error
                with response:
                    self.send_response(response.status)
                    for name, value in response.headers.items():
                        if name.lower() not in (
                            "connection",
                            "transfer-encoding",
                            "content-length",
                        ):
                            self.send_header(name, value)
                    self.send_header("connection", "close")
                    self.end_headers()
                    while data := response.read1(65536):
                        self.wfile.write(data)
                        self.wfile.flush()
                self.close_connection = True

            do_POST = _request
            do_GET = _request
            do_PUT = _request
            do_DELETE = _request
            do_PATCH = _request
            do_HEAD = _request

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        url = f"http://tca:{self.key}@127.0.0.1:{self._server.server_address[1]}"
        return {
            "HTTP_PROXY": url,
            "HTTPS_PROXY": url,
            "http_proxy": url,
            "https_proxy": url,
            "NO_PROXY": "",
            "no_proxy": "",
            "NODE_EXTRA_CA_CERTS": str(self._ca_file),
        }

    async def close(self) -> None:
        """Stop the proxy and close helper requests on Activity cancellation."""
        with self._lock:
            for connection in self._connections:
                try:
                    if isinstance(connection, socket.socket):
                        connection.shutdown(socket.SHUT_RDWR)
                    connection.close()
                except OSError:
                    pass
            self._connections.clear()
        if self._server is not None:
            await asyncio.to_thread(self._server.shutdown)
            self._server.server_close()
