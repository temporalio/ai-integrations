"""A deterministic local stand-in for the Anthropic Messages API.

The real Claude Code engine (bundled in claude-agent-sdk) talks to it through
``ANTHROPIC_BASE_URL``, so tests exercise the real engine with no account and no
network. It enforces the API's tool_use / tool_result pairing rules and records any
request that breaks them in ``errors``.
"""

from __future__ import annotations

import itertools
import json
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from temporalio.claude_agent_sdk.testing import Final, HistoryItem, Policy

PREFIX = "mcp__durable__"
ENGINE_TOOLS = frozenset(
    {"Agent", "Bash", "Edit", "Glob", "PowerShell", "Read", "Write"}
)
"""Claude Code tools a scripted policy may call by name (MCP tools: ``mcp__...``)."""
Decide = Callable[[dict[str, Any]], list[dict[str, Any]]]


def _blocks(message: dict[str, Any] | None) -> list[dict[str, Any]]:
    if message and isinstance(message.get("content"), list):
        return message["content"]
    return []


def validate(body: dict[str, Any]) -> str | None:
    """Return the problem if the request breaks the real API's tool pairing rules."""
    msgs = [
        m for m in body.get("messages", []) if m.get("role") in ("user", "assistant")
    ]
    for i, m in enumerate(msgs):
        if m["role"] == "assistant":
            uses = [b.get("id") for b in _blocks(m) if b.get("type") == "tool_use"]
            nxt = msgs[i + 1] if i + 1 < len(msgs) else None
            if uses and nxt is not None:
                got = {
                    b.get("tool_use_id")
                    for b in _blocks(nxt)
                    if b.get("type") == "tool_result"
                }
                missing = [u for u in uses if u not in got]
                if nxt["role"] != "user" or missing:
                    return (
                        "tool_use ids were found without tool_result blocks immediately "
                        f"after: {missing or uses}"
                    )
        else:
            results = [b for b in _blocks(m) if b.get("type") == "tool_result"]
            if not results:
                continue
            prev = msgs[i - 1] if i > 0 else None
            prev_uses = (
                {b.get("id") for b in _blocks(prev) if b.get("type") == "tool_use"}
                if prev is not None and prev["role"] == "assistant"
                else set()
            )
            orphans = [
                b.get("tool_use_id")
                for b in results
                if b.get("tool_use_id") not in prev_uses
            ]
            if orphans:
                return f"unexpected tool_use_id in tool_result blocks: {orphans}"
            kinds = [b.get("type") for b in _blocks(m)]
            first_other = next(
                (k for k, kind in enumerate(kinds) if kind != "tool_result"), None
            )
            if first_other is not None and "tool_result" in kinds[first_other:]:
                return "tool_result blocks must come first in the user message content"
    return None


def _text_of(content: Any) -> str:
    if isinstance(content, list):
        return " ".join(x.get("text", "") for x in content if isinstance(x, dict))
    return str(content)


def history_of(
    body: dict[str, Any],
) -> tuple[dict[str, Any], list[str], list[HistoryItem]]:
    """Return (tool_use id -> (name, input), user texts, finished calls) from a request."""
    uses: dict[str, tuple[str, dict[str, Any]]] = {}
    texts: list[str] = []
    history: list[HistoryItem] = []
    for message in body.get("messages", []):
        content = message.get("content")
        if isinstance(content, str):
            if message.get("role") == "user":
                texts.append(content)
            continue
        for block in content or []:
            kind = block.get("type")
            if kind == "text" and message.get("role") == "user":
                texts.append(block.get("text", ""))
            elif kind == "tool_use":
                uses[block["id"]] = (block["name"], block.get("input", {}))
            elif kind == "tool_result":
                name, args = uses.get(block.get("tool_use_id"), ("?", {}))
                raw = _text_of(block.get("content"))
                try:  # the engine may append a <system-reminder>; read only the JSON
                    value, end = json.JSONDecoder().raw_decode(raw.strip())
                    suffix = raw.strip()[end:].strip()
                    if suffix and not suffix.startswith("<system-reminder>"):
                        value = raw
                except ValueError:
                    value = raw
                history.append(
                    HistoryItem(
                        str(block.get("tool_use_id")),
                        name.removeprefix(PREFIX),
                        args,
                        value,
                        bool(block.get("is_error")),
                    )
                )
    return uses, texts, history


class FakeMessagesAPI:
    """A local Messages API server that answers with scripted content blocks."""

    def __init__(
        self,
        decide: Decide,
        *,
        strict: bool = True,
        helper_decide: Decide | None = None,
    ) -> None:
        """Create the server (call :meth:`start`).

        Args:
            decide: Returns the assistant content blocks for a request that offers
                durable tools. Other requests (the engine's side calls) get "ok".
            strict: Reject requests that break the tool pairing rules with a 400.
            helper_decide: Optional deterministic response for helper model requests.
        """
        self.decide = decide
        self.helper_decide = helper_decide
        self.strict = strict
        self.fail_status: int | None = None
        """When set, requests that offer durable tools get this HTTP error."""
        self.fail_message = "rejected"
        """The error message sent with ``fail_status``."""
        self.errors: list[str] = []
        self.requests: list[dict[str, Any]] = []
        self.request_headers: list[dict[str, str]] = []
        self._ids = itertools.count(1)
        self._server: ThreadingHTTPServer | None = None

    def next_id(self, prefix: str) -> str:
        """A fresh id such as ``toolu_mock0001``."""
        return f"{prefix}{next(self._ids):04d}"

    @property
    def base_url(self) -> str:
        """The URL to put in ``ANTHROPIC_BASE_URL``."""
        assert self._server is not None, "call start() first"
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    def start(self) -> FakeMessagesAPI:
        """Start serving in a background thread."""
        api = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: Any) -> None:
                del format, args

            def send_json(self, payload: dict[str, Any], status: int = 200) -> None:
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self) -> None:
                self.send_json({})

            def do_POST(self) -> None:
                length = int(self.headers.get("content-length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                if "count_tokens" in self.path:
                    return self.send_json({"input_tokens": 10})
                if not self.path.startswith("/v1/messages"):
                    return self.send_json({})
                api.requests.append(body)
                api.request_headers.append(dict(self.headers.items()))
                problem = validate(body) if api.strict else None
                if problem is not None:
                    api.errors.append(problem)
                    error = {"type": "invalid_request_error", "message": problem}
                    return self.send_json({"type": "error", "error": error}, status=400)
                tools = [str(t.get("name", "")) for t in body.get("tools") or []]
                durable = any(t.startswith(PREFIX) for t in tools)
                if durable and api.fail_status is not None:
                    error = {
                        "type": "invalid_request_error",
                        "message": api.fail_message,
                    }
                    return self.send_json(
                        {"type": "error", "error": error}, status=api.fail_status
                    )
                blocks = (
                    api.decide(body)
                    if durable
                    else api.helper_decide(body)
                    if api.helper_decide is not None
                    else [{"type": "text", "text": "ok"}]
                )
                api.write(self, body, blocks)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return self

    def stop(self) -> None:
        """Stop the server."""
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None

    def write(
        self, handler: Any, body: dict[str, Any], blocks: list[dict[str, Any]]
    ) -> None:
        """Answer as server-sent events (or one JSON message when not streaming)."""
        stop = (
            "tool_use" if any(b["type"] == "tool_use" for b in blocks) else "end_turn"
        )
        model = body.get("model", "claude-fake")
        if not body.get("stream"):
            usage = {"input_tokens": 12, "output_tokens": 20}
            return handler.send_json(
                {
                    "id": self.next_id("msg_fake"),
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": blocks,
                    "stop_reason": stop,
                    "stop_sequence": None,
                    "usage": usage,
                }
            )
        usage = {
            "input_tokens": 12,
            "output_tokens": 1,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        }
        events: list[dict[str, Any]] = [
            {
                "type": "message_start",
                "message": {
                    "id": self.next_id("msg_fake"),
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": usage,
                },
            }
        ]
        for i, block in enumerate(blocks):
            start: dict[str, Any]
            delta: dict[str, Any] = {}
            if block["type"] == "text":
                start = {"type": "text", "text": ""}
                delta = {"type": "text_delta", "text": block["text"]}
            elif block["type"] in ("tool_use", "server_tool_use"):
                start = {
                    "type": block["type"],
                    "id": block["id"],
                    "name": block["name"],
                    "input": {},
                }
                delta = {
                    "type": "input_json_delta",
                    "partial_json": json.dumps(block["input"]),
                }
            else:
                start = block
            events.append(
                {"type": "content_block_start", "index": i, "content_block": start}
            )
            if block["type"] in ("text", "tool_use", "server_tool_use"):
                events.append(
                    {"type": "content_block_delta", "index": i, "delta": delta}
                )
            events.append({"type": "content_block_stop", "index": i})
        events.append(
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop, "stop_sequence": None},
                "usage": {"output_tokens": 20},
            }
        )
        events.append({"type": "message_stop"})
        handler.send_response(200)
        handler.send_header("content-type", "text/event-stream")
        handler.send_header("connection", "close")
        handler.end_headers()
        for event in events:
            handler.wfile.write(
                f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode()
            )
            handler.wfile.flush()
        handler.close_connection = True

    def tool_use(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """A tool_use block for a durable tool."""
        return {
            "type": "tool_use",
            "id": self.next_id("toolu_fake"),
            "name": PREFIX + name,
            "input": args,
        }

    def call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """A tool_use block: a Claude Code tool or an MCP tool by its own name, else a
        durable tool."""
        if name in ENGINE_TOOLS or name.startswith("mcp__"):
            return {
                "type": "tool_use",
                "id": self.next_id("toolu_engine"),
                "name": name,
                "input": args,
            }
        return self.tool_use(name, args)


def policy_decider(api_ref: list[FakeMessagesAPI], policy: Policy) -> Decide:
    """Play a scripted policy (the same kind ``ScriptedClaude`` uses) through the real engine."""

    def decide(body: dict[str, Any]) -> list[dict[str, Any]]:
        _, texts, history = history_of(body)
        # Like ScriptedClaude: the latest user message, not the engine's reminders.
        said = [t for t in texts if not t.lstrip().startswith("<system-reminder>")]
        action = policy(said[-1] if said else "", history)
        if isinstance(action, Final):
            return [{"type": "text", "text": action.text}]
        calls = action if isinstance(action, list) else [action]
        return [api_ref[0].call(call.name, call.input) for call in calls]

    return decide


def start_with_policy(policy: Policy) -> FakeMessagesAPI:
    """Start a server that plays ``policy``."""
    ref: list[FakeMessagesAPI] = []
    api = FakeMessagesAPI(policy_decider(ref, policy))
    ref.append(api)
    return api.start()


def engine_env(api: FakeMessagesAPI, config_dir: str) -> dict[str, str]:
    """Environment that points the engine at the fake API, offline."""
    return {
        "ANTHROPIC_BASE_URL": api.base_url,
        "ANTHROPIC_API_KEY": "sk-ant-fake-not-real",
        "DISABLE_TELEMETRY": "1",
        "DISABLE_ERROR_REPORTING": "1",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "CLAUDE_CONFIG_DIR": config_dir,
        "NO_PROXY": "127.0.0.1,localhost",
        "no_proxy": "127.0.0.1,localhost",
    }
