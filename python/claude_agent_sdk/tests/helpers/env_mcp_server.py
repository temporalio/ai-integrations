"""A stdio MCP server whose one tool, ``show``, reports the server's model endpoint.

It answers the few JSON-RPC requests Claude Code sends (``initialize``,
``tools/list``, ``tools/call``), one JSON message per line.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any


def _send(message: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def main() -> None:
    """Serve until the input ends."""
    for line in sys.stdin:
        if not line.strip():
            continue
        request = json.loads(line)
        if "id" not in request:
            continue  # a notification
        result: dict[str, Any] = {}
        method = request.get("method")
        if method == "initialize":
            result = {
                "protocolVersion": request["params"].get("protocolVersion"),
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "env", "version": "1"},
            }
        elif method == "tools/list":
            schema = {"type": "object", "properties": {}}
            result = {"tools": [{"name": "show", "inputSchema": schema}]}
        elif method == "tools/call":
            seen = {
                "ANTHROPIC_BASE_URL": os.environ.get("ANTHROPIC_BASE_URL"),
                "TCA_HOOK_DIR": os.environ.get("TCA_HOOK_DIR"),
            }
            result = {"content": [{"type": "text", "text": json.dumps(seen)}]}
        _send({"jsonrpc": "2.0", "id": request["id"], "result": result})


if __name__ == "__main__":
    main()
