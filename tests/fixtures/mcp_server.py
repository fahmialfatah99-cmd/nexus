#!/usr/bin/env python3
"""Fixture: a minimal MCP server used by the test-suite.

Speaks JSON-RPC 2.0 over stdio: ``initialize``, ``tools/list``, ``tools/call``.
It also exercises the awkward parts a real client must survive:
  * a non-JSON banner printed on stdout before the protocol starts
  * log noise on stderr
  * an error response for an unknown tool
  * a tool that returns ``isError`` plus multi-part content
"""

from __future__ import annotations

import json
import sys

BANNER = "mcp-fixture starting up (this line is not JSON)"

TOOLS = [
    {
        "name": "echo",
        "description": "Echo the given text back.",
        "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}},
                        "required": ["text"]},
    },
    {
        "name": "add",
        "description": "Add two numbers.",
        "inputSchema": {"type": "object",
                        "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                        "required": ["a", "b"]},
    },
    {
        "name": "failing",
        "description": "Always reports an error result.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def send(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def main() -> int:
    print(BANNER, flush=True)
    log("fixture server ready")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            log(f"bad json: {line[:80]}")
            continue
        method = msg.get("method")
        msg_id = msg.get("id")
        params = msg.get("params") or {}
        if method == "initialize":
            send({"jsonrpc": "2.0", "id": msg_id, "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "mcp-fixture", "version": "0.1.0"}}})
        elif method == "notifications/initialized":
            log("initialized notification received")
        elif method == "tools/list":
            send({"jsonrpc": "2.0", "id": msg_id, "result": {"tools": TOOLS}})
        elif method == "tools/call":
            name = params.get("name")
            args = params.get("arguments") or {}
            if name == "echo":
                send({"jsonrpc": "2.0", "id": msg_id, "result": {
                    "content": [{"type": "text", "text": str(args.get("text", ""))}]}})
            elif name == "add":
                try:
                    total = float(args.get("a", 0)) + float(args.get("b", 0))
                except (TypeError, ValueError):
                    total = "error"
                send({"jsonrpc": "2.0", "id": msg_id, "result": {
                    "content": [{"type": "text", "text": f"result: {total}"},
                                {"type": "json", "json": {"sum": total}}]}})
            elif name == "failing":
                send({"jsonrpc": "2.0", "id": msg_id, "result": {
                    "isError": True,
                    "content": [{"type": "text", "text": "this tool always fails"}]}})
            else:
                send({"jsonrpc": "2.0", "id": msg_id,
                      "error": {"code": -32602, "message": f"unknown tool: {name}"}})
        elif method == "shutdown":
            send({"jsonrpc": "2.0", "id": msg_id, "result": {}})
            return 0
        elif msg_id is not None:
            send({"jsonrpc": "2.0", "id": msg_id,
                  "error": {"code": -32601, "message": f"unsupported method: {method}"}})
    return 0


if __name__ == "__main__":
    sys.exit(main())
