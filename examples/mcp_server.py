#!/usr/bin/env python3
"""Minimal MCP server you can adapt (JSON-RPC 2.0 over stdio).

Register it with:
    nexus mcp add demo -- python3 examples/mcp_server.py
    nexus mcp test demo

NEXUS exposes each tool as `mcp__demo__<tool>`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

TOOLS = [
    {
        "name": "workspace_stats",
        "description": "Count files by extension under a directory.",
        "inputSchema": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Directory to scan."},
                           "limit": {"type": "integer", "description": "Max files to scan."}},
            "required": ["path"],
        },
    },
    {
        "name": "now",
        "description": "Return the current UTC timestamp.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


def send(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def workspace_stats(args: dict) -> dict:
    root = Path(args.get("path", ".")).expanduser()
    if not root.is_dir():
        return {"isError": True, "content": [{"type": "text", "text": f"not a directory: {root}"}]}
    limit = int(args.get("limit") or 5000)
    counts: dict = {}
    for i, path in enumerate(root.rglob("*")):
        if i >= limit or not path.is_file():
            continue
        ext = path.suffix.lower() or "(none)"
        counts[ext] = counts.get(ext, 0) + 1
    top = sorted(counts.items(), key=lambda kv: -kv[1])[:15]
    body = "\n".join(f"{ext:<12} {n}" for ext, n in top)
    return {"content": [{"type": "text", "text": f"{root}\n{body or '(no files)'}"}]}


def now(_args: dict) -> dict:
    import datetime

    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    return {"content": [{"type": "text", "text": stamp}]}


HANDLERS = {"workspace_stats": workspace_stats, "now": now}


def main() -> int:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        method, msg_id, params = msg.get("method"), msg.get("id"), msg.get("params") or {}
        if method == "initialize":
            send({"jsonrpc": "2.0", "id": msg_id, "result": {
                "protocolVersion": "2024-11-05", "capabilities": {"tools": {}},
                "serverInfo": {"name": "nexus-demo-mcp", "version": "1.0"}}})
        elif method == "tools/list":
            send({"jsonrpc": "2.0", "id": msg_id, "result": {"tools": TOOLS}})
        elif method == "tools/call":
            handler = HANDLERS.get(params.get("name", ""))
            if handler is None:
                send({"jsonrpc": "2.0", "id": msg_id,
                      "error": {"code": -32602, "message": f"unknown tool {params.get('name')}"}})
                continue
            try:
                result = handler(params.get("arguments") or {})
            except Exception as exc:
                result = {"isError": True, "content": [{"type": "text", "text": str(exc)}]}
            send({"jsonrpc": "2.0", "id": msg_id, "result": result})
        elif msg_id is not None:
            send({"jsonrpc": "2.0", "id": msg_id,
                  "error": {"code": -32601, "message": f"unsupported method {method}"}})
    return 0


if __name__ == "__main__":
    sys.exit(main())
