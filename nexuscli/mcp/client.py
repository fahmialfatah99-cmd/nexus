"""Minimal Model Context Protocol (MCP) client over stdio.

Enough of the spec to be genuinely useful, implemented with the standard library
only (``subprocess`` + JSON-RPC 2.0 + a reader thread):

* ``initialize`` handshake, then ``notifications/initialized``
* ``tools/list`` -> each remote tool becomes a NEXUS :class:`Tool` named
  ``mcp__<server>__<tool>`` so it is unmistakable where a call is going
* ``tools/call`` with timeout, returning text content (and surfacing ``isError``)

Robustness rules: a server that never answers is timed out, not hung on; a server
that writes logs to stdout as non-JSON is tolerated; a server that dies is marked
dead and its tools report the failure instead of blocking the agent.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..core.logging_ import get_logger
from ..tools.base import ConfirmationRequest, Tool, ToolContext, ToolResult, ToolRegistry, clip

PROTOCOL_VERSION = "2024-11-05"


class MCPClient:
    def __init__(self, name: str, command: str, args: Sequence[str] = (), *, env: Optional[Dict[str, str]] = None,
                 cwd: Optional[Path] = None, timeout: float = 30.0, log: Any = None) -> None:
        self.name = name
        self.command = command
        self.args = list(args)
        self.env = {**os.environ, **(env or {})}
        self.cwd = str(cwd) if cwd else None
        self.timeout = timeout
        self.log = log or get_logger()
        self._proc: Optional[subprocess.Popen] = None
        self._queue: "queue.Queue[Optional[dict]]" = queue.Queue()
        self._reader: Optional[threading.Thread] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._id = 0
        self._lock = threading.Lock()
        self.tools: List[Dict[str, Any]] = []
        self.server_info: Dict[str, Any] = {}
        self.last_error: str = ""
        self.alive = False

    # ------------------------------------------------------------------ #
    def command_line(self) -> str:
        return " ".join([self.command, *self.args])

    def start(self) -> bool:
        try:
            self._proc = subprocess.Popen(
                [self.command, *self.args], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, bufsize=1, env=self.env, cwd=self.cwd,
                errors="replace",
            )
        except (OSError, ValueError) as exc:
            self.last_error = f"cannot start '{self.command}': {exc}"
            self.log.warning("mcp start failed", server=self.name, error=self.last_error)
            return False
        self._reader = threading.Thread(target=self._read_loop, daemon=True, name=f"mcp-{self.name}")
        self._reader.start()
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True,
                                               name=f"mcp-{self.name}-err")
        self._stderr_thread.start()
        try:
            result = self.request("initialize", {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "clientInfo": {"name": "nexus", "version": "1.0"},
            })
        except MCPError as exc:
            self.last_error = str(exc)
            self.close()
            return False
        if result is None:
            self.last_error = "server closed during initialize"
            self.close()
            return False
        self.server_info = (result or {}).get("serverInfo") or {}
        self.notify("notifications/initialized", {})
        self.alive = True
        try:
            listed = self.request("tools/list", {}) or {}
            self.tools = [t for t in (listed.get("tools") or []) if isinstance(t, dict) and t.get("name")]
        except MCPError as exc:
            self.last_error = f"tools/list failed: {exc}"
            self.tools = []
        self.log.info("mcp connected", server=self.name, tools=len(self.tools),
                      version=self.server_info.get("version", "?"))
        return True

    # ------------------------------------------------------------------ #
    def _next_id(self) -> int:
        with self._lock:
            self._id += 1
            return self._id

    def _send(self, payload: Dict[str, Any]) -> None:
        if self._proc is None or self._proc.stdin is None or self._proc.stdin.closed:
            raise MCPError(f"server '{self.name}' is not running")
        line = json.dumps(payload, ensure_ascii=False) + "\n"
        try:
            self._proc.stdin.write(line)
            self._proc.stdin.flush()
        except (OSError, ValueError) as exc:
            self.alive = False
            raise MCPError(f"cannot write to '{self.name}': {exc}") from exc

    def request(self, method: str, params: Optional[Dict[str, Any]] = None,
                *, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        msg_id = self._next_id()
        self._send({"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params or {}})
        deadline = time.monotonic() + (timeout if timeout is not None else self.timeout)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise MCPError(f"'{self.name}' timed out after {self.timeout:.0f}s on {method}")
            try:
                msg = self._queue.get(timeout=min(remaining, 1.0))
            except queue.Empty:
                if self._proc is not None and self._proc.poll() is not None:
                    self.alive = False
                    raise MCPError(f"'{self.name}' exited with code {self._proc.returncode}")
                continue
            if msg is None:
                self.alive = False
                raise MCPError(f"'{self.name}' closed the connection")
            if msg.get("id") != msg_id:
                continue  # notification or a stale reply: keep waiting
            if "error" in msg and msg["error"]:
                err = msg["error"]
                raise MCPError(f"'{self.name}' {method} error {err.get('code')}: {err.get('message')}")
            return msg.get("result")

    def notify(self, method: str, params: Optional[Dict[str, Any]] = None) -> None:
        try:
            self._send({"jsonrpc": "2.0", "method": method, "params": params or {}})
        except MCPError as exc:
            self.log.warning("mcp notify failed", server=self.name, error=str(exc))

    def _read_loop(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        try:
            for line in self._proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    # Some servers print banners/logs on stdout; ignore them.
                    self.log.debug("mcp: non-JSON stdout line", server=self.name, line=line[:160])
                    continue
                if isinstance(msg, dict):
                    self._queue.put(msg)
        except (OSError, ValueError):
            pass
        finally:
            self._queue.put(None)
            self.alive = False

    def _drain_stderr(self) -> None:
        assert self._proc is not None and self._proc.stderr is not None
        try:
            for line in self._proc.stderr:
                text = line.rstrip()
                if text:
                    self.log.debug("mcp stderr", server=self.name, line=text[:300])
        except (OSError, ValueError):
            pass

    # ------------------------------------------------------------------ #
    def tool_names(self) -> List[str]:
        return [str(t.get("name")) for t in self.tools]

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> ToolResult:
        if not self.alive:
            return ToolResult.fail(f"MCP server '{self.name}' is not running ({self.last_error or 'unknown reason'}).")
        try:
            result = self.request("tools/call", {"name": name, "arguments": arguments or {}})
        except MCPError as exc:
            self.alive = False
            return ToolResult.fail(f"MCP call '{self.name}/{name}' failed: {exc}")
        if result is None:
            return ToolResult.fail(f"MCP server '{self.name}' returned no result for '{name}'.")
        texts: List[str] = []
        for item in result.get("content") or []:
            if not isinstance(item, dict):
                continue
            kind = item.get("type")
            if kind == "text" and item.get("text"):
                texts.append(str(item["text"]))
            elif kind == "json" or "json" in item:
                payload = item.get("json", item)
                try:
                    texts.append(json.dumps(payload, ensure_ascii=False, indent=2))
                except (TypeError, ValueError):
                    texts.append(str(payload))
            elif kind == "image":
                texts.append(f"[image {item.get('mimeType', 'image')}]")
            elif kind == "resource":
                texts.append(json.dumps(item.get("resource", {}), ensure_ascii=False)[:4000])
        body = "\n".join(texts).strip() or json.dumps(result, ensure_ascii=False)[:4000]
        is_error = bool(result.get("isError"))
        return ToolResult(content=clip(body, 24_000), is_error=is_error,
                          data={"server": self.name, "tool": name})

    def close(self) -> None:
        self.alive = False
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            if proc.stdin and not proc.stdin.closed:
                proc.stdin.close()
        except OSError:
            pass
        try:
            proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                pass
        except Exception:
            pass


class MCPError(Exception):
    pass


# --------------------------------------------------------------------------- #
# Tool adapter
# --------------------------------------------------------------------------- #
class MCPTool(Tool):
    def __init__(self, client: MCPClient, spec: Dict[str, Any]) -> None:
        self.client = client
        self.remote_name = str(spec.get("name"))
        self.name = f"mcp__{client.name}__{self.remote_name}".replace("-", "_")
        raw_desc = str(spec.get("description") or "").strip()
        self.description = (f"[MCP {client.name}] {raw_desc}" if raw_desc
                            else f"[MCP {client.name}] remote tool {self.remote_name}")
        schema = spec.get("inputSchema") or spec.get("input_schema") or {}
        if not isinstance(schema, dict) or schema.get("type") != "object":
            schema = {"type": "object", "properties": schema.get("properties", {}) if isinstance(schema, dict) else {}}
        self.parameters = {
            "type": "object",
            "properties": schema.get("properties") or {},
            "required": [str(r) for r in (schema.get("required") or [])],
        }
        if schema.get("description") and not raw_desc:
            self.description += " " + str(schema["description"])
        self.category = "mcp"
        self.read_only = bool(spec.get("readOnlyHint"))
        self.needs_network = True
        self.concurrency_safe = not bool(spec.get("destructiveHint"))
        self.timeout = float(spec.get("timeout") or client.timeout)

    def confirmation(self, args: Dict[str, Any], ctx: ToolContext) -> Optional[ConfirmationRequest]:
        summary = ", ".join(f"{k}={clip(str(v), 40)}" for k, v in list(args.items())[:3])
        return ConfirmationRequest(title=f"{self.remote_name} via MCP '{self.client.name}'",
                                   detail=summary or "(no arguments)",
                                   risk="normal" if self.read_only else "elevated",
                                   key=f"{self.name}:*")

    def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        return self.client.call_tool(self.remote_name, args)


class MCPManager:
    def __init__(self, *, log: Any = None, timeout: float = 30.0) -> None:
        self.clients: List[MCPClient] = []
        self.log = log or get_logger()
        self.timeout = timeout

    def connect(self, name: str, spec: Dict[str, Any]) -> Optional[MCPClient]:
        command = spec.get("command")
        if not command:
            self.log.warning("mcp: server has no command", server=name)
            return None
        client = MCPClient(name, str(command), list(spec.get("args") or []),
                           env=dict(spec.get("env") or {}),
                           cwd=Path(spec["cwd"]) if spec.get("cwd") else None,
                           timeout=float(spec.get("timeout") or self.timeout), log=self.log)
        if not client.start():
            self.log.warning("mcp: server failed to start", server=name, error=client.last_error)
            return None
        self.clients.append(client)
        return client

    def connect_all(self, servers: Dict[str, Any], registry: ToolRegistry) -> str:
        """Connect every configured server and register its tools. Returns a summary."""
        registered = 0
        failed: List[str] = []
        for name, spec in (servers or {}).items():
            if not isinstance(spec, dict):
                failed.append(f"{name} (bad config)")
                continue
            if spec.get("disabled"):
                continue
            client = self.connect(str(name), spec)
            if client is None:
                failed.append(str(name))
                continue
            for tool_spec in client.tools:
                try:
                    registry.register(MCPTool(client, tool_spec), replace=True)
                    registered += 1
                except Exception as exc:
                    self.log.warning("mcp: cannot register tool", server=name,
                                     tool=tool_spec.get("name"), error=str(exc))
        parts = []
        if registered:
            parts.append(f"{registered} tool(s) from {len(self.clients)} server(s)")
        if failed:
            parts.append("failed: " + ", ".join(failed))
        return "; ".join(parts)

    def close_all(self) -> None:
        for client in self.clients:
            try:
                client.close()
            except Exception:
                pass
        self.clients.clear()


__all__ = ["MCPClient", "MCPManager", "MCPTool", "MCPError", "PROTOCOL_VERSION"]
