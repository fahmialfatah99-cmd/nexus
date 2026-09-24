"""MCP client tests, driven against a real subprocess server fixture."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from nexuscli.mcp.client import MCPClient, MCPError, MCPManager, MCPTool  # noqa: E402
from nexuscli.tools import build_registry  # noqa: E402
from nexuscli.tools.base import ToolContext  # noqa: E402
from nexuscli.core.permissions import PermissionEngine  # noqa: E402

FIXTURE = HERE / "fixtures" / "mcp_server.py"


def make_client(**kwargs) -> MCPClient:
    return MCPClient("fixture", sys.executable, [str(FIXTURE)], timeout=15.0, **kwargs)


class MCPTestBase(unittest.TestCase):
    def setUp(self):
        self.client = make_client()
        self.assertTrue(self.client.start(), self.client.last_error)
        self.addCleanup(self.client.close)


class TestHandshake(MCPTestBase):
    def test_initialize_and_tools_list(self):
        self.assertTrue(self.client.alive)
        self.assertEqual(self.client.server_info.get("name"), "mcp-fixture")
        self.assertEqual(self.client.tool_names(), ["echo", "add", "failing"])

    def test_non_json_banner_is_tolerated(self):
        # The fixture prints a banner line before speaking JSON; the client must
        # not treat it as a protocol error.
        self.assertTrue(self.client.alive)


class TestToolCalls(MCPTestBase):
    def test_echo(self):
        result = self.client.call_tool("echo", {"text": "hello mcp"})
        self.assertFalse(result.is_error)
        self.assertEqual(result.content, "hello mcp")

    def test_multi_part_content(self):
        result = self.client.call_tool("add", {"a": 2, "b": 3})
        self.assertIn("result: 5", result.content)
        self.assertIn('"sum"', result.content)

    def test_is_error_propagates(self):
        result = self.client.call_tool("failing", {})
        self.assertTrue(result.is_error)
        self.assertIn("always fails", result.content)

    def test_unknown_tool_returns_error_not_exception(self):
        result = self.client.call_tool("nope", {})
        self.assertTrue(result.is_error)
        self.assertIn("unknown tool", result.content)

    def test_timeout_is_enforced(self):
        # A server that never answers: point at a process that just sleeps.
        client = MCPClient("slow", sys.executable, ["-c", "import time; time.sleep(30)"], timeout=1.0)
        ok = client.start()
        self.assertFalse(ok, "initialize must time out")
        self.assertIn("timed out", client.last_error)
        client.close()


class TestFailureModes(unittest.TestCase):
    def test_missing_command(self):
        client = MCPClient("nope", "/definitely/not/a/real/binary")
        self.assertFalse(client.start())
        self.assertIn("cannot start", client.last_error)
        self.assertFalse(client.alive)
        result = client.call_tool("x", {})
        self.assertTrue(result.is_error)

    def test_server_that_exits_immediately(self):
        client = MCPClient("dead", sys.executable, ["-c", "pass"], timeout=3.0)
        self.assertFalse(client.start())
        self.assertFalse(client.alive)

    def test_server_that_prints_garbage(self):
        client = MCPClient("garbage", sys.executable,
                           ["-c", "print('not json'); import time; time.sleep(3)"], timeout=2.0)
        self.assertFalse(client.start())
        self.assertIn("timed out", client.last_error)
        client.close()

    def test_close_is_idempotent(self):
        client = make_client()
        self.assertTrue(client.start())
        client.close()
        client.close()  # must not raise
        self.assertFalse(client.alive)

    def test_call_after_close(self):
        client = make_client()
        client.start()
        client.close()
        result = client.call_tool("echo", {"text": "x"})
        self.assertTrue(result.is_error)


class TestManagerAndRegistry(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(self.tmp, ignore_errors=True))

    def test_connect_all_registers_tools(self):
        registry = build_registry()
        manager = MCPManager(timeout=15.0)
        summary = manager.connect_all(
            {"fixture": {"command": sys.executable, "args": [str(FIXTURE)]}}, registry)
        self.addCleanup(manager.close_all)
        self.assertIn("3 tool(s)", summary)
        for name in ("mcp__fixture__echo", "mcp__fixture__add", "mcp__fixture__failing"):
            self.assertTrue(registry.has(name), name)
        tool = registry.get("mcp__fixture__echo")
        self.assertIsInstance(tool, MCPTool)
        self.assertEqual(tool.category, "mcp")
        self.assertTrue(tool.needs_network)
        self.assertIn("MCP fixture", tool.description)
        self.assertEqual(tool.parameters["required"], ["text"])

    def test_registered_tool_is_callable_through_the_registry(self):
        registry = build_registry()
        manager = MCPManager(timeout=15.0)
        manager.connect_all({"fixture": {"command": sys.executable, "args": [str(FIXTURE)]}}, registry)
        self.addCleanup(manager.close_all)
        ctx = ToolContext(cwd=self.tmp, workspace_root=self.tmp,
                          permissions=PermissionEngine(mode="full-auto", workspace_root=self.tmp))
        result = registry.get("mcp__fixture__add").execute({"a": 20, "b": 22}, ctx)
        self.assertIn("result: 42", result.content)

    def test_disabled_server_is_skipped(self):
        registry = build_registry()
        manager = MCPManager(timeout=15.0)
        summary = manager.connect_all(
            {"fixture": {"command": sys.executable, "args": [str(FIXTURE)], "disabled": True}}, registry)
        self.addCleanup(manager.close_all)
        self.assertEqual(summary, "")
        self.assertFalse(registry.has("mcp__fixture__echo"))

    def test_broken_server_config_reported(self):
        registry = build_registry()
        manager = MCPManager(timeout=3.0)
        summary = manager.connect_all({"bad": {"command": "/no/such/binary"},
                                       "worse": "not-a-dict"}, registry)
        self.addCleanup(manager.close_all)
        self.assertIn("failed", summary)
        self.assertEqual(manager.clients, [])

    def test_confirmation_request_is_produced(self):
        registry = build_registry()
        manager = MCPManager(timeout=15.0)
        manager.connect_all({"fixture": {"command": sys.executable, "args": [str(FIXTURE)]}}, registry)
        self.addCleanup(manager.close_all)
        ctx = ToolContext(cwd=self.tmp, workspace_root=self.tmp,
                          permissions=PermissionEngine(mode="full-auto", workspace_root=self.tmp))
        tool = registry.get("mcp__fixture__echo")
        request = tool.confirmation({"text": "hi"}, ctx)
        self.assertIsNotNone(request)
        self.assertEqual(request.risk, "elevated")
        self.assertIn("MCP", request.title)


if __name__ == "__main__":
    unittest.main(verbosity=2)
