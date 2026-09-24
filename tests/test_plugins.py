"""Plugin loader tests (plugins are real Python files written to a temp dir)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nexuscli.plugins.loader import LOADED_MODULES, load_plugins  # noqa: E402
from nexuscli.tools import build_registry  # noqa: E402
from nexuscli.tools.base import Tool, ToolContext, ToolResult  # noqa: E402
from nexuscli.core.permissions import PermissionEngine  # noqa: E402

GOOD_PLUGIN = '''
"""A well-behaved plugin."""
from nexuscli.tools.base import Tool, ToolContext, ToolResult


class PingTool(Tool):
    name = "ping"
    description = "Return pong."
    parameters = {"type": "object", "properties": {"who": {"type": "string"}}, "required": ["who"]}
    category = "plugin"
    read_only = True

    def run(self, args, ctx: ToolContext) -> ToolResult:
        return ToolResult.ok(f"pong {args['who']}")


TOOLS = [PingTool()]

PERSONAS = [{"key": "plugged", "name": "Pluggy", "emoji": "P", "role": "the plugin persona",
             "style": "Terse.", "duties": "Ping things.", "temperature": 0.4}]

COMMANDS = [{"name": "pingcmd", "help": "ping", "category": "plugin",
             "handler": lambda args, raw: "pong"}]

REGISTERED = []


def register(registry, context):
    REGISTERED.append(context.get("registry") is registry)
'''

BROKEN_PLUGIN = '''
raise RuntimeError("this plugin is broken on purpose")
'''

BAD_CONTENT_PLUGIN = '''
TOOLS = ["not a tool", 42]
PERSONAS = [{"no_key_here": True}]
COMMANDS = [{"name": "x"}]
'''


class PluginTestBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name) / "plugins"
        self.dir.mkdir(parents=True)
        self.registry = build_registry()
        self.addCleanup(self._cleanup)

    def _cleanup(self):
        for name in list(LOADED_MODULES):
            del LOADED_MODULES[name]
        for name in [m for m in sys.modules if m.startswith("nexus_plugin_")]:
            del sys.modules[name]
        self._tmp.cleanup()

    def write_plugin(self, name: str, body: str) -> Path:
        path = self.dir / f"{name}.py"
        path.write_text(body, encoding="utf-8")
        return path


class TestLoading(PluginTestBase):
    def test_good_plugin_registers_everything(self):
        self.write_plugin("good", GOOD_PLUGIN)
        report = load_plugins(self.registry, dirs=[self.dir])
        self.assertEqual(report.names, ["good"])
        self.assertEqual(report.tools, ["ping"])
        self.assertEqual(report.personas, ["plugged"])
        self.assertEqual([c["name"] for c in report.commands], ["pingcmd"])
        self.assertEqual(report.errors, [])
        self.assertTrue(self.registry.has("ping"))
        self.assertIn("1 tool(s)", report.summary())
        self.assertIn("1 persona(s)", report.summary())
        self.assertEqual(report.failed, [])

    def test_registered_tool_is_callable(self):
        self.write_plugin("good", GOOD_PLUGIN)
        load_plugins(self.registry, dirs=[self.dir])
        tmp = Path(tempfile.mkdtemp())
        ctx = ToolContext(cwd=tmp, workspace_root=tmp,
                          permissions=PermissionEngine(mode="full-auto", workspace_root=tmp))
        result = self.registry.get("ping").execute({"who": "world"}, ctx)
        self.assertEqual(result.content, "pong world")
        self.assertFalse(result.is_error)

    def test_persona_is_usable(self):
        self.write_plugin("good", GOOD_PLUGIN)
        load_plugins(self.registry, dirs=[self.dir])
        from nexuscli.agents.persona import BUILTIN, get_persona

        self.addCleanup(BUILTIN.pop, "plugged", None)
        persona = get_persona("plugged")
        self.assertEqual(persona.name, "Pluggy")
        self.assertEqual(persona.temperature, 0.4)
        self.assertIn("Ping things", persona.system_prompt())

    def test_register_hook_is_called(self):
        self.write_plugin("good", GOOD_PLUGIN)
        load_plugins(self.registry, dirs=[self.dir])
        module = LOADED_MODULES["nexus_plugin_good"]
        self.assertEqual(module.REGISTERED, [True])

    def test_broken_plugin_is_isolated(self):
        self.write_plugin("good", GOOD_PLUGIN)
        self.write_plugin("broken", BROKEN_PLUGIN)
        report = load_plugins(self.registry, dirs=[self.dir])
        self.assertIn("good", report.names)
        self.assertEqual(report.failed, ["broken"], "a plugin that cannot import is reported as failed")
        self.assertNotIn("broken", report.names)
        self.assertEqual(len(report.errors), 1)
        self.assertIn("broken on purpose", report.errors[0])
        self.assertTrue(self.registry.has("ping"), "a broken plugin must not break the others")

    def test_invalid_content_is_reported(self):
        self.write_plugin("badcontent", BAD_CONTENT_PLUGIN)
        report = load_plugins(self.registry, dirs=[self.dir])
        self.assertEqual(report.names, ["badcontent"])
        self.assertEqual(report.tools, [])
        self.assertGreaterEqual(len(report.errors), 3)
        self.assertTrue(any("not a Tool" in e for e in report.errors))
        self.assertTrue(any("bad persona" in e for e in report.errors))
        self.assertTrue(any("bad command" in e for e in report.errors))

    def test_missing_directory_is_fine(self):
        report = load_plugins(self.registry, dirs=[self.dir / "does-not-exist"])
        self.assertEqual(report.names, [])
        self.assertEqual(report.errors, [])
        self.assertEqual(report.summary(), "no plugins found")

    def test_private_files_are_skipped(self):
        self.write_plugin("_helper", BROKEN_PLUGIN)
        report = load_plugins(self.registry, dirs=[self.dir])
        self.assertEqual(report.names, [])

    def test_no_duplicate_loading(self):
        self.write_plugin("good", GOOD_PLUGIN)
        first = load_plugins(self.registry, dirs=[self.dir])
        second = load_plugins(self.registry, dirs=[self.dir])
        self.assertEqual(first.names, ["good"])
        self.assertEqual(second.names, ["good"])  # served from sys.modules, no re-exec errors
        self.assertEqual(second.errors, [])

    def test_report_bool_and_summary(self):
        self.write_plugin("good", GOOD_PLUGIN)
        report = load_plugins(self.registry, dirs=[self.dir])
        self.assertTrue(report)
        self.assertIn("1 plugin(s)", report.summary())


if __name__ == "__main__":
    unittest.main(verbosity=2)
