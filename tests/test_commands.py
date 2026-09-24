"""Slash-command tests: every command is executed for real against an App.

This suite exists because a manual smoke test of all 48 commands found three
bugs that unit tests missed -- ``/config`` crashed on a two-token value, left
settings half-mutated when validation failed, and silently accepted unknown keys.
Commands are the surface users touch most, so they are all driven here.
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from nexuscli.app import App, EXIT, _get_path, _setting_exists  # noqa: E402
from nexuscli.core.config import Settings  # noqa: E402
from nexuscli.ui.render import Renderer  # noqa: E402
from nexuscli.ui.theme import Style  # noqa: E402


class CapturingRenderer(Renderer):
    """A renderer that records everything instead of writing to a terminal."""

    def __init__(self, style):
        stream = io.StringIO()
        super().__init__(style, stream=stream, err_stream=stream, live=True,
                         spinner_enabled=False, quiet=False, input_fn=lambda *_: "n")
        self.buffer = stream

    def write(self, text: str) -> None:
        self.buffer.write(text)

    @property
    def text(self) -> str:
        return self.buffer.getvalue()


class AppTestBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.work = Path(self._tmp.name) / "proj"
        self.work.mkdir()
        # A project marker keeps find_project_root() from walking above the
        # fixture directory (otherwise an unrelated .nexus higher up would win).
        (self.work / ".git").mkdir()
        (self.work / "src").mkdir()
        (self.work / "src" / "calc.py").write_text("def div(a, b):\n    return a / b\n")
        (self.work / "README.md").write_text("# demo project\n")
        settings = Settings()
        settings.default_provider = "mock"
        settings.default_model = "mock-1"
        settings.approval_mode = "full-auto"
        self.style = Style.create("dark", enabled=False, width=100)
        self.renderer = CapturingRenderer(self.style)
        self.app = App(settings=settings, cwd=self.work, quiet=False, renderer=self.renderer,
                       enable_plugins=False, enable_mcp=False)
        self.addCleanup(self._cleanup)

    def _cleanup(self):
        try:
            self.app.shutdown()
        finally:
            self._tmp.cleanup()

    def run_command(self, line: str):
        buf = io.StringIO()
        with redirect_stdout(buf):
            keep_going = self.app.handle_command(line)
        return keep_going, self.renderer.text + buf.getvalue()

    def reset_output(self):
        self.renderer.buffer.truncate(0)
        self.renderer.buffer.seek(0)


class TestCommandSurface(AppTestBase):
    def test_every_command_is_registered_and_dispatchable(self):
        self.assertGreaterEqual(len(self.app.commands), 45)
        for name, command in self.app.commands.items():
            with self.subTest(command=name):
                self.assertTrue(command.help)
                self.assertTrue(command.category)
                self.assertTrue(callable(command.handler))
                self.assertEqual(command.name, name)

    def test_aliases_resolve(self):
        for alias, target in (("?", "help"), ("q", "exit"), ("m", "model"), ("st", "status"),
                              ("reset", "clear"), ("temp", "temperature")):
            found = None
            for command in self.app.commands.values():
                if alias in command.aliases:
                    found = command.name
            self.assertEqual(found, target, alias)

    def test_unknown_command_is_helpful_not_fatal(self):
        keep, out = self.run_command("/definitely-not-a-command")
        self.assertTrue(keep)
        self.assertIn("Unknown command", out)
        self.assertIn("/help", out)

    def test_every_command_runs_without_crashing(self):
        """The core regression guard: no command may raise out of handle_command."""
        script = [
            "/help", "/help swarm", "/help keys", "/help tools", "/help modes",
            "/status", "/model", "/models mock", "/provider mock", "/providers",
            "/temperature 0.4", "/temperature", "/reasoning low", "/reasoning",
            "/failover", "/failover mock:mock-1", "/mode", "/plan", "/plan",
            "/tools", "/tool off bash", "/tool on bash", "/tool off nope",
            "/allow bash:git", "/deny bash:rm", "/rules", "/readonly", "/readonly",
            "/context", "/add src/calc.py", "/memory", "/remember use four spaces",
            "/project", "/usage", "/sessions", "/board", "/agents", "/cast",
            "/cast architect tester", "/swarm-mode", "/swarm-mode debate", "/swarm-mode hive",
            "/swarm-mode nope", "/rewind 1", "/compact", "/diff", "/undo",
            "/config", "/config ui.theme", "/config ui.theme dark", "/config nope 1",
            "/auth", "/mcp", "/keys", "/log 5", "/about", "/again", "/clear",
        ]
        for line in script:
            with self.subTest(command=line):
                keep, out = self.run_command(line)
                self.assertTrue(keep, f"{line} unexpectedly asked the REPL to exit")
                self.assertNotIn("Traceback", out)
                self.assertNotIn("unexpected error", out)

    def test_exit_command_stops_the_repl(self):
        keep, _ = self.run_command("/exit")
        self.assertFalse(keep)
        for alias in ("/quit", "/q"):
            keep, _ = self.run_command(alias)
            self.assertFalse(keep, alias)


class TestConfigCommand(AppTestBase):
    def test_two_token_value_does_not_crash(self):
        _, out = self.run_command("/config ui.theme light")
        self.assertNotIn("Traceback", out)
        self.assertEqual(_get_path(self.app.settings, "ui.theme"), "light")

    def test_values_with_spaces_are_preserved(self):
        self.run_command("/config default_model groq:llama-3.3-70b-versatile")
        self.assertEqual(_get_path(self.app.settings, "default_model"),
                         "groq:llama-3.3-70b-versatile")

    def test_invalid_value_leaves_settings_untouched(self):
        before = self.app.settings.to_dict()
        _, out = self.run_command("/config approval_mode bogus")
        self.assertIn("Cannot set", out)
        self.assertEqual(self.app.settings.to_dict(), before)

    def test_out_of_range_values_are_rejected(self):
        for line in ("/config temperature 99", "/config swarm.max_parallel 0",
                     "/config compaction.threshold 5", "/config max_turns -3"):
            with self.subTest(line=line):
                before = self.app.settings.to_dict()
                _, out = self.run_command(line)
                self.assertIn("Cannot set", out)
                self.assertEqual(self.app.settings.to_dict(), before)

    def test_unknown_keys_are_rejected_not_silently_accepted(self):
        before = self.app.settings.to_dict()
        for line in ("/config not.a.key x", "/config ui.nonexistent 1",
                     "/config providers.openai.api_kye z"):
            with self.subTest(line=line):
                _, out = self.run_command(line)
                self.assertIn("Unknown setting", out)
        self.assertEqual(self.app.settings.to_dict(), before)

    def test_open_mappings_are_settable(self):
        self.run_command("/config providers.groq.api_key gsk-1")
        self.assertEqual(self.app.settings.providers["groq"].api_key, "gsk-1")
        self.run_command("/config mcp.servers.fs.command npx")
        self.assertEqual(self.app.settings.mcp["servers"]["fs"]["command"], "npx")
        self.run_command("/config search.backend brave")
        self.assertEqual(self.app.settings.search["backend"], "brave")

    def test_list_values_accept_json_and_spaces(self):
        self.run_command('/config swarm.cast ["architect","tester"]')
        self.assertEqual(self.app.settings.swarm.cast, ["architect", "tester"])
        self.run_command("/config swarm.cast architect tester docs")
        self.assertEqual(self.app.settings.swarm.cast, ["architect", "tester", "docs"])

    def test_get_and_list(self):
        _, out = self.run_command("/config ui.theme")
        self.assertIn("ui.theme", out)
        _, out = self.run_command("/config")
        self.assertIn("default_provider", out)

    def test_setting_exists_helper(self):
        settings = Settings()
        for good in ("ui.theme", "swarm.max_parallel", "providers.openai.api_key",
                     "providers.x.extra.foo", "mcp.servers.fs", "search.backend",
                     "compaction.threshold", "permissions.deny"):
            self.assertTrue(_setting_exists(settings, good), good)
        for bad in ("not.a.key", "ui.nope", "swarm.bogus", "providers.openai.api_kye",
                    "providers.openai.a.b.c.d"):
            self.assertFalse(_setting_exists(settings, bad), bad)


class TestFileCommands(AppTestBase):
    def test_add_attaches_file_contents(self):
        _, out = self.run_command("/add src/calc.py")
        self.assertIn("Attached", out)
        joined = "\n".join(m.text for m in self.app.agent.history if m.role == "user")
        self.assertIn("def div", joined)

    def test_add_missing_path(self):
        _, out = self.run_command("/add nope.py")
        self.assertIn("Not found", out)

    def test_remember_writes_project_memory(self):
        self.run_command("/remember always use four spaces")
        memory = self.work / ".nexus" / "MEMORY.md"
        self.assertTrue(memory.is_file())
        self.assertIn("always use four spaces", memory.read_text())
        self.assertIn("always use four spaces", self.app.memory_text)

    def test_undo_and_diff_without_changes(self):
        _, out = self.run_command("/undo")
        self.assertIn("No checkpoints", out)
        self.reset_output()
        _, out = self.run_command("/diff")
        self.assertIn("No files modified", out)

    def test_undo_restores_a_real_change(self):
        self.app.registry.get("write_file").execute(
            {"path": "src/calc.py", "content": "REWRITTEN\n"}, self.app.services.tool_context("main"))
        self.assertEqual((self.work / "src" / "calc.py").read_text(), "REWRITTEN\n")
        self.app.session.note_touched([str(self.work / "src" / "calc.py")])
        _, out = self.run_command("/undo")
        self.assertIn("Restored", out)
        self.assertIn("def div", (self.work / "src" / "calc.py").read_text())

    def test_project_command_rebuilds_context(self):
        _, out = self.run_command("/project")
        self.assertTrue("rebuilt" in out or "No project context" in out, out[-300:])

    def test_context_reports_sizes(self):
        _, out = self.run_command("/context")
        self.assertIn("system prompt", out)
        self.assertIn("total (est.)", out)

    def test_export_writes_a_transcript(self):
        self.app.session.add_message(__import__("nexuscli.providers.base", fromlist=["Message"])
                                     .Message.user("hello transcript"))
        target = Path(self._tmp.name) / "out.md"
        _, out = self.run_command(f"/export {target}")
        self.assertIn("exported", out)
        self.assertIn("hello transcript", target.read_text())


class TestModeCommands(AppTestBase):
    def test_mode_switch(self):
        self.run_command("/mode yolo")
        self.assertEqual(self.app.permissions.mode, "yolo")
        self.run_command("/mode suggest")
        self.assertEqual(self.app.permissions.mode, "suggest")

    def test_invalid_mode(self):
        _, out = self.run_command("/mode nonsense")
        self.assertIn("Unknown mode", out)
        self.assertEqual(self.app.permissions.mode, "full-auto")

    def test_readonly_toggle(self):
        self.run_command("/readonly")
        self.assertEqual(self.app.permissions.mode, "read-only")
        self.run_command("/readonly")
        self.assertEqual(self.app.permissions.mode, "auto-edit")

    def test_plan_mode_toggle(self):
        self.assertFalse(self.app.plan_mode)
        self.run_command("/plan")
        self.assertTrue(self.app.plan_mode)
        self.run_command("/plan")
        self.assertFalse(self.app.plan_mode)

    def test_allow_and_deny_rules(self):
        self.run_command("/allow bash:git")
        self.run_command("/deny bash:rm -rf")
        self.assertIn("bash:git", self.app.permissions.rules.allow)
        self.assertTrue(any(r.startswith("bash:rm") for r in self.app.permissions.rules.deny))
        _, out = self.run_command("/rules")
        self.assertIn("permission rules", out)

    def test_tool_toggle(self):
        self.run_command("/tool off bash")
        self.assertNotIn("bash", [t.name for t in self.app.registry.enabled_tools()])
        self.run_command("/tool on bash")
        self.assertIn("bash", [t.name for t in self.app.registry.enabled_tools()])
        _, out = self.run_command("/tool off nope")
        self.assertIn("No such tool", out)


class TestModelCommands(AppTestBase):
    def test_model_switch(self):
        self.run_command("/model mock:mock-1")
        self.assertEqual(self.app.settings.default_model, "mock-1")
        self.assertEqual(self.app.agent.options.model_spec, "mock-1")

    def test_model_without_credentials_is_reported_cleanly(self):
        import os
        from unittest import mock

        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MISTRAL_API_KEY", None)
            _, out = self.run_command("/model mistral:mistral-large-latest")
        self.assertIn("API key", out, out[-400:])
        self.assertNotIn("Traceback", out)

    def test_temperature_and_reasoning(self):
        self.run_command("/temperature 0.9")
        self.assertEqual(self.app.settings.temperature, 0.9)
        self.assertEqual(self.app.agent.options.temperature, 0.9)
        _, out = self.run_command("/temperature abc")
        self.assertIn("Usage", out)
        _, out = self.run_command("/temperature 5")
        self.assertIn("between", out)
        self.run_command("/reasoning high")
        self.assertEqual(self.app.agent.options.reasoning_effort, "high")
        self.run_command("/reasoning off")
        self.assertIsNone(self.app.agent.options.reasoning_effort)

    def test_failover_chain(self):
        self.run_command("/failover mock:mock-1 mock:mock-mini")
        self.assertEqual(self.app.settings.failover, ["mock:mock-1", "mock:mock-mini"])
        _, out = self.run_command("/failover")
        self.assertIn("failover chain", out)

    def test_provider_switch(self):
        self.run_command("/provider mock")
        self.assertEqual(self.app.settings.default_provider, "mock")
        _, out = self.run_command("/provider nope")
        self.assertIn("Unknown provider", out)


class TestSwarmCommands(AppTestBase):
    def test_cast_and_mode(self):
        self.run_command("/cast architect tester")
        self.assertEqual(self.app.settings.swarm.cast, ["architect", "tester"])
        self.run_command("/swarm-mode pipeline")
        self.assertEqual(self.app.settings.swarm.mode, "pipeline")
        _, out = self.run_command("/swarm-mode nope")
        self.assertIn("Unknown swarm mode", out)

    def test_agents_lists_personas(self):
        _, out = self.run_command("/agents")
        for persona in ("orchestrator", "architect", "reviewer", "critic"):
            self.assertIn(persona, out)

    def test_board_empty_message(self):
        _, out = self.run_command("/board")
        self.assertIn("empty", out)

    def test_swarm_requires_objective(self):
        _, out = self.run_command("/swarm")
        self.assertIn("Usage", out)

    def test_debate_requires_question(self):
        _, out = self.run_command("/debate")
        self.assertIn("Usage", out)

    def test_agent_requires_two_arguments(self):
        _, out = self.run_command("/agent reviewer")
        self.assertIn("Usage", out)


class TestConversationCommands(AppTestBase):
    def test_clear_resets_history(self):
        self.app.agent.history.append(__import__("nexuscli.providers.base", fromlist=["Message"])
                                      .Message.user("x"))
        self.run_command("/clear")
        self.assertEqual(self.app.agent.history, [])

    def test_rewind_and_again_without_history(self):
        _, out = self.run_command("/rewind 5")
        self.assertIn("Nothing to rewind", out)
        self.reset_output()
        _, out = self.run_command("/again")
        self.assertIn("No previous request", out)

    def test_compact_without_history(self):
        _, out = self.run_command("/compact")
        self.assertIn("Nothing to compact", out)

    def test_help_topics(self):
        for topic in ("", "swarm", "keys", "tools", "modes"):
            with self.subTest(topic=topic):
                keep, out = self.run_command(f"/help {topic}".strip())
                self.assertTrue(keep)
                self.assertTrue(out.strip())

    def test_usage_and_status(self):
        _, out = self.run_command("/usage")
        self.assertIn("requests", out)
        self.reset_output()
        _, out = self.run_command("/status")
        for needle in ("version", "model", "approval", "workspace", "session", "tools"):
            self.assertIn(needle, out)

    def test_about_and_keys_and_log(self):
        _, out = self.run_command("/about")
        self.assertIn("1.0.0", out)
        self.reset_output()
        _, out = self.run_command("/keys")
        self.assertIn("Tab", out)
        self.reset_output()
        _, out = self.run_command("/log 3")
        self.assertIn("nexus.log", out)


class TestInputHandling(AppTestBase):
    def test_mention_expansion_file(self):
        cleaned, blocks = self.app.expand_mentions("fix @src/calc.py please")
        self.assertEqual(len(blocks), 1)
        self.assertIn("def div", blocks[0])
        self.assertIn("@src/calc.py", cleaned)

    def test_mention_expansion_directory(self):
        cleaned, blocks = self.app.expand_mentions("look at @src")
        self.assertIn("Directory listing of src", blocks[0])
        self.assertIn("calc.py", blocks[0])

    def test_mention_missing_file_is_left_alone(self):
        cleaned, blocks = self.app.expand_mentions("see @nope.txt")
        self.assertEqual(blocks, [])
        self.assertIn("@nope.txt", cleaned)

    def test_no_mentions(self):
        cleaned, blocks = self.app.expand_mentions("just a question")
        self.assertEqual(blocks, [])
        self.assertEqual(cleaned, "just a question")

    def test_email_addresses_are_not_mentions(self):
        cleaned, blocks = self.app.expand_mentions("mail me at user@example.com please")
        self.assertEqual(blocks, [], "an email must not be treated as a file mention")

    def test_shell_command_runner(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.app.run_shell("echo bang-works")
        self.assertIn("bang-works", self.renderer.text + buf.getvalue())

    def test_prompt_string_reflects_mode(self):
        self.app.permissions.set_mode("yolo")
        self.assertIn("!", self.app._prompt_string())
        self.app.permissions.set_mode("read-only")
        self.assertIn("r", self.app._prompt_string())


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestInteractiveMenus(AppTestBase):
    """The clickable-menu wiring behind /model, /mode, /cast, /tools, etc."""

    def setUp(self):
        super().setUp()
        self.app._interactive_override = True
        self.picked = []

    def patch_pick(self, result):
        def fake_pick(title, items, **kw):
            self.picked.append((title, [getattr(i, "value", i) for i in items], kw))
            return result
        self.app._pick = fake_pick

    def test_model_menu_switches_model(self):
        self.patch_pick("groq:llama-3.3-70b-versatile")
        _, out = self.run_command("/model")
        self.assertEqual(self.app.settings.default_model, "llama-3.3-70b-versatile")
        self.assertEqual(self.app.settings.default_provider, "groq")
        title, values, _ = self.picked[0]
        self.assertIn("model", title.lower())
        self.assertTrue(any(str(v).startswith("groq:") for v in values))

    def test_switching_to_a_keyless_provider_warns_but_applies(self):
        import os
        from unittest import mock

        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MISTRAL_API_KEY", None)
            _, out = self.run_command("/model mistral:mistral-large-latest")
        self.assertEqual(self.app.settings.default_provider, "mistral")
        self.assertIn("not usable yet", out)
        self.assertIn("auth login", out)

    def test_menus_follow_the_ui_language_setting(self):
        """`ui.language` must reach the menus, not just the engine."""
        from nexuscli.app import App
        from nexuscli.core.config import Settings
        from nexuscli.ui.i18n import get_lang, set_lang

        before = get_lang()
        try:
            set_lang("en")
            settings = Settings()
            settings.ui.language = "id"
            App(settings=settings)
            self.assertEqual(get_lang(), "id")

            set_lang("id")
            settings.ui.language = "en"
            App(settings=settings)
            self.assertEqual(get_lang(), "en")
        finally:
            set_lang(before)

    def test_language_comes_from_the_locale_when_unset(self):
        from nexuscli.app import App
        from nexuscli.core.config import Settings
        from nexuscli.ui.i18n import get_lang, set_lang

        before = get_lang()
        try:
            set_lang("en")
            import os
            from unittest import mock

            with mock.patch.dict(os.environ, {"LANG": "id_ID.UTF-8"}, clear=False):
                settings = Settings()
                settings.ui.language = ""      # not set explicitly -> autodetect
                App(settings=settings)
                self.assertEqual(get_lang(), "id")
        finally:
            set_lang(before)

    def test_unresolvable_model_is_a_hard_error(self):
        before = self.app.settings.default_model
        _, out = self.run_command("/model nosuchprovider-at-all:xyz")
        self.assertNotIn("Traceback", out)
        # an unknown provider prefix is treated as a plain model id, so it still
        # resolves against the default provider -- what must never happen is a crash
        self.assertTrue(out.strip())

    def test_model_menu_cancel_keeps_model(self):
        before = self.app.settings.default_model
        self.patch_pick(None)
        self.run_command("/model")
        self.assertEqual(self.app.settings.default_model, before)

    def test_provider_menu_switches_provider(self):
        self.patch_pick("mock")
        self.run_command("/provider")
        self.assertEqual(self.app.settings.default_provider, "mock")

    def test_mode_menu_switches_mode(self):
        self.patch_pick("suggest")
        self.run_command("/mode")
        self.assertEqual(self.app.permissions.mode, "suggest")
        self.assertEqual(self.app.settings.approval_mode, "suggest")

    def test_mode_menu_lists_every_mode(self):
        self.patch_pick(None)
        self.run_command("/mode")
        _title, values, _ = self.picked[0]
        self.assertEqual(values, ["read-only", "suggest", "auto-edit", "full-auto", "yolo"])

    def test_cast_menu_is_multi_select(self):
        self.patch_pick(["architect", "tester"])
        self.run_command("/cast")
        self.assertEqual(self.app.settings.swarm.cast, ["architect", "tester"])
        self.assertTrue(self.picked[0][2].get("multi"))

    def test_cast_menu_empty_clears_cast(self):
        self.app.settings.swarm.cast = ["architect"]
        self.patch_pick([])
        self.run_command("/cast")
        self.assertEqual(self.app.settings.swarm.cast, [])

    def test_swarm_mode_menu(self):
        self.patch_pick("debate")
        self.run_command("/swarm-mode")
        self.assertEqual(self.app.settings.swarm.mode, "debate")

    def test_tools_picker_toggles(self):
        self.patch_pick(["bash", "delete_path"])   # items to switch OFF
        self.run_command("/tools pick")
        enabled = {t.name for t in self.app.registry.enabled_tools()}
        self.assertNotIn("bash", enabled)
        self.assertNotIn("delete_path", enabled)
        self.assertIn("read_file", enabled)

    def test_tools_picker_cancel_changes_nothing(self):
        before = {t.name for t in self.app.registry.enabled_tools()}
        self.patch_pick(None)
        self.run_command("/tools pick")
        self.assertEqual({t.name for t in self.app.registry.enabled_tools()}, before)

    def test_command_menu_runs_the_chosen_command(self):
        import builtins

        self.patch_pick("temperature")
        original = builtins.input
        builtins.input = lambda *_a, **_k: "0.77"
        try:
            self.app.cmd_menu([], "")
        finally:
            builtins.input = original
        self.assertEqual(self.app.settings.temperature, 0.77)

    def test_command_menu_category_filter(self):
        self.patch_pick(None)
        self.app.cmd_menu(["swarm"], "")
        title, values, _ = self.picked[0]
        self.assertTrue(all(str(v) in ("swarm", "swarm-mode", "cast", "agents", "agent", "debate", "board")
                            for v in values), values)

    def test_command_menu_unknown_category(self):
        from nexuscli.ui.i18n import tr

        _, out = self.run_command("/menu nope")
        # language-independent: compare against the active translation
        self.assertIn(tr("warn.unknown_category", name="nope"), out)

    def test_command_menu_cancel_does_nothing(self):
        self.patch_pick(None)
        keep, out = self.run_command("/menu")
        self.assertTrue(keep)
        self.assertNotIn("Traceback", out)

    def test_slash_alone_opens_the_menu(self):
        import builtins

        self.patch_pick(None)
        buf = io.StringIO()
        original = builtins.input
        builtins.input = lambda *_a, **_k: ""
        try:
            with redirect_stdout(buf):
                keep = self.app.handle_command("/")
        finally:
            builtins.input = original
        self.assertTrue(keep, "a bare / must not exit the REPL")
        self.assertTrue(self.picked, "a bare / must open the command menu")

    def test_session_menu_resumes(self):
        from nexuscli.providers.base import Message

        other = self.app.session_store.create(cwd=self.work)
        other.add_message(Message.user("sesi lain"))
        other.close()
        self.patch_pick(other.meta.id)
        self.run_command("/sessions")
        self.assertEqual(self.app.session.meta.id, other.meta.id)

    def test_agents_menu_runs_a_persona(self):
        import builtins

        self.patch_pick("tester")
        original = builtins.input
        builtins.input = lambda *_a, **_k: "cek test-nya"
        try:
            _, out = self.run_command("/agent")
        finally:
            builtins.input = original
        self.assertNotIn("Traceback", out)
        self.assertIn("mock echo", out)

    def test_menus_are_skipped_when_not_interactive(self):
        self.app._interactive_override = False
        called = []
        self.app._pick = lambda *a, **k: called.append(a) or None
        self.run_command("/mode")
        self.assertEqual(called, [], "non-interactive sessions must keep the text output")
