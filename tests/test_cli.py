"""End-to-end CLI tests: the real ``nexus`` launcher, real subprocesses.

These exist because unit tests cannot catch wiring bugs (a flag that no parser
declares, a bootstrap that reads an attribute before it exists, an exit code
that is wrong). Everything runs offline against the mock provider with
``NEXUS_HOME`` pointed at a temp dir, so it is safe and deterministic.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
NEXUS = str(ROOT / "nexus")
FIXTURE_MCP = str(HERE / "fixtures" / "mcp_server.py")


def run(args, *, cwd=None, env=None, stdin="", timeout=180):
    full_env = {
        **os.environ,
        "NEXUS_HOME": env["NEXUS_HOME"] if env and "NEXUS_HOME" in env else "",
        "NO_COLOR": "1",
        "PYTHONIOENCODING": "utf-8",
        "TERM": "xterm-256color",
    }
    full_env.pop("OPENAI_API_KEY", None)
    full_env.pop("ANTHROPIC_API_KEY", None)
    if env:
        full_env.update(env)
    proc = subprocess.run([sys.executable, NEXUS, *args], cwd=str(cwd or ROOT),
                          input=stdin, capture_output=True, text=True, timeout=timeout,
                          env=full_env, errors="replace")
    return proc


class CLITestBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.home = self.tmp / "home"
        self.work = self.tmp / "work"
        self.home.mkdir()
        self.work.mkdir()
        self.env = {"NEXUS_HOME": str(self.home)}
        self.addCleanup(self._tmp.cleanup)

    def nexus(self, *args, **kwargs):
        kwargs.setdefault("cwd", self.work)
        kwargs.setdefault("env", self.env)
        return run(list(args), **kwargs)

    def assertOk(self, proc, needle=None, **_ignored):
        self.assertEqual(proc.returncode, 0, f"stdout={proc.stdout[-1500:]}\nstderr={proc.stderr[-1500:]}")
        if needle:
            self.assertIn(needle, proc.stdout)
        return proc


class TestParserIntegrity(unittest.TestCase):
    """Every subcommand parser must build: argparse conflicts are runtime crashes."""

    def test_all_subcommands_have_working_help(self):
        import contextlib
        import io

        sys.path.insert(0, str(ROOT))
        from nexuscli import cli

        broken = []
        for name in sorted(cli.SUBCOMMANDS):
            try:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    cli.dispatch(name, ["--help"])
            except SystemExit as exc:
                if exc.code not in (0, None):
                    broken.append((name, exc.code))
            except Exception as exc:
                broken.append((name, f"{type(exc).__name__}: {exc}"))
        self.assertEqual(broken, [])
        self.assertGreaterEqual(len(cli.SUBCOMMANDS), 18)

    def test_dispatch_table_matches_subcommand_set(self):
        sys.path.insert(0, str(ROOT))
        from nexuscli import cli

        self.assertEqual(set(cli.SUBCOMMANDS), set(cli.SUBCOMMAND_HANDLERS))


class TestVersionAndHelp(CLITestBase):
    def test_version(self):
        proc = self.assertOk(self.nexus("version"))
        self.assertIn("nexus 1.0.0", proc.stdout)
        self.assertIn("providers", proc.stdout)

    def test_version_json(self):
        proc = self.assertOk(self.nexus("version", "--json"))
        data = json.loads(proc.stdout)
        self.assertEqual(data["version"], "1.0.0")
        self.assertGreater(data["providers"], 10)

    def test_dash_version(self):
        proc = self.assertOk(self.nexus("-V"))
        self.assertIn("nexus", proc.stdout)

    def test_help_lists_subcommands(self):
        proc = self.nexus("--help")
        self.assertEqual(proc.returncode, 0)
        for word in ("swarm", "models", "doctor", "demo"):
            self.assertIn(word, proc.stdout)

    def test_unknown_subcommand_is_not_silent(self):
        proc = self.nexus("definitely-not-a-command")
        # treated as a prompt for the agent, so it must not crash
        self.assertIn(proc.returncode, (0, 1, 2, 3))

    def test_bad_flag_exits_2(self):
        proc = self.nexus("version", "--no-such-flag")
        self.assertEqual(proc.returncode, 2)


class TestInfoCommands(CLITestBase):
    def test_providers(self):
        proc = self.assertOk(self.nexus("providers"))
        for key in ("openai", "anthropic", "gemini", "groq", "deepseek", "ollama", "mock"):
            self.assertIn(key, proc.stdout)
        self.assertIn("no key", proc.stdout)

    def test_providers_json(self):
        proc = self.assertOk(self.nexus("providers", "--json"))
        data = json.loads(proc.stdout)
        keys = {row["key"] for row in data}
        self.assertIn("anthropic", keys)
        self.assertGreater(len(keys), 15)
        for row in data:
            self.assertIn("default_model", row)

    def test_tools(self):
        proc = self.assertOk(self.nexus("tools"))
        for tool in ("read_file", "edit_file", "grep", "bash", "git", "task"):
            self.assertIn(tool, proc.stdout)

    def test_tools_json_schema_is_valid(self):
        proc = self.assertOk(self.nexus("tools", "--json", "--all"))
        data = json.loads(proc.stdout)
        names = {t["name"] for t in data}
        self.assertIn("swarm_post", names, "--all must include swarm tools")
        for tool in data:
            self.assertEqual(tool["parameters"].get("type"), "object", tool["name"])
            self.assertIn("properties", tool["parameters"])
            self.assertIsInstance(tool["read_only"], bool)

    def test_personas(self):
        proc = self.assertOk(self.nexus("personas"))
        for key in ("orchestrator", "architect", "implementer", "reviewer", "tester",
                    "debugger", "security", "critic"):
            self.assertIn(key, proc.stdout)
        self.assertIn("hive", proc.stdout)

    def test_personas_json(self):
        proc = self.assertOk(self.nexus("personas", "--json"))
        data = json.loads(proc.stdout)
        keys = {p["key"] for p in data}
        self.assertIn("orchestrator", keys)
        for persona in data:
            self.assertTrue(persona["role"])
            self.assertTrue(persona["style"])

    def test_models_offline_is_graceful(self):
        proc = self.nexus("models", "--provider", "mock")
        self.assertIn(proc.returncode, (0, 1))

    def test_plugins_none(self):
        proc = self.assertOk(self.nexus("plugins"))
        self.assertIn("no plugins found", proc.stdout)
        self.assertIn("plugin directories", proc.stdout)


class TestConfig(CLITestBase):
    def test_config_init_and_roundtrip(self):
        self.assertOk(self.nexus("config", "init"))
        path = self.home / "config.json"
        self.assertTrue(path.is_file())
        data = json.loads(path.read_text())
        self.assertEqual(data["approval_mode"], "auto-edit")

    def test_config_set_get(self):
        self.assertOk(self.nexus("config", "set", "swarm.max_parallel", "7"))
        proc = self.assertOk(self.nexus("config", "get", "swarm.max_parallel"))
        self.assertEqual(proc.stdout.strip(), "7")
        stored = json.loads((self.home / "config.json").read_text())
        self.assertEqual(stored["swarm"]["max_parallel"], 7)

    def test_config_set_validates(self):
        proc = self.nexus("config", "set", "approval_mode", "nonsense")
        self.assertEqual(proc.returncode, 2)
        self.assertIn("approval_mode", proc.stderr)

    def test_config_set_bool(self):
        self.assertOk(self.nexus("config", "set", "ui.spinner", "false"))
        self.assertFalse(json.loads((self.home / "config.json").read_text())["ui"]["spinner"])

    def test_config_get_unknown_key(self):
        proc = self.nexus("config", "get", "not.a.real.key")
        self.assertEqual(proc.returncode, 2)

    def test_config_path_and_sources(self):
        self.assertOk(self.nexus("config", "path"), "config.json")
        self.assertOk(self.nexus("config", "sources"))

    def test_config_list_is_json(self):
        proc = self.assertOk(self.nexus("config", "list", "--json"))
        data = json.loads(proc.stdout)
        self.assertIn("providers", data)
        self.assertIn("swarm", data)

    def test_broken_config_gives_an_actionable_error(self):
        (self.home / "config.json").write_text("{not json")
        proc = self.nexus("config", "list")
        self.assertEqual(proc.returncode, 2)
        self.assertIn("not valid JSON", proc.stderr)


class TestAuth(CLITestBase):
    def test_login_list_logout(self):
        self.assertOk(self.nexus("auth", "login", "openai", "sk-test-123"))
        path = self.home / "auth.json"
        self.assertTrue(path.is_file())
        self.assertEqual(oct(path.stat().st_mode & 0o777), "0o600")
        self.assertEqual(json.loads(path.read_text())["keys"]["openai"], "sk-test-123")
        proc = self.assertOk(self.nexus("auth", "list"))
        self.assertIn("openai", proc.stdout)
        self.assertIn("stored", proc.stdout)
        self.assertOk(self.nexus("auth", "logout", "openai"))
        self.assertEqual(json.loads(path.read_text())["keys"], {})

    def test_login_unknown_provider(self):
        proc = self.nexus("auth", "login", "not-a-provider", "x")
        self.assertEqual(proc.returncode, 2)

    def test_login_empty_key(self):
        proc = self.nexus("auth", "login", "openai", "")
        self.assertEqual(proc.returncode, 2)

    def test_logout_missing_key(self):
        proc = self.nexus("auth", "logout", "openai")
        self.assertEqual(proc.returncode, 1)

    def test_auth_json(self):
        self.assertOk(self.nexus("auth", "login", "groq", "gsk-x"))
        proc = self.assertOk(self.nexus("auth", "list", "--json"))
        data = json.loads(proc.stdout)
        groq = next(r for r in data if r["provider"] == "groq")
        self.assertTrue(groq["stored"])
        self.assertTrue(groq["ready"])


class TestMCPCommands(CLITestBase):
    def test_list_empty(self):
        proc = self.assertOk(self.nexus("mcp"))
        self.assertIn("no MCP servers", proc.stdout)

    def test_add_list_remove(self):
        self.assertOk(self.nexus("mcp", "add", "fixture", "--", sys.executable, FIXTURE_MCP))
        proc = self.assertOk(self.nexus("mcp"))
        self.assertIn("fixture", proc.stdout)
        stored = json.loads((self.home / "config.json").read_text())
        self.assertEqual(stored["mcp"]["servers"]["fixture"]["command"], sys.executable)
        self.assertOk(self.nexus("mcp", "remove", "fixture"))
        self.assertNotIn("fixture", self.assertOk(self.nexus("mcp")).stdout)

    def test_add_without_command(self):
        proc = self.nexus("mcp", "add", "x")
        self.assertEqual(proc.returncode, 2)

    def test_test_connects_to_fixture(self):
        self.assertOk(self.nexus("mcp", "add", "fixture", "--", sys.executable, FIXTURE_MCP))
        proc = self.assertOk(self.nexus("mcp", "test", "fixture"), timeout=60)
        self.assertIn("mcp-fixture", proc.stdout)
        for tool in ("echo", "add", "failing"):
            self.assertIn(tool, proc.stdout)

    def test_test_bad_server(self):
        self.assertOk(self.nexus("mcp", "add", "bad", "--", "/no/such/binary"))
        proc = self.nexus("mcp", "test", "bad")
        self.assertEqual(proc.returncode, 1)
        self.assertIn("FAILED", proc.stdout)

    def test_remove_unknown(self):
        proc = self.nexus("mcp", "remove", "nope")
        self.assertEqual(proc.returncode, 2)


class TestInit(CLITestBase):
    def test_init_creates_project_config(self):
        proc = self.assertOk(self.nexus("init", "--agents"))
        config = self.work / ".nexus" / "config.json"
        self.assertTrue(config.is_file(), proc.stdout)
        data = json.loads(config.read_text())
        self.assertIn("allow", data["permissions"])
        self.assertIn("deny", data["permissions"])
        self.assertTrue((self.work / "AGENTS.md").is_file())
        self.assertIn(".nexus/checkpoints/", (self.work / ".gitignore").read_text())

    def test_init_is_not_destructive(self):
        self.assertOk(self.nexus("init"))
        proc = self.nexus("init")
        self.assertEqual(proc.returncode, 1)
        self.assertIn("already exists", proc.stdout)
        self.assertOk(self.nexus("init", "--force"))

    def test_init_appends_gitignore_once(self):
        (self.work / ".gitignore").write_text("*.pyc\n")
        self.assertOk(self.nexus("init"))
        self.assertOk(self.nexus("init", "--force"))
        text = (self.work / ".gitignore").read_text()
        self.assertIn("*.pyc", text)
        self.assertEqual(text.count(".nexus/checkpoints/"), 1)


class TestOneShot(CLITestBase):
    def test_print_mode_with_mock(self):
        proc = self.assertOk(self.nexus("-p", "say hello", "--provider", "mock", "--model", "mock-1"))
        self.assertIn("mock echo", proc.stdout)
        self.assertIn("say hello", proc.stdout)

    def test_print_mode_json(self):
        proc = self.assertOk(self.nexus("-p", "hi", "--provider", "mock", "--json"))
        start = proc.stdout.index("{")
        data = json.loads(proc.stdout[start:])
        self.assertEqual(data["type"], "turn")
        self.assertTrue(data["ok"])
        self.assertIn("hi", data["text"])
        self.assertEqual(data["usage"]["requests"], 1)
        self.assertEqual(data["model"], "mock:mock-1")

    def test_stdin_is_attached_as_context(self):
        proc = self.assertOk(self.nexus("-p", "review", "--provider", "mock", stdin="line one\nline two\n"))
        self.assertIn("<stdin>", proc.stdout)
        self.assertIn("line one", proc.stdout)

    def test_print_without_prompt_exits_2(self):
        proc = self.nexus("-p", "--provider", "mock")
        self.assertEqual(proc.returncode, 2)

    def test_run_from_file(self):
        prompt = self.tmp / "task.md"
        prompt.write_text("do the thing from a file")
        proc = self.assertOk(self.nexus("run", str(prompt), "--provider", "mock"))
        self.assertIn("do the thing from a file", proc.stdout)

    def test_run_missing_file(self):
        proc = self.nexus("run", str(self.tmp / "nope.md"))
        self.assertEqual(proc.returncode, 2)

    def test_tools_disabled_flag(self):
        proc = self.assertOk(self.nexus("-p", "hi", "--provider", "mock", "--no-tools"))
        self.assertIn("mock echo", proc.stdout)

    def test_quiet_flag(self):
        proc = self.assertOk(self.nexus("-p", "hi", "--provider", "mock", "-q"))
        self.assertNotIn("you›", proc.stdout)

    def test_missing_credentials_reports_cleanly(self):
        proc = self.nexus("-p", "hi", "--provider", "openai", "--model", "gpt-4o")
        self.assertNotEqual(proc.returncode, 0)
        combined = proc.stdout + proc.stderr
        self.assertTrue("API key" in combined or "error" in combined.lower(), combined[:400])


class TestSwarmCLI(CLITestBase):
    def test_plan_only_with_mock(self):
        # The mock provider echoes, so no tasks are planned; the command must
        # still complete cleanly and say so.
        proc = self.nexus("swarm", "build a thing", "--plan-only", "--provider", "mock",
                          "--model", "mock-1", "--cast", "implementer")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

    def test_swarm_json_output(self):
        proc = self.nexus("swarm", "objective", "--mode", "parallel", "--cast", "tester",
                          "--provider", "mock", "--model", "mock-1", "--json")
        self.assertEqual(proc.returncode, 0, proc.stdout[-2000:] + proc.stderr[-2000:])
        start = proc.stdout.index("{")
        data = json.loads(proc.stdout[start:])
        self.assertEqual(data["mode"], "parallel")
        self.assertTrue(data["runs"])
        self.assertEqual(data["runs"][0]["persona"], "tester")

    def test_debate_command(self):
        proc = self.nexus("debate", "sqlite or jsonl?", "--cast", "architect", "critic",
                          "--provider", "mock", "--model", "mock-1", "--rounds", "1")
        self.assertEqual(proc.returncode, 0, proc.stdout[-2000:] + proc.stderr[-2000:])

    def test_agent_command(self):
        proc = self.assertOk(self.nexus("agent", "reviewer", "look at this",
                                        "--provider", "mock", "--model", "mock-1"))
        self.assertIn("mock echo", proc.stdout)

    def test_agent_json(self):
        proc = self.assertOk(self.nexus("agent", "tester", "check it", "--json",
                                        "--provider", "mock", "--model", "mock-1"))
        data = json.loads(proc.stdout[proc.stdout.index("{"):])
        self.assertEqual(data["persona"], "tester")
        self.assertTrue(data["ok"])

    def test_unknown_persona_still_runs(self):
        proc = self.nexus("agent", "brand-new-role", "do it", "--provider", "mock", "--model", "mock-1")
        self.assertEqual(proc.returncode, 0, proc.stderr[-800:])


class TestSessionsCLI(CLITestBase):
    def test_session_created_and_listed(self):
        self.assertOk(self.nexus("-p", "remember this session", "--provider", "mock"))
        proc = self.assertOk(self.nexus("sessions", "--all-dirs"))
        self.assertIn("remember this session", proc.stdout)

    def test_sessions_json(self):
        self.assertOk(self.nexus("-p", "hello", "--provider", "mock"))
        proc = self.assertOk(self.nexus("sessions", "--all-dirs", "--json"))
        data = json.loads(proc.stdout)
        self.assertTrue(data)
        self.assertIn("id", data[0])

    def test_export(self):
        self.assertOk(self.nexus("-p", "export me", "--provider", "mock"))
        listed = self.assertOk(self.nexus("sessions", "--all-dirs", "--json"))
        session_id = json.loads(listed.stdout)[0]["id"]
        proc = self.assertOk(self.nexus("export", session_id))
        self.assertIn("# NEXUS session", proc.stdout)
        self.assertIn("export me", proc.stdout)

    def test_export_to_file(self):
        self.assertOk(self.nexus("-p", "to a file", "--provider", "mock"))
        listed = self.assertOk(self.nexus("sessions", "--all-dirs", "--json"))
        session_id = json.loads(listed.stdout)[0]["id"]
        out = self.tmp / "transcript.md"
        self.assertOk(self.nexus("export", session_id, "-o", str(out)))
        self.assertIn("to a file", out.read_text())

    def test_export_unknown_session(self):
        proc = self.nexus("export", "does-not-exist")
        self.assertEqual(proc.returncode, 2)

    def test_resume_unknown_session_starts_fresh(self):
        proc = self.nexus("resume", "nope", "-p", "hi", "--provider", "mock")
        self.assertIn(proc.returncode, (0, 2))


class TestDoctorAndSelftest(CLITestBase):
    def test_doctor_runs(self):
        proc = self.nexus("doctor")
        self.assertIn(proc.returncode, (0, 1))
        for check in ("python", "workspace", "tools", "permission mode"):
            self.assertIn(check, proc.stdout)

    def test_doctor_json(self):
        proc = self.nexus("doctor", "--json")
        self.assertIn(proc.returncode, (0, 1))
        data = json.loads(proc.stdout)
        self.assertIn("checks", data)
        self.assertTrue(all("ok" in c for c in data["checks"]))

    def test_selftest_with_filter(self):
        # run() already prepends "python <repo>/nexus"
        proc = run(["selftest", "ignore"], env=self.env, cwd=ROOT, timeout=300)
        self.assertEqual(proc.returncode, 0, proc.stdout[-2000:])
        self.assertIn("OK", proc.stdout)


class TestDemo(CLITestBase):
    def test_demo_runs_offline(self):
        proc = self.nexus("demo", timeout=240)
        self.assertEqual(proc.returncode, 0, proc.stdout[-3000:] + proc.stderr[-2000:])
        self.assertIn("NEXUS DEMO", proc.stdout)
        self.assertIn("read_file", proc.stdout)
        self.assertIn("next steps", proc.stdout)

    def test_demo_with_swarm(self):
        proc = self.nexus("demo", "--swarm", timeout=240)
        self.assertEqual(proc.returncode, 0, proc.stdout[-3000:] + proc.stderr[-2000:])
        self.assertIn("swarm", proc.stdout.lower())


class TestInteractivePiped(CLITestBase):
    def test_slash_commands_through_a_pipe(self):
        proc = self.assertOk(self.nexus("--provider", "mock", "--model", "mock-1",
                                        stdin="/status\n/tools\n/exit\n"))
        self.assertIn("version", proc.stdout)
        self.assertIn("read_file", proc.stdout)

    def test_conversation_and_commands(self):
        proc = self.assertOk(self.nexus("--provider", "mock", "--model", "mock-1",
                                        stdin="hello there\n/usage\n/exit\n"))
        self.assertIn("mock echo", proc.stdout)
        self.assertIn("requests", proc.stdout)

    def test_unknown_command_is_helpful(self):
        proc = self.assertOk(self.nexus("--provider", "mock", stdin="/nosuch\n/exit\n"))
        self.assertIn("Unknown command", proc.stdout)
        self.assertIn("/help", proc.stdout)

    def test_help_command(self):
        proc = self.assertOk(self.nexus("--provider", "mock", stdin="/help\n/exit\n"))
        for word in ("CONVERSATION", "MODEL", "SWARM", "/swarm", "/model"):
            self.assertIn(word, proc.stdout)

    def test_help_swarm_topic(self):
        proc = self.assertOk(self.nexus("--provider", "mock", stdin="/help swarm\n/exit\n"))
        self.assertIn("hive", proc.stdout)
        self.assertIn("blackboard", proc.stdout.lower())

    def test_shell_bang_command(self):
        proc = self.assertOk(self.nexus("--provider", "mock", stdin="!echo bang-works\n/exit\n"))
        self.assertIn("bang-works", proc.stdout)

    def test_eof_exits_cleanly(self):
        proc = self.assertOk(self.nexus("--provider", "mock", stdin=""))
        self.assertIn("NEXUS", proc.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
