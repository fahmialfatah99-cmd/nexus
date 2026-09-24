"""Configuration layering, validation and credential store tests."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nexuscli.core.config import (  # noqa: E402
    Settings,
    config_sources,
    deep_merge,
    env_overrides,
    load_auth,
    load_json,
    load_settings,
    remove_auth_key,
    save_auth_key,
)
from nexuscli.core.errors import ConfigError  # noqa: E402


class ConfigTestBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.home = self.tmp / "home"
        self.proj = self.tmp / "proj"
        (self.home).mkdir()
        (self.proj / ".nexus").mkdir(parents=True)
        self._saved_home = os.environ.get("NEXUS_HOME")
        os.environ["NEXUS_HOME"] = str(self.home)
        self.addCleanup(self._restore)

    def _restore(self):
        if self._saved_home is None:
            os.environ.pop("NEXUS_HOME", None)
        else:
            os.environ["NEXUS_HOME"] = self._saved_home
        self._tmp.cleanup()

    def write(self, path: Path, data: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")


class TestDefaults(ConfigTestBase):
    def test_defaults_are_usable(self):
        s = Settings()
        self.assertEqual(s.approval_mode, "auto-edit")
        self.assertEqual(s.ui.theme, "dark")
        self.assertEqual(s.swarm.mode, "hive")
        self.assertTrue(s.stream)
        self.assertIsNone(s.temperature)
        self.assertEqual(s.project_context.read_files[0], "README.md")

    def test_validate_accepts_defaults(self):
        Settings().validate()


class TestLayering(ConfigTestBase):
    def test_precedence_user_project_local_env_cli(self):
        self.write(self.home / "config.json",
                   {"default_model": "gpt-4o", "temperature": 0.7, "ui": {"theme": "light"}})
        self.write(self.proj / ".nexus" / "config.json",
                   {"default_model": "claude-sonnet-4-5", "swarm": {"max_parallel": 8}})
        self.write(self.proj / ".nexus" / "config.local.json", {"swarm": {"max_parallel": 2}})
        s = load_settings(cwd=self.proj, overrides={"max_turns": 7},
                          environ={"NEXUS_TEMPERATURE": "0.3"})
        self.assertEqual(s.default_model, "claude-sonnet-4-5", "project beats user")
        self.assertEqual(s.swarm.max_parallel, 2, "local beats project")
        self.assertEqual(s.temperature, 0.3, "env beats file")
        self.assertEqual(s.ui.theme, "light", "user value survives")
        self.assertEqual(s.max_turns, 7, "cli beats everything")
        self.assertGreaterEqual(len(config_sources(s)), 3)

    def test_missing_files_are_fine(self):
        s = load_settings(cwd=self.tmp / "nowhere", environ={})
        self.assertEqual(s.approval_mode, "auto-edit")

    def test_deep_merge_is_recursive(self):
        merged = deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"c": 3}, "d": 4})
        self.assertEqual(merged, {"a": {"b": 1, "c": 3}, "d": 4})

    def test_read_only_forces_approval_mode(self):
        s = load_settings(cwd=self.proj, overrides={"read_only": True})
        self.assertEqual(s.approval_mode, "read-only")


class TestEnvOverrides(ConfigTestBase):
    def test_shapes_and_casts(self):
        env = env_overrides({"NEXUS_SWARM_MODE": "debate", "NEXUS_COLOR": "0",
                             "NEXUS_MAX_TOKENS": "900", "NEXUS_MODEL": "",
                             "NEXUS_FAILOVER": "a, b", "NEXUS_TEMPERATURE": "not-a-number"})
        self.assertEqual(env, {"swarm": {"mode": "debate"}, "ui": {"color": False},
                               "max_tokens": 900, "failover": ["a", "b"]})

    def test_bool_parsing(self):
        for raw, expected in (("1", True), ("true", True), ("yes", True), ("on", True),
                              ("0", False), ("false", False), ("off", False)):
            self.assertEqual(env_overrides({"NEXUS_STREAM": raw})["stream"], expected, raw)


class TestProviders(ConfigTestBase):
    def test_provider_sections_merge(self):
        s = load_settings(cwd=self.proj, overrides={"providers": {
            "openai": {"base_url": "https://x/v1", "api_key": "sk-1", "extra": {"a": 1}},
            "groq": {"api_key_env": "GROQ_API_KEY"},
        }})
        self.assertEqual(s.providers["openai"].base_url, "https://x/v1")
        self.assertEqual(s.providers["openai"].extra, {"a": 1})
        self.assertEqual(s.providers["groq"].api_key_env, "GROQ_API_KEY")
        self.assertEqual(s.provider_configs()["openai"]["api_key"], "sk-1")
        self.assertNotIn("models", s.provider_configs()["openai"], "empty values are dropped")

    def test_roundtrip(self):
        s = load_settings(cwd=self.proj, overrides={"providers": {"openai": {"api_key": "k"}},
                                                    "swarm": {"cast": ["tester"]}})
        restored = Settings.from_dict(s.to_dict())
        self.assertEqual(restored.providers["openai"].api_key, "k")
        self.assertEqual(restored.swarm.cast, ["tester"])
        path = s.save(self.tmp / "out.json")
        self.assertEqual(load_json(path)["swarm"]["cast"], ["tester"])


class TestValidation(ConfigTestBase):
    def test_errors_name_the_exact_path(self):
        cases = [
            ({"approval_mode": "yolo1"}, "approval_mode"),
            ({"temperature": 5}, "temperature"),
            ({"max_turns": 0}, "max_turns"),
            ({"compaction": {"threshold": 2}}, "compaction.threshold"),
            ({"swarm": {"max_parallel": 0}}, "swarm.max_parallel"),
            ({"swarm": {"max_rounds": 0}}, "swarm.max_rounds"),
            ({"stream": "maybe"}, "stream"),
            ({"ui": {"width": "wide"}}, "ui.width"),
            ({"providers": {"openai": 5}}, "providers.openai"),
            ({"ui": 5}, "ui"),
        ]
        for payload, needle in cases:
            with self.subTest(payload=payload):
                with self.assertRaises(ConfigError) as ctx:
                    Settings.from_dict(payload)
                self.assertIn(needle, str(ctx.exception))

    def test_unknown_keys_warn_but_load(self):
        s = Settings.from_dict({"totally_new_key": 1, "ui": {"future": 2}})
        self.assertEqual(s.approval_mode, "auto-edit")

    def test_strict_mode_rejects_unknown_keys(self):
        with self.assertRaises(ConfigError):
            Settings.from_dict({"nope": 1}, strict=True)

    def test_broken_json_is_actionable(self):
        (self.tmp / "bad.json").write_text("{not json")
        with self.assertRaises(ConfigError) as ctx:
            load_json(self.tmp / "bad.json")
        self.assertIn("not valid JSON", str(ctx.exception))
        self.assertIn("line 1", str(ctx.exception))

    def test_non_object_root_rejected(self):
        (self.tmp / "arr.json").write_text("[1,2,3]")
        with self.assertRaises(ConfigError):
            load_json(self.tmp / "arr.json")

    def test_empty_file_is_ok(self):
        (self.tmp / "empty.json").write_text("")
        self.assertEqual(load_json(self.tmp / "empty.json"), {})


class TestAuthStore(ConfigTestBase):
    def test_save_load_remove_and_permissions(self):
        path = save_auth_key("openai", "sk-secret")
        self.assertEqual(load_auth()["openai"], "sk-secret")
        self.assertEqual(oct(path.stat().st_mode & 0o777), "0o600", "credentials must be 0600")
        save_auth_key("groq", "gsk-1")
        self.assertEqual(set(load_auth()), {"openai", "groq"})
        self.assertTrue(remove_auth_key("openai"))
        self.assertFalse(remove_auth_key("openai"))
        self.assertEqual(load_auth(), {"groq": "gsk-1"})

    def test_missing_file_is_empty(self):
        self.assertEqual(load_auth(), {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
