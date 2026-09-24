"""Project-context tests (stack detection, commands, git, tree, bounding)."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nexuscli.core.project import (  # noqa: E402
    build_project_context,
    build_tree,
    detect_commands,
    detect_stack,
    git_info,
)


def git(args, cwd):
    return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True)


class Ctx:
    """Minimal stand-in for Settings.project_context."""

    def __init__(self, **kw):
        self.enabled = kw.get("enabled", True)
        self.max_files = kw.get("max_files", 400)
        self.tree_depth = kw.get("tree_depth", 3)
        self.include_tree = kw.get("include_tree", True)
        self.include_git = kw.get("include_git", True)
        self.read_files = kw.get("read_files", ["AGENTS.md", "NEXUS.md", "CLAUDE.md"])
        self.max_chars = kw.get("max_chars", 12_000)


class SettingsStub:
    def __init__(self, **kw):
        self.project_context = Ctx(**kw)


class PyProjectFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls._tmp.name)
        git(["init", "-q", "-b", "main", str(cls.tmp)], cls.tmp.parent)
        git(["config", "user.email", "t@t"], cls.tmp)
        git(["config", "user.name", "t"], cls.tmp)
        (cls.tmp / "pyproject.toml").write_text('[project]\nname="demo"\n[tool.ruff]\nline-length=100\n')
        (cls.tmp / "Makefile").write_text("all: build\n\ntest:\n\tpytest\n\nbuild:\n\techo hi\n\nlint:\n\truff check .\n")
        (cls.tmp / "README.md").write_text("# Demo project\n\nThis is the readme.\n")
        (cls.tmp / "AGENTS.md").write_text("# Agent rules\nNever touch vendor/.\n")
        (cls.tmp / "src").mkdir()
        (cls.tmp / "src" / "main.py").write_text("print(1)\n")
        (cls.tmp / "tests").mkdir()
        (cls.tmp / "tests" / "test_a.py").write_text("def test_a(): assert True\n")
        (cls.tmp / "node_modules").mkdir()
        (cls.tmp / "node_modules" / "junk.js").write_text("x")
        (cls.tmp / ".gitignore").write_text("*.log\nbuild/\n")
        (cls.tmp / "secret.log").write_text("x")
        git(["add", "-A"], cls.tmp)
        git(["commit", "-qm", "initial commit"], cls.tmp)
        git(["commit", "-q", "--allow-empty", "-m", "second commit"], cls.tmp)
        (cls.tmp / "src" / "main.py").write_text("print(2)\n")
        (cls.tmp / "untracked.txt").write_text("new")
        (cls.tmp / "tests" / "test_b.py").write_text("def test_b(): pass\n")
        git(["add", "tests/test_b.py"], cls.tmp)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_git_info(self):
        info = git_info(self.tmp)
        self.assertTrue(info.repo)
        self.assertEqual(info.branch, "main")
        self.assertEqual(info.dirty, 1)
        self.assertEqual(info.untracked, 1)
        self.assertEqual(info.staged, 1)
        self.assertEqual(len(info.recent_commits), 2)
        self.assertIn("second commit", info.recent_commits[0])
        self.assertIn("1 staged, 1 modified, 1 untracked", info.summary())

    def test_git_info_outside_repo(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertFalse(git_info(Path(td)).repo)
            self.assertEqual(git_info(Path(td)).summary(), "")

    def test_stack_detection(self):
        stacks, markers = detect_stack(self.tmp)
        self.assertIn("python", stacks)
        self.assertIn("pyproject.toml", markers)
        self.assertIn("Makefile", markers)
        self.assertIn("tests/", markers)

    def test_command_detection(self):
        cmds = detect_commands(self.tmp)
        self.assertEqual(cmds["test"], "python -m pytest -q")
        self.assertEqual(cmds["lint"], "ruff check .")
        self.assertEqual(cmds["build"], "make build")

    def test_prompt_block_contents(self):
        ctx = build_project_context(self.tmp)
        self.assertIn("Detected stack:", ctx.text)
        self.assertIn("python", ctx.stacks)
        self.assertIn("AGENTS.md", ctx.text)
        self.assertIn("Never touch vendor/", ctx.text)
        self.assertIn("Git: branch: main", ctx.text)
        self.assertIn("Languages (file counts): python:", ctx.text)
        self.assertIn("Recent commits", ctx.text)
        self.assertIn("Commands found in config", ctx.text)
        self.assertTrue(ctx.test_command)

    def test_ignored_paths_never_appear(self):
        ctx = build_project_context(self.tmp)
        self.assertNotIn("node_modules", ctx.tree)
        self.assertNotIn("secret.log", ctx.tree)

    def test_output_is_bounded_by_max_chars(self):
        small = build_project_context(self.tmp, settings=SettingsStub(max_chars=600, read_files=["AGENTS.md"]))
        self.assertLessEqual(len(small.text), 700)
        self.assertIn("truncated", small.text.lower() + " ".join(small.truncated).lower())

    def test_can_be_disabled(self):
        off = build_project_context(self.tmp, settings=SettingsStub(enabled=False))
        self.assertEqual(off.text, "")
        self.assertEqual(off.stacks, [])

    def test_missing_directory_is_safe(self):
        ctx = build_project_context(self.tmp / "does-not-exist")
        self.assertEqual(ctx.text, "")

    def test_no_settings_uses_defaults(self):
        self.assertTrue(build_project_context(self.tmp, settings=None).text)


class TestOtherStacks(unittest.TestCase):
    def _ctx(self, files: dict) -> Path:
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(tmp, ignore_errors=True))
        for name, body in files.items():
            path = tmp / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(body)
        return tmp

    def test_node_package_manager_detection(self):
        tmp = self._ctx({"package.json": '{"scripts":{"test":"vitest run","build":"vite build","lint":"eslint ."}}',
                         "pnpm-lock.yaml": ""})
        cmds = detect_commands(tmp)
        self.assertTrue(cmds["test"].startswith("pnpm"), cmds)
        self.assertIn("lint", cmds)
        self.assertFalse(git_info(tmp).repo)

    def test_node_defaults_to_npm(self):
        tmp = self._ctx({"package.json": '{"scripts":{"test":"jest"}}'})
        self.assertTrue(detect_commands(tmp)["test"].startswith("npm"))

    def test_rust(self):
        tmp = self._ctx({"Cargo.toml": '[package]\nname="x"\n'})
        self.assertEqual(detect_commands(tmp)["test"], "cargo test")
        self.assertIn("rust", detect_stack(tmp)[0])

    def test_go(self):
        tmp = self._ctx({"go.mod": "module x\n"})
        self.assertEqual(detect_commands(tmp)["test"], "go test ./...")

    def test_makefile_targets(self):
        tmp = self._ctx({"Makefile": "all: build\n\ntest:\n\tpytest\n\nbuild:\n\techo hi\n\n"
                                     "NOT_A_TARGET = 1\n\n# comment\n"})
        cmds = detect_commands(tmp)
        self.assertEqual(cmds["test"], "make test")
        self.assertEqual(cmds["build"], "make build")


class TestTree(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(self.tmp, ignore_errors=True))
        for d in ("src/pkg/deep", "tests", "docs", "node_modules", "build"):
            (self.tmp / d).mkdir(parents=True)
        for f in ("src/a.py", "src/pkg/b.py", "src/pkg/deep/c.py", "tests/t.py", "docs/d.md",
                  "README.md", "x.log", "node_modules/j.js"):
            (self.tmp / f).write_text("x")
        (self.tmp / ".gitignore").write_text("*.log\nbuild/\n")

    def test_shape_and_alignment(self):
        text, count, truncated = build_tree(self.tmp, depth=4)
        self.assertFalse(truncated)
        self.assertNotIn("node_modules", text)
        self.assertNotIn("x.log", text)
        self.assertNotIn("build/", text)
        self.assertIn("│   ├── pkg/", text)
        self.assertIn("│   │   ├── deep/", text)
        self.assertIn("│   │   │   └── c.py", text)
        self.assertGreater(count, 8)

    def test_depth_limit(self):
        text, _, _ = build_tree(self.tmp, depth=1)
        self.assertNotIn("deep", text)

    def test_entry_limit_sets_truncated(self):
        text, count, truncated = build_tree(self.tmp, depth=4, max_entries=3)
        self.assertTrue(truncated)
        self.assertEqual(count, 3)

    def test_empty_dir(self):
        empty = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(empty, ignore_errors=True))
        text, count, truncated = build_tree(empty)
        self.assertEqual((text, count, truncated), ("", 0, False))


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestProjectRootDetection(unittest.TestCase):
    """Workspace containment depends on finding the right root.

    Regression guard: ``~/.nexus`` is the *global* config directory, so using
    ``.nexus`` as a plain project marker made every project under ``$HOME``
    resolve its workspace root to the home directory -- which silently widened
    the path guard to the whole home tree.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self._saved = os.environ.get("NEXUS_HOME")
        os.environ["NEXUS_HOME"] = str(self.root / ".nexus")
        (self.root / ".nexus").mkdir(parents=True)
        (self.root / ".nexus" / "config.json").write_text("{}")
        self.addCleanup(self._restore)

    def _restore(self):
        if self._saved is None:
            os.environ.pop("NEXUS_HOME", None)
        else:
            os.environ["NEXUS_HOME"] = self._saved
        self._tmp.cleanup()

    def test_project_without_markers_resolves_to_itself(self):
        from nexuscli.core.paths import find_project_root

        project = self.root / "projects" / "myapp"
        (project / "src").mkdir(parents=True)
        self.assertEqual(find_project_root(project), project)
        self.assertEqual(find_project_root(project / "src"), project / "src")

    def test_global_config_dir_is_never_the_workspace_root(self):
        from nexuscli.core.paths import find_project_root, home

        documents = self.root / "documents" / "notes"
        documents.mkdir(parents=True)
        self.assertEqual(home(), self.root / ".nexus")
        self.assertNotEqual(find_project_root(documents), self.root)
        self.assertEqual(find_project_root(documents), documents)

    def test_vcs_marker_wins(self):
        from nexuscli.core.paths import find_project_root

        project = self.root / "repo"
        (project / ".git").mkdir(parents=True)
        (project / "a" / "b").mkdir(parents=True)
        self.assertEqual(find_project_root(project / "a" / "b"), project)

    def test_project_nexus_config_is_a_valid_marker(self):
        from nexuscli.core.paths import find_project_root

        project = self.root / "other" / "app"
        (project / ".nexus").mkdir(parents=True)
        (project / ".nexus" / "config.json").write_text("{}")
        (project / "src").mkdir()
        self.assertEqual(find_project_root(project / "src"), project)

    def test_empty_nexus_dir_is_not_a_marker(self):
        from nexuscli.core.paths import find_project_root

        project = self.root / "third" / "app"
        (project / ".nexus").mkdir(parents=True)
        (project / "src").mkdir()
        self.assertEqual(find_project_root(project / "src"), project / "src")

    def test_manifest_markers(self):
        from nexuscli.core.paths import find_project_root

        for manifest in ("package.json", "pyproject.toml", "Cargo.toml", "go.mod", "Makefile"):
            project = self.root / "m" / manifest.replace(".", "_")
            project.mkdir(parents=True)
            (project / manifest).write_text("{}")
            (project / "deep").mkdir()
            self.assertEqual(find_project_root(project / "deep"), project, manifest)
