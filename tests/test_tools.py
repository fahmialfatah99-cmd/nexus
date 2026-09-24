"""Tool implementation tests: real files, real subprocesses, no network.

Every tool is exercised through its public ``execute()`` path, which means
argument coercion, validation, permission metadata and output clipping are all
covered -- not just the happy path of ``run()``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nexuscli.core.checkpoints import CheckpointStore  # noqa: E402
from nexuscli.core.errors import ValidationError  # noqa: E402
from nexuscli.core.permissions import PermissionEngine  # noqa: E402
from nexuscli.tools import build_registry  # noqa: E402
from nexuscli.tools.base import ToolContext, ToolResult, add_line_numbers, clip  # noqa: E402
from nexuscli.tools.builtin.files import read_text_file, unified_diff, write_text_file  # noqa: E402
from nexuscli.tools.builtin.git import classify_git  # noqa: E402
from nexuscli.tools.builtin.shell import classify_command, is_interactive  # noqa: E402


class ToolTestBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.registry = build_registry()
        self.permissions = PermissionEngine(mode="full-auto", workspace_root=self.root)
        self.ctx = ToolContext(cwd=self.root, workspace_root=self.root, permissions=self.permissions,
                               checkpoints=CheckpointStore(self.root, "test"), cancelled=threading.Event())
        self.addCleanup(self._tmp.cleanup)

    def call(self, name: str, args):
        return self.registry.get(name).execute(args, self.ctx)

    def write(self, rel: str, text: str) -> Path:
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path


class TestReadFile(ToolTestBase):
    def test_reads_with_line_numbers(self):
        self.write("a.py", "one\ntwo\nthree\n")
        result = self.call("read_file", {"path": "a.py"})
        self.assertFalse(result.is_error)
        self.assertIn("1\tone", result.content)
        self.assertIn("3\tthree", result.content)
        self.assertIn("3 lines", result.content)

    def test_offset_and_limit(self):
        self.write("b.txt", "\n".join(f"line{i}" for i in range(1, 21)))
        result = self.call("read_file", {"path": "b.txt", "offset": 5, "limit": 3})
        self.assertIn("5\tline5", result.content)
        self.assertIn("7\tline7", result.content)
        self.assertNotIn("line8", result.content)
        self.assertIn("offset=8", result.content, "must tell the model how to continue")

    def test_raw_mode(self):
        self.write("c.txt", "hello")
        self.assertEqual(self.call("read_file", {"path": "c.txt", "raw": True}).content.strip(), "hello")

    def test_missing_file_suggests_alternatives(self):
        self.write("config.json", "{}")
        result = self.call("read_file", {"path": "config.jsn"})
        self.assertTrue(result.is_error)
        self.assertIn("config.json", result.content)

    def test_directory_and_binary_are_rejected(self):
        (self.root / "d").mkdir()
        self.assertTrue(self.call("read_file", {"path": "d"}).is_error)
        (self.root / "x.bin").write_bytes(b"\x00\x01\x02\x03")
        result = self.call("read_file", {"path": "x.bin"})
        self.assertTrue(result.is_error)
        self.assertIn("binary", result.content)

    def test_outside_workspace_is_blocked(self):
        outside = Path(tempfile.mkdtemp()) / "secret.txt"
        outside.write_text("secret")
        result = self.call("read_file", {"path": str(outside)})
        self.assertTrue(result.is_error)
        self.assertIn("outside the workspace", result.content)

    def test_argument_coercion(self):
        self.write("d.txt", "a\nb\nc\n")
        result = self.call("read_file", {"path": "d.txt", "offset": "2", "limit": "1"})
        self.assertIn("2\tb", result.content)

    def test_invalid_arguments_raise_validation_error(self):
        with self.assertRaises(ValidationError):
            self.call("read_file", {"offset": 1})


class TestWriteEdit(ToolTestBase):
    def test_write_creates_and_checkpoints(self):
        result = self.call("write_file", {"path": "new/dir/f.txt", "content": "hello"})
        self.assertFalse(result.is_error)
        self.assertEqual((self.root / "new/dir/f.txt").read_text(), "hello")
        self.assertTrue(result.touched)

    def test_write_is_idempotent(self):
        self.write("f.txt", "same")
        result = self.call("write_file", {"path": "f.txt", "content": "same"})
        self.assertIn("No change", result.content)
        self.assertFalse(result.data.get("changed"))

    def test_append(self):
        self.write("log.txt", "line1\n")
        self.call("write_file", {"path": "log.txt", "content": "line2", "append": True})
        self.assertEqual((self.root / "log.txt").read_text(), "line1\nline2")

    def test_edit_exact(self):
        self.write("m.py", "x = 1\ny = 2\n")
        result = self.call("edit_file", {"path": "m.py", "old_text": "x = 1", "new_text": "x = 42"})
        self.assertFalse(result.is_error)
        self.assertEqual((self.root / "m.py").read_text(), "x = 42\ny = 2\n")
        self.assertEqual(result.data["match"], "exact")

    def test_edit_ambiguous_is_refused(self):
        self.write("n.py", "foo()\nfoo()\n")
        result = self.call("edit_file", {"path": "n.py", "old_text": "foo()", "new_text": "bar()"})
        self.assertTrue(result.is_error)
        self.assertIn("ambiguous", result.content)
        self.assertEqual((self.root / "n.py").read_text(), "foo()\nfoo()\n", "must not modify on ambiguity")

    def test_edit_replace_all(self):
        self.write("o.py", "foo()\nfoo()\n")
        self.call("edit_file", {"path": "o.py", "old_text": "foo()", "new_text": "bar()", "replace_all": True})
        self.assertEqual((self.root / "o.py").read_text(), "bar()\nbar()\n")

    def test_edit_whitespace_tolerant(self):
        self.write("p.py", "def f():\n    return 1\n")
        result = self.call("edit_file", {"path": "p.py", "old_text": "def f():\n  return 1",
                                        "new_text": "def f():\n    return 2"})
        self.assertFalse(result.is_error, result.content)
        self.assertIn("return 2", (self.root / "p.py").read_text())
        self.assertEqual(result.data["match"], "whitespace")

    def test_edit_fuzzy_match_is_labelled(self):
        self.write("q.py", "alpha = 1\nbeta = compute(1, 2, 3)\ngamma = 3\n")
        result = self.call("edit_file", {"path": "q.py", "old_text": "beta = compute(1, 2, 4)",
                                        "new_text": "beta = compute(9)"})
        self.assertFalse(result.is_error, result.content)
        self.assertEqual(result.data["match"], "fuzzy")
        self.assertIn("compute(9)", (self.root / "q.py").read_text())
        self.assertIn("verify the diff", result.content)

    def test_edit_no_match_gives_a_hint(self):
        self.write("r.py", "value = 1\n")
        result = self.call("edit_file", {"path": "r.py", "old_text": "value = 999", "new_text": "x"})
        self.assertTrue(result.is_error)
        self.assertIn("Closest lines", result.content)
        self.assertEqual((self.root / "r.py").read_text(), "value = 1\n")

    def test_edit_creates_no_file(self):
        result = self.call("edit_file", {"path": "missing.py", "old_text": "a", "new_text": "b"})
        self.assertTrue(result.is_error)
        self.assertFalse((self.root / "missing.py").exists())

    def test_crlf_is_preserved(self):
        path = self.root / "win.txt"
        path.write_bytes(b"a = 1\r\nb = 2\r\n")
        self.call("edit_file", {"path": "win.txt", "old_text": "a = 1", "new_text": "a = 9"})
        raw = path.read_bytes()
        self.assertEqual(raw, b"a = 9\r\nb = 2\r\n")

    def test_invalid_utf8_is_never_rewritten(self):
        path = self.root / "latin.txt"
        path.write_bytes(b"caf\xe9 = 1\n")
        result = self.call("edit_file", {"path": "latin.txt", "old_text": "caf", "new_text": "CAF"})
        self.assertTrue(result.is_error)
        self.assertIn("UTF-8", result.content)
        self.assertEqual(path.read_bytes(), b"caf\xe9 = 1\n", "file must be untouched")

    def test_multi_edit_is_atomic(self):
        self.write("s.py", "a = 1\nb = 2\nc = 3\n")
        result = self.call("multi_edit", {"path": "s.py", "edits": [
            {"old_text": "a = 1", "new_text": "a = 10"},
            {"old_text": "b = 2", "new_text": "b = 20"},
            {"old_text": "NOT PRESENT", "new_text": "x"},
        ]})
        self.assertTrue(result.is_error)
        self.assertEqual((self.root / "s.py").read_text(), "a = 1\nb = 2\nc = 3\n",
                         "a failed edit must roll back all of them")

    def test_multi_edit_applies_in_order(self):
        self.write("t.py", "value = 1\n")
        self.call("multi_edit", {"path": "t.py", "edits": [
            {"old_text": "value = 1", "new_text": "value = 2"},
            {"old_text": "value = 2", "new_text": "value = 3"}]})
        self.assertEqual((self.root / "t.py").read_text(), "value = 3\n")

    def test_undo_restores_previous_bytes(self):
        self.write("u.txt", "v1")
        self.call("write_file", {"path": "u.txt", "content": "v2"})
        self.assertEqual((self.root / "u.txt").read_text(), "v2")
        restore = self.ctx.checkpoints.restore()
        self.assertTrue(restore.ok)
        self.assertEqual((self.root / "u.txt").read_text(), "v1")

    def test_read_only_mode_blocks_writes(self):
        self.ctx.read_only_mode = True
        result = self.call("write_file", {"path": "z.txt", "content": "x"})
        self.assertTrue(result.is_error)
        self.assertIn("read-only", result.content)
        self.assertFalse((self.root / "z.txt").exists())


class TestListAndFind(ToolTestBase):
    def setUp(self):
        super().setUp()
        self.write("src/a.py", "x")
        self.write("src/deep/b.py", "y")
        self.write("docs/c.md", "z")
        self.write("node_modules/junk.js", "j")
        (self.root / ".gitignore").write_text("*.log\n")
        self.write("skip.log", "s")

    def test_list_dir_hides_ignored(self):
        result = self.call("list_dir", {"path": "."})
        self.assertIn("src/", result.content)
        self.assertIn("docs/", result.content)
        self.assertNotIn("junk.js", result.content)

    def test_list_dir_all_shows_ignored(self):
        result = self.call("list_dir", {"path": ".", "all": True})
        self.assertIn("skip.log", result.content)

    def test_tree(self):
        result = self.call("list_dir", {"path": ".", "tree": True, "depth": 3})
        self.assertIn("src", result.content)
        self.assertIn("b.py", result.content)
        self.assertNotIn("junk.js", result.content)

    def test_list_dir_on_file_is_an_error(self):
        self.assertTrue(self.call("list_dir", {"path": "src/a.py"}).is_error)

    def test_find_by_glob(self):
        result = self.call("find_files", {"pattern": "**/*.py"})
        self.assertIn("src/a.py", result.content)
        self.assertIn("src/deep/b.py", result.content)
        self.assertNotIn("junk.js", result.content)

    def test_find_by_basename_pattern(self):
        result = self.call("find_files", {"pattern": "*.md"})
        self.assertIn("docs/c.md", result.content)

    def test_find_no_match(self):
        result = self.call("find_files", {"pattern": "**/*.rs"})
        self.assertIn("No files match", result.content)

    def test_find_sort_by_mtime(self):
        result = self.call("find_files", {"pattern": "**/*.py", "sort": "mtime"})
        self.assertFalse(result.is_error)

    def test_file_info(self):
        result = self.call("file_info", {"path": "src/a.py"})
        self.assertIn("kind: file", result.content)
        self.assertIn("encoding: utf-8", result.content)


class TestGrep(ToolTestBase):
    def setUp(self):
        super().setUp()
        self.write("src/a.py", "import os\n\ndef helper():\n    return os.getcwd()\n")
        self.write("src/b.py", "def other():\n    pass\n")
        self.write("docs/c.md", "helper is documented here\n")
        self.write("node_modules/x.py", "def helper():\n")
        (self.root / "big.bin").write_bytes(b"\x00helper\x00")

    def test_basic_search(self):
        result = self.call("grep", {"pattern": "helper"})
        self.assertIn("src/a.py:3:def helper():", result.content)
        self.assertIn("docs/c.md", result.content)
        self.assertNotIn("node_modules", result.content, "ignored dirs must be skipped")
        self.assertNotIn("big.bin", result.content, "binary must be skipped")

    def test_regex_and_case(self):
        self.assertTrue(self.call("grep", {"pattern": "def\\s+\\w+\\("}).content.count("def") >= 1)
        self.assertIn("No matches", self.call("grep", {"pattern": "HELPER"}).content)
        self.assertIn("docs/c.md", self.call("grep", {"pattern": "HELPER", "ignore_case": True}).content)

    def test_literal_mode_escapes_regex(self):
        self.write("re.txt", "a.b\naxb\n")
        result = self.call("grep", {"pattern": "a.b", "literal": True})
        self.assertIn("a.b", result.content)
        self.assertNotIn("axb", result.content)

    def test_invalid_regex_is_reported(self):
        result = self.call("grep", {"pattern": "([unclosed"})
        self.assertTrue(result.is_error)
        self.assertIn("Invalid regular expression", result.content)

    def test_glob_filter(self):
        result = self.call("grep", {"pattern": "helper", "glob": "*.py"})
        self.assertIn("src/a.py", result.content)
        self.assertNotIn("docs/c.md", result.content)

    def test_context_lines(self):
        result = self.call("grep", {"pattern": "getcwd", "context": 1})
        self.assertIn("return os.getcwd()", result.content)
        self.assertIn("def helper()", result.content)

    def test_files_with_matches(self):
        result = self.call("grep", {"pattern": "def", "files_with_matches": True})
        self.assertIn("src/a.py", result.content)
        self.assertNotIn(":", result.content.split("\n")[-1])

    def test_max_results_cap(self):
        self.write("many.txt", "\n".join(f"needle {i}" for i in range(200)))
        result = self.call("grep", {"pattern": "needle", "max_results": 5})
        self.assertIn("stopped at 5 matches", result.content)

    def test_word_regexp(self):
        self.write("w.txt", "cat concatenate\n")
        result = self.call("grep", {"pattern": "cat", "word_regexp": True, "path": "w.txt"})
        self.assertIn("w.txt", result.content)

    def test_single_file_path(self):
        result = self.call("grep", {"pattern": "helper", "path": "src/a.py"})
        self.assertIn("def helper()", result.content)


class TestShell(ToolTestBase):
    def test_echo_and_exit_code(self):
        result = self.call("bash", {"command": "echo hello"})
        self.assertFalse(result.is_error)
        self.assertIn("hello", result.content)
        self.assertIn("exit=0", result.content)

    def test_nonzero_exit_is_an_error_result(self):
        result = self.call("bash", {"command": "echo oops >&2; exit 7"})
        self.assertTrue(result.is_error)
        self.assertIn("exit=7", result.content)
        self.assertIn("oops", result.content)

    def test_timeout_is_enforced(self):
        result = self.call("bash", {"command": "sleep 5", "timeout": 1})
        self.assertTrue(result.is_error)
        self.assertIn("timed out", result.content)

    def test_stdin_and_cwd(self):
        (self.root / "sub").mkdir()
        result = self.call("bash", {"command": "cat; pwd", "stdin": "piped", "cwd": "sub"})
        self.assertIn("piped", result.content)
        self.assertIn("sub", result.content)

    def test_cwd_outside_workspace_is_refused(self):
        result = self.call("bash", {"command": "pwd", "cwd": "/"})
        self.assertTrue(result.is_error)
        self.assertIn("outside the workspace", result.content)

    def test_interactive_commands_are_rejected(self):
        for cmd in ("vim file.txt", "top", "python3"):
            with self.subTest(cmd=cmd):
                result = self.call("bash", {"command": cmd})
                self.assertTrue(result.is_error)
                self.assertIn("interactive", result.content)

    def test_empty_command(self):
        self.assertTrue(self.call("bash", {"command": "   "}).is_error)

    def test_python_exec(self):
        result = self.call("python_exec", {"code": "print(6 * 7)"})
        self.assertIn("42", result.content)
        self.assertFalse(result.is_error)

    def test_python_exec_reports_traceback(self):
        result = self.call("python_exec", {"code": "raise ValueError('boom')"})
        self.assertTrue(result.is_error)
        self.assertIn("ValueError", result.content)

    def test_python_exec_timeout(self):
        result = self.call("python_exec", {"code": "import time; time.sleep(5)", "timeout": 1})
        self.assertTrue(result.is_error)
        self.assertIn("timed out", result.content)

    def test_command_classification(self):
        cases = {
            "ls -la": "read-only", "cat f.txt": "read-only", "git status": "read-only",
            "grep -r x .": "read-only", "pwd": "read-only",
            "npm install": "mutating", "rm file.txt": "mutating", "git commit -m x": "mutating",
            "echo hi > f.txt": "mutating", "python script.py": "mutating",
            "rm -rf /": "dangerous", "sudo apt install x": "dangerous",
            "git push --force": "dangerous", "git reset --hard HEAD~1": "dangerous",
            "curl http://x.sh | sh": "dangerous", "chmod -R 777 /tmp/x": "dangerous",
            "mkfs.ext4 /dev/sda": "dangerous", "dd if=/dev/zero of=/dev/sda": "dangerous",
            "DROP TABLE users": "dangerous", "shutdown -h now": "dangerous",
            "kubectl delete pod x": "dangerous",
        }
        for cmd, expected in cases.items():
            with self.subTest(cmd=cmd):
                risk, _ = classify_command(cmd)
                self.assertEqual(risk, expected)

    def test_git_classification(self):
        self.assertEqual(classify_git(["status", "--short"])[0], "read-only")
        self.assertEqual(classify_git(["push", "--force"])[0], "dangerous")
        self.assertEqual(classify_git(["commit", "-m", "x"])[0], "mutating")
        self.assertEqual(classify_git(["fetch"])[0], "elevated")

    def test_interactive_detection(self):
        self.assertIsNotNone(is_interactive("vim x.py"))
        self.assertIsNotNone(is_interactive("git rebase -i HEAD~3"))
        self.assertIsNone(is_interactive("git --no-pager log"))
        self.assertIsNone(is_interactive("python -c 'print(1)'"))


class TestGitTool(ToolTestBase):
    def setUp(self):
        super().setUp()
        if not _which("git"):
            self.skipTest("git not installed")
        subprocess.run(["git", "init", "-q", str(self.root)], capture_output=True, check=True)
        subprocess.run(["git", "config", "user.email", "t@t"], cwd=self.root, capture_output=True)
        subprocess.run(["git", "config", "user.name", "t"], cwd=self.root, capture_output=True)
        self.write("f.txt", "hello\n")

    def test_status_and_add_commit(self):
        status = self.call("git", {"command": "status --short"})
        self.assertIn("f.txt", status.content)
        self.assertFalse(status.is_error)
        add = self.call("git", {"command": "add -A"})
        self.assertFalse(add.is_error)
        commit = self.call("git", {"command": "commit -m 'first'"})
        self.assertFalse(commit.is_error, commit.content)
        log = self.call("git", {"command": "log --oneline"})
        self.assertIn("first", log.content)

    def test_not_a_repo(self):
        other = Path(tempfile.mkdtemp())
        ctx = ToolContext(cwd=other, workspace_root=other, permissions=self.permissions)
        result = self.registry.get("git").execute({"command": "status"}, ctx)
        self.assertTrue(result.is_error)
        self.assertIn("Not a git repository", result.content)

    def test_interactive_is_refused(self):
        result = self.call("git", {"command": "rebase -i HEAD~1"})
        self.assertTrue(result.is_error)
        self.assertIn("Interactive", result.content)

    def test_no_shell_injection(self):
        # A metacharacter-laden branch name must be passed as a literal argument.
        result = self.call("git", {"command": "branch 'x; touch PWNED'"})
        self.assertFalse((self.root / "PWNED").exists(), "shell metacharacters must not be interpreted")


class TestTodoAndMemory(ToolTestBase):
    def test_todo_write_and_read(self):
        result = self.call("todo_write", {"todos": [
            {"content": "step one", "priority": "high"},
            {"content": "step two", "status": "in_progress"}]})
        self.assertFalse(result.is_error)
        self.assertIn("step one", result.content)
        self.assertIn("0/2 done", result.content)
        listing = self.call("todo_read", {})
        self.assertIn("[~]", listing.content)
        self.assertIn("0/2 done", listing.content)

    def test_todo_merge_updates_existing(self):
        self.call("todo_write", {"todos": [{"content": "alpha"}]})
        board = self.ctx.vars["board"]
        task = board.all()[0]
        self.call("todo_write", {"merge": True, "todos": [{"content": "alpha", "status": "done"}]})
        self.assertEqual(board.get(task.id).status, "done")
        self.assertEqual(len(board.all()), 1, "merge must not duplicate")

    def test_todo_cycle_is_rejected(self):
        # depends_on may reference the other task by title (ids do not exist yet)
        result = self.call("todo_write", {"todos": [
            {"content": "a", "depends_on": ["b"]}, {"content": "b", "depends_on": ["a"]}]})
        self.assertTrue(result.is_error)
        self.assertIn("cycle", result.content.lower())

    def test_todo_dependencies_resolve_by_title_and_index(self):
        result = self.call("todo_write", {"todos": [
            {"content": "design"}, {"content": "build", "depends_on": ["design"]},
            {"content": "test", "depends_on": ["1", "2"]}]})
        self.assertFalse(result.is_error, result.content)
        board = self.ctx.vars["board"]
        design, build, test = board.all()
        self.assertEqual(build.depends_on, [design.id])
        self.assertEqual(test.depends_on, [design.id, build.id])
        self.assertEqual([t.id for t in board.ready()], [design.id])

    def test_todo_unknown_dependency_is_reported(self):
        result = self.call("todo_write", {"todos": [{"content": "x", "depends_on": ["nope"]}]})
        self.assertTrue(result.is_error)
        self.assertIn("nope", result.content)

    def test_memory_add_view_remove(self):
        added = self.call("memory", {"action": "add", "scope": "project", "section": "Conventions",
                                     "content": "Use 4-space indents"})
        self.assertFalse(added.is_error, added.content)
        memory_file = self.root / ".nexus" / "MEMORY.md"
        self.assertTrue(memory_file.is_file())
        self.assertIn("Use 4-space indents", memory_file.read_text())
        viewed = self.call("memory", {"action": "view"})
        self.assertIn("Use 4-space indents", viewed.content)
        removed = self.call("memory", {"action": "remove", "scope": "project", "match": "4-space"})
        self.assertFalse(removed.is_error)
        self.assertNotIn("4-space", memory_file.read_text())

    def test_memory_dedupes_identical_entries(self):
        for _ in range(3):
            self.call("memory", {"action": "add", "content": "same fact", "section": "Notes"})
        text = (self.root / ".nexus" / "MEMORY.md").read_text()
        self.assertEqual(text.count("same fact"), 1)

    def test_memory_requires_content(self):
        self.assertTrue(self.call("memory", {"action": "add"}).is_error)


class TestSubAgentTool(ToolTestBase):
    def test_unavailable_without_runtime(self):
        result = self.call("task", {"prompt": "do something"})
        self.assertTrue(result.is_error)
        self.assertIn("not available", result.content)

    def test_invokes_spawner(self):
        seen = {}

        def spawn(**kwargs):
            seen.update(kwargs)
            return "SUBAGENT REPORT"

        self.ctx.vars["spawn_subagent"] = spawn
        result = self.call("task", {"subagent_type": "reviewer", "prompt": "review x", "max_turns": 3})
        self.assertIn("SUBAGENT REPORT", result.content)
        self.assertEqual(seen["role"], "reviewer")
        self.assertEqual(seen["max_turns"], 3)

    def test_spawner_crash_is_contained(self):
        def boom(**kwargs):
            raise RuntimeError("kaboom")

        self.ctx.vars["spawn_subagent"] = boom
        result = self.call("task", {"prompt": "x"})
        self.assertTrue(result.is_error)
        self.assertIn("kaboom", result.content)

    def test_empty_prompt_rejected(self):
        self.ctx.vars["spawn_subagent"] = lambda **kw: "x"
        self.assertTrue(self.call("task", {"prompt": "  "}).is_error)


class TestHelpers(ToolTestBase):
    def test_clip_keeps_head_and_tail(self):
        text = "H" * 100 + "M" * 10_000 + "T" * 100
        out = clip(text, 1000)
        self.assertLessEqual(len(out), 1400)
        self.assertTrue(out.startswith("H"))
        self.assertTrue(out.endswith("T"))
        self.assertIn("truncated", out)

    def test_clip_noop_when_small(self):
        self.assertEqual(clip("abc", 100), "abc")
        self.assertEqual(clip("abc", 0), "abc")

    def test_add_line_numbers(self):
        self.assertEqual(add_line_numbers("a\nb", start=3), "3\ta\n4\tb")
        self.assertEqual(add_line_numbers("a\nb\n"), "1\ta\n2\tb")

    def test_text_io_roundtrip(self):
        path = self.root / "rt.txt"
        write_text_file(path, "a\nb\n", "\n")
        text, style, lossy = read_text_file(path)
        self.assertEqual((text, style, lossy), ("a\nb\n", "\n", False))
        path2 = self.root / "rt2.txt"
        write_text_file(path2, "a\nb\n", "\r\n")
        self.assertEqual(path2.read_bytes(), b"a\r\nb\r\n")
        text2, style2, _ = read_text_file(path2)
        self.assertEqual((text2, style2), ("a\nb\n", "\r\n"))

    def test_unified_diff(self):
        d = unified_diff("a\nb\n", "a\nc\n", "f.txt")
        self.assertIn("-b", d)
        self.assertIn("+c", d)

    def test_tool_result_helpers(self):
        ok = ToolResult.ok("fine", x=1)
        self.assertFalse(ok.is_error)
        self.assertEqual(ok.data["x"], 1)
        bad = ToolResult.fail("nope")
        self.assertTrue(bad.is_error)
        self.assertEqual(ok.with_touched("a", "a", "b").touched, ["a", "b"])

    def test_registry_behaviour(self):
        self.assertTrue(self.registry.has("read_file"))
        self.assertFalse(self.registry.has("nope"))
        with self.assertRaises(Exception):
            self.registry.get("nope")
        self.registry.set_enabled("bash", False)
        self.assertNotIn("bash", [t.name for t in self.registry.enabled_tools()])
        self.registry.set_enabled("bash", True)
        self.assertIn("bash", [t.name for t in self.registry.enabled_tools()])

    def test_offline_removes_network_tools(self):
        offline = build_registry(offline=True)
        names = {t.name for t in offline.enabled_tools()}
        self.assertNotIn("web_fetch", names)
        self.assertNotIn("web_search", names)
        self.assertIn("read_file", names)


class TestWebFetchSafety(ToolTestBase):
    def test_private_addresses_are_refused(self):
        from nexuscli.tools.builtin.web import screen_url

        for url in ("http://127.0.0.1/x", "http://localhost:8080/", "http://169.254.169.254/latest/meta-data/",
                    "http://10.0.0.5/", "http://192.168.1.1/", "http://[::1]/"):
            with self.subTest(url=url):
                _normalised, error = screen_url(url)
                self.assertIsNotNone(error, f"{url} must be refused")

    def test_schemes_are_restricted(self):
        from nexuscli.tools.builtin.web import screen_url

        for url in ("file:///etc/passwd", "ftp://example.com/x", "gopher://x/"):
            _u, error = screen_url(url)
            self.assertIsNotNone(error, url)

    def test_internal_names_are_refused(self):
        from nexuscli.tools.builtin.web import screen_url

        _u, error = screen_url("http://metadata.google.internal/computeMetadata")
        self.assertIsNotNone(error)

    def test_allow_private_opt_in(self):
        from nexuscli.tools.builtin.web import screen_url

        _u, error = screen_url("http://127.0.0.1:8080/", allow_private=True)
        self.assertIsNone(error)

    def test_html_to_text(self):
        from nexuscli.tools.builtin.web import html_to_text

        html = ("<html><head><title>T</title><style>x{}</style></head><body>"
                "<h1>Head</h1><p>Para <a href='http://x'>link</a></p>"
                "<script>evil()</script><ul><li>one</li><li>two</li></ul>"
                "<pre>code\nblock</pre></body></html>")
        text, title = html_to_text(html)
        self.assertEqual(title, "T")
        self.assertIn("Head", text)
        self.assertIn("link (http://x)", text)
        self.assertIn("- one", text)
        self.assertNotIn("evil()", text)
        self.assertNotIn("x{}", text)
        self.assertIn("code", text)

    def test_html_to_text_survives_garbage(self):
        from nexuscli.tools.builtin.web import html_to_text

        text, _ = html_to_text("<p>unclosed <b>tags &amp; entities")
        self.assertIn("unclosed", text)
        self.assertIn("&", text)


def _which(name: str) -> str:
    import shutil

    return shutil.which(name) or ""


if __name__ == "__main__":
    unittest.main(verbosity=2)
