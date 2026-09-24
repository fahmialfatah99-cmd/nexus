"""Ignore-rule / path-safety tests.

Covers the gitignore subset that code search depends on, including nested
ignore files, negation, anchoring, dir-only rules and ``**``.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nexuscli.core.ignore import (  # noqa: E402
    IgnoreMatcher,
    human_size,
    is_probably_binary,
    is_within,
    parse_rule,
    resolve_path,
    walk_files,
)


def build_tree(root: Path) -> None:
    (root / ".gitignore").write_text("*.log\nbuild/\n/dist\nnode_modules\n!keep.log\ntemp/**/*.pyc\nsrc/gen/\n")
    for sub in ("build", "dist", "src/gen", "temp/a", "keepdir", "node_modules",
                "a/b/c", "a/b/deep", "a/b/c/deep"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    files = {
        "app.py": "x", "a.log": "x", "keep.log": "x", "blob.bin": "\x00",
        "build/out.js": "x", "dist/i.js": "x", "src/gen/g.py": "x", "src/main.py": "x",
        "temp/a/b.pyc": "x", "temp/a/ok.py": "x", "keepdir/deep.log": "x",
        "node_modules/lib.js": "x", "a/secret.txt": "x", "a/b/secret.txt": "x",
        "a/b/c/secret.txt": "x", "a/b/deep/z.tmp": "x", "a/b/c/deep/y.tmp": "x",
        "a/b/c/ok.py": "x",
    }
    for rel, body in files.items():
        (root / rel).write_text(body)
    (root / "a" / "b" / ".gitignore").write_text("secret.txt\ndeep/*.tmp\n")


class TestIgnoreRules(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        build_tree(self.root)
        self.m = IgnoreMatcher(self.root)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def ign(self, rel: str, is_dir: bool = False) -> bool:
        return self.m.is_ignored(self.root / rel, is_dir=is_dir)

    def test_root_rules(self):
        self.assertTrue(self.ign("a.log"))
        self.assertFalse(self.ign("keep.log"), "negation must win")
        self.assertFalse(self.ign("app.py"))
        self.assertTrue(self.ign("keepdir/deep.log"), "basename rule applies at depth")

    def test_dir_only_rules(self):
        self.assertTrue(self.ign("build", is_dir=True))
        self.assertTrue(self.ign("build/out.js"))
        self.assertTrue(self.ign("dist", is_dir=True))
        self.assertTrue(self.ign("src/gen", is_dir=True))
        self.assertTrue(self.ign("src/gen/g.py"))
        self.assertFalse(self.ign("src/main.py"))

    def test_anchored_leading_slash(self):
        (self.root / ".gitignore").write_text("/rootonly\n")
        m = IgnoreMatcher(self.root)
        self.assertTrue(m.is_ignored(self.root / "rootonly"))
        (self.root / "sub").mkdir(exist_ok=True)
        self.assertFalse(m.is_ignored(self.root / "sub" / "rootonly"))

    def test_doublestar(self):
        self.assertTrue(self.ign("temp/a/b.pyc"))
        self.assertFalse(self.ign("temp/a/ok.py"))

    def test_always_skipped(self):
        self.assertTrue(self.ign("node_modules/lib.js"), "node_modules always skipped")
        self.assertTrue(self.ign(".git/config"))
        self.assertTrue(self.ign("blob.bin"), "binary content skipped")

    def test_nested_ignore_files(self):
        self.assertTrue(self.ign("a/b/secret.txt"))
        self.assertTrue(self.ign("a/b/c/secret.txt"), "slash-free nested rule applies to descendants")
        self.assertFalse(self.ign("a/secret.txt"), "nested rule must not leak upwards")
        self.assertTrue(self.ign("a/b/deep/z.tmp"), "anchored nested rule")
        self.assertFalse(self.ign("a/b/c/deep/y.tmp"), "anchored nested rule stays anchored")

    def test_respect_gitignore_flag(self):
        m = IgnoreMatcher(self.root, extra_patterns=["*.py"], respect_gitignore=False)
        self.assertTrue(m.is_ignored(self.root / "a" / "b" / "c" / "ok.py"))
        self.assertFalse(m.is_ignored(self.root / "a.log"))

    def test_walk_prunes_ignored_dirs(self):
        got = sorted(str(p.relative_to(self.root)) for p in walk_files(self.root))
        expected = sorted([
            ".gitignore", "app.py", "keep.log", "src/main.py", "temp/a/ok.py",
            "a/secret.txt", "a/b/.gitignore", "a/b/c/deep/y.tmp", "a/b/c/ok.py",
        ])
        self.assertEqual(got, expected)

    def test_walk_max_files(self):
        got = list(walk_files(self.root, max_files=3))
        self.assertLessEqual(len(got), 3)


class TestParseRule(unittest.TestCase):
    def test_comments_and_blanks(self):
        self.assertIsNone(parse_rule("# comment"))
        self.assertIsNone(parse_rule(""))
        self.assertIsNone(parse_rule("   "))

    def test_negation(self):
        r = parse_rule("!important.log")
        self.assertTrue(r.negated)
        self.assertTrue(r.basename)

    def test_anchored(self):
        r = parse_rule("/root-only")
        self.assertTrue(r.anchored)
        self.assertFalse(r.basename)

    def test_doublestar_prefix_is_not_basename(self):
        self.assertFalse(parse_rule("**/anywhere").basename)

    def test_escaped_specials(self):
        self.assertEqual(parse_rule("\\!escaped").pattern, "!escaped")
        self.assertEqual(parse_rule("\\#hash").pattern, "#hash")

    def test_dir_only(self):
        r = parse_rule("build/")
        self.assertTrue(r.dir_only)

    def test_nested_prefix_expansion(self):
        self.assertEqual(parse_rule("secret.txt", prefix="a/b/").pattern, "a/b/**/secret.txt")
        self.assertEqual(parse_rule("deep/*.tmp", prefix="a/b/").pattern, "a/b/deep/*.tmp")


class TestPathSafety(unittest.TestCase):
    def test_is_within(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self.assertTrue(is_within(root, root / "a" / "b.txt"))
            self.assertTrue(is_within(root, root))
            self.assertTrue(is_within(root, root / "sub" / ".." / "a.txt"), "lexical .. must resolve")
            self.assertFalse(is_within(root, root.parent))
            self.assertFalse(is_within(root, root / ".." / ".." / "etc"))

    def test_resolve_path(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self.assertEqual(resolve_path("../y", root), root.parent / "y")
            self.assertTrue(resolve_path("~/z", root).name == "z")
            self.assertTrue(str(resolve_path("$HOME/w", root)).endswith("w"))

    def test_binary_detection(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "t.txt").write_text("hello")
            (root / "b.bin").write_bytes(b"\x00\x01")
            (root / "i.png").write_bytes(b"\x89PNG\r\n")
            self.assertFalse(is_probably_binary(root / "t.txt"))
            self.assertTrue(is_probably_binary(root / "b.bin"))
            self.assertTrue(is_probably_binary(root / "i.png"))
            self.assertTrue(is_probably_binary(root / "missing.txt"))


class TestHumanSize(unittest.TestCase):
    def test_units(self):
        self.assertEqual(human_size(0), "0B")
        self.assertEqual(human_size(512), "512B")
        self.assertEqual(human_size(1536), "1.5KB")
        self.assertEqual(human_size(5 * 1024 * 1024), "5.0MB")


if __name__ == "__main__":
    unittest.main(verbosity=2)
