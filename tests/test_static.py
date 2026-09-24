"""Static analysis guard: no undefined names anywhere in the package.

Two real bugs motivated this file: ``cli.py`` used ``sessions_dir`` and
``app.py`` used ``visible_width`` without importing them. Both are runtime
``NameError``s on rarely-taken paths, exactly the kind of bug unit tests miss and
users find. This checker walks the AST of every module, collects every name that
is bound anywhere in the enclosing scopes (imports, assignments, defs, classes,
function arguments, comprehension targets, ``except as``, ``with as``, globals),
and fails on any load of a name that is not bound and not a builtin.

It is deliberately scope-insensitive (a name bound anywhere in the module counts
as bound) which keeps false positives near zero while still catching every
missing import.
"""

from __future__ import annotations

import ast
import builtins
import sys
import unittest
from pathlib import Path
from typing import Dict, List, Set

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "nexuscli"
sys.path.insert(0, str(ROOT))

BUILTIN_NAMES = set(dir(builtins)) | {"__file__", "__name__", "__doc__", "__package__",
                                      "__spec__", "__loader__", "__builtins__", "__class__",
                                      "__dict__", "__module__", "__qualname__", "__all__",
                                      "__version__", "__app_name__", "self", "cls"}


def module_files() -> List[Path]:
    return sorted(p for p in PACKAGE.rglob("*.py"))


def bound_names(tree: ast.AST) -> Set[str]:
    """Every name bound anywhere in the module (deliberately scope-insensitive)."""
    names: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.name == "*":
                    names.add("*")
                target = (alias.asname or alias.name).split(".")[0]
                names.add(target)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = node.args
                for arg in [*args.posonlyargs, *args.args, *args.kwonlyargs]:
                    names.add(arg.arg)
                if args.vararg:
                    names.add(args.vararg.arg)
                if args.kwarg:
                    names.add(args.kwarg.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
        elif isinstance(node, ast.Global) or isinstance(node, ast.Nonlocal):
            names.update(node.names)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.comprehension):
            for sub in ast.walk(node.target):
                if isinstance(sub, ast.Name):
                    names.add(sub.id)
        elif isinstance(node, ast.Attribute):
            pass
    return names


def star_imports(tree: ast.AST) -> List[str]:
    return [node.module or "" for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and any(a.name == "*" for a in node.names)]


def undefined_names(path: Path) -> List[str]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    bound = bound_names(tree)
    if star_imports(tree):
        # A wildcard import makes the bound set unknowable; only check that the
        # module still parses and skip name resolution.
        return []
    used: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            used.add(node.id)
    missing = sorted(used - bound - BUILTIN_NAMES)
    return missing


def attribute_roots(path: Path) -> List[str]:
    """Names used as ``x.y`` where ``x`` is never bound (another missing-import sign)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    bound = bound_names(tree)
    if star_imports(tree):
        return []
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            target = node
            while isinstance(target, ast.Attribute):
                target = target.value
            if isinstance(target, ast.Name):
                roots.add(target.id)
    return sorted(roots - bound - BUILTIN_NAMES)


class TestNoUndefinedNames(unittest.TestCase):
    def test_every_module_is_parseable(self):
        files = module_files()
        self.assertGreater(len(files), 20, "the package should have real content")
        for path in files:
            with self.subTest(module=path.name):
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def test_no_undefined_names(self):
        problems: Dict[str, List[str]] = {}
        for path in module_files():
            missing = undefined_names(path)
            if missing:
                problems[str(path.relative_to(ROOT))] = missing
        self.assertEqual(problems, {}, f"undefined names found: {problems}")

    def test_no_unbound_attribute_roots(self):
        problems: Dict[str, List[str]] = {}
        for path in module_files():
            missing = attribute_roots(path)
            if missing:
                problems[str(path.relative_to(ROOT))] = missing
        self.assertEqual(problems, {}, f"attribute access on unbound names: {problems}")

    def test_every_module_imports_cleanly(self):
        """Import each module for real: catches import-time errors and cycles."""
        import importlib

        for path in module_files():
            rel = path.relative_to(ROOT).with_suffix("")
            module = ".".join(rel.parts)
            if module.endswith("__main__"):
                continue
            with self.subTest(module=module):
                importlib.import_module(module)

    def test_no_syntax_warnings_from_duplicated_dict_keys(self):
        """Duplicated keys in a literal silently drop data; reject them."""
        for path in module_files():
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Dict):
                    seen = []
                    for key in node.keys:
                        if isinstance(key, ast.Constant):
                            if key.value in seen:
                                self.fail(f"{path.name}: duplicated dict key {key.value!r}")
                            seen.append(key.value)


if __name__ == "__main__":
    unittest.main(verbosity=2)
