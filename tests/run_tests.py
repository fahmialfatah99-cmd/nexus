#!/usr/bin/env python3
"""NEXUS test runner.

Runs every ``tests/test_*.py`` with the stdlib ``unittest`` loader and prints a
compact summary. Also reachable from the CLI as ``nexus selftest`` so a user can
verify their installation without installing pytest.

    python3 tests/run_tests.py            # everything
    python3 tests/run_tests.py -v         # verbose
    python3 tests/run_tests.py providers  # only tests/test_providers.py
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def _load_module(path: Path):
    """Load a test module by file path.

    Loading via ``spec_from_file_location`` with a synthetic module name avoids
    the dotted-name import machinery entirely, so an unrelated ``tests``
    package elsewhere on ``sys.path`` (e.g. in site-packages) can never shadow
    or conflict with this repo's test files.
    """
    module_name = f"nexus_tests_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot create import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


def build_suite(patterns: list) -> unittest.TestSuite:
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for path in sorted(HERE.glob("test_*.py")):
        if patterns and not any(p in path.stem for p in patterns):
            continue
        module_name = f"tests.{path.stem}"
        try:
            module = _load_module(path)
        except Exception as exc:  # import error must be reported, not swallowed
            print(f"!! could not import {module_name}: {exc}", file=sys.stderr)
            raise
        suite.addTests(loader.loadTestsFromModule(module))
    return suite


def main(argv: list) -> int:
    parser = argparse.ArgumentParser(description="Run the NEXUS test suite.")
    parser.add_argument("patterns", nargs="*", help="only run test files whose name contains these substrings")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-f", "--failfast", action="store_true")
    args = parser.parse_args(argv)

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    if str(HERE.parent) not in sys.path:
        sys.path.insert(0, str(HERE.parent))

    suite = build_suite(args.patterns)
    runner = unittest.TextTestRunner(verbosity=2 if args.verbose else 1, failfast=args.failfast,
                                     stream=sys.stdout)
    result = runner.run(suite)
    total = result.testsRun
    bad = len(result.failures) + len(result.errors)
    print()
    matched = [f for f in sorted(HERE.glob("test_*.py"))
               if not args.patterns or any(p in f.stem for p in args.patterns)]
    print(f"test files: {len(matched)}   classes: {len(suite._tests)}   tests: {total}   "
          f"failed: {bad}   skipped: {len(result.skipped)}")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
