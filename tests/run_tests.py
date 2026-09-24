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
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def build_suite(patterns: list) -> unittest.TestSuite:
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for path in sorted(HERE.glob("test_*.py")):
        if patterns and not any(p in path.stem for p in patterns):
            continue
        module_name = f"tests.{path.stem}"
        try:
            module = __import__(module_name, fromlist=[path.stem])
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
