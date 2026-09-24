#!/usr/bin/env python3
"""Fixture: drive a Menu inside a real pty and print the machine-readable result.

Usage::

    python3 tests/fixtures/menu_driver.py <scenario> [extra…]

Scenarios are declared so the test can pick one; each prints
``RESULT:<repr>`` on the last line and ``STATE:<json>`` for diagnostics.
The terminal must be left exactly as it was found, so the driver also prints
``RESTORED:yes`` when the raw-mode/mouse teardown ran.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

from nexuscli.ui.menu import Menu, MenuItem  # noqa: E402
from nexuscli.ui.theme import Style  # noqa: E402

ITEMS = [
    MenuItem("alpha", value="A", hint="first"),
    MenuItem("beta", value="B", hint="second"),
    MenuItem("gamma", value="C", hint="third"),
    MenuItem("delta", value="D", hint="fourth"),
    MenuItem("epsilon", value="E", hint="fifth"),
]


def build(scenario: str) -> Menu:
    # Auto-detect the width from the pty so the test can recompute the exact
    # layout with the same numbers.
    style = Style.create("dark", enabled=True)
    multi = scenario in ("multi_tab", "multi_click", "multi_all")
    allow_filter = scenario != "no_filter"
    menu = Menu("Pilih satu" if not multi else "Pilih beberapa",
                [MenuItem(i.label, i.value, hint=i.hint) for i in ITEMS],
                style=style, multi=multi, allow_filter=allow_filter, max_rows=3)
    return menu


def main() -> int:
    scenario = sys.argv[1] if len(sys.argv) > 1 else "single"
    menu = build(scenario)
    restored = {"value": False}

    # Wrap the terminal teardown so the test can prove it happened.
    from nexuscli.ui import input as input_mod

    original_disable = input_mod.RawTerminal.disable

    def disable(self):
        original_disable(self)
        restored["value"] = True

    input_mod.RawTerminal.disable = disable

    result = menu.show()
    print(f"RESULT:{result!r}", flush=True)
    print("STATE:" + json.dumps({
        "filter": menu.state.filter_text,
        "cursor": menu.state.cursor,
        "selected": sorted(menu.state.selected),
        "scroll": menu.state.scroll,
    }), flush=True)
    print(f"RESTORED:{'yes' if restored['value'] else 'no'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
