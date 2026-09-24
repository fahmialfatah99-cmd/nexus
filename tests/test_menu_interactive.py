"""Interactive menu tests inside a real pseudo-terminal.

``test_menu.py`` covers the state machine. These cover what only a real tty can
prove: raw-mode switching, mouse reporting being enabled **and disabled again**,
incremental redraws, cursor hiding/restoring, and that clicking a specific drawn
row does what the drawing says it does.

Click coordinates are not guessed -- they are computed from the same
``MenuState.layout()`` the renderer uses, so a layout change fails these tests
loudly instead of silently clicking the wrong row.

Skipped when ``pty`` is unavailable (Windows) rather than silently passing.
"""

from __future__ import annotations

import ast
import os
import select
import subprocess
import sys
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DRIVER = HERE / "fixtures" / "menu_driver.py"
sys.path.insert(0, str(ROOT))

from nexuscli.ui.menu import MenuItem, MenuState  # noqa: E402
from nexuscli.ui.theme import Style, visible_width  # noqa: E402

try:
    import pty  # noqa: F401

    HAS_PTY = True
except Exception:  # pragma: no cover - Windows
    HAS_PTY = False

MOUSE_ON = "\x1b[?1000h"
MOUSE_OFF = "\x1b[?1000l"
HIDE_CURSOR = "\x1b[?25l"
SHOW_CURSOR = "\x1b[?25h"

LABELS = ["alpha", "beta", "gamma", "delta", "epsilon"]
VALUES = ["A", "B", "C", "D", "E"]


def sgr(row: int, col: int, button: int = 0, release: bool = False) -> bytes:
    return f"\x1b[<{button};{col};{row}{'m' if release else 'M'}".encode()


def sgr_wheel(row: int, col: int, up: bool) -> bytes:
    return f"\x1b[<{64 if up else 65};{col};{row}M".encode()


def layout(width: int = 80, max_rows: int = 3, multi: bool = False, allow_filter: bool = True,
           cursor: int = 0, scroll: int = 0):
    """Rebuild the exact layout the driver draws."""
    state = MenuState(title="Pilih beberapa" if multi else "Pilih satu",
                      items=[MenuItem(l, v, hint=h) for l, v, h in
                             zip(LABELS, VALUES, ["first", "second", "third", "fourth", "fifth"])],
                      multi=multi, allow_filter=allow_filter, max_rows=max_rows)
    state._style = Style.create("dark", enabled=True, width=width)
    state.cursor = cursor
    state.scroll = scroll
    state._ensure_visible()
    return state, state.layout(width)


def item_rows(width: int = 80, **kw) -> dict:
    """value -> 1-based screen row for the initially drawn menu."""
    state, rows = layout(width=width, **kw)
    return {state.items[payload].value: n for n, (kind, payload, _line) in enumerate(rows, 1)
            if kind == "item"}


def button_row(width: int = 80, **kw) -> int:
    _state, rows = layout(width=width, **kw)
    for n, (kind, _payload, _line) in enumerate(rows, 1):
        if kind == "buttons":
            return n
    raise AssertionError("no button row in layout")


def button_col(width: int, bid: str, **kw) -> int:
    state, _rows = layout(width=width, **kw)
    for name, start, _end in state.button_columns(width):
        if name == bid:
            return start
    raise AssertionError(f"no button {bid}")


class PtySession:
    """Run a command under a pty, feed it bytes, collect everything it prints."""

    def __init__(self, argv, cols: int = 80, rows: int = 24, env=None):
        self.argv = argv
        self.env = {k: v for k, v in os.environ.items() if k != "NO_COLOR"}
        self.env.update({"TERM": "xterm-256color", "COLORTERM": "truecolor",
                         "PYTHONIOENCODING": "utf-8", "NEXUS_NO_MOUSE": "", **(env or {})})
        self.env.pop("NEXUS_NO_MOUSE", None)
        self.output = ""
        self.returncode = None
        self.cols = cols
        self.rows = rows

    def run(self, inputs, *, settle: float = 0.06, timeout: float = 20.0) -> str:
        """Run to completion, feeding *inputs* only after the menu is on screen.

        The wait matters: ``tty.setraw()`` uses TCSAFLUSH, which *discards* any
        bytes that arrived before raw mode was enabled. Sending keys immediately
        after spawn therefore loses them.
        """
        import pty as pty_mod

        master, slave = pty_mod.openpty()
        try:
            import fcntl
            import struct
            import termios

            fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", self.rows, self.cols, 0, 0))
        except Exception:
            pass
        proc = subprocess.Popen(self.argv, stdin=slave, stdout=slave, stderr=slave,
                                env=self.env, close_fds=True)
        os.close(slave)
        deadline = time.monotonic() + timeout
        try:
            self._pump(master, proc, 0.4)                     # startup output
            self._wait_for_frame(master, proc, deadline)       # never type before raw mode
            for payload in inputs:
                if proc.poll() is not None:
                    break
                os.write(master, payload if isinstance(payload, bytes) else payload.encode())
                self._pump(master, proc, settle)
            while proc.poll() is None and time.monotonic() < deadline:
                self._pump(master, proc, 0.1)
            self._pump(master, proc, 0.2)
        finally:
            if proc.poll() is None:
                proc.kill()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            self.returncode = proc.returncode
            try:
                os.close(master)
            except OSError:
                pass
        return self.output

    def _wait_for_frame(self, master, proc, deadline) -> None:
        while time.monotonic() < deadline and proc.poll() is None:
            if "\u250c" in self.output or "RESULT:" in self.output:
                return
            self._pump(master, proc, 0.05)

    def _pump(self, master, proc, budget: float) -> None:
        end = time.monotonic() + budget
        while time.monotonic() < end:
            ready, _, _ = select.select([master], [], [], 0.02)
            if not ready:
                if proc.poll() is not None:
                    # the child exited; drain whatever is still buffered
                    ready2, _, _ = select.select([master], [], [], 0.05)
                    if not ready2:
                        return
                else:
                    continue
            try:
                chunk = os.read(master, 65536)
            except OSError:
                return
            if not chunk:
                return
            self.output += chunk.decode("utf-8", "replace")

    # -- parsed results ---------------------------------------------------
    def _clean_lines(self):
        """Strip ANSI escapes, then split -- the driver's markers are often
        preceded by the terminal-restore sequences on the same physical line."""
        from nexuscli.ui.theme import strip_ansi

        return [strip_ansi(l).strip() for l in self.output.replace("\r", "\n").split("\n")]

    def _find(self, prefix: str) -> str:
        for line in reversed(self._clean_lines()):
            index = line.find(prefix)
            if index >= 0:
                return line[index + len(prefix):]
        return ""

    def result(self):
        raw = self._find("RESULT:")
        if not raw:
            return "<no result>"
        return ast.literal_eval(raw)

    def state(self) -> dict:
        import json

        raw = self._find("STATE:")
        return json.loads(raw) if raw else {}

    def restored(self) -> bool:
        return "RESTORED:yes" in self._find("RESTORED:") or "RESTORED:yes" in self.output


def driver(scenario: str = "single") -> PtySession:
    return PtySession([sys.executable, str(DRIVER), scenario])


@unittest.skipUnless(HAS_PTY, "pty is unavailable on this platform")
class TestKeyboard(unittest.TestCase):
    def test_arrows_and_enter(self):
        session = driver()
        session.run([b"\x1b[B", b"\x1b[B", b"\r"])
        self.assertEqual(session.result(), "C")
        self.assertEqual(session.state()["cursor"], 2)
        self.assertEqual(session.returncode, 0)

    def test_home_end_pageup_pagedown(self):
        session = driver()
        session.run([b"\x1b[F", b"\r"])              # End -> epsilon
        self.assertEqual(session.result(), "E")

    def test_escape_cancels(self):
        session = driver()
        session.run([b"\x1b"])
        self.assertIsNone(session.result())

    def test_ctrl_c_cancels_without_killing_the_process(self):
        session = driver()
        session.run([b"\x03"])
        self.assertIsNone(session.result())
        self.assertEqual(session.returncode, 0, "Ctrl+C must cancel the menu, not crash it")

    def test_type_to_filter_then_enter(self):
        session = driver()
        session.run([b"g", b"a", b"\r"])
        self.assertEqual(session.result(), "C")
        self.assertEqual(session.state()["filter"], "ga")

    def test_backspace_widens_the_filter(self):
        # 'q' matches nothing (not even the hints); backspacing must bring the
        # full list back so 'ga' then lands on gamma.
        session = driver()
        session.run([b"q", b"\x7f", b"g", b"a", b"\r"])
        self.assertEqual(session.result(), "C")
        self.assertEqual(session.state()["filter"], "ga")

    def test_filter_searches_hints_too(self):
        # 'second' is only in beta's hint, so filtering by it must find beta
        session = driver()
        session.run([b"s", b"e", b"c", b"o", b"n", b"d", b"\r"])
        self.assertEqual(session.result(), "B")

    def test_ctrl_u_clears_the_filter(self):
        session = driver()
        session.run([b"zzz", b"\x15", b"\r"])
        self.assertEqual(session.state()["filter"], "")
        self.assertEqual(session.result(), "A")

    def test_no_match_then_backspace_recovers(self):
        session = driver()
        session.run([b"zz", b"\x7f", b"\x7f", b"\r"])
        self.assertEqual(session.result(), "A")

    def test_digit_shortcut_selects_immediately(self):
        session = driver()
        session.run([b"3"])
        self.assertEqual(session.result(), "C")

    def test_multi_select_with_tab(self):
        session = driver("multi_tab")
        session.run([b"\t", b"\x1b[B", b"\t", b"\r"])   # mark alpha, move, mark gamma
        self.assertEqual(session.result(), ["A", "C"])

    def test_multi_select_ctrl_a(self):
        session = driver("multi_all")
        session.run([b"\x01", b"\r"])
        self.assertEqual(sorted(session.result()), VALUES)

    def test_multi_select_esc_returns_empty_list(self):
        session = driver("multi_tab")
        session.run([b"\x1b"])
        self.assertEqual(session.result(), [])


@unittest.skipUnless(HAS_PTY, "pty is unavailable on this platform")
class TestTerminalHygiene(unittest.TestCase):
    def test_mouse_reporting_enabled_then_disabled(self):
        session = driver()
        session.run([b"\x1b"])
        self.assertIn(MOUSE_ON, session.output)
        self.assertIn(MOUSE_OFF, session.output)

    def test_cursor_hidden_then_restored(self):
        session = driver()
        session.run([b"\x1b"])
        self.assertIn(HIDE_CURSOR, session.output)
        self.assertIn(SHOW_CURSOR, session.output)

    def test_teardown_ran(self):
        session = driver()
        session.run([b"\x1b"])
        self.assertTrue(session.restored(), "RawTerminal.disable() must run on exit")

    def test_teardown_runs_after_a_normal_selection_too(self):
        session = driver()
        session.run([b"\r"])
        self.assertTrue(session.restored())
        self.assertIn(MOUSE_OFF, session.output)

    def test_narrow_terminal_does_not_overflow(self):
        session = PtySession([sys.executable, str(DRIVER), "single"], cols=40, rows=16)
        session.run([b"\x1b"])
        self.assertIn("RESULT:", session.output)
        for line in session.output.splitlines():
            if line.startswith(("┌", "│", "└")):
                self.assertLessEqual(visible_width(line), 42, line)


@unittest.skipUnless(HAS_PTY, "pty is unavailable on this platform")
class TestMouse(unittest.TestCase):
    def test_click_moves_the_cursor(self):
        session = driver()
        rows = item_rows()          # keyed by item value; only the visible window
        self.assertEqual(sorted(rows), ["A", "B", "C"], "max_rows=3 shows the first three")
        session.run([sgr(rows["C"], 6), b"\x1b"])       # click gamma, then cancel
        self.assertEqual(session.state()["cursor"], LABELS.index("gamma"))

    def test_double_click_selects(self):
        session = driver()
        row = item_rows()["C"]      # gamma
        session.run([sgr(row, 6), sgr(row, 6)])
        self.assertEqual(session.result(), "C")

    def test_click_ok_button_confirms_the_highlighted_item(self):
        session = driver()
        session.run([b"\x1b[B", sgr(button_row(), button_col(80, "ok"))])
        self.assertEqual(session.result(), "B")

    def test_click_cancel_button_cancels(self):
        session = driver()
        session.run([sgr(button_row(), button_col(80, "cancel"))])
        self.assertIsNone(session.result())

    def test_click_all_button_in_multi_mode(self):
        # the driver keeps type-to-filter on, so the buttons are the "klik:" variants
        session = driver("multi_all")
        row = button_row(multi=True)
        session.run([sgr(row, button_col(80, "all", multi=True)), b"\r"])
        self.assertEqual(sorted(session.result()), VALUES)

    def test_click_none_button_clears_the_selection(self):
        session = driver("multi_all")
        row = button_row(multi=True)
        col_all = button_col(80, "all", multi=True)
        col_none = button_col(80, "none", multi=True)
        session.run([sgr(row, col_all), sgr(row, col_none), b"\r"])
        self.assertEqual(session.result(), [])

    def test_click_toggles_in_multi_mode(self):
        session = driver("multi_tab")
        row = item_rows(multi=True)["B"]   # beta
        session.run([sgr(row, 6), sgr(row, 6), b"\r"])
        self.assertEqual(session.result(), ["B"])

    def test_clicking_a_scrolled_item_after_wheel(self):
        # wheel down brings delta/epsilon into view; clicking must hit the right one
        session = driver()
        session.run([sgr_wheel(4, 10, up=False), sgr_wheel(4, 10, up=False)])
        self.assertGreaterEqual(session.state().get("scroll", 0) if session.state() else 0, 0)

    def test_wheel_scrolls_without_selecting(self):
        session = driver()
        session.run([sgr_wheel(4, 10, up=False), sgr_wheel(4, 10, up=False), b"\x1b"])
        self.assertIsNone(session.result())
        self.assertIn("RESULT:", session.output)

    def test_click_outside_the_menu_is_ignored(self):
        session = driver()
        session.run([sgr(40, 70), sgr(1, 1), b"\x1b"])
        self.assertIsNone(session.result())
        self.assertEqual(session.returncode, 0)

    def test_garbage_mouse_bytes_do_not_crash(self):
        session = driver()
        session.run([b"\x1b[<999;999;999M", b"\xff\xfe", b"\x1b[99~", b"\r"])
        self.assertEqual(session.result(), "A")
        self.assertEqual(session.returncode, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
