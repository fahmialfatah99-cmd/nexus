"""End-to-end tests: drive the real ``nexus`` CLI inside a pseudo-terminal.

``test_menu_interactive.py`` proves the menu widget works; this proves the *shipped
program* wires it up correctly -- startup, REPL, command dispatch, menu, effect on
settings, and the ``--lang`` / ``--no-menu`` flags.

Each test pays a real interpreter start (~3s), so the module is skipped when
``pty`` is missing and can be deselected with ``NEXUS_SKIP_E2E=1``.
"""

from __future__ import annotations

import fcntl
import os
import pty
import re
import select
import struct
import subprocess
import sys
import termios
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
NEXUS = ROOT / "nexus"

try:
    import pty as _pty  # noqa: F401

    HAS_PTY = True
except Exception:  # pragma: no cover - Windows
    HAS_PTY = False

ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\x1b[()][A-Za-z0-9]|\x1b[=>]|\x1b\[\?[0-9]+[hl]")

SKIP = os.environ.get("NEXUS_SKIP_E2E") or not HAS_PTY or not NEXUS.exists()


def clean(text: str) -> str:
    """Strip ANSI and collapse the repeated frames produced by in-place redraws."""
    text = ANSI.sub("", text).replace("\x1b", "").replace("\r", "\n")
    out = []
    for line in (l.rstrip() for l in text.split("\n")):
        if out and out[-1] == line:
            continue
        out.append(line)
    return "\n".join(out)


class CliSession:
    def __init__(self, extra_args=(), cols: int = 100, rows: int = 30):
        self.argv = [sys.executable, str(NEXUS), *extra_args, "-C", str(ROOT)]
        self.cols = cols
        self.rows = rows
        self.raw = ""

    def run(self, inputs, *, settle: float = 0.35, startup: float = 3.0, tail: float = 1.2,
            timeout: float = 45.0) -> str:
        master, slave = pty.openpty()
        try:
            fcntl.ioctl(slave, termios.TIOCSWINSZ,
                        struct.pack("HHHH", self.rows, self.cols, 0, 0))
        except Exception:
            pass
        env = {k: v for k, v in os.environ.items() if k != "NO_COLOR"}
        env.update({"TERM": "xterm-256color", "COLORTERM": "truecolor",
                    "PYTHONIOENCODING": "utf-8", "NEXUS_NO_MENU": "",
                    "LANG": "en_US.UTF-8", "LC_ALL": "en_US.UTF-8", "NEXUS_LANG": "en"})
        env.pop("NEXUS_NO_MENU", None)
        proc = subprocess.Popen(self.argv, stdin=slave, stdout=slave, stderr=slave,
                                env=env, close_fds=True, cwd=str(ROOT))
        os.close(slave)
        deadline = time.monotonic() + timeout

        def pump(budget: float) -> None:
            end = time.monotonic() + budget
            while time.monotonic() < end and time.monotonic() < deadline:
                ready, _, _ = select.select([master], [], [], 0.05)
                if not ready:
                    continue
                try:
                    chunk = os.read(master, 65536)
                except OSError:
                    return
                if not chunk:
                    return
                self.raw += chunk.decode("utf-8", "replace")

        try:
            pump(startup)
            for payload in inputs:
                os.write(master, payload if isinstance(payload, bytes) else payload.encode())
                pump(settle)
            pump(tail)
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
        return clean(self.raw)


@unittest.skipIf(SKIP, "pty / launcher unavailable, or NEXUS_SKIP_E2E=1")
class TestCliMenus(unittest.TestCase):
    maxDiff = None

    def assertMenu(self, text: str, needles, absent=()):
        for needle in needles:
            self.assertIn(needle, text, f"missing {needle!r} in:\n{text[-2000:]}")
        for needle in absent:
            self.assertNotIn(needle, text, f"unexpected {needle!r} in:\n{text[-2000:]}")

    def test_mode_menu_changes_the_approval_mode(self):
        text = CliSession().run(["/mode\n", b"\x1b[B", b"\x1b[B", b"\r"])
        self.assertMenu(text, ["Approval mode", "read-only", "suggest", "auto-edit",
                               "full-auto", "yolo", "Approval mode:"])

    def test_model_menu_with_a_matching_filter_applies(self):
        text = CliSession().run(["/model\n", "g", "e", "m", "\r"])
        self.assertMenu(text, ["Choose a model", "filter: gem", "Model set to"])

    def test_model_menu_with_no_match_does_not_select_anything(self):
        """Regression: Enter on an empty filter result used to pick an unrelated model."""
        text = CliSession().run(["/model\n", "z", "z", "z", "\r", "/exit\n"])
        self.assertMenu(text, ["no match"], absent=["Model set to"])

    def test_cast_menu_is_multi_select(self):
        text = CliSession().run(["/cast\n", b"\t", b"\x1b[B", b"\t", b"\r"])
        self.assertMenu(text, ["cast", "[x]"])

    def test_menu_command_browses_categories(self):
        text = CliSession().run(["/menu\n", b"\x1b[B", b"\r", b"\x1b"])
        self.assertMenu(text, ["All commands", "Commands ·"])

    def test_bare_slash_opens_the_menu(self):
        text = CliSession().run(["/\n", b"\x1b"])
        self.assertMenu(text, ["All commands"])

    def test_lang_flag_switches_the_menu_language(self):
        text = CliSession(extra_args=("--lang", "id")).run(["/mode\n", b"\r"])
        self.assertMenu(text, ["Mode persetujuan", "aktif sekarang", "enter pilih"])

    def test_no_menu_flag_keeps_the_text_interface(self):
        """--no-menu must give the documented text table, not an interactive menu.

        Only menu-specific chrome is asserted absent: box drawing also appears in
        the startup banner and in markdown tables, so it is not a valid signal.
        """
        text = CliSession(extra_args=("--no-menu",)).run(["/mode\n"])
        self.assertMenu(text, ["read-only", "Approval modes", "current: auto-edit"],
                        absent=["enter choose", "esc cancel"])

    def test_tools_picker_lists_tools(self):
        text = CliSession().run(["/tools pick\n", b"\x1b"])
        self.assertMenu(text, ["read_file", "bash", "DISABLE"])

    def test_mouse_reporting_is_switched_off_when_the_menu_closes(self):
        """Leaving mouse mode on would break text selection in the user's terminal."""
        session = CliSession()
        session.run(["/mode\n", b"\x1b"])
        self.assertIn("\x1b[?1000h", session.raw, "mouse reporting should be enabled")
        self.assertIn("\x1b[?1000l", session.raw, "and disabled again on exit")
        self.assertIn("\x1b[?25h", session.raw, "the cursor must be shown again")


if __name__ == "__main__":
    unittest.main(verbosity=2)
