"""Raw terminal input: keys and mouse events, pure stdlib.

Terminals deliver input as byte streams full of escape sequences. This module
turns those bytes into typed :class:`Event` objects so the rest of the UI never
has to think about ``\\x1b[<0;12;4M`` again.

Three properties matter and are all tested without a terminal:

* **Incremental parsing.** Bytes may arrive split anywhere -- including in the
  middle of a mouse sequence or a UTF-8 character -- so the parser is fed chunks
  and holds back anything incomplete.
* **Esc vs escape sequence.** A lone ``\\x1b`` is the Escape key; ``\\x1b[A`` is
  Up. The driver disambiguates by checking whether more bytes are already
  available.
* **Mouse reporting is opt-in and reversible.** Enabling it (SGR 1006 + button
  events) makes the terminal send clicks to us instead of doing its own text
  selection, so it must be turned off again -- including on crash and on
  ``SIGWINCH``/``SIGINT`` -- or the user's terminal is left in a broken state.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

# --------------------------------------------------------------------------- #
# Events
# --------------------------------------------------------------------------- #
KEY_NAMES = {
    "up": "\x1b[A", "down": "\x1b[B", "right": "\x1b[C", "left": "\x1b[D",
    "home": "\x1b[H", "end": "\x1b[F",
    "pageup": "\x1b[5~", "pagedown": "\x1b[6~", "delete": "\x1b[3~",
    "shift_tab": "\x1b[Z",
    "f1": "\x1bOP", "f2": "\x1bOQ", "f3": "\x1bOR", "f4": "\x1bOS",
    "f5": "\x1b[15~", "f6": "\x1b[17~", "f7": "\x1b[18~", "f8": "\x1b[19~",
    "f9": "\x1b[20~", "f10": "\x1b[21~", "f11": "\x1b[23~", "f12": "\x1b[24~",
}
_NAME_BY_SEQ = {v: k for k, v in KEY_NAMES.items()}
# xterm sends these older forms for Home/End and F1-F4
_LEGACY_SEQ = {
    "\x1b[1~": "home", "\x1b[4~": "end", "\x1b[7~": "home", "\x1b[8~": "end",
    "\x1b[11~": "f1", "\x1b[12~": "f2", "\x1b[13~": "f3", "\x1b[14~": "f4",
}
_CTRL = {0: "ctrl_space", 1: "ctrl_a", 3: "ctrl_c", 4: "ctrl_d", 5: "ctrl_e",
         8: "backspace", 9: "tab", 10: "enter", 11: "ctrl_k", 12: "ctrl_l",
         13: "enter", 21: "ctrl_u", 23: "ctrl_w", 26: "ctrl_z", 27: "esc",
         127: "backspace"}


@dataclass(frozen=True)
class KeyEvent:
    key: str          # 'up', 'enter', 'esc', 'ctrl_c', 'tab', or a literal char
    char: str = ""    # printable character, if any

    @property
    def is_printable(self) -> bool:
        return bool(self.char)


@dataclass(frozen=True)
class MouseEvent:
    kind: str         # 'press' | 'release' | 'motion' | 'wheel_up' | 'wheel_down'
    x: int = 0        # 1-based column
    y: int = 0        # 1-based row
    button: int = 0   # 0 left, 1 middle, 2 right
    shift: bool = False
    alt: bool = False
    ctrl: bool = False


@dataclass(frozen=True)
class ResizeEvent:
    cols: int
    rows: int


Event = object  # KeyEvent | MouseEvent | ResizeEvent


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #
class InputParser:
    """Feed bytes, get events. Holds back incomplete sequences."""

    def __init__(self) -> None:
        self._buf = b""
        #: set by the driver once a lone ESC has idled, so it is emitted as a key
        self.esc_pending = False

    def feed(self, data: bytes) -> Iterator[Event]:
        self._buf += data
        while self._buf:
            event, consumed = self._take()
            if consumed == 0:
                break  # incomplete: wait for more bytes
            self._buf = self._buf[consumed:]
            if event is not None:
                yield event

    # -- internals --------------------------------------------------------
    def _take(self) -> Tuple[Optional[Event], int]:
        buf = self._buf
        if buf[:1] == b"\x1b":
            return self._take_escape(buf)
        return self._take_plain(buf)

    def _take_plain(self, buf: bytes) -> Tuple[Optional[Event], int]:
        byte = buf[0]
        name = _CTRL.get(byte)
        if name == "esc":
            return self._take_escape(buf)
        if name is not None:
            return KeyEvent(name), 1
        if byte < 0x20:
            return None, 1  # unknown control byte: drop it rather than wedge
        # UTF-8 multi-byte character
        length = _utf8_len(byte)
        if length > 1:
            have = buf[:length]
            for i in range(1, len(have)):
                if not 0x80 <= have[i] <= 0xBF:
                    # Not a continuation byte: the lead byte is bogus. Drop only
                    # it and re-parse the rest, so a real key (like Enter) that
                    # happens to follow is not swallowed with it.
                    return None, 1
            if len(buf) < length:
                return None, 0  # well-formed so far, just incomplete
            try:
                char = buf[:length].decode("utf-8")
            except UnicodeDecodeError:
                return None, 1  # overlong or out of range: drop the lead byte
            return KeyEvent("char", char=char), length
        try:
            char = buf[:1].decode("utf-8")
        except UnicodeDecodeError:
            return None, 1
        return KeyEvent("char", char=char), 1

    def _take_escape(self, buf: bytes) -> Tuple[Optional[Event], int]:
        if len(buf) == 1:
            # Only one byte so far: it is either the start of a sequence or a
            # genuine Escape press. The driver decides by calling mark_esc() once
            # the byte has sat idle with nothing following it.
            if self.esc_pending:
                return KeyEvent("esc"), 1
            return None, 0
        second = buf[1:2]
        # Alt+<char> arrives as ESC followed by a printable byte
        if second not in (b"[", b"O", b"M", b"<"):
            if second[0] >= 0x20:
                length = _utf8_len(second[0])
                if len(buf) < 1 + length:
                    return None, 0  # incomplete Alt+multibyte char
                try:
                    char = buf[1 : 1 + length].decode("utf-8")
                except UnicodeDecodeError:
                    return KeyEvent("esc"), 1
                return KeyEvent("alt", char=char), 1 + length
            return KeyEvent("esc"), 1
        if second == b"O" and len(buf) >= 3:
            name = _NAME_BY_SEQ.get(buf[:3].decode("latin-1"))
            if name:
                return KeyEvent(name), 3
            return None, 3
        if second == b"[":
            third = buf[2:3]
            if third == b"M":
                # legacy X10 mouse: ESC [ M Cb Cx Cy  (each value is byte + 32)
                if len(buf) < 6:
                    return None, 0
                cb, cx, cy = buf[3] - 32, buf[4] - 32, buf[5] - 32
                return _mouse_from_codes(cb, cx, cy, sgr=False), 6
            # SGR mouse: ESC [ < Cb ; Cx ; Cy M|m
            if third == b"<":
                end = _find_terminator(buf, 3, b"Mm")
                if end < 0:
                    return None, 0
                body = buf[3:end].decode("latin-1")
                parts = body.split(";")
                if len(parts) != 3:
                    return None, end + 1
                try:
                    cb, cx, cy = int(parts[0]), int(parts[1]), int(parts[2])
                except ValueError:
                    return None, end + 1
                return _mouse_from_codes(cb, cx, cy, sgr=True,
                                          release=buf[end:end + 1] == b"m"), end + 1
            # CSI key sequence: ESC [ <params> <final byte>
            end = _find_terminator(buf, 2, b"ABCDEFGHPQRSZ~")
            if end < 0:
                return None, 0
            seq = buf[: end + 1].decode("latin-1")
            name = _NAME_BY_SEQ.get(seq) or _LEGACY_SEQ.get(seq)
            return (KeyEvent(name) if name else None), end + 1
        return KeyEvent("esc"), 1

    # -- driver hint ------------------------------------------------------
    def mark_esc(self) -> None:
        """Called by the driver when a lone ESC has sat idle: emit it as a key."""
        self.esc_pending = True

    def clear_esc(self) -> None:
        self.esc_pending = False


def _find_terminator(buf: bytes, start: int, finals: bytes) -> int:
    for i in range(start, len(buf)):
        if buf[i : i + 1] in finals:
            return i
    return -1


def _utf8_len(byte: int) -> int:
    """Expected length of a UTF-8 sequence from its lead byte.

    Invalid lead bytes (0x80-0xC1 and 0xF5-0xFF can never start a character)
    report length 1 so they are dropped immediately. Waiting for continuation
    bytes that will never arrive would swallow the next real key -- including
    the Enter that closes a menu.
    """
    if byte < 0x80:
        return 1
    if 0xC2 <= byte <= 0xDF:
        return 2
    if 0xE0 <= byte <= 0xEF:
        return 3
    if 0xF0 <= byte <= 0xF4:
        return 4
    return 1   # invalid lead byte: consume and discard


def _mouse_from_codes(cb: int, cx: int, cy: int, *, sgr: bool,
                      release: bool = False) -> MouseEvent:
    """Decode a mouse button code into an event (SGR and legacy X10)."""
    shift = bool(cb & 4)
    alt = bool(cb & 8)
    ctrl = bool(cb & 16)
    motion = bool(cb & 32)
    wheel = cb & 64
    if wheel:
        kind = "wheel_up" if (cb & 1) == 0 else "wheel_down"
        return MouseEvent(kind=kind, x=max(1, cx), y=max(1, cy), button=0,
                          shift=shift, alt=alt, ctrl=ctrl)
    button = cb & 3
    if motion:
        return MouseEvent(kind="motion", x=max(1, cx), y=max(1, cy), button=button if button != 3 else 0,
                          shift=shift, alt=alt, ctrl=ctrl)
    if button == 3 and not sgr:
        # legacy protocol reports release as button 3 with no button number
        return MouseEvent(kind="release", x=max(1, cx), y=max(1, cy), button=0,
                          shift=shift, alt=alt, ctrl=ctrl)
    return MouseEvent(kind="release" if release else "press", x=max(1, cx), y=max(1, cy),
                      button=button, shift=shift, alt=alt, ctrl=ctrl)


# --------------------------------------------------------------------------- #
# Mouse mode escape sequences
# --------------------------------------------------------------------------- #
#: Button events + SGR extended coordinates. Motion tracking is deliberately NOT
#: enabled: it floods stdin and breaks text selection for no benefit here.
MOUSE_ON = "\x1b[?1000h\x1b[?1002h\x1b[?1006h"
MOUSE_OFF = "\x1b[?1000l\x1b[?1002l\x1b[?1006l"
HIDE_CURSOR = "\x1b[?25l"
SHOW_CURSOR = "\x1b[?25h"
ALT_SCREEN_ON = "\x1b[?1049h"
ALT_SCREEN_OFF = "\x1b[?1049l"


def mouse_supported(*, stream=None) -> bool:
    """Mouse reporting needs a real terminal and a TERM that is not dumb."""
    stream = stream if stream is not None else sys.stdout
    if os.environ.get("NEXUS_NO_MOUSE"):
        return False
    if not getattr(stream, "isatty", lambda: False)():
        return False
    if os.name == "nt":
        # Windows Console mouse reporting works in Windows Terminal but the
        # stdin path differs; keyboard navigation is guaranteed instead.
        return os.environ.get("WT_SESSION") is not None
    term = (os.environ.get("TERM") or "").lower()
    return term not in ("", "dumb")


# --------------------------------------------------------------------------- #
# Raw terminal driver
# --------------------------------------------------------------------------- #
class RawTerminal:
    """Puts the tty into raw mode and yields parsed events.

    Usage::

        with RawTerminal() as term:
            for event in term.events():
                ...

    Restores the previous terminal state on exit even when an exception or
    ``Ctrl+C`` interrupts the block, and re-enables the cursor.
    """

    def __init__(self, *, want_mouse: Optional[bool] = None, stream=None) -> None:
        self.stream = stream if stream is not None else sys.stdout
        self.parser = InputParser()
        self._fd: Optional[int] = None
        self._saved = None
        self._mouse = mouse_supported(stream=self.stream) if want_mouse is None else bool(want_mouse)
        self._active = False
        self._is_posix = hasattr(sys.stdin, "fileno") and os.name != "nt"

    # -- context manager --------------------------------------------------
    def __enter__(self) -> "RawTerminal":
        self.enable()
        return self

    def __exit__(self, *exc) -> None:
        self.disable()

    def enable(self) -> None:
        if self._active:
            return
        if not self._enter_raw():
            return
        self._active = True
        self._write(HIDE_CURSOR)
        if self._mouse:
            self._write(MOUSE_ON)

    def disable(self) -> None:
        if not self._active:
            return
        self._active = False
        if self._mouse:
            self._write(MOUSE_OFF)
        self._write(SHOW_CURSOR)
        self._leave_raw()

    def _write(self, text: str) -> None:
        try:
            self.stream.write(text)
            self.stream.flush()
        except (ValueError, OSError):
            pass

    def _enter_raw(self) -> bool:
        if os.name == "nt":
            try:
                import msvcrt  # noqa: F401

                self._msvcrt = True
                return True
            except Exception:
                return False
        try:
            import termios
            import tty

            self._fd = sys.stdin.fileno()
            self._saved = termios.tcgetattr(self._fd)
            tty.setraw(self._fd)
            return True
        except Exception:
            self._fd = None
            self._saved = None
            return False

    def _leave_raw(self) -> None:
        if os.name == "nt":
            self._msvcrt = False
            return
        if self._fd is None or self._saved is None:
            return
        try:
            import termios

            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)
        except Exception:
            pass
        finally:
            self._fd = None
            self._saved = None

    # -- reading ----------------------------------------------------------
    @property
    def active(self) -> bool:
        return self._active

    def read_events(self, timeout: float = 0.05) -> List[Event]:
        """Read whatever is available within *timeout* and parse it."""
        if not self._active:
            return []
        data = self._read_available(timeout)
        if not data:
            return []
        return list(self.parser.feed(data))

    def _read_available(self, timeout: float) -> bytes:
        if os.name == "nt":
            return self._read_msvcrt(timeout)
        import select

        try:
            ready, _, _ = select.select([sys.stdin], [], [], timeout)
        except (OSError, ValueError):
            return b""
        if not ready:
            return b""
        try:
            return os.read(sys.stdin.fileno(), 4096)
        except OSError:
            return b""

    def _read_msvcrt(self, timeout: float) -> bytes:  # pragma: no cover - Windows
        import msvcrt

        out = bytearray()
        deadline = timeout
        while True:
            if not msvcrt.kbhit():
                if out or deadline <= 0:
                    break
                import time

                time.sleep(0.01)
                deadline -= 0.01
                continue
            ch = msvcrt.getwch()
            if ch in ("\x00", "\xe0"):
                code = msvcrt.getwch()
                out += _windows_arrow(code)
            elif ch == "\x03":
                raise KeyboardInterrupt
            else:
                out += ch.encode("utf-8", "replace")
        return bytes(out)

    def inject(self, data: bytes) -> None:
        """Push bytes back to the front of the parser buffer.

        Used when a control reply (e.g. the cursor-position report) arrives with
        user input glued behind it: the reply is consumed by us, the rest must
        still reach the parser.
        """
        self.parser._buf = data + self.parser._buf

    def query_cursor(self, stream, timeout: float = 0.25) -> Tuple[int, int]:
        """Ask the terminal where the cursor is (DSR ``ESC[6n``).

        Returns ``(row, col)``, 1-based. Falls back to ``(1, 1)`` when the
        terminal does not answer, which keeps menus usable on dumb terminals --
        at the cost of mouse clicks not resolving exactly.
        """
        import re as _re
        import time as _time

        try:
            stream.write("\x1b[6n")
            stream.flush()
        except (ValueError, OSError):
            return 1, 1
        buf = b""
        deadline = _time.monotonic() + timeout
        while _time.monotonic() < deadline:
            data = self._read_available(0.02)
            if data:
                buf += data
                match = _re.search(rb"\x1b\[(\d+);(\d+)R", buf)
                if match:
                    rest = buf[match.end():]
                    if rest:
                        self.inject(rest)
                    try:
                        return int(match.group(1)), int(match.group(2))
                    except ValueError:
                        return 1, 1
            elif buf:
                break
        if buf:
            self.inject(buf)
        return 1, 1

    def pending_esc(self) -> bool:
        """True when a lone ESC has been sitting in the buffer (so it is a key)."""
        return self.parser._buf == b"\x1b"

    def flush_esc(self) -> List[Event]:
        """Emit a held lone ESC as an Escape key press."""
        if self.parser._buf == b"\x1b":
            self.parser.mark_esc()
            events = list(self.parser.feed(b""))
            self.parser.clear_esc()
            self.parser._buf = b""
            return events
        return []

    @staticmethod
    def size() -> Tuple[int, int]:
        try:
            size = os.get_terminal_size(sys.stdout.fileno())
            return size.columns, size.lines
        except (OSError, ValueError, AttributeError):
            return 100, 30


def _windows_arrow(code: str) -> bytes:  # pragma: no cover - Windows
    return {
        "H": KEY_NAMES["up"].encode(), "P": KEY_NAMES["down"].encode(),
        "M": KEY_NAMES["right"].encode(), "K": KEY_NAMES["left"].encode(),
        "G": KEY_NAMES["home"].encode(), "O": KEY_NAMES["end"].encode(),
        "I": KEY_NAMES["pageup"].encode(), "Q": KEY_NAMES["pagedown"].encode(),
        "S": KEY_NAMES["delete"].encode(),
    }.get(code, b"")


__all__ = ["KeyEvent", "MouseEvent", "ResizeEvent", "InputParser", "RawTerminal",
           "KEY_NAMES", "MOUSE_ON", "MOUSE_OFF", "HIDE_CURSOR", "SHOW_CURSOR",
           "ALT_SCREEN_ON", "ALT_SCREEN_OFF", "mouse_supported"]
