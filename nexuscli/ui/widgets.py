"""Terminal widgets: spinner, boxes, tables, progress, formatting helpers."""

from __future__ import annotations

import itertools
import sys
import threading
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .markdown import render_table
from .theme import Style, pad, strip_ansi, truncate, visible_width, wrap_text

SPINNER_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
ASCII_SPINNER = ("|", "/", "-", "\\")


def format_duration(ms: float) -> str:
    if ms < 1000:
        return f"{int(ms)}ms"
    seconds = ms / 1000
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def format_number(n: float) -> str:
    n = float(n)
    if abs(n) >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if abs(n) >= 1_000:
        return f"{n / 1_000:.1f}k"
    return f"{int(n)}"


def format_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024 or unit == "GB":
            return f"{int(n)}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}GB"


def format_cost(usd: float) -> str:
    if usd == 0:
        return "$0"
    if usd < 0.01:
        return f"${usd:.4f}"
    return f"${usd:.3f}"


class Spinner:
    """Background spinner that never corrupts output when stdout is not a TTY."""

    def __init__(self, style: Style, message: str = "", *, stream=None, enabled: bool = True,
                 lock: Optional[threading.Lock] = None) -> None:
        self.style = style
        self.message = message
        self.stream = stream if stream is not None else sys.stdout
        self.enabled = bool(enabled and getattr(self.stream, "isatty", lambda: False)())
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        # Shared with the Renderer when provided: painting a frame and erasing
        # it must be atomic against application output, or the two writers
        # interleave (garbled lines under tmux/screen).
        self._lock = lock if lock is not None else threading.Lock()
        self._started = time.monotonic()
        self._frames = itertools.cycle(SPINNER_FRAMES if style.enabled else ASCII_SPINNER)

    def set_message(self, message: str) -> None:
        with self._lock:
            self.message = message

    def _paint(self, text: str) -> None:
        """Write one in-place update; caller must hold ``self._lock``."""
        try:
            self.stream.write(text)
            self.stream.flush()
        except (ValueError, OSError):
            pass

    def clear_line(self, stream=None) -> None:
        """Erase the spinner's current line. Locks, so it cannot race a frame."""
        with self._lock:
            if self.enabled:
                self._paint("\r\x1b[K")

    def start(self, message: Optional[str] = None) -> "Spinner":
        if message is not None:
            self.message = message
        if not self.enabled or (self._thread and self._thread.is_alive()):
            return self
        self._stop.clear()
        self._started = time.monotonic()
        self._thread = threading.Thread(target=self._spin, daemon=True, name="nexus-spinner")
        self._thread.start()
        return self

    def _spin(self) -> None:
        try:
            while not self._stop.is_set():
                with self._lock:
                    message = self.message
                    frame = next(self._frames)
                    elapsed = time.monotonic() - self._started
                    text = f"\r{self.style.paint('accent', frame)} {self.style.dim(truncate(message, 70))}"
                    if elapsed >= 2:
                        text += self.style.dim(f" {elapsed:.0f}s")
                    text += "\x1b[K"
                    self._paint(text)
                self._stop.wait(0.08)
        except Exception:  # a spinner must never break the app
            pass

    def stop(self, final: str = "") -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        with self._lock:
            if self.enabled:
                self._paint("\r\x1b[K")
            if final:
                self._paint(final + "\n")

    def __enter__(self) -> "Spinner":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


def box(title: str, lines: Sequence[str], style: Style, *, width: int = 0, role: str = "border",
        indent: str = "") -> str:
    """Rounded box, width-aware (handles CJK content correctly)."""
    width = width or style.width
    inner = max(10, min(width - 4, 100))
    body: List[str] = []
    for line in lines:
        if not line:
            body.append("")
            continue
        for wrapped in wrap_text(line, inner):
            body.append(wrapped)
    top_title = f" {title} " if title else ""
    # The title segment must fit inside the frame: otherwise the fill goes to
    # zero and the top border ends up wider than every other row (ragged box,
    # especially visible with CJK/emoji titles). Total width is always inner+4.
    top_title = truncate(top_title, max(0, inner))
    # "┌─" + title + fill + "┐" must total inner+4 cells: 2 + title + fill + 1.
    top = "┌─" + top_title + "─" * max(0, inner + 1 - visible_width(top_title)) + "┐"
    rows = [style.paint(role, top)]
    for line in body:
        fill = max(0, inner - visible_width(line))
        rows.append(style.paint(role, "│ ") + line + " " * fill + style.paint(role, " │"))
    rows.append(style.paint(role, "└" + "─" * (inner + 2) + "┘"))
    return "\n".join(indent + r for r in rows)


def table(headers: Sequence[str], rows: Sequence[Sequence[Any]], style: Style, *, width: int = 0,
          aligns: Optional[Sequence[str]] = None, inline: bool = False) -> str:
    """Data table. Inline markdown is OFF by default (cells are data, not prose)."""
    data = [[str(c) for c in row] for row in rows]
    if headers:
        data = [[str(h) for h in headers], *data]
    return "\n".join(render_table(data, style, width or style.width, list(aligns or []), inline=inline))


def kv(pairs: Sequence[Tuple[str, Any]], style: Style, *, width: int = 0, key_width: int = 0) -> str:
    """Aligned ``key  value`` listing with wrapped values."""
    width = width or style.width
    key_width = key_width or max((visible_width(str(k)) for k, _ in pairs), default=8)
    out: List[str] = []
    for key, value in pairs:
        prefix = style.paint("accent", pad(truncate(str(key), key_width), key_width)) + "  "
        cont = " " * (key_width + 2)
        wrapped = wrap_text(str(value), max(10, width - key_width - 2))
        for i, line in enumerate(wrapped):
            out.append((prefix if i == 0 else cont) + line)
    return "\n".join(out)


def progress(done: int, total: int, style: Style, *, width: int = 30, label: str = "") -> str:
    total = max(1, total)
    ratio = max(0.0, min(1.0, done / total))
    filled = int(round(width * ratio))
    bar = "█" * filled + "░" * (width - filled)
    color = "success" if ratio >= 1 else "accent"
    text = f"{style.paint(color, bar)} {int(ratio * 100):3d}% ({done}/{total})"
    return f"{label + ' ' if label else ''}{text}"


def rule(style: Style, title: str = "", *, width: int = 0, role: str = "border") -> str:
    width = width or style.width
    if not title:
        return style.paint(role, "─" * min(width - 1, 78))
    left = f"── {title} "
    fill = max(2, min(width - 1, 78) - visible_width(left))
    return style.paint(role, left + "─" * fill)


def badge(text: str, style: Style, role: str = "accent") -> str:
    return style.paint(role, f" {text.strip()} ")


def status_icon(ok: bool, style: Style) -> str:
    return style.paint("success" if ok else "error", "✓" if ok else "✗")


def columns(items: Sequence[str], style: Style, *, width: int = 0, min_col: int = 18) -> str:
    """Lay out a list of short strings in aligned columns (used by /help)."""
    width = width or style.width
    if not items:
        return ""
    longest = max(visible_width(i) for i in items)
    col_width = max(min_col, longest + 2)
    per_row = max(1, width // col_width)
    rows: List[str] = []
    for start in range(0, len(items), per_row):
        chunk = items[start : start + per_row]
        rows.append("".join(pad(truncate(c, col_width - 2), col_width) for c in chunk))
    return "\n".join(rows)


__all__ = ["Spinner", "box", "table", "kv", "progress", "rule", "badge", "columns",
           "format_duration", "format_number", "format_bytes", "format_cost", "status_icon"]
