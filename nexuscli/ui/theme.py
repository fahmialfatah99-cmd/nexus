"""ANSI primitives + themes.

Two things live here because everything visual depends on them:

1. **Width-correct text layout.** Terminals render CJK characters two cells wide,
   and ANSI escapes zero cells wide. Naive ``len()`` breaks every table, box and
   progress bar for non-Latin users, so :func:`visible_width` uses East-Asian
   width and all padding/truncation goes through it.
2. **Colour capability detection.** Honours ``NO_COLOR`` (the standard),
   ``FORCE_COLOR``, non-TTY output, ``TERM=dumb`` and Windows without VT
   support, then degrades 24-bit -> 256 -> 16 -> none. A theme is a mapping of
   semantic roles to colours, never hard-coded escapes in call sites.
"""

from __future__ import annotations

import os
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

ANSI_RE = re.compile(r"\x1b\[[0-9;:?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")
RESET = "\x1b[0m"


# --------------------------------------------------------------------------- #
# Text layout
# --------------------------------------------------------------------------- #
def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def char_width(ch: str) -> int:
    if ch == "\t":
        return 1  # wrap_text expands tabs; measure them as a single cell otherwise
    if unicodedata.combining(ch):
        return 0
    ea = unicodedata.east_asian_width(ch)
    if ea in ("W", "F"):
        return 2
    if unicodedata.category(ch) in ("Cc", "Cf"):
        return 0
    return 1


def visible_width(text: str) -> int:
    """Display width in terminal cells (ANSI-aware, wide-char aware)."""
    return sum(char_width(c) for c in strip_ansi(text))


def pad(text: str, width: int, align: str = "left", fill: str = " ") -> str:
    gap = max(0, width - visible_width(text))
    if align == "right":
        return fill * gap + text
    if align == "center":
        left = gap // 2
        return fill * left + text + fill * (gap - left)
    return text + fill * gap


def truncate(text: str, width: int, *, ellipsis: str = "…", keep_ansi: bool = True) -> str:
    """Cut to *width* cells. Preserves escape sequences when ``keep_ansi``."""
    if width <= 0:
        return ""
    if visible_width(text) <= width:
        return text
    limit = width - visible_width(ellipsis)
    if limit <= 0:
        return ellipsis[:width]
    out: List[str] = []
    used = 0
    i = 0
    while i < len(text):
        m = ANSI_RE.match(text, i)
        if m:
            if keep_ansi:
                out.append(m.group(0))
            i = m.end()
            continue
        ch = text[i]
        w = char_width(ch)
        if used + w > limit:
            break
        out.append(ch)
        used += w
        i += 1
    result = "".join(out) + ellipsis
    return result + RESET if (keep_ansi and "\x1b" in result and not result.endswith(RESET)) else result


def wrap_text(text: str, width: int, *, indent: str = "", subsequent_indent: str = "") -> List[str]:
    """Wrap to *width* cells without splitting ANSI sequences or wide chars."""
    width = max(4, width)
    lines: List[str] = []
    for raw_line in text.split("\n"):
        current = indent
        used = visible_width(indent)
        buf: List[str] = []
        i = 0
        line = raw_line
        first = True
        while i < len(line):
            m = ANSI_RE.match(line, i)
            if m:
                buf.append(m.group(0))
                i = m.end()
                continue
            ch = line[i]
            w = char_width(ch)
            if ch == "\t":
                ch, w = "    ", 4
            if used + w > width:
                out = current + "".join(buf)
                lines.append(out)
                buf = []
                first = False
                current = subsequent_indent or indent
                used = visible_width(current)
                if ch == " ":
                    i += 1
                    continue
            buf.append(ch)
            used += w
            i += 1
        tail = current + "".join(buf)
        if tail.strip() or first:
            lines.append(tail)
        elif buf:
            lines.append(tail)
    return lines if lines else [""]


def terminal_width(default: int = 100) -> int:
    """Best-effort output width.

    Order of precedence: ``NEXUS_WIDTH`` (explicit override, useful in
    tmux/screen where the detected size can be stale), then the real terminal,
    then ``COLUMNS``, then *default*. Anything we return is clamped to
    ``[40, 200]`` so boxes never collapse on very narrow terminals and never
    run away on ultrawide ones.
    """
    def clamp(value: int) -> int:
        return max(40, min(value, 200))
    for var in ("NEXUS_WIDTH", "COLUMNS"):
        try:
            value = int(os.environ.get(var, ""))
        except ValueError:
            continue
        if value > 20:
            return clamp(value)
    try:
        size = os.get_terminal_size(sys.stdout.fileno())
        if size.columns > 20:
            return clamp(size.columns)
    except (OSError, ValueError, AttributeError):
        pass
    return clamp(default)


# --------------------------------------------------------------------------- #
# Colour support
# --------------------------------------------------------------------------- #
def detect_color(*, stream=None, environ: Optional[Dict[str, str]] = None) -> str:
    """Return 'truecolor' | '256' | '16' | 'none'."""
    env = environ if environ is not None else os.environ
    if env.get("NO_COLOR"):
        return "none"
    if env.get("NEXUS_NO_COLOR"):
        return "none"
    forced = env.get("FORCE_COLOR") or env.get("CLICOLOR_FORCE")
    stream = stream if stream is not None else sys.stdout
    is_tty = bool(getattr(stream, "isatty", lambda: False)())
    if not is_tty and not forced:
        return "none"
    term = (env.get("TERM") or "").lower()
    if term in ("dumb", ""):
        return "none" if not forced else "16"
    if forced and term in ("dumb", ""):
        return "16"
    if env.get("COLORTERM") in ("truecolor", "24bit"):
        return "truecolor"
    if "256" in term or term in ("xterm-kitty", "alacritty", "wezterm", "xterm-ghostty"):
        return "256"
    if sys.platform == "win32":
        return "16"
    return "16"


class Color:
    """Builds SGR sequences for the detected capability level."""

    def __init__(self, level: str = "none") -> None:
        self.level = level if level in ("truecolor", "256", "16", "none") else "none"
        self.enabled = level != "none"

    def _wrap(self, code: str, text: str) -> str:
        if not self.enabled or not code:
            return text
        return f"\x1b[{code}m{text}{RESET}"

    def rgb(self, r: int, g: int, b: int, text: str, *, bg: bool = False) -> str:
        if not self.enabled:
            return text
        if self.level == "truecolor":
            return self._wrap(f"{'48' if bg else '38'};2;{r};{g};{b}", text)
        if self.level == "256":
            return self._wrap(f"{'48' if bg else '38'};5;{_rgb_to_256(r, g, b)}", text)
        return self._wrap(_ansi16(_rgb_to_16(r, g, b), bg), text)

    def basic(self, code: str, text: str) -> str:
        return self._wrap(code, text)

    def bold(self, text: str) -> str:
        return self._wrap("1", text)

    def dim(self, text: str) -> str:
        return self._wrap("2", text)

    def italic(self, text: str) -> str:
        return self._wrap("3", text)

    def underline(self, text: str) -> str:
        return self._wrap("4", text)

    def inverse(self, text: str) -> str:
        return self._wrap("7", text)


_BASIC_16 = [
    (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
    (0, 128, 128), (192, 192, 192), (128, 128, 128), (255, 0, 0), (0, 255, 0),
    (255, 255, 0), (0, 0, 255), (255, 0, 255), (0, 255, 255), (255, 255, 255),
]


def _ansi16(index: int, bg: bool = False) -> str:
    """Map a palette index 0-15 onto the right SGR code.

    0-7 are the normal colours (30-37 / 40-47) and 8-15 the bright ones
    (90-97 / 100-107). Concatenating "3" + index would produce 39 for bright
    red, which is *default foreground* -- a silent colour bug on every terminal
    that is not 256-colour capable.
    """
    index = max(0, min(15, int(index)))
    if index < 8:
        return str((40 if bg else 30) + index)
    return str((100 if bg else 90) + (index - 8))


def _rgb_to_16(r: int, g: int, b: int) -> int:
    best, dist = 0, None
    for i, (rr, gg, bb) in enumerate(_BASIC_16):
        d = (rr - r) ** 2 + (gg - g) ** 2 + (bb - b) ** 2
        if dist is None or d < dist:
            dist, best = d, i
    return best


def _rgb_to_256(r: int, g: int, b: int) -> int:
    if r == g == b:  # greyscale ramp
        if r < 8:
            return 16
        if r > 248:
            return 231
        return 232 + ((r - 8) * 24) // 247
    return 16 + 36 * (r // 51) + 6 * (g // 51) + (b // 51)


# --------------------------------------------------------------------------- #
# Themes
# --------------------------------------------------------------------------- #
DARK: Dict[str, Tuple[int, int, int]] = {
    "text": (223, 223, 223), "dim": (138, 145, 158), "bold": (255, 255, 255),
    "accent": (126, 196, 255), "accent2": (199, 146, 234), "success": (126, 231, 135),
    "warning": (255, 196, 92), "error": (255, 107, 107), "info": (110, 200, 255),
    "tool": (255, 179, 71), "agent": (199, 146, 234), "code": (174, 214, 241),
    "code_bg": (35, 38, 45), "border": (72, 78, 92), "keyword": (199, 146, 234),
    "string": (152, 195, 121), "number": (209, 154, 102), "comment": (108, 113, 128),
    "diff_add": (126, 231, 135), "diff_del": (255, 107, 107), "diff_hunk": (126, 196, 255),
    "user": (110, 231, 183),
}

LIGHT: Dict[str, Tuple[int, int, int]] = {
    "text": (40, 44, 52), "dim": (110, 118, 129), "bold": (0, 0, 0),
    "accent": (20, 90, 180), "accent2": (130, 60, 170), "success": (22, 130, 60),
    "warning": (176, 110, 0), "error": (200, 30, 40), "info": (20, 100, 170),
    "tool": (180, 100, 0), "agent": (130, 60, 170), "code": (30, 60, 110),
    "code_bg": (240, 241, 244), "border": (180, 186, 196), "keyword": (130, 60, 170),
    "string": (30, 110, 60), "number": (160, 80, 20), "comment": (130, 138, 148),
    "diff_add": (22, 130, 60), "diff_del": (200, 30, 40), "diff_hunk": (20, 90, 180),
    "user": (10, 120, 100),
}

MONO: Dict[str, Tuple[int, int, int]] = {k: (255, 255, 255) for k in DARK}

THEMES = {"dark": DARK, "light": LIGHT, "mono": MONO}


@dataclass
class Style:
    """Semantic styling: call sites ask for roles, never for raw escapes."""

    theme: Dict[str, Tuple[int, int, int]] = field(default_factory=lambda: dict(DARK))
    color: Color = field(default_factory=lambda: Color("none"))
    width: int = 100

    @staticmethod
    def create(theme_name: str = "dark", *, enabled: Optional[bool] = None,
               stream=None, width: int = 0, environ: Optional[Dict[str, str]] = None) -> "Style":
        """Build a style.

        ``enabled=None``  auto-detect (TTY, NO_COLOR, TERM)
        ``enabled=True``  force colour on -- used by ``--color`` so that piping
                          into ``less -R`` or a file still keeps the escapes
        ``enabled=False`` force colour off (``--no-color``)
        """
        env = environ if environ is not None else os.environ
        if enabled is False or env.get("NO_COLOR") or env.get("NEXUS_NO_COLOR"):
            level = "none"
        elif enabled is True:
            level = "truecolor" if env.get("COLORTERM") in ("truecolor", "24bit") else (
                "256" if "256" in (env.get("TERM") or "") else "16")
        else:
            level = detect_color(stream=stream, environ=env)
        color = Color(level)
        palette = dict(THEMES.get(theme_name, DARK))
        if level == "none":
            palette = dict(MONO)
        return Style(theme=palette, color=color, width=width or terminal_width())

    # -- semantic helpers -------------------------------------------------
    def paint(self, role: str, text: str) -> str:
        rgb = self.theme.get(role) or self.theme["text"]
        return self.color.rgb(*rgb, text)

    def text(self, s: str) -> str:
        return self.paint("text", s)

    def dim(self, s: str) -> str:
        return self.paint("dim", s)

    def bold(self, s: str) -> str:
        return self.color.bold(self.paint("bold", s))

    def accent(self, s: str) -> str:
        return self.paint("accent", s)

    def accent2(self, s: str) -> str:
        return self.paint("accent2", s)

    def success(self, s: str) -> str:
        return self.paint("success", s)

    def warning(self, s: str) -> str:
        return self.paint("warning", s)

    def error(self, s: str) -> str:
        return self.paint("error", s)

    def info(self, s: str) -> str:
        return self.paint("info", s)

    def tool(self, s: str) -> str:
        return self.paint("tool", s)

    def agent(self, s: str) -> str:
        return self.paint("agent", s)

    def user(self, s: str) -> str:
        return self.paint("user", s)

    def border(self, s: str) -> str:
        return self.paint("border", s)

    def code(self, s: str) -> str:
        return self.paint("code", s)

    @property
    def enabled(self) -> bool:
        return self.color.enabled


__all__ = [
    "strip_ansi", "visible_width", "char_width", "pad", "truncate", "wrap_text",
    "terminal_width", "detect_color", "Color", "Style", "THEMES", "DARK", "LIGHT",
    "MONO", "ANSI_RE", "RESET", "_ansi16",
]
