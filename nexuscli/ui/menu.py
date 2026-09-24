"""Interactive menus: arrow keys, type-to-filter, and real mouse clicks.

Split in two so the interesting part is testable without a terminal:

* :class:`MenuState` -- a pure state machine. It owns the items, filter text,
  cursor, scroll offset and selection, and it renders itself to a list of strings
  **plus a hit map** (screen row -> item index / button id). Feeding it events and
  checking the rendered lines is a normal unit test.
* :class:`Menu` -- the driver. It owns the terminal: raw mode, mouse reporting,
  flicker-free redraws (cursor-up + clear, no alternate screen so your scrollback
  survives), signal safety, and the non-TTY fallback.

Clicking works because modern terminals report mouse events when we enable SGR
tracking; :meth:`MenuState.hit_test` turns a click coordinate into an action.
Keyboard navigation always works, including when mouse reporting is unavailable
(Windows console, ``TERM=dumb``, ``NEXUS_NO_MOUSE=1``) or when stdin is a pipe --
in that last case a numbered text menu is used so scripts keep working.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .i18n import tr
from .input import KeyEvent, MouseEvent, RawTerminal
from .theme import Style, pad, strip_ansi, truncate, visible_width

BUTTON_OK = "ok"
BUTTON_CANCEL = "cancel"
BUTTON_ALL = "all"
BUTTON_NONE = "none"


@dataclass
class MenuItem:
    label: str
    value: Any = None
    hint: str = ""
    group: str = ""
    enabled: bool = True
    #: filled in by MenuState for the first nine visible items
    shortcut: str = ""

    def __post_init__(self) -> None:
        if self.value is None:
            self.value = self.label


@dataclass
class MenuState:
    """Pure menu state + rendering. No IO."""

    title: str = ""
    items: List[MenuItem] = field(default_factory=list)
    multi: bool = False
    allow_filter: bool = True
    prompt: str = ""
    filter_text: str = ""
    cursor: int = 0
    scroll: int = 0
    selected: List[int] = field(default_factory=list)   # indices into self.items
    max_rows: int = 12
    #: set when the user explicitly emptied the selection, so Enter must not
    #: "helpfully" re-pick the highlighted item right afterwards
    explicit_clear: bool = False

    def __post_init__(self) -> None:
        self.items = [i if isinstance(i, MenuItem) else MenuItem(str(i)) for i in self.items]
        if self.cursor >= len(self.items):
            self.cursor = max(0, len(self.items) - 1)

    # -- filtering --------------------------------------------------------
    def matches(self) -> List[int]:
        """Indices of items matching the filter, in display order."""
        needle = self.filter_text.strip().lower()
        if not needle:
            return list(range(len(self.items)))
        tokens = needle.split()
        out = []
        for i, item in enumerate(self.items):
            haystack = f"{item.label} {item.hint} {item.group}".lower()
            if all(t in haystack for t in tokens):
                out.append(i)
        return out

    def visible(self) -> List[int]:
        return self.matches()

    # -- cursor -----------------------------------------------------------
    def move(self, delta: int) -> None:
        matches = self.matches()
        if not matches:
            return
        try:
            pos = matches.index(self.cursor)
        except ValueError:
            pos = 0 if delta > 0 else len(matches) - 1
        pos = max(0, min(len(matches) - 1, pos + delta))
        self.cursor = matches[pos]
        self._ensure_visible()

    def goto(self, index: int) -> None:
        if 0 <= index < len(self.items):
            self.cursor = index
            self._ensure_visible()

    def home(self) -> None:
        matches = self.matches()
        if matches:
            self.goto(matches[0])

    def end(self) -> None:
        matches = self.matches()
        if matches:
            self.goto(matches[-1])

    def page(self, delta: int) -> None:
        self.move(delta * max(1, self.max_rows - 1))

    def _ensure_visible(self) -> None:
        matches = self.matches()
        if not matches:
            return
        try:
            pos = matches.index(self.cursor)
        except ValueError:
            return
        if pos < self.scroll:
            self.scroll = pos
        elif pos >= self.scroll + self.max_rows:
            self.scroll = pos - self.max_rows + 1
        self.scroll = max(0, min(self.scroll, max(0, len(matches) - self.max_rows)))

    def wheel(self, delta: int) -> None:
        matches = self.matches()
        self.scroll = max(0, min(self.scroll + delta, max(0, len(matches) - self.max_rows)))

    # -- selection --------------------------------------------------------
    def toggle(self, index: Optional[int] = None) -> None:
        index = self.cursor if index is None else index
        if not (0 <= index < len(self.items)) or not self.items[index].enabled:
            return
        self.explicit_clear = False
        if index in self.selected:
            self.selected.remove(index)
        else:
            self.selected.append(index)

    def select_all(self) -> None:
        self.explicit_clear = False
        self.selected = [i for i in self.matches() if self.items[i].enabled]

    def select_none(self) -> None:
        self.selected = []
        self.explicit_clear = True

    def result(self) -> Any:
        if self.multi:
            return [self.items[i].value for i in sorted(self.selected)]
        if not self.items or not self.matches():
            # With an empty filter result there is nothing to choose; returning the
            # cursor's item would silently pick something unrelated.
            return None
        item = self.items[self.cursor] if 0 <= self.cursor < len(self.items) else None
        return item.value if item and item.enabled else None

    # -- input ------------------------------------------------------------
    def type_char(self, char: str) -> None:
        if not self.allow_filter or not char or not char.isprintable():
            return
        self.filter_text += char
        self.scroll = 0
        matches = self.matches()
        if matches and self.cursor not in matches:
            self.cursor = matches[0]
        self._ensure_visible()

    def backspace(self) -> None:
        if self.filter_text:
            self.filter_text = self.filter_text[:-1]
            self.scroll = 0
            matches = self.matches()
            if matches and self.cursor not in matches:
                self.cursor = matches[0]
            self._ensure_visible()

    def clear_filter(self) -> None:
        self.filter_text = ""
        self.scroll = 0

    def by_shortcut(self, char: str) -> Optional[int]:
        """Map a digit key to an item. Computed from the live filter, so it
        works before the first render (and after every keystroke)."""
        if not char.isdigit() or char == "0":
            return None
        matches = self.matches()
        n = int(char)
        if 1 <= n <= min(9, len(matches)):
            return matches[n - 1]
        return None

    def assign_shortcuts(self) -> None:
        matches = self.matches()
        for item in self.items:
            item.shortcut = ""
        for n, index in enumerate(matches[:9], start=1):
            self.items[index].shortcut = str(n)

    # -- geometry / hit testing ------------------------------------------
    def layout(self, width: int) -> List[Tuple[str, Any, str]]:
        """Build the menu as ``(kind, payload, rendered_line)`` triples.

        ``rows()`` and ``hit_map()`` are both derived from this, so the click
        coordinates can never drift away from what was drawn.
        """
        self.assign_shortcuts()
        matches = self.matches()
        inner, total = self._geometry(width)
        out: List[Tuple[str, Any, str]] = [("frame", None, self._line("top", inner, total))]
        header = self.title or "Menu"
        if self.filter_text:
            header += f"  ·  filter: {self.filter_text}"
        out.append(("header", None, self._body(header, inner, total, "header")))
        if self.prompt:
            out.append(("prompt", None, self._body(self.prompt, inner, total, "prompt")))
        if not matches:
            out.append(("empty", None, self._body(tr("menu.no_match"), inner, total, "empty")))
        else:
            above = self.scroll
            below = max(0, len(matches) - self.scroll - self.max_rows)
            if above:
                out.append(("scroll", None,
                            self._body(tr("menu.more_above", n=above), inner, total, "scroll")))
            for index in matches[self.scroll : self.scroll + self.max_rows]:
                out.append(("item", index, self._row(index, inner, total)))
            if below:
                out.append(("scroll", None,
                            self._body(tr("menu.more_below", n=below), inner, total, "scroll")))
        out.append(("buttons", None, self._buttons(inner, total)))
        out.append(("frame", None, self._line("bottom", inner, total)))
        return out

    def rows(self, width: int) -> List[str]:
        """Rendered lines of the menu."""
        return [line for _kind, _payload, line in self.layout(width)]

    def hit_map(self, width: int) -> Dict[int, Tuple[str, Any]]:
        """1-based screen row -> (kind, payload), derived from :meth:`layout`."""
        return {n: (kind, payload)
                for n, (kind, payload, _line) in enumerate(self.layout(width), start=1)}

    def button_columns(self, width: int) -> List[Tuple[str, int, int]]:
        """``(button_id, start_col, end_col)`` on the button row, 1-based inclusive.

        Derived from the same geometry as :meth:`layout`, so the clickable
        regions always match the drawn captions.
        """
        _inner, total = self._geometry(width)
        cols: List[Tuple[str, int, int]] = []
        col = 3  # left border + one space
        for bid, text in self._button_labels(total):
            end = col + visible_width(text) - 1
            cols.append((bid, col, end))
            col = end + 3  # two spaces between buttons
        return cols

    def hit_test(self, row: int, col: int, width: int) -> Optional[Tuple[str, Any]]:
        mapping = self.hit_map(width)
        if row not in mapping:
            return None
        kind, payload = mapping[row]
        if kind != "buttons":
            return (kind, payload)
        for bid, start, end in self.button_columns(width):
            if start <= col <= end:
                return ("button", bid)
        return None

    # -- drawing helpers --------------------------------------------------
    _style: Any = None

    def _geometry(self, width: int) -> Tuple[int, int]:
        """(inner text width, total frame width). Every row uses these so the
        box can never come out ragged -- including the highlighted row."""
        total = max(12, min(width, 120))
        inner = total - 4
        return inner, total

    def _line(self, which: str, inner: int, total: int) -> str:
        if which == "top":
            head = truncate(f" {self.title or 'Menu'} ", max(4, total - 4))
            fill = max(0, total - 3 - visible_width(head))
            return self._paint("border", "┌─" + head + "─" * fill + "┐")
        return self._paint("border", "└" + "─" * (total - 2) + "┘")

    def _body(self, text: str, inner: int, total: int, kind: str) -> str:
        role = {"header": "bold", "prompt": "dim", "empty": "warning",
                "scroll": "dim"}.get(kind, "text")
        text = truncate(text, inner)
        body = self._paint(role, text) + " " * max(0, inner - visible_width(text))
        return self._paint("border", "│ ") + body + self._paint("border", " │")

    def _row(self, index: int, inner: int, total: int) -> str:
        item = self.items[index]
        active = index == self.cursor
        marker = ("[x] " if index in self.selected else "[ ] ") if self.multi else ""
        number = f"{item.shortcut}. " if item.shortcut else "   "
        label = item.label if item.enabled else item.label
        text = ("❯ " if active else "  ") + marker + number + label
        if item.hint:
            used = visible_width(text)
            budget = inner - used - 2
            if budget > 6:
                text += " " * max(1, inner - used - budget) + truncate(item.hint, budget)
        text = truncate(text, inner)
        padded = text + " " * max(0, inner - visible_width(text))
        if not item.enabled:
            padded = self._paint("dim", padded)
        elif active:
            padded = self._invert(padded)
        return self._paint("border", "│ ") + padded + self._paint("border", " │")

    def _button_labels(self, total: int) -> List[Tuple[str, str]]:
        """Button captions, shortened until the row fits the frame.

        When type-to-filter is on, letters belong to the filter, so the
        select-all/clear buttons are labelled click-only instead of advertising a
        keyboard shortcut that would not work.
        """
        if self.multi:
            if self.allow_filter:
                sets = [
                    [("ok", tr("button.ok")), ("all", tr("button.all_click")),
                     ("none", tr("button.none_click")), ("cancel", tr("button.cancel"))],
                    [("ok", tr("button.ok_short")), ("all", tr("button.all_short")),
                     ("none", tr("button.none_short")), ("cancel", tr("button.cancel_short"))],
                    [("ok", tr("button.ok_tiny")), ("all", tr("button.all_tiny")),
                     ("none", tr("button.none_tiny")), ("cancel", tr("button.cancel_tiny"))],
                ]
            else:
                sets = [
                    [("ok", tr("button.ok")), ("all", tr("button.all_key")),
                     ("none", tr("button.none_key")), ("cancel", tr("button.cancel"))],
                    [("ok", tr("button.ok_short")), ("all", tr("button.all_key")),
                     ("none", tr("button.none_key")), ("cancel", tr("button.cancel_short"))],
                    [("ok", tr("button.ok_tiny")), ("all", tr("button.all_tiny")),
                     ("none", tr("button.none_tiny")), ("cancel", tr("button.cancel_tiny"))],
                ]
        else:
            sets = [
                [("ok", tr("button.ok")), ("cancel", tr("button.cancel"))],
                [("ok", tr("button.ok_short")), ("cancel", tr("button.cancel_short"))],
                [("ok", tr("button.ok_tiny")), ("cancel", tr("button.cancel_tiny"))],
            ]

        def fits(labels: List[Tuple[str, str]]) -> bool:
            needed = sum(visible_width(t) for _, t in labels) + 2 * (len(labels) - 1) + 4
            return needed <= total

        for labels in sets:
            if fits(labels):
                return labels
        return sets[-1]

    def _buttons(self, inner: int, total: int) -> str:
        parts = [self._paint("accent" if bid == "ok" else "dim", text)
                 for bid, text in self._button_labels(total)]
        body = "  ".join(parts)
        body = truncate(body, inner)
        return (self._paint("border", "│ ") + body + " " * max(0, inner - visible_width(body))
                + self._paint("border", " │"))

    def _paint(self, role: str, text: str) -> str:
        if self._style is None:
            return text
        if role == "bold":
            return self._style.bold(text)
        return self._style.paint(role, text)

    def _invert(self, text: str) -> str:
        if self._style is None or not self._style.enabled:
            return text
        return self._style.color.inverse(text)

    # -- event handling ---------------------------------------------------
    def step(self, event: Any) -> Optional[str]:
        """Apply one keyboard event. Returns 'select', 'cancel', 'redraw' or None."""
        if not isinstance(event, KeyEvent):
            return None
        key = event.key
        if key == "up":
            self.move(-1)
        elif key == "down":
            self.move(1)
        elif key == "pageup":
            self.page(-1)
        elif key == "pagedown":
            self.page(1)
        elif key == "home":
            self.home()
        elif key == "end":
            self.end()
        elif key in ("enter", "ctrl_space"):
            return self._confirm_selection()
        elif key in ("esc", "ctrl_c"):
            return "cancel"
        elif key == "tab":
            if self.multi:
                self.toggle()
            self.move(1)
        elif key == "shift_tab":
            if self.multi:
                self.move(-1)
                self.toggle()
            else:
                self.move(-1)
        elif key == "backspace":
            self.backspace()
        elif key == "ctrl_u":
            self.clear_filter()
        elif key == "ctrl_a" and self.multi:
            self.select_all()
        elif key == "ctrl_l":
            return "redraw"
        elif key == "char":
            return self._char(event.char)
        return None

    def _confirm_selection(self) -> Optional[str]:
        if self.multi:
            # Enter with nothing marked means "the highlighted one" -- unless the
            # user has just deliberately cleared the selection, in which case an
            # empty result is what they asked for.
            if not self.selected and not self.explicit_clear and 0 <= self.cursor < len(self.items):
                if self.items[self.cursor].enabled:
                    self.selected.append(self.cursor)
            return "select"
        if not self.matches():
            # Enter on an empty result must not select an unrelated item
            return None
        item = self.items[self.cursor] if 0 <= self.cursor < len(self.items) else None
        if item is None or not item.enabled:
            return None
        return "select"

    def _char(self, char: str) -> Optional[str]:
        if not char:
            return None
        # With filtering off, a/n drive the select-all/clear buttons and digits
        # jump straight to an item. With filtering on, everything is typed.
        if not self.allow_filter:
            if self.multi and char == "a":
                self.select_all()
                return None
            if self.multi and char == "n":
                self.select_none()
                return None
            # vim-style motion, but only where letters are not needed for filtering
            if char == "j":
                self.move(1)
                return None
            if char == "k":
                self.move(-1)
                return None
        if char.isdigit() and not self.filter_text:
            index = self.by_shortcut(char)
            if index is not None:
                self.cursor = index
                if self.multi:
                    self.toggle(index)
                    return None
                return "select"
        if self.multi and char == " " and not self.allow_filter:
            self.toggle()
            self.move(1)
            return None
        self.type_char(char)
        return None

    def _step_mouse(self, event: MouseEvent, width: int = 100) -> Optional[str]:
        if event.kind == "wheel_up":
            self.wheel(-3)
        elif event.kind == "wheel_down":
            self.wheel(3)
        return None

    def mouse_click(self, row: int, col: int, width: int) -> Optional[str]:
        """Resolve a click into a state change / action."""
        hit = self.hit_test(row, col, width)
        if not hit:
            return None
        kind, payload = hit
        if kind == "item":
            if self.multi:
                if self.cursor == payload:
                    self.toggle(payload)
                else:
                    self.goto(payload)
            else:
                if self.cursor == payload:
                    return "select"       # second click on the highlighted row
                self.goto(payload)
            return None
        if kind == "button":
            if payload == BUTTON_OK:
                if self.multi and not self.selected and self.items:
                    self.toggle()
                return "select"
            if payload == BUTTON_CANCEL:
                return "cancel"
            if payload == BUTTON_ALL and self.multi:
                self.select_all()
            if payload == BUTTON_NONE and self.multi:
                self.select_none()
        return None


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
class Menu:
    """Interactive menu with keyboard and mouse support."""

    def __init__(self, title: str, items: Sequence[Any], *, style: Optional[Style] = None,
                 multi: bool = False, allow_filter: bool = True, prompt: str = "",
                 stream=None, max_rows: int = 12, fallback: bool = True) -> None:
        self.style = style or Style.create()
        self.stream = stream if stream is not None else sys.stdout
        self.multi = multi
        self.fallback = fallback
        menu_items = [i if isinstance(i, MenuItem) else MenuItem(str(i)) for i in items]
        self.state = MenuState(title=title, items=menu_items, multi=multi,
                               allow_filter=allow_filter, prompt=prompt, max_rows=max_rows)
        self.state._style = self.style
        self._width = min(self.style.width or 100, 100)
        self._origin_row = 1     # screen row of the first menu line
        self._term: Any = None

    # -- public -----------------------------------------------------------
    def show(self) -> Any:
        """Display the menu and return the selection (None when cancelled)."""
        if not self.state.items:
            self._print(tr("menu.no_options"))
            return [] if self.multi else None
        if not self._is_tty():
            return self._fallback()
        return self._interactive()

    # -- internals --------------------------------------------------------
    def _is_tty(self) -> bool:
        try:
            return bool(sys.stdin.isatty() and getattr(self.stream, "isatty", lambda: False)())
        except (OSError, ValueError, AttributeError):
            return False

    def _print(self, text: str) -> None:
        try:
            self.stream.write(text + "\n")
            self.stream.flush()
        except (ValueError, OSError):
            pass

    def _fallback(self) -> Any:
        """Numbered text menu for non-interactive use (pipes, scripts, CI)."""
        self._print(self.style.bold(self.state.title) if self.style.enabled else self.state.title)
        if self.state.prompt:
            self._print("  " + self.state.prompt)
        for n, item in enumerate(self.state.items, 1):
            mark = "" if item.enabled else "  (disabled)"
            hint = f"  -- {item.hint}" if item.hint else ""
            self._print(f"  {n:>2}. {item.label}{hint}{mark}")
        self._print(tr("menu.fallback_numbers" if self.multi else "menu.fallback_number"))
        try:
            raw = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            self._print("")
            return [] if self.multi else None
        if not raw:
            return [] if self.multi else None
        picks = []
        for token in raw.replace(",", " ").split():
            if token.isdigit() and 1 <= int(token) <= len(self.state.items):
                picks.append(int(token) - 1)
        if self.multi:
            return [self.state.items[i].value for i in dict.fromkeys(picks)]
        return self.state.items[picks[0]].value if picks else None

    def _interactive(self) -> Any:
        term = RawTerminal(stream=self.stream)
        lines_printed = 0
        try:
            term.enable()
            if not term.active:
                return self._fallback()
            self._term = term
            self._origin_row = self._place(term)
            self._draw(clear=0)
            lines_printed = len(self.state.rows(self._width))
            while True:
                events = term.read_events(timeout=0.05)
                if not events:
                    if term.pending_esc():
                        events = term.flush_esc() or [KeyEvent("esc")]
                    else:
                        continue
                for event in events:
                    action = self._handle(event, term)
                    if action == "select":
                        return self.state.result()
                    if action == "cancel":
                        return [] if self.multi else None
                self._draw(clear=lines_printed)
                lines_printed = len(self.state.rows(self._width))
        except KeyboardInterrupt:
            return [] if self.multi else None
        finally:
            self._draw(clear=lines_printed)
            term.disable()

    def _place(self, term: RawTerminal) -> int:
        """Work out which screen row the menu will start on.

        The terminal reports the cursor position (DSR), and if the menu would run
        off the bottom we scroll it into view first -- otherwise the terminal
        scrolls for us and every mouse coordinate we compute would be wrong.
        """
        row, _col = term.query_cursor(self.stream)
        try:
            _cols, screen_rows = os.get_terminal_size(self.stream.fileno())
        except (OSError, ValueError, AttributeError):
            screen_rows = 24
        needed = len(self.state.rows(self._width)) + 1
        if row + needed - 1 > screen_rows and screen_rows > needed:
            extra = row + needed - 1 - screen_rows
            try:
                self.stream.write("\n" * extra)
                self.stream.flush()
            except (ValueError, OSError):
                pass
            row, _col = term.query_cursor(self.stream)
        return max(1, row)

    def _handle(self, event: Any, term: RawTerminal) -> Optional[str]:
        if isinstance(event, MouseEvent):
            if event.kind in ("wheel_up", "wheel_down"):
                self.state.wheel(-3 if event.kind == "wheel_up" else 3)
                return None
            if event.kind != "press":
                return None
            # event.y is an absolute screen row; menus are 1-based from the origin
            row = event.y - self._origin_row + 1
            return self.state.mouse_click(row, event.x, self._width)
        return self.state.step(event)

    def _draw(self, clear: int = 0) -> None:
        lines = self.state.rows(self._width)
        out: List[str] = []
        if clear:
            out.append(f"\x1b[{clear}A")     # move up over the previous frame
            out.append("\x1b[J")             # clear from here to end of screen
        try:
            cols, rows = os.get_terminal_size(self.stream.fileno())
        except (OSError, ValueError, AttributeError):
            rows = 24
        # keep the whole menu on screen: if it would overflow, scroll first
        if len(lines) + 1 > rows:
            self.state.max_rows = max(3, rows - 6)
            self.state._ensure_visible()
            lines = self.state.rows(self._width)
        out.append("\n".join(lines))
        out.append("\n")
        try:
            self.stream.write("".join(out))
            self.stream.flush()
        except (ValueError, OSError):
            pass


def select(title: str, items: Sequence[Any], *, style: Optional[Style] = None,
           prompt: str = "", allow_filter: bool = True, multi: bool = False,
           stream=None, fallback: bool = True) -> Any:
    """Convenience wrapper: show a menu and return the chosen value(s)."""
    menu = Menu(title, items, style=style, multi=multi, allow_filter=allow_filter,
                prompt=prompt, stream=stream)
    menu.fallback = fallback
    return menu.show()


def confirm(title: str, message: str = "", *, style: Optional[Style] = None,
            default: bool = False, yes_label: str = "Ya", no_label: str = "Tidak",
            stream=None) -> bool:
    """A two-button clickable yes/no dialog."""
    items = [MenuItem(yes_label, True, hint="lanjutkan"), MenuItem(no_label, False, hint="batalkan")]
    menu = Menu(title, items, style=style, allow_filter=False, prompt=message,
                stream=stream, max_rows=2)
    menu.state.cursor = 0 if default else 1
    result = menu.show()
    if result is None:
        return default
    return bool(result)


__all__ = ["MenuItem", "MenuState", "Menu", "select", "confirm", "BUTTON_OK", "BUTTON_CANCEL",
           "BUTTON_ALL", "BUTTON_NONE"]
