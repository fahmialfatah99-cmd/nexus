"""Terminal markdown renderer with a streaming front-end.

Two layers:

* :func:`render_markdown` -- render a complete markdown document (used for
  ``/help``, docs, final reports).
* :class:`StreamingMarkdown` -- feed deltas as they arrive from the model and get
  back ready-to-print text. It buffers **one line** (plus whole tables and code
  fences) so line-oriented constructs render correctly while still feeling
  instant. This is why NEXUS output does not "jump" or redraw like a
  re-render-the-whole-block implementation.

Inline formatting is applied in a single regex pass, which avoids the classic
bug of nested substitutions fighting each other (e.g. bold markers inside a code
span getting re-processed).
"""

from __future__ import annotations

import re
from typing import Any, List, Optional, Sequence, Tuple

from .theme import RESET, Style, pad, strip_ansi, truncate, visible_width, wrap_text

CODE_SPAN_RE = re.compile(r"`+[^`\n]+`+")
INLINE_RE = re.compile(
    r"(?P<strong>\*\*[^*\n]+\*\*|__[^_\n]+__)"
    r"|(?P<em>\*[^*\n]+\*|(?<![A-Za-z0-9_])_[^_\n]+_(?![A-Za-z0-9_]))"
    r"|(?P<strike>~~[^~\n]+~~)"
    r"|(?P<link>\[[^\]\n]*\]\([^)\n]+\))"
    r"|(?P<url>(?<![(\w])https?://[^\s<)\]]+)"
)
_PLACEHOLDER_RE = re.compile(r"\x00(\d+)\x00")
#: a link label opened but not yet closed: "[tex"
_UNCLOSED_LINK_RE = re.compile(r"(?:^|[^!])\[[^\]]*$")
#: a link destination opened but not yet closed: "[text](http"
_OPEN_LINK_DEST_RE = re.compile(r"\]\([^)]*$")


CODE_FENCE_RE = re.compile(r"^\s*(```+|~~~+)\s*([\w+#.\-]*)\s*$")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
HR_RE = re.compile(r"^\s*([-*_])(\s*\1){2,}\s*$")
LIST_RE = re.compile(r"^(\s*)([-*+]|\d{1,9}[.)])\s+(.*)$")
QUOTE_RE = re.compile(r"^(\s*)>\s?(.*)$")
TABLE_SEP_RE = re.compile(r"^\s*\|?[\s:|-]+\|[\s:|-]*$")

KEYWORDS = {
    "python": r"\b(def|class|return|if|elif|else|for|while|import|from|as|with|try|except|finally|raise|"
              r"lambda|yield|pass|break|continue|and|or|not|in|is|None|True|False|self|async|await|"
              r"global|nonlocal|assert|del|match|case)\b",
    "js": r"\b(function|const|let|var|return|if|else|for|while|import|export|from|class|extends|new|"
          r"this|async|await|try|catch|finally|throw|typeof|instanceof|null|undefined|true|false|of|in)\b",
    "generic": r"\b(function|def|class|return|if|else|for|while|import|from|export|const|let|var|new|"
               r"try|catch|finally|throw|raise|async|await|public|private|static|void|int|str|bool|"
               r"true|false|null|None|True|False|self|this)\b",
}
TOKEN_RE_CACHE: dict = {}


def _token_re(lang: str) -> "re.Pattern[str]":
    key = lang or "generic"
    cached = TOKEN_RE_CACHE.get(key)
    if cached is not None:
        return cached
    family = "python" if key in ("python", "py", "python3") else "js" if key in (
        "js", "jsx", "ts", "tsx", "javascript", "typescript", "json") else "generic"
    kw = KEYWORDS[family]
    comment = r"#[^\n]*" if family == "python" else r"//[^\n]*|/\*.*?\*/"
    if key in ("sh", "bash", "shell", "zsh", "yaml", "yml", "toml", "ini", "ruby", "perl"):
        comment = r"#[^\n]*"
    pattern = re.compile(
        rf"(?P<string>\"\"\"[\s\S]*?\"\"\"|'''[\s\S]*?'''|\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'|`(?:\\.|[^`\\])*`)"
        rf"|(?P<comment>{comment})"
        rf"|(?P<kw>{kw})"
        rf"|(?P<num>\b\d[\d_]*\.?[\d_]*(?:e[+-]?\d+)?\b)"
        rf"|(?P<fn>\b[A-Za-z_][A-Za-z0-9_]*(?=\s*\())"
        rf"|(?P<dec>@[A-Za-z_][\w.]*)"
    )
    TOKEN_RE_CACHE[key] = pattern
    return pattern


def highlight(code_line: str, lang: str, style: Style) -> str:
    """Very small deterministic highlighter: strings, comments, keywords, numbers."""
    if not style.enabled or not code_line.strip():
        return code_line
    rx = _token_re(lang)

    def repl(m: "re.Match[str]") -> str:
        text = m.group(0)
        if m.group("string"):
            return style.paint("string", text)
        if m.group("comment"):
            return style.paint("comment", text)
        if m.group("kw"):
            return style.paint("keyword", style.color.bold(text))
        if m.group("num"):
            return style.paint("number", text)
        if m.group("dec"):
            return style.paint("accent2", text)
        if m.group("fn"):
            return style.paint("accent", text)
        return text

    return rx.sub(repl, code_line)


def render_inline(text: str, style: Style) -> str:
    """Inline markdown -> ANSI, in two passes.

    Code spans are extracted first so that nesting works: ``**bold with `code`
    inside**`` must keep the backtick span intact instead of leaking it as text
    (a single-pass regex cannot nest).
    """
    if not text:
        return text
    spans: List[str] = []

    def stash(m: "re.Match[str]") -> str:
        spans.append(m.group(0))
        return f"\x00{len(spans) - 1}\x00"

    protected = CODE_SPAN_RE.sub(stash, text)

    def repl(m: "re.Match[str]") -> str:
        if m.group("strong"):
            return style.bold(m.group("strong")[2:-2])
        if m.group("em"):
            return style.color.italic(style.paint("text", m.group("em")[1:-1]))
        if m.group("strike"):
            return style.dim("\u0336".join(m.group("strike")[2:-2]))
        if m.group("link"):
            full = m.group("link")
            label, _, url = full[1:-1].partition("](")
            url = url.rstrip(")")
            return style.paint("accent", label or url) + style.dim(f" ({url})")
        if m.group("url"):
            return style.paint("accent", m.group("url"))
        return m.group(0)

    rendered = INLINE_RE.sub(repl, protected)

    def restore(m: "re.Match[str]") -> str:
        raw = spans[int(m.group(1))]
        ticks = 1
        while raw.startswith("`" * (ticks + 1)) and raw.endswith("`" * (ticks + 1)):
            ticks += 1
        inner = raw[ticks:-ticks] if len(raw) > 2 * ticks else raw.strip("`")
        return style.code(inner)

    return _PLACEHOLDER_RE.sub(restore, rendered)


def _has_open_inline(text: str) -> bool:
    """True when *text* must not be emitted yet.

    Three cases: an unclosed span (``**bo``), a delimiter that could be the
    *start* of a span that has not completed yet (a lone ``*`` may still become
    ``*em*`` or ``**bold**`` -- chunk boundaries land in the middle of spans all
    the time), or a trailing delimiter character. Holding in all three cases
    keeps the invariant that an emitted chunk never ends inside (or leaks the
    raw marker of) a markdown construct.
    """
    if not text:
        return False
    if text.count("`") % 2 == 1:
        return True
    if text.count("**") % 2 == 1:
        return True
    if text.count("*") % 2 == 1:
        return True
    if text.count("~~") % 2 == 1:
        return True
    if text.count("_") % 2 == 1:
        return True
    if _UNCLOSED_LINK_RE.search(text):
        return True
    if _OPEN_LINK_DEST_RE.search(text):
        return True
    return text[-1] in "*`~_[]"


def _split_table_row(line: str) -> List[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    cells, current, escaped = [], "", False
    for ch in stripped:
        if escaped:
            current += ch
            escaped = False
        elif ch == "\\":
            escaped = True
        elif ch == "|":
            cells.append(current.strip())
            current = ""
        else:
            current += ch
    cells.append(current.strip())
    return cells


def render_table(rows: List[List[str]], style: Style, width: int, aligns: Optional[List[str]] = None,
                 inline: bool = True) -> List[str]:
    """Render a box table.

    ``inline`` controls markdown interpretation of cell text: parsed markdown
    tables want it (`` `mode` `` renders as code), but *data* tables must not --
    a tool description containing ``**/*.py`` would otherwise be mangled into
    bold markers.
    """
    if not rows:
        return []
    cols = max(len(r) for r in rows)
    rows = [r + [""] * (cols - len(r)) for r in rows]
    aligns = (aligns or ["left"] * cols) + ["left"] * cols
    def fmt(cell: Any) -> str:
        return render_inline(str(cell), style) if inline else str(cell)

    widths = [max(visible_width(fmt(rows[r][c])) for r in range(len(rows))) for c in range(cols)]
    # Shrink proportionally when the table is wider than the terminal, then
    # enforce the invariant exactly: a proportional scale plus a per-column
    # minimum can still overflow by a cell or two.
    chrome = 3 * cols + 1
    budget = max(cols, width - chrome)
    total = sum(widths) or 1
    if total > budget:
        scale = budget / total
        widths = [max(1, int(w * scale)) for w in widths]
    guard = 0
    while sum(widths) + chrome > width and max(widths) > 1 and guard < 10_000:
        widest = max(range(cols), key=lambda c: widths[c])
        widths[widest] -= 1
        guard += 1

    def line(left: str, mid: str, right: str, fill: str = "─") -> str:
        return style.border(left + mid.join(fill * (w + 2) for w in widths) + right)

    out = [line("┌", "┬", "┐")]
    for i, row in enumerate(rows):
        cells = []
        for c, cell in enumerate(row):
            rendered = fmt(cell)
            if i == 0:
                rendered = style.bold(rendered)
            cells.append(" " + pad(truncate(rendered, widths[c]), widths[c], aligns[c]) + " ")
        out.append(style.border("│") + style.border("│").join(cells) + style.border("│"))
        if i == 0:
            out.append(line("├", "┼", "┤"))
    out.append(line("└", "┴", "┘"))
    return out


def _looks_like_table(line: str) -> bool:
    s = line.strip()
    return s.startswith("|") and s.count("|") >= 2


def render_markdown(text: str, style: Style, *, width: int = 0, code_line_numbers: bool = False) -> str:
    """Render a complete markdown document to terminal text."""
    width = width or style.width
    lines = (text or "").replace("\r\n", "\n").split("\n")
    out: List[str] = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        fence = CODE_FENCE_RE.match(line)
        if fence:
            marker, lang = fence.group(1)[0] * 3, fence.group(2)
            i += 1
            block: List[str] = []
            while i < n:
                if lines[i].strip().startswith(marker):
                    i += 1
                    break
                block.append(lines[i])
                i += 1
            out.extend(_render_code_block(block, lang, style, width, code_line_numbers))
            continue
        if not line.strip():
            out.append("")
            i += 1
            continue
        heading = HEADING_RE.match(line)
        if heading:
            level, content = len(heading.group(1)), heading.group(2).strip()
            rendered = render_inline(content, style)
            if level <= 2:
                out.append(style.paint("accent", style.color.bold(rendered)))
                out.append(style.border("─" * min(width - 1, max(6, visible_width(rendered)))))
            else:
                out.append(style.paint("accent2", style.color.bold(f"{'#' * level} {rendered}")))
            i += 1
            continue
        if HR_RE.match(line):
            out.append(style.border("─" * max(8, min(width - 1, 60))))
            i += 1
            continue
        if _looks_like_table(line):
            block = []
            while i < n and _looks_like_table(lines[i]):
                block.append(lines[i])
                i += 1
            rows = [_split_table_row(b) for b in block]
            rows = [r for r in rows if not all(set(c) <= set("-: ") and c for c in r)]
            aligns: List[str] = []
            for b in block:
                if TABLE_SEP_RE.match(b) and "-" in b:
                    for cell in _split_table_row(b):
                        cell = cell.strip()
                        aligns.append("center" if cell.startswith(":") and cell.endswith(":")
                                      else "right" if cell.endswith(":") else "left")
                    break
            out.extend(render_table(rows, style, width, aligns))
            continue
        quote = QUOTE_RE.match(line)
        if quote:
            while i < n:
                q = QUOTE_RE.match(lines[i])
                if not q:
                    break
                content = render_inline(q.group(2), style)
                for wrapped in wrap_text(content, width - 3, indent=style.border("│ ") + " ",
                                         subsequent_indent=style.border("│ ") + " "):
                    out.append(wrapped)
                i += 1
            continue
        listing = LIST_RE.match(line)
        if listing:
            indent_raw, bullet, content = listing.groups()
            depth = len(indent_raw) // 2
            marker = "•" if bullet in "-*+" else bullet
            prefix = "  " * depth + style.paint("accent", marker) + " "
            rendered = render_inline(content, style)
            out.extend(wrap_text(rendered, width, indent=prefix,
                                 subsequent_indent=" " * (visible_width(prefix))))
            i += 1
            continue
        paragraph = [line]
        i += 1
        while i < n and lines[i].strip() and not any((
                CODE_FENCE_RE.match(lines[i]), HEADING_RE.match(lines[i]), HR_RE.match(lines[i]),
                LIST_RE.match(lines[i]), QUOTE_RE.match(lines[i]), _looks_like_table(lines[i]))):
            paragraph.append(lines[i])
            i += 1
        joined = " ".join(p.strip() for p in paragraph)
        out.extend(wrap_text(render_inline(joined, style), width))
    while len(out) > 1 and not out[-1].strip():
        out.pop()
    return "\n".join(out)


def _code_geometry(width: int) -> int:
    """Inner text width for a framed code block (total frame width == width).

    The floor is low enough that even a 12-column terminal does not overflow;
    a frame wider than the terminal would wrap and destroy the box.
    """
    return max(6, width - 4)


def _code_top(inner: int, lang: str, style: Style) -> str:
    head = f" {lang} " if lang else ""
    fill = max(0, inner + 1 - visible_width(head))
    return style.border("┌─" + head + "─" * fill + "┐")


def _code_bottom(inner: int, style: Style) -> str:
    return style.border("└" + "─" * (inner + 2) + "┘")


def _code_line(inner: int, body: str, style: Style) -> str:
    fill = max(0, inner - visible_width(body))
    return style.border("│ ") + body + " " * fill + style.border(" │")


def _render_code_block(block: List[str], lang: str, style: Style, width: int,
                       line_numbers: bool) -> List[str]:
    inner = _code_geometry(width)
    out: List[str] = [_code_top(inner, lang, style)]
    gutter = len(str(len(block))) if line_numbers else 0
    for idx, raw in enumerate(block, start=1):
        line = raw.replace("\t", "    ")
        prefix = style.dim(pad(str(idx), gutter, "right") + " │ ") if line_numbers else ""
        cont_prefix = (" " * (gutter + 3)) if line_numbers else ""
        pieces = wrap_text(highlight(line, lang, style), inner - visible_width(prefix)) or [""]
        for piece in pieces:
            out.append(_code_line(inner, prefix + piece, style))
            prefix = cont_prefix
    out.append(_code_bottom(inner, style))
    return out


class StreamingMarkdown:
    """Incremental markdown renderer for token streams.

    Contract: ``feed(delta)`` returns text safe to print *right now*; call
    :meth:`flush` at end of stream. Two rules make this both instant and correct:

    * **Prose streams character-by-character.** Holding a paragraph until its
      newline would make the CLI feel frozen, so partial lines are inline-formatted
      and emitted immediately. When the line finally completes, only the remainder
      is emitted -- never a duplicate.
    * **Line-oriented constructs are buffered.** Code fences, tables, headings,
      lists and quotes need the whole line (tables need every row) to render
      correctly, so those are held until the newline arrives.
    """

    #: a partial line beginning with one of these is held back until complete
    _HOLD_PREFIXES = ("#", ">", "|", "-", "*", "+", "```", "~~~", "1.", "2.", "3.", "4.",
                      "5.", "6.", "7.", "8.", "9.")

    def __init__(self, style: Style, *, width: int = 0, plain: bool = False) -> None:
        self.style = style
        self.width = width or style.width
        self.plain = plain
        self._partial = ""          # current line, not yet newline-terminated
        self._emitted = 0           # how many chars of _partial we already printed
        self._in_fence = False
        self._fence_lang = ""
        self._fence_marker = "```"
        self._table_rows: List[List[str]] = []
        self._table_active = False

    # -- api --------------------------------------------------------------
    def feed(self, delta: str) -> str:
        if not delta:
            return ""
        if self.plain:
            return delta
        self._partial += delta
        out: List[str] = []
        while "\n" in self._partial:
            line, self._partial = self._partial.split("\n", 1)
            out.extend(self._handle_line(line))
            self._emitted = 0
        out.append(self._emit_partial())
        return "".join(out)

    def flush(self) -> str:
        out: List[str] = []
        if self._partial and not self._in_fence and not self._table_active:
            held = self._partial[self._emitted:]
            stripped = self._partial.lstrip()
            structural = bool(stripped) and any(p.startswith(stripped) or stripped.startswith(p)
                                               for p in self._HOLD_PREFIXES)
            if held and not structural:
                # End of stream: whatever was held back for an unclosed span is
                # complete now, so it can be wrapped like a normal line.
                if self._emitted:
                    out.append(render_inline(held, self.style))
                else:
                    out.extend(wrap_text(render_inline(self._partial.strip(), self.style), self.width))
                self._emitted = len(self._partial)
            elif held and structural and not self._in_fence:
                # Held as a possible structural line but the stream ended: render it.
                out.extend(self._handle_line(self._partial))
                self._partial = ""
                self._emitted = 0
        if self._table_active:
            leftover = self._partial
            if leftover and leftover.lstrip().startswith("|"):
                self._table_rows.append(_split_table_row(leftover))
                self._partial = ""
                self._emitted = 0
            out.extend(self._flush_table())
        if self._partial:
            remainder = self._partial[self._emitted:]
            closing = self._in_fence and self._partial.strip().startswith(self._fence_marker)
            if closing:
                pass  # a closing fence with no trailing newline: just close the frame
            elif self._in_fence:
                out.extend(piece + "\n" for piece in self._fence_body(self._partial))
            elif remainder:
                out.append(render_inline(remainder, self.style))
                out.append("\n")
            self._partial = ""
            self._emitted = 0
        if self._in_fence:
            out.append(self._fence_bottom())
            self._in_fence = False
        text = "".join(out)
        return text

    # -- partial-line emission -------------------------------------------
    def _emit_partial(self) -> str:
        """Stream what we safely can of the unfinished line.

        Prose is emitted as-is: the terminal soft-wraps it, which keeps this code
        simple and free of column-tracking edge cases. Structural lines (fences,
        tables, headings, lists, quotes) are held until complete so they can be
        wrapped and framed correctly.
        """
        line = self._partial
        if self._in_fence or self._table_active:
            return ""                      # frames/tables need the complete line
        stripped = line.lstrip()
        if stripped and any(p.startswith(stripped) or stripped.startswith(p)
                            for p in self._HOLD_PREFIXES):
            # Either it already is a structural line, or it could still become
            # one ("``" may turn into "```"): wait for the newline either way.
            return ""
        pending = line[self._emitted:]
        if not pending:
            return ""
        if _has_open_inline(pending):
            # Emitting half of `**bo` would print the markers literally; wait for
            # the closing delimiter (flush() emits it at end of stream).
            return ""
        self._emitted = len(line)
        return render_inline(pending, self.style)

    # -- complete lines ---------------------------------------------------
    def _handle_line(self, line: str) -> List[str]:
        style = self.style
        already = self._emitted
        self._emitted = 0
        if already and not self._in_fence and not self._table_active:
            # We streamed this line live; emit only the tail to avoid duplicates.
            tail = line[already:]
            return [(render_inline(tail, style) if tail else "") + "\n"]

        if self._in_fence:
            if line.strip().startswith(self._fence_marker):
                self._in_fence = False
                return [self._fence_bottom() + "\n"]
            return [piece + "\n" for piece in self._fence_body(line)]

        fence = CODE_FENCE_RE.match(line)
        if fence:
            self._in_fence = True
            self._fence_marker = fence.group(1)[0] * 3
            self._fence_lang = fence.group(2) or ""
            return [_code_top(_code_geometry(self.width), self._fence_lang, style) + "\n"]

        if _looks_like_table(line):
            if TABLE_SEP_RE.match(line) and "-" in line and not self._table_rows:
                return []
            self._table_active = True
            self._table_rows.append(_split_table_row(line))
            return []
        if self._table_active:
            out = self._flush_table()
            out.extend(self._handle_line(line))
            return out

        if not line.strip():
            return ["\n"]
        heading = HEADING_RE.match(line)
        if heading:
            level, content = len(heading.group(1)), heading.group(2).strip()
            rendered = render_inline(content, style)
            if level <= 2:
                return [style.paint("accent", style.color.bold(rendered)) + "\n"
                        + style.border("─" * min(self.width - 1, max(6, visible_width(rendered)))) + "\n"]
            return [style.paint("accent2", style.color.bold(rendered)) + "\n"]
        if HR_RE.match(line):
            return [style.border("─" * max(8, min(self.width - 1, 60))) + "\n"]
        quote = QUOTE_RE.match(line)
        if quote:
            content = render_inline(quote.group(2), style)
            return [w + "\n" for w in wrap_text(content, self.width - 3,
                                                indent=style.border("│ ") + " ",
                                                subsequent_indent=style.border("│ ") + " ")]
        listing = LIST_RE.match(line)
        if listing:
            indent_raw, bullet, content = listing.groups()
            depth = len(indent_raw) // 2
            marker = "•" if bullet in "-*+" else bullet
            prefix = "  " * depth + style.paint("accent", marker) + " "
            rendered = render_inline(content, style)
            return [w + "\n" for w in wrap_text(rendered, self.width, indent=prefix,
                                                subsequent_indent=" " * visible_width(prefix))]
        rendered = render_inline(line.strip(), style)
        return [w + "\n" for w in wrap_text(rendered, self.width)]

    # -- frames -----------------------------------------------------------
    def _fence_body(self, line: str) -> List[str]:
        inner = _code_geometry(self.width)
        rendered = highlight(line.replace("\t", "    "), self._fence_lang, self.style)
        return [_code_line(inner, piece, self.style) for piece in (wrap_text(rendered, inner) or [""])]

    def _fence_bottom(self) -> str:
        return _code_bottom(_code_geometry(self.width), self.style)

    def _flush_table(self) -> List[str]:
        rows = self._table_rows
        self._table_rows = []
        self._table_active = False
        if not rows:
            return []
        rows = [r for r in rows if not all(set(c) <= set("-: ") and c for c in r)]
        return [line + "\n" for line in render_table(rows, self.style, self.width)]


__all__ = ["render_markdown", "render_inline", "render_table", "StreamingMarkdown", "highlight"]
