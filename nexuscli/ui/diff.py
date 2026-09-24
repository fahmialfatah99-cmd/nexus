"""Diff rendering + summary statistics."""

from __future__ import annotations

import difflib
from typing import Dict, List, Tuple

from .theme import Style, truncate, visible_width, wrap_text


def unified(old: str, new: str, path: str = "file", context: int = 3) -> str:
    return "\n".join(difflib.unified_diff(old.split("\n"), new.split("\n"),
                                          fromfile=f"a/{path}", tofile=f"b/{path}",
                                          lineterm="", n=context))


def render_diff(diff_text: str, style: Style, *, width: int = 0, max_lines: int = 400) -> str:
    """Colourise a unified diff and wrap it to the terminal width."""
    width = width or style.width
    if not diff_text or not diff_text.strip():
        return style.dim("(no textual changes)")
    out: List[str] = []
    for i, line in enumerate(diff_text.split("\n")):
        if i >= max_lines:
            out.append(style.dim(f"… ({len(diff_text.split(chr(10))) - max_lines} more diff lines)"))
            break
        if line.startswith("+++") or line.startswith("---"):
            rendered = style.bold(style.paint("bold", line))
        elif line.startswith("@@"):
            rendered = style.paint("diff_hunk", line)
        elif line.startswith("+"):
            rendered = style.paint("diff_add", line)
        elif line.startswith("-"):
            rendered = style.paint("diff_del", line)
        elif line.startswith("\\"):
            rendered = style.dim(line)
        else:
            rendered = style.dim(line)
        if visible_width(rendered) > width:
            out.extend(wrap_text(rendered, width, subsequent_indent="  "))
        else:
            out.append(rendered)
    return "\n".join(out)


def summarise(diff_text: str) -> Dict[str, object]:
    """Return +/- counts and the list of files touched."""
    added = removed = 0
    files: List[str] = []
    for line in (diff_text or "").split("\n"):
        if line.startswith("+++ ") and not line.startswith("+++ /dev/null"):
            files.append(line[4:].split("\t")[0].lstrip("b/"))
        elif line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed += 1
    return {"added": added, "removed": removed, "files": sorted(set(files)),
            "net": added - removed}


def format_summary(diff_text: str, style: Style) -> str:
    stats = summarise(diff_text)
    parts = []
    if stats["files"]:
        parts.append(", ".join(str(f) for f in stats["files"][:4]))
        if len(stats["files"]) > 4:
            parts[-1] += f" +{len(stats['files']) - 4} more"
    parts.append(style.paint("diff_add", f"+{stats['added']}"))
    parts.append(style.paint("diff_del", f"-{stats['removed']}"))
    return " ".join(parts)


def word_diff(old: str, new: str, style: Style, width: int = 0) -> str:
    """Intra-line diff, useful for one-line edits where +/- blocks hide the change."""
    width = width or style.width
    tokens = list(difflib.ndiff(old.split(" "), new.split(" ")))
    out: List[str] = []
    for token in tokens:
        tag, _, word = token.partition(" ")
        if tag == "+":
            out.append(style.paint("diff_add", word))
        elif tag == "-":
            out.append(style.paint("diff_del", style.color.underline(word)))
        elif word:
            out.append(style.dim(word))
    return truncate(" ".join(out), width * 6)


__all__ = ["unified", "render_diff", "summarise", "format_summary", "word_diff"]
