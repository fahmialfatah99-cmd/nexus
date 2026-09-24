"""Filesystem tools: read, write, edit, multi-edit, list, delete.

Design rules that keep these safe:

* **Never corrupt a file.** Text is read losslessly; if decoding required
  replacement characters the tool refuses to write back.
* **Preserve conventions.** The dominant line ending (LF vs CRLF) and the
  absence/presence of a trailing newline are detected and restored.
* **Every mutation is checkpointed** first, so ``/undo`` always works.
* **Edits must be unambiguous.** ``old_text`` has to match exactly once (or the
  caller must pass ``replace_all``); a fuzzy match is reported explicitly.
"""

from __future__ import annotations

import difflib
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ...core.errors import ValidationError
from ...core.ignore import (
    ALWAYS_SKIP_DIRS,
    IgnoreMatcher,
    glob_match,
    human_size,
    is_probably_binary,
    is_within,
    walk_files,
)
from ..base import ConfirmationRequest, Tool, ToolContext, ToolResult, add_line_numbers, clip

MAX_READ_BYTES = 2_000_000
MAX_WRITE_BYTES = 20_000_000


# --------------------------------------------------------------------------- #
# text IO helpers (lossless + convention preserving)
# --------------------------------------------------------------------------- #
def read_text_file(path: Path) -> Tuple[str, str, bool]:
    """Return ``(text_with_lf_endings, dominant_newline_style, lossy)``.

    Content is normalised to LF internally and restored to the file's dominant
    style on write, so round-tripping a CRLF file never rewrites every line.
    ``lossy`` is True when the bytes were not valid UTF-8 -- callers must refuse
    to write such files back.
    """
    raw = path.read_bytes()
    crlf = raw.count(b"\r\n")
    lf = raw.count(b"\n") - crlf
    style = "\r\n" if crlf > lf else "\n"
    try:
        text = raw.decode("utf-8")
        lossy = False
    except UnicodeDecodeError:
        text = raw.decode("utf-8", errors="replace")
        lossy = True
    text = text.replace("\r\n", "\n")
    if crlf == 0 and "\r" in text:  # classic-Mac line endings
        text = text.replace("\r", "\n")
    return text, style, lossy


def write_text_file(path: Path, text: str, style: str = "\n") -> int:
    if style == "\r\n":
        text = text.replace("\r\n", "\n").replace("\n", "\r\n")
    data = text.encode("utf-8")
    if len(data) > MAX_WRITE_BYTES:
        raise ValidationError(f"Refusing to write {human_size(len(data))} (> {human_size(MAX_WRITE_BYTES)}).")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".nexus-tmp")
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    # Preserve permissions of an existing file.
    try:
        if path.exists():
            os.chmod(tmp, os.stat(path).st_mode & 0o7777)
    except OSError:
        pass
    os.replace(tmp, path)
    return len(data)


def unified_diff(old: str, new: str, path: str, max_lines: int = 400) -> str:
    diff = list(difflib.unified_diff(old.split("\n"), new.split("\n"),
                                     fromfile=f"a/{path}", tofile=f"b/{path}", lineterm="", n=3))
    if len(diff) > max_lines:
        diff = diff[:max_lines] + [f"... ({len(diff) - max_lines} more diff lines)"]
    return "\n".join(diff)


def _guard_path(ctx: ToolContext, path: Path) -> Optional[str]:
    """Return an error message when the path is outside the workspace."""
    if is_within(ctx.workspace_root, path):
        return None
    allow = getattr(ctx.permissions, "outside_workspace_allowed", None)
    if callable(allow) and allow(path):
        return None
    return (f"Path '{path}' is outside the workspace ({ctx.workspace_root}). "
            "Use an absolute path inside the project, or re-run with --add-dir.")


def _suggest_name(path: Path, ctx: ToolContext) -> str:
    """Helpful 'did you mean' when a file does not exist."""
    parent = path.parent
    if not parent.is_dir():
        return ""
    try:
        names = [p.name for p in parent.iterdir()][:500]
    except OSError:
        return ""
    close = difflib.get_close_matches(path.name, names, n=3, cutoff=0.5)
    return f" Did you mean: {', '.join(close)}?" if close else ""


# --------------------------------------------------------------------------- #
# read_file
# --------------------------------------------------------------------------- #
class ReadFileTool(Tool):
    name = "read_file"
    description = (
        "Read a text file from the workspace and return its contents with line numbers.\n"
        "Use `offset` (1-based line) and `limit` for large files. Images return a note "
        "instead of bytes; binary files are rejected. Prefer this over `bash cat`."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path, relative to the workspace or absolute."},
            "offset": {"type": "integer", "description": "First line to read (1-based).", "minimum": 1},
            "limit": {"type": "integer", "description": "Maximum number of lines to return.", "minimum": 1},
            "raw": {"type": "boolean", "description": "Return only the file content: no header, no line numbers.", "default": False},
        },
        "required": ["path"],
    }
    category = "files"
    read_only = True

    def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        path = ctx.resolve(args["path"])
        if not path.exists():
            return ToolResult.fail(f"File not found: {args['path']}.{_suggest_name(path, ctx)}")
        if path.is_dir():
            return ToolResult.fail(f"'{args['path']}' is a directory. Use list_dir instead.")
        err = _guard_path(ctx, path)
        if err:
            return ToolResult.fail(err)
        size = path.stat().st_size
        if is_probably_binary(path):
            return ToolResult.fail(
                f"'{args['path']}' looks binary ({human_size(size)}). Cannot display as text."
            )
        if size > MAX_READ_BYTES:
            return ToolResult.fail(
                f"'{args['path']}' is {human_size(size)}, above the {human_size(MAX_READ_BYTES)} read limit. "
                "Use grep to search it, or read_file with offset/limit."
            )
        text, style, lossy = read_text_file(path)
        lines = text.split("\n")
        if lines and lines[-1] == "":
            lines.pop()  # a trailing newline does not create an extra line
        total = len(lines)
        offset = max(1, int(args.get("offset") or 1))
        limit = int(args.get("limit") or 0) or 2000
        window = lines[offset - 1 : offset - 1 + limit]
        body = "\n".join(window)
        truncated = total > offset - 1 + len(window)
        raw_mode = bool(args.get("raw"))
        if raw_mode:
            # `raw` means "just the bytes as text" -- no header, no line numbers --
            # so the output can be piped into another tool verbatim.
            return ToolResult(content=body, data={"path": str(path), "lines": total,
                                                 "shown": len(window), "newline": style,
                                                 "lossy": lossy, "raw": True}).with_touched(str(path))
        content = add_line_numbers(body, start=offset)
        note = []
        if lossy:
            note.append("WARNING: file is not valid UTF-8; content shown with replacement characters (writes are blocked).")
        if truncated:
            note.append(f"Showing lines {offset}-{offset + len(window) - 1} of {total}. "
                        f"Use offset={offset + len(window)} to continue.")
        header = f"{ctx.rel(path)} ({human_size(size)}, {total} lines, {style.replace(chr(10), 'LF').replace(chr(13) + 'LF', 'CRLF')})"
        return ToolResult(content=(header + ("\n" + "\n".join(note) if note else "") + "\n\n" + content).strip(),
                          data={"path": str(path), "lines": total, "shown": len(window), "newline": style,
                                "lossy": lossy}).with_touched(str(path))


# --------------------------------------------------------------------------- #
# write_file
# --------------------------------------------------------------------------- #
class WriteFileTool(Tool):
    name = "write_file"
    description = (
        "Create a file or completely replace its contents. Parent directories are created.\n"
        "For targeted changes to an existing file prefer edit_file (it is safer and cheaper)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Destination path."},
            "content": {"type": "string", "description": "Full file content."},
            "append": {"type": "boolean", "description": "Append instead of overwriting.", "default": False},
        },
        "required": ["path", "content"],
    }
    category = "files"

    def confirmation(self, args, ctx):
        path = ctx.resolve(args["path"])
        exists = path.exists()
        size = len((args.get("content") or "").encode("utf-8"))
        return ConfirmationRequest(
            title=("Append to" if args.get("append") else "Overwrite" if exists else "Create") + f" {ctx.rel(path)}",
            detail=f"{human_size(size)} of content",
            risk="normal",
            key=f"write_file:{ctx.rel(path)}",
        )

    def run(self, args, ctx):
        path = ctx.resolve(args["path"])
        err = _guard_path(ctx, path)
        if err:
            return ToolResult.fail(err)
        if path.is_dir():
            return ToolResult.fail(f"'{args['path']}' is a directory.")
        content = args.get("content") or ""
        old_text, style, lossy = ("", "\n", False)
        if path.exists():
            if is_probably_binary(path):
                return ToolResult.fail(f"Refusing to overwrite binary file '{args['path']}'.")
            old_text, style, lossy = read_text_file(path)
            if lossy:
                return ToolResult.fail(
                    f"'{args['path']}' is not valid UTF-8; refusing to write it back to avoid data loss."
                )
        if args.get("append") and path.exists():
            new_text = old_text + ("" if old_text.endswith("\n") or not old_text else "\n") + content
        else:
            new_text = content
        if new_text == old_text:
            return ToolResult.ok(f"No change: {ctx.rel(path)} already has identical content.",
                                 path=str(path), changed=False).with_touched(str(path))
        if ctx.checkpoints:
            ctx.checkpoints.snapshot([path], reason=f"write_file {ctx.rel(path)}")
        written = write_text_file(path, new_text, style)
        diff = unified_diff(old_text, new_text, ctx.rel(path)) if path.exists() or old_text else ""
        action = "Appended to" if args.get("append") and old_text else ("Updated" if old_text else "Created")
        summary = f"{action} {ctx.rel(path)} ({human_size(written)}, {new_text.count(chr(10)) + 1} lines)"
        return ToolResult(content=summary + (f"\n\n{clip(diff, 4000)}" if diff else ""),
                          data={"path": str(path), "bytes": written, "diff": diff, "created": not old_text,
                                "changed": True}).with_touched(str(path))


# --------------------------------------------------------------------------- #
# edit_file
# --------------------------------------------------------------------------- #
class EditFileTool(Tool):
    name = "edit_file"
    description = (
        "Replace a specific piece of text in an existing file (surgical edit).\n"
        "`old_text` must match exactly once unless `replace_all` is true. Include enough "
        "surrounding lines to make the match unique. Whitespace/indentation differences are "
        "tolerated and reported. Much cheaper and safer than rewriting a whole file."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "old_text": {"type": "string", "description": "Text to find. Use an empty string only with `new_text` to prepend."},
            "new_text": {"type": "string", "description": "Replacement text."},
            "replace_all": {"type": "boolean", "description": "Replace every occurrence.", "default": False},
        },
        "required": ["path", "old_text", "new_text"],
    }
    category = "files"

    def confirmation(self, args, ctx):
        path = ctx.resolve(args["path"])
        return ConfirmationRequest(title=f"Edit {ctx.rel(path)}",
                                   detail=f"{len(args.get('old_text') or '')} chars -> {len(args.get('new_text') or '')} chars",
                                   key=f"edit_file:{ctx.rel(path)}")

    def run(self, args, ctx):
        path = ctx.resolve(args["path"])
        err = _guard_path(ctx, path)
        if err:
            return ToolResult.fail(err)
        if not path.exists():
            return ToolResult.fail(f"File not found: {args['path']}.{_suggest_name(path, ctx)} Use write_file to create it.")
        if path.is_dir():
            return ToolResult.fail(f"'{args['path']}' is a directory.")
        old_text, style, lossy = read_text_file(path)
        if lossy:
            return ToolResult.fail(f"'{args['path']}' is not valid UTF-8; refusing to edit to avoid data loss.")
        target = args["old_text"]
        replacement = args["new_text"]
        if target == replacement:
            return ToolResult.ok(f"No change: old_text and new_text are identical in {ctx.rel(path)}.",
                                 path=str(path), changed=False).with_touched(str(path))
        if target == "":
            new_content = replacement + old_text
            kind, count = "prepend", 1
        else:
            new_content, kind, count = _apply_edit(old_text, target, replacement, bool(args.get("replace_all")))
            if count == 0:
                return ToolResult.fail(
                    f"old_text not found in {ctx.rel(path)}. {_edit_hint(old_text, target)}"
                )
            if count == -1:
                occurrences = old_text.count(target)
                return ToolResult.fail(
                    f"old_text matches {occurrences} places in {ctx.rel(path)}; the edit is ambiguous. "
                    "Add more surrounding context to make it unique, or pass replace_all=true."
                )
        if ctx.checkpoints:
            ctx.checkpoints.snapshot([path], reason=f"edit_file {ctx.rel(path)}")
        write_text_file(path, new_content, style)
        diff = unified_diff(old_text, new_content, ctx.rel(path))
        label = {"exact": "", "whitespace": " (matched with whitespace tolerance)",
                 "fuzzy": " (matched approximately -- verify the diff!)",
                 "all": f" ({count} occurrences replaced)", "prepend": ""}.get(kind, "")
        return ToolResult(
            content=f"Edited {ctx.rel(path)}{label}\n\n{clip(diff, 6000)}",
            data={"path": str(path), "diff": diff, "match": kind, "count": max(count, 1), "changed": True},
        ).with_touched(str(path))


def _apply_edit(text: str, target: str, replacement: str, replace_all: bool) -> Tuple[str, str, int]:
    """Return (new_text, match_kind, count). count == -1 means ambiguous."""
    n = text.count(target)
    if n == 1 or (replace_all and n > 1):
        return text.replace(target, replacement), ("all" if replace_all and n > 1 else "exact"), (n if replace_all else 1)
    if n > 1:
        return text, "exact", -1
    if n == 0:
        # 1) whitespace-insensitive match
        norm_text = _normalise_ws(text)
        norm_target = _normalise_ws(target)
        if norm_target and norm_text.count(norm_target) == 1:
            span = _map_span(text, norm_text, norm_text.index(norm_target), len(norm_target))
            if span:
                s, e = span
                return text[:s] + replacement + text[e:], "whitespace", 1
        # 2) fuzzy line-window match
        match = _fuzzy_find(text, target)
        if match:
            s, e, ratio = match
            return text[:s] + replacement + text[e:], "fuzzy", 1
    return text, "exact", 0


def _normalise_ws(text: str) -> str:
    return "\n".join(line.strip() for line in text.split("\n"))


def _map_span(orig: str, normalised: str, start: int, length: int) -> Optional[Tuple[int, int]]:
    """Map an offset range in the whitespace-normalised text back to the original."""
    # Build an index map: normalised char position -> original position.
    mapping: List[int] = []
    pos = 0
    for line in orig.split("\n"):
        stripped = line.strip()
        lead = len(line) - len(line.lstrip())
        for i in range(len(stripped)):
            mapping.append(pos + lead + i)
        mapping.append(pos + len(line))  # the newline itself
        pos += len(line) + 1
    if start >= len(mapping) or start + length - 1 >= len(mapping):
        return None
    return mapping[start], mapping[start + length - 1] + 1


def _fuzzy_find(text: str, target: str, threshold: float = 0.86) -> Optional[Tuple[int, int, float]]:
    """Locate the best matching window of lines using difflib similarity."""
    t_lines = target.split("\n")
    if not t_lines or len(t_lines) > 400:
        return None
    lines = text.split("\n")
    offsets: List[int] = [0]
    for ln in lines:
        offsets.append(offsets[-1] + len(ln) + 1)
    window = len(t_lines)
    best: Optional[Tuple[int, int, float]] = None
    matcher = difflib.SequenceMatcher(autojunk=False)
    for i in range(0, max(1, len(lines) - window + 1)):
        cand = "\n".join(lines[i : i + window])
        matcher.set_seq2(cand)
        matcher.set_seq1(target)
        ratio = matcher.ratio()
        if ratio >= threshold and (best is None or ratio > best[2]):
            best = (offsets[i], offsets[i + window] - 1, ratio)
    if best:
        return best
    # fall back to a single-anchor line search (handles inserted/removed lines)
    anchor = max((l.strip() for l in t_lines if l.strip()), key=len, default="")
    if len(anchor) < 8:
        return None
    for i, ln in enumerate(lines):
        if ln.strip() == anchor:
            s = offsets[i]
            e = offsets[min(i + window, len(lines)) ] - 1
            cand = text[s:e]
            matcher.set_seq1(target)
            matcher.set_seq2(cand)
            ratio = matcher.ratio()
            if ratio >= threshold - 0.06 and (best is None or ratio > best[2]):
                best = (s, e, ratio)
    return best


def _edit_hint(text: str, target: str) -> str:
    lines = text.split("\n")
    t_first = next((l.strip() for l in target.split("\n") if l.strip()), "")
    if not t_first:
        return "Provide the exact text as it appears in the file."
    close = difflib.get_close_matches(t_first, [l.strip() for l in lines], n=3, cutoff=0.5)
    if close:
        return "Closest lines in the file:\n  " + "\n  ".join(c[:160] for c in close)
    return "Provide the exact text as it appears in the file (use read_file first)."


# --------------------------------------------------------------------------- #
# multi_edit (atomic)
# --------------------------------------------------------------------------- #
class MultiEditTool(Tool):
    name = "multi_edit"
    description = (
        "Apply several edit_file operations to one file atomically: if any edit fails, "
        "none of them are written. Edits are applied in order."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "edits": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "properties": {
                        "old_text": {"type": "string"},
                        "new_text": {"type": "string"},
                        "replace_all": {"type": "boolean", "default": False},
                    },
                    "required": ["old_text", "new_text"],
                },
            },
        },
        "required": ["path", "edits"],
    }
    category = "files"

    def confirmation(self, args, ctx):
        path = ctx.resolve(args["path"])
        return ConfirmationRequest(title=f"Apply {len(args.get('edits') or [])} edits to {ctx.rel(path)}",
                                   risk="normal", key=f"multi_edit:{ctx.rel(path)}")

    def run(self, args, ctx):
        path = ctx.resolve(args["path"])
        err = _guard_path(ctx, path)
        if err:
            return ToolResult.fail(err)
        if not path.is_file():
            return ToolResult.fail(f"File not found: {args['path']}.{_suggest_name(path, ctx)}")
        original, style, lossy = read_text_file(path)
        if lossy:
            return ToolResult.fail(f"'{args['path']}' is not valid UTF-8; refusing to edit.")
        text = original
        applied: List[str] = []
        for i, edit in enumerate(args["edits"], start=1):
            target, replacement = edit.get("old_text", ""), edit.get("new_text", "")
            new_text, kind, count = _apply_edit(text, target, replacement, bool(edit.get("replace_all")))
            if count == 0:
                return ToolResult.fail(f"Edit #{i} failed: old_text not found. No changes were written. {_edit_hint(text, target)}")
            if count == -1:
                return ToolResult.fail(f"Edit #{i} failed: old_text is ambiguous (multiple matches). No changes were written.")
            text = new_text
            applied.append(f"#{i} {kind}")
        if text == original:
            return ToolResult.ok(f"No net change to {ctx.rel(path)}.", path=str(path), changed=False)
        if ctx.checkpoints:
            ctx.checkpoints.snapshot([path], reason=f"multi_edit {ctx.rel(path)}")
        write_text_file(path, text, style)
        diff = unified_diff(original, text, ctx.rel(path))
        return ToolResult(content=f"Applied {len(applied)} edits to {ctx.rel(path)} [{', '.join(applied)}]\n\n{clip(diff, 6000)}",
                          data={"path": str(path), "diff": diff, "edits": len(applied), "changed": True}
                          ).with_touched(str(path))


# --------------------------------------------------------------------------- #
# list_dir / tree
# --------------------------------------------------------------------------- #
class ListDirTool(Tool):
    name = "list_dir"
    description = "List files and directories at a path (gitignore-aware), with sizes. Use `tree` for a recursive view."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "default": "."},
            "tree": {"type": "boolean", "description": "Recursive tree view.", "default": False},
            "depth": {"type": "integer", "description": "Max depth for tree view.", "minimum": 1, "default": 3},
            "limit": {"type": "integer", "description": "Max entries to return.", "minimum": 1, "default": 400},
            "all": {"type": "boolean", "description": "Include ignored/hidden entries.", "default": False},
        },
    }
    category = "files"
    read_only = True

    def run(self, args, ctx):
        path = ctx.resolve(args.get("path") or ".")
        if not path.exists():
            return ToolResult.fail(f"Directory not found: {args.get('path')}")
        if path.is_file():
            return ToolResult.fail(f"'{args.get('path')}' is a file, not a directory. Use read_file.")
        limit = int(args.get("limit") or 400)
        matcher = None if args.get("all") else IgnoreMatcher(path)
        if not args.get("tree"):
            try:
                entries = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
            except OSError as exc:
                return ToolResult.fail(f"Cannot list directory: {exc}")
            shown, out = 0, []
            for e in entries:
                if matcher is not None and matcher.is_ignored(e, is_dir=e.is_dir()):
                    continue
                if shown >= limit:
                    out.append(f"... ({len(entries) - shown} more entries; raise limit)")
                    break
                try:
                    st = e.stat()
                except OSError:
                    continue
                shown += 1
                if e.is_dir():
                    out.append(f"  {e.name}/")
                else:
                    out.append(f"  {e.name}  ({human_size(st.st_size)})")
            return ToolResult(content=f"{ctx.rel(path)}/\n" + ("\n".join(out) or "  (empty)"),
                              data={"path": str(path), "entries": shown})
        depth = int(args.get("depth") or 3)
        lines: List[str] = []
        count = 0

        def walk(d: Path, level: int) -> None:
            nonlocal count
            if level > depth or count >= limit:
                return
            try:
                entries = sorted(d.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
            except OSError:
                return
            for e in entries:
                if count >= limit:
                    return
                if matcher is not None and matcher.is_ignored(e, is_dir=e.is_dir()):
                    continue
                count += 1
                lines.append(("│  " * (level - 1) + "├─ " if level else "") + e.name + ("/" if e.is_dir() else ""))
                if e.is_dir() and e.name not in ALWAYS_SKIP_DIRS:
                    walk(e, level + 1)

        lines.append(f"{ctx.rel(path)}/")
        walk(path, 0)
        return ToolResult(content="\n".join(lines), data={"path": str(path), "entries": count})


# --------------------------------------------------------------------------- #
# find_files (glob)
# --------------------------------------------------------------------------- #
class FindFilesTool(Tool):
    name = "find_files"
    description = (
        "Find files by glob pattern (e.g. '**/*.py', 'src/**/*.ts'), respecting ignore rules. "
        "Returns paths relative to the workspace, most recently modified first when `sort=mtime`."
    )
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Glob pattern. '**' spans directories."},
            "path": {"type": "string", "description": "Directory to search in.", "default": "."},
            "limit": {"type": "integer", "minimum": 1, "default": 200},
            "sort": {"type": "string", "enum": ["name", "mtime", "size"], "default": "name"},
        },
        "required": ["pattern"],
    }
    category = "search"
    read_only = True

    def run(self, args, ctx):
        root = ctx.resolve(args.get("path") or ".")
        if not root.is_dir():
            return ToolResult.fail(f"Not a directory: {args.get('path')}")
        pattern = args["pattern"]
        matcher = IgnoreMatcher(root)
        limit = int(args.get("limit") or 200)
        hits: List[Tuple[Any, Path]] = []
        for p in walk_files(root, matcher=matcher, max_files=60_000):
            rel = p.relative_to(root).as_posix()
            if not glob_match(pattern, rel):
                continue
            key: Any = rel
            if args.get("sort") == "mtime":
                try:
                    key = (-p.stat().st_mtime, rel)
                except OSError:
                    key = (0.0, rel)
            elif args.get("sort") == "size":
                try:
                    key = (-p.stat().st_size, rel)
                except OSError:
                    key = (0, rel)
            hits.append((key, p))
            if len(hits) >= max(limit * 4, limit):
                break
        hits.sort(key=lambda kv: kv[0])
        out = [ctx.rel(p) for _, p in hits[:limit]]
        more = f"\n... ({len(hits) - limit} more matches; raise limit or narrow the pattern)" if len(hits) > limit else ""
        return ToolResult(content=("\n".join(out) if out else f"No files match '{pattern}' under {ctx.rel(root)}.") + more,
                          data={"count": len(out), "pattern": pattern})


# --------------------------------------------------------------------------- #
# delete_path
# --------------------------------------------------------------------------- #
class DeletePathTool(Tool):
    name = "delete_path"
    description = "Delete a file or an empty directory. Refuses to delete non-empty directories unless `recursive` is true."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "recursive": {"type": "boolean", "default": False},
        },
        "required": ["path"],
    }
    category = "files"

    def confirmation(self, args, ctx):
        path = ctx.resolve(args["path"])
        recursive = bool(args.get("recursive"))
        n = 0
        if recursive and path.is_dir():
            n = sum(1 for _ in walk_files(path))
        return ConfirmationRequest(
            title=f"Delete {'directory' if path.is_dir() else 'file'} {ctx.rel(path)}" + (f" ({n} files)" if n else ""),
            detail="This is destructive. A checkpoint is taken first, so /undo can restore it.",
            risk="dangerous" if recursive else "elevated",
            key="delete_path:*" if recursive else f"delete_path:{ctx.rel(path)}",
        )

    def run(self, args, ctx):
        import shutil

        path = ctx.resolve(args["path"])
        err = _guard_path(ctx, path)
        if err:
            return ToolResult.fail(err)
        if not path.exists():
            return ToolResult.fail(f"Path not found: {args['path']}")
        if path == ctx.workspace_root:
            return ToolResult.fail("Refusing to delete the workspace root.")
        recursive = bool(args.get("recursive"))
        if ctx.checkpoints:
            targets = [p for p in walk_files(path)] if path.is_dir() else [path]
            ctx.checkpoints.snapshot(targets[:200], reason=f"delete_path {ctx.rel(path)}")
        try:
            if path.is_dir():
                if recursive:
                    shutil.rmtree(path)
                else:
                    path.rmdir()
            else:
                path.unlink()
        except OSError as exc:
            return ToolResult.fail(f"Delete failed: {exc}")
        return ToolResult(content=f"Deleted {ctx.rel(path)}", data={"path": str(path), "recursive": recursive}
                          ).with_touched(str(path))


def build_tools() -> List[Tool]:
    return [ReadFileTool(), WriteFileTool(), EditFileTool(), MultiEditTool(), ListDirTool(),
            FindFilesTool(), DeletePathTool()]
