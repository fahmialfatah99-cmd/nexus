"""Search tool: ``grep`` (content) powered by a ripgrep-style walker.

Pure-Python, but built for the things that actually matter in an agent loop:
regex *or* literal mode, per-file globs, context lines, binary/ignore skipping,
a hard result cap, and output formatted so a model can act on it directly.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ...core.ignore import IgnoreMatcher, glob_match, is_probably_binary, walk_files
from ..base import Tool, ToolContext, ToolResult, clip

MAX_FILE_BYTES = 4_000_000
MAX_RESULTS = 300
MAX_FILES_SCANNED = 20_000


class GrepTool(Tool):
    name = "grep"
    description = (
        "Search file *contents* with a regular expression (ripgrep-like). Skips ignored and binary "
        "files automatically.\n"
        "Returns `path:line: text`. Use `glob` to restrict file types ('*.py'), `files_with_matches` "
        "for a compact list, and `context` for surrounding lines. Prefer this over `bash grep -r`."
    )
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Regular expression (Python syntax) or literal text."},
            "path": {"type": "string", "description": "File or directory to search.", "default": "."},
            "glob": {"type": "string", "description": "Only search files matching this glob, e.g. '*.py'."},
            "ignore_case": {"type": "boolean", "default": False},
            "literal": {"type": "boolean", "description": "Treat pattern as literal text, not a regex.", "default": False},
            "context": {"type": "integer", "description": "Lines of context around each match.", "minimum": 0, "maximum": 20, "default": 0},
            "max_results": {"type": "integer", "minimum": 1, "maximum": 2000, "default": 100},
            "files_with_matches": {"type": "boolean", "description": "Only list file names.", "default": False},
            "word_regexp": {"type": "boolean", "description": "Match whole words only.", "default": False},
            "multiline": {"type": "boolean", "description": "Allow '.' to match newlines (DOTALL).", "default": False},
        },
        "required": ["pattern"],
    }
    category = "search"
    read_only = True

    def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        pattern = args["pattern"]
        flags = re.MULTILINE
        if args.get("ignore_case"):
            flags |= re.IGNORECASE
        if args.get("multiline"):
            flags |= re.DOTALL
        try:
            rx = re.compile(re.escape(pattern) if args.get("literal") else pattern, flags)
        except re.error as exc:
            return ToolResult.fail(f"Invalid regular expression '{pattern}': {exc}. Pass literal=true to search plain text.")
        if args.get("word_regexp"):
            rx = re.compile(rf"\b(?:{rx.pattern})\b", rx.flags)

        root = ctx.resolve(args.get("path") or ".")
        if not root.exists():
            return ToolResult.fail(f"Path not found: {args.get('path')}")
        matcher = IgnoreMatcher(root if root.is_dir() else root.parent)
        glob_filter = args.get("glob")
        max_results = int(args.get("max_results") or 100)
        context = int(args.get("context") or 0)
        only_files = bool(args.get("files_with_matches"))

        files: List[Path] = [root] if root.is_file() else list(
            _iter_files(root, matcher, glob_filter, MAX_FILES_SCANNED)
        )
        out: List[str] = []
        matched_files = 0
        total_matches = 0
        for f in files:
            if ctx.is_cancelled():
                break
            hits = _search_file(f, rx, context, max_results - total_matches if not only_files else 1)
            if not hits:
                continue
            matched_files += 1
            total_matches += len(hits)
            if only_files:
                out.append(ctx.rel(f))
                if len(out) >= max_results:
                    break
                continue
            out.extend(f"{ctx.rel(f)}:{line}{sep}{text}" for line, text, sep in hits)
            if total_matches >= max_results:
                out.append(f"... stopped at {max_results} matches (raise max_results or narrow the pattern)")
                break
        if not out:
            return ToolResult(
                content=f"No matches for '{pattern}' in {ctx.rel(root)} "
                        f"({len(files)} files scanned, ignored/binary files excluded).",
                data={"matches": 0, "files_scanned": len(files)},
            )
        header = f"{total_matches if not only_files else matched_files} match(es) in {matched_files} file(s), {len(files)} scanned"
        return ToolResult(content=f"{header}\n\n" + clip("\n".join(out), ctx.max_output_chars),
                          data={"matches": total_matches, "files": matched_files, "scanned": len(files)})


def _iter_files(root: Path, matcher: IgnoreMatcher, glob_filter: Optional[str], limit: int):
    count = 0
    for p in walk_files(root, matcher=matcher, max_files=limit):
        if glob_filter and not glob_match(glob_filter, p.relative_to(root).as_posix()):
            continue
        count += 1
        yield p
        if count >= limit:
            return


def _search_file(path: Path, rx: "re.Pattern[str]", context: int, budget: int) -> List[Tuple[int, str, str]]:
    if budget <= 0:
        return []
    try:
        if path.stat().st_size > MAX_FILE_BYTES or is_probably_binary(path):
            return []
        raw = path.read_bytes()
    except OSError:
        return []
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("utf-8", errors="replace")
    lines = text.split("\n")
    hits: List[Tuple[int, str, str]] = []
    reported: set = set()
    for i, line in enumerate(lines):
        if len(hits) >= budget:
            break
        if not rx.search(line):
            continue
        lo = max(0, i - context)
        hi = min(len(lines), i + context + 1)
        if context == 0:
            hits.append((i + 1, _trim(line), ":"))
            reported.add(i)
            continue
        for j in range(lo, hi):
            if j in reported:
                continue
            reported.add(j)
            sep = ":" if j == i else "-"
            hits.append((j + 1, _trim(lines[j]), sep))
        hits.append((-1, "--", ""))  # separator between match groups
    while hits and hits[-1][0] == -1:
        hits.pop()
    return hits


def _trim(line: str, width: int = 400) -> str:
    line = line.replace("\t", "    ").rstrip()
    if len(line) > width:
        return line[:width] + "…"
    return line


class FileStatsTool(Tool):
    name = "file_info"
    description = "Return metadata for a path: size, line count, modified time, encoding, and whether it is binary."
    parameters = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    }
    category = "files"
    read_only = True

    def run(self, args, ctx):
        path = ctx.resolve(args["path"])
        if not path.exists():
            return ToolResult.fail(f"Path not found: {args['path']}")
        st = path.stat()
        import time

        info: Dict[str, Any] = {
            "path": ctx.rel(path),
            "kind": "directory" if path.is_dir() else "file",
            "size": st.st_size,
            "modified": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime)),
        }
        if path.is_file():
            info["binary"] = is_probably_binary(path)
            if not info["binary"]:
                try:
                    with open(path, "rb") as fh:
                        data = fh.read(MAX_FILE_BYTES)
                    info["lines"] = data.count(b"\n") + (0 if data.endswith(b"\n") else 1)
                    info["encoding"] = "utf-8"
                    try:
                        data.decode("utf-8")
                    except UnicodeDecodeError:
                        info["encoding"] = "unknown (not utf-8)"
                except OSError:
                    pass
        else:
            try:
                entries = list(path.iterdir())
                info["entries"] = len(entries)
                info["dirs"] = sum(1 for e in entries if e.is_dir())
            except OSError:
                pass
        return ToolResult(content="\n".join(f"{k}: {v}" for k, v in info.items()), data=info)


def build_tools() -> List[Tool]:
    return [GrepTool(), FileStatsTool()]
