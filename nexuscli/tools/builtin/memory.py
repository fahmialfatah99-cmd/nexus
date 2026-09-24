"""Persistent memory tools.

Two scopes:
* **project** -- ``<workspace>/.nexus/MEMORY.md``, committed with the repo so the
  whole team (and every future session) benefits.
* **global** -- ``~/.nexus/MEMORY.md``, personal preferences across projects.

Memory is stored as markdown with ``## Section`` headings so individual facts can
be replaced or removed without rewriting the file. The app injects both files
into the system prompt at session start.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ...core.paths import memory_file as global_memory_file
from ..base import ConfirmationRequest, Tool, ToolContext, ToolResult

MAX_MEMORY_BYTES = 256 * 1024


def project_memory_path(ctx: ToolContext) -> Path:
    override = ctx.vars.get("project_memory")
    if override:
        return Path(override)
    return ctx.workspace_root / ".nexus" / "MEMORY.md"


def scope_path(ctx: ToolContext, scope: str) -> Path:
    return global_memory_file() if scope == "global" else project_memory_path(ctx)


def read_memory(path: Path) -> str:
    try:
        if path.is_file():
            return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        pass
    return ""


def write_memory(path: Path, text: str) -> None:
    if len(text.encode("utf-8")) > MAX_MEMORY_BYTES:
        raise ValueError(f"memory file would exceed {MAX_MEMORY_BYTES // 1024}KB")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def split_sections(text: str) -> List[Tuple[str, List[str]]]:
    """Split markdown into (heading, body_lines) pairs. Preamble heading is ''."""
    sections: List[Tuple[str, List[str]]] = []
    current: List[str] = []
    heading = ""
    for line in text.split("\n"):
        if line.startswith("## "):
            sections.append((heading, current))
            heading = line[3:].strip()
            current = []
        else:
            current.append(line)
    sections.append((heading, current))
    return sections


def render_sections(sections: List[Tuple[str, List[str]]]) -> str:
    out: List[str] = []
    for heading, body in sections:
        if heading:
            out.append(f"## {heading}")
        out.extend(body)
    text = "\n".join(out)
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text.strip() + "\n"


def upsert_section(text: str, heading: str, entry: str) -> str:
    sections = split_sections(text)
    stamp = time.strftime("%Y-%m-%d")
    line = f"- {entry.strip()}  <!-- {stamp} -->"
    for i, (h, body) in enumerate(sections):
        if h.strip().lower() == heading.strip().lower():
            body = [b for b in body if b.strip() and _strip_stamp(b) != _strip_stamp(line)]
            body.append(line)
            sections[i] = (h, body)
            return render_sections(sections)
    sections.append((heading, ["", line, ""]))
    return render_sections(sections)


def _strip_stamp(line: str) -> str:
    return re.sub(r"\s*<!--.*?-->\s*$", "", line).strip().lstrip("- ").strip()


def remove_entry(text: str, needle: str) -> Tuple[str, bool]:
    needle_norm = needle.strip().lower()
    sections = split_sections(text)
    removed = False
    for i, (h, body) in enumerate(sections):
        kept = []
        for line in body:
            if line.strip().startswith("-") and needle_norm in line.lower():
                removed = True
                continue
            kept.append(line)
        sections[i] = (h, kept)
    return (render_sections(sections) if removed else text), removed


class MemoryTool(Tool):
    name = "memory"
    description = (
        "Read or update long-term memory that is automatically loaded at the start of every session.\n"
        "Save durable facts only: user preferences, project conventions, architecture decisions, "
        "environment quirks, 'never do X' rules. Do NOT save task progress or anything derivable "
        "from the code itself.\n"
        "scope=project writes .nexus/MEMORY.md (shared with the team); scope=global writes your "
        "personal cross-project memory."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["view", "add", "replace_section", "remove"], "default": "add"},
            "scope": {"type": "string", "enum": ["project", "global"], "default": "project"},
            "section": {"type": "string", "description": "Markdown heading, e.g. 'Conventions' or 'User preferences'."},
            "content": {"type": "string", "description": "The fact to remember (one line preferred)."},
            "match": {"type": "string", "description": "Substring identifying the entry to remove."},
        },
    }
    category = "memory"

    def confirmation(self, args, ctx):
        if args.get("action") == "view":
            return None
        scope = args.get("scope", "project")
        return ConfirmationRequest(title=f"Write to {scope} memory",
                                   detail=(args.get("content") or args.get("match") or "")[:120],
                                   risk="normal", key=f"memory:{scope}")

    def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        action = args.get("action") or "add"
        scope = args.get("scope") or "project"
        path = scope_path(ctx, scope)
        text = read_memory(path)

        if action == "view":
            both = []
            for label, p in (("PROJECT", project_memory_path(ctx)), ("GLOBAL", global_memory_file())):
                body = read_memory(p)
                both.append(f"--- {label}: {p} ---\n{body.strip() if body.strip() else '(empty)'}")
            return ToolResult(content="\n\n".join(both), data={"path": str(path)})

        if action == "remove":
            match = (args.get("match") or "").strip()
            if not match:
                return ToolResult.fail("action=remove requires 'match'.")
            new_text, removed = remove_entry(text, match)
            if not removed:
                return ToolResult.fail(f"No memory entry matching '{match}' in {scope} memory.")
            write_memory(path, new_text)
            return ToolResult(content=f"Removed memory entry matching '{match}' from {scope} memory.",
                              data={"path": str(path)}).with_touched(str(path))

        content = (args.get("content") or "").strip()
        if not content:
            return ToolResult.fail("action=add/replace_section requires 'content'.")
        section = (args.get("section") or "Notes").strip()
        if action == "replace_section":
            sections = [(h, b) for h, b in split_sections(text) if h.strip().lower() != section.lower()]
            sections.append((section, ["", *[f"- {line.strip()}" for line in content.split("\n") if line.strip()], ""]))
            new_text = render_sections(sections)
        else:
            new_text = upsert_section(text, section, content)
        if new_text == text:
            return ToolResult.ok(f"Memory unchanged (entry already present in {scope} memory).")
        try:
            write_memory(path, new_text)
        except (OSError, ValueError) as exc:
            return ToolResult.fail(f"Could not write memory: {exc}")
        return ToolResult(content=f"Saved to {scope} memory ({path}): [{section}] {content[:160]}",
                          data={"path": str(path), "section": section}).with_touched(str(path))


def memory_paths(workspace_root: Optional[Path] = None, ctx: Optional[ToolContext] = None) -> List[Tuple[str, Path]]:
    """(label, path) pairs, in injection order. Works with or without a ToolContext."""
    if ctx is not None:
        return [("Project memory", project_memory_path(ctx)), ("User memory", global_memory_file())]
    root = Path(workspace_root) if workspace_root else Path.cwd()
    return [("Project memory", root / ".nexus" / "MEMORY.md"), ("User memory", global_memory_file())]


def load_memory_for_prompt(workspace_root: Optional[Path] = None,
                           ctx: Optional[ToolContext] = None) -> str:
    """Combine both scopes into the block injected into the system prompt."""
    chunks = []
    for label, path in memory_paths(workspace_root, ctx):
        body = read_memory(path).strip()
        if body:
            chunks.append(f"### {label} ({path.name})\n{body}")
    return "\n\n".join(chunks)


def build_tools() -> List[Tool]:
    return [MemoryTool()]
