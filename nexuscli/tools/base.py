"""Tool contract, registry and shared helpers.

A tool is a small, self-describing unit:

* ``parameters`` -- a JSON Schema validated (and coerced) *before* execution,
  so a model can never reach ``run()`` with wrong types.
* ``read_only`` -- drives the permission engine: read-only tools never prompt.
* ``confirmation()`` -- returns the human-facing description of the *side effect*
  about to happen (path, command, diff summary) that the approval UI shows.
* ``run()`` -- returns a :class:`ToolResult`; raising is reserved for
  programmer errors, expected failures come back as ``is_error=True`` so the
  model can react.
"""

from __future__ import annotations

import fnmatch
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from ..core.errors import ToolNotFound, ValidationError
from ..core.jsonschema import normalize_args
from ..core.logging_ import Logger, get_logger
from ..providers.base import ToolSpec

DEFAULT_MAX_OUTPUT_CHARS = 24_000


# --------------------------------------------------------------------------- #
# Result
# --------------------------------------------------------------------------- #
@dataclass
class ToolResult:
    content: str = ""
    is_error: bool = False
    data: Dict[str, Any] = field(default_factory=dict)
    #: paths touched by this call -- used for checkpoints, /undo and audit log
    touched: List[str] = field(default_factory=list)

    @staticmethod
    def ok(content: str = "", **data: Any) -> "ToolResult":
        return ToolResult(content=content, data=data)

    @staticmethod
    def fail(message: str, **data: Any) -> "ToolResult":
        return ToolResult(content=message, is_error=True, data=data)

    def with_touched(self, *paths: str) -> "ToolResult":
        for p in paths:
            if p and p not in self.touched:
                self.touched.append(p)
        return self


# --------------------------------------------------------------------------- #
# Context
# --------------------------------------------------------------------------- #
# ConfirmationRequest is defined once, in core.permissions, and re-exported here
# so tools and the permission engine share a single type (isinstance works).
from ..core.permissions import ConfirmationRequest  # noqa: E402


@dataclass
class ToolContext:
    """Everything a tool may need. Injected by the agent loop."""

    cwd: Path
    workspace_root: Path
    config: Any = None
    permissions: Any = None
    checkpoints: Any = None
    ui: Any = None
    session: Any = None
    events: Any = None
    transport: Any = None
    agent: str = "main"
    cancelled: Optional[threading.Event] = None
    log: Logger = field(default_factory=get_logger)
    vars: Dict[str, Any] = field(default_factory=dict)
    max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS
    read_only_mode: bool = False

    def is_cancelled(self) -> bool:
        return bool(self.cancelled is not None and self.cancelled.is_set())

    def resolve(self, raw: str) -> Path:
        from ..core.ignore import resolve_path

        return resolve_path(raw, self.cwd)

    def rel(self, path: Path) -> str:
        from ..core.ignore import relative_display

        return relative_display(path, self.workspace_root)


# --------------------------------------------------------------------------- #
# Tool base class
# --------------------------------------------------------------------------- #
class Tool:
    name: str = "tool"
    description: str = ""
    parameters: Dict[str, Any] = {"type": "object", "properties": {}}
    category: str = "general"
    read_only: bool = False
    #: safe to run concurrently with other tools (swarm/parallel execution)
    concurrency_safe: bool = True
    aliases: Sequence[str] = ()
    timeout: float = 120.0
    #: tools that need the network are auto-disabled in --offline mode
    needs_network: bool = False
    #: hidden tools do not appear in /help listings but stay callable
    hidden: bool = False

    def spec(self) -> ToolSpec:
        return ToolSpec(name=self.name, description=self.description.strip(), parameters=self.parameters)

    def confirmation(self, args: Dict[str, Any], ctx: ToolContext) -> Optional[ConfirmationRequest]:
        return None

    def validate(self, args: Dict[str, Any]) -> None:
        """Hook for cross-field validation beyond the JSON schema."""

    def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:  # pragma: no cover - abstract
        raise NotImplementedError(f"{type(self).__name__} must implement run()")

    def prepare(self, raw_args: Any, ctx: ToolContext) -> Dict[str, Any]:
        """Coerce + validate + semantic check. Raises ValidationError."""
        args, errors = normalize_args(raw_args, self.parameters)
        if errors:
            raise ValidationError(f"Invalid arguments for '{self.name}': " + "; ".join(errors))
        self.validate(args)
        return args

    def execute(self, raw_args: Any, ctx: ToolContext) -> ToolResult:
        """Full pipeline: validate -> guard read-only mode -> run -> clip output."""
        args = self.prepare(raw_args, ctx)
        if ctx.read_only_mode and not self.read_only:
            return ToolResult.fail(
                f"Tool '{self.name}' is disabled in read-only mode (--read-only). "
                "Ask the user to re-run without read-only mode."
            )
        result = self.run(args, ctx)
        if not isinstance(result, ToolResult):  # defensive: plugins may return str
            result = ToolResult.ok(str(result))
        result.content = clip(result.content, ctx.max_output_chars)
        return result


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}
        self._aliases: Dict[str, str] = {}
        self._disabled: set = set()
        self._lock = threading.RLock()

    def register(self, tool: Tool, *, replace: bool = False) -> None:
        with self._lock:
            if not isinstance(tool, Tool):
                raise ValidationError(f"register() expects a Tool instance, got {type(tool).__name__}")
            if not tool.name:
                raise ValidationError("Tool.name must be non-empty")
            if tool.name in self._tools and not replace:
                raise ValidationError(f"Tool '{tool.name}' is already registered")
            self._tools[tool.name] = tool
            for alias in tool.aliases:
                self._aliases[alias] = tool.name

    def register_many(self, tools: Iterable[Tool], *, replace: bool = False) -> None:
        for t in tools:
            self.register(t, replace=replace)

    def unregister(self, name: str) -> bool:
        with self._lock:
            real = self._aliases.get(name, name)
            tool = self._tools.pop(real, None)
            if tool is None:
                return False
            for alias, target in list(self._aliases.items()):
                if target == real:
                    del self._aliases[alias]
            self._disabled.discard(real)
            return True

    def get(self, name: str) -> Tool:
        with self._lock:
            real = self._aliases.get(name, name)
            tool = self._tools.get(real)
            if tool is None:
                close = _closest(real, list(self._tools))
                raise ToolNotFound(
                    f"Unknown tool '{real}'." + (f" Did you mean '{close}'?" if close else "")
                    + f" Available: {', '.join(sorted(self._tools))[:400]}"
                )
            return tool

    def has(self, name: str) -> bool:
        real = self._aliases.get(name, name)
        return real in self._tools

    def set_enabled(self, name: str, enabled: bool) -> None:
        real = self._aliases.get(name, name)
        with self._lock:
            if enabled:
                self._disabled.discard(real)
            else:
                self._disabled.add(real)

    def is_enabled(self, name: str) -> bool:
        real = self._aliases.get(name, name)
        return real not in self._disabled and real in self._tools

    def enabled_tools(self, *, offline: bool = False, categories: Optional[Sequence[str]] = None) -> List[Tool]:
        with self._lock:
            out = [t for n, t in self._tools.items() if n not in self._disabled]
        if offline:
            out = [t for t in out if not t.needs_network]
        if categories:
            cats = set(categories)
            out = [t for t in out if t.category in cats]
        return sorted(out, key=lambda t: (t.category, t.name))

    def specs(self, **kwargs: Any) -> List[ToolSpec]:
        return [t.spec() for t in self.enabled_tools(**kwargs)]

    def names(self) -> List[str]:
        with self._lock:
            return sorted(self._tools)

    def by_category(self) -> Dict[str, List[Tool]]:
        out: Dict[str, List[Tool]] = {}
        for t in self.enabled_tools():
            out.setdefault(t.category, []).append(t)
        return dict(sorted(out.items()))


def _closest(name: str, candidates: Sequence[str]) -> Optional[str]:
    import difflib

    matches = difflib.get_close_matches(name, list(candidates), n=1, cutoff=0.6)
    return matches[0] if matches else None


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def clip(text: str, max_chars: int = DEFAULT_MAX_OUTPUT_CHARS, head_ratio: float = 0.7) -> str:
    """Truncate huge output keeping head + tail (models need both ends)."""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    head = int(max_chars * head_ratio)
    tail = max_chars - head
    omitted = len(text) - head - tail
    return (
        text[:head]
        + f"\n\n... [{omitted:,} characters truncated by NEXUS; "
        f"re-run with narrower filters (offset/limit/pattern) to see more] ...\n\n"
        + (text[-tail:] if tail > 0 else "")
    )


def add_line_numbers(text: str, start: int = 1, width: Optional[int] = None) -> str:
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    w = width or len(str(start + len(lines)))
    return "\n".join(f"{i + start:>{w}}\t{line}" for i, line in enumerate(lines))


def match_any(patterns: Sequence[str], value: str) -> bool:
    return any(fnmatch.fnmatchcase(value, p) for p in patterns)


def requires(args: Dict[str, Any], key: str) -> Any:
    if key not in args or args[key] in (None, ""):
        raise ValidationError(f"Missing required argument '{key}'")
    return args[key]


__all__ = [
    "ToolResult", "ToolContext", "ConfirmationRequest", "Tool", "ToolRegistry",  # noqa: E402
    "clip", "add_line_numbers", "match_any", "requires", "DEFAULT_MAX_OUTPUT_CHARS",
]
