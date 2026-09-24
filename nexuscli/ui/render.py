"""The renderer: everything the user sees.

It implements the UI protocol the agent loop and swarm runner call (``on_text``,
``on_tool_start``, ``on_agent_end``, ...). Keeping the protocol in one class is
what lets the whole engine be unit-tested with a recording stub.

Two output strategies, chosen automatically:

* **live** (solo agent) -- deltas stream through :class:`StreamingMarkdown` as
  they arrive.
* **buffered** (swarm) -- parallel agents would interleave into unreadable soup,
  so each agent's output is collected and printed as one labelled block when it
  finishes, while a status line shows who is still running.
"""

from __future__ import annotations

import sys
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..core.permissions import ConfirmationRequest
from . import diff as diff_ui
from .markdown import StreamingMarkdown, render_inline, render_markdown
from .i18n import tr
from .theme import RESET, Style, strip_ansi, truncate, visible_width
from .widgets import (
    Spinner,
    badge,
    box,
    columns,
    format_cost,
    format_duration,
    format_number,
    kv,
    progress,
    rule,
    table,
)

RISK_ROLE = {"normal": "accent", "elevated": "warning", "dangerous": "error"}


class Renderer:
    def __init__(self, style: Style, *, stream=None, err_stream=None, live: bool = True,
                 spinner_enabled: bool = True, show_usage: bool = True, show_reasoning: bool = True,
                 quiet: bool = False, verbose: bool = False, input_fn=None) -> None:
        self.style = style
        self.out = stream if stream is not None else sys.stdout
        self.err = err_stream if err_stream is not None else sys.stderr
        self.live = live
        self.spinner_enabled = spinner_enabled
        self.show_usage = show_usage
        self.show_reasoning = show_reasoning
        self.quiet = quiet
        self.verbose = verbose
        self._input = input_fn or input
        self._lock = threading.RLock()
        self._stream_md: Dict[str, StreamingMarkdown] = {}
        self._buffers: Dict[str, List[str]] = {}
        self._reasoning_buffers: Dict[str, List[str]] = {}
        self._spinner: Optional[Spinner] = None
        self._streaming_agent: Optional[str] = None
        self._pending_newline = False
        self._tool_counts: Dict[str, int] = {}
        self._wrote_text: Dict[str, bool] = {}
        #: True when streamed text ended mid-line; the next block of output must
        #: start with a newline so tool lines never glue onto prose.
        self._partial_line = False

    # ------------------------------------------------------------------ #
    # low level
    # ------------------------------------------------------------------ #
    def write(self, text: str) -> None:
        if not text:
            return
        with self._lock:
            self._stop_spinner_transiently()
            try:
                self.out.write(text)
                self.out.flush()
            except (ValueError, OSError):
                pass

    def println(self, text: str = "") -> None:
        self._break_partial_line()
        self.write(text + "\n")

    def _break_partial_line(self) -> None:
        if self._partial_line:
            self._partial_line = False
            self.write("\n")

    def _stop_spinner_transiently(self) -> None:
        if self._spinner is not None and self._spinner.enabled:
            try:
                self.out.write("\r\x1b[K")
                self.out.flush()
            except (ValueError, OSError):
                pass

    def _restart_spinner(self) -> None:
        if self._spinner is not None and self._spinner.enabled:
            try:
                self.out.flush()
            except (ValueError, OSError):
                pass

    # ------------------------------------------------------------------ #
    # high level helpers
    # ------------------------------------------------------------------ #
    def markdown(self, text: str, *, width: int = 0) -> None:
        self.println(render_markdown(text, self.style, width=width or self.style.width))

    def info(self, text: str) -> None:
        self.println(self.style.paint("info", "ℹ ") + self.style.dim(text))

    def success(self, text: str) -> None:
        self.println(self.style.paint("success", "✓ ") + text)

    def warning(self, text: str) -> None:
        self.println(self.style.paint("warning", "! ") + self.style.paint("warning", text))

    def error(self, text: str, *, hint: str = "") -> None:
        self.println(self.style.paint("error", "✗ ") + self.style.paint("error", text))
        if hint:
            self.println("  " + self.style.dim(hint))

    def dim(self, text: str) -> None:
        self.println(self.style.dim(text))

    def rule(self, title: str = "") -> None:
        self.println(rule(self.style, title))

    def table(self, headers: Sequence[str], rows: Sequence[Sequence[Any]], *, aligns=None) -> None:
        self.println(table(headers, rows, self.style, aligns=aligns))

    def box(self, title: str, lines: Sequence[str], *, role: str = "border") -> None:
        self.println(box(title, lines, self.style, role=role))

    def kv(self, pairs: Sequence[Tuple[str, Any]]) -> None:
        self.println(kv(pairs, self.style))

    def echo(self, text: str) -> None:
        self.println(text)

    def blank(self) -> None:
        self.println("")

    # ------------------------------------------------------------------ #
    # spinner
    # ------------------------------------------------------------------ #
    def start_spinner(self, message: str = "") -> None:
        if not self.spinner_enabled or self.quiet:
            return
        with self._lock:
            if self._spinner is None:
                self._spinner = Spinner(self.style, message, stream=self.out,
                                        enabled=self.spinner_enabled)
            else:
                self._spinner.set_message(message)
            self._spinner.start(message)

    def update_spinner(self, message: str) -> None:
        with self._lock:
            if self._spinner is not None:
                self._spinner.set_message(message)

    def stop_spinner(self, final: str = "") -> None:
        with self._lock:
            if self._spinner is not None:
                self._spinner.stop(final)
                self._spinner = None

    # ------------------------------------------------------------------ #
    # agent-loop protocol
    # ------------------------------------------------------------------ #
    def on_turn_start(self, agent: str = "main", turn: int = 1, model: str = "") -> None:
        self._ensure_stream(agent)
        if self.live and turn > 1 and self._wrote_text.get(agent):
            # The model resumed after tool calls: start its continuation on a new
            # line instead of gluing it to the previous sentence.
            self.write("\n")
        if self.live:
            self.start_spinner(f"thinking ({model})" if model else "thinking")

    def on_turn_end(self, agent: str = "main", turn: int = 1, usage: Any = None,
                    finish_reason: str = "") -> None:
        self.stop_spinner()
        # The per-turn token line is verbose-only: `submit()` already prints a
        # richer summary, and duplicating it is noise.
        if self.show_usage and self.verbose and usage is not None and self.live:
            total = getattr(usage, "total_tokens", 0)
            if total:
                self._break_partial_line()   # never glue stats onto streamed prose
                self.write(self.style.dim(
                    f"  ⌁ turn {turn}: {format_number(getattr(usage, 'input_tokens', 0))}→"
                    f"{format_number(getattr(usage, 'output_tokens', 0))} tokens"
                    + (f" · {finish_reason}" if finish_reason not in ("stop", "") else "") + "\n"))

    def on_text(self, text: str, agent: str = "main") -> None:
        if self.quiet:
            return
        md = self._ensure_stream(agent)
        rendered = md.feed(text)
        if not rendered:
            return
        if self.live and (agent in ("main", "") or self._streaming_agent in (None, agent)):
            self.stop_spinner()
            self._streaming_agent = agent
            self._wrote_text[agent] = True
            self.write(rendered)
            self._partial_line = not rendered.endswith("\n")
        else:
            self._buffers.setdefault(agent, []).append(rendered)

    def on_reasoning(self, text: str, agent: str = "main") -> None:
        if not self.show_reasoning or self.quiet:
            return
        if self.live and agent in ("main", ""):
            self.stop_spinner()
            self.write(self.style.paint("dim", self.style.color.italic(
                "".join(ch for ch in text))))
        else:
            self._reasoning_buffers.setdefault(agent, []).append(text)

    def on_tool_call_stream(self, index: int, name: str, agent: str = "main") -> None:
        if self.live and name:
            self.update_spinner(f"calling {name}")

    def on_tool_start(self, name: str, args: Dict[str, Any], agent: str = "main") -> None:
        self._tool_counts[name] = self._tool_counts.get(name, 0) + 1
        if self.quiet:
            return
        summary = _summarise_args(name, args)
        prefix = self._agent_prefix(agent)
        line = f"{prefix}{self.style.paint('tool', '⚙')} {self.style.bold(name)}{self.style.dim(summary)}"
        if self.live:
            self.stop_spinner()
            self.println(line)
        else:
            self._buffers.setdefault(agent, []).append(line + "\n")

    def on_tool_end(self, name: str, result: Any, agent: str = "main") -> None:
        if self.quiet:
            return
        ok = not getattr(result, "is_error", False)
        content = str(getattr(result, "content", "") or "")
        elapsed = (getattr(result, "data", {}) or {}).get("elapsed_ms")
        detail = _summarise_result(name, content, ok)
        if elapsed:
            detail += self.style.dim(f" · {format_duration(elapsed)}")
        icon = self.style.paint("success" if ok else "error", "✓" if ok else "✗")
        prefix = self._agent_prefix(agent)
        line = f"{prefix}{icon} {self.style.dim(truncate(detail, max(30, self.style.width - 20)))}"
        if self.live:
            self.println(line)
        else:
            self._buffers.setdefault(agent, []).append(line + "\n")
            if not ok and content:
                for extra in content.split("\n")[:6]:
                    self._buffers[agent].append(self.style.dim("    " + truncate(extra, self.style.width - 8)) + "\n")

    def on_tool_denied(self, name: str, reason: str, agent: str = "main") -> None:
        line = (f"{self._agent_prefix(agent)}{self.style.paint('warning', '⊘')} "
                f"{self.style.paint('warning', name)} {self.style.dim('denied: ' + reason)}")
        self.println(line) if self.live else self._buffers.setdefault(agent, []).append(line + "\n")

    def on_context_budget(self, note: str, agent: str = "main") -> None:
        if self.quiet:
            return
        line = self.style.dim(f"⤵ context: {note}")
        self.println(line) if self.live else self._buffers.setdefault(agent, []).append(line + "\n")

    def on_error(self, message: str, agent: str = "main") -> None:
        first, _, rest = message.partition("\n")
        self.error(truncate(first, self.style.width - 4))
        if rest:
            self.println(self.style.dim(truncate(rest.split("\n")[0], self.style.width - 6)))

    # ------------------------------------------------------------------ #
    # swarm protocol
    # ------------------------------------------------------------------ #
    def set_live(self, live: bool) -> None:
        self.live = live

    def on_swarm_start(self, mode: str = "", objective: str = "", cast: Sequence[str] = ()) -> None:
        self.println(rule(self.style, f"SWARM · {mode}"))
        self.println(self.style.dim("objective: ") + truncate(objective, self.style.width - 12))
        if cast:
            self.println(self.style.dim("cast:      ") + ", ".join(cast))
        self.blank()

    def on_swarm_end(self, result: Any = None) -> None:
        if result is None:
            return
        runs = getattr(result, "runs", []) or []
        usage = getattr(result, "usage", None)
        stats = [
            ("agents", f"{len({r.agent for r in runs})} ({len(runs)} runs)"),
            ("rounds", str(getattr(result, "rounds", 0))),
            ("tasks", f"{getattr(result.board, 'percent_done', lambda: 0)():.0f}% done"
            if hasattr(result, "board") else ""),
            ("tokens", format_number(getattr(usage, "total_tokens", 0)) if usage else "0"),
            ("cost", format_cost(getattr(result, "cost", 0.0))),
            ("time", format_duration(getattr(result, "elapsed_ms", 0))),
        ]
        failed = [r for r in runs if not r.ok]
        self.blank()
        self.println(rule(self.style, "SWARM COMPLETE" + (f" · {len(failed)} failed" if failed else "")))
        self.println(kv([(k, v) for k, v in stats if v], self.style))
        if failed:
            for r in failed[:5]:
                self.println(self.style.paint("error", f"  ✗ {r.agent}: {truncate(r.error or 'failed', 90)}"))

    def on_agent_start(self, agent: str = "", persona: Any = None, task_id: str = "",
                       round_no: int = 0, prompt: str = "") -> None:
        label = _persona_label(persona, agent)
        line = (f"{self.style.paint('agent', '▶')} {label} "
                + (self.style.dim(f"[round {round_no}]") if round_no else "")
                + (self.style.dim(f" task {task_id}") if task_id else ""))
        self.println(line)
        if prompt:
            first = " ".join(prompt.split())[:110]
            self.println(self.style.dim(f"   ↳ {first}…"))
        self.start_spinner(f"{agent} working")

    def on_agent_end(self, agent: str = "", persona: Any = None, run: Any = None) -> None:
        self.stop_spinner()
        label = _persona_label(persona, agent)
        text = "".join(self._buffers.pop(agent, []))
        reasoning = "".join(self._reasoning_buffers.pop(agent, [])).strip()
        status = getattr(run, "report", {}).get("status", "") if run else ""
        ok = getattr(run, "ok", True) if run else True
        icon = self.style.paint("success" if ok else "error", "✓" if ok else "✗")
        head = f"{icon} {label}" + (self.style.dim(f" [{status}]") if status else "")
        usage = getattr(run, "usage", None) if run else None
        if usage is not None and getattr(usage, "total_tokens", 0):
            head += self.style.dim(f" · {format_number(usage.total_tokens)} tok"
                                   f" · {format_duration(getattr(run, 'latency_ms', 0))}")
        self.println(head)
        if reasoning and self.show_reasoning:
            self.println(self.style.dim("  " + truncate(reasoning.replace("\n", " "), self.style.width - 6)))
        body = strip_ansi(text).strip()
        if body:
            lines = [truncate(l, self.style.width - 6) for l in text.rstrip().split("\n")]
            shown = lines if len(lines) <= 40 else lines[:40] + [self.style.dim(f"  … {len(lines) - 40} more lines")]
            self.println(box(agent, [strip_ansi(s) for s in shown], self.style, role="border"))
        else:
            final = (getattr(run, "text", "") or "").strip()
            if final:
                self.println(render_markdown(truncate(final, 4000), self.style))
        if run is not None and getattr(run, "error", ""):
            self.println(self.style.paint("error", f"  error: {truncate(run.error, 120)}"))

    # ------------------------------------------------------------------ #
    # approval prompt (interactive)
    # ------------------------------------------------------------------ #
    def confirm(self, request: ConfirmationRequest, tool_name: str, args: Dict[str, Any]
                ) -> Tuple[bool, Optional[Tuple[str, str]]]:
        """Interactive approval as a clickable menu.

        Returns ``(allowed, remember)`` where remember is ``("allow", rule)`` or
        ``("deny", rule)`` so "always"/"never" persist. Falls back to a numbered
        text menu when stdin is not a terminal.
        """
        self.stop_spinner()
        role = RISK_ROLE.get(request.risk, "accent")
        lines = [self.style.paint(role, f"{tool_name}") + "  " + request.title]
        if request.detail:
            lines.append(self.style.dim(request.detail))
        if request.risk != "normal" and "risk=" not in (request.detail or ""):
            lines.append(self.style.paint(role, tr("approve.risk", risk=request.risk)))
        if request.diff:
            lines.append(diff_ui.render_diff(request.diff, self.style)[:1500])
        self.println(box(tr("approve.title"), lines, self.style, role=role))

        from .menu import Menu, MenuItem

        items = [
            MenuItem(tr("approve.yes"), "yes", hint=tr("approve.yes_hint")),
            MenuItem(tr("approve.always"), "always",
                     hint=tr("approve.always_hint", key=request.key or tool_name)),
            MenuItem(tr("approve.never"), "never",
                     hint=tr("approve.never_hint", key=request.key or tool_name)),
            MenuItem(tr("approve.no"), "no", hint=tr("approve.no_hint")),
        ]
        items = [i for i in items if i.enabled]
        if request.diff or args:
            items.insert(1, MenuItem(tr("approve.view"), "view", hint=tr("approve.view_hint")))
        menu = Menu(tr("approve.pick"), items, style=self.style, allow_filter=False,
                    prompt=f"{tool_name} · {tr('approve.risk', risk=request.risk)}",
                    stream=self.out, max_rows=6)
        menu.state.cursor = 0
        # Bounded on purpose: a "view details, then ask again" loop must never be
        # able to spin forever on a repeating answer or a misbehaving stdin.
        for _attempt in range(20):
            choice = menu.show()
            if choice is None or choice == "no":
                return False, None
            if choice == "yes":
                return True, None
            if choice == "always":
                return True, ("allow", request.key or f"{tool_name}:*")
            if choice == "never":
                return False, ("deny", request.key or f"{tool_name}:*")
            if choice == "view":
                self.println(box(tr("approve.detail_title"),
                                 [f"{k}: {truncate(str(v), 200)}" for k, v in (args or {}).items()]
                                 or ["(tidak ada argumen)"], self.style))
                if request.diff:
                    self.println(diff_ui.render_diff(request.diff, self.style))
                continue
            return False, None
        self.println(self.style.dim(tr("approve.too_many")))
        return False, None

    # ------------------------------------------------------------------ #
    def flush_agent(self, agent: str) -> None:
        text = "".join(self._buffers.pop(agent, []))
        if text.strip():
            self.println(text.rstrip())

    def _ensure_stream(self, agent: str) -> StreamingMarkdown:
        md = self._stream_md.get(agent)
        if md is None:
            md = StreamingMarkdown(self.style, width=self.style.width)
            self._stream_md[agent] = md
        return md

    def end_stream(self, agent: str = "main") -> None:
        """Flush any partial line for an agent's stream."""
        md = self._stream_md.pop(agent, None)
        if md is None:
            return
        tail = md.flush()
        if tail:
            if self.live and agent in ("main", ""):
                self.write(tail)
                self._partial_line = not tail.endswith("\n")
            else:
                self._buffers.setdefault(agent, []).append(tail)
        self._streaming_agent = None
        self._break_partial_line()

    def _agent_prefix(self, agent: str) -> str:
        if not agent or agent == "main" or self.live:
            return ""
        return self.style.paint("agent", f"[{agent}] ")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _persona_label(persona: Any, agent: str) -> str:
    if persona is None:
        return agent
    emoji = getattr(persona, "emoji", "") or ""
    name = getattr(persona, "name", agent)
    role = getattr(persona, "role", "")
    return f"{name} {emoji}".strip() + (f" · {role}" if role else "")


_SENSITIVE_KEYS = ("content", "code", "command", "old_text", "new_text", "text", "prompt", "query", "url", "path")


def _summarise_args(name: str, args: Dict[str, Any]) -> str:
    if not args:
        return ""
    parts: List[str] = []
    for key in _SENSITIVE_KEYS:
        if key in args and args[key] not in (None, "", [], {}, False):
            parts.append(f"{key}={_short(args[key])}")
    for key, value in args.items():
        if key in _SENSITIVE_KEYS or value in (None, "", [], {}, False):
            continue
        parts.append(f"{key}={_short(value)}")
        if len(parts) >= 4:
            break
    return "  " + " ".join(parts[:4]) if parts else ""


def _short(value: Any, limit: int = 60) -> str:
    if isinstance(value, (dict, list)):
        import json

        try:
            value = json.dumps(value, ensure_ascii=False)
        except (TypeError, ValueError):
            value = str(value)
    text = " ".join(str(value).split())
    return truncate(text, limit)


def _summarise_result(name: str, content: str, ok: bool) -> str:
    if not content:
        return "ok" if ok else "failed"
    first = content.strip().split("\n")[0]
    if name in ("read_file", "grep", "find_files", "list_dir") and ok:
        lines = content.count("\n") + 1
        return f"{lines} line(s): {truncate(first, 70)}"
    return truncate(first, 90)


__all__ = ["Renderer"]
