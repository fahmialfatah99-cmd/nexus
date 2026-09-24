"""The application layer: bootstrap, REPL and every slash command.

Layout of a running session::

    Settings -> Style -> Renderer -> Prompt
                    \\-> ToolRegistry (+plugins +MCP)
                    \\-> PermissionEngine -> CheckpointStore
                    \\-> ModelRouter -> Agent(Services)
                    \\-> Session (JSONL)
                    \\-> ProjectContext

The REPL is deliberately thin: parse a line, dispatch to a command handler or to
the agent, render. All state lives in the objects above, which is why the same
code path serves interactive, one-shot (``-p``), piped-stdin and swarm runs.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from . import __version__
from .agents.persona import MODE_DEFAULT_CAST, all_personas, get_persona, load_custom
from .agents.runtime import Agent, AgentOptions, Services
from .agents.swarm import MODES, SwarmConfig, SwarmRunner
from .core.checkpoints import CheckpointStore
from .core.config import Settings, config_sources, load_auth, load_settings, save_auth_key
from .core.context import ContextManager, count_tokens, summarise_history
from .core.errors import NexusError
from .core.ignore import human_size, is_within, walk_files
from .core.logging_ import configure as configure_logging
from .core.logging_ import get_logger
from .core.paths import data_dir, ensure_dirs, find_project_root, home, personas_dir, sessions_dir
from .core.permissions import MODES as APPROVAL_MODES
from .core.permissions import PermissionEngine, RuleSet
from .core.project import build_project_context
from .core.router import ModelRouter
from .core.session import SessionStore
from .core.taskboard import TaskBoard
from .core.usage import UsageLedger
from .providers.base import Message, RequestOptions
from .providers.registry import (MODEL_CATALOG, PROVIDERS, available_providers, model_info,
                                 resolve_model_id)
from .tools import build_registry
from .ui import diff as diff_ui
from .ui.i18n import autodetect, set_lang, tr
from .ui.markdown import render_markdown
from .ui.menu import Menu, MenuItem
from .ui.prompt import Prompt
from .ui.render import Renderer
from .ui.theme import Style, strip_ansi, truncate, visible_width
from .ui.widgets import box, columns, format_cost, format_duration, format_number, kv, rule

BANNER = r"""
 ███▄    █  ███████ ▐██▌  █    █  ██████
 ██ ▀█   █  ██       ████▌  █    █ ▀▀▀▀██
 ██  ▀█  █  █████     ██▌   █    █  ▄██▀
 ██   ▀█▀█  ██        ███▌  ▐█  █▌  ▀██▄
 ██    ▀██  ███████   ██▌    ▐██▀   █████
"""


@dataclass
class Command:
    name: str
    help: str
    category: str
    handler: Callable[..., Any]
    aliases: Tuple[str, ...] = ()
    args: str = ""
    dynamic: Optional[Callable[[], Sequence[str]]] = None
    hidden: bool = False

    @property
    def signature(self) -> str:
        return f"/{self.name}" + (f" {self.args}" if self.args else "")


class App:
    def __init__(self, *, settings: Settings, cwd: Optional[Path] = None, style: Optional[Style] = None,
                 renderer: Optional[Renderer] = None, quiet: bool = False,
                 session_id: str = "", resume: Optional[str] = None,
                 model_spec: str = "", offline: Optional[bool] = None,
                 approval_mode: str = "", extra_dirs: Sequence[str] = (),
                 transport: Any = None, enable_plugins: bool = True,
                 enable_mcp: bool = True) -> None:
        self.settings = settings
        # explicit ui.language wins, otherwise follow the user's locale
        set_lang(settings.ui.language or autodetect())
        self.cwd = Path(cwd) if cwd else Path.cwd()
        self.workspace_root = find_project_root(self.cwd)
        ensure_dirs()
        self.log = configure_logging_()
        self.style = style or Style.create(settings.ui.theme, enabled=None if settings.ui.color else False,
                                           width=settings.ui.width)
        self.quiet = quiet
        self.offline = bool(settings.offline if offline is None else offline)
        self.exit_code = 0
        self._cancelled = threading.Event()
        self._running = True

        self.renderer = renderer or Renderer(
            self.style, live=True, spinner_enabled=settings.ui.spinner and not quiet,
            show_usage=settings.ui.show_usage, show_reasoning=settings.ui.show_reasoning,
            quiet=quiet, verbose=bool(settings.verbose))

        # ---- tools ------------------------------------------------------
        self.registry = build_registry(offline=self.offline,
                                       disabled=list((settings.tools or {}).get("disabled") or []))
        self.plugin_tools: List[str] = []
        self.plugin_report: Any = None
        if enable_plugins:
            self._load_plugins()

        # ---- permissions & safety ---------------------------------------
        self.permissions = PermissionEngine(
            mode=approval_mode or settings.approval_mode,
            rules=RuleSet.from_dict(settings.permissions),
            workspace_root=self.workspace_root,
            extra_dirs=[self._resolve_dir(d) for d in list(settings.extra_dirs) + list(extra_dirs)],
            confirmer=None if quiet else self.renderer.confirm,
            allow_private_network=settings.allow_private_network,
            on_persist=self._persist_rule,
            log=self.log,
        )
        self.session_store = SessionStore(persist=settings.session.get("persist", True))
        self.session = self._open_session(session_id=session_id, resume=resume)
        self.checkpoints = CheckpointStore(self.workspace_root, self.session.meta.id,
                                          enabled=settings.session.get("checkpoints", True))

        # ---- model routing ----------------------------------------------
        self.router = ModelRouter(
            default_provider=settings.default_provider,
            default_model=model_spec or settings.default_model,
            provider_configs=settings.provider_configs(),
            failover=settings.failover,
            auth_store=load_auth(),
            transport=transport,
            timeout=300.0,
            max_retries=4,
            log=self.log,
            on_failover=self._on_failover,
        )
        if model_spec:
            settings.default_model = model_spec

        # ---- project awareness ------------------------------------------
        self.project_context = ""
        self.refresh_project_context()
        self.memory_text = self._load_memory()

        # ---- services + agent -------------------------------------------
        self.board = TaskBoard()
        self.services = Services(
            registry=self.registry, permissions=self.permissions, router=self.router,
            context=ContextManager(), ledger=UsageLedger(), checkpoints=self.checkpoints,
            ui=self.renderer, log=self.log, session=self.session, cwd=self.cwd,
            workspace_root=self.workspace_root, project_context=self.project_context,
            memory_text=self.memory_text, offline=self.offline, cancelled=self._cancelled,
            max_output_chars=int(settings.session.get("max_tool_output", 24_000)),
        )
        self.services.vars["config"] = settings
        self.services.vars["board"] = self.board
        self.services.vars["model_spec"] = settings.default_model
        self.services.vars["spawn_subagent"] = self.spawn_subagent
        self.services.spawn_subagent = self.spawn_subagent

        self.custom_personas = load_custom([personas_dir(), self.workspace_root / ".nexus" / "personas"])
        self.agent = Agent(services=self.services, persona=get_persona("main"),
                           options=self._agent_options())
        self.plan_mode = False
        self._interactive_override: Optional[bool] = None
        if os.environ.get("NEXUS_NO_MENU"):
            # --no-menu / NEXUS_NO_MENU=1: always use the numbered text list.
            # Set here (not earlier) so a later initialisation cannot undo it.
            self._interactive_override = False
        self.mcp_clients: List[Any] = []
        if enable_mcp:
            self._load_mcp()

        # One DEBUG line per start makes support requests answerable: it records
        # exactly which workspace, mode and model a session began with.
        self.log.debug("nexus start", version=__version__, cwd=str(self.cwd),
                       workspace=str(self.workspace_root), mode=self.permissions.mode,
                       model=settings.default_model, provider=settings.default_provider,
                       offline=self.offline, tools=len(self.registry.enabled_tools()),
                       session=self.session.meta.id)

        self.commands = self._build_commands()
        self.prompt = Prompt(
            self.style, history_file=home() / "history.txt",
            commands=[(c.name, c.help) for c in self.commands.values()],
            cwd=self.cwd,
            dynamic_completers={c.name: c.dynamic for c in self.commands.values() if c.dynamic},
        )

    # ------------------------------------------------------------------ #
    # bootstrap helpers
    # ------------------------------------------------------------------ #
    def _agent_options(self) -> AgentOptions:
        s = self.settings
        return AgentOptions(model_spec=s.default_model, temperature=s.temperature,
                            max_tokens=s.max_tokens, max_turns=s.max_turns, stream=s.stream and not self.quiet)

    def _resolve_dir(self, raw: str) -> Path:
        path = Path(raw).expanduser()
        return path if path.is_absolute() else (self.cwd / path).resolve()

    def _open_session(self, *, session_id: str = "", resume: Optional[str] = None):
        if resume:
            found = self.session_store.load(resume) if resume not in ("last", "latest") \
                else self.session_store.latest(cwd=self.cwd)
            if found is not None:
                if not self.quiet:
                    self.renderer.info(f"Resumed session {found.meta.id} "
                                       f"({found.meta.messages} messages, {found.meta.turns} turns)")
                return found
            if not self.quiet:
                self.renderer.warning(f"Session '{resume}' not found; starting a new one.")
        return self.session_store.create(cwd=self.cwd, model=self.settings.default_model,
                                        provider=self.settings.default_provider,
                                        approval_mode=self.settings.approval_mode,
                                        session_id=session_id)

    def refresh_project_context(self) -> None:
        try:
            ctx = build_project_context(self.workspace_root, settings=self.settings, log=self.log)
            self.project_context = ctx.text
        except Exception as exc:  # never fail startup because of context
            self.log.warning("project context failed", error=str(exc))
            self.project_context = ""

    def _load_memory(self) -> str:
        """Load MEMORY.md blocks. Called during bootstrap, before Services exists."""
        if not self.settings.memory.get("enabled", True):
            return ""
        try:
            from .tools.builtin.memory import load_memory_for_prompt

            return load_memory_for_prompt(self.workspace_root)
        except Exception as exc:
            self.log.warning("memory load failed", error=str(exc))
            return ""

    def _load_plugins(self) -> None:
        try:
            from .plugins.loader import load_plugins

            report = load_plugins(self.registry,
                                 dirs=[home() / "plugins", self.workspace_root / ".nexus" / "plugins"],
                                 log=self.log)
            self.plugin_report = report
            self.plugin_tools = list(report.tools)
            for error in report.errors:
                self.renderer.warning(f"plugin error: {error}")
            if report.names and not self.quiet:
                self.renderer.info(f"plugins: {', '.join(report.names)}"
                                   + (f" (+{len(report.tools)} tools)" if report.tools else ""))
        except Exception as exc:
            self.log.warning("plugin load failed", error=str(exc))

    def _load_mcp(self) -> None:
        servers = (self.settings.mcp or {}).get("servers") or {}
        if not servers:
            return
        try:
            from .mcp.client import MCPManager

            manager = MCPManager(log=self.log)
            added = manager.connect_all(servers, self.registry)
            self.mcp_clients = manager.clients
            if added and not self.quiet:
                self.renderer.info(f"mcp: {added}")
        except Exception as exc:
            self.log.warning("mcp load failed", error=str(exc))

    def _persist_rule(self, rule: str) -> None:
        """Persist an 'always allow' decision into the project config file."""
        try:
            path = self.workspace_root / ".nexus" / "config.json"
            data = {}
            if path.is_file():
                data = json.loads(path.read_text(encoding="utf-8") or "{}")
            perms = data.setdefault("permissions", {})
            allow = perms.setdefault("allow", [])
            if rule not in allow:
                allow.append(rule)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
            self.settings.permissions.setdefault("allow", []).append(rule)
        except (OSError, json.JSONDecodeError) as exc:
            self.log.warning("could not persist rule", rule=rule, error=str(exc))

    def _on_failover(self, failed, target, reason: str) -> None:
        if not self.quiet:
            self.renderer.warning(f"{failed.label} failed ({truncate(reason, 60)}) -> switching to {target.label}")

    # ------------------------------------------------------------------ #
    # REPL
    # ------------------------------------------------------------------ #
    def banner(self) -> None:
        if self.quiet:
            return
        target = self.router.resolve(self.settings.default_model)
        lines = [
            f"version {__version__} · python {sys.version.split()[0]} · zero dependencies",
            f"model    {self.style.paint('accent', target.label)}"
            + (f"  ({target.info.context_window:,} ctx)" if target.info.context_window else "  (context unknown)"),
            f"mode     {self.permissions.mode}   workspace {self.workspace_root}",
            f"session  {self.session.meta.id}   tools {len(self.registry.enabled_tools())}"
            + (f"   plugins {len(self.plugin_tools)}" if self.plugin_tools else ""),
        ]
        self.renderer.println(self.style.paint("accent", BANNER.rstrip("\n")))
        self.renderer.box("NEXUS", lines)
        self.renderer.println(self.style.dim(
            "  /help for commands · @file to attach · !cmd for shell · Ctrl+C cancels · /exit quits"))
        self.renderer.blank()

    def run(self) -> int:
        self.banner()
        interrupted = 0
        while self._running:
            try:
                prompt_str = self._prompt_string()
                text = self.prompt.read(prompt_str)
            except KeyboardInterrupt:
                self._cancelled.set()
                interrupted += 1
                self.renderer.println("")
                if interrupted >= 2:
                    self.renderer.dim("interrupted twice -- exiting. Use /exit for a clean shutdown.")
                    break
                self.renderer.warning("Cancelled. Press Ctrl+C again to exit.")
                continue
            except EOFError:
                break
            interrupted = 0
            if text is None:
                break
            text = text.strip()
            if not text:
                continue
            try:
                if text.startswith("/"):
                    keep_going = self.handle_command(text)
                    if not keep_going:
                        break
                elif text.startswith("!"):
                    self.run_shell(text[1:].strip())
                else:
                    self.submit(text)
            except NexusError as exc:
                self.renderer.error(str(exc), hint=exc.hint or "")
            except Exception as exc:  # a bug must be visible and logged, never silent
                self.log.exception("unhandled error in REPL", exc=exc)
                self.renderer.error(f"{type(exc).__name__}: {exc}")
                self.renderer.dim(f"details in {data_dir() / 'logs' / 'nexus.log'}")
            finally:
                self._cancelled.clear()
        self.shutdown()
        return self.exit_code

    def _prompt_string(self) -> str:
        mode = self.permissions.mode
        marker = {"yolo": "!", "full-auto": "»", "auto-edit": "›", "suggest": "?", "read-only": "r"}.get(mode, "›")
        label = self.style.paint("user", f"you{marker}")
        cost = self.services.ledger.totals()["cost_usd"]
        suffix = self.style.dim(f" [{format_cost(cost)}]") if cost else ""
        return f"{label}{suffix} "

    def shutdown(self) -> None:
        try:
            self.session.close()
        except Exception:
            pass
        try:
            self.prompt.save_history()
        except Exception:
            pass
        for client in self.mcp_clients:
            try:
                client.close()
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    # one-shot / piped
    # ------------------------------------------------------------------ #
    def run_once(self, text: str, *, swarm: bool = False, output_json: bool = False,
                 swarm_mode: str = "") -> int:
        try:
            if swarm or swarm_mode:
                result = self.run_swarm(text, mode=swarm_mode or self.settings.swarm.mode)
                payload = {"type": "swarm", "mode": result.mode, "final": result.final_text,
                           "runs": [{"agent": r.agent, "persona": r.persona_key, "ok": r.ok,
                                     "status": r.status, "summary": r.one_line(), "error": r.error}
                                    for r in result.runs],
                           "usage": result.usage.as_dict(), "cost": result.cost,
                           "elapsed_ms": result.elapsed_ms,
                           "tasks": result.board.to_dict()}
            else:
                turn = self.submit(text, return_result=True)
                payload = {"type": "turn", "text": turn.text, "ok": turn.ok, "errors": turn.errors,
                           "denied": turn.denied, "tool_calls": turn.tool_calls, "turns": turn.turns,
                           "usage": turn.usage.as_dict(), "cost": turn.cost,
                           "finish_reason": turn.finish_reason, "report": turn.report,
                           "model": turn.target.label if turn.target else "",
                           "elapsed_ms": turn.latency_ms}
            if output_json:
                print(json.dumps(payload, ensure_ascii=False, indent=2))
            self.exit_code = 0 if (payload.get("ok", True) and not payload.get("errors")) else 1
        except NexusError as exc:
            if output_json:
                print(json.dumps({"type": "error", "error": str(exc), "hint": exc.hint or ""},
                                 ensure_ascii=False))
            else:
                self.renderer.error(str(exc), hint=exc.hint or "")
            self.exit_code = 2
        except Exception as exc:
            self.log.exception("run_once failed", exc=exc)
            self.renderer.error(f"{type(exc).__name__}: {exc}")
            self.exit_code = 3
        finally:
            self.shutdown()
        return self.exit_code

    # ------------------------------------------------------------------ #
    # submitting work
    # ------------------------------------------------------------------ #
    def expand_mentions(self, text: str) -> Tuple[str, List[str]]:
        """Replace @path mentions with their (bounded) contents."""
        import re

        found: List[str] = []
        blocks: List[str] = []

        def repl(match: "re.Match[str]") -> str:
            raw = match.group(1)
            path = Path(raw).expanduser()
            if not path.is_absolute():
                path = self.cwd / raw
            if not path.exists():
                return match.group(0)
            try:
                rel = str(path.resolve().relative_to(self.workspace_root.resolve()))
            except ValueError:
                rel = str(path)
            found.append(rel)
            if path.is_dir():
                entries = []
                for i, p in enumerate(walk_files(path, max_files=60)):
                    if i >= 60:
                        entries.append("…")
                        break
                    entries.append(str(p.relative_to(path)))
                blocks.append(f"Directory listing of {rel}:\n" + "\n".join(entries))
                return f"@{rel}"
            from .core.ignore import is_probably_binary

            if is_probably_binary(path):
                blocks.append(f"{rel}: binary file ({human_size(path.stat().st_size)})")
                return f"@{rel}"
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                blocks.append(f"{rel}: unreadable ({exc})")
                return f"@{rel}"
            limit = 20_000
            body = content if len(content) <= limit else content[:limit] + "\n…(truncated)"
            blocks.append(f"File {rel}:\n```\n{body}\n```")
            return f"@{rel}"

        cleaned = re.sub(r"(?<![\w/])@([^\s@,;:!?'\"()\[\]]+)", repl, text)
        return cleaned, blocks

    def submit(self, text: str, *, return_result: bool = False, persona: str = "",
               images: Optional[Sequence[Any]] = None):
        cleaned, blocks = self.expand_mentions(text)
        if blocks:
            prefix = "\n\n".join(blocks)
            cleaned = f"{prefix}\n\n---\n\nUser request: {cleaned}"
        if not self.quiet:
            self.renderer.println(self.style.paint("user", "you› ")
                                  + self.style.dim(truncate(text, 90)))
        self.session.add_message(Message.user(text))
        self._cancelled.clear()
        agent = self.agent
        if persona:
            agent = Agent(services=self.services, persona=get_persona(persona), name=persona,
                          options=self._agent_options())
        if self.plan_mode:
            agent.inject("PLAN MODE: do not modify any file. Investigate and produce a step-by-step "
                         "plan with exact paths and commands. Ask for approval before implementing.")
        try:
            result = agent.send(cleaned, images=images)
        finally:
            self.renderer.end_stream(agent.name)
        self.session.add_messages([m for m in agent.history[-8:] if m.role in ("assistant",)])
        if result.aborted:
            self.renderer.warning("Interrupted.")
        for denied in result.denied[:3]:
            self.renderer.dim(f"  denied: {denied}")
        for err in result.errors[:3]:
            self.renderer.error(err)
        if not self.quiet and result.usage.total_tokens:
            totals = self.services.ledger.totals()
            self.renderer.println(self.style.dim(
                f"  ⌁ {result.turns} turn(s) · {result.tool_calls} tool(s) · "
                f"{format_number(result.usage.input_tokens)}→{format_number(result.usage.output_tokens)} tok · "
                f"{format_cost(result.cost)} this turn · {format_cost(totals['cost_usd'])} session · "
                f"{format_duration(result.latency_ms)}"))
        if persona:
            self.agent.history = agent.history  # keep a single conversation thread
        return result if return_result else None

    def run_shell(self, command: str) -> None:
        if not command:
            return
        try:
            proc = subprocess.run(command, shell=True, cwd=str(self.cwd), text=True,
                                  capture_output=True, timeout=120, errors="replace",
                                  env={**os.environ, "GIT_PAGER": "cat", "NO_COLOR": "1"})
        except subprocess.TimeoutExpired:
            self.renderer.warning("command timed out after 120s")
            return
        except OSError as exc:
            self.renderer.error(f"cannot run command: {exc}")
            return
        if proc.stdout.strip():
            self.renderer.println(truncate(proc.stdout.rstrip(), 20_000))
        if proc.stderr.strip():
            self.renderer.println(self.style.paint("error", truncate(proc.stderr.rstrip(), 4000)))
        self.renderer.dim(f"exit {proc.returncode}")

    def spawn_subagent(self, *, role: str = "general", prompt: str = "", context: str = "",
                       max_turns: int = 12, parent: str = "main") -> str:
        persona = self.custom_personas.get(role) or get_persona(role)
        agent = Agent(services=self.services, persona=persona, name=f"{persona.key}-sub",
                      options=AgentOptions(model_spec=self.settings.default_model,
                                           temperature=self.settings.temperature,
                                           max_turns=max_turns, stream=False),
                      system_extra=f"Requested by {parent}." + (f"\nContext: {context}" if context else ""))
        self.renderer.println(self.style.paint("agent", f"  ↳ sub-agent {persona.name} ({persona.key})"))
        result = agent.send(prompt)
        self.renderer.end_stream(agent.name)
        status = result.report.get("status", "done" if result.ok else "failed")
        self.renderer.println(self.style.dim(
            f"  ↳ {persona.name} finished [{status}] · {result.turns} turns · "
            f"{format_number(result.usage.total_tokens)} tok"))
        return result.text

    def run_swarm(self, objective: str, *, mode: str = "", cast: Optional[Sequence[str]] = None) -> Any:
        config = SwarmConfig(
            mode=mode or self.settings.swarm.mode,
            cast=list(cast or self.settings.swarm.cast),
            max_parallel=self.settings.swarm.max_parallel,
            max_rounds=self.settings.swarm.max_rounds,
            max_turns_per_agent=self.settings.swarm.max_turns_per_agent,
            debate_rounds=self.settings.swarm.debate_rounds,
            reviewer_gate=self.settings.swarm.reviewer_gate,
            model_specs=dict(self.settings.swarm.model_specs),
            default_model_spec=self.settings.default_model,
            temperature=self.settings.temperature,
            stream=not self.quiet,
        )
        runner = SwarmRunner(self.services, config, ui=self.renderer)
        was_live = self.renderer.live
        self.renderer.set_live(False)  # parallel agents must not interleave
        self.renderer.println(rule(self.style, f"swarm · {config.mode}"))
        self._cancelled.clear()
        try:
            result = runner.run(objective)
        finally:
            self.renderer.set_live(was_live)
        self.renderer.println("")
        self.renderer.markdown(result.final_text or "(no output)")
        self.session.note_event("swarm", mode=config.mode, runs=len(result.runs),
                                cost=result.cost, objective=objective[:200])
        return result

    # ------------------------------------------------------------------ #
    # command dispatch
    # ------------------------------------------------------------------ #
    def handle_command(self, line: str) -> bool:
        """Execute a slash command. Returns False when the REPL should exit."""
        try:
            parts = shlex.split(line)
        except ValueError:
            parts = line.split()
        if not parts:
            return True
        if line.strip() == "/":
            self.cmd_menu([], "")
            return True
        name = parts[0].lstrip("/").lower()
        args = parts[1:]
        raw_args = line.split(None, 1)[1] if " " in line else ""
        command = self.commands.get(name)
        if command is None:
            for candidate in self.commands.values():
                if name in candidate.aliases:
                    command = candidate
                    break
        if command is None:
            close = _closest(name, list(self.commands))
            self.renderer.error(f"Unknown command /{name}" + (f". Did you mean /{close}?" if close else ""))
            self.renderer.dim("/help lists every command.")
            return True
        self.log.info("command", name=name, args=len(args))
        outcome = command.handler(args, raw_args)
        return False if outcome is EXIT else True

    def _build_commands(self) -> Dict[str, Command]:
        cmds: List[Command] = [
            # ---- conversation ----
            Command("help", "Show this menu (or /help <topic>)", "conversation", self.cmd_help, ("?", "h"), "[topic]"),
            Command("menu", "Buka menu perintah interaktif (bisa diklik)", "conversation", self.cmd_menu,
                    ("m",), "[kategori]"),
            Command("clear", "Clear conversation history (keeps files & memory)", "conversation", self.cmd_clear, ("reset",)),
            Command("compact", "Summarise history to free context", "conversation", self.cmd_compact, (), "[instructions]"),
            Command("undo", "Revert file changes made by tools", "conversation", self.cmd_undo, (), "[checkpoint]"),
            Command("diff", "Show file changes made in this session", "conversation", self.cmd_diff),
            Command("rewind", "Rewind conversation to an earlier turn", "conversation", self.cmd_rewind, (), "[n]"),
            Command("again", "Re-run the last request", "conversation", self.cmd_again),
            Command("export", "Export the transcript as markdown", "conversation", self.cmd_export, (), "[path]"),
            Command("exit", "Quit NEXUS", "conversation", self.cmd_exit, ("quit", "q")),
            # ---- model ----
            Command("model", "Show or switch model", "model", self.cmd_model, ("m",), "[provider:]model",
                    dynamic=lambda: sorted({f"{p}:{m}" for p, m in _known_specs()})),
            Command("models", "List models for a provider", "model", self.cmd_models, (), "[provider] [--refresh]"),
            Command("provider", "Switch provider", "model", self.cmd_provider, (), "<provider>",
                    dynamic=lambda: sorted(PROVIDERS)),
            Command("providers", "List providers and their status", "model", self.cmd_providers),
            Command("temperature", "Set sampling temperature", "model", self.cmd_temperature, ("temp",), "<0.0-2.0>"),
            Command("reasoning", "Set reasoning effort", "model", self.cmd_reasoning, (), "<off|low|medium|high>",
                    dynamic=lambda: ["off", "low", "medium", "high"]),
            Command("failover", "Show or set the provider failover chain", "model", self.cmd_failover, (), "[specs]"),
            # ---- modes & tools ----
            Command("mode", "Show or set approval mode", "control", self.cmd_mode, (),
                    "|".join(APPROVAL_MODES), dynamic=lambda: list(APPROVAL_MODES)),
            Command("plan", "Toggle plan mode (no file changes)", "control", self.cmd_plan),
            Command("tools", "List tools (atau 'pick' untuk saklar klik)", "control", self.cmd_tools, (), "[pick]"),
            Command("tool", "Enable/disable a tool", "control", self.cmd_tool, (), "<on|off> <name>",
                    dynamic=lambda: self.registry.names()),
            Command("allow", "Add an allow rule", "control", self.cmd_allow, (), "<tool:pattern>"),
            Command("deny", "Add a deny rule", "control", self.cmd_deny, (), "<tool:pattern>"),
            Command("rules", "Show permission rules and audit trail", "control", self.cmd_rules),
            Command("readonly", "Toggle read-only mode", "control", self.cmd_readonly),
            # ---- context ----
            Command("context", "Show what is in context and its size", "context", self.cmd_context),
            Command("add", "Attach a file or directory to the context", "context", self.cmd_add, (), "<path>"),
            Command("memory", "View persistent memory", "context", self.cmd_memory),
            Command("remember", "Save a fact to project memory", "context", self.cmd_remember, (), "<text>"),
            Command("project", "Rebuild repository context", "context", self.cmd_project),
            Command("usage", "Token and cost usage", "context", self.cmd_usage),
            Command("status", "Full session status", "context", self.cmd_status, ("st",)),
            # ---- sessions ----
            Command("sessions", "List saved sessions", "session", self.cmd_sessions),
            Command("resume", "Resume another session", "session", self.cmd_resume, (), "[id]",
                    dynamic=lambda: [m.id for m in self.session_store.list(limit=20)]),
            # ---- swarm ----
            Command("swarm", "Run the swarm on an objective", "swarm", self.cmd_swarm, (), "<objective>"),
            Command("swarm-mode", "Show or set the swarm mode", "swarm", self.cmd_swarm_mode, (),
                    "|".join(MODES), dynamic=lambda: list(MODES)),
            Command("cast", "Show or set the swarm cast", "swarm", self.cmd_cast, (), "[persona...]"),
            Command("agents", "List available personas", "swarm", self.cmd_agents),
            Command("agent", "Run a single persona once", "swarm", self.cmd_agent, (), "<persona> <prompt>",
                    dynamic=lambda: sorted(self.custom_personas) + [p.key for p in all_personas()]),
            Command("debate", "Run a structured debate", "swarm", self.cmd_debate, (), "<question>"),
            Command("board", "Show the task board", "swarm", self.cmd_board),
            # ---- system ----
            Command("config", "Read or write configuration", "system", self.cmd_config, (), "[key] [value]"),
            Command("auth", "Store an API key", "system", self.cmd_auth, (), "<provider> [key]",
                    dynamic=lambda: sorted(PROVIDERS)),
            Command("doctor", "Diagnose the installation", "system", self.cmd_doctor),
            Command("selftest", "Run the built-in test suite", "system", self.cmd_selftest),
            Command("mcp", "List MCP servers and their tools", "system", self.cmd_mcp),
            Command("keys", "Show keyboard shortcuts", "system", self.cmd_keys),
            Command("log", "Show the log file location / tail it", "system", self.cmd_log, (), "[n]"),
            Command("about", "Version and credits", "system", self.cmd_about),
        ]
        return {c.name: c for c in cmds}

    # ------------------------------------------------------------------ #
    # interactive menus
    # ------------------------------------------------------------------ #
    @property
    def interactive(self) -> bool:
        """True when clickable menus should be offered.

        Overridable (``_interactive_override``) so the menu wiring can be tested
        without a terminal; ``NEXUS_FORCE_MENU=1`` does the same for debugging.
        """
        if self._interactive_override is not None:
            return self._interactive_override
        if os.environ.get("NEXUS_FORCE_MENU"):
            return True
        try:
            return bool(sys.stdin.isatty() and sys.stdout.isatty())
        except (OSError, ValueError, AttributeError):
            return False

    def _pick(self, title: str, items: Sequence[Any], *, multi: bool = False, prompt: str = "",
              allow_filter: bool = True, max_rows: int = 12) -> Any:
        """Show a clickable menu. Returns None (or []) when cancelled.

        Falls back to a numbered text menu automatically when stdin is not a
        terminal, so scripted sessions keep working.
        """
        if not items:
            self.renderer.info(tr("info.no_options"))
            return [] if multi else None
        self.renderer.stop_spinner()
        menu = Menu(title, items, style=self.style, multi=multi, prompt=prompt,
                    allow_filter=allow_filter, stream=sys.stdout, max_rows=max_rows)
        return menu.show()

    def _ask_arg(self, command: "Command") -> str:
        """Prompt for a command's argument after it was chosen from a menu."""
        if not command.args:
            return ""
        try:
            value = input(self.style.paint("accent", f"  argumen {command.signature}: ")).strip()
        except (EOFError, KeyboardInterrupt):
            self.renderer.println("")
            return ""
        return value

    def _menu_commands(self, category: str = "") -> None:
        """Command picker: click a command, optionally type its argument, run it."""
        commands = [c for c in self.commands.values() if not c.hidden
                    and (not category or c.category == category)]
        items = [MenuItem(f"/{c.name}" + (f" {c.args}" if c.args else ""), value=c.name,
                          hint=c.help, group=c.category)
                 for c in sorted(commands, key=lambda c: (c.category, c.name))]
        chosen = self._pick(tr("menu.commands") + (f" · {category}" if category else ""), items,
                            prompt=tr("menu.filter_hint"))
        if not chosen:
            return
        command = self.commands.get(chosen)
        if command is None:
            return
        argument = self._ask_arg(command)
        line = f"/{chosen}" + (f" {argument}" if argument else "")
        self.renderer.println(self.style.paint("user", "you› ") + self.style.dim(line))
        self.handle_command(line)

    # ------------------------------------------------------------------ #
    # command implementations
    # ------------------------------------------------------------------ #
    def cmd_menu(self, args: Sequence[str], raw: str) -> Any:
        """Interactive, clickable command browser."""
        categories = sorted({c.category for c in self.commands.values() if not c.hidden})
        if args:
            wanted = args[0].lower()
            if wanted in categories:
                return self._menu_commands(wanted)
            self.renderer.warning(tr("warn.unknown_category", name=wanted)
                                  + f": {', '.join(categories)}")
            return
        items = [MenuItem(f"{c.title()}", value=c,
                          hint=f"{sum(1 for x in self.commands.values() if x.category == c)} perintah")
                 for c in categories]
        items.insert(0, MenuItem(tr("menu.all_commands"), value="", hint=f"{len(self.commands)}"))
        chosen = self._pick("NEXUS", items, allow_filter=False, prompt=tr("menu.pick_hint"))
        if chosen is None:
            return
        self._menu_commands(chosen)

    def cmd_help(self, args: Sequence[str], raw: str) -> Any:
        topic = args[0].lower() if args else ""
        if topic == "swarm":
            self.renderer.markdown(_SWARM_HELP)
            return
        if topic in ("keys", "keybindings"):
            return self.cmd_keys(args, raw)
        if topic in ("tools",):
            return self.cmd_tools(args, raw)
        if topic in ("models", "providers"):
            return self.cmd_providers(args, raw)
        if topic in ("permissions", "mode", "modes"):
            self.renderer.markdown(_MODES_HELP)
            return
        categories: Dict[str, List[Command]] = {}
        for command in self.commands.values():
            if command.hidden:
                continue
            categories.setdefault(command.category, []).append(command)
        self.renderer.println(rule(self.style, "COMMANDS"))
        for category in ("conversation", "model", "control", "context", "session", "swarm", "system"):
            items = categories.get(category)
            if not items:
                continue
            self.renderer.println(self.style.paint("accent2", f"\n{category.upper()}"))
            width = max(visible_width(c.signature) for c in items)
            for command in items:
                aliases = f" ({', '.join('/' + a for a in command.aliases)})" if command.aliases else ""
                self.renderer.println("  " + self.style.bold(command.signature.ljust(width))
                                      + "  " + self.style.dim(command.help + aliases))
        self.renderer.println(self.style.dim(
            "\nInput that is not a command goes to the agent. Prefix with ! to run a shell command, "
            "use @path to attach a file. /help swarm for the swarm guide."))

    def cmd_clear(self, args, raw) -> Any:
        self.agent.reset()
        self.board.replace_all([])
        self.renderer.success("History cleared. Files, memory and checkpoints are untouched.")

    def cmd_compact(self, args, raw) -> Any:
        history = self.agent.history
        if len(history) < 4:
            self.renderer.info("Nothing to compact yet.")
            return
        target = self.router.resolve(self.settings.default_model)
        manager = self.services.context
        to_summarise, keep = manager.split_for_compaction(history, self.settings.compaction.keep_recent)
        if not to_summarise:
            self.renderer.info("History is already short.")
            return
        before = count_tokens(history, target.model)
        self.renderer.start_spinner("compacting history…")
        try:
            summary = summarise_history(self.router.provider(target.provider_key), to_summarise,
                                        model=target.model, max_tokens=self.settings.max_tokens or 1500)
        finally:
            self.renderer.stop_spinner()
        if not summary:
            self.renderer.error("Compaction failed (provider returned nothing). History unchanged.")
            return
        instruction = " ".join(args).strip()
        if instruction:
            summary += f"\n\n## Extra instructions from the user\n{instruction}"
        new_history = [Message.system(f"[Compacted conversation summary]\n{summary}")] + list(keep)
        self.agent.history = new_history
        self.session.note_summary(summary, len(to_summarise))
        after = count_tokens(new_history, target.model)
        self.renderer.success(f"Compacted {len(to_summarise)} messages into a summary: "
                              f"{format_number(before)} → {format_number(after)} est. tokens "
                              f"({100 - (after * 100 // max(1, before))}% saved).")

    def cmd_undo(self, args, raw) -> Any:
        seq = None
        if args and args[0].isdigit():
            seq = int(args[0])
        checkpoints = self.checkpoints.list()
        if not checkpoints:
            self.renderer.info("No checkpoints in this session (no files were modified yet).")
            return
        result = self.checkpoints.restore(seq)
        if result.restored:
            self.renderer.success(f"Restored {len(result.restored)} file(s): "
                                  + ", ".join(result.restored[:6]))
        if result.deleted:
            self.renderer.success(f"Removed {len(result.deleted)} created file(s): "
                                  + ", ".join(result.deleted[:6]))
        for err in result.errors:
            self.renderer.error(err)
        if result.skipped:
            self.renderer.warning(f"Skipped (too large to snapshot): {', '.join(result.skipped[:5])}")
        if not result.restored and not result.deleted:
            self.renderer.info("Nothing to undo.")

    def cmd_diff(self, args, raw) -> Any:
        touched = [p for p in self.session.meta.touched if Path(p).exists()]
        if not touched:
            self.renderer.info("No files modified in this session.")
            return
        if shutil.which("git") and (self.workspace_root / ".git").exists():
            self.run_shell("git --no-pager diff --stat -- " + " ".join(shlex.quote(p) for p in touched[:40]))
            self.run_shell("git --no-pager diff -- " + " ".join(shlex.quote(p) for p in touched[:12]))
            return
        checkpoints = list(reversed(self.checkpoints.list()))
        for path in touched[:12]:
            p = Path(path)
            rel = str(p.relative_to(self.workspace_root)) if is_within(self.workspace_root, p) else p.name
            original = None
            for cp in checkpoints:
                for entry in cp.files:
                    if entry.rel != rel or not entry.existed:
                        continue
                    stored = cp.dir / "files" / entry.rel
                    if stored.is_file():
                        try:
                            original = stored.read_text(encoding="utf-8", errors="replace")
                        except OSError:
                            original = None
                    break
                if original is not None:
                    break
            if original is None:
                self.renderer.println(self.style.dim(f"{path} (no checkpoint to diff against)"))
                continue
            current = p.read_text(encoding="utf-8", errors="replace")
            self.renderer.println(self.style.bold(path))
            self.renderer.println(diff_ui.render_diff(diff_ui.unified(original, current, path), self.style))

    def cmd_rewind(self, args, raw) -> Any:
        n = int(args[0]) if args and args[0].isdigit() else 1
        history = self.agent.history
        if len(history) <= 1:
            self.renderer.info("Nothing to rewind.")
            return
        cut = max(0, len(history) - n)
        while cut > 0 and history[cut].role == "tool":
            cut -= 1
        removed = len(history) - cut
        self.agent.history = history[:cut]
        self.renderer.success(f"Rewound {removed} message(s). History now has {cut}.")

    def cmd_again(self, args, raw) -> Any:
        last_user = None
        for msg in reversed(self.agent.history):
            if msg.role == "user":
                last_user = msg.text
                break
        if not last_user:
            self.renderer.info("No previous request to repeat.")
            return
        # drop the last exchange so the repeat is clean
        self.cmd_rewind(["2"], "")
        self.submit(last_user)

    def cmd_export(self, args, raw) -> Any:
        path = Path(args[0]).expanduser() if args else self.cwd / f"nexus-{self.session.meta.id}.md"
        text = self.session_store.export_markdown(self.session.meta.id)
        if not text:
            self.renderer.error("Nothing to export.")
            return
        try:
            path.write_text(text, encoding="utf-8")
        except OSError as exc:
            self.renderer.error(f"Cannot write {path}: {exc}")
            return
        self.renderer.success(f"Transcript exported to {path} ({human_size(path.stat().st_size)})")

    def cmd_exit(self, args, raw) -> Any:
        totals = self.services.ledger.totals()
        if totals["requests"] and not self.quiet:
            self.renderer.dim(f"session: {totals['requests']} request(s), "
                              f"{format_number(totals['total_tokens'])} tokens, "
                              f"{format_cost(totals['cost_usd'])}")
        return EXIT

    def _model_items(self) -> List[Any]:
        """Model picker entries: providers you can actually use come first."""
        seen: set = set()
        out: List[Any] = []
        order: List[str] = []
        if self.settings.default_provider:
            order.append(self.settings.default_provider)
        order += [p.key for p in available_providers(include_local=True) if p.key not in order]
        order += [k for k in sorted(PROVIDERS) if k not in order]
        current = f"{self.settings.default_provider}:{self.settings.default_model}" \
            if self.settings.default_provider else self.settings.default_model
        if current and current not in seen:
            seen.add(current)
            info = model_info(self.settings.default_model, self.settings.default_provider)
            out.append(MenuItem(current, value=current,
                                hint=tr("status.ready") if self._has_key(self.settings.default_provider or "")
                                else tr("status.nokey"), group="current"))
        for key in order[:14]:
            for model in sorted(m.id for m in MODEL_CATALOG.values() if m.provider == key):
                spec = f"{key}:{model}"
                if spec in seen:
                    continue
                seen.add(spec)
                info = model_info(model, key)
                hint = f"{info.context_window // 1000}k ctx" if info.context_window else "ctx ?"
                out.append(MenuItem(spec, value=spec, hint=hint, group=key))
        return out

    def cmd_model(self, args, raw) -> Any:
        if not args and self.interactive:
            chosen = self._pick(tr("model.pick"), self._model_items(),
                                prompt=tr("model.filter_hint"))
            if chosen:
                args = [str(chosen)]
        if not args:
            target = self.router.resolve(self.settings.default_model)
            info = target.info
            self.renderer.kv([
                ("model", target.label),
                ("context", f"{info.context_window:,}" if info.context_window else "unknown"),
                ("max output", f"{info.max_output_tokens:,}" if info.max_output_tokens else "unknown"),
                ("capabilities", ", ".join(filter(None, [
                    "tools" if info.supports_tools else "", "images" if info.supports_images else "",
                    "reasoning" if info.supports_reasoning else "", "streaming" if info.supports_streaming else ""]))
                or "none reported"),
                ("temperature", str(self.settings.temperature if self.settings.temperature is not None
                                    else f"persona default ({self.agent.persona.temperature})")),
            ])
            self.renderer.dim("Switch with /model <provider>:<model>, e.g. /model anthropic:sonnet")
            return
        spec = args[0]
        try:
            target = self.router.resolve(spec)
        except NexusError as exc:
            # An unresolvable spec is a hard error: nothing sensible to switch to.
            self.renderer.error(str(exc), hint=exc.hint or "")
            return
        credential_warning = ""
        try:
            self.router.provider(target.provider_key)
        except NexusError as exc:
            # Missing credentials should not block the switch -- the user may be
            # setting things up, and a clear warning beats a silent refusal.
            credential_warning = str(exc)
        except Exception as exc:
            credential_warning = f"{type(exc).__name__}: {exc}"
        self.settings.default_model = target.model
        self.settings.default_provider = target.provider_key
        self.agent.options.model_spec = target.model
        self.agent._target = None
        self.services.vars["model_spec"] = target.model
        self.renderer.success(f"Model set to {self.style.paint('accent', target.label)}"
                              + (f" ({target.info.context_window:,} ctx)" if target.info.context_window else ""))
        if credential_warning:
            self.renderer.warning(f"Provider '{target.provider_key}' is not usable yet: {credential_warning}")
            self.renderer.dim(f"Fix with: nexus auth login {target.provider_key}   (or /auth {target.provider_key})")
        if not target.info.context_window:
            self.renderer.dim("Context window unknown for this model: budgeting is disabled for it. "
                              "Add it to config providers.<key>.models or use /models --refresh.")

    def cmd_models(self, args, raw) -> Any:
        refresh = "--refresh" in args or "-r" in args
        provider_key = next((a for a in args if not a.startswith("-")), "") or self.settings.default_provider
        if not provider_key:
            provider_key = self.router.resolve().provider_key
        try:
            models = self.router.list_models(provider_key, refresh=refresh)
        except NexusError as exc:
            self.renderer.error(str(exc), hint=exc.hint or "")
            return
        if not models:
            self.renderer.warning(f"No models returned for '{provider_key}'.")
            self.renderer.dim("Check the API key (/auth) and base URL (/config providers."
                              f"{provider_key}.base_url).")
            return
        self.renderer.println(rule(self.style, f"{provider_key} · {len(models)} models"))
        self.renderer.println(columns(models[:120], self.style, width=self.style.width, min_col=28))
        if len(models) > 120:
            self.renderer.dim(f"… and {len(models) - 120} more")

    def cmd_provider(self, args, raw) -> Any:
        if not args and self.interactive:
            items = []
            for key in sorted(PROVIDERS):
                spec = PROVIDERS[key]
                status = tr("status.ready" if self._has_key(key)
                            else ("status.local" if spec.local else "status.nokey"))
                items.append(MenuItem(key, value=key,
                                      hint=f"{spec.display_name} · {status} · {spec.default_model}"))
            chosen = self._pick(tr("provider.pick"), items, prompt=tr("menu.filter_hint"))
            if chosen:
                args = [str(chosen)]
        if not args:
            return self.cmd_providers(args, raw)
        key = args[0].lower()
        if key not in PROVIDERS:
            self.renderer.error(f"Unknown provider '{key}'.", hint=f"Known: {', '.join(sorted(PROVIDERS))}")
            return
        spec = PROVIDERS[key]
        self.settings.default_provider = key
        if not self.settings.default_model or self.router.resolve().provider_key != key:
            self.settings.default_model = spec.default_model
        self.agent.options.model_spec = self.settings.default_model
        self.agent._target = None
        self.router.drop_cached(key)
        self.renderer.success(f"Provider set to {spec.display_name}; default model {spec.default_model}")
        if spec.requires_api_key and not self._has_key(key):
            self.renderer.warning(f"No API key found for {key}. Run: /auth {key} <key>")

    def _has_key(self, key: str) -> bool:
        spec = PROVIDERS.get(key)
        if not spec:
            return False
        if not spec.requires_api_key:
            return True
        if any(os.environ.get(e) for e in spec.env_keys):
            return True
        return bool(load_auth().get(key)) or bool((self.settings.providers.get(key).api_key
                                                   if key in self.settings.providers else ""))

    def cmd_providers(self, args, raw) -> Any:
        rows = []
        for key in sorted(PROVIDERS):
            spec = PROVIDERS[key]
            ready = self._has_key(key)
            rows.append([key, spec.display_name, "ready" if ready else ("local" if spec.local else "no key"),
                         spec.default_model or "-", spec.default_base_url or "-"])
        self.renderer.table(["key", "name", "status", "default model", "base url"], rows)
        self.renderer.dim(f"\nSet a key with: /auth <provider> <key>   (or export "
                          f"{PROVIDERS['openai'].env_keys[0]} / ANTHROPIC_API_KEY / GEMINI_API_KEY …)")

    def cmd_temperature(self, args, raw) -> Any:
        if not args:
            current = self.settings.temperature
            self.renderer.info(f"temperature = {current if current is not None else 'persona default'}")
            return
        try:
            value = float(args[0])
        except ValueError:
            self.renderer.error("Usage: /temperature <0.0-2.0>")
            return
        if not 0 <= value <= 2:
            self.renderer.error("temperature must be between 0.0 and 2.0")
            return
        self.settings.temperature = value
        self.agent.options.temperature = value
        self.renderer.success(f"temperature = {value}")

    def cmd_reasoning(self, args, raw) -> Any:
        choices = ("off", "low", "medium", "high")
        if not args:
            self.renderer.info(f"reasoning effort = {self.agent.options.reasoning_effort or 'model default'}")
            return
        value = args[0].lower()
        if value not in choices:
            self.renderer.error(f"Usage: /reasoning <{'|'.join(choices)}>")
            return
        self.agent.options.reasoning_effort = None if value == "off" else value
        self.renderer.success(f"reasoning effort = {value}")

    def cmd_failover(self, args, raw) -> Any:
        if not args:
            chain = self.router.failover_chain(self.router.resolve())
            self.renderer.println(rule(self.style, "failover chain"))
            for i, target in enumerate(chain, 1):
                self.renderer.println(f"  {i}. {target.label}")
            if len(chain) == 1:
                self.renderer.dim("No failover configured. Example: /failover anthropic:sonnet groq:llama-3.3-70b-versatile")
            return
        specs = list(args)
        self.settings.failover = specs
        self.router.failover = specs
        self.renderer.success("failover: " + " → ".join(specs))

    def cmd_mode(self, args, raw) -> Any:
        if not args and self.interactive:
            describe = {m: tr(f"mode.{m}") for m in APPROVAL_MODES}
            items = [MenuItem(m, value=m, hint=describe.get(m, ""),
                              group="aktif" if m == self.permissions.mode else "")
                     for m in APPROVAL_MODES]
            chosen = self._pick(tr("mode.pick"), items, allow_filter=False,
                                prompt=tr("mode.current", mode=self.permissions.mode))
            if chosen:
                args = [str(chosen)]
        if not args:
            self.renderer.markdown(_MODES_HELP)
            self.renderer.println(self.style.dim(f"current: {self.style.paint('accent', self.permissions.mode)}"))
            return
        value = args[0].lower()
        if value not in APPROVAL_MODES:
            self.renderer.error(f"Unknown mode '{value}'.", hint="Valid: " + ", ".join(APPROVAL_MODES))
            return
        self.permissions.set_mode(value)
        self.settings.approval_mode = value
        self.renderer.success(f"Approval mode: {value}")
        if value == "yolo":
            self.renderer.warning("yolo approves every operation without asking. Use only in disposable environments.")

    def cmd_plan(self, args, raw) -> Any:
        self.plan_mode = not self.plan_mode
        if self.plan_mode:
            self.renderer.success("Plan mode ON: the agent will investigate and propose, not modify.")
        else:
            self.renderer.success("Plan mode OFF.")

    def _menu_tools(self) -> Any:
        """Clickable tool switch board."""
        enabled = {t.name for t in self.registry.enabled_tools()}
        items = [MenuItem(name, value=name,
                          hint=self.registry.get(name).category,
                          group="on" if name in enabled else "off")
                 for name in self.registry.names()]
        chosen = self._pick(tr("tools.pick"), items, multi=True, prompt=tr("menu.multi_hint"))
        if chosen is None:
            return
        keep_off = {str(c) for c in chosen}
        for name in self.registry.names():
            self.registry.set_enabled(name, name not in keep_off)
        self.renderer.success(tr("tools.applied", off=len(keep_off),
                                 on=len(self.registry.enabled_tools())))

    def cmd_tools(self, args, raw) -> Any:
        if args and str(args[0]).lower() == "pick":
            return self._menu_tools()
        by_cat = self.registry.by_category()
        rows = []
        for category, tools in by_cat.items():
            for tool in tools:
                rows.append([tool.name, category,
                             "read" if tool.read_only else "write",
                             "net" if tool.needs_network else "-",
                             truncate(" ".join(tool.description.split())[:70], 70)])
        self.renderer.table(["tool", "category", "effect", "flags", "purpose"], rows)
        disabled = sorted(set(self.registry.names()) - {t.name for t in self.registry.enabled_tools()})
        if disabled:
            self.renderer.dim(f"\ndisabled: {', '.join(disabled)}")

    def cmd_tool(self, args, raw) -> Any:
        if len(args) < 2 or args[0].lower() not in ("on", "off", "enable", "disable"):
            self.renderer.error("Usage: /tool <on|off> <name>")
            return
        enable = args[0].lower() in ("on", "enable")
        name = args[1]
        if not self.registry.has(name):
            self.renderer.error(f"No such tool '{name}'.", hint=", ".join(self.registry.names()))
            return
        self.registry.set_enabled(name, enable)
        self.renderer.success(f"{name} {'enabled' if enable else 'disabled'}")

    def cmd_allow(self, args, raw) -> Any:
        if not args:
            return self.cmd_rules(args, raw)
        rule = args[0]
        self.permissions.add_rule("allow", rule)
        self.renderer.success(f"allow += {rule}")

    def cmd_deny(self, args, raw) -> Any:
        if not args:
            return self.cmd_rules(args, raw)
        rule = args[0]
        self.permissions.add_rule("deny", rule)
        self.renderer.success(f"deny += {rule}")

    def cmd_rules(self, args, raw) -> Any:
        rules = self.permissions.rules
        self.renderer.println(rule_(self.style, "permission rules"))
        for kind in ("deny", "ask", "allow"):
            items = getattr(rules, kind)
            self.renderer.println(f"  {self.style.paint('accent', kind):<16} "
                                  + (", ".join(items) if items else self.style.dim("(none)")))
        self.renderer.println(self.style.dim(f"\n{self.permissions.describe()}"))
        audit = self.permissions.audit(8)
        if audit:
            self.renderer.println(rule_(self.style, "recent decisions"))
            for entry in audit:
                icon = self.style.paint("success" if entry["allowed"] else "error",
                                        "✓" if entry["allowed"] else "✗")
                self.renderer.println(f"  {icon} {entry['tool']:<12} {truncate(entry['key'], 34):<34} "
                                      f"{self.style.dim(entry['source'] + ': ' + truncate(entry['reason'], 40))}")

    def cmd_readonly(self, args, raw) -> Any:
        if self.permissions.mode == "read-only":
            self.permissions.set_mode("auto-edit")
            self.settings.approval_mode = "auto-edit"
            self.renderer.success("Read-only OFF (mode: auto-edit)")
        else:
            self.permissions.set_mode("read-only")
            self.settings.approval_mode = "read-only"
            self.renderer.success("Read-only ON: every write/shell tool is blocked.")

    def cmd_context(self, args, raw) -> Any:
        target = self.router.resolve(self.settings.default_model)
        messages = self.agent.system_messages() + self.agent.history
        tokens = count_tokens(messages, target.model)
        limit = self.services.context.limit_for(target.info.context_window,
                                                self.settings.max_tokens or target.info.max_output_tokens)
        rows = [("system prompt", f"{count_tokens(self.agent.system_messages(), target.model):,}"),
                ("history", f"{count_tokens(self.agent.history, target.model):,}"),
                ("project context", f"{len(self.project_context):,} chars"),
                ("memory", f"{len(self.memory_text):,} chars"),
                ("total (est.)", f"{tokens:,}"),
                ("budget limit", f"{limit:,}" if limit else "unknown window"),
                ("tools exposed", str(len(self.agent.tool_specs())))]
        self.renderer.kv(rows)
        pct = (tokens * 100 // limit) if limit else 0
        if limit:
            self.renderer.println(self.style.dim(f"  usage {pct}% of budget"))
            if pct > 75:
                self.renderer.warning("Context is getting full. /compact summarises it, /clear starts fresh.")

    def cmd_add(self, args, raw) -> Any:
        if not args:
            self.renderer.error("Usage: /add <path>")
            return
        target = self._resolve_dir(args[0])
        if not target.exists():
            self.renderer.error(f"Not found: {args[0]}")
            return
        if not is_within(self.workspace_root, target) and not any(is_within(d, target) for d in self.permissions.extra_dirs):
            self.permissions.extra_dirs.append(target)
            self.renderer.info(f"Added {target} to the allowed directories.")
        cleaned, blocks = self.expand_mentions("@" + str(target))
        if not blocks:
            self.renderer.warning("Nothing readable at that path.")
            return
        text = "\n\n".join(blocks)
        self.agent.history.append(Message.user(f"[context added by user]\n{text}"))
        self.renderer.success(f"Attached {target.name} "
                              f"({format_number(count_tokens([Message.user(text)], ''))} est. tokens)")

    def cmd_memory(self, args, raw) -> Any:
        from .tools.builtin.memory import memory_paths, read_memory

        for label, path in memory_paths(self.workspace_root):
            body = read_memory(path).strip()
            self.renderer.println(rule_(self.style, f"{label} · {path}"))
            self.renderer.println(body if body else self.style.dim("(empty)"))
        self.renderer.dim("\nSave with /remember <text>, or let the agent use the memory tool.")

    def cmd_remember(self, args, raw) -> Any:
        text = raw.strip()
        if not text:
            self.renderer.error("Usage: /remember <fact to remember>")
            return
        from .tools.builtin.memory import MemoryTool

        tool = MemoryTool()
        result = tool.execute({"action": "add", "scope": "project", "section": "Notes", "content": text},
                              self.services.tool_context("main"))
        (self.renderer.success if not result.is_error else self.renderer.error)(result.content)
        self.memory_text = self._load_memory()
        self.services.memory_text = self.memory_text

    def cmd_project(self, args, raw) -> Any:
        self.refresh_project_context()
        self.services.project_context = self.project_context
        if not self.project_context:
            self.renderer.info("No project context detected (empty or non-project directory).")
            return
        self.renderer.success(f"Project context rebuilt ({format_number(len(self.project_context))} chars)")
        self.renderer.println(self.style.dim(truncate(self.project_context, 1200)))

    def cmd_usage(self, args, raw) -> Any:
        totals = self.services.ledger.totals()
        self.renderer.println(rule_(self.style, "usage"))
        self.renderer.kv([("requests", totals["requests"]),
                          ("input tokens", f"{totals['input_tokens']:,}"),
                          ("output tokens", f"{totals['output_tokens']:,}"),
                          ("cached", f"{totals['cached_tokens']:,}"),
                          ("total", f"{totals['total_tokens']:,}"),
                          ("cost", format_cost(totals["cost_usd"]) + " (estimate)"),
                          ("model time", format_duration(totals["latency_ms"])),
                          ("permissions", self.permissions.describe())])
        by_model = self.services.ledger.by_model()
        if by_model:
            self.renderer.println("")
            self.renderer.table(["model", "reqs", "tokens", "cost"],
                                [[m["key"], m["requests"], format_number(m["total_tokens"]),
                                  format_cost(m["cost_usd"])] for m in by_model])
        by_agent = [a for a in self.services.ledger.by_agent() if a["key"] != "main"]
        if by_agent:
            self.renderer.println("")
            self.renderer.table(["agent", "reqs", "tokens", "cost"],
                                [[a["key"], a["requests"], format_number(a["total_tokens"]),
                                  format_cost(a["cost_usd"])] for a in by_agent])

    def cmd_status(self, args, raw) -> Any:
        target = self.router.resolve(self.settings.default_model)
        totals = self.services.ledger.totals()
        checkpoints = self.checkpoints.list()
        self.renderer.println(rule_(self.style, "status"))
        self.renderer.kv([
            ("version", __version__),
            ("model", target.label + (f" · {target.info.context_window:,} ctx" if target.info.context_window else "")),
            ("approval", self.permissions.mode + ("  (plan mode)" if self.plan_mode else "")),
            ("workspace", str(self.workspace_root)),
            ("cwd", str(self.cwd)),
            ("session", self.session.meta.id),
            ("history", f"{len(self.agent.history)} messages · "
                        f"{format_number(count_tokens(self.agent.history, target.model))} est. tokens"),
            ("tools", f"{len(self.registry.enabled_tools())} enabled of {len(self.registry.names())}"),
            ("usage", f"{totals['requests']} requests · {format_number(totals['total_tokens'])} tok · "
                      f"{format_cost(totals['cost_usd'])}"),
            ("checkpoints", f"{len(checkpoints)} ({human_size(self.checkpoints.disk_usage())})"),
            ("files touched", str(len(self.session.meta.touched))),
            ("offline", str(self.offline)),
            ("config", ", ".join(config_sources(self.settings)) or "defaults only"),
        ])

    def cmd_sessions(self, args, raw) -> Any:
        metas = self.session_store.list(limit=20)
        if not metas:
            self.renderer.info("No saved sessions.")
            return
        if self.interactive:
            items = [MenuItem(m.id, value=m.id,
                              hint=f"{m.turns} turn · {m.input_tokens + m.output_tokens} tok · "
                                   f"{(m.title or '(tanpa judul)')[:40]}")
                     for m in metas if m.id != self.session.meta.id]
            if items:
                chosen = self._pick(tr("session.pick"), items, prompt=tr("menu.pick_hint"))
                if chosen:
                    return self.cmd_resume([str(chosen)], str(chosen))
        rows = [[m.id, time.strftime("%m-%d %H:%M", time.localtime(m.updated_at or m.started_at)),
                 str(m.turns), format_number(m.input_tokens + m.output_tokens), format_cost(m.cost_usd),
                 truncate(m.title or "(untitled)", 44), Path(m.cwd).name] for m in metas]
        self.renderer.table(["id", "when", "turns", "tokens", "cost", "title", "dir"], rows)
        self.renderer.dim("\n/resume <id> to continue one of them.")

    def cmd_resume(self, args, raw) -> Any:
        target = args[0] if args else "last"
        session = self.session_store.load(target) if target not in ("last", "latest") \
            else self.session_store.latest(cwd=self.cwd)
        if session is None:
            self.renderer.error(f"Session '{target}' not found.")
            return
        self.session.close()
        self.session = session
        self.services.session = session
        self.agent.history = list(session.messages)
        self.checkpoints = CheckpointStore(self.workspace_root, session.meta.id)
        self.services.checkpoints = self.checkpoints
        self.renderer.success(f"Resumed {session.meta.id}: {len(self.agent.history)} messages restored.")

    def cmd_swarm(self, args, raw) -> Any:
        objective = raw.strip()
        if not objective:
            self.renderer.error("Usage: /swarm <objective>")
            self.renderer.dim("Example: /swarm add a JSON config loader with tests and docs")
            return
        self.run_swarm(objective)

    def cmd_swarm_mode(self, args, raw) -> Any:
        if not args and self.interactive:
            shapes = {
                "hive": "plan -> gelombang paralel -> review gate -> integrasi",
                "pipeline": "urutan peran tetap, tiap tahap menerima hasil sebelumnya",
                "parallel": "semua peran menjawab sekaligus, hasil digabung",
                "debate": "beberapa ronde argumen, lalu hakim memutuskan",
                "council": "pendapat paralel + satu sintesis",
                "review": "reviewer saja, digabung",
                "build": "architect + implementer + tester",
                "debug": "debugger + tester + reviewer",
                "audit": "security + reviewer + devops",
            }
            items = [MenuItem(m, value=m, hint=shapes.get(m, ""),
                              group="aktif" if m == self.settings.swarm.mode else "")
                     for m in MODES]
            chosen = self._pick(tr("swarm_mode.pick"), items, allow_filter=False,
                                prompt=tr("mode.current", mode=self.settings.swarm.mode))
            if chosen:
                args = [str(chosen)]
        if not args:
            self.renderer.markdown(_SWARM_MODES_TABLE)
            self.renderer.println(self.style.dim(f"current: {self.settings.swarm.mode}"))
            return
        mode = args[0].lower()
        if mode not in MODES:
            self.renderer.error(f"Unknown swarm mode '{mode}'.", hint="Valid: " + ", ".join(MODES))
            return
        self.settings.swarm.mode = mode
        self.renderer.success(f"Swarm mode: {mode}")
        self.renderer.dim("Default cast: " + ", ".join(MODE_DEFAULT_CAST.get(mode, [])))

    def cmd_cast(self, args, raw) -> Any:
        if not args and self.interactive:
            known = {p.key: p for p in all_personas()}
            known.update(self.custom_personas)
            current = set(self.settings.swarm.cast or MODE_DEFAULT_CAST.get(self.settings.swarm.mode, []))
            items = [MenuItem(f"{p.emoji} {p.name}".strip(), value=key, hint=p.role,
                              group="dipilih" if key in current else "")
                     for key, p in sorted(known.items())]
            chosen = self._pick(tr("cast.pick"), items, multi=True, prompt=tr("menu.multi_hint"))
            if chosen is None:
                return
            args = [str(c) for c in chosen]
            if not args:
                self.settings.swarm.cast = []
                self.renderer.info(tr("info.cast_cleared"))
                return
        if not args:
            current = self.settings.swarm.cast or MODE_DEFAULT_CAST.get(self.settings.swarm.mode, [])
            self.renderer.println(self.style.dim("cast for mode ") + self.settings.swarm.mode + ":")
            for key in current:
                persona = get_persona(key)
                self.renderer.println(f"  {persona.emoji} {self.style.paint('agent', persona.name):<10} "
                                      f"{self.style.dim(persona.role)}")
            self.renderer.dim("\nSet with: /cast architect implementer tester")
            return
        keys = []
        for key in args:
            key = key.strip().lower()
            if key not in {p.key for p in all_personas()} and key not in self.custom_personas:
                self.renderer.warning(f"Unknown persona '{key}' (using a generic one).")
            keys.append(key)
        self.settings.swarm.cast = keys
        self.renderer.success("cast: " + ", ".join(keys))

    def cmd_agents(self, args, raw) -> Any:
        rows = []
        for persona in all_personas():
            rows.append([persona.key, f"{persona.emoji} {persona.name}", persona.role,
                         persona.model_pref, str(persona.temperature),
                         "all" if persona.tools is None else str(len(persona.tools))])
        for key, persona in sorted(self.custom_personas.items()):
            rows.append([key + " *", f"{persona.emoji} {persona.name}", persona.role, persona.model_pref,
                         str(persona.temperature), "all" if persona.tools is None else str(len(persona.tools))])
        self.renderer.table(["key", "name", "role", "model", "temp", "tools"], rows)
        self.renderer.dim("\n* = custom persona from ~/.nexus/personas or .nexus/personas")

    def cmd_agent(self, args, raw) -> Any:
        if self.interactive and len(args) < 2:
            known = {p.key: p for p in all_personas()}
            known.update(self.custom_personas)
            items = [MenuItem(f"{p.emoji} {p.name}".strip(), value=key, hint=p.role)
                     for key, p in sorted(known.items())]
            chosen = self._pick(tr("persona.pick"), items, prompt=tr("menu.pick_hint"))
            if not chosen:
                return
            try:
                task = input(self.style.paint("accent", tr("persona.task", name=chosen))).strip()
            except (EOFError, KeyboardInterrupt):
                self.renderer.println("")
                return
            if not task:
                self.renderer.info(tr("info.task_empty"))
                return
            args = [str(chosen), task]
            raw = f"{chosen} {task}"
        if len(args) < 2:
            self.renderer.error("Usage: /agent <persona> <prompt>")
            return
        persona_key = args[0].lower()
        prompt = raw.split(None, 1)[1].split(None, 1)[1] if raw.count(" ") >= 2 else " ".join(args[1:])
        runner = SwarmRunner(self.services, SwarmConfig(mode="parallel", stream=not self.quiet,
                                                        default_model_spec=self.settings.default_model),
                             ui=self.renderer)
        was_live = self.renderer.live
        self.renderer.set_live(False)
        try:
            run = runner.run_agent(persona_key, prompt)
        finally:
            self.renderer.set_live(was_live)
        if run.text:
            self.renderer.markdown(run.text)
        if run.error:
            self.renderer.error(run.error)

    def cmd_debate(self, args, raw) -> Any:
        question = raw.strip()
        if not question:
            self.renderer.error("Usage: /debate <question>")
            return
        self.settings.swarm.mode = "debate"
        self.run_swarm(question, mode="debate")

    def cmd_board(self, args, raw) -> Any:
        if not self.board.all():
            self.renderer.info("Task board is empty. It is filled by /swarm or by the todo_write tool.")
            return
        progress = self.board.progress()
        self.renderer.println(rule_(self.style, f"task board · {progress['done']}/{progress['total']} done"))
        self.renderer.println(self.board.to_markdown())

    def cmd_config(self, args, raw) -> Any:
        if not args:
            self.renderer.println(render_markdown("```json\n" + json.dumps(self.settings.to_dict(), indent=2)
                                                + "\n```", self.style))
            self.renderer.dim(f"sources: {', '.join(config_sources(self.settings)) or 'defaults'}")
            return
        if len(args) == 1:
            value = _get_path(self.settings, args[0])
            if value is _MISSING:
                self.renderer.error(f"Unknown setting '{args[0]}'.")
                return
            self.renderer.println(f"{args[0]} = {json.dumps(value, ensure_ascii=False)}")
            return
        # Keep the value verbatim (it may contain spaces or JSON), rather than
        # indexing a fixed split -- `/config ui.theme light` has only two tokens.
        key = args[0]
        parts = raw.split(None, 1)
        value = parts[1].strip() if len(parts) > 1 else ""
        if not value:
            self.renderer.error("Usage: /config <key> <value>")
            self.renderer.dim("Contoh: /config ui.theme light  ·  /config swarm.max_parallel 6"
                          "  ·  nilai list bisa JSON atau dipisah spasi")
            return
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = value
            # Convenience: a space-separated value for a list setting becomes a list.
            if isinstance(parsed, str) and " " in parsed.strip():
                current = _get_path(self.settings, key)
                if isinstance(current, list):
                    parsed = parsed.split()
        if not _setting_exists(self.settings, key):
            # Without this, a dict-shaped candidate would silently accept any key
            # ("✓ not.a.key = x") while changing nothing at all.
            self.renderer.error(f"Unknown setting '{key}'.")
            self.renderer.dim("Run /config with no arguments to see every setting.")
            return
        # Build the change on a *copy* and validate that, so a rejected value can
        # never leave the live settings half-mutated (an invalid approval_mode
        # would silently disable the permission engine's mode checks).
        candidate_data = self.settings.to_dict()
        try:
            _set_path(candidate_data, key, parsed)
            candidate = Settings.from_dict(candidate_data)
        except (NexusError, AttributeError, TypeError, ValueError) as exc:
            self.renderer.error(f"Cannot set {key}: {exc}")
            return
        for field_name in candidate.__dataclass_fields__:
            setattr(self.settings, field_name, getattr(candidate, field_name))
        self.renderer.success(f"{key} = {json.dumps(parsed, ensure_ascii=False)}")
        self.renderer.dim("Runtime settings apply now; to keep them, run: nexus config set "
                          f"{key} {json.dumps(parsed, ensure_ascii=False)}")

    def cmd_auth(self, args, raw) -> Any:
        if not args:
            rows = []
            for key in sorted(PROVIDERS):
                spec = PROVIDERS[key]
                source = "env" if any(os.environ.get(e) for e in spec.env_keys) else (
                    "stored" if load_auth().get(key) else "-")
                rows.append([key, source, ", ".join(spec.env_keys) or "(none)"])
            self.renderer.table(["provider", "key", "env vars"], rows)
            self.renderer.dim("\n/auth <provider> <key> stores it in ~/.nexus/auth.json (chmod 600).")
            return
        provider = args[0].lower()
        if provider not in PROVIDERS:
            self.renderer.error(f"Unknown provider '{provider}'.")
            return
        if len(args) > 1:
            secret = args[1]
        else:
            try:
                import getpass

                secret = getpass.getpass(f"{provider} API key: ").strip()
            except (EOFError, KeyboardInterrupt):
                self.renderer.println("")
                return
        if not secret:
            self.renderer.error("Empty key; nothing stored.")
            return
        path = save_auth_key(provider, secret)
        self.router.auth_store[provider] = secret
        self.router.drop_cached(provider)
        self.renderer.success(f"Stored {provider} key in {path}")

    def cmd_doctor(self, args, raw) -> Any:
        rows: List[List[str]] = []

        def check(name: str, ok: bool, detail: str) -> None:
            rows.append([self.style.paint("success" if ok else "error", "✓" if ok else "✗"), name, detail])

        check("python", sys.version_info >= (3, 9), sys.version.split()[0])
        check("workspace writable", os.access(self.workspace_root, os.W_OK), str(self.workspace_root))
        check("data dir", data_dir().is_dir() or data_dir().parent.is_dir(), str(data_dir()))
        check("log file", (data_dir() / "logs" / "nexus.log").parent.is_dir(), str(data_dir() / "logs"))
        check("git available", bool(shutil.which("git")), shutil.which("git") or "not on PATH")
        ready = [p.key for p in available_providers(include_local=True)]
        check("providers with keys", bool(ready), ", ".join(ready) or "none configured")
        target = None
        try:
            target = self.router.resolve()
            self.router.provider(target.provider_key)
            check("default model usable", True, target.label)
        except Exception as exc:
            check("default model usable", False, f"{target.label if target else '?'}: {exc}")
        check("tools registered", len(self.registry.names()) >= 15, f"{len(self.registry.names())} tools")
        check("permission mode", self.permissions.mode in APPROVAL_MODES, self.permissions.mode)
        check("session writable", self.session.persist, str(self.session.path))
        self.renderer.println(rule_(self.style, "doctor"))
        self.renderer.table(["", "check", "detail"], rows)
        network = _probe_network()
        self.renderer.println(self.style.dim(f"\nnetwork: {network}"))
        if not ready:
            self.renderer.println(self.style.dim("Next step: /auth <provider> <key>  (see /providers)"))

    def cmd_selftest(self, args, raw) -> Any:
        root = Path(__file__).resolve().parent.parent
        runner_path = root / "tests" / "run_tests.py"
        if not runner_path.is_file():
            self.renderer.error(f"Test suite not found at {runner_path} (packaged install?).")
            return
        self.renderer.info(f"Running the built-in suite: {runner_path}")
        try:
            proc = subprocess.run([sys.executable, str(runner_path), *args], cwd=str(root), text=True,
                                  capture_output=True, timeout=600)
        except subprocess.TimeoutExpired:
            self.renderer.error("Self-test timed out after 600s.")
            return
        output = (proc.stdout or "") + (proc.stderr or "")
        self.renderer.println(truncate(output.strip(), 6000))
        (self.renderer.success if proc.returncode == 0 else self.renderer.error)(
            f"exit {proc.returncode}")

    def cmd_mcp(self, args, raw) -> Any:
        if not self.mcp_clients:
            self.renderer.info("No MCP servers configured.")
            self.renderer.dim("Add one with: nexus mcp add <name> -- <command> [args…]")
            return
        rows = []
        for client in self.mcp_clients:
            rows.append([client.name, "connected" if client.alive else "dead", str(len(client.tool_names())),
                         truncate(client.command_line(), 50)])
        self.renderer.table(["server", "state", "tools", "command"], rows)

    def cmd_keys(self, args, raw) -> Any:
        self.renderer.markdown(_KEYS_HELP)

    def cmd_log(self, args, raw) -> Any:
        path = data_dir() / "logs" / "nexus.log"
        n = int(args[0]) if args and args[0].isdigit() else 20
        self.renderer.println(self.style.dim(str(path)))
        if not path.is_file():
            self.renderer.info("No log file yet.")
            return
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:]
        except OSError as exc:
            self.renderer.error(f"Cannot read log: {exc}")
            return
        for line in lines:
            self.renderer.println(self.style.dim(truncate(line, self.style.width - 2)))

    def cmd_about(self, args, raw) -> Any:
        self.renderer.box("NEXUS", [
            f"version {__version__}",
            "A zero-dependency agentic CLI: multi-provider, tool-using, checkpointed,",
            "with a multi-persona swarm mode.",
            "",
            f"providers: {len(PROVIDERS)}   tools: {len(self.registry.names())}   "
            f"personas: {len(all_personas()) + len(self.custom_personas)}   swarm modes: {len(MODES)}",
            f"python {sys.version.split()[0]} on {sys.platform}",
        ])


EXIT = object()


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def configure_logging_():
    level = "DEBUG" if os.environ.get("NEXUS_DEBUG") else "INFO"
    return configure_logging(level=level, echo=bool(os.environ.get("NEXUS_LOG_ECHO")))


def rule_(style: Style, title: str) -> str:
    return rule(style, title)


def _closest(name: str, candidates: Sequence[str]) -> Optional[str]:
    import difflib

    matches = difflib.get_close_matches(name, list(candidates), n=1, cutoff=0.6)
    return matches[0] if matches else None


def _known_specs() -> List[Tuple[str, str]]:
    from .providers.registry import MODEL_CATALOG

    return [(m.provider, m.id) for m in MODEL_CATALOG.values() if m.provider]


def _probe_network() -> str:
    import socket

    for host in ("api.openai.com", "api.anthropic.com"):
        try:
            socket.create_connection((host, 443), timeout=2).close()
            return f"reachable ({host})"
        except OSError:
            continue
    return "no outbound HTTPS detected (offline mode may be required)"


_MISSING = object()

#: Settings whose value is an open mapping: new keys are legitimate there.
OPEN_SETTING_PREFIXES = ("providers.", "mcp.servers.", "search.")


def _setting_exists(settings: Settings, dotted: str) -> bool:
    """True when *dotted* names a real setting.

    Dataclass paths are walked attribute by attribute. Open mappings
    (``providers.<name>.<field>``, ``mcp.servers.<name>``, ``search.<key>``)
    accept new keys, but a provider sub-key must still be a real
    :class:`ProviderConfig` field -- otherwise a typo like
    ``providers.openai.api_kye`` would be accepted and silently do nothing.
    """
    from .core.config import ProviderConfig

    if dotted.startswith("providers."):
        parts = dotted.split(".")
        if len(parts) == 2:
            return bool(parts[1])
        if len(parts) == 3:
            return parts[2] in ProviderConfig.__dataclass_fields__
        if len(parts) == 4:
            return parts[2] in ("extra", "headers") and bool(parts[3])
        return False
    if dotted.startswith(("mcp.servers.", "search.")):
        return len(dotted.split(".")) >= 2 and bool(dotted.split(".")[-1])
    node: Any = settings
    for part in dotted.split("."):
        if isinstance(node, dict):
            if part not in node:
                return False
            node = node[part]
        elif hasattr(node, part):
            node = getattr(node, part)
        else:
            return False
    return True





def _get_path(settings: Settings, dotted: str) -> Any:
    node: Any = settings
    for part in dotted.split("."):
        if isinstance(node, dict):
            if part not in node:
                return _MISSING
            node = node[part]
        elif hasattr(node, part):
            node = getattr(node, part)
        else:
            return _MISSING
    return node


def _set_path(settings: Settings, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node: Any = settings
    for part in parts[:-1]:
        if isinstance(node, dict):
            node = node.setdefault(part, {})
        else:
            node = getattr(node, part)
    leaf = parts[-1]
    if isinstance(node, dict):
        node[leaf] = value
        return
    current = getattr(node, leaf, None)
    if isinstance(current, bool):
        value = str(value).lower() in ("1", "true", "yes", "on")
    elif isinstance(current, int) and not isinstance(current, bool):
        value = int(value)
    elif isinstance(current, float):
        value = float(value)
    setattr(node, leaf, value)


# --------------------------------------------------------------------------- #
# help texts
# --------------------------------------------------------------------------- #
_MODES_HELP = """## Approval modes

| mode | reads | file writes | shell | dangerous |
|------|-------|-------------|-------|-----------|
| `read-only` | auto | blocked | blocked | blocked |
| `suggest` | auto | ask | ask | ask |
| `auto-edit` | auto | auto | ask | ask |
| `full-auto` | auto | auto | auto | ask |
| `yolo` | auto | auto | auto | auto |

Rules override modes: `deny` always wins, then `allow`, then `ask`.
Rule syntax is `tool:pattern` -- `bash:git`, `write_file:src/**`, `web_fetch:*.github.com`.
`*` matches anything (including `/`); use `**` for strict path globs.
Answering `a` (always) at a prompt writes an allow rule into `.nexus/config.json`.
"""

_SWARM_MODES_TABLE = """## Swarm modes

| mode | shape | use it for |
|------|-------|-----------|
| `hive` | plan → parallel waves → review gate → integrate | real features, multi-file work |
| `pipeline` | fixed role sequence, each stage builds on the last | design→build→test→document |
| `parallel` | every role answers at once, results merged | broad coverage, quick survey |
| `council` | parallel opinions + a synthesis pass | decisions needing several viewpoints |
| `debate` | N rounds of argument, then a judge | contentious design choices |
| `review` | reviewers only, merged | code review, audits |
| `build` | architect + implementer + tester | focused implementation |
| `debug` | debugger + tester + reviewer | hard-to-find failures |
| `audit` | security + reviewer + devops | security / release readiness |
"""

_SWARM_HELP = _SWARM_MODES_TABLE + """
## How a swarm runs (`hive`)

1. **Atlas** (orchestrator) reads the repo and writes a task board with dependencies.
2. Workers run in **waves**: tasks whose dependencies are done, up to `swarm.max_parallel`
   at a time. Each worker is a separate agent with its own context window and persona.
3. A **reviewer gate** inspects the result of each wave and opens `blocker:` / `should-fix:`
   tasks when something is wrong (bounded by `swarm.max_rounds`).
4. Atlas integrates everything into one final answer.

Agents coordinate through a **blackboard** (findings, questions, decisions, artifacts) and the
shared **task board**. Every run is accounted separately: `/usage` shows per-agent tokens.

## Examples

    /swarm add a JSON config loader with validation and tests
    /swarm-mode debate
    /debate should sessions be stored in SQLite or JSONL?
    /cast architect implementer tester security
    /swarm audit this repo for security issues before release
    /agent reviewer review the last commit

## Configuration

    nexus config set swarm.mode hive
    nexus config set swarm.max_parallel 4
    nexus config set swarm.max_rounds 3
    nexus config set swarm.model_specs '{"orchestrator":"anthropic:sonnet","tester":"groq:llama-3.3-70b-versatile"}'

Custom personas: drop JSON files in `~/.nexus/personas/` (see docs/SWARM.md).
"""

_KEYS_HELP = """## Keyboard

| keys | effect |
|------|--------|
| `Tab` | complete a `/command`, an argument, or an `@path` |
| `↑` `↓` | history (persisted in `~/.nexus/history.txt`) |
| `Ctrl+C` | cancel the running turn; twice in a row exits |
| `Ctrl+D` | exit on an empty prompt |
| `\\` at end of line | continue on the next line |
| a ``` fence | multi-line paste is detected automatically |

## Input prefixes

| prefix | effect |
|--------|--------|
| `/` | slash command (`/help`) |
| `!` | run a shell command in the workspace (`!git status`) |
| `@` | attach a file or directory (`@src/main.py fix the bug`) |
| anything else | talk to the agent |
"""
