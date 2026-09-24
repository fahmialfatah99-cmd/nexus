"""Command line interface.

Subcommand dispatch is explicit (not argparse-driven) so that the common case --
``nexus "fix the bug"`` or bare ``nexus`` -- keeps working exactly as expected and
never collides with a flag. Every subcommand builds its own small parser, which
keeps ``--help`` readable instead of a wall of mutually exclusive options.

Exit codes: 0 ok · 1 task/agent failure · 2 usage/config/provider error ·
3 unexpected internal error · 130 interrupted.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from . import __version__
from .core.config import (ProviderConfig, Settings, config_sources, load_auth, load_settings,
                          remove_auth_key, save_auth_key)
from .core.errors import ConfigError, NexusError
from .core.logging_ import get_logger
from .core.paths import (config_file, data_dir, ensure_dirs, find_project_root, home,
                         logs_dir, sessions_dir)
from .providers.registry import MODEL_CATALOG, PROVIDERS, available_providers, model_info
from .ui.theme import Style

SUBCOMMANDS = {
    "swarm", "debate", "agent", "run", "models", "providers", "tools", "personas",
    "sessions", "resume", "export", "config", "auth", "mcp", "plugins", "doctor",
    "selftest", "demo", "init", "version", "log",
}

EPILOG = """examples:
  nexus                                   start an interactive session
  nexus "why does this test fail?"        interactive, seeded with a first prompt
  nexus -p "summarise this repo"          one-shot answer, then exit
  git diff | nexus -p "review this"       pipe stdin as context
  nexus swarm "add CSV export with tests" run the multi-agent swarm
  nexus debate "SQLite or JSONL?"         structured debate with a judge
  nexus agent reviewer "review src/"      run one persona once
  nexus models --provider groq --refresh  list live models
  nexus doctor                            diagnose the installation
  nexus demo                              offline guided tour (no API key needed)
"""


# --------------------------------------------------------------------------- #
# shared flags
# --------------------------------------------------------------------------- #
def add_common_flags(parser: argparse.ArgumentParser, exclude: Sequence[str] = ()) -> None:
    """Attach the flags every subcommand shares.

    ``exclude`` lets a subcommand own a name itself (``nexus swarm --mode`` means
    the swarm mode, not the approval mode) instead of crashing argparse with a
    conflicting-option-string error.
    """
    skip = set(exclude)

    def want(*names: str) -> bool:
        return not any(n in skip for n in names)

    group = parser.add_argument_group("model")
    group.add_argument("-m", "--model", default="", help="model spec, e.g. anthropic:sonnet or gpt-4o")
    group.add_argument("--provider", default="", help="provider key (see `nexus providers`)")
    group.add_argument("--temperature", type=float, default=None)
    group.add_argument("--max-tokens", type=int, default=None)
    group.add_argument("--max-turns", type=int, default=None, help="tool-loop ceiling per request")
    group.add_argument("--failover", default="", help="comma separated fallback specs")
    group.add_argument("--reasoning", choices=["off", "low", "medium", "high"], default="")

    group = parser.add_argument_group("permissions")
    if want("--mode"):
        group.add_argument("--mode", "--approval-mode", dest="mode", default="",
                           choices=["", "read-only", "suggest", "auto-edit", "full-auto", "yolo"])
    group.add_argument("--read-only", action="store_true", help="block every write/shell tool")
    group.add_argument("--auto-edit", action="store_true", help="auto-approve file edits, ask for shell")
    group.add_argument("--full-auto", action="store_true", help="auto-approve everything except dangerous ops")
    group.add_argument("--yolo", "--dangerously-skip-permissions", dest="yolo", action="store_true",
                       help="approve everything without asking")
    group.add_argument("--add-dir", action="append", default=[], help="allow paths outside the workspace")
    group.add_argument("--allow-private-network", action="store_true",
                       help="let web_fetch reach private/internal addresses (SSRF protection off)")
    group.add_argument("--no-tools", action="store_true", help="disable all tools (chat only)")

    group = parser.add_argument_group("output")
    group.add_argument("--color", dest="color", action="store_true", default=None)
    group.add_argument("--no-color", dest="color", action="store_false")
    group.add_argument("--theme", default="", choices=["", "dark", "light", "mono"])
    group.add_argument("--lang", "--language", dest="lang", default="", choices=["", "en", "id"],
                       help="menu/dialog language (default: from your locale)")
    group.add_argument("--no-menu", action="store_true",
                       help="never open clickable menus; use the numbered text list")
    group.add_argument("-q", "--quiet", action="store_true", help="minimal output (for scripts)")
    group.add_argument("--no-spinner", action="store_true")
    group.add_argument("--compact", action="store_true",
                       help="compact output: fold tool results onto the tool line")
    group.add_argument("--no-markdown", action="store_true", help="plain text output")

    group = parser.add_argument_group("session")
    group.add_argument("-C", "--cwd", default="", help="run in this directory")
    group.add_argument("-s", "--session", default="", help="session id to use")
    group.add_argument("-r", "--resume", default="", help="resume a session id, or 'last'")
    group.add_argument("--offline", action="store_true", help="disable network tools")
    group.add_argument("--no-plugins", action="store_true")
    group.add_argument("--no-mcp", action="store_true")
    group.add_argument("--no-stream", action="store_true")
    group.add_argument("-v", "--verbose", action="store_true")
    group.add_argument("--debug", action="store_true", help="debug logging to stderr")


def settings_from_args(args: argparse.Namespace) -> Settings:
    cwd = Path(args.cwd).expanduser() if getattr(args, "cwd", "") else Path.cwd()
    overrides: Dict[str, Any] = {}
    if getattr(args, "model", ""):
        spec = args.model
        if ":" in spec:
            head, _, tail = spec.partition(":")
            if head.lower() in PROVIDERS and tail:
                overrides.setdefault("providers", {}).setdefault(head.lower(), {})
                overrides["default_provider"] = head.lower()
                overrides["default_model"] = tail
            else:
                overrides["default_model"] = spec
        else:
            overrides["default_model"] = spec
    if getattr(args, "provider", ""):
        overrides["default_provider"] = args.provider
    if getattr(args, "temperature", None) is not None:
        overrides["temperature"] = args.temperature
    if getattr(args, "max_tokens", None):
        overrides["max_tokens"] = args.max_tokens
    if getattr(args, "max_turns", None):
        overrides["max_turns"] = args.max_turns
    if getattr(args, "failover", ""):
        overrides["failover"] = [p.strip() for p in args.failover.split(",") if p.strip()]
    mode = ""
    for flag, name in (("yolo", "yolo"), ("full_auto", "full-auto"), ("auto_edit", "auto-edit"),
                       ("read_only", "read-only")):
        if getattr(args, flag, False):
            mode = name
    if getattr(args, "mode", ""):
        mode = args.mode
    if mode:
        overrides["approval_mode"] = mode
    if getattr(args, "add_dir", []):
        overrides["extra_dirs"] = list(args.add_dir)
    if getattr(args, "allow_private_network", False):
        overrides["allow_private_network"] = True
    if getattr(args, "offline", False):
        overrides["offline"] = True
    if getattr(args, "no_stream", False):
        overrides["stream"] = False
    if getattr(args, "verbose", False) or getattr(args, "debug", False):
        overrides["verbose"] = True
    ui: Dict[str, Any] = {}
    if getattr(args, "color", None) is not None:
        ui["color"] = bool(args.color)
    if getattr(args, "theme", ""):
        ui["theme"] = args.theme
    if getattr(args, "no_spinner", False):
        ui["spinner"] = False
    if getattr(args, "compact", False):
        ui["compact"] = True
    if getattr(args, "no_markdown", False):
        ui["markdown"] = False
    if getattr(args, "lang", ""):
        ui["language"] = args.lang
    if getattr(args, "no_menu", False):
        os.environ["NEXUS_NO_MENU"] = "1"
    if ui:
        overrides["ui"] = ui
    settings = load_settings(cwd=cwd, overrides=overrides)
    if getattr(args, "debug", False):
        os.environ["NEXUS_DEBUG"] = "1"
        os.environ["NEXUS_LOG_ECHO"] = "1"   # --debug means "show me", not just "record it"
    elif getattr(args, "verbose", False):
        os.environ.setdefault("NEXUS_LOG_ECHO", "1")
    return settings


def build_app(args: argparse.Namespace, *, quiet: Optional[bool] = None):
    from .app import App

    settings = settings_from_args(args)
    cwd = Path(args.cwd).expanduser() if getattr(args, "cwd", "") else Path.cwd()
    return App(settings=settings, cwd=cwd, quiet=quiet if quiet is not None else bool(args.quiet),
               session_id=getattr(args, "session", ""), resume=getattr(args, "resume", ""),
               model_spec=getattr(args, "model", ""), approval_mode=settings.approval_mode,
               extra_dirs=getattr(args, "add_dir", []),
               enable_plugins=not getattr(args, "no_plugins", False),
               enable_mcp=not getattr(args, "no_mcp", False))


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in SUBCOMMANDS:
        try:
            return dispatch(argv[0], argv[1:])
        except KeyboardInterrupt:
            print("", file=sys.stderr)
            return 130
        except ConfigError as exc:
            print(f"error: {exc}", file=sys.stderr)
            if exc.hint:
                print(f"hint: {exc.hint}", file=sys.stderr)
            return 2
        except NexusError as exc:
            print(f"error: {exc}", file=sys.stderr)
            if exc.hint:
                print(f"hint: {exc.hint}", file=sys.stderr)
            return 2
    try:
        return chat_main(argv)
    except KeyboardInterrupt:
        print("", file=sys.stderr)
        return 130
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        if exc.hint:
            print(f"hint: {exc.hint}", file=sys.stderr)
        return 2
    except NexusError as exc:
        print(f"error: {exc}", file=sys.stderr)
        if exc.hint:
            print(f"hint: {exc.hint}", file=sys.stderr)
        return 2
    except BrokenPipeError:
        return 0
    except Exception as exc:  # last resort: never dump a raw traceback on users
        log = get_logger()
        log.exception("fatal", exc=exc)
        print(f"nexus: unexpected error: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(f"details: {logs_dir() / 'nexus.log'}", file=sys.stderr)
        return 3


def dispatch(name: str, argv: List[str]) -> int:
    handler = SUBCOMMAND_HANDLERS.get(name)
    if handler is None:  # pragma: no cover - guarded by SUBCOMMANDS
        print(f"unknown subcommand: {name}", file=sys.stderr)
        return 2
    return handler(argv)


# --------------------------------------------------------------------------- #
# chat / one-shot
# --------------------------------------------------------------------------- #
def chat_main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="nexus", description="NEXUS -- a zero-dependency agentic CLI with multi-provider "
                                  "support and a multi-persona swarm mode.",
        epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("prompt", nargs="*", help="initial prompt (or the whole task with -p)")
    parser.add_argument("-p", "--print", dest="print_mode", action="store_true",
                        help="non-interactive: answer and exit (script friendly)")
    parser.add_argument("--output-format", choices=["text", "json"], default="")
    parser.add_argument("--json", action="store_true", help="machine readable output with -p")
    parser.add_argument("--swarm", action="store_true", help="run the prompt with the swarm")
    parser.add_argument("--swarm-mode", default="", help="swarm mode for --swarm/-p")
    parser.add_argument("--cast", nargs="*", default=[], help="personas for --swarm")
    parser.add_argument("--stdin", action="store_true", help="read extra context from stdin")
    parser.add_argument("-V", "--version", action="version", version=f"nexus {__version__}")
    add_common_flags(parser)
    args = parser.parse_args(argv)

    prompt_parts = list(args.prompt or [])
    stdin_text = ""
    # Consume stdin as context only when the user clearly wants it: with -p,
    # with an explicit --stdin, or when a prompt was given AND stdin is a pipe.
    # Otherwise a piped list of slash commands must reach the REPL intact.
    want_stdin = bool(args.stdin or args.print_mode
                      or (prompt_parts and not sys.stdin.isatty()))
    if want_stdin:
        try:
            stdin_text = sys.stdin.read()
        except (OSError, KeyboardInterrupt):
            stdin_text = ""
    if stdin_text.strip():
        limit = 200_000
        body = stdin_text if len(stdin_text) <= limit else stdin_text[:limit] + "\n…(stdin truncated)"
        prompt_parts.insert(0, f"<stdin>\n{body}\n</stdin>\n")
    prompt = "\n".join(p for p in prompt_parts if p).strip()

    app = build_app(args)
    if args.cast:
        app.settings.swarm.cast = list(args.cast)
    if args.no_tools:
        for tool_name in app.registry.names():
            app.registry.set_enabled(tool_name, False)
    if args.reasoning:
        app.agent.options.reasoning_effort = None if args.reasoning == "off" else args.reasoning
    if args.no_markdown:
        app.renderer.live = True
        app.settings.ui.markdown = False

    output_json = bool(args.json) or args.output_format == "json"
    if args.print_mode or (prompt and not sys.stdin.isatty() and not sys.stdout.isatty()):
        if not prompt:
            print("error: -p requires a prompt (or piped stdin)", file=sys.stderr)
            return 2
        return app.run_once(prompt, swarm=bool(args.swarm or args.swarm_mode),
                            output_json=output_json, swarm_mode=args.swarm_mode)
    app.renderer.quiet = bool(args.quiet)
    if prompt and not args.swarm:
        app.submit(prompt)
    elif prompt and args.swarm:
        app.run_swarm(prompt, mode=args.swarm_mode)
    return app.run()


# --------------------------------------------------------------------------- #
# subcommands
# --------------------------------------------------------------------------- #
def _simple_parser(name: str, description: str, argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog=f"nexus {name}", description=description)
    add_common_flags(parser)
    return parser.parse_args(argv)


def cmd_swarm(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(prog="nexus swarm",
                                     description="Run the multi-agent swarm on an objective.")
    parser.add_argument("objective", nargs="+", help="what the swarm should achieve")
    parser.add_argument("--swarm-mode", "--mode", dest="swarm_mode", default="",
                        help="hive|pipeline|parallel|debate|council|review|build|debug|audit")
    parser.add_argument("--cast", nargs="*", default=[], help="persona keys to use")
    parser.add_argument("--max-parallel", type=int, default=0)
    parser.add_argument("--max-rounds", type=int, default=0)
    parser.add_argument("--plan-only", action="store_true", help="plan the tasks, do not execute")
    parser.add_argument("--no-review-gate", action="store_true")
    parser.add_argument("--json", action="store_true")
    add_common_flags(parser, exclude=("--mode", "--approval-mode"))
    args = parser.parse_args(argv)
    app = build_app(args)
    objective = " ".join(args.objective).strip()
    if args.cast:
        app.settings.swarm.cast = list(args.cast)
    if args.max_parallel:
        app.settings.swarm.max_parallel = args.max_parallel
    if args.max_rounds:
        app.settings.swarm.max_rounds = args.max_rounds
    if args.no_review_gate:
        app.settings.swarm.reviewer_gate = False
    if args.plan_only:
        app.settings.swarm.mode = args.swarm_mode or "hive"
        from .agents.swarm import SwarmConfig, SwarmRunner

        runner = SwarmRunner(app.services, SwarmConfig(
            mode=app.settings.swarm.mode, cast=app.settings.swarm.cast, plan_only=True,
            default_model_spec=app.settings.default_model, max_turns_per_agent=12), ui=app.renderer)
        result = runner.run(objective)
        app.renderer.markdown(result.final_text)
        app.renderer.println(app.renderer.style.dim("\nPlanned tasks:"))
        app.renderer.println(runner.board.to_markdown())
        app.shutdown()
        return 0
    result = app.run_swarm(objective, mode=args.swarm_mode)
    if args.json:
        print(json.dumps({"mode": result.mode, "final": result.final_text,
                          "runs": [{"agent": r.agent, "persona": r.persona_key, "ok": r.ok,
                                    "status": r.status, "summary": r.one_line()} for r in result.runs],
                          "usage": result.usage.as_dict(), "cost": result.cost,
                          "tasks": result.board.to_dict()}, ensure_ascii=False, indent=2))
    app.shutdown()
    return 0 if not result.failures() else 1


def cmd_debate(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(prog="nexus debate",
                                     description="Run a structured debate and let a judge decide.")
    parser.add_argument("question", nargs="+")
    parser.add_argument("--cast", nargs="*", default=[])
    parser.add_argument("--rounds", type=int, default=2)
    add_common_flags(parser)
    args = parser.parse_args(argv)
    app = build_app(args)
    app.settings.swarm.debate_rounds = max(1, args.rounds)
    if args.cast:
        app.settings.swarm.cast = list(args.cast)
    app.run_swarm(" ".join(args.question).strip(), mode="debate")
    app.shutdown()
    return 0


def cmd_agent(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(prog="nexus agent", description="Run a single persona once.")
    parser.add_argument("persona", help="persona key (see `nexus personas`)")
    parser.add_argument("prompt", nargs="+")
    parser.add_argument("--json", action="store_true")
    add_common_flags(parser)
    args = parser.parse_args(argv)
    app = build_app(args)
    from .agents.swarm import SwarmConfig, SwarmRunner

    runner = SwarmRunner(app.services, SwarmConfig(mode="parallel", stream=not args.quiet,
                                                   default_model_spec=app.settings.default_model),
                         ui=app.renderer)
    app.renderer.set_live(False)
    run = runner.run_agent(args.persona, " ".join(args.prompt).strip())
    app.renderer.set_live(True)
    if args.json:
        print(json.dumps({"agent": run.agent, "persona": run.persona_key, "ok": run.ok,
                          "status": run.status, "text": run.text, "report": run.report,
                          "usage": run.usage.as_dict(), "cost": run.cost}, ensure_ascii=False, indent=2))
    else:
        app.renderer.markdown(run.text)
    app.shutdown()
    return 0 if run.ok else 1


def cmd_run(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(prog="nexus run", description="Run instructions from a file.")
    parser.add_argument("file", help="path to a prompt file (- for stdin)")
    parser.add_argument("--swarm", action="store_true")
    parser.add_argument("--json", action="store_true")
    add_common_flags(parser)
    args = parser.parse_args(argv)
    if args.file == "-":
        text = sys.stdin.read()
    else:
        path = Path(args.file).expanduser()
        if not path.is_file():
            print(f"error: no such file: {path}", file=sys.stderr)
            return 2
        text = path.read_text(encoding="utf-8", errors="replace")
    app = build_app(args)
    return app.run_once(text.strip(), swarm=bool(args.swarm), output_json=bool(args.json))


def cmd_models(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(prog="nexus models", description="List models.")
    parser.add_argument("--refresh", action="store_true", help="fetch the live list from the provider")
    parser.add_argument("--json", action="store_true")
    add_common_flags(parser)
    args = parser.parse_args(argv)
    settings = settings_from_args(args)
    from .core.router import ModelRouter

    router = ModelRouter(default_provider=settings.default_provider, default_model=settings.default_model,
                         provider_configs=settings.provider_configs(), auth_store=load_auth())
    provider_key = args.provider or settings.default_provider or router._detect_provider()
    live_error = ""
    if args.refresh or not args.json:
        try:
            live = router.list_models(provider_key, refresh=args.refresh)
        except NexusError as exc:
            live = []
            live_error = str(exc)
            print(f"warning: could not list live models ({exc})", file=sys.stderr)
    else:
        live = []
    known = sorted(m.id for m in MODEL_CATALOG.values() if not provider_key or m.provider == provider_key)
    models = sorted(set(live) | set(known))
    if args.json:
        payload = [{"id": m, "provider": provider_key, **{k: v for k, v in model_info(m, provider_key).__dict__.items()
                                                          if k not in ("id", "provider", "aliases")}} for m in models]
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    style = Style.create(settings.ui.theme, enabled=None if settings.ui.color else False)
    if not models:
        if live_error:
            print(f"Could not reach '{provider_key}' and it has no catalogue entries.")
            print(f"  error: {live_error}")
            spec = PROVIDERS.get(provider_key)
            if spec and spec.local:
                print(f"  is the server running? expected at {spec.default_base_url}")
            else:
                print(f"  check the key (`nexus auth list`) and base URL "
                      f"(`nexus config get providers.{provider_key}.base_url`)")
        else:
            print(f"No models known for '{provider_key}'. Try --refresh, or `nexus auth login {provider_key}`.")
        return 1
    print(f"{provider_key}: {len(models)} models" + (" (live)" if live else " (catalogue)"))
    for m in models:
        info = model_info(m, provider_key)
        ctx = f"{info.context_window // 1000}k" if info.context_window else "?"
        caps = "".join(c for c, ok in (("T", info.supports_tools), ("I", info.supports_images),
                                       ("R", info.supports_reasoning)) if ok)
        print(f"  {m:<46} ctx={ctx:<6} {caps}")
    print("\nTI=tools/images/reasoning · use as: nexus -m " + f"{provider_key}:<model>")
    return 0


def cmd_providers(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(prog="nexus providers", description="List providers and key status.")
    parser.add_argument("--json", action="store_true")
    add_common_flags(parser)
    args = parser.parse_args(argv)
    settings = settings_from_args(args)
    auth = load_auth()
    rows = []
    for key in sorted(PROVIDERS):
        spec = PROVIDERS[key]
        env_key = next((e for e in spec.env_keys if os.environ.get(e)), "")
        source = "env:" + env_key if env_key else ("stored" if auth.get(key) else ("local" if spec.local else "-"))
        rows.append({"key": key, "name": spec.display_name, "ready": source != "-", "key_source": source,
                     "default_model": spec.default_model, "base_url": spec.default_base_url,
                     "env_vars": list(spec.env_keys), "note": spec.note, "local": spec.local})
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return 0
    print(f"{'key':<12} {'name':<26} {'status':<10} {'default model':<34} base url")
    for row in rows:
        status = row["key_source"] if row["ready"] else "no key"
        print(f"{row['key']:<12} {row['name']:<26} {status:<10} {row['default_model']:<34} {row['base_url']}")
    ready = [r["key"] for r in rows if r["ready"]]
    print(f"\nready: {', '.join(ready) if ready else 'none -- run `nexus auth login <provider>`'}")
    return 0


def cmd_tools(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(prog="nexus tools", description="List the built-in tools.")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--all", action="store_true", help="include swarm-only tools")
    add_common_flags(parser)
    args = parser.parse_args(argv)
    from .tools import tool_catalog

    catalog = tool_catalog()
    if not args.all:
        catalog = [t for t in catalog if t["category"] != "swarm"]
    if args.json:
        print(json.dumps(catalog, ensure_ascii=False, indent=2))
        return 0
    current = ""
    for tool in catalog:
        if tool["category"] != current:
            current = tool["category"]
            print(f"\n{current.upper()}")
        flags = ("read-only" if tool["read_only"] else "writes") + (" · network" if tool["needs_network"] else "")
        print(f"  {tool['name']:<14} {flags:<20} {tool['description'][:96]}")
    print(f"\n{len(catalog)} tools. Disable with: nexus config set tools.disabled '[\"bash\"]'")
    return 0


def cmd_personas(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(prog="nexus personas", description="List agent personas.")
    parser.add_argument("--json", action="store_true")
    add_common_flags(parser)
    args = parser.parse_args(argv)
    from .agents.persona import MODE_DEFAULT_CAST, all_personas, load_custom
    from .core.paths import personas_dir

    custom = load_custom([personas_dir(), find_project_root() / ".nexus" / "personas"])
    personas = list(all_personas()) + list(custom.values())
    if args.json:
        print(json.dumps([p.to_dict() for p in personas], ensure_ascii=False, indent=2))
        return 0
    print(f"{'key':<14} {'name':<10} {'role':<34} {'model':<9} temp  tools")
    for p in personas:
        marker = "*" if p.key in custom else " "
        tools = "all" if p.tools is None else str(len(p.tools))
        print(f"{p.key + marker:<14} {(p.emoji + ' ' + p.name).strip():<10} {p.role[:34]:<34} "
              f"{p.model_pref:<9} {p.temperature:<5} {tools}")
    print("\n* = custom (from ~/.nexus/personas)")
    print("\nswarm modes:")
    for mode, cast in MODE_DEFAULT_CAST.items():
        print(f"  {mode:<10} {', '.join(cast)}")
    return 0


def cmd_sessions(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(prog="nexus sessions", description="List saved sessions.")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--all-dirs", action="store_true", help="do not filter by the current directory")
    parser.add_argument("--json", action="store_true")
    add_common_flags(parser)
    args = parser.parse_args(argv)
    from .core.session import SessionStore

    store = SessionStore()
    metas = store.list(limit=args.limit, cwd=None if args.all_dirs else Path.cwd())
    if args.json:
        print(json.dumps([m.to_dict() for m in metas], ensure_ascii=False, indent=2, default=str))
        return 0
    if not metas:
        print("No sessions found" + ("" if args.all_dirs else " for this directory (try --all-dirs)"))
        return 0
    import time

    print(f"{'id':<26} {'when':<12} {'turns':>5} {'tokens':>8} {'cost':>9}  title")
    for m in metas:
        when = time.strftime("%m-%d %H:%M", time.localtime(m.updated_at or m.started_at))
        print(f"{m.id:<26} {when:<12} {m.turns:>5} {m.input_tokens + m.output_tokens:>8} "
              f"{m.cost_usd:>8.4f}$  {(m.title or '(untitled)')[:52]}")
    print(f"\nstorage: {sessions_dir()}")
    print("resume:  nexus --resume <id>   ·   export: nexus export <id>")
    return 0


def cmd_resume(argv: List[str]) -> int:
    """``nexus resume [id] [-- extra chat flags]`` -> interactive session on that history."""
    parser = argparse.ArgumentParser(prog="nexus resume", add_help=False)
    parser.add_argument("session", nargs="?", default="last")
    known, rest = parser.parse_known_args(argv)
    return chat_main([*rest, "--resume", known.session])


def cmd_export(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(prog="nexus export", description="Export a session as markdown.")
    parser.add_argument("session", nargs="?", default="last")
    parser.add_argument("-o", "--output", default="")
    add_common_flags(parser)
    args = parser.parse_args(argv)
    from .core.session import SessionStore

    store = SessionStore()
    session = store.load(args.session) if args.session != "last" else store.latest(cwd=Path.cwd())
    if session is None:
        print(f"error: session '{args.session}' not found", file=sys.stderr)
        return 2
    text = store.export_markdown(session.meta.id)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        print(text)
    return 0


def cmd_config(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(prog="nexus config", description="Inspect or edit configuration.")
    parser.add_argument("action", nargs="?", default="list",
                        choices=["list", "get", "set", "path", "sources", "init", "edit"])
    parser.add_argument("key", nargs="?")
    parser.add_argument("value", nargs="?")
    parser.add_argument("--json", action="store_true")
    add_common_flags(parser)
    args = parser.parse_args(argv)
    settings = settings_from_args(args)
    from .app import _get_path, _set_path, _MISSING

    if args.action == "path":
        print(config_file())
        return 0
    if args.action == "sources":
        sources = config_sources(settings)
        if not sources:
            print("(defaults only -- no config file, environment variable or flag overrode anything)")
        for source in sources:
            print(source)
        return 0
    if args.action == "init":
        path = config_file()
        if path.is_file() and not os.environ.get("NEXUS_FORCE"):
            print(f"{path} already exists (set NEXUS_FORCE=1 to overwrite)")
            return 1
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(settings.to_dict(), indent=2) + "\n", encoding="utf-8")
        print(f"wrote {path}")
        return 0
    if args.action == "edit":
        editor = os.environ.get("EDITOR") or ("notepad" if os.name == "nt" else "vi")
        path = config_file()
        if not path.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(settings.to_dict(), indent=2) + "\n", encoding="utf-8")
        return subprocess.call([editor, str(path)])
    if args.action == "list":
        payload = settings.to_dict()
        print(json.dumps(payload, indent=2, ensure_ascii=False) if args.json
              else _as_yaml_ish(payload))
        return 0
    if not args.key:
        print("error: this action needs a key", file=sys.stderr)
        return 2
    if args.action == "get":
        value = _get_path(settings, args.key)
        if value is _MISSING:
            print(f"error: unknown setting '{args.key}'", file=sys.stderr)
            return 2
        print(json.dumps(value, ensure_ascii=False, indent=2) if args.json or isinstance(value, (dict, list))
              else str(value))
        return 0
    if args.action == "set":
        if args.value is None:
            print("error: set needs KEY VALUE", file=sys.stderr)
            return 2
        try:
            parsed = json.loads(args.value)
        except json.JSONDecodeError:
            parsed = args.value
        try:
            _set_path(settings, args.key, parsed)
            settings.validate()
        except (NexusError, AttributeError, TypeError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        path = config_file()
        data = {}
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8") or "{}")
            except json.JSONDecodeError:
                data = {}
        node = data
        parts = args.key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                print(f"error: cannot nest under '{part}'", file=sys.stderr)
                return 2
        node[parts[-1]] = parsed
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"{args.key} = {json.dumps(parsed, ensure_ascii=False)}  -> {path}")
        return 0
    return 2


def _as_yaml_ish(data: Any, indent: int = 0) -> str:
    pad = "  " * indent
    lines: List[str] = []
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, (dict, list)) and value:
                lines.append(f"{pad}{key}:")
                lines.append(_as_yaml_ish(value, indent + 1))
            else:
                lines.append(f"{pad}{key}: {json.dumps(value, ensure_ascii=False)}")
    elif isinstance(data, list):
        for item in data:
            lines.append(f"{pad}- {json.dumps(item, ensure_ascii=False)}")
    else:
        lines.append(f"{pad}{json.dumps(data, ensure_ascii=False)}")
    return "\n".join(l for l in lines if l)


def cmd_auth(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(prog="nexus auth", description="Manage stored API keys.")
    parser.add_argument("action", nargs="?", default="list", choices=["list", "login", "logout", "status"])
    parser.add_argument("provider", nargs="?")
    parser.add_argument("key", nargs="?")
    parser.add_argument("--json", action="store_true")
    add_common_flags(parser)
    args = parser.parse_args(argv)
    auth = load_auth()
    if args.action in ("list", "status"):
        rows = []
        for key in sorted(PROVIDERS):
            spec = PROVIDERS[key]
            env_hit = next((e for e in spec.env_keys if os.environ.get(e)), "")
            rows.append({"provider": key, "stored": key in auth, "env": env_hit,
                         "ready": bool(env_hit or auth.get(key) or spec.local),
                         "env_vars": list(spec.env_keys)})
        if args.json:
            print(json.dumps(rows, ensure_ascii=False, indent=2))
            return 0
        for row in rows:
            if row["ready"] or row["stored"] or row["env"]:
                source = f"env:{row['env']}" if row["env"] else ("stored" if row["stored"] else "local")
                print(f"  {row['provider']:<12} {source}")
        if not any(r["ready"] for r in rows):
            print("\nno credentials yet. Try: nexus auth login openai <key>")
        return 0
    if not args.provider:
        print("error: provider required", file=sys.stderr)
        return 2
    provider = args.provider.lower()
    if provider not in PROVIDERS:
        print(f"error: unknown provider '{provider}' (see `nexus providers`)", file=sys.stderr)
        return 2
    if args.action == "logout":
        removed = remove_auth_key(provider)
        print(f"removed stored key for {provider}" if removed else f"no stored key for {provider}")
        return 0 if removed else 1
    secret = args.key
    if secret is None:
        try:
            import getpass

            secret = getpass.getpass(f"{provider} API key: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 130
    if not secret:
        print("error: empty key", file=sys.stderr)
        return 2
    path = save_auth_key(provider, secret)
    print(f"stored {provider} key in {path} (0600)")
    return 0


def cmd_mcp(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(prog="nexus mcp", description="Manage MCP servers.")
    parser.add_argument("action", nargs="?", default="list", choices=["list", "add", "remove", "test"])
    parser.add_argument("name", nargs="?")
    parser.add_argument("rest", nargs="*")
    parser.add_argument("--json", action="store_true")
    add_common_flags(parser)
    args = parser.parse_args(argv)
    settings = settings_from_args(args)
    path = config_file()
    data: Dict[str, Any] = {}
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8") or "{}")
        except json.JSONDecodeError as exc:
            print(f"error: {path} is not valid JSON: {exc}", file=sys.stderr)
            return 2
    servers = data.setdefault("mcp", {}).setdefault("servers", {})

    if args.action == "add":
        if not args.name:
            print("error: nexus mcp add <name> -- <command> [args…]", file=sys.stderr)
            return 2
        rest = list(args.rest)
        if rest and rest[0] == "--":
            rest = rest[1:]
        if not rest:
            print("error: no command given (use -- before the command)", file=sys.stderr)
            return 2
        servers[args.name] = {"command": rest[0], "args": rest[1:]}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"added MCP server '{args.name}': {' '.join(rest)}")
        print("restart nexus to connect (or run `nexus mcp test " + args.name + "`)")
        return 0
    if args.action == "remove":
        if args.name in servers:
            del servers[args.name]
            path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print(f"removed '{args.name}'")
            return 0
        print(f"error: no server named '{args.name}'", file=sys.stderr)
        return 2
    if args.action == "test":
        from .mcp.client import MCPClient

        name = args.name or next(iter(servers), "")
        spec = servers.get(name)
        if not spec:
            print(f"error: no MCP server named '{name}'", file=sys.stderr)
            return 2
        client = MCPClient(name, spec.get("command", ""), list(spec.get("args") or []),
                           env=dict(spec.get("env") or {}))
        ok = client.start()
        if not ok:
            print(f"FAILED: {client.last_error}")
            return 1
        print(f"connected to {client.server_info.get('name', name)} "
              f"{client.server_info.get('version', '')}".strip())
        for tool_name in client.tool_names():
            print(f"  - {tool_name}")
        client.close()
        return 0
    if args.json:
        print(json.dumps(servers, ensure_ascii=False, indent=2))
        return 0
    if not servers:
        print("no MCP servers configured.")
        print("add one:  nexus mcp add files -- npx -y @modelcontextprotocol/server-filesystem /path")
        return 0
    for name, spec in servers.items():
        command = " ".join([spec.get("command", "?"), *(spec.get("args") or [])])
        print(f"  {name:<16} {command}")
    return 0


def cmd_plugins(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(prog="nexus plugins", description="List installed plugins.")
    parser.add_argument("--json", action="store_true")
    add_common_flags(parser)
    args = parser.parse_args(argv)
    from .plugins.loader import load_plugins
    from .tools import build_registry

    registry = build_registry()
    dirs = [home() / "plugins", find_project_root() / ".nexus" / "plugins"]
    report = load_plugins(registry, dirs=dirs)
    if args.json:
        print(json.dumps({"plugins": report.names, "tools": report.tools, "personas": report.personas,
                          "commands": [c.get("name") for c in report.commands],
                          "errors": report.errors}, ensure_ascii=False, indent=2))
        return 0
    print("plugin directories:")
    for directory in dirs:
        exists = "found" if directory.is_dir() else "missing"
        print(f"  {directory}  ({exists})")
    print(f"\n{report.summary()}")
    for name in report.names:
        print(f"  ✓ {name}")
    for error in report.errors:
        print(f"  ✗ {error}")
    return 0 if not report.errors else 1


def cmd_doctor(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(prog="nexus doctor", description="Diagnose the installation.")
    parser.add_argument("--json", action="store_true")
    add_common_flags(parser)
    args = parser.parse_args(argv)
    app = build_app(args, quiet=True)
    style = Style.create(app.settings.ui.theme, enabled=None if app.settings.ui.color else False)
    checks: List[Dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "ok": ok, "detail": detail})

    add("python >= 3.9", sys.version_info >= (3, 9), sys.version.split()[0])
    add("workspace", os.access(app.workspace_root, os.W_OK), str(app.workspace_root))
    add("data dir", data_dir().parent.is_dir(), str(data_dir()))
    add("log dir", logs_dir().parent.is_dir(), str(logs_dir()))
    add("config file", True, str(config_file()) + ("" if config_file().is_file() else "  (not created yet -- defaults)"))
    add("git", bool(_which("git")), _which("git") or "not on PATH")
    ready = [p.key for p in available_providers(include_local=True, probe_local=True)]
    add("providers ready", bool(ready), ", ".join(ready) or "none")
    add("credentials stored", bool(load_auth()), ", ".join(sorted(load_auth())) or "none")
    try:
        target = app.router.resolve()
        app.router.provider(target.provider_key)
        add("default model", True, target.label)
    except Exception as exc:
        add("default model", False, str(exc)[:120])
    add("tools", len(app.registry.names()) >= 15, f"{len(app.registry.names())} registered")
    add("permission mode", app.permissions.mode in ("read-only", "suggest", "auto-edit", "full-auto", "yolo"),
        app.permissions.mode)
    add("session writable", app.session.persist, str(app.session.path))
    add("checkpoints", app.checkpoints.enabled, str(app.checkpoints.base))
    add("plugins", not bool(app.plugin_tools) or True, ", ".join(app.plugin_tools) or "none")
    network = _network_probe()
    add("network", network.startswith("ok"), network)
    app.shutdown()
    if args.json:
        print(json.dumps({"checks": checks, "ok": all(c["ok"] for c in checks)}, ensure_ascii=False, indent=2))
        return 0 if all(c["ok"] for c in checks) else 1
    width = max(len(c["check"]) for c in checks) + 2
    for check in checks:
        icon = style.paint("success" if check["ok"] else "error", "✓" if check["ok"] else "✗")
        print(f"  {icon} {check['check']:<{width}} {style.dim(check['detail'])}")
    failed = [c for c in checks if not c["ok"]]
    print()
    if failed:
        print(f"{len(failed)} check(s) failed. Next: nexus auth login <provider>, or nexus config init")
        return 1
    print("all checks passed.")
    return 0


def _which(name: str) -> str:
    import shutil

    return shutil.which(name) or ""


def _network_probe() -> str:
    import socket

    for host in ("api.openai.com", "api.anthropic.com", "generativelanguage.googleapis.com"):
        try:
            socket.create_connection((host, 443), timeout=2).close()
            return f"ok ({host})"
        except OSError:
            continue
    return "no outbound HTTPS"


def cmd_selftest(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(prog="nexus selftest", description="Run the built-in test suite.")
    parser.add_argument("patterns", nargs="*", help="only run matching test files")
    parser.add_argument("-v", "--verbose", action="store_true")
    args, _unknown = parser.parse_known_args(argv)
    root = Path(__file__).resolve().parent.parent
    runner = root / "tests" / "run_tests.py"
    if not runner.is_file():
        print(f"error: test suite not found at {runner}", file=sys.stderr)
        print("hint: run it from a source checkout, not a packaged install", file=sys.stderr)
        return 2
    extra = ["-v"] if args.verbose else []
    return subprocess.call([sys.executable, str(runner), *extra, *args.patterns], cwd=str(root))


def cmd_demo(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(prog="nexus demo",
                                     description="Offline guided tour using the built-in mock provider.")
    parser.add_argument("--swarm", action="store_true", help="include a swarm demo")
    add_common_flags(parser)
    args = parser.parse_args(argv)
    from .app import App

    os.environ["NEXUS_HOME"] = os.environ.get("NEXUS_HOME") or str(Path(home()))
    settings = settings_from_args(args)
    settings.default_provider = "mock"
    settings.default_model = "mock-1"
    # The tour is non-interactive: edits must not block on an approval prompt.
    settings.approval_mode = "full-auto"
    if "mock" not in settings.providers:
        settings.providers["mock"] = ProviderConfig()
    tmp = Path(_demo_workspace())
    app = App(settings=settings, cwd=tmp, quiet=False, enable_plugins=False, enable_mcp=False)
    app.renderer.println(style_box(app.style, "NEXUS DEMO", [
        "No API key needed: this tour uses the built-in deterministic mock provider.",
        f"Workspace: {tmp}",
        "Everything you see is real NEXUS behaviour -- tools, checkpoints, context",
        "budgeting and the swarm scheduler all run for real.",
        "Approval mode is full-auto here so nothing blocks; in normal use a write",
        "shows an approval prompt with y/n/always/never.",
    ]))
    app.renderer.blank()
    _demo_script(app, include_swarm=bool(args.swarm))
    app.shutdown()
    return 0


def _demo_workspace() -> str:
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="nexus-demo-"))
    (tmp / "src").mkdir()
    (tmp / "src" / "calc.py").write_text("def add(a, b):\n    return a + b\n\n\ndef div(a, b):\n    return a / b\n")
    (tmp / "tests").mkdir()
    (tmp / "tests" / "test_calc.py").write_text("from src.calc import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n")
    (tmp / "README.md").write_text("# demo\n\nA tiny project used by `nexus demo`.\n")
    return str(tmp)


def _demo_script(app, *, include_swarm: bool) -> None:
    from .agents.persona import get_persona
    from .providers.mock import MockProvider

    style = app.style
    provider = app.router.provider("mock")
    assert isinstance(provider, MockProvider)

    app.renderer.println(style_box(style, "1 · tools + checkpoints", [
        "The agent reads and edits real files. Every write is snapshotted first,",
        "so /undo restores the previous bytes and removes files it created.",
    ]))
    provider.queue(
        {"text": "Let me look at the code first.",
         "tool_calls": [("read_file", {"path": "src/calc.py"})]},
        {"text": "`div` has no zero guard. Fixing it now.",
         "tool_calls": [("edit_file", {
             "path": "src/calc.py",
             "old_text": "def div(a, b):\n    return a / b",
             "new_text": "def div(a, b):\n    if b == 0:\n        raise ValueError('division by zero')\n    return a / b"})]},
        {"text": "Fixed and verified by reading the file back.",
         "tool_calls": [("read_file", {"path": "src/calc.py"})]},
        {"text": "Done.\n\n```report\nstatus: done\nsummary: added a zero guard to div\n"
                 "artifacts: src/calc.py\nfollowups: add a regression test\nconfidence: high\n```"},
    )
    app.submit("Fix the division-by-zero bug in src/calc.py")
    guard = "division by zero" in (app.services.cwd / "src" / "calc.py").read_text()
    app.renderer.println(style.paint("success" if guard else "error",
                                     f"  -> file on disk contains the guard: {guard}"))
    app.renderer.println(style.dim("  -> /diff shows the change, /undo reverts it"))
    app.renderer.blank()

    app.renderer.println(style_box(style, "2 · context budget", [
        "History is measured before every request and reduced in a graded ladder.",
        "If it still does not fit, NEXUS says so instead of silently truncating.",
    ]))
    app.cmd_context([], "")
    app.renderer.blank()

    if include_swarm:
        app.renderer.println(style_box(style, "3 · swarm (hive)", [
            "Atlas plans the task board, specialists execute in parallel waves,",
            "Sable gates the result, Atlas integrates. Each has its own context window.",
        ]))
        app.settings.swarm.cast = ["architect", "implementer", "tester", "docs", "reviewer"]
        app.settings.swarm.max_parallel = 3
        state = {"planned": False}

        def persona_of(messages) -> str:
            system = messages[0].text if messages and messages[0].role == "system" else ""
            for key in ("orchestrator", "architect", "implementer", "reviewer", "tester",
                        "docs", "security", "debugger"):
                if get_persona(key).name in system:
                    return key
            return "main"

        def dispatcher(messages):
            who = persona_of(messages)
            user = next((m.text for m in reversed(messages) if m.role == "user"), "")
            if who == "orchestrator":
                if not state["planned"]:
                    state["planned"] = True
                    return {"text": "", "tool_calls": [
                        ("swarm_task", {"action": "add", "title": "Add a regression test for div() zero guard"}),
                        ("swarm_task", {"action": "add", "title": "Document div() behaviour in the README"})]}
                return ("## Result\n\n`div()` now raises `ValueError` on a zero divisor. "
                        "Probe added `tests/test_calc.py::test_div_zero` and Quill documented the "
                        "contract in `README.md`. Sable found no blockers.\n\n"
                        "```report\nstatus: done\nsummary: div() guarded, tested and documented\n"
                        "artifacts: src/calc.py, tests/test_calc.py, README.md\n"
                        "followups: none\nconfidence: high\n```")
            task_id = ""
            for line in user.split("\n"):
                if line.startswith("YOUR TASK ("):
                    task_id = line.split("(")[1].split(")")[0]
            done = {
                "tester": ("Wrote `tests/test_calc.py::test_div_zero` asserting `ValueError`; "
                           "ran it and it passes."),
                "docs": "Added a 'Error handling' section to README.md documenting the ValueError contract.",
                "implementer": "Reviewed src/calc.py; the guard is in place, no further change needed.",
                "architect": "Contract confirmed: div(a, b) raises ValueError on b == 0.",
            }
            summary = done.get(who, "Work complete.")
            if who == "reviewer":
                return ("Checked the guard, the new test and the README section. "
                        "No blockers: the error type is explicit and the test would fail "
                        "if the guard were removed.")
            return {"text": summary + "\n\n```report\nstatus: done\nsummary: " + summary[:80]
                            + "\nartifacts: none\nfollowups: none\nconfidence: high\n```",
                    "tool_calls": [("swarm_task", {"action": "complete", "task_id": task_id,
                                                   "result": summary[:200]})]}

        provider.dispatcher = dispatcher
        app.run_swarm("Make div() safe, tested and documented")
        provider.dispatcher = None
        app.renderer.blank()

    app.renderer.println(style_box(style, "4 · what just happened under the hood", [
        "27 providers through 3 wire protocols (OpenAI-compatible, Anthropic, Gemini)",
        "21 tools, JSON-Schema validated and coerced before they run",
        "permission engine: 5 modes + allow/deny/ask rules + full audit trail",
        "checkpoints behind every mutation, context budgeting before every request",
        "sessions persisted as JSONL with crash recovery, MCP servers, plugins",
    ]))
    app.renderer.blank()
    app.renderer.println(style_box(style, "next steps", [
        "1. Add a real key:   nexus auth login anthropic   (or openai / gemini / groq / deepseek …)",
        "2. Verify setup:     nexus doctor",
        "3. Start working:    nexus",
        "4. Read the guide:   /help  and  /help swarm   inside a session",
        "5. Run the suite:    nexus selftest        (554 offline tests)",
    ]))


def style_box(style: Style, title: str, lines: Sequence[str]) -> str:
    from .ui.widgets import box

    return box(title, list(lines), style)


def cmd_init(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(prog="nexus init",
                                     description="Create a project .nexus/ configuration.")
    parser.add_argument("--agents", action="store_true", help="also write an AGENTS.md template")
    parser.add_argument("--force", action="store_true")
    add_common_flags(parser)
    args = parser.parse_args(argv)
    root = find_project_root(Path(args.cwd).expanduser() if args.cwd else Path.cwd())
    nexus_dir = root / ".nexus"
    config_path = nexus_dir / "config.json"
    if config_path.exists() and not args.force:
        print(f"{config_path} already exists (use --force to overwrite)")
        return 1
    nexus_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "default_model": "",
        "approval_mode": "auto-edit",
        "permissions": {"allow": ["read_file:*", "grep:*", "find_files:*", "list_dir:*", "git:status",
                                  "git:diff", "git:log"],
                        "deny": ["bash:rm -rf *", "write_file:.env*"]},
        "project_context": {"enabled": True, "tree_depth": 3, "max_files": 300},
        "swarm": {"mode": "hive", "max_parallel": 4, "max_rounds": 3},
        "compaction": {"enabled": True, "threshold": 0.82},
        "tools": {"disabled": []},
        "mcp": {"servers": {}},
    }
    config_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {config_path}")
    gitignore = root / ".gitignore"
    entry = ".nexus/checkpoints/"
    if gitignore.is_file():
        text = gitignore.read_text(encoding="utf-8")
        if entry not in text:
            gitignore.write_text(text.rstrip("\n") + f"\n{entry}\n", encoding="utf-8")
            print(f"added {entry} to .gitignore")
    else:
        gitignore.write_text(entry + "\n", encoding="utf-8")
        print(f"wrote {gitignore}")
    if args.agents:
        agents = root / "AGENTS.md"
        if not agents.exists() or args.force:
            agents.write_text(_AGENTS_TEMPLATE, encoding="utf-8")
            print(f"wrote {agents}")
    print("\nnext: nexus auth login <provider>   then   nexus")
    return 0


_AGENTS_TEMPLATE = """# Project guide for AI agents

NEXUS reads this file automatically at session start. Keep it short and factual.

## Commands
- test: `<how to run the tests>`
- lint: `<how to lint>`
- build: `<how to build>`

## Conventions
- <language style, formatting rules, import order>
- <naming rules>

## Architecture
- <the 3-6 modules that matter and what each owns>

## Do not touch
- <generated files, vendor directories, migrations already applied>

## Definition of done
- <tests pass, linter clean, docs updated, no new warnings>
"""


def cmd_log(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(prog="nexus log",
                                     description="Show the log file location and its last lines.")
    parser.add_argument("lines", nargs="?", type=int, default=25, help="how many lines to show")
    parser.add_argument("--path", action="store_true", help="print only the log file path")
    parser.add_argument("-f", "--follow", action="store_true", help="tail -f the log")
    args = parser.parse_args(argv)
    path = logs_dir() / "nexus.log"
    if args.path:
        print(path)
        return 0
    print(str(path))
    if not path.is_file():
        print("(no log file yet)")
        return 0
    if args.follow:
        import time as _time

        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                fh.seek(0, 2)
                while True:
                    line = fh.readline()
                    if not line:
                        _time.sleep(0.3)
                        continue
                    print(line.rstrip(), flush=True)
        except KeyboardInterrupt:
            return 130
        except OSError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()[-max(1, args.lines):]
    except OSError as exc:
        print(f"error: cannot read log: {exc}", file=sys.stderr)
        return 2
    for line in lines:
        print(line)
    return 0


def cmd_version(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(prog="nexus version")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    info = {"name": "nexus", "version": __version__, "python": sys.version.split()[0],
            "platform": sys.platform, "providers": len(PROVIDERS), "catalogue_models": len(MODEL_CATALOG),
            "home": str(home()), "data": str(data_dir())}
    if args.json:
        print(json.dumps(info, ensure_ascii=False, indent=2))
    else:
        print(f"nexus {__version__}  python {info['python']}  {info['platform']}")
        print(f"providers {info['providers']} · catalogue models {info['catalogue_models']}")
        print(f"home {info['home']}")
        print(f"data {info['data']}")
    return 0


SUBCOMMAND_HANDLERS: Dict[str, Any] = {
    "swarm": cmd_swarm, "debate": cmd_debate, "agent": cmd_agent, "run": cmd_run,
    "models": cmd_models, "providers": cmd_providers, "tools": cmd_tools, "personas": cmd_personas,
    "sessions": cmd_sessions, "resume": cmd_resume, "export": cmd_export, "config": cmd_config,
    "auth": cmd_auth, "mcp": cmd_mcp, "plugins": cmd_plugins, "doctor": cmd_doctor,
    "selftest": cmd_selftest, "demo": cmd_demo, "init": cmd_init, "version": cmd_version,
    "log": cmd_log,
}


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
