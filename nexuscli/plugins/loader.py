"""Plugin loader.

A plugin is a single Python file dropped into ``~/.nexus/plugins/`` or
``<project>/.nexus/plugins/``. It may expose any of:

``TOOLS = [Tool(), ...]``        registered into the tool registry
``PERSONAS = [ {...}, ... ]``    merged into the persona list (dict or Persona)
``def register(registry, ctx)``  full control: register tools, patch anything
``COMMANDS = [ {...} ]``         extra slash commands (name/help/handler)

Plugins run with your user privileges -- they are code, not data. The loader
never lets a broken plugin take the session down: every failure is captured and
reported by ``nexus plugins`` / ``/doctor``.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..agents.persona import BUILTIN, Persona
from ..core.logging_ import get_logger
from ..tools.base import Tool, ToolRegistry

LOADED_MODULES: Dict[str, Any] = {}


@dataclass
class PluginReport:
    #: plugins that were imported successfully (they may still have reported errors)
    names: List[str] = field(default_factory=list)
    #: plugin file names that could not be imported at all
    failed: List[str] = field(default_factory=list)
    tools: List[str] = field(default_factory=list)
    personas: List[str] = field(default_factory=list)
    commands: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.names)

    def summary(self) -> str:
        if not self.names and not self.errors:
            return "no plugins found"
        parts = [f"{len(self.names)} plugin(s)"]
        if self.failed:
            parts.append(f"{len(self.failed)} failed to load")
        if self.tools:
            parts.append(f"{len(self.tools)} tool(s)")
        if self.personas:
            parts.append(f"{len(self.personas)} persona(s)")
        if self.commands:
            parts.append(f"{len(self.commands)} command(s)")
        if self.errors:
            parts.append(f"{len(self.errors)} error(s)")
        return ", ".join(parts)


def _import_module(path: Path) -> Any:
    module_name = f"nexus_plugin_{path.stem}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot create an import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    LOADED_MODULES[module_name] = module
    return module


def load_plugins(registry: ToolRegistry, *, dirs: Sequence[Path] = (),
                 log: Any = None, extra_context: Optional[Dict[str, Any]] = None) -> PluginReport:
    """Import every plugin file and merge what it exposes. Never raises."""
    log = log or get_logger()
    report = PluginReport()
    context = {"registry": registry, "log": log, **(extra_context or {})}
    seen: set = set()
    for directory in dirs:
        directory = Path(directory)
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.py")):
            if path.name.startswith("_") or path.resolve() in seen:
                continue
            seen.add(path.resolve())
            try:
                module = _import_module(path)
            except Exception as exc:
                message = f"{path.name}: {type(exc).__name__}: {exc}"
                report.errors.append(message)
                report.failed.append(path.stem)
                log.error("plugin import failed", plugin=path.name, error=str(exc))
                continue
            report.names.append(path.stem)
            try:
                _absorb(module, registry, report, context, log)
            except Exception as exc:
                message = f"{path.name}: {type(exc).__name__}: {exc}"
                report.errors.append(message)
                log.error("plugin registration failed", plugin=path.name, error=str(exc))
    return report


def _absorb(module: Any, registry: ToolRegistry, report: PluginReport,
            context: Dict[str, Any], log: Any) -> None:
    tools = getattr(module, "TOOLS", None) or []
    if isinstance(tools, Tool):
        tools = [tools]
    for tool in tools:
        if not isinstance(tool, Tool):
            report.errors.append(f"{module.__name__}: TOOLS entry is not a Tool: {tool!r}")
            continue
        try:
            registry.register(tool, replace=True)
            report.tools.append(tool.name)
            log.info("plugin tool registered", tool=tool.name, module=module.__name__)
        except Exception as exc:
            report.errors.append(f"{module.__name__}: cannot register tool {getattr(tool, 'name', '?')}: {exc}")

    personas = getattr(module, "PERSONAS", None) or []
    if isinstance(personas, dict):
        personas = [personas]
    for entry in personas:
        try:
            persona = entry if isinstance(entry, Persona) else Persona.from_dict(dict(entry))
            BUILTIN.setdefault(persona.key, persona)
            report.personas.append(persona.key)
        except (TypeError, ValueError) as exc:
            report.errors.append(f"{module.__name__}: bad persona {entry!r}: {exc}")

    commands = getattr(module, "COMMANDS", None) or []
    for entry in commands:
        if isinstance(entry, dict) and entry.get("name") and callable(entry.get("handler")):
            report.commands.append(entry)
        else:
            report.errors.append(f"{module.__name__}: bad command entry {entry!r}")

    hook = getattr(module, "register", None)
    if callable(hook):
        hook(registry, context)


__all__ = ["load_plugins", "PluginReport", "LOADED_MODULES"]
