"""Tool registry assembly.

``build_registry()`` is the single place that decides which tools exist in a
session: built-ins, plugins, MCP servers, and the swarm-only coordination tools.
Feature flags (offline, read-only, disabled lists) are applied here so no other
module has to know about them.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence

from ..core.logging_ import get_logger
from .base import Tool, ToolRegistry
from .builtin import agent as agent_tools
from .builtin import files as file_tools
from .builtin import git as git_tools
from .builtin import memory as memory_tools
from .builtin import search as search_tools
from .builtin import shell as shell_tools
from .builtin import tasks as task_tools
from .builtin import web as web_tools

BUILTIN_BUILDERS = (
    file_tools.build_tools,
    search_tools.build_tools,
    shell_tools.build_tools,
    web_tools.build_tools,
    git_tools.build_tools,
    task_tools.build_tools,
    memory_tools.build_tools,
    agent_tools.build_tools,
)


def build_registry(
    *,
    offline: bool = False,
    include_swarm_tools: bool = False,
    extra_tools: Optional[Iterable[Tool]] = None,
    disabled: Sequence[str] = (),
    only: Optional[Sequence[str]] = None,
    log: Any = None,
) -> ToolRegistry:
    """Create a registry with all built-ins (plus plugins/MCP tools if given)."""
    log = log or get_logger()
    registry = ToolRegistry()
    for builder in BUILTIN_BUILDERS:
        for tool in builder():
            try:
                registry.register(tool)
            except Exception as exc:  # a broken tool must not kill the session
                log.error("failed to register tool", tool=getattr(tool, "name", "?"), error=str(exc))
    if include_swarm_tools:
        for tool in agent_tools.build_swarm_tools():
            registry.register(tool, replace=True)
    for tool in extra_tools or ():
        try:
            registry.register(tool, replace=True)
        except Exception as exc:
            log.error("failed to register extra tool", tool=getattr(tool, "name", "?"), error=str(exc))
    for name in disabled:
        registry.set_enabled(name, False)
    if only:
        keep = set(only)
        for name in registry.names():
            registry.set_enabled(name, name in keep)
    if offline:
        for tool in registry.enabled_tools():
            if tool.needs_network:
                registry.set_enabled(tool.name, False)
    return registry


def tool_catalog() -> List[Dict[str, Any]]:
    """Machine-readable catalogue used by ``nexus tools`` and the docs."""
    registry = build_registry(include_swarm_tools=True)
    out = []
    for tool in registry.enabled_tools():
        out.append({
            "name": tool.name,
            "category": tool.category,
            "description": tool.description.strip().split("\n")[0],
            "read_only": tool.read_only,
            "needs_network": tool.needs_network,
            "parameters": tool.parameters,
        })
    return out


__all__ = ["build_registry", "tool_catalog", "ToolRegistry", "Tool", "BUILTIN_BUILDERS"]
