"""Shared test harness: a recording UI that also defines the UI contract.

The real renderer (nexuscli/ui/render.py) must implement every method this
recorder implements -- that is what keeps the agent loop and the UI decoupled
and lets the whole loop be tested without a terminal.
"""
from __future__ import annotations

import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nexuscli.agents.persona import get_persona
from nexuscli.agents.runtime import Agent, AgentOptions, Services
from nexuscli.core.checkpoints import CheckpointStore
from nexuscli.core.context import ContextManager
from nexuscli.core.permissions import PermissionEngine
from nexuscli.core.router import ModelRouter
from nexuscli.core.taskboard import TaskBoard
from nexuscli.core.usage import UsageLedger
from nexuscli.providers.base import RequestOptions, Usage
from nexuscli.providers.mock import MockProvider
from nexuscli.tools import build_registry


class RecordingUI:
    """Implements the UI protocol used by the agent loop and swarm runner."""

    def __init__(self) -> None:
        self.text: List[str] = []
        self.reasoning: List[str] = []
        self.events: List[tuple] = []
        self.tools_started: List[str] = []
        self.tools_ended: List[tuple] = []
        self.denied: List[tuple] = []
        self.errors: List[str] = []
        self.budgets: List[str] = []
        self.turns: List[tuple] = []
        self.concurrent_peak = 0
        self._live = 0
        self._lock = threading.Lock()

    # -- protocol -------------------------------------------------------
    def on_turn_start(self, agent: str = "", turn: int = 0, model: str = "") -> None:
        self.events.append(("turn_start", agent, turn, model))
        self.turns.append((agent, turn))

    def on_text(self, text: str, agent: str = "") -> None:
        self.text.append(text)

    def on_reasoning(self, text: str, agent: str = "") -> None:
        self.reasoning.append(text)

    def on_tool_call_stream(self, index: int, name: str, agent: str = "") -> None:
        self.events.append(("tool_call_stream", index, name, agent))

    def on_turn_end(self, agent: str = "", turn: int = 0, usage: Any = None, finish_reason: str = "") -> None:
        self.events.append(("turn_end", agent, turn, finish_reason))

    def on_tool_start(self, name: str, args: Dict[str, Any], agent: str = "") -> None:
        with self._lock:
            self._live += 1
            self.concurrent_peak = max(self.concurrent_peak, self._live)
        self.tools_started.append(name)
        self.events.append(("tool_start", name, agent))

    def on_tool_end(self, name: str, result: Any, agent: str = "") -> None:
        with self._lock:
            self._live = max(0, self._live - 1)
        self.tools_ended.append((name, result.is_error))
        self.events.append(("tool_end", name, agent, result.is_error))

    def on_tool_denied(self, name: str, reason: str, agent: str = "") -> None:
        self.denied.append((name, reason))
        self.events.append(("tool_denied", name, agent))

    def on_error(self, message: str, agent: str = "") -> None:
        self.errors.append(message)
        self.events.append(("error", message, agent))

    def on_context_budget(self, note: str, agent: str = "") -> None:
        self.budgets.append(note)

    @property
    def text_joined(self) -> str:
        return "".join(self.text)


def make_env(*, scripts: Optional[List[Any]] = None, mode: str = "full-auto",
             rules: Optional[Dict[str, Any]] = None, confirmer=None,
             persona: str = "main", model_spec: str = "mock:mock-1",
             max_turns: int = 8, offline: bool = False, workspace: Optional[Path] = None,
             tools: Optional[List[str]] = None):
    """Build a fully wired Agent with an offline mock provider."""
    tmp = workspace or Path(tempfile.mkdtemp(prefix="nexus-test-"))
    provider = MockProvider(scripts=scripts or [])
    router = ModelRouter(default_provider="mock", default_model="mock-1")
    router.register_provider("mock", provider)
    registry = build_registry(offline=offline)
    if tools:
        for name in registry.names():
            registry.set_enabled(name, name in tools)
    from nexuscli.core.permissions import RuleSet

    permissions = PermissionEngine(mode=mode, rules=RuleSet.from_dict(rules or {}),
                                  workspace_root=tmp, confirmer=confirmer)
    services = Services(
        registry=registry, permissions=permissions, router=router,
        context=ContextManager(), ledger=UsageLedger(),
        checkpoints=CheckpointStore(tmp, "test-session"), ui=RecordingUI(),
        cwd=tmp, workspace_root=tmp, cancelled=threading.Event(), offline=offline,
    )
    services.vars["board"] = TaskBoard()
    services.vars["config"] = None
    agent = Agent(services=services, persona=get_persona(persona), options=AgentOptions(
        model_spec=model_spec, max_turns=max_turns, temperature=0.1))
    return agent, provider, tmp
