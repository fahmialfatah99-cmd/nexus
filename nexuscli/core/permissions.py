"""Permission engine.

Four approval modes plus a rule list, evaluated in a fixed, explainable order:

1. **deny rules** always win (hard blocks, e.g. ``bash:rm -rf *``)
2. **read-only mode** blocks every tool that is not ``read_only``
3. **allow rules** short-circuit the prompt (what "always allow" writes)
4. **mode policy** decides whether the remaining cases need a human
5. otherwise the injected ``confirmer`` asks the user (blocking) and may
   return "always", which persists a new allow rule

Every decision is logged with the rule/mode that produced it, so ``/audit``
can explain exactly why something ran without asking.
"""

from __future__ import annotations

import fnmatch
import threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from ..core.logging_ import get_logger
from .ignore import is_within

MODES = ("read-only", "suggest", "auto-edit", "full-auto", "yolo")
DEFAULT_MODE = "auto-edit"

RISK_NORMAL, RISK_ELEVATED, RISK_DANGEROUS = "normal", "elevated", "dangerous"


@dataclass
class Decision:
    allowed: bool
    reason: str = ""
    source: str = "mode"  # rule | mode | user | denied
    remember: Optional[str] = None  # rule key to persist when the user chose "always"

    def __bool__(self) -> bool:  # convenience
        return self.allowed


@dataclass
class ConfirmationRequest:
    title: str
    detail: str = ""
    risk: str = RISK_NORMAL
    diff: Optional[str] = None
    key: Optional[str] = None


#: confirmer(request, tool_name, args) -> (allowed, remember)
#: ``remember`` is None, a rule string (persisted as allow), or a
#: ``(kind, rule)`` tuple so "never ask again" can persist a deny rule.
Confirmer = Callable[[ConfirmationRequest, str, Dict[str, Any]], Tuple[bool, Optional[str]]]


class RuleSet:
    """Allow/deny/ask rule lists with ``tool:key`` glob patterns."""

    def __init__(self, allow: Sequence[str] = (), deny: Sequence[str] = (), ask: Sequence[str] = ()) -> None:
        self.allow = [r for r in allow if r]
        self.deny = [r for r in deny if r]
        self.ask = [r for r in ask if r]

    @staticmethod
    def parse(rule: str) -> Tuple[str, str]:
        if ":" in rule:
            tool, _, key = rule.partition(":")
            return tool.strip(), key.strip() or "*"
        return rule.strip(), "*"

    def matches(self, rules: Iterable[str], tool: str, key: str) -> Optional[str]:
        for rule in rules:
            r_tool, r_key = self.parse(rule)
            if r_tool not in ("*", tool):
                continue
            if _key_match(r_key, key):
                return rule
        return None

    def to_dict(self) -> Dict[str, List[str]]:
        return {"allow": list(self.allow), "deny": list(self.deny), "ask": list(self.ask)}

    @staticmethod
    def from_dict(data: Optional[Dict[str, Any]]) -> "RuleSet":
        data = data or {}
        return RuleSet(allow=list(data.get("allow") or []), deny=list(data.get("deny") or []),
                       ask=list(data.get("ask") or []))


def _key_match(pattern: str, key: str) -> bool:
    """Match a rule key against a request key.

    ``*``              -> everything for that tool
    ``git``            -> the command itself and any subcommand ("git commit")
    ``npm *``          -> flat glob: '*' matches anything including '/' and spaces
    ``src/**/*.py``    -> strict path glob: '*' stays inside a segment, '**' spans
    """
    if pattern == "*" or pattern == key:
        return True
    if "**" in pattern:
        # Strict path semantics: '*' does not cross '/', '**' does.
        from .ignore import glob_match

        if glob_match(pattern, key):
            return True
    # Flat semantics (fnmatch): '*' matches anything, including spaces and '/'.
    # This is what people expect for command rules such as "bash:rm -rf *".
    if fnmatch.fnmatchcase(key, pattern):
        return True
    if " " not in pattern and "*" not in pattern and key.startswith(pattern + " "):
        return True
    return False


class PermissionEngine:
    def __init__(
        self,
        *,
        mode: str = DEFAULT_MODE,
        rules: Optional[RuleSet] = None,
        workspace_root: Optional[Path] = None,
        extra_dirs: Sequence[Path] = (),
        confirmer: Optional[Confirmer] = None,
        allow_private_network: bool = False,
        on_persist: Optional[Callable[[str], None]] = None,
        log: Any = None,
    ) -> None:
        self.mode = mode if mode in MODES else DEFAULT_MODE
        self.rules = rules or RuleSet()
        self.workspace_root = Path(workspace_root) if workspace_root else Path.cwd()
        self.extra_dirs = [Path(d) for d in extra_dirs]
        self.confirmer = confirmer
        self.allow_private_network = allow_private_network
        self.on_persist = on_persist
        self.log = log or get_logger()
        self._audit: List[Dict[str, Any]] = []
        self._lock = threading.RLock()
        self.auto_approved = 0
        self.prompted = 0
        self.denied = 0

    # -- configuration ----------------------------------------------------
    def set_mode(self, mode: str) -> str:
        if mode not in MODES:
            raise ValueError(f"Unknown approval mode '{mode}'. Valid: {', '.join(MODES)}")
        self.mode = mode
        return self.mode

    def add_rule(self, kind: str, rule: str, *, persist: bool = True) -> None:
        if kind not in ("allow", "deny", "ask"):
            raise ValueError(f"Unknown rule kind '{kind}'")
        target = getattr(self.rules, kind)
        if rule not in target:
            target.append(rule)
        if persist and self.on_persist:
            self.on_persist(rule)

    def remove_rule(self, kind: str, rule: str) -> bool:
        target = getattr(self.rules, kind, None)
        if not target or rule not in target:
            return False
        target.remove(rule)
        return True

    def outside_workspace_allowed(self, path: Path) -> bool:
        if is_within(self.workspace_root, path):
            return True
        return any(is_within(d, path) for d in self.extra_dirs)

    # -- the decision -----------------------------------------------------
    @staticmethod
    def _subject(tool_name: str, request: Optional[ConfirmationRequest]) -> str:
        """Normalise a request key into the *subject* part used for rule matching.

        Tools build keys as ``"<tool>:<subject>"`` (nice for display); rules are
        written as ``"<tool>:<subject>"`` too, so the engine strips the tool
        prefix once here and both sides compare on the same footing.
        """
        raw = (request.key if request and request.key else "") or "*"
        prefix = tool_name + ":"
        if raw.startswith(prefix):
            raw = raw[len(prefix):] or "*"
        return raw

    def check(self, tool_name: str, args: Dict[str, Any], request: Optional[ConfirmationRequest],
              *, read_only: bool = False) -> Decision:
        key = self._subject(tool_name, request)
        risk = (request.risk if request else RISK_NORMAL) or RISK_NORMAL

        deny = self.rules.matches(self.rules.deny, tool_name, key)
        if deny:
            return self._record(Decision(False, f"blocked by deny rule '{deny}'", "denied"), tool_name, key, risk)

        if self.mode == "read-only" and not read_only:
            return self._record(Decision(False, "read-only mode is active (--read-only)", "denied"),
                                tool_name, key, risk)

        allow = self.rules.matches(self.rules.allow, tool_name, key)
        if allow:
            return self._record(Decision(True, f"allowed by rule '{allow}'", "rule"), tool_name, key, risk)

        ask = self.rules.matches(self.rules.ask, tool_name, key)
        if not ask:
            auto, why = self._mode_policy(read_only, risk)
            if auto:
                return self._record(Decision(True, why, "mode"), tool_name, key, risk)

        # Needs a human.
        if self.confirmer is None:
            return self._record(
                Decision(False, "approval required but no interactive prompt is available "
                                "(non-interactive mode). Re-run with --auto-edit/--full-auto, or add an allow rule.",
                         "denied"),
                tool_name, key, risk)

        req = request or ConfirmationRequest(title=f"{tool_name} {key}", risk=risk, key=f"{tool_name}:{key}")
        with self._lock:
            self.prompted += 1
        try:
            allowed, remember = self.confirmer(req, tool_name, args)
        except KeyboardInterrupt:  # Ctrl+C at a prompt means "no"
            allowed, remember = False, None
        if remember:
            # The confirmer may ask for either kind of rule to be persisted:
            #   "bash:git"            -> allow rule (legacy/simple form)
            #   ("allow", "bash:git") -> explicit kind
            #   ("deny", "bash:rm *") -> explicit deny (the "never ask again" answer)
            if isinstance(remember, (tuple, list)) and len(remember) == 2:
                kind, rule = str(remember[0]), str(remember[1])
                if kind in ("allow", "deny", "ask"):
                    self.add_rule(kind, rule)
            elif allowed:
                self.add_rule("allow", str(remember))
        remembered = None
        if allowed and remember:
            remembered = remember[1] if isinstance(remember, (tuple, list)) and len(remember) == 2 else str(remember)
        return self._record(Decision(bool(allowed), "user decision", "user", remember=remembered),
                            tool_name, key, risk)

    def _mode_policy(self, read_only: bool, risk: str) -> Tuple[bool, str]:
        if read_only:
            return True, "read-only tool (no side effects)"
        if self.mode == "yolo":
            return True, "yolo mode (all approvals skipped)"
        if risk == RISK_DANGEROUS:
            return False, "dangerous operation always requires approval"
        if self.mode == "full-auto":
            return True, "full-auto mode"
        if self.mode == "auto-edit":
            return False, "auto-edit mode only auto-approves read-only tools"
        return False, "suggest mode requires approval for every side effect"

    def _record(self, decision: Decision, tool: str, key: str, risk: str) -> Decision:
        """Single place where counters and the audit trail are updated."""
        import time

        entry = {"ts": time.time(), "tool": tool, "key": key, "risk": risk, "allowed": decision.allowed,
                 "reason": decision.reason, "source": decision.source, "mode": self.mode}
        with self._lock:  # swarm workers share one engine: counters must be atomic
            if decision.allowed:
                if decision.source in ("rule", "mode"):
                    self.auto_approved += 1
            else:
                self.denied += 1
            self._audit.append(entry)
            if len(self._audit) > 5000:
                self._audit = self._audit[-2500:]
        self.log.info("permission", tool=tool, key=key, allowed=decision.allowed, source=decision.source,
                      reason=decision.reason)
        return decision

    # -- introspection ----------------------------------------------------
    def audit(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._audit[-limit:])

    def stats(self) -> Dict[str, int]:
        return {"auto_approved": self.auto_approved, "prompted": self.prompted, "denied": self.denied}

    def describe(self) -> str:
        return (f"mode={self.mode}  allow={len(self.rules.allow)} deny={len(self.rules.deny)} "
                f"ask={len(self.rules.ask)}  auto={self.auto_approved} prompted={self.prompted} denied={self.denied}")


def path_scope_error(path: Path, root: Path) -> str:
    return f"'{path}' is outside the workspace '{root}'."


__all__ = ["PermissionEngine", "RuleSet", "Decision", "ConfirmationRequest", "MODES", "DEFAULT_MODE",
           "RISK_NORMAL", "RISK_ELEVATED", "RISK_DANGEROUS"]
