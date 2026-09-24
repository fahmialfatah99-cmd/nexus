"""The swarm blackboard.

A thread-safe shared space where swarm members exchange messages, publish
artifacts and record decisions. Everything an agent learns that another agent
might need goes here, which is what makes parallel agents converge instead of
duplicating or contradicting each other.

Message kinds:
``result``   finished work product
``finding``  a fact others should know (path, API shape, constraint)
``question`` blocked, needs an answer
``handoff``  asks a specific role to continue
``decision`` a choice that is now binding for the swarm
``note``     anything else
"""

from __future__ import annotations

import itertools
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

KINDS = ("result", "finding", "question", "handoff", "decision", "note")
_ids = itertools.count(1)


@dataclass
class Entry:
    id: int
    ts: float
    sender: str
    kind: str
    content: str
    to: str = ""
    refs: List[str] = field(default_factory=list)
    read_by: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def age(self) -> float:
        return time.time() - self.ts

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "ts": self.ts, "sender": self.sender, "kind": self.kind,
                "content": self.content, "to": self.to, "refs": list(self.refs),
                "read_by": list(self.read_by), "meta": dict(self.meta)}


class Blackboard:
    def __init__(self, *, max_entries: int = 2000) -> None:
        self._entries: List[Entry] = []
        self._artifacts: Dict[str, Dict[str, Any]] = {}
        self._decisions: List[str] = []
        self._lock = threading.RLock()
        self.max_entries = max_entries

    # -- messaging --------------------------------------------------------
    def post(self, *, sender: str, content: str, kind: str = "note", to: str = "",
             refs: Optional[Sequence[str]] = None, meta: Optional[Dict[str, Any]] = None) -> Entry:
        kind = kind if kind in KINDS else "note"
        entry = Entry(id=next(_ids), ts=time.time(), sender=sender, kind=kind,
                      content=(content or "").strip(), to=to or "", refs=list(refs or []),
                      meta=dict(meta or {}))
        with self._lock:
            self._entries.append(entry)
            if len(self._entries) > self.max_entries:
                self._entries = self._entries[-self.max_entries:]
            if kind == "decision":
                self._decisions.append(f"[{sender}] {entry.content}")
        return entry

    def query(self, *, limit: int = 25, kind: str = "", sender: str = "", to: str = "",
              since_id: int = 0, unread_for: str = "") -> List[Entry]:
        with self._lock:
            items = list(self._entries)
        if kind:
            items = [e for e in items if e.kind == kind]
        if sender:
            items = [e for e in items if e.sender == sender]
        if to:
            items = [e for e in items if not e.to or e.to == to]
        if since_id:
            items = [e for e in items if e.id > since_id]
        if unread_for:
            items = [e for e in items if unread_for not in e.read_by and e.sender != unread_for]
        return items[-limit:] if limit else items

    def mark_read(self, reader: str, ids: Iterable[int]) -> None:
        wanted = set(ids)
        with self._lock:
            for e in self._entries:
                if e.id in wanted and reader not in e.read_by:
                    e.read_by.append(reader)

    def unanswered_questions(self, agent: str = "") -> List[Entry]:
        """Questions nobody has answered yet (answers reference the question id)."""
        with self._lock:
            entries = list(self._entries)
        answered = set()
        for e in entries:
            for qid in e.meta.get("answers") or []:
                answered.add(qid)
        out = []
        for e in entries:
            if e.kind != "question" or e.id in answered:
                continue
            if agent and e.to and e.to != agent:
                continue
            out.append(e)
        return out

    # -- artifacts --------------------------------------------------------
    def publish_artifact(self, path: str, *, producer: str, note: str = "") -> None:
        with self._lock:
            prev = self._artifacts.get(path, {"producers": [], "notes": []})
            if producer not in prev["producers"]:
                prev["producers"].append(producer)
            if note and note not in prev["notes"]:
                prev["notes"].append(note)
            prev["updated"] = time.time()
            self._artifacts[path] = prev

    def artifacts(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return {k: dict(v) for k, v in self._artifacts.items()}

    # -- decisions --------------------------------------------------------
    def decide(self, sender: str, decision: str) -> None:
        self.post(sender=sender, content=decision, kind="decision")

    def decisions(self) -> List[str]:
        with self._lock:
            return list(self._decisions)

    # -- rendering --------------------------------------------------------
    def render(self, entries: Optional[Sequence[Entry]] = None, *, width: int = 100) -> str:
        items = list(entries) if entries is not None else self.query(limit=50)
        if not items:
            return "(blackboard is empty)"
        lines = []
        for e in items:
            head = f"#{e.id} [{e.kind}] {e.sender}" + (f" -> {e.to}" if e.to else "")
            body = e.content if len(e.content) <= width * 3 else e.content[: width * 3] + "…"
            lines.append(f"{head}\n  {body}")
            if e.refs:
                lines.append(f"  refs: {', '.join(e.refs[:8])}")
        return "\n".join(lines)

    def digest(self, *, per_agent: int = 6, max_chars: int = 12_000) -> str:
        """Compact view injected into every agent prompt (bounded size)."""
        with self._lock:
            entries = list(self._entries)
        by_sender: Dict[str, List[Entry]] = {}
        for e in entries:
            by_sender.setdefault(e.sender, []).append(e)
        lines: List[str] = []
        for sender, items in by_sender.items():
            recent = items[-per_agent:]
            lines.append(f"### {sender}")
            for e in recent:
                body = " ".join(e.content.split())
                if len(body) > 400:
                    body = body[:400] + "…"
                lines.append(f"- [{e.kind}] {body}")
        arts = self.artifacts()
        if arts:
            lines.append("### Artifacts")
            for path, info in sorted(arts.items())[:40]:
                lines.append(f"- {path} (by {', '.join(info['producers'])})")
        text = "\n".join(lines)
        return text[:max_chars]

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            kinds: Dict[str, int] = {}
            for e in self._entries:
                kinds[e.kind] = kinds.get(e.kind, 0) + 1
            return {"entries": len(self._entries), "by_kind": kinds,
                    "artifacts": len(self._artifacts), "decisions": len(self._decisions),
                    "agents": len({e.sender for e in self._entries})}

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._artifacts.clear()
            self._decisions.clear()


__all__ = ["Blackboard", "Entry", "KINDS"]
