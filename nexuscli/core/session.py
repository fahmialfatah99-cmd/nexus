"""Session persistence (JSONL append-only).

Every session is a directory-friendly single file::

    <data>/sessions/<YYYY-MM-DD>/<session-id>.jsonl

Record types: ``meta`` (header, rewritten on close), ``msg`` (a conversation
message), ``turn`` (usage/cost for one model call), ``touch`` (files modified),
``summary`` (compaction note), ``event`` (swarm/permission audit markers).

Append-only JSONL means a crash loses at most the last line, and any line that
does not parse is skipped on load instead of corrupting the session. Images are
*not* persisted (a placeholder is stored) so history files stay small.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from ..providers.base import Message, TextBlock, ToolCall, Usage, content_to_text
from .logging_ import get_logger
from .paths import sessions_dir

SCHEMA = 1
MAX_TITLE_LEN = 80


def new_session_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]


@dataclass
class SessionMeta:
    id: str
    cwd: str
    started_at: float
    updated_at: float = 0.0
    title: str = ""
    model: str = ""
    provider: str = ""
    approval_mode: str = ""
    turns: int = 0
    messages: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    agents: List[str] = field(default_factory=list)
    touched: List[str] = field(default_factory=list)
    schema: int = SCHEMA

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "SessionMeta":
        allowed = set(SessionMeta.__dataclass_fields__)  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in data.items() if k in allowed}
        kwargs.setdefault("id", new_session_id())
        kwargs.setdefault("cwd", os.getcwd())
        kwargs.setdefault("started_at", time.time())
        return SessionMeta(**kwargs)

    @property
    def duration_s(self) -> float:
        return max(0.0, (self.updated_at or self.started_at) - self.started_at)


def message_to_dict(msg: Message) -> Dict[str, Any]:
    data: Dict[str, Any] = {"role": msg.role}
    if isinstance(msg.content, str):
        data["content"] = msg.content
    else:
        blocks = []
        for b in msg.content:
            if isinstance(b, TextBlock):
                blocks.append({"type": "text", "text": b.text})
            else:
                # Images are intentionally not persisted: sessions would balloon
                # and the model already consumed them in the original turn.
                blocks.append({"type": "image_placeholder", "mime": getattr(b, "mime", "image/png")})
        data["content"] = blocks
    if msg.tool_calls:
        data["tool_calls"] = [{"id": tc.id, "name": tc.name, "arguments": tc.arguments,
                              "extra": tc.extra} for tc in msg.tool_calls]
    if msg.tool_call_id:
        data["tool_call_id"] = msg.tool_call_id
    if msg.name:
        data["name"] = msg.name
    if msg.meta:
        safe = {k: v for k, v in msg.meta.items() if _jsonable(v)}
        if safe:
            data["meta"] = safe
    return data


def message_from_dict(data: Dict[str, Any]) -> Optional[Message]:
    role = data.get("role")
    if role not in ("system", "user", "assistant", "tool"):
        return None
    raw = data.get("content", "")
    content: Any
    if isinstance(raw, list):
        blocks = []
        for b in raw:
            if not isinstance(b, dict):
                continue
            if b.get("type") == "text":
                blocks.append(TextBlock(str(b.get("text", ""))))
            elif b.get("type") == "image_placeholder":
                blocks.append(TextBlock(f"[image omitted from session history ({b.get('mime', 'image')})]"))
        content = blocks if blocks else ""
    else:
        content = str(raw)
    calls = []
    for tc in data.get("tool_calls") or []:
        if isinstance(tc, dict) and tc.get("name"):
            calls.append(ToolCall(id=str(tc.get("id") or ""), name=str(tc["name"]),
                                  arguments=str(tc.get("arguments") or "{}"),
                                  extra=tc.get("extra") if isinstance(tc.get("extra"), dict) else {}))
    return Message(role=role, content=content, tool_calls=calls,
                   tool_call_id=data.get("tool_call_id"), name=data.get("name"),
                   meta=data.get("meta") if isinstance(data.get("meta"), dict) else {})


def _jsonable(value: Any) -> bool:
    try:
        json.dumps(value)
        return True
    except (TypeError, ValueError):
        return False


class Session:
    """One conversation, persisted incrementally."""

    def __init__(self, path: Path, meta: SessionMeta, *, log: Any = None, persist: bool = True) -> None:
        self.path = Path(path)
        self.meta = meta
        self.log = log or get_logger()
        self.persist = persist
        self._messages: List[Message] = []
        self._fh = None
        self._closed = False
        if self.persist:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._fh = open(self.path, "a", encoding="utf-8", buffering=1)
            except OSError as exc:
                self.log.warning("session: cannot open file, continuing in memory", error=str(exc))
                self.persist = False

    # -- writing ----------------------------------------------------------
    def append(self, record_type: str, **fields: Any) -> None:
        if not self.persist or self._closed or self._fh is None:
            return
        record = {"t": record_type, "ts": time.time(), **fields}
        try:
            self._fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except (OSError, TypeError, ValueError) as exc:
            self.log.warning("session: write failed", error=str(exc))
            self.persist = False

    def add_message(self, msg: Message) -> None:
        self._messages.append(msg)
        self.meta.messages = len(self._messages)
        self.append("msg", **message_to_dict(msg))
        if not self.meta.title and msg.role == "user":
            self.meta.title = _derive_title(content_to_text(msg.content))

    def add_messages(self, msgs: Iterable[Message]) -> None:
        for m in msgs:
            self.add_message(m)

    def record_turn(self, agent: Any, result: Any) -> None:
        self.meta.turns += 1
        usage = getattr(result, "usage", None) or Usage(requests=0)
        self.meta.input_tokens += usage.input_tokens
        self.meta.output_tokens += usage.output_tokens
        self.meta.cost_usd = round(self.meta.cost_usd + float(getattr(result, "cost", 0.0) or 0.0), 6)
        name = getattr(agent, "name", "main")
        if name not in self.meta.agents:
            self.meta.agents.append(name)
        target = getattr(result, "target", None)
        if target is not None:
            self.meta.provider = getattr(target, "provider_key", self.meta.provider)
            self.meta.model = getattr(target, "model", self.meta.model)
        self.append("turn", agent=name, usage=usage.as_dict(), cost=getattr(result, "cost", 0.0),
                    turns=getattr(result, "turns", 1), finish=getattr(result, "finish_reason", ""),
                    ok=bool(getattr(result, "ok", True)))
        self.touch()

    def note_touched(self, paths: Sequence[str]) -> None:
        for p in paths:
            if p and p not in self.meta.touched:
                self.meta.touched.append(p)
        if len(self.meta.touched) > 500:
            self.meta.touched = self.meta.touched[-500:]

    def note_summary(self, text: str, replaced: int) -> None:
        self.append("summary", text=text[:20_000], replaced=replaced)

    def note_event(self, kind: str, **fields: Any) -> None:
        self.append("event", kind=kind, **fields)

    def touch(self) -> None:
        self.meta.updated_at = time.time()

    def flush_meta(self) -> None:
        self.touch()
        self.append("meta", **self.meta.to_dict())

    # -- reading ----------------------------------------------------------
    @property
    def messages(self) -> List[Message]:
        return list(self._messages)

    def replace_messages(self, msgs: Sequence[Message]) -> None:
        """Used after compaction: rewrite history and record the new baseline."""
        self._messages = list(msgs)
        self.meta.messages = len(self._messages)
        self.append("replace_history", messages=[message_to_dict(m) for m in self._messages])

    def close(self) -> None:
        if self._closed:
            return
        # Flush the closing meta record BEFORE marking closed, otherwise append()
        # short-circuits and the session loses its final totals/agent list.
        self.flush_meta()
        self._closed = True
        if self._fh is not None:
            try:
                self._fh.close()
            finally:
                self._fh = None

    def __enter__(self) -> "Session":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _derive_title(text: str) -> str:
    clean = " ".join((text or "").split())
    return (clean[:MAX_TITLE_LEN] + "…") if len(clean) > MAX_TITLE_LEN else clean


class SessionStore:
    def __init__(self, root: Optional[Path] = None, *, log: Any = None, persist: bool = True) -> None:
        self.root = Path(root) if root else sessions_dir()
        self.log = log or get_logger()
        self.persist = persist

    # -- paths ------------------------------------------------------------
    def path_for(self, session_id: str, when: Optional[float] = None) -> Path:
        day = time.strftime("%Y-%m-%d", time.localtime(when or time.time()))
        return self.root / day / f"{session_id}.jsonl"

    def _find(self, session_id: str) -> Optional[Path]:
        if not self.root.is_dir():
            return None
        for path in sorted(self.root.glob(f"*/{session_id}.jsonl"), reverse=True):
            return path
        for path in sorted(self.root.glob(f"**/{session_id}*.jsonl"), reverse=True):
            return path
        return None

    # -- lifecycle --------------------------------------------------------
    def create(self, *, cwd: Optional[Path] = None, model: str = "", provider: str = "",
               approval_mode: str = "", session_id: str = "") -> Session:
        sid = session_id or new_session_id()
        now = time.time()
        meta = SessionMeta(id=sid, cwd=str(cwd or Path.cwd()), started_at=now, updated_at=now,
                           model=model, provider=provider, approval_mode=approval_mode)
        path = self.path_for(sid, now)
        session = Session(path, meta, log=self.log, persist=self.persist)
        session.append("meta", **meta.to_dict())
        return session

    def load(self, session_id: str) -> Optional[Session]:
        path = self._find(session_id)
        if path is None:
            return None
        return self._read(path)

    def latest(self, cwd: Optional[Path] = None) -> Optional[Session]:
        metas = self.list(limit=25)
        if not metas:
            return None
        if cwd:
            wanted = str(Path(cwd).resolve())
            for m in metas:
                if str(Path(m.cwd).resolve()) == wanted:
                    return self.load(m.id)
        return self.load(metas[0].id)

    def _read(self, path: Path) -> Session:
        meta = SessionMeta(id=path.stem, cwd=os.getcwd(), started_at=path.stat().st_mtime)
        messages: List[Message] = []
        skipped = 0
        # Running totals from `turn` records: they are the source of truth when
        # the process died before the closing `meta` record was written.
        turn_totals = {"turns": 0, "input": 0, "output": 0, "cost": 0.0}
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as exc:
            self.log.warning("session: cannot read", path=str(path), error=str(exc))
            lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1  # torn last line after a crash: skip, do not fail
                continue
            if not isinstance(rec, dict):
                skipped += 1
                continue
            kind = rec.get("t")
            if kind == "meta":
                meta = SessionMeta.from_dict(rec)
            elif kind == "msg":
                msg = message_from_dict(rec)
                if msg is not None:
                    messages.append(msg)
            elif kind == "replace_history":
                rebuilt = [message_from_dict(m) for m in rec.get("messages") or []]
                messages = [m for m in rebuilt if m is not None]
            elif kind == "turn":
                usage = rec.get("usage") if isinstance(rec.get("usage"), dict) else {}
                turn_totals["turns"] += 1  # one record == one model round-trip
                turn_totals["input"] += int(usage.get("input_tokens") or 0)
                turn_totals["output"] += int(usage.get("output_tokens") or 0)
                try:
                    turn_totals["cost"] += float(rec.get("cost") or 0.0)
                except (TypeError, ValueError):
                    pass
        if skipped:
            self.log.info("session: skipped unparseable lines", path=str(path), count=skipped)
        if meta.turns == 0 and turn_totals["turns"]:
            meta.turns = turn_totals["turns"]
            meta.input_tokens = turn_totals["input"]
            meta.output_tokens = turn_totals["output"]
            meta.cost_usd = round(turn_totals["cost"], 6)
        session = Session(path, meta, log=self.log, persist=self.persist)
        session._messages = messages  # restore in-memory history
        session.meta.messages = len(messages)
        if not session.meta.title:
            for m in messages:
                if m.role == "user":
                    session.meta.title = _derive_title(content_to_text(m.content))
                    break
        return session

    def list(self, *, limit: int = 50, cwd: Optional[Path] = None) -> List[SessionMeta]:
        if not self.root.is_dir():
            return []
        out: List[SessionMeta] = []
        wanted = str(Path(cwd).resolve()) if cwd else None
        paths = sorted(self.root.glob("*/*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
        for path in paths:
            try:
                meta = self._read_meta_only(path)
            except OSError:
                continue
            if wanted and str(Path(meta.cwd).resolve()) != wanted:
                continue
            out.append(meta)
            if len(out) >= limit:
                break
        return out

    def _read_meta_only(self, path: Path) -> SessionMeta:
        """Cheap header read: scan from the end for the newest meta record."""
        meta = SessionMeta(id=path.stem, cwd=os.getcwd(), started_at=path.stat().st_mtime)
        try:
            with open(path, "rb") as fh:
                fh.seek(0, os.SEEK_END)
                size = fh.tell()
                fh.seek(max(0, size - 64 * 1024))
                tail = fh.read().decode("utf-8", errors="replace")
        except OSError:
            return meta
        for line in reversed(tail.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict) and rec.get("t") == "meta":
                return SessionMeta.from_dict(rec)
        # fall back to the header
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                first = fh.readline()
            rec = json.loads(first)
            if isinstance(rec, dict) and rec.get("t") == "meta":
                return SessionMeta.from_dict(rec)
        except (OSError, json.JSONDecodeError):
            pass
        return meta

    def delete(self, session_id: str) -> bool:
        path = self._find(session_id)
        if path is None:
            return False
        try:
            path.unlink()
            return True
        except OSError:
            return False

    def prune(self, keep: int = 50) -> int:
        paths = sorted(self.root.glob("*/*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
        removed = 0
        for path in paths[keep:]:
            try:
                path.unlink()
                removed += 1
            except OSError:
                continue
        return removed

    def export_markdown(self, session_id: str) -> str:
        session = self.load(session_id)
        if session is None:
            return ""
        meta = session.meta
        lines = [f"# NEXUS session {meta.id}",
                 f"- started: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(meta.started_at))}",
                 f"- cwd: {meta.cwd}",
                 f"- model: {meta.provider}:{meta.model}" if meta.model else "- model: (unknown)",
                 f"- turns: {meta.turns}  tokens in/out: {meta.input_tokens}/{meta.output_tokens}"
                 f"  cost: ${meta.cost_usd:.4f}",
                 f"- agents: {', '.join(meta.agents) or 'main'}", ""]
        if meta.touched:
            lines += ["## Files touched", *[f"- {p}" for p in meta.touched], ""]
        lines.append("## Transcript")
        for msg in session.messages:
            body = content_to_text(msg.content)
            if msg.role == "assistant" and msg.tool_calls:
                calls = "\n".join(f"  - `{tc.name}({tc.arguments[:200]})`" for tc in msg.tool_calls)
                body = (body + "\n" if body.strip() else "") + "tool calls:\n" + calls
            label = {"system": "SYSTEM", "user": "USER", "assistant": "NEXUS", "tool": "TOOL"}[msg.role]
            prefix = f"**{label}**" + (f" ({msg.name})" if msg.name and msg.role == "tool" else "")
            lines.append(f"\n{prefix}\n\n{body}\n")
        session.close()
        return "\n".join(lines)


__all__ = ["Session", "SessionMeta", "SessionStore", "new_session_id", "message_to_dict",
           "message_from_dict"]
