"""File checkpoints -- the safety net behind ``/undo``.

Every mutating tool snapshots the files it is about to touch *before* touching
them. A checkpoint records both the previous bytes and whether the file existed
at all, so restoring can also delete files that were created afterwards (which
is what users expect ``undo`` to mean).

Storage::

    <workspace>/.nexus/checkpoints/<session>/cp-0001/
        manifest.json
        files/<relative path>          # byte-identical copy
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .logging_ import get_logger

MAX_SNAPSHOT_BYTES = 8 * 1024 * 1024
MAX_CHECKPOINTS = 200


@dataclass
class CheckpointFile:
    rel: str
    existed: bool
    sha: str = ""
    size: int = 0
    stored: bool = True  # False when the file was too large / unreadable


@dataclass
class Checkpoint:
    seq: int
    reason: str
    ts: float
    agent: str = "main"
    files: List[CheckpointFile] = field(default_factory=list)
    dir: Path = field(default_factory=Path)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["dir"] = str(self.dir)
        return data


@dataclass
class RestoreResult:
    restored: List[str] = field(default_factory=list)
    deleted: List[str] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    seq: int = 0

    @property
    def ok(self) -> bool:
        return not self.errors

    def summary(self) -> str:
        parts = []
        if self.restored:
            parts.append(f"restored {len(self.restored)} file(s)")
        if self.deleted:
            parts.append(f"deleted {len(self.deleted)} created file(s)")
        if self.skipped:
            parts.append(f"skipped {len(self.skipped)}")
        if self.errors:
            parts.append(f"{len(self.errors)} error(s)")
        return ", ".join(parts) or "nothing to do"


class CheckpointStore:
    def __init__(self, workspace_root: Path, session_id: str, *, enabled: bool = True,
                 max_file_bytes: int = MAX_SNAPSHOT_BYTES, log: Any = None) -> None:
        self.root = Path(workspace_root)
        self.session_id = session_id
        self.enabled = enabled
        self.max_file_bytes = max_file_bytes
        self.log = log or get_logger()
        self.base = self.root / ".nexus" / "checkpoints" / session_id
        self._seq = self._discover_seq()
        # Swarm workers snapshot concurrently; the sequence counter must be atomic
        # or two checkpoints would collide on the same directory.
        self._lock = threading.Lock()

    # -- internals --------------------------------------------------------
    def _discover_seq(self) -> int:
        if not self.base.is_dir():
            return 0
        highest = 0
        for entry in self.base.iterdir():
            if entry.is_dir() and entry.name.startswith("cp-"):
                try:
                    highest = max(highest, int(entry.name[3:]))
                except ValueError:
                    continue
        return highest

    def _dir_for(self, seq: int) -> Path:
        return self.base / f"cp-{seq:04d}"

    # -- api --------------------------------------------------------------
    def snapshot(self, paths: Sequence[Path], reason: str, agent: str = "main") -> Optional[Checkpoint]:
        if not self.enabled or not paths:
            return None
        with self._lock:
            self._seq += 1
            seq = self._seq
        cp_dir = self._dir_for(seq)
        files_dir = cp_dir / "files"
        entries: List[CheckpointFile] = []
        try:
            files_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self.log.warning("checkpoint: cannot create dir", error=str(exc))
            with self._lock:
                self._seq -= 1
            return None
        for path in paths:
            p = Path(path)
            try:
                rel = str(p.resolve().relative_to(self.root.resolve()))
            except (ValueError, OSError):
                rel = p.name  # outside workspace: store by name only
            exists = p.is_file()
            entry = CheckpointFile(rel=rel, existed=exists)
            if exists:
                try:
                    data = p.read_bytes()
                    entry.size = len(data)
                    entry.sha = hashlib.sha256(data).hexdigest()[:16]
                    if len(data) > self.max_file_bytes:
                        entry.stored = False
                    else:
                        dest = files_dir / rel
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        dest.write_bytes(data)
                except OSError as exc:
                    entry.stored = False
                    self.log.warning("checkpoint: read failed", path=rel, error=str(exc))
            entries.append(entry)
        cp = Checkpoint(seq=seq, reason=reason, ts=time.time(), agent=agent, files=entries, dir=cp_dir)
        try:
            (cp_dir / "manifest.json").write_text(json.dumps(cp.to_dict(), indent=2), encoding="utf-8")
        except OSError as exc:
            self.log.warning("checkpoint: manifest write failed", error=str(exc))
            return None
        # Pruning scans the directory, so doing it on every snapshot would make
        # long sessions O(n^2). Every 16th snapshot is plenty.
        if seq % 16 == 0:
            self.prune()
        self.log.debug("checkpoint", seq=seq, files=len(entries), reason=reason)
        return cp

    def list(self) -> List[Checkpoint]:
        out: List[Checkpoint] = []
        if not self.base.is_dir():
            return out
        for entry in sorted(self.base.iterdir()):
            cp = self._load(entry)
            if cp:
                out.append(cp)
        return out

    def latest(self) -> Optional[Checkpoint]:
        cps = self.list()
        return cps[-1] if cps else None

    def get(self, seq: int) -> Optional[Checkpoint]:
        return self._load(self._dir_for(seq))

    def _load(self, cp_dir: Path) -> Optional[Checkpoint]:
        manifest = cp_dir / "manifest.json"
        if not manifest.is_file():
            return None
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        files = [CheckpointFile(**{k: v for k, v in f.items() if k in CheckpointFile.__dataclass_fields__})
                 for f in data.get("files") or []]
        return Checkpoint(seq=int(data.get("seq") or 0), reason=str(data.get("reason") or ""),
                          ts=float(data.get("ts") or 0), agent=str(data.get("agent") or "main"),
                          files=files, dir=cp_dir)

    def restore(self, seq: Optional[int] = None) -> RestoreResult:
        """Restore to the state *before* checkpoint ``seq`` (default: latest).

        Restores every checkpoint from the newest down to ``seq`` so that
        ``restore(1)`` undoes the whole session.
        """
        cps = self.list()
        if not cps:
            return RestoreResult(errors=["no checkpoints recorded for this session"])
        target = seq if seq is not None else cps[-1].seq
        chosen = [c for c in cps if c.seq >= target]
        if not chosen:
            return RestoreResult(errors=[f"checkpoint {target} not found"])
        result = RestoreResult(seq=target)
        for cp in sorted(chosen, key=lambda c: -c.seq):
            self._apply(cp, result)
        return result

    def _apply(self, cp: Checkpoint, result: RestoreResult) -> None:
        files_dir = cp.dir / "files"
        for entry in cp.files:
            target = self.root / entry.rel
            if entry.existed:
                src = files_dir / entry.rel
                if not entry.stored or not src.is_file():
                    result.skipped.append(entry.rel)
                    continue
                try:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, target)
                    result.restored.append(entry.rel)
                except OSError as exc:
                    result.errors.append(f"{entry.rel}: {exc}")
            else:
                # Did not exist at snapshot time -> remove it if the agent created it.
                if target.is_file():
                    try:
                        target.unlink()
                        result.deleted.append(entry.rel)
                    except OSError as exc:
                        result.errors.append(f"{entry.rel}: {exc}")

    def prune(self, keep: int = MAX_CHECKPOINTS) -> int:
        cps = self.list()
        removed = 0
        for cp in cps[:-keep] if len(cps) > keep else []:
            try:
                shutil.rmtree(cp.dir, ignore_errors=True)
                removed += 1
            except OSError:
                pass
        return removed

    def clear(self) -> int:
        if not self.base.is_dir():
            return 0
        shutil.rmtree(self.base, ignore_errors=True)
        self._seq = 0
        return 1

    def disk_usage(self) -> int:
        total = 0
        if not self.base.is_dir():
            return 0
        for dirpath, _dirnames, filenames in os.walk(self.base):
            for name in filenames:
                try:
                    total += (Path(dirpath) / name).stat().st_size
                except OSError:
                    continue
        return total


__all__ = ["CheckpointStore", "Checkpoint", "CheckpointFile", "RestoreResult"]
