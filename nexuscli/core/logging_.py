"""File-only logging.

NEXUS owns stdout completely (it is a TUI), so *nothing* is ever written there
by the logging layer. Diagnostics go to ``~/.nexus/logs/nexus.log`` with size
based rotation implemented by hand (stdlib ``logging`` handlers are fine, but
we keep a single well understood code path).
"""

from __future__ import annotations

import os
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Optional

from .paths import data_dir

_LEVELS = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}
_LEVEL_NAMES = {10: "DEBUG", 20: "INFO", 30: "WARN", 40: "ERROR", 50: "FATAL"}

_MAX_BYTES = 4 * 1024 * 1024
_BACKUP_COUNT = 3


class Logger:
    def __init__(self, path: Optional[Path] = None, level: str = "INFO", echo: bool = False) -> None:
        self.path = path
        self.level = _LEVELS.get(level.upper(), 20)
        self.echo = echo
        self._lock = threading.Lock()
        self._fh = None
        self.disabled = path is None

    # -- lifecycle ---------------------------------------------------------
    def open(self) -> None:
        if self.disabled or self._fh is not None:
            return
        try:
            assert self.path is not None
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(self.path, "a", encoding="utf-8", buffering=1)
        except OSError:
            # Logging must never break the application.
            self.disabled = True
            self._fh = None

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                try:
                    self._fh.close()
                finally:
                    self._fh = None

    def _rotate_if_needed(self) -> None:
        assert self.path is not None
        try:
            if self.path.stat().st_size < _MAX_BYTES:
                return
        except OSError:
            return
        try:
            if self._fh is not None:
                self._fh.close()
                self._fh = None
            for i in range(_BACKUP_COUNT - 1, 0, -1):
                src = self.path.with_suffix(f".log.{i}")
                dst = self.path.with_suffix(f".log.{i + 1}")
                if src.exists():
                    os.replace(src, dst)
            os.replace(self.path, self.path.with_suffix(".log.1"))
            self._fh = open(self.path, "a", encoding="utf-8", buffering=1)
        except OSError:
            self.disabled = True

    # -- api ---------------------------------------------------------------
    def set_level(self, level: str) -> None:
        self.level = _LEVELS.get(level.upper(), self.level)

    def log(self, level: str, msg: str, *, exc: Optional[BaseException] = None, **fields) -> None:
        num = _LEVELS.get(level.upper(), 20)
        if num < self.level:
            return
        parts = [f"{time.strftime('%Y-%m-%dT%H:%M:%S')}", f"{_LEVEL_NAMES.get(num, level.upper()):5s}", msg]
        if fields:
            parts.append(" ".join(f"{k}={_fmt(v)}" for k, v in fields.items()))
        if exc is not None:
            parts.append("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).rstrip())
        line = " | ".join(parts)
        if self.echo:
            print(line, file=sys.stderr)
        if self.disabled:
            return
        self.open()
        with self._lock:
            if self._fh is None:
                return
            try:
                self._fh.write(line + "\n")
                self._rotate_if_needed()
            except OSError:
                self.disabled = True

    def debug(self, msg: str, **kw) -> None:
        self.log("DEBUG", msg, **kw)

    def info(self, msg: str, **kw) -> None:
        self.log("INFO", msg, **kw)

    def warning(self, msg: str, **kw) -> None:
        self.log("WARNING", msg, **kw)

    def error(self, msg: str, **kw) -> None:
        self.log("ERROR", msg, **kw)

    def exception(self, msg: str, exc: Optional[BaseException] = None, **kw) -> None:
        self.log("ERROR", msg, exc=exc or sys.exc_info()[1], **kw)


def _fmt(v) -> str:
    s = repr(v) if not isinstance(v, str) else v
    return s if len(s) <= 200 else s[:197] + "..."


_default: Optional[Logger] = None


def get_logger() -> Logger:
    global _default
    if _default is None:
        _default = Logger(data_dir() / "logs" / "nexus.log")
    return _default


def configure(level: str = "INFO", echo: bool = False, path: Optional[Path] = None) -> Logger:
    global _default
    _default = Logger(path if path is not None else data_dir() / "logs" / "nexus.log", level=level, echo=echo)
    return _default
