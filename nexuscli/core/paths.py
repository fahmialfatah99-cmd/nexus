"""Filesystem locations used by NEXUS.

Honours XDG on Linux/macOS and ``%APPDATA%`` on Windows, and can be fully
redirected with ``NEXUS_HOME`` (used heavily by the test-suite).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

APP_DIRNAME = ".nexus"


def home() -> Path:
    env = os.environ.get("NEXUS_HOME")
    if env:
        return Path(env).expanduser()
    return Path.home() / APP_DIRNAME


def data_dir() -> Path:
    """Root for logs / caches / sessions."""
    if os.environ.get("NEXUS_HOME"):
        return home()
    if sys.platform == "win32":
        base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
        return Path(base) / "nexus"
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / "nexus"


def config_file() -> Path:
    return home() / "config.json"


def auth_file() -> Path:
    return home() / "auth.json"


def memory_file() -> Path:
    return home() / "MEMORY.md"


def sessions_dir() -> Path:
    return data_dir() / "sessions"


def logs_dir() -> Path:
    return data_dir() / "logs"


def plugins_dir() -> Path:
    return home() / "plugins"


def personas_dir() -> Path:
    return home() / "personas"


def project_dir(cwd: Path | None = None) -> Path:
    return (cwd or Path.cwd()) / APP_DIRNAME


def project_config_file(cwd: Path | None = None) -> Path:
    return project_dir(cwd) / "config.json"


def ensure_dirs() -> None:
    for p in (home(), data_dir(), sessions_dir(), logs_dir(), plugins_dir(), personas_dir()):
        try:
            p.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass


#: Unambiguous project markers. ``.nexus`` is handled separately because the
#: *global* config directory is ``~/.nexus`` -- treating it as a project marker
#: would make every project under $HOME resolve its workspace root to the home
#: directory, which would silently widen the path-containment guard.
PROJECT_MARKERS = (".git", ".hg", ".svn", "pyproject.toml", "package.json", "setup.py",
                   "Cargo.toml", "go.mod", "composer.json", "Gemfile", "pom.xml",
                   "build.gradle", "CMakeLists.txt", "Makefile")


def find_project_root(start: Path | None = None) -> Path:
    """Walk upwards looking for a VCS / project marker.

    A ``.nexus`` directory only counts when it is a *project* one -- i.e. not the
    global config directory and it actually holds configuration.
    """
    cur = (start or Path.cwd()).resolve()
    try:
        global_nexus = home().resolve()
    except OSError:  # pragma: no cover - unreadable HOME
        global_nexus = None
    for parent in [cur, *cur.parents]:
        if any((parent / m).exists() for m in PROJECT_MARKERS):
            return parent
        candidate = parent / APP_DIRNAME
        if candidate.exists() and (candidate / "config.json").is_file():
            try:
                if global_nexus is None or candidate.resolve() != global_nexus:
                    return parent
            except OSError:
                return parent
        if parent == parent.parent:
            break
    return cur
