"""Path safety + ignore rules.

Two responsibilities that every file/search/shell tool depends on:

1. **Containment** -- resolve a user/model supplied path and decide whether it
   lives inside the workspace. Used by the permission engine.
2. **Ignoring** -- a small, dependency-free implementation of the subset of
   ``.gitignore`` semantics that matters for code search: globs, ``**``,
   directory-only rules, negation, and per-directory ignore files.

Deliberately *not* a full gitignore clone (no ``[a-z]`` character classes beyond
what :mod:`fnmatch` gives us, no sparse-checkout interaction) -- but it is
deterministic and unit-tested, which is what search quality actually needs.
"""

from __future__ import annotations

import fnmatch
import os
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, List, Optional, Sequence

#: Directories never worth walking, regardless of ignore files.
ALWAYS_SKIP_DIRS = {
    ".git", ".hg", ".svn", ".nexus", "__pycache__", "node_modules", ".venv", "venv",
    ".tox", ".nox", ".mypy_cache", ".pytest_cache", ".ruff_cache", "dist", "build",
    "target", ".next", ".nuxt", ".output", ".turbo", ".cache", ".npm", ".yarn",
    ".gradle", ".idea", ".vs", "coverage", ".terraform", ".eggs", "site-packages",
}

ALWAYS_SKIP_FILES = {".DS_Store", "Thumbs.db", ".nexus.lock"}

BINARY_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp", ".tiff", ".pdf", ".zip",
    ".gz", ".bz2", ".xz", ".7z", ".rar", ".tar", ".exe", ".dll", ".so", ".dylib", ".bin",
    ".class", ".jar", ".war", ".pyc", ".pyo", ".o", ".a", ".obj", ".woff", ".woff2",
    ".ttf", ".otf", ".eot", ".mp3", ".mp4", ".avi", ".mov", ".mkv", ".wav", ".flac",
    ".db", ".sqlite", ".sqlite3", ".parquet", ".arrow", ".npy", ".npz", ".pkl",
    ".lockb", ".wasm", ".ipynb_checkpoints",
}


# --------------------------------------------------------------------------- #
# Path safety
# --------------------------------------------------------------------------- #
def resolve_path(raw: str | os.PathLike, cwd: Path) -> Path:
    """Expand ``~``/env vars and make the path absolute (no symlink resolution)."""
    p = Path(os.path.expandvars(str(raw))).expanduser()
    if not p.is_absolute():
        p = cwd / p
    return Path(os.path.normpath(str(p)))


def is_within(root: Path, target: Path) -> bool:
    """True when *target* is *root* itself or lives underneath it.

    Uses fully resolved realpaths so ``..`` and symlink tricks cannot escape.
    """
    try:
        r = root.resolve()
        t = target.resolve()
    except OSError:
        return False
    try:
        t.relative_to(r)
        return True
    except ValueError:
        return False


def relative_display(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except (ValueError, OSError):
        return str(path)


def safe_relpath(raw: str, cwd: Path) -> str:
    p = resolve_path(raw, cwd)
    try:
        return str(p.relative_to(cwd))
    except ValueError:
        return str(p)


# --------------------------------------------------------------------------- #
# Ignore rules
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Rule:
    """One parsed ignore rule.

    ``anchored`` is True when the pattern contains a slash (other than a
    trailing one), which pins it to the directory of the ignore file.
    ``basename`` is True for slash-free patterns, which match a path component
    at *any* depth -- the gitignore behaviour people rely on.
    """

    pattern: str
    negated: bool = False
    dir_only: bool = False
    anchored: bool = False
    basename: bool = False

    def matches(self, rel: str, is_dir: bool) -> bool:
        parts = rel.split("/")
        if self.basename:
            last = len(parts) - 1
            for i, part in enumerate(parts):
                if not _regex_match(self.pattern, part):
                    continue
                if i == last:
                    # The rule targets the path itself: honour dir-only rules.
                    if self.dir_only and not is_dir:
                        return False
                    return True
                # An ancestor directory matched -> everything under it is ignored.
                return True
            return False
        # Anchored: match the full relative path, or any ancestor directory of it.
        for i in range(len(parts), 0, -1):
            candidate = "/".join(parts[:i])
            if not _regex_match(self.pattern, candidate):
                continue
            if i == len(parts):
                if self.dir_only and not is_dir:
                    return False
                return True
            return True
        return False


_REGEX_CACHE: Dict[str, "object"] = {}


def _regex_match(pattern: str, path: str) -> bool:
    """Glob -> regex translation implementing ``*``, ``?``, ``**`` and ``[...]``."""
    import re

    rx = _REGEX_CACHE.get(pattern)
    if rx is None:
        out: List[str] = []
        i = 0
        n = len(pattern)
        while i < n:
            c = pattern[i]
            if c == "*":
                if pattern[i : i + 3] == "**/":
                    out.append("(?:.*/)?")  # zero or more leading directories
                    i += 3
                    continue
                if pattern[i : i + 2] == "**":
                    out.append(".*")
                    i += 2
                    continue
                out.append("[^/]*")
                i += 1
                continue
            if c == "?":
                out.append("[^/]")
                i += 1
                continue
            if c == "[":
                j = pattern.find("]", i + 1)
                if j > i:
                    cls = pattern[i + 1 : j]
                    if cls.startswith("!"):
                        cls = "^" + cls[1:]
                    out.append(f"[{cls}]")
                    i = j + 1
                    continue
            out.append(re.escape(c))
            i += 1
        rx = re.compile("".join(out) + r"\Z")
        _REGEX_CACHE[pattern] = rx
    return bool(rx.match(path))  # type: ignore[union-attr]


@dataclass
class IgnoreMatcher:
    """Evaluates a stack of ignore files for a directory tree walk."""

    root: Path
    extra_patterns: Sequence[str] = ()
    respect_gitignore: bool = True
    always_skip: bool = True
    _rules: List[Rule] = field(default_factory=list)
    _loaded: Dict[Path, bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        for pat in self.extra_patterns:
            rule = parse_rule(pat)
            if rule:
                self._rules.append(rule)
        if self.respect_gitignore:
            self._load_dir(self.root)

    # -- loading ----------------------------------------------------------
    def _load_chain(self, directory: Path) -> None:
        """Load ignore files for *directory* and every ancestor up to root.

        A ``.gitignore`` in a mid-level directory must also apply to files
        deeper down, so loading only the immediate parent is not enough.
        """
        if not self.respect_gitignore:
            return
        try:
            rel = directory.resolve().relative_to(self.root.resolve())
        except (ValueError, OSError):
            return
        current = self.root
        self._load_dir(current)
        for part in rel.parts:
            current = current / part
            self._load_dir(current)

    def _load_dir(self, directory: Path) -> None:
        if directory in self._loaded:
            return
        self._loaded[directory] = True
        for name in (".gitignore", ".nexusignore", ".ignore"):
            f = directory / name
            try:
                if not f.is_file():
                    continue
                text = f.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for line in text.splitlines():
                rule = parse_rule(line, prefix=_rel_prefix(directory, self.root))
                if rule:
                    self._rules.append(rule)

    # -- matching ---------------------------------------------------------
    def is_ignored(self, path: Path, *, is_dir: Optional[bool] = None) -> bool:
        p = Path(path)
        if is_dir is None:
            try:
                is_dir = p.is_dir()
            except OSError:
                is_dir = False
        name = p.name
        if self.always_skip:
            if is_dir and name in ALWAYS_SKIP_DIRS:
                return True
            if not is_dir and (name in ALWAYS_SKIP_FILES or p.suffix.lower() in BINARY_EXTENSIONS):
                return True
        try:
            rel = str(p.resolve().relative_to(self.root.resolve()).as_posix())
        except (ValueError, OSError):
            return False
        if not rel or rel == ".":
            return False
        if self.always_skip:
            # Direct path checks must also honour always-skipped ancestors,
            # otherwise is_ignored(".git/config") would return False even though
            # the walker never descends into .git.
            parts = rel.split("/")
            if any(part in ALWAYS_SKIP_DIRS for part in parts[:-1]):
                return True
        self._load_chain(p.parent)
        ignored = False
        for rule in self._rules:  # last matching rule wins (gitignore semantics)
            if rule.matches(rel, is_dir):
                ignored = not rule.negated
        return ignored

    def filter(self, paths: Iterable[Path]) -> List[Path]:
        return [p for p in paths if not self.is_ignored(p)]


def _rel_prefix(directory: Path, root: Path) -> str:
    try:
        rel = directory.resolve().relative_to(root.resolve()).as_posix()
    except (ValueError, OSError):
        return ""
    return "" if rel in (".", "") else rel + "/"


def parse_rule(line: str, prefix: str = "") -> Optional[Rule]:
    line = line.rstrip("\n").rstrip("\r")
    if not line.strip() or line.lstrip().startswith("#"):
        return None
    negated = False
    if line.startswith("!"):
        negated = True
        line = line[1:]
    line = line.strip()
    if not line:
        return None
    if line.startswith("\\") and len(line) > 1 and line[1] in ("!", "#"):
        line = line[1:]
    dir_only = line.endswith("/")
    if dir_only:
        line = line[:-1]
    had_leading_slash = line.startswith("/")
    line = line.lstrip("/")
    body = line[:-1] if line.endswith("/") else line
    basename = "/" not in body and not had_leading_slash
    if body.startswith("**/"):
        basename = False
    if basename and prefix:
        # A slash-free rule inside a nested ignore file applies at *any* depth
        # below that directory -> "a/b/**/secret.txt".
        pattern = f"{prefix}**/{line}"
        basename = False
    else:
        pattern = prefix + line
    anchored = not basename
    return Rule(pattern=pattern, negated=negated, dir_only=dir_only,
                anchored=anchored, basename=basename)


# --------------------------------------------------------------------------- #
# Tree walking
# --------------------------------------------------------------------------- #
def walk_files(
    root: Path,
    *,
    matcher: Optional[IgnoreMatcher] = None,
    max_files: int = 200_000,
    follow_symlinks: bool = False,
) -> Iterable[Path]:
    """Yield files under *root*, pruning ignored directories (never descends)."""
    root = Path(root)
    matcher = matcher or IgnoreMatcher(root)
    count = 0
    stack: List[Path] = [root]
    while stack:
        current = stack.pop()
        try:
            entries = list(os.scandir(current))
        except (OSError, PermissionError):
            continue
        dirs: List[Path] = []
        for entry in entries:
            try:
                p = Path(entry.path)
                is_dir = entry.is_dir(follow_symlinks=follow_symlinks)
            except OSError:
                continue
            if is_dir:
                if entry.is_symlink() and not follow_symlinks:
                    continue
                if matcher.is_ignored(p, is_dir=True):
                    continue
                dirs.append(p)
            else:
                if matcher.is_ignored(p, is_dir=False):
                    continue
                count += 1
                if count > max_files:
                    return
                yield p
        stack.extend(sorted(dirs, key=lambda x: x.name))


def is_probably_binary(path: Path, sample: int = 4096) -> bool:
    if path.suffix.lower() in BINARY_EXTENSIONS:
        return True
    try:
        with open(path, "rb") as fh:
            chunk = fh.read(sample)
    except OSError:
        return True
    if b"\x00" in chunk:
        return True
    return False


def glob_match(pattern: str, path: str) -> bool:
    """Public glob matcher (``*`` ``?`` ``**`` ``[...]``) used by the search tools.

    A slash-free pattern also matches the basename, mirroring shell/gitignore
    intuition (``*.py`` matches ``src/a/b.py``).
    """
    pattern = pattern.replace("\\", "/")
    path = path.replace("\\", "/")
    if _regex_match(pattern, path):
        return True
    if "/" not in pattern.strip("/"):
        return _regex_match(pattern, PurePosixPath(path).name)
    if not pattern.startswith("**/"):
        return _regex_match("**/" + pattern, path)
    return False


def human_size(num: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num) < 1024 or unit == "TB":
            return f"{num:.0f}{unit}" if unit == "B" else f"{num:.1f}{unit}"
        num /= 1024
    return f"{num:.1f}TB"
