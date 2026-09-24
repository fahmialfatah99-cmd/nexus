"""Project awareness: build the repository context block for the system prompt.

This is what lets the agent answer "add a test" without first spending five tool
calls discovering that the project uses pytest, or that ``npm test`` exists. It is
deliberately **deterministic and bounded**: a depth-limited tree, detected stack
metadata, the actual build/test/lint commands found in config files, git state,
and excerpts of the project's own agent instructions (AGENTS.md / NEXUS.md /
README.md).

No embeddings, no indexing daemon, no guessing: everything here comes from a file
that exists. If detection fails, that section is simply omitted.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .ignore import ALWAYS_SKIP_DIRS, IgnoreMatcher, walk_files
from .logging_ import get_logger

STACK_MARKERS = {
    "pyproject.toml": "python", "setup.py": "python", "setup.cfg": "python",
    "requirements.txt": "python", "Pipfile": "python", "poetry.lock": "python",
    "uv.lock": "python", "tox.ini": "python", "pytest.ini": "python",
    "package.json": "node", "pnpm-lock.yaml": "node", "yarn.lock": "node",
    "package-lock.json": "node", "bun.lockb": "node", "tsconfig.json": "typescript",
    "Cargo.toml": "rust", "go.mod": "go", "composer.json": "php", "Gemfile": "ruby",
    "pom.xml": "java", "build.gradle": "java", "build.gradle.kts": "java",
    "CMakeLists.txt": "c/c++", "Makefile": "make", "Dockerfile": "docker",
    "docker-compose.yml": "docker", "docker-compose.yaml": "docker",
    "flake.nix": "nix", "shell.nix": "nix", ".csproj": "c#", "mix.exs": "elixir",
}

DOC_FILES = ("AGENTS.md", "NEXUS.md", "CLAUDE.md", ".cursorrules", ".github/copilot-instructions.md")
TEST_DIRS = ("tests", "test", "spec", "__tests__")


@dataclass
class GitInfo:
    repo: bool = False
    branch: str = ""
    dirty: int = 0
    staged: int = 0
    untracked: int = 0
    ahead: int = 0
    behind: int = 0
    recent_commits: List[str] = field(default_factory=list)
    remotes: List[str] = field(default_factory=list)

    def summary(self) -> str:
        if not self.repo:
            return ""
        parts = [f"branch: {self.branch or '(detached)'}"]
        if self.dirty or self.staged or self.untracked:
            parts.append(f"changes: {self.staged} staged, {self.dirty} modified, {self.untracked} untracked")
        else:
            parts.append("working tree clean")
        if self.ahead or self.behind:
            parts.append(f"ahead {self.ahead} / behind {self.behind}")
        if self.remotes:
            parts.append(f"remotes: {', '.join(self.remotes[:3])}")
        return "; ".join(parts)


@dataclass
class ProjectContext:
    root: Path
    stacks: List[str] = field(default_factory=list)
    markers: List[str] = field(default_factory=list)
    commands: Dict[str, str] = field(default_factory=dict)
    tree: str = ""
    file_count: int = 0
    git: GitInfo = field(default_factory=GitInfo)
    docs: Dict[str, str] = field(default_factory=dict)
    test_command: str = ""
    languages: Dict[str, int] = field(default_factory=dict)
    text: str = ""
    truncated: List[str] = field(default_factory=list)

    def as_prompt_block(self) -> str:
        return self.text


# --------------------------------------------------------------------------- #
# git
# --------------------------------------------------------------------------- #
def git_info(root: Path, *, timeout: float = 4.0) -> GitInfo:
    info = GitInfo()
    if not (root / ".git").exists():
        return info
    info.repo = True

    def run(args: Sequence[str]) -> str:
        try:
            proc = subprocess.run(["git", "--no-pager", *args], cwd=str(root), capture_output=True,
                                  text=True, timeout=timeout, errors="replace",
                                  env={**os.environ, "GIT_OPTIONAL_LOCKS": "0", "GIT_PAGER": "cat"})
        except (OSError, subprocess.TimeoutExpired):
            return ""
        return proc.stdout.strip() if proc.returncode == 0 else ""

    head = (root / ".git" / "HEAD")
    try:
        raw = head.read_text(encoding="utf-8", errors="replace").strip()
        info.branch = raw.split("/")[-1] if raw.startswith("ref:") else raw[:12]
    except OSError:
        info.branch = run(["rev-parse", "--abbrev-ref", "HEAD"])
    status = run(["status", "--porcelain=v1", "-b"])
    for line in status.splitlines():
        if line.startswith("##"):
            m = re.search(r"\[ahead (\d+)", line)
            if m:
                info.ahead = int(m.group(1))
            m = re.search(r"\[behind (\d+)", line)
            if m:
                info.behind = int(m.group(1))
            continue
        if not line.strip():
            continue
        # porcelain v1: XY where X = index status, Y = worktree status
        x, y = line[0], line[1]
        if x == "?":
            info.untracked += 1
            continue
        if x != " ":
            info.staged += 1
        if y != " ":
            info.dirty += 1
    commits = run(["log", "--oneline", "--no-decorate", "-6"])
    info.recent_commits = [c for c in commits.splitlines() if c.strip()][:6]
    remotes = run(["remote"])
    info.remotes = [r for r in remotes.splitlines() if r.strip()][:5]
    return info


# --------------------------------------------------------------------------- #
# stack + commands
# --------------------------------------------------------------------------- #
def detect_stack(root: Path) -> tuple:
    markers: List[str] = []
    stacks: List[str] = []
    for name in sorted(STACK_MARKERS):
        if (root / name).exists():
            markers.append(name)
            stack = STACK_MARKERS[name]
            if stack not in stacks:
                stacks.append(stack)
    for pattern, stack in (("*.csproj", "c#"), ("*.sln", "c#")):
        if list(root.glob(pattern)):
            markers.append(pattern)
            if stack not in stacks:
                stacks.append(stack)
    if any((root / d).is_dir() for d in TEST_DIRS):
        if "tests" not in markers:
            markers.append("tests/")
    return stacks, sorted(set(markers))


def detect_commands(root: Path) -> Dict[str, str]:
    """Find the real build/test/lint commands from config files."""
    cmds: Dict[str, str] = {}
    pkg = root / "package.json"
    if pkg.is_file():
        try:
            data = json.loads(pkg.read_text(encoding="utf-8", errors="replace"))
        except json.JSONDecodeError:
            data = {}
        scripts = data.get("scripts") or {}
        manager = "pnpm" if (root / "pnpm-lock.yaml").exists() else "yarn" if (root / "yarn.lock").exists() \
            else "bun" if (root / "bun.lockb").exists() else "npm"
        run = f"{manager} run" if manager in ("npm", "pnpm") else manager
        for key in ("test", "build", "lint", "typecheck", "dev", "start", "format", "check"):
            if key in scripts:
                cmds[key] = f"{run} {key}" if key not in ("start", "dev") else f"{run} {key}"
        if "test" not in cmds and manager == "npm":
            cmds["test"] = "npm test"
    if (root / "pyproject.toml").is_file():
        text = _read(root / "pyproject.toml")
        if "pytest" in text or (root / "pytest.ini").is_file() or (root / "tests").is_dir():
            cmds.setdefault("test", "python -m pytest -q")
        if "[tool.ruff]" in text:
            cmds.setdefault("lint", "ruff check .")
        if "[tool.black]" in text:
            cmds.setdefault("format", "black .")
        if "[tool.mypy]" in text:
            cmds.setdefault("typecheck", "mypy .")
        if "[tool.poetry]" in text:
            cmds.setdefault("install", "poetry install")
        elif (root / "uv.lock").is_file():
            cmds.setdefault("install", "uv sync")
        elif (root / "requirements.txt").is_file():
            cmds.setdefault("install", "pip install -r requirements.txt")
    if (root / "Makefile").is_file() or (root / "makefile").is_file():
        targets = _makefile_targets(root / "Makefile" if (root / "Makefile").is_file() else root / "makefile")
        for wanted in ("test", "build", "lint", "check", "all", "install"):
            if wanted in targets:
                cmds.setdefault(wanted, f"make {wanted}")
        if targets and "test" not in cmds:
            cmds.setdefault("make_targets", ", ".join(sorted(targets)[:12]))
    if (root / "Cargo.toml").is_file():
        cmds.setdefault("test", "cargo test")
        cmds.setdefault("build", "cargo build")
        cmds.setdefault("lint", "cargo clippy")
    if (root / "go.mod").is_file():
        cmds.setdefault("test", "go test ./...")
        cmds.setdefault("build", "go build ./...")
    if (root / "composer.json").is_file():
        cmds.setdefault("test", "vendor/bin/phpunit")
    if (root / "Gemfile").is_file():
        cmds.setdefault("test", "bundle exec rspec")
    if (root / "tox.ini").is_file():
        cmds.setdefault("test", "tox")
    return cmds


def _read(path: Path, limit: int = 200_000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[:limit]
    except OSError:
        return ""


def _makefile_targets(path: Path) -> List[str]:
    text = _read(path, 60_000)
    targets = []
    for line in text.splitlines():
        if line.startswith(("\t", " ", "#")) or "=" in line.split(":")[0]:
            continue
        m = re.match(r"^([A-Za-z0-9_.\-]+)\s*:(?!=)", line)
        if m:
            targets.append(m.group(1))
    return targets


# --------------------------------------------------------------------------- #
# tree
# --------------------------------------------------------------------------- #
LANGUAGE_EXTENSIONS = {
    ".py": "python", ".js": "javascript", ".jsx": "javascript", ".ts": "typescript",
    ".tsx": "typescript", ".rs": "rust", ".go": "go", ".java": "java", ".kt": "kotlin",
    ".c": "c", ".h": "c", ".cpp": "cc", ".cc": "cc", ".hpp": "cc", ".cs": "c#",
    ".rb": "ruby", ".php": "php", ".swift": "swift", ".m": "objc", ".sh": "shell",
    ".sql": "sql", ".md": "markdown", ".html": "html", ".css": "css", ".scss": "scss",
    ".vue": "vue", ".svelte": "svelte", ".lua": "lua", ".r": "r", ".dart": "dart",
    ".ex": "elixir", ".exs": "elixir", ".zig": "zig", ".toml": "toml", ".yaml": "yaml",
    ".yml": "yaml", ".json": "json",
}


def build_tree(root: Path, *, depth: int = 3, max_entries: int = 220,
               matcher: Optional[IgnoreMatcher] = None) -> tuple:
    """Render a depth-limited, ignore-aware directory tree.

    Returns ``(text, entry_count, truncated)``. Uses proper box-drawing prefixes
    so nested entries line up under their parent branch.
    """
    matcher = matcher or IgnoreMatcher(root)
    lines: List[str] = []
    count = 0
    truncated = False

    def walk(directory: Path, prefix: str, level: int) -> None:
        nonlocal count, truncated
        if level > depth or count >= max_entries:
            return
        try:
            entries = sorted(directory.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except OSError:
            return
        visible = [e for e in entries if not matcher.is_ignored(e, is_dir=e.is_dir())]
        for i, entry in enumerate(visible):
            if count >= max_entries:
                truncated = True
                return
            last = i == len(visible) - 1
            count += 1
            lines.append(prefix + ("└── " if last else "├── ") + entry.name + ("/" if entry.is_dir() else ""))
            if entry.is_dir() and entry.name not in ALWAYS_SKIP_DIRS:
                walk(entry, prefix + ("    " if last else "│   "), level + 1)

    walk(root, "", 0)
    return "\n".join(lines), count, truncated


def language_histogram(root: Path, matcher: IgnoreMatcher, *, max_files: int = 4000) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for path in walk_files(root, matcher=matcher, max_files=max_files):
        lang = LANGUAGE_EXTENSIONS.get(path.suffix.lower())
        if lang:
            counts[lang] = counts.get(lang, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1])[:8])


# --------------------------------------------------------------------------- #
# main builder
# --------------------------------------------------------------------------- #
def build_project_context(root: Path, *, settings: Any = None, include_git: bool = True,
                          log: Any = None) -> ProjectContext:
    """Build the whole block. Never raises: a partial context beats no context."""
    log = log or get_logger()
    cfg = getattr(settings, "project_context", None)
    enabled = True if cfg is None else bool(getattr(cfg, "enabled", True))
    max_files = int(getattr(cfg, "max_files", 400) if cfg else 400)
    depth = int(getattr(cfg, "tree_depth", 3) if cfg else 3)
    include_tree = bool(getattr(cfg, "include_tree", True) if cfg else True)
    doc_names = list(getattr(cfg, "read_files", DOC_FILES) if cfg else DOC_FILES)
    max_chars = int(getattr(cfg, "max_chars", 12_000) if cfg else 12_000)
    ctx = ProjectContext(root=root)
    if not enabled or not root.is_dir():
        return ctx

    try:
        matcher = IgnoreMatcher(root)
        ctx.stacks, ctx.markers = detect_stack(root)
        ctx.commands = detect_commands(root)
        ctx.test_command = ctx.commands.get("test", "")
        ctx.languages = language_histogram(root, matcher, max_files=max_files)
        if include_tree:
            ctx.tree, ctx.file_count, truncated = build_tree(root, depth=depth, max_entries=max_files,
                                                             matcher=matcher)
            if truncated:
                ctx.truncated.append("tree")
        if include_git:
            ctx.git = git_info(root)
        for name in doc_names + ["README.md", "readme.md", "README.rst"]:
            path = root / name
            if path.is_file():
                text = _read(path, 20_000).strip()
                if text:
                    ctx.docs[name] = text
    except Exception as exc:  # never break startup because of context building
        log.warning("project context build failed", error=str(exc))

    ctx.text = _render(ctx, max_chars)
    return ctx


def _render(ctx: ProjectContext, max_chars: int) -> str:
    parts: List[str] = []
    if ctx.stacks or ctx.markers:
        parts.append("Detected stack: " + ", ".join(ctx.stacks or ["unknown"])
                     + (f"\nMarkers: {', '.join(ctx.markers[:14])}" if ctx.markers else ""))
    if ctx.languages:
        parts.append("Languages (file counts): "
                     + ", ".join(f"{k}:{v}" for k, v in ctx.languages.items()))
    if ctx.commands:
        parts.append("Commands found in config:\n"
                     + "\n".join(f"- {k}: `{v}`" for k, v in sorted(ctx.commands.items())))
    if ctx.git.repo:
        block = ["Git: " + ctx.git.summary()]
        if ctx.git.recent_commits:
            block.append("Recent commits:\n" + "\n".join(f"- {c}" for c in ctx.git.recent_commits))
        parts.append("\n".join(block))
    if ctx.file_count:
        parts.append(f"Layout ({ctx.file_count} entries, ignored files excluded):\n{ctx.tree}")
    for name, text in ctx.docs.items():
        budget = max(400, (max_chars - sum(len(p) for p in parts)) // max(1, len(ctx.docs)))
        body = text if len(text) <= budget else text[:budget] + "\n…(truncated)"
        parts.append(f"--- {name} ---\n{body}")
    out = "\n\n".join(parts).strip()
    if len(out) > max_chars:
        out = out[:max_chars].rstrip() + "\n…(project context truncated)"
        if "context" not in " ".join(ctx.truncated):
            ctx.truncated.append("context")
    return out


__all__ = ["ProjectContext", "GitInfo", "build_project_context", "git_info", "detect_stack",
           "detect_commands", "build_tree", "language_histogram", "DOC_FILES"]
