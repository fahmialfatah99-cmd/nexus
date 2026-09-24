"""Shell + Python execution tools.

Safety model:
* Every command is classified (``read-only`` / ``mutating`` / ``dangerous``)
  before it runs; the classification drives the permission prompt and the audit
  log. Classification is *advisory* -- the permission engine has the final say.
* Execution is a real subprocess with a hard timeout, capped output, and no
  interactive TTY (interactive programs are detected and rejected with a
  helpful message instead of hanging the agent forever).
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..base import ConfirmationRequest, Tool, ToolContext, ToolResult, clip

DEFAULT_TIMEOUT = 120.0
MAX_TIMEOUT = 1800.0
OUTPUT_LIMIT = 200_000

DANGEROUS_PATTERNS: List[Tuple[str, str]] = [
    (r"\brm\s+(-[a-zA-Z]*[rf][a-zA-Z]*\s+)+(/|~|\$HOME)(\s|$)", "recursive delete of a root/home path"),
    (r"\brm\s+-[a-zA-Z]*r[a-zA-Z]*\s+-[a-zA-Z]*f[a-zA-Z]*\s+/(?!tmp)", "recursive force delete from /"),
    (r"\bmkfs(\.\w+)?\b", "filesystem format"),
    (r"\bdd\b.*\bof=/dev/", "raw device write"),
    (r":\(\)\s*\{\s*:\|:\s*&\s*\}\s*;:", "fork bomb"),
    (r"\b(curl|wget)\b[^|]*\|\s*(sudo\s+)?(ba)?sh\b", "piping a remote script into a shell"),
    (r"\bgit\s+push\b.*(--force|-f)\b", "force push (rewrites remote history)"),
    (r"\bgit\s+reset\s+--hard\b", "hard reset (discards local work)"),
    (r"\bgit\s+clean\s+-[a-zA-Z]*f", "git clean -f (deletes untracked files)"),
    (r"\bchmod\s+(-R\s+)?0?777\b", "chmod 777"),
    (r"\b(shutdown|reboot|halt|poweroff)\b", "power state change"),
    (r">\s*/dev/(sd|nvme|disk)", "raw device overwrite"),
    (r"\bDROP\s+(TABLE|DATABASE)\b", "destructive SQL"),
    (r"\bTRUNCATE\s+TABLE\b", "destructive SQL"),
    (r"\bsudo\b", "privilege escalation"),
    (r"\bsu\s+-?\s*$", "user switch"),
    (r"\bkill(all)?\s+-9\s+(-1|1)\b", "kill all processes"),
    (r"\bhistory\s+-c\b", "clearing shell history"),
    (r"\bnpm\s+publish\b|\btwine\s+upload\b|\bcargo\s+publish\b", "publishing a package"),
    (r"\baws\s+.*\brm\b|\bgcloud\s+.*\bdelete\b|\baz\s+.*\bdelete\b", "cloud resource deletion"),
    (r"\bkubectl\s+delete\b", "kubernetes resource deletion"),
    (r"\bdocker\s+(system\s+prune|rm|volume\s+rm)", "docker destructive action"),
]

MUTATING_PREFIXES = (
    "rm", "mv", "cp", "mkdir", "touch", "chmod", "chown", "ln", "rmdir", "sed", "tee",
    "git", "npm", "yarn", "pnpm", "pip", "pip3", "uv", "cargo", "go", "make", "cmake",
    "docker", "kubectl", "terraform", "apt", "apt-get", "brew", "poetry", "ruff", "black",
    "isort", "pytest", "python", "python3", "node", "deno", "bun", "sh", "bash", "zsh",
    "curl", "wget", "nc", "scp", "rsync", "ssh", "tar", "unzip", "zip", "gzip", "echo",
    "printf", "cat", "kill", "pkill", "systemctl", "service", "psql", "mysql", "sqlite3",
)

#: Commands that cannot modify anything by themselves. Interpreters (python,
#: node, ruby, perl) are deliberately NOT here: they execute arbitrary code, so
#: they are always treated as mutating.
READONLY_COMMANDS = {
    "ls", "ll", "la", "pwd", "cat", "head", "tail", "wc", "find", "grep",
    "egrep", "fgrep", "rg", "ag", "tree", "file", "stat", "du", "df", "which", "whereis",
    "whoami", "id", "uname", "date", "env", "printenv", "echo", "diff", "cmp", "sort",
    "uniq", "cut", "awk", "jq", "yq", "ps", "free", "uptime", "hostname", "type", "test",
    "true", "false", "history", "man", "help", "basename", "dirname", "realpath",
    "readlink", "xargs", "seq", "tr", "column", "nl", "tac", "rev", "md5sum", "sha256sum",
}

#: Sub-commands of container tools that are safe to read.
READONLY_SUBCOMMANDS = {
    "docker": {"ps", "images", "logs", "inspect", "version", "info", "stats", "history", "diff"},
    "kubectl": {"get", "describe", "logs", "top", "version", "explain", "api-resources", "config"},
}

#: Flags that turn an otherwise read-only command into a write.
MUTATING_FLAGS = {"sed": {"-i", "--in-place"}}

READONLY_GIT = {"status", "log", "diff", "show", "branch", "tag", "remote", "ls-files",
                "blame", "shortlog", "describe", "rev-parse", "config", "stash"}

INTERACTIVE_PATTERNS = [
    r"^\s*(vim?|nvim|nano|emacs|pico)\b", r"^\s*(top|htop|less|more|man)\s*$",
    r"\bgit\s+(rebase|add\s+-p|commit)\s*($|.*-i\b)", r"^\s*python[23]?\s*(-i)?\s*$",
    r"^\s*node\s*(-i)?\s*$",
    r"^\s*(ftp|telnet|ssh)\b(?!.*-)", r"\b--interactive\b", r"\bnpm\s+login\b",
]


def classify_command(command: str) -> Tuple[str, str]:
    """Return (risk, reason) for a shell command. risk in read-only|mutating|dangerous."""
    cmd = (command or "").strip()
    if not cmd:
        return "mutating", "empty command"
    for pattern, reason in DANGEROUS_PATTERNS:
        if re.search(pattern, cmd, re.IGNORECASE):
            return "dangerous", reason
    # split on shell operators and evaluate each segment
    segments = [s.strip() for s in re.split(r"&&|\|\||;|\||\$\(|`", cmd) if s.strip()]
    mutating = False
    for seg in segments:
        try:
            parts = shlex.split(seg, posix=True)
        except ValueError:
            parts = seg.split()
        if not parts:
            continue
        head = Path(parts[0]).name
        if head in ("sudo", "su", "doas"):
            return "dangerous", "privilege escalation"
        if head == "git" and len(parts) > 1:
            sub = parts[1]
            if sub.startswith("-"):
                sub = parts[2] if len(parts) > 2 else ""
            if sub in READONLY_GIT:
                continue
            mutating = True
            continue
        if head in ("curl", "wget"):
            mutating = True
            continue
        if head in READONLY_COMMANDS and not _writes_to_file(seg):
            if head in MUTATING_FLAGS and any(a in MUTATING_FLAGS[head] for a in parts[1:]):
                mutating = True
                continue
            continue
        if head in READONLY_SUBCOMMANDS and len(parts) > 1 and parts[1] in READONLY_SUBCOMMANDS[head]:
            continue
        mutating = True
    return ("mutating" if mutating else "read-only"), ("mutates state" if mutating else "read-only command")


def _writes_to_file(segment: str) -> bool:
    return bool(re.search(r"(^|[^>])>{1,2}\s*\S", segment))


def is_interactive(command: str) -> Optional[str]:
    for pattern in INTERACTIVE_PATTERNS:
        if re.search(pattern, command.strip()):
            return pattern
    return None


class BashTool(Tool):
    name = "bash"
    description = (
        "Run a shell command in the workspace and return stdout, stderr and the exit code.\n"
        "Non-interactive only (no vim/top/less). Long-running commands are killed after `timeout` "
        "seconds (default 120, max 1800). Output is capped. Use this for builds, tests and git; "
        "use read_file/grep/find_files for searching code because they are faster and ignore-aware."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Shell command to execute (bash -c)."},
            "timeout": {"type": "number", "description": "Seconds before the command is killed.", "default": 120},
            "cwd": {"type": "string", "description": "Working directory (must be inside the workspace)."},
            "stdin": {"type": "string", "description": "Optional data piped to stdin."},
            "env": {"type": "object", "description": "Extra environment variables."},
        },
        "required": ["command"],
    }
    category = "shell"
    concurrency_safe = False

    def confirmation(self, args, ctx):
        command = args.get("command", "")
        risk, reason = classify_command(command)
        return ConfirmationRequest(
            title=f"Run: {_oneline(command)}",
            detail=f"risk={risk} ({reason})",
            risk=risk,
            key=f"bash:{_command_key(command)}",
        )

    def run(self, args, ctx):
        command = args.get("command") or ""
        if not command.strip():
            return ToolResult.fail("Empty command.")
        interactive = is_interactive(command)
        if interactive:
            return ToolResult.fail(
                f"'{command.strip().split()[0]}' is interactive and cannot run in this environment. "
                "Use a non-interactive equivalent (e.g. `git --no-pager log`, `python -c ...`)."
            )
        cwd = ctx.resolve(args.get("cwd") or ".")
        if not cwd.is_dir():
            return ToolResult.fail(f"Working directory does not exist: {cwd}")
        from ...core.ignore import is_within

        if not is_within(ctx.workspace_root, cwd):
            return ToolResult.fail(f"cwd {cwd} is outside the workspace.")
        timeout = min(float(args.get("timeout") or DEFAULT_TIMEOUT), MAX_TIMEOUT)
        env = dict(os.environ)
        env["NEXUS_AGENT"] = ctx.agent
        env.setdefault("GIT_PAGER", "cat")
        env.setdefault("PAGER", "cat")
        env.setdefault("NO_COLOR", "1")
        env["PYTHONUNBUFFERED"] = "1"
        for k, v in (args.get("env") or {}).items():
            env[str(k)] = str(v)
        try:
            proc = subprocess.run(
                command,
                shell=True,
                executable=_shell(),
                cwd=str(cwd),
                env=env,
                input=(args.get("stdin") or None),
                capture_output=True,
                text=True,
                timeout=timeout,
                errors="replace",
            )
        except subprocess.TimeoutExpired as exc:
            partial = _combine(exc.stdout, exc.stderr)
            return ToolResult.fail(
                f"Command timed out after {timeout:.0f}s and was killed.\n"
                f"Partial output:\n{clip(partial, 4000)}"
            )
        except OSError as exc:
            return ToolResult.fail(f"Failed to start command: {exc}")
        out = _combine(proc.stdout, proc.stderr)
        head = f"exit={proc.returncode} cwd={ctx.rel(cwd)}"
        body = clip(out, ctx.max_output_chars) if out.strip() else "(no output)"
        data = {"exit_code": proc.returncode, "stdout": proc.stdout or "", "stderr": proc.stderr or ""}
        result = ToolResult(content=f"{head}\n{body}", data=data)
        if proc.returncode != 0:
            result.is_error = True
        return result


class PythonExecTool(Tool):
    name = "python_exec"
    description = (
        "Execute Python code in a fresh subprocess (workspace as cwd) and return stdout/stderr. "
        "Use for data crunching, quick calculations or inspecting structures. The workspace's own "
        "interpreter is used; no packages are pre-installed beyond the standard library."
    )
    parameters = {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Python source code to execute."},
            "timeout": {"type": "number", "default": 60},
        },
        "required": ["code"],
    }
    category = "shell"
    concurrency_safe = False

    def confirmation(self, args, ctx):
        first = (args.get("code") or "").strip().splitlines()
        return ConfirmationRequest(title=f"Run Python ({len(args.get('code') or '')} chars)",
                                   detail=first[0][:120] if first else "", risk="mutating",
                                   key="python_exec:*")

    def run(self, args, ctx):
        code = args.get("code") or ""
        if not code.strip():
            return ToolResult.fail("Empty code.")
        timeout = min(float(args.get("timeout") or 60), MAX_TIMEOUT)
        try:
            proc = subprocess.run([sys.executable, "-I", "-c", code], cwd=str(ctx.cwd),
                                  capture_output=True, text=True, timeout=timeout, errors="replace",
                                  env={**os.environ, "PYTHONUNBUFFERED": "1", "NO_COLOR": "1"})
        except subprocess.TimeoutExpired:
            return ToolResult.fail(f"Python execution timed out after {timeout:.0f}s.")
        except OSError as exc:
            return ToolResult.fail(f"Could not start python: {exc}")
        out = _combine(proc.stdout, proc.stderr)
        result = ToolResult(content=f"exit={proc.returncode}\n{clip(out, ctx.max_output_chars) if out.strip() else '(no output)'}",
                            data={"exit_code": proc.returncode})
        result.is_error = proc.returncode != 0
        return result


def _shell() -> Optional[str]:
    for candidate in (os.environ.get("SHELL"), "/bin/bash", "/bin/sh"):
        if candidate and os.path.exists(candidate):
            return candidate
    return None


def _combine(stdout: Optional[Any], stderr: Optional[Any]) -> str:
    def text(v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, bytes):
            return v.decode("utf-8", errors="replace")
        return str(v)

    out, err = text(stdout).strip(), text(stderr).strip()
    if out and err:
        return f"{out}\n--- stderr ---\n{err}"
    return out or err


def _oneline(command: str, width: int = 100) -> str:
    one = " ".join(command.split())
    return one if len(one) <= width else one[: width - 1] + "…"


def _command_key(command: str) -> str:
    """Stable identity for 'always allow' rules: first word + safe args."""
    try:
        parts = shlex.split(command)
    except ValueError:
        parts = command.split()
    if not parts:
        return "*"
    head = Path(parts[0]).name
    if head in ("git", "docker", "kubectl", "npm", "yarn", "cargo", "make") and len(parts) > 1:
        return f"{head} {parts[1]}"
    return head


def build_tools() -> List[Tool]:
    return [BashTool(), PythonExecTool()]
