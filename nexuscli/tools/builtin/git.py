"""Git tool.

Deliberately *not* implemented as ``bash("git ...")``: arguments are passed as an
argv list to ``subprocess`` with ``shell=False``, so shell metacharacters in a
model-produced branch name or commit message can never be interpreted. Read-only
subcommands are auto-approved; mutating ones go through the permission engine.
"""

from __future__ import annotations

import os
import subprocess
from typing import Any, Dict, List, Optional, Tuple

from ..base import ConfirmationRequest, Tool, ToolContext, ToolResult, clip

READONLY_SUBCOMMANDS = {
    "status", "log", "diff", "show", "branch", "tag", "remote", "ls-files", "blame",
    "shortlog", "describe", "rev-parse", "cat-file", "rev-list", "whatchanged",
    "count-objects", "ls-tree", "reflog", "config", "stash", "worktree", "symbolic-ref",
}
MUTATING_SUBCOMMANDS = {
    "add", "commit", "checkout", "switch", "restore", "merge", "rebase", "pull", "fetch",
    "push", "reset", "clean", "rm", "mv", "cherry-pick", "revert", "apply", "stash",
    "tag", "init", "clone", "am", "commit-tree", "update-index", "worktree",
}
#: subcommands that change remote state -> always elevated risk
REMOTE_AFFECTING = {"push", "fetch", "pull", "clone", "ls-remote"}
DANGEROUS_ARGS = {
    ("push", "--force"), ("push", "--force-with-lease"), ("push", "-f"),
    ("reset", "--hard"), ("clean", "-f"), ("clean", "-fd"), ("clean", "-fdx"),
    ("checkout", "-f"), ("rebase", "--abort"),
}


def classify_git(argv: List[str]) -> Tuple[str, str]:
    """Return ``(risk, reason)`` for parsed git args (without the leading 'git').

    Subcommands that are read-only *or* mutating depending on their flags are
    resolved explicitly -- guessing here is what makes agents either nag the
    user for `git status` or silently run `git reset --hard`.
    """
    positional = [a for a in argv if not a.startswith("-")]
    sub = positional[0] if positional else ""
    if not sub:
        return "mutating", "no subcommand given"

    for pair in DANGEROUS_ARGS:
        if argv[0] == pair[0] and pair[1] in argv:
            return "dangerous", f"git {pair[0]} {pair[1]}"
    if sub == "push" and any(a in ("--force", "-f", "--force-with-lease") for a in argv):
        return "dangerous", "force push rewrites remote history"

    if sub == "config":
        read_form = len(positional) <= 1 or any(
            a in ("--get", "--get-all", "--get-regexp", "--list", "-l") for a in argv[1:])
        return ("read-only", "git config (read)") if read_form else ("mutating", "git config writes settings")
    if sub == "stash":
        if len(argv) > 1 and argv[1] in ("list", "show"):
            return "read-only", "git stash (inspect)"
        return "mutating", "git stash modifies the working tree"
    if sub == "tag":
        if len(positional) <= 1 and not any(a in ("-d", "--delete") for a in argv):
            return "read-only", "git tag (list)"
        return "mutating", "git tag modifies refs"
    if sub == "branch":
        if any(a in ("-d", "-D", "-m", "-M", "--delete", "--move", "--set-upstream-to") for a in argv):
            return "mutating", "git branch modifies refs"
        return "read-only", "git branch (list)"
    if sub == "worktree":
        if len(argv) > 1 and argv[1] == "list":
            return "read-only", "git worktree list"
        return "mutating", "git worktree modifies the repository"
    if sub == "remote":
        if len(positional) <= 1 or (len(argv) > 1 and argv[1] in ("-v", "show", "get-url")):
            return "read-only", "git remote (inspect)"
        return "mutating", "git remote modifies configuration"
    if sub in READONLY_SUBCOMMANDS and sub not in MUTATING_SUBCOMMANDS:
        return "read-only", f"git {sub} does not modify anything"
    if sub in REMOTE_AFFECTING:
        return "elevated", f"git {sub} contacts the remote"
    if sub in MUTATING_SUBCOMMANDS:
        return "mutating", f"git {sub} modifies the repository"
    return "mutating", f"unknown git subcommand '{sub}'"


class GitTool(Tool):
    name = "git"
    description = (
        "Run a git command safely (argv based, no shell injection). Examples:\n"
        "  git status --short\n  git diff HEAD~1 -- src/\n  git log --oneline -20\n"
        "  git add -A && git commit -m 'msg'   (pass as ONE command string; it is split safely)\n"
        "Read-only commands run without asking; commits/pushes require approval. "
        "Never use -i/--interactive or a pager."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Git arguments, e.g. 'status --short' or 'commit -m \"fix\"'."},
            "cwd": {"type": "string", "description": "Repository directory (defaults to the workspace)."},
            "timeout": {"type": "number", "default": 60},
        },
        "required": ["command"],
    }
    category = "vcs"

    def _parse(self, args: Dict[str, Any]) -> Tuple[List[str], Optional[str]]:
        import shlex

        raw = (args.get("command") or "").strip()
        if raw.startswith("git "):
            raw = raw[4:]
        try:
            argv = shlex.split(raw, posix=True)
        except ValueError as exc:
            return [], f"Could not parse git command: {exc}"
        if not argv:
            return [], "Empty git command."
        if argv[0] == "git":
            argv = argv[1:]
        return argv, None

    def confirmation(self, args, ctx):
        argv, err = self._parse(args)
        if err:
            return ConfirmationRequest(title="git (unparseable)", detail=err, risk="mutating")
        risk, reason = classify_git(argv)
        return ConfirmationRequest(title=f"git {' '.join(argv)}"[:140], detail=reason, risk=risk,
                                   key=f"git:{argv[0]}" if risk == "read-only" else f"git:{' '.join(argv)}"[:120])

    def run(self, args, ctx):
        argv, err = self._parse(args)
        if err:
            return ToolResult.fail(err)
        if any(a in ("-i", "--interactive") or a.endswith("-i") for a in argv):
            return ToolResult.fail("Interactive git commands are not supported (they would hang).")
        cwd = ctx.resolve(args.get("cwd") or ".")
        if not (cwd / ".git").exists() and not cwd.name == ".git":
            parent = _find_repo_root(cwd)
            if parent is None:
                return ToolResult.fail(f"Not a git repository: {cwd} (run `git init` first).")
            cwd = parent
        env = dict(os.environ)
        env.update({"GIT_PAGER": "cat", "PAGER": "cat", "GIT_TERMINAL_PROMPT": "0",
                    "GIT_OPTIONAL_LOCKS": "0", "NO_COLOR": "1", "NEXUS_AGENT": ctx.agent})
        full = ["git", "--no-pager", *argv]
        timeout = float(args.get("timeout") or 60)
        try:
            proc = subprocess.run(full, cwd=str(cwd), env=env, capture_output=True, text=True,
                                  timeout=timeout, errors="replace")
        except subprocess.TimeoutExpired:
            return ToolResult.fail(f"git {' '.join(argv)} timed out after {timeout:.0f}s.")
        except FileNotFoundError:
            return ToolResult.fail("git is not installed or not on PATH.")
        except OSError as exc:
            return ToolResult.fail(f"Failed to run git: {exc}")
        out = (proc.stdout or "").strip()
        err_out = (proc.stderr or "").strip()
        body = out if out else ""
        if err_out:
            body = f"{body}\n--- stderr ---\n{err_out}" if body else f"--- stderr ---\n{err_out}"
        head = f"exit={proc.returncode} cwd={ctx.rel(cwd)}"
        result = ToolResult(content=f"{head}\n{clip(body, ctx.max_output_chars) if body.strip() else '(no output)'}",
                            data={"exit_code": proc.returncode, "stdout": out, "stderr": err_out,
                                  "argv": argv})
        result.is_error = proc.returncode != 0
        return result


def _find_repo_root(start) -> Optional[Any]:
    from pathlib import Path

    cur = Path(start).resolve()
    for parent in [cur, *cur.parents]:
        if (parent / ".git").exists():
            return parent
    return None


def build_tools() -> List[Tool]:
    return [GitTool()]
