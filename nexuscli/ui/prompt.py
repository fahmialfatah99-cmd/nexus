"""Interactive input: history, tab completion, multi-line editing.

Built on stdlib ``readline`` when available (Linux/macOS, and Windows via the
``pyreadline`` shim if the user has it) with a pure-``input()`` fallback, so the
CLI degrades gracefully instead of breaking in exotic environments.

Completion covers:
* ``/command`` names (fuzzy, with the description shown inline)
* ``@path`` file references, respecting ignore rules
* argument values for commands that declare them (models, agents, modes)
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .theme import Style, truncate, visible_width

try:  # pragma: no cover - platform dependent
    import readline  # type: ignore
except Exception:  # pragma: no cover
    readline = None  # type: ignore

# Feature-detect instead of trusting the import: some platforms ship an
# editline-backed `readline` that is missing readline()/set_completer entirely.
_HAS_READLINE = bool(readline is not None and hasattr(readline, "readline"))
_HAS_COMPLETER = bool(_HAS_READLINE and hasattr(readline, "set_completer")
                      and hasattr(readline, "parse_and_bind"))
_HAS_HISTORY = bool(readline is not None and hasattr(readline, "read_history_file")
                    and hasattr(readline, "write_history_file"))


class Prompt:
    def __init__(
        self,
        style: Style,
        *,
        history_file: Optional[Path] = None,
        commands: Sequence[Tuple[str, str]] = (),
        cwd: Optional[Path] = None,
        dynamic_completers: Optional[Dict[str, Callable[[], Iterable[str]]]] = None,
        history_length: int = 5000,
    ) -> None:
        self.style = style
        self.history_file = history_file
        self.commands = list(commands)
        self.cwd = Path(cwd) if cwd else Path.cwd()
        self.dynamic = dict(dynamic_completers or {})
        self.history_length = history_length
        self._matches: List[str] = []
        self._configured = False
        self.multiline_hint = "… "

    # ------------------------------------------------------------------ #
    @property
    def capabilities(self) -> Dict[str, bool]:
        return {"readline": _HAS_READLINE, "completion": _HAS_COMPLETER, "history": _HAS_HISTORY}

    def configure(self) -> None:
        if self._configured or readline is None:
            return
        self._configured = True
        try:
            if _HAS_COMPLETER:
                readline.parse_and_bind("tab: complete")
                readline.set_completer(self._complete)
                readline.set_completer_delims("")  # we do our own tokenisation
            if _HAS_HISTORY and self.history_file is not None:
                self.history_file.parent.mkdir(parents=True, exist_ok=True)
                try:
                    readline.read_history_file(str(self.history_file))
                except (OSError, ValueError):
                    pass
                if hasattr(readline, "set_history_length"):
                    readline.set_history_length(self.history_length)
        except Exception:
            pass  # completion/history are conveniences; never fail the session

    def save_history(self) -> None:
        if not _HAS_HISTORY or self.history_file is None:
            return
        try:
            self.history_file.parent.mkdir(parents=True, exist_ok=True)
            readline.write_history_file(str(self.history_file))
        except (OSError, ValueError, AttributeError):
            pass

    # ------------------------------------------------------------------ #
    def read(self, prompt_str: str = "> ", *, multiline: bool = True) -> Optional[str]:
        """Read one logical line. Returns None on EOF, raises KeyboardInterrupt."""
        self.configure()
        buffer_lines: List[str] = []
        while True:
            shown = prompt_str if not buffer_lines else self.multiline_hint
            try:
                if _HAS_READLINE:
                    line = readline.readline(shown)
                    if line == "":  # EOF
                        if buffer_lines:
                            return "\n".join(buffer_lines)
                        return None
                    line = line.rstrip("\n")
                else:
                    line = input(shown).rstrip("\n")
            except EOFError:
                return "\n".join(buffer_lines) if buffer_lines else None
            if not self._is_tty():
                # Nothing echoes typed characters when stdin is a pipe; print the
                # line ourselves so scripted sessions remain readable.
                try:
                    sys.stdout.write(line + "\n")
                    sys.stdout.flush()
                except (OSError, ValueError):
                    pass
            if not buffer_lines and not line.strip():
                continue  # ignore blank input at the primary prompt
            if multiline and self._continues(line, buffer_lines):
                buffer_lines.append(line[:-1] if line.endswith("\\") else line)
                continue
            buffer_lines.append(line)
            text = "\n".join(buffer_lines).strip()
            if text and readline is not None and hasattr(readline, "add_history"):
                try:
                    readline.add_history(text)
                except Exception:
                    pass
            return text

    def _is_tty(self) -> bool:
        try:
            return bool(sys.stdin.isatty() and sys.stdout.isatty())
        except (OSError, ValueError, AttributeError):
            return False

    @staticmethod
    def _continues(line: str, buffer_lines: Sequence[str]) -> bool:
        if line.endswith("\\"):
            return True
        joined = "\n".join([*buffer_lines, line])
        fences = len(re.findall(r"^\s*```", joined, re.MULTILINE))
        return fences % 2 == 1  # inside an unterminated code fence

    # ------------------------------------------------------------------ #
    def _complete(self, text: str, state: int) -> Optional[str]:
        if state == 0:
            self._matches = self._candidates()
        if state < len(self._matches):
            return self._matches[state]
        return None

    def _candidates(self) -> List[str]:
        if not _HAS_COMPLETER:
            return []
        try:
            line = readline.get_line_buffer()
            begin = readline.get_begidx()
            end = readline.get_endidx()
        except Exception:
            return []
        prefix = line[begin:end]
        before = line[:begin]

        # @file references anywhere in the line
        at = _last_at_token(before + prefix)
        if at is not None:
            return self._complete_files(at, keep_prefix=True)

        stripped = line.strip()
        # slash command names
        if stripped.startswith("/") and " " not in stripped and not before.strip():
            needle = stripped[1:]
            out = []
            for name, help_text in self.commands:
                if name.lstrip("/").startswith(needle):
                    out.append("/" + name.lstrip("/"))
            return sorted(set(out))

        # command argument values
        parts = stripped.split()
        if parts and parts[0].startswith("/"):
            cmd = parts[0].lstrip("/")
            completer = self.dynamic.get(cmd)
            if completer is not None:
                try:
                    values = [str(v) for v in completer()]
                except Exception:
                    values = []
                return sorted({v for v in values if v.startswith(prefix)})
        return self._complete_files(prefix, keep_prefix=False)

    def _complete_files(self, prefix: str, *, keep_prefix: bool) -> List[str]:
        if not prefix and not keep_prefix:
            return []
        raw = prefix[1:] if keep_prefix and prefix.startswith("@") else prefix
        target = (self.cwd / raw) if raw else self.cwd
        parent = target if target.is_dir() and raw.endswith("/") else target.parent
        stem = "" if (target.is_dir() and raw.endswith("/")) else target.name
        try:
            entries = sorted(parent.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except OSError:
            return []
        from ..core.ignore import ALWAYS_SKIP_DIRS

        out: List[str] = []
        for entry in entries[:400]:
            if not entry.name.startswith(stem):
                continue
            if entry.name in ALWAYS_SKIP_DIRS or entry.name.startswith(".git"):
                continue
            base = raw[: len(raw) - len(stem)] if stem else raw
            completion = ("@" if keep_prefix else "") + base + entry.name + ("/" if entry.is_dir() else "")
            out.append(completion)
        return out[:80]

    # ------------------------------------------------------------------ #
    def ask(self, question: str, *, default: Optional[str] = None,
            choices: Optional[Sequence[str]] = None) -> str:
        """Simple blocking question with validation."""
        suffix = ""
        if choices:
            suffix = f" [{'/'.join(choices)}]"
        if default:
            suffix += f" (default: {default})"
        while True:
            try:
                answer = input(f"{question}{suffix}: ").strip()
            except EOFError:
                return default or ""
            if not answer and default is not None:
                return default
            if choices and answer.lower() not in [c.lower() for c in choices]:
                self.style and print(self.style.dim(f"  choose one of: {', '.join(choices)}"))
                continue
            return answer

    def yes_no(self, question: str, *, default: bool = False) -> bool:
        answer = self.ask(question, default="y" if default else "n", choices=("y", "n", "yes", "no"))
        return answer.strip().lower() in ("y", "yes")


def _last_at_token(text: str) -> Optional[str]:
    match = re.search(r"(?:^|\s)@([^\s]*)$", text)
    return match.group(0).strip() if match else None


__all__ = ["Prompt"]
