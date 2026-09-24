"""Token estimation and context-window budgeting.

**Estimation.** Uses ``tiktoken`` when the user happens to have it installed
(exact for OpenAI models) and otherwise falls back to a calibrated heuristic
that treats CJK characters as ~1 token each. That matters: a naive
``len(text)/4`` badly *under*-estimates Chinese/Japanese/Korean input, and
under-estimating is exactly the direction that causes context overflow.

**Budgeting.** A graded, deterministic ladder:

1. squeeze the biggest tool outputs outside the protected tail
2. prune old tool outputs entirely (leaving a re-runnable marker)
3. drop the oldest turns -- an assistant turn takes its tool results with it,
   because providers hard-reject orphaned ``tool`` messages
4. escalate into the protected tail (squeeze -> prune old -> prune newest)

If even that does not fit, ``BudgetReport.overflow`` is set and the caller must
say so. NEXUS never silently truncates the user's current request.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..providers.base import Completion, Message, content_to_text

# --------------------------------------------------------------------------- #
# Token estimation
# --------------------------------------------------------------------------- #
try:  # pragma: no cover - optional dependency
    import tiktoken  # type: ignore

    _TIKTOKEN: Any = tiktoken
except Exception:  # pragma: no cover
    _TIKTOKEN = None

_ENCODINGS: Dict[str, Any] = {}
_CJK_RE = re.compile(
    "[\u3000-\u303f\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff00-\uffef"
    "\uac00-\ud7af\u1100-\u11ff\u3130-\u318f]"
)
_PER_MESSAGE_OVERHEAD = 4


def _encoding_for(model: str) -> Any:
    if _TIKTOKEN is None:
        return None
    key = model or "default"
    if key in _ENCODINGS:
        return _ENCODINGS[key]
    enc = None
    try:  # pragma: no cover - depends on optional dep
        enc = _TIKTOKEN.encoding_for_model(key)
    except Exception:
        try:
            enc = _TIKTOKEN.get_encoding("cl100k_base")
        except Exception:
            enc = None
    _ENCODINGS[key] = enc
    return enc


def estimate_tokens(text: str, model: str = "") -> int:
    """Estimate the token count of *text*. Never raises."""
    if not text:
        return 0
    enc = _encoding_for(model)
    if enc is not None:
        try:  # pragma: no cover - depends on optional dep
            return len(enc.encode(text, disallowed_special=()))
        except Exception:
            pass
    cjk = len(_CJK_RE.findall(text))
    other = len(text) - cjk
    return int(cjk * 1.0 + other / 3.6 + 1)


def message_tokens(msg: Message, model: str = "") -> int:
    total = _PER_MESSAGE_OVERHEAD
    total += estimate_tokens(content_to_text(msg.content), model)
    if msg.name:
        total += estimate_tokens(msg.name, model)
    for tc in msg.tool_calls:
        total += estimate_tokens(tc.name + tc.arguments, model) + 6
    return total


def count_tokens(messages: Sequence[Message], model: str = "") -> int:
    return sum(message_tokens(m, model) for m in messages)


# --------------------------------------------------------------------------- #
# Budgeting
# --------------------------------------------------------------------------- #
BIG_OUTPUT_CHARS = 4_000
PROTECT_RECENT = 6
PRUNED_MARKER = "[output pruned to save context -- re-run the tool if you need it again]"


@dataclass
class ContextConfig:
    reserve_output: int = 8_192
    reserve_ratio: float = 0.92
    protect_recent: int = PROTECT_RECENT
    big_output_chars: int = BIG_OUTPUT_CHARS
    shrink_to_chars: int = 1_200
    compaction_threshold: float = 0.82


@dataclass
class BudgetReport:
    model: str = ""
    window: int = 0
    limit: int = 0
    before: int = 0
    after: int = 0
    actions: List[str] = field(default_factory=list)
    overflow: bool = False
    #: True when the model's context window is unknown, so nothing was enforced.
    skipped: bool = False

    @property
    def changed(self) -> bool:
        return bool(self.actions)

    def describe(self) -> str:
        if not self.actions:
            return f"{self.after:,} est. tokens (limit {self.limit:,})"
        return (f"{self.before:,} -> {self.after:,} est. tokens (limit {self.limit:,}); "
                + "; ".join(self.actions))


class ContextManager:
    """Assembles the exact message list sent to the model, within budget."""

    def __init__(self, config: Optional[ContextConfig] = None) -> None:
        self.config = config or ContextConfig()

    # -- budget arithmetic ------------------------------------------------
    def limit_for(self, window: int, max_output: int = 0) -> int:
        if window <= 0:
            return 0  # unknown window -> never guess a limit
        reserve = max(max_output or 0, self.config.reserve_output)
        usable = int(window * self.config.reserve_ratio) - reserve
        return max(usable, window // 4)

    # -- main entry -------------------------------------------------------
    def assemble(
        self,
        messages: Sequence[Message],
        *,
        window: int = 0,
        max_output: int = 0,
        model: str = "",
        limit: Optional[int] = None,
    ) -> Tuple[List[Message], BudgetReport]:
        report = BudgetReport(model=model, window=window)
        work = [m.clone() for m in messages]
        report.before = count_tokens(work, model)
        report.limit = limit if limit is not None else self.limit_for(window, max_output)
        if report.limit <= 0:
            report.after = report.before
            report.skipped = True  # unknown window: not an "action", so stay quiet
            return work, report
        if report.before <= report.limit:
            report.after = report.before
            return work, report

        protect = max(2, self.config.protect_recent)

        squeezed = self._squeeze_tool_outputs(work, protect, report.limit, model)
        if squeezed:
            report.actions.append(f"squeezed {squeezed} tool output(s)")
        if self._fits(work, report.limit, model):
            return self._finish(work, report, model)

        pruned = self._prune_old_tool_outputs(work, protect)
        if pruned:
            report.actions.append(f"pruned {pruned} old tool output(s)")
        if self._fits(work, report.limit, model):
            return self._finish(work, report, model)

        dropped = self._drop_oldest(work, protect, report.limit, model)
        if dropped:
            report.actions.append(f"dropped {dropped} oldest message(s)")
        if self._fits(work, report.limit, model):
            return self._finish(work, report, model)

        note = self._escalate(work, report.limit, model)
        if note:
            report.actions.append(note)
        return self._finish(work, report, model, force_overflow_check=True)

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def _fits(work: Sequence[Message], limit: int, model: str) -> bool:
        return count_tokens(work, model) <= limit

    def _finish(self, work: List[Message], report: BudgetReport, model: str,
                *, force_overflow_check: bool = False) -> Tuple[List[Message], BudgetReport]:
        report.after = count_tokens(work, model)
        report.overflow = report.after > report.limit
        if report.overflow:
            report.actions.append("STILL OVER BUDGET")
        return work, report

    @staticmethod
    def _replace_tool(work: List[Message], i: int, content: str) -> None:
        old = work[i]
        work[i] = Message(role="tool", content=content, tool_call_id=old.tool_call_id,
                          name=old.name, meta=dict(old.meta))

    def _tail_cut(self, work: Sequence[Message], protect: int) -> int:
        return max(0, len(work) - protect)

    def _squeeze_tool_outputs(self, work: List[Message], protect: int, limit: int, model: str) -> int:
        from ..tools.base import clip

        cut = self._tail_cut(work, protect)
        idxs = [i for i in range(cut) if work[i].role == "tool"]
        idxs.sort(key=lambda i: -len(content_to_text(work[i].content)))
        squeezed = 0
        for i in idxs:
            text = content_to_text(work[i].content)
            if len(text) <= self.config.shrink_to_chars:
                continue
            self._replace_tool(work, i, clip(text, self.config.shrink_to_chars))
            squeezed += 1
            if self._fits(work, limit, model):
                break
        return squeezed

    def _prune_old_tool_outputs(self, work: List[Message], protect: int) -> int:
        cut = self._tail_cut(work, protect)
        pruned = 0
        for i in range(cut):
            if work[i].role != "tool":
                continue
            text = content_to_text(work[i].content)
            if not text or text == PRUNED_MARKER:
                continue
            self._replace_tool(work, i, PRUNED_MARKER)
            pruned += 1
        return pruned

    def _drop_oldest(self, work: List[Message], protect: int, limit: int, model: str) -> int:
        dropped = 0
        while not self._fits(work, limit, model) and len(work) > protect + 1:
            candidate: Optional[int] = None
            boundary = len(work) - protect
            for i, m in enumerate(work):
                if i >= boundary:
                    break
                if m.role == "system":
                    continue
                candidate = i
                break
            if candidate is None:
                break
            removed = work.pop(candidate)
            dropped += 1
            if removed.role == "assistant" and removed.tool_calls:
                ids = {tc.id for tc in removed.tool_calls}
                for j in range(len(work) - 1, -1, -1):
                    if work[j].role == "tool" and work[j].tool_call_id in ids:
                        work.pop(j)
                        dropped += 1
        return dropped

    def _escalate(self, work: List[Message], limit: int, model: str) -> str:
        """Last resort: act on the protected tail, newest information last."""
        from ..tools.base import clip

        tool_idx = [i for i, m in enumerate(work) if m.role == "tool"]
        if not tool_idx:
            return ""
        notes: List[str] = []

        squeezed = 0
        for i in sorted(tool_idx, key=lambda i: -len(content_to_text(work[i].content))):
            text = content_to_text(work[i].content)
            if len(text) > self.config.shrink_to_chars:
                self._replace_tool(work, i, clip(text, self.config.shrink_to_chars))
                squeezed += 1
                if self._fits(work, limit, model):
                    break
        if squeezed:
            notes.append(f"squeezed {squeezed} recent tool output(s)")
        if self._fits(work, limit, model):
            return "escalation: " + ", ".join(notes)

        pruned = 0
        for i in tool_idx[:-1]:
            text = content_to_text(work[i].content)
            if text and text != PRUNED_MARKER:
                self._replace_tool(work, i, PRUNED_MARKER)
                pruned += 1
        if pruned:
            notes.append(f"pruned {pruned} recent tool output(s)")
        if self._fits(work, limit, model):
            return "escalation: " + ", ".join(notes)

        last = tool_idx[-1]
        if content_to_text(work[last].content) != PRUNED_MARKER:
            self._replace_tool(work, last, PRUNED_MARKER)
            notes.append("pruned the newest tool output")
        return "escalation: " + ", ".join(notes) if notes else ""

    # -- compaction -------------------------------------------------------
    def needs_compaction(self, messages: Sequence[Message], window: int, max_output: int = 0,
                         model: str = "") -> bool:
        limit = self.limit_for(window, max_output)
        if limit <= 0:
            return False
        return count_tokens(messages, model) >= limit * self.config.compaction_threshold

    @staticmethod
    def compaction_prompt() -> str:
        return (
            "You are the context-compression module of an agentic CLI. Summarise the conversation so far "
            "into a compact handoff note that lets another agent continue with zero loss of operative detail.\n\n"
            "Use exactly this structure:\n"
            "## Goal\n<what the user is trying to achieve, including verbatim requirements>\n"
            "## Decisions\n<choices already made and why>\n"
            "## State\n<files created/modified/deleted with exact paths, commands run and their outcome>\n"
            "## Open items\n<what is unfinished, blocked or still to verify>\n"
            "## Constraints\n<user preferences, style rules, do-not-touch areas>\n\n"
            "Be terse and factual. Never invent anything not present in the conversation. "
            "Preserve exact file paths, identifiers, error messages and version numbers."
        )

    @staticmethod
    def split_for_compaction(messages: Sequence[Message], keep_recent: int = 6
                             ) -> Tuple[List[Message], List[Message]]:
        """Split into ``(to_summarise, to_keep)`` without orphaning tool results."""
        msgs = list(messages)
        systems = [m for m in msgs if m.role == "system"]
        rest = [m for m in msgs if m.role != "system"]
        cut = max(0, len(rest) - keep_recent)
        while cut < len(rest) and rest[cut].role == "tool":
            cut += 1
        while cut > 0 and rest[cut - 1].role == "assistant" and rest[cut - 1].tool_calls:
            cut -= 1
        return systems + rest[:cut], rest[cut:]


def summarise_history(provider: Any, messages: Sequence[Message], *, model: str, max_tokens: int = 1500,
                      on_event: Any = None) -> Optional[str]:
    """Ask *provider* to compress *messages* into a handoff note."""
    from ..providers.base import RequestOptions

    transcript = render_transcript(messages)
    prompt_msgs = [Message.system(ContextManager.compaction_prompt()),
                   Message.user(f"<conversation>\n{transcript}\n</conversation>")]
    try:
        result: Completion = provider.complete(
            prompt_msgs, (), RequestOptions(model=model, temperature=0.1, max_tokens=max_tokens, stream=False),
            on_event=on_event)
    except Exception:
        return None
    text = result.message.text.strip()
    return text or None


def render_transcript(messages: Sequence[Message], max_chars: int = 120_000) -> str:
    from ..tools.base import clip

    lines: List[str] = []
    for m in messages:
        if m.role == "system":
            continue
        body = content_to_text(m.content)
        if m.role == "assistant" and m.tool_calls:
            calls = ", ".join(f"{tc.name}({clip(tc.arguments, 300)})" for tc in m.tool_calls)
            body = (body + "\n" if body else "") + f"[tool calls] {calls}"
        if m.role == "tool":
            body = f"[{m.name or 'tool'} result] {body}"
        lines.append(f"{m.role.upper()}: {clip(body, 6000)}")
    return clip("\n\n".join(lines), max_chars)


__all__ = [
    "estimate_tokens", "message_tokens", "count_tokens", "ContextManager", "ContextConfig",
    "BudgetReport", "summarise_history", "render_transcript", "PRUNED_MARKER",
]
