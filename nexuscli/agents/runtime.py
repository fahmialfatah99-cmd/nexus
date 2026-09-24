"""The agent loop.

One turn = user message in, final assistant text out, with as many tool
round-trips in between as the task needs.

Correctness details that are easy to get wrong and are handled here:

* **No orphaned tool calls.** If the user aborts (Ctrl+C) or a tool explodes in
  the middle of a batch, every outstanding ``tool_call`` still receives a
  ``tool`` message. Providers hard-reject histories where they do not.
* **Tool errors are data, not exceptions.** A validation error, a permission
  denial or a crashing tool becomes a tool result the model can read and react
  to; the loop only dies on provider/programming errors.
* **Parallel only when safe.** A batch runs concurrently only if *every* call is
  read-only and marked concurrency-safe; otherwise order is preserved.
* **Loop detection.** Repeating the identical call is caught and broken instead
  of burning tokens until max_turns.
* **Budget enforcement** via :class:`ContextManager` before every request.
"""

from __future__ import annotations

import hashlib
import platform
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..core.context import ContextManager, count_tokens
from ..core.errors import (
    ContextOverflowError,
    NexusError,
    PermissionDenied,
    ToolNotFound,
    ValidationError,
)
from ..core.logging_ import get_logger
from ..core.permissions import PermissionEngine
from ..core.router import ModelRouter, RouteTarget
from ..core.usage import UsageLedger
from ..providers.base import (
    ImageBlock,
    Message,
    ReasoningDelta,
    RequestOptions,
    TextDelta,
    ToolCall,
    ToolCallStart,
    ToolSpec,
    Usage,
    UsageEvent,
)
from ..tools.base import ToolContext, ToolRegistry, ToolResult
from .persona import Persona, get_persona

MAX_PARALLEL_TOOLS = 6
IDENTICAL_CALL_LIMIT = 3


# --------------------------------------------------------------------------- #
# Options / results
# --------------------------------------------------------------------------- #
@dataclass
class AgentOptions:
    model_spec: str = ""
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    max_turns: int = 40
    reasoning_effort: Optional[str] = None
    json_mode: bool = False
    top_p: Optional[float] = None
    stream: bool = True
    parallel_tools: bool = True
    seed: Optional[int] = None
    allow_failover: bool = True


@dataclass
class TurnResult:
    text: str = ""
    reasoning: str = ""
    usage: Usage = field(default_factory=lambda: Usage(requests=0))
    cost: float = 0.0
    tool_calls: int = 0
    turns: int = 0
    finish_reason: str = "stop"
    aborted: bool = False
    denied: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    latency_ms: int = 0
    target: Optional[RouteTarget] = None
    report: Dict[str, Any] = field(default_factory=dict)
    budget: str = ""

    @property
    def ok(self) -> bool:
        return not self.errors and not self.aborted


# --------------------------------------------------------------------------- #
# Shared services
# --------------------------------------------------------------------------- #
@dataclass
class Services:
    registry: ToolRegistry
    permissions: PermissionEngine
    router: ModelRouter
    context: ContextManager = field(default_factory=ContextManager)
    ledger: UsageLedger = field(default_factory=UsageLedger)
    checkpoints: Any = None
    ui: Any = None
    log: Any = field(default_factory=get_logger)
    session: Any = None
    cwd: Path = field(default_factory=Path.cwd)
    workspace_root: Path = field(default_factory=Path.cwd)
    project_context: str = ""
    memory_text: str = ""
    offline: bool = False
    cancelled: Optional[threading.Event] = None
    vars: Dict[str, Any] = field(default_factory=dict)
    spawn_subagent: Optional[Callable[..., str]] = None
    max_output_chars: int = 24_000

    def tool_context(self, agent: str) -> ToolContext:
        if self.spawn_subagent is not None:
            self.vars.setdefault("spawn_subagent", self.spawn_subagent)
        return ToolContext(
            cwd=self.cwd,
            workspace_root=self.workspace_root,
            config=self.vars.get("config"),
            permissions=self.permissions,
            checkpoints=self.checkpoints,
            ui=self.ui,
            session=self.session,
            events=self.vars.get("events"),
            transport=self.vars.get("transport"),
            agent=agent,
            cancelled=self.cancelled,
            log=self.log,
            vars=self.vars,
            max_output_chars=self.max_output_chars,
            read_only_mode=self.permissions.mode == "read-only",
        )


# --------------------------------------------------------------------------- #
# Report parsing
# --------------------------------------------------------------------------- #
REPORT_KEYS = ("status", "summary", "artifacts", "followups", "confidence")


def parse_report(text: str) -> Dict[str, Any]:
    """Extract the ```report block every persona is required to emit."""
    if not text:
        return {}
    blocks = re.findall(r"```report\s*\n(.*?)```", text, re.DOTALL)
    raw = blocks[-1] if blocks else ""
    if not raw:
        # tolerate agents that used a plain fenced block at the very end
        tail = text.strip()[-1200:]
        if re.search(r"^\s*status:\s*(done|blocked|needs-review)", tail, re.MULTILINE):
            raw = tail
    out: Dict[str, Any] = {}
    for line in raw.split("\n"):
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().lower()
        if key in REPORT_KEYS:
            out[key] = value.strip()
    if "artifacts" in out:
        arts = [a.strip() for a in re.split(r"[,\n]", out["artifacts"]) if a.strip() and a.strip().lower() != "none"]
        out["artifact_list"] = arts
    return out


# --------------------------------------------------------------------------- #
# Environment block
# --------------------------------------------------------------------------- #
def environment_block(cwd: Path, workspace_root: Path, mode: str, model_label: str,
                      extra: Optional[Dict[str, str]] = None) -> str:
    lines = [
        f"Platform: {platform.system()} {platform.release()} ({platform.machine()})",
        f"Python: {sys.version.split()[0]}",
        f"Working directory: {cwd}",
        f"Workspace root: {workspace_root}",
        f"Today: {time.strftime('%Y-%m-%d %H:%M %Z').strip()}",
        f"Approval mode: {mode}",
        f"Model: {model_label}",
    ]
    branch = _git_branch(workspace_root)
    if branch:
        lines.append(f"Git branch: {branch}")
    for k, v in (extra or {}).items():
        lines.append(f"{k}: {v}")
    return "\n".join(lines)


def _git_branch(root: Path) -> str:
    head = root / ".git" / "HEAD"
    try:
        if head.is_file():
            text = head.read_text(encoding="utf-8", errors="replace").strip()
            if text.startswith("ref:"):
                return text.split("/")[-1]
            return text[:12]
    except OSError:
        pass
    return ""


# --------------------------------------------------------------------------- #
# The agent
# --------------------------------------------------------------------------- #
class Agent:
    def __init__(
        self,
        *,
        services: Services,
        persona: Optional[Persona] = None,
        name: str = "",
        options: Optional[AgentOptions] = None,
        history: Optional[List[Message]] = None,
        system_extra: str = "",
    ) -> None:
        self.services = services
        self.persona = persona or get_persona("main")
        self.name = name or self.persona.key
        self.options = options or AgentOptions()
        self.history: List[Message] = list(history or [])
        self.system_extra = system_extra
        self.log = services.log
        self.usage = Usage(requests=0)
        self.cost = 0.0
        self.turns_used = 0
        self._target: Optional[RouteTarget] = None
        self._continue_nudges = 0

    # ------------------------------------------------------------------ #
    @property
    def target(self) -> Optional[RouteTarget]:
        return self._target

    def model_label(self) -> str:
        if self._target:
            return self._target.label
        return self.options.model_spec or "default"

    def system_messages(self) -> List[Message]:
        env = environment_block(self.services.cwd, self.services.workspace_root,
                                self.services.permissions.mode, self.model_label(),
                                {"Agent": f"{self.persona.name} ({self.persona.key})"})
        body = self.persona.system_prompt(
            cwd=str(self.services.cwd),
            project_context=self.services.project_context,
            memory=self.services.memory_text,
            extra=(f"## Environment\n{env}" + (f"\n\n{self.system_extra}" if self.system_extra else "")),
        )
        msgs = [Message.system(body)]
        tool_guide = self._tool_guide()
        if tool_guide:
            msgs.append(Message.system(tool_guide))
        return msgs

    def _tool_guide(self) -> str:
        tools = self.available_tools()
        if not tools:
            return ""
        names = ", ".join(t.name for t in tools)
        return (
            "## Tools\n"
            f"Available: {names}.\n"
            "- Call tools whenever you need facts about the repository; never guess file contents.\n"
            "- Multiple independent tool calls may be issued in a single turn (they run in parallel when read-only).\n"
            "- A tool result marked is_error is feedback, not a failure of the task: fix the input and retry once.\n"
            "- Stop calling tools as soon as you can answer; then give the final answer.\n"
            "- Never mention tool names or internal mechanics in the final user-facing answer unless asked."
        )

    def available_tools(self) -> List[Any]:
        allowed = self.persona.tools
        tools = self.services.registry.enabled_tools(offline=self.services.offline)
        if allowed is None:
            return tools
        keep = set(allowed)
        return [t for t in tools if t.name in keep]

    def tool_specs(self) -> List[ToolSpec]:
        return [t.spec() for t in self.available_tools()]

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        self.history.clear()
        self.usage = Usage(requests=0)
        self.cost = 0.0
        self.turns_used = 0
        self._continue_nudges = 0

    def add_user(self, content: Any) -> None:
        self.history.append(Message.user(content))

    def token_estimate(self) -> int:
        return count_tokens(self.system_messages() + self.history, self._target.model if self._target else "")

    # ------------------------------------------------------------------ #
    def send(self, content: Any, *, images: Optional[Sequence[ImageBlock]] = None) -> TurnResult:
        started = time.monotonic()
        if images:
            blocks: List[Any] = []
            if isinstance(content, str) and content:
                from ..providers.base import TextBlock

                blocks.append(TextBlock(content))
            blocks.extend(images)
            self.history.append(Message.user(blocks))
        elif isinstance(content, Message):
            self.history.append(content)
        else:
            self.history.append(Message.user(str(content)))

        result = TurnResult()
        call_count = 0
        identical: Dict[str, int] = {}
        turns = 0
        max_turns = max(1, self.options.max_turns)
        text_parts: List[str] = []
        reasoning_parts: List[str] = []

        while turns < max_turns:
            if self._aborted():
                result.aborted = True
                break
            turns += 1
            self.turns_used += 1
            ui = self.services.ui
            try:
                messages = self._assemble(result)
            except ContextOverflowError as exc:
                result.errors.append(f"ContextOverflowError: {exc}")
                if ui:
                    ui.on_error(f"{exc}\n{exc.hint or ''}", agent=self.name)
                break
            tools = self.tool_specs()
            opts = self._request_options(messages, tools)
            if ui:
                ui.on_turn_start(agent=self.name, turn=turns, model=self.model_label())

            streamed: List[str] = []
            streamed_reasoning: List[str] = []

            def on_event(ev: Any) -> None:
                if isinstance(ev, TextDelta):
                    streamed.append(ev.text)
                    if ui:
                        ui.on_text(ev.text, agent=self.name)
                elif isinstance(ev, ReasoningDelta):
                    streamed_reasoning.append(ev.text)
                    if ui:
                        ui.on_reasoning(ev.text, agent=self.name)
                elif isinstance(ev, ToolCallStart):
                    if ui:
                        ui.on_tool_call_stream(ev.index, ev.name, agent=self.name)
                elif isinstance(ev, UsageEvent):
                    pass

            try:
                completion, target = self.services.router.complete(
                    messages, tools, spec=self.options.model_spec, options=opts,
                    on_event=on_event, allow_failover=self.options.allow_failover,
                )
            except NexusError as exc:
                result.errors.append(f"{type(exc).__name__}: {exc}")
                if ui:
                    ui.on_error(str(exc), agent=self.name)
                break
            except Exception as exc:  # unexpected
                self.log.exception("provider call failed", exc=exc)
                result.errors.append(f"unexpected provider error: {exc}")
                if ui:
                    ui.on_error(str(exc), agent=self.name)
                break

            self._target = target
            self.usage.merge(completion.usage)
            self.cost += target.info.cost(completion.usage)
            self.services.ledger.record(provider=target.provider_key, model=target.model,
                                        usage=completion.usage, info=target.info, agent=self.name,
                                        latency_ms=completion.latency_ms)
            result.usage = Usage(input_tokens=self.usage.input_tokens, output_tokens=self.usage.output_tokens,
                                 cached_tokens=self.usage.cached_tokens,
                                 reasoning_tokens=self.usage.reasoning_tokens,
                                 requests=self.usage.requests)
            result.cost = self.cost
            result.target = target

            assistant = completion.message
            text = assistant.text
            if text:
                text_parts.append(text)
            if completion.reasoning_text:
                reasoning_parts.append(completion.reasoning_text)
            self.history.append(assistant)
            result.finish_reason = completion.finish_reason
            if ui:
                ui.on_turn_end(agent=self.name, turn=turns, usage=completion.usage,
                               finish_reason=completion.finish_reason)

            if not assistant.tool_calls:
                if completion.finish_reason == "length" and self._continue_nudges < 1:
                    self._continue_nudges += 1
                    self.history.append(Message.user(
                        "[system] Your previous message hit the output token limit. Continue exactly "
                        "where you stopped, without repeating what you already wrote."))
                    continue
                break

            # ---- execute tools -------------------------------------------
            executed = self._execute_tools(assistant.tool_calls, result, identical)
            call_count += executed
            result.tool_calls = call_count
            if self._aborted():
                result.aborted = True
                break
            if any("repeated the identical" in e for e in result.errors):
                break
        else:
            result.errors.append(f"reached max_turns ({max_turns}) without a final answer")

        result.text = "\n".join(p for p in text_parts if p).strip()
        result.reasoning = "\n".join(p for p in reasoning_parts if p).strip()
        result.turns = turns
        result.latency_ms = int((time.monotonic() - started) * 1000)
        result.report = parse_report(result.text)
        if self.services.session is not None:
            try:
                self.services.session.record_turn(self, result)
            except Exception:  # session persistence must never break a turn
                self.log.warning("session.record_turn failed")
        return result

    # ------------------------------------------------------------------ #
    def _assemble(self, result: TurnResult) -> List[Message]:
        target = self._target
        model = target.model if target else ""
        window = target.info.context_window if target else 0
        max_out = self.options.max_tokens or (target.info.max_output_tokens if target else 0)
        messages, report = self.services.context.assemble(
            self.system_messages() + self.history, window=window, max_output=max_out, model=model)
        if report.actions and self.services.ui:
            self.services.ui.on_context_budget(report.describe(), agent=self.name)
        result.budget = report.describe()
        if report.overflow:
            raise ContextOverflowError(
                f"The conversation needs ~{report.after:,} tokens but the limit is {report.limit:,} "
                f"even after pruning ({'; '.join(report.actions)}).",
                hint="Run /compact to summarise the history, /clear to start fresh, or switch to a "
                     "model with a larger context window (/model).",
            )
        return messages

    def _request_options(self, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> RequestOptions:
        opts = RequestOptions(
            model=self.options.model_spec,
            temperature=self.options.temperature if self.options.temperature is not None else self.persona.temperature,
            max_tokens=self.options.max_tokens,
            top_p=self.options.top_p,
            json_mode=self.options.json_mode,
            reasoning_effort=self.options.reasoning_effort,
            seed=self.options.seed,
            stream=self.options.stream and not self.options.json_mode,
        )
        return opts

    def _aborted(self) -> bool:
        return bool(self.services.cancelled is not None and self.services.cancelled.is_set())

    # ------------------------------------------------------------------ #
    def _execute_tools(self, calls: Sequence[ToolCall], result: TurnResult,
                       identical: Dict[str, int]) -> int:
        """Run a batch of tool calls, guaranteeing one tool message per call."""
        ctx = self.services.tool_context(self.name)
        ui = self.services.ui
        resolved: List[Tuple[ToolCall, Any, Dict[str, Any], ToolResult, str]] = []
        blocked_signatures: List[str] = []

        # 1) resolve + validate + permission (cheap, serial, order preserving)
        for call in calls:
            signature = _signature(call)
            identical[signature] = identical.get(signature, 0) + 1
            if identical[signature] > IDENTICAL_CALL_LIMIT:
                blocked_signatures.append(signature)
            try:
                tool = self.services.registry.get(call.name)
            except ToolNotFound as exc:
                resolved.append((call, None, {}, ToolResult.fail(str(exc)), "missing-tool"))
                continue
            try:
                args = tool.prepare(call.parsed_args() if _is_json(call.arguments) else _repair_args(call), ctx)
            except ValidationError as exc:
                resolved.append((call, tool, {}, ToolResult.fail(
                    f"{exc}\nFix the arguments and call the tool again."), "invalid-args"))
                continue
            except Exception as exc:
                resolved.append((call, tool, {}, ToolResult.fail(f"Could not parse arguments: {exc}"), "invalid-args"))
                continue
            request = None
            try:
                request = tool.confirmation(args, ctx)
            except Exception as exc:
                self.log.warning("confirmation() failed", tool=tool.name, error=str(exc))
            decision = self.services.permissions.check(tool.name, args, request, read_only=tool.read_only)
            if not decision.allowed:
                result.denied.append(f"{tool.name}: {decision.reason}")
                resolved.append((call, tool, args, ToolResult.fail(
                    f"Permission denied: {decision.reason}. Do not retry this exact call; "
                    "either choose a different approach or ask the user for approval."), "denied"))
                if ui:
                    ui.on_tool_denied(tool.name, decision.reason, agent=self.name)
                continue
            resolved.append((call, tool, args, None, "pending"))  # type: ignore[arg-type]

        # 2) execute
        run_now = [r for r in resolved if r[4] == "pending"]
        parallel_ok = (self.options.parallel_tools and len(run_now) > 1
                       and all(r[1].read_only and r[1].concurrency_safe for r in run_now))
        outcomes: Dict[int, ToolResult] = {}

        def run_one(index: int, tool: Any, args: Dict[str, Any]) -> ToolResult:
            if ui:
                ui.on_tool_start(tool.name, args, agent=self.name)
            t0 = time.monotonic()
            try:
                out = tool.execute(args, ctx)
            except PermissionDenied as exc:
                out = ToolResult.fail(f"Blocked by policy: {exc}")
            except ValidationError as exc:
                out = ToolResult.fail(f"{exc}")
            except NexusError as exc:
                out = ToolResult.fail(f"{type(exc).__name__}: {exc}")
            except Exception as exc:
                self.log.exception("tool crashed", exc=exc, tool=tool.name)
                out = ToolResult.fail(f"Tool '{tool.name}' crashed: {type(exc).__name__}: {exc}")
            out.data.setdefault("elapsed_ms", int((time.monotonic() - t0) * 1000))
            if ui:
                ui.on_tool_end(tool.name, out, agent=self.name)
            return out

        if parallel_ok:
            with ThreadPoolExecutor(max_workers=min(MAX_PARALLEL_TOOLS, len(run_now))) as pool:
                futures = {}
                for i, (call, tool, args, _res, state) in enumerate(resolved):
                    if state != "pending":
                        continue
                    futures[pool.submit(run_one, i, tool, args)] = i
                for fut in futures:
                    idx = futures[fut]
                    try:
                        outcomes[idx] = fut.result()
                    except Exception as exc:  # pragma: no cover - run_one catches everything
                        self.log.exception("parallel tool failure", exc=exc)
                        outcomes[idx] = ToolResult.fail(f"Tool execution failed: {exc}")
        else:
            for i, (call, tool, args, res, state) in enumerate(resolved):
                if state != "pending":
                    continue
                if self._aborted():
                    outcomes[i] = ToolResult.fail("Aborted by user before execution.")
                    continue
                outcomes[i] = run_one(i, tool, args)

        # 3) append results -- one per call, in the original order
        executed = 0
        for i, (call, tool, args, res, state) in enumerate(resolved):
            if state == "pending":
                out = outcomes.get(i) or ToolResult.fail("Tool did not run (internal error).")
                executed += 1
            else:
                out = res
            name = tool.name if tool else call.name
            if state == "denied" or state in ("invalid-args", "missing-tool"):
                pass
            if blocked_signatures and _signature(call) in blocked_signatures:
                out = ToolResult.fail(
                    f"You repeated the identical call to '{name}' {IDENTICAL_CALL_LIMIT}+ times with the same "
                    "arguments. Stop and try a different approach, or report that you are blocked.")
                result.errors.append(f"repeated the identical {name} call")
            self.history.append(Message.tool_result(call.id or f"call_{i}", name, out.content,
                                                    is_error=bool(out.is_error)))
            if out.touched and self.services.session is not None:
                try:
                    self.services.session.note_touched(out.touched)
                except Exception:
                    pass
        # safety net: guarantee no orphan tool_calls remain in history
        self._heal_history()
        return executed

    def _heal_history(self) -> None:
        """Append placeholder tool results for any unanswered tool_call."""
        answered: set = set()
        for m in self.history:
            if m.role == "tool" and m.tool_call_id:
                answered.add(m.tool_call_id)
        for m in self.history:
            if m.role == "assistant" and m.tool_calls:
                for tc in m.tool_calls:
                    if tc.id and tc.id not in answered:
                        self.history.append(Message.tool_result(
                            tc.id, tc.name, "[no result: the turn was interrupted]", is_error=True))
                        answered.add(tc.id)

    def inject(self, text: str) -> None:
        """Add a system-style nudge to the conversation (used by the swarm)."""
        self.history.append(Message.user(f"[system] {text}"))


def _is_json(text: str) -> bool:
    t = (text or "").strip()
    return t.startswith("{") or t.startswith("[") or t == ""


def _repair_args(call: ToolCall) -> Dict[str, Any]:
    from ..providers.base import repair_json

    obj = repair_json(call.arguments or "")
    return obj if isinstance(obj, dict) else {"_raw": call.arguments}


def _signature(call: ToolCall) -> str:
    raw = (call.arguments or "").strip()
    digest = hashlib.sha1(f"{call.name}|{raw}".encode("utf-8", "replace")).hexdigest()[:16]
    return f"{call.name}:{digest}"


__all__ = ["Agent", "AgentOptions", "Services", "TurnResult", "parse_report", "environment_block"]
