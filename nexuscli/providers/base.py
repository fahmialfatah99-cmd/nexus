"""Provider abstraction layer.

NEXUS speaks **one** internal message format. Every backend (OpenAI-compatible,
Anthropic Messages, Google Gemini) is an adapter that translates to/from it.
That single seam is what makes multi-provider support reliable instead of a
pile of special cases leaking into the agent loop.

Internal format
---------------
``Message(role, content, tool_calls, tool_call_id, name)`` where content is
either a ``str`` or a list of :class:`TextBlock` / :class:`ImageBlock`.

Streaming is normalised too: providers emit :class:`StreamEvent` objects and
:class:`StreamAggregator` reassembles the final message, so the UI and the agent
loop never see provider-specific delta shapes.
"""

from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Union

from ..core.errors import ProviderError, ValidationError
from ..transport.http import Transport

# --------------------------------------------------------------------------- #
# Content blocks
# --------------------------------------------------------------------------- #


@dataclass
class TextBlock:
    text: str

    def to_plain(self) -> str:
        return self.text


@dataclass
class ImageBlock:
    data: Optional[bytes] = None
    url: Optional[str] = None
    mime: str = "image/png"
    detail: str = "auto"

    def as_data_url(self) -> str:
        if self.url and not self.data:
            return self.url
        b64 = base64.b64encode(self.data or b"").decode("ascii")
        return f"data:{self.mime};base64,{b64}"

    def to_plain(self) -> str:
        return "[image]"


Block = Union[TextBlock, ImageBlock]
Content = Union[str, List[Block]]


def content_to_text(content: Content) -> str:
    if isinstance(content, str):
        return content
    out = []
    for b in content:
        if isinstance(b, TextBlock):
            out.append(b.text)
        else:
            out.append("[image]")
    return "\n".join(p for p in out if p)


@dataclass
class ToolCall:
    id: str
    name: str
    #: Raw JSON text. Starts EMPTY (not "{}") so that streamed argument
    #: fragments concatenate cleanly; normalised to "{}" when consumed.
    arguments: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def parsed_args(self) -> Dict[str, Any]:
        """Parse arguments defensively: models occasionally emit junk."""
        raw = (self.arguments or "").strip()
        if not raw:
            return {}
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            obj = repair_json(raw)
            if obj is None:
                raise ValidationError(f"Tool '{self.name}' received malformed JSON arguments: {raw[:200]}")
        if isinstance(obj, list) and len(obj) == 1 and isinstance(obj[0], dict):
            obj = obj[0]
        if not isinstance(obj, dict):
            raise ValidationError(f"Tool '{self.name}' arguments must be a JSON object, got {type(obj).__name__}")
        return obj


def repair_json(raw: str) -> Optional[Any]:
    """Best-effort recovery of truncated / decorated JSON emitted by models."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
        if "```" in text:
            text = text.split("```")[0].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Trim to the outermost braces / brackets, then balance them.
    start = min([i for i in (text.find("{"), text.find("[")) if i >= 0], default=-1)
    if start < 0:
        return None
    text = text[start:]
    for candidate in (text, _strip_trailing_commas(text), _balance(text)):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return _repair_by_truncation(text)


def _repair_by_truncation(text: str, max_candidates: int = 80) -> Optional[Any]:
    """Recover JSON whose tail is an incomplete key/value pair.

    ``{"path": "a.txt", "lim`` cannot be fixed by bracket balancing alone -- the
    dangling key has no value. We therefore retry from the end, cutting at each
    "safe" boundary (``,`` ``}`` ``]`` ``"``) and re-balancing, keeping the
    longest prefix that parses.
    """
    cuts = [len(text)]
    cuts += [i for i, ch in enumerate(text) if ch in ",}]\""]
    cuts.sort(reverse=True)
    for i in cuts[:max_candidates]:
        head = text[:i].rstrip()
        if not head:
            continue
        for candidate in (head, _strip_trailing_commas(head), _balance(_strip_trailing_commas(head))):
            try:
                obj = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, (dict, list)):
                return obj
    return None


def _balance(text: str) -> str:
    """Close open strings/brackets -- handles truncated streaming output."""
    out: List[str] = []
    stack: List[str] = []
    in_str = False
    esc = False
    for ch in text:
        out.append(ch)
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]" and stack:
            stack.pop()
    fixed = _strip_trailing_commas("".join(out))
    if in_str:
        if esc:
            fixed = fixed[:-1]
        fixed += '"'
    if fixed.rstrip().endswith(","):
        fixed = fixed.rstrip()[:-1]
    return _strip_trailing_commas(fixed + "".join(reversed(stack)))


def _strip_trailing_commas(text: str) -> str:
    """Remove ``,`` immediately before a closing bracket, outside of strings.

    Models emit ``{"a": 1,}`` often enough that tolerating it saves a whole
    retry round-trip. Index-based scan so skipping a comma cannot desync.
    """
    out: List[str] = []
    in_str = False
    esc = False
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if in_str:
            out.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            i += 1
            continue
        if ch == '"':
            in_str = True
            out.append(ch)
            i += 1
            continue
        if ch == ",":
            j = i + 1
            while j < n and text[j] in " \t\r\n":
                j += 1
            if j < n and text[j] in "}]":
                i += 1  # drop the comma
                continue
        out.append(ch)
        i += 1
    return "".join(out)


# --------------------------------------------------------------------------- #
# Messages
# --------------------------------------------------------------------------- #


@dataclass
class Message:
    role: str  # system | user | assistant | tool
    content: Content = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    tool_call_id: Optional[str] = None
    name: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    # -- helpers ----------------------------------------------------------
    @property
    def text(self) -> str:
        return content_to_text(self.content)

    def has_images(self) -> bool:
        return isinstance(self.content, list) and any(isinstance(b, ImageBlock) for b in self.content)

    def append_text(self, text: str) -> None:
        if isinstance(self.content, str):
            self.content += text
        else:
            if self.content and isinstance(self.content[-1], TextBlock):
                self.content[-1].text += text
            else:
                self.content.append(TextBlock(text))

    def clone(self) -> "Message":
        content: Content
        if isinstance(self.content, str):
            content = self.content
        else:
            content = list(self.content)
        return Message(
            role=self.role,
            content=content,
            tool_calls=[ToolCall(c.id, c.name, c.arguments, dict(c.extra)) for c in self.tool_calls],
            tool_call_id=self.tool_call_id,
            name=self.name,
            meta=dict(self.meta),
        )

    def is_blank(self) -> bool:
        return not self.text.strip() and not self.tool_calls

    @staticmethod
    def system(text: str) -> "Message":
        return Message(role="system", content=text)

    @staticmethod
    def user(text: Union[str, List[Block]]) -> "Message":
        return Message(role="user", content=text)

    @staticmethod
    def tool_result(tool_call_id: str, name: str, text: str, *, is_error: bool = False) -> "Message":
        return Message(role="tool", content=text, tool_call_id=tool_call_id, name=name,
                       meta={"is_error": is_error})


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}})

    def to_json_schema(self) -> Dict[str, Any]:
        return {"type": "object", "properties": self.parameters.get("properties", {}),
                "required": self.parameters.get("required", [])}


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    requests: int = 1

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def merge(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cached_tokens += other.cached_tokens
        self.reasoning_tokens += other.reasoning_tokens
        self.requests += other.requests

    def as_dict(self) -> Dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_tokens": self.cached_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "total_tokens": self.total_tokens,
            "requests": self.requests,
        }


@dataclass
class ModelInfo:
    id: str
    provider: str = ""
    context_window: int = 0
    max_output_tokens: int = 0
    supports_tools: bool = True
    supports_images: bool = False
    supports_streaming: bool = True
    supports_reasoning: bool = False
    input_cost_per_mtok: float = 0.0
    output_cost_per_mtok: float = 0.0
    aliases: List[str] = field(default_factory=list)
    family: str = ""

    def cost(self, usage: Usage) -> float:
        return (usage.input_tokens * self.input_cost_per_mtok + usage.output_tokens * self.output_cost_per_mtok) / 1_000_000


@dataclass
class RequestOptions:
    model: str = ""
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    stop: Optional[List[str]] = None
    json_mode: bool = False
    json_schema: Optional[Dict[str, Any]] = None
    reasoning_effort: Optional[str] = None  # low|medium|high|None
    tool_choice: Optional[str] = None  # auto|none|required
    seed: Optional[int] = None
    stream: bool = True
    extra_body: Dict[str, Any] = field(default_factory=dict)
    timeout: Optional[float] = None


@dataclass
class Completion:
    message: Message
    usage: Usage = field(default_factory=Usage)
    finish_reason: str = "stop"
    model: str = ""
    provider: str = ""
    latency_ms: int = 0
    reasoning_text: str = ""
    raw: Optional[Any] = None

    @property
    def wants_tools(self) -> bool:
        return bool(self.message.tool_calls)


# --------------------------------------------------------------------------- #
# Stream events
# --------------------------------------------------------------------------- #


@dataclass
class TextDelta:
    text: str


@dataclass
class ReasoningDelta:
    text: str


@dataclass
class ToolCallStart:
    index: int
    id: str = ""
    name: str = ""


@dataclass
class ToolCallArgsDelta:
    index: int
    args: str


@dataclass
class UsageEvent:
    usage: Usage


@dataclass
class FinishEvent:
    reason: str


StreamEvent = Union[TextDelta, ReasoningDelta, ToolCallStart, ToolCallArgsDelta, UsageEvent, FinishEvent]
EventHandler = Optional[Callable[[StreamEvent], None]]


class StreamAggregator:
    """Reassembles normalised stream events into a final :class:`Completion`."""

    def __init__(self) -> None:
        self.text: List[str] = []
        self.reasoning: List[str] = []
        self.calls: Dict[int, ToolCall] = {}
        self.order: List[int] = []
        self.usage = Usage(requests=0)
        self.finish_reason = "stop"

    def handle(self, ev: StreamEvent) -> None:
        if isinstance(ev, TextDelta):
            self.text.append(ev.text)
        elif isinstance(ev, ReasoningDelta):
            self.reasoning.append(ev.text)
        elif isinstance(ev, ToolCallStart):
            if ev.index not in self.calls:
                self.calls[ev.index] = ToolCall(id=ev.id, name=ev.name)
                self.order.append(ev.index)
            else:
                if ev.id:
                    self.calls[ev.index].id = ev.id
                if ev.name:
                    self.calls[ev.index].name = ev.name
        elif isinstance(ev, ToolCallArgsDelta):
            call = self.calls.get(ev.index)
            if call is None:
                call = ToolCall(id=f"call_{ev.index}", name="")
                self.calls[ev.index] = call
                self.order.append(ev.index)
            call.arguments += ev.args
        elif isinstance(ev, UsageEvent):
            # Providers may send usage twice (mid + final); keep the maximums.
            u = ev.usage
            self.usage.input_tokens = max(self.usage.input_tokens, u.input_tokens)
            self.usage.output_tokens = max(self.usage.output_tokens, u.output_tokens)
            self.usage.cached_tokens = max(self.usage.cached_tokens, u.cached_tokens)
            self.usage.reasoning_tokens = max(self.usage.reasoning_tokens, u.reasoning_tokens)
            self.usage.requests = 1
        elif isinstance(ev, FinishEvent):
            if ev.reason:
                self.finish_reason = ev.reason

    def tool_calls(self) -> List[ToolCall]:
        calls = [self.calls[i] for i in sorted(self.order)]
        cleaned: List[ToolCall] = []
        for i, c in enumerate(calls):
            if not c.id:
                c.id = f"call_{int(time.time() * 1000)}_{i}"
            if not c.arguments.strip():
                c.arguments = "{}"  # normalise empty (providers differ)
            cleaned.append(c)
        return cleaned

    def build(self, *, model: str = "", provider: str = "", latency_ms: int = 0, raw: Any = None) -> Completion:
        calls = self.tool_calls()
        reason = self.finish_reason
        if calls and reason in ("stop", "", "end_turn"):
            reason = "tool_calls"
        msg = Message(role="assistant", content="".join(self.text), tool_calls=calls)
        if self.usage.requests == 0:
            self.usage.requests = 1
        return Completion(message=msg, usage=self.usage, finish_reason=reason or "stop", model=model,
                          provider=provider, latency_ms=latency_ms, reasoning_text="".join(self.reasoning), raw=raw)


# --------------------------------------------------------------------------- #
# Base provider
# --------------------------------------------------------------------------- #


class BaseProvider:
    """Adapter base class. Subclasses implement :meth:`_stream`/_meth:`_invoke`."""

    key: str = "base"
    display_name: str = "Base"
    default_base_url: str = ""
    default_model: str = ""
    requires_api_key: bool = True
    env_keys: Sequence[str] = ()
    models_url: str = ""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        transport: Optional[Transport] = None,
        timeout: float = 300.0,
        max_retries: int = 4,
        headers: Optional[Dict[str, str]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        from ..transport.http import HTTPTransport, RetryPolicy

        self.api_key = api_key or ""
        self.base_url = (base_url or self.default_base_url or "").rstrip("/")
        self.timeout = timeout
        self.headers = dict(headers or {})
        self.extra = dict(extra or {})
        self.transport: Transport = transport or HTTPTransport(RetryPolicy(max_attempts=max_retries))
        if self.requires_api_key and not self.api_key:
            raise ProviderError(
                f"{self.display_name} requires an API key.",
                provider=self.key,
                hint=f"export {self.env_keys[0] if self.env_keys else 'API_KEY'}=... or run `nexus auth login {self.key}`",
            )

    # -- response validation ----------------------------------------------
    def _raise_for_status(self, resp) -> None:
        """Map a non-2xx response onto the NEXUS error hierarchy.

        Lives on the provider (not only in the HTTP transport) so that *every*
        transport -- including the offline mock used by tests -- produces the
        same exceptions for the same status codes.
        """
        if 200 <= resp.status < 300:
            return
        from ..transport.http import http_error

        raise http_error(resp.status, resp.body, resp.headers, getattr(resp, "url", ""))

    # -- to be implemented by subclasses ----------------------------------
    def _invoke(self, messages: Sequence[Message], tools: Sequence[ToolSpec], opts: RequestOptions) -> Completion:
        raise NotImplementedError

    def _stream(
        self, messages: Sequence[Message], tools: Sequence[ToolSpec], opts: RequestOptions, emit: Callable[[StreamEvent], None]
    ) -> Completion:
        raise NotImplementedError

    def list_models(self) -> List[str]:
        return []

    # -- shared entry point ------------------------------------------------
    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] = (),
        options: Optional[RequestOptions] = None,
        on_event: EventHandler = None,
    ) -> Completion:
        opts = options or RequestOptions()
        self._validate(messages, opts)
        started = time.monotonic()
        try:
            if opts.stream:
                def emit(ev: StreamEvent) -> None:
                    if on_event is not None:
                        on_event(ev)

                result = self._stream(messages, tools, opts, emit)
            else:
                result = self._invoke(messages, tools, opts)
                if on_event is not None:
                    if result.message.text:
                        on_event(TextDelta(result.message.text))
                    for i, tc in enumerate(result.message.tool_calls):
                        on_event(ToolCallStart(i, tc.id, tc.name))
                        on_event(ToolCallArgsDelta(i, tc.arguments))
                    on_event(UsageEvent(result.usage))
                    on_event(FinishEvent(result.finish_reason))
        finally:
            pass
        result.latency_ms = int((time.monotonic() - started) * 1000)
        result.provider = result.provider or self.key
        if result.usage.requests < 1:
            result.usage.requests = 1  # exactly one API call produced this result
        return result

    def _validate(self, messages: Sequence[Message], opts: RequestOptions) -> None:
        if not messages:
            raise ValidationError("Cannot send an empty message list to the provider.")
        if any(m.role not in ("system", "user", "assistant", "tool") for m in messages):
            bad = next(m.role for m in messages if m.role not in ("system", "user", "assistant", "tool"))
            raise ValidationError(f"Unknown message role: {bad!r}")
        # Every tool result must reference an existing assistant tool_call.
        seen: set = set()
        for m in messages:
            if m.role == "assistant":
                seen.update(tc.id for tc in m.tool_calls)
            elif m.role == "tool":
                if m.tool_call_id and seen and m.tool_call_id not in seen:
                    raise ValidationError(f"Orphan tool result for id {m.tool_call_id!r}")

    # -- helpers -----------------------------------------------------------
    def _headers(self) -> Dict[str, str]:
        h = dict(self.headers)
        if self.api_key:
            h.setdefault("Authorization", f"Bearer {self.api_key}")
        return h

    def _url(self, path: str) -> str:
        if path.startswith("http"):
            return path
        return f"{self.base_url}/{path.lstrip('/')}"


__all__ = [
    "TextBlock", "ImageBlock", "Block", "Content", "content_to_text",
    "ToolCall", "repair_json", "Message", "ToolSpec", "Usage", "ModelInfo",
    "RequestOptions", "Completion", "TextDelta", "ReasoningDelta", "ToolCallStart",
    "ToolCallArgsDelta", "UsageEvent", "FinishEvent", "StreamEvent", "EventHandler",
    "StreamAggregator", "BaseProvider",
]
