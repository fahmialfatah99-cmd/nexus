"""Mock/offline provider.

Three jobs:
1. Lets the whole agent loop (streaming, tool calls, retries, failover, swarm)
   be unit-tested with zero network access and zero API keys.
2. Powers ``nexus demo`` so a first-time user can explore the UI immediately.
3. Acts as the ``echo`` fallback when no credentials are configured.

Scripts are plain data, e.g.::

    MockProvider(scripts=["hello", {"tool_calls": [("read_file", {"path": "x"})]}])
"""

from __future__ import annotations

import json
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from ..core.errors import ProviderError
from .base import (
    BaseProvider,
    Completion,
    FinishEvent,
    Message,
    RequestOptions,
    StreamAggregator,
    TextDelta,
    ToolCall,
    ToolCallArgsDelta,
    ToolCallStart,
    ToolSpec,
    Usage,
    UsageEvent,
    content_to_text,
)

Script = Union[str, Dict[str, Any], Callable[..., Any], BaseException]


class MockProvider(BaseProvider):
    key = "mock"
    display_name = "Mock (offline)"
    default_base_url = "mock://local"
    default_model = "mock-1"
    requires_api_key = False

    def __init__(self, *, scripts: Optional[Sequence[Script]] = None, chunk_sleep: float = 0.0,
                 chunk_size: int = 6,
                 dispatcher: Optional[Callable[[Sequence[Message]], Script]] = None,
                 **kwargs: Any) -> None:
        kwargs.setdefault("api_key", "mock")
        super().__init__(**kwargs)
        self.scripts: List[Script] = list(scripts or [])
        self.chunk_sleep = chunk_sleep
        self.chunk_size = max(1, chunk_size)
        #: When set, the reply is computed from the messages instead of the FIFO
        #: queue. This is how swarm tests give each persona its own behaviour
        #: even when workers run concurrently.
        self.dispatcher = dispatcher
        self.requests: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    def queue(self, *scripts: Script) -> "MockProvider":
        self.scripts.extend(scripts)
        return self

    def _next_script(self, messages, tools, opts) -> Script:
        if self.dispatcher is not None:
            return self.dispatcher(messages)
        if self.scripts:
            return self.scripts.pop(0)
        last = ""
        for m in reversed(messages):
            if m.role == "user":
                last = content_to_text(m.content)
                break
        return f"[mock echo] {last.strip()[:500]}"

    def _coerce(self, script: Script, messages, tools, opts) -> Dict[str, Any]:
        if isinstance(script, BaseException):
            raise script
        if isinstance(script, Exception):  # pragma: no cover - defensive
            raise script
        if callable(script):
            script = script(messages, tools, opts)
            if isinstance(script, BaseException):
                raise script
        if isinstance(script, str):
            return {"text": script}
        if isinstance(script, dict):
            return script
        raise ProviderError(f"Unsupported mock script: {script!r}", provider=self.key)

    def _invoke(self, messages, tools, opts) -> Completion:
        self.requests.append({"messages": [m.clone() for m in messages], "tools": [t.name for t in tools], "opts": opts})
        spec = self._coerce(self._next_script(messages, tools, opts), messages, tools, opts)
        return _completion_from_spec(spec, opts, self.key, streamed=False)

    def _stream(self, messages, tools, opts, emit) -> Completion:
        self.requests.append({"messages": [m.clone() for m in messages], "tools": [t.name for t in tools], "opts": opts})
        spec = self._coerce(self._next_script(messages, tools, opts), messages, tools, opts)
        agg = StreamAggregator()
        text = spec.get("text", "")
        for i in range(0, len(text), self.chunk_size):
            piece = text[i : i + self.chunk_size]
            agg.handle(TextDelta(piece))
            emit(TextDelta(piece))
            if self.chunk_sleep:
                time.sleep(self.chunk_sleep)
        for i, tc in enumerate(_tool_calls(spec.get("tool_calls") or [])):
            start = ToolCallStart(i, tc.id, tc.name)
            agg.handle(start)
            emit(start)
            d = ToolCallArgsDelta(i, tc.arguments)
            agg.handle(d)
            emit(d)
        usage = _usage(spec.get("usage"))
        agg.handle(UsageEvent(usage))
        emit(UsageEvent(usage))
        finish = FinishEvent(spec.get("finish_reason") or ("tool_calls" if spec.get("tool_calls") else "stop"))
        agg.handle(finish)
        emit(finish)
        result = agg.build(model=opts.model or self.default_model, provider=self.key)
        result.raw = spec
        return result

    def list_models(self) -> List[str]:
        return ["mock-1", "mock-mini", "mock-pro"]


def _tool_calls(raw: Sequence[Any]) -> List[ToolCall]:
    out: List[ToolCall] = []
    for i, item in enumerate(raw):
        if isinstance(item, ToolCall):
            out.append(item)
        elif isinstance(item, dict):
            args = item.get("arguments", item.get("args", {}))
            out.append(ToolCall(id=item.get("id") or f"mock_call_{i}", name=item.get("name") or "",
                                arguments=args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)))
        elif isinstance(item, (tuple, list)) and len(item) >= 2:
            name, args = item[0], item[1]
            out.append(ToolCall(id=f"mock_call_{i}", name=str(name),
                                arguments=args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)))
        else:
            raise ProviderError(f"Unsupported mock tool_call entry: {item!r}", provider="mock")
    return out


def _usage(raw: Any) -> Usage:
    if isinstance(raw, Usage):
        return raw
    if isinstance(raw, dict):
        return Usage(input_tokens=int(raw.get("input_tokens") or raw.get("prompt_tokens") or 0),
                     output_tokens=int(raw.get("output_tokens") or raw.get("completion_tokens") or 0),
                     cached_tokens=int(raw.get("cached_tokens") or 0), requests=0)
    return Usage(input_tokens=10, output_tokens=20, requests=0)


def _completion_from_spec(spec: Dict[str, Any], opts: RequestOptions, provider: str, *, streamed: bool) -> Completion:
    calls = _tool_calls(spec.get("tool_calls") or [])
    msg = Message(role="assistant", content=spec.get("text", ""), tool_calls=calls)
    reason = spec.get("finish_reason") or ("tool_calls" if calls else "stop")
    return Completion(message=msg, usage=_usage(spec.get("usage")), finish_reason=reason,
                      model=opts.model or "mock-1", provider=provider, raw=spec)


__all__ = ["MockProvider", "Script"]
