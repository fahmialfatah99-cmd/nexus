"""OpenAI-compatible provider.

One adapter covers every service that speaks the OpenAI Chat Completions wire
protocol: OpenAI, Azure OpenAI, Groq, OpenRouter, DeepSeek, Together, Fireworks,
xAI, Mistral, Cerebras, Moonshot, Qwen/DashScope (compatible-mode), vLLM,
LM Studio, Ollama and any self-hosted gateway.

Provider-specific quirks are handled explicitly rather than guessed:
  * ``max_tokens`` vs ``max_completion_tokens`` (OpenAI reasoning models)
  * temperature/top_p rejected by reasoning models
  * ``reasoning_content`` / ``reasoning`` deltas (DeepSeek, Groq, Qwen)
  * ``stream_options.include_usage`` (ignored by servers that do not know it)
  * empty-string assistant content alongside tool_calls (widest compatibility)
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..core.errors import ProviderError
from ..core.logging_ import get_logger
from .base import (
    BaseProvider,
    Completion,
    FinishEvent,
    ImageBlock,
    Message,
    ModelInfo,
    ReasoningDelta,
    RequestOptions,
    StreamAggregator,
    TextBlock,
    TextDelta,
    ToolCallArgsDelta,
    ToolCallStart,
    ToolSpec,
    Usage,
    UsageEvent,
    content_to_text,
)

REASONING_PREFIXES = ("o1", "o3", "o4", "gpt-5")


def _is_reasoning_model(model: str) -> bool:
    m = (model or "").lower().split("/")[-1]
    return any(m.startswith(p) for p in REASONING_PREFIXES) or "-thinking" in m or m.endswith(":reasoning")


class OpenAICompatProvider(BaseProvider):
    key = "openai"
    display_name = "OpenAI"
    default_base_url = "https://api.openai.com/v1"
    default_model = "gpt-4o"
    env_keys = ("OPENAI_API_KEY",)

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.log = get_logger()
        self.chat_path: str = self.extra.get("chat_path", "/chat/completions")
        self.models_path: str = self.extra.get("models_path", "/models")
        self.send_stream_options: bool = bool(self.extra.get("stream_options", True))

    # ------------------------------------------------------------------ #
    # Request construction
    # ------------------------------------------------------------------ #
    def _wire_messages(self, messages: Sequence[Message]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        system_parts: List[str] = []
        for m in messages:
            if m.role == "system":
                system_parts.append(content_to_text(m.content))
                continue
            if m.role == "assistant":
                text = content_to_text(m.content)
                entry: Dict[str, Any] = {"role": "assistant", "content": text}
                if m.tool_calls:
                    entry["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.name, "arguments": tc.arguments or "{}"},
                        }
                        for tc in m.tool_calls
                    ]
                if m.name:
                    entry["name"] = m.name
                out.append(entry)
            elif m.role == "tool":
                entry = {"role": "tool", "tool_call_id": m.tool_call_id or "", "content": content_to_text(m.content)}
                if m.name:
                    entry["name"] = m.name
                out.append(entry)
            else:  # user
                out.append({"role": "user", "content": self._wire_content(m.content)})
        if system_parts:
            out.insert(0, {"role": "system", "content": "\n\n".join(p for p in system_parts if p)})
        return out

    def _wire_content(self, content: Any) -> Any:
        if isinstance(content, str):
            return content
        parts: List[Dict[str, Any]] = []
        for b in content:
            if isinstance(b, TextBlock):
                if b.text:
                    parts.append({"type": "text", "text": b.text})
            elif isinstance(b, ImageBlock):
                img: Dict[str, Any] = {"url": b.as_data_url()}
                if b.detail and b.detail != "auto":
                    img["detail"] = b.detail
                parts.append({"type": "image_url", "image_url": img})
        if not parts:
            return ""
        if len(parts) == 1 and parts[0]["type"] == "text":
            return parts[0]["text"]
        return parts

    def _wire_tools(self, tools: Sequence[ToolSpec]) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters or {"type": "object", "properties": {}},
                },
            }
            for t in tools
        ]

    def _body(self, messages: Sequence[Message], tools: Sequence[ToolSpec], opts: RequestOptions, stream: bool) -> Dict[str, Any]:
        body: Dict[str, Any] = {"model": opts.model, "messages": self._wire_messages(messages)}
        reasoning = _is_reasoning_model(opts.model)
        if stream:
            body["stream"] = True
            if self.send_stream_options:
                body["stream_options"] = {"include_usage": True}
        if tools:
            body["tools"] = self._wire_tools(tools)
            if opts.tool_choice:
                body["tool_choice"] = opts.tool_choice
            elif not reasoning:
                body["tool_choice"] = "auto"
        if opts.temperature is not None and not reasoning:
            body["temperature"] = opts.temperature
        if opts.top_p is not None and not reasoning:
            body["top_p"] = opts.top_p
        if opts.stop:
            body["stop"] = opts.stop[:4]
        if opts.seed is not None and not reasoning:
            body["seed"] = opts.seed
        if opts.max_tokens:
            key = "max_completion_tokens" if reasoning or self.extra.get("use_max_completion_tokens") else "max_tokens"
            body[key] = opts.max_tokens
        if opts.reasoning_effort and (reasoning or self.extra.get("allow_reasoning_effort")):
            body["reasoning_effort"] = opts.reasoning_effort
        if opts.json_schema:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": opts.json_schema.get("name", "response"),
                                "schema": opts.json_schema.get("schema", opts.json_schema), "strict": True},
            }
        elif opts.json_mode:
            body["response_format"] = {"type": "json_object"}
        body.update(self.extra.get("extra_body", {}) or {})
        body.update(opts.extra_body or {})
        return body

    def _headers(self) -> Dict[str, str]:
        h = super()._headers()
        if self.extra.get("api_version"):
            h["api-version"] = str(self.extra["api_version"])
        if self.extra.get("org"):
            h["OpenAI-Organization"] = str(self.extra["org"])
        if self.extra.get("project"):
            h["OpenAI-Project"] = str(self.extra["project"])
        # OpenRouter attribution (harmless for other gateways).
        if "openrouter" in self.base_url:
            h.setdefault("HTTP-Referer", "https://nexus.local")
            h.setdefault("X-Title", "NEXUS CLI")
        return h

    # ------------------------------------------------------------------ #
    # Non-streaming
    # ------------------------------------------------------------------ #
    def _invoke(self, messages, tools, opts) -> Completion:
        body = self._body(messages, tools, opts, stream=False)
        resp, _ = self.transport.request(
            "POST", self._url(self.chat_path), headers=self._headers(), json_body=body,
            timeout=opts.timeout or self.timeout,
        )
        self._raise_for_status(resp)
        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            raise ProviderError(f"Provider returned no choices: {json.dumps(data)[:400]}", provider=self.key)
        choice = choices[0]
        msg = choice.get("message") or {}
        calls = []
        for i, tc in enumerate(msg.get("tool_calls") or []):
            fn = tc.get("function") or {}
            calls.append(_tool_call_from_wire(tc.get("id") or f"call_{i}", fn.get("name") or "", fn.get("arguments") or "{}"))
        out = Message(role="assistant", content=msg.get("content") or "", tool_calls=calls)
        reasoning_text = msg.get("reasoning_content") or msg.get("reasoning") or ""
        usage = _usage_from_wire(data.get("usage") or {})
        return Completion(message=out, usage=usage, finish_reason=_norm_finish(choice.get("finish_reason")),
                          model=data.get("model") or opts.model, provider=self.key,
                          reasoning_text=reasoning_text, raw=data)

    # ------------------------------------------------------------------ #
    # Streaming
    # ------------------------------------------------------------------ #
    def _stream(self, messages, tools, opts, emit) -> Completion:
        body = self._body(messages, tools, opts, stream=True)
        resp, handle = self.transport.request(
            "POST", self._url(self.chat_path), headers=self._headers(), json_body=body,
            timeout=opts.timeout or self.timeout, stream=True,
        )
        agg = StreamAggregator()
        model_name = opts.model
        assert handle is not None
        try:
            from ..transport.http import iter_sse

            for ev in iter_sse(handle):
                if ev.data.strip() == "[DONE]":
                    break
                try:
                    chunk = ev.json()
                except json.JSONDecodeError:
                    self.log.warning("openai: skipping malformed SSE chunk", data=ev.data[:200])
                    continue
                if isinstance(chunk, dict) and chunk.get("error"):
                    err = chunk["error"]
                    raise ProviderError(f"Stream error: {err.get('message', err)}", provider=self.key)
                if chunk.get("model"):
                    model_name = chunk["model"]
                if chunk.get("usage"):
                    u = _usage_from_wire(chunk["usage"])
                    agg.handle(UsageEvent(u))
                    emit(UsageEvent(u))
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    text = delta.get("content")
                    if text:
                        agg.handle(TextDelta(text))
                        emit(TextDelta(text))
                    think = delta.get("reasoning_content") or delta.get("reasoning")
                    if think:
                        agg.handle(ReasoningDelta(think))
                        emit(ReasoningDelta(think))
                    for tc in delta.get("tool_calls") or []:
                        idx = tc.get("index")
                        if idx is None:
                            idx = len(agg.order)
                        fn = tc.get("function") or {}
                        if tc.get("id") or fn.get("name"):
                            start = ToolCallStart(int(idx), tc.get("id") or "", fn.get("name") or "")
                            agg.handle(start)
                            emit(start)
                        args = fn.get("arguments")
                        if args:
                            d = ToolCallArgsDelta(int(idx), args)
                            agg.handle(d)
                            emit(d)
                    if choice.get("finish_reason"):
                        f = FinishEvent(_norm_finish(choice["finish_reason"]))
                        agg.handle(f)
                        emit(f)
        finally:
            handle.close()
        result = agg.build(model=model_name, provider=self.key)
        result.raw = None
        return result

    # ------------------------------------------------------------------ #
    def list_models(self) -> List[str]:
        if not self.models_path:
            return []
        resp, _ = self.transport.request("GET", self._url(self.models_path), headers=self._headers(), timeout=30)
        self._raise_for_status(resp)
        data = resp.json()
        items = data.get("data") or data.get("models") or []
        ids = []
        for it in items:
            if isinstance(it, str):
                ids.append(it)
            elif isinstance(it, dict):
                mid = it.get("id") or it.get("name")
                if mid:
                    ids.append(str(mid))
        return sorted(set(ids))


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _tool_call_from_wire(tc_id: str, name: str, arguments: Any) -> Any:
    from .base import ToolCall

    if not isinstance(arguments, str):
        arguments = json.dumps(arguments, ensure_ascii=False)
    return ToolCall(id=tc_id, name=name, arguments=arguments)


def _usage_from_wire(u: Dict[str, Any]) -> Usage:
    details_in = u.get("prompt_tokens_details") or {}
    details_out = u.get("completion_tokens_details") or {}
    inp = int(u.get("prompt_tokens") or u.get("input_tokens") or 0)
    outp = int(u.get("completion_tokens") or u.get("output_tokens") or 0)
    return Usage(
        input_tokens=inp,
        output_tokens=outp,
        cached_tokens=int(details_in.get("cached_tokens") or u.get("cached_tokens") or 0),
        reasoning_tokens=int(details_out.get("reasoning_tokens") or 0),
        requests=0,
    )


def _norm_finish(reason: Optional[str]) -> str:
    mapping = {
        "stop": "stop",
        "end_turn": "stop",
        "tool_calls": "tool_calls",
        "function_call": "tool_calls",
        "length": "length",
        "max_tokens": "length",
        "content_filter": "content_filter",
        "null": "stop",
    }
    if reason is None:
        return "stop"
    return mapping.get(str(reason).lower(), str(reason))


# Re-exported for other modules that build OpenAI-shaped payloads.
__all__ = ["OpenAICompatProvider", "ModelInfo", "content_to_text"]
