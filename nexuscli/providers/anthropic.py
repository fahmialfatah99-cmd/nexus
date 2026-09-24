"""Anthropic Messages API provider (Claude).

Handled explicitly because Anthropic's wire format differs from OpenAI in ways
that silently break naive adapters:

* ``max_tokens`` is **required**.
* ``system`` is a top-level field, not a message role.
* Tool results are ``tool_result`` content blocks inside a **user** message.
* Messages must alternate; consecutive same-role turns must be merged.
* Extended-thinking models require the ``thinking`` blocks (with their
  ``signature``) to be replayed verbatim when returning tool results, otherwise
  the API rejects the request.
* Streaming tool arguments arrive as ``input_json_delta.partial_json``.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

from ..core.errors import ProviderError
from ..core.logging_ import get_logger
from .base import (
    BaseProvider,
    Completion,
    FinishEvent,
    ImageBlock,
    Message,
    ReasoningDelta,
    RequestOptions,
    StreamAggregator,
    TextBlock,
    TextDelta,
    ToolCall,
    ToolCallArgsDelta,
    ToolCallStart,
    ToolSpec,
    Usage,
    UsageEvent,
    content_to_text,
)

ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_MAX_TOKENS = 4096

_FINISH_MAP = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
    "pause_turn": "stop",
    "refusal": "content_filter",
}


class AnthropicProvider(BaseProvider):
    key = "anthropic"
    display_name = "Anthropic"
    default_base_url = "https://api.anthropic.com"
    default_model = "claude-sonnet-4-5"
    env_keys = ("ANTHROPIC_API_KEY",)

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.log = get_logger()
        self.version = str(self.extra.get("anthropic_version", ANTHROPIC_VERSION))
        self.beta_headers: List[str] = list(self.extra.get("beta_headers", []) or [])

    # ------------------------------------------------------------------ #
    def _headers(self) -> Dict[str, str]:
        h = {k: v for k, v in self.headers.items()}
        h["x-api-key"] = self.api_key
        h["anthropic-version"] = self.version
        h.pop("Authorization", None)
        if self.beta_headers:
            h["anthropic-beta"] = ",".join(self.beta_headers)
        if self.extra.get("use_bedrock_token"):
            h.pop("x-api-key")
        return h

    def _url_messages(self) -> str:
        return self._url(self.extra.get("messages_path", "/v1/messages"))

    # ------------------------------------------------------------------ #
    # Wire conversion
    # ------------------------------------------------------------------ #
    def _split_system(self, messages: Sequence[Message]) -> Any:
        parts: List[Dict[str, Any]] = []
        for m in messages:
            if m.role != "system":
                continue
            text = content_to_text(m.content)
            if not text.strip():
                continue
            block: Dict[str, Any] = {"type": "text", "text": text}
            if m.meta.get("cache"):
                block["cache_control"] = {"type": m.meta["cache"]}
            parts.append(block)
        if not parts:
            return None
        if len(parts) == 1 and not parts[0].get("cache_control"):
            return parts[0]["text"]
        return parts

    def _wire_messages(self, messages: Sequence[Message]) -> List[Dict[str, Any]]:
        """Convert + normalise into strictly alternating user/assistant turns."""
        out: List[Dict[str, Any]] = []
        pending_tool_results: List[Dict[str, Any]] = []

        def flush_tool_results() -> None:
            if pending_tool_results:
                _append(out, "user", list(pending_tool_results))
                pending_tool_results.clear()

        for m in messages:
            if m.role == "system":
                continue
            if m.role == "tool":
                block: Dict[str, Any] = {
                    "type": "tool_result",
                    "tool_use_id": m.tool_call_id or "",
                    "content": content_to_text(m.content) or "(no output)",
                }
                if m.meta.get("is_error"):
                    block["is_error"] = True
                pending_tool_results.append(block)
                continue

            flush_tool_results()
            if m.role == "assistant":
                blocks: List[Dict[str, Any]] = []
                for th in m.meta.get("thinking_blocks") or []:
                    if isinstance(th, dict) and th.get("type") in ("thinking", "redacted_thinking"):
                        blocks.append(th)
                text = content_to_text(m.content)
                if text.strip():
                    blocks.append({"type": "text", "text": text})
                for tc in m.tool_calls:
                    blocks.append({"type": "tool_use", "id": tc.id, "name": tc.name, "input": _safe_json(tc.arguments)})
                if not blocks:
                    blocks = [{"type": "text", "text": ""}]
                _append(out, "assistant", blocks)
            else:  # user
                blocks = self._user_blocks(m.content)
                if not blocks:
                    blocks = [{"type": "text", "text": ""}]
                _append(out, "user", blocks)
        flush_tool_results()

        # Anthropic requires the first message to be from the user.
        while out and out[0]["role"] != "user":
            head = out.pop(0)
            if out:
                _prepend_into(out[0], head)
        return out

    def _user_blocks(self, content: Any) -> List[Dict[str, Any]]:
        if isinstance(content, str):
            return [{"type": "text", "text": content}] if content else []
        blocks: List[Dict[str, Any]] = []
        for b in content:
            if isinstance(b, TextBlock):
                if b.text:
                    blocks.append({"type": "text", "text": b.text})
            elif isinstance(b, ImageBlock):
                if b.data:
                    import base64

                    blocks.append({
                        "type": "image",
                        "source": {"type": "base64", "media_type": b.mime or "image/png",
                                   "data": base64.b64encode(b.data).decode("ascii")},
                    })
                elif b.url:
                    blocks.append({"type": "image", "source": {"type": "url", "url": b.url}})
        return blocks

    def _wire_tools(self, tools: Sequence[ToolSpec]) -> List[Dict[str, Any]]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.parameters or {"type": "object", "properties": {}},
            }
            for t in tools
        ]

    def _body(self, messages: Sequence[Message], tools: Sequence[ToolSpec], opts: RequestOptions, stream: bool) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "model": opts.model,
            "messages": self._wire_messages(messages),
            "max_tokens": int(opts.max_tokens or self.extra.get("default_max_tokens", DEFAULT_MAX_TOKENS)),
        }
        system = self._split_system(messages)
        if system is not None:
            body["system"] = system
        if tools:
            body["tools"] = self._wire_tools(tools)
            if opts.tool_choice == "required":
                body["tool_choice"] = {"type": "any"}
            elif opts.tool_choice == "none":
                body["tool_choice"] = {"type": "none"} if self.extra.get("support_tool_choice_none", True) else {"type": "auto"}
            else:
                body["tool_choice"] = {"type": "auto"}
        if opts.temperature is not None:
            body["temperature"] = opts.temperature
        if opts.top_p is not None:
            body["top_p"] = opts.top_p
        if opts.stop:
            body["stop_sequences"] = list(opts.stop)
        if opts.json_mode or opts.json_schema:
            body.setdefault("metadata", {})
        if stream:
            body["stream"] = True
        thinking_budget = self.extra.get("thinking_budget")
        if thinking_budget and not opts.tool_choice == "required":
            body["thinking"] = {"type": "enabled", "budget_tokens": int(thinking_budget)}
            body.pop("temperature", None)
        body.update(self.extra.get("extra_body", {}) or {})
        body.update(opts.extra_body or {})
        return body

    # ------------------------------------------------------------------ #
    def _invoke(self, messages, tools, opts) -> Completion:
        body = self._body(messages, tools, opts, stream=False)
        resp, _ = self.transport.request("POST", self._url_messages(), headers=self._headers(), json_body=body,
                                         timeout=opts.timeout or self.timeout)
        self._raise_for_status(resp)
        data = resp.json()
        return self._parse_response(data, opts)

    def _parse_response(self, data: Dict[str, Any], opts: RequestOptions) -> Completion:
        text_parts: List[str] = []
        calls: List[ToolCall] = []
        thinking_blocks: List[Dict[str, Any]] = []
        reasoning_parts: List[str] = []
        for block in data.get("content") or []:
            btype = block.get("type")
            if btype == "text":
                text_parts.append(block.get("text") or "")
            elif btype == "tool_use":
                calls.append(ToolCall(id=block.get("id") or f"toolu_{len(calls)}", name=block.get("name") or "",
                                      arguments=json.dumps(block.get("input") or {}, ensure_ascii=False),
                                      extra={"raw_input": block.get("input")}))
            elif btype == "thinking":
                thinking_blocks.append({"type": "thinking", "thinking": block.get("thinking") or "",
                                        "signature": block.get("signature") or ""})
                reasoning_parts.append(block.get("thinking") or "")
            elif btype == "redacted_thinking":
                thinking_blocks.append({"type": "redacted_thinking", "data": block.get("data") or ""})
        msg = Message(role="assistant", content="".join(text_parts), tool_calls=calls)
        if thinking_blocks:
            msg.meta["thinking_blocks"] = thinking_blocks
        usage = _usage(data.get("usage") or {})
        return Completion(message=msg, usage=usage, finish_reason=_FINISH_MAP.get(data.get("stop_reason") or "", data.get("stop_reason") or "stop"),
                          model=data.get("model") or opts.model, provider=self.key,
                          reasoning_text="\n".join(p for p in reasoning_parts if p), raw=data)

    # ------------------------------------------------------------------ #
    def _stream(self, messages, tools, opts, emit) -> Completion:
        body = self._body(messages, tools, opts, stream=True)
        resp, handle = self.transport.request("POST", self._url_messages(), headers=self._headers(), json_body=body,
                                             timeout=opts.timeout or self.timeout, stream=True)
        assert handle is not None
        agg = StreamAggregator()
        model_name = opts.model
        block_kind: Dict[int, str] = {}
        block_meta: Dict[int, Dict[str, Any]] = {}
        thinking_blocks: List[Dict[str, Any]] = []
        stop_reason = ""
        from ..transport.http import iter_sse

        try:
            for ev in iter_sse(handle):
                try:
                    data = ev.json()
                except json.JSONDecodeError:
                    self.log.warning("anthropic: skipping malformed SSE chunk", data=ev.data[:200])
                    continue
                etype = ev.event or data.get("type") or ""
                if etype == "message_start":
                    msg = data.get("message") or {}
                    model_name = msg.get("model") or model_name
                    agg.handle(UsageEvent(_usage(msg.get("usage") or {})))
                elif etype == "content_block_start":
                    idx = int(data.get("index") or 0)
                    block = data.get("content_block") or {}
                    kind = block.get("type") or "text"
                    block_kind[idx] = kind
                    block_meta[idx] = {"id": block.get("id") or "", "name": block.get("name") or "",
                                       "signature": block.get("signature") or "", "text": []}
                    if kind == "tool_use":
                        s = ToolCallStart(idx, block.get("id") or "", block.get("name") or "")
                        agg.handle(s)
                        emit(s)
                elif etype == "content_block_delta":
                    idx = int(data.get("index") or 0)
                    delta = data.get("delta") or {}
                    dtype = delta.get("type")
                    if dtype == "text_delta":
                        t = delta.get("text") or ""
                        block_meta.setdefault(idx, {"text": []})["text"].append(t)
                        agg.handle(TextDelta(t))
                        emit(TextDelta(t))
                    elif dtype == "input_json_delta":
                        p = delta.get("partial_json") or ""
                        d = ToolCallArgsDelta(idx, p)
                        agg.handle(d)
                        emit(d)
                    elif dtype == "thinking_delta":
                        t = delta.get("thinking") or ""
                        block_meta.setdefault(idx, {"text": []})["text"].append(t)
                        agg.handle(ReasoningDelta(t))
                        emit(ReasoningDelta(t))
                    elif dtype == "signature_delta":
                        block_meta.setdefault(idx, {})["signature"] = delta.get("signature") or ""
                elif etype == "content_block_stop":
                    idx = int(data.get("index") or 0)
                    kind = block_kind.get(idx)
                    meta = block_meta.get(idx) or {}
                    if kind in ("thinking", "redacted_thinking"):
                        thinking_blocks.append({"type": kind, "thinking": "".join(meta.get("text") or []),
                                                "signature": meta.get("signature") or "", "data": meta.get("data") or ""})
                elif etype == "message_delta":
                    delta = data.get("delta") or {}
                    if delta.get("stop_reason"):
                        stop_reason = delta["stop_reason"]
                    u = data.get("usage") or {}
                    if u:
                        usage_ev = UsageEvent(Usage(output_tokens=int(u.get("output_tokens") or 0), requests=0))
                        # Anthropic sends cumulative output tokens in message_delta.
                        agg.usage.output_tokens = max(agg.usage.output_tokens, usage_ev.usage.output_tokens)
                        emit(usage_ev)
                elif etype == "message_stop":
                    pass
                elif etype == "error":
                    err = data.get("error") or {}
                    raise ProviderError(f"Anthropic stream error: {err.get('message', err)}", provider=self.key)
                elif etype == "ping":
                    continue
        finally:
            handle.close()

        result = agg.build(model=model_name, provider=self.key)
        if stop_reason:
            result.finish_reason = _FINISH_MAP.get(stop_reason, stop_reason)
            if result.message.tool_calls and result.finish_reason == "stop":
                result.finish_reason = "tool_calls"
        if thinking_blocks:
            result.message.meta["thinking_blocks"] = [
                {k: v for k, v in tb.items() if v} for tb in thinking_blocks
            ]
        result.usage.input_tokens = agg.usage.input_tokens
        return result

    # ------------------------------------------------------------------ #
    def list_models(self) -> List[str]:
        resp, _ = self.transport.request("GET", self._url("/v1/models?limit=100"), headers=self._headers(), timeout=30)
        self._raise_for_status(resp)
        data = resp.json()
        return sorted({m.get("id") for m in (data.get("data") or []) if isinstance(m, dict) and m.get("id")})


# --------------------------------------------------------------------------- #
def _append(out: List[Dict[str, Any]], role: str, blocks: List[Dict[str, Any]]) -> None:
    if out and out[-1]["role"] == role:
        out[-1]["content"].extend(blocks)
    else:
        out.append({"role": role, "content": blocks})


def _prepend_into(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    target["content"] = list(source.get("content") or []) + list(target.get("content") or [])


def _safe_json(raw: str) -> Any:
    if not raw or not raw.strip():
        return {}
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        from .base import repair_json

        obj = repair_json(raw)
        if obj is None:
            return {"_raw": raw}
    return obj if isinstance(obj, dict) else {"value": obj}


def _usage(u: Dict[str, Any]) -> Usage:
    return Usage(
        input_tokens=int(u.get("input_tokens") or 0),
        output_tokens=int(u.get("output_tokens") or 0),
        cached_tokens=int((u.get("cache_read_input_tokens") or 0)),
        requests=0,
    )


__all__ = ["AnthropicProvider", "ANTHROPIC_VERSION"]
