"""Google Gemini (Generative Language API v1beta) provider.

Gemini specifics handled here:
* ``systemInstruction`` is separate from ``contents``; roles are user/model only.
* Tools are declared as ``functionDeclarations``; results come back as
  ``functionResponse`` parts inside a **user** turn.
* The API issues **no tool-call ids**, so NEXUS synthesises stable ones and maps
  them back to function names when replaying history.
* Thinking models stream ``part.thought == true`` (reasoning) and return a
  ``thoughtSignature`` that must be replayed verbatim on the matching
  ``functionCall`` part -- losing it makes tool loops fail on Gemini 2.5.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

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

_FINISH_MAP = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "SAFETY": "content_filter",
    "RECITATION": "content_filter",
    "PROHIBITED_CONTENT": "content_filter",
    "OTHER": "stop",
    "BLOCKLIST": "content_filter",
    "SPII": "content_filter",
}


class GeminiProvider(BaseProvider):
    key = "gemini"
    display_name = "Google Gemini"
    default_base_url = "https://generativelanguage.googleapis.com"
    default_model = "gemini-2.5-pro"
    env_keys = ("GEMINI_API_KEY", "GOOGLE_API_KEY")

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.log = get_logger()
        self.api_version = str(self.extra.get("api_version", "v1beta"))

    # ------------------------------------------------------------------ #
    def _headers(self) -> Dict[str, str]:
        h = {k: v for k, v in self.headers.items()}
        h["x-goog-api-key"] = self.api_key
        h.pop("Authorization", None)
        return h

    def _endpoint(self, model: str, stream: bool) -> str:
        action = "streamGenerateContent?alt=sse" if stream else "generateContent"
        return self._url(f"/{self.api_version}/models/{model}:{action}")

    # ------------------------------------------------------------------ #
    # Wire conversion
    # ------------------------------------------------------------------ #
    def _system_instruction(self, messages: Sequence[Message]) -> Optional[Dict[str, Any]]:
        parts = [content_to_text(m.content) for m in messages if m.role == "system"]
        parts = [p for p in parts if p.strip()]
        if not parts:
            return None
        return {"parts": [{"text": "\n\n".join(parts)}]}

    def _contents(self, messages: Sequence[Message]) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
        """Return (contents, id->function name map for tool results)."""
        contents: List[Dict[str, Any]] = []
        id_to_name: Dict[str, str] = {}
        pending_responses: List[Dict[str, Any]] = []

        def flush() -> None:
            if pending_responses:
                _append_turn(contents, "user", list(pending_responses))
                pending_responses.clear()

        for m in messages:
            if m.role == "system":
                continue
            if m.role == "assistant":
                flush()
                parts: List[Dict[str, Any]] = []
                text = content_to_text(m.content)
                if text:
                    parts.append({"text": text})
                for i, tc in enumerate(m.tool_calls):
                    id_to_name[tc.id] = tc.name
                    part: Dict[str, Any] = {"functionCall": {"name": tc.name, "arguments": _as_obj(tc.arguments)}}
                    sig = tc.extra.get("thoughtSignature")
                    if sig:
                        part["thoughtSignature"] = sig
                    parts.append(part)
                if not parts:
                    parts = [{"text": ""}]
                _append_turn(contents, "model", parts)
            elif m.role == "tool":
                name = m.name or id_to_name.get(m.tool_call_id or "", "")
                payload = {"content": content_to_text(m.content)}
                if m.meta.get("is_error"):
                    payload["is_error"] = True
                if m.tool_call_id:
                    payload["call_id"] = m.tool_call_id
                pending_responses.append({"functionResponse": {"name": name or "tool", "response": payload}})
            else:
                flush()
                _append_turn(contents, "user", self._user_parts(m.content))
        flush()
        if not contents:
            contents.append({"role": "user", "parts": [{"text": ""}]})
        # Gemini rejects a leading model turn.
        while len(contents) > 1 and contents[0]["role"] != "user":
            contents.pop(0)
        return contents, id_to_name

    def _user_parts(self, content: Any) -> List[Dict[str, Any]]:
        if isinstance(content, str):
            return [{"text": content}] if content else [{"text": ""}]
        parts: List[Dict[str, Any]] = []
        for b in content:
            if isinstance(b, TextBlock):
                if b.text:
                    parts.append({"text": b.text})
            elif isinstance(b, ImageBlock):
                if b.data:
                    import base64

                    parts.append({"inlineData": {"mimeType": b.mime or "image/png",
                                                "data": base64.b64encode(b.data).decode("ascii")}})
                elif b.url:
                    parts.append({"fileData": {"fileUri": b.url, "mimeType": b.mime or "image/png"}})
        return parts or [{"text": ""}]

    def _tools(self, tools: Sequence[ToolSpec]) -> Optional[List[Dict[str, Any]]]:
        if not tools:
            return None
        decls = []
        for t in tools:
            decls.append({
                "name": t.name,
                "description": t.description,
                "parameters": _clean_schema(t.parameters or {"type": "object", "properties": {}}),
            })
        return [{"functionDeclarations": decls}]

    def _body(self, messages: Sequence[Message], tools: Sequence[ToolSpec], opts: RequestOptions) -> Dict[str, Any]:
        contents, _ = self._contents(messages)
        body: Dict[str, Any] = {"contents": contents}
        sysinstr = self._system_instruction(messages)
        if sysinstr:
            body["systemInstruction"] = sysinstr
        decls = self._tools(tools)
        if decls:
            body["tools"] = decls
            mode = {"auto": "AUTO", "required": "ANY", "none": "NONE"}.get(opts.tool_choice or "auto", "AUTO")
            body["toolConfig"] = {"functionCallingConfig": {"mode": mode}}
        cfg: Dict[str, Any] = {}
        if opts.temperature is not None:
            cfg["temperature"] = opts.temperature
        if opts.top_p is not None:
            cfg["topP"] = opts.top_p
        if opts.max_tokens:
            cfg["maxOutputTokens"] = opts.max_tokens
        if opts.stop:
            cfg["stopSequences"] = list(opts.stop)[:5]
        if opts.json_mode or opts.json_schema:
            cfg["responseMimeType"] = "application/json"
            if opts.json_schema:
                cfg["responseSchema"] = _clean_schema(opts.json_schema.get("schema", opts.json_schema))
        thinking = self.extra.get("thinking_budget")
        if thinking is not None:
            cfg["thinkingConfig"] = {"thinkingBudget": int(thinking), "includeThoughts": bool(self.extra.get("include_thoughts", True))}
        if cfg:
            body["generationConfig"] = cfg
        body.update(self.extra.get("extra_body", {}) or {})
        body.update(opts.extra_body or {})
        return body

    # ------------------------------------------------------------------ #
    def _invoke(self, messages, tools, opts) -> Completion:
        body = self._body(messages, tools, opts)
        resp, _ = self.transport.request("POST", self._endpoint(opts.model, False), headers=self._headers(),
                                         json_body=body, timeout=opts.timeout or self.timeout)
        self._raise_for_status(resp)
        data = resp.json()
        agg = StreamAggregator()
        self._absorb(data, agg, emit=None)
        return agg.build(model=data.get("modelVersion") or opts.model, provider=self.key, raw=data)

    def _stream(self, messages, tools, opts, emit) -> Completion:
        body = self._body(messages, tools, opts)
        resp, handle = self.transport.request("POST", self._endpoint(opts.model, True), headers=self._headers(),
                                              json_body=body, timeout=opts.timeout or self.timeout, stream=True)
        assert handle is not None
        agg = StreamAggregator()
        from ..transport.http import iter_sse

        try:
            for ev in iter_sse(handle):
                try:
                    data = ev.json()
                except json.JSONDecodeError:
                    self.log.warning("gemini: skipping malformed SSE chunk", data=ev.data[:200])
                    continue
                if isinstance(data, dict) and data.get("error"):
                    err = data["error"]
                    raise ProviderError(f"Gemini stream error: {err.get('message', err)}", provider=self.key)
                self._absorb(data, agg, emit=emit)
        finally:
            handle.close()
        return agg.build(model=opts.model, provider=self.key)

    def _absorb(self, data: Dict[str, Any], agg: StreamAggregator, emit: Optional[Any]) -> None:
        for cand in data.get("candidates") or []:
            content = cand.get("content") or {}
            for part in content.get("parts") or []:
                if "functionCall" in part:
                    fc = part["functionCall"] or {}
                    idx = len(agg.order)
                    call_id = f"gemini_{idx}_{fc.get('name', 'fn')}"
                    start = ToolCallStart(idx, call_id, fc.get("name") or "")
                    agg.handle(start)
                    if emit:
                        emit(start)
                    args = json.dumps(fc.get("args") or {}, ensure_ascii=False)
                    d = ToolCallArgsDelta(idx, args)
                    agg.handle(d)
                    if emit:
                        emit(d)
                    sig = part.get("thoughtSignature")
                    if sig and idx in agg.calls:
                        agg.calls[idx].extra["thoughtSignature"] = sig
                    continue
                text = part.get("text")
                if not text:
                    continue
                if part.get("thought"):
                    agg.handle(ReasoningDelta(text))
                    if emit:
                        emit(ReasoningDelta(text))
                else:
                    agg.handle(TextDelta(text))
                    if emit:
                        emit(TextDelta(text))
            if cand.get("finishReason"):
                reason = _FINISH_MAP.get(cand["finishReason"], cand["finishReason"].lower())
                f = FinishEvent(reason)
                agg.handle(f)
                if emit:
                    emit(f)
            if cand.get("promptFeedback", {}).get("blockReason"):
                f = FinishEvent("content_filter")
                agg.handle(f)
                if emit:
                    emit(f)
        um = data.get("usageMetadata") or {}
        if um:
            u = Usage(
                input_tokens=int(um.get("promptTokenCount") or 0),
                output_tokens=int((um.get("candidatesTokenCount") or 0) + (um.get("thoughtsTokenCount") or 0)),
                cached_tokens=int(um.get("cachedContentTokenCount") or 0),
                reasoning_tokens=int(um.get("thoughtsTokenCount") or 0),
                requests=0,
            )
            agg.handle(UsageEvent(u))
            if emit:
                emit(UsageEvent(u))

    # ------------------------------------------------------------------ #
    def list_models(self) -> List[str]:
        resp, _ = self.transport.request(
            "GET", self._url(f"/{self.api_version}/models?pageSize=200"), headers=self._headers(), timeout=30
        )
        self._raise_for_status(resp)
        data = resp.json()
        out = []
        for m in data.get("models") or []:
            name = str(m.get("name") or "")
            if name.startswith("models/"):
                name = name.split("/", 1)[1]
            actions = m.get("supportedGenerationMethods") or []
            if name and (not actions or "generateContent" in actions):
                out.append(name)
        return sorted(set(out))


# --------------------------------------------------------------------------- #
def _append_turn(contents: List[Dict[str, Any]], role: str, parts: List[Dict[str, Any]]) -> None:
    if contents and contents[-1]["role"] == role:
        contents[-1]["parts"].extend(parts)
    else:
        contents.append({"role": role, "parts": parts})


def _as_obj(raw: str) -> Dict[str, Any]:
    if not raw or not raw.strip():
        return {}
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        from .base import repair_json

        obj = repair_json(raw)
        if obj is None:
            return {"_raw": raw}
    if isinstance(obj, dict):
        return obj
    return {"value": obj}


_UNSUPPORTED_SCHEMA_KEYS = {"$schema", "$id", "$ref", "additionalProperties", "$defs", "definitions", "$comment"}


def _clean_schema(schema: Any, _depth: int = 0) -> Any:
    """Recursively strip JSON-Schema keywords Gemini rejects."""
    if _depth > 25 or not isinstance(schema, dict):
        return schema if not isinstance(schema, dict) else {}
    out: Dict[str, Any] = {}
    for k, v in schema.items():
        if k in _UNSUPPORTED_SCHEMA_KEYS:
            continue
        if k == "properties" and isinstance(v, dict):
            props = {}
            for pk, pv in v.items():
                props[pk] = _clean_schema(pv, _depth + 1)
            out["properties"] = props
        elif k == "items":
            out["items"] = _clean_schema(v, _depth + 1)
        elif k in ("anyOf", "oneOf", "allOf") and isinstance(v, list):
            cleaned = [c for c in (_clean_schema(i, _depth + 1) for i in v) if isinstance(c, dict) and c.get("type") != "null"]
            if cleaned:
                out[k] = cleaned
        elif k == "required" and isinstance(v, list):
            out["required"] = [str(x) for x in v]
        elif k == "enum" and isinstance(v, list):
            out["enum"] = list(v)
        elif k in ("type", "description", "format", "default", "nullable", "minimum", "maximum",
                   "minItems", "maxItems", "pattern", "title", "propertyOrdering"):
            out[k] = v
        elif k == "type" and isinstance(v, list):
            out["type"] = v[0]
    t = out.get("type")
    if isinstance(t, list):
        out["type"] = t[0] if t else "string"
    if out.get("type") == "object" and "properties" not in out:
        out["properties"] = {}
    return out


__all__ = ["GeminiProvider", "content_to_text"]
