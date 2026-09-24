"""Provider adapter contract tests (offline, deterministic).

Every assertion here corresponds to a real wire-format requirement of the
upstream API. If a provider adapter regresses, this file fails.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nexuscli.core.errors import AuthError, ModelNotFoundError, RateLimitError, ValidationError  # noqa: E402
from nexuscli.providers.base import (  # noqa: E402
    ImageBlock,
    Message,
    RequestOptions,
    TextBlock,
    ToolCall,
    ToolSpec,
    repair_json,
)
from nexuscli.providers.registry import create_provider, model_info, resolve_model_id  # noqa: E402
from nexuscli.transport.http import MockTransport  # noqa: E402

READ_TOOL = ToolSpec(
    name="read_file",
    description="Read a file",
    parameters={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
)
MSGS = [Message.system("You are helpful."), Message.user("read x.txt")]


def opts(**kw):
    kw.setdefault("model", "gpt-4o")
    return RequestOptions(**kw)


class TestOpenAICompat(unittest.TestCase):
    def _provider(self, queue_item):
        t = MockTransport()
        t.queue(queue_item)
        return create_provider("openai", api_key="k", transport=t), t

    def test_non_stream_tool_call(self):
        p, t = self._provider((200, {
            "id": "c1", "model": "gpt-4o",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi", "tool_calls": [
                {"id": "call_1", "type": "function",
                 "function": {"name": "read_file", "arguments": '{"path":"x.txt"}'}}]},
                "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7,
                      "prompt_tokens_details": {"cached_tokens": 3},
                      "completion_tokens_details": {"reasoning_tokens": 2}},
        }))
        c = p.complete(MSGS, [READ_TOOL], opts(max_tokens=100, stream=False))
        self.assertEqual(c.message.text, "Hi")
        self.assertEqual(c.finish_reason, "tool_calls")
        self.assertEqual(c.message.tool_calls[0].parsed_args(), {"path": "x.txt"})
        self.assertEqual((c.usage.input_tokens, c.usage.cached_tokens, c.usage.reasoning_tokens), (11, 3, 2))
        body = t.calls[0]["json"]
        self.assertEqual(body["messages"][0], {"role": "system", "content": "You are helpful."})
        self.assertEqual(body["tools"][0]["function"]["name"], "read_file")
        self.assertEqual(body["max_tokens"], 100)
        self.assertEqual(body["tool_choice"], "auto")
        self.assertNotIn("temperature", body)  # opts() leaves temperature unset

    def test_stream_fragmented_tool_args(self):
        chunks = [
            {"choices": [{"index": 0, "delta": {"role": "assistant", "content": "Let me "}}]},
            {"choices": [{"index": 0, "delta": {"content": "check."}}]},
            {"choices": [{"index": 0, "delta": {"tool_calls": [
                {"index": 0, "id": "call_a", "type": "function",
                 "function": {"name": "read_file", "arguments": '{"pa'}}]}}]},
            {"choices": [{"index": 0, "delta": {"tool_calls": [
                {"index": 0, "function": {"arguments": 'th": "x.t'}}]}}]},
            {"choices": [{"index": 0, "delta": {"tool_calls": [
                {"index": 0, "function": {"arguments": 'xt"}'}}]}, "finish_reason": "tool_calls"}]},
            {"choices": [], "usage": {"prompt_tokens": 20, "completion_tokens": 9}},
        ]
        p, _ = self._provider({"sse": chunks + ["[DONE]"]})
        events = []
        c = p.complete(MSGS, [READ_TOOL], opts(max_tokens=100), on_event=events.append)
        self.assertEqual(c.message.text, "Let me check.")
        self.assertEqual(len(c.message.tool_calls), 1)
        self.assertEqual(c.message.tool_calls[0].id, "call_a")
        self.assertEqual(c.message.tool_calls[0].parsed_args(), {"path": "x.txt"})
        self.assertEqual((c.usage.input_tokens, c.usage.output_tokens), (20, 9))
        self.assertEqual(c.finish_reason, "tool_calls")
        self.assertTrue(events)

    def test_stream_parallel_tool_calls(self):
        chunks = [
            {"choices": [{"index": 0, "delta": {"tool_calls": [
                {"index": 0, "id": "a", "type": "function", "function": {"name": "f1", "arguments": "{}"}}]}}]},
            {"choices": [{"index": 0, "delta": {"tool_calls": [
                {"index": 1, "id": "b", "type": "function", "function": {"name": "f2", "arguments": '{"x"'}}]}}]},
            {"choices": [{"index": 0, "delta": {"tool_calls": [
                {"index": 1, "function": {"arguments": ":1}"}}]}, "finish_reason": "tool_calls"}]},
        ]
        p, _ = self._provider({"sse": chunks + ["[DONE]"]})
        c = p.complete(MSGS, [], opts(max_tokens=50))
        self.assertEqual([(x.id, x.name, x.parsed_args()) for x in c.message.tool_calls],
                         [("a", "f1", {}), ("b", "f2", {"x": 1})])

    def test_reasoning_content_delta(self):
        p, _ = self._provider({"sse": [
            {"choices": [{"index": 0, "delta": {"reasoning_content": "thinking..."}}]},
            {"choices": [{"index": 0, "delta": {"content": "answer"}, "finish_reason": "stop"}]},
            "[DONE]"]})
        c = p.complete(MSGS, [], opts(max_tokens=50))
        self.assertEqual(c.reasoning_text, "thinking...")
        self.assertEqual(c.message.text, "answer")

    def test_reasoning_model_param_handling(self):
        p, t = self._provider((200, {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}], "usage": {}}))
        p.complete(MSGS, [], opts(model="o3", temperature=0.5, max_tokens=99, stream=False))
        body = t.calls[0]["json"]
        self.assertNotIn("temperature", body)
        self.assertEqual(body["max_completion_tokens"], 99)

    def test_image_content_parts(self):
        p, t = self._provider((200, {"choices": [{"message": {"content": "pic"}, "finish_reason": "stop"}], "usage": {}}))
        p.complete([Message.user([TextBlock("what?"), ImageBlock(data=b"\x89PNG\r\n", mime="image/png")])], [],
                   opts(stream=False))
        content = t.calls[0]["json"]["messages"][0]["content"]
        self.assertEqual(content[0], {"type": "text", "text": "what?"})
        self.assertTrue(content[1]["image_url"]["url"].startswith("data:image/png;base64,"))

    def test_tool_history_roundtrip(self):
        p, t = self._provider((200, {"choices": [{"message": {"content": "done"}, "finish_reason": "stop"}], "usage": {}}))
        hist = [Message.user("go"),
                Message(role="assistant", content="", tool_calls=[ToolCall("c1", "read_file", '{"path":"a"}')]),
                Message.tool_result("c1", "read_file", "CONTENTS")]
        p.complete(hist, [READ_TOOL], opts(stream=False))
        wire = t.calls[0]["json"]["messages"]
        self.assertEqual(wire[1]["role"], "assistant")
        self.assertEqual(wire[1]["content"], "")  # empty string, not null (widest compat)
        self.assertEqual(wire[1]["tool_calls"][0]["function"]["arguments"], '{"path":"a"}')
        self.assertEqual(wire[2], {"role": "tool", "tool_call_id": "c1", "content": "CONTENTS", "name": "read_file"})

    def test_json_mode_and_schema(self):
        p, t = self._provider((200, {"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}], "usage": {}}))
        p.complete(MSGS, [], opts(json_mode=True, stream=False))
        self.assertEqual(t.calls[0]["json"]["response_format"], {"type": "json_object"})
        t.queue((200, {"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}], "usage": {}}))
        p.complete(MSGS, [], opts(json_schema={"name": "out", "schema": {"type": "object"}}, stream=False))
        self.assertEqual(t.calls[1]["json"]["response_format"]["type"], "json_schema")

    def test_list_models(self):
        p, t = self._provider((200, {"data": [{"id": "gpt-4o"}, {"id": "gpt-4o-mini"}]}))
        self.assertEqual(p.list_models(), ["gpt-4o", "gpt-4o-mini"])

    def test_error_mapping(self):
        for status, exc in ((401, AuthError), (429, RateLimitError), (404, ModelNotFoundError)):
            with self.subTest(status=status):
                p, _ = self._provider((status, {"error": {"message": "nope"}}))
                with self.assertRaises(exc):
                    p.complete(MSGS, [], opts(stream=False))


class TestAnthropic(unittest.TestCase):
    def _provider(self, queue_item):
        t = MockTransport()
        t.queue(queue_item)
        return create_provider("anthropic", api_key="k", transport=t), t

    def test_non_stream(self):
        p, t = self._provider((200, {
            "id": "m1", "model": "claude-sonnet-4-5", "stop_reason": "tool_use",
            "content": [{"type": "thinking", "thinking": "hmm", "signature": "SIG"},
                        {"type": "text", "text": "Sure"},
                        {"type": "tool_use", "id": "toolu_1", "name": "read_file", "input": {"path": "x.txt"}}],
            "usage": {"input_tokens": 30, "output_tokens": 12, "cache_read_input_tokens": 5}}))
        c = p.complete(MSGS, [READ_TOOL], RequestOptions(model="claude-sonnet-4-5", max_tokens=1024, stream=False))
        body = t.calls[0]["json"]
        self.assertEqual(body["system"], "You are helpful.")
        self.assertEqual(body["max_tokens"], 1024)
        self.assertEqual(body["tools"][0]["input_schema"]["required"], ["path"])
        self.assertEqual(body["messages"], [{"role": "user", "content": [{"type": "text", "text": "read x.txt"}]}])
        self.assertEqual(t.calls[0]["headers"]["anthropic-version"], "2023-06-01")
        self.assertEqual(t.calls[0]["headers"]["x-api-key"], "k")
        self.assertNotIn("Authorization", t.calls[0]["headers"])
        self.assertEqual(c.finish_reason, "tool_calls")
        self.assertEqual(c.message.tool_calls[0].id, "toolu_1")
        self.assertEqual(c.message.tool_calls[0].parsed_args(), {"path": "x.txt"})
        self.assertEqual((c.usage.input_tokens, c.usage.cached_tokens), (30, 5))
        self.assertEqual(c.message.meta["thinking_blocks"][0]["signature"], "SIG")

    def test_max_tokens_defaults_when_unset(self):
        p, t = self._provider((200, {"content": [{"type": "text", "text": "x"}], "stop_reason": "end_turn", "usage": {}}))
        p.complete(MSGS, [], RequestOptions(model="claude-sonnet-4-5", stream=False))
        self.assertGreater(t.calls[0]["json"]["max_tokens"], 0)

    def test_stream_tool_use(self):
        sse = [
            {"event": "message_start", "data": {"type": "message_start", "message": {
                "id": "m", "model": "claude-sonnet-4-5", "usage": {"input_tokens": 42, "output_tokens": 1}}}},
            {"event": "content_block_start", "data": {"type": "content_block_start", "index": 0, "content_block": {
                "type": "tool_use", "id": "toolu_9", "name": "read_file", "input": {}}}},
            {"event": "content_block_delta", "data": {"type": "content_block_delta", "index": 0, "delta": {
                "type": "input_json_delta", "partial_json": '{"path"'}}},
            {"event": "content_block_delta", "data": {"type": "content_block_delta", "index": 0, "delta": {
                "type": "input_json_delta", "partial_json": ':"y.txt"}'}}},
            {"event": "content_block_stop", "data": {"type": "content_block_stop", "index": 0}},
            {"event": "message_delta", "data": {"type": "message_delta", "delta": {"stop_reason": "tool_use"},
                                                "usage": {"output_tokens": 18}}},
            {"event": "message_stop", "data": {"type": "message_stop"}},
        ]
        p, _ = self._provider({"sse": sse})
        c = p.complete(MSGS, [READ_TOOL], RequestOptions(model="claude-sonnet-4-5", max_tokens=1024))
        self.assertEqual((c.usage.input_tokens, c.usage.output_tokens), (42, 18))
        tc = c.message.tool_calls[0]
        self.assertEqual((tc.id, tc.name, tc.parsed_args()), ("toolu_9", "read_file", {"path": "y.txt"}))
        self.assertEqual(c.finish_reason, "tool_calls")

    def test_alternation_and_tool_results(self):
        hist = [Message.system("s"), Message.user("a"), Message.user("b"),
                Message(role="assistant", content="", tool_calls=[
                    ToolCall("toolu_1", "read_file", '{"path":"x"}'), ToolCall("toolu_2", "ls", "{}")]),
                Message.tool_result("toolu_1", "read_file", "FILE CONTENTS"),
                Message.tool_result("toolu_2", "ls", "DIR", is_error=True)]
        p, t = self._provider((200, {"content": [{"type": "text", "text": "done"}],
                                     "stop_reason": "end_turn", "usage": {"input_tokens": 5, "output_tokens": 2}}))
        c = p.complete(hist, [], RequestOptions(model="claude-sonnet-4-5", max_tokens=64, stream=False))
        wire = t.calls[0]["json"]["messages"]
        self.assertEqual([m["role"] for m in wire], ["user", "assistant", "user"])
        self.assertEqual(wire[0]["content"], [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}])
        self.assertEqual(wire[1]["content"][0]["type"], "tool_use")
        self.assertEqual(wire[1]["content"][1]["id"], "toolu_2")
        self.assertEqual(wire[2]["content"][0]["tool_use_id"], "toolu_1")
        self.assertTrue(wire[2]["content"][1]["is_error"])
        self.assertEqual(c.finish_reason, "stop")

    def test_thinking_blocks_replayed(self):
        hist = [Message.user("q"),
                Message(role="assistant", content="a", tool_calls=[ToolCall("t1", "f", "{}")],
                        meta={"thinking_blocks": [{"type": "thinking", "thinking": "deep", "signature": "S1"}]}),
                Message.tool_result("t1", "f", "res")]
        p, t = self._provider((200, {"content": [{"type": "text", "text": "k"}], "stop_reason": "end_turn", "usage": {}}))
        p.complete(hist, [], RequestOptions(model="claude-sonnet-4-5", max_tokens=64, stream=False))
        blocks = t.calls[0]["json"]["messages"][1]["content"]
        self.assertEqual(blocks[0], {"type": "thinking", "thinking": "deep", "signature": "S1"})

    def test_leading_assistant_turn_is_repaired(self):
        hist = [Message(role="assistant", content="prior"), Message.user("now")]
        p, t = self._provider((200, {"content": [{"type": "text", "text": "k"}], "stop_reason": "end_turn", "usage": {}}))
        p.complete(hist, [], RequestOptions(model="claude-sonnet-4-5", max_tokens=64, stream=False))
        wire = t.calls[0]["json"]["messages"]
        self.assertEqual(wire[0]["role"], "user")

    def test_list_models(self):
        p, _ = self._provider((200, {"data": [{"id": "claude-sonnet-4-5"}, {"id": "claude-opus-4-1"}]}))
        self.assertEqual(p.list_models(), ["claude-opus-4-1", "claude-sonnet-4-5"])


class TestGemini(unittest.TestCase):
    def _provider(self, queue_item):
        t = MockTransport()
        t.queue(queue_item)
        return create_provider("gemini", api_key="k", transport=t), t

    def test_non_stream(self):
        p, t = self._provider((200, {
            "modelVersion": "gemini-2.5-pro",
            "candidates": [{"content": {"role": "model", "parts": [
                {"text": "Let me"}, {"thought": True, "text": "hmm"},
                {"functionCall": {"name": "read_file", "args": {"path": "x.txt"}}, "thoughtSignature": "TSIG"}]},
                "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 15, "candidatesTokenCount": 8,
                              "thoughtsTokenCount": 3, "cachedContentTokenCount": 2}}))
        c = p.complete(MSGS, [READ_TOOL], RequestOptions(model="gemini-2.5-pro", stream=False, max_tokens=200))
        body = t.calls[0]["json"]
        self.assertEqual(body["systemInstruction"], {"parts": [{"text": "You are helpful."}]})
        self.assertEqual(body["tools"][0]["functionDeclarations"][0]["name"], "read_file")
        self.assertEqual(body["generationConfig"]["maxOutputTokens"], 200)
        self.assertEqual(t.calls[0]["headers"]["x-goog-api-key"], "k")
        self.assertEqual(c.message.text, "Let me")
        self.assertEqual(c.reasoning_text, "hmm")
        self.assertEqual(c.message.tool_calls[0].parsed_args(), {"path": "x.txt"})
        self.assertEqual(c.message.tool_calls[0].extra["thoughtSignature"], "TSIG")
        self.assertEqual((c.usage.input_tokens, c.usage.output_tokens, c.usage.reasoning_tokens), (15, 11, 3))
        # A functionCall in the response means the model wants tools, even when
        # Gemini reports finishReason=STOP -- normalised for the agent loop.
        self.assertEqual(c.finish_reason, "tool_calls")

    def test_stream(self):
        sse = [
            {"candidates": [{"content": {"role": "model", "parts": [{"text": "Hel"}]}}]},
            {"candidates": [{"content": {"role": "model", "parts": [{"text": "lo"}]}}]},
            {"candidates": [{"content": {"role": "model", "parts": [{"functionCall": {"name": "ls", "args": {}}}]}}]},
            {"candidates": [{"content": {"role": "model", "parts": []}, "finishReason": "STOP"}],
             "usageMetadata": {"promptTokenCount": 9, "candidatesTokenCount": 4}},
        ]
        p, _ = self._provider({"sse": sse})
        c = p.complete(MSGS, [], RequestOptions(model="gemini-2.5-pro"))
        self.assertEqual(c.message.text, "Hello")
        self.assertEqual(len(c.message.tool_calls), 1)
        self.assertEqual(c.message.tool_calls[0].name, "ls")
        self.assertEqual(c.usage.input_tokens, 9)

    def test_function_response_roundtrip(self):
        hist = [Message.user("go"),
                Message(role="assistant", content="ok", tool_calls=[
                    ToolCall("gemini_0_read_file", "read_file", '{"path":"x.txt"}',
                             extra={"thoughtSignature": "TSIG"})]),
                Message.tool_result("gemini_0_read_file", "read_file", "CONTENTS")]
        p, t = self._provider((200, {"candidates": [{"content": {"parts": [{"text": "done"}]},
                                                     "finishReason": "STOP"}], "usageMetadata": {}}))
        p.complete(hist, [], RequestOptions(model="gemini-2.5-pro", stream=False))
        contents = t.calls[0]["json"]["contents"]
        self.assertEqual(contents[1]["role"], "model")
        self.assertEqual(contents[1]["parts"][1]["functionCall"]["name"], "read_file")
        self.assertEqual(contents[1]["parts"][1]["thoughtSignature"], "TSIG")
        self.assertEqual(contents[2]["role"], "user")
        fr = contents[2]["parts"][0]["functionResponse"]
        self.assertEqual(fr["name"], "read_file")
        self.assertEqual(fr["response"]["content"], "CONTENTS")

    def test_schema_sanitising(self):
        bad = ToolSpec(name="t", description="d", parameters={
            "type": "object", "additionalProperties": False,
            "$schema": "http://json-schema.org/draft-07/schema#",
            "properties": {"a": {"type": ["string", "null"], "description": "x"},
                           "b": {"type": "array", "items": {"type": "integer"}}},
            "required": ["a"]})
        p, t = self._provider((200, {"candidates": [{"content": {"parts": [{"text": "ok"}]},
                                                     "finishReason": "STOP"}], "usageMetadata": {}}))
        p.complete(MSGS, [bad], RequestOptions(model="gemini-2.5-pro", stream=False))
        decl = t.calls[0]["json"]["tools"][0]["functionDeclarations"][0]["parameters"]
        self.assertNotIn("$schema", decl)
        self.assertNotIn("additionalProperties", decl)
        self.assertEqual(decl["properties"]["a"]["type"], "string")
        self.assertEqual(decl["properties"]["b"]["items"], {"type": "integer"})

    def test_json_mode_sets_response_mime(self):
        p, t = self._provider((200, {"candidates": [{"content": {"parts": [{"text": "{}"}]},
                                                     "finishReason": "STOP"}], "usageMetadata": {}}))
        p.complete(MSGS, [], RequestOptions(model="gemini-2.5-pro", json_mode=True, stream=False))
        self.assertEqual(t.calls[0]["json"]["generationConfig"]["responseMimeType"], "application/json")

    def test_stream_error_event(self):
        p, _ = self._provider({"sse": [{"error": {"code": 400, "message": "bad request"}}]})
        with self.assertRaises(Exception):
            p.complete(MSGS, [], RequestOptions(model="gemini-2.5-pro"))


class TestJsonRepair(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(repair_json('{"a": 1}'), {"a": 1})

    def test_fenced(self):
        self.assertEqual(repair_json('```json\n{"a": 1}\n```'), {"a": 1})
        self.assertEqual(repair_json('```\n{"a": 1}\n```'), {"a": 1})

    def test_truncated_object(self):
        self.assertEqual(repair_json('{"path": "a.txt", "lim'), {"path": "a.txt"})
        self.assertEqual(repair_json('{"a": {"b": 1}, "c":'), {"a": {"b": 1}})
        self.assertEqual(repair_json('{"items": [1, 2, 3'), {"items": [1, 2, 3]})

    def test_truncated_string(self):
        self.assertEqual(repair_json('{"path": "a.txt'), {"path": "a.txt"})

    def test_trailing_comma(self):
        self.assertEqual(repair_json('{"a": 1,}'), {"a": 1})
        self.assertEqual(repair_json('{"a": [1, 2,],}'), {"a": [1, 2]})

    def test_prefix_junk(self):
        self.assertEqual(repair_json('Sure! {"a": 1}'), {"a": 1})

    def test_list_wrapped(self):
        tc = ToolCall("i", "n", '[{"a": 1}]')
        self.assertEqual(tc.parsed_args(), {"a": 1})

    def test_garbage_raises(self):
        with self.assertRaises(ValidationError):
            ToolCall("i", "n", "not json at all").parsed_args()


class TestRegistry(unittest.TestCase):
    def test_aliases(self):
        self.assertEqual(resolve_model_id("sonnet"), "claude-sonnet-4-5")
        self.assertEqual(resolve_model_id("unknown-model"), "unknown-model")

    def test_unknown_model_has_zero_context(self):
        info = model_info("totally-new-model-2030", "openai")
        self.assertEqual(info.context_window, 0)  # unknown must never be guessed

    def test_dated_model_matches_catalog_prefix(self):
        info = model_info("claude-sonnet-4-5-20250929", "anthropic")
        self.assertEqual(info.context_window, 200_000)
        self.assertEqual(info.id, "claude-sonnet-4-5-20250929")

    def test_mock_provider_needs_no_key(self):
        p = create_provider("mock")
        c = p.complete([Message.user("hi")], [], RequestOptions(model="mock-1", stream=False))
        self.assertIn("hi", c.message.text)

    def test_missing_key_raises_with_hint(self):
        import os

        saved = {k: os.environ.pop(k, None) for k in ("OPENAI_API_KEY", "NEXUS_OPENAI_API_KEY")}
        try:
            with self.assertRaises(Exception) as ctx:
                create_provider("openai")
            self.assertIn("API key", str(ctx.exception))
        finally:
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v


if __name__ == "__main__":
    unittest.main(verbosity=2)
