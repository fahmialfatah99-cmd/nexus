#!/usr/bin/env python3
"""A stand-in for a running local OpenAI-compatible gateway (9Router / Ollama / vLLM).

Used for manual verification of the real CLI against the real HTTP stack::

    python3 tests/fixtures/fake_gateway.py [port]      # default 20128 (9Router's port)

    nexus models --provider 9router --refresh
    nexus -p "create a file" --provider 9router -m kr/claude-sonnet-4.5

It exposes ``/v1/models`` and ``/v1/chat/completions`` (streaming and not), and
asks for one ``write_file`` tool call with arguments fragmented across three SSE
chunks, then answers plainly once the tool result comes back -- the same shape a
real model produces.
"""

from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

MODELS = ["kr/claude-sonnet-4.5", "glm/glm-5.1", "oc/auto", "cc/gpt-5.5", "minimax/MiniMax-M2.7"]
TOOL_CALL = {"name": "write_file",
             "arguments": {"path": "created-by-gateway.txt",
                           "content": "This file was written by a tool call that came\n"
                                      "through a local OpenAI-compatible gateway.\n"}}
ARG_FRAGMENTS = ['{"path": "created-by-ga', 'teway.txt", "content": "This file was written by a tool '
                 'call that came\\nthrough a local OpenAI-compatible gateway.\\n"}']


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _json(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/").endswith("/models"):
            self._json(200, {"object": "list",
                             "data": [{"id": m, "object": "model", "owned_by": "fake-gateway"}
                                      for m in MODELS]})
        else:
            self._json(404, {"error": {"message": f"no route for {self.path}"}})

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            body = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            self._json(400, {"error": {"message": "invalid json body"}})
            return
        if not self.path.rstrip("/").endswith("/chat/completions"):
            self._json(404, {"error": {"message": f"no route for {self.path}"}})
            return
        answered = any(m.get("role") == "tool" for m in body.get("messages") or [])
        if body.get("stream"):
            self._stream(body.get("model", ""), answered)
        else:
            self._complete(body.get("model", ""), answered)

    def _complete(self, model, answered):
        if answered:
            message = {"role": "assistant",
                       "content": "Done. I created `created-by-gateway.txt` via a tool call routed "
                                  "through the local gateway, then read the result back."}
            reason = "stop"
        else:
            message = {"role": "assistant", "content": "Creating the file now.", "tool_calls": [
                {"id": "gw_1", "type": "function",
                 "function": {"name": TOOL_CALL["name"],
                              "arguments": json.dumps(TOOL_CALL["arguments"])}}]}
            reason = "tool_calls"
        self._json(200, {"id": "chatcmpl-fake", "object": "chat.completion", "model": model,
                         "choices": [{"index": 0, "message": message, "finish_reason": reason}],
                         "usage": {"prompt_tokens": 150, "completion_tokens": 40}})

    def _stream(self, model, answered):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def send(obj):
            payload = f"data: {json.dumps(obj)}\n\n".encode()
            self.wfile.write(b"%X\r\n%s\r\n" % (len(payload), payload))
            self.wfile.flush()

        def chunk(delta, finish=None, usage=None):
            obj = {"id": "chatcmpl-fake", "object": "chat.completion.chunk", "model": model,
                   "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
            if usage is not None:
                obj["usage"] = usage
            send(obj)

        if answered:
            for piece in ("Done. ", "I created `created-by-gateway.txt` ", "through the gateway."):
                chunk({"content": piece})
            chunk({}, finish="stop")
            send({"id": "chatcmpl-fake", "object": "chat.completion.chunk", "model": model,
                  "choices": [], "usage": {"prompt_tokens": 90, "completion_tokens": 12}})
        else:
            for piece in ("I will ", "create the ", "file now."):
                chunk({"content": piece})
            chunk({"tool_calls": [{"index": 0, "id": "gw_stream_1", "type": "function",
                                   "function": {"name": TOOL_CALL["name"],
                                                "arguments": ARG_FRAGMENTS[0]}}]})
            for frag in ARG_FRAGMENTS[1:]:
                chunk({"tool_calls": [{"index": 0, "function": {"arguments": frag}}]})
            chunk({}, finish="tool_calls")
            send({"id": "chatcmpl-fake", "object": "chat.completion.chunk", "model": model,
                  "choices": [], "usage": {"prompt_tokens": 210, "completion_tokens": 33}})
        done = b"data: [DONE]\n\n"
        self.wfile.write(b"%X\r\n%s\r\n" % (len(done), done))
        self.wfile.write(b"0\r\n\r\n")   # terminate the chunked body
        self.wfile.flush()


def main() -> int:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 20128
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"fake local gateway listening on http://127.0.0.1:{port}/v1", flush=True)
    print(f"models: {', '.join(MODELS)}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
