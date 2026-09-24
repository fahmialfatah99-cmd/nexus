"""Real-socket tests for HTTPTransport.

Everything else in the suite uses ``MockTransport``. These tests run an actual
``http.server`` on a loopback port and drive the production transport through it,
so gzip decoding, chunked streaming, SSE framing across packet boundaries,
``Retry-After`` handling, status mapping and timeouts are all verified for real.
"""

from __future__ import annotations

import gzip
import json
import socket
import sys
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nexuscli.core.errors import AuthError, ModelNotFoundError, ProviderError, RateLimitError, TimeoutError_  # noqa: E402
from nexuscli.transport.http import HTTPTransport, RetryPolicy, SSEParser, iter_sse  # noqa: E402

STATE = {"flaky_hits": 0, "server_error_hits": 0}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # silence the server
        pass

    # -- helpers ----------------------------------------------------------
    def _send(self, code: int, body: bytes, ctype: str = "application/json", extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _send_gzip(self, code: int, payload: dict):
        raw = json.dumps(payload).encode()
        body = gzip.compress(raw)
        self._send(code, body, extra={"Content-Encoding": "gzip"})

    def _sse(self, events, chunk_size: int = 7, delay: float = 0.0):
        parts = []
        for e in events:
            if isinstance(e, dict) and "data" in e:
                prefix = f"event: {e['event']}\n" if "event" in e else ""
                parts.append(f"{prefix}data: {json.dumps(e['data'])}\n\n")
            else:
                parts.append(f"data: {e}\n\n")
        raw = "".join(parts).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        for i in range(0, len(raw), chunk_size):
            piece = raw[i : i + chunk_size]
            self.wfile.write(b"%X\r\n%s\r\n" % (len(piece), piece))
            self.wfile.flush()
            if delay:
                time.sleep(delay)
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    # -- routes -----------------------------------------------------------
    def do_GET(self):
        if self.path == "/json":
            self._send(200, json.dumps({"ok": True, "value": 42}).encode())
        elif self.path == "/gzip":
            self._send_gzip(200, {"compressed": True, "text": "héllo → wörld"})
        elif self.path == "/models":
            self._send(200, json.dumps({"data": [{"id": "m1"}, {"id": "m2"}]}).encode())
        elif self.path == "/unauthorized":
            self._send(401, json.dumps({"error": {"message": "bad key"}}).encode())
        elif self.path == "/missing":
            self._send(404, json.dumps({"error": {"message": "no such model"}}).encode())
        elif self.path == "/ratelimit":
            self._send(429, json.dumps({"error": {"message": "slow down"}}).encode(),
                       extra={"Retry-After": "0"})
        elif self.path == "/flaky":
            STATE["flaky_hits"] += 1
            if STATE["flaky_hits"] < 3:
                self._send(503, json.dumps({"error": {"message": "unavailable"}}).encode())
            else:
                self._send(200, json.dumps({"recovered": True, "attempts": STATE["flaky_hits"]}).encode())
        elif self.path == "/server-error":
            STATE["server_error_hits"] += 1
            self._send(500, json.dumps({"error": {"message": "boom"}}).encode())
        elif self.path == "/slow":
            time.sleep(3)
            self._send(200, b"{}")
        elif self.path == "/sse":
            self._sse([
                {"data": {"n": 1}},
                {"data": {"n": 2, "text": "héllo → wörld 日本語"}},
                {"event": "done", "data": {"n": 3}},
            ])
        elif self.path == "/sse-done":
            raw = b'data: {"a": 1}\n\ndata: [DONE]\n\n'
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
        elif self.path == "/sse-no-trailing-blank":
            raw = b'data: {"tail": true}'
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
        elif self.path == "/sse-crlf":
            raw = b": keep-alive\r\nevent: x\r\ndata: {\"crlf\": 1}\r\n\r\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
        elif self.path == "/bad-chunks":
            # announce chunked, then write a body that is not valid chunk framing
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            # declare a 256-byte chunk, send 5 bytes, then hang up
            self.wfile.write(b"100\r\nshort\r\n")
            self.wfile.flush()
            self.close_connection = True
            try:
                self.connection.close()
            except OSError:
                pass
        elif self.path == "/context-overflow":
            self._send(400, json.dumps({"error": {
                "message": "This model's maximum context length is 8192 tokens"}}).encode())
        else:
            self._send(404, b'{"error":{"message":"no route"}}')

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        if self.path == "/echo":
            try:
                payload = json.loads(body.decode("utf-8"))
            except json.JSONDecodeError:
                payload = {"raw": body.decode("utf-8", "replace")}
            self._send(200, json.dumps({"echo": payload,
                                        "ctype": self.headers.get("Content-Type"),
                                        "auth": self.headers.get("Authorization"),
                                        "custom": self.headers.get("X-Custom")}).encode())
        elif self.path == "/sse":
            self._sse([{"data": {"chunk": i}} for i in range(5)], chunk_size=3)
        else:
            self._send(404, b'{"error":{"message":"no route"}}')


def start_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


class TransportTestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server, cls.base = start_server()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        STATE["flaky_hits"] = 0
        STATE["server_error_hits"] = 0
        self.transport = HTTPTransport(RetryPolicy(max_attempts=4, base_delay=0.01, max_delay=0.05,
                                                   jitter=0.0))

    def url(self, path: str) -> str:
        return self.base + path


class TestBasicRequests(TransportTestBase):
    def test_get_json(self):
        resp, handle = self.transport.request("GET", self.url("/json"))
        self.assertEqual(resp.status, 200)
        self.assertIsNone(handle)
        self.assertEqual(resp.json(), {"ok": True, "value": 42})

    def test_gzip_is_transparent(self):
        resp, _ = self.transport.request("GET", self.url("/gzip"))
        self.assertEqual(resp.json()["compressed"], True)
        self.assertEqual(resp.json()["text"], "héllo → wörld")

    def test_post_json_and_headers(self):
        resp, _ = self.transport.request(
            "POST", self.url("/echo"), json_body={"a": 1, "b": [1, 2]},
            headers={"Authorization": "Bearer sk-test", "X-Custom": "yes"})
        data = resp.json()
        self.assertEqual(data["echo"], {"a": 1, "b": [1, 2]})
        self.assertEqual(data["ctype"], "application/json")
        self.assertEqual(data["auth"], "Bearer sk-test")
        self.assertEqual(data["custom"], "yes")

    def test_charset_is_honoured(self):
        resp, _ = self.transport.request("GET", self.url("/json"))
        self.assertIsInstance(resp.text(), str)

    def test_header_lookup_is_case_insensitive(self):
        resp, _ = self.transport.request("GET", self.url("/json"))
        self.assertIn("json", (resp.header("content-type") or "").lower())
        self.assertIn("json", (resp.header("CONTENT-TYPE") or "").lower())
        self.assertIsNone(resp.header("nope-not-here"))


class TestErrorMapping(TransportTestBase):
    def test_401_maps_to_auth_error(self):
        with self.assertRaises(AuthError):
            self.transport.request("GET", self.url("/unauthorized"))

    def test_404_maps_to_model_not_found(self):
        with self.assertRaises(ModelNotFoundError):
            self.transport.request("GET", self.url("/missing"))

    def test_429_maps_to_rate_limit_with_retry_after(self):
        with self.assertRaises(RateLimitError) as ctx:
            self.transport.request("GET", self.url("/ratelimit"))
        self.assertIsNotNone(ctx.exception.retry_after)

    def test_context_overflow_message_is_preserved(self):
        with self.assertRaises(ProviderError) as ctx:
            self.transport.request("GET", self.url("/context-overflow"))
        self.assertIn("context length", str(ctx.exception))

    def test_500_exhausts_retries(self):
        with self.assertRaises(ProviderError):
            self.transport.request("GET", self.url("/server-error"))
        self.assertEqual(STATE["server_error_hits"], 4, "should have retried up to max_attempts")

    def test_503_then_success(self):
        resp, _ = self.transport.request("GET", self.url("/flaky"))
        self.assertEqual(resp.json()["recovered"], True)
        self.assertEqual(STATE["flaky_hits"], 3)

    def test_timeout_raises(self):
        transport = HTTPTransport(RetryPolicy(max_attempts=1))
        with self.assertRaises(TimeoutError_):
            transport.request("GET", self.url("/slow"), timeout=0.4)

    def test_malformed_chunked_stream_maps_to_network_error(self):
        """A server that breaks chunked framing must surface as NetworkError.

        `http.client.IncompleteRead` is an HTTPException, not an OSError, so it
        used to escape the transport's mapping and show up to users as an
        "unexpected provider error".
        """
        from nexuscli.core.errors import NetworkError

        with self.assertRaises(NetworkError):
            HTTPTransport(RetryPolicy(max_attempts=1, base_delay=0.0)).request(
                "GET", self.url("/bad-chunks"))

    def test_connection_refused_maps_to_network_error(self):
        from nexuscli.core.errors import NetworkError

        port = _free_port()
        with self.assertRaises(NetworkError):
            HTTPTransport(RetryPolicy(max_attempts=1)).request("GET", f"http://127.0.0.1:{port}/x")


class TestStreaming(TransportTestBase):
    def test_sse_events_over_chunked_transfer(self):
        resp, handle = self.transport.request("GET", self.url("/sse"), stream=True)
        self.assertIsNotNone(handle)
        events = list(iter_sse(handle))
        handle.close()
        self.assertEqual([e.json()["n"] for e in events], [1, 2, 3])
        self.assertEqual(events[1].json()["text"], "héllo → wörld 日本語")
        self.assertEqual(events[2].event, "done")

    def test_sse_split_across_tiny_packet_boundaries(self):
        # the POST route streams 5 events in 3-byte chunks, splitting frames
        resp, handle = self.transport.request("POST", self.url("/sse"), json_body={"x": 1}, stream=True)
        events = list(iter_sse(handle))
        handle.close()
        self.assertEqual([e.json()["chunk"] for e in events], [0, 1, 2, 3, 4])

    def test_done_sentinel(self):
        resp, handle = self.transport.request("GET", self.url("/sse-done"), stream=True)
        datas = [e.data for e in iter_sse(handle)]
        handle.close()
        self.assertEqual(datas, ['{"a": 1}', "[DONE]"])

    def test_event_without_trailing_blank_line_is_flushed(self):
        resp, handle = self.transport.request("GET", self.url("/sse-no-trailing-blank"), stream=True)
        events = list(iter_sse(handle))
        handle.close()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].json(), {"tail": True})

    def test_crlf_and_comments(self):
        resp, handle = self.transport.request("GET", self.url("/sse-crlf"), stream=True)
        events = list(iter_sse(handle))
        handle.close()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].json(), {"crlf": 1})
        self.assertEqual(events[0].event, "x")

    def test_stream_handle_context_manager(self):
        resp, handle = self.transport.request("GET", self.url("/sse"), stream=True)
        with handle as h:
            first = next(iter(iter_sse(h)))
            self.assertEqual(first.json()["n"], 1)


class TestRetryPolicy(unittest.TestCase):
    def test_exponential_backoff(self):
        policy = RetryPolicy(base_delay=1.0, max_delay=10.0, jitter=0.0)
        self.assertEqual(policy.delay_for(1), 1.0)
        self.assertEqual(policy.delay_for(2), 2.0)
        self.assertEqual(policy.delay_for(3), 4.0)
        self.assertEqual(policy.delay_for(9), 10.0, "must cap at max_delay")

    def test_retry_after_wins(self):
        policy = RetryPolicy(base_delay=1.0, max_delay=20.0, jitter=0.0)
        self.assertEqual(policy.delay_for(1, retry_after=3.0), 3.0)
        self.assertEqual(policy.delay_for(1, retry_after=999.0), 20.0, "still capped")
        self.assertEqual(policy.delay_for(1, retry_after=0.0), 1.0, "zero falls back to backoff")

    def test_jitter_stays_bounded(self):
        policy = RetryPolicy(base_delay=1.0, jitter=0.25)
        for _ in range(50):
            self.assertLessEqual(abs(policy.delay_for(2) - 2.0), 0.51)

    def test_retry_after_parsing(self):
        from nexuscli.transport.http import _parse_retry_after

        self.assertEqual(_parse_retry_after("3"), 3.0)
        self.assertEqual(_parse_retry_after(None), None)
        self.assertEqual(_parse_retry_after("garbage"), None)
        self.assertEqual(_parse_retry_after("0"), 0.0)


class TestSSEParserEdgeCases(unittest.TestCase):
    def feed(self, chunks):
        parser = SSEParser()
        out = []
        for chunk in chunks:
            out.extend(parser.feed(chunk))
        tail = parser.close()
        if tail is not None:
            out.append(tail)
        return out

    def test_split_multibyte_character(self):
        raw = 'data: {"t":"héllo→"}\n\n'.encode()
        events = self.feed([raw[i : i + 3] for i in range(0, len(raw), 3)])
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].json(), {"t": "héllo→"})

    def test_multiline_data_and_comments(self):
        events = self.feed([b": ping\r\nevent: m\r\ndata: a\r\ndata: b\r\n\r\n"])
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].data, "a\nb")
        self.assertEqual(events[0].event, "m")

    def test_lone_cr_line_endings(self):
        events = self.feed([b"data: a\rdata: b\r\r"])
        self.assertEqual([e.data for e in events], ["a\nb"])

    def test_bom_and_fields(self):
        events = self.feed([b"\xef\xbb\xbfid: 7\nretry: 1500\ndata: hi\n\n"])
        self.assertEqual((events[0].id, events[0].retry, events[0].data), ("7", 1500, "hi"))

    def test_colon_inside_value(self):
        events = self.feed([b'data: {"url":"http://x:8080"}\n\n'])
        self.assertEqual(events[0].json(), {"url": "http://x:8080"})

    def test_empty_data_field_only_events_are_not_dispatched(self):
        events = self.feed([b"\n\n\ndata: x\n\n"])
        self.assertEqual([e.data for e in events], ["x"])

    def test_byte_by_byte(self):
        raw = b"data: {\"a\": 1}\n\ndata: {\"b\": 2}\n\n"
        events = self.feed([raw[i : i + 1] for i in range(len(raw))])
        self.assertEqual([e.json() for e in events], [{"a": 1}, {"b": 2}])


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


if __name__ == "__main__":
    unittest.main(verbosity=2)
