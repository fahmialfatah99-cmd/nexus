"""Local OpenAI-compatible gateway tests (9Router, Ollama, vLLM, LM Studio, custom).

These run a stub gateway on a real loopback port and drive the **production**
stack through it: `HTTPTransport` over real sockets → `OpenAICompatProvider` →
`ModelRouter` → `Agent` → real tools writing real files. That is the exact path a
9Router user takes, so it is verified rather than assumed.

9Router specifics that are asserted here:
  * base URL `http://localhost:<port>/v1` with `/chat/completions` and `/models`
  * the gateway does not validate the API key but requires a non-empty one, so
    NEXUS sends `Authorization: Bearer local`
  * model ids carry a provider prefix (`kr/claude-sonnet-4.5`, `glm/glm-5.1`)
  * streaming tool calls arrive fragmented across SSE chunks
"""

from __future__ import annotations

import json
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from nexuscli.agents.persona import get_persona  # noqa: E402
from nexuscli.agents.runtime import Agent, AgentOptions, Services  # noqa: E402
from nexuscli.core.checkpoints import CheckpointStore  # noqa: E402
from nexuscli.core.context import ContextManager  # noqa: E402
from nexuscli.core.permissions import PermissionEngine  # noqa: E402
from nexuscli.core.router import ModelRouter  # noqa: E402
from nexuscli.core.usage import UsageLedger  # noqa: E402
from nexuscli.providers.base import Message, RequestOptions  # noqa: E402
from nexuscli.providers.registry import create_provider, get_spec  # noqa: E402
from nexuscli.tools import build_registry  # noqa: E402

import tempfile  # noqa: E402

GATEWAY_MODELS = ["kr/claude-sonnet-4.5", "glm/glm-5.1", "oc/auto", "minimax/MiniMax-M2.7"]
REQUESTS: list = []


class GatewayHandler(BaseHTTPRequestHandler):
    """A minimal stand-in for a local OpenAI-compatible router."""

    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _json(self, code: int, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        REQUESTS.append({"method": "GET", "path": self.path, "auth": self.headers.get("Authorization")})
        if self.path.rstrip("/").endswith("/models"):
            self._json(200, {"object": "list", "data": [{"id": m, "object": "model"} for m in GATEWAY_MODELS]})
        else:
            self._json(404, {"error": {"message": "no route"}})

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            body = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            self._json(400, {"error": {"message": "invalid json"}})
            return
        REQUESTS.append({"method": "POST", "path": self.path, "auth": self.headers.get("Authorization"),
                         "body": body})
        if not self.path.rstrip("/").endswith("/chat/completions"):
            self._json(404, {"error": {"message": "no route"}})
            return
        # The scripted behaviour is chosen by the model name, so one server can
        # exercise several paths.
        model = body.get("model", "")
        if body.get("stream"):
            self._stream(model, body)
        else:
            self._complete(model, body)

    @staticmethod
    def _has_tool_result(body: dict) -> bool:
        return any(m.get("role") == "tool" for m in body.get("messages") or [])

    def _complete(self, model: str, body: dict):
        if model.endswith("plain") or self._has_tool_result(body):
            message = {"role": "assistant", "content": "plain answer"}
        else:
            message = {"role": "assistant", "content": "",
                       "tool_calls": [{"id": "gw_1", "type": "function", "function": {
                           "name": "write_file",
                           "arguments": json.dumps({"path": "made-by-gateway.txt",
                                                    "content": "written through the gateway"})}}]}
        self._json(200, {"id": "chatcmpl-gw", "object": "chat.completion", "model": model,
                         "choices": [{"index": 0, "message": message,
                                      "finish_reason": "tool_calls" if message.get("tool_calls") else "stop"}],
                         "usage": {"prompt_tokens": 120, "completion_tokens": 25,
                                   "prompt_tokens_details": {"cached_tokens": 40}}})

    def _stream(self, model: str, body: dict):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def send(obj):
            payload = f"data: {json.dumps(obj)}\n\n".encode()
            self.wfile.write(b"%X\r\n%s\r\n" % (len(payload), payload))
            self.wfile.flush()

        if self._has_tool_result(body):
            # Second round: the tool ran, so answer and stop calling tools
            # (exactly what a real model does).
            for piece in ("Done. ", "The file ", "was created."):
                send({"id": "c", "object": "chat.completion.chunk", "model": model,
                      "choices": [{"index": 0, "delta": {"content": piece}}]})
            send({"id": "c", "object": "chat.completion.chunk", "model": model,
                  "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
            send({"id": "c", "object": "chat.completion.chunk", "model": model, "choices": [],
                  "usage": {"prompt_tokens": 90, "completion_tokens": 8}})
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
            return

        # 1) streamed prose
        for piece in ("Written ", "through ", "the gateway."):
            send({"id": "c", "object": "chat.completion.chunk", "model": model,
                  "choices": [{"index": 0, "delta": {"content": piece}}]})
        # 2) a tool call fragmented across three chunks (the classic failure mode)
        args = json.dumps({"path": "streamed.txt", "content": "hello from stream"})
        fragments = ['{"path": "str', 'eamed.txt", "cont', 'ent": "hello from stream"}']
        assert "".join(fragments) == args, "fragments must reassemble to the full JSON"
        send({"id": "c", "object": "chat.completion.chunk", "model": model,
              "choices": [{"index": 0, "delta": {"tool_calls": [
                  {"index": 0, "id": "gw_stream_1", "type": "function",
                   "function": {"name": "write_file", "arguments": fragments[0]}}]}}]})
        for frag in fragments[1:]:
            send({"id": "c", "object": "chat.completion.chunk", "model": model,
                  "choices": [{"index": 0, "delta": {"tool_calls": [
                      {"index": 0, "function": {"arguments": frag}}]}}]})
        send({"id": "c", "object": "chat.completion.chunk", "model": model,
              "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]})
        send({"id": "c", "object": "chat.completion.chunk", "model": model, "choices": [],
              "usage": {"prompt_tokens": 200, "completion_tokens": 30}})
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()


def start_gateway():
    server = ThreadingHTTPServer(("127.0.0.1", 0), GatewayHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}/v1"


class TestProviderRegistration(unittest.TestCase):
    def test_9router_is_a_known_local_provider(self):
        spec = get_spec("9router")
        self.assertEqual(spec.default_base_url, "http://localhost:20128/v1")
        self.assertTrue(spec.local)
        self.assertFalse(spec.requires_api_key, "a local gateway must not demand an API key")

    def test_placeholder_key_is_sent(self):
        provider = create_provider("9router")
        self.assertEqual(provider._headers().get("Authorization"), "Bearer local")

    def test_base_url_is_overridable(self):
        provider = create_provider("9router", base_url="http://127.0.0.1:9999/v1")
        self.assertEqual(provider.base_url, "http://127.0.0.1:9999/v1")


class GatewayTestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server, cls.base_url = start_gateway()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        REQUESTS.clear()
        self._tmp = tempfile.TemporaryDirectory()
        self.work = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def router(self) -> ModelRouter:
        return ModelRouter(default_provider="9router", default_model="kr/claude-sonnet-4.5",
                           provider_configs={"9router": {"base_url": self.base_url}},
                           max_retries=1)

    def agent(self, model: str = "kr/claude-sonnet-4.5") -> Agent:
        permissions = PermissionEngine(mode="full-auto", workspace_root=self.work)
        services = Services(registry=build_registry(), permissions=permissions, router=self.router(),
                            context=ContextManager(), ledger=UsageLedger(),
                            checkpoints=CheckpointStore(self.work, "gw-test"),
                            cwd=self.work, workspace_root=self.work)
        return Agent(services=services, persona=get_persona("main"),
                     options=AgentOptions(model_spec=f"9router:{model}", max_turns=4))


class TestGatewayListing(GatewayTestBase):
    def test_models_are_listed_from_the_gateway(self):
        router = self.router()
        models = router.list_models("9router", refresh=True)
        self.assertEqual(models, sorted(GATEWAY_MODELS), "list_models returns a sorted, de-duplicated list")

    def test_models_request_carries_the_placeholder_key(self):
        self.router().list_models("9router", refresh=True)
        self.assertEqual(REQUESTS[-1]["auth"], "Bearer local")


class TestGatewayCompletion(GatewayTestBase):
    def test_non_streaming_tool_call_writes_a_real_file(self):
        provider = create_provider("9router", base_url=self.base_url)
        result = provider.complete([Message.user("create the file")], (),
                                   RequestOptions(model="kr/claude-sonnet-4.5", stream=False))
        self.assertEqual(result.finish_reason, "tool_calls")
        self.assertEqual(result.message.tool_calls[0].name, "write_file")
        self.assertEqual(result.message.tool_calls[0].parsed_args()["path"], "made-by-gateway.txt")
        self.assertEqual(result.usage.input_tokens, 120)
        self.assertEqual(result.usage.cached_tokens, 40)

    def test_request_body_is_openai_shaped(self):
        provider = create_provider("9router", base_url=self.base_url)
        from nexuscli.providers.base import ToolSpec

        provider.complete(
            [Message.system("be brief"), Message.user("hi")],
            [ToolSpec(name="t", description="d", parameters={"type": "object", "properties": {}})],
            RequestOptions(model="glm/glm-5.1", stream=False, temperature=0.3))
        body = REQUESTS[-1]["body"]
        self.assertEqual(body["model"], "glm/glm-5.1")
        self.assertEqual(body["messages"][0], {"role": "system", "content": "be brief"})
        self.assertEqual(body["tools"][0]["type"], "function")
        self.assertEqual(body["temperature"], 0.3)

    def test_streaming_fragmented_tool_call_is_reassembled(self):
        provider = create_provider("9router", base_url=self.base_url)
        events = []
        result = provider.complete([Message.user("go")], (),
                                   RequestOptions(model="kr/claude-sonnet-4.5", stream=True),
                                   on_event=events.append)
        self.assertEqual(result.message.text, "Written through the gateway.")
        self.assertEqual(len(result.message.tool_calls), 1)
        call = result.message.tool_calls[0]
        self.assertEqual(call.id, "gw_stream_1")
        self.assertEqual(call.name, "write_file")
        self.assertEqual(call.parsed_args(), {"path": "streamed.txt", "content": "hello from stream"})
        self.assertEqual(result.usage.input_tokens, 200)
        self.assertEqual(result.usage.output_tokens, 30)
        self.assertEqual(result.finish_reason, "tool_calls")
        self.assertTrue(events)


class TestGatewayAgentLoop(GatewayTestBase):
    def test_full_turn_executes_the_tool_and_answers(self):
        agent = self.agent()
        result = agent.send("please create streamed.txt")
        self.assertTrue(result.ok, result.errors)
        target = self.work / "streamed.txt"
        self.assertTrue(target.is_file(), "the gateway tool call must have written a real file")
        self.assertEqual(target.read_text(), "hello from stream")
        self.assertEqual(result.target.provider_key, "9router")
        self.assertEqual(result.target.model, "kr/claude-sonnet-4.5")
        self.assertGreaterEqual(result.usage.input_tokens, 200)
        roles = [m.role for m in agent.history]
        self.assertEqual(roles[:4], ["user", "assistant", "tool", "assistant"])

    def test_model_spec_with_slash_prefix_resolves(self):
        router = self.router()
        target = router.resolve("9router:minimax/MiniMax-M2.7")
        self.assertEqual(target.provider_key, "9router")
        self.assertEqual(target.model, "minimax/MiniMax-M2.7")
        self.assertEqual(target.info.context_window, 0, "unknown models must stay unknown")

    def test_checkpoint_and_undo_through_the_gateway(self):
        agent = self.agent()
        agent.send("create the file")
        self.assertTrue((self.work / "streamed.txt").is_file())
        restore = agent.services.checkpoints.restore()
        self.assertTrue(restore.ok, restore.errors)
        self.assertFalse((self.work / "streamed.txt").exists(),
                         "a file created through the gateway must be removable by /undo")


class TestGatewayFailover(GatewayTestBase):
    def test_dead_gateway_fails_over_to_the_mock(self):
        from nexuscli.providers.mock import MockProvider

        router = ModelRouter(default_provider="9router", default_model="kr/claude-sonnet-4.5",
                             provider_configs={"9router": {"base_url": "http://127.0.0.1:1/v1"}},
                             failover=["mock:mock-1"], max_retries=1)
        router.register_provider("mock", MockProvider(scripts=["answered by the fallback"]))
        switches = []
        router.on_failover = lambda failed, target, reason: switches.append((failed.label, target.label))
        result, target = router.complete([Message.user("hi")], (),
                                         options=RequestOptions(model="kr/claude-sonnet-4.5", stream=False))
        self.assertEqual(target.provider_key, "mock")
        self.assertEqual(result.message.text, "answered by the fallback")
        self.assertTrue(switches, "the switch must be observable")

    def test_dead_gateway_failover_reason_mentions_connection(self):
        from nexuscli.providers.mock import MockProvider

        router = ModelRouter(default_provider="9router", default_model="kr/claude-sonnet-4.5",
                             provider_configs={"9router": {"base_url": "http://127.0.0.1:1/v1"}},
                             failover=["mock:mock-1"], max_retries=1)
        router.register_provider("mock", MockProvider(scripts=["fallback answered"]))
        result, target = router.complete([Message.user("hi")], (),
                                         options=RequestOptions(model="kr/claude-sonnet-4.5", stream=False))
        self.assertEqual(target.provider_key, "mock")
        self.assertIsNone(router.last_error, "a successful failover clears the error state")

    def test_request_level_errors_do_not_fail_over(self):
        """A malformed request must surface, not silently hop providers."""
        from nexuscli.core.errors import ValidationError
        from nexuscli.providers.mock import MockProvider

        router = ModelRouter(default_provider="9router", default_model="m",
                             provider_configs={"9router": {"base_url": self.base_url}},
                             failover=["mock:mock-1"], max_retries=1)
        router.register_provider("mock", MockProvider(scripts=["never used"]))
        with self.assertRaises(ValidationError):
            router.complete([], (), options=RequestOptions(model="m", stream=False))


if __name__ == "__main__":
    unittest.main(verbosity=2)
