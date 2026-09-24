"""Agent-loop tests: the behaviour that decides whether NEXUS is trustworthy.

Covers: plain turns, real tool execution against a temp workspace, invalid
arguments fed back for self-correction, unknown tools, permission denial,
parallel vs serial execution, abort mid-batch (no orphaned tool calls),
identical-call loop detection, max_turns, context overflow, usage accounting,
output-limit continuation, and wire-format validity of the produced history for
all three provider families.
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from _harness import RecordingUI, make_env  # noqa: E402

from nexuscli.providers.base import Message, RequestOptions, ToolCall  # noqa: E402
from nexuscli.providers.registry import create_provider  # noqa: E402
from nexuscli.transport.http import MockTransport  # noqa: E402


def tool_call_script(name: str, args: dict, cid: str = "c1") -> dict:
    return {"text": "", "tool_calls": [(name, args)], "usage": {"input_tokens": 5, "output_tokens": 5}}


class TestBasicTurn(unittest.TestCase):
    def test_plain_answer(self):
        agent, provider, tmp = make_env(scripts=["All good."])
        result = agent.send("hi")
        self.assertEqual(result.text, "All good.")
        self.assertTrue(result.ok)
        self.assertEqual(result.turns, 1)
        self.assertEqual([m.role for m in agent.history], ["user", "assistant"])

    def test_usage_and_cost_are_recorded(self):
        agent, provider, tmp = make_env(scripts=[{"text": "x", "usage": {"input_tokens": 100, "output_tokens": 50}}])
        result = agent.send("hi")
        self.assertEqual(result.usage.input_tokens, 100)
        self.assertEqual(result.usage.output_tokens, 50)
        self.assertEqual(result.usage.requests, 1)
        self.assertEqual(agent.services.ledger.totals()["requests"], 1)

    def test_system_prompt_contains_environment(self):
        agent, _, tmp = make_env(scripts=["ok"])
        text = agent.system_messages()[0].text
        self.assertIn(str(tmp), text)
        self.assertIn("Platform:", text)
        self.assertIn("Approval mode:", text)
        self.assertIn("```report", text)

    def test_report_block_is_parsed(self):
        agent, _, _ = make_env(scripts=["done\n```report\nstatus: done\nsummary: shipped\n"
                                        "artifacts: a.py, b.py\nfollowups: none\nconfidence: high\n```"])
        result = agent.send("go")
        self.assertEqual(result.report["status"], "done")
        self.assertEqual(result.report["artifact_list"], ["a.py", "b.py"])


class TestToolExecution(unittest.TestCase):
    def test_write_then_read_roundtrip(self):
        agent, provider, tmp = make_env(scripts=[
            tool_call_script("write_file", {"path": "hello.txt", "content": "Hello NEXUS"}),
            tool_call_script("read_file", {"path": "hello.txt"}, cid="c2"),
            {"text": "The file says Hello NEXUS."},
        ])
        result = agent.send("create and verify hello.txt")
        self.assertTrue(result.ok, result.errors)
        self.assertEqual(result.tool_calls, 2)
        self.assertIn("Hello NEXUS", result.text)
        self.assertTrue((tmp / "hello.txt").is_file())
        roles = [m.role for m in agent.history]
        self.assertEqual(roles, ["user", "assistant", "tool", "assistant", "tool", "assistant"])

    def test_edit_file_actually_edits(self):
        agent, provider, tmp = make_env(scripts=[
            tool_call_script("edit_file", {"path": "a.py", "old_text": "x = 1", "new_text": "x = 2"}),
            {"text": "changed"},
        ])
        (tmp / "a.py").write_text("import os\nx = 1\nprint(x)\n")
        agent.send("bump x")
        body = (tmp / "a.py").read_text()
        self.assertIn("x = 2", body)
        self.assertNotIn("x = 1", body)
        self.assertIn("print(x)", body, "untouched lines must survive")

    def test_checkpoint_and_undo(self):
        agent, provider, tmp = make_env(scripts=[
            tool_call_script("write_file", {"path": "f.txt", "content": "v2"}),
            {"text": "done"},
        ])
        (tmp / "f.txt").write_text("v1")
        agent.send("update")
        self.assertEqual((tmp / "f.txt").read_text(), "v2")
        restore = agent.services.checkpoints.restore()
        self.assertTrue(restore.ok, restore.errors)
        self.assertEqual((tmp / "f.txt").read_text(), "v1")

    def test_invalid_arguments_are_fed_back_for_self_correction(self):
        agent, provider, tmp = make_env(scripts=[
            tool_call_script("read_file", {"path": 123, "limit": "abc"}, cid="bad1"),
            tool_call_script("read_file", {"path": "ok.txt"}, cid="good1"),
            {"text": "recovered"},
        ])
        (tmp / "ok.txt").write_text("content")
        result = agent.send("read it")
        self.assertTrue(result.ok, result.errors)
        self.assertIn("recovered", result.text)
        tool_msgs = [m for m in agent.history if m.role == "tool"]
        self.assertTrue(tool_msgs[0].meta.get("is_error"), "first call must be reported as an error")
        self.assertFalse(tool_msgs[1].meta.get("is_error"))

    def test_unknown_tool_is_reported_not_fatal(self):
        agent, provider, tmp = make_env(scripts=[
            tool_call_script("teleport_file", {"path": "x"}),
            {"text": "ok, used another way"},
        ])
        result = agent.send("do it")
        self.assertTrue(result.ok, result.errors)
        tool_msg = [m for m in agent.history if m.role == "tool"][0]
        self.assertIn("Unknown tool", tool_msg.text)
        self.assertIn("read_file", tool_msg.text)  # suggests what does exist

    def test_malformed_json_arguments_are_repaired(self):
        agent, provider, tmp = make_env(scripts=[
            {"text": "", "tool_calls": [("write_file", '{"path": "m.txt", "content": "hi"')]},
            {"text": "wrote it"},
        ])
        agent.send("write")
        self.assertTrue((tmp / "m.txt").is_file(), "truncated JSON should be repaired")

    def test_tool_crash_becomes_error_result(self):
        agent, provider, tmp = make_env(scripts=[
            tool_call_script("bash", {"command": "exit 3"}),
            {"text": "handled"},
        ])
        result = agent.send("run failing command")
        self.assertTrue(result.ok)
        tool_msg = [m for m in agent.history if m.role == "tool"][0]
        self.assertIn("exit=3", tool_msg.text)
        self.assertTrue(tool_msg.meta.get("is_error"))


class TestPermissions(unittest.TestCase):
    def test_denied_without_confirmer(self):
        agent, provider, tmp = make_env(
            scripts=[tool_call_script("write_file", {"path": "x.txt", "content": "x"}), {"text": "blocked"}],
            mode="suggest",
        )
        result = agent.send("write a file")
        self.assertTrue(result.denied)
        self.assertFalse((tmp / "x.txt").exists())
        tool_msg = [m for m in agent.history if m.role == "tool"][0]
        self.assertIn("Permission denied", tool_msg.text)

    def test_confirmer_is_consulted_and_can_remember(self):
        calls = []

        def confirmer(req, tool, args):
            calls.append((tool, req.risk))
            return True, f"{tool}:*"

        agent, provider, tmp = make_env(
            scripts=[tool_call_script("write_file", {"path": "a.txt", "content": "1"}, cid="c1"),
                     tool_call_script("write_file", {"path": "b.txt", "content": "2"}, cid="c2"),
                     {"text": "done"}],
            mode="suggest", confirmer=confirmer,
        )
        agent.send("write two files")
        self.assertEqual(len(calls), 1, "second write should hit the remembered allow rule")
        self.assertTrue((tmp / "a.txt").exists() and (tmp / "b.txt").exists())

    def test_read_only_mode_blocks_writes(self):
        agent, provider, tmp = make_env(
            scripts=[tool_call_script("write_file", {"path": "x.txt", "content": "x"}), {"text": "no"}],
            mode="read-only",
        )
        agent.send("write")
        self.assertFalse((tmp / "x.txt").exists())
        self.assertTrue(agent.services.permissions.denied >= 1)

    def test_deny_rule_wins_over_full_auto(self):
        agent, provider, tmp = make_env(
            scripts=[tool_call_script("write_file", {"path": ".env", "content": "secret"}), {"text": "blocked"}],
            mode="full-auto", rules={"deny": ["write_file:.env*"]},
        )
        agent.send("write env")
        self.assertFalse((tmp / ".env").exists())


class TestExecutionStrategy(unittest.TestCase):
    def test_read_only_batch_runs_in_parallel(self):
        agent, provider, tmp = make_env(scripts=[
            {"text": "", "tool_calls": [("read_file", {"path": f"f{i}.txt"}) for i in range(4)]},
            {"text": "read all"},
        ])
        for i in range(4):
            (tmp / f"f{i}.txt").write_text(f"content {i}")
        agent.send("read four files")
        self.assertEqual(len(agent.services.ui.tools_started), 4)
        self.assertEqual(len([m for m in agent.history if m.role == "tool"]), 4)

    def test_mixed_batch_runs_serially_in_order(self):
        agent, provider, tmp = make_env(scripts=[
            {"text": "", "tool_calls": [("read_file", {"path": "a.txt"}),
                                        ("write_file", {"path": "b.txt", "content": "x"}),
                                        ("read_file", {"path": "a.txt"})]},
            {"text": "done"},
        ])
        (tmp / "a.txt").write_text("A")
        agent.send("mixed")
        order = [name for name, _ in agent.services.ui.tools_ended]
        self.assertEqual(order, ["read_file", "write_file", "read_file"])

    def test_identical_call_loop_is_detected(self):
        same = tool_call_script("read_file", {"path": "loop.txt"})
        agent, provider, tmp = make_env(scripts=[same, same, same, same, {"text": "gave up"}])
        (tmp / "loop.txt").write_text("x")
        result = agent.send("loop")
        self.assertTrue(any("repeated the identical" in e for e in result.errors), result.errors)
        self.assertLessEqual(result.turns, 5)


class TestAbortAndLimits(unittest.TestCase):
    def test_abort_leaves_no_orphan_tool_calls(self):
        agent, provider, tmp = make_env(scripts=[
            {"text": "", "tool_calls": [("read_file", {"path": "a.txt"}), ("read_file", {"path": "b.txt"})]},
            {"text": "never"},
        ])
        (tmp / "a.txt").write_text("A")
        (tmp / "b.txt").write_text("B")

        def cancel_soon():
            time.sleep(0.01)
            agent.services.cancelled.set()

        threading.Thread(target=cancel_soon, daemon=True).start()
        agent.send("read both")
        answered = {m.tool_call_id for m in agent.history if m.role == "tool"}
        requested = {tc.id for m in agent.history if m.role == "assistant" for tc in m.tool_calls}
        self.assertEqual(requested - answered, set(), "every tool_call must have a result")

    def test_max_turns_is_reported(self):
        scripts = [tool_call_script("read_file", {"path": f"f{i}.txt"}, cid=f"c{i}") for i in range(30)]
        agent, provider, tmp = make_env(scripts=scripts, max_turns=3)
        for i in range(30):
            (tmp / f"f{i}.txt").write_text("x")
        result = agent.send("read everything")
        self.assertTrue(any("max_turns" in e for e in result.errors), result.errors)
        self.assertLessEqual(result.turns, 3)

    def test_context_overflow_is_graceful(self):
        agent, provider, tmp = make_env(scripts=["never"])
        agent.options.max_turns = 2
        big = "z" * 200_000
        agent.history.append(Message.user(big))
        # force a tiny window so budgeting must fail
        agent._target = None
        from nexuscli.core.router import RouteTarget
        from nexuscli.providers.base import ModelInfo

        agent._target = RouteTarget(provider_key="mock", model="mock-1",
                                    info=ModelInfo(id="mock-1", provider="mock", context_window=2000,
                                                   max_output_tokens=500))
        result = agent.send("and more")
        self.assertTrue(any("ContextOverflowError" in e for e in result.errors), result.errors)
        self.assertTrue(agent.services.ui.errors)

    def test_output_limit_triggers_one_continuation(self):
        agent, provider, tmp = make_env(scripts=[
            {"text": "part one ", "finish_reason": "length"},
            {"text": "part two", "finish_reason": "stop"},
        ])
        result = agent.send("long answer")
        self.assertIn("part one", result.text)
        self.assertIn("part two", result.text)
        self.assertEqual(result.turns, 2)


class TestHistoryValidity(unittest.TestCase):
    """The history NEXUS builds must be accepted by every provider adapter."""

    RESPONSES = {
        "openai": {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}], "usage": {}},
        "anthropic": {"content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn", "usage": {}},
        "gemini": {"candidates": [{"content": {"parts": [{"text": "ok"}]}, "finishReason": "STOP"}],
                   "usageMetadata": {}},
    }
    MODELS = {"openai": "gpt-4o", "anthropic": "claude-sonnet-4-5", "gemini": "gemini-2.5-pro"}

    def _history(self):
        agent, provider, tmp = make_env(scripts=[
            tool_call_script("write_file", {"path": "f.txt", "content": "hello"}),
            tool_call_script("read_file", {"path": "f.txt"}, cid="c2"),
            {"text": "all done"},
        ])
        agent.send("create and verify")
        return agent.system_messages() + agent.history

    def test_all_providers_accept_the_history(self):
        history = self._history()
        for name, body in self.RESPONSES.items():
            with self.subTest(provider=name):
                transport = MockTransport()
                transport.queue((200, body))
                provider = create_provider(name, api_key="k", transport=transport)
                result = provider.complete(history, (), RequestOptions(
                    model=self.MODELS[name], max_tokens=64, stream=False))
                self.assertEqual(result.message.text, "ok")

    def test_no_orphans_in_final_history(self):
        history = self._history()
        seen = set()
        for m in history:
            if m.role == "assistant":
                seen |= {tc.id for tc in m.tool_calls}
            elif m.role == "tool":
                self.assertIn(m.tool_call_id, seen)


class TestSubAgent(unittest.TestCase):
    def test_task_tool_reports_unavailable_when_no_runtime(self):
        agent, provider, tmp = make_env(scripts=[
            tool_call_script("task", {"prompt": "do research"}),
            {"text": "fell back to doing it myself"},
        ])
        result = agent.send("delegate")
        tool_msg = [m for m in agent.history if m.role == "tool"][0]
        self.assertIn("not available", tool_msg.text)
        self.assertTrue(result.ok)

    def test_task_tool_uses_injected_spawner(self):
        agent, provider, tmp = make_env(scripts=[
            tool_call_script("task", {"subagent_type": "reviewer", "prompt": "review x"}),
            {"text": "reviewed"},
        ])
        seen = {}

        def spawn(**kwargs):
            seen.update(kwargs)
            return "REVIEW: looks good"

        agent.services.vars["spawn_subagent"] = spawn
        agent.services.spawn_subagent = spawn
        agent.send("delegate")
        tool_msg = [m for m in agent.history if m.role == "tool"][0]
        self.assertIn("REVIEW: looks good", tool_msg.text)
        self.assertEqual(seen["role"], "reviewer")
        self.assertEqual(seen["parent"], "main")


if __name__ == "__main__":
    unittest.main(verbosity=2)
