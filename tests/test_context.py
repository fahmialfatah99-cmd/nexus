"""Token estimation + context budgeting tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nexuscli.core.context import (  # noqa: E402
    PRUNED_MARKER,
    ContextManager,
    count_tokens,
    estimate_tokens,
    message_tokens,
    render_transcript,
)
from nexuscli.providers.base import Message, ToolCall  # noqa: E402


def big_history(turns: int = 40, tool_chars: int = 20_000) -> list:
    msgs = [Message.system("sys" * 500)]
    for i in range(turns):
        msgs.append(Message.user(f"question {i} " + "x" * 500))
        msgs.append(Message(role="assistant", content="",
                            tool_calls=[ToolCall(f"c{i}", "bash", f'{{"command":"cmd {i}"}}')]))
        msgs.append(Message.tool_result(f"c{i}", "bash", "y" * tool_chars))
    return msgs


def assert_no_orphans(testcase: unittest.TestCase, msgs) -> None:
    seen = set()
    for m in msgs:
        if m.role == "assistant":
            seen |= {t.id for t in m.tool_calls}
        elif m.role == "tool":
            testcase.assertIn(m.tool_call_id, seen, f"orphan tool result {m.tool_call_id}")


class TestEstimation(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(estimate_tokens(""), 0)

    def test_english_is_roughly_four_chars_per_token(self):
        text = "The quick brown fox jumps over the lazy dog. " * 10
        est = estimate_tokens(text)
        self.assertTrue(0.15 * len(text) < est < 0.40 * len(text), est)

    def test_cjk_is_not_underestimated(self):
        text = "你好世界测试中文分词估计" * 10
        self.assertGreater(estimate_tokens(text), len(text) * 0.8)

    def test_monotonic(self):
        self.assertLess(estimate_tokens("abc"), estimate_tokens("abc" * 100))

    def test_message_tokens_counts_tool_calls(self):
        plain = Message.user("hello")
        with_calls = Message(role="assistant", content="hello",
                             tool_calls=[ToolCall("a", "bash", '{"command":"ls -la"}')])
        self.assertGreater(message_tokens(with_calls), message_tokens(plain))


class TestBudgeting(unittest.TestCase):
    def setUp(self):
        self.cm = ContextManager()

    def test_limit_for_unknown_window_is_zero(self):
        self.assertEqual(self.cm.limit_for(0), 0)

    def test_limit_reserves_output(self):
        limit = self.cm.limit_for(128_000, 8_192)
        self.assertGreater(limit, 0)
        self.assertLess(limit, 128_000)

    def test_under_budget_is_untouched(self):
        msgs = [Message.system("s"), Message.user("hi")]
        out, report = self.cm.assemble(msgs, window=128_000, max_output=4096, model="gpt-4o")
        self.assertEqual(report.before, report.after)
        self.assertFalse(report.actions)
        self.assertFalse(report.overflow)
        self.assertEqual(len(out), 2)

    def test_unknown_window_skips_budgeting(self):
        msgs = big_history()
        out, report = self.cm.assemble(msgs, window=0, model="gpt-4o")
        self.assertEqual(report.limit, 0)
        self.assertEqual(count_tokens(out, "gpt-4o"), count_tokens(msgs, "gpt-4o"))

    def test_shrinks_to_reachable_budget(self):
        msgs = big_history()
        before = count_tokens(msgs, "gpt-4o")
        for limit in (3000, 1500):
            out, report = self.cm.assemble(msgs, model="gpt-4o", limit=limit)
            self.assertLessEqual(report.after, limit)
            self.assertLess(report.after, before)
            self.assertFalse(report.overflow)
            assert_no_orphans(self, out)
            self.assertEqual(out[0].role, "system")

    def test_escalation_reaches_tiny_budgets(self):
        msgs = big_history()
        out, report = self.cm.assemble(msgs, model="gpt-4o", limit=1200)
        self.assertLessEqual(report.after, 1200)
        assert_no_orphans(self, out)

    def test_impossible_budget_flags_overflow_instead_of_truncating(self):
        msgs = big_history()
        out, report = self.cm.assemble(msgs, model="gpt-4o", limit=10)
        self.assertTrue(report.overflow, "must report overflow rather than silently truncating")
        self.assertTrue(out)
        self.assertEqual(out[0].role, "system")
        self.assertIn("STILL OVER BUDGET", report.actions)

    def test_never_orphans_tool_results(self):
        for limit in (5000, 2000, 900, 300):
            out, _ = self.cm.assemble(big_history(), model="gpt-4o", limit=limit)
            assert_no_orphans(self, out)

    def test_pruned_marker_is_used(self):
        out, report = self.cm.assemble(big_history(), model="gpt-4o", limit=1500)
        self.assertTrue(any(PRUNED_MARKER in (m.text or "") for m in out if m.role == "tool")
                        or any("squeezed" in a for a in report.actions))

    def test_report_describe(self):
        _, report = self.cm.assemble(big_history(), model="gpt-4o", limit=2000)
        self.assertIn("est. tokens", report.describe())


class TestCompaction(unittest.TestCase):
    def setUp(self):
        self.cm = ContextManager()

    def test_needs_compaction(self):
        self.assertTrue(self.cm.needs_compaction(big_history(), 8_000, 1_000, "gpt-4o"))
        self.assertFalse(self.cm.needs_compaction([Message.user("hi")], 128_000, 4_096, "gpt-4o"))

    def test_needs_compaction_unknown_window(self):
        self.assertFalse(self.cm.needs_compaction(big_history(), 0, 0, "gpt-4o"))

    def test_split_keeps_tool_pairs_together(self):
        msgs = big_history()
        to_summarise, keep = self.cm.split_for_compaction(msgs, keep_recent=6)
        self.assertTrue(to_summarise)
        self.assertTrue(keep)
        assert_no_orphans(self, keep)
        self.assertEqual(to_summarise[0].role, "system")
        # nothing is lost
        self.assertEqual(len([m for m in msgs if m.role != "system"]),
                         len([m for m in to_summarise if m.role != "system"]) + len(keep))

    def test_split_with_few_messages(self):
        msgs = [Message.system("s"), Message.user("hi")]
        to_summarise, keep = self.cm.split_for_compaction(msgs, keep_recent=6)
        self.assertEqual(to_summarise, [msgs[0]])
        self.assertEqual(keep, [msgs[1]])

    def test_prompt_is_structured(self):
        prompt = ContextManager.compaction_prompt()
        for section in ("## Goal", "## Decisions", "## State", "## Open items", "## Constraints"):
            self.assertIn(section, prompt)


class TestTranscript(unittest.TestCase):
    def test_renders_roles_and_tool_calls(self):
        msgs = [Message.system("s"), Message.user("q"),
                Message(role="assistant", content="thinking",
                        tool_calls=[ToolCall("c1", "bash", '{"command":"ls"}')]),
                Message.tool_result("c1", "bash", "file.txt")]
        text = render_transcript(msgs)
        self.assertIn("USER: q", text)
        self.assertIn("[tool calls] bash(", text)
        self.assertIn("[bash result] file.txt", text)
        self.assertNotIn("SYSTEM:", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
