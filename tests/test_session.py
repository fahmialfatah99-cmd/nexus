"""Session persistence tests, including crash recovery."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nexuscli.core.session import (  # noqa: E402
    Session,
    SessionStore,
    message_from_dict,
    message_to_dict,
    new_session_id,
)
from nexuscli.providers.base import ImageBlock, Message, TextBlock, ToolCall, Usage  # noqa: E402


class FakeTarget:
    provider_key = "openai"
    model = "gpt-4o"


class FakeResult:
    usage = Usage(input_tokens=100, output_tokens=50, requests=1)
    cost = 0.0123
    turns = 2
    finish_reason = "stop"
    ok = True
    target = FakeTarget()


class FakeAgent:
    name = "main"


class SessionTestBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.store = SessionStore(self.tmp)
        self.addCleanup(self._tmp.cleanup)

    def sample_session(self, close: bool = True) -> Session:
        s = self.store.create(cwd=self.tmp, model="gpt-4o", provider="openai",
                              approval_mode="auto-edit")
        s.add_message(Message.user("Please refactor the parser and add tests"))
        s.add_message(Message(role="assistant", content="ok",
                              tool_calls=[ToolCall("c1", "edit_file", '{"path":"p.py"}')]))
        s.add_message(Message.tool_result("c1", "edit_file", "done"))
        s.add_message(Message.user([TextBlock("look"), ImageBlock(data=b"\x89PNG", mime="image/png")]))
        s.record_turn(FakeAgent(), FakeResult())
        s.note_touched(["p.py", "t.py"])
        if close:
            s.close()
        return s


class TestSessionLifecycle(SessionTestBase):
    def test_ids_are_unique_and_sortable(self):
        ids = {new_session_id() for _ in range(50)}
        self.assertEqual(len(ids), 50)

    def test_roundtrip_preserves_everything(self):
        s = self.sample_session()
        loaded = self.store.load(s.meta.id)
        self.assertIsNotNone(loaded)
        self.assertEqual(len(loaded.messages), 4)
        self.assertEqual(loaded.messages[1].tool_calls[0].name, "edit_file")
        self.assertEqual(loaded.messages[2].tool_call_id, "c1")
        self.assertEqual(loaded.messages[2].role, "tool")
        self.assertFalse(loaded.messages[2].meta.get("is_error"))
        self.assertEqual(loaded.meta.turns, 1)
        self.assertEqual(loaded.meta.input_tokens, 100)
        self.assertEqual(loaded.meta.output_tokens, 50)
        self.assertAlmostEqual(loaded.meta.cost_usd, 0.0123, places=6)
        self.assertEqual(loaded.meta.touched, ["p.py", "t.py"])
        self.assertEqual(loaded.meta.model, "gpt-4o")
        self.assertEqual(loaded.meta.agents, ["main"])
        self.assertGreaterEqual(loaded.meta.updated_at, loaded.meta.started_at)

    def test_title_derived_from_first_user_message(self):
        s = self.sample_session()
        loaded = self.store.load(s.meta.id)
        self.assertEqual(loaded.meta.title, "Please refactor the parser and add tests")

    def test_long_title_is_truncated(self):
        s = self.store.create(cwd=self.tmp)
        s.add_message(Message.user("x" * 400))
        s.close()
        self.assertLessEqual(len(self.store.load(s.meta.id).meta.title), 81)

    def test_images_are_not_persisted(self):
        s = self.sample_session()
        loaded = self.store.load(s.meta.id)
        self.assertIn("image omitted", loaded.messages[3].text)
        self.assertLess(s.path.stat().st_size, 20_000)

    def test_listing_and_latest(self):
        s = self.sample_session()
        metas = self.store.list()
        self.assertEqual(len(metas), 1)
        self.assertEqual(metas[0].id, s.meta.id)
        self.assertEqual(metas[0].turns, 1)
        self.assertEqual(self.store.latest(cwd=self.tmp).meta.id, s.meta.id)

    def test_latest_filters_by_cwd(self):
        self.sample_session()
        other = SessionStore(self.tmp)
        s2 = other.create(cwd=self.tmp / "elsewhere")
        s2.add_message(Message.user("other project"))
        s2.close()
        found = self.store.latest(cwd=self.tmp / "elsewhere")
        self.assertEqual(found.meta.id, s2.meta.id)

    def test_delete_and_prune(self):
        s = self.sample_session()
        for i in range(4):
            extra = self.store.create(cwd=self.tmp)
            extra.add_message(Message.user(f"m{i}"))
            extra.close()
        self.assertEqual(len(self.store.list()), 5)
        self.assertTrue(self.store.delete(s.meta.id))
        self.assertFalse(self.store.delete(s.meta.id))
        self.assertEqual(self.store.prune(keep=2), 2)
        self.assertEqual(len(self.store.list()), 2)


class TestCrashRecovery(SessionTestBase):
    def test_torn_last_line_is_skipped(self):
        s = self.sample_session()
        lines = s.path.read_text().splitlines()
        s.path.write_text("\n".join(lines[:-1]) + '\n{"t":"msg","role":"user","content":"torn')
        loaded = self.store.load(s.meta.id)
        self.assertIsNotNone(loaded)
        self.assertEqual(len(loaded.messages), 4)

    def test_totals_recovered_when_close_never_ran(self):
        s = self.sample_session(close=False)
        # simulate a hard crash: file handle dropped without the closing meta record
        s._fh.close()
        s._fh = None
        s._closed = True
        loaded = self.store.load(s.meta.id)
        self.assertEqual(loaded.meta.turns, 1)
        self.assertEqual(loaded.meta.input_tokens, 100)

    def test_garbage_file_does_not_raise(self):
        day = self.tmp / "2020-01-01"
        day.mkdir()
        (day / "broken.jsonl").write_text("\x00\x01not json\n{}\n[]\n")
        loaded = self.store.load("broken")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.messages, [])

    def test_unwritable_path_falls_back_to_memory(self):
        store = SessionStore(self.tmp / "nope" / "deep")
        s = store.create(cwd=self.tmp)
        s.add_message(Message.user("still works"))
        s.close()
        self.assertEqual(len(s.messages), 1)


class TestHistoryRewrite(SessionTestBase):
    def test_compaction_replaces_history(self):
        s = self.sample_session()
        s2 = self.store.load(s.meta.id)
        s2.replace_messages([Message.system("summary of everything"), Message.user("next?")])
        s2.close()
        s3 = self.store.load(s.meta.id)
        self.assertEqual(len(s3.messages), 2)
        self.assertTrue(s3.messages[0].text.startswith("summary"))


class TestSerialisation(SessionTestBase):
    def test_non_json_meta_is_dropped(self):
        msg = Message(role="assistant", content="x",
                      tool_calls=[ToolCall("i", "n", "{a:1}", extra={"thoughtSignature": "S"})],
                      meta={"thinking_blocks": [{"type": "thinking"}], "bad": object()})
        data = message_to_dict(msg)
        self.assertNotIn("bad", data.get("meta", {}))
        restored = message_from_dict(data)
        self.assertEqual(restored.tool_calls[0].extra, {"thoughtSignature": "S"})
        self.assertEqual(restored.meta["thinking_blocks"], [{"type": "thinking"}])

    def test_unknown_role_is_rejected(self):
        self.assertIsNone(message_from_dict({"role": "weird"}))
        self.assertIsNone(message_from_dict({"role": 5}))

    def test_block_content_roundtrip(self):
        msg = Message.user([TextBlock("a"), TextBlock("b")])
        restored = message_from_dict(message_to_dict(msg))
        self.assertEqual([b.text for b in restored.content], ["a", "b"])

    def test_empty_content_roundtrip(self):
        msg = Message(role="assistant", content="")
        self.assertEqual(message_from_dict(message_to_dict(msg)).content, "")

    def test_export_markdown(self):
        s = self.sample_session()
        md = self.store.export_markdown(s.meta.id)
        self.assertIn("# NEXUS session", md)
        self.assertIn("## Transcript", md)
        self.assertIn("## Files touched", md)
        self.assertIn("edit_file", md)
        self.assertEqual(self.store.export_markdown("does-not-exist"), "")


class TestInMemoryMode(SessionTestBase):
    def test_persist_disabled_writes_nothing(self):
        store = SessionStore(self.tmp / "mem", persist=False)
        s = store.create(cwd=self.tmp)
        s.add_message(Message.user("hi"))
        s.close()
        self.assertFalse((self.tmp / "mem").exists())
        self.assertEqual(len(s.messages), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
