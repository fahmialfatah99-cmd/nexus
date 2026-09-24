"""Terminal input parser tests.

Covers every byte sequence a real terminal can send: printable characters
(including multi-byte UTF-8), control keys, all named keys plus legacy xterm
forms, the SGR and X10 mouse protocols with wheel and modifier bits, sequences
split across read boundaries byte by byte, the lone-ESC vs escape-sequence
ambiguity, and malformed input that must never wedge the parser.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nexuscli.ui.input import KEY_NAMES, InputParser, KeyEvent, MouseEvent  # noqa: E402


def ev(data: bytes, esc: bool = False):
    """Parse a whole buffer at once."""
    parser = InputParser()
    if esc:
        parser.mark_esc()
    return list(parser.feed(data))


def bytewise(data: bytes):
    """Parse one byte at a time -- the worst case for a stream parser."""
    parser = InputParser()
    out = []
    for i in range(len(data)):
        out.extend(parser.feed(data[i : i + 1]))
    return out


class TestPrintableAndControl(unittest.TestCase):
    def test_ascii_character(self):
        self.assertEqual(ev(b"a"), [KeyEvent("char", char="a")])

    def test_multibyte_utf8(self):
        self.assertEqual(ev("日本語".encode()),
                         [KeyEvent("char", "日"), KeyEvent("char", "本"), KeyEvent("char", "語")])

    def test_accented_character(self):
        self.assertEqual(ev("é".encode()), [KeyEvent("char", char="é")])

    def test_control_keys(self):
        cases = [(b"\r", "enter"), (b"\n", "enter"), (b"\x7f", "backspace"), (b"\x08", "backspace"),
                 (b"\t", "tab"), (b"\x03", "ctrl_c"), (b"\x04", "ctrl_d"), (b"\x0c", "ctrl_l"),
                 (b"\x15", "ctrl_u"), (b"\x01", "ctrl_a"), (b"\x05", "ctrl_e"), (b"\x17", "ctrl_w"),
                 (b"\x0b", "ctrl_k"), (b"\x1a", "ctrl_z"), (b"\x00", "ctrl_space")]
        for data, name in cases:
            with self.subTest(name=name):
                self.assertEqual(ev(data), [KeyEvent(name)])

    def test_batched_input(self):
        self.assertEqual(ev(b"ab\r"),
                         [KeyEvent("char", "a"), KeyEvent("char", "b"), KeyEvent("enter")])


class TestNamedKeys(unittest.TestCase):
    def test_every_named_key(self):
        for name, seq in KEY_NAMES.items():
            with self.subTest(key=name):
                self.assertEqual(ev(seq.encode()), [KeyEvent(name)])

    def test_legacy_xterm_forms(self):
        cases = [(b"\x1b[1~", "home"), (b"\x1b[4~", "end"), (b"\x1b[7~", "home"), (b"\x1b[8~", "end"),
                 (b"\x1b[11~", "f1"), (b"\x1b[12~", "f2"), (b"\x1b[13~", "f3"), (b"\x1b[14~", "f4")]
        for seq, name in cases:
            with self.subTest(name=name):
                self.assertEqual(ev(seq), [KeyEvent(name)])

    def test_two_arrows_in_one_buffer(self):
        self.assertEqual(ev(b"\x1b[A\x1b[B"), [KeyEvent("up"), KeyEvent("down")])


class TestIncrementalParsing(unittest.TestCase):
    def test_sgr_mouse_split_byte_by_byte(self):
        events = bytewise(b"\x1b[<0;12;4M")
        self.assertEqual(events, [MouseEvent("press", 12, 4, 0)])

    def test_utf8_split_byte_by_byte(self):
        events = bytewise("日本語".encode() + b"\x1b[B")
        self.assertEqual([e.char for e in events if getattr(e, "char", "")], ["日", "本", "語"])
        self.assertEqual(events[-1], KeyEvent("down"))

    def test_x10_mouse_split_byte_by_byte(self):
        events = bytewise(b"\x1b[M" + bytes([32, 35, 41]))
        self.assertEqual(len(events), 1)
        self.assertEqual((events[0].kind, events[0].x, events[0].y), ("press", 3, 9))

    def test_named_key_split_byte_by_byte(self):
        self.assertEqual(bytewise(KEY_NAMES["pagedown"].encode()), [KeyEvent("pagedown")])

    def test_partial_sequence_is_held_not_emitted(self):
        parser = InputParser()
        self.assertEqual(list(parser.feed(b"\x1b[")), [])
        self.assertEqual(list(parser.feed(b"A")), [KeyEvent("up")])


class TestMouseSGR(unittest.TestCase):
    def test_buttons_and_kinds(self):
        cases = [
            (b"\x1b[<0;5;7M", ("press", 5, 7, 0)),
            (b"\x1b[<0;5;7m", ("release", 5, 7, 0)),
            (b"\x1b[<1;5;7M", ("press", 5, 7, 1)),
            (b"\x1b[<2;5;7M", ("press", 5, 7, 2)),
            (b"\x1b[<32;5;7M", ("motion", 5, 7, 0)),
            (b"\x1b[<64;5;7M", ("wheel_up", 5, 7, 0)),
            (b"\x1b[<65;5;7M", ("wheel_down", 5, 7, 0)),
        ]
        for data, expect in cases:
            with self.subTest(data=data):
                m = ev(data)[0]
                self.assertEqual((m.kind, m.x, m.y, m.button), expect)

    def test_modifiers(self):
        self.assertTrue(ev(b"\x1b[<4;5;7M")[0].shift)
        self.assertTrue(ev(b"\x1b[<8;5;7M")[0].alt)
        self.assertTrue(ev(b"\x1b[<16;5;7M")[0].ctrl)
        self.assertFalse(ev(b"\x1b[<0;5;7M")[0].shift)

    def test_coordinates_are_never_below_one(self):
        m = ev(b"\x1b[<0;0;0M")[0]
        self.assertEqual((m.x, m.y), (1, 1))

    def test_large_coordinates(self):
        m = ev(b"\x1b[<0;400;200M")[0]
        self.assertEqual((m.x, m.y), (400, 200))


class TestMouseLegacyX10(unittest.TestCase):
    def test_press_and_release(self):
        press = ev(b"\x1b[M" + bytes([32 + 0, 32 + 3, 32 + 9]))[0]
        self.assertEqual((press.kind, press.x, press.y), ("press", 3, 9))
        release = ev(b"\x1b[M" + bytes([32 + 3, 32 + 3, 32 + 9]))[0]
        self.assertEqual(release.kind, "release")

    def test_wheel(self):
        self.assertEqual(ev(b"\x1b[M" + bytes([32 + 64, 33, 33]))[0].kind, "wheel_up")
        self.assertEqual(ev(b"\x1b[M" + bytes([32 + 65, 33, 33]))[0].kind, "wheel_down")

    def test_motion(self):
        self.assertEqual(ev(b"\x1b[M" + bytes([32 + 32, 33, 33]))[0].kind, "motion")


class TestEscapeAmbiguity(unittest.TestCase):
    def test_lone_esc_is_held_until_the_driver_decides(self):
        parser = InputParser()
        self.assertEqual(list(parser.feed(b"\x1b")), [])

    def test_idle_esc_becomes_a_key(self):
        parser = InputParser()
        list(parser.feed(b"\x1b"))
        parser.mark_esc()
        self.assertEqual(list(parser.feed(b"")), [KeyEvent("esc")])

    def test_esc_immediately_before_a_sequence_is_not_a_key(self):
        parser = InputParser()
        parser.mark_esc()
        self.assertEqual(list(parser.feed(b"\x1b[A")), [KeyEvent("up")])

    def test_esc_with_pending_flag_emits_once(self):
        self.assertEqual(ev(b"\x1b", esc=True), [KeyEvent("esc")])


class TestAltCombinations(unittest.TestCase):
    def test_alt_letter(self):
        self.assertEqual(ev(b"\x1bx"), [KeyEvent("alt", char="x")])

    def test_alt_multibyte(self):
        self.assertEqual(ev(b"\x1b" + "é".encode()), [KeyEvent("alt", char="é")])

    def test_alt_multibyte_split(self):
        self.assertEqual(bytewise(b"\x1b" + "é".encode()), [KeyEvent("alt", char="é")])


class TestUtf8Recovery(unittest.TestCase):
    """Invalid bytes must be dropped, never allowed to swallow the next key.

    A parser that waits forever for continuation bytes that will never arrive
    turns a stray 0xff into a menu that can no longer be confirmed.
    """

    def test_bogus_continuation_preserves_the_next_key(self):
        self.assertEqual(ev(b"\xe6\x97\r"), [KeyEvent("enter")])

    def test_four_byte_lead_with_bogus_continuation(self):
        self.assertEqual(ev(b"\xf0\x9f\r"), [KeyEvent("enter")])

    def test_invalid_lead_bytes_are_dropped(self):
        for bad in (b"\x80", b"\x81", b"\xc0", b"\xc1", b"\xf5", b"\xff"):
            with self.subTest(byte=bad):
                self.assertEqual(ev(bad + b"a"), [KeyEvent("char", "a")])

    def test_overlong_sequence_is_dropped(self):
        self.assertEqual(ev(b"\xc0\x80a"), [KeyEvent("char", "a")])

    def test_out_of_range_sequence_is_dropped(self):
        self.assertEqual(ev(b"\xf5\x80\x80\x80z"), [KeyEvent("char", "z")])

    def test_garbage_interleaved_with_real_keys(self):
        self.assertEqual(ev(b"\xff" + b"\x1b[A" + b"\xfe" + b"x" + b"\r"),
                         [KeyEvent("up"), KeyEvent("char", "x"), KeyEvent("enter")])

    def test_valid_multibyte_still_decodes(self):
        self.assertEqual(ev("é".encode()), [KeyEvent("char", "é")])
        self.assertEqual(ev("日本語".encode()),
                         [KeyEvent("char", "日"), KeyEvent("char", "本"), KeyEvent("char", "語")])

    def test_four_byte_emoji(self):
        self.assertEqual(ev("🎉".encode()), [KeyEvent("char", "🎉")])

    def test_emoji_split_byte_by_byte(self):
        self.assertEqual(bytewise("é🎉".encode()),
                         [KeyEvent("char", "é"), KeyEvent("char", "🎉")])

    def test_incomplete_but_valid_tail_is_held_then_completed(self):
        parser = InputParser()
        self.assertEqual(list(parser.feed(b"\xe6\x97")), [])
        self.assertEqual(list(parser.feed(b"\xa5")), [KeyEvent("char", "日")])


class TestMalformedInput(unittest.TestCase):
    def test_never_raises_and_recovers(self):
        for junk in (b"\x01\x02\x07", b"\x1b[999~", b"\x1b[<bad;dataM", b"\xff\xfe", b"\x1bOZ",
                     b"\x1b[M\x01\x02", b"\x1b[<0;1", b"\xe6\x97", b"\x1b\x1b\x1b", b""):
            with self.subTest(junk=junk):
                parser = InputParser()
                try:
                    list(parser.feed(junk))
                    list(parser.feed(b"a"))  # must still work afterwards
                except Exception as exc:
                    self.fail(f"{junk!r} raised {type(exc).__name__}: {exc}")

    def test_unknown_csi_final_byte_is_dropped(self):
        events = ev(b"\x1b[9Q" + b"x")
        self.assertEqual([e for e in events if getattr(e, "char", "")], [KeyEvent("char", "x")])


if __name__ == "__main__":
    unittest.main(verbosity=2)
