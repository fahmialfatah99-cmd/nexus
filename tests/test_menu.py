"""Interactive menu tests.

`MenuState` is pure, so the whole interaction model -- cursor, filtering,
selection, scrolling, rendering, and mouse hit-testing -- is verified here
without a terminal. The driver's non-TTY fallback is tested with an injected
stdin.
"""

from __future__ import annotations

import io
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nexuscli.ui.input import KeyEvent, MouseEvent  # noqa: E402
from nexuscli.ui.i18n import set_lang, tr  # noqa: E402
from nexuscli.ui.menu import (BUTTON_ALL, BUTTON_CANCEL, BUTTON_NONE, BUTTON_OK,  # noqa: E402
                              Menu, MenuItem, MenuState)
from nexuscli.ui.theme import Style, strip_ansi, visible_width  # noqa: E402

WIDTHS = (30, 46, 60, 80, 100)


def plain(width: int = 80) -> Style:
    return Style.create("dark", enabled=False, width=width)


def items(n: int = 6, prefix: str = "item") -> list:
    return [MenuItem(f"{prefix} {i}", value=f"{prefix}-{i}", hint=f"hint {i}") for i in range(1, n + 1)]


def state(**kw) -> MenuState:
    kw.setdefault("title", "Pick one")
    kw.setdefault("items", items())
    kw.setdefault("max_rows", 4)
    st = MenuState(**kw)
    st._style = plain()
    return st


def render(st: MenuState, width: int = 80) -> list:
    st._style = plain(width)
    return st.rows(width)


def text_lines(st: MenuState, width: int = 80) -> list:
    return [strip_ansi(l) for l in render(st, width)]


class TestConstruction(unittest.TestCase):
    def test_strings_become_items(self):
        st = MenuState(title="t", items=["a", "b"])
        self.assertEqual([i.label for i in st.items], ["a", "b"])
        self.assertEqual([i.value for i in st.items], ["a", "b"])

    def test_value_defaults_to_label(self):
        self.assertEqual(MenuItem("x").value, "x")

    def test_empty_menu(self):
        st = MenuState(title="t", items=[])
        self.assertIsNone(st.result())
        self.assertEqual(st.matches(), [])
        self.assertTrue(st.rows(60))  # must still render a frame

    def test_cursor_clamped_on_construction(self):
        st = MenuState(title="t", items=items(3), cursor=99)
        self.assertEqual(st.cursor, 2)


class TestCursor(unittest.TestCase):
    def test_move_down_and_up(self):
        st = state()
        self.assertEqual(st.cursor, 0)
        st.move(1)
        self.assertEqual(st.cursor, 1)
        st.move(-1)
        self.assertEqual(st.cursor, 0)

    def test_clamped_at_both_ends(self):
        st = state()
        st.move(-5)
        self.assertEqual(st.cursor, 0)
        st.move(99)
        self.assertEqual(st.cursor, len(st.items) - 1)

    def test_home_and_end(self):
        st = state()
        st.end()
        self.assertEqual(st.cursor, len(st.items) - 1)
        st.home()
        self.assertEqual(st.cursor, 0)

    def test_page_moves_by_viewport(self):
        st = state(max_rows=3)
        st.page(1)
        self.assertEqual(st.cursor, 2)

    def test_disabled_items_are_not_selectable(self):
        st = state(items=[MenuItem("a", enabled=False), MenuItem("b")])
        st.cursor = 0
        self.assertIsNone(st.step(KeyEvent("enter")))
        st.move(1)
        self.assertEqual(st.step(KeyEvent("enter")), "select")
        self.assertEqual(st.result(), "b")


class TestFiltering(unittest.TestCase):
    def test_filter_narrows_matches(self):
        st = state(items=[MenuItem("alpha"), MenuItem("beta"), MenuItem("alphabet")])
        st.type_char("a")
        st.type_char("l")
        self.assertEqual([st.items[i].label for i in st.matches()], ["alpha", "alphabet"])

    def test_multi_token_and_matching(self):
        st = state(items=[MenuItem("read file", hint="files"), MenuItem("write file"), MenuItem("grep search")])
        for ch in "file read":
            st.type_char(ch)
        self.assertEqual([st.items[i].label for i in st.matches()], ["read file"])

    def test_case_insensitive_and_searches_hints(self):
        st = state(items=[MenuItem("Small", hint="BIG thing")])
        for ch in "big":
            st.type_char(ch)
        self.assertEqual(len(st.matches()), 1)

    def test_backspace_widens(self):
        st = state(items=[MenuItem("alpha"), MenuItem("zulu")])
        st.type_char("z")
        self.assertEqual(len(st.matches()), 1)
        st.backspace()
        self.assertEqual(len(st.matches()), 2)

    def test_clear_filter(self):
        st = state()
        st.type_char("z")
        self.assertEqual(st.matches(), [])
        st.clear_filter()
        self.assertEqual(len(st.matches()), 6)

    def test_cursor_moves_into_matches(self):
        st = state(items=[MenuItem("aaa"), MenuItem("bbb"), MenuItem("abc")])
        st.type_char("b")
        self.assertEqual(st.items[st.cursor].label, "bbb")

    def test_no_match_renders_a_hint(self):
        st = state()
        st.type_char("zzzz")
        self.assertIn(tr("menu.no_match").strip("()"), " ".join(text_lines(st)))

    def test_filter_text_shown_in_header(self):
        st = state()
        st.type_char("it")
        self.assertIn("filter: it", " ".join(text_lines(st)))

    def test_control_u_clears(self):
        st = state()
        st.type_char("x")
        self.assertEqual(st.step(KeyEvent("ctrl_u")), None)
        self.assertEqual(st.filter_text, "")

    def test_filter_can_be_disabled(self):
        st = state(allow_filter=False)
        st.type_char("z")
        self.assertEqual(st.filter_text, "")
        self.assertEqual(len(st.matches()), 6)


class TestSelection(unittest.TestCase):
    def test_single_select_returns_cursor_value(self):
        st = state()
        st.move(2)
        self.assertEqual(st.step(KeyEvent("enter")), "select")
        self.assertEqual(st.result(), "item-3")

    def test_multi_toggle(self):
        st = state(multi=True)
        st.toggle()
        st.move(2)
        st.toggle()
        self.assertEqual(st.result(), ["item-1", "item-3"])

    def test_multi_toggle_off(self):
        st = state(multi=True)
        st.toggle()
        st.toggle()
        self.assertEqual(st.result(), [])

    def test_multi_select_all_respects_filter(self):
        st = state(items=[MenuItem("alpha one"), MenuItem("zulu two"), MenuItem("alpha three")], multi=True)
        for ch in "alpha":
            st.type_char(ch)
        st.select_all()
        self.assertEqual(sorted(st.result()), ["alpha one", "alpha three"])

    def test_multi_select_none(self):
        st = state(multi=True)
        st.select_all()
        st.select_none()
        self.assertEqual(st.result(), [])

    def test_tab_toggles_and_advances_in_multi(self):
        st = state(multi=True)
        st.step(KeyEvent("tab"))
        self.assertEqual(st.cursor, 1)
        self.assertEqual(st.selected, [0])

    def test_shift_tab_goes_back(self):
        st = state()
        st.move(2)
        st.step(KeyEvent("shift_tab"))
        self.assertEqual(st.cursor, 1)

    def test_enter_with_empty_multi_selection_picks_highlighted(self):
        st = state(multi=True)
        st.move(1)
        self.assertEqual(st.step(KeyEvent("enter")), "select")
        self.assertEqual(st.result(), ["item-2"])

    def test_enter_with_no_matches_selects_nothing(self):
        """Typing a filter that matches nothing then pressing Enter must not
        silently choose an unrelated item."""
        st = state()
        st.type_char("zzz")
        self.assertEqual(st.matches(), [])
        self.assertIsNone(st.step(KeyEvent("enter")))
        self.assertIsNone(st.result())

    def test_backspacing_to_a_real_match_then_entering_works(self):
        st = state()
        for ch in "zzz":
            st.type_char(ch)
        st.backspace()
        st.backspace()
        st.backspace()
        self.assertEqual(st.step(KeyEvent("enter")), "select")
        self.assertEqual(st.result(), "item-1")

    def test_ok_button_with_no_matches_does_not_select(self):
        st = state()
        st.type_char("zzz")
        row = [r for r, (k, _) in st.hit_map(80).items() if k == "buttons"][0]
        col = [c for b, c, _ in st.button_columns(80) if b == BUTTON_OK][0]
        self.assertEqual(st.mouse_click(row, col, 80), "select")
        self.assertIsNone(st.result(), "the click confirms, but there is nothing to confirm")

    def test_enter_after_explicit_clear_returns_empty(self):
        st = state(multi=True)
        st.select_all()
        self.assertEqual(len(st.selected), 6)
        st.select_none()
        self.assertEqual(st.step(KeyEvent("enter")), "select")
        self.assertEqual(st.result(), [], "an explicit clear must not be undone by Enter")

    def test_enter_without_any_selection_picks_the_highlighted_item(self):
        st = state(multi=True)
        st.move(2)
        self.assertEqual(st.step(KeyEvent("enter")), "select")
        self.assertEqual(st.result(), ["item-3"])

    def test_toggling_clears_the_explicit_flag(self):
        st = state(multi=True)
        st.select_none()
        st.toggle(1)
        self.assertFalse(st.explicit_clear)
        self.assertEqual(st.result(), ["item-2"])

    def test_disabled_items_cannot_be_selected(self):
        st = state(items=[MenuItem("a", enabled=False), MenuItem("b")], multi=True)
        st.toggle(0)
        self.assertEqual(st.selected, [])
        st.select_all()
        self.assertEqual(st.result(), ["b"])

    def test_result_order_is_stable(self):
        st = state(multi=True)
        st.toggle(4)
        st.toggle(1)
        self.assertEqual(st.result(), ["item-2", "item-5"])


class TestShortcuts(unittest.TestCase):
    def test_first_nine_visible_items_get_numbers(self):
        st = state(items=items(12), max_rows=12)
        st.assign_shortcuts()
        self.assertEqual([i.shortcut for i in st.items[:9]], [str(n) for n in range(1, 10)])
        self.assertEqual(st.items[9].shortcut, "")

    def test_digit_selects_in_single_mode(self):
        st = state(items=items(5))
        self.assertEqual(st.step(KeyEvent("char", char="3")), "select")
        self.assertEqual(st.result(), "item-3")

    def test_digit_toggles_in_multi_mode(self):
        st = state(items=items(5), multi=True)
        self.assertIsNone(st.step(KeyEvent("char", char="2")))
        self.assertEqual(st.result(), ["item-2"])

    def test_digit_types_when_a_filter_is_active(self):
        st = state(items=items(5))
        st.type_char("item")
        self.assertIsNone(st.step(KeyEvent("char", char="1")))
        self.assertEqual(st.filter_text, "item1")

    def test_shortcuts_follow_the_filter(self):
        st = state(items=[MenuItem("alpha"), MenuItem("beta"), MenuItem("alphabet")])
        st.type_char("al")
        st.assign_shortcuts()
        self.assertEqual(st.items[0].shortcut, "1")
        self.assertEqual(st.items[1].shortcut, "")
        self.assertEqual(st.items[2].shortcut, "2")


class TestScrolling(unittest.TestCase):
    def test_window_shows_max_rows(self):
        st = state(items=items(10), max_rows=4)
        st.end()
        lines = [l for l in text_lines(st) if l.startswith(("│", "┌", "└"))]
        body = [l for l in lines if "item" in l]
        self.assertLessEqual(len(body), 4)

    def test_indicators_appear(self):
        st = state(items=items(10), max_rows=3)
        st.move(1)
        joined = " ".join(text_lines(st))
        self.assertIn("▼", joined)
        st.end()
        joined = " ".join(text_lines(st))
        self.assertIn("▲", joined)

    def test_wheel_scrolls(self):
        st = state(items=items(20), max_rows=4)
        before = st.scroll
        st.step(MouseEvent(kind="wheel_down"))
        self.assertEqual(st.scroll, 0)  # step() ignores mouse; the driver routes it
        st.wheel(3)
        self.assertEqual(st.scroll, before + 3)
        st.wheel(-99)
        self.assertEqual(st.scroll, 0)

    def test_scroll_never_goes_negative_or_past_the_end(self):
        st = state(items=items(6), max_rows=4)
        st.wheel(-10)
        self.assertEqual(st.scroll, 0)
        st.wheel(100)
        self.assertLessEqual(st.scroll, 2)


class TestRendering(unittest.TestCase):
    def test_never_exceeds_width(self):
        long_items = [MenuItem("a very long label " * 4, hint="and a long hint " * 4) for _ in range(5)]
        for width in WIDTHS:
            st = state(items=long_items, max_rows=4)
            st._style = plain(width)
            for line in st.rows(width):
                self.assertLessEqual(visible_width(line), width, (width, line))

    def test_cjk_labels_do_not_break_alignment(self):
        cjk = [MenuItem(" Berkas konfigurasi", hint="ubah"), MenuItem("数据模型", hint="字段")]
        for width in WIDTHS:
            st = state(items=cjk, max_rows=4)
            st._style = plain(width)
            widths = {visible_width(l) for l in st.rows(width)}
            self.assertEqual(len(widths), 1, (width, widths))
            self.assertLessEqual(widths.pop(), width)

    def test_frame_is_consistent(self):
        st = state()
        lines = text_lines(st)
        self.assertTrue(lines[0].startswith("┌"))
        self.assertTrue(lines[-1].startswith("└"))
        self.assertEqual(len({len(l) for l in lines}), 1, "all rows must have equal width")

    def test_cursor_marker(self):
        st = state()
        st.move(1)
        lines = text_lines(st)
        marked = [l for l in lines if "❯" in l]
        self.assertEqual(len(marked), 1)
        self.assertIn("item 2", marked[0])

    def test_checkboxes_in_multi_mode(self):
        st = state(multi=True)
        st.toggle(1)
        joined = "\n".join(text_lines(st))
        self.assertIn("[x]", joined)
        self.assertIn("[ ]", joined)

    def test_hints_are_shown(self):
        st = state()
        self.assertIn("hint 1", " ".join(text_lines(st)))

    def test_button_labels_single_vs_multi(self):
        single = " ".join(text_lines(state()))
        self.assertIn(tr("button.ok"), single)
        self.assertIn(tr("button.cancel"), single)
        self.assertNotIn(tr("button.all_click"), single)
        multi = " ".join(text_lines(state(multi=True)))
        self.assertIn(tr("button.all_click"), multi,
                      "filter is on, so the button must be labelled click-only")
        no_filter = " ".join(text_lines(state(multi=True, allow_filter=False)))
        self.assertIn(tr("button.all_key"), no_filter)

    def test_colored_rendering_keeps_widths(self):
        style = Style.create("dark", enabled=True, width=60)
        st = state()
        st._style = style
        for line in st.rows(60):
            self.assertLessEqual(visible_width(line), 60)


class TestHitTesting(unittest.TestCase):
    def test_every_item_row_is_clickable(self):
        st = state(items=items(4), max_rows=4)
        mapping = st.hit_map(80)
        item_rows = [r for r, (kind, _) in mapping.items() if kind == "item"]
        self.assertEqual(len(item_rows), 4)
        for row in item_rows:
            kind, index = st.hit_test(row, 5, 80)
            self.assertEqual(kind, "item")
            self.assertIsInstance(index, int)

    def test_clicking_selects_on_second_click_single_mode(self):
        st = state(items=items(4), max_rows=4)
        mapping = st.hit_map(80)
        rows = sorted(r for r, (k, _) in mapping.items() if k == "item")
        first = st.mouse_click(rows[2], 5, 80)   # move the cursor
        self.assertIsNone(first)
        self.assertEqual(st.cursor, 2)
        second = st.mouse_click(rows[2], 5, 80)  # click again to choose
        self.assertEqual(second, "select")
        self.assertEqual(st.result(), "item-3")

    def test_clicking_toggles_in_multi_mode(self):
        st = state(items=items(4), max_rows=4, multi=True)
        rows = sorted(r for r, (k, _) in st.hit_map(80).items() if k == "item")
        st.mouse_click(rows[1], 5, 80)
        self.assertEqual(st.cursor, 1)
        st.mouse_click(rows[1], 5, 80)
        self.assertEqual(st.selected, [1])

    def test_buttons_are_clickable(self):
        st = state(items=items(3), max_rows=3)
        button_row = [r for r, (k, _) in st.hit_map(80).items() if k == "buttons"][0]
        cols = st.button_columns(80)
        ids = {bid for bid, _, _ in cols}
        self.assertEqual(ids, {BUTTON_OK, BUTTON_CANCEL})
        for bid, start, end in cols:
            self.assertEqual(st.hit_test(button_row, start, 80), ("button", bid))
            self.assertEqual(st.hit_test(button_row, end, 80), ("button", bid))
            mid = (start + end) // 2
            self.assertEqual(st.hit_test(button_row, mid, 80), ("button", bid))

    def test_ok_button_selects_and_cancel_cancels(self):
        st = state(items=items(3), max_rows=3)
        row = [r for r, (k, _) in st.hit_map(80).items() if k == "buttons"][0]
        ok_col = [c for b, c, _ in st.button_columns(80) if b == BUTTON_OK][0]
        cancel_col = [c for b, c, _ in st.button_columns(80) if b == BUTTON_CANCEL][0]
        self.assertEqual(st.mouse_click(row, ok_col, 80), "select")
        self.assertEqual(st.mouse_click(row, cancel_col, 80), "cancel")

    def test_multi_buttons_all_and_none(self):
        st = state(items=items(3), max_rows=3, multi=True)
        row = [r for r, (k, _) in st.hit_map(80).items() if k == "buttons"][0]
        cols = {b: c for b, c, _ in st.button_columns(80)}
        self.assertEqual(st.mouse_click(row, cols[BUTTON_ALL], 80), None)
        self.assertEqual(len(st.selected), 3)
        self.assertEqual(st.mouse_click(row, cols[BUTTON_NONE], 80), None)
        self.assertEqual(st.selected, [])

    def test_button_columns_do_not_overlap_and_fit(self):
        for width in WIDTHS:
            st = state(items=items(3), max_rows=3, multi=True)
            cols = st.button_columns(width)
            previous_end = 0
            for bid, start, end in cols:
                self.assertGreater(start, previous_end, (width, bid))
                self.assertLessEqual(end, width, (width, bid, end))
                previous_end = end

    def test_click_outside_the_menu_is_ignored(self):
        st = state(items=items(3), max_rows=3)
        self.assertIsNone(st.hit_test(999, 1, 80))
        self.assertIsNone(st.mouse_click(999, 1, 80))

    def test_hit_map_matches_rendered_rows(self):
        for width in WIDTHS:
            st = state(items=items(8), max_rows=4)
            st.move(3)
            lines = st.rows(width)
            mapping = st.hit_map(width)
            # rows are 1-based, so the highest mapped row is the bottom border
            self.assertEqual(max(mapping), len(lines), (width, max(mapping), len(lines)))
            for row, (kind, payload) in mapping.items():
                if kind == "item":
                    self.assertIn(st.items[payload].label.split()[0], strip_ansi(lines[row - 1]))


class TestKeyboardProtocol(unittest.TestCase):
    def test_esc_and_ctrl_c_cancel(self):
        for key in ("esc", "ctrl_c"):
            st = state()
            self.assertEqual(st.step(KeyEvent(key)), "cancel")

    def test_ctrl_l_requests_redraw(self):
        self.assertEqual(state().step(KeyEvent("ctrl_l")), "redraw")

    def test_ctrl_a_selects_all_in_multi(self):
        st = state(multi=True)
        st.step(KeyEvent("ctrl_a"))
        self.assertEqual(len(st.selected), 6)

    def test_arrows_move(self):
        st = state()
        st.step(KeyEvent("down"))
        st.step(KeyEvent("down"))
        self.assertEqual(st.cursor, 2)
        st.step(KeyEvent("up"))
        self.assertEqual(st.cursor, 1)

    def test_space_toggles_in_multi_without_filter(self):
        st = state(multi=True, allow_filter=False)
        st.step(KeyEvent("char", char=" "))
        self.assertEqual(st.selected, [0])
        self.assertEqual(st.cursor, 1)

    def test_a_and_n_buttons_when_filter_disabled(self):
        st = state(multi=True, allow_filter=False)
        st.step(KeyEvent("char", char="a"))
        self.assertEqual(len(st.selected), 6)
        st.step(KeyEvent("char", char="n"))
        self.assertEqual(st.selected, [])

    def test_vim_keys_move_only_when_filtering_is_off(self):
        st = state(allow_filter=False)
        st.step(KeyEvent("char", char="j"))
        self.assertEqual(st.cursor, 1)
        st.step(KeyEvent("char", char="k"))
        self.assertEqual(st.cursor, 0)
        st2 = state(allow_filter=True)
        st2.step(KeyEvent("char", char="j"))
        self.assertEqual(st2.cursor, 0, "with a filter on, j must type, not move")
        self.assertEqual(st2.filter_text, "j")

    def test_letters_type_when_filter_enabled(self):
        st = state(multi=True, allow_filter=True)
        st.step(KeyEvent("char", char="a"))
        self.assertEqual(st.filter_text, "a")
        self.assertEqual(st.selected, [])

    def test_mouse_events_do_not_select_via_step(self):
        st = state()
        self.assertIsNone(st.step(MouseEvent(kind="press", x=1, y=1)))
        st.step(MouseEvent(kind="wheel_down"))
        self.assertEqual(st.scroll, 0)


class TestLanguage(unittest.TestCase):
    """The same state machine must render correctly in every supported language."""

    def test_indonesian(self):
        set_lang("id")
        try:
            self.assertIn("pilih", " ".join(text_lines(state())))
            self.assertIn("tidak ada yang cocok",
                          " ".join(text_lines(state(filter_text="zzzz"))))
        finally:
            set_lang("en")

    def test_english(self):
        set_lang("en")
        self.assertIn("choose", " ".join(text_lines(state())))

    def test_widths_hold_in_both_languages(self):
        for lang in ("en", "id"):
            set_lang(lang)
            try:
                for width in WIDTHS:
                    st = state(items=items(9), max_rows=4, multi=True)
                    st._style = plain(width)
                    for line in st.rows(width):
                        self.assertLessEqual(visible_width(line), width, (lang, width, line))
            finally:
                set_lang("en")


class TestFallback(unittest.TestCase):
    """Non-TTY path: a numbered text menu so scripts and pipes keep working."""

    def run_menu(self, items_, answer: str, **kw):
        out = io.StringIO()
        menu = Menu("Choose", items_, style=plain(80), stream=out, **kw)
        import builtins

        original = builtins.input
        builtins.input = lambda *_a, **_k: answer
        try:
            result = menu.show()
        finally:
            builtins.input = original
        return result, out.getvalue()

    def test_single_choice(self):
        result, out = self.run_menu(["alpha", "beta", "gamma"], "2")
        self.assertEqual(result, "beta")
        self.assertIn("1. alpha", out)
        self.assertIn("3. gamma", out)

    def test_multi_choice(self):
        result, _ = self.run_menu(["a", "b", "c"], "1 3", multi=True)
        self.assertEqual(result, ["a", "c"])

    def test_multi_choice_with_commas(self):
        result, _ = self.run_menu(["a", "b", "c"], "2,3", multi=True)
        self.assertEqual(result, ["b", "c"])

    def test_empty_answer_cancels(self):
        result, _ = self.run_menu(["a", "b"], "")
        self.assertIsNone(result)
        result, _ = self.run_menu(["a", "b"], "", multi=True)
        self.assertEqual(result, [])

    def test_out_of_range_and_garbage_are_ignored(self):
        result, _ = self.run_menu(["a", "b"], "99 xyz")
        self.assertIsNone(result)

    def test_eof_cancels(self):
        out = io.StringIO()
        menu = Menu("Choose", ["a"], style=plain(80), stream=out)
        import builtins

        original = builtins.input

        def raise_eof(*_a, **_k):
            raise EOFError

        builtins.input = raise_eof
        try:
            self.assertIsNone(menu.show())
        finally:
            builtins.input = original

    def test_disabled_items_are_marked(self):
        _, out = self.run_menu([MenuItem("a"), MenuItem("b", enabled=False)], "1")
        self.assertIn("(disabled)", out)

    def test_hints_are_shown(self):
        _, out = self.run_menu([MenuItem("a", hint="the first")], "1")
        self.assertIn("the first", out)

    def test_empty_menu_returns_immediately(self):
        out = io.StringIO()
        menu = Menu("Choose", [], style=plain(80), stream=out)
        self.assertIsNone(menu.show())
        self.assertEqual(Menu("Choose", [], style=plain(80), stream=out, multi=True).show(), [])


class TestConfirm(unittest.TestCase):
    def test_confirm_default_no(self):
        from nexuscli.ui.menu import confirm

        out = io.StringIO()
        import builtins

        original = builtins.input
        builtins.input = lambda *_a, **_k: "2"
        try:
            self.assertFalse(confirm("Really?", "this changes files", style=plain(80), stream=out))
        finally:
            builtins.input = original
        self.assertIn("this changes files", out.getvalue())

    def test_confirm_yes(self):
        from nexuscli.ui.menu import confirm

        import builtins

        original = builtins.input
        builtins.input = lambda *_a, **_k: "1"
        try:
            self.assertTrue(confirm("Really?", style=plain(80), stream=io.StringIO()))
        finally:
            builtins.input = original


class TestSelectHelper(unittest.TestCase):
    def test_select_returns_value(self):
        import builtins

        original = builtins.input
        builtins.input = lambda *_a, **_k: "1"
        try:
            from nexuscli.ui.menu import select

            result = select("Model", [MenuItem("gpt-4o", "openai:gpt-4o"),
                                      MenuItem("sonnet", "anthropic:sonnet")],
                            style=plain(80), stream=io.StringIO())
        finally:
            builtins.input = original
        self.assertEqual(result, "openai:gpt-4o")


if __name__ == "__main__":
    unittest.main(verbosity=2)
