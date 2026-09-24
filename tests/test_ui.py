"""UI layer tests: layout, markdown, streaming, tables, diff, widgets, prompt."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nexuscli.ui import diff as diff_ui  # noqa: E402
from nexuscli.ui.markdown import (  # noqa: E402
    StreamingMarkdown,
    highlight,
    render_inline,
    render_markdown,
    render_table,
)
from nexuscli.ui.prompt import Prompt  # noqa: E402
from nexuscli.ui.theme import (  # noqa: E402
    Color,
    Style,
    char_width,
    detect_color,
    pad,
    strip_ansi,
    terminal_width,
    truncate,
    visible_width,
    wrap_text,
)
from nexuscli.ui.widgets import (  # noqa: E402
    Spinner,
    box,
    columns,
    format_bytes,
    format_cost,
    format_duration,
    format_number,
    kv,
    progress,
    rule,
    table,
)

WIDTHS = (24, 40, 80, 120)


def plain(theme: str = "dark", width: int = 80) -> Style:
    return Style.create(theme, enabled=False, width=width)


class TestLayout(unittest.TestCase):
    def test_visible_width(self):
        self.assertEqual(visible_width("abc"), 3)
        self.assertEqual(visible_width("你好"), 4)
        self.assertEqual(visible_width("\x1b[31mred\x1b[0m"), 3)
        self.assertEqual(visible_width("a\u0301"), 1, "combining marks are zero width")
        self.assertEqual(visible_width("日本語テキスト"), 14)

    def test_char_width(self):
        self.assertEqual(char_width("\t"), 1)
        self.assertEqual(char_width("\x1b"), 0)
        self.assertEqual(char_width("あ"), 2)
        self.assertEqual(char_width("a"), 1)

    def test_pad(self):
        self.assertEqual(pad("ab", 5), "ab   ")
        self.assertEqual(visible_width(pad("你好", 8)), 8)
        self.assertEqual(pad("ab", 5, "right"), "   ab")
        self.assertEqual(visible_width(pad("ab", 6, "center")), 6)
        self.assertEqual(pad("abcdef", 2), "abcdef", "never truncates")

    def test_truncate(self):
        self.assertEqual(visible_width(truncate("abcdefgh", 5)), 5)
        self.assertTrue(truncate("abcdefgh", 5).endswith("…"))
        self.assertEqual(truncate("abc", 10), "abc")
        self.assertEqual(visible_width(truncate("你好世界测试", 7)), 7)
        self.assertEqual(truncate("abc", 0), "")
        self.assertEqual(truncate("abc", 1), "…")

    def test_truncate_preserves_escapes(self):
        out = truncate("\x1b[31mred text here\x1b[0m", 8)
        self.assertLessEqual(visible_width(out), 8)
        self.assertTrue(strip_ansi(out).endswith("…"))
        self.assertIn("\x1b[31m", out)

    def test_wrap_never_exceeds_width(self):
        text = "The quick brown fox jumps over the lazy dog near the riverbank"
        for width in WIDTHS:
            lines = wrap_text(text, width)
            self.assertTrue(all(visible_width(l) <= width for l in lines), (width, lines))
            self.assertEqual("".join(strip_ansi(l) for l in lines).replace(" ", ""),
                             text.replace(" ", ""))

    def test_wrap_cjk_and_ansi(self):
        for width in (12, 24, 40):
            for text in ("你好世界这是中文换行测试需要正确宽度",
                         "\x1b[31mred words that need wrapping across lines\x1b[0m",
                         "supercalifragilisticexpialidocious"):
                lines = wrap_text(text, width)
                self.assertTrue(all(visible_width(l) <= width for l in lines), (width, text, lines))

    def test_wrap_multiline_and_empty(self):
        self.assertEqual(wrap_text("", 20), [""])
        self.assertGreaterEqual(len(wrap_text("short\n\nanother line here that is longer", 20)), 3)

    def test_terminal_width_is_sane(self):
        self.assertGreaterEqual(terminal_width(), 20)
        self.assertLessEqual(terminal_width(), 200)


class TestColorDetection(unittest.TestCase):
    class NotTTY:
        @staticmethod
        def isatty():
            return False

    class TTY:
        @staticmethod
        def isatty():
            return True

    def test_no_color_wins(self):
        self.assertEqual(detect_color(environ={"NO_COLOR": "1"}), "none")
        self.assertEqual(detect_color(environ={"NEXUS_NO_COLOR": "1"}), "none")
        self.assertEqual(detect_color(stream=self.TTY(), environ={"NO_COLOR": "1"}), "none")

    def test_non_tty_is_none(self):
        self.assertEqual(detect_color(stream=self.NotTTY(), environ={}), "none")

    def test_dumb_terminal(self):
        self.assertEqual(detect_color(environ={"TERM": "dumb"}), "none")

    def test_force_color(self):
        self.assertEqual(detect_color(stream=self.TTY(), environ={"FORCE_COLOR": "1", "TERM": "dumb"}), "16")

    def test_levels(self):
        self.assertEqual(detect_color(stream=self.TTY(),
                                      environ={"COLORTERM": "truecolor", "TERM": "xterm-256color"}), "truecolor")
        self.assertEqual(detect_color(stream=self.TTY(), environ={"TERM": "xterm-256color"}), "256")
        self.assertEqual(detect_color(stream=self.TTY(), environ={"TERM": "xterm"}), "16")

    def test_color_builders(self):
        self.assertEqual(Color("none").bold("x"), "x")
        self.assertEqual(Color("none").rgb(1, 2, 3, "y"), "y")
        self.assertEqual(Color("truecolor").rgb(255, 0, 0, "R"), "\x1b[38;2;255;0;0mR\x1b[0m")
        self.assertIn("38;5;", Color("256").rgb(255, 0, 0, "R"))
        # bright red is 91, never 39 (which means "default foreground")
        self.assertEqual(Color("16").rgb(255, 0, 0, "R"), "\x1b[91mR\x1b[0m")
        self.assertEqual(Color("16").rgb(0, 0, 0, "K"), "\x1b[30mK\x1b[0m")
        self.assertEqual(Color("16").rgb(255, 255, 255, "W"), "\x1b[97mW\x1b[0m")
        self.assertIn("\x1b[41m", Color("16").rgb(128, 0, 0, "r", bg=True))

    def test_every_16_palette_index_maps_to_a_real_code(self):
        from nexuscli.ui.theme import _ansi16

        for i in range(16):
            code = _ansi16(i)
            self.assertNotIn(code, ("39", "49"), f"index {i} must not map to default colour")
            self.assertIn(int(code), list(range(30, 38)) + list(range(40, 48))
                          + list(range(90, 98)) + list(range(100, 108)))

    def test_style_forcing(self):
        # environ={} on purpose: these tests must not depend on the ambient
        # environment (NO_COLOR is set when running through `nexus selftest`).
        clean: dict = {}
        self.assertTrue(Style.create("dark", enabled=True, environ=clean).color.enabled)
        self.assertFalse(Style.create("dark", enabled=False, environ=clean).color.enabled)
        self.assertFalse(Style.create("dark", enabled=True, environ={"NO_COLOR": "1"}).color.enabled,
                         "NO_COLOR must win over an explicit --color")
        self.assertFalse(Style.create("dark", enabled=True, environ={"NEXUS_NO_COLOR": "1"}).color.enabled)
        self.assertEqual(Style.create("dark", enabled=True, environ={"COLORTERM": "24bit"}).color.level,
                         "truecolor")
        self.assertEqual(Style.create("dark", enabled=None, environ=clean).color.level, "none",
                         "auto-detect with no TTY means no colour")

    def test_all_roles_render_single_width(self):
        style = Style.create("dark", enabled=True, width=80, environ={})
        for role in ("text", "dim", "accent", "accent2", "success", "warning", "error", "info",
                     "tool", "agent", "border", "code", "user"):
            self.assertEqual(visible_width(getattr(style, role)("X")), 1, role)
        self.assertEqual(visible_width(style.bold("X")), 1)
        self.assertEqual(visible_width(style.paint("no-such-role", "X")), 1)

    def test_palettes_cover_the_same_roles(self):
        from nexuscli.ui.theme import DARK, LIGHT, MONO

        self.assertEqual(set(DARK), set(LIGHT))
        self.assertEqual(set(DARK), set(MONO))


class TestMarkdown(unittest.TestCase):
    DOC = """# Title
Some **bold** and *italic* and `code` and [link](https://x.y).

- item one
- item two
  - nested
1. numbered

> quoted line

| col a | col b |
|-------|------:|
| 1     | 2.5  |

```python
def f(x):
    return x + 1  # comment
```

---
End.
"""

    def test_all_constructs_render(self):
        out = render_markdown(self.DOC, plain())
        text = strip_ansi(out)
        for frag in ("Title", "bold", "italic", "code", "link (https://x.y)", "item one", "nested",
                     "numbered", "quoted line", "col a", "2.5", "def f(x):", "End."):
            self.assertIn(frag, text, frag)

    def test_never_exceeds_width(self):
        for width in WIDTHS:
            out = render_markdown(self.DOC, plain(width=width), width=width)
            for line in out.split("\n"):
                self.assertLessEqual(visible_width(line), width, (width, line))

    def test_cjk_document(self):
        doc = "# 标题\n\n这是一段中文说明，需要正确的宽度计算才能正常换行显示。" * 3
        for width in (30, 40, 60):
            out = render_markdown(doc, plain(width=width), width=width)
            for line in out.split("\n"):
                self.assertLessEqual(visible_width(line), width)

    def test_inline_code_span_not_reformatted(self):
        self.assertIn("a**b", strip_ansi(render_inline("use `a**b` verbatim", plain())))

    def test_snake_case_is_not_italicised(self):
        self.assertEqual(strip_ansi(render_inline("call my_function_name now", plain())),
                         "call my_function_name now")

    def test_bold_and_link(self):
        self.assertIn("x", strip_ansi(render_inline("**x**", plain())))
        self.assertIn("label (http://y)", strip_ansi(render_inline("[label](http://y)", plain())))

    def test_bare_url_detected(self):
        self.assertIn("https://a.b", strip_ansi(render_inline("see https://a.b now", plain())))

    def test_highlight_is_deterministic_and_safe(self):
        for lang in ("python", "js", "bash", "", "unknown-lang"):
            out = highlight("def f(x): return 'str'  # comment", lang, plain())
            self.assertIn("def f", strip_ansi(out))

    def test_code_frame_geometry_is_consistent(self):
        doc = "```python\ndef f(x):\n    return x+1\n```\n"
        out = render_markdown(doc, plain(width=80), width=80)
        frame = [l for l in out.split("\n") if l.startswith(("┌", "│", "└"))]
        self.assertEqual({visible_width(l) for l in frame}, {80})

    def test_code_frame_with_cjk(self):
        out = render_markdown("```python\n变量 = 1\n```\n", plain(width=40), width=40)
        frame = [l for l in out.split("\n") if l.startswith(("┌", "│", "└"))]
        self.assertEqual({visible_width(l) for l in frame}, {40})

    def test_line_numbers_option(self):
        out = render_markdown("```\na\nbb\nccc\n```\n", plain(width=60), width=60, code_line_numbers=True)
        self.assertIn("1", out)
        self.assertIn("3", out)

    def test_unterminated_fence(self):
        out = render_markdown("```python\ndef f():\n", plain(width=60), width=60)
        self.assertIn("def f", out)

    def test_empty_document(self):
        self.assertEqual(render_markdown("", plain()).strip(), "")


class TestTables(unittest.TestCase):
    ROWS = [["name", "status"], ["alpha", "ok"], ["very long value here " * 3, "failed"]]

    def test_fits_every_width(self):
        for width in (20, 24, 30, 40, 80, 120):
            out = render_table(self.ROWS, plain(width=width), width)
            for line in out:
                self.assertLessEqual(visible_width(line), width, (width, line))

    def test_cjk_cells(self):
        rows = [["名称", "数值"], ["中文字段很长", "一二三四五"], ["x", "y"]]
        for width in (24, 40, 60):
            out = render_table(rows, plain(width=width), width)
            for line in out:
                self.assertLessEqual(visible_width(line), width)

    def test_alignment(self):
        out = render_table([["l", "c", "r"], ["1", "22222", "3"]], plain(width=60), 60,
                           ["left", "center", "right"])
        self.assertIn("┌", out[0])

    def test_inline_markdown_toggle(self):
        rows = [["pattern"], ["**/*.py"]]
        self.assertIn("*/*.py", strip_ansi("\n".join(render_table(rows, plain(), 60, inline=False))))
        self.assertNotIn("**", strip_ansi("\n".join(render_table(rows, plain(), 60, inline=True))))

    def test_empty_table(self):
        self.assertEqual(render_table([], plain(), 60), [])

    def test_ragged_rows_are_padded(self):
        out = render_table([["a", "b", "c"], ["1"]], plain(), 60)
        self.assertTrue(out)


class TestStreaming(unittest.TestCase):
    """Streaming must never lose content, leak markdown markers, or break frames."""

    DOCS = {
        "prose+structure": "Heading\n" + ("lorem ipsum dolor sit amet " * 8) + "\n- a\n- b\n"
                           "```py\nx=1\n```\nEnd.\n",
        "table": "| a | b |\n|---|---|\n| 1 | 2 |\n",
        "unterminated fence": "text\n```py\ncode line\n",
        "partial fence at end": "abc\n``",
        "inline spans": "see **bold** and `code` and [link](http://x) and ~~struck~~\n",
        "inline no newline": "trailing **bold** without newline",
        "nested": "**bold with `code` inside** and *em*\n",
        "cjk": "\u6807\u9898\n\u8fd9\u662f\u4e2d\u6587\u8bf4\u660e\u9700\u8981\u6362\u884c\n- \u9879\u76ee\n",
        "numbers": "1. first\n2. second\n1000 rows processed\n",
        "quote": "> quoted\n\npara\n",
        "empty": "",
        "only newlines": "\n\n\n",
        "hr": "text\n\n---\n\nmore\n",
        "h1": "# Big heading\n\nbody\n",
        "h3+list": "### Sub\n\n- **a** item\n- `b` item\n",
        # a closing fence with no trailing newline must not be rendered as content
        "closing fence, no newline": "Done.\n\n```report\nstatus: done\nconfidence: high\n```",
        "fence then text": "```py\nx=1\n```\n\nafter the fence\n",
    }
    LEAKS = ("**", "```", "~~", "[link](", "`code", "__")
    FRAME = ("\u250c", "\u2502", "\u2514", "\u251c")
    WIDTHS = (12, 24, 40, 80, 120)
    STEPS = (1, 2, 3, 5, 7, 13, 25, 200, 5000)

    @staticmethod
    def alnum(text: str) -> str:
        import re

        return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]", "", text)

    def test_streaming_matrix(self):
        cases = 0
        for name, doc in self.DOCS.items():
            for width in self.WIDTHS:
                style = plain(width=width)
                for step in self.STEPS:
                    sm = StreamingMarkdown(style, width=width)
                    out = "".join(sm.feed(doc[i:i + step]) for i in range(0, len(doc), step)) + sm.flush()
                    text = strip_ansi(out)
                    with self.subTest(doc=name, width=width, step=step):
                        self.assertEqual(self.alnum(text), self.alnum(doc), "content lost or duplicated")
                        for marker in self.LEAKS:
                            self.assertNotIn(marker, text, f"leaked {marker}")
                        for line in out.split("\n"):
                            if line.startswith(self.FRAME):
                                self.assertLessEqual(visible_width(line), width)
                    cases += 1
        self.assertGreater(cases, 500)

    def test_block_matrix(self):
        for name, doc in self.DOCS.items():
            for width in self.WIDTHS:
                style = plain(width=width)
                out = render_markdown(doc, style, width=width)
                text = strip_ansi(out)
                with self.subTest(doc=name, width=width):
                    self.assertEqual(self.alnum(text), self.alnum(doc))
                    for marker in self.LEAKS:
                        self.assertNotIn(marker, text, f"leaked {marker}")
                    for line in out.split("\n"):
                        self.assertLessEqual(visible_width(line), width, line)

    def test_prose_streams_incrementally(self):
        sm = StreamingMarkdown(plain(), width=80)
        chunks = [sm.feed(c) for c in "Hello world, this is streaming text."]
        self.assertGreater(len([c for c in chunks if c]), 5)
        full = "".join(chunks) + sm.flush()
        self.assertEqual(strip_ansi(full).strip(), "Hello world, this is streaming text.")

    def test_no_duplication_across_newlines(self):
        sm = StreamingMarkdown(plain(), width=80)
        text = "Line one is here.\nLine two follows.\n"
        out = "".join(sm.feed(text[i:i + 3]) for i in range(0, len(text), 3)) + sm.flush()
        self.assertEqual(strip_ansi(out).count("Line one"), 1)
        self.assertEqual(strip_ansi(out).count("Line two"), 1)

    def test_unclosed_inline_span_is_held_back(self):
        sm = StreamingMarkdown(plain(width=80), width=80)
        self.assertEqual(sm.feed("see **bol"), "", "half a bold span must not print literally")
        rest = sm.feed("d** text\n") + sm.flush()
        self.assertIn("bold", strip_ansi(rest))
        self.assertNotIn("**", strip_ansi(rest))

    def test_lone_delimiter_is_held_back(self):
        sm = StreamingMarkdown(plain(width=80), width=80)
        self.assertEqual(sm.feed("a *"), "", "a lone * may start ** so it must wait")
        out = sm.feed("*bold**\n") + sm.flush()
        self.assertIn("bold", strip_ansi(out))
        self.assertNotIn("*", strip_ansi(out))

    def test_closing_fence_is_not_rendered_as_content(self):
        text = "Done.\n\n```report\nstatus: done\n```"
        for chunk in (1, 3, 6, len(text)):
            sm = StreamingMarkdown(plain(width=100), width=100)
            out = "".join(sm.feed(text[i:i + chunk]) for i in range(0, len(text), chunk)) + sm.flush()
            rendered = strip_ansi(out)
            with self.subTest(chunk=chunk):
                self.assertNotIn("```", rendered)
                self.assertIn("status: done", rendered)
                self.assertEqual(rendered.count("\u250c"), 1)
                self.assertEqual(rendered.count("\u2514"), 1)

    def test_unclosed_code_span_is_held_back(self):
        sm = StreamingMarkdown(plain(width=80), width=80)
        self.assertEqual(sm.feed("run `nexus"), "")
        out = sm.feed(" demo`\n") + sm.flush()
        self.assertIn("nexus demo", strip_ansi(out))
        self.assertNotIn("`", strip_ansi(out))

    def test_unterminated_fence_and_table_at_flush(self):
        sm = StreamingMarkdown(plain(width=60), width=60)
        self.assertIn("code", strip_ansi(sm.feed("```\ncode") + sm.flush()))
        sm = StreamingMarkdown(plain(width=60), width=60)
        self.assertIn("a", strip_ansi(sm.feed("| a | b |") + sm.flush()))

    def test_plain_mode_passthrough(self):
        sm = StreamingMarkdown(plain(), plain=True)
        self.assertEqual(sm.feed("**x**"), "**x**")

    def test_empty_feed(self):
        sm = StreamingMarkdown(plain())
        self.assertEqual(sm.feed(""), "")
        self.assertEqual(sm.flush(), "")


class TestInlineNesting(unittest.TestCase):
    def test_code_inside_bold(self):
        out = strip_ansi(render_inline("**bold with `code` inside**", plain()))
        self.assertIn("bold with", out)
        self.assertIn("code", out)
        self.assertNotIn("`", out)
        self.assertNotIn("**", out)

    def test_bold_inside_code_is_literal(self):
        out = strip_ansi(render_inline("`code with **stars**`", plain()))
        self.assertIn("code with **stars**", out)
        self.assertNotIn("`", out)

    def test_link_and_url(self):
        self.assertIn("l (http://u)", strip_ansi(render_inline("[l](http://u)", plain())))
        self.assertIn("https://a.b", strip_ansi(render_inline("see https://a.b now", plain())))

    def test_snake_case_not_italicised(self):
        self.assertEqual(strip_ansi(render_inline("call my_function_name now", plain())),
                         "call my_function_name now")

    def test_strike(self):
        # strikethrough is drawn with a combining overstrike, so compare with it removed
        out = strip_ansi(render_inline("~~gone~~", plain()))
        self.assertEqual(out.replace("\u0336", ""), "gone")
        self.assertNotIn("~~", out)


class TestDiff(unittest.TestCase):
    def test_unified_and_summary(self):
        d = diff_ui.unified("a\nb\nc\n", "a\nB\nc\nd\n", "f.py")
        stats = diff_ui.summarise(d)
        self.assertEqual(stats["added"], 2)
        self.assertEqual(stats["removed"], 1)
        self.assertEqual(stats["files"], ["f.py"])

    def test_render_contains_changes(self):
        d = diff_ui.unified("a\nb\n", "a\nB\n", "f.py")
        out = strip_ansi(diff_ui.render_diff(d, plain(), width=80))
        self.assertIn("+B", out)
        self.assertIn("-b", out)

    def test_empty_diff(self):
        self.assertIn("no textual changes", diff_ui.render_diff("", plain()))

    def test_long_diff_is_capped(self):
        d = "\n".join(f"+line {i} " + "x" * 50 for i in range(500))
        out = diff_ui.render_diff(d, plain(width=60), width=60, max_lines=50)
        self.assertIn("more diff lines", strip_ansi(out))
        for line in out.split("\n"):
            self.assertLessEqual(visible_width(line), 60)

    def test_word_diff(self):
        self.assertIn("slow", strip_ansi(diff_ui.word_diff("the quick fox", "the slow fox", plain())))

    def test_format_summary(self):
        d = diff_ui.unified("a\n", "b\n", "f.py")
        self.assertIn("+1", strip_ansi(diff_ui.format_summary(d, plain())))


class TestWidgets(unittest.TestCase):
    def test_box_fits_and_wraps(self):
        for width in (30, 50, 80, 120):
            out = box("Title", ["line one", "long " * 30, "中文内容也需要正确处理宽度"],
                      plain(width=width), width=width)
            for line in out.split("\n"):
                self.assertLessEqual(visible_width(line), width)
            self.assertIn("Title", strip_ansi(out))

    def test_table_widget_does_not_interpret_markdown(self):
        out = table(["cmd"], [["**/*.py"]], plain())
        self.assertIn("**/*.py", strip_ansi(out))

    def test_kv_wraps_values(self):
        for width in (40, 60):
            out = kv([("model", "gpt-4o"), ("long", "x" * 200), ("中文", "值" * 40)],
                     plain(width=width), width=width)
            for line in out.split("\n"):
                self.assertLessEqual(visible_width(line), width + 2)
            self.assertIn("model", strip_ansi(out))

    def test_progress(self):
        self.assertIn("30%", strip_ansi(progress(3, 10, plain(), width=20, label="tasks")))
        self.assertIn("100%", strip_ansi(progress(5, 5, plain())))
        self.assertIn("0%", strip_ansi(progress(0, 10, plain())))

    def test_rule_and_columns(self):
        for width in (40, 80):
            self.assertLessEqual(visible_width(rule(plain(width=width), "section")), width)
        self.assertIn("section", strip_ansi(rule(plain(), "section")))
        self.assertIn("/help", strip_ansi(columns(["/help", "/model", "/clear"], plain(), width=60)))

    def test_formatters(self):
        self.assertEqual(format_duration(500), "500ms")
        self.assertEqual(format_duration(1500), "1.5s")
        self.assertEqual(format_duration(90_000), "1m30s")
        self.assertEqual(format_duration(3_700_000), "1h01m")
        self.assertEqual(format_bytes(2048), "2.0KB")
        self.assertEqual(format_bytes(500), "500B")
        self.assertEqual(format_cost(0.0012), "$0.0012")
        self.assertEqual(format_cost(0), "$0")
        self.assertEqual(format_cost(1.23456), "$1.235")
        self.assertEqual(format_number(1500), "1.5k")
        self.assertEqual(format_number(2_500_000), "2.50M")
        self.assertEqual(format_number(42), "42")

    def test_spinner_disabled_writes_no_frames(self):
        import io

        buf = io.StringIO()
        spinner = Spinner(plain(), "working", stream=buf, enabled=False)
        spinner.start()
        spinner.set_message("other")
        spinner.stop()
        self.assertEqual(buf.getvalue(), "", "a disabled spinner must not draw anything")
        with Spinner(plain(), "ctx", stream=buf, enabled=False):
            pass
        self.assertEqual(buf.getvalue(), "")

    def test_spinner_final_message_is_written_once(self):
        import io

        buf = io.StringIO()
        spinner = Spinner(plain(), "working", stream=buf, enabled=False)
        spinner.start()
        spinner.stop("done")
        self.assertEqual(buf.getvalue(), "done\n")


class TestPrompt(unittest.TestCase):
    def test_continuation_detection(self):
        self.assertTrue(Prompt._continues("text \\", []))
        self.assertFalse(Prompt._continues("text", []))
        self.assertTrue(Prompt._continues("```python", []))
        self.assertFalse(Prompt._continues("```", ["```python", "x = 1"]))

    def test_capabilities_reported(self):
        prompt = Prompt(plain())
        caps = prompt.capabilities
        self.assertEqual(set(caps), {"readline", "completion", "history"})
        for value in caps.values():
            self.assertIsInstance(value, bool)

    def test_file_completion(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(tmp, ignore_errors=True))
        (tmp / "alpha.py").write_text("x")
        (tmp / "beta.py").write_text("x")
        (tmp / "sub").mkdir()
        (tmp / "sub" / "gamma.py").write_text("x")
        prompt = Prompt(plain(), cwd=tmp)
        matches = prompt._complete_files("al", keep_prefix=False)
        self.assertEqual(matches, ["alpha.py"])
        matches = prompt._complete_files("@s", keep_prefix=True)
        self.assertIn("@sub/", matches)
        matches = prompt._complete_files("sub/", keep_prefix=False)
        self.assertIn("sub/gamma.py", matches)

    def test_file_completion_ignores_noise(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(tmp, ignore_errors=True))
        (tmp / "node_modules").mkdir()
        (tmp / ".git").mkdir()
        (tmp / "keep.py").write_text("x")
        prompt = Prompt(plain(), cwd=tmp)
        matches = prompt._complete_files("", keep_prefix=False)
        self.assertEqual(matches, [])
        matches = prompt._complete_files("k", keep_prefix=False)
        self.assertEqual(matches, ["keep.py"])

    def test_missing_directory_is_safe(self):
        prompt = Prompt(plain(), cwd=Path("/definitely/not/here"))
        self.assertEqual(prompt._complete_files("x", keep_prefix=False), [])

    def test_ask_and_yes_no_use_injected_input(self):
        import builtins
        import contextlib
        import io

        answers = iter(["maybe", "y"])
        prompt = Prompt(plain())
        original = builtins.input
        builtins.input = lambda *_a, **_k: next(answers)
        try:
            with contextlib.redirect_stdout(io.StringIO()) as captured:
                self.assertTrue(prompt.yes_no("continue?"))
            self.assertIn("choose one of", captured.getvalue(), "an invalid answer must be re-asked")
        finally:
            builtins.input = original

    def test_read_handles_eof(self):
        prompt = Prompt(plain())
        import builtins

        original = builtins.input

        def raise_eof(*_a, **_k):
            raise EOFError

        builtins.input = raise_eof
        try:
            if not prompt.capabilities["readline"]:
                self.assertIsNone(prompt.read("> "))
        finally:
            builtins.input = original


if __name__ == "__main__":
    unittest.main(verbosity=2)
