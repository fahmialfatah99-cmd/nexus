"""i18n tests: key parity between languages, fallbacks, and autodetection.

A missing translation must degrade to English (then to the key), never crash a
menu -- and the two tables must not drift apart silently.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nexuscli.ui import i18n  # noqa: E402
from nexuscli.ui.i18n import EN, ID, autodetect, get_lang, set_lang, tr  # noqa: E402


class Localed(unittest.TestCase):
    """Restore the previous language after every test."""

    def setUp(self):
        self._previous = get_lang()

    def tearDown(self):
        set_lang(self._previous)


class TestTables(unittest.TestCase):
    def test_key_parity(self):
        """Every key must exist in both tables -- no half-translated menus."""
        self.assertEqual(set(EN), set(ID))

    def test_no_empty_values(self):
        for table in (EN, ID):
            for key, value in table.items():
                self.assertTrue(value.strip(), key)

    def test_placeholders_match(self):
        import re

        for key in EN:
            en = set(re.findall(r"{(\w+)}", EN[key]))
            idn = set(re.findall(r"{(\w+)}", ID[key]))
            self.assertEqual(en, idn, f"{key} placeholders differ: {en} vs {idn}")

    def test_supported_languages(self):
        self.assertEqual(set(i18n.SUPPORTED), {"en", "id"})


class TestTranslate(Localed):
    def test_english(self):
        set_lang("en")
        self.assertEqual(tr("button.ok"), EN["button.ok"])

    def test_indonesian(self):
        set_lang("id")
        self.assertEqual(tr("button.ok"), ID["button.ok"])

    def test_interpolation(self):
        set_lang("en")
        self.assertEqual(tr("menu.more_above", n=7), "▲ 7 more above")
        set_lang("id")
        self.assertEqual(tr("menu.more_below", n=3), "▼ 3 lagi di bawah")

    def test_unknown_key_returns_the_key(self):
        set_lang("en")
        self.assertEqual(tr("nope.not.a.key"), "nope.not.a.key")

    def test_missing_placeholder_does_not_raise(self):
        set_lang("en")
        self.assertIn("{mode}", tr("mode.current"))

    def test_extra_kwargs_are_ignored(self):
        set_lang("en")
        self.assertTrue(tr("button.ok", unused=1))

    def test_unknown_language_falls_back_to_english(self):
        set_lang("zz")
        self.assertEqual(get_lang(), "en")

    def test_legacy_indonesian_locale_code(self):
        """`in` is the old ISO code for Indonesian and still appears in the wild."""
        set_lang("in")
        self.assertEqual(get_lang(), "id")

    def test_language_is_case_and_space_insensitive(self):
        set_lang("  ID ")
        self.assertEqual(get_lang(), "id")


class TestAutodetect(Localed):
    def test_explicit_env_wins(self):
        self.assertEqual(autodetect({"NEXUS_LANG": "id", "LANG": "en_US.UTF-8"}), "id")

    def test_locale_with_codeset(self):
        self.assertEqual(autodetect({"LANG": "id_ID.UTF-8"}), "id")
        self.assertEqual(autodetect({"LC_ALL": "en_GB.UTF-8"}), "en")

    def test_locale_with_modifier(self):
        self.assertEqual(autodetect({"LANG": "id_ID@euro"}), "id")

    def test_legacy_code(self):
        self.assertEqual(autodetect({"LANG": "in_ID"}), "id")

    def test_unknown_locale_defaults_to_english(self):
        self.assertEqual(autodetect({"LANG": "fr_FR.UTF-8"}), "en")

    def test_empty_environment(self):
        self.assertEqual(autodetect({}), "en")

    def test_empty_values_are_skipped(self):
        self.assertEqual(autodetect({"NEXUS_LANG": "", "LANG": "", "LC_ALL": "id_ID"}), "id")


class TestUsedByMenus(Localed):
    """The menu renderer must actually follow the language."""

    def test_rendered_menu_switches_language(self):
        from nexuscli.ui.menu import MenuItem, MenuState
        from nexuscli.ui.theme import Style

        def draw():
            state = MenuState(title="T", items=[MenuItem("a"), MenuItem("b")], max_rows=2)
            state._style = Style.create("dark", enabled=False, width=60)
            return " ".join(state.rows(60))

        set_lang("en")
        english = draw()
        set_lang("id")
        indonesian = draw()
        self.assertNotEqual(english, indonesian)
        self.assertIn("choose", english)
        self.assertIn("pilih", indonesian)


if __name__ == "__main__":
    unittest.main(verbosity=2)
