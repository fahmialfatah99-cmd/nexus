"""Tiny i18n for the interactive UI.

Only user-facing menu/dialog strings go through here -- logs, errors from the
engine and documentation stay in English. Two languages ship by default
(``en``, ``id``) and adding one is a dict entry.

The default is ``en``; set ``ui.language`` in the config (or ``NEXUS_LANG``) to
``id`` for Indonesian. Unknown keys fall back to English, then to the key itself,
so a missing translation can never crash a menu.
"""

from __future__ import annotations

import os
from typing import Any, Dict

EN: Dict[str, str] = {
    "menu.select_one": "Select one",
    "menu.select_many": "Select one or more",
    "menu.commands": "Commands",
    "menu.all_commands": "All commands",
    "menu.filter_hint": "type to filter · arrows/click to move · enter to choose · esc to cancel",
    "menu.multi_hint": "tab/space marks · enter applies · ctrl+a all",
    "menu.pick_hint": "click or use the arrows · esc cancels",
    "menu.no_match": "(no match -- backspace to widen the filter)",
    "menu.more_above": "▲ {n} more above",
    "menu.more_below": "▼ {n} more below",
    "menu.no_options": "(no options available)",
    "menu.fallback_number": "Enter a number (empty = cancel):",
    "menu.fallback_numbers": "Enter numbers separated by spaces (empty = none):",
    "button.ok": "[ enter choose ]",
    "button.cancel": "[ esc cancel ]",
    "button.all_click": "[ click: all ]",
    "button.none_click": "[ click: none ]",
    "button.all_key": "[ a all ]",
    "button.none_key": "[ n none ]",
    "button.ok_short": "[enter]",
    "button.cancel_short": "[esc]",
    "button.all_short": "[all]",
    "button.none_short": "[none]",
    "button.ok_tiny": "[ok]",
    "button.cancel_tiny": "[x]",
    "button.all_tiny": "[+]",
    "button.none_tiny": "[-]",
    "approve.title": "APPROVAL REQUIRED",
    "approve.pick": "Choose an action",
    "approve.yes": "Yes, run it once",
    "approve.yes_hint": "this time only",
    "approve.always": "Always allow",
    "approve.always_hint": "remember: {key}",
    "approve.never": "Always deny",
    "approve.never_hint": "add a deny rule: {key}",
    "approve.no": "No",
    "approve.no_hint": "deny this time",
    "approve.view": "View arguments",
    "approve.view_hint": "show details, then ask again",
    "approve.risk": "risk: {risk}",
    "approve.too_many": "  too many attempts -- treated as denied",
    "approve.detail_title": "details",
    "model.pick": "Choose a model",
    "model.filter_hint": "type to filter, e.g. sonnet or groq",
    "provider.pick": "Choose a provider",
    "mode.pick": "Approval mode",
    "mode.current": "current: {mode}",
    "cast.pick": "Choose the swarm cast",
    "persona.pick": "Run a persona",
    "persona.task": "  task for {name}: ",
    "session.pick": "Choose a session to resume",
    "tools.pick": "Tools: mark the ones to DISABLE",
    "tools.applied": "{off} tool(s) disabled · {on} still enabled",
    "swarm_mode.pick": "Swarm mode",
    "mode.read-only": "nothing may be modified",
    "mode.suggest": "ask before every change",
    "mode.auto-edit": "file edits automatic, shell still asks (default)",
    "mode.full-auto": "everything automatic except dangerous operations",
    "mode.yolo": "everything automatic, no exceptions -- dangerous",
    "status.ready": "ready",
    "status.local": "local",
    "status.nokey": "no key",
    "info.no_options": "No options to choose from.",
    "info.cast_cleared": "Cast cleared; the mode default will be used.",
    "info.task_empty": "Empty task; cancelled.",
    "warn.unknown_category": "Unknown category '{name}'",
}

ID: Dict[str, str] = {
    "menu.select_one": "Pilih satu",
    "menu.select_many": "Pilih satu atau lebih",
    "menu.commands": "Perintah",
    "menu.all_commands": "Semua perintah",
    "menu.filter_hint": "ketik untuk memfilter · panah/klik untuk pindah · enter pilih · esc batal",
    "menu.multi_hint": "tab/spasi menandai · enter terapkan · ctrl+a semua",
    "menu.pick_hint": "klik atau pakai panah · esc membatalkan",
    "menu.no_match": "(tidak ada yang cocok -- backspace untuk memperluas)",
    "menu.more_above": "▲ {n} lagi di atas",
    "menu.more_below": "▼ {n} lagi di bawah",
    "menu.no_options": "(tidak ada pilihan)",
    "menu.fallback_number": "Masukkan nomor (kosong = batal):",
    "menu.fallback_numbers": "Masukkan nomor dipisah spasi (kosong = tidak ada):",
    "button.ok": "[ enter pilih ]",
    "button.cancel": "[ esc batal ]",
    "button.all_click": "[ klik: semua ]",
    "button.none_click": "[ klik: kosongkan ]",
    "button.all_key": "[ a semua ]",
    "button.none_key": "[ n kosongkan ]",
    "button.ok_short": "[enter]",
    "button.cancel_short": "[esc]",
    "button.all_short": "[semua]",
    "button.none_short": "[kosong]",
    "button.ok_tiny": "[ok]",
    "button.cancel_tiny": "[x]",
    "button.all_tiny": "[+]",
    "button.none_tiny": "[-]",
    "approve.title": "PERSETUJUAN DIPERLUKAN",
    "approve.pick": "Pilih tindakan",
    "approve.yes": "Ya, jalankan sekali",
    "approve.yes_hint": "hanya kali ini",
    "approve.always": "Selalu izinkan",
    "approve.always_hint": "ingat: {key}",
    "approve.never": "Selalu tolak",
    "approve.never_hint": "buat deny rule: {key}",
    "approve.no": "Tidak",
    "approve.no_hint": "tolak kali ini",
    "approve.view": "Lihat argumen",
    "approve.view_hint": "tampilkan detail, lalu tanya lagi",
    "approve.risk": "risiko: {risk}",
    "approve.too_many": "  terlalu banyak percobaan -- dianggap ditolak",
    "approve.detail_title": "detail",
    "model.pick": "Pilih model",
    "model.filter_hint": "ketik untuk memfilter, mis. sonnet atau groq",
    "provider.pick": "Pilih provider",
    "mode.pick": "Mode persetujuan",
    "mode.current": "aktif sekarang: {mode}",
    "cast.pick": "Pilih cast swarm",
    "persona.pick": "Jalankan persona",
    "persona.task": "  tugas untuk {name}: ",
    "session.pick": "Pilih sesi untuk dilanjutkan",
    "tools.pick": "Tool: tandai yang ingin DIMATIKAN",
    "tools.applied": "{off} tool dimatikan · {on} masih aktif",
    "swarm_mode.pick": "Mode swarm",
    "mode.read-only": "tidak boleh mengubah apa pun",
    "mode.suggest": "tanya untuk setiap perubahan",
    "mode.auto-edit": "edit file otomatis, shell tetap tanya (default)",
    "mode.full-auto": "semua otomatis kecuali operasi berbahaya",
    "mode.yolo": "semua otomatis tanpa kecuali -- berbahaya",
    "status.ready": "siap",
    "status.local": "lokal",
    "status.nokey": "tanpa key",
    "info.no_options": "Tidak ada pilihan.",
    "info.cast_cleared": "Cast dikosongkan; cast default mode akan dipakai.",
    "info.task_empty": "Tugas kosong; dibatalkan.",
    "warn.unknown_category": "Kategori '{name}' tidak dikenal",
}

TABLES: Dict[str, Dict[str, str]] = {"en": EN, "id": ID}
SUPPORTED = tuple(TABLES)

_lang = "en"


def set_lang(lang: str) -> str:
    """Set the UI language. Unknown values fall back to English."""
    global _lang
    key = (lang or "en").strip().lower()
    if key.startswith("in"):       # legacy locale code for Indonesian
        key = "id"
    _lang = key if key in TABLES else "en"
    return _lang


def get_lang() -> str:
    return _lang


def autodetect(environ: Dict[str, Any] | None = None) -> str:
    """Pick a language from NEXUS_LANG / LANG / LC_ALL."""
    env = environ if environ is not None else os.environ
    for var in ("NEXUS_LANG", "NEXUS_LANGUAGE", "LC_ALL", "LANG"):
        value = str(env.get(var) or "").strip().lower()
        if not value:
            continue
        head = value.split(".")[0].split("_")[0].split("-")[0]
        if head in TABLES:
            return head
        if head in ("in",):
            return "id"
    return "en"


def tr(key: str, **kwargs: Any) -> str:
    """Translate *key*, interpolating kwargs. Never raises."""
    table = TABLES.get(_lang, EN)
    text = table.get(key) or EN.get(key) or key
    if kwargs:
        try:
            return text.format(**kwargs)
        except (KeyError, IndexError, ValueError):
            return text
    return text


__all__ = ["tr", "set_lang", "get_lang", "autodetect", "SUPPORTED", "EN", "ID"]
