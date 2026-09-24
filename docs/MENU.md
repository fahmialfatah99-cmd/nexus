# Menu interaktif NEXUS

Semua pilihan di NEXUS bisa dipilih **tanpa mengetik**: pakai panah, klik mouse, atau
langsung ketik untuk menyaring. Halaman ini menjelaskan cara pakainya, cara kerjanya
di balik layar, dan apa yang terjadi kalau terminal Anda tidak mendukung salah satunya.

---

## 1. Ringkasan kontrol

| Aksi | Keyboard | Mouse |
|---|---|---|
| Pindah pilihan | `↑` `↓`, `PgUp` `PgDn`, `Home` `End`, dan `j` `k` *hanya di menu tanpa filter* | klik satu kali (memindahkan kursor) |
| Pilih | `Enter` | klik dua kali pada baris, atau klik `[ enter pilih ]` |
| Batal | `Esc` atau `Ctrl+C` | klik `[ esc batal ]` |
| Saring daftar | ketik saja (mis. `sonnet`) | — |
| Hapus saringan | `Backspace`, `Ctrl+U` (hapus semua) | — |
| Lompat ke item ke-n | angka `1`–`9` | klik barisnya |
| Tandai (mode multi) | `Tab` atau `Space` | klik barisnya |
| Pilih semua (multi) | `Ctrl+A` | klik `[ semua ]` |
| Kosongkan (multi) | — | klik `[ kosongkan ]` |
| Gambar ulang | `Ctrl+L` | — |

Catatan: saat **filter aktif**, huruf dipakai untuk menyaring — jadi `j`/`k` ikut
menyaring (bukan memindah), dan tombol "pilih semua/kosongkan" ditandai `[ klik: … ]`
sehingga memang hanya bisa diklik.
Kalau filter dimatikan (dialog ya/tidak), tombolnya jadi `[ a semua ]` / `[ n kosongkan ]`
dan huruf `a`/`n` berfungsi sebagai shortcut.

---

## 2. Command yang membuka menu

| Command | Menu yang muncul |
|---|---|
| `/` (sendirian) atau `/menu` | browser perintah: pilih kategori → pilih perintah → isi argumen → jalankan |
| `/menu <kategori>` | langsung ke perintah dalam kategori itu |
| `/model` | pilih model (provider yang bisa dipakai muncul paling atas, model aktif selalu ada) |
| `/provider` | pilih provider + status kuncinya |
| `/mode` | mode persetujuan (`read-only` … `yolo`) |
| `/swarm-mode` | mode swarm (hive, pipeline, debate, …) |
| `/cast` | **multi-select** persona untuk swarm |
| `/agent` | pilih persona → tulis tugasnya |
| `/sessions` | pilih sesi untuk dilanjutkan |
| `/tools pick` | papan saklar tool: tandai yang ingin dimatikan |
| prompt persetujuan | `[ ya ] [ lihat argumen ] [ selalu izinkan ] [ selalu tolak ] [ tidak ]` |

Command lama yang menerima argumen tetap jalan seperti biasa
(`/mode full-auto`, `/model openai:gpt-4o`) — menu hanya muncul kalau argumennya kosong
**dan** Anda sedang di terminal interaktif.

---

## 3. Bahasa menu

```bash
nexus --lang id        # bahasa Indonesia
nexus --lang en        # English (default)
```

Atau permanen di `~/.nexus/config.json`:

```json
{ "ui": { "language": "id" } }
```

Kalau tidak diset, NEXUS membaca `NEXUS_LANG`, lalu `LC_ALL` / `LANG`
(`id_ID.UTF-8` → Indonesia, `in_ID` juga dikenali). String mesin, log, dan pesan error
tetap bahasa Inggris — hanya menu dan dialog yang diterjemahkan.

---

## 4. Tanpa menu (skrip, CI, pipe)

Kalau stdin bukan terminal, menu otomatis berubah jadi daftar bernomor:

```
Choose a model
  type to filter, e.g. sonnet or groq
   1. gemini:gemini-2.0-flash  -- 1048k ctx
   2. anthropic:claude-sonnet-4-5  -- 200k ctx
Enter a number (empty = cancel):
```

Ini membuat `nexus -p "..." < /dev/null`, pipe, dan CI tetap berfungsi tanpa pernah
menggantung menunggu klik. Untuk memaksa perilaku ini walau di terminal:

```bash
nexus --no-menu          # atau: NEXUS_NO_MENU=1 nexus
```

EOF pada daftar bernomor dianggap "batal", jadi skrip yang menutup stdin tidak macet.

---

## 5. Klik mouse: apa yang diperlukan

Klik bekerja di terminal yang mendukung *mouse reporting*: iTerm2, GNOME Terminal,
Konsole, Kitty, Alacritty, WezTerm, Windows Terminal, tmux (dengan `set -g mouse on`).

NEXUS hanya menyalakan pelacakan **tombol** (bukan gerakan), dan selalu
mematikannya lagi saat menu tutup — termasuk kalau Anda menekan `Ctrl+C` atau
programnya error. Jadi seleksi-teks/copy di terminal Anda tidak rusak permanen.

Kalau klik tidak jalan tapi panah jalan, itu normal pada:

| Kondisi | Perilaku |
|---|---|
| `TERM=dumb` atau kosong | keyboard saja |
| Console Windows lama (bukan Windows Terminal) | keyboard saja |
| `NEXUS_NO_MOUSE=1` | keyboard saja (paksa) |
| stdin/stdout bukan tty (pipe, CI) | daftar bernomor |
| di dalam `screen`/`tmux` tanpa mouse | keyboard saja |

---

## 6. Cara kerjanya (untuk yang mau mengubah)

```
nexuscli/ui/input.py    byte stream -> KeyEvent / MouseEvent / ResizeEvent
nexuscli/ui/menu.py     MenuState (murni, tanpa IO) + Menu (driver terminal)
nexuscli/ui/i18n.py     tabel string en/id
```

**`MenuState` tidak menyentuh terminal sama sekali.** Ia menyimpan item, filter, kursor,
scroll, dan pilihan; lalu `layout(width)` menghasilkan daftar
`(jenis, payload, baris_tergambar)`. Dari satu sumber yang sama:

- `rows()` → baris untuk digambar,
- `hit_map()` → nomor baris layar → item mana,
- `button_columns()` → kolom tombol → tombol mana.

Karena ketiganya berasal dari satu fungsi, **koordinat klik tidak mungkin bergeser**
dari gambar. Ini alasan bug "klik kena baris yang salah" sulit muncul lagi.

**`Menu`** yang memegang terminal: raw mode, mouse reporting, redraw tanpa kedip
(naik N baris + clear, **bukan** alternate screen supaya scrollback Anda tetap utuh),
dan pemulihan terminal di `finally`.

Posisi menu di layar ditanyakan ke terminal lewat DSR (`ESC[6n`) sebelum gambar pertama,
lalu kalau menu tidak muat, digulir dulu secukupnya — supaya koordinat klik absolut
tetap cocok. Balasan DSR yang datang berbarengan dengan tombol user dikembalikan ke
parser (`RawTerminal.inject`), jadi tidak ada keystroke yang hilang.

Detail yang sudah ditangani dan diuji:
- escape sequence yang terpotong di tengah (diberi byte-per-byte di test),
- `Esc` sendirian vs `Esc[A` (diambiguasi dengan menunda satu putaran baca),
- byte UTF-8 rusak (`0xff`, continuation salah) dibuang **tanpa** menelan tombol berikutnya,
- protokol mouse SGR (`ESC[<b;x;yM`) dan X10 lama (`ESC[M` + 3 byte),
- lebar CJK dan ANSI supaya kotak tidak pernah miring,
- tombol diperpendek otomatis (`[ enter pilih ]` → `[enter]` → `[ok]`) saat terminal sempit,
- `Enter` saat filter tidak cocok **tidak** memilih apa pun,
- "kosongkan" lalu `Enter` tidak diam-diam memilih ulang item di bawah kursor.

---

## 7. Menjalankan testnya

```bash
python3 tests/run_tests.py menu             # state machine + fallback + i18n
python3 tests/run_tests.py menu_interactive # di pty asli: tombol, klik, wheel, kebersihan terminal
python3 tests/run_tests.py e2e_cli          # CLI sungguhan: /mode /model /cast /menu --lang --no-menu
python3 tests/run_tests.py input            # parser escape sequence & mouse
```

Semuanya offline dan deterministik; tidak butuh API key atau jaringan.

---

## 8. Menambah menu baru

```python
from nexuscli.ui.menu import Menu, MenuItem

menu = Menu(
    "Pilih environment",                       # judul
    [MenuItem("staging", value="stg", hint="aman"),
     MenuItem("production", value="prd", hint="hati-hati")],
    style=self.style,
    multi=False,                               # True = boleh pilih banyak
    allow_filter=True,                         # False = huruf jadi shortcut, bukan filter
    prompt="klik atau pakai panah",
)
choice = menu.show()      # None kalau dibatalkan; list kalau multi=True
if choice is None:
    return
```

Atau lewat helper di `App`:

```python
choice = self._pick("Pilih environment", items, multi=False, prompt="...")
```

`_pick` sudah menangani: daftar kosong, spinner yang harus dihentikan lebih dulu,
dan fallback ke daftar bernomor saat bukan terminal.
