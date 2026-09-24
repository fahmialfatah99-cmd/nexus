# Panduan Setup NEXUS — dari nol sampai sesi pertama

Panduan ini berurutan dan setiap perintah di dalamnya sudah dijalankan sungguhan.
Ikuti dari atas; tiap bagian punya cara memverifikasi bahwa langkahnya berhasil.

---

## 0. Yang Anda butuhkan

| Kebutuhan | Wajib? | Cek |
|---|---|---|
| Python 3.9 atau lebih baru | **wajib** | `python3 --version` |
| `git` | disarankan (untuk `/diff`, konteks repo, tool `git`) | `git --version` |
| Koneksi internet | hanya untuk model cloud | — |
| API key **atau** model lokal | salah satu | lihat Bagian 3 |
| Paket Python pihak ketiga | **tidak ada** | tidak perlu `pip install` apa pun |

Kalau `python3 --version` di bawah 3.9, perbarui Python dulu — itu satu-satunya
prasyarat keras.

---

## 1. Ambil dan pasang

NEXUS adalah satu folder. Tidak ada build, tidak ada instalasi paket.

```bash
cd nexus            # folder hasil download/clone
./install.sh        # cek prasyarat + jadikan ./nexus executable
```

Output yang benar:

```
NEXUS installer

  ✓ python 3.11.2 (>= 3.9 required)
  ✓ package files present
  ✓ zero third-party dependencies (verified by AST scan)
  ✓ launcher executable: /path/to/nexus/nexus
```

### Supaya `nexus` bisa dipanggil dari mana saja

```bash
./install.sh --link                 # symlink ke ~/.local/bin
export PATH="$HOME/.local/bin:$PATH"   # tambahkan ke ~/.bashrc atau ~/.zshrc bila belum
nexus version
```

Pilihan lain:

```bash
./install.sh --link --bin-dir=/usr/local/bin   # butuh izin tulis di situ
./install.sh --verify                          # sekaligus menjalankan 554 test
./install.sh --uninstall                       # hapus symlink saja; data Anda utuh
```

### Tanpa installer sama sekali

```bash
chmod +x nexus && ./nexus version          # langsung dari folder
python3 -m nexuscli version                # atau sebagai modul
```

**Verifikasi bagian ini:** `nexus version` menampilkan `nexus 1.0.0` dan jumlah provider.

---

## 2. Verifikasi instalasi (3 perintah, tanpa API key)

```bash
nexus doctor      # 15 pemeriksaan: python, git, key, model, tools, jaringan…
nexus selftest    # 554 test bawaan, semuanya offline
nexus demo        # tur berpandu pakai provider mock — tidak butuh key
```

`doctor` yang sehat terlihat seperti ini (baris `providers ready` dan
`credentials stored` masih merah — itu normal sebelum Bagian 3):

```
  ✓ python >= 3.9        3.11.2
  ✓ workspace            /home/you/projects/myapp
  ✓ data dir             /home/you/.local/share/nexus
  ✓ log dir              /home/you/.local/share/nexus/logs
  ✓ config file          /home/you/.nexus/config.json
  ✓ git                  /usr/bin/git
  ✗ providers ready      none
  ✗ credentials stored   none
  ✓ default model        mock:mock-1
  ✓ tools                18 registered
  ✓ permission mode      auto-edit
  ✓ session writable     …/sessions/2026-09-22/….jsonl
  ✓ checkpoints          …/.nexus/checkpoints/…
  ✓ plugins              none
  ✓ network              ok (api.openai.com)
```

`nexus demo` menunjukkan agent membaca file, mengeditnya, checkpoint terbentuk,
lalu swarm berjalan (plan → workers paralel → reviewer gate → integrasi).
Semuanya nyata kecuali modelnya — provider mock bersifat deterministik.

> **Kalau `selftest` gagal:** jangan lanjut. Itu berarti checkout-nya rusak atau
> Python-nya terlalu tua. Lihat Bagian 13.

---

## 3. Pilih jalur model

Ada empat jalur. Pilih satu untuk mulai; sisanya bisa ditambah nanti.

### Opsi A — Cloud, kualitas terbaik (butuh key berbayar)

```bash
nexus auth login anthropic        # muncul prompt tersembunyi, tempel key Anda
# atau langsung:
nexus auth login anthropic sk-ant-...
```

Lalu set model default:

```bash
nexus config set default_provider anthropic
nexus config set default_model claude-sonnet-4-5
```

Provider cloud lain yang didukung: `openai`, `gemini`, `deepseek`, `mistral`,
`xai`, `together`, `fireworks`, `cerebras`, `perplexity`, `qwen`, `moonshot`,
`zhipu`, `siliconflow`, `huggingface`, `github`, `novita`, `chutes`, `sambanova`.

### Opsi B — Cloud murah / cepat

```bash
nexus auth login groq gsk-...
nexus config set default_provider groq
nexus config set default_model llama-3.3-70b-versatile
```

Groq sangat cepat untuk pekerjaan massal; cocok dipasang sebagai *fallback*
(lihat Bagian 10).

### Opsi C — Lokal, gratis, tanpa key

**Ollama** ([ollama.com](https://ollama.com)):

```bash
ollama pull llama3.2
nexus --provider ollama -m llama3.2
```

**LM Studio**: aktifkan server lokalnya (default port 1234), lalu

```bash
nexus --provider lmstudio
```

**vLLM / server OpenAI-compatible lain:**

```bash
nexus config set providers.custom.base_url "http://localhost:8000/v1"
nexus -m custom:nama-model
```

Model lokal kecil biasanya lemah dalam *tool calling*. Untuk agent yang benar-benar
memakai tools, gunakan model ≥ 8B yang mendukung function calling.

### Opsi D — Gateway lokal (9Router, satu pintu ke banyak provider)

[9Router](https://github.com/decolua/9router) menjalankan gateway
OpenAI-compatible di `http://localhost:20128/v1` dan merutekan ke 40+ provider
dengan auto-fallback.

```bash
# nyalakan 9Router dulu, atur provider/combo di dashboardnya
nexus models --provider 9router --refresh     # lihat model yang tersedia
nexus --provider 9router -m kr/claude-sonnet-4.5
```

Tidak perlu API key di sisi NEXUS (9Router yang memegang kredensial Anda);
NEXUS mengirim placeholder `Bearer local` karena 9Router mensyaratkan header
key non-empty. Port/host berbeda:

```bash
nexus config set providers.9router.base_url "http://127.0.0.1:20128/v1"
```

### Memakai environment variable (alternatif `auth login`)

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
export GEMINI_API_KEY=...
export GROQ_API_KEY=...
export DEEPSEEK_API_KEY=...
```

Urutan pencarian key: config `providers.<key>.api_key` → `api_key_env` →
environment variable provider → `~/.nexus/auth.json`.

**Verifikasi bagian ini:**

```bash
nexus auth list        # provider mana yang sudah "stored" / "env"
nexus providers        # tabel 27 provider + statusnya
nexus models --refresh # daftar model live dari provider Anda
```

`auth.json` disimpan dengan izin `0600` (hanya Anda yang bisa membaca).

---

## 4. Sesi pertama

```bash
cd proyek-anda
nexus
```

Anda akan melihat banner, model aktif, mode persetujuan, workspace, dan jumlah tool.

**Semua pengaturan bisa dipilih lewat menu -- tidak perlu dihafal.** Ketik `/` lalu
`Enter` (atau `/menu`) untuk membuka browser perintah; `/model`, `/provider`, `/mode`,
`/swarm-mode`, `/cast`, `/agent`, `/sessions`, dan `/tools pick` masing-masing membuka
menu sendiri. Navigasinya pakai panah **atau klik mouse**, ketik untuk menyaring,
`Enter` untuk memilih, `Esc` untuk batal. Prompt persetujuan juga berupa tombol yang
bisa diklik, bukan pertanyaan y/n.

Bahasa menu mengikuti locale Anda; paksa dengan `nexus --lang id` (atau
`"ui": {"language": "id"}` di config). Kalau stdin bukan terminal, menu otomatis jadi
daftar bernomor sehingga skrip dan CI tidak pernah menggantung -- `--no-menu` memaksa
perilaku itu di mana saja. Rinciannya di `docs/MENU.md`.

Lalu:

```
you› apa yang dilakukan proyek ini?
```

Agent akan memakai `list_dir`, `grep`, `read_file` sendiri — Anda melihat tiap tool
dipanggil dan hasilnya. Coba juga:

```
you› @src/main.py jelaskan fungsi utamanya
you› !git status
you› /status
you› /help
```

Tiga prefix input:

| prefix | arti |
|---|---|
| `/` | slash command |
| `!` | jalankan perintah shell di workspace |
| `@` | lampirkan file/folder ke pesan |
| selain itu | bicara ke agent |

Keluar dengan `/exit` (atau `Ctrl+D` pada prompt kosong).

**Verifikasi bagian ini:** `/status` menampilkan model yang benar dan `tools 18 enabled`.

---

## 5. Setup per proyek

Di dalam folder proyek:

```bash
nexus init --agents
```

Yang dibuat:

```
.nexus/config.json    mode persetujuan, allow/deny rules, pengaturan swarm
.gitignore            ditambah .nexus/checkpoints/
AGENTS.md             template panduan yang dibaca agent tiap sesi
```

Isi `AGENTS.md` dengan hal yang tidak bisa ditebak dari kode:

```markdown
## Commands
- test: `python3 tests/run_tests.py`
- lint: `ruff check .`

## Conventions
- docstring gaya Google, komentar berbahasa Inggris

## Do not touch
- migrations/ yang sudah diterapkan
- vendor/

## Definition of done
- test hijau, linter bersih, tidak ada warning baru
```

File ini di-inject ke system prompt otomatis, jadi agent tidak perlu menebak cara
menjalankan test Anda. NEXUS juga membaca `README.md`, `NEXUS.md`, `CLAUDE.md`,
dan `.cursorrules` bila ada.

**Verifikasi:** `/project` menampilkan stack terdeteksi, perintah build/test, status
git, dan isi `AGENTS.md`.

---

## 6. Mode persetujuan — pilih sesuai kenyamanan

| mode | baca file | tulis file | shell | operasi berbahaya |
|---|---|---|---|---|
| `read-only` | otomatis | **diblokir** | **diblokir** | diblokir |
| `suggest` | otomatis | tanya | tanya | tanya |
| `auto-edit` *(default)* | otomatis | otomatis | tanya | tanya |
| `full-auto` | otomatis | otomatis | otomatis | tanya |
| `yolo` | otomatis | otomatis | otomatis | **otomatis** |

```bash
nexus --read-only -p "jelaskan main.py"    # paling aman, untuk eksplorasi
nexus --auto-edit                          # default
nexus --full-auto "perbaiki semua test"    # untuk tugas panjang
nexus --yolo                               # hanya di lingkungan sekali-pakai
```

Di dalam sesi: `/mode`, `/readonly`, `/plan` (agent hanya merencanakan, tidak mengubah).

**Yang selalu ditanya, di mode apa pun kecuali `yolo`:** `rm -rf /`, `sudo`,
`git push --force`, `git reset --hard`, `git clean -f`, `curl … | sh`, `chmod 777`,
`mkfs`, `dd of=/dev/…`, `DROP TABLE`, `shutdown`, `kubectl delete`,
`docker system prune`, `npm publish`.

Saat prompt persetujuan muncul:

```
  [y]es  [n]o  [a]lways allow  [N]ever allow (deny rule)  [v]iew details
```

- `a` → menulis **allow rule** ke `.nexus/config.json` (tidak ditanya lagi)
- `d` → menulis **deny rule** (selalu ditolak)

Rule juga bisa ditulis manual: `/allow bash:git`, `/deny write_file:.env*`, `/rules`.
Sintaks `tool:pola`; `*` cocok apa saja termasuk `/`, `**` adalah glob path ketat.

---

## 7. Alur kerja sehari-hari

```
you› tambahkan validasi email di form registrasi, lengkap dengan test

  ⚙ grep  pattern=register
  ✓ 4 match(es) in 2 file(s)
  ⚙ read_file  path=src/forms.py
  ✓ 210 line(s)
  ⚙ edit_file  old_text=… new_text=…
  ✓ Edited src/forms.py
  ⚙ bash  command=python -m pytest tests/test_forms.py -q
  ✓ exit=0
```

Perintah yang paling sering dipakai:

| perintah | fungsi |
|---|---|
| `/undo` | kembalikan perubahan file (checkpoint) |
| `/diff` | lihat perubahan sesi ini |
| `/compact` | ringkas riwayat untuk membebaskan konteks |
| `/context` | berapa token yang terpakai dari budget |
| `/usage` | token + biaya per model dan per agent |
| `/status` | ringkasan lengkap sesi |
| `/add <path>` | lampirkan file/folder ke konteks |
| `/remember <fakta>` | simpan ke memori proyek permanen |
| `/model <spec>` | ganti model di tengah sesi |
| `/tools` · `/tool off bash` | lihat/matikan tool |
| `/sessions` · `/resume <id>` | daftar & lanjutkan sesi lama |
| `/export out.md` | ekspor transkrip ke markdown |
| `/plan` | mode perencanaan (tidak mengubah file) |

Untuk skrip dan CI:

```bash
nexus -p "ringkas perubahan di diff ini" --json          # output JSON
git diff | nexus -p "review" --read-only                  # pipe sebagai konteks
nexus run tugas.md --full-auto                            # instruksi dari file
nexus -p "jalankan test" --output-format json | jq .ok
```

---

## 8. Swarm (banyak agent, satu tujuan)

```bash
nexus swarm "tambah ekspor CSV dengan validasi dan test"
```

Yang terjadi: Atlas (orchestrator) memeriksa repo dan menyusun task board berdependensi
→ pekerja berjalan paralel per gelombang → reviewer gate memeriksa tiap gelombang dan
membuka task perbaikan → Atlas mengintegrasikan hasil akhir.

```bash
nexus personas                                     # 14 persona + peran masing-masing
nexus swarm "refactor parser" --swarm-mode pipeline
nexus debate "SQLite atau JSONL untuk session?" --rounds 3
nexus agent reviewer "review commit terakhir"
nexus swarm "audit sebelum rilis" --swarm-mode audit
nexus swarm "perubahan besar" --plan-only          # hanya rencananya
```

9 mode: `hive`, `pipeline`, `parallel`, `debate`, `council`, `review`, `build`,
`debug`, `audit`. Detail lengkap: `docs/SWARM.md`.

Persona tambahan cukup file JSON di `~/.nexus/personas/` (contoh siap pakai:
`examples/persona_dba.json`).

**Verifikasi:** `/usage` setelah swarm menampilkan rincian token per agent.

---

## 9. Sesi, undo, dan diff

Semua otomatis. Yang perlu Anda tahu:

- Tiap sesi disimpan sebagai JSONL di `~/.local/share/nexus/sessions/<tanggal>/`.
  Tahan crash: baris terakhir yang rusak dilewati, dan total token dipulihkan dari
  record `turn` bila proses mati sebelum menutup file.
- Tiap tool yang mengubah file membuat checkpoint **sebelum** menulis.
  `/undo` mengembalikan isi lama **dan** menghapus file yang baru dibuat agent.
- `/resume last` melanjutkan sesi terakhir di folder ini.

```bash
nexus sessions --all-dirs      # semua sesi
nexus resume <id>              # lanjutkan
nexus export <id> -o out.md    # ekspor transkrip
```

---

## 10. Konfigurasi

Urutan prioritas (yang terakhir menang):

```
~/.nexus/config.json                       (pribadi, lintas proyek)
<proyek>/.nexus/config.json                (tim, ikut di-commit)
<proyek>/.nexus/config.local.json          (pribadi, per proyek)
environment variable NEXUS_*
flag command line
```

```bash
nexus config list                       # semua pengaturan efektif
nexus config sources                    # file mana saja yang digabung
nexus config get swarm.max_parallel
nexus config set default_model anthropic:claude-sonnet-4-5
nexus config set swarm.max_parallel 6
nexus config set permissions.deny '["bash:rm -rf *","write_file:.env*"]'
nexus config edit                       # buka di $EDITOR
nexus config init                       # tulis config global dari default
```

Config awal yang berguna (contoh lengkap: `examples/config.json`):

```json
{
  "default_model": "anthropic:claude-sonnet-4-5",
  "approval_mode": "auto-edit",
  "failover": ["groq:llama-3.3-70b-versatile", "openai:gpt-4o-mini"],
  "permissions": {
    "allow": ["read_file:*", "grep:*", "find_files:*", "list_dir:*", "git:status", "git:diff"],
    "deny":  ["bash:rm -rf *", "write_file:.env*"]
  },
  "swarm": {
    "max_parallel": 4,
    "model_specs": { "orchestrator": "anthropic:sonnet", "tester": "groq:llama-3.3-70b-versatile" }
  }
}
```

`failover` penting: kalau provider utama kena rate limit, key-nya bermasalah, atau
**servernya mati** (misalnya 9Router belum dinyalakan), NEXUS pindah ke berikutnya
dan memberitahu Anda — bukan berhenti.

Di dalam sesi, `/config <key> <value>` mengubah sementara; nilai yang ditolak
(key salah, di luar rentang) **tidak** mengubah apa pun.

---

## 11. Ekstensi (opsional)

**Plugin** — satu file Python di `~/.nexus/plugins/` atau `<proyek>/.nexus/plugins/`:

```bash
cp examples/plugin_example.py ~/.nexus/plugins/
nexus plugins          # harus muncul: 1 plugin(s), 2 tool(s), 1 persona(s)
```

Plugin yang rusak dilaporkan dan dilewati — tidak menjatuhkan sesi.

**MCP server** — tool dari server MCP mana pun muncul sebagai `mcp__<server>__<tool>`:

```bash
nexus mcp add filesystem -- npx -y @modelcontextprotocol/server-filesystem .
nexus mcp test filesystem     # harus connect + menampilkan daftar tool
nexus mcp list
```

Contoh server MCP minimal untuk dipelajari: `examples/mcp_server.py`.

---

## 12. Lokasi data

| isi | lokasi |
|---|---|
| config global, auth, memori global, plugin, persona, history prompt | `~/.nexus/` |
| session, log | `~/.local/share/nexus/` (Linux/macOS) atau `%APPDATA%\nexus` (Windows) |
| config proyek, memori proyek, checkpoint | `<proyek>/.nexus/` |

Semua bisa dipindah dengan satu variabel:

```bash
export NEXUS_HOME=/path/lain        # config + auth + plugin + session + log jadi satu di sini
```

Backup yang layak: `~/.nexus/config.json`, `~/.nexus/auth.json` (rahasia!),
`~/.nexus/MEMORY.md`, dan `<proyek>/.nexus/`. Checkpoint boleh dibuang.

```bash
nexus log            # lokasi + 25 baris terakhir
nexus log 100        # lebih banyak
nexus log --path     # hanya pathnya
nexus log -f         # ikuti log secara live
```

Di dalam sesi, `/log 20` melakukan hal yang sama.

---

## 13. Troubleshooting

| gejala | penyebab | solusi |
|---|---|---|
| `python3: command not found` | Python belum terpasang | pasang Python ≥ 3.9 |
| `…requires an API key` | key belum ada | `nexus auth login <provider>` atau `export <VAR>` |
| `Authentication failed (401)` | key salah/kedaluwarsa | `nexus auth list`, lalu login ulang |
| `Rate limited (429)` | kuota provider habis | tunggu, atau set `failover` |
| `Connection refused … :20128` | 9Router belum jalan | nyalakan 9Router, atau ganti provider |
| `No models known for 'x'` | provider tidak merespons | `nexus models --provider x --refresh` |
| `Not a git repository` | folder bukan repo | `git init`, atau abaikan tool `git` |
| agent tidak memakai tool | model lemah untuk function calling | ganti model; cek `nexus models --refresh` |
| `Permission denied: …` saat non-interaktif | tidak ada prompt untuk approve | pakai `--full-auto` atau tambah allow rule |
| output tanpa warna padahal di terminal | `NO_COLOR` aktif / `TERM=dumb` | `unset NO_COLOR` atau pakai `--color` |
| completion `Tab` tidak jalan | `readline` platform tidak lengkap | normal — prompt otomatis turun ke `input()` biasa |
| `ContextOverflowError` | riwayat terlalu panjang | `/compact`, `/clear`, atau model ber-context besar |
| perilaku aneh | perlu diagnostik | `nexus doctor`, lalu `nexus log 100` |

Untuk bug yang tidak terjelaskan, jalankan dengan debug dan sertakan lognya:

```bash
nexus --debug -p "..."      # level DEBUG + dicetak ke stderr
cat ~/.local/share/nexus/logs/nexus.log | tail -50
```

---

## 14. Update dan uninstall

```bash
# update: ganti foldernya (tidak ada state di dalam folder)
rm -rf nexus && <ekstrak versi baru> && ./install.sh --link

# uninstall: hapus symlink; data Anda tidak disentuh
./install.sh --uninstall
rm -rf ~/.nexus ~/.local/share/nexus      # hapus data bila memang mau bersih total
```

Config Anda kompatibel ke depan: key yang tidak dikenal versi baru hanya
memunculkan warning, tidak membuat gagal start.

---

## Checklist 5 menit

```bash
./install.sh                                   # 1
nexus doctor                                   # 2
nexus demo                                     # 3  (opsional, tanpa key)
nexus auth login anthropic                     # 4
nexus config set default_model claude-sonnet-4-5   # 5
cd proyek-anda && nexus init --agents          # 6
# isi AGENTS.md                                # 7
nexus                                          # 8  mulai bekerja
```

Setelah itu: `/help` untuk semua perintah, `/help swarm` untuk panduan swarm,
`docs/COMMANDS.md` untuk referensi lengkap, `docs/PROVIDERS.md` untuk daftar
provider dan failover, `docs/ARCHITECTURE.md` untuk cara kerjanya di dalam.
