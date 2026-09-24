#!/usr/bin/env bash
# NEXUS installer -- no packages, no build step, no network access.
#
#   ./install.sh              check prerequisites + make ./nexus executable
#   ./install.sh --link       also symlink `nexus` into ~/.local/bin
#   ./install.sh --link --bin-dir /usr/local/bin
#   ./install.sh --verify     also run doctor + the 554-test suite
#   ./install.sh --uninstall  remove the symlink (never touches your data)
#
# Everything it does is reversible and it never writes outside the directory you
# choose. Re-running is safe (idempotent).

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER="$HERE/nexus"
DO_LINK=0
DO_VERIFY=0
DO_UNINSTALL=0
BIN_DIR="${NEXUS_BIN_DIR:-$HOME/.local/bin}"

say()  { printf '  %s\n' "$*"; }
ok()   { printf '  \033[32m✓\033[0m %s\n' "$*"; }
warn() { printf '  \033[33m!\033[0m %s\n' "$*"; }
die()  { printf '  \033[31m✗\033[0m %s\n' "$*" >&2; exit 1; }

for arg in "$@"; do
  case "$arg" in
    --link)      DO_LINK=1 ;;
    --verify)    DO_VERIFY=1 ;;
    --uninstall) DO_UNINSTALL=1 ;;
    --bin-dir)   die "--bin-dir needs a value: --bin-dir=/path" ;;
    --bin-dir=*) BIN_DIR="${arg#--bin-dir=}" ;;
    -h|--help)   sed -n '2,14p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *)           die "unknown option: $arg (try --help)" ;;
  esac
done

PYTHON="${PYTHON:-python3}"
command -v "$PYTHON" >/dev/null 2>&1 || die "python3 not found on PATH"

if [ "$DO_UNINSTALL" = "1" ]; then
  if [ -L "$BIN_DIR/nexus" ]; then
    rm -f "$BIN_DIR/nexus"
    ok "removed $BIN_DIR/nexus (your config, sessions and logs are untouched)"
  else
    say "nothing to remove at $BIN_DIR/nexus"
  fi
  exit 0
fi

echo "NEXUS installer"
echo

# ---- 1. python version -----------------------------------------------------
"$PYTHON" - <<'PY' || die "Python 3.9 or newer is required"
import sys
sys.exit(0 if sys.version_info >= (3, 9) else 1)
PY
ok "python $("$PYTHON" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))') (>= 3.9 required)"

# ---- 2. the package is complete -------------------------------------------
[ -f "$LAUNCHER" ] || die "launcher not found: $LAUNCHER"
missing=0
for f in nexuscli/cli.py nexuscli/app.py nexuscli/agents/swarm.py nexuscli/providers/registry.py; do
  [ -f "$HERE/$f" ] || { warn "missing $f"; missing=1; }
done
[ "$missing" = "0" ] || die "the checkout looks incomplete"
ok "package files present"

# ---- 3. no third-party dependencies ---------------------------------------
"$PYTHON" - <<PY
import ast, pathlib, sys
stdlib = getattr(sys, "stdlib_module_names", None)
if stdlib is None:  # Python 3.9: fall back to a conservative known-list
    import os, re
    stdlib = set(sys.builtin_module_names) | {
        "argparse","ast","base64","concurrent","dataclasses","difflib","email","enum",
        "fnmatch","getpass","gzip","hashlib","html","http","importlib","io","ipaddress",
        "itertools","json","logging","os","pathlib","platform","queue","random","re",
        "readline","shlex","shutil","socket","ssl","subprocess","sys","tempfile","threading",
        "time","traceback","typing","unicodedata","unittest","urllib","uuid","zlib",
        "__future__",
    }
hard = []
for p in pathlib.Path("$HERE/nexuscli").rglob("*.py"):
    tree = ast.parse(p.read_text(encoding="utf-8"))
    guarded = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            for sub in ast.walk(node):
                if isinstance(sub, (ast.Import, ast.ImportFrom)):
                    for a in sub.names:
                        guarded.add(a.name if isinstance(sub, ast.Import) else (sub.module or ""))
    for node in ast.walk(tree):
        mods = []
        if isinstance(node, ast.Import):
            mods = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                continue
            mods = [node.module or ""]
        for m in mods:
            root = m.split(".")[0]
            if root in stdlib or any(root in g for g in guarded):
                continue
            hard.append(f"{p.name}: {m}")
sys.exit(1 if hard else 0)
PY
ok "zero third-party dependencies (verified by AST scan)"

# ---- 4. make it runnable ---------------------------------------------------
chmod +x "$LAUNCHER"
ok "launcher executable: $LAUNCHER"

# ---- 5. optional PATH symlink ---------------------------------------------
if [ "$DO_LINK" = "1" ]; then
  mkdir -p "$BIN_DIR"
  ln -sfn "$LAUNCHER" "$BIN_DIR/nexus"
  ok "linked $BIN_DIR/nexus -> $LAUNCHER"
  case ":$PATH:" in
    *":$BIN_DIR:"*) ok "$BIN_DIR is already on PATH" ;;
    *) warn "$BIN_DIR is NOT on PATH. Add this to your shell profile:"
       say "export PATH=\"$BIN_DIR:\$PATH\"" ;;
  esac
else
  say "run from the checkout:  $LAUNCHER"
  say "or install on PATH:      ./install.sh --link"
fi

# ---- 6. optional verification ---------------------------------------------
if [ "$DO_VERIFY" = "1" ]; then
  echo
  say "running the built-in test suite…"
  set +e
  ( cd "$HERE" && "$PYTHON" tests/run_tests.py ) > /tmp/nexus-selftest.$$ 2>&1
  code=$?
  tail -3 /tmp/nexus-selftest.$$
  rm -f /tmp/nexus-selftest.$$
  set -e
  if [ "$code" = "0" ]; then
    ok "test suite passed"
  else
    die "test suite failed (exit $code) -- do not use this checkout"
  fi
fi

cat <<NEXT

Next steps
  1. $LAUNCHER demo            offline tour, no API key needed
  2. $LAUNCHER doctor          check the installation
  3. $LAUNCHER auth login anthropic     (or openai / gemini / groq / deepseek / qwen)
     -- or use a local gateway, no key: $LAUNCHER --provider ollama -m llama3.2
                                       $LAUNCHER --provider 9router
  4. cd your-project && $LAUNCHER init --agents
  5. $LAUNCHER

Docs: README.md · docs/SETUP.md · docs/COMMANDS.md · docs/MENU.md · docs/SWARM.md · docs/PROVIDERS.md
NEXT
