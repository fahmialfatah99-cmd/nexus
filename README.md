# NEXUS — agentic CLI, zero dependencies

A terminal coding agent that runs on **pure Python standard library** — nothing to install,
nothing to break. 27 model providers, 21 tools, checkpointed edits, a permission engine,
context budgeting, MCP support, plugins, and a **multi-persona swarm mode**.

```
 ███▄    █  ███████ ▐██▌  █    █  ██████
 ██ ▀█   █  ██       ████▌  █    █ ▀▀▀▀██
 ██  ▀█  █  █████     ██▌   █    █  ▄██▀
 ██   ▀█▀█  ██        ███▌  ▐█  █▌  ▀██▄
 ██    ▀██  ███████   ██▌    ▐██▀   █████
```

| | |
|---|---|
| **Dependencies** | none — Python 3.9+ standard library only |
| **Providers** | 27 (OpenAI, Anthropic, Gemini, Groq, DeepSeek, OpenRouter, Qwen, 9Router, Ollama, …) |
| **Tools** | 21 (files, search, shell, git, web, planning, memory, sub-agents, swarm) |
| **Personas** | 14 specialists + your own JSON personas |
| **Swarm modes** | 9 (hive, pipeline, parallel, debate, council, review, build, debug, audit) |
| **Slash commands** | 48 |
| **Tests** | 554, all offline and deterministic (`nexus selftest`) |

---

## 1. Install

No build step. Clone or copy the directory, then run it:

```bash
cd nexus
./nexus                     # interactive session
```

Optional: put it on your `PATH`.

```bash
ln -s "$PWD/nexus" ~/.local/bin/nexus     # Linux/macOS
```

Or run it as a module:

```bash
python3 -m nexuscli.cli version
```

Verify the installation:

```bash
nexus doctor        # 15 checks: python, git, keys, model, tools, network…
nexus selftest      # runs the full 554-test suite
nexus demo          # offline guided tour — no API key needed
```

## 2. Add a model key

```bash
nexus auth login anthropic          # prompts securely, stored 0600 in ~/.nexus/auth.json
nexus auth login openai sk-...      # or pass it inline
nexus auth list
```

Environment variables work too and take precedence:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
export GEMINI_API_KEY=...
export GROQ_API_KEY=...
export DEEPSEEK_API_KEY=...
```

Local models and local gateways need no key at all:

```bash
nexus --provider ollama -m llama3.2
nexus --provider lmstudio
nexus --provider 9router -m kr/claude-sonnet-4.5   # local multi-provider gateway
```

## 3. Use it

```bash
nexus                                    # interactive REPL
nexus "why does this test fail?"         # REPL seeded with a first prompt
nexus -p "summarise this repo"           # one-shot, print, exit (script friendly)
git diff | nexus -p "review this"        # pipe stdin as context
nexus -p "list the TODOs" --json         # machine readable output
nexus run task.md                        # instructions from a file
nexus --read-only -p "explain main.py"   # cannot modify anything
```

Inside a session:

| input | effect |
|---|---|
| `text` | talk to the agent |
| `/help` | every command, grouped |
| `@src/main.py fix this` | attach a file (or directory) to the message |
| `!git status` | run a shell command directly |
| `Tab` | complete commands, arguments and `@paths` |
| `Ctrl+C` | cancel the running turn; twice exits |

## 3b. Menus you can click

Nothing has to be typed twice. `/model`, `/provider`, `/mode`, `/swarm-mode`,
`/cast`, `/agent`, `/sessions` and `/tools pick` open a real menu: arrows **or
mouse clicks**, type-to-filter, `Enter` to choose, `Esc` to cancel. A bare `/`
opens the whole command browser, and approval prompts are clickable buttons
(`[ yes ] [ view ] [ always allow ] [ always deny ] [ no ]`) instead of a y/n
question.

```
┌─ Approval mode ────────────────────────────────────────────┐
│ Approval mode                                              │
│ current: auto-edit                                         │
│   1. read-only  nothing may be modified                    │
│   2. suggest  ask before every change                      │
│ ❯ 3. auto-edit  file edits automatic, shell still asks     │
│   4. full-auto  everything automatic except dangerous ops  │
│   5. yolo  everything automatic, no exceptions             │
│ [ enter choose ]  [ esc cancel ]                           │
└────────────────────────────────────────────────────────────┘
```

Language follows your locale (`--lang id` / `--lang en`, or `ui.language` in the
config). When stdin is not a terminal the same menus degrade to a numbered text
list, so scripts and CI never hang; `--no-menu` forces that everywhere.
Details in `docs/MENU.md`.

## 4. Swarm mode

Multiple agents with distinct **personalities, duties, tools and temperatures** working one
objective — each in its own context window, coordinating through a shared blackboard and a
dependency-aware task board.

```bash
nexus swarm "add a JSON config loader with validation, tests and docs"
nexus swarm "make the parser safe" --swarm-mode hive --cast architect implementer tester
nexus debate "should sessions live in SQLite or JSONL?" --rounds 3
nexus agent reviewer "review the last commit"
nexus swarm "audit this repo before release" --swarm-mode audit
nexus swarm "big refactor" --plan-only          # plan the task graph, do not execute
```

Inside a session: `/swarm <objective>`, `/swarm-mode`, `/cast`, `/agents`, `/debate`, `/board`.

**How `hive` runs:** Atlas (orchestrator) inspects the repo and writes a task board with
dependencies → workers execute in parallel waves, capped by `swarm.max_parallel` → a reviewer
gate inspects each wave and opens `blocker:` / `should-fix:` tasks → Atlas integrates the
result. A crashing worker produces a failed run, never a crashed swarm.

The cast:

| persona | name | role |
|---|---|---|
| `main` | Nexus | primary engineering agent (solo sessions) |
| `orchestrator` | Atlas 🧭 | plans, delegates, verifies, integrates — never codes |
| `architect` | Vega 🏛️ | boundaries, interfaces, failure modes, trade-offs |
| `implementer` | Forge ⚒️ | production code that matches local conventions |
| `reviewer` | Sable 🔍 | line-by-line, hunts the 3am bug, severity-classified |
| `tester` | Probe 🧪 | writes tests that can fail, runs them, pastes real output |
| `debugger` | Trace 🐞 | reproduces first, root cause over symptom |
| `security` | Warden 🛡️ | injection, SSRF, path traversal, secrets — with attack scenarios |
| `docs` | Quill 📝 | documents what the code does, verified examples only |
| `researcher` | Scout 🧭 | evidence and sources, fact vs inference |
| `refactorer` | Chisel 🪓 | structure changes, behaviour proven identical |
| `critic` | Nemesis ⚔️ | strongest case against, cheapest failure mode |
| `devops` | Rigger 🛠️ | CI, dependencies, reproducible from a clean checkout |
| `data` | Ledger 📊 | types, nulls, encodings, idempotency |

Custom personas: drop a JSON file in `~/.nexus/personas/` — see `docs/SWARM.md`.

## 5. Safety model

Four layers, each independently tested:

1. **Approval modes** — `read-only`, `suggest`, `auto-edit` (default), `full-auto`, `yolo`.
   Set with `--mode`, `--full-auto`, `--yolo`, or `/mode`.
2. **Rules** — `deny` always wins, then `allow`, then `ask`. Syntax `tool:pattern`:
   `bash:git`, `write_file:src/**`, `web_fetch:*.github.com`. Answering `a` (always) at a
   prompt persists an allow rule into `.nexus/config.json`; `d` persists a deny rule.
3. **Command classification** — every shell command and git invocation is classified
   `read-only` / `mutating` / `elevated` / `dangerous`. `rm -rf /`, `sudo`, `git push --force`,
   `git reset --hard`, `curl … | sh`, `chmod 777`, `DROP TABLE`, `kubectl delete` and friends
   always require approval, in every mode except `yolo`.
4. **Checkpoints** — every mutating tool snapshots the target files *before* touching them.
   `/undo` restores them and deletes files the agent created. `git` runs through argv, never
   `shell=True`, so metacharacters cannot be injected.

Also: `web_fetch` refuses private/loopback/link-local/metadata addresses unless you pass
`--allow-private-network` (SSRF protection), non-UTF-8 files are never rewritten, CRLF and
trailing-newline conventions are preserved, and `multi_edit` is all-or-nothing.

## 6. Context management

Nothing is silently dropped:

- Token counts are estimated with a CJK-aware heuristic (or `tiktoken` if you have it).
- Before every request the history is measured against the model's window and reduced in a
  graded ladder: squeeze big tool outputs → prune old tool outputs → drop oldest turns (an
  assistant turn always takes its tool results with it, since providers reject orphans) →
  escalate into the protected tail.
- If it still does not fit, NEXUS **says so** and suggests `/compact` instead of truncating
  your request behind your back.
- `/compact` summarises the history into a structured handoff note (goal, decisions, state,
  open items, constraints).
- Unknown context windows are never guessed — budgeting is skipped and reported.

## 7. Configuration

Layered, later wins: `~/.nexus/config.json` → `<project>/.nexus/config.json` →
`<project>/.nexus/config.local.json` → `NEXUS_*` env vars → CLI flags.

```bash
nexus init                       # create .nexus/config.json + AGENTS.md + .gitignore entry
nexus config list                # effective settings
nexus config get swarm.max_parallel
nexus config set default_model anthropic:sonnet
nexus config set swarm.mode debate
nexus config set permissions.deny '["bash:rm -rf *"]'
nexus config edit                # $EDITOR
nexus config sources             # which files were merged
```

A useful starter config:

```json
{
  "default_model": "anthropic:claude-sonnet-4-5",
  "approval_mode": "auto-edit",
  "failover": ["groq:llama-3.3-70b-versatile", "openai:gpt-4o-mini"],
  "permissions": {
    "allow": ["read_file:*", "grep:*", "find_files:*", "list_dir:*", "git:status", "git:diff"],
    "deny": ["bash:rm -rf *", "write_file:.env*", "write_file:**/secrets/**"]
  },
  "swarm": { "mode": "hive", "max_parallel": 4, "max_rounds": 3,
             "model_specs": { "orchestrator": "anthropic:sonnet", "tester": "groq:llama-3.3-70b-versatile" } },
  "compaction": { "enabled": true, "threshold": 0.82 }
}
```

## 8. Extending

**Plugins** — a Python file in `~/.nexus/plugins/` or `.nexus/plugins/`:

```python
from nexuscli.tools.base import Tool, ToolContext, ToolResult

class DeployTool(Tool):
    name = "deploy"
    description = "Deploy to staging."
    parameters = {"type": "object", "properties": {"env": {"type": "string"}}, "required": ["env"]}
    category = "ops"

    def run(self, args, ctx: ToolContext) -> ToolResult:
        return ToolResult.ok(f"deployed to {args['env']}")

TOOLS = [DeployTool()]
PERSONAS = [{"key": "release", "name": "Rocket", "role": "the release engineer",
             "style": "Cautious.", "duties": "Ship safely."}]
```

A broken plugin is reported and skipped; it never takes the session down. `nexus plugins`.

**MCP servers** — any stdio MCP server becomes a set of `mcp__<server>__<tool>` tools:

```bash
nexus mcp add filesystem -- npx -y @modelcontextprotocol/server-filesystem /path/to/project
nexus mcp test filesystem
nexus mcp list
```

## 9. Documentation

| file | contents |
|---|---|
| `docs/SETUP.md` | step-by-step setup guide, from nothing to a working session |
| `docs/ARCHITECTURE.md` | layers, data flow, the invariants that keep it bug-free |
| `docs/COMMANDS.md` | every CLI flag and all 49 slash commands |
| `docs/SWARM.md` | personas, modes, blackboard protocol, custom personas, tuning |
| `docs/PROVIDERS.md` | all 27 providers, env vars, base URLs, aliases, failover |
| `docs/MENU.md` | interactive menus: keys, mouse, languages, how it works |

## 10. What it is not

Being straight about the limits matters more than a feature list:

- **No embeddings / semantic index.** Project awareness is deterministic: a real directory
  tree, detected stack, actual build/test commands, git state and your `AGENTS.md`. Nothing
  is inferred that was not read from a file.
- **The model catalogue is best-effort.** Context windows and prices for known models are
  public specs; unknown models report `0` (unknown) rather than an invented number, and
  budgeting is then skipped rather than guessed. `nexus models --refresh` pulls the live list.
- **Costs are estimates**, computed from token counts and catalogue prices.
- **Plugins and MCP servers run with your user privileges.** They are code, not data.
- **`yolo` mode approves everything.** Use it in disposable environments only.
- **Windows** works but is not where it was developed: colour falls back to 16 colours and
  `readline` completion may be unavailable (the prompt degrades to plain `input()`).
- **Tests run on Python 3.11**; the code targets 3.9+ (no third-party imports, no 3.10-only
  syntax), but only 3.11 is continuously verified here.

## 11. Development

```
nexus/
├── nexus                     # launcher (executable, no install needed)
├── nexuscli/
│   ├── cli.py                # argparse subcommands + exit codes
│   ├── app.py                # bootstrap, REPL, 49 slash commands
│   ├── core/                 # config, context, permissions, checkpoints,
│   │                         # session, router, taskboard, project, ignore,
│   │                         # jsonschema, usage, paths, logging, errors
│   ├── transport/http.py     # urllib transport + spec-correct SSE parser
│   ├── providers/            # base (normalised message model), openai_compat,
│   │                         # anthropic, gemini, mock, registry
│   ├── tools/                # base + builtin/{files,search,shell,git,web,
│   │                         # tasks,memory,agent}
│   ├── agents/               # runtime (the loop), persona, swarm, blackboard
│   ├── ui/                   # theme, markdown, diff, widgets, render, prompt
│   ├── mcp/client.py         # MCP over stdio (JSON-RPC 2.0)
│   └── plugins/loader.py
├── examples/                 # working plugin, persona, MCP server and config
├── tests/                    # 554 tests + run_tests.py + fixtures/
└── docs/
```

```bash
python3 tests/verify_examples.py       # prove every file in examples/ really works
python3 tests/run_tests.py             # everything
python3 tests/run_tests.py providers   # one area
python3 tests/run_tests.py -v          # verbose
nexus selftest                         # same suite through the CLI
```

Test areas: `static` (AST-level undefined-name guard), `providers`, `ignore`, `context`,
`config`, `session`, `project`, `tools`, `agent_loop`, `swarm`, `mcp`, `plugins`, `ui`, `cli`.
