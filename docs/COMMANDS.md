# Command reference

Everything below is generated from the running program, so it cannot drift from the code.

## CLI

```
nexus [prompt…]                    interactive session (optionally seeded with a first prompt)
nexus -p "prompt"                  one-shot: answer, print, exit
nexus run <file|->                 run instructions from a file (or stdin)
nexus swarm <objective>            run the multi-agent swarm
nexus debate <question>            structured debate with a judge
nexus agent <persona> <prompt>     run one persona once
nexus models                       list models (catalogue + live with --refresh)
nexus providers                    list providers and key status
nexus tools                        list tools and their JSON schemas
nexus personas                     list personas and swarm modes
nexus sessions | resume | export   session management
nexus config list|get|set|path|sources|init|edit
nexus auth list|login|logout
nexus mcp list|add|remove|test     MCP servers
nexus plugins                      installed plugins
nexus init                         create .nexus/ project configuration
nexus doctor                       15 installation checks
nexus selftest [filter]            run the built-in test suite
nexus demo [--swarm]               offline guided tour (mock provider, no key)
nexus version
```

### Exit codes

| code | meaning |
|---|---|
| 0 | success |
| 1 | the agent/task reported failure (errors, failed swarm runs, doctor found problems) |
| 2 | usage, configuration or provider error (bad flag, invalid config value, missing credentials) |
| 3 | unexpected internal error (always logged with a traceback) |
| 130 | interrupted (Ctrl+C) |

### Global flags

| flag | effect |
|---|---|
| `-m, --model SPEC` | `provider:model` or a bare model id / alias |
| `--provider KEY` | provider key (see `nexus providers`) |
| `--temperature F`, `--max-tokens N`, `--max-turns N` | sampling and loop limits |
| `--reasoning off\|low\|medium\|high` | reasoning effort where the model supports it |
| `--failover "a:m1,b:m2"` | provider failover chain |
| `--mode M` | `read-only`, `suggest`, `auto-edit`, `full-auto`, `yolo` |
| `--read-only`, `--auto-edit`, `--full-auto`, `--yolo` | shorthands for `--mode` |
| `--lang en\|id` | menu and dialog language (default: from your locale) |
| `--no-menu` | never open clickable menus; use the numbered text list |
| `--add-dir PATH` | allow tools to touch a directory outside the workspace (repeatable) |
| `--allow-private-network` | let `web_fetch` reach private/internal addresses (SSRF guard off) |
| `--no-tools` | chat only |
| `--color`, `--no-color`, `--theme T` | colour control (`dark`, `light`, `mono`) |
| `-q, --quiet` | minimal output for scripts |
| `--no-spinner`, `--no-markdown` | plainer output |
| `--json` | machine readable output (with `-p` and most subcommands) |
| `-C, --cwd DIR` | run in another directory |
| `-s, --session ID`, `-r, --resume ID\|last` | session control |
| `--offline` | disable network tools |
| `--no-plugins`, `--no-mcp` | skip extension loading |
| `--no-stream` | disable streaming |
| `-v, --verbose`, `--debug` | more output / debug logging to stderr |

### Environment variables

| variable | effect |
|---|---|
| `NEXUS_HOME` | config/auth/plugins/personas directory (default `~/.nexus`) |
| `NEXUS_MODEL`, `NEXUS_PROVIDER` | default model / provider |
| `NEXUS_MODE`, `NEXUS_APPROVAL_MODE` | approval mode |
| `NEXUS_TEMPERATURE`, `NEXUS_MAX_TOKENS`, `NEXUS_MAX_TURNS` | sampling and loop limits |
| `NEXUS_STREAM`, `NEXUS_OFFLINE`, `NEXUS_READ_ONLY`, `NEXUS_VERBOSE` | booleans |
| `NEXUS_THEME`, `NEXUS_COLOR` | appearance |
| `NEXUS_LANG` | menu/dialog language (`en`, `id`); falls back to `LC_ALL`/`LANG` |
| `NEXUS_NO_MENU` | `1` = never open clickable menus, use the numbered list |
| `NEXUS_NO_MOUSE` | `1` = keep mouse reporting off (keyboard-only menus) |
| `NEXUS_SWARM_MODE`, `NEXUS_SWARM_PARALLEL`, `NEXUS_SWARM_ROUNDS` | swarm defaults |
| `NEXUS_FAILOVER` | comma separated failover chain |
| `NEXUS_DEBUG`, `NEXUS_LOG_ECHO` | debug logging / mirror logs to stderr |
| `NO_COLOR`, `FORCE_COLOR`, `COLORTERM`, `TERM` | standard colour conventions |
| provider keys | `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `GROQ_API_KEY`, … |

## Input prefixes

| prefix | effect |
|---|---|
| `/` | slash command |
| `!` | run a shell command in the workspace |
| `@` | attach a file or directory to the message |
| anything else | talk to the agent |

Multi-line input: end a line with `\`, or paste a fenced block (an unterminated ``` keeps
collecting lines). `Tab` completes commands, their arguments and `@paths`.

## Slash commands

### Conversation

| command | arguments | what it does |
|---|---|---|
| `/again` |  | Re-run the last request |
| `/clear` `/reset` |  | Clear conversation history (keeps files & memory) |
| `/compact` | `[instructions]` | Summarise history to free context |
| `/diff` |  | Show file changes made in this session |
| `/exit` `/quit` `/q` |  | Quit NEXUS |
| `/export` | `[path]` | Export the transcript as markdown |
| `/help` `/?` `/h` | `[topic]` | Show this menu (or /help <topic>) |
| `/menu` `/m` | `[category]` | Open the clickable command browser |
| `/rewind` | `[n]` | Rewind conversation to an earlier turn |
| `/undo` | `[checkpoint]` | Revert file changes made by tools |

### Model

| command | arguments | what it does |
|---|---|---|
| `/failover` | `[specs]` | Show or set the provider failover chain |
| `/model` `/m` | `[provider:]model` | Show or switch model |
| `/models` | `[provider] [--refresh]` | List models for a provider |
| `/provider` | `<provider>` | Switch provider |
| `/providers` |  | List providers and their status |
| `/reasoning` | `<off|low|medium|high>` | Set reasoning effort |
| `/temperature` `/temp` | `<0.0-2.0>` | Set sampling temperature |

### Control

| command | arguments | what it does |
|---|---|---|
| `/allow` | `<tool:pattern>` | Add an allow rule |
| `/deny` | `<tool:pattern>` | Add a deny rule |
| `/mode` | `read-only|suggest|auto-edit|full-auto|yolo` | Show or set approval mode |
| `/plan` |  | Toggle plan mode (no file changes) |
| `/readonly` |  | Toggle read-only mode |
| `/rules` |  | Show permission rules and audit trail |
| `/tool` | `<on|off> <name>` | Enable/disable a tool |
| `/tools` |  | List tools |

### Context

| command | arguments | what it does |
|---|---|---|
| `/add` | `<path>` | Attach a file or directory to the context |
| `/context` |  | Show what is in context and its size |
| `/memory` |  | View persistent memory |
| `/project` |  | Rebuild repository context |
| `/remember` | `<text>` | Save a fact to project memory |
| `/status` `/st` |  | Full session status |
| `/usage` |  | Token and cost usage |

### Session

| command | arguments | what it does |
|---|---|---|
| `/resume` | `[id]` | Resume another session |
| `/sessions` |  | List saved sessions |

### Swarm

| command | arguments | what it does |
|---|---|---|
| `/agent` | `<persona> <prompt>` | Run a single persona once |
| `/agents` |  | List available personas |
| `/board` |  | Show the task board |
| `/cast` | `[persona...]` | Show or set the swarm cast |
| `/debate` | `<question>` | Run a structured debate |
| `/swarm` | `<objective>` | Run the swarm on an objective |
| `/swarm-mode` | `hive|pipeline|parallel|debate|council|review|build|debug|audit` | Show or set the swarm mode |

### System

| command | arguments | what it does |
|---|---|---|
| `/about` |  | Version and credits |
| `/auth` | `<provider> [key]` | Store an API key |
| `/config` | `[key] [value]` | Read or write configuration |
| `/doctor` |  | Diagnose the installation |
| `/keys` |  | Show keyboard shortcuts |
| `/log` | `[n]` | Show the log file location / tail it |
| `/mcp` |  | List MCP servers and their tools |
| `/selftest` |  | Run the built-in test suite |

## Keyboard

| keys | effect |
|---|---|
| `Tab` | complete a command, an argument value, or an `@path` |
| `↑` `↓` | history (persisted in `~/.nexus/history.txt`) |
| `Ctrl+C` | cancel the running turn; twice in a row exits |
| `Ctrl+D` | exit on an empty prompt |
| `\` at end of line | continue on the next line |

### Inside a menu

| keys | effect |
|---|---|
| `↑` `↓` `PgUp` `PgDn` `Home` `End` | move the cursor (or click an item) |
| `j` `k` | move the cursor, but only in menus where filtering is off |
| `Enter` | choose (or click `[ enter choose ]` / double-click a row) |
| `Esc`, `Ctrl+C` | cancel (or click `[ esc cancel ]`) |
| any letter | filter the list; `Backspace` widens, `Ctrl+U` clears |
| `1`-`9` | jump straight to that item (only while no filter is typed) |
| `Tab`, `Space` | mark an item in multi-select menus (`/cast`, `/tools pick`) |
| `Ctrl+A` | mark everything currently matching |
| `Ctrl+L` | repaint |
| mouse wheel | scroll the list |

Mouse clicks need a terminal with mouse reporting (iTerm2, GNOME Terminal, Kitty,
Alacritty, WezTerm, Windows Terminal, tmux with `set -g mouse on`). NEXUS enables it
only while a menu is open and always disables it again on exit -- including on
`Ctrl+C` and on errors. Set `NEXUS_NO_MOUSE=1` to keep it off entirely.

Commands that open a menu when called without arguments: `/menu` (and a bare `/`),
`/model`, `/provider`, `/mode`, `/swarm-mode`, `/cast`, `/agent`, `/sessions`,
`/tools pick`. Passing arguments still works exactly as before
(`/mode full-auto`), and when stdin is not a terminal every menu degrades to a
numbered text list so scripts and CI never hang. See `docs/MENU.md`.

## Tools

List them live with `nexus tools` (or `/tools` in a session). Summary:

| category | tools |
|---|---|
| files | `read_file`, `write_file`, `edit_file`, `multi_edit`, `list_dir`, `find_files`, `delete_path`, `file_info` |
| search | `grep` |
| shell | `bash`, `python_exec` |
| vcs | `git` |
| web | `web_fetch`, `web_search` |
| planning | `todo_write`, `todo_read` |
| memory | `memory` |
| agents | `task` (sub-agent in a fresh context) |
| swarm | `swarm_post`, `swarm_read`, `swarm_task` (registered during a swarm run) |

Every tool's arguments are described by a JSON Schema, validated **and coerced** before
execution (`{"limit": "20"}` becomes `20`), so a sloppy model gets a precise correction instead
of a crash. Read-only tools never prompt; writers go through the permission engine; a batch runs
in parallel only when every call is read-only and concurrency-safe.
