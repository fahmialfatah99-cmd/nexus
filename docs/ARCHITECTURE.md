# Architecture

## Design goals

1. **Zero dependencies.** Only the Python standard library. This removes an entire class of
   failures (resolver conflicts, ABI mismatches, half-installed extras) and makes the tool
   runnable anywhere Python 3.9+ is.
2. **One internal message model.** Every provider is an adapter to a single normalised format.
   Provider quirks are handled once, in one file each, instead of leaking into the agent loop.
3. **Nothing silent.** Unknown models report `unknown`, not a guess. An unaffordable
   conversation raises instead of truncating your request. A failing worker becomes a failed
   run, not a crashed swarm. A broken plugin is reported and skipped.
4. **Testable without a network.** Every layer accepts an injected transport, provider, UI and
   confirmer, so the whole stack — including streaming tool calls and swarm scheduling — is
   covered by deterministic offline tests.

## Layers

```
             ┌──────────────────────────────────────────────┐
   entry     │ cli.py  (argparse subcommands, exit codes)   │
             │ app.py  (bootstrap, REPL, 49 slash commands) │
             └───────────────────┬──────────────────────────┘
                                 │
             ┌───────────────────▼──────────────────────────┐
   agents    │ runtime.py   Agent loop (the turn state machine)
             │ swarm.py     Orchestration: hive/pipeline/…  │
             │ persona.py   Personalities as data           │
             │ blackboard.py  Shared knowledge, thread-safe │
             └───────────────────┬──────────────────────────┘
                                 │
             ┌───────────────────▼──────────────────────────┐
   services  │ router      provider+model resolution, failover
             │ context     token estimation, budget ladder  │
             │ permissions modes, rules, audit, confirmer   │
             │ checkpoints snapshot/restore/undo            │
             │ session     JSONL persistence, crash-safe    │
             │ project     repo context block               │
             │ taskboard   dependency graph, thread-safe    │
             │ config      layered settings, validation     │
             └───────┬───────────────────────┬──────────────┘
                     │                       │
             ┌───────▼────────┐      ┌───────▼──────────────┐
   capability│ tools/         │      │ providers/           │
             │  21 tools +    │      │  base (normalised    │
             │  registry +    │      │   message model)     │
             │  JSON-Schema   │      │  openai_compat       │
             │  validation    │      │  anthropic, gemini   │
             └───────┬────────┘      │  mock, registry      │
                     │               └───────┬──────────────┘
             ┌───────▼───────────────────────▼──────────────┐
   substrate │ transport/http.py  retries, gzip, SSE parser │
             │ ui/  theme, markdown, diff, widgets, render, │
             │      menu, input, i18n                       │
             └──────────────────────────────────────────────┘
```

Dependencies point downward only. `ui` never imports `agents`; `tools` never import `app`;
`providers` know nothing about the UI. That is what makes the recording-UI stub in
`tests/_harness.py` able to drive the entire engine.

## The turn state machine

`Agent.send()` is the heart. One user message in, one final assistant answer out, with as many
tool round-trips as needed:

```
user message
   │
   ▼
┌───────────────────────────── loop, at most max_turns ─────────────────────────────┐
│ 1. ContextManager.assemble(system + history)  → budgeted messages + report        │
│    · overflow ⇒ ContextOverflowError surfaced with a hint, never silent truncation │
│ 2. ModelRouter.complete(messages, tools)        → streaming events to the UI      │
│    · retries/backoff in the transport, failover across providers in the router    │
│ 3. record usage + cost, append the assistant message                              │
│ 4. no tool calls? ── finish_reason == "length" ⇒ one continuation nudge, else DONE │
│ 5. for each tool call: resolve → coerce+validate args → permission check → run     │
│    · read-only batch runs in parallel; any writer ⇒ strictly serial, in order      │
│    · every outcome becomes a tool message: success, validation error, denial,      │
│      crash. Errors are data the model can react to, not exceptions                 │
│ 6. identical call repeated > 3× ⇒ break with an explicit "stop repeating" result   │
└───────────────────────────────────────────────────────────────────────────────────┘
```

### Invariant: no orphaned tool calls

Providers hard-reject a history where an assistant `tool_calls` entry has no matching `tool`
message. NEXUS guarantees the pairing in three places:

- after every batch (`_heal_history` appends placeholders for anything unanswered),
- on abort mid-batch (pending calls get an "interrupted" result),
- in the context budget ladder (dropping an assistant turn drops its tool results too).

`tests/test_agent_loop.py` asserts the invariant for all three provider wire formats.

## The normalised message model

```python
Message(role, content, tool_calls, tool_call_id, name, meta)
content = str | [TextBlock | ImageBlock]
ToolCall(id, name, arguments: str, extra: dict)     # arguments is always JSON *text*
```

Each adapter translates to and from it:

| concern | OpenAI-compatible | Anthropic | Gemini |
|---|---|---|---|
| system prompt | a `system` message | top-level `system` field | `systemInstruction` |
| roles | system/user/assistant/tool | user/assistant only | user/model only |
| tool call | `tool_calls[].function` | `tool_use` content block | `functionCall` part |
| tool result | a `tool` message | `tool_result` block in a **user** turn | `functionResponse` part in a **user** turn |
| call ids | provided | provided | **absent** → synthesised and mapped back |
| streaming args | `delta.tool_calls[i].function.arguments` fragments | `input_json_delta.partial_json` | complete `args` object |
| reasoning | `delta.reasoning_content` | `thinking` blocks + `signature` | `part.thought` + `thoughtSignature` |
| max output | `max_tokens` / `max_completion_tokens` | `max_tokens` (**required**) | `generationConfig.maxOutputTokens` |

Details that are easy to miss and are handled explicitly:

- Anthropic requires alternating turns → consecutive same-role messages are merged, and a
  leading assistant turn is folded into the first user turn.
- Extended-thinking models require the `thinking` blocks (with signatures) to be replayed when
  returning tool results, so they are kept in `Message.meta` and re-emitted.
- Gemini 2.5 thinking models return a `thoughtSignature` that must be echoed back on the
  matching `functionCall`, or the tool loop fails; it is preserved in `ToolCall.extra`.
- Gemini rejects several JSON-Schema keywords (`$schema`, `additionalProperties`, `$ref`,
  `type: [...]`), so schemas are recursively sanitised on the way out.
- Reasoning models reject `temperature`/`top_p` and need `max_completion_tokens`, which the
  OpenAI adapter applies by model family.

## Streaming

`transport/http.py` implements the WHATWG event-stream algorithm properly: multi-line `data:`
fields joined with `\n`, CRLF/CR/LF endings, comment lines, `id`/`retry` fields, a UTF-8
BOM, events split across TCP chunks, a multi-byte character split across a chunk boundary, and
a trailing event with no blank line (flushed on close).

Providers emit normalised `StreamEvent`s (`TextDelta`, `ReasoningDelta`, `ToolCallStart`,
`ToolCallArgsDelta`, `UsageEvent`, `FinishEvent`) and `StreamAggregator` reassembles them, so
the agent loop and UI never see provider-specific delta shapes. Tool arguments arrive in
fragments and are concatenated **per index**, which is why `ToolCall.arguments` starts empty
rather than `"{}"` — a default of `"{}"` silently produced `{}{"path":…}` and was one of the
first bugs the test-suite caught.

## Context budgeting

`ContextManager.assemble()` returns the exact message list plus a `BudgetReport`. The ladder:

1. squeeze the largest tool outputs outside the protected tail (head + tail preserved);
2. replace older tool outputs with a re-runnable marker;
3. drop the oldest turns, taking orphaned tool results with them;
4. escalate into the protected tail: squeeze all → prune older → prune newest.

If it still does not fit, `BudgetReport.overflow` is set and the caller must say so. The user's
current request is never truncated silently. When the model's context window is unknown
(catalogue returns `0`), budgeting is skipped and reported rather than guessed.

Token estimation uses `tiktoken` when present and otherwise a calibrated heuristic that counts
CJK characters as ~1 token each — `len/4` under-estimates CJK by 2–3×, which is precisely the
direction that causes overflow.

## Permission engine

```
deny rules  →  read-only mode  →  allow rules  →  ask rules  →  mode policy  →  confirmer
```

Every decision is recorded with the rule or mode that produced it (`/rules`, `/audit`).
Counters are updated under a lock because swarm workers share one engine. `ConfirmationRequest`
is defined once (in `core/permissions.py`) and re-exported by `tools/base.py` so `isinstance`
works across the boundary.

Rule keys are normalised in one place: tools build `"<tool>:<subject>"` keys for display, and
the engine strips the tool prefix so both sides compare on the same footing. Pattern semantics
are documented and tested: `*` is flat (matches `/` and spaces, so `bash:rm -rf *` works) and
`**` is a strict path glob (`write_file:src/**/*.py`).

## Swarm

Three primitives, one scheduler:

- `TaskBoard` — tasks with owners, statuses, priorities and `depends_on`; `ready()` returns
  tasks whose dependencies are done or cancelled; `has_cycle()` prevents scheduler deadlocks.
- `Blackboard` — thread-safe messages (`result`, `finding`, `question`, `handoff`, `decision`,
  `note`), an artifact registry and a bounded `digest()` injected into every agent prompt.
- `Agent` instances — one per worker, each with its own history, persona, model and tool subset.

`SwarmRunner` schedules waves with a capped `ThreadPoolExecutor`, reconciles task states from
each run's `report` block, runs the reviewer gate, converts `blocker:` / `should-fix:` lines
into follow-up tasks, and finally asks the orchestrator to integrate. Concurrency is safe
because the shared services are: registry (locked), permissions (locked counters + audit),
checkpoints (atomic sequence counter — verified with 8 threads × 40 snapshots), ledger
(locked), blackboard and board (locked).

Parallel agents would interleave unreadably on one terminal, so the renderer switches to
**buffered** mode during a swarm: each agent's output is collected and printed as one labelled
block when it finishes.

## UI

- `theme.py` — width-correct layout primitives. `visible_width` uses East-Asian widths and
  strips ANSI, so tables and boxes align for CJK users too. Colour capability is detected
  (`NO_COLOR`, `FORCE_COLOR`, TTY, `TERM`, `COLORTERM`) and degrades truecolor → 256 → 16 →
  none. The 16-colour mapping is explicit: indices 8–15 are codes 90–97/100–107, not `3<index>`
  (which would silently produce "default foreground").
- `markdown.py` — a block renderer and a `StreamingMarkdown`. Inline formatting is a two-pass
  transform (code spans are protected first) so nesting works. Streaming holds back a partial
  line when it could still become a structural line or an inline span, which gives the
  invariant: **an emitted chunk never ends inside a markdown construct**. 750 combinations of
  document × width × chunk size are asserted to lose nothing, leak no markers and never
  overflow a frame.
- `render.py` — implements the UI protocol (`on_text`, `on_tool_start`, `on_agent_end`, …).
  It tracks a partial line so tool output never glues onto streamed prose.
- `prompt.py` — feature-detects `readline`. Some platforms ship an editline-backed module
  without `readline()` or `set_completer()`; the prompt degrades to plain `input()` instead of
  crashing (this was found by actually running it, not by reading docs).

## Error handling

`core/errors.py` defines one hierarchy. Expected failures carry `user_facing = True` and an
optional `hint`; the REPL prints them cleanly. Anything else is logged with a traceback,
reported as an unexpected error with the log path, and never dumped raw on the user.

Transport-level HTTP statuses are mapped once (`http_error`) **and** re-checked by every
provider (`_raise_for_status`), so error semantics do not depend on which transport is in use —
including the offline mock used by tests.

## Testing strategy

554 tests, all offline:

| suite | what it protects |
|---|---|
| `test_static` | AST scan: no undefined names, no unbound attribute roots, every module imports, no duplicate dict keys. Caught three real missing-import bugs. |
| `test_providers` | wire formats and streaming for all three families, against recorded payload shapes |
| `test_ignore` | gitignore subset: nesting, negation, anchoring, `**`, always-skipped dirs |
| `test_context` | budget ladder, orphan prevention, overflow honesty, compaction split |
| `test_tools` | real files and real subprocesses: edits, CRLF, non-UTF-8 refusal, atomicity, grep, classification |
| `test_agent_loop` | the turn machine: self-correction, denials, parallel vs serial, abort healing, loop detection |
| `test_swarm` | all five mode shapes, dependency waves, reviewer gate, plan-only, failure isolation, cancellation |
| `test_mcp` | a real subprocess MCP server: handshake, calls, timeouts, dead servers |
| `test_local_gateway` | a real loopback OpenAI-compatible gateway (9Router/Ollama/vLLM shape): listing, fragmented streaming tool calls, a full agent turn writing a real file, undo, and failover from a dead gateway |
| `test_cli` | the actual launcher in a subprocess: every subcommand, exit codes, JSON output, piped REPL |
| `test_ui` | layout, colour, markdown matrix (750 cases), widgets, prompt |
| `test_config` / `test_session` / `test_project` | layering + validation; crash recovery; repo detection |

`tests/_harness.py` contains the recording UI, which doubles as the executable definition of the
UI protocol: if the engine calls a method the renderer does not have, tests fail.
