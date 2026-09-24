# Guide for AI agents working on NEXUS

NEXUS reads this file at session start (see `core/project.py` → `DOC_FILES`).

## Commands
- tests: `python3 tests/run_tests.py` (or `./nexus selftest`)
- one area: `python3 tests/run_tests.py providers`
- static guard: `python3 tests/run_tests.py static` (undefined names, import health)
- no linter/formatter is configured on purpose: zero dependencies is a hard rule

## Hard rules
1. **Standard library only.** Adding a third-party import is a regression. Optional
   integrations must be guarded (`try: import tiktoken` / `try: import readline`) and must
   degrade gracefully — see `core/context.py` and `ui/prompt.py`.
2. **Never write to stdout except through `ui/`.** The renderer owns stdout; logging goes to
   `~/.nexus/logs/nexus.log` (`core/logging_.py`). A stray `print()` corrupts the TUI.
3. **Provider quirks belong in the adapter.** `providers/{openai_compat,anthropic,gemini}.py`
   are the only files allowed to know about wire formats. The agent loop sees the normalised
   `Message`/`ToolCall` model from `providers/base.py`.
4. **Preserve the no-orphan-tool-calls invariant.** Every assistant `tool_calls` entry must have
   a matching `tool` message: after each batch, on abort, and when the context ladder drops a
   turn (`agents/runtime.py::_heal_history`, `core/context.py::_drop_oldest`).
5. **Never guess a number.** Unknown context windows, prices and model capabilities stay `0` /
   `unknown` and are reported as such. Inventing a limit is worse than not enforcing one.
6. **Errors are data for the model, exceptions for the user.** A tool failure returns
   `ToolResult(is_error=True)`; only `NexusError` subclasses escape to the REPL handler.
7. **Width-correct text.** Any layout code must use `ui/theme.py::visible_width` (CJK is two
   cells, ANSI is zero). Never `len()` on a string that may contain escapes or CJK.
8. **Thread safety where the swarm touches it.** Registry, permission counters/audit,
   checkpoint sequence, usage ledger, blackboard and task board are all shared by concurrent
   workers — keep their locks.

## Architecture
See `docs/ARCHITECTURE.md`. Layers point downward only: `cli`/`app` → `agents` → `core`
services → `tools`/`providers` → `transport`/`ui`. `ui` never imports `agents`; `tools` never
import `app`.

## Testing conventions
- Every test must run offline and deterministically. Use `MockTransport` for providers and
  `tests/_harness.py::make_env` for the agent loop.
- `tests/_harness.py::RecordingUI` is the executable definition of the UI protocol: if the
  engine calls a renderer method, add it there too.
- When a test fails, first decide whether the *code* or the *expectation* is wrong. Several
  behaviours look like bugs and are correct: a `full-auto` normal-risk operation does not
  prompt; a `report`-fenced block is rendered as a code box; streaming prose is soft-wrapped by
  the terminal; an impossible context budget must report `overflow=True` instead of truncating
  the user's request.

## Definition of done
- `python3 tests/run_tests.py` is green (554 tests at the time of writing)
- `./nexus doctor` passes, `./nexus demo` runs clean
- Every subcommand builds: `TestParserIntegrity` covers argparse conflicts
- No new `print()` outside `ui/`, no new third-party import
- Docs updated when behaviour changes (`docs/COMMANDS.md` tables are generated from the code —
  regenerate rather than hand-editing)
