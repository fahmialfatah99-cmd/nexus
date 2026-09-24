# Swarm mode

Multiple agents, one objective. Each agent has its own **personality, duties, tool subset,
temperature and context window**, and they coordinate through two shared structures.

```
                      ┌────────────────────────────┐
   objective ────────▶│  Atlas · orchestrator      │  plans, verifies, integrates
                      └─────────────┬──────────────┘
                                    │ tasks (with dependencies)
                      ┌─────────────▼──────────────┐
                      │  TaskBoard                 │  ready() → next wave
                      └─────────────┬──────────────┘
        ┌───────────────┬───────────┼───────────┬───────────────┐
        ▼               ▼           ▼           ▼               ▼
   ┌─────────┐    ┌───────────┐ ┌────────┐ ┌────────┐    ┌──────────┐
   │ Vega    │    │ Forge     │ │ Probe  │ │ Warden │    │ Sable    │
   │architect│    │implementer│ │ tester │ │security│    │ reviewer │
   └────┬────┘    └─────┬─────┘ └───┬────┘ └───┬────┘    └────┬─────┘
        └───────────────┴─────┬─────┴───────────┴──────────────┘
                              ▼
                  ┌────────────────────────┐
                  │  Blackboard            │  findings, questions, decisions,
                  └────────────────────────┘  artifacts, results
```

## The two shared structures

**TaskBoard** — tasks with an id, title, description, status (`pending`, `in_progress`, `done`,
`blocked`, `cancelled`), owner, priority and `depends_on`. `ready()` returns tasks whose
dependencies are done or cancelled, ordered by priority then age. `has_cycle()` prevents a
dependency cycle from deadlocking the scheduler. It is the *same* board the `todo_write` tool
uses in solo sessions.

**Blackboard** — thread-safe messages with kinds `result`, `finding`, `question`, `handoff`,
`decision`, `note`; an artifact registry (path → producers); and a bounded `digest()` that is
injected into every agent prompt so nobody works from stale assumptions. Questions can be
answered by referencing their id, and `unanswered_questions()` finds the ones nobody resolved.

Agents touch both through three tools: `swarm_post`, `swarm_read`, `swarm_task`
(`list` / `ready` / `claim` / `complete` / `block` / `add`). These are registered only during a
swarm run.

## The nine modes

| mode | shape | synthesises? | default cast | use it for |
|---|---|---|---|---|
| `hive` | plan → dependency waves → reviewer gate → integrate | yes | orchestrator, architect, implementer, reviewer, tester | real multi-file features |
| `pipeline` | fixed role sequence, each stage receives the previous output | last stage is the output | architect, implementer, reviewer, tester, docs | design → build → test → document |
| `parallel` | every role answers the same objective at once | merged, no judge | architect, implementer, reviewer, security, tester, docs | broad coverage fast |
| `debate` | N rounds; each agent reads the others' positions, then a judge decides | judge verdict | architect, critic, security | contentious design choices |
| `council` | parallel opinions + a synthesis pass | yes | architect, implementer, reviewer, security, tester, docs | decisions needing several viewpoints |
| `review` | reviewers in parallel + a merged report | yes | reviewer, security, tester | code review |
| `build` | focused implementation | yes | architect, implementer, tester | small, well-understood features |
| `debug` | failure hunting | yes | debugger, tester, reviewer | hard-to-find bugs |
| `audit` | security / release readiness | yes | security, reviewer, devops | pre-release checks |

```bash
nexus swarm "add CSV export with tests"                      # hive (default)
nexus swarm "refactor the parser" --swarm-mode pipeline
nexus debate "SQLite or JSONL for sessions?" --rounds 3
nexus swarm "review src/" --swarm-mode review
nexus swarm "big change" --plan-only                         # task graph only
```

In a session: `/swarm <objective>`, `/swarm-mode <mode>`, `/cast <personas…>`, `/debate <q>`,
`/agent <persona> <prompt>`, `/board`, `/agents`.

## What a `hive` run actually does

1. **Plan.** Atlas gets the objective plus the roster, is told to inspect the repository first,
   and must create tasks with `swarm_task action=add` (3–8 tasks, dependencies where needed, no
   overlapping files). If it produces no board tasks, one is derived from its plan so the run
   still makes progress. Duplicate titles are collapsed — a planner that re-adds work cannot
   multiply the board (or the bill).
2. **Execute waves.** `board.ready()` gives the runnable tasks; each is assigned to the best-fit
   persona by keyword score with a deterministic tie-break towards the more specific specialist
   ("write unit tests for the parser" → `tester`, not `implementer`). Up to `max_parallel`
   workers run concurrently, each a separate `Agent` with its own history.
3. **Reconcile.** A worker that called `swarm_task action=complete` has already updated the
   board; otherwise its `report` block decides (`done` / `blocked` / `needs-review`), and a
   crashed worker leaves the task `blocked` with the error recorded.
4. **Reviewer gate.** After each wave (while the round budget lasts) Sable reviews the *actual*
   state of the code and emits `blocker:` / `should-fix:` lines. Those become new high-priority
   tasks, assigned to the right specialist, and the loop continues.
5. **Integrate.** Atlas writes the final answer: what changed (exact paths), how it was verified,
   what is open, what the risks are.

## The reporting contract

Every persona's system prompt ends with the same contract, so heterogeneous agents produce
machine-parseable output:

````
```report
status: done|blocked|needs-review
summary: <one line, what actually happened>
artifacts: <comma separated paths, or "none">
followups: <what remains, or "none">
confidence: high|medium|low
```
````

`parse_report()` extracts it; the swarm uses `status` to reconcile tasks and `artifacts` to
populate the blackboard's artifact registry. In solo sessions you see it rendered as a box.

The contract also states the rules of engagement: never claim unverified work, prefer reading a
file over guessing its contents, report exact paths and error messages, and say precisely what
you need when blocked.

## Custom personas

A persona is data. Drop a JSON file in `~/.nexus/personas/` (personal) or
`<project>/.nexus/personas/` (team):

```json
{
  "key": "dba",
  "name": "Dbora",
  "emoji": "🗄️",
  "role": "the database specialist",
  "style": "You think in transactions, indexes and migrations. You never run a destructive statement without an explicit rollback plan.",
  "duties": "- Review every schema change for locking, index and migration cost.\n- Write migrations that are reversible.\n- Verify query plans for anything touching a hot table.",
  "focus": ["Is this migration reversible?", "What happens to in-flight writes?"],
  "temperature": 0.2,
  "model_pref": "smart",
  "max_turns": 18,
  "tools": ["read_file", "grep", "find_files", "bash", "git", "swarm_post", "swarm_read", "swarm_task"]
}
```

Fields: `key` (required), `name`, `emoji`, `role`, `style`, `duties`, `focus[]`, `temperature`,
`model_pref` (`smart`/`fast`/`balanced`), `max_turns`, `color`, `tools[]` (`null` or omitted =
all tools). A file may contain one object or an array. `swarm_*` and `todo_*` tools are always
added automatically during a swarm run.

```bash
nexus personas                     # built-in and custom personas
nexus agent dba "review this migration"
nexus swarm "add a users table" --cast architect dba tester
```

A malformed persona file is skipped with a log entry, not a crash.

## Per-role models

Give the orchestrator and reviewer a strong model and the bulk workers a fast cheap one:

```json
"swarm": {
  "mode": "hive",
  "max_parallel": 4,
  "max_rounds": 3,
  "model_specs": {
    "orchestrator": "anthropic:claude-sonnet-4-5",
    "reviewer":     "anthropic:claude-sonnet-4-5",
    "implementer":  "groq:llama-3.3-70b-versatile",
    "tester":       "groq:llama-3.3-70b-versatile"
  }
}
```

```bash
nexus config set swarm.model_specs '{"orchestrator":"anthropic:sonnet","tester":"groq:llama-3.3-70b-versatile"}'
```

`/usage` breaks tokens and cost down per agent, so you can see which role is expensive.

## Tuning

| setting | default | effect |
|---|---|---|
| `swarm.mode` | `hive` | orchestration shape |
| `swarm.cast` | mode default | which personas participate |
| `swarm.max_parallel` | 4 | concurrent workers (1 = fully serial, useful for debugging) |
| `swarm.max_rounds` | 3 | reviewer-gate budget; waves are capped at `max_rounds * 3` |
| `swarm.max_turns_per_agent` | 16 | tool-loop ceiling per worker |
| `swarm.debate_rounds` | 2 | rounds in `debate` mode |
| `swarm.reviewer_gate` | true | set false to skip verification waves |
| `swarm.model_specs` | {} | per-persona model override |

## Failure behaviour

- A worker that raises produces `AgentRun(ok=False, error=…)`; the swarm continues.
- Cancellation (`Ctrl+C` in a session, or the cancel event) is checked between waves; remaining
  tasks stay on the board and the result is marked `aborted`.
- A worker that repeats an identical call more than three times is stopped with an explicit
  "try a different approach" result instead of burning tokens to `max_turns`.
- Completing or claiming an already-finished task returns "already done, move on" rather than
  silently re-running it.
- Parallel output is buffered per agent and printed as one labelled block, because interleaved
  streams from four agents are unreadable.

## Honest limits

- **Agents do not see each other's context.** They share the blackboard and the task board, not
  their transcripts. A worker's brief must be self-contained; the orchestrator prompt says so.
- **No automatic conflict resolution on files.** Two workers editing the same file will both
  write. The planning prompt forbids overlapping file ownership and the reviewer gate catches
  the damage, but there is no merge engine.
- **Cost scales with the cast.** `hive` runs at least a planner, the workers, a reviewer and an
  integrator. Check `/usage` after your first run and lower `max_parallel` / `max_rounds` or use
  cheaper `model_specs` for bulk roles.
- **Quality depends on the models.** Personas shape behaviour; they cannot make a weak model
  produce strong reviews. Assign `model_pref`/`model_specs` accordingly.
