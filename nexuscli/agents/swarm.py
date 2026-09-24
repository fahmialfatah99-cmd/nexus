"""Swarm orchestration.

Five coordination modes, all built on the same three primitives -- a
:class:`~nexuscli.core.taskboard.TaskBoard` (who does what, in which order), a
:class:`~nexuscli.agents.blackboard.Blackboard` (what everybody learned) and
per-role :class:`~nexuscli.agents.runtime.Agent` instances with their own
context window:

``hive``      orchestrator plans -> workers execute waves in parallel respecting
              dependencies -> reviewer gate -> fix wave -> integrated summary.
``pipeline``  fixed sequence of roles, each receiving the previous output.
``parallel``  every role attacks the same objective at once, results merged.
``debate``    N rounds of argument where each agent reads the others' positions,
              then a judge synthesises the decision.
``council``   parallel opinions + a synthesis pass (parallel with a judge).

Failure isolation: a crashing worker produces a failed run, never a crashed
swarm. Cancellation is checked between waves. Concurrency is capped.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..core.errors import NexusError
from ..core.taskboard import Task, TaskBoard
from ..providers.base import Usage
from ..tools.builtin.agent import build_swarm_tools
from .blackboard import Blackboard
from .persona import MODE_DEFAULT_CAST, Persona, get_persona
from .runtime import Agent, AgentOptions, Services, TurnResult, parse_report

MODES = ("hive", "pipeline", "parallel", "debate", "council", "review", "build", "debug", "audit")
#: Tie-break order for task assignment: specific specialists before generalists.
ROLE_PRIORITY = ("debugger", "security", "tester", "docs", "devops", "data", "refactorer",
                 "researcher", "architect", "critic", "reviewer", "implementer")
SWARM_TOOLS = ("swarm_post", "swarm_read", "swarm_task", "todo_write", "todo_read")


@dataclass
class SwarmConfig:
    mode: str = "hive"
    cast: List[str] = field(default_factory=list)
    max_parallel: int = 4
    max_rounds: int = 3
    max_turns_per_agent: int = 16
    debate_rounds: int = 2
    reviewer_gate: bool = True
    model_specs: Dict[str, str] = field(default_factory=dict)
    default_model_spec: str = ""
    temperature: Optional[float] = None
    plan_only: bool = False
    stream: bool = True

    def effective_cast(self) -> List[str]:
        if self.cast:
            return list(dict.fromkeys(self.cast))
        return list(MODE_DEFAULT_CAST.get(self.mode, MODE_DEFAULT_CAST["hive"]))


@dataclass
class AgentRun:
    persona_key: str
    agent: str
    task_id: str = ""
    prompt: str = ""
    text: str = ""
    report: Dict[str, Any] = field(default_factory=dict)
    ok: bool = True
    error: str = ""
    turns: int = 0
    tool_calls: int = 0
    usage: Usage = field(default_factory=lambda: Usage(requests=0))
    cost: float = 0.0
    latency_ms: int = 0
    round: int = 0

    @property
    def status(self) -> str:
        return str(self.report.get("status", "")).lower()

    def one_line(self) -> str:
        summary = self.report.get("summary") or " ".join(self.text.split())[:160]
        return summary


@dataclass
class SwarmResult:
    objective: str
    mode: str
    final_text: str = ""
    runs: List[AgentRun] = field(default_factory=list)
    board: TaskBoard = field(default_factory=TaskBoard)
    blackboard: Blackboard = field(default_factory=Blackboard)
    usage: Usage = field(default_factory=lambda: Usage(requests=0))
    cost: float = 0.0
    rounds: int = 0
    aborted: bool = False
    plan_only: bool = False
    elapsed_ms: int = 0

    def by_agent(self) -> Dict[str, List[AgentRun]]:
        out: Dict[str, List[AgentRun]] = {}
        for r in self.runs:
            out.setdefault(r.agent, []).append(r)
        return out

    def failures(self) -> List[AgentRun]:
        return [r for r in self.runs if not r.ok]


class SwarmRunner:
    def __init__(self, services: Services, config: Optional[SwarmConfig] = None,
                 *, ui: Any = None) -> None:
        self.services = services
        self.config = config or SwarmConfig()
        self.ui = ui or services.ui
        self.blackboard = Blackboard()
        self.board = TaskBoard()
        services.vars["blackboard"] = self.blackboard
        services.vars["board"] = self.board
        # Swarm coordination tools live in the same registry the workers use.
        for tool in build_swarm_tools():
            services.registry.register(tool, replace=True)
        self.log = services.log
        self._counter = 0
        self._counter_lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Agent construction
    # ------------------------------------------------------------------ #
    def _next_index(self) -> int:
        with self._counter_lock:
            self._counter += 1
            return self._counter

    def make_agent(self, persona_key: str, *, name: str = "", system_extra: str = "") -> Agent:
        persona = get_persona(persona_key)
        tools = persona.tools
        if tools is not None:
            tools = tuple(dict.fromkeys(list(tools) + list(SWARM_TOOLS)))
        persona = replace(persona, tools=tools)
        idx = self._next_index()
        agent_name = name or f"{persona.key}-{idx}"
        options = AgentOptions(
            model_spec=self.config.model_specs.get(persona_key) or self.config.default_model_spec
            or self.services.vars.get("model_spec", ""),
            temperature=self.config.temperature,
            max_turns=self.config.max_turns_per_agent,
            stream=self.config.stream,
            parallel_tools=True,
        )
        return Agent(services=self.services, persona=persona, name=agent_name, options=options,
                     system_extra=system_extra)

    def _emit(self, method: str, *args: Any, **kwargs: Any) -> None:
        handler = getattr(self.ui, method, None) if self.ui else None
        if callable(handler):
            try:
                handler(*args, **kwargs)
            except Exception as exc:  # UI must never break a swarm
                self.log.warning("swarm UI callback failed", method=method, error=str(exc))

    # ------------------------------------------------------------------ #
    # Running a single worker
    # ------------------------------------------------------------------ #
    def run_agent(self, persona_key: str, prompt: str, *, task_id: str = "", round_no: int = 0,
                  system_extra: str = "", agent_name: str = "") -> AgentRun:
        started = time.monotonic()
        agent = self.make_agent(persona_key, name=agent_name, system_extra=system_extra)
        persona = agent.persona
        run = AgentRun(persona_key=persona_key, agent=agent.name, task_id=task_id, prompt=prompt,
                       round=round_no)
        self._emit("on_agent_start", agent=agent.name, persona=persona, task_id=task_id,
                   round_no=round_no, prompt=prompt)
        try:
            result: TurnResult = agent.send(prompt)
            run.text = result.text
            run.report = result.report
            run.turns = result.turns
            run.tool_calls = result.tool_calls
            run.usage = result.usage
            run.cost = result.cost
            run.ok = result.ok and not result.aborted
            if result.aborted:
                run.error = "aborted"
            elif result.errors:
                run.error = "; ".join(result.errors)
        except NexusError as exc:
            run.ok = False
            run.error = f"{type(exc).__name__}: {exc}"
        except Exception as exc:  # isolate a crashing worker
            self.log.exception("swarm worker crashed", exc=exc, agent=agent.name)
            run.ok = False
            run.error = f"{type(exc).__name__}: {exc}"
        run.latency_ms = int((time.monotonic() - started) * 1000)
        self.blackboard.post(sender=agent.name, kind="result" if run.ok else "note",
                             content=run.one_line() or run.error, refs=[task_id] if task_id else [],
                             meta={"persona": persona_key, "status": run.status or ("ok" if run.ok else "error")})
        for path in run.report.get("artifact_list") or []:
            self.blackboard.publish_artifact(path, producer=agent.name)
        self._emit("on_agent_end", agent=agent.name, persona=persona, run=run)
        return run

    def run_parallel(self, jobs: Sequence[Dict[str, Any]]) -> List[AgentRun]:
        """Execute worker jobs concurrently (capped), preserving input order."""
        jobs = list(jobs)
        if not jobs:
            return []
        results: List[Optional[AgentRun]] = [None] * len(jobs)
        workers = max(1, min(self.config.max_parallel, len(jobs)))
        if workers == 1 or len(jobs) == 1:
            for i, job in enumerate(jobs):
                if self._aborted():
                    break
                results[i] = self.run_agent(**job)
            return [r for r in results if r]
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="swarm") as pool:
            futures = {}
            for i, job in enumerate(jobs):
                futures[pool.submit(self.run_agent, **job)] = i
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    results[i] = fut.result()
                except Exception as exc:  # pragma: no cover - run_agent isolates
                    self.log.exception("swarm job failed", exc=exc)
                    job = jobs[i]
                    results[i] = AgentRun(persona_key=job.get("persona_key", ""), agent="?",
                                          ok=False, error=str(exc))
        return [r for r in results if r]

    def _aborted(self) -> bool:
        return bool(self.services.cancelled is not None and self.services.cancelled.is_set())

    # ------------------------------------------------------------------ #
    # Entry point
    # ------------------------------------------------------------------ #
    def run(self, objective: str, *, context: str = "") -> SwarmResult:
        started = time.monotonic()
        mode = self.config.mode if self.config.mode in MODES else "hive"
        result = SwarmResult(objective=objective, mode=mode, blackboard=self.blackboard, board=self.board)
        self._emit("on_swarm_start", mode=mode, objective=objective, cast=self.config.effective_cast())
        self.blackboard.decide("swarm", f"mode={mode}; objective={objective[:400]}")
        try:
            if mode == "pipeline":
                self._mode_pipeline(objective, context, result)
            elif mode == "parallel":
                self._mode_parallel(objective, context, result)
            elif mode == "debate":
                self._mode_debate(objective, context, result)
            elif mode in ("council", "review", "audit", "build", "debug"):
                # Same shape, different default cast: specialists answer in
                # parallel, then the orchestrator synthesises one report.
                self._mode_council(objective, context, result, cast=self.config.effective_cast())
            else:
                self._mode_hive(objective, context, result)
        except NexusError as exc:
            result.final_text = result.final_text or f"Swarm stopped: {exc}"
            self.log.warning("swarm stopped", error=str(exc))
        except Exception as exc:
            self.log.exception("swarm crashed", exc=exc)
            result.final_text = result.final_text or f"Swarm crashed: {type(exc).__name__}: {exc}"
        result.aborted = self._aborted()
        result.rounds = max((r.round for r in result.runs), default=0)
        for r in result.runs:
            result.usage.merge(r.usage)
            result.cost += r.cost
        result.elapsed_ms = int((time.monotonic() - started) * 1000)
        if not result.final_text:
            result.final_text = self._fallback_summary(result)
        self._emit("on_swarm_end", result=result)
        return result

    # ------------------------------------------------------------------ #
    # hive
    # ------------------------------------------------------------------ #
    def _mode_hive(self, objective: str, context: str, result: SwarmResult) -> None:
        cast = [c for c in self.config.effective_cast() if c != "orchestrator"]
        # 1) plan
        plan_prompt = self._plan_prompt(objective, context, cast)
        plan_run = self.run_agent("orchestrator", plan_prompt, round_no=0)
        result.runs.append(plan_run)
        self._dedupe_board()
        if self.config.plan_only or self._aborted():
            result.plan_only = self.config.plan_only
            result.final_text = plan_run.text
            return
        if not self.board.all():
            # The orchestrator did not use the board: derive one task per cast member
            # from its plan so the swarm still makes progress.
            self.board.add(Task(title=objective[:160], description=plan_run.text[:4000], priority="high"))
            self.log.info("swarm: orchestrator produced no board tasks; created one from the plan")

        # 2) execute waves
        wave = 0
        while wave < max(1, self.config.max_rounds) * 3 and not self._aborted():
            ready = self.board.ready()
            if not ready:
                break
            wave += 1
            batch = ready[: max(1, self.config.max_parallel) * 2]
            jobs = []
            for task in batch:
                owner = task.owner or self._assign(task, cast, wave)
                self.board.update(task.id, status="in_progress", owner=owner)
                jobs.append({"persona_key": owner, "prompt": self._task_prompt(task, objective, context),
                             "task_id": task.id, "round_no": wave})
            runs = self.run_parallel(jobs)
            result.runs.extend(runs)
            self._reconcile_tasks(runs, batch)

            # 3) reviewer gate: verify the wave before moving on. It runs on
            #    successful waves too -- that is the point of a gate -- but only
            #    while the round budget lasts (disable with reviewer_gate=False).
            if self.config.reviewer_gate and wave <= self.config.max_rounds:
                gate = self.run_agent("reviewer", self._review_prompt(runs, objective), round_no=wave,
                                      agent_name=f"reviewer-gate-{wave}")
                result.runs.append(gate)
                if self._pending_fixes(runs):
                    self.blackboard.post(sender=gate.agent, kind="finding",
                                         content="wave had failures; reviewer verdict recorded")
                fix_tasks = self._extract_fix_tasks(gate, cast)
                if fix_tasks:
                    self.board.add_many(fix_tasks)
                    self.blackboard.post(sender=gate.agent, kind="decision",
                                         content=f"opened {len(fix_tasks)} fix task(s) from review")
                    continue
            if not self.board.ready() and not [t for t in self.board.all() if t.status == "pending"]:
                break

        # 4) integrate
        if not self._aborted():
            final = self.run_agent("orchestrator", self._synthesis_prompt(objective, result), round_no=wave + 1,
                                   agent_name="orchestrator-final")
            result.runs.append(final)
            result.final_text = final.text

    def _dedupe_board(self) -> int:
        """Collapse duplicate task titles.

        A planner that calls ``swarm_task add`` more than once for the same work
        would otherwise multiply the board (and the cost). Keeping the first
        occurrence is deterministic and safe.
        """
        seen: Dict[str, Task] = {}
        removed = 0
        for task in self.board.all():
            key = " ".join(task.title.lower().split())
            if key in seen:
                keeper = seen[key]
                for dep in list(task.depends_on):
                    pass
                for other in self.board.all():
                    other.depends_on = [keeper.id if d == task.id else d for d in other.depends_on]
                self.board.remove(task.id)
                removed += 1
            else:
                seen[key] = task
        if removed:
            self.log.info("swarm: removed duplicate tasks", count=removed)
        return removed

    def _assign(self, task: Task, cast: List[str], wave: int) -> str:
        """Pick the best-fit persona for a task (keyword match, then round robin)."""
        text = f"{task.title} {task.description} {' '.join(task.tags)}".lower()
        score: Dict[str, int] = {}
        keywords = {
            "architect": ("design", "architect", "interface", "schema", "api contract", "plan"),
            "implementer": ("implement", "write", "code", "feature", "add", "create", "fix bug", "refactor"),
            "reviewer": ("review", "audit", "check", "verify", "quality"),
            "tester": ("test", "spec", "coverage", "pytest", "unittest", "verify behaviour"),
            "debugger": ("debug", "crash", "error", "traceback", "flaky", "reproduce", "bug"),
            "security": ("security", "vulnerab", "cve", "injection", "secret", "auth", "permission"),
            "docs": ("doc", "readme", "comment", "guide", "changelog"),
            "researcher": ("research", "investigate", "compare", "evaluate", "find out"),
            "refactorer": ("refactor", "clean up", "simplify", "deduplicate"),
            "devops": ("ci", "build", "deploy", "docker", "release", "pipeline", "dependency"),
            "data": ("data", "csv", "json schema", "migration", "etl", "database"),
        }
        for persona_key in cast:
            words = keywords.get(persona_key, ())
            score[persona_key] = sum(1 for w in words if w in text)
        if score:
            best_score = max(score.values())
            if best_score > 0:
                # Tie-break towards the more specific specialist: "write unit
                # tests" scores for both implementer ("write") and tester
                # ("test"), and tester is the right owner.
                for preferred in ROLE_PRIORITY:
                    if score.get(preferred, 0) == best_score and preferred in cast:
                        return preferred
                return max(score.items(), key=lambda kv: kv[1])[0]
        if not cast:
            return "implementer"
        return cast[(hash(task.id) + wave) % len(cast)]

    def _reconcile_tasks(self, runs: Sequence[AgentRun], batch: Sequence[Task]) -> None:
        """Make sure every dispatched task leaves the board in a terminal/known state."""
        by_task = {r.task_id: r for r in runs if r.task_id}
        for task in batch:
            run = by_task.get(task.id)
            current = self.board.get(task.id)
            if current is None or current.status != "in_progress":
                continue  # the agent already updated it via swarm_task
            if run is None:
                self.board.update(task.id, status="pending", owner="")
                continue
            status = run.status
            if status in ("done", "complete", "completed"):
                self.board.update(task.id, status="done", result=run.one_line())
            elif status in ("blocked", "needs-review"):
                self.board.update(task.id, status="blocked", result=run.error or run.one_line())
            elif run.ok:
                self.board.update(task.id, status="done", result=run.one_line())
            else:
                self.board.update(task.id, status="blocked", result=run.error or "worker failed")

    def _pending_fixes(self, runs: Sequence[AgentRun]) -> bool:
        return any((not r.ok) or r.status in ("blocked", "needs-review") for r in runs)

    def _extract_fix_tasks(self, gate: AgentRun, cast: List[str]) -> List[Task]:
        """Turn a reviewer verdict into concrete follow-up tasks."""
        tasks: List[Task] = []
        existing = {t.title.lower() for t in self.board.all()}
        for line in gate.text.split("\n"):
            low = line.strip().lower()
            if not low:
                continue
            marker = None
            for token in ("blocker:", "should-fix:", "fix:", "- [ ]"):
                if token in low:
                    marker = token
                    break
            if not marker:
                continue
            title = line.split(marker)[-1].strip(" -*\t")
            if not title or len(title) < 8 or title.lower() in existing:
                continue
            owner = "implementer" if "implementer" in cast else (cast[0] if cast else "implementer")
            if any(w in low for w in ("test", "coverage")) and "tester" in cast:
                owner = "tester"
            if any(w in low for w in ("security", "vulnerab", "secret")) and "security" in cast:
                owner = "security"
            tasks.append(Task(title=title[:200], description=gate.text[:1500], owner="",
                              priority="high", tags=[owner]))
            existing.add(title.lower())
            if len(tasks) >= 6:
                break
        return tasks

    # ------------------------------------------------------------------ #
    # pipeline / parallel / debate / council
    # ------------------------------------------------------------------ #
    def _mode_pipeline(self, objective: str, context: str, result: SwarmResult) -> None:
        cast = [c for c in self.config.effective_cast() if c != "orchestrator"]
        previous = ""
        for step, persona_key in enumerate(cast, start=1):
            if self._aborted():
                break
            persona = get_persona(persona_key)
            prompt = (
                f"Objective:\n{objective}\n\n"
                + (f"Context:\n{context}\n\n" if context else "")
                + (f"Output of the previous stage ({cast[step - 2] if step > 1 else 'user'}):\n{previous}\n\n"
                   if previous else "")
                + f"You are stage {step}/{len(cast)}. Do your part completely, then hand off. "
                  "Do not repeat the previous stage's work."
            )
            run = self.run_agent(persona_key, prompt, round_no=step, task_id=f"stage-{step}")
            result.runs.append(run)
            previous = run.text or previous
            self.blackboard.post(sender=run.agent, kind="handoff",
                                 content=f"stage {step} ({persona.name}) complete", meta={"stage": step})
        result.final_text = previous

    def _mode_parallel(self, objective: str, context: str, result: SwarmResult) -> None:
        cast = [c for c in self.config.effective_cast() if c != "orchestrator"]
        jobs = [{"persona_key": key,
                 "prompt": self._parallel_prompt(objective, context, key, len(cast)),
                 "task_id": f"parallel-{key}", "round_no": 1} for key in cast]
        runs = self.run_parallel(jobs)
        result.runs.extend(runs)
        result.final_text = self._merge_parallel(runs)

    def _mode_debate(self, objective: str, context: str, result: SwarmResult) -> None:
        cast = [c for c in self.config.effective_cast() if c != "orchestrator"]
        rounds = max(1, self.config.debate_rounds)
        positions: Dict[str, str] = {}
        for rnd in range(1, rounds + 1):
            if self._aborted():
                break
            jobs = []
            for key in cast:
                others = "\n\n".join(f"--- {name} ---\n{text}" for name, text in positions.items()
                                     if name != key)
                prompt = (
                    f"Proposition under debate:\n{objective}\n\n"
                    + (f"Context:\n{context}\n\n" if context else "")
                    + (f"Other positions so far (round {rnd}):\n{others}\n\n" if others else "")
                    + (f"This is round {rnd} of {rounds}. "
                       + ("State your opening position with concrete evidence from the codebase."
                          if rnd == 1 else
                          "Rebut the strongest opposing point with evidence, then refine your position. "
                          "Concede explicitly anything the evidence contradicts.")
                       + " End with: POSITION: <one paragraph>.")
                )
                jobs.append({"persona_key": key, "prompt": prompt, "task_id": f"debate-r{rnd}-{key}",
                             "round_no": rnd})
            runs = self.run_parallel(jobs)
            result.runs.extend(runs)
            for run in runs:
                positions[run.persona_key] = run.text
        judge = self.run_agent("orchestrator", self._judge_prompt(objective, positions), round_no=rounds + 1,
                               agent_name="judge")
        result.runs.append(judge)
        result.final_text = judge.text

    def _mode_council(self, objective: str, context: str, result: SwarmResult, *, cast: Sequence[str],
                      synthesise: bool = True) -> None:
        cast = [c for c in cast if c != "orchestrator"]
        jobs = [{"persona_key": key,
                 "prompt": self._council_prompt(objective, context, key),
                 "task_id": f"council-{key}", "round_no": 1} for key in cast]
        runs = self.run_parallel(jobs)
        result.runs.extend(runs)
        if not synthesise:
            result.final_text = self._merge_parallel(runs)
            return
        final = self.run_agent("orchestrator", self._synthesis_prompt(objective, result), round_no=2,
                               agent_name="synthesiser")
        result.runs.append(final)
        result.final_text = final.text

    # ------------------------------------------------------------------ #
    # Prompts
    # ------------------------------------------------------------------ #
    def _plan_prompt(self, objective: str, context: str, cast: Sequence[str]) -> str:
        roster = "\n".join(f"- {k}: {get_persona(k).role}" for k in cast)
        return (
            f"Objective:\n{objective}\n\n"
            + (f"Context:\n{context}\n\n" if context else "")
            + f"Your team:\n{roster}\n\n"
            "Plan the work. Inspect the repository first (read_file/grep/list_dir) so the plan matches "
            "reality, then create the task board:\n"
            "1. Call swarm_task with action=add once per task. Give each task a specific, verifiable title "
            "(imperative, <=120 chars) and set depends_on for tasks that must wait.\n"
            "2. Keep the number of tasks small (3-8). No task may overlap another's files.\n"
            "3. Every task must be independently completable by one specialist with the information in its title.\n"
            "4. Do NOT implement anything yourself.\n"
            "After the board is complete, reply with a short plan summary and the report block."
        )

    def _task_prompt(self, task: Task, objective: str, context: str) -> str:
        digest = self.blackboard.digest(per_agent=4, max_chars=4000)
        board = self.board.to_markdown()
        return (
            f"Overall objective:\n{objective}\n\n"
            + (f"Context:\n{context}\n\n" if context else "")
            + f"YOUR TASK ({task.id}): {task.title}\n"
            + (f"Details:\n{task.description}\n" if task.description else "")
            + (f"Depends on: {', '.join(task.depends_on)} (already completed)\n" if task.depends_on else "")
            + f"\nShared task board:\n{board}\n\n"
            + (f"Blackboard (what the team knows):\n{digest}\n\n" if digest.strip() else "")
            + "Do the work completely and verify it. Stay inside your task's scope -- other agents own "
            "the other tasks. When finished call swarm_task with action=complete and result=<what you did "
            "and how you verified it>. Post anything the team must know with swarm_post (kind=finding). "
            "If you cannot finish, call swarm_task action=block with the reason."
        )

    def _review_prompt(self, runs: Sequence[AgentRun], objective: str) -> str:
        summaries = "\n".join(f"- [{r.persona_key}] {r.one_line()}" for r in runs)
        artifacts = "\n".join(f"- {p}" for p in sorted(self.blackboard.artifacts()))
        return (
            f"Objective:\n{objective}\n\nWork reported by the team:\n{summaries}\n\n"
            f"Artifacts touched:\n{artifacts or '(none recorded)'}\n\n"
            "Review the ACTUAL state of the code (read the files, run the tests you can). "
            "Report integration problems: inconsistencies between agents, broken callers, missing "
            "error handling, untested claims, style clashes.\n"
            "For each problem emit a line starting with exactly 'blocker:' or 'should-fix:' followed by a "
            "specific, actionable instruction. If the work is sound, say so and emit no such lines."
        )

    def _synthesis_prompt(self, objective: str, result: SwarmResult) -> str:
        board = self.board.to_markdown()
        runs = "\n".join(
            f"- [{r.persona_key} round {r.round} {'ok' if r.ok else 'FAILED'}] {r.one_line()}"
            for r in result.runs)
        digest = self.blackboard.digest(per_agent=5, max_chars=6000)
        return (
            f"Objective:\n{objective}\n\nTask board:\n{board}\n\nAgent reports:\n{runs}\n\n"
            f"Blackboard:\n{digest}\n\n"
            "Produce the final integrated answer for the user: what was built/changed (exact paths), "
            "how it was verified, what is still open, and any risks. Verify claims against the "
            "repository where cheap to do so. Do not invent work that was not reported."
        )

    def _judge_prompt(self, objective: str, positions: Dict[str, str]) -> str:
        body = "\n\n".join(f"--- {name} ({get_persona(name).role}) ---\n{text[:6000]}"
                          for name, text in positions.items())
        return (
            f"Proposition:\n{objective}\n\nPositions argued:\n{body}\n\n"
            "You are the judge. Weigh the arguments by evidence, not rhetoric. State: the decision, the "
            "decisive reasons, which objections survive and must be handled, and the concrete next step. "
            "Be explicit about what remains genuinely uncertain."
        )

    def _parallel_prompt(self, objective: str, context: str, persona_key: str, total: int) -> str:
        persona = get_persona(persona_key)
        return (
            f"Objective:\n{objective}\n\n" + (f"Context:\n{context}\n\n" if context else "")
            + f"You are one of {total} specialists working on this objective simultaneously, as the "
            f"{persona.role}. Do your part from your own angle; other agents cover the other angles, so "
            "do not duplicate their work. Report concretely with paths and evidence."
        )

    def _council_prompt(self, objective: str, context: str, persona_key: str) -> str:
        persona = get_persona(persona_key)
        return (
            f"Question for the council:\n{objective}\n\n" + (f"Context:\n{context}\n\n" if context else "")
            + f"Answer strictly as the {persona.role}: what you see, what you would do, and what could go "
            "wrong. Ground every claim in something you actually read or ran."
        )

    # ------------------------------------------------------------------ #
    def _merge_parallel(self, runs: Sequence[AgentRun]) -> str:
        parts = []
        for run in runs:
            persona = get_persona(run.persona_key)
            status = "OK" if run.ok else f"FAILED ({run.error})"
            parts.append(f"## {persona.name} {persona.emoji} -- {persona.role} [{status}]\n\n{run.text.strip()}")
        return "\n\n---\n\n".join(parts)

    def _fallback_summary(self, result: SwarmResult) -> str:
        if not result.runs:
            return "The swarm did not run any agent."
        return self._merge_parallel(result.runs[-3:])


__all__ = ["SwarmRunner", "SwarmConfig", "SwarmResult", "AgentRun", "MODES", "ROLE_PRIORITY",
           "MODE_DEFAULT_CAST"]
