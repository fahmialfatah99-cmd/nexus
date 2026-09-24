"""Swarm orchestration tests.

Uses a dispatcher-driven mock provider so each persona answers according to its
role, which exercises the real scheduling logic (planning, dependency waves,
parallelism, reviewer gate, synthesis) without any network access.
"""

from __future__ import annotations

import sys
import threading
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from _harness import make_env  # noqa: E402

from nexuscli.agents.blackboard import Blackboard  # noqa: E402
from nexuscli.agents.persona import MODE_DEFAULT_CAST, get_persona  # noqa: E402
from nexuscli.agents.swarm import MODES, SwarmConfig, SwarmRunner  # noqa: E402
from nexuscli.core.taskboard import Task, TaskBoard  # noqa: E402
from nexuscli.providers.base import Message  # noqa: E402
from nexuscli.providers.mock import MockProvider  # noqa: E402


def persona_of(messages) -> str:
    """Identify the calling persona from its system prompt."""
    system = messages[0].text if messages and messages[0].role == "system" else ""
    for key in ("orchestrator", "architect", "implementer", "reviewer", "tester", "debugger",
                "security", "docs", "critic", "researcher", "refactorer", "devops", "data", "main"):
        if get_persona(key).name in system:
            return key
    return "main"


def add_task_calls(specs):
    """Build the tool-call script an orchestrator uses to populate the board."""
    return {"text": "", "tool_calls": [("swarm_task", {"action": "add", "title": t, "depends_on": d})
                                       for t, d in specs]}


def make_swarm(*, mode="hive", cast=None, max_parallel=2, max_rounds=2, dispatcher=None,
               workspace=None, scripts=None, plan_only=False, reviewer_gate=True):
    agent, provider, tmp = make_env(scripts=scripts or [], workspace=workspace, max_turns=6)
    if dispatcher is not None:
        provider.dispatcher = dispatcher
    services = agent.services
    config = SwarmConfig(mode=mode, cast=cast or [], max_parallel=max_parallel, max_rounds=max_rounds,
                         max_turns_per_agent=4, plan_only=plan_only, reviewer_gate=reviewer_gate,
                         stream=False, debate_rounds=2)
    runner = SwarmRunner(services, config, ui=services.ui)
    return runner, provider, tmp


class TestBlackboard(unittest.TestCase):
    def test_post_and_query(self):
        bb = Blackboard()
        e1 = bb.post(sender="a", content="found the parser", kind="finding")
        bb.post(sender="b", content="need the schema", kind="question", to="a")
        self.assertEqual(len(bb.query()), 2)
        self.assertEqual(len(bb.query(kind="question")), 1)
        self.assertEqual(len(bb.query(sender="a")), 1)
        self.assertEqual(bb.query(to="a")[0].content, "found the parser")
        bb.mark_read("a", [e1.id])
        self.assertEqual(bb.query(unread_for="a")[0].kind, "question")

    def test_unknown_kind_is_normalised(self):
        bb = Blackboard()
        self.assertEqual(bb.post(sender="a", content="x", kind="bogus").kind, "note")

    def test_artifacts_and_decisions(self):
        bb = Blackboard()
        bb.publish_artifact("src/a.py", producer="forge", note="added")
        bb.publish_artifact("src/a.py", producer="sable")
        arts = bb.artifacts()
        self.assertEqual(arts["src/a.py"]["producers"], ["forge", "sable"])
        bb.decide("atlas", "use JSON")
        self.assertEqual(bb.decisions(), ["[atlas] use JSON"])

    def test_digest_is_bounded_and_thread_safe(self):
        bb = Blackboard()
        errors = []

        def writer(n):
            try:
                for i in range(50):
                    bb.post(sender=f"agent{n}", content=f"message {i} " + "x" * 100, kind="finding")
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        threads = [threading.Thread(target=writer, args=(n,)) for n in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        self.assertEqual(bb.stats()["entries"], 300)
        self.assertLessEqual(len(bb.digest(max_chars=2000)), 2000)

    def test_unanswered_questions(self):
        bb = Blackboard()
        q = bb.post(sender="tester", content="where are fixtures?", kind="question")
        self.assertEqual([e.id for e in bb.unanswered_questions()], [q.id])
        bb.post(sender="impl", content="tests/fixtures", kind="finding", meta={"answers": [q.id]})
        self.assertEqual(bb.unanswered_questions(), [])


class TestTaskBoard(unittest.TestCase):
    def test_dependencies_gate_readiness(self):
        board = TaskBoard()
        a = board.add(Task(title="design"))
        b = board.add(Task(title="implement", depends_on=[a.id]))
        self.assertEqual([t.id for t in board.ready()], [a.id])
        board.update(a.id, status="done")
        self.assertEqual([t.id for t in board.ready()], [b.id])

    def test_cycle_detection(self):
        board = TaskBoard()
        a = board.add(Task(title="a"))
        b = board.add(Task(title="b", depends_on=[a.id]))
        board.update(a.id, depends_on=[b.id])
        self.assertTrue(board.has_cycle())

    def test_cancelled_dependency_unblocks(self):
        board = TaskBoard()
        a = board.add(Task(title="a"))
        b = board.add(Task(title="b", depends_on=[a.id]))
        board.update(a.id, status="cancelled")
        self.assertEqual([t.id for t in board.ready()], [b.id])

    def test_priority_ordering(self):
        board = TaskBoard()
        board.add(Task(title="low", priority="low"))
        crit = board.add(Task(title="crit", priority="critical"))
        ready = board.ready()
        self.assertEqual(ready[0].id, crit.id)

    def test_serialisation_roundtrip(self):
        board = TaskBoard()
        one = board.add(Task(title="one", tags=["x"], result="done thing", status="done"))
        board.add(Task(title="two", status="blocked", depends_on=[one.id]))
        restored = TaskBoard.from_dict(board.to_dict())
        self.assertEqual([t.title for t in restored.all()], ["one", "two"])
        self.assertEqual(restored.all()[0].result, "done thing")
        self.assertEqual(restored.all()[0].status, "done")
        self.assertEqual(restored.all()[1].depends_on, [one.id])
        self.assertIn("[x]", restored.to_markdown())
        self.assertIn("[!]", restored.to_markdown())

    def test_remove_cleans_dependencies(self):
        board = TaskBoard()
        a = board.add(Task(title="a"))
        b = board.add(Task(title="b", depends_on=[a.id]))
        board.remove(a.id)
        self.assertEqual(board.get(b.id).depends_on, [])

    def test_progress_and_percent(self):
        board = TaskBoard()
        board.add(Task(title="a", status="done"))
        board.add(Task(title="b"))
        self.assertEqual(board.progress()["total"], 2)
        self.assertEqual(board.percent_done(), 50.0)


class TestSwarmModes(unittest.TestCase):
    def test_all_modes_have_a_default_cast_of_real_personas(self):
        for mode, cast in MODE_DEFAULT_CAST.items():
            for key in cast:
                self.assertIsNotNone(get_persona(key), f"{mode}/{key}")
        for mode in MODES:
            self.assertTrue(SwarmConfig(mode=mode).effective_cast())

    def test_pipeline_runs_each_stage_in_order(self):
        seen = []

        def dispatcher(messages):
            persona = persona_of(messages)
            seen.append(persona)
            last_user = [m for m in messages if m.role == "user"][-1].text
            return f"{persona.upper()} output (received {len(last_user)} chars)"

        runner, provider, tmp = make_swarm(mode="pipeline",
                                           cast=["architect", "implementer", "tester"],
                                           dispatcher=dispatcher)
        result = runner.run("Build a parser")
        self.assertEqual(seen, ["architect", "implementer", "tester"])
        self.assertEqual(len(result.runs), 3)
        self.assertIn("TESTER output", result.final_text)
        # each stage received the previous stage's output
        self.assertIn("ARCHITECT output", result.runs[1].prompt)
        self.assertIn("IMPLEMENTER output", result.runs[2].prompt)

    def test_parallel_runs_every_role(self):
        def dispatcher(messages):
            return f"opinion from {persona_of(messages)}"

        runner, _, _ = make_swarm(mode="parallel", cast=["architect", "security", "tester"],
                                  max_parallel=3, dispatcher=dispatcher)
        result = runner.run("How should we store sessions?")
        self.assertEqual(len(result.runs), 3)
        for key in ("architect", "security", "tester"):
            self.assertIn(f"opinion from {key}", result.final_text)

    def test_debate_shows_opponents_positions(self):
        rounds = {}

        def dispatcher(messages):
            persona = persona_of(messages)
            user = [m for m in messages if m.role == "user"][-1].text
            rounds.setdefault(persona, []).append(user)
            return f"POSITION of {persona}"

        runner, _, _ = make_swarm(mode="debate", cast=["architect", "critic"], dispatcher=dispatcher)
        runner.config.debate_rounds = 2
        result = runner.run("Should we use SQLite or JSON files?")
        # judge run is the last one
        self.assertEqual(result.runs[-1].agent, "judge")
        self.assertIn("POSITION of architect", result.runs[-1].prompt)
        # in round 2 each debater saw the other's round-1 position
        round2 = [p for p in rounds["architect"] if "POSITION of critic" in p]
        self.assertTrue(round2, "debaters must see opposing positions in later rounds")

    def test_council_synthesises(self):
        def dispatcher(messages):
            persona = persona_of(messages)
            if persona == "orchestrator":
                return "FINAL SYNTHESIS: do X"
            return f"{persona} says Y"

        runner, _, _ = make_swarm(mode="council", cast=["architect", "tester"], dispatcher=dispatcher)
        result = runner.run("Pick a caching strategy")
        self.assertEqual(result.final_text, "FINAL SYNTHESIS: do X")
        self.assertGreaterEqual(len(result.runs), 3)

    def test_review_mode_merges_findings(self):
        def dispatcher(messages):
            persona = persona_of(messages)
            if persona == "orchestrator":
                return "MERGED REVIEW"
            return f"review from {persona}"

        runner, _, _ = make_swarm(mode="review", cast=["reviewer", "security"], dispatcher=dispatcher)
        result = runner.run("Review the diff")
        keys = [r.persona_key for r in result.runs]
        self.assertIn("reviewer", keys)
        self.assertIn("security", keys)
        # a synthesis pass merges the findings into one report
        self.assertEqual(result.final_text, "MERGED REVIEW")


class TestHiveMode(unittest.TestCase):
    def _dispatcher(self, tasks, board_ref, review_text="All good, no blockers."):
        state = {"planned": False}

        def dispatcher(messages):
            persona = persona_of(messages)
            user_msgs = [m.text for m in messages if m.role == "user"]
            user = user_msgs[-1] if user_msgs else ""
            if persona == "orchestrator" and not state["planned"]:
                state["planned"] = True
                return {"text": "Plan ready.", "tool_calls": [
                    ("swarm_task", {"action": "add", "title": t}) for t in tasks]}
            if persona == "orchestrator":
                board = board_ref["board"]
                return ("FINAL: integrated summary\n"
                        + board.to_markdown())
            if persona == "reviewer":
                return review_text
            tid = ""
            for line in user.split("\n"):
                if line.startswith("YOUR TASK ("):
                    tid = line.split("(")[1].split(")")[0]
            done = board_ref.setdefault("done", set())
            if tid in done:
                # second turn: the worker has nothing left to do and answers plainly
                return f"{persona} finished {tid}."
            done.add(tid)
            board_ref["executed"].append((persona, tid))
            return {"text": f"{persona} completed {tid}\n```report\nstatus: done\n"
                            f"summary: {tid} done\nartifacts: none\nfollowups: none\nconfidence: high\n```",
                    "tool_calls": [("swarm_task", {"action": "complete", "task_id": tid,
                                                   "result": f"{persona} verified"})]}

        return dispatcher

    def test_hive_plans_executes_and_synthesises(self):
        holder = {"board": None, "executed": []}
        inner = self._dispatcher(["Write parser", "Write tests", "Write docs"], holder)

        def dispatcher(messages):
            return inner(messages)

        runner, _, tmp = make_swarm(mode="hive", cast=["architect", "implementer", "tester", "docs"],
                                    max_parallel=2, dispatcher=dispatcher)
        holder["board"] = runner.board
        result = runner.run("Add a config parser with tests and docs")
        self.assertFalse(result.aborted)
        self.assertEqual(len(runner.board.all()), 3, runner.board.to_markdown())
        self.assertEqual(len([t for t in runner.board.all() if t.status == "done"]), 3,
                         runner.board.to_markdown())
        self.assertIn("FINAL: integrated summary", result.final_text)
        executed_titles = [t for _, t in holder["executed"]]
        self.assertEqual(sorted(executed_titles), sorted(t.id for t in runner.board.all()))
        personas_used = {p for p, _ in holder["executed"]}
        self.assertTrue(personas_used & {"implementer", "tester", "docs", "architect"})

    def test_hive_respects_dependencies(self):
        """A task with depends_on must not be dispatched before its parent is done."""
        holder = {"board": None, "executed": [], "planned": False}

        def dispatcher(messages):
            persona = persona_of(messages)
            user = [m.text for m in messages if m.role == "user"][-1]
            if persona == "orchestrator" and not holder["planned"]:
                holder["planned"] = True
                return {"text": "planned", "tool_calls": [
                    ("swarm_task", {"action": "add", "title": "design schema"}),
                    ("swarm_task", {"action": "add", "title": "implement after design"}),
                ]}
            if persona == "orchestrator":
                return "FINAL"
            if persona == "reviewer":
                return "clean"
            tid = ""
            for line in user.split("\n"):
                if line.startswith("YOUR TASK ("):
                    tid = line.split("(")[1].split(")")[0]
            # link the second task to the first the first time a worker runs
            board = holder["board"]
            tasks = board.all()
            if len(tasks) == 2 and not tasks[1].depends_on:
                board.update(tasks[1].id, depends_on=[tasks[0].id])
            holder["executed"].append(tid)
            return {"text": "done", "tool_calls": [
                ("swarm_task", {"action": "complete", "task_id": tid, "result": "ok"})]}

        runner, _, _ = make_swarm(mode="hive", cast=["architect", "implementer"], dispatcher=dispatcher,
                                  reviewer_gate=False, max_parallel=1)
        holder["board"] = runner.board
        runner.run("Design then implement")
        board = runner.board
        tasks = board.all()
        self.assertEqual(len(tasks), 2)
        self.assertEqual([t.status for t in tasks], ["done", "done"])
        # the dependent task must have been executed after its dependency
        self.assertEqual(holder["executed"][0], tasks[0].id)
        self.assertEqual(holder["executed"][-1], tasks[1].id)

    def test_board_deduplicates_repeated_planning(self):
        holder = {"board": None, "executed": []}
        inner = self._dispatcher(["Dup task"], holder)

        def dispatcher(messages):
            # a misbehaving planner that re-adds the same task on every turn
            holder["planned"] = False
            return inner(messages)

        runner, _, _ = make_swarm(mode="hive", cast=["implementer"], dispatcher=dispatcher,
                                  reviewer_gate=False, plan_only=True)
        holder["board"] = runner.board
        runner.run("plan twice")
        titles = [t.title for t in runner.board.all()]
        self.assertEqual(len(titles), len(set(titles)), f"duplicates survived: {titles}")

    def test_reviewer_gate_creates_fix_tasks(self):
        holder = {"board": None, "executed": []}
        review = "Problems found:\nblocker: parser crashes on empty input, add a guard\nshould-fix: add a test for unicode\n"
        inner = self._dispatcher(["Write parser"], holder, review_text=review)

        def dispatcher(messages):
            return inner(messages)

        runner, _, _ = make_swarm(mode="hive", cast=["implementer", "tester"], max_parallel=1,
                                  dispatcher=dispatcher, reviewer_gate=True)
        holder["board"] = runner.board
        result = runner.run("Add a parser")
        titles = [t.title.lower() for t in runner.board.all()]
        self.assertTrue(any("parser crashes on empty input" in t for t in titles), titles)
        self.assertTrue(any("unicode" in t for t in titles), titles)
        self.assertTrue(any(r.persona_key == "reviewer" for r in result.runs))

    def test_plan_only_stops_after_planning(self):
        holder = {"board": None, "executed": []}
        inner = self._dispatcher(["Task A", "Task B"], holder)

        def dispatcher(messages):
            return inner(messages)

        runner, _, _ = make_swarm(mode="hive", cast=["implementer"], plan_only=True, dispatcher=dispatcher)
        holder["board"] = runner.board
        result = runner.run("Plan only")
        self.assertTrue(result.plan_only)
        self.assertEqual(holder["executed"], [], "no worker may run in plan-only mode")
        self.assertEqual(len(runner.board.all()), 2)

    def test_worker_failure_does_not_crash_the_swarm(self):
        calls = {"n": 0}

        def dispatcher(messages):
            persona = persona_of(messages)
            calls["n"] += 1
            if persona == "orchestrator" and calls["n"] == 1:
                return {"text": "", "tool_calls": [("swarm_task", {"action": "add", "title": "do work"})]}
            if persona == "orchestrator":
                return "FINAL SUMMARY"
            raise RuntimeError("simulated worker crash")

        runner, _, _ = make_swarm(mode="hive", cast=["implementer"], dispatcher=dispatcher,
                                  reviewer_gate=False)
        result = runner.run("Build it")
        self.assertTrue(any(not r.ok for r in result.runs), "the crash must be recorded as a failed run")
        self.assertIn("FINAL SUMMARY", result.final_text)

    def test_cancellation_stops_the_swarm(self):
        holder = {"board": None, "executed": []}
        state = {"workers": 0}
        inner = self._dispatcher(["A", "B", "C", "D"], holder)

        def dispatcher(messages):
            persona = persona_of(messages)
            if persona not in ("orchestrator", "reviewer"):
                state["workers"] += 1
                if state["workers"] >= 2:
                    # deterministic cancellation: the user pressed Ctrl+C mid-swarm
                    holder["runner"].services.cancelled.set()
            return inner(messages)

        runner, _, _ = make_swarm(mode="hive", cast=["implementer"], max_parallel=1, dispatcher=dispatcher)
        holder["board"] = runner.board
        holder["runner"] = runner
        result = runner.run("Long job")
        self.assertTrue(result.aborted)
        self.assertLess(state["workers"], 4, "cancellation must stop the remaining work")

    def test_assignment_prefers_matching_specialist(self):
        runner, _, _ = make_swarm(mode="hive", cast=["architect", "implementer", "tester", "security"])
        cases = {
            "Write unit tests for the parser": "tester",
            "Audit the code for SQL injection vulnerabilities": "security",
            "Design the module interfaces and schema": "architect",
            "Implement the CSV exporter feature": "implementer",
        }
        for title, expected in cases.items():
            got = runner._assign(Task(title=title, id="x"), ["architect", "implementer", "tester", "security"], 1)
            self.assertEqual(got, expected, title)


class TestSwarmAccounting(unittest.TestCase):
    def test_usage_and_cost_aggregate(self):
        def dispatcher(messages):
            return {"text": "x", "usage": {"input_tokens": 10, "output_tokens": 5}}

        runner, _, _ = make_swarm(mode="parallel", cast=["architect", "tester"], dispatcher=dispatcher)
        result = runner.run("question")
        self.assertEqual(result.usage.input_tokens, 20)
        self.assertEqual(result.usage.output_tokens, 10)
        self.assertEqual(result.usage.requests, 2)

    def test_blackboard_records_every_run(self):
        def dispatcher(messages):
            return "some finding"

        runner, _, _ = make_swarm(mode="parallel", cast=["architect", "tester"], dispatcher=dispatcher)
        result = runner.run("question")
        results = runner.blackboard.query(kind="result", limit=50)
        self.assertEqual(len(results), 2)
        self.assertEqual(result.blackboard.stats()["agents"] >= 2, True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
