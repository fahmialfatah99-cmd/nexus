"""Sub-agent and swarm-communication tools.

``task`` is the single most valuable tool in an agentic CLI: it lets the main
agent delegate a self-contained chunk of work to a fresh context window (a
"sub-agent"), keeping its own context clean. The sub-agent runs with the same
permission engine and tool registry, but its own history and its own persona.

``swarm_post`` / ``swarm_read`` / ``swarm_task`` expose the shared blackboard so
swarm members can coordinate; they are only registered while a swarm is running.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..base import ConfirmationRequest, Tool, ToolContext, ToolResult, clip


class TaskTool(Tool):
    name = "task"
    description = (
        "Delegate a self-contained piece of work to a specialist sub-agent running in a FRESH context "
        "window, and get back only its final report. Use it for: deep research across many files, "
        "writing a full module, an independent review, or any subtask whose intermediate noise you do "
        "not want in your own context.\n"
        "The sub-agent sees the workspace and the same tools but NOT your conversation, so the prompt "
        "must be fully self-contained: state the goal, the relevant paths, the constraints and exactly "
        "what to report back. Run one task per call; independent tasks can be issued in parallel."
    )
    parameters = {
        "type": "object",
        "properties": {
            "subagent_type": {
                "type": "string",
                "description": "Specialist to use. Common: general, architect, implementer, reviewer, "
                               "tester, debugger, security, docs, researcher. Use 'general' when unsure.",
                "default": "general",
            },
            "prompt": {"type": "string", "description": "Complete, self-contained instructions."},
            "context": {"type": "string", "description": "Extra background the sub-agent needs."},
            "max_turns": {"type": "integer", "minimum": 1, "maximum": 60, "default": 12},
        },
        "required": ["prompt"],
    }
    category = "agents"
    concurrency_safe = True

    def confirmation(self, args, ctx):
        return ConfirmationRequest(title=f"Spawn sub-agent '{args.get('subagent_type', 'general')}'",
                                   detail=(args.get("prompt") or "")[:160], risk="normal",
                                   key=f"task:{args.get('subagent_type', 'general')}")

    def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        spawn = ctx.vars.get("spawn_subagent")
        if not callable(spawn):
            return ToolResult.fail(
                "Sub-agents are not available in this context (no runtime attached). "
                "Do the work directly with your own tools."
            )
        prompt = (args.get("prompt") or "").strip()
        if not prompt:
            return ToolResult.fail("'prompt' must not be empty.")
        try:
            report = spawn(
                role=args.get("subagent_type") or "general",
                prompt=prompt,
                context=args.get("context") or "",
                max_turns=int(args.get("max_turns") or 12),
                parent=ctx.agent,
            )
        except Exception as exc:  # a failing sub-agent must not kill the parent
            ctx.log.warning("subagent failed", role=args.get("subagent_type"), error=str(exc))
            return ToolResult.fail(f"Sub-agent '{args.get('subagent_type')}' failed: {exc}")
        text = (report or "").strip() or "(sub-agent returned no report)"
        return ToolResult(content=clip(text, ctx.max_output_chars),
                          data={"role": args.get("subagent_type", "general"), "chars": len(text)})


# --------------------------------------------------------------------------- #
# Swarm coordination
# --------------------------------------------------------------------------- #
class SwarmPostTool(Tool):
    name = "swarm_post"
    description = (
        "Post a message to the swarm blackboard so other agents (and the orchestrator) can read it. "
        "Use kind='result' for finished work, 'question' when blocked, 'finding' for information others "
        "need, 'handoff' to request the next step. Keep it short and factual."
    )
    parameters = {
        "type": "object",
        "properties": {
            "content": {"type": "string"},
            "kind": {"type": "string", "enum": ["result", "question", "finding", "handoff", "note"], "default": "note"},
            "to": {"type": "string", "description": "Target agent name, or empty to broadcast."},
            "refs": {"type": "array", "items": {"type": "string"}, "description": "Related file paths or task ids."},
        },
        "required": ["content"],
    }
    category = "swarm"
    concurrency_safe = True

    def run(self, args, ctx):
        board = ctx.vars.get("blackboard")
        if board is None:
            return ToolResult.fail("swarm_post is only available inside a swarm run.")
        entry = board.post(sender=ctx.agent, content=args["content"], kind=args.get("kind", "note"),
                           to=args.get("to") or "", refs=args.get("refs") or [])
        return ToolResult(content=f"Posted {entry.kind} #{entry.id} to the blackboard"
                                   + (f" (to {entry.to})" if entry.to else " (broadcast)"),
                          data={"id": entry.id})


class SwarmReadTool(Tool):
    name = "swarm_read"
    description = "Read recent blackboard messages, optionally filtered by sender, recipient or kind."
    parameters = {
        "type": "object",
        "properties": {
            "limit": {"type": "integer", "minimum": 1, "default": 25},
            "kind": {"type": "string"},
            "from": {"type": "string", "description": "Only messages from this agent."},
            "unread_only": {"type": "boolean", "default": False, "description": "Only messages addressed to me that I have not read."},
        },
    }
    category = "swarm"
    read_only = True
    concurrency_safe = True

    def run(self, args, ctx):
        board = ctx.vars.get("blackboard")
        if board is None:
            return ToolResult.fail("swarm_read is only available inside a swarm run.")
        entries = board.query(limit=int(args.get("limit") or 25), kind=args.get("kind") or "",
                              sender=args.get("from") or "", to=(ctx.agent if args.get("unread_only") else ""))
        if args.get("unread_only"):
            board.mark_read(ctx.agent, [e.id for e in entries])
        if not entries:
            return ToolResult.ok("Blackboard has no matching messages.")
        return ToolResult(content=board.render(entries), data={"count": len(entries)})


class SwarmTaskTool(Tool):
    name = "swarm_task"
    description = (
        "Inspect or update the shared swarm task board. action=list shows all tasks; action=ready shows "
        "tasks whose dependencies are satisfied; action=claim marks a task in_progress for you; "
        "action=complete records the result."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["list", "ready", "claim", "complete", "block", "add"], "default": "list"},
            "task_id": {"type": "string"},
            "title": {"type": "string", "description": "For action=add."},
            "result": {"type": "string", "description": "For action=complete."},
            "reason": {"type": "string", "description": "For action=block."},
            "depends_on": {"type": "array", "items": {"type": "string"}},
        },
    }
    category = "swarm"
    concurrency_safe = True

    def run(self, args, ctx):
        board = ctx.vars.get("board")
        if board is None:
            return ToolResult.fail("swarm_task is only available inside a swarm run.")
        action = args.get("action", "list")
        if action == "list":
            return ToolResult.ok(board.to_markdown(), tasks=board.to_dict())
        if action == "ready":
            ready = board.ready(owner=ctx.agent)
            return ToolResult.ok("\n".join(f"{t.id} {t.title}" for t in ready) or "(no ready tasks)")
        if action == "add":
            title = (args.get("title") or "").strip()
            if not title:
                return ToolResult.fail("action=add requires title.")
            new = board.add(_make_task(title, args.get("depends_on") or []))
            return ToolResult.ok(f"Added task {new.id}: {new.title}\n\n{board.to_markdown()}",
                                 task_id=new.id)
        task_id = (args.get("task_id") or "").strip()
        if not task_id:
            return ToolResult.fail(f"action={action} requires task_id.")
        task = board.get(task_id) or board.find(title=task_id)
        if task is None:
            return ToolResult.fail(f"Unknown task '{task_id}'. Use action=list to see ids.")
        if task.is_terminal and action in ("claim", "complete"):
            # Repeating a finished task wastes turns; tell the agent plainly.
            progress = board.progress()
            return ToolResult.ok(
                f"Task {task.id} is already {task.status} -- nothing to do. "
                f"Board: {progress['done']}/{progress['total']} done. "
                "Move on to your next task or write your final report.",
                already=task.status)
        if action == "claim":
            board.update(task.id, status="in_progress", owner=ctx.agent, attempts=task.attempts + 1)
            return ToolResult.ok(f"Claimed {task.id}: {task.title}\n\n{task.description}")
        if action == "complete":
            board.update(task.id, status="done", owner=ctx.agent, result=(args.get("result") or "")[:4000])
            return ToolResult.ok(f"Completed {task.id}. Remaining: "
                                 f"{board.progress()['total'] - board.progress()['done']} task(s).")
        if action == "block":
            board.update(task.id, status="blocked", owner=ctx.agent,
                         result=(args.get("reason") or "blocked")[:2000])
            return ToolResult.ok(f"Marked {task.id} blocked: {args.get('reason', '')}")
        return ToolResult.fail(f"Unsupported action '{action}'.")


def _make_task(title: str, depends_on: List[str]):
    from ...core.taskboard import Task

    return Task(title=title, depends_on=list(depends_on), owner="")


def build_tools() -> List[Tool]:
    return [TaskTool()]


def build_swarm_tools() -> List[Tool]:
    return [SwarmPostTool(), SwarmReadTool(), SwarmTaskTool()]
