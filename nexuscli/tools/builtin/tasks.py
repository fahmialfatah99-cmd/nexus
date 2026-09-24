"""Planning tools backed by the shared :class:`TaskBoard`.

``todo_write`` is the same board the swarm scheduler uses, so a single agent's
plan and a swarm's task graph are literally the same object -- which is what
makes "promote this todo into a swarm task" possible without translation.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ...core.taskboard import VALID_PRIORITY, VALID_STATUS, Task, TaskBoard
from ..base import Tool, ToolContext, ToolResult


def board_of(ctx: ToolContext) -> TaskBoard:
    board = ctx.vars.get("board")
    if board is None:
        board = TaskBoard()
        ctx.vars["board"] = board
    return board


class TodoWriteTool(Tool):
    name = "todo_write"
    description = (
        "Create or update the task plan for the current job. Pass the FULL list you want to exist "
        "(it replaces the current plan) or use `merge=true` to add/update individual items.\n"
        "Use this for any multi-step work: it keeps you on track and shows the user live progress. "
        "Set exactly one task to in_progress at a time, and mark tasks done as soon as they are."
    )
    parameters = {
        "type": "object",
        "properties": {
            "todos": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "description": "Existing task id when updating."},
                        "content": {"type": "string", "description": "Short imperative description of the task."},
                        "status": {"type": "string", "enum": list(VALID_STATUS), "default": "pending"},
                        "priority": {"type": "string", "enum": list(VALID_PRIORITY), "default": "normal"},
                        "owner": {"type": "string", "description": "Agent name (swarm mode)."},
                        "depends_on": {"type": "array", "items": {"type": "string"}, "description": "Task ids that must finish first."},
                        "result": {"type": "string", "description": "Outcome note, filled when completing a task."},
                    },
                    "required": ["content"],
                },
            },
            "merge": {"type": "boolean", "description": "Update/add instead of replacing the whole plan.", "default": False},
        },
        "required": ["todos"],
    }
    category = "planning"
    concurrency_safe = False

    def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        board = board_of(ctx)
        items: List[Dict[str, Any]] = args.get("todos") or []
        if not items:
            return ToolResult.fail("No todos supplied.")
        if args.get("merge"):
            changed = 0
            for item in items:
                title = (item.get("content") or "").strip()
                if not title:
                    continue
                existing = board.find(task_id=item.get("id"), title=title)
                if existing is None:
                    board.add(Task(title=title, description=item.get("description", ""),
                                   status=item.get("status", "pending"), priority=item.get("priority", "normal"),
                                   owner=item.get("owner", ctx.agent), depends_on=item.get("depends_on") or [],
                                   result=item.get("result", "")))
                else:
                    board.update(existing.id, status=item.get("status"), priority=item.get("priority"),
                                 owner=item.get("owner"), result=item.get("result"),
                                 depends_on=item.get("depends_on"))
                changed += 1
            action = f"Merged {changed} task(s)"
        else:
            tasks = [Task(title=(i.get("content") or "").strip() or "(untitled)",
                          description=i.get("description", ""),
                          status=i.get("status", "pending"), priority=i.get("priority", "normal"),
                          owner=i.get("owner", "") or ctx.agent, result=i.get("result", ""))
                     for i in items]
            # The model cannot know ids before the tasks exist, so accepts ids,
            # titles or 1-based positions and resolves them to the real ids.
            unresolved = _resolve_dependencies(items, tasks)
            board.replace_all(tasks)
            action = f"Plan set to {len(tasks)} task(s)"
            if unresolved:
                return ToolResult.fail(
                    "Unknown depends_on reference(s): " + ", ".join(unresolved)
                    + ". Use a task title, its 1-based position in this list, or an existing task id.\n\n"
                    + board.to_markdown())
        if board.has_cycle():
            return ToolResult.fail("Dependency cycle detected in the plan; fix depends_on. Current plan:\n"
                                   + board.to_markdown())
        progress = board.progress()
        ctx.vars["board_dirty"] = True
        return ToolResult(content=f"{action} ({progress['done']}/{progress['total']} done)\n\n{board.to_markdown()}",
                          data={"progress": progress, "tasks": board.to_dict()})


def _resolve_dependencies(items: List[Dict[str, Any]], tasks: List[Task]) -> List[str]:
    """Map each payload's depends_on entries onto real task ids."""
    by_payload_id = {str(i.get("id")): t.id for i, t in zip(items, tasks) if i.get("id")}
    by_title = {" ".join(t.title.lower().split()): t.id for t in tasks}
    by_index = {str(n): t.id for n, t in enumerate(tasks, start=1)}
    unresolved: List[str] = []
    for item, task in zip(items, tasks):
        resolved: List[str] = []
        for dep in item.get("depends_on") or []:
            key = str(dep).strip()
            target = (by_payload_id.get(key) or by_title.get(" ".join(key.lower().split()))
                      or by_index.get(key))
            if target:
                if target != task.id:
                    resolved.append(target)
            else:
                unresolved.append(f"{key} (in '{task.title}')")
        task.depends_on = resolved
    return unresolved


class TodoReadTool(Tool):
    name = "todo_read"
    description = "Show the current task plan with statuses, owners and dependencies."
    parameters = {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["", *VALID_STATUS], "description": "Filter by status.", "default": ""},
        },
    }
    category = "planning"
    read_only = True

    def run(self, args, ctx):
        board = board_of(ctx)
        status = (args.get("status") or "").strip()
        tasks = board.by_status(status) if status else board.all()
        if not tasks:
            return ToolResult.ok("Plan is empty. Use todo_write to create one.")
        icons = {"pending": "[ ]", "in_progress": "[~]", "done": "[x]", "blocked": "[!]", "cancelled": "[-]"}
        lines = [f"{icons.get(t.status, '[ ]')} {t.id} {t.title}"
                 + (f" @{t.owner}" if t.owner else "")
                 + (f" (after {', '.join(t.depends_on)})" if t.depends_on else "")
                 + (f"\n      -> {t.result}" if t.result else "") for t in tasks]
        progress = board.progress()
        head = f"{progress['done']}/{progress['total']} done ({board.percent_done():.0f}%)"
        return ToolResult(content=f"{head}\n\n" + "\n".join(lines), data={"progress": progress})


def build_tools() -> List[Tool]:
    return [TodoWriteTool(), TodoReadTool()]
