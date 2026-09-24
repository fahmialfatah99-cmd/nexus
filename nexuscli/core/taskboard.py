"""Thread-safe task board with dependencies.

Shared by three features:
* the ``todo_write`` / ``todo_read`` tools (single-agent planning),
* the swarm orchestrator (task graph with owners and dependencies),
* session persistence (serialisable to/from plain dicts).

Keeping this in ``core`` means the swarm scheduler and the todo tool can never
drift apart in how they represent work.
"""

from __future__ import annotations

import itertools
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

VALID_STATUS = ("pending", "in_progress", "done", "blocked", "cancelled")
VALID_PRIORITY = ("low", "normal", "high", "critical")

_counter = itertools.count(1)


def _next_id() -> str:
    return f"t{next(_counter)}"


@dataclass
class Task:
    title: str
    description: str = ""
    id: str = field(default_factory=_next_id)
    status: str = "pending"
    owner: str = ""
    priority: str = "normal"
    depends_on: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    result: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    attempts: int = 0

    def __post_init__(self) -> None:
        if self.status not in VALID_STATUS:
            self.status = "pending"
        if self.priority not in VALID_PRIORITY:
            self.priority = "normal"
        self.depends_on = [d for d in (self.depends_on or []) if d]

    @property
    def is_terminal(self) -> bool:
        return self.status in ("done", "cancelled")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Task":
        allowed = {f for f in Task.__dataclass_fields__}  # type: ignore[attr-defined]
        return Task(**{k: v for k, v in data.items() if k in allowed})


class TaskBoard:
    def __init__(self, tasks: Optional[Iterable[Task]] = None) -> None:
        self._tasks: Dict[str, Task] = {}
        self._order: List[str] = []
        self._lock = threading.RLock()
        for t in tasks or []:
            self.add(t)

    # -- mutation ---------------------------------------------------------
    def add(self, task: Task) -> Task:
        with self._lock:
            if task.id in self._tasks:
                task.id = _next_id()
                while task.id in self._tasks:
                    task.id = _next_id()
            self._tasks[task.id] = task
            self._order.append(task.id)
            return task

    def add_many(self, tasks: Iterable[Task]) -> List[Task]:
        return [self.add(t) for t in tasks]

    def update(self, task_id: str, **fields: Any) -> Optional[Task]:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            for k, v in fields.items():
                if not hasattr(task, k) or v is None:
                    continue
                if k == "status" and v not in VALID_STATUS:
                    continue
                if k == "priority" and v not in VALID_PRIORITY:
                    continue
                setattr(task, k, v)
            task.updated_at = time.time()
            return task

    def remove(self, task_id: str) -> bool:
        with self._lock:
            if task_id not in self._tasks:
                return False
            del self._tasks[task_id]
            self._order = [i for i in self._order if i != task_id]
            for t in self._tasks.values():
                t.depends_on = [d for d in t.depends_on if d != task_id]
            return True

    def replace_all(self, tasks: Sequence[Task]) -> List[Task]:
        with self._lock:
            self._tasks.clear()
            self._order.clear()
            return [self.add(t) for t in tasks]

    # -- queries ----------------------------------------------------------
    def get(self, task_id: str) -> Optional[Task]:
        with self._lock:
            return self._tasks.get(task_id)

    def find(self, *, title: Optional[str] = None, task_id: Optional[str] = None) -> Optional[Task]:
        with self._lock:
            if task_id and task_id in self._tasks:
                return self._tasks[task_id]
            if title:
                needle = title.strip().lower()
                for tid in self._order:
                    t = self._tasks[tid]
                    if t.title.strip().lower() == needle:
                        return t
                for tid in self._order:
                    t = self._tasks[tid]
                    if needle in t.title.strip().lower():
                        return t
            return None

    def all(self) -> List[Task]:
        with self._lock:
            return [self._tasks[i] for i in self._order if i in self._tasks]

    def by_status(self, status: str) -> List[Task]:
        return [t for t in self.all() if t.status == status]

    def ready(self, *, owner: Optional[str] = None) -> List[Task]:
        """Tasks whose dependencies are satisfied and that nobody has claimed."""
        with self._lock:
            done = {t.id for t in self._tasks.values() if t.status == "done"}
            cancelled = {t.id for t in self._tasks.values() if t.status == "cancelled"}
            out = []
            for tid in self._order:
                t = self._tasks.get(tid)
                if t is None or t.status != "pending":
                    continue
                if owner and t.owner and t.owner != owner:
                    continue
                blocked = [d for d in t.depends_on if d in self._tasks and d not in done and d not in cancelled]
                if blocked:
                    continue
                out.append(t)
            rank = {p: i for i, p in enumerate(("critical", "high", "normal", "low"))}
            out.sort(key=lambda t: (rank.get(t.priority, 2), t.created_at))
            return out

    def has_cycle(self) -> bool:
        """Detect dependency cycles (a cycle would deadlock the scheduler)."""
        with self._lock:
            graph = {tid: list(t.depends_on) for tid, t in self._tasks.items()}
        state: Dict[str, int] = {}

        def visit(node: str) -> bool:
            if state.get(node) == 1:
                return True
            if state.get(node) == 2:
                return False
            state[node] = 1
            for dep in graph.get(node, []):
                if dep in graph and visit(dep):
                    return True
            state[node] = 2
            return False

        return any(visit(n) for n in list(graph))

    def progress(self) -> Dict[str, int]:
        tasks = self.all()
        counts: Dict[str, int] = {s: 0 for s in VALID_STATUS}
        for t in tasks:
            counts[t.status] = counts.get(t.status, 0) + 1
        counts["total"] = len(tasks)
        return counts

    def percent_done(self) -> float:
        tasks = self.all()
        if not tasks:
            return 0.0
        return 100.0 * sum(1 for t in tasks if t.status == "done") / len(tasks)

    # -- serialisation ----------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {"order": list(self._order), "tasks": [self._tasks[i].to_dict() for i in self._order if i in self._tasks]}

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "TaskBoard":
        board = TaskBoard()
        for item in (data or {}).get("tasks") or []:
            try:
                board.add(Task.from_dict(item))
            except (TypeError, ValueError):
                continue
        return board

    def to_markdown(self) -> str:
        icons = {"pending": "[ ]", "in_progress": "[~]", "done": "[x]", "blocked": "[!]", "cancelled": "[-]"}
        lines = []
        for t in self.all():
            deps = f" (after {', '.join(t.depends_on)})" if t.depends_on else ""
            owner = f" @{t.owner}" if t.owner else ""
            lines.append(f"{icons.get(t.status, '[ ]')} {t.id} {t.title}{owner}{deps}")
            if t.result:
                lines.append(f"      -> {t.result.splitlines()[0][:160]}")
        return "\n".join(lines) if lines else "(no tasks)"


__all__ = ["Task", "TaskBoard", "VALID_STATUS", "VALID_PRIORITY"]
