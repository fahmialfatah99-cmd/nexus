"""Usage & cost ledger.

Accumulates tokens/cost per provider, model and agent, so ``/usage`` can answer
"what did this session cost and which swarm member burned it". Costs come from
the model catalogue and are labelled as estimates -- unknown models report 0
rather than a made-up number.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..providers.base import ModelInfo, Usage


@dataclass
class LedgerEntry:
    ts: float
    provider: str
    model: str
    agent: str
    usage: Usage
    cost: float
    latency_ms: int = 0
    turns: int = 1


@dataclass
class UsageLedger:
    entries: List[LedgerEntry] = field(default_factory=list)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def record(self, *, provider: str, model: str, usage: Usage, info: Optional[ModelInfo] = None,
               agent: str = "main", latency_ms: int = 0, turns: int = 1) -> LedgerEntry:
        cost = info.cost(usage) if info else 0.0
        entry = LedgerEntry(ts=time.time(), provider=provider, model=model, agent=agent,
                            usage=usage, cost=cost, latency_ms=latency_ms, turns=turns)
        with self._lock:
            self.entries.append(entry)
            if len(self.entries) > 20_000:
                self.entries = self.entries[-10_000:]
        return entry

    # -- aggregates -------------------------------------------------------
    def totals(self) -> Dict[str, Any]:
        with self._lock:
            entries = list(self.entries)
        usage = Usage(requests=0)
        cost = 0.0
        latency = 0
        for e in entries:
            usage.input_tokens += e.usage.input_tokens
            usage.output_tokens += e.usage.output_tokens
            usage.cached_tokens += e.usage.cached_tokens
            usage.reasoning_tokens += e.usage.reasoning_tokens
            usage.requests += e.usage.requests
            cost += e.cost
            latency += e.latency_ms
        return {"requests": usage.requests, "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens, "cached_tokens": usage.cached_tokens,
                "reasoning_tokens": usage.reasoning_tokens, "total_tokens": usage.total_tokens,
                "cost_usd": round(cost, 6), "latency_ms": latency, "entries": len(entries)}

    def by_model(self) -> List[Dict[str, Any]]:
        return self._group(lambda e: f"{e.provider}:{e.model}")

    def by_agent(self) -> List[Dict[str, Any]]:
        return self._group(lambda e: e.agent)

    def _group(self, keyfn) -> List[Dict[str, Any]]:
        with self._lock:
            entries = list(self.entries)
        buckets: Dict[str, Dict[str, Any]] = {}
        for e in entries:
            k = keyfn(e)
            b = buckets.setdefault(k, {"key": k, "requests": 0, "input_tokens": 0, "output_tokens": 0,
                                       "total_tokens": 0, "cost_usd": 0.0, "latency_ms": 0})
            b["requests"] += e.usage.requests
            b["input_tokens"] += e.usage.input_tokens
            b["output_tokens"] += e.usage.output_tokens
            b["total_tokens"] += e.usage.total_tokens
            b["cost_usd"] += e.cost
            b["latency_ms"] += e.latency_ms
        out = list(buckets.values())
        for b in out:
            b["cost_usd"] = round(b["cost_usd"], 6)
        out.sort(key=lambda b: -b["total_tokens"])
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {"totals": self.totals(), "by_model": self.by_model(), "by_agent": self.by_agent()}

    def reset(self) -> None:
        with self._lock:
            self.entries.clear()


__all__ = ["UsageLedger", "LedgerEntry"]
