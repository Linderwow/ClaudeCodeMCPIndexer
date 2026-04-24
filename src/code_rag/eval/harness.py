"""Offline retrieval eval.

Input:  a JSON file of {query, expected: [{path, symbol?}]} pairs.
Output: Recall@1/3/10, MRR, p50/p95 latency, per-query breakdown.

Design goals
------------
- Self-contained: point it at an already-indexed store; it doesn't rebuild.
- Non-flaky: deterministic ordering given same index contents.
- Cheap to run: 30-pair suite finishes in seconds; suitable for CI or pre-commit.
- Signal-preserving: expected match is "path matches" OR "path+symbol matches"
  so fixtures are easy to author without over-specifying line ranges.
"""
from __future__ import annotations

import asyncio
import json
import statistics
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from code_rag.logging import get
from code_rag.models import SearchHit
from code_rag.retrieval.search import HybridSearcher, SearchParams

log = get(__name__)


@dataclass
class ExpectedHit:
    path: str                     # exact path match (posix-style)
    symbol: str | None = None     # optional — if present, symbol must match exactly

    def matches(self, hit: SearchHit) -> bool:
        if hit.chunk.path != self.path:
            return False
        if self.symbol is None:
            return True
        return hit.chunk.symbol == self.symbol


@dataclass
class EvalCase:
    query: str
    expected: list[ExpectedHit]

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EvalCase:
        return cls(
            query=str(d["query"]),
            expected=[
                ExpectedHit(path=str(e["path"]), symbol=e.get("symbol"))
                for e in d["expected"]
            ],
        )


@dataclass
class EvalResult:
    query: str
    rank: int | None              # 1-based rank of first expected hit, None if miss
    latency_ms: float
    top_paths: list[str] = field(default_factory=list)

    @property
    def hit(self) -> bool:
        return self.rank is not None


@dataclass
class EvalReport:
    cases: list[EvalResult]

    @property
    def recall_at_1(self) -> float:
        return self._recall_at(1)

    @property
    def recall_at_3(self) -> float:
        return self._recall_at(3)

    @property
    def recall_at_10(self) -> float:
        return self._recall_at(10)

    @property
    def mrr(self) -> float:
        if not self.cases:
            return 0.0
        total = sum(1.0 / r.rank if r.rank is not None else 0.0 for r in self.cases)
        return total / len(self.cases)

    @property
    def p50_latency_ms(self) -> float:
        lat = sorted(r.latency_ms for r in self.cases)
        return statistics.median(lat) if lat else 0.0

    @property
    def p95_latency_ms(self) -> float:
        lat = sorted(r.latency_ms for r in self.cases)
        if not lat:
            return 0.0
        idx = max(0, round(0.95 * (len(lat) - 1)))
        return lat[idx]

    def _recall_at(self, k: int) -> float:
        if not self.cases:
            return 0.0
        return sum(1 for r in self.cases if r.rank is not None and r.rank <= k) / len(self.cases)

    def summary(self) -> dict[str, Any]:
        return {
            "n":              len(self.cases),
            "recall@1":       round(self.recall_at_1, 4),
            "recall@3":       round(self.recall_at_3, 4),
            "recall@10":      round(self.recall_at_10, 4),
            "mrr":            round(self.mrr, 4),
            "p50_latency_ms": round(self.p50_latency_ms, 1),
            "p95_latency_ms": round(self.p95_latency_ms, 1),
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary(),
            "cases": [
                {"query": c.query, "rank": c.rank, "latency_ms": round(c.latency_ms, 1),
                 "top_paths": c.top_paths[:5]}
                for c in self.cases
            ],
        }


def load_cases(path: Path) -> list[EvalCase]:
    data = json.loads(path.read_text("utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"eval file must be a JSON array, got {type(data).__name__}")
    return [EvalCase.from_dict(d) for d in data]


async def run_eval(
    cases: Iterable[EvalCase],
    searcher: HybridSearcher,
    *,
    top_k: int = 10,
    rerank_in: int = 50,
) -> EvalReport:
    params = SearchParams(k_final=top_k, k_vector=rerank_in, k_lexical=rerank_in,
                          k_rerank_in=rerank_in)
    results: list[EvalResult] = []
    for case in cases:
        t0 = time.monotonic()
        hits = await searcher.search(case.query, params)
        dur_ms = (time.monotonic() - t0) * 1000

        rank: int | None = None
        for i, h in enumerate(hits, start=1):
            if any(e.matches(h) for e in case.expected):
                rank = i
                break
        results.append(EvalResult(
            query=case.query,
            rank=rank,
            latency_ms=dur_ms,
            top_paths=[h.chunk.path for h in hits],
        ))
    return EvalReport(cases=results)


# ---- synchronous convenience for CLI --------------------------------------


def run_eval_sync(cases: Iterable[EvalCase], searcher: HybridSearcher, **kwargs: Any) -> EvalReport:
    return asyncio.run(run_eval(cases, searcher, **kwargs))
