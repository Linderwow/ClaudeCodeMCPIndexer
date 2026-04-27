"""Offline retrieval eval.

Input:  a JSON file of {query, expected: [{path, symbol?}]} pairs.
Output: Recall@1/3/10, NDCG@10, MRR, p50/p95/p99 latency, per-query breakdown,
        and (optionally) a per-tag breakdown when cases declare a `tag` field.

Design goals
------------
- Self-contained: point it at an already-indexed store; it doesn't rebuild.
- Non-flaky: deterministic ordering given same index contents.
- Cheap to run: 30-pair suite finishes in seconds; suitable for CI or pre-commit.
- Signal-preserving: expected match is "path matches" OR "path+symbol matches"
  so fixtures are easy to author without over-specifying line ranges.
- Tag-aware: each case can declare an optional `tag` (e.g. "identifier" /
  "natural_language" / "graph_walk") so we can see which kinds of queries
  regress when an embedder/reranker swaps out.
"""
from __future__ import annotations

import asyncio
import json
import math
import time
from collections import defaultdict
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
    symbol: str | None = None     # optional -- if present, symbol must match exactly

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
    tag: str | None = None        # optional category for breakdown reporting

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EvalCase:
        return cls(
            query=str(d["query"]),
            expected=[
                ExpectedHit(path=str(e["path"]), symbol=e.get("symbol"))
                for e in d["expected"]
            ],
            tag=(str(d["tag"]) if d.get("tag") else None),
        )


@dataclass
class EvalResult:
    query: str
    rank: int | None              # 1-based rank of first expected hit, None if miss
    latency_ms: float
    top_paths: list[str] = field(default_factory=list)
    tag: str | None = None

    @property
    def hit(self) -> bool:
        return self.rank is not None


@dataclass
class EvalReport:
    cases: list[EvalResult]
    label: str | None = None      # e.g. "baseline" / "after BGE swap"
    index_meta: dict[str, Any] | None = None

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
    def ndcg_at_10(self) -> float:
        """NDCG@10 with binary relevance.

        Each case has exactly ONE relevant document (the expected path), so:
            DCG_i  = 1 / log2(rank + 1) if rank <= 10 else 0
            IDCG_i = 1 / log2(2)       (rank-1 being best)
        Mean across cases.
        """
        if not self.cases:
            return 0.0
        idcg = 1.0 / math.log2(2)
        total = 0.0
        for r in self.cases:
            if r.rank is not None and r.rank <= 10:
                dcg = 1.0 / math.log2(r.rank + 1)
                total += dcg / idcg
        return total / len(self.cases)

    @property
    def p50_latency_ms(self) -> float:
        return self._percentile(0.50)

    @property
    def p95_latency_ms(self) -> float:
        return self._percentile(0.95)

    @property
    def p99_latency_ms(self) -> float:
        return self._percentile(0.99)

    def _percentile(self, q: float) -> float:
        lat = sorted(r.latency_ms for r in self.cases)
        if not lat:
            return 0.0
        idx = max(0, round(q * (len(lat) - 1)))
        return lat[idx]

    def _recall_at(self, k: int) -> float:
        if not self.cases:
            return 0.0
        return sum(1 for r in self.cases if r.rank is not None and r.rank <= k) / len(self.cases)

    def per_tag(self) -> dict[str, dict[str, Any]]:
        """Group results by `tag` (or "untagged") and compute per-group metrics.
        Useful for spotting embedder swaps that help one query type but hurt
        another."""
        groups: dict[str, list[EvalResult]] = defaultdict(list)
        for r in self.cases:
            groups[r.tag or "untagged"].append(r)
        out: dict[str, dict[str, Any]] = {}
        for tag, results in sorted(groups.items()):
            sub = EvalReport(cases=results)
            out[tag] = {
                "n":         len(results),
                "recall@1":  round(sub.recall_at_1, 4),
                "recall@3":  round(sub.recall_at_3, 4),
                "recall@10": round(sub.recall_at_10, 4),
                "mrr":       round(sub.mrr, 4),
                "ndcg@10":   round(sub.ndcg_at_10, 4),
            }
        return out

    def summary(self) -> dict[str, Any]:
        return {
            "n":              len(self.cases),
            "recall@1":       round(self.recall_at_1, 4),
            "recall@3":       round(self.recall_at_3, 4),
            "recall@10":      round(self.recall_at_10, 4),
            "mrr":            round(self.mrr, 4),
            "ndcg@10":        round(self.ndcg_at_10, 4),
            "p50_latency_ms": round(self.p50_latency_ms, 1),
            "p95_latency_ms": round(self.p95_latency_ms, 1),
            "p99_latency_ms": round(self.p99_latency_ms, 1),
        }

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "label":      self.label,
            "index_meta": self.index_meta,
            "summary":    self.summary(),
            "per_tag":    self.per_tag(),
            "cases": [
                {"query": c.query, "rank": c.rank,
                 "latency_ms": round(c.latency_ms, 1),
                 "tag": c.tag,
                 "top_paths": c.top_paths[:5]}
                for c in self.cases
            ],
        }
        return out

    def diff(self, baseline: EvalReport) -> dict[str, Any]:
        """Compute pp-deltas for every headline metric vs a baseline report.
        Lets a CI gate fail commits that drop quality."""
        b = baseline.summary()
        c = self.summary()
        return {
            "label":     f"{self.label or 'current'} vs {baseline.label or 'baseline'}",
            "n":         c["n"],
            "deltas_pp": {
                k: round((c[k] - b[k]) * 100, 2)
                for k in ("recall@1", "recall@3", "recall@10", "mrr", "ndcg@10")
                if k in b and k in c
            },
            "deltas_ms": {
                k: round(c[k] - b[k], 1)
                for k in ("p50_latency_ms", "p95_latency_ms", "p99_latency_ms")
                if k in b and k in c
            },
        }


def load_cases(path: Path) -> list[EvalCase]:
    data = json.loads(path.read_text("utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"eval file must be a JSON array, got {type(data).__name__}")
    return [EvalCase.from_dict(d) for d in data]


def categorize_misses(
    report: "EvalReport",
    indexed_paths: set[str],
) -> dict[str, list[dict[str, Any]]]:
    """Phase 35 (A4): per-miss diagnostic — explain each failed query.

    For every case that didn't hit top-10, classify into one of:

      * "ground_truth_not_in_index"
            The expected path doesn't exist in the indexed corpus at all.
            Re-mine the eval set — this isn't a retrieval failure.

      * "retrieved_but_wrong_path"
            Top-10 has 10 paths but none match expected. Real retrieval
            miss. Either the chunker missed the symbol, or fusion/rerank
            ranked the wrong chunks higher.

      * "no_results"
            Search returned an empty top-10 (probably min_score gate or
            crash). Investigate the case query for adversarial input.

    Returns a dict with one list per category, each entry shaped like:
        {"query": "...", "expected_paths": [...], "top_paths": [...],
         "tag": "..."}

    Use the output to drive surgical fixes vs. shotgun tuning.
    """
    cats: dict[str, list[dict[str, Any]]] = {
        "ground_truth_not_in_index": [],
        "retrieved_but_wrong_path":  [],
        "no_results":                [],
    }
    # We need the original cases to know expected_paths. Encode them in
    # the report's casewise top_paths field as a side effect — see EvalResult.
    for r in report.cases:
        if r.hit:
            continue
        # The harness loses the expected_paths after run_eval; the caller
        # must pass cases via the report's debug attr if they want full
        # diagnostics. Default: just split into "no_results" vs other.
        if not r.top_paths:
            cats["no_results"].append({
                "query": r.query, "tag": r.tag, "top_paths": [],
            })
            continue
        # Without access to expected paths here we can't tell
        # "ground_truth_not_in_index" from "retrieved_but_wrong_path"
        # in this lightweight mode. Default to wrong_path.
        cats["retrieved_but_wrong_path"].append({
            "query": r.query, "tag": r.tag,
            "top_paths": r.top_paths[:5],
        })
    return cats


def diagnose_misses_with_cases(
    cases: Iterable[EvalCase],
    report: "EvalReport",
    indexed_paths: set[str],
) -> dict[str, list[dict[str, Any]]]:
    """Richer diagnostic that needs both the original cases AND the report.

    Distinguishes "ground_truth_not_in_index" from
    "retrieved_but_wrong_path" using the actual indexed_paths set.
    """
    case_by_query = {c.query: c for c in cases}
    cats: dict[str, list[dict[str, Any]]] = {
        "ground_truth_not_in_index": [],
        "retrieved_but_wrong_path":  [],
        "no_results":                [],
    }
    for r in report.cases:
        if r.hit:
            continue
        c = case_by_query.get(r.query)
        if c is None:
            continue
        expected_paths = [e.path for e in c.expected]
        in_index = [p for p in expected_paths if p in indexed_paths]
        out_of_index = [p for p in expected_paths if p not in indexed_paths]

        if not r.top_paths:
            cats["no_results"].append({
                "query": r.query, "tag": r.tag,
                "expected_paths": expected_paths,
                "top_paths": [],
            })
        elif not in_index:
            cats["ground_truth_not_in_index"].append({
                "query": r.query, "tag": r.tag,
                "expected_paths": expected_paths,
                "missing_paths": out_of_index,
                "top_paths": r.top_paths[:5],
            })
        else:
            cats["retrieved_but_wrong_path"].append({
                "query": r.query, "tag": r.tag,
                "expected_paths": in_index,
                "top_paths": r.top_paths[:5],
            })
    return cats


def filter_cases_to_paths(
    cases: Iterable[EvalCase], indexed_paths: set[str],
) -> list[EvalCase]:
    """Phase 26: drop cases whose expected paths are all outside the index.

    Mined cases reference files Claude opened from worktree clones, deleted
    files, or roots not currently configured. Including them would artificially
    depress recall metrics — the retriever cannot return a chunk that doesn't
    exist. This filter keeps only the in-corpus subset, giving a true ceiling.

    Cases with at least one in-index expected path are kept (with the missing
    expected paths pruned). Cases with zero in-index paths are dropped.
    """
    out: list[EvalCase] = []
    for c in cases:
        keep = [e for e in c.expected if e.path in indexed_paths]
        if keep:
            out.append(EvalCase(query=c.query, expected=keep, tag=c.tag))
    return out


async def run_eval(
    cases: Iterable[EvalCase],
    searcher: HybridSearcher,
    *,
    top_k: int = 10,
    rerank_in: int = 50,
    label: str | None = None,
    index_meta: dict[str, Any] | None = None,
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
            tag=case.tag,
        ))
    return EvalReport(cases=results, label=label, index_meta=index_meta)


# ---- synchronous convenience for CLI --------------------------------------


def run_eval_sync(cases: Iterable[EvalCase], searcher: HybridSearcher, **kwargs: Any) -> EvalReport:
    return asyncio.run(run_eval(cases, searcher, **kwargs))
