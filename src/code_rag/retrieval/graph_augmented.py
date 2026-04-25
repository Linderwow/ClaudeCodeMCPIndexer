"""Phase 21: graph-augmented retrieval.

Pure semantic search ranks chunks by literal similarity to the query.
Pure graph traversal returns "everything within N hops of X" but is
unranked. Combining them -- search + graph radius -- answers questions like:

    "What helpers does OnBarUpdate use that touch session phase?"
    "Find regime-classification functions called by FindTrade"
    "Show me code Audited by RunGhostEvaluation that writes CSVs"

Algorithm
---------
Given (query, anchor_symbol, hops):
    1. Resolve `anchor_symbol` to {path, symbol} via the graph store.
    2. BFS up to `hops` along call+import+contains edges → set of related
       (path, symbol) tuples.
    3. Run the regular hybrid search on `query`.
    4. Boost hits whose (path, symbol) is in the related set, decaying by
       graph distance.
    5. Re-rank by boosted score.

`anchor_symbol` is optional: if None, fall back to plain hybrid search
(making this MCP tool a strict superset of `search_code`). The boost
formula is intentionally gentle: a hit at distance 1 gets ~2x its base
score, distance 2 ~1.5x, etc. -- enough to surface relevant code but not
so aggressive that it drowns out genuinely better semantic matches.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from code_rag.interfaces.graph_store import GraphStore, SymbolRef
from code_rag.logging import get
from code_rag.models import SearchHit
from code_rag.retrieval.search import HybridSearcher, SearchParams, SearchResponse

log = get(__name__)


@dataclass
class AnchoredSearchResult:
    response: SearchResponse
    anchor_symbol: str | None
    anchor_resolved: list[SymbolRef]
    related_set_size: int  # how many (path, symbol) tuples were within radius

    def as_dict(self) -> dict[str, object]:
        return {
            "anchor_symbol": self.anchor_symbol,
            "anchor_resolved": [
                {"path": s.path, "symbol": s.symbol, "kind": s.kind}
                for s in self.anchor_resolved
            ],
            "related_set_size": self.related_set_size,
            "no_confident_match": self.response.no_confident_match,
            "elapsed_ms": round(self.response.elapsed_ms, 1),
            "hits": [
                {
                    "chunk_id": h.chunk.id,
                    "path": h.chunk.path,
                    "symbol": h.chunk.symbol,
                    "kind": h.chunk.kind.value,
                    "start_line": h.chunk.start_line,
                    "end_line": h.chunk.end_line,
                    "score": round(h.score, 4),
                    "match_reason": h.match_reason,
                    "text": h.chunk.text,
                }
                for h in self.response.hits
            ],
        }


class GraphAugmentedSearcher:
    """Wraps a HybridSearcher and re-ranks results using graph-distance to
    an anchor symbol. Falls back to plain hybrid search when no anchor is
    given OR the anchor doesn't resolve."""

    def __init__(self, hybrid: HybridSearcher, graph: GraphStore | None) -> None:
        self._hybrid = hybrid
        self._graph = graph

    async def search(
        self,
        query: str,
        params: SearchParams,
        anchor_symbol: str | None = None,
        hops: int = 1,
        anchor_path: str | None = None,
    ) -> AnchoredSearchResult:
        # 1) Plain hybrid search first.
        response = await self._hybrid.search_full(query, params)

        # 2) If no anchor or no graph, return as-is.
        if not anchor_symbol or self._graph is None:
            return AnchoredSearchResult(
                response=response,
                anchor_symbol=anchor_symbol,
                anchor_resolved=[],
                related_set_size=0,
            )

        # 3) Resolve anchor + collect graph radius.
        try:
            anchors = self._graph.find_symbol(anchor_symbol, path=anchor_path)
        except Exception as e:
            log.warning("graph_augmented.resolve_failed", err=str(e))
            return AnchoredSearchResult(
                response=response,
                anchor_symbol=anchor_symbol,
                anchor_resolved=[],
                related_set_size=0,
            )
        if not anchors:
            log.debug("graph_augmented.anchor_unresolved", anchor=anchor_symbol)
            return AnchoredSearchResult(
                response=response,
                anchor_symbol=anchor_symbol,
                anchor_resolved=[],
                related_set_size=0,
            )

        related = self._collect_radius(anchors, hops=max(1, min(hops, 4)))

        # 4) Re-score hits with graph proximity boost.
        boosted = self._apply_graph_boost(response.hits, related)

        # 5) Repackage. Hits are now in re-ranked order.
        new_response = SearchResponse(
            hits=boosted,
            no_confident_match=response.no_confident_match,
            elapsed_ms=response.elapsed_ms,
            neighborhood=response.neighborhood,
        )
        return AnchoredSearchResult(
            response=new_response,
            anchor_symbol=anchor_symbol,
            anchor_resolved=anchors,
            related_set_size=len(related),
        )

    def _collect_radius(
        self, anchors: list[SymbolRef], hops: int,
    ) -> dict[tuple[str, str], int]:
        """BFS the graph from each anchor, returning {(path, symbol): distance}.

        Uses callers + callees as undirected edges; the question "what
        related code matters" rarely cares about direction. Distance 0 =
        the anchor itself.
        """
        assert self._graph is not None
        visited: dict[tuple[str, str], int] = {}
        q: deque[tuple[SymbolRef, int]] = deque()
        for a in anchors:
            key = (a.path, a.symbol)
            if key not in visited:
                visited[key] = 0
                q.append((a, 0))

        while q:
            sym, d = q.popleft()
            if d >= hops:
                continue
            try:
                # Outgoing (callees) AND incoming (callers) -- undirected.
                neighbors = list(self._graph.callees_of(sym.symbol, path=sym.path))
                neighbors.extend(self._graph.callers_of(sym.symbol, path=sym.path))
            except Exception as e:
                log.debug("graph_augmented.neighbor_fetch_failed",
                          symbol=sym.symbol, err=str(e))
                continue
            for n in neighbors:
                key = (n.path, n.symbol)
                if key in visited:
                    continue
                visited[key] = d + 1
                q.append((n, d + 1))
        return visited

    @staticmethod
    def _apply_graph_boost(
        hits: list[SearchHit], related: dict[tuple[str, str], int],
    ) -> list[SearchHit]:
        """Boost hits whose (path, symbol) is in the radius, weighted by
        inverse distance. Resort by boosted score."""
        boosted: list[SearchHit] = []
        for h in hits:
            key = (h.chunk.path, h.chunk.symbol or "")
            d = related.get(key)
            # Also try without the symbol -- sometimes the anchor's PATH
            # is what matters (e.g. "all chunks in the same file").
            if d is None:
                d = related.get((h.chunk.path, ""))
            if d is None:
                # Try any symbol in the same path. If we found something at
                # the path level, it's a weaker (path-only) match.
                for (p, _s), dist in related.items():
                    if p == h.chunk.path:
                        d = dist + 1  # path-level match is weaker than symbol-level
                        break
            if d is None:
                boosted.append(h)
                continue
            # Boost = (1 + 1/(1 + d)) -- distance 0 → 2x, distance 1 → 1.5x,
            # distance 2 → 1.33x, etc. Bounded so it can't dwarf the
            # original score's signal entirely.
            factor = 1.0 + 1.0 / (1.0 + d)
            new_score = h.score * factor
            reason = (h.match_reason or "") + f" | graph d={d} x{factor:.2f}"
            boosted.append(SearchHit(
                chunk=h.chunk,
                score=new_score,
                source="hybrid",
                match_reason=reason.strip(),
            ))
        boosted.sort(key=lambda x: x.score, reverse=True)
        return boosted
