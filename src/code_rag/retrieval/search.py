from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

from code_rag.interfaces.embedder import Embedder
from code_rag.interfaces.graph_store import GraphStore, SymbolRef
from code_rag.interfaces.lexical_store import LexicalStore
from code_rag.interfaces.reranker import Reranker
from code_rag.interfaces.vector_store import VectorStore
from code_rag.logging import get
from code_rag.models import Chunk, SearchHit
from code_rag.retrieval.fusion import reciprocal_rank_fusion

log = get(__name__)


@dataclass(frozen=True)
class SearchParams:
    k_final: int = 8           # what the caller gets back
    k_vector: int = 50         # per-list fan-out before fusion
    k_lexical: int = 50
    k_rerank_in: int = 50      # the reranker's input size (capped by available hits)
    rrf_k: int = 60

    # ---- filters ------------------------------------------------------------
    path_glob: str | None = None
    language: str | None = None
    symbol: str | None = None

    # ---- Phase 9 polish -----------------------------------------------------

    # Confidence threshold: hits below this are dropped. Applied AFTER rerank
    # (using the cross-encoder score) OR after fusion (using RRF score if the
    # reranker is a no-op/fallback). Default 0.0 = no threshold (current behavior).
    # Typical tuned values: 0.2-0.4 for rerank, 0.01 for RRF.
    min_score: float = 0.0

    # Per-hit context truncation. 0 = no truncation (return full chunk text).
    # When set, chunks longer than this are truncated with a `... (truncated N chars)`
    # marker so callers get bounded context.
    max_chars_per_hit: int = 0

    # Attach 1-hop graph neighborhood (callers + callees) to each hit. Off by
    # default because the graph walk adds latency and the caller can always do
    # get_callers/get_callees separately if they want it.
    attach_neighbors: bool = False


@dataclass
class SearchResponse:
    """Envelope for a search call. Lets us return 'no confident match' as a
    first-class result rather than a bare empty list."""

    hits: list[SearchHit] = field(default_factory=list)
    no_confident_match: bool = False   # True iff min_score filter emptied the list
    elapsed_ms: float = 0.0
    # Per-hit neighborhood (keyed by chunk id). Populated iff attach_neighbors=True.
    neighborhood: dict[str, dict[str, list[SymbolRef]]] = field(default_factory=dict)


class HybridSearcher:
    """Vector + Lexical -> RRF -> Reranker -> top-k, with confidence gate,
    smart truncation, and optional graph neighborhood attach.

    All polish flags default to OFF so the search API stays backward-compatible
    with Phase 2 callers; tools that want the "work-of-art" experience opt in.
    """

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        lexical_store: LexicalStore,
        reranker: Reranker,
        graph_store: GraphStore | None = None,
        hyde_plan: object | None = None,  # HydeRetrieverPlan, optional
    ) -> None:
        self._embed = embedder
        self._vec = vector_store
        self._lex = lexical_store
        self._rerank = reranker
        self._graph = graph_store
        # Phase 19 HyDE retriever plan. If set, the searcher consults
        # `plan.plan(query)` to (optionally) add a hypothetical-doc query
        # arm for natural-language questions. None = pure literal search.
        self._hyde = hyde_plan

    async def search(self, query: str, params: SearchParams) -> list[SearchHit]:
        """Back-compat entry point. Returns a bare list.

        New callers should prefer `search_full()` which returns the structured
        SearchResponse envelope (carries no_confident_match, latency, neighbors).
        """
        return (await self.search_full(query, params)).hits

    async def search_full(self, query: str, params: SearchParams) -> SearchResponse:
        t0 = time.monotonic()

        # Phase 19: HyDE plan. Identifier queries get a single-arm plan
        # (no LLM call); NL/mixed queries get a 2-arm plan with a
        # hypothetical-doc embedding fused alongside the literal one.
        if self._hyde is not None:
            plan = await self._hyde.plan(query)  # type: ignore[attr-defined]
        else:
            plan = [(query, 1.0)]

        # 1) Kick off vector queries (one per plan arm) + lexical (single,
        #    using the LITERAL query — BM25 over a hypothetical chunk would
        #    add noise without recall lift).
        embed_inputs = [arm_text for arm_text, _w in plan]
        q_vec_task = asyncio.create_task(self._embed.embed(embed_inputs))
        lex_task = asyncio.create_task(
            asyncio.to_thread(self._lex.query, query, params.k_lexical),
        )

        q_vecs = await q_vec_task
        where = self._build_where(params)

        # Run vector queries for each plan arm in parallel.
        vec_tasks = [
            asyncio.create_task(asyncio.to_thread(
                self._vec.query, qv, params.k_vector, where,
            ))
            for qv in q_vecs
        ]
        vec_hits_per_arm = [self._apply_filters(await t, params) for t in vec_tasks]
        lex_hits = self._apply_filters(await lex_task, params)

        # For symmetry with the previous logging shape, expose the union
        # candidate count of all arms as `vec_hits`.
        all_vec_hits = [h for arm in vec_hits_per_arm for h in arm]

        # 2) Filters already applied above.
        vec_hits = all_vec_hits

        # 3) Fuse — every plan arm contributes a ranked list, plus lexical.
        fused = reciprocal_rank_fusion(
            [*vec_hits_per_arm, lex_hits],
            k=params.rrf_k,
            top_k=max(params.k_rerank_in, params.k_final),
        )

        # 4) Rerank top candidates. If the reranker is a no-op/fallback, the
        #    hits come back with their fusion score intact.
        rerank_in = fused[: params.k_rerank_in]
        reranked = await self._rerank.rerank(query, rerank_in, params.k_final)

        # 5) Confidence gate.
        no_match = False
        if params.min_score > 0.0:
            gated = [h for h in reranked if h.score >= params.min_score]
            if not gated:
                no_match = True
            reranked = gated

        # 6) Smart per-hit truncation.
        if params.max_chars_per_hit > 0:
            reranked = [self._truncate(h, params.max_chars_per_hit) for h in reranked]

        # 7) Optional neighborhood attach.
        neighborhood: dict[str, dict[str, list[SymbolRef]]] = {}
        if params.attach_neighbors and self._graph is not None:
            neighborhood = await asyncio.to_thread(self._attach_neighbors, reranked)

        dur_ms = (time.monotonic() - t0) * 1000
        log.info(
            "search.done",
            query_preview=query[:80],
            vec=len(vec_hits),
            lex=len(lex_hits),
            fused=len(fused),
            returned=len(reranked),
            no_confident_match=no_match,
            ms=round(dur_ms, 1),
        )
        return SearchResponse(
            hits=list(reranked),
            no_confident_match=no_match,
            elapsed_ms=dur_ms,
            neighborhood=neighborhood,
        )

    # ---- filters ------------------------------------------------------------

    @staticmethod
    def _build_where(params: SearchParams) -> dict[str, object] | None:
        """Vector store (Chroma) metadata filter.

        path_glob is NOT usable here — Chroma supports exact $eq / $in, not
        fnmatch. We handle path_glob post-hoc on the vector hits too.
        """
        clauses: list[dict[str, object]] = []
        if params.language:
            clauses.append({"language": params.language})
        if params.symbol:
            clauses.append({"symbol": params.symbol})
        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

    @staticmethod
    def _apply_filters(hits: Sequence[SearchHit], params: SearchParams) -> list[SearchHit]:
        import fnmatch
        out = []
        for h in hits:
            if params.language and h.chunk.language != params.language:
                continue
            if params.symbol and h.chunk.symbol != params.symbol:
                continue
            if params.path_glob and not fnmatch.fnmatchcase(h.chunk.path, params.path_glob):
                continue
            out.append(h)
        return out

    # ---- per-hit polish -----------------------------------------------------

    @staticmethod
    def _truncate(h: SearchHit, limit: int) -> SearchHit:
        """Return a copy of the hit with chunk.text shortened to `limit` chars.

        Preserves the first `limit - marker_len` chars so the signature and
        first body lines stay visible; the tail is replaced with a comment
        marker that says how many chars were dropped.
        """
        text = h.chunk.text
        if len(text) <= limit:
            return h
        marker = f"\n// ... (truncated {len(text) - limit} chars; ask get_chunk_text for full body) ...\n"
        head = text[: max(1, limit - len(marker))]
        new_chunk = Chunk(
            id=h.chunk.id,
            repo=h.chunk.repo,
            path=h.chunk.path,
            language=h.chunk.language,
            symbol=h.chunk.symbol,
            kind=h.chunk.kind,
            start_line=h.chunk.start_line,
            end_line=h.chunk.end_line,
            text=head + marker,
        )
        return h.model_copy(update={"chunk": new_chunk})

    def _attach_neighbors(
        self, hits: Sequence[SearchHit],
    ) -> dict[str, dict[str, list[SymbolRef]]]:
        """For each hit with a resolved symbol, fetch 1-hop graph neighborhood.

        Returned shape: {chunk_id: {"callers": [...], "callees": [...]}}.

        We ask for callers WITHOUT a path filter because the graph extractor
        (see code_rag/graph/extractor.py) creates Calls edges to extern nodes
        for unresolved callees — so scoping by target path would miss the
        in-corpus callers whose edges terminate on the extern placeholder. For
        callees we scope by source path, which is always the hit's own file.
        """
        assert self._graph is not None
        out: dict[str, dict[str, list[SymbolRef]]] = {}
        for h in hits:
            if not h.chunk.symbol:
                out[h.chunk.id] = {"callers": [], "callees": []}
                continue
            # Strip the parent prefix for callers-lookup — calls in source land
            # the bare method name (e.g., `helper()`, not `Bar.helper()`).
            short = h.chunk.symbol.rsplit(".", 1)[-1]
            try:
                callers = self._graph.callers_of(short, path=None)
                callees = self._graph.callees_of(h.chunk.symbol, path=h.chunk.path)
            except Exception as e:  # graph errors shouldn't kill the search pipeline
                log.warning("search.neighbor_fail", symbol=h.chunk.symbol, err=str(e))
                callers, callees = [], []
            out[h.chunk.id] = {"callers": callers, "callees": callees}
        return out
