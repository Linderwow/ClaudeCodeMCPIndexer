from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from code_rag.interfaces.embedder import Embedder
from code_rag.interfaces.graph_store import GraphStore, SymbolRef
from code_rag.interfaces.lexical_store import LexicalStore
from code_rag.interfaces.reranker import Reranker
from code_rag.interfaces.vector_store import VectorStore
from code_rag.logging import get
from code_rag.models import Chunk, SearchHit
from code_rag.retrieval.fusion import (
    boost_exact_matches,
    extract_identifiers,
    mmr_diversify,
    reciprocal_rank_fusion,
)

log = get(__name__)


# Phase 38: cached repo roots for the recent_files_only_days filter.
# `load_settings()` reads config + the dynamic_roots.json registry — fast
# (~10 ms) but no need to hammer it on every chunk. We refresh on a TTL
# so newly auto-discovered roots become visible without a process restart.
_ROOTS_CACHE: tuple[tuple, float] = ((), 0.0)
_ROOTS_TTL_S = 60.0


def _cached_roots() -> tuple:
    """Return a tuple of resolved Path roots, refreshed at most every 60s."""
    global _ROOTS_CACHE
    now = time.time()
    cached, expiry = _ROOTS_CACHE
    if cached and now < expiry:
        return cached
    try:
        from code_rag.config import load_settings
        roots = tuple(p for p in load_settings().all_roots())
    except Exception:
        roots = ()
    _ROOTS_CACHE = (roots, now + _ROOTS_TTL_S)
    return roots


def _is_path_recent(rel_or_abs_path: str, cutoff: float) -> bool:
    """Return True iff the file at `rel_or_abs_path` was modified at or after
    `cutoff` (Unix epoch seconds). Tries the path as-is first (covers absolute
    paths), then each configured root as a prefix. Missing files / OS errors
    are treated as too-old."""
    from pathlib import Path as _PathCls
    p = _PathCls(rel_or_abs_path)
    candidates: list[_PathCls] = [p]
    if not p.is_absolute():
        for root in _cached_roots():
            candidates.append(root / rel_or_abs_path)
    for c in candidates:
        try:
            if c.exists():
                return c.stat().st_mtime >= cutoff
        except OSError:
            continue
    return False


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

    # ---- Phase 27 MMR diversity --------------------------------------------

    # MMR lambda applied between fusion and rerank. 1.0 = pure relevance
    # (current behavior, no diversity), 0.7 = balanced relevance/diversity
    # (default — picks a fresh file over a 4th hit from the same one),
    # 0.5 = diversity-heavy, 0.0 = max diversity. Disable by setting 1.0.
    mmr_lambda: float = 0.7

    # Phase 28: multiplicative boost applied to hits whose text contains
    # identifier-shaped tokens from the query. score *= (1 + factor * matches).
    # 0.0 = disabled; 0.5 = +50% per matched identifier (capped at 3 per hit).
    exact_match_boost: float = 0.5

    # ---- Phase 37-K: filter UX --------------------------------------------

    # If set to a positive int, drop hits whose source file's mtime is
    # older than N days. `None` or `0` disables the filter (no-op). Typical
    # values: 1 (today), 7 (this week), 30 (this month).
    # Most code-search use cases benefit: agents asking "where did we change
    # X recently" don't want chunks from files that haven't been touched in
    # a year showing up. File-system stat() is cheap (cached at OS level).
    recent_files_only_days: int | None = None

    # If set, multiplicatively boost hits whose path is under this root.
    # Useful for steering an agent currently editing repo A toward A's
    # files vs unrelated repo B. score *= (1 + prefer_root_boost) for
    # hits in the preferred root; others untouched. Path comparison is
    # case-insensitive and works on either absolute or relative paths.
    prefer_root: str | None = None
    prefer_root_boost: float = 0.25


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
        rewriter: object | None = None,   # LMStudioQueryRewriter, optional
        decomposer: object | None = None, # Phase 37-A LMStudioQueryDecomposer
        reflector: object | None = None,  # Phase 37-B LMStudioReflector
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
        # Phase 30 query rewriter. If set, the searcher expands identifier-
        # shaped queries into casing variants (and, with an LLM, related
        # identifiers). Composes with HyDE — both contribute arms. None =
        # pure literal search.
        self._rewriter = rewriter
        # Phase 37-A query decomposer. Splits multi-part questions into
        # 2-3 sub-queries, each becoming an extra retrieval arm. Heuristic
        # gate skips the LLM call for short / single-part queries.
        self._decomposer = decomposer
        # Phase 37-B post-rerank reflector. Re-orders the top-K cross-
        # encoder hits using a small chat model that scores how directly
        # each chunk answers the query. Confidence-gated (skipped when
        # cross-encoder top-1 is already strong).
        self._reflector = reflector

    async def search(self, query: str, params: SearchParams) -> list[SearchHit]:
        """Back-compat entry point. Returns a bare list.

        New callers should prefer `search_full()` which returns the structured
        SearchResponse envelope (carries no_confident_match, latency, neighbors).
        """
        return (await self.search_full(query, params)).hits

    async def search_full(self, query: str, params: SearchParams) -> SearchResponse:
        t0 = time.monotonic()

        # Phase 37-A + Phase 30 + Phase 19 plan composition. Order:
        #   1. Decomposer (if any) splits multi-part queries into sub-
        #      queries that each become their own retrieval arm.
        #   2. Rewriter (if any) seeds the arm list with original + casing
        #      variants + LLM-suggested expansions.
        #   3. HyDE (if any) appends a hypothetical-doc arm on NL queries.
        # If none are configured we fall back to the single-arm literal
        # search — identical to pre-Phase-19 behavior.
        plan: list[tuple[str, float]] = []
        if self._rewriter is not None:
            try:
                rw = await self._rewriter.rewrite(query)  # type: ignore[attr-defined]
                plan = list(rw.arms)
            except Exception as e:  # rewriter is best-effort
                log.warning("search.rewrite_fail", err=f"{type(e).__name__}: {e}")
        if not plan:
            plan = [(query, 1.0)]
        # Phase 37-A: decomposition runs AFTER the rewriter so casing
        # variants of the original query stay in the plan; sub-queries
        # are appended without replacing them. The decomposer gates on
        # multi-part-ness internally (returns just `original` otherwise).
        if self._decomposer is not None:
            try:
                decomp = await self._decomposer.decompose(query)  # type: ignore[attr-defined]
                literal_norm = query.strip().lower()
                # `decomp.arms` always includes the original; skip it
                # (already in `plan`) and append only the new sub-queries.
                seen = {arm_text.strip().lower() for arm_text, _w in plan}
                for arm_text, arm_w in decomp.arms:
                    norm = arm_text.strip().lower()
                    if norm == literal_norm or norm in seen:
                        continue
                    plan.append((arm_text, arm_w))
                    seen.add(norm)
            except Exception as e:
                log.warning("search.decompose_fail", err=f"{type(e).__name__}: {e}")
        if self._hyde is not None:
            try:
                hyde_arms = await self._hyde.plan(query)  # type: ignore[attr-defined]
                # Drop the literal arm from HyDE (we already have it) and
                # keep the hypothetical-doc arms.
                literal_norm = query.strip()
                for arm_text, arm_w in hyde_arms:
                    if arm_text.strip() != literal_norm:
                        plan.append((arm_text, arm_w))
            except Exception as e:
                log.warning("search.hyde_fail", err=f"{type(e).__name__}: {e}")

        # 1) Kick off vector queries (one per plan arm) + lexical (single,
        #    using the LITERAL query — BM25 over a hypothetical chunk would
        #    add noise without recall lift).
        # Phase 37 audit fix: previously a failure in the embedder or any
        # vector-query future would orphan the lexical task and other arms
        # ("Task was destroyed but it is pending"). Wrap the awaits so any
        # raise inside one await still cancels + drains the rest.
        embed_inputs = [arm_text for arm_text, _w in plan]
        q_vec_task = asyncio.create_task(self._embed.embed(embed_inputs))
        lex_task = asyncio.create_task(
            asyncio.to_thread(self._lex.query, query, params.k_lexical),
        )

        vec_tasks: list[asyncio.Task[Sequence[SearchHit]]] = []
        vec_tasks: list[asyncio.Task[Any]] = []
        try:
            q_vecs = await q_vec_task
            where = self._build_where(params)

            # Run vector queries for each plan arm in parallel.
            vec_tasks = [
                asyncio.create_task(asyncio.to_thread(
                    self._vec.query, qv, params.k_vector, where,
                ))
                for qv in q_vecs
            ]
            # Phase 38: return_exceptions=True so a single failed arm doesn't
            # orphan its siblings. Failed arms degrade gracefully to empty
            # results (the union across remaining arms still produces hits).
            raw_vec_results = await asyncio.gather(
                *vec_tasks, return_exceptions=True,
            )
            vec_results: list[list[SearchHit]] = []
            for i, r in enumerate(raw_vec_results):
                if isinstance(r, BaseException):
                    log.warning(
                        "search.vec_arm_failed",
                        arm=i, err=f"{type(r).__name__}: {r}",
                    )
                    vec_results.append([])
                else:
                    vec_results.append(r)  # type: ignore[arg-type]
            vec_hits_per_arm = [self._apply_filters(r, params) for r in vec_results]
            try:
                lex_raw = await lex_task
                lex_hits = self._apply_filters(lex_raw, params)
            except Exception as e:
                log.warning("search.lex_arm_failed",
                            err=f"{type(e).__name__}: {e}")
                lex_hits = []
        except BaseException:
            # Cancel any in-flight tasks so we don't leak coroutines on the
            # error path; suppress within the cleanup so the original
            # exception is what propagates to the caller.
            for t in (q_vec_task, lex_task, *vec_tasks):
                if not t.done():
                    t.cancel()
            for t in (q_vec_task, lex_task, *vec_tasks):
                with contextlib.suppress(BaseException):
                    await t
            raise

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

        # 3a) Phase 28: exact-identifier boost BEFORE diversity. If the query
        # contains identifier-shaped tokens (`OnBarUpdate`, `ensure_lm_studio_ready`),
        # chunks containing those tokens get a multiplicative score lift.
        # Runs ahead of MMR so the boosted hit is the one MMR considers
        # canonical for its file.
        if params.exact_match_boost > 0.0:
            idents = extract_identifiers(query)
            if idents:
                fused = boost_exact_matches(
                    fused, idents, factor=params.exact_match_boost,
                )

        # 3b) Phase 27: MMR diversity. Penalizes near-duplicates (same path,
        # same symbol) so the reranker sees a more diverse candidate slate.
        # Skipped when lambda >= 1.0 (relevance-only) so the path is a no-op
        # for callers that haven't opted in.
        if params.mmr_lambda < 1.0:
            fused = mmr_diversify(
                fused,
                lam=params.mmr_lambda,
                top_k=max(params.k_rerank_in, params.k_final),
            )

        # 4) Rerank top candidates. If the reranker is a no-op/fallback, the
        #    hits come back with their fusion score intact.
        rerank_in = fused[: params.k_rerank_in]
        reranked = await self._rerank.rerank(query, rerank_in, params.k_final)

        # 4b) Phase 37-B reflection. Confidence-gated inside the reflector:
        # if the cross-encoder top-1 is already strong, this is a no-op.
        # Best-effort — any LLM failure leaves the rerank order intact.
        if self._reflector is not None and reranked:
            try:
                reranked = await self._reflector.reflect(query, list(reranked))  # type: ignore[attr-defined]
            except Exception as e:
                log.warning("search.reflect_fail", err=f"{type(e).__name__}: {e}")

        # 4c) Phase 37-K: prefer-root boost. Runs after rerank so the boost
        # applies to scores the user actually sees in match_reason. Cheap:
        # one pure-Python prefix check per hit. Sort-stable so non-boosted
        # hits keep their relative rerank order.
        if params.prefer_root:
            reranked = self._apply_root_boost(reranked, params)

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
        # Phase 37-K: pre-compute the recency cutoff once outside the loop.
        # mtime comparison uses os.path.getmtime which is microseconds and
        # OS-cached after the first call per path.
        cutoff: float | None = None
        if params.recent_files_only_days is not None and params.recent_files_only_days > 0:
            import time
            cutoff = time.time() - params.recent_files_only_days * 86400.0

        out = []
        for h in hits:
            if params.language and h.chunk.language != params.language:
                continue
            if params.symbol and h.chunk.symbol != params.symbol:
                continue
            if params.path_glob and not fnmatch.fnmatchcase(h.chunk.path, params.path_glob):
                continue
            # Phase 38: chunk.path is repo-relative (e.g. 'src/foo.py').
            # Plain Path(...).exists() resolves vs cwd, which is wrong
            # when the watcher / dashboard / search CLI has a different
            # cwd than any indexed root. Try all configured roots in
            # priority order; first that resolves wins.
            if cutoff is not None and not _is_path_recent(h.chunk.path, cutoff):
                continue
            out.append(h)
        return out

    @staticmethod
    def _apply_root_boost(
        hits: Sequence[SearchHit], params: SearchParams,
    ) -> list[SearchHit]:
        """Phase 37-K: multiplicatively boost hits under `prefer_root`.

        Comparison is case-insensitive and uses both forward and back
        slashes interchangeably so Windows / POSIX paths work.
        """
        if not params.prefer_root or params.prefer_root_boost <= 0:
            return list(hits)
        prefix = params.prefer_root.replace("\\", "/").rstrip("/").lower()
        boost = 1.0 + params.prefer_root_boost
        out: list[SearchHit] = []
        for h in hits:
            path = h.chunk.path.replace("\\", "/").lower()
            if path.startswith(prefix + "/") or path == prefix:
                # Stamp the boost in match_reason so the agent sees why this
                # rose in the rankings — debuggability matters for trust.
                reason = (h.match_reason or "") + f" +prefer_root({params.prefer_root_boost:+.2f})"
                out.append(h.model_copy(update={
                    "score": h.score * boost,
                    "match_reason": reason.strip(),
                }))
            else:
                out.append(h)
        # Re-sort because boosting may have reshuffled ordering.
        out.sort(key=lambda h: h.score, reverse=True)
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
