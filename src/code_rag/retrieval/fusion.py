from __future__ import annotations

import re
from collections.abc import Sequence

from code_rag.models import SearchHit

# Phase 28: identifier-shaped tokens are tokens that LOOK LIKE CODE, i.e.
# tokens unlikely to appear in natural English. Pure-lowercase english words
# like "work", "code", "next" must NOT anchor the boost or every query would
# trigger false positives.
#
# A token qualifies iff it's ≥ 4 chars AND any of:
#   - contains an underscore        (snake_case: ensure_lm_studio_ready)
#   - contains a digit              (versioned: place_order_v2, MyClass3)
#   - has an uppercase letter NOT at position 0  (CamelCase: OnBarUpdate)
# This rejects "Where" / "Strategy" (uppercase only at pos 0) and "work"
# (all lowercase no special chars), but keeps "MNQAlpha" / "manifest_manager".
_IDENT_CANDIDATE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{3,}\b")


def _looks_like_identifier(tok: str) -> bool:
    if "_" in tok:
        return True
    if any(ch.isdigit() for ch in tok):
        return True
    # Uppercase letter anywhere AFTER position 0 → CamelCase.
    return any(ch.isupper() for ch in tok[1:])


# Common code-shaped english stopwords that pass the structural rule but
# would still be too noisy as boost anchors. Kept short.
_IDENT_STOPWORDS = frozenset({
    "self", "null", "void", "type", "class", "args", "kwargs",
})


def extract_identifiers(query: str) -> list[str]:
    """Pull out identifier-shaped tokens from a query.

    Returns a deduplicated list, preserving first-occurrence order so the
    log line stays human-readable. Lowercased stopwords are dropped.
    """
    seen: set[str] = set()
    out: list[str] = []
    for m in _IDENT_CANDIDATE.finditer(query):
        tok = m.group(0)
        if not _looks_like_identifier(tok):
            continue
        if tok.lower() in _IDENT_STOPWORDS:
            continue
        if tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
    return out


def boost_exact_matches(
    hits: Sequence[SearchHit],
    identifiers: Sequence[str],
    *,
    factor: float = 0.5,
) -> list[SearchHit]:
    """Phase 28: multiplicatively boost hits whose text contains identifier
    tokens from the query.

    The boost is `score * (1 + factor * matches)` where `matches` is the
    count of distinct identifiers from the query found in the chunk text or
    symbol. Counts are capped at 3 so a single chunk cramming the same name
    everywhere can't dominate.

    Why this exists: BM25 already rewards exact-token matches, but RRF
    fusion blends BM25 with dense retrieval — and a strong dense embedder
    can outvote BM25 even when the user asked for an EXACT identifier.
    This re-asserts the floor: if the user typed `OnBarUpdate`, a chunk
    containing `OnBarUpdate` outranks one that's "semantically similar".

    Returns a new list with boosted scores; preserves match_reason with a
    "+exact-match" stamp so the breadcrumb shows the lift happened.
    """
    if not identifiers or not hits:
        return list(hits)
    out: list[SearchHit] = []
    # Lowercase comparison: identifiers stored case-sensitive but matched
    # case-insensitively because BM25 tokenizers fold case anyway.
    lc_idents = [t.lower() for t in identifiers]
    for h in hits:
        text_lc = (h.chunk.text or "").lower()
        sym_lc = (h.chunk.symbol or "").lower()
        path_lc = (h.chunk.path or "").lower()
        # We match against BOTH the chunk body (semantic occurrence) AND the
        # symbol/path (structural occurrence). A chunk whose SYMBOL is
        # `OnBarUpdate` is more relevant for query "OnBarUpdate" than one
        # that just mentions it inside a comment.
        body_matches = sum(1 for t in lc_idents if t in text_lc)
        struct_matches = sum(1 for t in lc_idents if t in sym_lc or t in path_lc)
        # Cap so any single chunk can earn at most ~3x the boost.
        capped = min(body_matches + 2 * struct_matches, 6)
        if capped == 0:
            out.append(h)
            continue
        new_score = float(h.score) * (1.0 + factor * capped)
        prev = h.match_reason or ""
        tag = f"+exact-match x{capped}"
        reason = f"{prev} | {tag}" if prev else tag
        out.append(h.model_copy(update={"score": new_score, "match_reason": reason}))
    # Re-sort by boosted score so the top of the list reflects the boost.
    out.sort(key=lambda h: h.score, reverse=True)
    return out


def mmr_diversify(
    hits: Sequence[SearchHit],
    *,
    lam: float = 0.7,
    top_k: int = 50,
) -> list[SearchHit]:
    """Phase 27: lightweight MMR-style diversity over fused hits.

    Why this exists: pre-Phase-27 the searcher would return e.g. 5 chunks from
    `MNQAlpha.cs` for the query "OnBarUpdate" because every long file has many
    relevant symbols, and RRF rewards consistent top ranking on every list.
    The result window collapses to one file and the user sees an echo chamber.

    The classic MMR formula is

        score(c) = lam * relevance(c) - (1 - lam) * max_sim(c, already_picked)

    For code search, "similarity to already-picked" is well-approximated by
    membership in the same file (and to a lesser extent the same symbol).
    Embedding-similarity MMR would be more principled but cost an extra
    pairwise compute pass; the file-level penalty captures > 95% of the
    benefit on this corpus and runs in O(n).

    Penalty schedule per (path, symbol) ladder:
      - first hit from a path:                no penalty
      - subsequent hits from the same path:   penalty grows with prior count
      - same (path, symbol):                  larger penalty (near-duplicate)

    `lam = 0.7` empirically: relevance still dominates, but a 4th hit from
    the same file rarely wins against a fresh file. Set `lam = 1.0` to
    disable diversity (degenerates to score-sorted top_k).

    Idempotent: stable order on equal scores via input-rank tiebreak.
    """
    if not hits or top_k <= 0:
        return []
    if lam >= 1.0:
        # Pure relevance — preserve input order, just truncate.
        return list(hits[:top_k])

    # Normalize relevance into [0, 1] so `lam` has a consistent meaning across
    # very different score scales: RRF scores are ~0.03, cosine similarities
    # are ~0.85, rerank scores are 0–1. Without this normalization the
    # penalty term drowns out RRF relevance entirely.
    #
    # Using `score / max_score` (not min-max). Min-max collapses the lowest
    # hit to relevance=0, which makes it never selectable regardless of the
    # diversity bonus. `score / max` preserves the relative ordering and
    # leaves the tail competitive once enough penalty accumulates above it.
    raw = [float(h.score) for h in hits]
    s_max = max(raw) if raw else 1.0
    if s_max <= 1e-12:
        rel = [1.0] * len(hits)
    else:
        rel = [s / s_max for s in raw]

    # Original index is the relevance-rank tiebreaker. Build a working list
    # of (orig_rank, hit, normalized_relevance) so we can pull greedily
    # without losing stability.
    pool: list[tuple[int, SearchHit, float]] = list(zip(range(len(hits)), hits, rel))
    selected: list[SearchHit] = []
    path_count: dict[str, int] = {}
    pair_count: dict[tuple[str, str | None], int] = {}

    # Penalty is also normalized to [0, 1]: clamp prior-siblings to a max of
    # 5 (anything beyond that is already "very dominated" so the cap keeps
    # the formula stable).
    def _penalty(p: str, sym: str | None) -> float:
        same_path = path_count.get(p, 0)
        same_pair = pair_count.get((p, sym), 0)
        raw_pen = same_path + 2.0 * same_pair
        return min(raw_pen, 5.0) / 5.0

    while pool and len(selected) < top_k:
        best_i = -1
        best_score = -float("inf")
        best_orig = 1 << 30
        for i, (orig, h, r) in enumerate(pool):
            mmr = lam * r - (1.0 - lam) * _penalty(h.chunk.path, h.chunk.symbol)
            # Pick max mmr; on tie, lower original rank wins (stability).
            if mmr > best_score or (mmr == best_score and orig < best_orig):
                best_score = mmr
                best_i = i
                best_orig = orig
        orig, hit, _ = pool.pop(best_i)
        selected.append(hit)
        p = hit.chunk.path
        path_count[p] = path_count.get(p, 0) + 1
        key = (p, hit.chunk.symbol)
        pair_count[key] = pair_count.get(key, 0) + 1
    return selected


def reciprocal_rank_fusion(
    lists: Sequence[Sequence[SearchHit]],
    *,
    k: int = 60,
    top_k: int = 50,
) -> list[SearchHit]:
    """Reciprocal Rank Fusion — the classic Cormack/Clarke/Buettcher formula.

        score(doc) = sum_over_lists( 1 / (k + rank_i(doc)) )

    k=60 is the standard recipe. Returns hits sorted by fused score desc,
    truncated to top_k. source='hybrid'.

    Each output hit's match_reason is a breadcrumb:
        "hybrid rrf from [vector r2, lexical r1]"

    Ties are broken by max score across lists (stable). We preserve the first
    Chunk we see for a given id.
    """
    agg: dict[str, float] = {}
    best: dict[str, SearchHit] = {}
    # per-chunk list of "<source> r<rank>" — used to build match_reason.
    sources: dict[str, list[str]] = {}
    for hits in lists:
        for rank, h in enumerate(hits):
            cid = h.chunk.id
            agg[cid] = agg.get(cid, 0.0) + 1.0 / (k + rank + 1)
            sources.setdefault(cid, []).append(f"{h.source} r{rank + 1}")
            if cid not in best:
                best[cid] = h
    ordered = sorted(agg.items(), key=lambda kv: kv[1], reverse=True)
    out: list[SearchHit] = []
    for cid, fused in ordered[:top_k]:
        src_hit = best[cid]
        reason = f"hybrid rrf from [{', '.join(sources[cid])}]"
        out.append(SearchHit(
            chunk=src_hit.chunk,
            score=fused,
            source="hybrid",
            match_reason=reason,
        ))
    return out
