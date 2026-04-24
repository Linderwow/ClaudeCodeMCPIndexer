from __future__ import annotations

from collections.abc import Sequence

from code_rag.models import SearchHit


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
