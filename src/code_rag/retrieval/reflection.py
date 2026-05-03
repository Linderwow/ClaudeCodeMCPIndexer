"""Phase 37-B: post-retrieval reflection.

Why this exists
---------------
NVIDIA's RAG blueprint exposes "An optional reflection step can further
validate or refine the answer against the retrieved context." The same
idea, applied at retrieval time rather than answer time:

After fusion + cross-encoder rerank we have the top-K candidates. The
cross-encoder scores text-to-query relevance, but it doesn't reason
about whether the chunk *actually answers* the question. A small chat
model can:

  - Demote chunks that LOOK relevant (share tokens) but actually answer
    a different question.
  - Spot the rare case where the top-1 is a near-miss (e.g. a test
    fixture for the function rather than the function itself).

We use it as a soft *re-ranker on top of the cross-encoder*: it can
adjust scores within the existing top-K but never invents new candidates.
That preserves the cross-encoder's recall floor and only spends LLM
cycles on the chunks the user is actually about to see.

Design
------
- Operates on `list[SearchHit]` returned by the cross-encoder.
- Calls a small chat model with all candidates in a single prompt
  (listwise scoring, like the lm_chat reranker but for relevance not
  ordering).
- Output is a JSON map of `{chunk_id: score 0-1}`. We blend with the
  existing reranker score: `final = alpha * rerank + (1 - alpha) * reflection`.
- alpha defaults to 0.6 — keep cross-encoder dominant; let reflection
  nudge.
- HEURISTIC GATE: only fires when the top-1 cross-encoder score is
  ambiguous (score < 0.6). Confident matches don't need a second
  opinion and reflection just adds latency.
- NEVER raises into the searcher. Any failure returns the input list
  unchanged.
"""
from __future__ import annotations

import asyncio
import json

import httpx

from code_rag.logging import get
from code_rag.models import SearchHit

log = get(__name__)


# Top-K to reflect on. Larger => more LLM context, more latency. The MCP
# tools cap final results at 8 anyway, so 8 covers everything the user
# will see.
_REFLECT_TOP_K = 8

# Max chars per candidate sent to the LLM. The reflection prompt sees
# the chunk text, so 400 covers signature + first body lines without
# blowing past the small model's context.
_MAX_CHARS_PER_DOC = 400

# Confidence threshold above which we skip reflection (top-1 is clearly
# good already). Cross-encoder scores are roughly 0-1 normalized.
_SKIP_IF_TOP1_ABOVE = 0.7

# Score-blend weight: final = alpha * rerank + (1 - alpha) * reflection.
_BLEND_ALPHA = 0.6


class LMStudioReflector:
    """Post-rerank LLM relevance check on the top-K candidates.

    Best-effort: any failure returns hits unchanged. Heuristic gate
    skips reflection when the cross-encoder is already confident.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_s: float = 15.0,
        top_k: int = _REFLECT_TOP_K,
        max_chars_per_doc: int = _MAX_CHARS_PER_DOC,
        blend_alpha: float = _BLEND_ALPHA,
        skip_if_top1_above: float = _SKIP_IF_TOP1_ABOVE,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = httpx.Timeout(timeout_s, connect=3.0)
        self._top_k = top_k
        self._max_chars = max_chars_per_doc
        self._alpha = blend_alpha
        self._skip_above = skip_if_top1_above
        self._client: httpx.AsyncClient | None = None
        self._lock = asyncio.Lock()

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            async with self._lock:
                if self._client is None:
                    self._client = httpx.AsyncClient(
                        base_url=self._base_url, timeout=self._timeout,
                    )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def reflect(
        self, query: str, hits: list[SearchHit],
    ) -> list[SearchHit]:
        """Return hits re-ordered by blended (rerank, reflection) score.

        Length is preserved; we only re-rank within the input.
        """
        if not self._model or not hits:
            return hits
        # Confidence gate: skip the LLM call when top-1 is already
        # clearly relevant. Saves ~1s per query on confident searches.
        if hits[0].score >= self._skip_above:
            return hits

        # Only reflect on the top-K. The tail is left in original order
        # so we don't drop chunks the user might still want to see.
        head = hits[: self._top_k]
        tail = hits[self._top_k:]

        scores = await self._score(query, head)
        if not scores:
            return hits  # LLM failed; pass through unchanged.

        # Blend cross-encoder score with reflection score. Both scales
        # are roughly 0-1; if the cross-encoder kind is "noop" the
        # rerank score is whatever fusion produced (RRF) which is much
        # smaller — but we still blend so reflection doesn't dominate
        # absent any other signal.
        rescored: list[tuple[float, SearchHit]] = []
        for h in head:
            r_score = scores.get(h.chunk.id, 0.5)
            blended = self._alpha * h.score + (1.0 - self._alpha) * r_score
            rescored.append((blended, h.model_copy(update={"score": blended})))

        rescored.sort(key=lambda t: t[0], reverse=True)
        return [h for _, h in rescored] + tail

    async def _score(
        self, query: str, hits: list[SearchHit],
    ) -> dict[str, float]:
        """Ask the LLM for a relevance score per candidate.

        Returns {chunk_id: score 0..1} or {} on any failure.
        """
        items = []
        for i, h in enumerate(hits, start=1):
            text = (h.chunk.text or "")[: self._max_chars]
            items.append({
                "id": h.chunk.id,
                "label": f"[{i}] {h.chunk.path}:{h.chunk.symbol or '?'}",
                "text": text,
            })

        prompt = (
            "You score how directly each code/doc snippet ANSWERS the "
            "developer's question. A snippet that mentions the topic but "
            "doesn't answer the question scores low.\n"
            "Output ONE JSON object. No prose, no markdown.\n\n"
            f"Question: {query}\n\n"
            "Candidates:\n"
            + "\n\n".join(
                f"{it['label']}\n{it['text']}"
                for it in items
            )
            + "\n\n"
            'Return: {"scores": [{"id": <id>, "score": <0.0..1.0>}, ...]}\n'
            "Rules:\n"
            "- One entry per candidate (use the exact `id` from the prompt).\n"
            "- score 1.0 = directly answers the question.\n"
            "- score 0.5 = related but doesn't answer.\n"
            "- score 0.0 = unrelated."
        )
        schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "reflection_scores",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "scores": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "string"},
                                    "score": {"type": "number"},
                                },
                                "required": ["id", "score"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["scores"],
                    "additionalProperties": False,
                },
            },
        }

        try:
            client = await self._ensure_client()
            r = await client.post("/chat/completions", json={
                "model": self._model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 600,
                "response_format": schema,
            })
            r.raise_for_status()
            data = r.json()
            content = data["choices"][0]["message"]["content"]
        except Exception as e:
            log.warning("reflection.llm_fail", err=f"{type(e).__name__}: {e}")
            return {}

        try:
            obj = json.loads(content) if isinstance(content, str) else None
        except json.JSONDecodeError:
            return {}
        if not isinstance(obj, dict):
            return {}
        arr = obj.get("scores")
        if not isinstance(arr, list):
            return {}
        out: dict[str, float] = {}
        for entry in arr:
            if not isinstance(entry, dict):
                continue
            cid = entry.get("id")
            score = entry.get("score")
            if not isinstance(cid, str):
                continue
            if not isinstance(score, int | float):
                continue
            # Clamp to [0, 1] so a misbehaving model can't poison the blend.
            out[cid] = max(0.0, min(1.0, float(score)))
        return out
