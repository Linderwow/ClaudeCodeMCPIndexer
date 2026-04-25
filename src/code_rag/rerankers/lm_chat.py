"""Phase 24: chat-completion-based reranker.

LM Studio doesn't implement /v1/rerank — calling it returns
`{"error":"Unexpected endpoint or method. (POST /v1/rerank)"}`. The original
LMStudioReranker silently fell back to no-op for the lifetime of the project.

This module ships a *listwise* reranker that uses /v1/chat/completions, which
LM Studio DOES implement. The flow:

    1. Take the top-N RRF candidates from the hybrid searcher (N capped at
       `max_candidates` so the prompt stays inside the chat model's context
       and latency budget).
    2. Truncate each candidate's text to `max_chars_per_doc` (signature + a
       few body lines is enough for a 1.7B model to judge relevance).
    3. Send ONE chat-completion call asking the LLM to rank the candidates by
       relevance to the query. Force JSON output via `response_format`.
    4. Parse the JSON; on any failure (network, bad JSON, hallucinated indices)
       fall back to the input order so the search pipeline never breaks.

Why qwen/qwen3-1.7b
-------------------
- Already loaded for HyDE (kept loaded as a side effect; no extra RAM cost).
- 1.7B is large enough to judge relevance; small enough that one call ≤ 2 s
  on consumer hardware. Bigger models (4B, 7B) would help quality but at the
  cost of unacceptable per-query latency.
- This is the same model loaded by the user's autostart bootstrap.

Why listwise (not pointwise)
----------------------------
- Pointwise (one call per pair) would be >= 50 calls per search -> 10-30 s.
- Listwise gets ranking quality close to a real cross-encoder at one call.
- Position bias is a known concern but minor for our top-20-in / top-8-out
  ratios; if the eval flags it, we can pre-shuffle the candidate order.

Latency budget
--------------
Default settings (`max_candidates=20`, `max_chars_per_doc=300`):
    prompt size ≈ 6 K chars ≈ 1.5 K tokens
    qwen3-1.7b prompt processing ~ 1.5 K tps  → 1 s prompt
    generation ~ 50 tokens of JSON @ 60 tps    → 0.8 s gen
    end-to-end                                  ≈ 2 s per query
"""
from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Sequence
from typing import Any

import httpx

from code_rag.interfaces.reranker import Reranker
from code_rag.logging import get
from code_rag.models import SearchHit

log = get(__name__)


class LMStudioChatReranker(Reranker):
    """Listwise reranker via /v1/chat/completions on a small chat model.

    Invariants:
      - NEVER raises into the caller. All failures fall back to input order.
      - Order-preserving: hits not chosen by the LLM (or beyond `top_k`)
        keep their input rank when padding.
      - Idempotent: same (query, candidate set) -> same ranking (temperature 0).
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_s: float = 30.0,
        max_candidates: int = 20,
        max_chars_per_doc: int = 300,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = httpx.Timeout(timeout_s, connect=5.0)
        self._max_candidates = max_candidates
        self._max_chars = max_chars_per_doc
        self._client: httpx.AsyncClient | None = None
        self._lock = asyncio.Lock()

    @property
    def model(self) -> str:
        return self._model

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            async with self._lock:
                if self._client is None:
                    self._client = httpx.AsyncClient(
                        base_url=self._base_url, timeout=self._timeout,
                    )
        return self._client

    async def health(self) -> None:
        client = await self._ensure_client()
        r = await client.get("/models")
        r.raise_for_status()
        models = {m.get("id") for m in r.json().get("data", [])}
        if self._model not in models:
            raise RuntimeError(
                f"Chat reranker model {self._model!r} not loaded in LM Studio. "
                f"Loaded: {sorted(x for x in models if x)}"
            )

    async def rerank(
        self, query: str, hits: Sequence[SearchHit], top_k: int,
    ) -> list[SearchHit]:
        if not hits or top_k <= 0:
            return []
        if len(hits) == 1:
            return [self._stamp_rerank_reason(hits[0], 0)]

        # Window the input — LLM sees at most max_candidates. Keeps the prompt
        # bounded so latency is predictable regardless of upstream top_k_in.
        window = list(hits[: self._max_candidates])
        prompt = self._build_prompt(query, window)

        ranking = await self._call_llm(prompt, len(window))
        if ranking is None:
            return [self._stamp_rerank_reason(h, i, fallback=True)
                    for i, h in enumerate(window[:top_k])]

        # Compose final order: LLM-ranked first (deduped), then any window
        # candidate the LLM omitted (keeps coverage at top_k).
        seen: set[int] = set()
        final_order: list[int] = []
        for idx in ranking:
            if 0 <= idx < len(window) and idx not in seen:
                final_order.append(idx)
                seen.add(idx)
        for i in range(len(window)):
            if i not in seen:
                final_order.append(i)

        out: list[SearchHit] = []
        for new_rank, idx in enumerate(final_order[:top_k]):
            out.append(self._stamp_rerank_reason(window[idx], new_rank))
        return out

    # ---- LLM I/O ------------------------------------------------------------

    async def _call_llm(self, prompt: str, n_candidates: int) -> list[int] | None:
        # LM Studio rejects `response_format.type = "json_object"` (the
        # OpenAI-spec value); it requires `"json_schema"` or `"text"`. We pin
        # an exact schema so the model can't return an unparseable shape.
        schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "rerank_ranking",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "ranking": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "minItems": 1,
                            "maxItems": n_candidates,
                        },
                    },
                    "required": ["ranking"],
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
                "max_tokens": 256,
                "response_format": schema,
            })
            r.raise_for_status()
            data = r.json()
            content = data["choices"][0]["message"]["content"]
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as e:
            log.warning("rerank.chat.http_fail", err=f"{type(e).__name__}: {e}")
            return None

        return self._parse_ranking(content, n_candidates)

    def _build_prompt(self, query: str, hits: Sequence[SearchHit]) -> str:
        # One line per candidate to keep the model's positional grounding tight.
        lines = [
            "You re-rank code search results by relevance to a query.",
            "Output ONE valid JSON object only. No prose, no markdown.",
            "",
            f"Query: {query}",
            "",
            "Candidates (id :: path :: symbol :: snippet):",
        ]
        for i, h in enumerate(hits):
            sym = h.chunk.symbol or "?"
            text = (h.chunk.text or "").strip()
            text = text.replace("\r", " ").replace("\n", "  ")
            text = text[: self._max_chars]
            lines.append(f"[{i}] {h.chunk.path} :: {sym} :: {text}")
        lines += [
            "",
            'Return: {"ranking": [<ids most-relevant-first>]}',
            "Use ONLY ids from the list above. Include every id exactly once.",
        ]
        return "\n".join(lines)

    @staticmethod
    def _parse_ranking(content: Any, n_candidates: int) -> list[int] | None:
        """Robustly extract a list of ints from `content`.

        Handles three response shapes seen in the wild from small models:
            1. {"ranking": [3, 1, 7, ...]}                  (preferred)
            2. [3, 1, 7, ...]                                (bare array)
            3. JSON wrapped in markdown fences or with prose (small models leak)
        Returns None on total failure so the caller can choose fallback.
        """
        if not isinstance(content, str):
            return None
        text = content.strip()

        # Strip ```json fences a small model might emit despite json_object.
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

        for candidate in (text, "{" + text + "}", _grab_json_blob(text)):
            if not candidate:
                continue
            try:
                obj = json.loads(candidate)
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(obj, dict):
                arr = obj.get("ranking") or obj.get("ids") or obj.get("order")
            elif isinstance(obj, list):
                arr = obj
            else:
                arr = None
            if not isinstance(arr, list):
                continue
            ranked = [int(x) for x in arr
                      if isinstance(x, (int, float)) and 0 <= int(x) < n_candidates]
            if ranked:
                return ranked

        # Last-ditch: scrape ints in order.
        ints = [int(m) for m in re.findall(r"\b\d+\b", text)]
        scraped = [i for i in ints if 0 <= i < n_candidates]
        return scraped or None

    # ---- helpers ------------------------------------------------------------

    @staticmethod
    def _stamp_rerank_reason(
        h: SearchHit, new_rank: int, *, fallback: bool = False,
    ) -> SearchHit:
        prev = h.match_reason or ""
        tag = f"rerank chat (fallback) r{new_rank + 1}" if fallback \
              else f"rerank chat r{new_rank + 1}"
        reason = f"{prev} | {tag}" if prev else tag
        # Use a synthetic monotone score: top result -> 1.0, last -> ~0.
        # Downstream code only uses the order, but we keep score informative.
        score = max(0.0, 1.0 - new_rank * 0.05)
        return SearchHit(
            chunk=h.chunk, score=score, source="rerank", match_reason=reason,
        )

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


def _grab_json_blob(text: str) -> str:
    """If the model wrapped JSON in prose, find the first {...} or [...] blob."""
    obj_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if obj_match:
        return obj_match.group(0)
    arr_match = re.search(r"\[.*\]", text, flags=re.DOTALL)
    if arr_match:
        return arr_match.group(0)
    return ""
