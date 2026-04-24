from __future__ import annotations

import asyncio
from collections.abc import Sequence

import httpx

from code_rag.interfaces.reranker import Reranker
from code_rag.logging import get
from code_rag.models import SearchHit

log = get(__name__)


class LMStudioReranker(Reranker):
    """Cross-encoder reranker via LM Studio.

    LM Studio exposes rerankers through an OpenAI-compatible /v1/rerank endpoint
    (Jina-style). If the configured model doesn't expose that endpoint, we fall
    back to no-op (keep input order) and log a warning rather than crash the
    whole search pipeline.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_s: float = 60.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = httpx.Timeout(timeout_s, connect=5.0)
        self._client: httpx.AsyncClient | None = None
        self._lock = asyncio.Lock()
        self._endpoint_ok: bool | None = None  # lazily discovered

    @property
    def model(self) -> str:
        return self._model

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            async with self._lock:
                if self._client is None:
                    self._client = httpx.AsyncClient(base_url=self._base_url, timeout=self._timeout)
        return self._client

    async def health(self) -> None:
        client = await self._ensure_client()
        r = await client.get("/models")
        r.raise_for_status()
        models = {m.get("id") for m in r.json().get("data", [])}
        if self._model not in models:
            raise RuntimeError(
                f"Reranker model {self._model!r} not loaded in LM Studio. "
                f"Loaded: {sorted(x for x in models if x)}"
            )

    async def rerank(
        self, query: str, hits: Sequence[SearchHit], top_k: int,
    ) -> list[SearchHit]:
        if not hits:
            return []
        # Fast path: once we've proven the endpoint is absent, don't retry
        # per-query — just fall back to input order.
        if self._endpoint_ok is False:
            return list(hits[:top_k])

        client = await self._ensure_client()

        # On the VERY first rerank, probe /models with a short timeout to catch
        # "user has reranker model unloaded" without stalling 60s on POST /rerank.
        if self._endpoint_ok is None:
            try:
                r = await client.get("/models", timeout=httpx.Timeout(5.0, connect=2.0))
                r.raise_for_status()
                loaded = {m.get("id") for m in r.json().get("data", [])}
                if self._model not in loaded:
                    log.warning("reranker.model_not_loaded", model=self._model,
                                loaded=sorted(x for x in loaded if x))
                    self._endpoint_ok = False
                    return list(hits[:top_k])
            except (httpx.HTTPError, ValueError) as e:
                log.warning("reranker.probe_failed", err=str(e))
                self._endpoint_ok = False
                return list(hits[:top_k])

        payload = {
            "model": self._model,
            "query": query,
            "documents": [h.chunk.text for h in hits],
            "top_n": top_k,
        }
        try:
            r = await client.post("/rerank", json=payload)
            r.raise_for_status()
            data = r.json()
            self._endpoint_ok = True
        except (httpx.HTTPError, ValueError) as e:
            # Route not supported or malformed response. Fall back.
            if self._endpoint_ok is not False:
                log.warning("reranker.fallback_noop", err=str(e))
            self._endpoint_ok = False
            return list(hits[:top_k])

        # Jina-style response: {"results": [{"index": i, "relevance_score": f}, ...]}
        results = data.get("results") or []
        if not results:
            return list(hits[:top_k])
        ordered: list[SearchHit] = []
        for r_ in results[:top_k]:
            idx = int(r_.get("index", 0))
            score = float(r_.get("relevance_score", 0.0))
            if 0 <= idx < len(hits):
                src = hits[idx]
                # Append rerank breadcrumb to whatever earlier stages wrote.
                prev = src.match_reason or ""
                reason = f"{prev} | rerank cross-encoder {score:.3f}" if prev else \
                         f"rerank cross-encoder {score:.3f}"
                ordered.append(SearchHit(
                    chunk=src.chunk,
                    score=score,
                    source="rerank",
                    match_reason=reason,
                ))
        return ordered

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
