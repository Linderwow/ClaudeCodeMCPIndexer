from __future__ import annotations

import asyncio
from collections.abc import Sequence

import httpx

from code_rag.interfaces.embedder import Embedder
from code_rag.logging import get

log = get(__name__)


class LMStudioEmbedder(Embedder):
    """OpenAI-compatible /v1/embeddings client pointed at LM Studio.

    Dimension is probed on first embed() call (or explicit health()) and cached.
    Batches input to avoid blowing past the server's per-request limit.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        dim: int = 0,
        timeout_s: float = 60.0,
        batch: int = 32,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._dim = dim
        self._timeout = httpx.Timeout(timeout_s, connect=5.0)
        self._batch = max(1, batch)
        self._client: httpx.AsyncClient | None = None
        self._lock = asyncio.Lock()

    @property
    def model(self) -> str:
        return self._model

    @property
    def dim(self) -> int:
        if self._dim <= 0:
            raise RuntimeError("Embedder dim not yet probed. Call health() or embed() first.")
        return self._dim

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            async with self._lock:
                if self._client is None:
                    # Phase 60-O (audit P2): explicit connection pool tuning.
                    # Defaults (20 keepalive / 100 max) are enough but the
                    # 5s default keepalive_expiry tears down idle connections
                    # between bursts -- on a re-index hitting /embeddings at
                    # 44 req/s, that's 2-5ms TCP/TLS handshake on every batch
                    # boundary. Bump keepalive_expiry to 120s so connections
                    # survive the indexer's per-file gaps. http2=True lets
                    # parallel arms multiplex on a single connection during
                    # search.
                    limits = httpx.Limits(
                        max_keepalive_connections=16,
                        max_connections=16,
                        keepalive_expiry=120.0,
                    )
                    self._client = httpx.AsyncClient(
                        base_url=self._base_url,
                        timeout=self._timeout,
                        limits=limits,
                    )
        return self._client

    async def _raw_embed_with_retry(self, texts: Sequence[str]) -> list[list[float]]:
        # Phase 60-O (audit IPC#4): retry transient 5xx with backoff so a
        # single vLLM 503 (GPU pressure spike) doesn't abort an entire
        # multi-batch embed call mid-stream. Pass-through on 4xx (those are
        # client-side bugs and retrying just papers them over).
        last_exc: Exception | None = None
        for attempt in (0, 1, 2):
            try:
                return await self._raw_embed(texts)
            except httpx.HTTPStatusError as e:
                last_exc = e
                if e.response.status_code < 500 or attempt == 2:
                    raise
                import asyncio as _aio
                await _aio.sleep(1.0 * (2 ** attempt))  # 1s, 2s
            except (httpx.HTTPError, httpx.RemoteProtocolError) as e:
                last_exc = e
                if attempt == 2:
                    raise
                import asyncio as _aio
                await _aio.sleep(1.0 * (2 ** attempt))
        # Unreachable, but mypy demands.
        assert last_exc is not None
        raise last_exc

    async def health(self) -> None:
        """Ping /v1/models and, if dim is unknown, probe it via a 1-token embedding."""
        client = await self._ensure_client()
        r = await client.get("/models")
        r.raise_for_status()
        data = r.json()
        models = {m.get("id") for m in data.get("data", [])}
        if self._model not in models:
            raise RuntimeError(
                f"Embedder model {self._model!r} not loaded in LM Studio. "
                f"Loaded: {sorted(x for x in models if x)}"
            )
        if self._dim <= 0:
            vecs = await self._raw_embed(["probe"])
            self._dim = len(vecs[0])
            log.info("embedder.dim_probed", model=self._model, dim=self._dim)

    async def _raw_embed(self, texts: Sequence[str]) -> list[list[float]]:
        client = await self._ensure_client()
        payload = {"model": self._model, "input": list(texts)}
        r = await client.post("/embeddings", json=payload)
        r.raise_for_status()
        data = r.json()
        # OpenAI spec: data is a list of {index, embedding}; preserve request order.
        items = sorted(data["data"], key=lambda d: d["index"])
        return [item["embedding"] for item in items]

    async def _raw_embed_with_400_split(
        self, texts: Sequence[str],
    ) -> list[list[float]]:
        # Phase 60-O hotfix (round 2): when vLLM returns 400 (typically
        # "context length 8192 tokens" exceeded by ONE oversized chunk in
        # the batch), the entire batch fails. Previously this discarded up
        # to 32 chunks for one bad input. Now: on 400 with batch>1, fall
        # back to per-item embedding so good inputs still land. Bad inputs
        # are zero-padded with the embedder's dim so the caller can still
        # tell stride from index-by-position. The DocChunker hard-split
        # fix should make this never trigger -- this is the last-line
        # defense if a code chunk somehow exceeds max_model_len.
        try:
            return await self._raw_embed_with_retry(texts)
        except httpx.HTTPStatusError as e:
            if e.response.status_code != 400 or len(texts) <= 1:
                raise
            log.warning("embedder.batch_400_split",
                        size=len(texts),
                        body=e.response.text[:200])
            out: list[list[float]] = []
            for t in texts:
                try:
                    one = await self._raw_embed_with_retry([t])
                    out.append(one[0])
                except httpx.HTTPStatusError as inner:
                    if inner.response.status_code == 400:
                        # Bad chunk -- skip with zero vector so caller's
                        # stride stays right; len() will tell them.
                        log.warning("embedder.chunk_400_skipped",
                                    chars=len(t), body=inner.response.text[:160])
                        out.append([0.0] * self._dim)
                    else:
                        raise
            return out

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        if self._dim <= 0:
            await self.health()
        out: list[list[float]] = []
        for i in range(0, len(texts), self._batch):
            chunk = texts[i : i + self._batch]
            vecs = await self._raw_embed_with_400_split(chunk)
            if any(len(v) != self._dim for v in vecs):
                raise RuntimeError(
                    f"Embedding dim drift: expected {self._dim}, got {[len(v) for v in vecs]}"
                )
            out.extend(vecs)
        return out

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
