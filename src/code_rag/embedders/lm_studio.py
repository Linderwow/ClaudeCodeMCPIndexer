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
                    self._client = httpx.AsyncClient(base_url=self._base_url, timeout=self._timeout)
        return self._client

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

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        if self._dim <= 0:
            await self.health()
        out: list[list[float]] = []
        for i in range(0, len(texts), self._batch):
            chunk = texts[i : i + self._batch]
            vecs = await self._raw_embed(chunk)
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
