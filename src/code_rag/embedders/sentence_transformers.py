"""Phase 34: sentence-transformers embedder backend.

Why this exists alongside `LMStudioEmbedder`
--------------------------------------------
LM Studio only loads GGUF-format models. Many state-of-the-art code-
specialized embedders (BAAI/bge-code-v1, codesage/codesage-large-v2,
nomic-embed-code) are released as SafeTensors only — LM Studio refuses
to load them. The MTEB Code leaderboard's top spots are held by these
models; until now they were inaccessible.

This backend bypasses LM Studio entirely. It loads the model directly
via sentence-transformers (the same dependency we already have for the
cross-encoder reranker), runs inference on GPU when CUDA is available,
and exposes the same `Embedder` interface so the rest of the pipeline
doesn't care.

Design:
  - Lazy model load on first health() / embed() call.
  - Model + tokenizer kept in memory for the process lifetime.
  - encode() runs on a thread pool so the asyncio loop isn't blocked.
  - GPU autopilot (cuda > mps > cpu); device override via config.

Deps:
  - sentence-transformers < 5     (Phase 29 cross-encoder also pinned here)
  - transformers          < 5     (compatibility)
  - huggingface_hub       < 1     (compatibility — v1.x has the httpx bug)
  - torch (CPU works; CUDA strongly preferred for speed)

Memory cost:
  bge-code-v1     1024d, ~1.3 GB on GPU
  codesage-large  1024d, ~1.6 GB on GPU
  qwen3-emb-8b    4096d, ~5.5 GB on GPU

These sit alongside the LM Studio chat / HyDE / reranker models — total
budget on a 22 GB card is comfortable with the Phase 33 VRAM trims.
"""
from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from code_rag.interfaces.embedder import Embedder
from code_rag.logging import get

log = get(__name__)


class SentenceTransformersEmbedder(Embedder):
    """Embeds text via a local sentence-transformers model.

    Invariants:
      - `model` property returns the configured HF id verbatim (used for
        IndexMeta stamping; mismatch on open => refuse to query).
      - `dim` is probed on first health() call by encoding a short string.
      - `embed()` batches internally (sentence-transformers handles micro-
        batching), no need to pre-chunk.
      - Single shared model instance per process (lazy-loaded under a lock).
    """

    def __init__(
        self,
        model: str,
        *,
        device: str | None = None,
        batch_size: int = 32,
        normalize: bool = True,
        trust_remote_code: bool = False,
    ) -> None:
        # Verify the dep is importable up front so the user gets a clear
        # error at config-load time, not at first query.
        try:
            import sentence_transformers  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "sentence_transformers embedder backend requires the "
                "`sentence-transformers` package. Install with "
                "`pip install -e \".[cross-encoder]\"` from the code-rag-mcp "
                "repo (the cross-encoder optional dep already includes it), "
                "then restart."
            ) from e

        self._model_id = model
        self._device = device
        self._batch_size = batch_size
        self._normalize = normalize
        self._trust_remote_code = trust_remote_code
        self._st_model: Any | None = None
        self._dim: int = 0
        self._lock = asyncio.Lock()

    @property
    def model(self) -> str:
        return self._model_id

    @property
    def dim(self) -> int:
        if self._dim <= 0:
            raise RuntimeError(
                "Embedder dim not yet probed. Call health() or embed() first."
            )
        return self._dim

    async def health(self) -> None:
        """Lazy-load the model + probe dim by encoding a short string.

        Failures here are fatal — there's no fallback embedder, and the
        index meta needs a valid dim before any chunks can be upserted.
        """
        await self._ensure_loaded()
        if self._dim <= 0:
            # Probe dim by embedding a known-short string.
            vec = await self.embed(["probe"])
            self._dim = len(vec[0])
            log.info("st_embedder.dim_probed", model=self._model_id, dim=self._dim)

    async def _ensure_loaded(self) -> Any:
        if self._st_model is not None:
            return self._st_model
        async with self._lock:
            if self._st_model is not None:
                return self._st_model
            self._st_model = await asyncio.to_thread(self._load_model)
            log.info(
                "st_embedder.loaded",
                model=self._model_id,
                device=self._device or "auto",
            )
            return self._st_model

    def _load_model(self) -> Any:
        # Imported here so the module-level import stays cheap.
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(
            self._model_id,
            device=self._device,
            trust_remote_code=self._trust_remote_code,
        )

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        st_model = await self._ensure_loaded()
        # `encode` is CPU/GPU-bound; run on a thread to avoid blocking the loop.
        # `convert_to_numpy=True` returns a 2D ndarray; we cast to plain list[list[float]]
        # at the boundary so downstream stores see a stable type.
        try:
            arr = await asyncio.to_thread(
                st_model.encode,
                list(texts),
                batch_size=self._batch_size,
                normalize_embeddings=self._normalize,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        except Exception as e:
            log.error("st_embedder.encode_fail", err=f"{type(e).__name__}: {e}")
            raise
        # arr.shape == (len(texts), dim). tolist() preserves order.
        return [list(map(float, row)) for row in arr]

    async def aclose(self) -> None:
        """Drop the model from memory. Safe to call multiple times.

        We don't unload from GPU explicitly — Python GC + torch's allocator
        will release VRAM on next major collection. Forcing it via
        `torch.cuda.empty_cache()` is rarely worth the complexity.
        """
        self._st_model = None
